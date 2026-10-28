//! SelTranslate: "Audio file" tab — streaming transcription of mp3/m4a/flac/ogg/wav.
//!
//! file → `decoder` (16 kHz frames) → `chunker` (Silero VAD, 20–30 s chunks)
//! → `TranscriptionManager::transcribe` chunk by chunk, on a background
//! thread. Memory holds one chunk at a time, whatever the file length.

pub mod chunker;
pub mod decoder;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

use crate::engine_supervisor::EngineError;
use crate::managers::transcription::TranscriptionManager;
use crate::settings::{get_settings, ModelUnloadTimeout};
use chunker::{Chunk, VadChunker};
use decoder::StreamingDecoder;

pub const EVENT_PROGRESS: &str = "audio-file-progress";
pub const EVENT_DONE: &str = "audio-file-done";
pub const EVENT_ERROR: &str = "audio-file-error";
pub const EVENT_CANCELLED: &str = "audio-file-cancelled";

/// Live `ModelHold` guards. `TranscriptionManager::maybe_unload_immediately`
/// checks `model_held()` (one added line there — minimal diff to upstream).
static MODEL_HOLDS: AtomicUsize = AtomicUsize::new(0);

/// While alive, the `Immediately` unload setting doesn't unload the model, so
/// a multi-chunk job loads it once. The idle-timeout unload needs no guard:
/// every chunk's `transcribe()` refreshes the idle timer.
pub struct ModelHold(());

impl Drop for ModelHold {
    fn drop(&mut self) {
        MODEL_HOLDS.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn hold_model() -> ModelHold {
    MODEL_HOLDS.fetch_add(1, Ordering::AcqRel);
    ModelHold(())
}

pub fn model_held() -> bool {
    MODEL_HOLDS.load(Ordering::Acquire) > 0
}

/// Managed state: at most one file job at a time.
#[derive(Default)]
pub struct AudioFileJob {
    running: AtomicBool,
    cancel: AtomicBool,
    /// A chunk is inside `transcribe()` right now.
    in_chunk: AtomicBool,
}

#[derive(Clone, Serialize)]
struct ProgressEvent {
    /// "loading_model" | "transcribing"
    phase: &'static str,
    processed_secs: f64,
    total_secs: Option<f64>,
    chunk_index: usize,
    /// Text of the chunk just finished (empty for phase-only updates).
    chunk_text: String,
}

#[derive(Clone, Serialize)]
struct DoneEvent {
    text: String,
    elapsed_secs: f64,
    audio_secs: f64,
    /// File name in the transcript library (stage H), when it was saved.
    saved_as: Option<String>,
    /// Why the library save failed; the text is still in the event.
    save_error: Option<String>,
}

#[derive(Clone, Serialize)]
struct ErrorEvent {
    message: String,
}

enum Outcome {
    Done {
        chunks: Vec<String>,
        audio_secs: f64,
    },
    Cancelled,
}

/// What the job has to do with the loaded model.
#[derive(Debug, PartialEq, Eq)]
struct ModelPlan {
    /// Model to load before the first chunk; `None` = use what is loaded.
    load: Option<String>,
    /// The job runs on a model other than the dictation one, so it is
    /// unloaded afterwards (dictation would otherwise keep using it).
    separate: bool,
    /// The dictation model was loaded when the job began; bring it back.
    restore_dictation: bool,
}

/// `requested`: the model picked on the tab (`None`/empty = as dictation).
/// `loaded`: the model in the engine right now.
fn plan_model(requested: Option<&str>, selected: &str, loaded: Option<&str>) -> ModelPlan {
    match requested.filter(|m| !m.is_empty() && *m != selected) {
        None => ModelPlan {
            load: loaded.is_none().then(|| selected.to_string()),
            separate: false,
            restore_dictation: false,
        },
        Some(model) => ModelPlan {
            load: (loaded != Some(model)).then(|| model.to_string()),
            separate: true,
            restore_dictation: loaded == Some(selected),
        },
    }
}

/// Runs after the job on every exit path (including a panic): puts the
/// engine back the way dictation expects it.
struct ModelRestore {
    tm: Arc<TranscriptionManager>,
    plan: ModelPlan,
    unload_timeout: ModelUnloadTimeout,
}

impl Drop for ModelRestore {
    fn drop(&mut self) {
        if !self.plan.separate {
            self.tm.maybe_unload_immediately("audio file");
            return;
        }
        log::info!("Audio file: unloading the file model");
        if let Err(e) = self.tm.unload_model() {
            log::warn!("Audio file: failed to unload the file model: {e:#}");
        }
        // `Immediately` means dictation loads its model per use anyway.
        if self.plan.restore_dictation && self.unload_timeout != ModelUnloadTimeout::Immediately {
            self.tm.initiate_model_load();
        }
    }
}

/// Starts transcribing `path` on a background thread; returns at once.
/// `model_id`: model for this file; `None` = the dictation model.
pub fn start(app: &AppHandle, path: PathBuf, model_id: Option<String>) -> Result<(), String> {
    let job = app.state::<Arc<AudioFileJob>>().inner().clone();
    if crate::translate::file::is_running() {
        return Err("A file is being translated: wait for it to finish".to_string());
    }
    if job.running.swap(true, Ordering::AcqRel) {
        return Err("A file is already being transcribed".to_string());
    }
    job.cancel.store(false, Ordering::Release);

    let app = app.clone();
    let thread_job = job.clone();
    let spawned = std::thread::Builder::new()
        .name("audio-file".to_string())
        .spawn(move || {
            let job = thread_job;
            // Dictation and a file job would fight over the one loaded model:
            // the hotkeys stay off until this thread ends.
            let _hotkeys_off = crate::hotkeys_pause::job_lock(&app);
            let started = Instant::now();
            log::info!("Audio file: transcribing {}", path.display());
            let result = run(&app, &job, &path, model_id.as_deref());
            job.in_chunk.store(false, Ordering::Release);
            match result {
                Ok(Outcome::Done { chunks, audio_secs }) => {
                    let elapsed_secs = started.elapsed().as_secs_f64();
                    log::info!("Audio file: done, {audio_secs:.1}s of audio in {elapsed_secs:.1}s");
                    let saved = crate::library::save_transcript(&app, &path, audio_secs, &chunks);
                    let _ = app.emit(
                        EVENT_DONE,
                        DoneEvent {
                            text: chunks.join(" "),
                            elapsed_secs,
                            audio_secs,
                            saved_as: saved.saved_as,
                            save_error: saved.error,
                        },
                    );
                }
                Ok(Outcome::Cancelled) => {
                    log::info!("Audio file: cancelled");
                    let _ = app.emit(EVENT_CANCELLED, ());
                }
                Err(e) => {
                    log::error!("Audio file: {e:#}");
                    let _ = app.emit(
                        EVENT_ERROR,
                        ErrorEvent {
                            message: format!("{e:#}"),
                        },
                    );
                }
            }
            job.running.store(false, Ordering::Release);
        });
    if let Err(e) = spawned {
        job.running.store(false, Ordering::Release);
        return Err(format!("Failed to start transcription: {e}"));
    }
    Ok(())
}

/// Requests cancellation; the job stops between packets/chunks, and a chunk
/// already in the model is interrupted (transcribe-cpp models).
pub fn cancel(app: &AppHandle) {
    let job = app.state::<Arc<AudioFileJob>>();
    if !job.running.load(Ordering::Acquire) {
        return;
    }
    job.cancel.store(true, Ordering::Release);
    if job.in_chunk.load(Ordering::Acquire) {
        // Also stops a dictation queued behind our chunk — accepted.
        app.state::<Arc<TranscriptionManager>>()
            .cancel_transcription();
    }
}

fn run(
    app: &AppHandle,
    job: &AudioFileJob,
    path: &Path,
    model_id: Option<&str>,
) -> Result<Outcome> {
    let tm = app.state::<Arc<TranscriptionManager>>().inner().clone();
    let vad_path = app
        .path()
        .resolve(
            "resources/models/silero_vad_v4.onnx",
            tauri::path::BaseDirectory::Resource,
        )
        .map_err(|e| anyhow!("Failed to resolve VAD path: {e}"))?;

    let mut decoder = StreamingDecoder::open(path)?;
    let total_secs = decoder.duration_secs;
    let mut chunker = VadChunker::new(&vad_path)?;
    log::info!(
        "Audio file: {} {} Hz x{}, {}",
        decoder.codec,
        decoder.sample_rate,
        decoder.channels,
        total_secs.map_or("unknown length".to_string(), |s| format!("{s:.1}s"))
    );

    let emit_progress = |phase, processed_secs: f64, chunk_index, chunk_text: String| {
        let _ = app.emit(
            EVENT_PROGRESS,
            ProgressEvent {
                phase,
                processed_secs: total_secs.map_or(processed_secs, |t| processed_secs.min(t)),
                total_secs,
                chunk_index,
                chunk_text,
            },
        );
    };

    let settings = get_settings(app);
    if settings.selected_model.is_empty() {
        return Err(anyhow!("No model selected. Choose one in Models."));
    }
    let loaded = tm
        .is_model_loaded()
        .then(|| tm.get_current_model())
        .flatten();
    let plan = plan_model(model_id, &settings.selected_model, loaded.as_deref());
    log::info!(
        "Audio file: model {:?}, loaded {loaded:?}, plan {plan:?}",
        model_id.unwrap_or("(as dictation)")
    );
    let to_load = plan.load.clone();
    // Declared before the hold: dropped after it, on every exit path.
    let _restore = ModelRestore {
        tm: tm.clone(),
        plan,
        unload_timeout: settings.model_unload_timeout,
    };
    if let Some(model) = to_load {
        emit_progress("loading_model", 0.0, 0, String::new());
        load_file_model(&tm, job, &model)?;
    }
    // Held for the whole job, so `Immediately` doesn't unload between chunks.
    let hold = hold_model();
    emit_progress("transcribing", 0.0, 0, String::new());

    let mut texts: Vec<String> = Vec::new();
    let mut chunk_index = 0;
    let mut ready: Vec<Chunk> = Vec::new();
    let result = 'job: loop {
        if job.cancel.load(Ordering::Acquire) {
            break Ok(Outcome::Cancelled);
        }
        let more = match decoder.next_frames(|f| ready.extend(chunker.push_frame(f))) {
            Ok(more) => more,
            Err(e) => break Err(e),
        };
        if !more {
            ready.extend(chunker.finish());
        }

        for chunk in ready.drain(..) {
            let end_secs = chunk.start_secs + chunk.duration_secs();
            log::debug!(
                "Audio file: chunk {chunk_index} at {:.2}s, {:.2}s long, {} speech frames",
                chunk.start_secs,
                chunk.duration_secs(),
                chunk.speech_frames
            );
            match transcribe_chunk(&tm, job, chunk) {
                Ok(Some(text)) => {
                    let text = text.trim().to_string();
                    if !text.is_empty() {
                        texts.push(text.clone());
                    }
                    emit_progress("transcribing", end_secs, chunk_index, text);
                    chunk_index += 1;
                }
                Ok(None) => break 'job Ok(Outcome::Cancelled),
                Err(e) => break 'job Err(e),
            }
        }
        if !more {
            break Ok(Outcome::Done {
                chunks: std::mem::take(&mut texts),
                audio_secs: decoder.decoded_secs(),
            });
        }
    };

    drop(hold);
    result
}

/// Loads `model` unless another load is running; then waits for that one and
/// loads only if it left something else in the engine.
fn load_file_model(tm: &TranscriptionManager, job: &AudioFileJob, model: &str) -> Result<()> {
    let waited = Instant::now();
    let _loading = loop {
        if let Some(guard) = tm.try_start_loading() {
            break guard;
        }
        if job.cancel.load(Ordering::Acquire) || waited.elapsed().as_secs() > 120 {
            return Err(anyhow!("Another model is still loading"));
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    };
    if tm.is_model_loaded() && tm.get_current_model().as_deref() == Some(model) {
        return Ok(());
    }
    tm.load_model(model)
        .map_err(|e| e.context(format!("Failed to load model '{model}'")))
}

/// `Ok(None)` = cancelled by our own Cancel. Any other failure is retried
/// once: the dictation Cancel stops whatever runs in the engine, and an ONNX
/// engine reads as "not loaded" while a dictation is using it.
fn transcribe_chunk(
    tm: &TranscriptionManager,
    job: &AudioFileJob,
    chunk: Chunk,
) -> Result<Option<String>> {
    let mut samples = chunk.samples;
    for attempt in 0..2 {
        // Keep a copy (≤ 1.9 MB) only while a retry is still possible.
        let audio = if attempt == 0 {
            samples.clone()
        } else {
            std::mem::take(&mut samples)
        };
        job.in_chunk.store(true, Ordering::Release);
        let result = tm.transcribe(audio);
        job.in_chunk.store(false, Ordering::Release);
        let e = match result {
            Ok(text) => return Ok(Some(text)),
            Err(e) => e,
        };
        if job.cancel.load(Ordering::Acquire) {
            return Ok(None);
        }
        if attempt == 1 {
            return Err(e.context(format!("Chunk at {:.1}s failed twice", chunk.start_secs)));
        }
        log::warn!(
            "Audio file: chunk at {:.1}s failed ({e:#}), retrying once",
            chunk.start_secs
        );
        if !is_cancelled(&e) {
            // Give a concurrent dictation time to hand the engine back.
            std::thread::sleep(std::time::Duration::from_secs(1));
            if job.cancel.load(Ordering::Acquire) {
                return Ok(None);
            }
        }
    }
    unreachable!("loop returns on the second attempt")
}

fn is_cancelled(e: &anyhow::Error) -> bool {
    matches!(
        e.downcast_ref::<EngineError>(),
        Some(EngineError::Cancelled)
    )
}

/// Duration and codec, read without decoding.
pub fn probe(path: &Path) -> Result<(Option<f64>, String)> {
    let d = StreamingDecoder::open(path)?;
    Ok((d.duration_secs, d.codec))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_as_dictation_loads_only_when_nothing_is_loaded() {
        for requested in [None, Some(""), Some("small")] {
            let idle = plan_model(requested, "small", None);
            assert_eq!(idle.load.as_deref(), Some("small"));
            assert!(!idle.separate && !idle.restore_dictation);

            let ready = plan_model(requested, "small", Some("small"));
            assert_eq!(ready.load, None);
            assert!(!ready.separate);
        }
    }

    #[test]
    fn other_model_is_loaded_and_unloaded_afterwards() {
        let p = plan_model(Some("large"), "small", Some("small"));
        assert_eq!(p.load.as_deref(), Some("large"));
        assert!(p.separate);
        assert!(
            p.restore_dictation,
            "dictation model was loaded: bring it back"
        );
    }

    #[test]
    fn dictation_model_is_not_restored_if_it_was_not_loaded() {
        let p = plan_model(Some("large"), "small", None);
        assert_eq!(p.load.as_deref(), Some("large"));
        assert!(p.separate && !p.restore_dictation);
    }

    #[test]
    fn file_model_already_loaded_is_not_reloaded_but_still_unloaded() {
        let p = plan_model(Some("large"), "small", Some("large"));
        assert_eq!(p.load, None);
        assert!(p.separate && !p.restore_dictation);
    }
}
