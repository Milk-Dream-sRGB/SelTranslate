//! SelTranslate: "Speakers" on a History entry. Who spoke when (Sortformer,
//! `diarizer`), then each turn transcribed on its own with the selected
//! speech model, so every line gets its speaker and its start time:
//!
//! ```text
//! [00:00:04] Speaker 1: …
//! [00:00:17] Speaker 2: …
//! ```
//!
//! Nothing of Handy's pipeline is changed: turns go through Handy's own
//! background transcription (`TranscriptionManager::transcribe_in_chunks`,
//! which yields to dictation and loads the model), with the language the
//! user picked for this file. That language reaches Handy's settings reads
//! on the job's thread only ([`apply_language_override`]); the stored
//! settings are never touched. The result is only shown under the entry;
//! nothing is written to Handy's history.
//!
//! One job at a time. The diarizer file is downloaded on the first run of a
//! variant, into the Hugging Face cache like the translation models.

pub mod catalog;
mod diarizer;
pub mod turns;

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Result};
use hf_hub::api::tokio::CancellationToken;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_store::StoreExt;

use crate::managers::history::HistoryManager;
use crate::managers::model::{HttpDownloadEvent, HttpDownloadOutcome, ModelManager};
use crate::managers::transcription::{ChunkControl, TranscriptionManager};
use crate::settings::AppSettings;
use turns::Turn;

const STORE_KEY: &str = "seltranslate_speakers";
/// The most speakers Sortformer tells apart.
pub const MAX_SPEAKERS: u32 = 4;
const DICTATION_POLL: Duration = Duration::from_millis(200);
const RATE: i64 = 16_000;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// The panel's choices, remembered between runs.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Type)]
#[serde(default)]
pub struct SpeakersOptions {
    /// A `catalog::VARIANTS` id.
    pub variant: String,
    /// Spoken language of the file: a code, or "" for Handy's setting.
    pub language: String,
    /// How many people talk; 0 when the diarizer decides.
    pub speakers: u32,
}

impl Default for SpeakersOptions {
    fn default() -> Self {
        Self {
            variant: catalog::DEFAULT_VARIANT.into(),
            language: String::new(),
            speakers: 0,
        }
    }
}

impl SpeakersOptions {
    fn sanitized(mut self) -> Self {
        if catalog::variant(&self.variant).is_none() {
            self.variant = catalog::DEFAULT_VARIANT.into();
        }
        if self.speakers > MAX_SPEAKERS {
            self.speakers = 0;
        }
        self.language = self.language.trim().to_string();
        self
    }
}

fn load_options(app: &AppHandle) -> SpeakersOptions {
    app.store(crate::portable::store_path(
        crate::settings::SETTINGS_STORE_PATH,
    ))
    .ok()
    .and_then(|store| store.get(STORE_KEY))
    .and_then(|value| serde_json::from_value::<SpeakersOptions>(value).ok())
    .unwrap_or_default()
    .sanitized()
}

fn save_options(app: &AppHandle, options: &SpeakersOptions) -> Result<(), String> {
    let store = app
        .store(crate::portable::store_path(
            crate::settings::SETTINGS_STORE_PATH,
        ))
        .map_err(|e| format!("Failed to open the settings store: {e}"))?;
    let value = serde_json::to_value(options).map_err(|e| e.to_string())?;
    store.set(STORE_KEY, value);
    Ok(())
}

/// The Hugging Face cache the translation setup uses.
fn hub(app: &AppHandle) -> Result<PathBuf, String> {
    crate::translate::settings::load(app);
    crate::translate::settings::setup_roots()
        .map(|(_, hub)| hub)
        .ok_or_else(|| "The app data folder is unavailable".to_string())
}

// ---------------------------------------------------------------------------
// The language of this job's thread
// ---------------------------------------------------------------------------

thread_local! {
    static LANGUAGE: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Called at the end of `settings::get_settings`: on a Speakers job's thread,
/// the language picked for the file replaces the stored one in what Handy
/// reads. Every other thread sees the settings as stored.
pub fn apply_language_override(settings: &mut AppSettings) {
    LANGUAGE.with(|language| {
        if let Some(code) = language.borrow().as_ref() {
            settings.selected_language = code.clone();
        }
    });
}

/// Sets this thread's language until dropped.
struct LanguageGuard;

impl LanguageGuard {
    fn set(code: &str) -> Option<Self> {
        if code.is_empty() {
            return None;
        }
        LANGUAGE.with(|language| *language.borrow_mut() = Some(code.to_string()));
        Some(Self)
    }
}

impl Drop for LanguageGuard {
    fn drop(&mut self) {
        LANGUAGE.with(|language| *language.borrow_mut() = None);
    }
}

// ---------------------------------------------------------------------------
// The job
// ---------------------------------------------------------------------------

/// The entry being worked on; -1 when none is.
static RUNNING: AtomicI64 = AtomicI64::new(-1);
static CANCEL: AtomicBool = AtomicBool::new(false);
/// What a cancel has to reach besides the flag: the download and the
/// diarizer run in progress.
static DOWNLOAD: Mutex<Option<CancellationToken>> = Mutex::new(None);
static DIARIZE: Mutex<Option<transcribe_cpp::CancelToken>> = Mutex::new(None);

fn cancelled() -> bool {
    CANCEL.load(Ordering::SeqCst)
}

#[derive(Serialize, Clone, Copy, Debug, PartialEq, Type)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// `done` / `total` in bytes.
    Downloading,
    /// No counts: one run over the whole file.
    Detecting,
    /// `done` / `total` in turns.
    Transcribing,
}

#[derive(Serialize, Clone, Type)]
struct ProgressEvent {
    entry_id: i64,
    phase: Phase,
    done: f64,
    total: f64,
}

#[derive(Serialize, Clone, Debug, PartialEq, Type)]
pub struct TurnText {
    pub start_ms: f64,
    pub speaker: u32,
    pub text: String,
}

#[derive(Serialize, Clone, Type)]
struct DoneEvent {
    entry_id: i64,
    turns: Vec<TurnText>,
    speakers: u32,
}

#[derive(Serialize, Clone, Type)]
struct ErrorEvent {
    entry_id: i64,
    message: String,
}

#[derive(Serialize, Clone, Type)]
struct CancelledEvent {
    entry_id: i64,
}

fn progress(app: &AppHandle, entry_id: i64, phase: Phase, done: u64, total: u64) {
    let _ = app.emit(
        "speakers-progress",
        ProgressEvent {
            entry_id,
            phase,
            done: done as f64,
            total: total as f64,
        },
    );
}

/// Stops on cancel and waits out dictation, like Handy's file jobs.
struct Control<'a> {
    app: &'a AppHandle,
}

impl ChunkControl for Control<'_> {
    fn proceed(&self) -> bool {
        loop {
            if cancelled() {
                return false;
            }
            if !crate::tray::dictation_in_progress(self.app) {
                return true;
            }
            std::thread::sleep(DICTATION_POLL);
        }
    }

    fn progress(&self, _done: usize, _total: usize) {}
}

/// The diarizer file of `variant`, downloading it first if needed. `None`
/// when cancelled.
fn model_file(app: &AppHandle, entry_id: i64, variant: &str) -> Result<Option<PathBuf>> {
    let item = catalog::variant(variant).ok_or_else(|| anyhow!("Unknown model {variant}"))?;
    let hub = hub(app).map_err(|e| anyhow!(e))?;
    if let Some(path) = catalog::downloaded(&hub, item) {
        return Ok(Some(path));
    }
    let partial = crate::translate::setup::catalog::model_partial(&hub, item)
        .ok_or_else(|| anyhow!("The model has no cache folder"))?;
    if let Some(dir) = partial.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let token = CancellationToken::new();
    *DOWNLOAD.lock().unwrap_or_else(|e| e.into_inner()) = Some(token.clone());
    if cancelled() {
        return Ok(None);
    }
    info!("Speakers: downloading {}", item.file_name);
    progress(app, entry_id, Phase::Downloading, 0, item.size);
    let outcome = tauri::async_runtime::block_on(async {
        let url = crate::translate::setup::install::resolve_redirects(item.url)
            .await
            .map_err(|e| anyhow!("Download failed: {}", e.message))?;
        ModelManager::download_http_resumable_with_events(
            item.id,
            &url,
            &partial,
            Some(item.size),
            Some(item.sha256),
            &token,
            &|event| {
                if let HttpDownloadEvent::Progress(p) = event {
                    progress(app, entry_id, Phase::Downloading, p.downloaded, item.size);
                }
            },
        )
        .await
    });
    *DOWNLOAD.lock().unwrap_or_else(|e| e.into_inner()) = None;
    match outcome? {
        HttpDownloadOutcome::Cancelled => Ok(None),
        HttpDownloadOutcome::Completed => Ok(Some(catalog::finalize(&hub, item)?)),
    }
}

fn ms_to_samples(ms: i64, len: usize) -> usize {
    ((ms.max(0) * RATE / 1000) as usize).min(len)
}

/// The turns with their text; `None` when cancelled.
fn run(app: &AppHandle, entry_id: i64, options: &SpeakersOptions) -> Result<Option<Vec<TurnText>>> {
    let history = app.state::<Arc<HistoryManager>>().inner().clone();
    let entry = tauri::async_runtime::block_on(history.get_entry_by_id(entry_id))?
        .ok_or_else(|| anyhow!("The History entry is gone"))?;
    let audio_path = history.get_audio_file_path(&entry.file_name);
    if !audio_path.is_file() {
        return Err(anyhow!("The recording of this entry was deleted"));
    }

    let Some(model) = model_file(app, entry_id, &options.variant)? else {
        return Ok(None);
    };
    if cancelled() {
        return Ok(None);
    }

    let samples = crate::audio_toolkit::read_wav_samples(&audio_path)
        .map_err(|e| anyhow!("Failed to load audio: {e}"))?;
    if samples.is_empty() {
        return Err(anyhow!("Recording has no audio samples"));
    }
    let audio_ms = samples.len() as i64 * 1000 / RATE;

    progress(app, entry_id, Phase::Detecting, 0, 0);
    let token = transcribe_cpp::CancelToken::new();
    *DIARIZE.lock().unwrap_or_else(|e| e.into_inner()) = Some(token.clone());
    let detected = if cancelled() {
        Ok(diarizer::Outcome::Cancelled)
    } else {
        diarizer::diarize(&model, &samples, &token)
    };
    *DIARIZE.lock().unwrap_or_else(|e| e.into_inner()) = None;
    let segments = match detected? {
        diarizer::Outcome::Segments(segments) => segments,
        diarizer::Outcome::Cancelled => return Ok(None),
    };
    let limit = (options.speakers > 0).then_some(options.speakers);
    let plan: Vec<Turn> = turns::plan(&segments, limit, audio_ms);
    if plan.is_empty() {
        return Err(anyhow!("Recording contains no speech"));
    }
    info!(
        "Speakers: {} segments → {} turns of {} speakers",
        segments.len(),
        plan.len(),
        plan.iter().map(|t| t.speaker).max().unwrap_or(0)
    );

    let tm = app.state::<Arc<TranscriptionManager>>().inner().clone();
    let control = Control { app };
    let _language = LanguageGuard::set(&options.language);
    let mut out = Vec::with_capacity(plan.len());
    for i in 0..plan.len() {
        progress(
            app,
            entry_id,
            Phase::Transcribing,
            i as u64,
            plan.len() as u64,
        );
        let (from, to) = turns::padded_range(&plan, i, audio_ms);
        let piece =
            samples[ms_to_samples(from, samples.len())..ms_to_samples(to, samples.len())].to_vec();
        let Some(transcription) = tm.transcribe_in_chunks(piece, &control)? else {
            return Ok(None);
        };
        // Handy puts a paragraph break between chunks; a turn is one line.
        let text = transcription
            .text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        if !text.is_empty() {
            out.push(TurnText {
                start_ms: plan[i].start_ms as f64,
                speaker: plan[i].speaker,
                text,
            });
        }
    }
    if out.is_empty() {
        return Err(anyhow!("Recording contains no speech"));
    }
    Ok(Some(out))
}

/// Speakers numbered 1, 2, … again over the turns that kept some text.
fn renumber(turns: &mut [TurnText]) -> u32 {
    let mut order: Vec<u32> = Vec::new();
    for t in turns.iter_mut() {
        let n = match order.iter().position(|&s| s == t.speaker) {
            Some(p) => p,
            None => {
                order.push(t.speaker);
                order.len() - 1
            }
        };
        t.speaker = n as u32 + 1;
    }
    order.len() as u32
}

fn finish(app: &AppHandle, entry_id: i64, result: Result<Option<Vec<TurnText>>>) {
    let tm = app.state::<Arc<TranscriptionManager>>();
    if !crate::tray::dictation_in_progress(app) {
        tm.maybe_unload_immediately("speakers");
    }
    match result {
        Ok(Some(mut turns)) => {
            let speakers = renumber(&mut turns);
            let _ = app.emit(
                "speakers-done",
                DoneEvent {
                    entry_id,
                    turns,
                    speakers,
                },
            );
        }
        Ok(None) => {
            info!("Speakers: cancelled");
            let _ = app.emit("speakers-cancelled", CancelledEvent { entry_id });
        }
        Err(_) if cancelled() => {
            let _ = app.emit("speakers-cancelled", CancelledEvent { entry_id });
        }
        Err(e) => {
            warn!("Speakers of entry {entry_id} failed: {e:#}");
            let _ = app.emit(
                "speakers-error",
                ErrorEvent {
                    entry_id,
                    message: format!("{e:#}"),
                },
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

#[derive(Serialize, Debug, Clone, PartialEq, Type)]
pub struct VariantView {
    pub id: String,
    /// f64 for the TypeScript bindings, as elsewhere.
    pub size_bytes: f64,
    pub downloaded: bool,
}

#[derive(Serialize, Debug, Clone, PartialEq, Type)]
pub struct SpeakersState {
    pub options: SpeakersOptions,
    pub variants: Vec<VariantView>,
    pub license: String,
    /// The entry being worked on, if any.
    pub running: Option<i64>,
}

#[tauri::command]
#[specta::specta]
pub fn speakers_state(app: AppHandle) -> Result<SpeakersState, String> {
    let hub = hub(&app)?;
    let running = RUNNING.load(Ordering::SeqCst);
    Ok(SpeakersState {
        options: load_options(&app),
        variants: catalog::VARIANTS
            .iter()
            .map(|v| VariantView {
                id: v.id.into(),
                size_bytes: v.size as f64,
                downloaded: catalog::downloaded(&hub, v).is_some(),
            })
            .collect(),
        license: catalog::VARIANTS[0].license.into(),
        running: (running >= 0).then_some(running),
    })
}

#[tauri::command]
#[specta::specta]
pub fn speakers_set_options(app: AppHandle, options: SpeakersOptions) -> Result<(), String> {
    save_options(&app, &options.sanitized())
}

#[tauri::command]
#[specta::specta]
pub fn speakers_delete_variant(app: AppHandle, id: String) -> Result<(), String> {
    if RUNNING.load(Ordering::SeqCst) >= 0 {
        return Err("Speaker detection is running".into());
    }
    let item = catalog::variant(&id).ok_or("Unknown model")?;
    catalog::delete(&hub(&app)?, item).map_err(|e| e.to_string())
}

/// Starts the job with the saved options; progress and the outcome come as
/// `speakers-progress` / `-done` / `-error` / `-cancelled`.
#[tauri::command]
#[specta::specta]
pub fn speakers_start(app: AppHandle, entry_id: i64) -> Result<(), String> {
    if RUNNING
        .compare_exchange(-1, entry_id, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("Speaker detection is already running".into());
    }
    CANCEL.store(false, Ordering::SeqCst);
    let options = load_options(&app);
    let spawned = std::thread::Builder::new()
        .name("speakers".into())
        .spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run(&app, entry_id, &options)
            }))
            .unwrap_or_else(|_| Err(anyhow!("Speaker detection crashed")));
            RUNNING.store(-1, Ordering::SeqCst);
            finish(&app, entry_id, result);
        });
    if let Err(e) = spawned {
        RUNNING.store(-1, Ordering::SeqCst);
        return Err(format!("Failed to start speaker detection: {e}"));
    }
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn speakers_cancel() {
    if RUNNING.load(Ordering::SeqCst) < 0 {
        return;
    }
    info!("Speakers: cancel requested");
    CANCEL.store(true, Ordering::SeqCst);
    if let Some(token) = DOWNLOAD.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        token.cancel();
    }
    if let Some(token) = DIARIZE.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        token.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_are_sanitized() {
        let options = SpeakersOptions {
            variant: "nope".into(),
            language: " ru ".into(),
            speakers: 9,
        }
        .sanitized();
        assert_eq!(options.variant, catalog::DEFAULT_VARIANT);
        assert_eq!(options.language, "ru");
        assert_eq!(options.speakers, 0);
    }

    #[test]
    fn the_language_reaches_only_the_guarded_thread() {
        let mut settings = crate::settings::get_default_settings();
        settings.selected_language = "auto".into();
        {
            let _guard = LanguageGuard::set("ru");
            let mut here = settings.clone();
            apply_language_override(&mut here);
            assert_eq!(here.selected_language, "ru");

            let elsewhere = std::thread::spawn({
                let mut s = settings.clone();
                move || {
                    apply_language_override(&mut s);
                    s.selected_language
                }
            })
            .join()
            .unwrap();
            assert_eq!(elsewhere, "auto");
        }
        apply_language_override(&mut settings);
        assert_eq!(settings.selected_language, "auto");
        assert!(LanguageGuard::set("").is_none());
    }

    #[test]
    fn speakers_are_renumbered_over_the_turns_with_text() {
        let mut turns = vec![
            TurnText {
                start_ms: 0.0,
                speaker: 2,
                text: "a".into(),
            },
            TurnText {
                start_ms: 1.0,
                speaker: 3,
                text: "b".into(),
            },
            TurnText {
                start_ms: 2.0,
                speaker: 2,
                text: "c".into(),
            },
        ];
        assert_eq!(renumber(&mut turns), 2);
        let speakers: Vec<u32> = turns.iter().map(|t| t.speaker).collect();
        assert_eq!(speakers, [1, 2, 1]);
    }
}
