//! B8: translate a whole text file (a transcript, say) through llama-server and
//! save the result next to the source as `<name>.<lang>.txt`.
//!
//! Own `FILE_BUSY` / `FILE_CANCEL` flags, separate from the selection
//! translation in `action`. While a file runs it is a long job
//! (`hotkeys_pause::job_lock`): dictation, the translate hotkey and double
//! Ctrl show a "wait" popup instead of working, and an audio file cannot be
//! started. User's decision after trying both: llama-server with `-np 1` does
//! queue a selection between two pieces (a live run showed waits of at most
//! 2.5 s), but a long job that nothing can disturb was preferred. A selection
//! translation that started before the file keeps working; its cancel does
//! not stop the server under the file. Pieces are the same as for a selection
//! (`prompt::split_for_translation`), without the 20 000 character limit.

use std::cell::RefCell;
use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{info, warn};
use serde::Serialize;
use specta::Type;
use tauri::{AppHandle, Emitter, Manager};

use super::action::{self, local_provider, translate_piece, CANCELLED};
use super::llama_server::{LlamaServer, ServerState};
use super::popup::ErrorCode;
use super::prompt::{self, Direction, PromptChoice, Segment};
use super::settings;

/// Larger files are refused: this is for text, not for dumps.
const MAX_FILE_BYTES: u64 = 10 * 1024 * 1024;
/// The direction is decided once per file, from its beginning.
const DIRECTION_SAMPLE_CHARS: usize = 5_000;
/// `name (2).txt` … `name (N).txt` are tried before giving up on a free name.
const MAX_COPIES: u32 = 1_000;

const NOT_TEXT: &str = "File is not UTF-8 or UTF-16 text";
const TOO_BIG: &str = "File is larger than 10 MB";
const NOTHING: &str = "Nothing to translate";

/// One file at a time; a second start is an error.
static FILE_BUSY: AtomicBool = AtomicBool::new(false);
/// Set by `translate_file_cancel`: the pipeline sends nothing more, drops the
/// request in flight and saves nothing.
static FILE_CANCEL: AtomicBool = AtomicBool::new(false);

fn file_cancelled() -> bool {
    FILE_CANCEL.load(Ordering::SeqCst)
}

/// Whether a file translation is running (the selection translation keeps
/// llama-server alive for it when it is cancelled itself).
pub fn is_running() -> bool {
    FILE_BUSY.load(Ordering::SeqCst)
}

pub(super) struct BusyGuard;

/// Held while the engine is switched: no file translation starts meanwhile.
/// `None`: one is running.
pub(super) fn hold_running() -> Option<BusyGuard> {
    (!FILE_BUSY.swap(true, Ordering::SeqCst)).then_some(BusyGuard)
}

impl Drop for BusyGuard {
    fn drop(&mut self) {
        FILE_BUSY.store(false, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// Pure parts (unit-tested without a network or Tauri)
// ---------------------------------------------------------------------------

/// Text of a file and whether it used `\r\n`. UTF-8 (with or without BOM) and
/// UTF-16 LE/BE with BOM; anything else (Windows-1251, …) is an error — no
/// charset crates here. Line endings are unified to `\n` (a `\r` inside a
/// paragraph would go to the model).
fn decode_text(bytes: &[u8]) -> Result<(String, bool), String> {
    let utf8 = |b: &[u8]| String::from_utf8(b.to_vec()).map_err(|_| NOT_TEXT.to_string());
    let utf16 = |b: &[u8], unit: fn([u8; 2]) -> u16| -> Result<String, String> {
        let (pairs, odd) = b.as_chunks::<2>();
        if !odd.is_empty() {
            return Err(NOT_TEXT.into());
        }
        let units: Vec<u16> = pairs.iter().map(|&pair| unit(pair)).collect();
        String::from_utf16(&units).map_err(|_| NOT_TEXT.to_string())
    };
    let text = if let Some(rest) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        utf8(rest)?
    } else if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        utf16(rest, u16::from_le_bytes)?
    } else if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        utf16(rest, u16::from_be_bytes)?
    } else {
        utf8(bytes)?
    };
    // UTF-16 without a BOM is "valid" UTF-8 full of NULs: not text for us.
    if text.contains('\0') {
        return Err(NOT_TEXT.into());
    }
    let crlf = text.contains("\r\n");
    Ok((text.replace("\r\n", "\n").replace('\r', "\n"), crlf))
}

/// `\n` back to `\r\n` when the source used it.
fn restore_line_endings(text: &str, crlf: bool) -> String {
    let unified = text.replace("\r\n", "\n");
    if crlf {
        unified.replace('\n', "\r\n")
    } else {
        unified
    }
}

/// `<stem>.<tgt>.txt` next to `src`; if `exists`, `<stem>.<tgt> (2).txt`, `(3)` …
fn output_path(src: &Path, tgt: &str, exists: impl Fn(&Path) -> bool) -> PathBuf {
    let stem = src
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "translation".into());
    let candidate = |n: u32| {
        src.with_file_name(if n <= 1 {
            format!("{stem}.{tgt}.txt")
        } else {
            format!("{stem}.{tgt} ({n}).txt")
        })
    };
    (1..MAX_COPIES)
        .map(candidate)
        .find(|p| !exists(p))
        .unwrap_or_else(|| candidate(MAX_COPIES))
}

/// Remaining time by characters: `elapsed / done_chars × remaining_chars`.
fn eta_secs(elapsed: Duration, done_chars: usize, total_chars: usize) -> Option<u64> {
    if done_chars == 0 {
        return None;
    }
    let remaining = total_chars.saturating_sub(done_chars);
    let secs = elapsed.as_secs_f64() / done_chars as f64 * remaining as f64;
    Some(secs.round() as u64)
}

/// A file split for translation.
#[derive(Debug)]
struct Prepared {
    dir: Direction,
    segments: Vec<Segment>,
    pieces: usize,
    chars: usize,
    /// Characters of the pieces that go to the model (for the ETA).
    piece_chars: usize,
}

fn prepare(text: &str) -> Result<Prepared, String> {
    let sample: String = text.chars().take(DIRECTION_SAMPLE_CHARS).collect();
    let dir = prompt::detect_direction(&sample);
    let segments = prompt::split_for_translation(text);
    let pieces = action::count_translatable(&segments);
    if pieces == 0 {
        return Err(NOTHING.into());
    }
    let piece_chars = segments
        .iter()
        .map(|s| match s {
            Segment::Translate(piece) => piece.chars().count(),
            Segment::Keep(_) => 0,
        })
        .sum();
    Ok(Prepared {
        dir,
        segments,
        pieces,
        chars: text.chars().count(),
        piece_chars,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Progress {
    done: usize,
    total: usize,
    done_chars: usize,
    total_chars: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct FileResult {
    text: String,
    /// Pieces left in the original language (they did not fit the context).
    untranslated: usize,
}

#[derive(Debug, PartialEq, Eq)]
enum FileFailure {
    Cancelled,
    /// 1-based number of the piece that failed twice.
    Piece {
        index: usize,
        total: usize,
        message: String,
    },
}

impl FileFailure {
    fn message(&self) -> String {
        match self {
            FileFailure::Cancelled => CANCELLED.into(),
            FileFailure::Piece {
                index,
                total,
                message,
            } => format!("Piece {index} of {total}: {message}"),
        }
    }
}

/// Translates the pieces one by one.
/// - A piece that does not fit the context stays as it was; the file goes on.
/// - Any other error: `reconnect` (a dead or unloaded server comes back, maybe
///   on a new port) and one more try; a second error ends the file, naming
///   the piece.
/// - Cancel (`is_cancelled`, or the request ending with `CANCELLED`) returns
///   `Cancelled` and nothing else is sent.
fn translate_file_segments(
    segments: &[Segment],
    dir: Direction,
    style: &PromptChoice,
    mut chat: impl FnMut(String) -> Result<Option<String>, String>,
    mut reconnect: impl FnMut() -> Result<(), String>,
    mut progress: impl FnMut(Progress),
    is_cancelled: impl Fn() -> bool,
) -> Result<FileResult, FileFailure> {
    let total = action::count_translatable(segments);
    let total_chars: usize = segments
        .iter()
        .map(|s| match s {
            Segment::Translate(piece) => piece.chars().count(),
            Segment::Keep(_) => 0,
        })
        .sum();
    let mut translations: Vec<String> = Vec::with_capacity(total);
    let mut untranslated = 0;
    let mut done_chars = 0;

    for segment in segments {
        let Segment::Translate(piece) = segment else {
            continue;
        };
        if is_cancelled() {
            return Err(FileFailure::Cancelled);
        }
        let index = translations.len() + 1;
        let stopped = |f: &action::Failure| is_cancelled() || f.detail == CANCELLED;
        let piece_failure = |message: String| {
            if is_cancelled() {
                FileFailure::Cancelled
            } else {
                FileFailure::Piece {
                    index,
                    total,
                    message,
                }
            }
        };

        let mut outcome = translate_piece(piece, dir, style, &mut chat);
        if let Err(first) = &outcome {
            if !stopped(first) && first.code != ErrorCode::TextTooLong {
                warn!(
                    "Translate file: piece {index} of {total} failed ({}), trying once more",
                    first.detail
                );
                reconnect().map_err(piece_failure)?;
                outcome = translate_piece(piece, dir, style, &mut chat);
            }
        }
        match outcome {
            Ok(answer) => translations.push(answer),
            Err(f) if stopped(&f) => return Err(FileFailure::Cancelled),
            Err(f) if f.code == ErrorCode::TextTooLong => {
                warn!(
                    "Translate file: piece {index} of {total} does not fit the context, left as is"
                );
                untranslated += 1;
                translations.push(piece.clone());
            }
            Err(f) => return Err(piece_failure(f.detail)),
        }
        done_chars += piece.chars().count();
        progress(Progress {
            done: index,
            total,
            done_chars,
            total_chars,
        });
    }
    Ok(FileResult {
        text: prompt::assemble(segments, &translations),
        untranslated,
    })
}

/// Writes `text` (UTF-8, no BOM) to the first free `output_path`. Never
/// overwrites: the file is created with `create_new`, so a name taken between
/// the check and the write just moves on to the next one.
fn save_next_to(src: &Path, tgt: &str, text: &str) -> Result<PathBuf, String> {
    let mut taken: Vec<PathBuf> = Vec::new();
    for _ in 0..20 {
        let path = output_path(src, tgt, |p| p.exists() || taken.iter().any(|t| t == p));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                return file
                    .write_all(text.as_bytes())
                    .map(|_| path.clone())
                    .map_err(|e| format!("Failed to write {}: {e}", path.display()));
            }
            Err(e) if e.kind() == ErrorKind::AlreadyExists => taken.push(path),
            Err(e) => return Err(format!("Failed to create {}: {e}", path.display())),
        }
    }
    Err("Could not find a free file name".into())
}

fn read_source(path: &Path) -> Result<(String, bool), String> {
    let meta =
        std::fs::metadata(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    if !meta.is_file() {
        return Err(format!("{} is not a file", path.display()));
    }
    if meta.len() > MAX_FILE_BYTES {
        return Err(TOO_BIG.into());
    }
    let bytes =
        std::fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    decode_text(&bytes)
}

// ---------------------------------------------------------------------------
// Tauri side: commands, events, the worker thread
// ---------------------------------------------------------------------------

#[derive(Serialize, Type)]
pub struct FileProbe {
    chars: u32,
    pieces: u32,
    src: String,
    tgt: String,
}

#[derive(Serialize, Clone)]
struct ProgressEvent {
    /// `starting_model` | `translating`
    phase: &'static str,
    done: usize,
    total: usize,
    done_chars: usize,
    total_chars: usize,
    elapsed_ms: u64,
    eta_secs: Option<u64>,
}

#[derive(Serialize, Clone)]
struct DoneEvent {
    /// Where the result was saved; `None` when `save_error` says why not.
    output_path: Option<String>,
    save_error: Option<String>,
    /// The translation (with the source's line endings): for Copy and Save as.
    text: String,
    pieces: usize,
    untranslated: usize,
    elapsed_secs: f64,
    src: String,
    tgt: String,
}

#[derive(Serialize, Clone)]
struct ErrorEvent {
    message: String,
}

fn emit_error(app: &AppHandle, message: String) {
    warn!("Translate file: {message}");
    let _ = app.emit("translate-file-error", ErrorEvent { message });
}

/// Language pair and size of a file, before the start.
#[tauri::command]
#[specta::specta]
pub async fn translate_file_probe(path: String) -> Result<FileProbe, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let (text, _) = read_source(Path::new(&path))?;
        let prepared = prepare(&text)?;
        Ok(FileProbe {
            chars: u32::try_from(prepared.chars).unwrap_or(u32::MAX),
            pieces: u32::try_from(prepared.pieces).unwrap_or(u32::MAX),
            src: prepared.dir.src.into(),
            tgt: prepared.dir.tgt.into(),
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Starts the translation in the background and returns at once; the answer
/// comes as `translate-file-progress` / `-done` / `-error` / `-cancelled`.
#[tauri::command]
#[specta::specta]
pub fn translate_file_start(app: AppHandle, path: String) -> Result<(), String> {
    if FILE_BUSY.swap(true, Ordering::SeqCst) {
        return Err("A file translation is already running".into());
    }
    // One long job at a time (audio file transcription is the other).
    if crate::hotkeys_pause::job_running(&app) {
        FILE_BUSY.store(false, Ordering::SeqCst);
        return Err("A file is being transcribed: wait for it to finish".into());
    }
    FILE_CANCEL.store(false, Ordering::SeqCst);
    let spawned = std::thread::Builder::new()
        .name("translate-file".into())
        .spawn(move || {
            let _busy = BusyGuard;
            // Dictation and translation hotkeys say "wait" until this ends.
            let _job = crate::hotkeys_pause::job_lock(&app);
            run(&app, PathBuf::from(path));
        });
    if let Err(e) = spawned {
        FILE_BUSY.store(false, Ordering::SeqCst);
        return Err(format!("Failed to start the translation: {e}"));
    }
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn translate_file_cancel(app: AppHandle) -> Result<(), String> {
    if !is_running() {
        return Ok(());
    }
    info!("Translate file: cancel requested");
    FILE_CANCEL.store(true, Ordering::SeqCst);
    // A server start in progress blocks the worker for up to 90 s: end it from
    // here, unless a selection translation waits for the same start.
    let starting = app
        .try_state::<Arc<LlamaServer>>()
        .is_some_and(|server| server.status().state == ServerState::Starting);
    if starting && !action::is_busy() {
        super::stop_server_in_background(&app);
    }
    Ok(())
}

fn run(app: &AppHandle, path: PathBuf) {
    let started = Instant::now();
    let (text, crlf) = match read_source(&path) {
        Ok(read) => read,
        Err(e) => return emit_error(app, e),
    };
    let prepared = match prepare(&text) {
        Ok(prepared) => prepared,
        Err(e) => return emit_error(app, e),
    };
    let Prepared {
        dir,
        segments,
        pieces,
        chars,
        piece_chars,
    } = prepared;
    info!(
        "Translate file: {} — {chars} chars, {} → {}, {pieces} pieces",
        path.display(),
        dir.src,
        dir.tgt
    );

    let emit_progress = |phase: &'static str, p: Progress, elapsed: Duration| {
        let _ = app.emit(
            "translate-file-progress",
            ProgressEvent {
                phase,
                done: p.done,
                total: p.total,
                done_chars: p.done_chars,
                total_chars: p.total_chars,
                elapsed_ms: elapsed.as_millis() as u64,
                eta_secs: eta_secs(elapsed, p.done_chars, p.total_chars),
            },
        );
    };
    let start = Progress {
        done: 0,
        total: pieces,
        done_chars: 0,
        total_chars: piece_chars,
    };

    let server = app.state::<Arc<LlamaServer>>().inner().clone();
    // Held for the whole file: the idle watcher does not stop the server.
    let _use = server.begin_use();
    // The slot is fixed for the whole file (it cannot change under a long job).
    let cfg = settings::load(app);
    if !cfg.setup_ready() {
        return emit_error(app, action::NOT_SET_UP.into());
    }
    let slot = cfg.resolve_slot(cfg.file_slot);
    info!(
        "Translate file: {slot:?} slot ({}, {:?})",
        super::models::display_name(cfg.model_for(slot)),
        cfg.prompt_for(slot).style
    );
    if super::needs_model_start(app, slot) {
        emit_progress("starting_model", start, Duration::ZERO);
    }
    let cancelled_out = |server: &LlamaServer| {
        // A selection translation in progress needs the server; its own cancel
        // is not ours to take.
        if !action::is_busy() {
            server.stop();
        }
        info!(
            "Translate file: cancelled after {} ms",
            started.elapsed().as_millis()
        );
        let _ = app.emit("translate-file-cancelled", ());
    };
    let (port, key) = match super::ensure_server(app, slot) {
        Ok(connection) => connection,
        Err(_) if file_cancelled() => return cancelled_out(&server),
        Err(e) => return emit_error(app, e),
    };
    if file_cancelled() {
        return cancelled_out(&server);
    }

    let provider = RefCell::new(local_provider(port));
    let api_key = RefCell::new(key);
    let work_started = Instant::now();
    emit_progress("translating", start, Duration::ZERO);

    let style = cfg.prompt_for(slot);
    let result = translate_file_segments(
        &segments,
        dir,
        &style,
        |prompt| {
            action::send_cancellable(
                &provider.borrow(),
                &api_key.borrow(),
                prompt,
                file_cancelled,
            )
        },
        || {
            let (port, key) = super::ensure_server(app, slot)?;
            *provider.borrow_mut() = local_provider(port);
            *api_key.borrow_mut() = key;
            Ok(())
        },
        |p| emit_progress("translating", p, work_started.elapsed()),
        file_cancelled,
    );

    let result = match result {
        Ok(result) => result,
        Err(FileFailure::Cancelled) => return cancelled_out(&server),
        Err(failure) => return emit_error(app, failure.message()),
    };

    let out = restore_line_endings(&result.text, crlf);
    let (output_path, save_error) = match save_next_to(&path, dir.tgt, &out) {
        Ok(saved) => (Some(saved.display().to_string()), None),
        Err(e) => {
            warn!("Translate file: {e}");
            (None, Some(e))
        }
    };
    info!(
        "Translate file: done in {:.1} s, {} left untranslated, saved to {}",
        started.elapsed().as_secs_f64(),
        result.untranslated,
        output_path.as_deref().unwrap_or("(not saved)")
    );
    let _ = app.emit(
        "translate-file-done",
        DoneEvent {
            output_path,
            save_error,
            text: out,
            pieces,
            untranslated: result.untranslated,
            elapsed_secs: started.elapsed().as_secs_f64(),
            src: dir.src.into(),
            tgt: dir.tgt.into(),
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    const RU_EN: Direction = Direction {
        src: "ru",
        tgt: "en",
    };

    fn utf16(text: &str, big_endian: bool) -> Vec<u8> {
        let mut bytes = if big_endian {
            vec![0xFE, 0xFF]
        } else {
            vec![0xFF, 0xFE]
        };
        for unit in text.encode_utf16() {
            let pair = if big_endian {
                unit.to_be_bytes()
            } else {
                unit.to_le_bytes()
            };
            bytes.extend_from_slice(&pair);
        }
        bytes
    }

    fn no_reconnect() -> Result<(), String> {
        Ok(())
    }

    #[test]
    fn decodes_utf8_with_and_without_bom() {
        assert_eq!(
            decode_text("Привет\nмир".as_bytes()).unwrap(),
            ("Привет\nмир".into(), false)
        );
        let mut with_bom = vec![0xEF, 0xBB, 0xBF];
        with_bom.extend_from_slice("Привет".as_bytes());
        assert_eq!(decode_text(&with_bom).unwrap(), ("Привет".into(), false));
    }

    #[test]
    fn decodes_utf16_le_and_be_with_bom() {
        let text = "Привет, мир!\r\nВторая строка 😀";
        for big_endian in [false, true] {
            let (decoded, crlf) = decode_text(&utf16(text, big_endian)).unwrap();
            assert_eq!(decoded, "Привет, мир!\nВторая строка 😀");
            assert!(crlf);
        }
    }

    #[test]
    fn rejects_other_encodings_and_binary() {
        // "Привет" in Windows-1251.
        let cp1251 = [0xCF, 0xF0, 0xE8, 0xE2, 0xE5, 0xF2];
        assert_eq!(decode_text(&cp1251).unwrap_err(), NOT_TEXT);
        // UTF-16LE without a BOM is ASCII with NULs.
        assert_eq!(decode_text(b"H\0i\0").unwrap_err(), NOT_TEXT);
        // An odd number of bytes after a UTF-16 BOM, and a lone surrogate.
        assert!(decode_text(&[0xFF, 0xFE, 0x41]).is_err());
        assert!(decode_text(&[0xFF, 0xFE, 0x00, 0xD8]).is_err());
    }

    #[test]
    fn line_endings_go_there_and_back() {
        let (text, crlf) = decode_text(b"one\r\ntwo\r\n\r\nthree\rfour").unwrap();
        assert_eq!(text, "one\ntwo\n\nthree\nfour");
        assert!(crlf);
        assert_eq!(
            restore_line_endings("a\nb\n\nc", crlf),
            "a\r\nb\r\n\r\nc",
            "the source used CRLF"
        );
        let (text, crlf) = decode_text(b"one\ntwo").unwrap();
        assert_eq!((text.as_str(), crlf), ("one\ntwo", false));
        assert_eq!(restore_line_endings("a\r\nb", false), "a\nb");
        // A model answer that already has CRLF is not doubled.
        assert_eq!(restore_line_endings("a\r\nb", true), "a\r\nb");
    }

    #[test]
    fn output_name_is_next_to_the_source_and_never_taken() {
        // Built with `join`, so the test holds on every platform.
        let notes = Path::new("notes");
        let src = notes.join("lecture.txt");
        assert_eq!(
            output_path(&src, "en", |_| false),
            notes.join("lecture.en.txt")
        );
        let taken = [
            notes.join("lecture.en.txt"),
            notes.join("lecture.en (2).txt"),
        ];
        assert_eq!(
            output_path(&src, "en", |p| taken.iter().any(|t| t == p)),
            notes.join("lecture.en (3).txt")
        );
        assert_eq!(
            output_path(Path::new("talk.md"), "ru", |_| false),
            Path::new("talk.ru.txt")
        );
    }

    #[test]
    fn saving_does_not_overwrite_and_writes_utf8_without_bom() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("lecture.txt");
        std::fs::write(&src, "x").unwrap();
        std::fs::write(dir.path().join("lecture.en.txt"), "mine").unwrap();
        let saved = save_next_to(&src, "en", "Hello\r\nмир").unwrap();
        assert_eq!(saved.file_name().unwrap(), "lecture.en (2).txt");
        assert_eq!(std::fs::read(&saved).unwrap(), "Hello\r\nмир".as_bytes());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("lecture.en.txt")).unwrap(),
            "mine"
        );
        let second = save_next_to(&src, "en", "again").unwrap();
        assert_eq!(second.file_name().unwrap(), "lecture.en (3).txt");
    }

    #[test]
    fn eta_is_by_characters() {
        assert_eq!(eta_secs(Duration::from_secs(10), 0, 100), None);
        assert_eq!(eta_secs(Duration::from_secs(10), 100, 400), Some(30));
        assert_eq!(eta_secs(Duration::from_secs(10), 400, 400), Some(0));
        assert_eq!(eta_secs(Duration::from_secs(10), 500, 400), Some(0));
    }

    #[test]
    fn a_file_without_letters_has_nothing_to_translate() {
        assert_eq!(prepare("").unwrap_err(), NOTHING);
        assert_eq!(prepare("2024 — 😀\n\n-----").unwrap_err(), NOTHING);
        let prepared = prepare("Привет.\n\nКак дела?").unwrap();
        assert_eq!((prepared.dir.src, prepared.dir.tgt), ("ru", "en"));
        assert_eq!(prepared.pieces, 2);
    }

    fn big_text() -> String {
        let paragraph = "Сегодня мы разберём, как устроена система, почему она работает именно так и что из этого следует для практики. ";
        (0..90)
            .map(|i| {
                if i % 9 == 4 {
                    format!("- пункт {i} списка;\n- ещё один пункт {i};\n- и третий пункт {i}.")
                } else {
                    format!("{i}. {}", paragraph.repeat(7))
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    #[test]
    fn a_big_file_is_split_without_the_selection_limit() {
        let text = big_text();
        assert!(text.chars().count() > 60_000, "{}", text.chars().count());
        assert!(text.chars().count() > prompt::MAX_SELECTION_CHARS);
        let prepared = prepare(&text).unwrap();
        let sent: String = prepared
            .segments
            .iter()
            .map(|s| match s {
                Segment::Keep(t) | Segment::Translate(t) => t.as_str(),
            })
            .collect();
        assert_eq!(sent, text, "the pieces add up to the whole file");
        assert_eq!(prompt::assemble(&prepared.segments, &[]), text);
        assert_eq!(prepared.chars, text.chars().count());
        assert!(prepared.pieces > 80);
    }

    #[test]
    fn pieces_come_back_in_order_with_progress_by_pieces_and_chars() {
        let segments = prompt::split_for_translation("Раз.\n\n- Два\n- три\n\nЧетыре пять.");
        let mut calls = 0;
        let mut steps = Vec::new();
        let result = translate_file_segments(
            &segments,
            RU_EN,
            &PromptChoice::default(),
            |_| {
                calls += 1;
                Ok(Some(format!("answer {calls}")))
            },
            no_reconnect,
            |p| steps.push(p),
            || false,
        )
        .unwrap();
        assert_eq!(result.text, "answer 1\n\nanswer 2\n\nanswer 3");
        assert_eq!(result.untranslated, 0);
        assert_eq!(
            steps,
            vec![
                Progress {
                    done: 1,
                    total: 3,
                    done_chars: 4,
                    total_chars: 4 + 11 + 12
                },
                Progress {
                    done: 2,
                    total: 3,
                    done_chars: 15,
                    total_chars: 27
                },
                Progress {
                    done: 3,
                    total: 3,
                    done_chars: 27,
                    total_chars: 27
                },
            ]
        );
    }

    #[test]
    fn a_piece_over_the_context_is_left_and_the_file_goes_on() {
        let segments = prompt::split_for_translation("Раз.\n\nДва.\n\nТри.");
        let mut calls = 0;
        let mut retries = 0;
        let result = translate_file_segments(
            &segments,
            RU_EN,
            &PromptChoice::default(),
            |_| {
                calls += 1;
                if calls == 2 {
                    Err(
                        "API request failed with status 400 Bad Request: exceed_context_size_error"
                            .into(),
                    )
                } else {
                    Ok(Some(format!("answer {calls}")))
                }
            },
            || {
                retries += 1;
                Ok(())
            },
            |_| {},
            || false,
        )
        .unwrap();
        assert_eq!(result.text, "answer 1\n\nДва.\n\nanswer 3");
        assert_eq!(result.untranslated, 1);
        assert_eq!(retries, 0, "an overflow is not retried");
    }

    #[test]
    fn a_failed_piece_is_tried_once_more_after_reconnecting() {
        let segments = prompt::split_for_translation("Раз.\n\nДва.");
        let mut calls = 0;
        let reconnects = Cell::new(0);
        let result = translate_file_segments(
            &segments,
            RU_EN,
            &PromptChoice::default(),
            |_| {
                calls += 1;
                if calls == 1 {
                    Err("connection refused".into())
                } else {
                    Ok(Some(format!("answer {calls}")))
                }
            },
            || {
                reconnects.set(reconnects.get() + 1);
                Ok(())
            },
            |_| {},
            || false,
        )
        .unwrap();
        assert_eq!(result.text, "answer 2\n\nanswer 3");
        assert_eq!(reconnects.get(), 1);
    }

    #[test]
    fn a_piece_that_fails_twice_ends_the_file_naming_the_piece() {
        let segments = prompt::split_for_translation("Раз.\n\nДва.\n\nТри.");
        let mut calls = 0;
        let err = translate_file_segments(
            &segments,
            RU_EN,
            &PromptChoice::default(),
            |_| {
                calls += 1;
                if calls >= 2 {
                    Err("connection refused".into())
                } else {
                    Ok(Some("ok".into()))
                }
            },
            no_reconnect,
            |_| {},
            || false,
        )
        .unwrap_err();
        assert_eq!(calls, 3, "one try, the retry, nothing after");
        assert_eq!(
            err,
            FileFailure::Piece {
                index: 2,
                total: 3,
                message: "connection refused".into()
            }
        );
        assert_eq!(err.message(), "Piece 2 of 3: connection refused");

        // The server does not come back: the reconnect error is the reason.
        let err = translate_file_segments(
            &segments,
            RU_EN,
            &PromptChoice::default(),
            |_| Err("down".into()),
            || Err("llama-server not found".into()),
            |_| {},
            || false,
        )
        .unwrap_err();
        assert_eq!(err.message(), "Piece 1 of 3: llama-server not found");
    }

    #[test]
    fn an_empty_answer_is_a_piece_error() {
        let segments = prompt::split_for_translation("Раз.");
        let mut calls = 0;
        let err = translate_file_segments(
            &segments,
            RU_EN,
            &PromptChoice::default(),
            |_| {
                calls += 1;
                Ok(Some("  ".into()))
            },
            no_reconnect,
            |_| {},
            || false,
        )
        .unwrap_err();
        assert_eq!(calls, 2, "retried once");
        assert!(matches!(err, FileFailure::Piece { index: 1, .. }));
    }

    #[test]
    fn cancel_in_the_middle_sends_nothing_more_and_returns_nothing() {
        let segments = prompt::split_for_translation("Раз.\n\nДва.\n\nТри.\n\nЧетыре.");
        let cancel = Cell::new(false);
        let mut calls = 0;
        let err = translate_file_segments(
            &segments,
            RU_EN,
            &PromptChoice::default(),
            |_| {
                calls += 1;
                if calls == 2 {
                    cancel.set(true);
                }
                Ok(Some("x".into()))
            },
            no_reconnect,
            |_| {},
            || cancel.get(),
        )
        .unwrap_err();
        assert_eq!(err, FileFailure::Cancelled);
        assert_eq!(calls, 2, "no request after the cancel");
    }

    #[test]
    fn a_request_stopped_on_purpose_is_a_cancel_not_a_retry() {
        let segments = prompt::split_for_translation("Раз.\n\nДва.");
        let mut calls = 0;
        let reconnects = Cell::new(0);
        let err = translate_file_segments(
            &segments,
            RU_EN,
            &PromptChoice::default(),
            |_| {
                calls += 1;
                Err(CANCELLED.into())
            },
            || {
                reconnects.set(reconnects.get() + 1);
                Ok(())
            },
            |_| {},
            || false,
        )
        .unwrap_err();
        assert_eq!(err, FileFailure::Cancelled);
        assert_eq!((calls, reconnects.get()), (1, 0));
    }

    /// Live run on the GPU: a 60–80 thousand character transcript with lists
    /// and paragraphs, through the real server and `send_cancellable`. Paths:
    /// settings defaults or `SELTRANSLATE_LLAMA_EXE` / `SELTRANSLATE_MODEL`.
    #[test]
    #[ignore]
    fn real_server_translates_a_big_file() {
        use super::super::settings::SelTranslateSettings;

        let mut cfg = SelTranslateSettings::default();
        if let Ok(exe) = std::env::var("SELTRANSLATE_LLAMA_EXE") {
            cfg.llama_server_path = exe;
        }
        if let Ok(model) = std::env::var("SELTRANSLATE_MODEL") {
            cfg.model_path = model;
        }
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("transcript.txt");
        let text = big_text();
        std::fs::write(&source, text.replace('\n', "\r\n")).unwrap();

        let (text, crlf) = read_source(&source).unwrap();
        assert!(crlf);
        let prepared = prepare(&text).unwrap();
        println!(
            "{} chars, {} pieces, {} → {}",
            prepared.chars, prepared.pieces, prepared.dir.src, prepared.dir.tgt
        );

        let server = LlamaServer::default();
        let (port, key) = server
            .ensure_running(
                &cfg.server_config(settings::Slot::Quality, dir.path().join("llama-server.log")),
            )
            .expect("server start");
        let provider = local_provider(port);
        let begun = Instant::now();
        let last = Cell::new(0);
        let file_done = std::sync::atomic::AtomicBool::new(false);
        // A second client, like the hotkey during a file: llama-server runs
        // with `-np 1` and has to queue its requests between the file's pieces.
        let (result, hotkey_waits) = std::thread::scope(|scope| {
            let hotkey = scope.spawn(|| {
                let mut waits = Vec::new();
                std::thread::sleep(Duration::from_secs(10));
                while !file_done.load(Ordering::SeqCst) {
                    let t = Instant::now();
                    let answer = action::send_cancellable(
                        &provider,
                        &key,
                        prompt::render_prompt("ru", "en", "Привет, мир"),
                        || false,
                    );
                    waits.push((t.elapsed(), answer));
                    std::thread::sleep(Duration::from_secs(15));
                }
                waits
            });
            let result = translate_file_segments(
                &prepared.segments,
                prepared.dir,
                &PromptChoice::default(),
                |p| action::send_cancellable(&provider, &key, p, || false),
                || Ok(()),
                |p| {
                    if p.done % 10 == 0 {
                        println!(
                            "{}/{} pieces, {}/{} chars, {:?} elapsed, eta {:?}",
                            p.done,
                            p.total,
                            p.done_chars,
                            p.total_chars,
                            begun.elapsed(),
                            eta_secs(begun.elapsed(), p.done_chars, p.total_chars)
                        );
                    }
                    last.set(p.done);
                },
                || false,
            )
            .expect("translation");
            file_done.store(true, Ordering::SeqCst);
            (result, hotkey.join().unwrap())
        });
        for (wait, answer) in &hotkey_waits {
            println!("hotkey request during the file: {wait:?} -> {answer:?}");
        }
        assert!(!hotkey_waits.is_empty(), "the file ran long enough");
        assert!(
            hotkey_waits
                .iter()
                .all(|(wait, answer)| answer.is_ok() && *wait < Duration::from_secs(10)),
            "a selection waits between pieces, not for the whole file"
        );
        println!(
            "done in {:.1} s, {} untranslated",
            begun.elapsed().as_secs_f64(),
            result.untranslated
        );
        server.stop();

        let out = restore_line_endings(&result.text, crlf);
        let saved = save_next_to(&source, prepared.dir.tgt, &out).unwrap();
        println!("saved to {}", saved.display());
        assert_eq!(last.get(), prepared.pieces);
        assert_eq!(result.untranslated, 0);
        assert_eq!(
            out.matches("\r\n\r\n").count(),
            text.matches("\n\n").count(),
            "paragraphs kept"
        );
        assert_eq!(
            out.lines().filter(|l| l.starts_with("- ")).count(),
            text.lines().filter(|l| l.starts_with("- ")).count(),
            "list markers kept"
        );
        let numbered = |t: &str| {
            t.lines()
                .filter(|l| {
                    let digits = l.bytes().take_while(u8::is_ascii_digit).count();
                    digits > 0 && l[digits..].starts_with(". ")
                })
                .count()
        };
        println!(
            "numbered paragraphs: source {}, result {}",
            numbered(&text),
            numbered(&out)
        );
        let head: String = out.chars().take(600).collect();
        println!("--- head ---\n{head}");
    }

    /// B9: an instruction-like enumeration used to come back as a chat answer
    /// ("Okay, here's the status update" and bullets). Through the real server.
    #[test]
    #[ignore]
    fn real_server_translates_an_enumeration_instead_of_answering_it() {
        use super::super::settings::SelTranslateSettings;

        let mut cfg = SelTranslateSettings::default();
        if let Ok(exe) = std::env::var("SELTRANSLATE_LLAMA_EXE") {
            cfg.llama_server_path = exe;
        }
        if let Ok(model) = std::env::var("SELTRANSLATE_MODEL") {
            cfg.model_path = model;
        }
        let dir = tempfile::tempdir().unwrap();
        let text = "Готово, когда (любой пункт, явно отложенный мной вопросом да или нет, считается \
выполненным): в DESIGN.md есть раздел F с потоком, списком файлов, событиями, интерфейсом карточки, \
таблицей URL, размеров и SHA-256 движка и моделей, выбором модели по видеопамяти, безопасностью, \
рисками и шагами; написан TASK_F2.md для Sonnet; подготовлен новый NEXT_GOAL.md под F2 (обычный \
текст до 4000 знаков, без форматирования); обновлена таблица моделей в PLAN.md; итог записан в \
PROGRESS.md и INSTALLED.md; TASK_F1.md удалён; temp прибран; в ответе итог, принятые решения и что \
осталось на потом.\n\nОграничения: ничего не скачивать и не ставить без моего подтверждения; \
метаданные релизов и страниц читать можно, архивы, exe и модели не скачивать. Новых крейтов нет. \
Вызывать get_usage в начале и после каждого пункта; при 80 процентах лимита дописать PROGRESS.md \
и остановиться на границе пункта. В начале и в конце напомнить модель: F1 Opus, F2 и F3 Sonnet.";
        let prepared = prepare(text).unwrap();
        println!("{} chars, {} pieces", prepared.chars, prepared.pieces);

        let server = LlamaServer::default();
        let (port, key) = server
            .ensure_running(
                &cfg.server_config(settings::Slot::Quality, dir.path().join("llama-server.log")),
            )
            .expect("server start");
        let provider = local_provider(port);
        let result = translate_file_segments(
            &prepared.segments,
            prepared.dir,
            &PromptChoice::default(),
            |p| action::send_cancellable(&provider, &key, p, || false),
            || Ok(()),
            |_| {},
            || false,
        )
        .expect("translation");
        server.stop();

        println!("--- result ---\n{}", result.text);
        assert_eq!(result.untranslated, 0);
        assert!(!result.text.contains("Okay"), "no chat opener");
        assert!(!result.text.contains("*   "), "no bullets");
        assert!(!result.text.contains('`'), "no backticks");
        assert!(result.text.contains("TASK_F2.md"));
        assert_eq!(result.text.matches("\n\n").count(), 1, "two paragraphs");
    }
}
