//! `TranslateAction` (B5): the `translate_selection` hotkey. Captures the
//! selection, translates it paragraph by paragraph through llama-server and
//! shows the result in the popup or puts it over the selection (Replace).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use log::{info, warn};
use tauri::{AppHandle, Manager};

use super::cache;
use super::llama_server::{base_url_for_port, LlamaServer, ServerState};
use super::popup::{self, ErrorCode, PopupView};
use super::prompt::{self, Direction, PromptChoice, Segment};
use super::selection::{self, CaptureError, ClipboardOutcome};
use super::settings::{self, OutputMode, Slot};
use crate::actions::ShortcutAction;
use crate::settings::PostProcessProvider;

/// llama-server ignores the model name; any non-empty string will do.
pub(super) const MODEL: &str = "translategemma";

/// One translation at a time: a second press while it runs is ignored.
static BUSY: AtomicBool = AtomicBool::new(false);

/// Set by the popup's Cancel / x / Esc while a translation works: the pipeline
/// sends nothing more, drops the request in flight and shows nothing.
static CANCEL: AtomicBool = AtomicBool::new(false);

/// How often a waiting request looks at `CANCEL`.
const CANCEL_POLL: Duration = Duration::from_millis(50);
/// The error text of a request that was stopped on purpose.
pub(super) const CANCELLED: &str = "Cancelled";

fn cancelled() -> bool {
    CANCEL.load(Ordering::SeqCst)
}

/// Whether a selection translation is running (the file translation looks at
/// it before it stops llama-server on its own cancel).
pub fn is_busy() -> bool {
    BUSY.load(Ordering::SeqCst)
}

/// Stops the translation that is running, if any. The pipeline then stops
/// llama-server itself (see `run`); a server start in progress, which blocks
/// the pipeline for up to 90 s, is ended from here, unless a file translation
/// waits for the same start.
pub fn request_cancel(app: &AppHandle) {
    if !BUSY.load(Ordering::SeqCst) {
        return;
    }
    info!("Translate: cancel requested");
    CANCEL.store(true, Ordering::SeqCst);
    let starting = app
        .try_state::<Arc<LlamaServer>>()
        .is_some_and(|server| server.status().state == ServerState::Starting);
    if starting && !super::file::is_running() {
        super::stop_server_in_background(app);
    }
}

/// Set while "Better translation" works: Cancel / x / Esc then bring the Fast
/// result back instead of closing the popup (G3).
static IMPROVING: AtomicBool = AtomicBool::new(false);

/// Whether the popup's Working view belongs to "Better translation".
pub fn is_improving() -> bool {
    IMPROVING.load(Ordering::SeqCst)
}

pub(super) struct BusyGuard;

/// Held while the engine is switched: no translation starts meanwhile (a
/// hotkey pressed in that time is ignored). `None`: one is running.
pub(super) fn hold_busy() -> Option<BusyGuard> {
    (!BUSY.swap(true, Ordering::SeqCst)).then_some(BusyGuard)
}

impl Drop for BusyGuard {
    fn drop(&mut self) {
        IMPROVING.store(false, Ordering::SeqCst);
        BUSY.store(false, Ordering::SeqCst);
    }
}

/// Where the model of a translation comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelChoice {
    /// The "Model for the hotkey" setting (Fast, Quality or Auto).
    FromSettings,
    /// The second hotkey: always the Quality model.
    Quality,
}

pub struct TranslateAction;

impl ShortcutAction for TranslateAction {
    fn start(&self, app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        // Right away, while the hotkey's Alt is still held (DESIGN B.4a).
        selection::mask_menu_activation(app);
        trigger(app);
    }

    fn stop(&self, _app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {}
}

/// The `translate_selection_quality` hotkey: the same, with the Quality model.
pub struct TranslateQualityAction;

impl ShortcutAction for TranslateQualityAction {
    fn start(&self, app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        selection::mask_menu_activation(app);
        trigger_with(app, ModelChoice::Quality);
    }

    fn stop(&self, _app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {}
}

/// Starts a translation of the current selection: the hotkey's `start`, minus
/// the menu mask (only an Alt hotkey needs it), and double Ctrl. A second
/// call while one runs is ignored.
pub fn trigger(app: &AppHandle) {
    trigger_with(app, ModelChoice::FromSettings);
}

pub fn trigger_with(app: &AppHandle, choice: ModelChoice) {
    // A long job (audio file, file translation) runs for minutes and nothing
    // may run next to it (decided with the user, B8).
    if crate::hotkeys_pause::job_running(app) {
        info!("Translate: a long job is running, trigger refused");
        show_busy(app);
        return;
    }
    if BUSY.swap(true, Ordering::SeqCst) {
        info!("Translate: already running, trigger ignored");
        return;
    }
    CANCEL.store(false, Ordering::SeqCst);
    let app = app.clone();
    let spawned = std::thread::Builder::new()
        .name("translate".into())
        .spawn(move || {
            let _busy = BusyGuard;
            run(&app, choice);
        });
    if let Err(e) = spawned {
        BUSY.store(false, Ordering::SeqCst);
        warn!("Translate: failed to start the thread: {e}");
    }
}

/// "A long job is running, wait": the popup for a hotkey pressed meanwhile
/// (dictation, translation, double Ctrl). Off the caller's thread: opening the
/// window waits for the UI thread.
pub fn show_busy(app: &AppHandle) {
    let app = app.clone();
    std::thread::spawn(move || {
        show_failure(
            &app,
            Failure::new(ErrorCode::JobRunning, "A long job is running"),
            0,
        )
    });
}

/// Why a translation did not produce text; shown in the popup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Failure {
    pub(super) code: ErrorCode,
    pub(super) detail: String,
}

impl Failure {
    pub(super) fn new(code: ErrorCode, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

impl From<CaptureError> for Failure {
    fn from(e: CaptureError) -> Self {
        let code = match e {
            CaptureError::NoSelection => ErrorCode::NoSelection,
            CaptureError::ClipboardBusy => ErrorCode::ClipboardBusy,
            CaptureError::NoAccessibility => ErrorCode::NoAccessibility,
            CaptureError::Other(_) => ErrorCode::Other,
        };
        Failure::new(code, e.to_string())
    }
}

fn show_failure(app: &AppHandle, failure: Failure, hwnd: isize) {
    warn!("Translate: {:?}: {}", failure.code, failure.detail);
    popup::open(
        app,
        PopupView::Error {
            code: failure.code,
            detail: failure.detail,
        },
        hwnd,
    );
}

/// What a translation of one captured text needs. "Better translation" runs
/// the same job again with the Quality slot.
#[derive(Clone)]
struct Job {
    text: String,
    truncated: bool,
    clipboard_lost: bool,
    hwnd: isize,
    mode: OutputMode,
    slot: Slot,
    /// "Translate again": ask the model even when the cache has this text.
    fresh: bool,
    /// Source language picked in the popup; `None`: detected from the text.
    src: Option<&'static str>,
}

/// The last translation shown in the popup: its job for "Better translation"
/// and "Translate again", and the view to bring back when that is cancelled
/// or fails. Stays until a new translation starts or a new result replaces it.
#[derive(Clone)]
struct Retry {
    job: Job,
    shown: PopupView,
}

static RETRY: Mutex<Option<Retry>> = Mutex::new(None);

fn set_retry(retry: Option<Retry>) {
    *RETRY.lock().unwrap_or_else(|e| e.into_inner()) = retry;
}

/// The shown result again after "Better translation" or "Translate again" was
/// cancelled (`failed` = false) or failed (`failed` = true, the page adds a
/// short note).
fn fast_view_again(shown: &PopupView, failed: bool) -> PopupView {
    let mut view = shown.clone();
    if let PopupView::Done { improve_failed, .. } = &mut view {
        *improve_failed = failed;
    }
    view
}

/// "Better translation" in the popup: the same text through the Quality slot,
/// the result replaces the one shown. Returns at once; the work runs in a thread.
pub fn improve(app: &AppHandle) -> Result<(), String> {
    rerun(app, Rerun::Better)
}

/// "Translate again" in the popup, offered under a result that came from the
/// cache: the same text through the same model, asked afresh; the new answer
/// replaces the one shown and the cache.
pub fn translate_again(app: &AppHandle) -> Result<(), String> {
    rerun(app, Rerun::Again)
}

/// The source language picked in the popup when the detected one was wrong:
/// the same text through the same model with the picked language.
pub fn change_source(app: &AppHandle, code: &str) -> Result<(), String> {
    let (code, _) = prompt::source_languages()
        .into_iter()
        .find(|(c, _)| *c == code)
        .ok_or("Unknown language")?;
    rerun(app, Rerun::Source(code))
}

#[derive(Clone, Copy)]
enum Rerun {
    Better,
    Again,
    Source(&'static str),
}

fn rerun(app: &AppHandle, what: Rerun) -> Result<(), String> {
    if crate::hotkeys_pause::job_running(app) {
        return Err("A long job is running".into());
    }
    if BUSY.swap(true, Ordering::SeqCst) {
        return Err("A translation is already running".into());
    }
    // Left in place: Cancel or an error brings the Fast result back.
    let retry = RETRY.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let Some(Retry { mut job, shown }) = retry else {
        BUSY.store(false, Ordering::SeqCst);
        return Err("No translation to run again".into());
    };
    match what {
        Rerun::Better => {
            job.slot = Slot::Quality;
            job.fresh = false;
        }
        Rerun::Again => job.fresh = true,
        Rerun::Source(code) => {
            job.src = Some(code);
            job.fresh = false;
        }
    }
    CANCEL.store(false, Ordering::SeqCst);
    IMPROVING.store(true, Ordering::SeqCst);
    let app = app.clone();
    let spawned = std::thread::Builder::new()
        .name("translate-improve".into())
        .spawn(move || {
            let _busy = BusyGuard;
            translate_job(&app, job, Some(shown));
        });
    if let Err(e) = spawned {
        IMPROVING.store(false, Ordering::SeqCst);
        BUSY.store(false, Ordering::SeqCst);
        return Err(format!("Failed to start the translation: {e}"));
    }
    Ok(())
}

/// Text of the "not set up" failure; `file.rs` reports it too.
pub(super) const NOT_SET_UP: &str = "Translation is not set up yet";

/// A hotkey before the setup has nothing to run: the popup sends the user to
/// the Translation settings (F.6). Checked before the selection is copied.
fn not_set_up(cfg: &settings::SelTranslateSettings) -> Option<Failure> {
    (!cfg.setup_ready()).then(|| Failure::new(ErrorCode::NotSetUp, NOT_SET_UP))
}

fn run(app: &AppHandle, choice: ModelChoice) {
    set_retry(None);
    let cfg = settings::load(app);
    if let Some(failure) = not_set_up(&cfg) {
        return show_failure(app, failure, 0);
    }
    let report = selection::capture_selection_report(app);
    let clipboard_lost = report.clipboard == ClipboardOutcome::RestoreFailed;
    let captured = match report.result {
        Ok(captured) => captured,
        Err(e) => return show_failure(app, e.into(), 0),
    };
    let (text, truncated) = prompt::truncate_selection(&captured.text);
    let slot = slot_for(choice, &cfg, &text, super::quality_loaded(app));
    let job = Job {
        text,
        truncated,
        clipboard_lost,
        hwnd: captured.hwnd,
        mode: cfg.output_mode,
        slot,
        fresh: false,
        src: None,
    };
    translate_job(app, job, None);
}

/// The slot for a captured `text`: the second hotkey is always Quality.
/// Already resolved (a missing model falls back), so `Working` names the
/// model that really runs.
fn slot_for(
    choice: ModelChoice,
    cfg: &settings::SelTranslateSettings,
    text: &str,
    quality_loaded: bool,
) -> Slot {
    match choice {
        ModelChoice::FromSettings => cfg.slot_for_selection(text, quality_loaded),
        ModelChoice::Quality => cfg.resolve_slot(Slot::Quality),
    }
}

/// Translates `job.text` with `job.slot` and shows the result. `fast_shown`:
/// the popup is on screen already with this Fast result (Better translation),
/// so its view is updated, and this result comes back if the work is cancelled
/// or fails.
fn translate_job(app: &AppHandle, job: Job, fast_shown: Option<PopupView>) {
    let popup_open = fast_shown.is_some();
    let started = Instant::now();
    let Job {
        text,
        truncated,
        clipboard_lost,
        hwnd,
        mode,
        slot,
        fresh,
        src,
    } = job.clone();
    let dir = src
        .and_then(prompt::direction_from_source)
        .unwrap_or_else(|| prompt::detect_direction(&text));
    let segments = prompt::split_for_translation(&text);
    let total = count_translatable(&segments);
    let cfg = settings::load(app);
    let model = super::models::display_name(cfg.model_for(slot));
    let style = cfg.prompt_for(slot);
    info!(
        "Translate: {} chars{}, {} → {}, {total} pieces, {mode:?}, {slot:?} ({model}, {:?})",
        text.chars().count(),
        if truncated { " (truncated)" } else { "" },
        dir.src,
        dir.tgt,
        style.style
    );

    // The same text through the same model and style is answered from memory
    // (no server start, no waiting); "Translate again" skips the lookup.
    let cache_key = cache::key(&text, cfg.model_for(slot), &format!("{style:?}"), dir.src);
    let cached = if fresh || total == 0 {
        None
    } else {
        cache::lookup(&cache_key)
    };
    let from_cache = cached.is_some();
    if from_cache {
        info!("Translate: answered from the cache");
    }
    let server = app.state::<Arc<LlamaServer>>().inner().clone();
    let _use = (!from_cache).then(|| server.begin_use());
    let starting_model = total > 0 && super::needs_model_start(app, slot);
    let working = |done: usize, starting_model: bool| PopupView::Working {
        src: dir.src.into(),
        tgt: dir.tgt.into(),
        slot,
        model: model.clone(),
        starting_model,
        done: done as u32,
        total: total as u32,
    };
    if !from_cache {
        match mode {
            OutputMode::Popup if popup_open => popup::update(app, working(0, starting_model)),
            OutputMode::Popup => popup::open(app, working(0, starting_model), hwnd),
            OutputMode::Replace => crate::overlay::show_processing_overlay(app),
        }
    }

    // Text without letters (numbers, dashes, emoji) needs no model.
    let result = if let Some(translation) = cached {
        Ok(translation)
    } else if total == 0 {
        Ok(prompt::assemble(&segments, &[]))
    } else {
        super::ensure_server(app, slot)
            .map_err(|e| Failure::new(ErrorCode::ServerFailed, e))
            .and_then(|(port, api_key)| {
                let provider = local_provider(port);
                translate_segments(
                    &segments,
                    dir,
                    &style,
                    |prompt| send_cancellable(&provider, &api_key, prompt, cancelled),
                    |done| {
                        if mode == OutputMode::Popup {
                            popup::update(app, working(done, false));
                        }
                    },
                )
            })
    };

    if mode == OutputMode::Replace {
        crate::overlay::hide_recording_overlay(app);
    }
    if cancelled() {
        // Dropping the connection does not make llama-server stop generating
        // (checked on b11623: a request cut after 1.5 s kept the GPU busy until
        // `-n`), so the process is stopped; the next translation starts it
        // again (~3 s). Done before this thread frees `BUSY`, so a new start
        // cannot be caught by the stop. A file translation in progress keeps
        // the server: its own cancel stops it.
        if total > 0 && !super::file::is_running() {
            server.stop();
        }
        info!(
            "Translate: cancelled after {} ms",
            started.elapsed().as_millis()
        );
        // Cancel of "Better translation": the popup stays with the Fast result.
        if let Some(shown) = &fast_shown {
            popup::update(app, fast_view_again(shown, false));
        }
        return;
    }
    // The popup names the model that failed (a GGUF picked with Browse may
    // not start or not answer).
    let result = result.map_err(|f| Failure {
        detail: format!("{model}: {}", f.detail),
        ..f
    });
    let translation = match result {
        Ok(translation) => translation,
        Err(failure) => {
            return match &fast_shown {
                Some(shown) => {
                    warn!("Translate: {:?}: {}", failure.code, failure.detail);
                    popup::update(app, fast_view_again(shown, true));
                }
                None => show_failure(app, failure, hwnd),
            };
        }
    };
    info!("Translate: done in {} ms", started.elapsed().as_millis());
    if total > 0 && !from_cache {
        cache::remember(cache_key, translation.clone());
    }

    // Replace never drops text: a truncated selection goes to the popup instead.
    if mode == OutputMode::Replace && !truncated {
        if let Err(e) = selection::paste_text(app, &translation) {
            show_failure(app, Failure::new(ErrorCode::ReplaceFailed, e), hwnd);
        }
        return;
    }
    let can_improve = can_improve(mode, slot, settings::load(app).quality_differs_from_fast());
    let done = PopupView::Done {
        src: dir.src.into(),
        tgt: dir.tgt.into(),
        slot,
        model,
        text: translation,
        truncated,
        clipboard_lost,
        can_improve,
        improve_failed: false,
        from_cache,
    };
    set_retry(Some(Retry {
        job: Job {
            fresh: false,
            ..job
        },
        shown: done.clone(),
    }));
    match mode {
        // A cache hit never showed the Working view: the popup opens now.
        OutputMode::Popup if popup_open || !from_cache => popup::update(app, done),
        OutputMode::Popup | OutputMode::Replace => popup::open(app, done, hwnd),
    }
}

/// "Better translation" is offered for a Fast result shown in the popup when
/// Quality can take over with another model file (never in Replace mode,
/// never for a Quality result).
fn can_improve(mode: OutputMode, slot: Slot, quality_differs: bool) -> bool {
    mode == OutputMode::Popup && slot == Slot::Fast && quality_differs
}

/// Waits for the answer on `rx`, looking at `is_cancelled` every `CANCEL_POLL`;
/// on cancel `abort` drops the request in flight (closing the connection makes
/// llama-server stop generating) and the wait ends with `CANCELLED`.
fn wait_or_cancel<T>(
    rx: &mpsc::Receiver<T>,
    is_cancelled: impl Fn() -> bool,
    abort: impl FnOnce(),
) -> Result<T, String> {
    loop {
        match rx.recv_timeout(CANCEL_POLL) {
            Ok(answer) => return Ok(answer),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if is_cancelled() {
                    abort();
                    return Err(CANCELLED.into());
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err("The request task ended without an answer".into());
            }
        }
    }
}

/// One chat request that `is_cancelled` can stop in the middle (the selection
/// translation passes `request_cancel`'s flag, the file translation its own).
pub(super) fn send_cancellable(
    provider: &PostProcessProvider,
    api_key: &str,
    prompt: String,
    is_cancelled: fn() -> bool,
) -> Result<Option<String>, String> {
    if is_cancelled() {
        return Err(CANCELLED.into());
    }
    let (tx, rx) = mpsc::channel();
    let provider = provider.clone();
    let api_key = api_key.to_string();
    let task = tauri::async_runtime::spawn(async move {
        let answer =
            crate::llm_client::send_chat_completion(&provider, api_key, MODEL, prompt, false).await;
        let _ = tx.send(answer);
    });
    wait_or_cancel(&rx, is_cancelled, || task.abort())?
}

pub(super) fn local_provider(port: u16) -> PostProcessProvider {
    PostProcessProvider {
        id: "seltranslate-local".into(),
        label: "Local".into(),
        base_url: base_url_for_port(port),
        allow_base_url_edit: false,
        models_endpoint: None,
        supports_structured_output: false,
    }
}

pub(super) fn count_translatable(segments: &[Segment]) -> usize {
    segments
        .iter()
        .filter(|s| matches!(s, Segment::Translate(_)))
        .count()
}

/// Translates one piece through `chat`: the prompt, the model's answer cleaned
/// and matched to the piece's edges and list markers. When the answer cannot be
/// a translation (`prompt::answer_problem`) the piece is asked once more with
/// the strict prompt and the second answer wins if it is fine (B9). An empty
/// answer is a failure, so is a request error (`TextTooLong` when the piece
/// does not fit the server's context).
pub(super) fn translate_piece(
    piece: &str,
    dir: Direction,
    style: &PromptChoice,
    chat: &mut impl FnMut(String) -> Result<Option<String>, String>,
) -> Result<String, Failure> {
    let mut ask = |prompt_text: String| -> Result<String, Failure> {
        let answer = chat(prompt_text).map_err(|e| {
            if prompt::is_context_overflow(&e) {
                Failure::new(ErrorCode::TextTooLong, e)
            } else {
                Failure::new(ErrorCode::TranslationFailed, e)
            }
        })?;
        let answer = prompt::match_source_edges(
            piece,
            &prompt::clean_output(answer.as_deref().unwrap_or("")),
        );
        let answer = prompt::drop_added_backticks(piece, &answer);
        Ok(prompt::restore_list_markers(piece, &answer))
    };
    let mut answer = ask(prompt::render_styled(style, dir.src, dir.tgt, piece, false))?;
    // An empty answer is a failure below (the file translation retries those).
    let problem = prompt::answer_problem(piece, &answer).filter(|_| !answer.is_empty());
    if let Some(problem) = problem {
        warn!("Translation looks wrong ({problem}); asking again with the strict prompt");
        let retry = ask(prompt::render_styled(style, dir.src, dir.tgt, piece, true))?;
        match prompt::answer_problem(piece, &retry) {
            None => answer = retry,
            Some(problem) => warn!("The second answer is no better ({problem}); keeping the first"),
        }
    }
    if answer.is_empty() {
        return Err(Failure::new(
            ErrorCode::TranslationFailed,
            "The model returned an empty answer",
        ));
    }
    Ok(answer)
}

/// Sends every `Translate` piece through `chat` in order and puts the answers
/// back between the kept separators. `progress(n)` after the n-th piece.
fn translate_segments(
    segments: &[Segment],
    dir: Direction,
    style: &PromptChoice,
    mut chat: impl FnMut(String) -> Result<Option<String>, String>,
    mut progress: impl FnMut(usize),
) -> Result<String, Failure> {
    let mut translations = Vec::new();
    for segment in segments {
        let Segment::Translate(piece) = segment else {
            continue;
        };
        translations.push(translate_piece(piece, dir, style, &mut chat)?);
        progress(translations.len());
    }
    Ok(prompt::assemble(segments, &translations))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RU_EN: Direction = Direction {
        src: "ru",
        tgt: "en",
    };

    const STANDARD: PromptChoice = PromptChoice {
        style: prompt::PromptStyle::Standard,
        custom: String::new(),
    };

    #[test]
    fn a_hotkey_before_the_setup_says_not_set_up() {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("llama-server.exe");
        let model = dir.path().join("m.gguf");
        std::fs::write(&exe, b"x").unwrap();
        std::fs::write(&model, b"x").unwrap();
        let ready = settings::SelTranslateSettings {
            llama_server_path: exe.display().to_string(),
            model_path: model.display().to_string(),
            fast_model_path: String::new(),
            ..Default::default()
        };
        assert_eq!(not_set_up(&ready), None);
        let no_models = settings::SelTranslateSettings {
            model_path: String::new(),
            ..ready.clone()
        };
        assert_eq!(
            not_set_up(&no_models),
            Some(Failure::new(ErrorCode::NotSetUp, NOT_SET_UP))
        );
        let no_engine = settings::SelTranslateSettings {
            llama_server_path: String::new(),
            ..ready
        };
        assert!(not_set_up(&no_engine).is_some());
    }

    #[test]
    fn a_wrong_answer_is_asked_again_with_the_strict_prompt() {
        let mut prompts: Vec<String> = Vec::new();
        let out = translate_piece(
            "Привет, мир",
            RU_EN,
            &STANDARD,
            &mut |p: String| {
                prompts.push(p);
                Ok(Some(if prompts.len() == 1 {
                    "Okay, here's the translation:\n\n* Hello".to_string()
                } else {
                    "Hello, world".to_string()
                }))
            },
        )
        .unwrap();
        assert_eq!(out, "Hello, world");
        assert_eq!(prompts.len(), 2);
        assert!(!prompts[0].contains("Never summarize"));
        assert!(prompts[1].contains("Never summarize"));
    }

    #[test]
    fn the_first_answer_is_kept_when_the_retry_is_no_better() {
        let mut asked = 0;
        let out = translate_piece(
            "Привет, мир",
            RU_EN,
            &STANDARD,
            &mut |_: String| {
                asked += 1;
                Ok(Some(format!("Okay, here's try {asked}: hello")))
            },
        )
        .unwrap();
        assert_eq!(out, "Okay, here's try 1: hello");
        assert_eq!(asked, 2);
    }

    #[test]
    fn a_good_answer_is_asked_once() {
        let mut asked = 0;
        let out = translate_piece(
            "Привет, мир",
            RU_EN,
            &STANDARD,
            &mut |_: String| {
                asked += 1;
                Ok(Some("Hello, world.".to_string()))
            },
        )
        .unwrap();
        assert_eq!(out, "Hello, world");
        assert_eq!(asked, 1);
    }

    #[test]
    fn pieces_are_translated_in_order_and_reassembled() {
        let segments = prompt::split_for_translation("Привет.\n\n- 2024 -\n\nКак дела?");
        let mut prompts = Vec::new();
        let mut steps = Vec::new();
        let out = translate_segments(
            &segments,
            RU_EN,
            &STANDARD,
            |p| {
                prompts.push(p);
                Ok(Some(format!(" <think>x</think>answer {}\n", prompts.len())))
            },
            |n| steps.push(n),
        )
        .unwrap();
        assert_eq!(out, "answer 1\n\n- 2024 -\n\nanswer 2");
        assert_eq!(steps, vec![1, 2]);
        assert_eq!(prompts.len(), 2);
        assert!(prompts[0].ends_with("\n\n\nПривет."), "{}", prompts[0]);
        assert!(prompts[1].contains("Russian (ru) to English (en)"));
    }

    #[test]
    fn numbered_list_markers_survive_the_model() {
        let segments = prompt::split_for_translation("1) Купить хлеб\n\n2) Купить молоко");
        let mut answers = ["1. Buy bread", "2. Buy milk"].into_iter();
        let out = translate_segments(
            &segments,
            RU_EN,
            &STANDARD,
            |_| Ok(answers.next().map(String::from)),
            |_| {},
        )
        .unwrap();
        assert_eq!(out, "1) Buy bread\n\n2) Buy milk");
    }

    #[test]
    fn a_cancelled_request_stops_the_translation_at_once() {
        let segments = prompt::split_for_translation("Раз.\n\nДва.\n\nТри.");
        let mut sent = 0;
        let err = translate_segments(
            &segments,
            RU_EN,
            &STANDARD,
            |_| {
                sent += 1;
                if sent == 2 {
                    Err(CANCELLED.into())
                } else {
                    Ok(Some("x".into()))
                }
            },
            |_| {},
        )
        .unwrap_err();
        assert_eq!(sent, 2, "nothing is sent after the cancel");
        assert_eq!(err.detail, CANCELLED);
    }

    #[test]
    fn waiting_returns_the_answer_without_aborting() {
        let (tx, rx) = mpsc::channel();
        tx.send(7).unwrap();
        let mut aborted = false;
        let got = wait_or_cancel(&rx, || false, || aborted = true);
        assert_eq!(got, Ok(7));
        assert!(!aborted);
    }

    #[test]
    fn waiting_aborts_the_request_when_cancelled() {
        let (_tx, rx) = mpsc::channel::<u32>();
        let polls = std::cell::Cell::new(0);
        let mut aborted = false;
        let got = wait_or_cancel(
            &rx,
            || {
                polls.set(polls.get() + 1);
                polls.get() >= 3
            },
            || aborted = true,
        );
        assert_eq!(got, Err(CANCELLED.to_string()));
        assert!(aborted);
        assert_eq!(polls.get(), 3);
    }

    #[test]
    fn waiting_reports_a_task_that_died() {
        let (tx, rx) = mpsc::channel::<u32>();
        drop(tx);
        assert!(wait_or_cancel(&rx, || false, || {}).is_err());
    }

    #[test]
    fn context_overflow_is_text_too_long() {
        let segments = prompt::split_for_translation("Текст");
        let err = translate_segments(
            &segments,
            RU_EN,
            &STANDARD,
            |_| {
                Err("API request failed with status 400 Bad Request: {\"error\":{\"type\":\"exceed_context_size_error\"}}".into())
            },
            |_| {},
        )
        .unwrap_err();
        assert_eq!(err.code, ErrorCode::TextTooLong);
    }

    #[test]
    fn other_errors_and_empty_answers_fail_the_translation() {
        let segments = prompt::split_for_translation("Текст");
        let err = translate_segments(
            &segments,
            RU_EN,
            &STANDARD,
            |_| Err("timeout".into()),
            |_| {},
        )
        .unwrap_err();
        assert_eq!(err, Failure::new(ErrorCode::TranslationFailed, "timeout"));
        for answer in [None, Some("  \u{200B} ".to_string())] {
            let err =
                translate_segments(&segments, RU_EN, &STANDARD, |_| Ok(answer.clone()), |_| {})
                    .unwrap_err();
            assert_eq!(err.code, ErrorCode::TranslationFailed);
        }
    }

    #[test]
    fn text_without_letters_needs_no_requests() {
        let segments = prompt::split_for_translation("2024 — 😀");
        assert_eq!(count_translatable(&segments), 0);
        let out = translate_segments(
            &segments,
            RU_EN,
            &STANDARD,
            |_| panic!("no request"),
            |_| {},
        )
        .unwrap();
        assert_eq!(out, "2024 — 😀");
    }

    /// Live run on the GPU: splitting + prompt + llm_client on a letter with a
    /// list (compare with B1). Paths: settings defaults or `SELTRANSLATE_LLAMA_EXE`
    /// / `SELTRANSLATE_MODEL`.
    #[test]
    #[ignore]
    fn real_server_translates_a_letter_with_a_list() {
        let mut cfg = settings::SelTranslateSettings::default();
        if let Ok(exe) = std::env::var("SELTRANSLATE_LLAMA_EXE") {
            cfg.llama_server_path = exe;
        }
        if let Ok(model) = std::env::var("SELTRANSLATE_MODEL") {
            cfg.model_path = model;
        }
        let dir = tempfile::tempdir().unwrap();
        let server = LlamaServer::default();
        let (port, api_key) = server
            .ensure_running(&cfg.server_config(Slot::Quality, dir.path().join("llama-server.log")))
            .expect("server start");
        let provider = local_provider(port);
        let letter = "Привет, Пётр!\n\nСписок на завтра:\n- купить молоко;\n- позвонить в банк;\n- забрать посылку.\n\n1. Сначала банк.\n2) Потом почта.\n\n-----\n\nСпасибо,\nАлексей";
        for text in [
            letter,
            "Hello, Peter!\n\n* buy milk\n* call the bank\n\nThanks",
        ] {
            let dir = prompt::detect_direction(text);
            let segments = prompt::split_for_translation(text);
            let t = Instant::now();
            let out = translate_segments(
                &segments,
                dir,
                &PromptChoice::default(),
                |p| {
                    tauri::async_runtime::block_on(crate::llm_client::send_chat_completion(
                        &provider,
                        api_key.clone(),
                        MODEL,
                        p,
                        false,
                    ))
                },
                |_| {},
            )
            .expect("translation");
            println!(
                "--- {} → {}, {} pieces, {} ms ---\n{out}\n",
                dir.src,
                dir.tgt,
                count_translatable(&segments),
                t.elapsed().as_millis()
            );
            assert_eq!(
                out.matches("\n\n").count(),
                text.matches("\n\n").count(),
                "paragraphs kept"
            );
        }
        server.stop();
    }

    /// Live run (I): every model of `SELTRANSLATE_MODELS` (`;`-separated) ×
    /// every style of `SELTRANSLATE_STYLES` (default standard, natural,
    /// gaming, software) on the texts of `SELTRANSLATE_COMPARE_DIR`, through
    /// the hotkey's own path (splitting, check, one retry). Appends Markdown
    /// to `<dir>/styles.md`. Needs `SELTRANSLATE_LLAMA_EXE`.
    #[test]
    #[ignore]
    fn real_models_with_prompt_styles() {
        use prompt::PromptStyle;
        use std::fmt::Write as _;
        let var = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("{name} not set"));
        let dir = std::path::PathBuf::from(var("SELTRANSLATE_COMPARE_DIR"));
        let read = |name: &str| std::fs::read_to_string(dir.join(name)).unwrap();
        let texts: Vec<(&str, String)> = vec![
            ("libra", read("libra.txt")),
            (
                "slang",
                "Honestly, the patch nerfed my main into the ground, so I'm just gonna grind ranked on alts till they fix it.".into(),
            ),
            ("yes-no", "Ожидание ответа только вопросом да или нет.".into()),
            (
                "idioms",
                "Не в бровь, а в глаз: он сразу понял, где собака зарыта, и не стал ходить вокруг да около.".into(),
            ),
            ("photoshop", read("photoshop.txt")),
            ("windows", read("windows.txt")),
        ];
        let styles: Vec<PromptStyle> = std::env::var("SELTRANSLATE_STYLES")
            .unwrap_or_else(|_| "standard,natural,gaming,software".into())
            .split(',')
            .map(|s| serde_json::from_value(serde_json::json!(s.trim())).unwrap())
            .collect();
        // The Custom prompt, when `custom` is among the styles.
        let custom = std::env::var("SELTRANSLATE_CUSTOM").unwrap_or_default();
        let out_path = dir.join("styles.md");
        let server = LlamaServer::default();
        for model in var("SELTRANSLATE_MODELS")
            .split(';')
            .filter(|m| !m.is_empty())
        {
            let cfg = settings::SelTranslateSettings {
                llama_server_path: var("SELTRANSLATE_LLAMA_EXE"),
                model_path: model.into(),
                ..Default::default()
            };
            let config = cfg.server_config(Slot::Quality, dir.join("server-styles.log"));
            let t = Instant::now();
            let (port, key) = match server.ensure_running(&config) {
                Ok(endpoint) => endpoint,
                // A GGUF that is no chat model: what the popup would say.
                Err(e) => {
                    let out = format!(
                        "\n## {} ({:?}): ServerFailed — {}\n",
                        super::super::models::display_name(model),
                        config.chat_format,
                        e.lines().next().unwrap_or_default()
                    );
                    print!("{out}");
                    let mut file = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&out_path)
                        .unwrap();
                    std::io::Write::write_all(&mut file, out.as_bytes()).unwrap();
                    continue;
                }
            };
            let mut out = format!(
                "\n## {} ({:?}), start {:.1} s, VRAM {:?} MiB\n",
                super::super::models::display_name(model),
                config.chat_format,
                t.elapsed().as_secs_f64(),
                vram_used_mib()
            );
            let provider = local_provider(port);
            for style in &styles {
                let choice = PromptChoice {
                    style: *style,
                    custom: custom.clone(),
                };
                let _ = writeln!(out, "\n### {style:?}\n");
                for (name, text) in &texts {
                    let dir_ = prompt::detect_direction(text);
                    let segments = prompt::split_for_translation(text);
                    let mut calls = 0;
                    let t = Instant::now();
                    let result = translate_segments(
                        &segments,
                        dir_,
                        &choice,
                        |p| {
                            calls += 1;
                            tauri::async_runtime::block_on(crate::llm_client::send_chat_completion(
                                &provider,
                                key.clone(),
                                MODEL,
                                p,
                                false,
                            ))
                        },
                        |_| {},
                    );
                    let pieces = count_translatable(&segments);
                    let text =
                        result.unwrap_or_else(|f| format!("ERROR {:?}: {}", f.code, f.detail));
                    let _ = writeln!(
                        out,
                        "**{name}** {} → {}, {:.2} s, {pieces} pieces, {} retries\n```text\n{text}\n```",
                        dir_.src,
                        dir_.tgt,
                        t.elapsed().as_secs_f64(),
                        calls - pieces
                    );
                }
            }
            print!("{out}");
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&out_path)
                .unwrap();
            std::io::Write::write_all(&mut file, out.as_bytes()).unwrap();
            server.stop();
            std::thread::sleep(Duration::from_secs(2));
        }
    }

    /// Live run (I.5), one step after another, never at the same time:
    /// (1) translate with a big model → dictation starts (`yield_to_dictation`)
    /// → the speech model loads and transcribes → the next translation starts
    /// the server again; (2) the speech model stays loaded → translate with
    /// the big model again, speed compared with (1). Env: `SELTRANSLATE_LLAMA_EXE`,
    /// `SELTRANSLATE_MODEL` (the big GGUF), `SELTRANSLATE_WHISPER` (a Whisper
    /// GGUF of Handy), `SELTRANSLATE_SPEECH` (an audio file, first 20 s used),
    /// `HANDY_TRANSCRIBE_WORKER_EXE` (a built handy.exe, run as the worker).
    #[test]
    #[ignore]
    fn real_dictation_pushes_out_a_big_model() {
        use crate::engine_supervisor::{DeviceSelector, EngineSupervisor, LoadSpec};
        let var = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("{name} not set"));
        let cfg = settings::SelTranslateSettings {
            llama_server_path: var("SELTRANSLATE_LLAMA_EXE"),
            model_path: var("SELTRANSLATE_MODEL"),
            ..Default::default()
        };
        let log_dir = tempfile::tempdir().unwrap();
        let config = cfg.server_config(Slot::Quality, log_dir.path().join("llama-server.log"));
        let server = LlamaServer::default();
        let text = std::fs::read_to_string(
            std::path::Path::new(&var("SELTRANSLATE_COMPARE_DIR")).join("libra.txt"),
        )
        .unwrap();
        let translate = |server: &LlamaServer| {
            let t = Instant::now();
            let (port, key) = server.ensure_running(&config).expect("server start");
            let started = t.elapsed();
            let provider = local_provider(port);
            let segments = prompt::split_for_translation(&text);
            let t = Instant::now();
            let out = translate_segments(
                &segments,
                prompt::detect_direction(&text),
                &PromptChoice::default(),
                |p| {
                    tauri::async_runtime::block_on(crate::llm_client::send_chat_completion(
                        &provider,
                        key.clone(),
                        MODEL,
                        p,
                        false,
                    ))
                },
                |_| {},
            )
            .expect("translation");
            let took = t.elapsed();
            println!(
                "  server start {:.1} s, translation {:.2} s ({:.0} chars/s), VRAM {:?} MiB",
                started.as_secs_f64(),
                took.as_secs_f64(),
                out.chars().count() as f64 / took.as_secs_f64(),
                vram_used_mib()
            );
            took
        };
        let speech = {
            let mut decoder = crate::audio_file::decoder::StreamingDecoder::open(
                std::path::Path::new(&var("SELTRANSLATE_SPEECH")),
            )
            .unwrap();
            let mut pcm: Vec<f32> = Vec::new();
            while pcm.len() < 20 * 16_000
                && decoder.next_frames(|f| pcm.extend_from_slice(f)).unwrap()
            {}
            pcm.truncate(20 * 16_000);
            pcm
        };

        println!("VRAM at the start: {:?} MiB", vram_used_mib());
        println!("(1) translation with the big model, nothing else loaded:");
        let alone = translate(&server);

        println!("(1) dictation starts:");
        let t = Instant::now();
        assert!(server.yield_to_dictation(), "the big model must leave");
        std::thread::sleep(Duration::from_millis(500));
        println!(
            "  llama-server stopped in {} ms, VRAM {:?} MiB",
            t.elapsed().as_millis(),
            vram_used_mib()
        );
        assert_eq!(server.status().state, ServerState::Stopped);
        let engine = EngineSupervisor::new(true);
        let t = Instant::now();
        let info = engine
            .load(LoadSpec {
                path: var("SELTRANSLATE_WHISPER").into(),
                backend: transcribe_cpp::Backend::Auto,
                device: DeviceSelector::Auto,
            })
            .expect("speech model load");
        println!(
            "  speech model loaded in {:.1} s on '{}' ({}), VRAM {:?} MiB",
            t.elapsed().as_secs_f64(),
            info.backend,
            info.device,
            vram_used_mib()
        );
        let t = Instant::now();
        let transcript = engine
            .transcribe(speech, transcribe_cpp::RunOptions::default())
            .expect("transcription");
        println!(
            "  20 s of speech transcribed in {:.2} s: {}",
            t.elapsed().as_secs_f64(),
            transcript.text.chars().take(120).collect::<String>()
        );
        assert!(!transcript.text.trim().is_empty());

        println!("(2) the speech model stays loaded, translation with the big model again:");
        let shared = translate(&server);
        println!(
            "  translation took {:.2}x the time it took alone",
            shared.as_secs_f64() / alone.as_secs_f64()
        );
        server.stop();
        engine.unload().wait();
    }

    fn vram_used_mib() -> Option<u32> {
        let out = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }

    /// Live run (G2): one server, Fast → Quality → Fast. Prints the time of
    /// every slot change (restart + first answer) and the video memory used.
    /// Defaults from the settings; `SELTRANSLATE_LLAMA_EXE` overrides the exe.
    #[test]
    #[ignore]
    fn real_server_switches_between_slots() {
        let mut cfg = settings::SelTranslateSettings::default();
        if let Ok(exe) = std::env::var("SELTRANSLATE_LLAMA_EXE") {
            cfg.llama_server_path = exe;
        }
        assert!(
            cfg.fast_available(),
            "Fast model not found: {}",
            cfg.fast_model_path
        );
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("llama-server.log");
        let server = LlamaServer::default();
        let text = "Мы вчера обсудили план релиза, и всё должно быть готово к пятнице.";
        let dir_ru_en = prompt::detect_direction(text);
        let before = vram_used_mib();
        println!("VRAM before: {before:?} MiB");
        for slot in [Slot::Fast, Slot::Quality, Slot::Fast, Slot::Quality] {
            let t = Instant::now();
            let (port, key) = server
                .ensure_running(&cfg.server_config(slot, log.clone()))
                .expect("server start");
            let switched = t.elapsed();
            let provider = local_provider(port);
            let t2 = Instant::now();
            let answer = translate_piece(text, dir_ru_en, &cfg.prompt_for(slot), &mut |p| {
                tauri::async_runtime::block_on(crate::llm_client::send_chat_completion(
                    &provider,
                    key.clone(),
                    MODEL,
                    p,
                    false,
                ))
            })
            .expect("translation");
            println!(
                "{slot:?}: slot change {:.2} s, first answer {:.2} s, VRAM {:?} MiB (+{:?}) -> {answer}",
                switched.as_secs_f64(),
                t2.elapsed().as_secs_f64(),
                vram_used_mib(),
                vram_used_mib().zip(before).map(|(a, b)| a as i64 - b as i64),
            );
            // Same slot again: no restart.
            let t3 = Instant::now();
            server
                .ensure_running(&cfg.server_config(slot, log.clone()))
                .expect("reuse");
            assert!(
                t3.elapsed() < Duration::from_secs(1),
                "same slot must not restart"
            );
        }
        server.stop();
        std::thread::sleep(Duration::from_secs(1));
        println!("VRAM after stop: {:?} MiB", vram_used_mib());
    }

    #[test]
    fn better_translation_is_offered_only_for_a_fast_result_in_the_popup() {
        assert!(can_improve(OutputMode::Popup, Slot::Fast, true));
        assert!(!can_improve(OutputMode::Popup, Slot::Fast, false));
        assert!(!can_improve(OutputMode::Popup, Slot::Quality, true));
        assert!(!can_improve(OutputMode::Replace, Slot::Fast, true));
    }

    fn fast_done() -> PopupView {
        PopupView::Done {
            src: "ru".into(),
            tgt: "en".into(),
            slot: Slot::Fast,
            model: "TranslateGemma 4B".into(),
            text: "Hello".into(),
            truncated: false,
            clipboard_lost: false,
            can_improve: true,
            improve_failed: false,
            from_cache: false,
        }
    }

    #[test]
    fn the_fast_result_comes_back_after_cancel_or_an_error() {
        let shown = fast_done();
        // Cancel: the same result, still with the button.
        assert_eq!(fast_view_again(&shown, false), shown);
        // Error: the same result plus the note.
        let PopupView::Done {
            text,
            slot,
            can_improve,
            improve_failed,
            ..
        } = fast_view_again(&shown, true)
        else {
            panic!("a result view");
        };
        assert_eq!((text.as_str(), slot), ("Hello", Slot::Fast));
        assert!(can_improve && improve_failed);
        // A later cancel clears the note of an earlier error.
        let failed = fast_view_again(&shown, true);
        assert_eq!(fast_view_again(&failed, false), shown);
    }

    #[test]
    fn the_second_hotkey_is_always_quality_and_the_first_follows_the_setting() {
        let dir = tempfile::tempdir().unwrap();
        let fast = dir.path().join("small.gguf");
        std::fs::write(&fast, b"x").unwrap();
        let big = dir.path().join("big.gguf");
        std::fs::write(&big, b"x").unwrap();
        let cfg = settings::SelTranslateSettings {
            selection_slot: settings::SelectionSlot::Fast,
            fast_model_path: fast.display().to_string(),
            model_path: big.display().to_string(),
            ..Default::default()
        };
        // A Fast-only setup: the job carries the slot that really runs, so
        // the popup does not say "larger model" for Fast.
        let fast_only = settings::SelTranslateSettings {
            model_path: String::new(),
            ..cfg.clone()
        };
        assert_eq!(
            slot_for(ModelChoice::Quality, &fast_only, "Привет", false),
            Slot::Fast
        );
        let long = "слово ".repeat(100);
        assert_eq!(
            slot_for(ModelChoice::Quality, &cfg, "Привет", false),
            Slot::Quality
        );
        assert_eq!(
            slot_for(ModelChoice::FromSettings, &cfg, &long, false),
            Slot::Fast
        );
        let auto = settings::SelTranslateSettings {
            selection_slot: settings::SelectionSlot::Auto,
            ..cfg
        };
        assert_eq!(
            slot_for(ModelChoice::FromSettings, &auto, "Привет", false),
            Slot::Fast
        );
        assert_eq!(
            slot_for(ModelChoice::FromSettings, &auto, &long, false),
            Slot::Quality
        );
        assert_eq!(
            slot_for(ModelChoice::Quality, &auto, "Привет", false),
            Slot::Quality
        );
        // Quality already running: Auto keeps it for a short text too.
        assert_eq!(
            slot_for(ModelChoice::FromSettings, &auto, "Привет", true),
            Slot::Quality
        );
    }

    #[test]
    fn better_translation_is_not_offered_when_auto_already_chose_quality() {
        // Auto → Quality for a long text: no button; Auto → Fast: button.
        let can = |text: &str| {
            let slot = settings::pick_auto_slot(text, true);
            can_improve(OutputMode::Popup, slot, true)
        };
        assert!(can("Привет"));
        assert!(!can(&"слово ".repeat(100)));
    }

    #[test]
    fn capture_errors_map_to_popup_codes() {
        assert_eq!(
            Failure::from(CaptureError::NoSelection),
            Failure::new(ErrorCode::NoSelection, "No text selected")
        );
        assert_eq!(
            Failure::from(CaptureError::ClipboardBusy).code,
            ErrorCode::ClipboardBusy
        );
        assert_eq!(
            Failure::from(CaptureError::NoAccessibility).code,
            ErrorCode::NoAccessibility
        );
        assert_eq!(
            Failure::from(CaptureError::Other("x".into())),
            Failure::new(ErrorCode::Other, "x")
        );
    }
}
