//! Settings of the translation module, kept in their own `"seltranslate"` key of
//! Handy's settings store, so Handy's `AppSettings` stays untouched.
//!
//! Every field falls back to its default when missing or invalid, so a store
//! written by an older build (or edited by hand) never fails the whole load.
//! The first load writes the key (and a repaired one is written back), so the
//! settings can be edited in `settings_store.json` before the UI exists.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use log::warn;
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::AppHandle;
use tauri_plugin_store::StoreExt;

use super::llama_server::{chat_format_for, ServerConfig, GPU_LAYERS_AUTO};
use super::prompt::PromptChoice;

const STORE_KEY: &str = "seltranslate";

// How the translation is shown: a popup near the cursor, or pasted over the selection.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type, Default)]
#[serde(rename_all = "lowercase")]
pub enum OutputMode {
    #[default]
    Popup,
    Replace,
}

/// Which model translates: Fast (small, quick) or Quality (the big one). G2.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "lowercase")]
pub enum Slot {
    Fast,
    Quality,
}

/// The model choice of the hotkey / double Ctrl: a fixed slot, or Auto (G3),
/// which picks Fast for a short text and Quality for a long one. A separate
/// type: `Slot` is what the server and the file translation work with.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "lowercase")]
pub enum SelectionSlot {
    Fast,
    Quality,
    Auto,
}

impl SelectionSlot {
    /// The slot to warm up without a text (the Test button): Auto starts Fast.
    pub fn without_text(self) -> Slot {
        match self {
            SelectionSlot::Quality => Slot::Quality,
            SelectionSlot::Fast | SelectionSlot::Auto => Slot::Fast,
        }
    }
}

/// Auto sends a text to Fast only when it has at most this many characters...
pub const AUTO_MAX_CHARS: usize = 400;
/// ...and at most this many sentences (a line break also ends one).
pub const AUTO_MAX_SENTENCES: usize = 3;

/// Number of sentences: a run of `.!?…` followed by a space or the end of the
/// text ends one (so `3.5` and `a.b` do not), and so does every line break.
/// A tail without a terminator counts as one more.
fn count_sentences(text: &str) -> usize {
    let mut count = 0;
    let mut open = false; // text since the last boundary
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\n' || c == '\r' {
            count += usize::from(open);
            open = false;
        } else if matches!(c, '.' | '!' | '?' | '…') {
            while chars
                .next_if(|n| matches!(n, '.' | '!' | '?' | '…'))
                .is_some()
            {}
            if chars.peek().is_none_or(|n| n.is_whitespace()) {
                count += usize::from(open);
                open = false;
            }
        } else if !c.is_whitespace() {
            open = true;
        }
    }
    count + usize::from(open)
}

/// The Auto rule: Fast for a short text (up to `AUTO_MAX_CHARS` characters and
/// `AUTO_MAX_SENTENCES` sentences), Quality for the rest or when Fast is missing.
pub fn pick_auto_slot(text: &str, fast_available: bool) -> Slot {
    let text = text.trim();
    let short =
        text.chars().count() <= AUTO_MAX_CHARS && count_sentences(text) <= AUTO_MAX_SENTENCES;
    if fast_available && short {
        Slot::Fast
    } else {
        Slot::Quality
    }
}

// `idle_unload_minutes = 0` keeps the server running until Handy exits.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Type)]
#[serde(default)]
pub struct SelTranslateSettings {
    pub llama_server_path: String,
    /// The Quality model (the one that was the only model before G2).
    pub model_path: String,
    /// The Fast model; the slot is usable only while this file exists.
    pub fast_model_path: String,
    /// Model of the hotkey / double Ctrl translation: Fast, Quality or Auto.
    pub selection_slot: SelectionSlot,
    /// Slot of the file translation.
    pub file_slot: Slot,
    pub gpu_layers: u32,
    pub ctx_size: u32,
    pub idle_unload_minutes: u32,
    pub output_mode: OutputMode,
    /// Language to translate into (L): "auto" (Russian → English, everything
    /// else → Russian) or an ISO 639-1 code from the language table.
    pub target_language: String,
    pub double_ctrl_enabled: bool,
    pub auto_pause_fullscreen: bool,
    /// Prompt style per model (I), keyed by `prompt_key` of the model file:
    /// the file name in lower case, so a moved file keeps its style.
    pub prompt_styles: BTreeMap<String, PromptChoice>,
    /// Model files picked before, newest first (for the model list, I).
    pub recent_models: Vec<String>,
}

/// How many earlier model files the list remembers.
pub const MAX_RECENT_MODELS: usize = 8;

/// The key of a model in `prompt_styles`: its file name in lower case.
pub fn prompt_key(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

/// Whether two paths name the same file (canonical paths; the text when a
/// file is missing).
pub fn same_file(a: &str, b: &str) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => !a.is_empty() && a == b,
    }
}

#[cfg(not(any(windows, target_os = "macos")))]
const MODEL_FILE: &str = "translategemma-12b-it.Q4_K_M.gguf";
#[cfg(not(any(windows, target_os = "macos")))]
const FAST_MODEL_FILE: &str = "translategemma-4b-it.Q4_K_M.gguf";

/// Roots of the default paths of a fresh store (DESIGN F.4): the app data
/// folder (the engine) and the Hugging Face cache (the models, next to
/// Handy's speech models). Set by the first `load`; unset in unit tests.
struct DefaultRoots {
    app_data: PathBuf,
    hf_hub: PathBuf,
}

static DEFAULT_ROOTS: OnceLock<DefaultRoots> = OnceLock::new();

fn init_default_roots(app: &AppHandle) {
    if DEFAULT_ROOTS.get().is_some() {
        return;
    }
    let Ok(app_data) = crate::portable::app_data_dir(app) else {
        return;
    };
    let _ = DEFAULT_ROOTS.set(DefaultRoots {
        app_data,
        hf_hub: hf_hub::Cache::from_env().path().clone(),
    });
}

/// The app data folder and the Hugging Face cache (`hub`) the setup uses.
pub fn setup_roots() -> Option<(PathBuf, PathBuf)> {
    DEFAULT_ROOTS
        .get()
        .map(|r| (r.app_data.clone(), r.hf_hub.clone()))
}

/// Where the setup puts llama-server (Vulkan on Windows, Metal on a Mac) and
/// the models; a stored path is never replaced by these.
#[cfg(any(windows, target_os = "macos"))]
fn default_paths() -> (String, String, String) {
    use super::setup::catalog;
    let (app_data, hub) = setup_roots().unwrap_or_default();
    let model = |id| {
        catalog::item(id)
            .and_then(|item| catalog::model_file(&hub, item))
            .map(|p| p.display().to_string())
            .unwrap_or_default()
    };
    (
        catalog::engine_exe(&app_data, catalog::DEFAULT_ENGINE)
            .display()
            .to_string(),
        model(catalog::MODEL_QUALITY),
        model(catalog::MODEL_FAST),
    )
}

/// Linux: `llama-server` from a package manager; the model sits in the home folder.
#[cfg(not(any(windows, target_os = "macos")))]
fn default_paths() -> (String, String, String) {
    let home = std::env::var("HOME").unwrap_or_default();
    (
        "/opt/homebrew/bin/llama-server".to_string(),
        format!("{home}/SelTranslate-tools/models/{MODEL_FILE}"),
        format!("{home}/SelTranslate-tools/models/{FAST_MODEL_FILE}"),
    )
}

/// For a new store: Auto needs the Fast model; without its file everything
/// goes to Quality. A stored value is never replaced by this.
fn default_selection_slot(fast_model_path: &str) -> SelectionSlot {
    if std::path::Path::new(fast_model_path).is_file() {
        SelectionSlot::Auto
    } else {
        SelectionSlot::Quality
    }
}

impl Default for SelTranslateSettings {
    fn default() -> Self {
        let (llama_server_path, model_path, fast_model_path) = default_paths();
        let selection_slot = default_selection_slot(&fast_model_path);
        Self {
            llama_server_path,
            model_path,
            fast_model_path,
            selection_slot,
            file_slot: Slot::Quality,
            gpu_layers: GPU_LAYERS_AUTO,
            ctx_size: 4096,
            idle_unload_minutes: 5,
            output_mode: OutputMode::Popup,
            target_language: "auto".into(),
            double_ctrl_enabled: true,
            // On a Mac the green full-screen button is how people work in Safari
            // or Pages; pausing there would switch translation off (DESIGN J.7).
            auto_pause_fullscreen: cfg!(windows),
            prompt_styles: BTreeMap::new(),
            recent_models: Vec::new(),
        }
    }
}

/// Smallest context that still fits a ~1000-character chunk, its prompt and answer.
pub const MIN_CTX_SIZE: u32 = 1024;

impl SelTranslateSettings {
    /// Builds settings from a stored value field by field: a missing or broken
    /// field gets its default instead of discarding the rest. The flag says
    /// whether the stored value differs from the result (worth writing back).
    fn from_stored(value: &serde_json::Value) -> (Self, bool) {
        if let Ok(settings) = serde_json::from_value::<Self>(value.clone()) {
            let complete = serde_json::to_value(&settings).ok().as_ref() == Some(value);
            return (settings, !complete);
        }
        let mut merged = serde_json::to_value(Self::default()).expect("settings serialize");
        if let (Some(stored), Some(target)) = (value.as_object(), merged.as_object_mut()) {
            for (key, field) in stored {
                if !target.contains_key(key) {
                    continue;
                }
                let mut probe = target.clone();
                probe.insert(key.clone(), field.clone());
                if serde_json::from_value::<Self>(serde_json::Value::Object(probe)).is_ok() {
                    target.insert(key.clone(), field.clone());
                } else {
                    warn!("Translate settings: invalid '{key}', using the default");
                }
            }
        }
        (serde_json::from_value(merged).unwrap_or_default(), true)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.ctx_size < MIN_CTX_SIZE {
            return Err(format!("Context size must be at least {MIN_CTX_SIZE}"));
        }
        for (model, choice) in &self.prompt_styles {
            if choice.style == super::prompt::PromptStyle::Custom {
                if let Some(problem) = super::prompt::custom_prompt_problem(&choice.custom) {
                    return Err(format!("Custom prompt of {model}: {problem}"));
                }
            }
        }
        Ok(())
    }

    /// Whether the Fast model file is there (Quality is always "available":
    /// a missing file is reported when the server starts, as before G2).
    pub fn fast_available(&self) -> bool {
        !self.fast_model_path.is_empty() && std::path::Path::new(&self.fast_model_path).is_file()
    }

    /// Whether the Quality model file is there (for "Better translation").
    pub fn quality_available(&self) -> bool {
        !self.model_path.is_empty() && std::path::Path::new(&self.model_path).is_file()
    }

    /// The slot that will really be used for `requested`: a slot without its
    /// model file falls back to the other one when that one has its file
    /// (Fast → Quality as in G2; Quality → Fast for a Fast-only setup, F.2).
    pub fn resolve_slot(&self, requested: Slot) -> Slot {
        match requested {
            Slot::Fast if !self.fast_available() => Slot::Quality,
            Slot::Quality if !self.quality_available() && self.fast_available() => Slot::Fast,
            slot => slot,
        }
    }

    /// The slot the hotkey / double Ctrl uses for `text` (Auto looks at the
    /// text). Auto stays on Quality while that model is already running
    /// (`quality_loaded`): a short text then costs no reload of the small model
    /// (the big one would be stopped for it and started again later).
    pub fn slot_for_selection(&self, text: &str, quality_loaded: bool) -> Slot {
        self.resolve_slot(match self.selection_slot {
            SelectionSlot::Fast => Slot::Fast,
            SelectionSlot::Quality => Slot::Quality,
            SelectionSlot::Auto if quality_loaded => Slot::Quality,
            SelectionSlot::Auto => pick_auto_slot(text, self.fast_available()),
        })
    }

    /// Whether translation can run: the engine file and at least one model
    /// (DESIGN F.4). Without it a hotkey shows "not set up".
    pub fn setup_ready(&self) -> bool {
        Path::new(&self.llama_server_path).is_file()
            && (self.fast_available() || self.quality_available())
    }

    pub fn model_for(&self, slot: Slot) -> &str {
        match slot {
            Slot::Fast => &self.fast_model_path,
            Slot::Quality => &self.model_path,
        }
    }

    /// Server configuration of a slot; only the model (and with it the chat
    /// format, I) differs between slots.
    pub fn server_config(&self, slot: Slot, log_path: std::path::PathBuf) -> ServerConfig {
        let model_path = PathBuf::from(self.model_for(slot));
        ServerConfig {
            exe_path: self.llama_server_path.clone().into(),
            chat_format: chat_format_for(&model_path),
            model_path,
            gpu_layers: self.gpu_layers,
            ctx_size: self.ctx_size,
            log_path,
        }
    }

    /// The prompt style of the model in `slot` (Standard when none is stored).
    pub fn prompt_for(&self, slot: Slot) -> PromptChoice {
        self.prompt_styles
            .get(&prompt_key(self.model_for(slot)))
            .cloned()
            .unwrap_or_default()
    }

    /// "Better translation" makes sense only when Quality is there and is
    /// another file than Fast (I: any model may sit in any slot).
    pub fn quality_differs_from_fast(&self) -> bool {
        self.quality_available() && !same_file(&self.model_path, &self.fast_model_path)
    }

    /// Puts the models of both slots and those of `older` (earlier lists)
    /// in front of `recent_models`: no repeats, at most `MAX_RECENT_MODELS`.
    pub fn remember_models(&mut self, older: &[String]) {
        let mut list: Vec<String> = Vec::new();
        let candidates = [self.fast_model_path.clone(), self.model_path.clone()]
            .into_iter()
            .chain(self.recent_models.iter().cloned())
            .chain(older.iter().cloned());
        for path in candidates {
            if path.is_empty() || list.iter().any(|p| same_file(p, &path)) {
                continue;
            }
            list.push(path);
        }
        list.truncate(MAX_RECENT_MODELS);
        self.recent_models = list;
    }

    pub fn idle_unload_ms(&self) -> u64 {
        u64::from(self.idle_unload_minutes) * 60_000
    }
}

pub fn load(app: &AppHandle) -> SelTranslateSettings {
    init_default_roots(app);
    let Ok(store) = app.store(crate::portable::store_path(
        crate::settings::SETTINGS_STORE_PATH,
    )) else {
        warn!("Translate settings: store unavailable, using defaults");
        return SelTranslateSettings::default();
    };
    let (settings, write_back) = match store.get(STORE_KEY) {
        Some(value) => SelTranslateSettings::from_stored(&value),
        None => (SelTranslateSettings::default(), true),
    };
    if write_back {
        if let Ok(value) = serde_json::to_value(&settings) {
            store.set(STORE_KEY, value);
        }
    }
    settings
}

pub fn save(app: &AppHandle, settings: &SelTranslateSettings) -> Result<(), String> {
    let store = app
        .store(crate::portable::store_path(
            crate::settings::SETTINGS_STORE_PATH,
        ))
        .map_err(|e| format!("Failed to open the settings store: {e}"))?;
    let value = serde_json::to_value(settings).map_err(|e| e.to_string())?;
    store.set(STORE_KEY, value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_match_the_design() {
        let s = SelTranslateSettings::default();
        let exe = if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        };
        assert_eq!(
            std::path::Path::new(&s.llama_server_path).file_name(),
            Some(std::ffi::OsStr::new(exe))
        );
        assert_eq!(
            std::path::Path::new(&s.model_path).file_name(),
            Some(std::ffi::OsStr::new("translategemma-12b-it.Q4_K_M.gguf"))
        );
        assert_eq!(
            (s.gpu_layers, s.ctx_size, s.idle_unload_minutes),
            (99, 4096, 5)
        );
        assert_eq!(
            std::path::Path::new(&s.fast_model_path).file_name(),
            Some(std::ffi::OsStr::new("translategemma-4b-it.Q4_K_M.gguf"))
        );
        assert_eq!(s.file_slot, Slot::Quality);
        assert_eq!(s.output_mode, OutputMode::Popup);
        assert!(s.double_ctrl_enabled);
        assert_eq!(s.auto_pause_fullscreen, cfg!(windows));
        assert_eq!(s.idle_unload_ms(), 300_000);
    }

    #[test]
    fn partial_store_keeps_valid_fields_and_defaults_the_rest() {
        let stored = json!({
            "gpu_layers": 20,
            "output_mode": "replace",
            "ctx_size": "big",
            "unknown_field": 1
        });
        let (s, write_back) = SelTranslateSettings::from_stored(&stored);
        assert!(write_back);
        assert_eq!(s.gpu_layers, 20);
        assert_eq!(s.output_mode, OutputMode::Replace);
        assert_eq!(s.ctx_size, 4096);
        assert_eq!(s.idle_unload_minutes, 5);
    }

    #[test]
    fn round_trips_through_json() {
        let s = SelTranslateSettings {
            idle_unload_minutes: 0,
            output_mode: OutputMode::Replace,
            ..Default::default()
        };
        let (back, write_back) =
            SelTranslateSettings::from_stored(&serde_json::to_value(&s).unwrap());
        assert_eq!(back, s);
        assert!(!write_back, "a complete value is not rewritten");

        let partial = json!({ "gpu_layers": 99 });
        let (back, write_back) = SelTranslateSettings::from_stored(&partial);
        assert_eq!(back, SelTranslateSettings::default());
        assert!(write_back, "missing fields are filled in the store");
    }

    #[test]
    fn tiny_context_is_rejected() {
        let s = SelTranslateSettings {
            ctx_size: 256,
            ..Default::default()
        };
        assert!(s.validate().is_err());
        assert!(SelTranslateSettings::default().validate().is_ok());
    }

    #[test]
    fn a_store_from_before_slots_reads_with_slot_defaults() {
        let old = json!({
            "llama_server_path": "X:\\llama-server.exe",
            "model_path": "X:\\big.gguf",
            "gpu_layers": 99,
            "ctx_size": 4096,
            "idle_unload_minutes": 5,
            "output_mode": "replace",
            "double_ctrl_enabled": false,
            "auto_pause_fullscreen": true
        });
        let (s, write_back) = SelTranslateSettings::from_stored(&old);
        assert!(write_back, "the new fields are written to the store");
        assert_eq!(s.model_path, "X:\\big.gguf");
        assert_eq!(s.output_mode, OutputMode::Replace);
        let d = SelTranslateSettings::default();
        assert_eq!(s.fast_model_path, d.fast_model_path);
        assert_eq!(s.selection_slot, d.selection_slot);
        assert_eq!(s.file_slot, Slot::Quality);
        assert_eq!(s.target_language, "auto");
    }

    #[test]
    fn a_broken_slot_value_falls_back_to_the_default() {
        let (s, write_back) = SelTranslateSettings::from_stored(&json!({
            "selection_slot": "turbo", "file_slot": "fast"
        }));
        assert!(write_back);
        assert_eq!(
            s.selection_slot,
            SelTranslateSettings::default().selection_slot
        );
        assert_eq!(s.file_slot, Slot::Fast);
    }

    #[test]
    fn stored_fast_and_quality_are_read_as_they_are() {
        for (stored, expected) in [
            ("fast", SelectionSlot::Fast),
            ("quality", SelectionSlot::Quality),
            ("auto", SelectionSlot::Auto),
        ] {
            let (s, write_back) =
                SelTranslateSettings::from_stored(&json!({ "selection_slot": stored }));
            assert!(write_back, "the other fields are missing");
            assert_eq!(s.selection_slot, expected);
        }
        // A complete store from G2 keeps the user's choice and is not rewritten.
        let mut g2 = serde_json::to_value(SelTranslateSettings::default()).unwrap();
        g2["selection_slot"] = json!("fast");
        let (s, write_back) = SelTranslateSettings::from_stored(&g2);
        assert_eq!(s.selection_slot, SelectionSlot::Fast);
        assert!(!write_back);
    }

    #[test]
    fn a_new_store_defaults_to_auto_only_with_the_fast_model() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("small.gguf");
        std::fs::write(&file, b"x").unwrap();
        assert_eq!(
            default_selection_slot(&file.display().to_string()),
            SelectionSlot::Auto
        );
        assert_eq!(
            default_selection_slot(&dir.path().join("none.gguf").display().to_string()),
            SelectionSlot::Quality
        );
        assert_eq!(default_selection_slot(""), SelectionSlot::Quality);
    }

    #[test]
    fn sentences_are_counted_by_terminators_and_line_breaks() {
        assert_eq!(count_sentences(""), 0);
        assert_eq!(count_sentences("   "), 0);
        assert_eq!(count_sentences("Привет"), 1);
        assert_eq!(count_sentences("Привет."), 1);
        assert_eq!(count_sentences("Привет. Как дела?"), 2);
        assert_eq!(count_sentences("Привет. Как дела? Хорошо"), 3);
        assert_eq!(count_sentences("Wait... what?!"), 2);
        assert_eq!(count_sentences("Pi is 3.5 or so."), 1);
        assert_eq!(count_sentences("one\ntwo\r\nthree"), 3);
        assert_eq!(count_sentences("one.\n\ntwo."), 2);
    }

    #[test]
    fn auto_sends_short_text_to_fast_and_the_rest_to_quality() {
        let pick = |text: &str| pick_auto_slot(text, true);
        assert_eq!(pick("Привет"), Slot::Fast);
        assert_eq!(pick("Привет, мир."), Slot::Fast);
        assert_eq!(pick("Привет, мир. Как дела?"), Slot::Fast);
        assert_eq!(pick("Привет. Как дела? Всё хорошо."), Slot::Fast);
        assert_eq!(
            pick("Привет. Как дела? Всё хорошо. Пока."),
            Slot::Quality,
            "four sentences"
        );
        assert_eq!(
            pick(&"слово ".repeat(80)),
            Slot::Quality,
            "one long sentence"
        );
        assert_eq!(pick("a\nb\nc"), Slot::Fast, "three lines");
        assert_eq!(pick("a\nb\nc\nd"), Slot::Quality, "four lines");
        assert_eq!(pick(""), Slot::Fast);
        // The limit itself is still short; one more character is not.
        assert_eq!(pick(&"я".repeat(AUTO_MAX_CHARS)), Slot::Fast);
        assert_eq!(pick(&"я".repeat(AUTO_MAX_CHARS + 1)), Slot::Quality);
        // Surrounding blanks do not count.
        assert_eq!(
            pick(&format!("  {}  \n", "я".repeat(AUTO_MAX_CHARS))),
            Slot::Fast
        );
    }

    #[test]
    fn auto_without_the_fast_model_is_quality() {
        assert_eq!(pick_auto_slot("Привет", false), Slot::Quality);
        let s = SelTranslateSettings {
            selection_slot: SelectionSlot::Auto,
            fast_model_path: String::new(),
            ..Default::default()
        };
        assert_eq!(s.slot_for_selection("Привет", false), Slot::Quality);
    }

    #[test]
    fn explicit_selection_slots_ignore_the_text() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("small.gguf");
        std::fs::write(&file, b"x").unwrap();
        let mut s = with_fast(&file);
        let big = dir.path().join("big.gguf");
        std::fs::write(&big, b"x").unwrap();
        s.model_path = big.display().to_string();
        let long = "слово ".repeat(100);
        s.selection_slot = SelectionSlot::Fast;
        assert_eq!(s.slot_for_selection(&long, false), Slot::Fast);
        s.selection_slot = SelectionSlot::Quality;
        assert_eq!(s.slot_for_selection("Привет", false), Slot::Quality);
        s.selection_slot = SelectionSlot::Auto;
        assert_eq!(s.slot_for_selection("Привет", false), Slot::Fast);
        assert_eq!(s.slot_for_selection(&long, false), Slot::Quality);
        // Quality already running: a short text stays on it; Fast is never forced.
        assert_eq!(s.slot_for_selection("Привет", true), Slot::Quality);
        s.selection_slot = SelectionSlot::Fast;
        assert_eq!(s.slot_for_selection("Привет", true), Slot::Fast);
        s.selection_slot = SelectionSlot::Auto;
        // Fast chosen by hand falls back as before when its file is gone.
        s.selection_slot = SelectionSlot::Fast;
        s.fast_model_path = String::new();
        assert_eq!(s.slot_for_selection("Привет", false), Slot::Quality);
        assert_eq!(SelectionSlot::Auto.without_text(), Slot::Fast);
        assert_eq!(SelectionSlot::Quality.without_text(), Slot::Quality);
    }

    fn with_fast(path: &std::path::Path) -> SelTranslateSettings {
        SelTranslateSettings {
            model_path: "big.gguf".into(),
            fast_model_path: path.display().to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn fast_falls_back_to_quality_when_its_file_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let missing = with_fast(&dir.path().join("none.gguf"));
        assert!(!missing.fast_available());
        assert_eq!(missing.resolve_slot(Slot::Fast), Slot::Quality);
        assert_eq!(missing.resolve_slot(Slot::Quality), Slot::Quality);

        let file = dir.path().join("small.gguf");
        std::fs::write(&file, b"x").unwrap();
        let present = with_fast(&file);
        assert!(present.fast_available());
        assert_eq!(present.resolve_slot(Slot::Fast), Slot::Fast);

        let empty = SelTranslateSettings {
            fast_model_path: String::new(),
            ..Default::default()
        };
        assert!(!empty.fast_available(), "an empty path switches Fast off");
    }

    #[test]
    fn quality_falls_back_to_fast_in_a_fast_only_setup() {
        let dir = tempfile::tempdir().unwrap();
        let small = dir.path().join("small.gguf");
        std::fs::write(&small, b"x").unwrap();
        let mut s = with_fast(&small);
        s.model_path = String::new();
        assert!(!s.quality_available());
        assert_eq!(s.resolve_slot(Slot::Quality), Slot::Fast);
        assert_eq!(s.resolve_slot(Slot::Fast), Slot::Fast);
        // Auto, Quality and a long text all run on Fast (no Better translation).
        let long = "слово ".repeat(100);
        for sel in [
            SelectionSlot::Auto,
            SelectionSlot::Quality,
            SelectionSlot::Fast,
        ] {
            s.selection_slot = sel;
            assert_eq!(s.slot_for_selection(&long, false), Slot::Fast, "{sel:?}");
        }
        // Neither file: Quality stays (the server reports the missing file).
        let none = SelTranslateSettings {
            model_path: String::new(),
            fast_model_path: String::new(),
            ..Default::default()
        };
        assert_eq!(none.resolve_slot(Slot::Quality), Slot::Quality);
        assert_eq!(none.resolve_slot(Slot::Fast), Slot::Quality);
    }

    #[test]
    fn setup_is_ready_with_the_engine_and_one_model() {
        let dir = tempfile::tempdir().unwrap();
        let file = |name: &str| {
            let p = dir.path().join(name);
            std::fs::write(&p, b"x").unwrap();
            p.display().to_string()
        };
        let exe = file("llama-server.exe");
        let small = file("small.gguf");
        let big = file("big.gguf");
        let s = |exe: &str, fast: &str, quality: &str| SelTranslateSettings {
            llama_server_path: exe.into(),
            fast_model_path: fast.into(),
            model_path: quality.into(),
            ..Default::default()
        };
        let missing = dir.path().join("none").display().to_string();
        assert!(s(&exe, &small, &big).setup_ready());
        assert!(s(&exe, &small, "").setup_ready(), "Fast only");
        assert!(s(&exe, "", &big).setup_ready(), "Quality only");
        assert!(!s(&exe, "", "").setup_ready(), "no model");
        assert!(!s(&exe, &missing, &missing).setup_ready());
        assert!(!s("", &small, &big).setup_ready(), "no engine");
        assert!(!s(&missing, &small, &big).setup_ready());
        assert!(
            !s(dir.path().to_str().unwrap(), &small, &big).setup_ready(),
            "a folder"
        );
    }

    #[test]
    fn fresh_defaults_point_into_the_setup_folders() {
        if !cfg!(windows) {
            return;
        }
        let s = SelTranslateSettings::default();
        // Without the roots (unit tests) the paths are relative.
        assert!(s
            .llama_server_path
            .ends_with(r"translate\engine\vulkan-b11623\llama-server.exe"));
        assert!(s
            .model_path
            .starts_with(r"models--mradermacher--translategemma-12b-it-GGUF\snapshots\"));
        assert!(s
            .fast_model_path
            .starts_with(r"models--mradermacher--translategemma-4b-it-GGUF\snapshots\"));
        assert!(!s.setup_ready());
    }

    #[test]
    fn a_store_from_g3_reads_with_standard_prompts_and_no_recent_models() {
        let mut g3 = serde_json::to_value(SelTranslateSettings::default()).unwrap();
        let obj = g3.as_object_mut().unwrap();
        obj.remove("prompt_styles");
        obj.remove("recent_models");
        obj.insert("selection_slot".into(), json!("fast"));
        let (s, write_back) = SelTranslateSettings::from_stored(&g3);
        assert!(write_back, "the new fields are written to the store");
        assert_eq!(s.selection_slot, SelectionSlot::Fast);
        assert!(s.prompt_styles.is_empty() && s.recent_models.is_empty());
        assert_eq!(s.prompt_for(Slot::Quality), PromptChoice::default());
        // A broken style map does not take the other fields with it.
        let (s, _) = SelTranslateSettings::from_stored(&json!({
            "prompt_styles": {"x.gguf": {"style": "turbo"}}, "ctx_size": 2048
        }));
        assert!(s.prompt_styles.is_empty());
        assert_eq!(s.ctx_size, 2048);
    }

    #[test]
    fn the_prompt_style_belongs_to_the_model_file_name() {
        use crate::translate::prompt::PromptStyle;
        assert_eq!(
            prompt_key("/m/Qwen3.5-9B-Q4_K_M.gguf"),
            "qwen3.5-9b-q4_k_m.gguf"
        );
        let mut s = SelTranslateSettings {
            fast_model_path: "/a/Qwen3.5-9B-Q4_K_M.gguf".into(),
            model_path: "/b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf".into(),
            ..Default::default()
        };
        s.prompt_styles.insert(
            "qwen3.5-9b-q4_k_m.gguf".into(),
            PromptChoice {
                style: PromptStyle::Gaming,
                custom: String::new(),
            },
        );
        assert_eq!(s.prompt_for(Slot::Fast).style, PromptStyle::Gaming);
        assert_eq!(s.prompt_for(Slot::Quality).style, PromptStyle::Standard);
        // The same file moved to another folder (or slot) keeps its style.
        s.model_path = "/elsewhere/QWEN3.5-9B-Q4_K_M.GGUF".into();
        assert_eq!(s.prompt_for(Slot::Quality).style, PromptStyle::Gaming);
    }

    #[test]
    fn a_custom_prompt_without_text_is_refused_on_save() {
        use crate::translate::prompt::PromptStyle;
        let mut s = SelTranslateSettings::default();
        let custom = |text: &str| PromptChoice {
            style: PromptStyle::Custom,
            custom: text.into(),
        };
        s.prompt_styles
            .insert("m.gguf".into(), custom("Translate: {text}"));
        assert!(s.validate().is_ok());
        s.prompt_styles
            .insert("m.gguf".into(), custom("Translate this"));
        assert!(s.validate().unwrap_err().contains("{text}"));
        s.prompt_styles
            .insert("m.gguf".into(), custom("{text} {text}"));
        assert!(s.validate().is_err());
        // A preset never needs the field.
        s.prompt_styles.insert(
            "m.gguf".into(),
            PromptChoice {
                style: PromptStyle::Natural,
                custom: String::new(),
            },
        );
        assert!(s.validate().is_ok());
    }

    #[test]
    fn better_translation_needs_another_file_in_quality() {
        let dir = tempfile::tempdir().unwrap();
        let file = |name: &str| {
            let p = dir.path().join(name);
            std::fs::write(&p, b"x").unwrap();
            p.display().to_string()
        };
        let (small, big) = (file("small.gguf"), file("big.gguf"));
        let s = |fast: &str, quality: &str| SelTranslateSettings {
            fast_model_path: fast.into(),
            model_path: quality.into(),
            ..Default::default()
        };
        assert!(s(&small, &big).quality_differs_from_fast());
        assert!(
            !s(&big, &big).quality_differs_from_fast(),
            "one file in both slots"
        );
        // The same file under another spelling of its path.
        let other_spelling = dir.path().join(".").join("big.gguf").display().to_string();
        assert!(!s(&other_spelling, &big).quality_differs_from_fast());
        assert!(!s(&small, "").quality_differs_from_fast(), "no Quality");
    }

    #[test]
    fn picked_models_are_remembered_newest_first_without_repeats() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = SelTranslateSettings {
            fast_model_path: "/m/a.gguf".into(),
            model_path: "/m/b.gguf".into(),
            recent_models: vec!["/m/c.gguf".into(), "/m/a.gguf".into()],
            ..Default::default()
        };
        s.remember_models(&["/m/d.gguf".into(), "/m/b.gguf".into(), String::new()]);
        assert_eq!(
            s.recent_models,
            ["/m/a.gguf", "/m/b.gguf", "/m/c.gguf", "/m/d.gguf"]
        );
        let many: Vec<String> = (0..20)
            .map(|i| dir.path().join(format!("{i}.gguf")).display().to_string())
            .collect();
        s.remember_models(&many);
        assert_eq!(s.recent_models.len(), MAX_RECENT_MODELS);
        assert_eq!(s.recent_models[0], "/m/a.gguf");
    }

    #[test]
    fn the_chat_format_follows_the_slot_model() {
        use crate::translate::llama_server::ChatFormat;
        let s = SelTranslateSettings {
            fast_model_path: "/m/translategemma-4b-it.Q4_K_M.gguf".into(),
            model_path: "/m/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf".into(),
            ..Default::default()
        };
        let log = std::path::PathBuf::from("l.log");
        assert_eq!(
            s.server_config(Slot::Fast, log.clone()).chat_format,
            ChatFormat::TranslateGemma
        );
        assert_eq!(
            s.server_config(Slot::Quality, log).chat_format,
            ChatFormat::Chat
        );
    }

    #[test]
    fn server_config_differs_between_slots_only_in_the_model() {
        let dir = tempfile::tempdir().unwrap();
        let s = with_fast(&dir.path().join("small.gguf"));
        let log = dir.path().join("l.log");
        let fast = s.server_config(Slot::Fast, log.clone());
        let quality = s.server_config(Slot::Quality, log);
        assert_eq!(quality.model_path, std::path::PathBuf::from("big.gguf"));
        assert_eq!(fast.model_path, dir.path().join("small.gguf"));
        assert_ne!(fast, quality);
        assert_eq!(
            ServerConfig {
                model_path: quality.model_path.clone(),
                ..fast.clone()
            },
            quality
        );
    }
}
