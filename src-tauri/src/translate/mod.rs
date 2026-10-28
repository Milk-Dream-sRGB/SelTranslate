//! Translation of the selected text (SelTranslate). B2: the llama-server
//! lifecycle and the module settings, plus commands for the future UI (B6).
//! B3: capture of the selection (`selection::capture_selection_report`).
//! B5: the `translate_selection` hotkey (`action`) and the result popup (`popup`).
//! B8: translation of a whole text file (`file`).
//! I: the model list of the slots (`models`), prompt styles (`prompt`).

pub mod action;
mod cache;
pub mod double_tap;
pub mod file;
pub mod fullscreen;
pub mod llama_server;
#[cfg(target_os = "macos")]
mod macos;
pub mod models;
pub mod popup;
pub mod prompt;
pub mod selection;
pub mod settings;
pub mod setup;

use std::sync::Arc;

use tauri::{AppHandle, Manager};

use llama_server::{LlamaServer, ServerStatus};
use settings::{SelTranslateSettings, Slot};

const SERVER_LOG_FILE: &str = "llama-server.log";

fn server_config(
    app: &AppHandle,
    settings: &SelTranslateSettings,
    slot: Slot,
) -> Result<llama_server::ServerConfig, String> {
    let log_dir = crate::portable::app_log_dir(app)
        .map_err(|e| format!("Failed to get log directory: {e}"))?;
    Ok(settings.server_config(slot, log_dir.join(SERVER_LOG_FILE)))
}

/// Starts llama-server for `slot` if needed (blocking) and returns its port and
/// the API key for `llm_client` (new on every start). Another model than the
/// running one means a restart (one server at a time, G2); Fast without its
/// model file is Quality.
pub fn ensure_server(app: &AppHandle, slot: Slot) -> Result<(u16, String), String> {
    let server = app.state::<Arc<LlamaServer>>();
    let settings = settings::load(app);
    server.set_idle_unload_ms(settings.idle_unload_ms());
    server.spawn_idle_watcher();
    server.ensure_running(&server_config(app, &settings, settings.resolve_slot(slot))?)
}

/// Whether using `slot` now has to start (or restart) the server, for the
/// "Starting model…" view.
pub fn needs_model_start(app: &AppHandle, slot: Slot) -> bool {
    let settings = settings::load(app);
    let wanted = std::path::PathBuf::from(settings.model_for(settings.resolve_slot(slot)));
    app.state::<Arc<LlamaServer>>().running_model().as_ref() != Some(&wanted)
}

/// Whether the server is running with the Quality model right now (Auto then
/// keeps it for short texts instead of swapping to Fast).
pub fn quality_loaded(app: &AppHandle) -> bool {
    let settings = settings::load(app);
    let quality =
        std::path::PathBuf::from(settings.model_for(settings.resolve_slot(Slot::Quality)));
    app.state::<Arc<LlamaServer>>().running_model().as_ref() == Some(&quality)
}

/// Dictation starts (I.5): dictation comes first, so a big translation model
/// (over `BIG_MODEL_BYTES`, Gemma 4) leaves the video card before the speech
/// model loads. Never mid-answer; small models stay. With the speech model
/// already loaded nothing is stopped: the server then started next to it and
/// fits (`-ngl auto`), and stopping it would only delay the recording (~1 s).
/// The next translation starts the server again.
pub fn before_dictation(app: &AppHandle) {
    let speech_loaded = app
        .try_state::<Arc<crate::managers::transcription::TranscriptionManager>>()
        .is_some_and(|tm| tm.is_model_loaded());
    if speech_loaded {
        return;
    }
    if let Some(server) = app.try_state::<Arc<LlamaServer>>() {
        server.yield_to_dictation();
    }
}

/// `RunEvent::Exit`: the Job Object would kill the server anyway; this is the
/// clean path. The Double Ctrl listener (a keyboard hook) goes too.
pub fn on_exit(app: &AppHandle) {
    double_tap::shutdown();
    fullscreen::shutdown();
    stop_server(app);
}

pub fn stop_server(app: &AppHandle) {
    if let Some(server) = app.try_state::<Arc<LlamaServer>>() {
        server.stop();
    }
}

/// Tray "Unload model": frees the VRAM of the translation model too. Off the
/// UI thread, since stopping waits for a start in progress to notice the cancel.
pub fn stop_server_in_background(app: &AppHandle) {
    let app = app.clone();
    std::thread::spawn(move || stop_server(&app));
}

#[tauri::command]
#[specta::specta]
pub fn get_translate_settings(app: AppHandle) -> SelTranslateSettings {
    settings::load(&app)
}

// A change of a server field stops a running server; the next use starts it with the new config.
#[tauri::command]
#[specta::specta]
pub async fn set_translate_settings(
    app: AppHandle,
    settings: SelTranslateSettings,
) -> Result<(), String> {
    settings.validate()?;
    let old = settings::load(&app);
    let mut settings = settings;
    settings.remember_models(&old.recent_models);
    settings::save(&app, &settings)?;
    if old.double_ctrl_enabled != settings.double_ctrl_enabled {
        // Joins the listener thread (~50 ms): not on the async runtime.
        let app = app.clone();
        tauri::async_runtime::spawn_blocking(move || double_tap::sync(&app))
            .await
            .map_err(|e| e.to_string())?;
    }
    if old.auto_pause_fullscreen != settings.auto_pause_fullscreen {
        let app = app.clone();
        tauri::async_runtime::spawn_blocking(move || fullscreen::sync(&app))
            .await
            .map_err(|e| e.to_string())?;
    }
    let server = app.state::<Arc<LlamaServer>>().inner().clone();
    server.set_idle_unload_ms(settings.idle_unload_ms());
    let server_changed = old.llama_server_path != settings.llama_server_path
        || old.model_path != settings.model_path
        || old.fast_model_path != settings.fast_model_path
        || old.gpu_layers != settings.gpu_layers
        || old.ctx_size != settings.ctx_size;
    if server_changed {
        tauri::async_runtime::spawn_blocking(move || server.stop())
            .await
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// The models the slots can choose from (I): next to the current ones, the
/// setup's models in the Hugging Face cache, the files picked before.
#[tauri::command]
#[specta::specta]
pub async fn translate_list_models(app: AppHandle) -> Vec<models::ModelEntry> {
    let cfg = settings::load(&app);
    tauri::async_runtime::spawn_blocking(move || {
        let mut extra: Vec<std::path::PathBuf> = Vec::new();
        if let Some((_, hub)) = settings::setup_roots() {
            for id in [setup::catalog::MODEL_FAST, setup::catalog::MODEL_QUALITY] {
                if let Some(file) =
                    setup::catalog::item(id).and_then(|item| setup::catalog::model_file(&hub, item))
                {
                    extra.push(file);
                }
            }
        }
        extra.extend(cfg.recent_models.iter().map(std::path::PathBuf::from));
        models::list_models(&[&cfg.fast_model_path, &cfg.model_path], &extra)
    })
    .await
    .unwrap_or_default()
}

/// The prompt of a preset with its placeholders: what Custom starts from (I).
#[tauri::command]
#[specta::specta]
pub fn translate_prompt_template(style: prompt::PromptStyle) -> String {
    prompt::preset_template(style)
}

/// Whether the Fast model file exists (the settings page hints when it does not).
#[tauri::command]
#[specta::specta]
pub fn translate_fast_available(app: AppHandle) -> bool {
    settings::load(&app).fast_available()
}

#[tauri::command]
#[specta::specta]
pub async fn translate_server_start(app: AppHandle) -> Result<ServerStatus, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let server = app.state::<Arc<LlamaServer>>().inner().clone();
        let _use = server.begin_use();
        // The Test button checks the slot the hotkey uses (Auto: Fast).
        ensure_server(&app, settings::load(&app).selection_slot.without_text())?;
        Ok(server.status())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
#[specta::specta]
pub async fn translate_server_stop(app: AppHandle) -> Result<(), String> {
    let server = app.state::<Arc<LlamaServer>>().inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        server.stop();
    })
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn translate_server_status(app: AppHandle) -> ServerStatus {
    app.state::<Arc<LlamaServer>>().status()
}
