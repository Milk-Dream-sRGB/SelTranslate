//! SelTranslate: commands for the Audio file tab. Progress arrives as
//! `audio-file-progress` / `-done` / `-error` / `-cancelled` events.

use std::path::PathBuf;

use serde::Serialize;
use specta::Type;
use tauri::AppHandle;

use crate::audio_file;

// duration_secs: `None` when the container doesn't say (e.g. VBR mp3
// without a header); format: codec short name ("mp3", "aac", "flac", …).
#[derive(Serialize, Type)]
pub struct AudioFileInfo {
    duration_secs: Option<f64>,
    format: String,
}

#[tauri::command]
#[specta::specta]
pub fn probe_audio_file(path: String) -> Result<AudioFileInfo, String> {
    let (duration_secs, format) =
        audio_file::probe(&PathBuf::from(path)).map_err(|e| format!("{e:#}"))?;
    Ok(AudioFileInfo {
        duration_secs,
        format,
    })
}

/// Starts the job in the background and returns at once. `model_id`: a
/// downloaded model for this file; `None` = the dictation model.
#[tauri::command]
#[specta::specta]
pub fn transcribe_audio_file(
    app: AppHandle,
    path: String,
    model_id: Option<String>,
) -> Result<(), String> {
    audio_file::start(&app, PathBuf::from(path), model_id)
}

#[tauri::command]
#[specta::specta]
pub fn cancel_audio_file_transcription(app: AppHandle) -> Result<(), String> {
    audio_file::cancel(&app);
    Ok(())
}

/// Writes the transcript to a path picked in the Save dialog (the fs plugin
/// scope only covers app data).
#[tauri::command]
#[specta::specta]
pub fn save_audio_file_text(path: String, text: String) -> Result<(), String> {
    std::fs::write(&path, text).map_err(|e| format!("Failed to save {path}: {e}"))
}
