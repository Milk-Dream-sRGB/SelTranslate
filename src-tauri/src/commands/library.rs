//! SelTranslate: commands of the transcript library (Audio file tab, stage H).
//! Every command takes a bare file name; `library::existing_path` keeps it
//! inside the library folder. Changes are announced with `library-changed`.

use tauri::AppHandle;
use tauri_plugin_opener::OpenerExt;

use crate::library::{self, LibraryEntry, LibraryInfo, LibrarySettings};

fn dir(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    library::resolve_dir(app, &library::load_settings(app))
}

#[tauri::command]
#[specta::specta]
pub fn library_get_settings(app: AppHandle) -> LibraryInfo {
    library::info(&app)
}

/// `dir`: empty = the default folder. Files are not moved to a new folder.
#[tauri::command]
#[specta::specta]
pub fn library_set_settings(
    app: AppHandle,
    enabled: bool,
    dir: String,
) -> Result<LibraryInfo, String> {
    library::save_settings(&app, &LibrarySettings { enabled, dir })?;
    library::notify_changed(&app);
    Ok(library::info(&app))
}

#[tauri::command]
#[specta::specta]
pub fn library_list(app: AppHandle) -> Result<Vec<LibraryEntry>, String> {
    Ok(library::list(&dir(&app)?))
}

/// Returns the new file name.
#[tauri::command]
#[specta::specta]
pub fn library_rename(
    app: AppHandle,
    file_name: String,
    new_title: String,
) -> Result<String, String> {
    let new_name = library::rename(&dir(&app)?, &file_name, &new_title)?;
    library::notify_changed(&app);
    Ok(new_name)
}

#[tauri::command]
#[specta::specta]
pub fn library_delete(app: AppHandle, file_name: String) -> Result<(), String> {
    library::delete(&dir(&app)?, &file_name)?;
    library::notify_changed(&app);
    Ok(())
}

/// Opens the file in its default program (from Rust: the opener's JS scope
/// would not cover a folder the user picked).
#[tauri::command]
#[specta::specta]
pub fn library_open(app: AppHandle, file_name: String) -> Result<(), String> {
    let path = library::existing_path(&dir(&app)?, &file_name)?;
    app.opener()
        .open_path(path.to_string_lossy(), None::<String>)
        .map_err(|e| format!("Failed to open {file_name}: {e}"))
}

#[tauri::command]
#[specta::specta]
pub fn library_reveal(app: AppHandle, file_name: String) -> Result<(), String> {
    let path = library::existing_path(&dir(&app)?, &file_name)?;
    app.opener()
        .reveal_item_in_dir(&path)
        .map_err(|e| format!("Failed to show {file_name}: {e}"))
}
