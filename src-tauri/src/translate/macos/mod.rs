//! macOS side of the selection translation (DESIGN J): the pasteboard with the
//! user's contents snapshotted and restored, the keyboard chords, and a few
//! small helpers around the frontmost application and the main thread.

pub mod keys;
pub mod memory;
pub mod pasteboard;
pub mod screen;

use std::sync::mpsc;
use std::time::Duration;

use objc2_app_kit::{NSApplicationActivationOptions, NSRunningApplication, NSWorkspace};
use tauri::AppHandle;

/// How long `on_main` waits for the main thread before giving up.
const MAIN_THREAD_WAIT: Duration = Duration::from_secs(2);

/// Runs `f` on the main thread and returns its result. The Text Input Sources
/// API (keycode of a letter in the current layout) and the synthetic key
/// events must come from there. The caller is a worker thread (never the main
/// one), so waiting cannot deadlock; called on the main thread itself (a sync
/// command) it just runs `f`.
pub fn on_main<T: Send + 'static>(
    app: &AppHandle,
    f: impl FnOnce() -> T + Send + 'static,
) -> Result<T, String> {
    if objc2::MainThreadMarker::new().is_some() {
        return Ok(f());
    }
    let (tx, rx) = mpsc::channel();
    app.run_on_main_thread(move || {
        let _ = tx.send(f());
    })
    .map_err(|e| format!("The main thread is not available: {e}"))?;
    rx.recv_timeout(MAIN_THREAD_WAIT)
        .map_err(|_| "The main thread did not answer".to_string())
}

/// Pid of the frontmost application (0 if there is none). Stored in the
/// `hwnd` field of the capture result: "the window to return to" on macOS.
pub fn frontmost_pid() -> i32 {
    NSWorkspace::sharedWorkspace()
        .frontmostApplication()
        .map_or(0, |app| app.processIdentifier())
}

/// Brings the application with this pid to the front (fallback when the popup
/// is an ordinary window that took the focus away).
pub fn activate_pid(pid: i32) -> bool {
    NSRunningApplication::runningApplicationWithProcessIdentifier(pid)
        .is_some_and(|app| app.activateWithOptions(NSApplicationActivationOptions::empty()))
}

/// Whether this app may post and observe keyboard events (Accessibility).
pub fn accessibility_trusted() -> bool {
    handy_keys::check_accessibility()
}
