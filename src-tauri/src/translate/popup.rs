//! Translation popup (B5): a frameless always-on-top window near the cursor
//! with the progress, the result (Copy / Replace / Close) or an error.
//!
//! The current view lives here; the page asks for it on load
//! (`translate_popup_state`) and then follows `translate-popup-state` events,
//! so a view sent before the webview finished loading is not lost.

use std::sync::Mutex;
use std::time::Duration;

use log::{info, warn};
use serde::Serialize;
use specta::Type;
#[cfg(not(target_os = "macos"))]
use tauri::WebviewWindowBuilder;
use tauri::{AppHandle, Emitter, Manager, WebviewUrl};
use tauri_plugin_clipboard_manager::ClipboardExt;

use super::settings::Slot;

#[cfg(target_os = "macos")]
#[allow(clippy::unused_unit)] // the macro wants an explicit `-> ()`
mod nspanel {
    use tauri::{AppHandle, Manager, Wry};
    use tauri_nspanel::PanelHandle;

    tauri_nspanel::tauri_panel! {
        // A panel that does not activate SelTranslate (the app the selection came
        // from stays active, so Replace needs no way back) but takes the keyboard.
        panel!(TranslatePopupPanel {
            config: {
                can_become_key_window: true,
                is_floating_panel: true
            }
        })
        panel_event!(TranslatePopupEvents {
            window_did_resign_key(notification: &NSNotification) -> ()
        })
    }

    /// Losing the key window (a click elsewhere) closes the popup, like losing
    /// the focus does elsewhere.
    pub fn close_on_resign_key(panel: &PanelHandle<Wry>, app: &AppHandle) {
        let events = TranslatePopupEvents::new();
        let closer = app.clone();
        events.window_did_resign_key(move |_| super::translate_popup_close(closer.clone()));
        panel.set_event_handler(Some(events.as_ref()));
        // The panel keeps only a weak reference to its delegate.
        std::mem::forget(events);
    }
}

#[cfg(target_os = "macos")]
use nspanel::TranslatePopupPanel;

pub const LABEL: &str = "translate_popup";
const EVENT: &str = "translate-popup-state";
/// Logical size of the working / result view; the text area scrolls.
const WIDTH: f64 = 440.0;
const HEIGHT: f64 = 260.0;
/// An error whose message alone says it all (one or two lines).
const ERROR_SIZE: (f64, f64) = (360.0, 104.0);
/// "Translation is not set up yet" with the Open settings button.
const NOT_SET_UP_SIZE: (f64, f64) = (360.0, 140.0);
/// An error with a backend detail under the message (scrolls if longer).
const ERROR_DETAIL_SIZE: (f64, f64) = (420.0, 200.0);
/// Logical distance between the cursor and the popup.
const CURSOR_GAP: f64 = 14.0;
/// How long Replace waits for the original window to come to the front.
const FOCUS_WAIT: Duration = Duration::from_millis(500);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    NoSelection,
    ClipboardBusy,
    TextTooLong,
    ServerFailed,
    TranslationFailed,
    ReplaceFailed,
    /// A long job (audio file, file translation) is running; wait (B8).
    JobRunning,
    /// The engine or the models are not installed yet (F.6): Open settings.
    NotSetUp,
    /// macOS: the Accessibility permission is missing (J.3): Open settings.
    NoAccessibility,
    Other,
}

#[derive(Debug, Clone, PartialEq, Serialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PopupView {
    Working {
        src: String,
        tgt: String,
        /// The model doing the work (the page says "larger model" for Quality).
        slot: Slot,
        /// Short name of the slot's model, shown next to the slot (I).
        model: String,
        starting_model: bool,
        done: u32,
        total: u32,
    },
    Done {
        src: String,
        tgt: String,
        slot: Slot,
        model: String,
        text: String,
        truncated: bool,
        clipboard_lost: bool,
        /// Show "Better translation": the same text through the Quality slot.
        can_improve: bool,
        /// "Better translation" or "Translate again" failed or was cancelled and
        /// the earlier result is shown again (G3).
        improve_failed: bool,
        /// The model was not asked: the same text was translated before. The
        /// page says so and offers "Translate again".
        from_cache: bool,
    },
    Error {
        code: ErrorCode,
        detail: String,
    },
}

/// Error codes whose backend detail the page does not show (keep in step with
/// `SELF_EXPLANATORY` in `TranslatePopup.tsx`).
fn error_shows_detail(code: ErrorCode, detail: &str) -> bool {
    !detail.is_empty()
        && !matches!(
            code,
            ErrorCode::NoSelection
                | ErrorCode::ClipboardBusy
                | ErrorCode::TextTooLong
                | ErrorCode::JobRunning
                | ErrorCode::NotSetUp
                | ErrorCode::NoAccessibility
        )
}

/// Logical window size for a view: a short error does not need the 440×260
/// box of a translation.
fn view_size(view: &PopupView) -> (f64, f64) {
    match view {
        PopupView::Error {
            code: ErrorCode::NotSetUp | ErrorCode::NoAccessibility,
            ..
        } => NOT_SET_UP_SIZE,
        PopupView::Error { code, detail } if error_shows_detail(*code, detail) => ERROR_DETAIL_SIZE,
        PopupView::Error { .. } => ERROR_SIZE,
        _ => (WIDTH, HEIGHT),
    }
}

struct Current {
    view: Option<PopupView>,
    /// Window the selection came from (target of Replace).
    hwnd: isize,
}

static CURRENT: Mutex<Current> = Mutex::new(Current {
    view: None,
    hwnd: 0,
});

fn set_current(view: PopupView, hwnd: Option<isize>) {
    let mut current = CURRENT.lock().unwrap_or_else(|e| e.into_inner());
    current.view = Some(view);
    if let Some(hwnd) = hwnd {
        current.hwnd = hwnd;
    }
}

/// Shows `view` in the popup next to the cursor and focuses it.
pub fn open(app: &AppHandle, view: PopupView, hwnd: isize) {
    set_current(view.clone(), Some(hwnd));
    let Some(window) = window(app) else {
        return;
    };
    let size = view_size(&view);
    let _ = window.set_size(tauri::LogicalSize::new(size.0, size.1));
    place_near_cursor(app, &window, size);
    show_and_focus(app, &window);
    let _ = window.emit_to(LABEL, EVENT, view);
}

#[cfg(not(target_os = "macos"))]
fn show_and_focus(_app: &AppHandle, window: &tauri::WebviewWindow) {
    let _ = window.show();
    let _ = window.set_focus();
}

/// The panel is shown on the main thread and becomes the key window without
/// activating the app.
#[cfg(target_os = "macos")]
fn show_and_focus(app: &AppHandle, _window: &tauri::WebviewWindow) {
    use tauri_nspanel::ManagerExt;
    let handle = app.clone();
    let _ = super::macos::on_main(app, move || {
        if let Ok(panel) = handle.get_webview_panel(LABEL) {
            panel.show_and_make_key();
        }
    });
}

/// Replaces the view of an open popup; a popup the user closed stays closed.
pub fn update(app: &AppHandle, view: PopupView) {
    set_current(view.clone(), None);
    if let Some(window) = app.get_webview_window(LABEL) {
        // The popup keeps its top-left corner when the view needs another size
        // (an error after "Starting model…").
        let size = view_size(&view);
        let _ = window.set_size(tauri::LogicalSize::new(size.0, size.1));
        let _ = window.emit_to(LABEL, EVENT, view);
    }
}

#[cfg(not(target_os = "macos"))]
fn hide(app: &AppHandle) {
    if let Some(window) = app.get_webview_window(LABEL) {
        let _ = window.hide();
    }
}

#[cfg(target_os = "macos")]
fn hide(app: &AppHandle) {
    use tauri_nspanel::ManagerExt;
    let handle = app.clone();
    let _ = super::macos::on_main(app, move || {
        if let Ok(panel) = handle.get_webview_panel(LABEL) {
            panel.hide();
        }
    });
}

/// The popup window, created hidden on first use.
fn window(app: &AppHandle) -> Option<tauri::WebviewWindow> {
    if let Some(window) = app.get_webview_window(LABEL) {
        return Some(window);
    }
    #[cfg(target_os = "macos")]
    return create_panel(app);
    #[cfg(not(target_os = "macos"))]
    create_window(app)
}

#[cfg(not(target_os = "macos"))]
fn create_window(app: &AppHandle) -> Option<tauri::WebviewWindow> {
    let mut builder = WebviewWindowBuilder::new(
        app,
        LABEL,
        WebviewUrl::App("src/translate-popup/index.html".into()),
    )
    .title("Translation")
    .inner_size(WIDTH, HEIGHT)
    .resizable(false)
    .maximizable(false)
    .minimizable(false)
    .decorations(false)
    .always_on_top(true)
    .skip_taskbar(true)
    .accept_first_mouse(true)
    .visible(false);
    if let Some(data_dir) = crate::portable::data_dir() {
        builder = builder.data_directory(data_dir.join("webview"));
    }
    match builder.build() {
        Ok(window) => Some(window),
        Err(e) => {
            warn!("Translate popup: failed to create the window: {e}");
            None
        }
    }
}

/// macOS: the popup is a non-activating floating panel (DESIGN J.5), also
/// visible over a full-screen app. Built on the main thread; losing the key
/// status (a click elsewhere) closes it like losing the focus does elsewhere.
#[cfg(target_os = "macos")]
fn create_panel(app: &AppHandle) -> Option<tauri::WebviewWindow> {
    use tauri_nspanel::{CollectionBehavior, PanelBuilder, PanelLevel, StyleMask};
    let handle = app.clone();
    let built = super::macos::on_main(app, move || {
        let panel = PanelBuilder::<_, TranslatePopupPanel>::new(&handle, LABEL)
            .url(WebviewUrl::App("src/translate-popup/index.html".into()))
            .title("Translation")
            .size(tauri::Size::Logical(tauri::LogicalSize {
                width: WIDTH,
                height: HEIGHT,
            }))
            .level(PanelLevel::Floating)
            .has_shadow(true)
            .style_mask(StyleMask::empty().borderless().nonactivating_panel())
            .collection_behavior(
                CollectionBehavior::new()
                    .can_join_all_spaces()
                    .full_screen_auxiliary(),
            )
            .with_window(|w| {
                let w = w
                    .decorations(false)
                    .resizable(false)
                    .skip_taskbar(true)
                    .accept_first_mouse(true)
                    .visible(false);
                match crate::portable::data_dir() {
                    Some(data_dir) => w.data_directory(data_dir.join("webview")),
                    None => w,
                }
            })
            .build();
        match panel {
            Ok(panel) => {
                nspanel::close_on_resign_key(&panel, &handle);
                panel.hide();
                true
            }
            Err(e) => {
                warn!("Translate popup: failed to create the panel: {e}");
                false
            }
        }
    });
    match built {
        Ok(true) => app.get_webview_window(LABEL),
        _ => None,
    }
}

/// Top-left corner for a `size` box next to `cursor` inside `area` (x, y, w, h):
/// below-right of the cursor, above it if there is no room below, shifted to
/// stay inside the area. All in physical pixels.
fn popup_origin(
    cursor: (i32, i32),
    gap: i32,
    size: (i32, i32),
    area: (i32, i32, i32, i32),
) -> (i32, i32) {
    let (cx, cy) = cursor;
    let (w, h) = size;
    let (ax, ay, aw, ah) = area;
    let mut x = cx + gap;
    let mut y = cy + gap;
    if y + h > ay + ah {
        y = cy - gap - h;
    }
    if x + w > ax + aw {
        x = ax + aw - w;
    }
    (x.max(ax), y.max(ay))
}

#[cfg(windows)]
fn place_near_cursor(app: &AppHandle, window: &tauri::WebviewWindow, logical: (f64, f64)) {
    use windows::Win32::UI::WindowsAndMessaging::{SetWindowPos, SWP_NOACTIVATE, SWP_NOZORDER};

    let Some(cursor) = crate::input::get_cursor_position(app) else {
        return;
    };
    // Cursor (GetCursorPos) and monitor bounds are both physical pixels.
    let monitor = app.available_monitors().ok().and_then(|monitors| {
        monitors.into_iter().find(|m| {
            let (p, s) = (m.position(), m.size());
            cursor.0 >= p.x
                && cursor.0 < p.x + s.width as i32
                && cursor.1 >= p.y
                && cursor.1 < p.y + s.height as i32
        })
    });
    let Some(monitor) = monitor else {
        return;
    };
    let scale = monitor.scale_factor();
    let area = monitor.work_area();
    let size = ((logical.0 * scale) as i32, (logical.1 * scale) as i32);
    let (x, y) = popup_origin(
        cursor,
        (CURSOR_GAP * scale) as i32,
        size,
        (
            area.position.x,
            area.position.y,
            area.size.width as i32,
            area.size.height as i32,
        ),
    );
    // SetWindowPos in physical pixels: Tauri would convert a PhysicalPosition
    // with the scale of the monitor the window is on now (see overlay.rs).
    if let Ok(hwnd) = window.hwnd() {
        // SAFETY: a valid window handle of this process.
        if let Err(e) = unsafe {
            SetWindowPos(
                hwnd,
                None,
                x,
                y,
                size.0,
                size.1,
                SWP_NOACTIVATE | SWP_NOZORDER,
            )
        } {
            warn!("Translate popup: SetWindowPos failed: {e}");
        }
    }
}

/// macOS: the cursor of enigo, the monitors and their work areas (the screen
/// without the menu bar and the Dock) are all converted to logical points,
/// where `popup_origin` works as it does in physical pixels on Windows.
#[cfg(target_os = "macos")]
fn place_near_cursor(app: &AppHandle, window: &tauri::WebviewWindow, logical: (f64, f64)) {
    let Some(cursor) = crate::input::get_cursor_position(app) else {
        return;
    };
    let monitor = app.available_monitors().ok().and_then(|monitors| {
        monitors.into_iter().find(|m| {
            let (p, s, k) = (m.position(), m.size(), m.scale_factor());
            let (x, y) = (f64::from(p.x) / k, f64::from(p.y) / k);
            let (w, h) = (f64::from(s.width) / k, f64::from(s.height) / k);
            let (cx, cy) = (f64::from(cursor.0), f64::from(cursor.1));
            cx >= x && cx < x + w && cy >= y && cy < y + h
        })
    });
    let Some(monitor) = monitor else {
        let _ = window.center();
        return;
    };
    let k = monitor.scale_factor();
    let area = monitor.work_area();
    let (x, y) = popup_origin(
        cursor,
        CURSOR_GAP as i32,
        (logical.0 as i32, logical.1 as i32),
        (
            (f64::from(area.position.x) / k) as i32,
            (f64::from(area.position.y) / k) as i32,
            (f64::from(area.size.width) / k) as i32,
            (f64::from(area.size.height) / k) as i32,
        ),
    );
    let _ = window.set_position(tauri::LogicalPosition::new(f64::from(x), f64::from(y)));
}

#[cfg(not(any(windows, target_os = "macos")))]
fn place_near_cursor(_app: &AppHandle, window: &tauri::WebviewWindow, _logical: (f64, f64)) {
    let _ = window.center();
}

/// Brings the window the selection came from back to the front; true once it is.
#[cfg(windows)]
fn focus_original(hwnd: isize) -> bool {
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, IsWindow, SetForegroundWindow,
    };
    let hwnd = HWND(hwnd as *mut core::ffi::c_void);
    // SAFETY: plain Win32 calls on a handle we only pass back to the OS.
    unsafe {
        if hwnd.is_invalid() || !IsWindow(Some(hwnd)).as_bool() {
            return false;
        }
        let _ = SetForegroundWindow(hwnd);
        let deadline = std::time::Instant::now() + FOCUS_WAIT;
        while GetForegroundWindow() != hwnd {
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    true
}

/// macOS: the panel never took the focus from the app (hwnd is its pid); wait
/// for it to be in front anyway, and bring it back if it is not.
#[cfg(target_os = "macos")]
fn focus_original(hwnd: isize) -> bool {
    use super::macos::{activate_pid, frontmost_pid};
    let pid = hwnd as i32;
    if pid <= 0 {
        return false;
    }
    let deadline = std::time::Instant::now() + FOCUS_WAIT;
    let mut asked = false;
    while frontmost_pid() != pid {
        if std::time::Instant::now() >= deadline {
            return false;
        }
        if !asked {
            asked = activate_pid(pid);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    true
}

#[cfg(not(any(windows, target_os = "macos")))]
fn focus_original(_hwnd: isize) -> bool {
    false
}

fn done_text() -> Option<(String, isize)> {
    let current = CURRENT.lock().unwrap_or_else(|e| e.into_inner());
    match &current.view {
        Some(PopupView::Done { text, .. }) => Some((text.clone(), current.hwnd)),
        _ => None,
    }
}

#[tauri::command]
#[specta::specta]
pub fn translate_popup_state() -> Option<PopupView> {
    CURRENT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .view
        .clone()
}

/// Close, x, Esc and focus loss. While the translation works this is Cancel:
/// the work stops instead of running on, invisibly, with the hotkey blocked.
/// While "Better translation" works the popup stays open: the work stops and
/// the Fast result comes back (G3).
#[tauri::command]
#[specta::specta]
pub fn translate_popup_close(app: AppHandle) {
    let working = matches!(
        CURRENT.lock().unwrap_or_else(|e| e.into_inner()).view,
        Some(PopupView::Working { .. })
    );
    if working {
        if !super::action::is_improving() {
            hide(&app);
        }
        super::action::request_cancel(&app);
    } else {
        hide(&app);
    }
}

/// "Better translation": translates the shown text again with the Quality model.
#[tauri::command]
#[specta::specta]
pub fn translate_popup_improve(app: AppHandle) -> Result<(), String> {
    super::action::improve(&app)
}

/// "Translate again": the shown text through the same model, asked afresh.
#[tauri::command]
#[specta::specta]
pub fn translate_popup_again(app: AppHandle) -> Result<(), String> {
    super::action::translate_again(&app)
}

/// A source language the popup offers when the detected one was wrong.
#[derive(Debug, Clone, Serialize, Type)]
pub struct LanguageOption {
    pub code: String,
    pub name: String,
}

#[tauri::command]
#[specta::specta]
pub fn translate_popup_languages() -> Vec<LanguageOption> {
    super::prompt::source_languages()
        .into_iter()
        .map(|(code, name)| LanguageOption {
            code: code.into(),
            name: name.into(),
        })
        .collect()
}

/// The shown text again with the source language picked in the popup.
#[tauri::command]
#[specta::specta]
pub fn translate_popup_set_source(app: AppHandle, code: String) -> Result<(), String> {
    super::action::change_source(&app, &code)
}

/// The shown text again with the target language picked in the popup.
#[tauri::command]
#[specta::specta]
pub fn translate_popup_set_target(app: AppHandle, code: String) -> Result<(), String> {
    super::action::change_target(&app, &code)
}

/// Copies the translation to the clipboard (on purpose: the user asked) and closes.
#[tauri::command]
#[specta::specta]
pub fn translate_popup_copy(app: AppHandle) -> Result<(), String> {
    let (text, _) = done_text().ok_or("No translation to copy")?;
    app.clipboard()
        .write_text(text)
        .map_err(|e| format!("Failed to copy: {e}"))?;
    hide(&app);
    Ok(())
}

/// Closes the popup, returns to the original window and puts the translation
/// over its selection.
#[tauri::command]
#[specta::specta]
pub async fn translate_popup_replace(app: AppHandle) -> Result<(), String> {
    let (text, hwnd) = done_text().ok_or("No translation to paste")?;
    hide(&app);
    tauri::async_runtime::spawn_blocking(move || {
        if !focus_original(hwnd) {
            let detail = "Could not return to the original window".to_string();
            warn!("Translate popup: {detail}");
            open(
                &app,
                PopupView::Error {
                    code: ErrorCode::ReplaceFailed,
                    detail: detail.clone(),
                },
                hwnd,
            );
            return Err(detail);
        }
        info!("Translate popup: replacing the selection");
        super::selection::paste_text(&app, &text)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;

    const AREA: (i32, i32, i32, i32) = (0, 0, 1920, 1040);

    #[test]
    fn popup_goes_below_right_of_the_cursor() {
        assert_eq!(popup_origin((100, 100), 10, (400, 200), AREA), (110, 110));
    }

    #[test]
    fn popup_flips_above_and_shifts_left_at_the_edges() {
        // Bottom edge: above the cursor.
        assert_eq!(popup_origin((100, 1000), 10, (400, 200), AREA), (110, 790));
        // Right edge: shifted left to fit.
        assert_eq!(popup_origin((1800, 100), 10, (400, 200), AREA), (1520, 110));
        // Bottom-right corner.
        assert_eq!(
            popup_origin((1900, 1030), 10, (400, 200), AREA),
            (1520, 820)
        );
    }

    #[test]
    fn popup_stays_inside_a_monitor_left_or_above_the_primary() {
        let area = (-1920, -200, 1920, 1080);
        assert_eq!(
            popup_origin((-1900, -190), 10, (400, 200), area),
            (-1890, -180)
        );
        // Too tall to fit above or below: pinned to the top of the area.
        assert_eq!(
            popup_origin((-100, -100), 10, (400, 2000), area),
            (-400, -200)
        );
    }

    fn error(code: ErrorCode, detail: &str) -> PopupView {
        PopupView::Error {
            code,
            detail: detail.into(),
        }
    }

    #[test]
    fn short_errors_get_a_compact_window() {
        for code in [
            ErrorCode::NoSelection,
            ErrorCode::ClipboardBusy,
            ErrorCode::TextTooLong,
            ErrorCode::JobRunning,
        ] {
            // The detail is not shown for these, so it does not enlarge the box.
            assert_eq!(view_size(&error(code, "")), ERROR_SIZE);
            assert_eq!(view_size(&error(code, "ignored")), ERROR_SIZE);
        }
        assert_eq!(
            view_size(&error(ErrorCode::TranslationFailed, "")),
            ERROR_SIZE
        );
    }

    #[test]
    fn not_set_up_has_room_for_the_button_and_no_detail() {
        assert_eq!(
            view_size(&error(ErrorCode::NotSetUp, "ignored")),
            NOT_SET_UP_SIZE
        );
        assert!(!error_shows_detail(ErrorCode::NotSetUp, "ignored"));
        assert!(ERROR_SIZE.1 < NOT_SET_UP_SIZE.1 && NOT_SET_UP_SIZE.1 < ERROR_DETAIL_SIZE.1);
    }

    #[test]
    fn error_with_a_shown_detail_gets_room_for_it() {
        assert_eq!(
            view_size(&error(ErrorCode::ServerFailed, "llama-server exited")),
            ERROR_DETAIL_SIZE
        );
        assert!(ERROR_SIZE.1 < ERROR_DETAIL_SIZE.1 && ERROR_DETAIL_SIZE.1 < HEIGHT);
    }

    #[test]
    fn working_and_result_views_keep_the_full_size() {
        let working = PopupView::Working {
            src: "ru".into(),
            tgt: "en".into(),
            slot: Slot::Quality,
            model: "Gemma 4 26B".into(),
            starting_model: true,
            done: 0,
            total: 1,
        };
        let done = PopupView::Done {
            src: "ru".into(),
            tgt: "en".into(),
            slot: Slot::Fast,
            model: "Qwen3.5 9B".into(),
            text: "Hello".into(),
            truncated: false,
            clipboard_lost: false,
            can_improve: true,
            improve_failed: false,
            from_cache: false,
        };
        assert_eq!(view_size(&working), (WIDTH, HEIGHT));
        assert_eq!(view_size(&done), (WIDTH, HEIGHT));
    }

    #[test]
    fn view_serializes_with_a_kind_tag() {
        let view = PopupView::Error {
            code: ErrorCode::TextTooLong,
            detail: "x".into(),
        };
        assert_eq!(
            serde_json::to_value(view).unwrap(),
            serde_json::json!({"kind": "error", "code": "text_too_long", "detail": "x"})
        );
    }
}
