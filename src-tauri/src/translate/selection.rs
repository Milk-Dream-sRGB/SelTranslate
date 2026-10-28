//! Capture of the selected text (DESIGN B.1): Ctrl+C confirmed by the
//! clipboard sequence number, with the user's clipboard snapshotted (every
//! byte-copyable format) before the copy and restored after it.
//!
//! The flow lives in `capture` over the `Backend` trait, so it is testable with
//! a fake clipboard; the Win32 backend is in `win`, the macOS one in `mac`.
// The Win32 clipboard helpers (CF_* ids, format selection, console classes) are
// tested everywhere but only used by `win`.
#![cfg_attr(not(windows), allow(dead_code))]

use std::fmt;
use std::time::{Duration, Instant};

use log::{info, warn};
use serde::Serialize;
use specta::Type;
use tauri::AppHandle;

// Standard clipboard format ids (winuser.h), as plain numbers so the pure
// logic below does not depend on the Win32 bindings.
const CF_BITMAP: u32 = 2;
const CF_METAFILEPICT: u32 = 3;
const CF_DIB: u32 = 8;
const CF_PALETTE: u32 = 9;
const CF_UNICODETEXT: u32 = 13;
const CF_ENHMETAFILE: u32 = 14;
const CF_DIBV5: u32 = 17;
const CF_OWNERDISPLAY: u32 = 0x80;
const CF_DSPTEXT: u32 = 0x81;
const CF_DSPBITMAP: u32 = 0x82;
const CF_DSPMETAFILEPICT: u32 = 0x83;
const CF_DSPENHMETAFILE: u32 = 0x8E;
/// CF_PRIVATEFIRST..=CF_GDIOBJLAST: handles of unknown kind / GDI objects.
const CF_PRIVATE_AND_GDI: std::ops::RangeInclusive<u32> = 0x200..=0x3FF;

/// Above this total the snapshot keeps only text and the bitmap.
const SNAPSHOT_LIMIT_BYTES: usize = 64 * 1024 * 1024;
const OPEN_ATTEMPTS: u32 = 10;
const OPEN_RETRY: Duration = Duration::from_millis(20);
const CHORD_HOLD_MS: u64 = 40;
/// Reads of CF_UNICODETEXT while the source app is still writing formats.
const READ_ATTEMPTS: u32 = 3;

struct Timings {
    modifiers: Duration,
    copy: Duration,
    step: Duration,
    reread: Duration,
}

const TIMINGS: Timings = Timings {
    // Held Alt/Shift/Win are released synthetically before the chord and Ctrl
    // by the chord itself, so waiting for the user's fingers only costs time.
    modifiers: Duration::from_millis(100),
    copy: Duration::from_millis(400),
    step: Duration::from_millis(10),
    reread: Duration::from_millis(30),
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedSelection {
    pub text: String,
    /// Foreground window before Ctrl+C (`GetForegroundWindow`), for Replace.
    /// On macOS: the pid of the frontmost application.
    pub hwnd: isize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureError {
    NoSelection,
    ClipboardBusy,
    /// macOS: no Accessibility permission, so Cmd+C cannot be sent (J.3).
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    NoAccessibility,
    Other(String),
}

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CaptureError::NoSelection => write!(f, "No text selected"),
            CaptureError::ClipboardBusy => {
                write!(f, "The clipboard is busy (another program is holding it)")
            }
            CaptureError::NoAccessibility => {
                write!(f, "SelTranslate needs the Accessibility permission")
            }
            CaptureError::Other(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for CaptureError {}

/// What happened to the user's clipboard during a capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ClipboardOutcome {
    /// The copy never happened (nothing selected) or we stopped before it.
    Untouched,
    Restored,
    /// The clipboard changed after our read (the user copied something): left as is.
    KeptNewerCopy,
    RestoreFailed,
}

pub struct CaptureReport {
    pub result: Result<CapturedSelection, CaptureError>,
    pub clipboard: ClipboardOutcome,
}

/// Window classes of consoles, where Ctrl+C without a selection interrupts the
/// running program; Ctrl+Insert only copies there (B5).
const CONSOLE_CLASSES: [&str; 3] = [
    "CASCADIA_HOSTING_WINDOW_CLASS", // Windows Terminal
    "ConsoleWindowClass",            // conhost
    "mintty",                        // Git Bash, MSYS2, Cygwin
];

/// Whether the copy chord for a window of this class is Ctrl+Insert (not Ctrl+C).
fn copies_with_ctrl_insert(window_class: &str) -> bool {
    CONSOLE_CLASSES.contains(&window_class)
}

/// Copies the selection of the foreground app and puts the user's clipboard
/// back. Blocking (up to ~1.5 s); call it off the UI thread.
pub fn capture_selection_report(app: &AppHandle) -> CaptureReport {
    #[cfg(windows)]
    {
        use tauri::Manager;
        let Some(enigo) = app.try_state::<crate::input::EnigoState>() else {
            return CaptureReport {
                result: Err(CaptureError::Other("Input is not initialized".into())),
                clipboard: ClipboardOutcome::Untouched,
            };
        };
        capture(&win::Win32 { enigo: &enigo }, &TIMINGS)
    }
    #[cfg(target_os = "macos")]
    {
        use tauri::Manager;
        // Without Accessibility the synthetic Cmd+C is dropped silently:
        // say so, and leave the pasteboard alone.
        if !super::macos::accessibility_trusted()
            || app.try_state::<crate::input::EnigoState>().is_none()
        {
            return CaptureReport {
                result: Err(CaptureError::NoAccessibility),
                clipboard: ClipboardOutcome::Untouched,
            };
        }
        capture(&mac::Mac { app }, &TIMINGS)
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        let _ = app;
        CaptureReport {
            result: Err(CaptureError::Other(
                "Selection capture is implemented only on Windows and macOS".into(),
            )),
            clipboard: ClipboardOutcome::Untouched,
        }
    }
}

/// Unassigned virtual key (like AutoHotkey's `#MenuMaskKey`): pressed while
/// the hotkey's Alt is still down, it makes the app see "Alt + another key"
/// instead of a bare Alt tap, so Alt↑ does not open the menu bar / KeyTips and
/// steal our Ctrl+C (DESIGN B.4a).
const VK_MENU_MASK: u32 = 0xE8;

/// Call from the hotkey handler, before the capture thread starts.
pub fn mask_menu_activation(app: &AppHandle) {
    #[cfg(windows)]
    {
        use enigo::{Direction, Key, Keyboard};
        use tauri::Manager;
        let Some(enigo) = app.try_state::<crate::input::EnigoState>() else {
            return;
        };
        let Ok(mut enigo) = enigo.0.lock() else {
            return;
        };
        // Consoles have no Alt menu, and Windows Terminal drops the selection on any key.
        if copies_with_ctrl_insert(&win::foreground_class()) {
            return;
        }
        if let Err(e) = enigo.key(Key::Other(VK_MENU_MASK), Direction::Click) {
            warn!("Selection: failed to send the menu mask key: {e}");
        }
    }
    #[cfg(not(windows))]
    let _ = app;
}

/// How Replace puts the translation over the selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PasteRoute {
    /// Typed with SendInput; the clipboard is not touched.
    Type,
    /// Through the clipboard with this chord; the user's clipboard is restored.
    CtrlV,
    CtrlShiftV,
    ShiftInsert,
}

/// Only Handy's Paste Method is honoured (B5): its dictation extras (trailing
/// space, auto submit, copy to clipboard) do not apply to a translation.
/// Typing turns "\n" into Enter, which sends the message in chats, so
/// multi-line text always goes through the clipboard.
fn paste_route(method: crate::settings::PasteMethod, text: &str) -> PasteRoute {
    use crate::settings::PasteMethod;
    match method {
        PasteMethod::Direct if !text.contains(['\n', '\r']) => PasteRoute::Type,
        PasteMethod::CtrlShiftV => PasteRoute::CtrlShiftV,
        PasteMethod::ShiftInsert => PasteRoute::ShiftInsert,
        // CtrlV, multi-line Direct, and None / ExternalScript (Replace was asked for explicitly).
        _ => PasteRoute::CtrlV,
    }
}

/// Least time the target app gets to read the pasted text before the user's
/// clipboard is put back.
const PASTE_SETTLE_MS: u64 = 300;

/// Replaces the selection of the foreground app with `text`. Blocking.
pub fn paste_text(app: &AppHandle, text: &str) -> Result<(), String> {
    #[cfg(windows)]
    {
        use tauri::Manager;
        let settings = crate::settings::get_settings(app);
        let route = paste_route(settings.paste_method, text);
        let enigo = app
            .try_state::<crate::input::EnigoState>()
            .ok_or("Input is not initialized")?;
        let with_enigo = |f: &dyn Fn(&mut enigo::Enigo) -> Result<(), String>| {
            let mut enigo = enigo
                .0
                .lock()
                .map_err(|_| "Input state is poisoned".to_string())?;
            f(&mut enigo)
        };
        let chord = match route {
            PasteRoute::Type => {
                return with_enigo(&|e| crate::input::paste_text_direct(e, text));
            }
            PasteRoute::CtrlV => crate::input::send_paste_ctrl_v,
            PasteRoute::CtrlShiftV => crate::input::send_paste_ctrl_shift_v,
            PasteRoute::ShiftInsert => crate::input::send_paste_shift_insert,
        };
        let snapshot = win::take_snapshot().map_err(|e| e.to_string())?;
        let ours = win::put_text(text).map_err(|e| e.to_string())?;
        std::thread::sleep(Duration::from_millis(settings.paste_delay_ms));
        let pasted = with_enigo(&|e| chord(e, CHORD_HOLD_MS));
        std::thread::sleep(Duration::from_millis(
            settings.paste_delay_after_ms.max(PASTE_SETTLE_MS),
        ));
        match win::restore_snapshot(&snapshot, Some(ours)) {
            Ok(true) => {}
            Ok(false) => info!("Replace: the clipboard changed meanwhile, left as is"),
            Err(e) => warn!("Replace: failed to restore the clipboard: {e}"),
        }
        info!(
            "Replace: pasted {} chars via {route:?}",
            text.chars().count()
        );
        pasted
    }
    #[cfg(target_os = "macos")]
    {
        use super::macos::{self, pasteboard};
        use tauri::Manager;
        let settings = crate::settings::get_settings(app);
        // Typing keeps the pasteboard untouched; every other route is Cmd+V
        // (there is no Ctrl+Shift+V or Shift+Insert on a Mac).
        let route = paste_route(settings.paste_method, text);
        let handle = app.clone();
        if route == PasteRoute::Type {
            let text = text.to_string();
            return macos::on_main(app, move || {
                let state = handle
                    .try_state::<crate::input::EnigoState>()
                    .ok_or("Input is not initialized")?;
                let mut enigo = state
                    .0
                    .lock()
                    .map_err(|_| "Input state is poisoned".to_string())?;
                crate::input::paste_text_direct(&mut enigo, &text)
            })?;
        }
        let snapshot = pasteboard::take_snapshot()?;
        let ours = pasteboard::put_text(text)?;
        std::thread::sleep(Duration::from_millis(settings.paste_delay_ms));
        let pasted = macos::on_main(app, move || {
            let state = handle
                .try_state::<crate::input::EnigoState>()
                .ok_or("Input is not initialized")?;
            let mut enigo = state
                .0
                .lock()
                .map_err(|_| "Input state is poisoned".to_string())?;
            crate::input::send_paste_ctrl_v(&mut enigo, CHORD_HOLD_MS)
        })
        .and_then(|result| result);
        std::thread::sleep(Duration::from_millis(
            settings.paste_delay_after_ms.max(PASTE_SETTLE_MS),
        ));
        match pasteboard::restore_snapshot(&snapshot, ours) {
            Ok(true) => {}
            Ok(false) => info!("Replace: the pasteboard changed meanwhile, left as is"),
            Err(e) => warn!("Replace: failed to restore the pasteboard: {e}"),
        }
        info!(
            "Replace: pasted {} chars via {route:?}",
            text.chars().count()
        );
        pasted
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        let _ = (app, text);
        Err("Replace is implemented only on Windows and macOS".into())
    }
}

/// Everything the capture needs from the OS.
trait Backend {
    type Snapshot;
    fn modifiers_down(&self) -> bool;
    /// Names of the modifiers held now, for the log.
    fn held_modifiers(&self) -> String {
        String::new()
    }
    fn foreground_window(&self) -> isize;
    fn snapshot(&self) -> Result<Self::Snapshot, CaptureError>;
    fn sequence(&self) -> u32;
    fn send_copy(&self) -> Result<(), String>;
    /// The copy on the clipboard now is an editor's "no selection → copy the
    /// line" (VS Code's `vscode-editor-data` with `isFromEmptySelection`).
    fn copied_from_empty_selection(&self) -> bool {
        false
    }
    /// CF_UNICODETEXT (None if absent) and the sequence number after reading
    /// it (reading may make the owner render a delayed format).
    fn read_text(&self) -> Result<(Option<String>, u32), CaptureError>;
    /// Writes the snapshot back unless the sequence number is no longer
    /// `expected` (checked with the clipboard open). `Ok(false)` = skipped.
    fn restore(&self, snapshot: &Self::Snapshot, expected: u32) -> Result<bool, CaptureError>;
}

fn capture<B: Backend>(backend: &B, timings: &Timings) -> CaptureReport {
    let started = Instant::now();
    let untouched = |result| CaptureReport {
        result,
        clipboard: ClipboardOutcome::Untouched,
    };

    if poll_until(timings.modifiers, timings.step, || {
        (!backend.modifiers_down()).then_some(())
    })
    .is_none()
    {
        info!(
            "Selection: modifiers still held after {} ms ({}), releasing them",
            timings.modifiers.as_millis(),
            backend.held_modifiers()
        );
    }
    let hwnd = backend.foreground_window();
    let snapshot = match backend.snapshot() {
        Ok(snapshot) => snapshot,
        Err(e) => return untouched(Err(e)),
    };
    let seq0 = backend.sequence();
    let chord = backend.send_copy();
    let copied = poll_until(timings.copy, timings.step, || {
        (backend.sequence() != seq0).then_some(())
    });
    if copied.is_none() {
        let result = match chord {
            Ok(()) => Err(CaptureError::NoSelection),
            Err(e) => Err(CaptureError::Other(format!("Failed to send Ctrl+C: {e}"))),
        };
        info!(
            "Selection: clipboard did not change ({} ms), nothing selected",
            started.elapsed().as_millis()
        );
        return untouched(result);
    }

    // VS Code copies the whole line when nothing is selected; it marks that
    // copy in its own clipboard format. Checked before the text is read.
    let (result, seq_ours) = if backend.copied_from_empty_selection() {
        info!("Selection: the app copied its current line (empty selection)");
        (Err(CaptureError::NoSelection), backend.sequence())
    } else {
        match read_copied_text(backend, timings) {
            Ok((text, seq)) => (selection_text(text).ok_or(CaptureError::NoSelection), seq),
            Err(e) => (Err(e), backend.sequence()),
        }
    };
    let clipboard = match backend.restore(&snapshot, seq_ours) {
        Ok(true) => ClipboardOutcome::Restored,
        Ok(false) => ClipboardOutcome::KeptNewerCopy,
        Err(e) => {
            warn!("Selection: failed to restore the clipboard: {e}");
            ClipboardOutcome::RestoreFailed
        }
    };
    info!(
        "Selection: {} in {} ms, clipboard {:?}",
        match &result {
            Ok(text) => format!("{} chars", text.chars().count()),
            Err(e) => e.to_string(),
        },
        started.elapsed().as_millis(),
        clipboard
    );
    CaptureReport {
        result: result.map(|text| CapturedSelection { text, hwnd }),
        clipboard,
    }
}

/// Reads the copied text; if it is not there yet but the clipboard keeps
/// changing (an app writing its formats in several goes), reads again.
fn read_copied_text<B: Backend>(
    backend: &B,
    timings: &Timings,
) -> Result<(Option<String>, u32), CaptureError> {
    let mut attempt = 1;
    loop {
        let (text, seq) = backend.read_text()?;
        if text.is_some() || attempt >= READ_ATTEMPTS {
            return Ok((text, seq));
        }
        std::thread::sleep(timings.reread);
        if backend.sequence() == seq {
            return Ok((None, seq));
        }
        attempt += 1;
    }
}

/// Calls `f` until it returns `Some` or `timeout` passes (at least once).
fn poll_until<T>(timeout: Duration, step: Duration, mut f: impl FnMut() -> Option<T>) -> Option<T> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(value) = f() {
            return Some(value);
        }
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(step);
    }
}

/// Text worth translating: present and not only whitespace.
fn selection_text(text: Option<String>) -> Option<String> {
    text.filter(|t| !t.trim().is_empty())
}

/// Formats whose handle is plain global memory. GDI handles (bitmaps,
/// metafiles, palettes) are skipped: Windows synthesizes them from CF_DIB.
fn is_byte_copyable(format: u32) -> bool {
    !matches!(
        format,
        CF_BITMAP
            | CF_METAFILEPICT
            | CF_PALETTE
            | CF_ENHMETAFILE
            | CF_OWNERDISPLAY
            | CF_DSPTEXT
            | CF_DSPBITMAP
            | CF_DSPMETAFILEPICT
            | CF_DSPENHMETAFILE
    ) && !CF_PRIVATE_AND_GDI.contains(&format)
}

/// Which of the `(format, size)` pairs go into the snapshot, in clipboard
/// order: all of them within `limit`, otherwise only text and the bitmap.
fn select_formats(sizes: &[(u32, usize)], limit: usize) -> Vec<u32> {
    let total = sizes
        .iter()
        .fold(0usize, |sum, &(_, size)| sum.saturating_add(size));
    if total <= limit {
        return sizes.iter().map(|&(format, _)| format).collect();
    }
    let mut kept = Vec::new();
    let mut used = 0usize;
    for priority in [CF_UNICODETEXT, CF_DIB, CF_DIBV5] {
        if let Some(&(format, size)) = sizes.iter().find(|&&(format, _)| format == priority) {
            if used.saturating_add(size) <= limit {
                kept.push(format);
                used += size;
            }
        }
    }
    sizes
        .iter()
        .map(|&(format, _)| format)
        .filter(|format| kept.contains(format))
        .collect()
}

/// Whether a payload of the `vscode-editor-data` clipboard format (the format
/// VS Code adds to every copy; UTF-8 JSON, maybe NUL-padded) says the copy
/// came from an empty selection, i.e. VS Code copied the whole line.
fn is_empty_selection_copy(editor_data: &[u8]) -> bool {
    let end = editor_data
        .iter()
        .rposition(|&b| b != 0)
        .map_or(0, |last| last + 1);
    serde_json::from_slice::<serde_json::Value>(&editor_data[..end])
        .ok()
        .and_then(|data| data.get("isFromEmptySelection")?.as_bool())
        .unwrap_or(false)
}

/// CF_UNICODETEXT bytes (UTF-16LE, NUL-terminated) to a string.
fn utf16_text(bytes: &[u8]) -> String {
    let units: Vec<u16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&pair| u16::from_le_bytes(pair))
        .take_while(|&unit| unit != 0)
        .collect();
    String::from_utf16_lossy(&units)
}

#[cfg(target_os = "macos")]
mod mac {
    use super::*;
    use crate::translate::macos::{self, keys, pasteboard};

    pub struct Mac<'a> {
        pub app: &'a AppHandle,
    }

    impl Backend for Mac<'_> {
        type Snapshot = pasteboard::Snapshot;

        fn modifiers_down(&self) -> bool {
            keys::modifiers_down()
        }

        fn held_modifiers(&self) -> String {
            keys::held_modifiers()
        }

        fn foreground_window(&self) -> isize {
            macos::frontmost_pid() as isize
        }

        fn snapshot(&self) -> Result<Self::Snapshot, CaptureError> {
            pasteboard::take_snapshot().map_err(CaptureError::Other)
        }

        fn sequence(&self) -> u32 {
            pasteboard::change_count()
        }

        fn send_copy(&self) -> Result<(), String> {
            keys::send_copy(self.app)
        }

        fn read_text(&self) -> Result<(Option<String>, u32), CaptureError> {
            let text = pasteboard::read_text();
            Ok((text, pasteboard::change_count()))
        }

        fn restore(&self, snapshot: &Self::Snapshot, expected: u32) -> Result<bool, CaptureError> {
            pasteboard::restore_snapshot(snapshot, expected).map_err(CaptureError::Other)
        }
    }
}

#[cfg(windows)]
mod win {
    use std::time::Instant;

    use enigo::{Direction, Key, Keyboard};
    use log::info;
    use windows::core::w;
    use windows::Win32::Foundation::{GlobalFree, HANDLE, HGLOBAL};
    use windows::Win32::System::DataExchange::{
        CloseClipboard, EmptyClipboard, EnumClipboardFormats, GetClipboardData,
        GetClipboardSequenceNumber, OpenClipboard, RegisterClipboardFormatW, SetClipboardData,
    };
    use windows::Win32::System::Memory::{
        GlobalAlloc, GlobalLock, GlobalSize, GlobalUnlock, GMEM_MOVEABLE,
    };
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, VIRTUAL_KEY, VK_LCONTROL, VK_LMENU, VK_LSHIFT, VK_LWIN, VK_RCONTROL,
        VK_RMENU, VK_RSHIFT, VK_RWIN,
    };
    use windows::Win32::UI::WindowsAndMessaging::{GetClassNameW, GetForegroundWindow};

    use super::{
        copies_with_ctrl_insert, is_byte_copyable, select_formats, utf16_text, Backend,
        CaptureError, CF_UNICODETEXT, CHORD_HOLD_MS, OPEN_ATTEMPTS, OPEN_RETRY,
        SNAPSHOT_LIMIT_BYTES,
    };
    use crate::input::EnigoState;

    const VK_C: u32 = 0x43;
    const VK_INSERT: u32 = 0x2D;

    pub struct Win32<'a> {
        pub enigo: &'a EnigoState,
    }

    /// Byte copies of the clipboard formats, in clipboard order.
    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    pub struct Snapshot {
        pub formats: Vec<(u32, Vec<u8>)>,
    }

    /// Open clipboard; closed on drop.
    struct Open;

    impl Drop for Open {
        fn drop(&mut self) {
            // SAFETY: only constructed after a successful OpenClipboard.
            let _ = unsafe { CloseClipboard() };
        }
    }

    fn open() -> Result<Open, CaptureError> {
        for attempt in 1..=OPEN_ATTEMPTS {
            // SAFETY: plain Win32 call; a NULL owner associates the clipboard with this task.
            if unsafe { OpenClipboard(None) }.is_ok() {
                return Ok(Open);
            }
            if attempt < OPEN_ATTEMPTS {
                std::thread::sleep(OPEN_RETRY);
            }
        }
        Err(CaptureError::ClipboardBusy)
    }

    pub fn sequence() -> u32 {
        // SAFETY: no arguments, no preconditions.
        unsafe { GetClipboardSequenceNumber() }
    }

    fn exclude_from_history_format() -> u32 {
        // SAFETY: valid NUL-terminated wide string literal.
        unsafe { RegisterClipboardFormatW(w!("ExcludeClipboardContentFromMonitorProcessing")) }
    }

    /// Bytes of a global-memory clipboard handle. Clipboard must be open.
    unsafe fn global_bytes(handle: HANDLE) -> Option<Vec<u8>> {
        let hg = HGLOBAL(handle.0);
        let size = GlobalSize(hg);
        if size == 0 {
            return None;
        }
        let ptr = GlobalLock(hg) as *const u8;
        if ptr.is_null() {
            return None;
        }
        let data = std::slice::from_raw_parts(ptr, size).to_vec();
        let _ = GlobalUnlock(hg);
        Some(data)
    }

    pub fn take_snapshot() -> Result<Snapshot, CaptureError> {
        let started = Instant::now();
        let _open = open()?;
        // SAFETY: the clipboard is open for the whole block; handles from
        // GetClipboardData stay valid until it is closed and are only read.
        unsafe {
            let mut handles = Vec::new();
            let mut format = 0u32;
            loop {
                format = EnumClipboardFormats(format);
                if format == 0 {
                    break;
                }
                if !is_byte_copyable(format) {
                    continue;
                }
                if let Ok(handle) = GetClipboardData(format) {
                    let size = GlobalSize(HGLOBAL(handle.0));
                    if size > 0 {
                        handles.push((format, handle, size));
                    }
                }
            }
            let sizes: Vec<(u32, usize)> = handles.iter().map(|&(f, _, s)| (f, s)).collect();
            let selected = select_formats(&sizes, SNAPSHOT_LIMIT_BYTES);
            let total: usize = sizes.iter().map(|&(_, s)| s).sum();
            if selected.len() < sizes.len() {
                log::warn!(
                    "Selection: clipboard is {total} bytes, over the snapshot limit; keeping only text and bitmap"
                );
            }
            let formats: Vec<(u32, Vec<u8>)> = handles
                .iter()
                .filter(|(format, _, _)| selected.contains(format))
                .filter_map(|&(format, handle, _)| Some((format, global_bytes(handle)?)))
                .collect();
            info!(
                "Selection: clipboard snapshot of {} formats, {} bytes in {} ms",
                formats.len(),
                formats.iter().map(|(_, d)| d.len()).sum::<usize>(),
                started.elapsed().as_millis()
            );
            Ok(Snapshot { formats })
        }
    }

    /// Empties the clipboard and puts `formats` on it (plus the clipboard
    /// history opt-out marker if `exclude_from_history`). Clipboard must be open.
    unsafe fn write_formats(
        formats: &[(u32, Vec<u8>)],
        exclude_from_history: bool,
    ) -> Result<(), String> {
        EmptyClipboard().map_err(|e| format!("EmptyClipboard failed: {e}"))?;
        if formats.is_empty() {
            return Ok(());
        }
        let marker = exclude_from_history_format();
        let mut all: Vec<(u32, &[u8])> = formats.iter().map(|(f, d)| (*f, d.as_slice())).collect();
        let marker_value = 1u32.to_le_bytes();
        if exclude_from_history && marker != 0 && !formats.iter().any(|(f, _)| *f == marker) {
            all.push((marker, &marker_value));
        }
        for (format, data) in all {
            if data.is_empty() {
                continue;
            }
            let Ok(hg) = GlobalAlloc(GMEM_MOVEABLE, data.len()) else {
                return Err(format!("GlobalAlloc of {} bytes failed", data.len()));
            };
            let ptr = GlobalLock(hg) as *mut u8;
            if ptr.is_null() {
                let _ = GlobalFree(Some(hg));
                return Err("GlobalLock failed".into());
            }
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            let _ = GlobalUnlock(hg);
            // On success the clipboard owns the memory.
            if let Err(e) = SetClipboardData(format, Some(HANDLE(hg.0))) {
                let _ = GlobalFree(Some(hg));
                log::warn!("Selection: SetClipboardData({format}) failed: {e}");
            }
        }
        Ok(())
    }

    pub fn restore_snapshot(
        snapshot: &Snapshot,
        expected: Option<u32>,
    ) -> Result<bool, CaptureError> {
        let _open = open()?;
        if expected.is_some_and(|seq| sequence() != seq) {
            return Ok(false);
        }
        // SAFETY: the clipboard is open.
        unsafe { write_formats(&snapshot.formats, true) }.map_err(CaptureError::Other)?;
        Ok(true)
    }

    /// Puts `text` on the clipboard (with the history opt-out marker) and
    /// returns the sequence number after the clipboard is closed.
    pub fn put_text(text: &str) -> Result<u32, CaptureError> {
        let bytes: Vec<u8> = text
            .encode_utf16()
            .chain(std::iter::once(0))
            .flat_map(u16::to_le_bytes)
            .collect();
        {
            let _open = open()?;
            // SAFETY: the clipboard is open.
            unsafe { write_formats(&[(CF_UNICODETEXT, bytes)], true) }
                .map_err(CaptureError::Other)?;
        }
        Ok(sequence())
    }

    /// `vscode-editor-data` of the clipboard says "copied from an empty selection".
    fn clipboard_has_empty_selection_marker() -> bool {
        // SAFETY: valid NUL-terminated wide string literal.
        let format = unsafe { RegisterClipboardFormatW(w!("vscode-editor-data")) };
        if format == 0 {
            return false;
        }
        let Ok(_open) = open() else {
            return false;
        };
        // SAFETY: the clipboard is open; the handle is only read while it is.
        unsafe {
            GetClipboardData(format)
                .ok()
                .and_then(|handle| global_bytes(handle))
                .is_some_and(|bytes| super::is_empty_selection_copy(&bytes))
        }
    }

    fn read_unicode_text() -> Result<(Option<String>, u32), CaptureError> {
        let _open = open()?;
        // SAFETY: the clipboard is open; the handle is only read while it is.
        let text = unsafe {
            GetClipboardData(CF_UNICODETEXT)
                .ok()
                .and_then(|handle| global_bytes(handle))
                .map(|bytes| utf16_text(&bytes))
        };
        Ok((text, sequence()))
    }

    /// Alt, Shift and Win still held by the user (the hotkey's keys) would turn
    /// our Ctrl+C into Ctrl+Alt+C — AltGr+C, which types "с" over the
    /// selection in Qt apps (Telegram). Sends their key-up (after the menu
    /// mask, so a lone Alt/Win release opens no menu; not in consoles, see
    /// `mask_menu_activation`); the physical release later is a no-op. Ctrl
    /// itself is released by the chord.
    fn release_held_modifiers(enigo: &mut enigo::Enigo, console: bool) {
        let held: Vec<(VIRTUAL_KEY, &str)> = MODIFIERS
            .iter()
            .copied()
            .filter(|&(vk, _)| vk != VK_LCONTROL && vk != VK_RCONTROL && is_down(vk))
            .collect();
        if held.is_empty() {
            return;
        }
        let names: Vec<&str> = held.iter().map(|&(_, name)| name).collect();
        info!("Selection: releasing {} before copying", names.join("+"));
        if !console {
            let _ = enigo.key(Key::Other(super::VK_MENU_MASK), Direction::Click);
        }
        for (vk, name) in held {
            if let Err(e) = enigo.key(Key::Other(vk.0 as u32), Direction::Release) {
                log::warn!("Selection: failed to release {name}: {e}");
            }
        }
    }

    /// Class name of the foreground window ("" if unknown).
    pub fn foreground_class() -> String {
        let mut buf = [0u16; 256];
        // SAFETY: plain Win32 calls; the buffer outlives the call.
        let len = unsafe { GetClassNameW(GetForegroundWindow(), &mut buf) };
        String::from_utf16_lossy(&buf[..len.max(0) as usize])
    }

    /// Ctrl+C, or Ctrl+Insert in consoles (see `CONSOLE_CLASSES`).
    fn send_copy_chord(enigo: &EnigoState) -> Result<(), String> {
        let class = foreground_class();
        let console = copies_with_ctrl_insert(&class);
        let (key, name) = if console {
            (VK_INSERT, "Insert")
        } else {
            (VK_C, "C")
        };
        if key == VK_INSERT {
            info!("Selection: console window ({class}), copying with Ctrl+Insert");
        }
        let mut enigo = enigo
            .0
            .lock()
            .map_err(|_| "Input state is poisoned".to_string())?;
        release_held_modifiers(&mut enigo, console);
        enigo
            .key(Key::Control, Direction::Press)
            .map_err(|e| format!("Failed to press Ctrl: {e}"))?;
        let click = enigo
            .key(Key::Other(key), Direction::Click)
            .map_err(|e| format!("Failed to click {name}: {e}"));
        std::thread::sleep(std::time::Duration::from_millis(CHORD_HOLD_MS));
        // Release even if the click failed, so Ctrl is never left down.
        let release = enigo
            .key(Key::Control, Direction::Release)
            .map_err(|e| format!("Failed to release Ctrl: {e}"));
        click.and(release)
    }

    const MODIFIERS: [(VIRTUAL_KEY, &str); 8] = [
        (VK_LCONTROL, "LCtrl"),
        (VK_RCONTROL, "RCtrl"),
        (VK_LSHIFT, "LShift"),
        (VK_RSHIFT, "RShift"),
        (VK_LMENU, "LAlt"),
        (VK_RMENU, "RAlt"),
        (VK_LWIN, "LWin"),
        (VK_RWIN, "RWin"),
    ];

    fn is_down(vk: VIRTUAL_KEY) -> bool {
        // SAFETY: plain Win32 call; the high bit means "down now".
        (unsafe { GetAsyncKeyState(vk.0 as i32) }) as u16 & 0x8000 != 0
    }

    impl Backend for Win32<'_> {
        type Snapshot = Snapshot;

        fn modifiers_down(&self) -> bool {
            MODIFIERS.iter().any(|&(vk, _)| is_down(vk))
        }

        fn held_modifiers(&self) -> String {
            let held: Vec<&str> = MODIFIERS
                .iter()
                .filter(|&&(vk, _)| is_down(vk))
                .map(|&(_, name)| name)
                .collect();
            held.join("+")
        }

        fn foreground_window(&self) -> isize {
            // SAFETY: no arguments, no preconditions.
            unsafe { GetForegroundWindow() }.0 as isize
        }

        fn snapshot(&self) -> Result<Snapshot, CaptureError> {
            take_snapshot()
        }

        fn sequence(&self) -> u32 {
            sequence()
        }

        fn send_copy(&self) -> Result<(), String> {
            send_copy_chord(self.enigo)
        }

        fn copied_from_empty_selection(&self) -> bool {
            clipboard_has_empty_selection_marker()
        }

        fn read_text(&self) -> Result<(Option<String>, u32), CaptureError> {
            read_unicode_text()
        }

        fn restore(&self, snapshot: &Snapshot, expected: u32) -> Result<bool, CaptureError> {
            restore_snapshot(snapshot, Some(expected))
        }
    }

    #[cfg(test)]
    pub mod testing {
        use super::*;
        use windows::Win32::System::DataExchange::CountClipboardFormats;

        /// Puts `formats` on the clipboard like an ordinary app copy (no history marker).
        pub fn set_formats(formats: &[(u32, Vec<u8>)]) {
            let _open = open().expect("open clipboard");
            unsafe { write_formats(formats, false) }.expect("write formats");
        }

        pub fn format_count() -> i32 {
            let _open = open().expect("open clipboard");
            unsafe { CountClipboardFormats() }
        }

        pub fn register(name: &str) -> u32 {
            let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
            unsafe { RegisterClipboardFormatW(windows::core::PCWSTR(wide.as_ptr())) }
        }

        pub fn marker() -> u32 {
            exclude_from_history_format()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    #[test]
    fn gdi_and_owner_display_formats_are_skipped() {
        for format in [
            CF_BITMAP,
            CF_METAFILEPICT,
            CF_PALETTE,
            CF_ENHMETAFILE,
            CF_OWNERDISPLAY,
            CF_DSPBITMAP,
            CF_DSPENHMETAFILE,
            0x200,
            0x300,
            0x3FF,
        ] {
            assert!(!is_byte_copyable(format), "{format:#x}");
        }
        // Text, locale, DIB, DIBV5, HDROP, a registered format (HTML, RTF, …).
        for format in [
            1,
            7,
            CF_DIB,
            CF_UNICODETEXT,
            15,
            16,
            CF_DIBV5,
            0x400,
            0xC0A1,
        ] {
            assert!(is_byte_copyable(format), "{format:#x}");
        }
    }

    #[test]
    fn snapshot_keeps_everything_within_the_limit() {
        let sizes = [(0xC100, 50), (CF_UNICODETEXT, 20), (CF_DIB, 30)];
        assert_eq!(
            select_formats(&sizes, 100),
            vec![0xC100, CF_UNICODETEXT, CF_DIB]
        );
        assert_eq!(select_formats(&[], 100), Vec::<u32>::new());
    }

    #[test]
    fn snapshot_over_the_limit_keeps_text_and_bitmap_in_clipboard_order() {
        let sizes = [
            (CF_DIBV5, 40),
            (0xC100, 500),
            (CF_UNICODETEXT, 20),
            (CF_DIB, 40),
        ];
        assert_eq!(
            select_formats(&sizes, 100),
            vec![CF_DIBV5, CF_UNICODETEXT, CF_DIB]
        );
        // Both bitmaps do not fit: text first, then CF_DIB.
        assert_eq!(select_formats(&sizes, 70), vec![CF_UNICODETEXT, CF_DIB]);
        // A huge bitmap alone is dropped, the text stays.
        assert_eq!(
            select_formats(&[(CF_DIB, 500), (CF_UNICODETEXT, 10)], 100),
            vec![CF_UNICODETEXT]
        );
        assert_eq!(
            select_formats(&[(1, usize::MAX), (CF_UNICODETEXT, 10)], 100),
            vec![CF_UNICODETEXT]
        );
    }

    #[test]
    fn consoles_copy_with_ctrl_insert() {
        for class in [
            "CASCADIA_HOSTING_WINDOW_CLASS",
            "ConsoleWindowClass",
            "mintty",
        ] {
            assert!(copies_with_ctrl_insert(class), "{class}");
        }
        // Notepad, Chrome/Electron, Word, Explorer, unknown.
        for class in [
            "Notepad",
            "Chrome_WidgetWin_1",
            "OpusApp",
            "CabinetWClass",
            "",
        ] {
            assert!(!copies_with_ctrl_insert(class), "{class}");
        }
    }

    #[test]
    fn replace_honours_only_the_paste_method() {
        use crate::settings::PasteMethod;
        assert_eq!(
            paste_route(PasteMethod::Direct, "one line"),
            PasteRoute::Type
        );
        assert_eq!(paste_route(PasteMethod::Direct, "a\nb"), PasteRoute::CtrlV);
        assert_eq!(
            paste_route(PasteMethod::Direct, "a\r\nb"),
            PasteRoute::CtrlV
        );
        assert_eq!(paste_route(PasteMethod::CtrlV, "a"), PasteRoute::CtrlV);
        assert_eq!(
            paste_route(PasteMethod::CtrlShiftV, "a\nb"),
            PasteRoute::CtrlShiftV
        );
        assert_eq!(
            paste_route(PasteMethod::ShiftInsert, "a"),
            PasteRoute::ShiftInsert
        );
        assert_eq!(paste_route(PasteMethod::None, "a"), PasteRoute::CtrlV);
        assert_eq!(
            paste_route(PasteMethod::ExternalScript, "a"),
            PasteRoute::CtrlV
        );
    }

    #[test]
    fn utf16_text_stops_at_nul() {
        let mut bytes: Vec<u8> = "Привет 🙂"
            .encode_utf16()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        bytes.extend([0, 0, b'x', 0, 7]);
        assert_eq!(utf16_text(&bytes), "Привет 🙂");
        assert_eq!(utf16_text(&[b'a', 0, b'b']), "a");
        assert_eq!(utf16_text(&[]), "");
    }

    #[test]
    fn whitespace_only_is_no_selection() {
        assert_eq!(selection_text(None), None);
        assert_eq!(selection_text(Some(" \r\n\t ".into())), None);
        assert_eq!(selection_text(Some(" a ".into())), Some(" a ".into()));
    }

    #[test]
    fn poll_until_returns_first_value_or_times_out() {
        let calls = Cell::new(0);
        let found = poll_until(Duration::from_secs(1), Duration::from_millis(1), || {
            calls.set(calls.get() + 1);
            (calls.get() == 3).then_some(calls.get())
        });
        assert_eq!(found, Some(3));

        let started = Instant::now();
        let none: Option<()> =
            poll_until(Duration::from_millis(30), Duration::from_millis(5), || None);
        assert_eq!(none, None);
        assert!(started.elapsed() >= Duration::from_millis(30));

        // A zero timeout still tries once.
        assert_eq!(
            poll_until(Duration::ZERO, Duration::from_millis(5), || Some(1)),
            Some(1)
        );
    }

    #[test]
    fn error_messages_are_english() {
        assert_eq!(CaptureError::NoSelection.to_string(), "No text selected");
        assert!(CaptureError::ClipboardBusy.to_string().contains("busy"));
        assert_eq!(CaptureError::Other("x".into()).to_string(), "x");
    }

    const FAST: Timings = Timings {
        modifiers: Duration::from_millis(20),
        copy: Duration::from_millis(20),
        step: Duration::from_millis(1),
        reread: Duration::from_millis(1),
    };

    /// What the foreground app does on our Ctrl+C.
    #[derive(Clone, Copy, PartialEq)]
    enum OnCopy {
        Nothing,
        Text(&'static str),
        Image,
        /// Empties the clipboard first and puts the text on it on the next read.
        TextLate(&'static str),
    }

    struct Fake {
        on_copy: OnCopy,
        snapshot_busy: bool,
        modifiers_held: bool,
        /// The copy is marked "from an empty selection" (VS Code).
        empty_selection: bool,
        /// The user copies something right after our read.
        user_copies_after_read: bool,
        seq: Cell<u32>,
        text: RefCell<Option<String>>,
        restores: Cell<u32>,
        copies: Cell<u32>,
        pending_late: Cell<Option<&'static str>>,
    }

    impl Fake {
        fn new(on_copy: OnCopy) -> Self {
            Fake {
                on_copy,
                snapshot_busy: false,
                modifiers_held: false,
                empty_selection: false,
                user_copies_after_read: false,
                seq: Cell::new(100),
                text: RefCell::new(Some("user clipboard".into())),
                restores: Cell::new(0),
                copies: Cell::new(0),
                pending_late: Cell::new(None),
            }
        }

        fn bump(&self) {
            self.seq.set(self.seq.get() + 1);
        }
    }

    impl Backend for Fake {
        type Snapshot = Option<String>;

        fn modifiers_down(&self) -> bool {
            self.modifiers_held
        }
        fn foreground_window(&self) -> isize {
            42
        }
        fn snapshot(&self) -> Result<Option<String>, CaptureError> {
            if self.snapshot_busy {
                return Err(CaptureError::ClipboardBusy);
            }
            Ok(self.text.borrow().clone())
        }
        fn sequence(&self) -> u32 {
            self.seq.get()
        }
        fn send_copy(&self) -> Result<(), String> {
            self.copies.set(self.copies.get() + 1);
            match self.on_copy {
                OnCopy::Nothing => {}
                OnCopy::Text(text) => {
                    *self.text.borrow_mut() = Some(text.into());
                    self.bump();
                }
                OnCopy::Image => {
                    *self.text.borrow_mut() = None;
                    self.bump();
                }
                OnCopy::TextLate(text) => {
                    *self.text.borrow_mut() = None;
                    self.pending_late.set(Some(text));
                    self.bump();
                }
            }
            Ok(())
        }
        fn copied_from_empty_selection(&self) -> bool {
            self.empty_selection
        }
        fn read_text(&self) -> Result<(Option<String>, u32), CaptureError> {
            let result = (self.text.borrow().clone(), self.seq.get());
            if let Some(late) = self.pending_late.take() {
                *self.text.borrow_mut() = Some(late.into());
                self.bump();
            }
            if self.user_copies_after_read {
                *self.text.borrow_mut() = Some("user copied".into());
                self.bump();
            }
            Ok(result)
        }
        fn restore(&self, snapshot: &Option<String>, expected: u32) -> Result<bool, CaptureError> {
            if self.seq.get() != expected {
                return Ok(false);
            }
            self.restores.set(self.restores.get() + 1);
            *self.text.borrow_mut() = snapshot.clone();
            self.bump();
            Ok(true)
        }
    }

    #[test]
    fn captures_text_and_restores_the_clipboard() {
        let fake = Fake::new(OnCopy::Text("Hello"));
        let report = capture(&fake, &FAST);
        assert_eq!(
            report.result,
            Ok(CapturedSelection {
                text: "Hello".into(),
                hwnd: 42
            })
        );
        assert_eq!(report.clipboard, ClipboardOutcome::Restored);
        assert_eq!(fake.text.borrow().as_deref(), Some("user clipboard"));
    }

    #[test]
    fn nothing_selected_leaves_the_clipboard_untouched() {
        let fake = Fake::new(OnCopy::Nothing);
        let report = capture(&fake, &FAST);
        assert_eq!(report.result, Err(CaptureError::NoSelection));
        assert_eq!(report.clipboard, ClipboardOutcome::Untouched);
        assert_eq!(fake.restores.get(), 0);
        assert_eq!(fake.seq.get(), 100);
    }

    #[test]
    fn copied_image_is_no_selection_but_the_clipboard_is_restored() {
        let fake = Fake::new(OnCopy::Image);
        let report = capture(&fake, &FAST);
        assert_eq!(report.result, Err(CaptureError::NoSelection));
        assert_eq!(report.clipboard, ClipboardOutcome::Restored);
        assert_eq!(fake.text.borrow().as_deref(), Some("user clipboard"));
    }

    #[test]
    fn line_copied_from_an_empty_selection_is_no_selection_and_restored() {
        // VS Code with nothing selected copies the current line and says so.
        let mut fake = Fake::new(OnCopy::Text("the whole current line"));
        fake.empty_selection = true;
        let report = capture(&fake, &FAST);
        assert_eq!(report.result, Err(CaptureError::NoSelection));
        assert_eq!(report.clipboard, ClipboardOutcome::Restored);
        assert_eq!(fake.text.borrow().as_deref(), Some("user clipboard"));
    }

    #[test]
    fn editor_data_flag_is_read_from_vscode_json() {
        let empty = br#"{"version":1,"isFromEmptySelection":true,"multicursorText":null,"mode":"plaintext"}"#;
        assert!(is_empty_selection_copy(empty));
        // NUL padding and spaces are fine.
        assert!(is_empty_selection_copy(
            b"{ \"isFromEmptySelection\" : true }\0\0"
        ));
        // A real selection, a missing flag, junk and empty payloads are not.
        assert!(!is_empty_selection_copy(
            br#"{"version":1,"isFromEmptySelection":false,"mode":"rust"}"#
        ));
        assert!(!is_empty_selection_copy(br#"{"version":1}"#));
        assert!(!is_empty_selection_copy(
            br#"{"isFromEmptySelection":"true"}"#
        ));
        assert!(!is_empty_selection_copy(b"not json"));
        assert!(!is_empty_selection_copy(b""));
        assert!(!is_empty_selection_copy(b"\0\0"));
    }

    #[test]
    fn production_waits_briefly_for_modifiers() {
        // Held Alt/Shift/Win are released synthetically, so a long wait only delays the copy.
        assert!(TIMINGS.modifiers <= Duration::from_millis(150));
    }

    #[test]
    fn whitespace_selection_is_no_selection_and_restored() {
        let fake = Fake::new(OnCopy::Text("  \r\n"));
        let report = capture(&fake, &FAST);
        assert_eq!(report.result, Err(CaptureError::NoSelection));
        assert_eq!(report.clipboard, ClipboardOutcome::Restored);
    }

    #[test]
    fn a_newer_user_copy_is_not_overwritten() {
        let mut fake = Fake::new(OnCopy::Text("Hello"));
        fake.user_copies_after_read = true;
        let report = capture(&fake, &FAST);
        assert_eq!(report.result.map(|s| s.text), Ok("Hello".into()));
        assert_eq!(report.clipboard, ClipboardOutcome::KeptNewerCopy);
        assert_eq!(fake.text.borrow().as_deref(), Some("user copied"));
    }

    #[test]
    fn text_written_after_the_first_change_is_read_again() {
        let fake = Fake::new(OnCopy::TextLate("Late"));
        let report = capture(&fake, &FAST);
        assert_eq!(report.result.map(|s| s.text), Ok("Late".into()));
        assert_eq!(report.clipboard, ClipboardOutcome::Restored);
        assert_eq!(fake.text.borrow().as_deref(), Some("user clipboard"));
    }

    #[test]
    fn busy_clipboard_stops_before_ctrl_c() {
        let mut fake = Fake::new(OnCopy::Text("Hello"));
        fake.snapshot_busy = true;
        let report = capture(&fake, &FAST);
        assert_eq!(report.result, Err(CaptureError::ClipboardBusy));
        assert_eq!(report.clipboard, ClipboardOutcome::Untouched);
        assert_eq!(fake.copies.get(), 0);
    }

    #[test]
    fn held_modifiers_delay_but_do_not_block_the_copy() {
        let mut fake = Fake::new(OnCopy::Text("Hello"));
        fake.modifiers_held = true;
        let started = Instant::now();
        let report = capture(&fake, &FAST);
        assert!(started.elapsed() >= FAST.modifiers);
        assert_eq!(report.result.map(|s| s.text), Ok("Hello".into()));
    }

    /// Uses the real clipboard: the user's content is saved and put back at
    /// the end (also on failure). Run: cargo test --lib real_clipboard -- --ignored
    #[cfg(windows)]
    #[test]
    #[ignore]
    fn real_clipboard_snapshot_round_trip() {
        use win::testing::{format_count, marker, register, set_formats};
        use win::{restore_snapshot, sequence, take_snapshot};

        struct PutBack(win::Snapshot);
        impl Drop for PutBack {
            fn drop(&mut self) {
                let _ = restore_snapshot(&self.0, None);
            }
        }
        let _user = PutBack(take_snapshot().expect("snapshot of the user's clipboard"));

        let utf16 = |s: &str| -> Vec<u8> {
            s.encode_utf16()
                .chain(std::iter::once(0))
                .flat_map(|u| u.to_le_bytes())
                .collect()
        };
        let html_format = register("HTML Format");
        let html =
            b"Version:0.9\r\nStartHTML:0000000000\r\nEndHTML:0000000000\r\n<b>Hi</b>\0".to_vec();
        // 2x2 32-bit BI_RGB DIB: BITMAPINFOHEADER + 16 bytes of pixels.
        let mut dib = Vec::new();
        for v in [40u32, 2, 2] {
            dib.extend(v.to_le_bytes());
        }
        dib.extend(1u16.to_le_bytes());
        dib.extend(32u16.to_le_bytes());
        for v in [0u32, 16, 0, 0, 0, 0] {
            dib.extend(v.to_le_bytes());
        }
        dib.extend([
            10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 1, 2, 3, 255,
        ]);
        let text = utf16("Привет, SelTranslate 🙂");

        set_formats(&[
            (CF_UNICODETEXT, text.clone()),
            (html_format, html.clone()),
            (CF_DIB, dib.clone()),
        ]);
        let before = take_snapshot().unwrap();
        let get = |snap: &win::Snapshot, f: u32| {
            snap.formats
                .iter()
                .find(|(format, _)| *format == f)
                .map(|(_, d)| d.clone())
        };
        assert!(get(&before, CF_UNICODETEXT).unwrap().starts_with(&text));
        assert!(get(&before, html_format).unwrap().starts_with(&html));
        assert!(get(&before, CF_DIB).unwrap().starts_with(&dib));
        assert!(
            get(&before, CF_DIBV5).is_some(),
            "synthesized DIBV5 is snapshotted"
        );
        println!(
            "formats before: {:?}",
            before
                .formats
                .iter()
                .map(|(f, d)| (*f, d.len()))
                .collect::<Vec<_>>()
        );

        // Overwrite, then restore: every format comes back byte for byte, plus the marker.
        set_formats(&[(CF_UNICODETEXT, utf16("overwritten"))]);
        assert!(restore_snapshot(&before, Some(sequence())).unwrap());
        let after = take_snapshot().unwrap();
        let marker = marker();
        assert_eq!(get(&after, marker), Some(1u32.to_le_bytes().to_vec()));
        let sorted = |formats: Vec<(u32, Vec<u8>)>| {
            let mut formats: Vec<_> = formats.into_iter().filter(|(f, _)| *f != marker).collect();
            formats.sort();
            formats
        };
        assert_eq!(
            sorted(after.formats.clone()),
            sorted(before.formats.clone())
        );

        // The sequence moved past the expected one: a newer copy is kept.
        let stale = sequence();
        set_formats(&[(CF_UNICODETEXT, utf16("user copy"))]);
        assert!(!restore_snapshot(&before, Some(stale)).unwrap());
        assert!(get(&take_snapshot().unwrap(), CF_UNICODETEXT)
            .unwrap()
            .starts_with(&utf16("user copy")));

        // An empty clipboard is a valid snapshot: restoring it clears the clipboard.
        set_formats(&[]);
        assert_eq!(format_count(), 0);
        let empty = take_snapshot().unwrap();
        assert!(empty.formats.is_empty());
        set_formats(&[(CF_UNICODETEXT, utf16("temp"))]);
        assert!(restore_snapshot(&empty, Some(sequence())).unwrap());
        assert_eq!(format_count(), 0);
    }
}
