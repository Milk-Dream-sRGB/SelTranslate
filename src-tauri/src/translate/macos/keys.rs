//! Keyboard side of the capture (DESIGN J.3): which modifiers are held, the
//! keycode of a letter under Command in the current layout, and the Cmd+C chord.

use std::ffi::c_void;
use std::time::Duration;

use enigo::{Direction, Key, Keyboard};
use log::{debug, warn};
use tauri::{AppHandle, Manager};

use crate::input::EnigoState;

// CGEventFlags (CGEventTypes.h).
const FLAG_SHIFT: u64 = 0x0002_0000;
const FLAG_CONTROL: u64 = 0x0004_0000;
const FLAG_OPTION: u64 = 0x0008_0000;
const FLAG_COMMAND: u64 = 0x0010_0000;
const MODIFIER_FLAGS: [(u64, &str); 4] = [
    (FLAG_CONTROL, "Ctrl"),
    (FLAG_OPTION, "Option"),
    (FLAG_SHIFT, "Shift"),
    (FLAG_COMMAND, "Cmd"),
];

/// kCGEventSourceStateHIDSystemState: the state of the physical keyboard.
const HID_SYSTEM_STATE: i32 = 1;

// kVK_ANSI_C: the fallback if the layout cannot be read.
const ANSI_C_KEYCODE: u16 = 8;
const KEYCODE_COUNT: u16 = 128;
const UC_KEY_ACTION_DISPLAY: u16 = 3;
const UC_KEY_TRANSLATE_NO_DEAD_KEYS_MASK: u32 = 1;
// Carbon's cmdKey is bit 8; UCKeyTranslate wants it shifted right by 8.
const COMMAND_MODIFIER_STATE: u32 = 1;
/// How long Command stays down after the letter, for apps that poll the state.
const CHORD_HOLD: Duration = Duration::from_millis(40);

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGEventSourceFlagsState(state_id: i32) -> u64;
}

type TisInputSourceRef = *const c_void;
type CfDataRef = *const c_void;
type CfStringRef = *const c_void;

#[link(name = "Carbon", kind = "framework")]
unsafe extern "C" {
    fn TISCopyCurrentKeyboardLayoutInputSource() -> TisInputSourceRef;
    fn TISGetInputSourceProperty(
        input_source: TisInputSourceRef,
        property_key: CfStringRef,
    ) -> CfDataRef;
    static kTISPropertyUnicodeKeyLayoutData: CfStringRef;
    fn UCKeyTranslate(
        key_layout: *const u8,
        virtual_key_code: u16,
        key_action: u16,
        modifier_key_state: u32,
        keyboard_type: u32,
        key_translate_options: u32,
        dead_key_state: *mut u32,
        max_string_length: usize,
        actual_string_length: *mut usize,
        unicode_string: *mut u16,
    ) -> i32;
    fn LMGetKbdType() -> u8;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFDataGetBytePtr(data: CfDataRef) -> *const u8;
    fn CFRelease(value: *const c_void);
}

/// Names of the modifiers set in CGEventFlags, "Ctrl+Shift" (empty if none).
fn modifier_names(flags: u64) -> String {
    MODIFIER_FLAGS
        .iter()
        .filter(|(flag, _)| flags & flag != 0)
        .map(|(_, name)| *name)
        .collect::<Vec<_>>()
        .join("+")
}

fn held_flags() -> u64 {
    // SAFETY: CGEventSourceFlagsState takes a state id and has no pointers.
    unsafe { CGEventSourceFlagsState(HID_SYSTEM_STATE) }
}

/// Whether Ctrl, Option, Shift or Command are physically down now.
pub fn modifiers_down() -> bool {
    !modifier_names(held_flags()).is_empty()
}

/// Names of the modifiers held now, for the log.
pub fn held_modifiers() -> String {
    modifier_names(held_flags())
}

struct InputSource(TisInputSourceRef);

impl Drop for InputSource {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: balances the retain of TISCopyCurrentKeyboardLayoutInputSource.
            unsafe { CFRelease(self.0) };
        }
    }
}

/// The physical key that macOS reads as `letter` while Command is held in the
/// current layout (Russian layouts map Cmd+letter to the ANSI key, Dvorak does
/// not). A copy of the approach of `input.rs::macos`, for any letter. Main
/// thread only (TIS).
fn resolve_command_keycode(letter: u8) -> Result<u16, String> {
    // SAFETY: main thread; the source follows the Create Rule and is released
    // by InputSource::drop.
    let source = InputSource(unsafe { TISCopyCurrentKeyboardLayoutInputSource() });
    if source.0.is_null() {
        return Err("macOS returned no current keyboard layout input source".into());
    }
    // SAFETY: the source stays retained during the scan.
    let layout_data =
        unsafe { TISGetInputSourceProperty(source.0, kTISPropertyUnicodeKeyLayoutData) };
    if layout_data.is_null() {
        return Err("the keyboard layout has no Unicode layout data".into());
    }
    // SAFETY: layout_data is owned by the retained source.
    let layout = unsafe { CFDataGetBytePtr(layout_data) };
    if layout.is_null() {
        return Err("the keyboard layout data is empty".into());
    }
    // SAFETY: no arguments.
    let keyboard_type = u32::from(unsafe { LMGetKbdType() });
    (0..KEYCODE_COUNT)
        .find(|&keycode| {
            let mut dead_key_state = 0;
            let mut chars = [0_u16; 4];
            let mut length = 0_usize;
            // SAFETY: layout points to valid UCKeyboardLayout bytes while the
            // source is retained; the outputs are local and of the declared size.
            let status = unsafe {
                UCKeyTranslate(
                    layout,
                    keycode,
                    UC_KEY_ACTION_DISPLAY,
                    COMMAND_MODIFIER_STATE,
                    keyboard_type,
                    UC_KEY_TRANSLATE_NO_DEAD_KEYS_MASK,
                    &mut dead_key_state,
                    chars.len(),
                    &mut length,
                    chars.as_mut_ptr(),
                )
            };
            status == 0 && length == 1 && chars[0] == u16::from(letter)
        })
        .ok_or_else(|| format!("could not map Cmd+{} in the current layout", letter as char))
}

fn command_keycode(letter: u8, fallback: u16) -> u16 {
    match resolve_command_keycode(letter) {
        Ok(keycode) => {
            debug!("Resolved Cmd+{} to keycode {keycode}", letter as char);
            keycode
        }
        Err(e) => {
            warn!(
                "Could not resolve Cmd+{} ({e}); using the ANSI keycode {fallback}",
                letter as char
            );
            fallback
        }
    }
}

/// Presses and releases Command+C through the shared `Enigo`. On the main
/// thread; Command is released even if the letter fails. The user's physically
/// held Ctrl/Shift do not leak into the chord: enigo gives every event its own
/// flags.
pub fn send_copy(app: &AppHandle) -> Result<(), String> {
    let handle = app.clone();
    super::on_main(app, move || {
        let state = handle
            .try_state::<EnigoState>()
            .ok_or("Input is not initialized")?;
        let mut enigo = state
            .0
            .lock()
            .map_err(|_| "Input state is poisoned".to_string())?;
        let keycode = command_keycode(b'c', ANSI_C_KEYCODE);
        enigo
            .key(Key::Meta, Direction::Press)
            .map_err(|e| format!("Failed to press Command: {e}"))?;
        let clicked = enigo
            .key(Key::Other(u32::from(keycode)), Direction::Click)
            .map_err(|e| format!("Failed to click C: {e}"));
        std::thread::sleep(CHORD_HOLD);
        let released = enigo
            .key(Key::Meta, Direction::Release)
            .map_err(|e| format!("Failed to release Command: {e}"));
        clicked.and(released)
    })?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modifier_flags_are_named_in_a_fixed_order() {
        assert_eq!(modifier_names(0), "");
        assert_eq!(modifier_names(FLAG_SHIFT | FLAG_CONTROL), "Ctrl+Shift");
        assert_eq!(
            modifier_names(FLAG_COMMAND | FLAG_OPTION | FLAG_SHIFT | FLAG_CONTROL),
            "Ctrl+Option+Shift+Cmd"
        );
    }

    #[test]
    fn other_flags_are_not_modifiers() {
        // Caps Lock (0x10000), Fn (0x800000) and the numeric pad bit.
        assert_eq!(modifier_names(0x0001_0000 | 0x0080_0000 | 0x0020_0000), "");
    }
}
