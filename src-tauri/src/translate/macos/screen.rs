//! Is the frontmost app covering a whole screen, menu bar included (DESIGN J.7)?
//! Only window bounds, layer and owner pid are read: they need no Screen
//! Recording permission (that one is for window names and contents).

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString};

/// A rectangle in global screen points, origin top-left.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// A window as `CGWindowListCopyWindowInfo` describes it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Window {
    pub pid: i32,
    pub layer: i32,
    pub bounds: Rect,
}

/// Slack for rounding of the edges, in points.
const EDGE_SLACK: f64 = 1.5;

/// Whether `rect` covers `screen` whole.
fn covers(rect: Rect, screen: Rect) -> bool {
    rect.x <= screen.x + EDGE_SLACK
        && rect.y <= screen.y + EDGE_SLACK
        && rect.x + rect.w >= screen.x + screen.w - EDGE_SLACK
        && rect.y + rect.h >= screen.y + screen.h - EDGE_SLACK
}

/// A normal window (layer 0) of `pid` covers a whole screen, the strip of the
/// menu bar included: a full-screen Space or a full-screen game. An ordinary
/// maximized window starts below the menu bar and does not.
pub fn covers_screen(windows: &[Window], screens: &[Rect], pid: i32) -> bool {
    pid > 0
        && windows
            .iter()
            .filter(|w| w.pid == pid && w.layer == 0)
            .any(|w| screens.iter().any(|s| covers(w.bounds, *s)))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CGPoint {
    x: f64,
    y: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CGSize {
    width: f64,
    height: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CGRect {
    origin: CGPoint,
    size: CGSize,
}

// kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements
const LIST_OPTIONS: u32 = 1 | (1 << 4);
const MAX_DISPLAYS: usize = 16;

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGWindowListCopyWindowInfo(option: u32, relative_to_window: u32) -> *mut AnyObject;
    fn CGGetActiveDisplayList(max: u32, displays: *mut u32, count: *mut u32) -> i32;
    fn CGDisplayBounds(display: u32) -> CGRect;
}

/// The bounds of every active display.
pub fn screens() -> Vec<Rect> {
    let mut ids = [0u32; MAX_DISPLAYS];
    let mut count = 0u32;
    // SAFETY: `ids` has room for MAX_DISPLAYS entries, `count` is a local.
    let status =
        unsafe { CGGetActiveDisplayList(MAX_DISPLAYS as u32, ids.as_mut_ptr(), &mut count) };
    if status != 0 {
        return Vec::new();
    }
    ids.iter()
        .take(count as usize)
        .map(|&id| {
            // SAFETY: an id from CGGetActiveDisplayList.
            let r = unsafe { CGDisplayBounds(id) };
            Rect {
                x: r.origin.x,
                y: r.origin.y,
                w: r.size.width,
                h: r.size.height,
            }
        })
        .collect()
}

fn number(dict: &NSDictionary<NSString, AnyObject>, key: &str) -> Option<f64> {
    let value = dict.objectForKey(&NSString::from_str(key))?;
    Some(value.downcast_ref::<NSNumber>()?.doubleValue())
}

/// The on-screen windows with their pid, layer and bounds.
pub fn windows() -> Vec<Window> {
    // SAFETY: a "Create" call; the array is ours (+1) and CFArray/CFDictionary
    // are toll-free bridged to NSArray/NSDictionary.
    let raw = unsafe { CGWindowListCopyWindowInfo(LIST_OPTIONS, 0) };
    let Some(array) =
        (unsafe { Retained::from_raw(raw.cast::<NSArray<NSDictionary<NSString, AnyObject>>>()) })
    else {
        return Vec::new();
    };
    array
        .iter()
        .filter_map(|dict| {
            let pid = number(&dict, "kCGWindowOwnerPID")? as i32;
            let layer = number(&dict, "kCGWindowLayer")? as i32;
            let bounds = dict.objectForKey(&NSString::from_str("kCGWindowBounds"))?;
            let bounds = bounds.downcast_ref::<NSDictionary>()?;
            // SAFETY: the type parameters of NSDictionary are phantom; the keys
            // are strings, as CGWindowListCopyWindowInfo documents.
            let bounds = unsafe {
                &*(bounds as *const NSDictionary).cast::<NSDictionary<NSString, AnyObject>>()
            };
            Some(Window {
                pid,
                layer,
                bounds: Rect {
                    x: number(bounds, "X")?,
                    y: number(bounds, "Y")?,
                    w: number(bounds, "Width")?,
                    h: number(bounds, "Height")?,
                },
            })
        })
        .collect()
}

/// The frontmost application covers a whole screen now.
pub fn frontmost_is_fullscreen() -> bool {
    covers_screen(&windows(), &screens(), super::frontmost_pid())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCREEN: Rect = Rect {
        x: 0.0,
        y: 0.0,
        w: 1728.0,
        h: 1117.0,
    };

    fn window(pid: i32, layer: i32, x: f64, y: f64, w: f64, h: f64) -> Window {
        Window {
            pid,
            layer,
            bounds: Rect { x, y, w, h },
        }
    }

    #[test]
    fn a_full_screen_window_covers_the_screen() {
        let w = [window(10, 0, 0.0, 0.0, 1728.0, 1117.0)];
        assert!(covers_screen(&w, &[SCREEN], 10));
    }

    #[test]
    fn a_maximized_window_below_the_menu_bar_does_not() {
        let w = [window(10, 0, 0.0, 38.0, 1728.0, 1079.0)];
        assert!(!covers_screen(&w, &[SCREEN], 10));
    }

    #[test]
    fn only_normal_windows_of_the_frontmost_app_count() {
        let big = |pid, layer| window(pid, layer, 0.0, 0.0, 1728.0, 1117.0);
        assert!(!covers_screen(&[big(11, 0)], &[SCREEN], 10), "another app");
        assert!(
            !covers_screen(&[big(10, 25)], &[SCREEN], 10),
            "the menu bar layer"
        );
        assert!(
            !covers_screen(&[big(10, -2147483623)], &[SCREEN], 10),
            "the desktop"
        );
        assert!(
            !covers_screen(&[big(0, 0)], &[SCREEN], 0),
            "no frontmost app"
        );
    }

    #[test]
    fn any_of_several_screens_will_do() {
        let second = Rect {
            x: 1728.0,
            y: -200.0,
            w: 2560.0,
            h: 1440.0,
        };
        let w = [window(10, 0, 1728.0, -200.0, 2560.0, 1440.0)];
        assert!(covers_screen(&w, &[SCREEN, second], 10));
        assert!(!covers_screen(&w, &[SCREEN], 10));
    }

    #[test]
    fn the_real_screen_list_is_not_empty() {
        let screens = screens();
        assert!(!screens.is_empty());
        assert!(screens.iter().all(|s| s.w > 0.0 && s.h > 0.0));
        // Reading the windows works without any permission.
        let _ = (windows(), frontmost_is_fullscreen());
    }
}
