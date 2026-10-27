//! Auto-pause of every hotkey while a full-screen app (game, video,
//! slideshow) is in the foreground — the `auto` reason of `hotkeys_pause`.
//!
//! The state comes from `SHQueryUserNotificationState`: a D3D full-screen
//! app (`QUNS_RUNNING_D3D_FULL_SCREEN`) or any other full-screen window
//! (`QUNS_BUSY`). `QUNS_PRESENTATION_MODE` is deliberately not a trigger: it
//! is a notification setting the user turns on by hand, not a full-screen app.
//! Borderless-window games are not reported by Windows as full screen; for
//! them the tray "Pause hotkeys" stays.
//!
//! The poll thread exists only while `auto_pause_fullscreen` is on: with the
//! setting off there is no thread and no call. Switching it needs no restart
//! (`sync`), and turning it off clears the `auto` reason (never `manual`).

use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::time::Duration;

use log::{info, warn};
use tauri::AppHandle;

/// Poll period; the stop signal interrupts the wait at once.
const POLL: Duration = Duration::from_secs(1);
/// Consecutive "full screen" readings needed to pause (a flicker of
/// `QUNS_BUSY` while a window switches is not a game). Leaving is immediate.
const ENTER_AFTER: u32 = 2;

// Values of QUERY_USER_NOTIFICATION_STATE (shellapi.h).
const QUNS_BUSY: i32 = 2;
const QUNS_RUNNING_D3D_FULL_SCREEN: i32 = 3;

#[cfg_attr(not(windows), allow(dead_code))]
fn state_is_fullscreen(raw_state: i32) -> bool {
    matches!(raw_state, QUNS_BUSY | QUNS_RUNNING_D3D_FULL_SCREEN)
}

/// Turns raw readings into the "fullscreen now" flag: pausing needs
/// [`ENTER_AFTER`] readings in a row, resuming only one.
#[derive(Default)]
struct Debounce {
    streak: u32,
    active: bool,
}

impl Debounce {
    /// Feeds one reading; returns the new state when it changed.
    fn feed(&mut self, fullscreen: bool) -> Option<bool> {
        if !fullscreen {
            self.streak = 0;
            return std::mem::take(&mut self.active).then_some(false);
        }
        self.streak = self.streak.saturating_add(1);
        if !self.active && self.streak >= ENTER_AFTER {
            self.active = true;
            return Some(true);
        }
        None
    }
}

#[cfg(windows)]
fn fullscreen_now() -> bool {
    use windows::Win32::UI::Shell::SHQueryUserNotificationState;
    // SAFETY: no arguments; the call only reads the shell's state.
    match unsafe { SHQueryUserNotificationState() } {
        Ok(state) => state_is_fullscreen(state.0),
        Err(e) => {
            warn!("Fullscreen watch: cannot query the shell state: {e}");
            false
        }
    }
}

/// macOS: the frontmost app covers a whole screen, menu bar included (J.7).
#[cfg(target_os = "macos")]
fn fullscreen_now() -> bool {
    super::macos::screen::frontmost_is_fullscreen()
}

#[cfg(not(any(windows, target_os = "macos")))]
fn fullscreen_now() -> bool {
    false
}

struct Running {
    /// Dropping the sender wakes the thread and ends it.
    stop: Sender<()>,
    thread: JoinHandle<()>,
}

static WATCHER: Mutex<Option<Running>> = Mutex::new(None);

/// Brings the watcher in line with the setting: a thread exists if and only if
/// `auto_pause_fullscreen` is on (Windows and macOS). Stopping also clears the
/// `auto` pause. Cheap when nothing changed; may block ~1 ms on stop.
pub fn sync(app: &AppHandle) {
    let wanted =
        cfg!(any(windows, target_os = "macos")) && super::settings::load(app).auto_pause_fullscreen;
    let mut slot = WATCHER.lock().unwrap_or_else(|e| e.into_inner());
    match (wanted, slot.is_some()) {
        (true, false) => match start(app) {
            Ok(running) => {
                *slot = Some(running);
                info!("Fullscreen watch: started");
            }
            Err(e) => warn!("Fullscreen watch: cannot start: {e}"),
        },
        (false, true) => {
            if let Some(running) = slot.take() {
                stop(running);
                // After the join: the thread can no longer set the reason again.
                crate::hotkeys_pause::set_auto(app, false);
                info!("Fullscreen watch: stopped");
            }
        }
        _ => {}
    }
}

/// `RunEvent::Exit`: no thread outlives the app.
pub fn shutdown() {
    let running = WATCHER.lock().unwrap_or_else(|e| e.into_inner()).take();
    if let Some(running) = running {
        stop(running);
    }
}

fn stop(running: Running) {
    drop(running.stop);
    let _ = running.thread.join();
}

fn start(app: &AppHandle) -> Result<Running, String> {
    let (stop, stopped) = mpsc::channel::<()>();
    let app = app.clone();
    let thread = std::thread::Builder::new()
        .name("fullscreen-watch".into())
        .spawn(move || {
            let mut debounce = Debounce::default();
            // `Timeout` = time to poll; anything else = the sender is gone.
            while let Err(RecvTimeoutError::Timeout) = stopped.recv_timeout(POLL) {
                if let Some(on) = debounce.feed(fullscreen_now()) {
                    crate::hotkeys_pause::set_auto(&app, on);
                }
            }
        })
        .map_err(|e| e.to_string())?;
    Ok(Running { stop, thread })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_full_screen_states_count() {
        // NOT_PRESENT, BUSY, D3D_FULL_SCREEN, PRESENTATION_MODE, ACCEPTS_NOTIFICATIONS, QUIET_TIME, APP
        let verdicts: Vec<bool> = (1..=7).map(state_is_fullscreen).collect();
        assert_eq!(
            verdicts,
            [false, true, true, false, false, false, false],
            "presentation mode, quiet time and the lock screen are not full screen"
        );
        assert!(!state_is_fullscreen(0));
    }

    #[test]
    fn pausing_needs_a_steady_full_screen_resuming_does_not() {
        let mut d = Debounce::default();
        assert_eq!(d.feed(false), None);
        assert_eq!(d.feed(true), None, "one reading is a flicker");
        assert_eq!(d.feed(false), None);
        assert_eq!(d.feed(true), None);
        assert_eq!(d.feed(true), Some(true));
        assert_eq!(d.feed(true), None, "no repeat while it lasts");
        assert_eq!(d.feed(false), Some(false), "leaving is immediate");
        assert_eq!(d.feed(false), None);
    }

    #[test]
    fn a_flicker_while_paused_does_not_leave_a_stale_streak() {
        let mut d = Debounce::default();
        d.feed(true);
        assert_eq!(d.feed(true), Some(true));
        assert_eq!(d.feed(false), Some(false));
        assert_eq!(d.feed(true), None, "the streak starts over");
        assert_eq!(d.feed(true), Some(true));
    }
}
