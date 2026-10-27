//! Double Ctrl (C2): a second way to start the translation, for places where the
//! Alt hotkey does not work (Windows Terminal drops its selection on any key).
//!
//! Two parts:
//! - [`DoubleTap`] — a pure state machine over (time, Ctrl pressed/released,
//!   any other key) events; no OS, no Tauri, unit-tested.
//! - [`sync`] — starts or stops a passive `handy_keys::KeyboardListener` (a
//!   low-level keyboard hook that only observes: it was created without
//!   blocking hotkeys, so it never swallows a key) in its own thread. The
//!   listener exists only while `double_ctrl_enabled` is on and the hotkeys
//!   are not paused (Pause hotkeys, audio-file job); otherwise the thread is
//!   joined and the hooks are gone, so a game sees no hook of ours.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use handy_keys::{Error as KeysError, KeyEvent, KeyboardListener, Modifiers};
use log::{info, warn};
use tauri::AppHandle;

/// A tap is Ctrl pressed and released within this time.
const MAX_HOLD: Duration = Duration::from_millis(300);
/// The second tap must start within this time after the first Ctrl release.
const MAX_GAP: Duration = Duration::from_millis(350);
/// After a trigger nothing is accepted for this long: our own copy chord
/// (Ctrl+C / Ctrl+Insert) and a clumsy third tap must not fire again.
const COOLDOWN: Duration = Duration::from_millis(600);
/// How often the listener thread looks at its stop flag.
const POLL: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    CtrlDown(Side),
    CtrlUp(Side),
    /// Any other key, modifier or mouse button: a Ctrl chord, not a tap.
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    /// First Ctrl is down.
    Down1 {
        at: Duration,
    },
    /// First tap done, waiting for the second.
    Gap {
        up_at: Duration,
    },
    /// Second Ctrl is down.
    Down2 {
        at: Duration,
    },
}

/// Time is a monotonic offset (`Duration` since the listener started).
#[derive(Debug)]
pub struct DoubleTap {
    phase: Phase,
    left: bool,
    right: bool,
    cooldown_until: Option<Duration>,
}

impl Default for DoubleTap {
    fn default() -> Self {
        Self {
            phase: Phase::Idle,
            left: false,
            right: false,
            cooldown_until: None,
        }
    }
}

impl DoubleTap {
    fn held(&self, side: Side) -> bool {
        match side {
            Side::Left => self.left,
            Side::Right => self.right,
        }
    }

    fn set_held(&mut self, side: Side, value: bool) {
        match side {
            Side::Left => self.left = value,
            Side::Right => self.right = value,
        }
    }

    fn other_held(&self, side: Side) -> bool {
        match side {
            Side::Left => self.right,
            Side::Right => self.left,
        }
    }

    /// Feeds one event; `true` means "the second tap just ended: translate".
    pub fn feed(&mut self, now: Duration, input: Input) -> bool {
        // Which Ctrl keys are down is tracked even during the cooldown, so a
        // key held across it is not mistaken for a new press afterwards.
        let cooling = match self.cooldown_until {
            Some(until) if now < until => true,
            Some(_) => {
                self.cooldown_until = None;
                false
            }
            None => false,
        };
        if cooling {
            match input {
                Input::CtrlDown(side) => self.set_held(side, true),
                Input::CtrlUp(side) => self.set_held(side, false),
                Input::Other => {}
            }
            self.phase = Phase::Idle;
            return false;
        }

        match input {
            Input::Other => {
                self.phase = Phase::Idle;
                false
            }
            Input::CtrlDown(side) => {
                if self.held(side) {
                    // Auto-repeat of a held key.
                    return false;
                }
                self.set_held(side, true);
                if self.other_held(side) {
                    // Left and right Ctrl at once is not a tap.
                    self.phase = Phase::Idle;
                    return false;
                }
                self.phase = match self.phase {
                    Phase::Gap { up_at } if now.saturating_sub(up_at) < MAX_GAP => {
                        Phase::Down2 { at: now }
                    }
                    _ => Phase::Down1 { at: now },
                };
                false
            }
            Input::CtrlUp(side) => {
                if !self.held(side) {
                    return false;
                }
                self.set_held(side, false);
                if self.other_held(side) {
                    self.phase = Phase::Idle;
                    return false;
                }
                match std::mem::replace(&mut self.phase, Phase::Idle) {
                    Phase::Down1 { at } if now.saturating_sub(at) < MAX_HOLD => {
                        self.phase = Phase::Gap { up_at: now };
                        false
                    }
                    Phase::Down2 { at } if now.saturating_sub(at) < MAX_HOLD => {
                        self.cooldown_until = Some(now + COOLDOWN);
                        true
                    }
                    _ => false,
                }
            }
        }
    }
}

/// Maps a raw key event to the machine's input. A Ctrl press or release with
/// any other modifier held (Ctrl+Shift, Win+Ctrl…) is not a clean tap.
fn classify(event: &KeyEvent) -> Input {
    let Some(changed) = event.changed_modifier else {
        return Input::Other;
    };
    if changed.is_empty() || !Modifiers::CTRL.contains(changed) {
        return Input::Other;
    }
    if !(event.modifiers & !Modifiers::CTRL).is_empty() {
        return Input::Other;
    }
    let side = if changed == Modifiers::CTRL_LEFT {
        Side::Left
    } else {
        Side::Right
    };
    if event.is_key_down {
        Input::CtrlDown(side)
    } else {
        Input::CtrlUp(side)
    }
}

struct Running {
    stop: Arc<AtomicBool>,
    thread: JoinHandle<()>,
}

static LISTENER: Mutex<Option<Running>> = Mutex::new(None);

/// Brings the listener in line with the settings and the pause state: running
/// if and only if Double Ctrl is enabled and the hotkeys are not paused.
/// Cheap when nothing changed. Call after any of those change (startup, the
/// settings command, Pause hotkeys); may block ~50 ms while a listener stops.
pub fn sync(app: &AppHandle) {
    let wanted =
        super::settings::load(app).double_ctrl_enabled && !crate::hotkeys_pause::is_paused(app);
    let mut slot = LISTENER.lock().unwrap_or_else(|e| e.into_inner());

    // A listener whose hook could not be installed ends by itself; forget it,
    // so the next sync tries again.
    if slot.as_ref().is_some_and(|r| r.thread.is_finished()) {
        if let Some(finished) = slot.take() {
            let _ = finished.thread.join();
        }
    }

    match (wanted, slot.is_some()) {
        (true, false) => match start(app) {
            Ok(running) => {
                *slot = Some(running);
                info!("Double Ctrl: listener started");
            }
            Err(e) => {
                warn!("Double Ctrl: cannot start the keyboard listener: {e}");
                #[cfg(target_os = "macos")]
                permission_waiter::ensure(app);
            }
        },
        (false, true) => {
            if let Some(running) = slot.take() {
                stop(running);
                info!("Double Ctrl: listener stopped");
            }
        }
        _ => {}
    }
}

/// `RunEvent::Exit`: no listener outlives the app.
pub fn shutdown() {
    #[cfg(target_os = "macos")]
    permission_waiter::stop();
    let running = LISTENER.lock().unwrap_or_else(|e| e.into_inner()).take();
    if let Some(running) = running {
        stop(running);
    }
}

fn stop(running: Running) {
    running.stop.store(true, Ordering::SeqCst);
    // The thread drops the listener on its way out, which joins the hook thread.
    let _ = running.thread.join();
}

fn start(app: &AppHandle) -> Result<Running, String> {
    let listener = KeyboardListener::new().map_err(|e| e.to_string())?;
    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = stop.clone();
    let app = app.clone();
    let thread = std::thread::Builder::new()
        .name("double-ctrl".into())
        .spawn(move || listen(&app, listener, &thread_stop))
        .map_err(|e| e.to_string())?;
    Ok(Running { stop, thread })
}

fn listen(app: &AppHandle, listener: KeyboardListener, stop: &AtomicBool) {
    let started = Instant::now();
    let mut machine = DoubleTap::default();
    while !stop.load(Ordering::SeqCst) {
        match listener.recv_timeout(POLL) {
            Ok(event) => {
                if machine.feed(started.elapsed(), classify(&event)) {
                    info!("Double Ctrl: triggered");
                    super::action::trigger(app);
                }
            }
            Err(KeysError::Timeout) => {}
            Err(e) => {
                // The hook thread is gone (the hook could not be installed).
                warn!("Double Ctrl: keyboard listener stopped: {e}");
                return;
            }
        }
    }
}

/// macOS: the keyboard hook needs the Accessibility permission. Without it
/// `sync` fails once and would not try again, so a single thread polls the
/// permission and calls `sync` as soon as it appears (Double Ctrl comes alive
/// right after onboarding, without a restart).
#[cfg(target_os = "macos")]
mod permission_waiter {
    use super::*;

    const CHECK_EVERY: Duration = Duration::from_secs(2);

    static WAITER: Mutex<Option<(Arc<AtomicBool>, JoinHandle<()>)>> = Mutex::new(None);

    /// Starts the waiter unless the permission is there (the failure was
    /// something else) or a waiter is already running.
    pub fn ensure(app: &AppHandle) {
        if handy_keys::check_accessibility() {
            return;
        }
        let mut slot = WAITER.lock().unwrap_or_else(|e| e.into_inner());
        if slot.as_ref().is_some_and(|(_, t)| !t.is_finished()) {
            return;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let (thread_stop, app) = (stop.clone(), app.clone());
        match std::thread::Builder::new()
            .name("double-ctrl-permission".into())
            .spawn(move || wait(&app, &thread_stop))
        {
            Ok(thread) => {
                info!("Double Ctrl: waiting for the Accessibility permission");
                *slot = Some((stop, thread));
            }
            Err(e) => warn!("Double Ctrl: cannot start the permission waiter: {e}"),
        }
    }

    /// `RunEvent::Exit`: the waiter does not outlive the app.
    pub fn stop() {
        let waiter = WAITER.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some((stop, thread)) = waiter {
            stop.store(true, Ordering::SeqCst);
            let _ = thread.join();
        }
    }

    fn wait(app: &AppHandle, stop: &AtomicBool) {
        let mut waited = Duration::ZERO;
        while !stop.load(Ordering::SeqCst) {
            if waited >= CHECK_EVERY {
                waited = Duration::ZERO;
                if handy_keys::check_accessibility() {
                    info!("Double Ctrl: Accessibility permission granted");
                    sync(app);
                    return;
                }
            }
            std::thread::sleep(POLL);
            waited += POLL;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Input::{CtrlDown, CtrlUp, Other};
    use Side::{Left, Right};

    const fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// Runs (time ms, input) events; returns the times at which it fired.
    fn run(events: &[(u64, Input)]) -> Vec<u64> {
        let mut machine = DoubleTap::default();
        events
            .iter()
            .filter(|(t, input)| machine.feed(ms(*t), *input))
            .map(|(t, _)| *t)
            .collect()
    }

    #[test]
    fn double_tap_fires_on_the_second_release() {
        let fired = run(&[
            (0, CtrlDown(Left)),
            (80, CtrlUp(Left)),
            (200, CtrlDown(Left)),
            (280, CtrlUp(Left)),
        ]);
        assert_eq!(fired, vec![280]);
    }

    #[test]
    fn right_ctrl_works_and_sides_may_differ() {
        assert_eq!(
            run(&[
                (0, CtrlDown(Right)),
                (60, CtrlUp(Right)),
                (150, CtrlDown(Right)),
                (210, CtrlUp(Right)),
            ]),
            vec![210]
        );
        assert_eq!(
            run(&[
                (0, CtrlDown(Left)),
                (60, CtrlUp(Left)),
                (150, CtrlDown(Right)),
                (210, CtrlUp(Right)),
            ]),
            vec![210]
        );
    }

    #[test]
    fn single_tap_does_nothing() {
        assert!(run(&[(0, CtrlDown(Left)), (80, CtrlUp(Left))]).is_empty());
    }

    #[test]
    fn ctrl_c_is_not_a_tap() {
        // Ctrl+C, then Ctrl+C again quickly: a key sits between ↓ and ↑ each time.
        let fired = run(&[
            (0, CtrlDown(Left)),
            (30, Other),
            (60, Other),
            (90, CtrlUp(Left)),
            (150, CtrlDown(Left)),
            (180, Other),
            (210, Other),
            (240, CtrlUp(Left)),
        ]);
        assert!(fired.is_empty());
    }

    #[test]
    fn a_tap_after_ctrl_c_does_not_pair_with_it() {
        // Ctrl+C then one tap: only one clean tap, no trigger.
        let fired = run(&[
            (0, CtrlDown(Left)),
            (30, Other),
            (60, Other),
            (90, CtrlUp(Left)),
            (150, CtrlDown(Left)),
            (210, CtrlUp(Left)),
        ]);
        assert!(fired.is_empty());
    }

    #[test]
    fn a_key_between_the_taps_resets() {
        let fired = run(&[
            (0, CtrlDown(Left)),
            (60, CtrlUp(Left)),
            (100, Other),
            (150, CtrlDown(Left)),
            (210, CtrlUp(Left)),
        ]);
        assert!(fired.is_empty());
    }

    #[test]
    fn a_slow_tap_is_a_hold() {
        // First Ctrl held 300 ms (the limit is exclusive).
        assert!(run(&[
            (0, CtrlDown(Left)),
            (300, CtrlUp(Left)),
            (350, CtrlDown(Left)),
            (400, CtrlUp(Left)),
        ])
        .is_empty());
        // Second Ctrl held too long.
        assert!(run(&[
            (0, CtrlDown(Left)),
            (100, CtrlUp(Left)),
            (150, CtrlDown(Left)),
            (500, CtrlUp(Left)),
        ])
        .is_empty());
        // Just under the limit passes.
        assert_eq!(
            run(&[
                (0, CtrlDown(Left)),
                (299, CtrlUp(Left)),
                (310, CtrlDown(Left)),
                (609, CtrlUp(Left)),
            ]),
            vec![609]
        );
    }

    #[test]
    fn a_long_gap_is_two_single_taps() {
        // Gap 350 ms or more.
        assert!(run(&[
            (0, CtrlDown(Left)),
            (50, CtrlUp(Left)),
            (400, CtrlDown(Left)),
            (450, CtrlUp(Left)),
        ])
        .is_empty());
        // The late tap can still be the first of a new pair.
        assert_eq!(
            run(&[
                (0, CtrlDown(Left)),
                (50, CtrlUp(Left)),
                (400, CtrlDown(Left)),
                (450, CtrlUp(Left)),
                (550, CtrlDown(Left)),
                (600, CtrlUp(Left)),
            ]),
            vec![600]
        );
    }

    #[test]
    fn auto_repeat_while_held_is_ignored() {
        // Held Ctrl repeats ↓; it is one long hold, not a double tap.
        let fired = run(&[
            (0, CtrlDown(Left)),
            (500, CtrlDown(Left)),
            (530, CtrlDown(Left)),
            (560, CtrlDown(Left)),
            (600, CtrlUp(Left)),
            (650, CtrlDown(Left)),
            (700, CtrlUp(Left)),
        ]);
        assert!(fired.is_empty());
        // Repeats inside a short tap do not break it.
        let fired = run(&[
            (0, CtrlDown(Left)),
            (20, CtrlDown(Left)),
            (60, CtrlUp(Left)),
            (120, CtrlDown(Left)),
            (140, CtrlDown(Left)),
            (180, CtrlUp(Left)),
        ]);
        assert_eq!(fired, vec![180]);
    }

    #[test]
    fn triple_tap_fires_once() {
        let fired = run(&[
            (0, CtrlDown(Left)),
            (50, CtrlUp(Left)),
            (120, CtrlDown(Left)),
            (170, CtrlUp(Left)),
            (240, CtrlDown(Left)),
            (290, CtrlUp(Left)),
        ]);
        assert_eq!(fired, vec![170]);
    }

    #[test]
    fn quadruple_tap_fires_once_and_cooldown_ends() {
        let fired = run(&[
            (0, CtrlDown(Left)),
            (50, CtrlUp(Left)),
            (120, CtrlDown(Left)),
            (170, CtrlUp(Left)),
            (240, CtrlDown(Left)),
            (290, CtrlUp(Left)),
            (360, CtrlDown(Left)),
            (410, CtrlUp(Left)),
            // After the cooldown (170 + 600) a new pair works again.
            (1000, CtrlDown(Left)),
            (1050, CtrlUp(Left)),
            (1120, CtrlDown(Left)),
            (1170, CtrlUp(Left)),
        ]);
        assert_eq!(fired, vec![170, 1170]);
    }

    #[test]
    fn our_copy_chord_after_a_trigger_does_not_fire_again() {
        // The action sends Ctrl+Insert / Ctrl+C right after the trigger.
        let fired = run(&[
            (0, CtrlDown(Left)),
            (50, CtrlUp(Left)),
            (120, CtrlDown(Left)),
            (170, CtrlUp(Left)),
            // synthetic chord: Ctrl↓ key↓ key↑ Ctrl↑, then the user's tap
            (260, CtrlDown(Left)),
            (262, Other),
            (300, Other),
            (302, CtrlUp(Left)),
        ]);
        assert_eq!(fired, vec![170]);
    }

    #[test]
    fn left_and_right_together_are_not_a_tap() {
        let fired = run(&[
            (0, CtrlDown(Left)),
            (20, CtrlDown(Right)),
            (60, CtrlUp(Left)),
            (80, CtrlUp(Right)),
            (150, CtrlDown(Left)),
            (200, CtrlUp(Left)),
        ]);
        assert!(fired.is_empty());
    }

    #[test]
    fn overlapping_sides_inside_the_second_tap_cancel_it() {
        let fired = run(&[
            (0, CtrlDown(Left)),
            (40, CtrlUp(Left)),
            (100, CtrlDown(Left)),
            (110, CtrlDown(Right)),
            (150, CtrlUp(Left)),
            (160, CtrlUp(Right)),
        ]);
        assert!(fired.is_empty());
    }

    #[test]
    fn stray_release_is_ignored() {
        // Ctrl was held before the listener started: its ↑ must not count.
        let fired = run(&[
            (0, CtrlUp(Left)),
            (50, CtrlDown(Left)),
            (90, CtrlUp(Left)),
            (150, CtrlDown(Left)),
            (190, CtrlUp(Left)),
        ]);
        assert_eq!(fired, vec![190]);
    }

    #[test]
    fn ctrl_held_through_the_cooldown_is_not_a_new_press() {
        // Second tap triggers; the user then presses and holds Ctrl inside the
        // cooldown, releases after it: no new first tap, so one more tap is
        // not enough for another trigger.
        let fired = run(&[
            (0, CtrlDown(Left)),
            (50, CtrlUp(Left)),
            (100, CtrlDown(Left)),
            (150, CtrlUp(Left)),
            (300, CtrlDown(Left)),
            (900, CtrlUp(Left)),
            (950, CtrlDown(Left)),
            (1000, CtrlUp(Left)),
        ]);
        assert_eq!(fired, vec![150]);
    }

    fn event(modifiers: Modifiers, key_down: bool, changed: Option<Modifiers>) -> KeyEvent {
        KeyEvent {
            modifiers,
            key: None,
            is_key_down: key_down,
            changed_modifier: changed,
        }
    }

    #[test]
    fn classify_bare_ctrl() {
        assert_eq!(
            classify(&event(
                Modifiers::CTRL_LEFT,
                true,
                Some(Modifiers::CTRL_LEFT)
            )),
            CtrlDown(Left)
        );
        assert_eq!(
            classify(&event(
                Modifiers::empty(),
                false,
                Some(Modifiers::CTRL_RIGHT)
            )),
            CtrlUp(Right)
        );
    }

    #[test]
    fn classify_everything_else_as_other() {
        // Another modifier changed.
        assert_eq!(
            classify(&event(
                Modifiers::SHIFT_LEFT,
                true,
                Some(Modifiers::SHIFT_LEFT)
            )),
            Other
        );
        // Ctrl pressed while Shift is held: a chord.
        assert_eq!(
            classify(&event(
                Modifiers::CTRL_LEFT | Modifiers::SHIFT_LEFT,
                true,
                Some(Modifiers::CTRL_LEFT)
            )),
            Other
        );
        // A regular key (changed_modifier is None).
        let key = KeyEvent {
            modifiers: Modifiers::CTRL_LEFT,
            key: Some(handy_keys::Key::C),
            is_key_down: true,
            changed_modifier: None,
        };
        assert_eq!(classify(&key), Other);
    }

    #[test]
    fn ctrl_shift_chord_never_fires() {
        // Shift↓ Ctrl↓ Ctrl↑ Ctrl↓ Ctrl↑ Shift↑ through `classify`.
        let events = [
            event(Modifiers::SHIFT_LEFT, true, Some(Modifiers::SHIFT_LEFT)),
            event(
                Modifiers::SHIFT_LEFT | Modifiers::CTRL_LEFT,
                true,
                Some(Modifiers::CTRL_LEFT),
            ),
            event(Modifiers::SHIFT_LEFT, false, Some(Modifiers::CTRL_LEFT)),
            event(
                Modifiers::SHIFT_LEFT | Modifiers::CTRL_LEFT,
                true,
                Some(Modifiers::CTRL_LEFT),
            ),
            event(Modifiers::SHIFT_LEFT, false, Some(Modifiers::CTRL_LEFT)),
        ];
        let mut machine = DoubleTap::default();
        let mut fired = false;
        for (i, e) in events.iter().enumerate() {
            fired |= machine.feed(ms(i as u64 * 60), classify(e));
        }
        assert!(!fired);
    }
}
