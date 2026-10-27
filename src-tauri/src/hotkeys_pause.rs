//! "Pause hotkeys" (tray): switches every Handy global hotkey off, e.g. for
//! games, where a registered combination would be swallowed by the OS-level
//! hook even if Handy ignored the event.
//!
//! Pausing *unregisters* the shortcuts (`shortcut::suspend_all_shortcuts`),
//! it does not just drop events. While paused, `shortcut::register_shortcut`
//! refuses to register anything (see [`refuse_register`]) so no UI path
//! (recording a new hotkey, changing a binding, post-processing toggle) can
//! arm a hotkey again. Resuming re-registers everything from settings.
//! The manual state is not persisted: after a restart hotkeys work as usual.
//!
//! The hotkeys are off for two independent reasons: the user's tray toggle
//! (`manual`) and a full-screen app in the foreground (`auto`, set by
//! `fullscreen_watch`; no windows may pop up in a game, so the hotkeys are
//! released like for `manual`). Either reason pauses; clearing one never
//! clears the other.
//!
//! A long job (an audio file being transcribed, a text file being translated;
//! see [`job_lock`]) is a different thing, decided with the user in B8: nothing
//! else may run next to it, and pressing a hotkey must say so instead of doing
//! nothing. So a job leaves the hotkeys registered and `job_running` makes
//! `shortcut::handler` (dictation) and `translate::action::trigger`
//! (translation, double Ctrl) show a "wait" notice and refuse.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use log::info;
use tauri::{AppHandle, Manager};

use crate::settings::ShortcutBinding;

#[derive(Default)]
pub struct HotkeysPaused {
    /// The tray "Pause hotkeys" check item.
    manual: AtomicBool,
    /// A full-screen app is in the foreground (`fullscreen_watch`).
    auto: AtomicBool,
    /// Long jobs in progress (see [`job_lock`]).
    jobs: AtomicUsize,
    /// `manual || auto`: what the hotkeys are actually doing right now.
    paused: AtomicBool,
    /// Serializes state transitions (tray clicks run on worker threads).
    transition: Mutex<()>,
}

impl HotkeysPaused {
    /// The hotkeys should be off. Long jobs are not a reason (see the module docs).
    fn wanted(&self) -> bool {
        self.manual.load(Ordering::Acquire) || self.auto.load(Ordering::Acquire)
    }

    fn flip_manual(&self) {
        self.manual.fetch_xor(true, Ordering::AcqRel);
    }

    fn set_auto(&self, on: bool) {
        self.auto.store(on, Ordering::Release);
    }
}

/// Hotkeys are off right now (for any reason).
pub fn is_paused(app: &AppHandle) -> bool {
    app.try_state::<HotkeysPaused>()
        .is_some_and(|s| s.paused.load(Ordering::Acquire))
}

/// The user's own pause — what the tray check item shows.
pub fn is_manually_paused(app: &AppHandle) -> bool {
    app.try_state::<HotkeysPaused>()
        .is_some_and(|s| s.manual.load(Ordering::Acquire))
}

pub fn toggle(app: &AppHandle) {
    update(app, HotkeysPaused::flip_manual);
}

/// A full-screen app came to / left the foreground (`fullscreen_watch`).
pub fn set_auto(app: &AppHandle, on: bool) {
    update(app, |s| s.set_auto(on));
}

/// A long job is running (audio file transcription, text file translation).
pub fn job_running(app: &AppHandle) -> bool {
    app.try_state::<HotkeysPaused>()
        .is_some_and(|s| s.jobs.load(Ordering::Acquire) > 0)
}

/// Marks a long job while the guard lives.
pub struct JobLock(AppHandle);

pub fn job_lock(app: &AppHandle) -> JobLock {
    if let Some(state) = app.try_state::<HotkeysPaused>() {
        state.jobs.fetch_add(1, Ordering::AcqRel);
    }
    JobLock(app.clone())
}

impl Drop for JobLock {
    fn drop(&mut self) {
        if let Some(state) = self.0.try_state::<HotkeysPaused>() {
            state.jobs.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

fn update(app: &AppHandle, change: impl FnOnce(&HotkeysPaused)) {
    let Some(state) = app.try_state::<HotkeysPaused>() else {
        return;
    };
    let _transition = state.transition.lock().unwrap_or_else(|e| e.into_inner());
    change(&state);
    let paused = state.wanted();

    // The flag is flipped first: while the shortcuts are being switched, any
    // concurrent registration attempt already sees the new state.
    if state.paused.swap(paused, Ordering::AcqRel) != paused {
        if paused {
            crate::shortcut::suspend_all_shortcuts(app);
            // `suspend_all_shortcuts` leaves "cancel" alone (it is dynamic and
            // only registered while recording); a pause must release it too.
            crate::shortcut::unregister_cancel_shortcut(app);
        } else {
            crate::shortcut::resume_all_shortcuts(app);
        }
        let state_name = match (paused, state.manual.load(Ordering::Acquire)) {
            (false, _) => "resumed",
            (true, true) => "paused",
            (true, false) => "paused (fullscreen)",
        };
        info!("Hotkeys {state_name}");
        // Double Ctrl has its own keyboard hook: off while paused (no hook of
        // ours in a game), back on resume.
        crate::translate::double_tap::sync(app);
    }
    crate::tray::update_tray_menu(app);
}

/// Called at the top of `shortcut::register_shortcut`. While paused the
/// registration is skipped and reported as successful, so settings changes
/// still go through and take effect on resume. "cancel" reports an error
/// instead: its registered/unregistered bookkeeping must not go out of sync.
pub fn refuse_register(app: &AppHandle, binding: &ShortcutBinding) -> Option<Result<(), String>> {
    if !is_paused(app) {
        return None;
    }
    Some(if binding.id == "cancel" {
        Err("hotkeys are paused".to_string())
    } else {
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_is_paused_by_default() {
        assert!(!HotkeysPaused::default().wanted());
    }

    #[test]
    fn manual_and_auto_each_pause_on_their_own() {
        let s = HotkeysPaused::default();
        s.flip_manual();
        assert!(s.wanted());
        s.flip_manual();
        assert!(!s.wanted());

        s.set_auto(true);
        assert!(s.wanted());
        s.set_auto(false);
        assert!(!s.wanted());
    }

    #[test]
    fn clearing_one_reason_keeps_the_other() {
        let s = HotkeysPaused::default();
        s.flip_manual();
        s.set_auto(true);
        s.set_auto(false);
        assert!(
            s.wanted(),
            "leaving full screen must not undo the tray pause"
        );

        s.set_auto(true);
        s.flip_manual();
        assert!(
            s.wanted(),
            "unticking the tray item must not undo the auto pause"
        );
        s.set_auto(false);
        assert!(!s.wanted());
    }

    #[test]
    fn setting_auto_twice_is_harmless() {
        let s = HotkeysPaused::default();
        s.set_auto(true);
        s.set_auto(true);
        s.set_auto(false);
        assert!(!s.wanted());
    }

    #[test]
    fn long_jobs_do_not_pause() {
        let s = HotkeysPaused::default();
        s.jobs.fetch_add(2, Ordering::AcqRel);
        assert!(!s.wanted());
        s.set_auto(true);
        s.jobs.fetch_sub(2, Ordering::AcqRel);
        assert!(s.wanted(), "a finished job must not undo the auto pause");
    }
}
