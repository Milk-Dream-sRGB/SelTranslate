import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { type as osType } from "@tauri-apps/plugin-os";
import { requestAccessibilityPermission } from "tauri-plugin-macos-permissions-api";
import React, { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { commands } from "@/bindings";
import type { ErrorCode, LanguageOption, PopupView } from "@/bindings";
import { syncLanguageFromSettings } from "@/i18n";
import "./TranslatePopup.css";

// Errors whose backend detail adds nothing to the localized message.
const SELF_EXPLANATORY: ErrorCode[] = [
  "no_selection",
  "clipboard_busy",
  "text_too_long",
  "job_running",
  "not_set_up",
  "no_accessibility",
];

// Copy shortcut: Cmd+C on macOS, Ctrl+C elsewhere.
const copyModifier = (event: KeyboardEvent) =>
  osType() === "macos" ? event.metaKey : event.ctrlKey;

const close = () => void commands.translatePopupClose();
const copy = () => void commands.translatePopupCopy();
const replace = () => void commands.translatePopupReplace();
const improve = () => void commands.translatePopupImprove();
const again = () => void commands.translatePopupAgain();
const setSource = (code: string) => void commands.translatePopupSetSource(code);
// "Not set up": the main window on the Translation section; the popup closes.
const openSettings = () => {
  void commands.translateOpenSettings();
  close();
};

// "No Accessibility": the system prompt that leads to the Privacy settings.
const openAccessibility = () => {
  void requestAccessibilityPermission();
  close();
};

const TranslatePopup: React.FC = () => {
  const { t } = useTranslation();
  const [view, setView] = useState<PopupView | null>(null);
  const viewRef = useRef<PopupView | null>(null);
  viewRef.current = view;
  // The source language list: picked when the detected language was wrong.
  const [languages, setLanguages] = useState<LanguageOption[]>([]);
  const [pickerOpen, setPickerOpen] = useState(false);
  const pickerRef = useRef(false);
  pickerRef.current = pickerOpen;

  useEffect(() => {
    void commands.translatePopupLanguages().then(setLanguages);
  }, []);

  // A new view (a new result) closes the list.
  useEffect(() => {
    setPickerOpen(false);
  }, [view]);

  // Views come as events; the stored one covers events sent before this page loaded.
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    void (async () => {
      await syncLanguageFromSettings();
      const off = await listen<PopupView>("translate-popup-state", (event) =>
        setView(event.payload),
      );
      if (cancelled) {
        off();
        return;
      }
      unlisten = off;
      const current = await commands.translatePopupState();
      if (!cancelled && current) {
        setView((latest) => latest ?? current);
      }
    })();
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  // Close on focus loss once there is something final to show.
  useEffect(() => {
    const unlisten = getCurrentWindow().onFocusChanged(({ payload }) => {
      if (!payload && viewRef.current?.kind !== "working") {
        close();
      }
    });
    return () => {
      void unlisten.then((off) => off());
    };
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const done = viewRef.current?.kind === "done";
      if (event.key === "Escape") {
        event.preventDefault();
        if (pickerRef.current) {
          setPickerOpen(false);
        } else {
          close();
        }
      } else if (event.key === "Enter" && done) {
        event.preventDefault();
        replace();
      } else if (
        copyModifier(event) &&
        event.code === "KeyC" &&
        done &&
        !window.getSelection()?.toString()
      ) {
        // Ctrl+C (Cmd+C on macOS) without a selection copies the whole translation.
        event.preventDefault();
        copy();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (!view) {
    return <div className="tp-root" />;
  }

  const canPick = view.kind === "done" && languages.length > 0;
  // Which slot and model did the work (I): "Fast · Qwen3.5 9B".
  const model =
    view.kind === "error"
      ? null
      : t("translatePopup.slotModel", {
          slot: t(`translatePopup.slot.${view.slot}`),
          model: view.model,
        });

  return (
    <div className="tp-root">
      <div className="tp-header">
        <span className="tp-direction">
          {view.kind !== "error" &&
            (canPick ? (
              <button
                className="tp-lang-button"
                onClick={() => setPickerOpen((open) => !open)}
                title={t("translatePopup.sourceLanguage")}
                aria-label={t("translatePopup.sourceLanguage")}
                aria-expanded={pickerOpen}
              >
                {view.src.toUpperCase()}
              </button>
            ) : (
              view.src.toUpperCase()
            ))}
          {view.kind !== "error" && ` → ${view.tgt.toUpperCase()}`}
          {model && <span className="tp-model"> · {model}</span>}
        </span>
        <button
          className="tp-icon-button"
          onClick={close}
          title={t("translatePopup.close")}
          aria-label={t("translatePopup.close")}
        >
          ×
        </button>
      </div>

      {pickerOpen && (
        <ul className="tp-lang-list" role="listbox">
          {languages.map((language) => (
            <li key={language.code}>
              <button
                className={
                  language.code === (view.kind === "done" ? view.src : "")
                    ? "tp-lang-option tp-lang-current"
                    : "tp-lang-option"
                }
                onClick={() => {
                  setPickerOpen(false);
                  setSource(language.code);
                }}
              >
                {language.name}
              </button>
            </li>
          ))}
        </ul>
      )}

      {view.kind === "working" && (
        <>
          <div className="tp-status">
            <span className="tp-spinner" />
            {view.starting_model
              ? t(
                  view.slot === "quality"
                    ? "translatePopup.startingBigModel"
                    : "translatePopup.startingModel",
                )
              : t("translatePopup.translating", {
                  done: Math.min(view.done + 1, view.total),
                  total: view.total,
                })}
          </div>
          <div className="tp-actions">
            <button className="tp-button" onClick={close}>
              {t("translatePopup.cancel")}
            </button>
          </div>
        </>
      )}

      {view.kind === "done" && (
        <>
          <div className="tp-text">{view.text}</div>
          {(view.truncated ||
            view.clipboard_lost ||
            view.improve_failed ||
            view.from_cache) && (
            <div className="tp-note">
              {view.improve_failed && (
                <div>{t("translatePopup.improveFailed")}</div>
              )}
              {view.from_cache && <div>{t("translatePopup.fromCache")}</div>}
              {view.truncated && <div>{t("translatePopup.truncated")}</div>}
              {view.clipboard_lost && (
                <div>{t("translatePopup.clipboardLost")}</div>
              )}
            </div>
          )}
          <div className="tp-actions">
            {view.from_cache && (
              <button className="tp-button" onClick={again}>
                {t("translatePopup.again")}
              </button>
            )}
            {view.can_improve && (
              <button className="tp-button" onClick={improve}>
                {t("translatePopup.better")}
              </button>
            )}
            <button className="tp-button" onClick={copy}>
              {t("translatePopup.copy")}
            </button>
            <button className="tp-button tp-primary" onClick={replace}>
              {t("translatePopup.replace")}
            </button>
          </div>
        </>
      )}

      {view.kind === "error" && (
        <div className="tp-error">
          <div className="tp-error-message">
            {t(`translatePopup.errors.${view.code}`)}
          </div>
          {!SELF_EXPLANATORY.includes(view.code) && view.detail && (
            <div className="tp-error-detail">{view.detail}</div>
          )}
          {view.code === "no_accessibility" && (
            <div className="tp-actions">
              <button
                className="tp-button tp-primary"
                onClick={openAccessibility}
              >
                {t("translatePopup.openSettings")}
              </button>
            </div>
          )}
          {view.code === "not_set_up" && (
            <div className="tp-actions">
              <button className="tp-button tp-primary" onClick={openSettings}>
                {t("translatePopup.openSettings")}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TranslatePopup;
