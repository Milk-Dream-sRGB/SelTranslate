// SelTranslate: "Translate" on a History entry, and the translation shown
// under it. Kept out of Handy's HistorySettings, which only places the two.
import React, { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { save } from "@tauri-apps/plugin-dialog";
import { Languages, X } from "lucide-react";
import { commands } from "@/bindings";
import { useFlash } from "@/hooks/useFlash";
import { notify } from "@/stores/noticeStore";
import { Button } from "../../ui/Button";
import { copyToClipboard } from "../history/clipboard";
import { formatTime } from "./format";
import { useHistoryTranslateStore } from "./historyTranslateStore";
import { isRunning, useTranslateFileStore } from "./translateFileStore";

// Prefixes of the backend's error texts (translate/file.rs, action.rs).
const ERROR_KEYS: [string, string][] = [
  ["Translation is not set up yet", "translateSetup.setUpFirst"],
  [
    "The file is already in the target language",
    "historyTranslate.sameLanguage",
  ],
  ["Nothing to translate", "historyTranslate.nothing"],
];

/** The icon next to Copy; History's own buttons look the same. */
export const TranslateEntryButton: React.FC<{
  entryId: number;
  disabled: boolean;
}> = ({ entryId, disabled }) => {
  const { t } = useTranslation();
  const { init, start, running } = useHistoryTranslateStore();
  const fileRunning = isRunning(useTranslateFileStore((s) => s.status));
  useEffect(() => init(), [init]);

  return (
    <button
      onClick={() => void start(entryId)}
      disabled={disabled || running !== null || fileRunning}
      className="p-1.5 rounded-md flex items-center justify-center transition-colors cursor-pointer disabled:cursor-not-allowed disabled:text-text/20 text-ink-3 hover:text-logo-primary"
      title={t("historyTranslate.translate")}
    >
      <Languages width={16} height={16} />
    </button>
  );
};

/** `name.ru.txt` from the entry's file, `translation.ru.txt` for dictation. */
const defaultFileName = (sourceName: string | null, tgt: string | null) => {
  const stem = sourceName?.replace(/\.[^.]+$/, "") || "translation";
  return `${stem}.${tgt ?? "txt"}.txt`;
};

/** Progress, the result or the error of the entry's translation; nothing
 * when it has none. */
export const EntryTranslationPanel: React.FC<{
  entryId: number;
  sourceName: string | null;
}> = ({ entryId, sourceName }) => {
  const { t } = useTranslation();
  const translation = useHistoryTranslateStore((s) => s.byEntry[entryId]);
  const { cancel, close } = useHistoryTranslateStore();
  const [flashed, flash] = useFlash();

  if (!translation) return null;
  const { status, done, total, etaSecs, text, src, tgt, untranslated, error } =
    translation;

  const handleCopy = async () => {
    if (!text) return;
    if (await copyToClipboard(text)) flash("copied");
    else notify({ title: t("translateSettings.file.copyFailed") });
  };

  const handleSave = async () => {
    if (!text) return;
    const target = await save({
      defaultPath: defaultFileName(sourceName, tgt),
      filters: [
        { name: t("translateSettings.file.filter"), extensions: ["txt"] },
      ],
    });
    if (!target) return;
    const saved = await commands.saveTextFile(target, text);
    if (saved.status === "ok") flash("saved");
    else
      notify({
        title: t("translateSettings.file.saveAsFailed", { error: saved.error }),
      });
  };

  const closeButton = (
    <button
      onClick={() => close(entryId)}
      className="p-1 rounded-md text-ink-3 hover:text-logo-primary cursor-pointer"
      title={t("historyTranslate.close")}
    >
      <X width={14} height={14} />
    </button>
  );

  if (status === "starting_model" || status === "translating") {
    let line = t("translateSettings.file.startingModel");
    if (status === "translating") {
      line = t("historyTranslate.progress", { done, total });
      if (etaSecs != null) {
        line += ` · ${t("translateSettings.file.eta", { eta: formatTime(etaSecs) })}`;
      }
    }
    const percent = total > 0 ? Math.min(100, (done / total) * 100) : 0;
    return (
      <div className="flex flex-col gap-2 rounded-lg border border-line px-3 py-2">
        <div className="flex items-center gap-3">
          <span className="text-sm text-ink-3 grow">{line}</span>
          <Button variant="secondary" size="sm" onClick={() => void cancel()}>
            {t("translateSettings.file.cancel")}
          </Button>
        </div>
        <div className="h-1.5 w-full rounded bg-line overflow-hidden">
          <div
            className={`h-full bg-logo-primary transition-[width] duration-300 ${
              status === "starting_model" ? "w-full animate-pulse" : ""
            }`}
            style={
              status === "starting_model" ? undefined : { width: `${percent}%` }
            }
          />
        </div>
      </div>
    );
  }

  if (status === "error") {
    const key = ERROR_KEYS.find(([prefix]) => error?.startsWith(prefix))?.[1];
    return (
      <div className="flex items-start gap-2 rounded-lg border border-line px-3 py-2">
        <p className="text-sm text-error break-words grow">
          {key ? t(key) : error}
        </p>
        {closeButton}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-line bg-wash px-3 py-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-ink-3 grow">
          {t("historyTranslate.title", {
            src: (src ?? "").toUpperCase(),
            tgt: (tgt ?? "").toUpperCase(),
          })}
        </span>
        {closeButton}
      </div>
      <p className="text-sm text-text select-text cursor-text whitespace-pre-wrap break-words">
        {text}
      </p>
      {untranslated > 0 && (
        <p className="text-xs text-warning">
          {t("translateSettings.file.untranslated", { count: untranslated })}
        </p>
      )}
      <div className="flex gap-2">
        <Button variant="secondary" size="sm" onClick={() => void handleCopy()}>
          {flashed === "copied"
            ? t("translateSettings.file.copied")
            : t("translateSettings.file.copy")}
        </Button>
        <Button variant="secondary" size="sm" onClick={() => void handleSave()}>
          {flashed === "saved"
            ? t("translateSettings.file.saved")
            : t("translateSettings.file.saveAs")}
        </Button>
      </div>
    </div>
  );
};
