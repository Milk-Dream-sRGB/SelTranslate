// SelTranslate: Translate file — a whole text file through the local model,
// saved next to the source as <name>.<lang>.txt.
import React, { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { open, save } from "@tauri-apps/plugin-dialog";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { toast } from "sonner";
import { commands } from "@/bindings";
import type { Slot } from "@/bindings";
import { Button } from "../../ui/Button";
import { Dropdown } from "../../ui/Dropdown";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { copyToClipboard } from "../history/clipboard";
import { isRunning, useTranslateFileStore } from "./translateFileStore";

// Prefixes of the backend's error texts (translate/file.rs).
const ERROR_KEYS: [string, string][] = [
  ["File is not UTF-8 or UTF-16 text", "translateSettings.file.notText"],
  ["File is larger than 10 MB", "translateSettings.file.tooBig"],
  ["Nothing to translate", "translateSettings.file.nothing"],
  ["Translation is not set up yet", "translateSetup.setUpFirst"],
];

const formatTime = (secs: number) => {
  const s = Math.max(0, Math.round(secs));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
};

const baseName = (path: string) => path.split(/[\\/]/).pop() ?? path;

interface TranslateFileProps {
  /** The engine and a model are installed (the card above). */
  ready: boolean;
  slot: Slot;
  onSlotChange: (slot: Slot) => void;
}

export const TranslateFile: React.FC<TranslateFileProps> = ({
  ready,
  slot,
  onSlotChange,
}) => {
  const { t } = useTranslation();
  const {
    path,
    probe,
    status,
    done,
    total,
    etaSecs,
    result,
    error,
    init,
    selectFile,
    start,
    cancel,
  } = useTranslateFileStore();

  useEffect(() => init(), [init]);

  const running = isRunning(status);

  const handleSelect = async () => {
    const picked = await open({
      multiple: false,
      directory: false,
      filters: [
        {
          name: t("translateSettings.file.filter"),
          extensions: ["txt", "md", "srt", "vtt", "text"],
        },
      ],
    });
    if (typeof picked === "string") await selectFile(picked);
  };

  const handleCopy = async () => {
    if (!result) return;
    if (await copyToClipboard(result.text)) {
      toast.success(t("translateSettings.file.copied"));
    } else {
      toast.error(t("translateSettings.file.copyFailed"));
    }
  };

  const handleReveal = async () => {
    if (!result?.output_path) return;
    try {
      await revealItemInDir(result.output_path);
    } catch (e) {
      toast.error(t("translateSettings.file.showFailed", { error: String(e) }));
    }
  };

  const handleSaveAs = async () => {
    if (!result || !path) return;
    const target = await save({
      defaultPath: `${baseName(path).replace(/\.[^.]+$/, "")}.${result.tgt}.txt`,
      filters: [
        { name: t("translateSettings.file.filter"), extensions: ["txt"] },
      ],
    });
    if (!target) return;
    const saved = await commands.saveAudioFileText(target, result.text);
    if (saved.status === "ok") {
      toast.success(t("translateSettings.file.saved"));
    } else {
      toast.error(
        t("translateSettings.file.saveAsFailed", { error: saved.error }),
      );
    }
  };

  const fileLine = probe
    ? t("translateSettings.file.info", {
        name: path ? baseName(path) : "",
        chars: probe.chars,
        pieces: probe.pieces,
        src: probe.src.toUpperCase(),
        tgt: probe.tgt.toUpperCase(),
      })
    : path
      ? baseName(path)
      : t("translateSettings.file.noFile");

  const percent = total > 0 ? Math.min(100, (done / total) * 100) : 0;

  let statusLine = "";
  if (status === "starting_model") {
    statusLine = t("translateSettings.file.startingModel");
  } else if (status === "translating") {
    statusLine = t("translateSettings.file.progress", { done, total });
    if (etaSecs != null) {
      statusLine += ` · ${t("translateSettings.file.eta", {
        eta: formatTime(etaSecs),
      })}`;
    }
  } else if (status === "done" && result) {
    statusLine = t("translateSettings.file.done", {
      elapsed: formatTime(result.elapsed_secs),
    });
  } else if (status === "cancelled") {
    statusLine = t("translateSettings.file.cancelled");
  }

  const errorKey = error
    ? ERROR_KEYS.find(([prefix]) => error.startsWith(prefix))?.[1]
    : undefined;

  return (
    <SettingsGroup
      title={t("translateSettings.file.title")}
      description={t("translateSettings.file.description")}
    >
      <div className="px-4 py-3 flex items-center gap-3">
        <Button
          variant="secondary"
          onClick={() => void handleSelect()}
          disabled={running || !ready}
        >
          {t("translateSettings.file.select")}
        </Button>
        <span className="text-sm truncate" title={path ?? undefined}>
          {fileLine}
        </span>
      </div>

      <div className="px-4 py-3 space-y-2">
        <div className="flex items-center gap-3">
          {running ? (
            <Button variant="danger" onClick={() => void cancel()}>
              {t("translateSettings.file.cancel")}
            </Button>
          ) : (
            <Button
              variant="primary"
              onClick={() => void start()}
              disabled={!ready || !path || !probe}
            >
              {t("translateSettings.file.translate")}
            </Button>
          )}
          <Dropdown
            options={[
              { value: "fast", label: t("translateSettings.slot.fast") },
              { value: "quality", label: t("translateSettings.slot.quality") },
            ]}
            selectedValue={slot}
            onSelect={(value) => onSlotChange(value as Slot)}
            disabled={running}
          />
          <span className="text-sm text-mid-gray">
            {ready ? statusLine : t("translateSetup.setUpFirst")}
          </span>
        </div>
        {running && (
          <div className="h-2 w-full rounded bg-mid-gray/20 overflow-hidden">
            <div
              className={`h-full bg-logo-primary transition-[width] duration-300 ${
                status === "starting_model" ? "w-full animate-pulse" : ""
              }`}
              style={
                status === "starting_model"
                  ? undefined
                  : { width: `${percent}%` }
              }
            />
          </div>
        )}
        {status === "error" && error && (
          <p className="text-sm text-red-400 break-words whitespace-pre-wrap">
            {errorKey ? t(errorKey) : error}
          </p>
        )}
      </div>

      {status === "done" && result && (
        <div className="px-4 py-3 space-y-3">
          {result.output_path ? (
            <p className="text-sm break-all">
              {t("translateSettings.file.savedTo", {
                path: result.output_path,
              })}
            </p>
          ) : (
            <p className="text-sm text-red-400 break-words">
              {t("translateSettings.file.saveFailed", {
                error: result.save_error ?? "",
              })}
            </p>
          )}
          {result.untranslated > 0 && (
            <p className="text-sm text-warning">
              {t("translateSettings.file.untranslated", {
                count: result.untranslated,
              })}
            </p>
          )}
          <div className="flex gap-2 flex-wrap">
            {result.output_path && (
              <Button variant="secondary" onClick={() => void handleReveal()}>
                {t("translateSettings.file.showInFolder")}
              </Button>
            )}
            <Button variant="secondary" onClick={() => void handleCopy()}>
              {t("translateSettings.file.copy")}
            </Button>
            {!result.output_path && (
              <Button variant="primary" onClick={() => void handleSaveAs()}>
                {t("translateSettings.file.saveAs")}
              </Button>
            )}
          </div>
        </div>
      )}
    </SettingsGroup>
  );
};
