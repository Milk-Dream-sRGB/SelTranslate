// SelTranslate: Audio file tab — transcribe a recording with the selected model.
import React, { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { open, save } from "@tauri-apps/plugin-dialog";
import { toast } from "sonner";
import { commands } from "@/bindings";
import { useModelStore } from "../../../stores/modelStore";
import { Button } from "../../ui/Button";
import { Dropdown } from "../../ui/Dropdown";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { copyToClipboard } from "../history/clipboard";
import { isRunning, useAudioFileStore } from "./audioFileStore";
import { Library } from "./Library";

// symphonia 0.5 has no Opus decoder (.opus is not offered) and no mkv/webm
// demuxer in this build; mp4/m4v/mov use the audio track of the video.
const AUDIO_EXTENSIONS = [
  "mp3",
  "m4a",
  "aac",
  "flac",
  "ogg",
  "oga",
  "wav",
  "mp4",
  "m4v",
  "mov",
];

// Prefix of decoder::NO_AUDIO_TRACK in the backend.
const NO_AUDIO_TRACK = "No supported audio track.";

const formatTime = (secs: number) => {
  const s = Math.max(0, Math.round(secs));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
};

const baseName = (path: string) => path.split(/[\\/]/).pop() ?? path;

export const AudioFileSettings: React.FC = () => {
  const { t } = useTranslation();
  const { currentModel, models } = useModelStore();
  const {
    path,
    info,
    modelId,
    status,
    processedSecs,
    totalSecs,
    text,
    transcribeStartedAt,
    elapsedSecs,
    error,
    init,
    selectFile,
    setModel,
    start,
    cancel,
  } = useAudioFileStore();

  useEffect(() => init(), [init]);

  const running = isRunning(status);
  const modelName = models.find((m) => m.id === currentModel)?.name;
  const downloaded = models.filter((m) => m.is_downloaded);
  const chosen = downloaded.find(
    (m) => m.id === modelId && m.id !== currentModel,
  );
  const modelOptions = [
    {
      value: "",
      label: t("audioFile.sameAsDictation", { model: modelName ?? "—" }),
    },
    ...downloaded
      .filter((m) => m.id !== currentModel)
      .map((m) => ({ value: m.id, label: m.name })),
  ];

  const handleSelect = async () => {
    const picked = await open({
      multiple: false,
      directory: false,
      filters: [
        { name: t("audioFile.audioFilter"), extensions: AUDIO_EXTENSIONS },
      ],
    });
    if (typeof picked === "string") await selectFile(picked);
  };

  const handleCopy = async () => {
    if (await copyToClipboard(text)) toast.success(t("audioFile.copied"));
    else toast.error(t("audioFile.copyFailed"));
  };

  const handleSave = async () => {
    const target = await save({
      defaultPath: path
        ? baseName(path).replace(/\.[^.]+$/, "") + ".txt"
        : undefined,
      filters: [{ name: t("audioFile.textFilter"), extensions: ["txt"] }],
    });
    if (!target) return;
    const result = await commands.saveAudioFileText(target, text);
    if (result.status === "ok") toast.success(t("audioFile.saved"));
    else toast.error(t("audioFile.saveFailed", { error: result.error }));
  };

  const fileLine = info
    ? `${path ? baseName(path) : ""} · ${
        info.duration_secs != null
          ? formatTime(info.duration_secs)
          : t("audioFile.unknownDuration")
      } · ${info.format}`
    : path
      ? baseName(path)
      : t("audioFile.noFile");

  const percent =
    totalSecs && totalSecs > 0
      ? Math.min(100, (processedSecs / totalSecs) * 100)
      : null;

  let etaSecs: number | null = null;
  if (status === "transcribing" && totalSecs && transcribeStartedAt) {
    const spent = (Date.now() - transcribeStartedAt) / 1000;
    if (processedSecs > 0 && spent > 0) {
      etaSecs = ((totalSecs - processedSecs) * spent) / processedSecs;
    }
  }

  let statusLine = "";
  if (status === "loading_model") statusLine = t("audioFile.loadingModel");
  else if (status === "transcribing") {
    statusLine = totalSecs
      ? `${formatTime(processedSecs)} / ${formatTime(totalSecs)}`
      : formatTime(processedSecs);
    if (etaSecs != null)
      statusLine += ` · ${t("audioFile.eta", { eta: formatTime(etaSecs) })}`;
  } else if (status === "done")
    statusLine = t("audioFile.done", {
      audio: formatTime(processedSecs),
      elapsed: formatTime(elapsedSecs ?? 0),
    });
  else if (status === "cancelled") statusLine = t("audioFile.cancelled");

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <SettingsGroup
        title={t("audioFile.title")}
        description={t("audioFile.description")}
      >
        <div className="px-4 py-3 space-y-2">
          <div className="flex items-center gap-3">
            <span className="text-sm shrink-0">
              {t("audioFile.modelLabel")}
            </span>
            <Dropdown
              className="min-w-0 flex-1"
              options={modelOptions}
              selectedValue={chosen ? chosen.id : ""}
              onSelect={setModel}
              disabled={running}
            />
          </div>
          <p
            className={`text-xs ${modelName ? "text-mid-gray" : "text-warning"}`}
          >
            {modelName ? t("audioFile.modelHint") : t("audioFile.noModel")}
          </p>
        </div>

        <div className="px-4 py-3 flex items-center gap-3">
          <Button variant="secondary" onClick={handleSelect} disabled={running}>
            {t("audioFile.selectFile")}
          </Button>
          <span className="text-sm truncate" title={path ?? undefined}>
            {fileLine}
          </span>
        </div>

        <div className="px-4 py-3 space-y-2">
          <div className="flex items-center gap-3">
            {running ? (
              <Button variant="danger" onClick={cancel}>
                {t("audioFile.cancel")}
              </Button>
            ) : (
              <Button
                variant="primary"
                onClick={start}
                disabled={!path || !info || !modelName}
              >
                {t("audioFile.transcribe")}
              </Button>
            )}
            <span className="text-sm text-mid-gray">{statusLine}</span>
          </div>
          {running && (
            <p className="text-xs text-mid-gray">{t("audioFile.hotkeysOff")}</p>
          )}
          {running && (
            <div className="h-2 w-full rounded bg-mid-gray/20 overflow-hidden">
              <div
                className={`h-full bg-logo-primary transition-[width] duration-300 ${
                  percent == null ? "w-full animate-pulse" : ""
                }`}
                style={percent == null ? undefined : { width: `${percent}%` }}
              />
            </div>
          )}
          {status === "error" && error && (
            <p className="text-sm text-red-400 break-words">
              {error.startsWith(NO_AUDIO_TRACK)
                ? t("audioFile.noAudioTrack")
                : error}
            </p>
          )}
        </div>
      </SettingsGroup>

      {(text || status === "done") && (
        <SettingsGroup title={t("audioFile.result")}>
          <div className="px-4 py-3 space-y-3">
            <textarea
              readOnly
              value={text}
              className="w-full min-h-[240px] px-3 py-2 text-sm bg-mid-gray/10 border border-mid-gray/20 rounded-md resize-y focus:outline-none"
            />
            <div className="flex gap-2">
              <Button variant="secondary" onClick={handleCopy} disabled={!text}>
                {t("audioFile.copy")}
              </Button>
              <Button
                variant="secondary"
                onClick={handleSave}
                disabled={!text || running}
              >
                {t("audioFile.saveTxt")}
              </Button>
            </div>
          </div>
        </SettingsGroup>
      )}

      <Library />
    </div>
  );
};
