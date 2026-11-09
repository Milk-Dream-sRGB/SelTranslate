// SelTranslate: "Speakers" on a History entry: who spoke when, with a time
// stamp per line. Kept out of Handy's HistorySettings, which only places the
// button and the panel.
import React, { useEffect, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { save } from "@tauri-apps/plugin-dialog";
import { Users, X } from "lucide-react";
import { commands } from "@/bindings";
import { useFlash } from "@/hooks/useFlash";
import { notify } from "@/stores/noticeStore";
import { LANGUAGES, languageName } from "@/lib/constants/languages";
import { Button } from "../../ui/Button";
import { Dropdown } from "../../ui/Dropdown";
import { copyToClipboard } from "../history/clipboard";
import { useHistorySpeakersStore } from "./historySpeakersStore";
import { formatTranscript, titleOf } from "./speakersFormat";

// Prefixes of the backend's error texts (src-tauri/src/speakers).
const ERROR_KEYS: [string, string][] = [
  ["Recording contains no speech", "historySpeakers.noSpeech"],
  ["The recording of this entry was deleted", "historySpeakers.noAudio"],
  ["Download failed", "historySpeakers.downloadFailed"],
];

const VARIANT_NAMES: Record<string, string> = {
  q8_0: "Q8_0",
  f16: "F16",
  f32: "F32",
};

const mb = (bytes: number) => Math.round(bytes / 1_000_000);

/** The icon next to Translate; History's own buttons look the same. */
export const SpeakersEntryButton: React.FC<{
  entryId: number;
  disabled: boolean;
}> = ({ entryId, disabled }) => {
  const { t } = useTranslation();
  const { init, open } = useHistorySpeakersStore();
  useEffect(() => init(), [init]);

  return (
    <button
      onClick={() => open(entryId)}
      disabled={disabled}
      className="p-1.5 rounded-md flex items-center justify-center transition-colors cursor-pointer disabled:cursor-not-allowed disabled:text-text/20 text-ink-3 hover:text-logo-primary"
      title={t("historySpeakers.button")}
    >
      <Users width={16} height={16} />
    </button>
  );
};

const CloseButton: React.FC<{ onClick: () => void }> = ({ onClick }) => {
  const { t } = useTranslation();
  return (
    <button
      onClick={onClick}
      className="p-1 rounded-md text-ink-3 hover:text-logo-primary cursor-pointer"
      title={t("historySpeakers.close")}
    >
      <X width={14} height={14} />
    </button>
  );
};

/** The choices before a run: model variant, language, number of speakers. */
const SpeakersOptionsForm: React.FC<{ entryId: number }> = ({ entryId }) => {
  const { t, i18n } = useTranslation();
  const { setup, running, setOptions, deleteVariant, start, close } =
    useHistorySpeakersStore();

  const languageOptions = useMemo(
    () => [
      { value: "", label: t("historySpeakers.languageFromSettings") },
      ...LANGUAGES.map((l) => ({
        value: l.code,
        label: `${languageName(l.code, i18n.language)} · ${l.native}`,
      })),
    ],
    [t, i18n.language],
  );

  if (!setup) return null;
  const { options, variants } = setup;
  const selected = variants.find((v) => v.id === options.variant);
  const busy = running !== null;

  const variantOptions = variants.map((v) => ({
    value: v.id,
    label: `${VARIANT_NAMES[v.id] ?? v.id} · ${mb(v.size_bytes)} MB`,
    description: v.downloaded
      ? t("historySpeakers.downloaded")
      : t("historySpeakers.notDownloaded"),
  }));
  const speakerOptions = [
    { value: "0", label: t("historySpeakers.speakersAuto") },
    ...[1, 2, 3, 4].map((n) => ({ value: String(n), label: String(n) })),
  ];

  const handleDelete = async () => {
    if (!selected) return;
    const error = await deleteVariant(selected.id);
    if (error) notify({ title: error });
  };

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-line px-3 py-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-ink-3 grow">
          {t("historySpeakers.title")}
        </span>
        <CloseButton onClick={() => close(entryId)} />
      </div>

      <label className="flex flex-col gap-1">
        <span className="text-sm">{t("historySpeakers.language")}</span>
        <Dropdown
          options={languageOptions}
          selectedValue={options.language}
          onSelect={(language) => void setOptions({ language })}
          tone="neutral"
        />
        <span className="text-xs text-ink-3">
          {t("historySpeakers.languageHint")}
        </span>
      </label>

      <div className="flex flex-wrap gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-sm">{t("historySpeakers.speakers")}</span>
          <Dropdown
            options={speakerOptions}
            selectedValue={String(options.speakers)}
            onSelect={(value) => void setOptions({ speakers: Number(value) })}
            tone="neutral"
          />
        </label>
        <label className="flex flex-col gap-1 grow">
          <span className="text-sm">{t("historySpeakers.model")}</span>
          <div className="flex items-center gap-2">
            <Dropdown
              options={variantOptions}
              selectedValue={options.variant}
              onSelect={(variant) => void setOptions({ variant })}
              className="grow"
              tone="neutral"
            />
            {selected?.downloaded && (
              <Button
                variant="secondary"
                size="sm"
                disabled={busy}
                onClick={() => void handleDelete()}
              >
                {t("historySpeakers.delete")}
              </Button>
            )}
          </div>
        </label>
      </div>
      <p className="text-xs text-ink-3">
        {t("historySpeakers.speakersHint", { license: setup.license })}
      </p>

      <div>
        <Button
          variant="primary"
          size="sm"
          disabled={busy}
          onClick={() => void start(entryId)}
        >
          {selected && !selected.downloaded
            ? t("historySpeakers.downloadAndStart", {
                size: mb(selected.size_bytes),
              })
            : t("historySpeakers.start")}
        </Button>
        {busy && running !== entryId && (
          <span className="ml-3 text-xs text-ink-3">
            {t("historySpeakers.otherRunning")}
          </span>
        )}
      </div>
    </div>
  );
};

/** The options, progress, result or error of the entry's Speakers; nothing
 * when the panel is closed. */
export const EntrySpeakersPanel: React.FC<{
  entryId: number;
  sourceName: string | null;
}> = ({ entryId, sourceName }) => {
  const { t } = useTranslation();
  const result = useHistorySpeakersStore((s) => s.byEntry[entryId]);
  const { cancel, close, reset } = useHistorySpeakersStore();
  const [flashed, flash] = useFlash();

  if (!result) return null;
  const { status, done, total, turns, speakers, error } = result;

  if (status === "options") return <SpeakersOptionsForm entryId={entryId} />;

  if (
    status === "downloading" ||
    status === "detecting" ||
    status === "transcribing"
  ) {
    let line = t("historySpeakers.detecting");
    if (status === "downloading") {
      line = t("historySpeakers.downloading", {
        done: mb(done),
        total: mb(total),
      });
    } else if (status === "transcribing") {
      line = t("historySpeakers.transcribing", { done: done + 1, total });
    }
    const counted = status !== "detecting" && total > 0;
    const percent = counted ? Math.min(100, (done / total) * 100) : 0;
    return (
      <div className="flex flex-col gap-2 rounded-lg border border-line px-3 py-2">
        <div className="flex items-center gap-3">
          <span className="text-sm text-ink-3 grow">{line}</span>
          <Button variant="secondary" size="sm" onClick={() => void cancel()}>
            {t("historySpeakers.cancel")}
          </Button>
        </div>
        <div className="h-1.5 w-full rounded bg-line overflow-hidden">
          <div
            className={`h-full bg-logo-primary transition-[width] duration-300 ${
              counted ? "" : "w-full animate-pulse"
            }`}
            style={counted ? { width: `${percent}%` } : undefined}
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
        <CloseButton onClick={() => close(entryId)} />
      </div>
    );
  }

  const title = titleOf(sourceName);
  const text = formatTranscript(turns, title, (n) =>
    t("historySpeakers.speakerLabel", { n }),
  );

  const handleCopy = async () => {
    if (await copyToClipboard(text)) flash("copied");
    else notify({ title: t("translateSettings.file.copyFailed") });
  };

  const handleSave = async () => {
    const target = await save({
      defaultPath: `${title ?? "speakers"}.speakers.txt`,
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

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-line bg-wash px-3 py-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-ink-3 grow">
          {t("historySpeakers.resultTitle", { count: speakers })}
        </span>
        <CloseButton onClick={() => close(entryId)} />
      </div>
      <p className="text-sm text-text select-text cursor-text whitespace-pre-wrap break-words max-h-96 overflow-y-auto">
        {text}
      </p>
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
        <Button variant="secondary" size="sm" onClick={() => reset(entryId)}>
          {t("historySpeakers.again")}
        </Button>
      </div>
    </div>
  );
};
