// SelTranslate: Translation section — llama-server, model, output and hotkey.
import React, { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { open } from "@tauri-apps/plugin-dialog";
import { type as osType } from "@tauri-apps/plugin-os";
import { toast } from "sonner";
import { commands } from "@/bindings";
import type {
  OutputMode,
  SelectionSlot,
  SelTranslateSettings,
  ServerStatus,
  Slot,
} from "@/bindings";
import { Button } from "../../ui/Button";
import { Dropdown } from "../../ui/Dropdown";
import { Input } from "../../ui/Input";
import { SettingContainer } from "../../ui/SettingContainer";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { ToggleSwitch } from "../../ui/ToggleSwitch";
import { ShortcutInput } from "../ShortcutInput";
import { AccessibilityRow } from "./AccessibilityRow";
import { ModelSlots } from "./ModelSlots";
import { TranslateFile } from "./TranslateFile";
import { TranslateSetup } from "./TranslateSetup";
import { useTranslateSetupStore } from "./translateSetupStore";

// Plain-language choices behind two technical numbers (G3.1). The stored
// values stay numbers; one the list does not know (set by hand in the store)
// is listed as "Custom" so it is neither lost nor rewritten.
const CTX_PRESETS = [
  { value: 2048, key: "compact" },
  { value: 4096, key: "standard" },
  { value: 8192, key: "extended" },
] as const;
const CTX_STANDARD = 4096;
const GPU_PRESETS = [
  { value: 99, key: "gpu" },
  { value: 0, key: "cpu" },
] as const;
const GPU_ALL = 99;
const STATUS_POLL_MS = 3000;

type Save = (patch: Partial<SelTranslateSettings>) => Promise<boolean>;

interface PathFieldProps {
  title: string;
  description: string;
  value: string;
  filterName: string;
  /** Without one (an executable on macOS) the dialog shows every file. */
  extension?: string;
  browseLabel: string;
  onCommit: (value: string) => Promise<boolean>;
}

// A path with Browse: typed text is saved on blur or Enter, a picked file at once.
const PathField: React.FC<PathFieldProps> = ({
  title,
  description,
  value,
  filterName,
  extension,
  browseLabel,
  onCommit,
}) => {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);

  const commit = async (next: string) => {
    const trimmed = next.trim();
    if (trimmed === value) {
      setDraft(value);
      return;
    }
    if (!(await onCommit(trimmed))) setDraft(value);
  };

  const browse = async () => {
    const picked = await open({
      multiple: false,
      directory: false,
      defaultPath: value || undefined,
      filters: extension ? [{ name: filterName, extensions: [extension] }] : [],
    });
    if (typeof picked === "string") await commit(picked);
  };

  return (
    <SettingContainer
      title={title}
      description={description}
      descriptionMode="inline"
      grouped={true}
      layout="stacked"
    >
      <div className="flex items-center gap-2">
        <Input
          type="text"
          variant="compact"
          className="min-w-0 flex-1 font-mono"
          value={draft}
          spellCheck={false}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => void commit(draft)}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.currentTarget.blur();
          }}
        />
        <Button variant="secondary" onClick={() => void browse()}>
          {browseLabel}
        </Button>
      </div>
    </SettingContainer>
  );
};

interface NumberFieldProps {
  title: string;
  description: string;
  value: number;
  min: number;
  max: number;
  suffix?: string;
  note?: string;
  onCommit: (value: number) => Promise<boolean>;
}

// A whole number kept as text while typing; saved on blur or Enter when valid.
const NumberField: React.FC<NumberFieldProps> = ({
  title,
  description,
  value,
  min,
  max,
  suffix,
  note,
  onCommit,
}) => {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);

  const commit = async () => {
    const parsed = Number(draft);
    const valid =
      draft.trim() !== "" &&
      Number.isInteger(parsed) &&
      parsed >= min &&
      parsed <= max;
    if (!valid || parsed === value) {
      setDraft(String(value));
      return;
    }
    if (!(await onCommit(parsed))) setDraft(String(value));
  };

  return (
    <SettingContainer
      title={title}
      description={description}
      descriptionMode="tooltip"
      grouped={true}
    >
      <div className="flex items-center gap-2">
        <Input
          type="number"
          variant="compact"
          className="w-24 [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
          min={min}
          max={max}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => void commit()}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.currentTarget.blur();
          }}
        />
        {suffix && <span className="text-sm">{suffix}</span>}
        {note && <span className="text-sm text-mid-gray">{note}</span>}
      </div>
    </SettingContainer>
  );
};

export const TranslateSettings: React.FC = () => {
  const { t } = useTranslation();
  const [settings, setSettings] = useState<SelTranslateSettings | null>(null);
  const [status, setStatus] = useState<ServerStatus | null>(null);
  const [testing, setTesting] = useState(false);
  const [testError, setTestError] = useState<string | null>(null);
  const [fastAvailable, setFastAvailable] = useState(true);
  const [languages, setLanguages] = useState<{ code: string; name: string }[]>(
    [],
  );
  useEffect(() => {
    void commands.translatePopupLanguages().then(setLanguages);
  }, []);
  const mounted = useRef(true);
  // Unknown while the plan loads: the buttons stay usable, the backend refuses
  // a translation that cannot run anyway.
  const setupReady = useTranslateSetupStore((s) => s.plan?.ready ?? true);
  const refreshSetup = useTranslateSetupStore((s) => s.refresh);

  const refreshStatus = useCallback(async () => {
    const next = await commands.translateServerStatus();
    if (mounted.current) setStatus(next);
  }, []);

  useEffect(() => {
    mounted.current = true;
    void commands.getTranslateSettings().then((loaded) => {
      if (mounted.current) setSettings(loaded);
    });
    void refreshStatus();
    void commands.translateFastAvailable().then((ok) => {
      if (mounted.current) setFastAvailable(ok);
    });
    const timer = window.setInterval(
      () => void refreshStatus(),
      STATUS_POLL_MS,
    );
    return () => {
      mounted.current = false;
      window.clearInterval(timer);
    };
  }, [refreshStatus]);

  // Changing the server, model, GPU layers or context stops a running server
  // (the next translation starts it with the new values).
  const save: Save = async (patch) => {
    if (!settings) return false;
    const next = { ...settings, ...patch };
    const result = await commands.setTranslateSettings(next);
    if (result.status !== "ok") {
      toast.error(t("translateSettings.saveFailed", { error: result.error }));
      return false;
    }
    setSettings(next);
    setTestError(null);
    void commands.translateFastAvailable().then((ok) => {
      if (mounted.current) setFastAvailable(ok);
    });
    void refreshStatus();
    void refreshSetup();
    return true;
  };

  const test = async () => {
    setTesting(true);
    setTestError(null);
    const result = await commands.translateServerStart();
    if (!mounted.current) return;
    setTesting(false);
    if (result.status === "ok") setStatus(result.data);
    else {
      setTestError(result.error);
      void refreshStatus();
    }
  };

  const stop = async () => {
    await commands.translateServerStop();
    void refreshStatus();
  };

  if (!settings) {
    return <div className="max-w-3xl w-full mx-auto" />;
  }

  const outputOptions: { value: OutputMode; label: string }[] = [
    { value: "popup", label: t("translateSettings.output.popup") },
    { value: "replace", label: t("translateSettings.output.replace") },
  ];

  const targetOptions = [
    { value: "auto", label: t("translateSettings.target.auto") },
    ...languages.map((l) => ({ value: l.code, label: l.name })),
  ];

  const slotOptions: { value: Slot; label: string }[] = [
    { value: "fast", label: t("translateSettings.slot.fast") },
    { value: "quality", label: t("translateSettings.slot.quality") },
  ];

  const selectionSlotOptions: { value: SelectionSlot; label: string }[] = [
    ...slotOptions,
    { value: "auto", label: t("translateSettings.slot.auto") },
  ];

  const gpuLayers = settings.gpu_layers ?? GPU_ALL;
  const gpuOptions = [
    ...GPU_PRESETS.map(({ value, key }) => ({
      value: String(value),
      label: t(
        `translateSettings.gpuLayers.options.${key === "gpu" && osType() === "macos" ? "gpuMac" : key}`,
      ),
    })),
    ...(GPU_PRESETS.some((p) => p.value === gpuLayers)
      ? []
      : [
          {
            value: String(gpuLayers),
            label: t("translateSettings.gpuLayers.options.custom", {
              n: gpuLayers,
            }),
          },
        ]),
  ];

  const ctxSize = settings.ctx_size ?? CTX_STANDARD;
  const ctxOptions = [
    ...CTX_PRESETS.map(({ value, key }) => ({
      value: String(value),
      label: t(`translateSettings.ctxSize.options.${key}`),
    })),
    ...(CTX_PRESETS.some((p) => p.value === ctxSize)
      ? []
      : [
          {
            value: String(ctxSize),
            label: t("translateSettings.ctxSize.options.custom", {
              n: ctxSize,
            }),
          },
        ]),
  ];

  const state = testing ? "starting" : (status?.state ?? "stopped");
  let statusLine = t(`translateSettings.status.${state}`);
  if (state === "running" && status?.port != null) {
    statusLine = t("translateSettings.status.runningOn", {
      port: status.port,
    });
  }
  if (state === "running" && status?.last_start_secs != null) {
    statusLine += ` · ${t("translateSettings.status.startedIn", {
      secs: status.last_start_secs.toFixed(1),
    })}`;
  }

  const idle = settings.idle_unload_minutes ?? 0;

  const pathFields = (
    <>
      <PathField
        title={t("translateSettings.serverPath.title")}
        description={t("translateSettings.serverPath.description")}
        value={settings.llama_server_path ?? ""}
        filterName={t("translateSettings.serverPath.filter")}
        extension={osType() === "windows" ? "exe" : undefined}
        browseLabel={t("translateSettings.browse")}
        onCommit={(value) => save({ llama_server_path: value })}
      />
      <PathField
        title={t("translateSettings.modelPath.title")}
        description={t("translateSettings.modelPath.description")}
        value={settings.model_path ?? ""}
        filterName={t("translateSettings.modelPath.filter")}
        extension="gguf"
        browseLabel={t("translateSettings.browse")}
        onCommit={(value) => save({ model_path: value })}
      />
      <PathField
        title={t("translateSettings.fastModelPath.title")}
        description={
          fastAvailable
            ? t("translateSettings.fastModelPath.description")
            : t("translateSettings.fastModelPath.missing")
        }
        value={settings.fast_model_path ?? ""}
        filterName={t("translateSettings.modelPath.filter")}
        extension="gguf"
        browseLabel={t("translateSettings.browse")}
        onCommit={(value) => save({ fast_model_path: value })}
      />
    </>
  );

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      {osType() === "macos" && <AccessibilityRow />}
      <TranslateSetup advanced={pathFields} />

      <SettingsGroup title={t("translateSettings.hotkey.title")}>
        <ShortcutInput
          shortcutId="translate_selection"
          descriptionMode="tooltip"
          grouped={true}
        />
        <ShortcutInput
          shortcutId="translate_selection_quality"
          descriptionMode="tooltip"
          grouped={true}
        />
        <ToggleSwitch
          checked={settings.double_ctrl_enabled ?? false}
          onChange={(checked) => void save({ double_ctrl_enabled: checked })}
          label={t("translateSettings.doubleCtrl.title")}
          description={t("translateSettings.doubleCtrl.description")}
          descriptionMode="tooltip"
          grouped={true}
        />
        <ToggleSwitch
          checked={settings.auto_pause_fullscreen ?? false}
          onChange={(checked) => void save({ auto_pause_fullscreen: checked })}
          label={t("translateSettings.autoPauseFullscreen.title")}
          description={t(
            osType() === "macos"
              ? "translateSettings.autoPauseFullscreen.descriptionMac"
              : "translateSettings.autoPauseFullscreen.description",
          )}
          descriptionMode="tooltip"
          grouped={true}
        />
        <SettingContainer
          title={t("translateSettings.target.title")}
          description={t("translateSettings.target.description")}
          descriptionMode="tooltip"
          grouped={true}
        >
          <Dropdown
            options={targetOptions}
            selectedValue={settings.target_language ?? "auto"}
            onSelect={(value) => void save({ target_language: value })}
          />
        </SettingContainer>
        <SettingContainer
          title={t("translateSettings.output.title")}
          description={t("translateSettings.output.description")}
          descriptionMode="tooltip"
          grouped={true}
        >
          <Dropdown
            options={outputOptions}
            selectedValue={settings.output_mode ?? "popup"}
            onSelect={(value) =>
              void save({ output_mode: value as OutputMode })
            }
          />
        </SettingContainer>
        <SettingContainer
          title={t("translateSettings.slot.selectionTitle")}
          description={t("translateSettings.slot.selectionDescription")}
          descriptionMode="tooltip"
          grouped={true}
        >
          <Dropdown
            options={selectionSlotOptions}
            selectedValue={settings.selection_slot ?? "fast"}
            onSelect={(value) =>
              void save({ selection_slot: value as SelectionSlot })
            }
          />
        </SettingContainer>
      </SettingsGroup>

      <TranslateFile
        ready={setupReady}
        slot={settings.file_slot ?? "quality"}
        onSlotChange={(slot) => void save({ file_slot: slot })}
      />

      <SettingsGroup
        title={t("translateSettings.model.title")}
        description={t("translateSettings.model.description")}
      >
        <ModelSlots settings={settings} save={save} />
        <SettingContainer
          title={t("translateSettings.gpuLayers.title")}
          description={t(
            osType() === "macos"
              ? "translateSettings.gpuLayers.descriptionMac"
              : "translateSettings.gpuLayers.description",
          )}
          descriptionMode="tooltip"
          grouped={true}
        >
          <Dropdown
            options={gpuOptions}
            selectedValue={String(gpuLayers)}
            onSelect={(value) => void save({ gpu_layers: Number(value) })}
          />
        </SettingContainer>
        <SettingContainer
          title={t("translateSettings.ctxSize.title")}
          description={t("translateSettings.ctxSize.description")}
          descriptionMode="tooltip"
          grouped={true}
        >
          <Dropdown
            options={ctxOptions}
            selectedValue={String(ctxSize)}
            onSelect={(value) => void save({ ctx_size: Number(value) })}
          />
        </SettingContainer>
        <NumberField
          title={t("translateSettings.idle.title")}
          description={t("translateSettings.idle.description")}
          value={idle}
          min={0}
          max={1440}
          suffix={t("translateSettings.idle.unit")}
          note={idle === 0 ? t("translateSettings.idle.never") : undefined}
          onCommit={(value) => save({ idle_unload_minutes: value })}
        />
      </SettingsGroup>

      <SettingsGroup title={t("translateSettings.test.title")}>
        <div className="px-4 py-3 space-y-2">
          <div className="flex items-center gap-3">
            <Button
              variant="primary"
              onClick={() => void test()}
              disabled={testing || !setupReady}
            >
              {t("translateSettings.test.button")}
            </Button>
            {status?.state === "running" && !testing && (
              <Button variant="secondary" onClick={() => void stop()}>
                {t("translateSettings.test.stop")}
              </Button>
            )}
            <span className="text-sm text-mid-gray">
              {setupReady ? statusLine : t("translateSetup.setUpFirst")}
            </span>
          </div>
          {testError && (
            <p className="text-sm text-red-400 break-words whitespace-pre-wrap">
              {testError}
            </p>
          )}
        </div>
      </SettingsGroup>
    </div>
  );
};
