// SelTranslate (I): the model of the Fast and Quality slots, chosen from a
// list (or any GGUF with Browse…), and the prompt style of each model.
import React, { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { open } from "@tauri-apps/plugin-dialog";
import { type as osType } from "@tauri-apps/plugin-os";
import { commands } from "@/bindings";
import type {
  ModelEntry,
  PromptChoice,
  PromptStyle,
  SelTranslateSettings,
  Slot,
} from "@/bindings";
import { Button } from "../../ui/Button";
import { Dropdown } from "../../ui/Dropdown";
import { SettingContainer } from "../../ui/SettingContainer";
import { Textarea } from "../../ui/Textarea";

// On a Mac the model shares one memory with everything else (DESIGN J.9).
const sizeKey =
  osType() === "macos"
    ? "translateSettings.models.sizeMac"
    : "translateSettings.models.size";

const BROWSE = "__browse__";
const GIB = 1024 * 1024 * 1024;
// File size + about 1 GB for the context and buffers (an honest guess).
const VRAM_OVERHEAD = GIB;
const MAX_CUSTOM_CHARS = 4000;
const STYLES: PromptStyle[] = [
  "standard",
  "natural",
  "gaming",
  "software",
  "literal",
  "custom",
];

type Save = (patch: Partial<SelTranslateSettings>) => Promise<boolean>;

const fileName = (path: string) => path.split(/[\\/]/).pop() ?? "";
// The key of `prompt_styles`, as `prompt_key` in settings.rs.
const promptKey = (path: string) => fileName(path).toLowerCase();
const isTranslateGemma = (path: string) =>
  fileName(path).toLowerCase().includes("translategemma");
const gb = (bytes: number) => (bytes / GIB).toFixed(1);

/** Same rules as `custom_prompt_problem` in prompt.rs. */
const customProblem = (
  prompt: string,
): "noText" | "twoTexts" | "tooLong" | null => {
  const count = prompt.split("{text}").length - 1;
  if (count === 0) return "noText";
  if (count > 1) return "twoTexts";
  if ([...prompt].length > MAX_CUSTOM_CHARS) return "tooLong";
  return null;
};

interface SlotProps {
  slot: Slot;
  settings: SelTranslateSettings;
  models: ModelEntry[];
  save: Save;
  onModelsChanged: () => void;
}

const SlotModel: React.FC<SlotProps> = ({
  slot,
  settings,
  models,
  save,
  onModelsChanged,
}) => {
  const { t } = useTranslation();
  const path =
    (slot === "fast" ? settings.fast_model_path : settings.model_path) ?? "";
  const key = promptKey(path);
  const choice: PromptChoice = settings.prompt_styles?.[key] ?? {};
  const style: PromptStyle = choice.style ?? "standard";
  const [draft, setDraft] = useState(choice.custom ?? "");
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setDraft(choice.custom ?? "");
    setError(null);
  }, [key, choice.custom]);

  const setPath = async (next: string) => {
    if (next === path) return;
    const ok = await save(
      slot === "fast" ? { fast_model_path: next } : { model_path: next },
    );
    if (ok) onModelsChanged();
  };

  const browse = async () => {
    const picked = await open({
      multiple: false,
      directory: false,
      defaultPath: path || undefined,
      filters: [
        { name: t("translateSettings.modelPath.filter"), extensions: ["gguf"] },
      ],
    });
    if (typeof picked === "string") await setPath(picked);
  };

  const saveChoice = (next: PromptChoice) =>
    save({ prompt_styles: { ...settings.prompt_styles, [key]: next } });

  const selectStyle = async (next: PromptStyle) => {
    if (next === style) return;
    if (next !== "custom") {
      await saveChoice({ style: next, custom: choice.custom ?? "" });
      return;
    }
    // Custom starts from the full prompt of the style chosen so far.
    const custom = choice.custom?.trim()
      ? choice.custom
      : await commands.translatePromptTemplate(style);
    await saveChoice({ style: "custom", custom });
  };

  const saveCustom = async () => {
    const problem = customProblem(draft);
    if (problem) {
      setError(
        t(`translateSettings.promptStyle.errors.${problem}`, {
          max: MAX_CUSTOM_CHARS,
        }),
      );
      return;
    }
    setError(null);
    if (!(await saveChoice({ style: "custom", custom: draft }))) {
      setDraft(choice.custom ?? "");
    }
  };

  const reset = async () => {
    setError(null);
    await saveChoice({ style: "standard", custom: "" });
  };

  const known = models.find((m) => m.path === path);
  const options = models.map((m) => ({
    value: m.path,
    label: m.name,
    description: t(sizeKey, {
      size: gb(m.size_bytes),
      vram: gb(m.size_bytes + VRAM_OVERHEAD),
    }),
  }));
  if (path && !known) {
    options.unshift({
      value: path,
      label: t("translateSettings.models.notFound", {
        name: fileName(path).replace(/\.gguf$/i, ""),
      }),
      description: "",
    });
  }
  options.push({
    value: BROWSE,
    label: t("translateSettings.browse"),
    description: t("translateSettings.models.browseHint"),
  });

  const styleOptions = STYLES.map((s) => ({
    value: s,
    label: t(`translateSettings.promptStyle.options.${s}.label`),
    description: t(`translateSettings.promptStyle.options.${s}.description`),
  }));
  const dirty = draft !== (choice.custom ?? "");

  return (
    <>
      <SettingContainer
        title={t(`translateSettings.models.${slot}.title`)}
        description={t(`translateSettings.models.${slot}.description`)}
        descriptionMode="tooltip"
        grouped={true}
      >
        <div className="flex flex-col items-end gap-1 min-w-0">
          <Dropdown
            className="min-w-[240px]"
            options={options}
            selectedValue={path || null}
            placeholder={t("translateSettings.models.none")}
            onOpen={onModelsChanged}
            onSelect={(value) =>
              void (value === BROWSE ? browse() : setPath(value))
            }
          />
          {known && (
            <span className="text-xs text-mid-gray">
              {t(sizeKey, {
                size: gb(known.size_bytes),
                vram: gb(known.size_bytes + VRAM_OVERHEAD),
              })}
            </span>
          )}
          {path && (
            <span
              className="text-xs text-mid-gray font-mono truncate max-w-[340px]"
              title={path}
            >
              {path}
            </span>
          )}
        </div>
      </SettingContainer>
      {path && (
        <SettingContainer
          title={t("translateSettings.promptStyle.title")}
          description={
            isTranslateGemma(path)
              ? `${t("translateSettings.promptStyle.description")} ${t("translateSettings.promptStyle.translateGemmaHint")}`
              : t("translateSettings.promptStyle.description")
          }
          descriptionMode="tooltip"
          grouped={true}
          layout={style === "custom" ? "stacked" : "horizontal"}
        >
          <div className="flex flex-col gap-2 w-full">
            <Dropdown
              className="min-w-[240px] self-end"
              options={styleOptions}
              selectedValue={style}
              onSelect={(value) => void selectStyle(value as PromptStyle)}
            />
            {style === "custom" && (
              <>
                <Textarea
                  className="w-full font-mono font-normal min-h-[160px]"
                  value={draft}
                  spellCheck={false}
                  onChange={(e) => setDraft(e.target.value)}
                />
                <p className="text-xs text-mid-gray">
                  {t("translateSettings.promptStyle.placeholders")}
                </p>
                {error && <p className="text-sm text-red-400">{error}</p>}
                <div className="flex gap-2 justify-end">
                  <Button variant="secondary" onClick={() => void reset()}>
                    {t("translateSettings.promptStyle.reset")}
                  </Button>
                  <Button
                    variant="primary"
                    disabled={!dirty}
                    onClick={() => void saveCustom()}
                  >
                    {t("translateSettings.promptStyle.save")}
                  </Button>
                </div>
              </>
            )}
          </div>
        </SettingContainer>
      )}
    </>
  );
};

interface ModelSlotsProps {
  settings: SelTranslateSettings;
  save: Save;
}

export const ModelSlots: React.FC<ModelSlotsProps> = ({ settings, save }) => {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const refresh = useCallback(() => {
    void commands.translateListModels().then(setModels);
  }, []);
  useEffect(refresh, [refresh]);

  return (
    <>
      {(["fast", "quality"] as const).map((slot) => (
        <SlotModel
          key={slot}
          slot={slot}
          settings={settings}
          models={models}
          save={save}
          onModelsChanged={refresh}
        />
      ))}
    </>
  );
};
