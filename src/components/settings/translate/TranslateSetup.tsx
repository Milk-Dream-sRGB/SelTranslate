// SelTranslate: the "Set up translation" card (DESIGN F.5) — one Download
// click fetches the llama.cpp engine and the TranslateGemma models picked by
// the video memory. Nothing is downloaded without a click.
import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { openUrl } from "@tauri-apps/plugin-opener";
import { commands } from "@/bindings";
import type { EngineFlavor, ErrorKind, PlanItem, SetupPlan } from "@/bindings";
import { Alert } from "../../ui/Alert";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { isSetupRunning, useTranslateSetupStore } from "./translateSetupStore";

const ENGINE_VULKAN = "engine-vulkan";
const ENGINE_CUDA = "engine-cuda";
const ENGINE_CUDART = "engine-cudart";
const ENGINE_METAL = "engine-metal";
const BUSY_POLL_MS = 2000;
const MODEL_FAST = "model-fast";
const MODEL_QUALITY = "model-quality";
const GEMMA_TERMS_URL = "https://ai.google.dev/gemma/terms";
const GEMMA_PROHIBITED_URL =
  "https://ai.google.dev/gemma/prohibited_use_policy";

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
// Kept free besides the download, and what the unpacked Vulkan engine takes
// (the same numbers as the backend check; the backend has the last word).
const SPACE_MARGIN = 512 * MIB;
const UNPACKED_VULKAN = 100 * MIB;
const UNPACKED_CUDA = 1300 * MIB;
const UNPACKED_METAL = 32 * MIB;

export const formatBytes = (bytes: number): string => {
  if (bytes >= GIB) return `${(bytes / GIB).toFixed(2)} GB`;
  if (bytes >= MIB) return `${(bytes / MIB).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
};

const byId = (plan: SetupPlan, id: string): PlanItem | undefined =>
  plan.items.find((item) => item.id === id);

const remaining = (item: PlanItem) =>
  Math.max(0, item.size - item.partial_bytes);

interface Space {
  /** Bytes still to download. */
  download: number;
  /** A volume has too little room for it. */
  short: boolean;
}

// Bytes to fetch for `queue`, and whether the volumes have room for them (the
// models' volume and the engine's, or one volume for both: the sum).
const spaceFor = (plan: SetupPlan, queue: PlanItem[]): Space => {
  let models = 0;
  let engine = 0;
  for (const item of queue) {
    if (item.kind === "model") models += remaining(item);
    else engine += remaining(item);
  }
  if (queue.some((item) => item.id === ENGINE_CUDA)) engine += UNPACKED_CUDA;
  else if (queue.some((item) => item.id === ENGINE_METAL))
    engine += UNPACKED_METAL;
  else if (queue.some((item) => item.kind === "engine"))
    engine += UNPACKED_VULKAN;
  const room = (need: number, free: number | null) =>
    need === 0 || free === null || free >= need + SPACE_MARGIN;
  const short =
    plan.models_volume === plan.engine_volume
      ? !room(models + engine, plan.free_models_bytes)
      : !room(models, plan.free_models_bytes) ||
        !room(engine, plan.free_engine_bytes);
  return {
    download: queue.reduce((sum, item) => sum + remaining(item), 0),
    short,
  };
};

const formatEta = (secs: number, minutes: string, hours: string): string => {
  const m = Math.max(1, Math.round(secs / 60));
  return m >= 60
    ? `${Math.floor(m / 60)} ${hours} ${m % 60} ${minutes}`
    : `${m} ${minutes}`;
};

interface RowProps {
  title: string;
  meta: string;
  note?: string;
  checked: boolean;
  /** Not a choice: the box shows what will be fetched. */
  locked?: boolean;
  status?: string;
  onChange?: (checked: boolean) => void;
}

const Row: React.FC<RowProps> = ({
  title,
  meta,
  note,
  checked,
  locked,
  status,
  onChange,
}) => (
  <label className="flex items-start gap-3 px-4 py-2">
    <input
      type="checkbox"
      className="mt-1 h-4 w-4 accent-logo-primary"
      checked={checked}
      disabled={locked || status === "installed"}
      onChange={(e) => onChange?.(e.target.checked)}
    />
    <span className="min-w-0 flex-1">
      <span className="block text-sm font-medium">{title}</span>
      <span className="block text-xs text-mid-gray">{meta}</span>
      {note && <span className="block text-xs text-mid-gray">{note}</span>}
    </span>
  </label>
);

interface ConfirmProps {
  open: boolean;
  title: string;
  text: string;
  confirmLabel: string;
  cancelLabel: string;
  onConfirm: () => void;
  onClose: () => void;
}

export const Confirm: React.FC<ConfirmProps> = ({
  open,
  title,
  text,
  confirmLabel,
  cancelLabel,
  onConfirm,
  onClose,
}) => (
  <Dialog
    open={open}
    title={title}
    closeLabel={cancelLabel}
    onOpenChange={(next) => !next && onClose()}
    footer={
      <div className="flex justify-end gap-2">
        <Button variant="secondary" onClick={onClose}>
          {cancelLabel}
        </Button>
        <Button variant="danger" onClick={onConfirm}>
          {confirmLabel}
        </Button>
      </div>
    }
  >
    <p className="text-sm">{text}</p>
  </Dialog>
);

interface TranslateSetupProps {
  /** The path fields of the "I already have the files" block. */
  advanced: React.ReactNode;
}

export const TranslateSetup: React.FC<TranslateSetupProps> = ({ advanced }) => {
  const { t } = useTranslation();
  const {
    plan,
    view,
    notices,
    skipped,
    lastItems,
    accepted,
    startError,
    init,
    setAccepted,
    start,
    cancel,
    discard,
    dismissNotices,
    dismissError,
    refresh,
  } = useTranslateSetupStore();
  // The user's changes to the model boxes; the rest follows the plan.
  const [picked, setPicked] = useState<Record<string, boolean>>({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [adding, setAdding] = useState(false);
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  // F2d: the engine chosen before Download, a switch that needs a download,
  // the deletions in Advanced, and whether a translation is running now.
  const [engineChoice, setEngineChoice] = useState<EngineFlavor | null>(null);
  const [pendingEngine, setPendingEngine] = useState<EngineFlavor | null>(null);
  const [engineMessage, setEngineMessage] = useState<string | null>(null);
  const [confirmDeleteEngine, setConfirmDeleteEngine] = useState(false);
  const [confirmDeleteModels, setConfirmDeleteModels] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => init(), [init]);

  // The engine cannot be switched under a running translation: the backend
  // refuses too; this only greys the buttons.
  useEffect(() => {
    let alive = true;
    const tick = () =>
      void commands.translateSetupBusy().then((b) => alive && setBusy(b));
    tick();
    const timer = window.setInterval(tick, BUSY_POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  if (!plan) return null;

  const running = isSetupRunning(view);
  const hasEngine = plan.engine.current !== "missing";
  // The Windows card chooses between Vulkan and CUDA; a Mac has Metal only.
  const hasEngineChoice = !!byId(plan, ENGINE_CUDA);
  const chosenEngine: EngineFlavor =
    engineChoice ?? (plan.unified_memory ? "metal" : "vulkan");
  const fast = byId(plan, MODEL_FAST);
  const quality = byId(plan, MODEL_QUALITY);
  // Quality is offered up front on cards that can use it; on a weak computer
  // it comes later through "Add the Quality model".
  const offersQuality = plan.tier === "both" || plan.tier === "partial";
  const modelOn = (item: PlanItem | undefined) =>
    !!item && !item.installed && (picked[item.id] ?? item.recommended);

  const engineItems = (flavor: EngineFlavor): PlanItem[] =>
    (flavor === "cuda"
      ? [ENGINE_CUDA, ENGINE_CUDART]
      : [flavor === "metal" ? ENGINE_METAL : ENGINE_VULKAN]
    )
      .map((id) => byId(plan, id))
      .filter((item): item is PlanItem => !!item);

  const defaultQueue = (): PlanItem[] => {
    const queue: PlanItem[] = [];
    if (!hasEngine) queue.push(...engineItems(chosenEngine));
    if (modelOn(fast) && fast) queue.push(fast);
    if (offersQuality && modelOn(quality) && quality) queue.push(quality);
    return queue;
  };
  // What Resume and Retry continue: the last click, else the half-downloaded
  // files and the rest of the default choice.
  const resumeQueue = (): PlanItem[] => {
    const wanted = new Set(lastItems);
    for (const item of plan.items) {
      if (item.partial_bytes > 0 && !item.installed) wanted.add(item.id);
    }
    if (wanted.size === 0) return defaultQueue();
    return plan.items.filter((item) => wanted.has(item.id) && !item.installed);
  };

  // `engine-cuda` brings its runtime by itself; the card lists it for the sizes.
  const download = async (queue: PlanItem[]) => {
    const ok = await start(
      queue.map((item) => item.id).filter((id) => id !== ENGINE_CUDART),
    );
    if (ok) {
      setAdding(false);
      setPendingEngine(null);
    }
    return ok;
  };

  const termsBox = (queue: PlanItem[]) =>
    queue.some((item) => item.kind === "model") && (
      <label className="flex items-start gap-3 px-4 py-2 text-sm">
        <input
          type="checkbox"
          className="mt-1 h-4 w-4 accent-logo-primary"
          checked={accepted}
          onChange={(e) => setAccepted(e.target.checked)}
        />
        <span>
          {t("translateSetup.terms.agree")}{" "}
          <button
            type="button"
            className="underline cursor-pointer"
            onClick={() => void openUrl(GEMMA_TERMS_URL)}
          >
            {t("translateSetup.terms.termsOfUse")}
          </button>{" "}
          {t("translateSetup.terms.and")}{" "}
          <button
            type="button"
            className="underline cursor-pointer"
            onClick={() => void openUrl(GEMMA_PROHIBITED_URL)}
          >
            {t("translateSetup.terms.prohibitedUse")}
          </button>
        </span>
      </label>
    );

  const totals = (queue: PlanItem[]) => {
    const space = spaceFor(plan, queue);
    const free = (volume: string, bytes: number | null) =>
      bytes === null
        ? ""
        : volume === ""
          ? t("translateSetup.freeUnified", { free: formatBytes(bytes) })
          : t("translateSetup.free", { volume, free: formatBytes(bytes) });
    const parts = [
      t("translateSetup.total", { size: formatBytes(space.download) }),
      free(plan.models_volume, plan.free_models_bytes),
    ];
    if (plan.engine_volume !== plan.models_volume) {
      parts.push(free(plan.engine_volume, plan.free_engine_bytes));
    }
    return (
      <p className={`px-4 py-2 text-sm ${space.short ? "text-red-400" : ""}`}>
        {parts.filter(Boolean).join(" · ")}
        {space.short && ` — ${t("translateSetup.notEnoughSpace")}`}
      </p>
    );
  };

  const meta = (item: PlanItem, what: string) =>
    [what, formatBytes(item.size), item.host, item.license].join(" · ");

  const status = (item: PlanItem) =>
    item.installed
      ? t("translateSetup.installed")
      : item.partial_bytes > 0
        ? t("translateSetup.partial", {
            done: formatBytes(item.partial_bytes),
            total: formatBytes(item.size),
          })
        : undefined;

  const hostOf = (itemId: string) =>
    itemId.startsWith("engine") ? "github.com" : "huggingface.co";

  const errorText = (kind: ErrorKind, itemId: string, message: string) =>
    t(
      `translateSetup.errors.${kind === "tar" && plan.unified_memory ? "tar_mac" : kind}`,
      {
        host: hostOf(itemId),
        detail: message,
      },
    );

  const engineName = (flavor: EngineFlavor) =>
    t(`translateSetup.engine.${flavor}`);
  const engineShort = (flavor: EngineFlavor) =>
    t(`translateSetup.engine.short.${flavor}`);

  const switchEngine = async (flavor: EngineFlavor) => {
    setEngineMessage(null);
    const downloaded =
      flavor === "cuda"
        ? plan.engine.cuda_downloaded
        : plan.engine.vulkan_downloaded;
    if (!downloaded) {
      setPendingEngine(flavor);
      return;
    }
    const result = await commands.translateSetupUseEngine(flavor);
    if (result.status !== "ok") setEngineMessage(result.error);
    else if (result.data === "cuda_failed") {
      setEngineMessage(t("translateSetup.errors.cuda_failed"));
    } else if (result.data) {
      setEngineMessage(t("translateSetup.engine.startFailed"));
    }
    await refresh();
  };

  // Engine: Vulkan / CUDA (F.10) — in Ready and in Advanced.
  const engineSwitch = () => {
    if (!hasEngineChoice) return null;
    const locked = busy || running;
    const size = (flavor: EngineFlavor) =>
      formatBytes(engineItems(flavor).reduce((sum, i) => sum + i.size, 0));
    const pendingQueue = pendingEngine ? engineItems(pendingEngine) : [];
    return (
      <div className="px-4 py-3 space-y-2">
        <p className="text-sm font-medium">
          {t("translateSetup.engine.title")}
        </p>
        <div className="flex flex-wrap gap-x-6 gap-y-1">
          {(["vulkan", "cuda"] as EngineFlavor[]).map((flavor) => (
            <label key={flavor} className="flex items-center gap-2 text-sm">
              <input
                type="radio"
                name="translate-engine"
                className="h-4 w-4 accent-logo-primary"
                checked={plan.engine.current === flavor}
                disabled={locked || (flavor === "cuda" && !plan.nvidia)}
                onChange={() => void switchEngine(flavor)}
              />
              {engineName(flavor)}
              {plan.engine.current === flavor && plan.engine.own_files
                ? ` ${t("translateSetup.engine.own")}`
                : ""}
            </label>
          ))}
        </div>
        {!plan.nvidia && (
          <p className="text-xs text-mid-gray">
            {t("translateSetup.engine.nvidiaOnly")}
          </p>
        )}
        {busy && (
          <p className="text-xs text-mid-gray">
            {t("translateSetup.engine.busy")}
          </p>
        )}
        {engineMessage && (
          <p className="text-sm text-red-400 break-words">{engineMessage}</p>
        )}
        {pendingEngine && (
          <div className="rounded-lg border border-mid-gray/20">
            <p className="px-4 py-2 text-sm">
              {t("translateSetup.engine.downloadQuestion", {
                name: engineShort(pendingEngine),
                size: size(pendingEngine),
              })}
            </p>
            {plan.engine.own_files && (
              <p className="px-4 pb-2 text-xs text-mid-gray">
                {t("translateSetup.engine.ownReplace")}
              </p>
            )}
            {totals(pendingQueue)}
            <div className="px-4 pb-3 flex gap-2">
              <Button
                variant="primary"
                onClick={() => void download(pendingQueue)}
                disabled={locked || spaceFor(plan, pendingQueue).short}
              >
                {t("translateSetup.download")}
              </Button>
              <Button
                variant="secondary"
                onClick={() => setPendingEngine(null)}
              >
                {t("translateSetup.cancel")}
              </Button>
            </div>
          </div>
        )}
      </div>
    );
  };

  const unusedNames = plan.engine.unused.map(engineShort).join(", ");
  const deletions = (
    <div className="px-4 py-3 space-y-2">
      <div className="flex flex-wrap gap-2">
        {plan.engine.unused.length > 0 && (
          <Button
            variant="secondary"
            disabled={busy || running}
            onClick={() => setConfirmDeleteEngine(true)}
          >
            {t("translateSetup.deleteEngine")}
          </Button>
        )}
        {plan.models_deletable_bytes > 0 && (
          <Button
            variant="secondary"
            disabled={busy || running}
            onClick={() => setConfirmDeleteModels(true)}
          >
            {t("translateSetup.deleteModels")}
          </Button>
        )}
      </div>
      <Confirm
        open={confirmDeleteEngine}
        title={t("translateSetup.deleteEngine")}
        text={t("translateSetup.deleteEngineConfirm", { names: unusedNames })}
        confirmLabel={t("translateSetup.delete")}
        cancelLabel={t("translateSetup.cancel")}
        onClose={() => setConfirmDeleteEngine(false)}
        onConfirm={() => {
          setConfirmDeleteEngine(false);
          void commands.translateSetupDeleteUnusedEngine().then((r) => {
            setEngineMessage(r.status === "ok" ? null : r.error);
            return refresh();
          });
        }}
      />
      <Confirm
        open={confirmDeleteModels}
        title={t("translateSetup.deleteModels")}
        text={t("translateSetup.deleteModelsConfirm", {
          size: formatBytes(plan.models_deletable_bytes),
        })}
        confirmLabel={t("translateSetup.delete")}
        cancelLabel={t("translateSetup.cancel")}
        onClose={() => setConfirmDeleteModels(false)}
        onConfirm={() => {
          setConfirmDeleteModels(false);
          void commands.translateSetupDeleteModels().then((r) => {
            setEngineMessage(r.status === "ok" ? null : r.error);
            return refresh();
          });
        }}
      />
    </div>
  );

  const advancedToggle = (
    <div className="px-4 py-2">
      <button
        type="button"
        className="text-sm underline cursor-pointer text-mid-gray hover:text-text"
        onClick={() => setShowAdvanced((v) => !v)}
        aria-expanded={showAdvanced}
      >
        {plan.ready
          ? t("translateSetup.advanced")
          : t("translateSetup.haveFiles")}
      </button>
    </div>
  );

  // ----- Downloading -----
  if (running && view.state === "running") {
    const p = view.progress;
    const percent =
      p.overall_total > 0
        ? Math.min(100, (p.overall_done / p.overall_total) * 100)
        : 0;
    const phase =
      p.phase === "retrying"
        ? t("translateSetup.phase.retrying", {
            secs: p.retry_in_secs ?? 0,
            attempt: p.attempt,
          })
        : t(`translateSetup.phase.${p.phase}`, {
            item: t(`translateSetup.items.${p.item}.name`),
          });
    let line = t("translateSetup.progress", {
      done: formatBytes(p.overall_done),
      total: formatBytes(p.overall_total),
    });
    if (p.phase === "downloading" && p.bytes_per_sec > 0) {
      line += ` · ${formatBytes(p.bytes_per_sec)}/s`;
      if (p.eta_secs !== null) {
        line += ` · ${t("translateSetup.eta", {
          eta: formatEta(
            p.eta_secs,
            t("translateSetup.units.min"),
            t("translateSetup.units.h"),
          ),
        })}`;
      }
    }
    return (
      <SettingsGroup title={t("translateSetup.title")}>
        <div className="px-4 py-3 space-y-2">
          <p className="text-sm">{phase}</p>
          <div className="h-2 w-full rounded bg-mid-gray/20 overflow-hidden">
            <div
              className="h-full bg-logo-primary transition-[width] duration-300"
              style={{ width: `${percent}%` }}
            />
          </div>
          <div className="flex items-center gap-3">
            <Button variant="danger" onClick={() => void cancel()}>
              {t("translateSetup.cancel")}
            </Button>
            <span className="text-sm text-mid-gray">{line}</span>
          </div>
        </div>
      </SettingsGroup>
    );
  }

  const partials = plan.items.filter(
    (item) => item.partial_bytes > 0 && !item.installed,
  );
  const paused = view.state === "idle" && partials.length > 0;

  const errorBlock = view.state === "error" && (
    <div className="px-4 py-3 space-y-2">
      <Alert variant="error" contained>
        {errorText(view.kind, view.item, view.message)}
      </Alert>
      {view.kind !== "disk_full" && view.kind !== "network" && (
        <p className="text-xs text-mid-gray break-words">{view.message}</p>
      )}
      {termsBox(resumeQueue())}
      <div className="flex gap-2">
        {view.kind !== "cuda_failed" && (
          <Button
            variant="primary"
            onClick={() => void download(resumeQueue())}
            disabled={
              resumeQueue().length === 0 ||
              (resumeQueue().some((i) => i.kind === "model") && !accepted)
            }
          >
            {t("translateSetup.retry")}
          </Button>
        )}
        {(plan.ready || view.kind === "cuda_failed") && (
          <Button variant="secondary" onClick={() => void dismissError()}>
            {t("translateSetup.dismiss")}
          </Button>
        )}
      </div>
    </div>
  );

  const startErrorLine = startError && (
    <p className="px-4 pb-2 text-sm text-red-400 break-words">{startError}</p>
  );

  // ----- Paused -----
  const pausedBlock = paused && (
    <div className="px-4 py-3 space-y-2">
      {(() => {
        const queue = resumeQueue();
        const done = queue.reduce((sum, i) => sum + i.partial_bytes, 0);
        const total = queue.reduce((sum, i) => sum + i.size, 0);
        const space = spaceFor(plan, queue);
        return (
          <>
            <p className="text-sm">
              {t("translateSetup.paused", {
                done: formatBytes(done),
                total: formatBytes(total),
              })}
            </p>
            {termsBox(queue)}
            {space.short && (
              <p className="text-sm text-red-400">
                {t("translateSetup.notEnoughSpace")}
              </p>
            )}
            <div className="flex gap-2">
              <Button
                variant="primary"
                onClick={() => void download(queue)}
                disabled={
                  space.short ||
                  (queue.some((i) => i.kind === "model") && !accepted)
                }
              >
                {t("translateSetup.resume")}
              </Button>
              <Button
                variant="secondary"
                onClick={() => setConfirmDiscard(true)}
              >
                {t("translateSetup.discard")}
              </Button>
            </div>
          </>
        );
      })()}
      <Confirm
        open={confirmDiscard}
        title={t("translateSetup.discard")}
        text={t("translateSetup.discardConfirm")}
        confirmLabel={t("translateSetup.discard")}
        cancelLabel={t("translateSetup.cancel")}
        onConfirm={() => {
          setConfirmDiscard(false);
          void discard();
        }}
        onClose={() => setConfirmDiscard(false)}
      />
    </div>
  );

  // ----- Ready -----
  const readyBlock = plan.ready && (
    <div className="px-4 py-3 space-y-2">
      <p className="text-sm">
        {plan.engine.own_files
          ? t("translateSetup.ready.own")
          : t("translateSetup.ready.line", {
              models: t(
                `translateSetup.ready.models.${
                  plan.has_fast && plan.has_quality
                    ? "both"
                    : plan.has_fast
                      ? "fast"
                      : "quality"
                }`,
              ),
              engine: t(`translateSetup.ready.engine.${plan.engine.current}`),
            })}
      </p>
      {notices.includes("no_gpu") && (
        <Alert variant="warning" contained>
          {t("translateSetup.notices.noGpu")}
        </Alert>
      )}
      {notices.includes("cuda_failed") && (
        <Alert variant="warning" contained>
          {t("translateSetup.notices.cudaFailed")}
        </Alert>
      )}
      {skipped.length > 0 && notices.length === 0 && (
        <p className="text-xs text-mid-gray">
          {t("translateSetup.notices.skipped")}
        </p>
      )}
      {notices.length > 0 && (
        <Button variant="ghost" size="sm" onClick={dismissNotices}>
          {t("translateSetup.dismiss")}
        </Button>
      )}
      {engineSwitch()}
      {!plan.has_quality && quality && !adding && (
        <div>
          <Button variant="secondary" onClick={() => setAdding(true)}>
            {t("translateSetup.addQuality")}
          </Button>
        </div>
      )}
      {adding && quality && (
        <div className="rounded-lg border border-mid-gray/20">
          <Row
            title={t("translateSetup.items.model-quality.name")}
            meta={meta(quality, t("translateSetup.items.model-quality.what"))}
            note={
              plan.tier === "partial"
                ? t("translateSetup.partialNote")
                : plan.tier === "both"
                  ? undefined
                  : t("translateSetup.slowNote")
            }
            checked
            locked
          />
          {termsBox([quality])}
          {totals([quality])}
          <div className="px-4 pb-3 flex gap-2">
            <Button
              variant="primary"
              onClick={() => void download([quality])}
              disabled={!accepted || spaceFor(plan, [quality]).short}
            >
              {t("translateSetup.download")}
            </Button>
            <Button variant="secondary" onClick={() => setAdding(false)}>
              {t("translateSetup.cancel")}
            </Button>
          </div>
        </div>
      )}
    </div>
  );

  // ----- Not set up -----
  const queue = defaultQueue();
  const hasModelQueued = queue.some((item) => item.kind === "model");
  const modelsPresent = plan.has_fast || plan.has_quality;
  const space = spaceFor(plan, queue);
  const canDownload =
    queue.length > 0 &&
    (hasModelQueued || modelsPresent) &&
    !space.short &&
    (!hasModelQueued || accepted);
  const gpuLine =
    plan.tier === "cpu" || !plan.adapter_name
      ? t("translateSetup.gpu.none")
      : plan.unified_memory
        ? t("translateSetup.gpu.unified", {
            name: plan.adapter_name,
            gb: Math.round(plan.dedicated_mib / 1024),
          })
        : t("translateSetup.gpu.found", {
            name: plan.adapter_name,
            gb: Math.round(plan.dedicated_mib / 1024),
          });
  const notSetUpBlock = !plan.ready && !paused && view.state !== "error" && (
    <>
      <div className="px-4 py-3 space-y-1">
        <p className="text-sm">{t("translateSetup.description")}</p>
        <p className="text-sm text-mid-gray">{gpuLine}</p>
      </div>
      {hasEngine ? (
        <Row
          title={t("translateSetup.engine.title")}
          meta={t("translateSetup.installed")}
          checked={false}
          status="installed"
        />
      ) : (
        <div className="px-4 py-2 space-y-1">
          <p className="text-sm font-medium">
            {t("translateSetup.engine.title")}
          </p>
          {(hasEngineChoice
            ? (["vulkan", "cuda"] as EngineFlavor[])
            : [chosenEngine]
          ).map((flavor) => {
            const items = engineItems(flavor);
            const bytes = items.reduce((sum, i) => sum + i.size, 0);
            return (
              <label key={flavor} className="flex items-start gap-3 text-sm">
                {hasEngineChoice && (
                  <input
                    type="radio"
                    name="translate-engine-choice"
                    className="mt-1 h-4 w-4 accent-logo-primary"
                    checked={chosenEngine === flavor}
                    disabled={flavor === "cuda" && !plan.nvidia}
                    onChange={() => setEngineChoice(flavor)}
                  />
                )}
                <span>
                  {engineName(flavor)}
                  <span className="block text-xs text-mid-gray">
                    {[
                      t(
                        flavor === "metal"
                          ? "translateSetup.items.engine-metal.what"
                          : "translateSetup.items.engine-vulkan.what",
                      ),
                      formatBytes(bytes),
                      items[0]?.host,
                      items[0]?.license,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </span>
                </span>
              </label>
            );
          })}
          {hasEngineChoice && !plan.nvidia && (
            <p className="text-xs text-mid-gray">
              {t("translateSetup.engine.nvidiaOnly")}
            </p>
          )}
        </div>
      )}
      {fast && (
        <Row
          title={t("translateSetup.items.model-fast.name")}
          meta={meta(fast, t("translateSetup.items.model-fast.what"))}
          note={status(fast)}
          checked={fast.installed || modelOn(fast)}
          status={fast.installed ? "installed" : undefined}
          onChange={(on) => setPicked({ ...picked, [fast.id]: on })}
        />
      )}
      {offersQuality && quality && (
        <Row
          title={t("translateSetup.items.model-quality.name")}
          meta={meta(quality, t("translateSetup.items.model-quality.what"))}
          note={
            status(quality) ??
            (plan.tier === "partial"
              ? t(
                  plan.unified_memory
                    ? "translateSetup.partialNoteUnified"
                    : "translateSetup.partialNote",
                )
              : undefined)
          }
          checked={quality.installed || modelOn(quality)}
          status={quality.installed ? "installed" : undefined}
          onChange={(on) => setPicked({ ...picked, [quality.id]: on })}
        />
      )}
      {totals(queue)}
      {termsBox(queue)}
      <div className="px-4 py-3 space-y-2">
        <Button
          variant="primary"
          onClick={() => void download(queue)}
          disabled={!canDownload}
        >
          {t("translateSetup.download")}
        </Button>
        <p className="text-xs text-mid-gray">{t("translateSetup.credits")}</p>
      </div>
    </>
  );

  return (
    <SettingsGroup title={t("translateSetup.title")}>
      {errorBlock}
      {pausedBlock}
      {readyBlock}
      {notSetUpBlock}
      {startErrorLine}
      {advancedToggle}
      {showAdvanced && (
        <div className="divide-y divide-mid-gray/20">
          {!plan.ready && hasEngine && engineSwitch()}
          {advanced}
          {(plan.engine.unused.length > 0 || plan.models_deletable_bytes > 0) &&
            deletions}
          {engineMessage && plan.ready && (
            <p className="px-4 py-2 text-sm text-red-400 break-words">
              {engineMessage}
            </p>
          )}
        </div>
      )}
    </SettingsGroup>
  );
};
