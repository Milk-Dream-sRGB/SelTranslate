//! Set up translation (DESIGN F): one Download click fetches the llama.cpp
//! engine and the TranslateGemma models picked by the video memory. The state
//! lives here (the card may be reopened mid-download); nothing is downloaded
//! without a click, and an interrupted download stays as `.partial` (Resume).

pub mod catalog;
pub mod install;
pub mod tier;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use hf_hub::api::tokio::CancellationToken;
use log::{info, warn};
use serde::Serialize;
use specta::Type;
use tauri::{AppHandle, Emitter, Manager};

use crate::managers::model::{HttpDownloadEvent, HttpDownloadOutcome, ModelManager};

use super::llama_server::LlamaServer;
use super::settings::{self, SelTranslateSettings};
use catalog::{EngineFlavor, Item, ItemKind, SERVER_EXE};
use install::{ErrorKind, SetupError};
use tier::{Tier, TierPick};

/// Pauses before the automatic retries of a network error (DESIGN F.3).
const RETRY_DELAYS_SECS: [u32; 3] = [2, 5, 15];
/// A damaged file is downloaded once more from scratch.
const HASH_RETRIES: u32 = 1;
/// Kept free on a volume besides what the download needs.
const SPACE_MARGIN: u64 = 512 << 20;
/// Window of the speed shown on the card.
const SPEED_WINDOW: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Downloading,
    Verifying,
    Extracting,
    CheckingEngine,
    Retrying,
}

#[derive(Debug, Clone, PartialEq, Serialize, Type)]
pub struct SetupProgress {
    pub item: String,
    pub phase: Phase,
    pub item_done: u64,
    pub item_total: u64,
    pub overall_done: u64,
    pub overall_total: u64,
    pub bytes_per_sec: u64,
    pub eta_secs: Option<u64>,
    pub retry_in_secs: Option<u32>,
    /// Number of the automatic retry (1–3) while `phase` is `retrying`.
    pub attempt: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Type)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum SetupStateView {
    Idle,
    Running {
        progress: SetupProgress,
    },
    Error {
        item: String,
        kind: ErrorKind,
        message: String,
    },
}

/// Things the card says after a successful run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum Notice {
    /// The engine does not see the video card: the processor, Quality skipped.
    NoGpu,
    /// CUDA did not start; Vulkan was installed instead.
    CudaFailed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Type)]
pub struct SetupDone {
    pub installed: Vec<String>,
    pub gpu_layers: u32,
    pub engine_sees_gpu: Option<bool>,
    pub skipped: Vec<String>,
    pub notices: Vec<Notice>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Type)]
pub struct SetupErrorEvent {
    pub item: String,
    pub kind: ErrorKind,
    pub message: String,
}

pub struct SetupState {
    running: AtomicBool,
    token: Mutex<Option<CancellationToken>>,
    view: Mutex<SetupStateView>,
}

impl Default for SetupState {
    fn default() -> Self {
        Self {
            running: AtomicBool::new(false),
            token: Mutex::new(None),
            view: Mutex::new(SetupStateView::Idle),
        }
    }
}

impl SetupState {
    fn set_view(&self, view: SetupStateView) {
        *self.view.lock().unwrap_or_else(|e| e.into_inner()) = view;
    }

    fn view(&self) -> SetupStateView {
        self.view.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

/// Which engine `llama_server_path` is, by the DLL next to it (F.10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum EngineKind {
    Vulkan,
    Cuda,
    /// macOS: the official build with Metal built in.
    Metal,
    Cpu,
    Missing,
}

pub fn engine_kind(exe: &Path) -> EngineKind {
    if !exe.is_file() {
        return EngineKind::Missing;
    }
    if cfg!(target_os = "macos") {
        return EngineKind::Metal;
    }
    let dir = exe.parent().unwrap_or(Path::new(""));
    if dir.join("ggml-cuda.dll").is_file() {
        EngineKind::Cuda
    } else if dir.join("ggml-vulkan.dll").is_file() {
        EngineKind::Vulkan
    } else {
        EngineKind::Cpu
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Type)]
pub struct EngineInfo {
    pub current: EngineKind,
    /// The engine is not one the setup installed (a path chosen by hand).
    pub own_files: bool,
    pub vulkan_downloaded: bool,
    pub cuda_downloaded: bool,
    /// Engines the setup downloaded that `llama_server_path` is not in: what
    /// "Delete the unused engine" removes.
    pub unused: Vec<EngineFlavor>,
}

pub fn engine_info(app_data: &Path, exe: &Path) -> EngineInfo {
    EngineInfo {
        current: engine_kind(exe),
        own_files: !exe.starts_with(catalog::engines_dir(app_data)),
        vulkan_downloaded: catalog::engine_exe(app_data, EngineFlavor::Vulkan).is_file(),
        cuda_downloaded: catalog::engine_exe(app_data, EngineFlavor::Cuda).is_file(),
        unused: [
            EngineFlavor::Vulkan,
            EngineFlavor::Cuda,
            EngineFlavor::Metal,
        ]
        .into_iter()
        .filter(|f| {
            catalog::engine_exe(app_data, *f).is_file()
                && !exe.starts_with(catalog::engine_dir(app_data, *f))
        })
        .collect(),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Type)]
pub struct PlanItem {
    pub id: String,
    pub kind: ItemKind,
    pub size: u64,
    pub installed: bool,
    pub partial_bytes: u64,
    pub recommended: bool,
    pub host: String,
    pub license: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Type)]
pub struct SetupPlan {
    pub tier: Tier,
    pub adapter_name: Option<String>,
    pub dedicated_mib: u64,
    pub nvidia: bool,
    /// macOS: the video memory is the shared memory (`dedicated_mib` is all of it).
    pub unified_memory: bool,
    pub items: Vec<PlanItem>,
    pub models_volume: String,
    pub engine_volume: String,
    pub free_models_bytes: Option<u64>,
    pub free_engine_bytes: Option<u64>,
    pub ready: bool,
    pub has_fast: bool,
    pub has_quality: bool,
    pub engine: EngineInfo,
    /// Size of the translation models in the Hugging Face cache that the
    /// settings use ("Delete translation models"); 0: nothing to delete.
    pub models_deletable_bytes: u64,
    pub state: SetupStateView,
}

fn partial_len(app_data: &Path, hub: &Path, item: &Item) -> u64 {
    std::fs::metadata(catalog::partial_path(app_data, hub, item)).map_or(0, |m| m.len())
}

/// The catalog as the card shows it: what is on disk already, what is half
/// downloaded, what is ticked by default.
fn plan_items(
    app_data: &Path,
    hub: &Path,
    s: &SelTranslateSettings,
    pick: &TierPick,
) -> Vec<PlanItem> {
    catalog::offered()
        .map(|item| {
            let installed = match item.id {
                catalog::ENGINE_VULKAN => {
                    catalog::engine_exe(app_data, EngineFlavor::Vulkan).is_file()
                }
                catalog::ENGINE_METAL => {
                    catalog::engine_exe(app_data, EngineFlavor::Metal).is_file()
                }
                catalog::ENGINE_CUDA | catalog::ENGINE_CUDART => {
                    catalog::engine_exe(app_data, EngineFlavor::Cuda).is_file()
                }
                catalog::MODEL_FAST => s.fast_available() || model_cached(hub, item),
                _ => s.quality_available() || model_cached(hub, item),
            };
            let recommended = match item.id {
                catalog::ENGINE_VULKAN | catalog::ENGINE_METAL | catalog::MODEL_FAST => true,
                catalog::MODEL_QUALITY => pick.quality_default,
                _ => false,
            };
            PlanItem {
                id: item.id.into(),
                kind: item.kind,
                size: item.size,
                installed,
                partial_bytes: if installed {
                    0
                } else {
                    partial_len(app_data, hub, item)
                },
                recommended,
                host: item.host().into(),
                license: item.license.into(),
            }
        })
        .collect()
}

fn model_cached(hub: &Path, item: &Item) -> bool {
    catalog::model_file(hub, item).is_some_and(|p| p.is_file())
}

/// Bytes still to write on the models' and the engine's volume for `queue`.
fn needed_bytes(app_data: &Path, hub: &Path, queue: &[&Item]) -> (u64, u64) {
    let mut models = 0;
    let mut engine = 0;
    let mut flavors = Vec::new();
    for item in queue {
        let rest = item.size.saturating_sub(partial_len(app_data, hub, item));
        match item.kind {
            ItemKind::Model => models += rest,
            ItemKind::Engine => {
                engine += rest;
                if let Some(flavor) = flavor_of(item) {
                    if !flavors.contains(&flavor) {
                        flavors.push(flavor);
                        engine += catalog::unpacked_engine_bytes(flavor);
                    }
                }
            }
        }
    }
    (models, engine)
}

fn flavor_of(item: &Item) -> Option<EngineFlavor> {
    match item.id {
        catalog::ENGINE_VULKAN => Some(EngineFlavor::Vulkan),
        catalog::ENGINE_METAL => Some(EngineFlavor::Metal),
        catalog::ENGINE_CUDA | catalog::ENGINE_CUDART => Some(EngineFlavor::Cuda),
        _ => None,
    }
}

fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / f64::from(1u32 << 30))
}

/// Fails with `disk_full` when a volume has less than it needs (one volume
/// for both: the sum).
fn check_space(
    needs: (u64, u64),
    volumes: (&str, &str),
    free: (Option<u64>, Option<u64>),
) -> Result<(), SetupError> {
    let (models, engine) = needs;
    let mut checks = vec![(volumes.0, models, free.0)];
    if volumes.0 == volumes.1 {
        checks[0].1 += engine;
    } else {
        checks.push((volumes.1, engine, free.1));
    }
    for (volume, need, free) in checks {
        if need == 0 {
            continue;
        }
        if let Some(free) = free {
            if free < need + SPACE_MARGIN {
                return Err(SetupError::new(
                    ErrorKind::DiskFull,
                    format!(
                        "{volume} need {}, free {}",
                        gb(need + SPACE_MARGIN),
                        gb(free)
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Ids from the card → the catalog items to fetch, in order: the engine
/// (CUDA brings its runtime zip), then Fast, then Quality.
fn expand_queue(ids: &[String]) -> Result<Vec<&'static Item>, String> {
    let mut wanted = Vec::new();
    for id in ids {
        let item = catalog::offered()
            .find(|i| i.id == id)
            .ok_or_else(|| format!("Unknown item {id}"))?;
        wanted.push(item.id);
        if item.id == catalog::ENGINE_CUDA {
            wanted.push(catalog::ENGINE_CUDART);
        }
    }
    if wanted.contains(&catalog::ENGINE_VULKAN) && wanted.contains(&catalog::ENGINE_CUDA) {
        return Err("Choose one engine".into());
    }
    Ok(catalog::offered()
        .filter(|item| wanted.contains(&item.id))
        .collect())
}

/// What a finished run changes in the settings.
#[derive(Debug, Default, Clone, PartialEq)]
struct RunResult {
    engine_exe: Option<PathBuf>,
    engine_sees_gpu: Option<bool>,
    fast: Option<PathBuf>,
    quality: Option<PathBuf>,
}

/// Writes only the paths, slots and `gpu_layers` (DESIGN F.3, step 7). An
/// engine switch on a working setup keeps the slots and the layers unless
/// the new engine does not see the card.
fn apply_result(s: &mut SelTranslateSettings, r: &RunResult, tier: Tier, ready_before: bool) {
    if let Some(exe) = &r.engine_exe {
        s.llama_server_path = exe.display().to_string();
    }
    if let Some(p) = &r.fast {
        s.fast_model_path = p.display().to_string();
    }
    if let Some(p) = &r.quality {
        s.model_path = p.display().to_string();
    }
    let models_changed = r.fast.is_some() || r.quality.is_some();
    if models_changed || !ready_before {
        let (fast, quality) = (s.fast_available(), s.quality_available());
        if models_changed {
            if !fast {
                s.fast_model_path.clear();
            }
            if !quality {
                s.model_path.clear();
            }
        }
        let wanted = tier::settings_for(tier, fast, quality, r.engine_sees_gpu.unwrap_or(true));
        if models_changed {
            s.selection_slot = wanted.selection_slot;
            s.file_slot = wanted.file_slot;
        }
        if r.engine_sees_gpu.is_some() || !tier.uses_gpu() {
            s.gpu_layers = wanted.gpu_layers;
        }
    } else if r.engine_sees_gpu == Some(false) {
        s.gpu_layers = 0;
    }
}

/// The running download: emits progress and keeps the view for the card.
struct Ctx {
    app: AppHandle,
    token: CancellationToken,
    app_data: PathBuf,
    hub: PathBuf,
    meter: Mutex<Meter>,
}

struct Meter {
    done_before: u64,
    total: u64,
    item: &'static str,
    item_total: u64,
    item_done: u64,
    samples: VecDeque<(Instant, u64)>,
}

/// Shows `progress` on the card: the view for a reopened page and the event.
fn publish(app: &AppHandle, progress: SetupProgress) {
    app.state::<SetupState>().set_view(SetupStateView::Running {
        progress: progress.clone(),
    });
    let _ = app.emit("translate-setup-progress", progress);
}

fn tag(item: &'static str) -> impl Fn(SetupError) -> (String, SetupError) {
    move |e| (item.to_string(), e)
}

impl Ctx {
    fn emit(&self, phase: Phase, retry: Option<(u32, u32)>) {
        publish(&self.app, self.snapshot(phase, retry));
    }

    fn snapshot(&self, phase: Phase, retry: Option<(u32, u32)>) -> SetupProgress {
        let mut m = self.meter.lock().unwrap_or_else(|e| e.into_inner());
        let overall = m.done_before + m.item_done;
        let now = Instant::now();
        m.samples.push_back((now, overall));
        while m
            .samples
            .front()
            .is_some_and(|(t, _)| now.duration_since(*t) > SPEED_WINDOW)
        {
            m.samples.pop_front();
        }
        let bytes_per_sec = match (m.samples.front(), m.samples.back()) {
            (Some((t0, b0)), Some((t1, b1))) if t1 > t0 && phase == Phase::Downloading => {
                ((b1 - b0) as f64 / t1.duration_since(*t0).as_secs_f64()) as u64
            }
            _ => 0,
        };
        SetupProgress {
            item: m.item.into(),
            phase,
            item_done: m.item_done,
            item_total: m.item_total,
            overall_done: overall,
            overall_total: m.total,
            bytes_per_sec,
            eta_secs: (bytes_per_sec > 0).then(|| m.total.saturating_sub(overall) / bytes_per_sec),
            retry_in_secs: retry.map(|(secs, _)| secs),
            attempt: retry.map_or(0, |(_, n)| n),
        }
    }

    fn start_item(&self, item: &'static Item) {
        let mut m = self.meter.lock().unwrap_or_else(|e| e.into_inner());
        m.item = item.id;
        m.item_total = item.size;
        m.item_done = 0;
        m.samples.clear();
    }

    fn finish_item(&self) {
        let mut m = self.meter.lock().unwrap_or_else(|e| e.into_inner());
        m.done_before += m.item_total;
        m.item_done = 0;
    }

    fn add_total(&self, bytes: u64) {
        self.meter.lock().unwrap_or_else(|e| e.into_inner()).total += bytes;
    }

    fn on_download_event(&self, event: HttpDownloadEvent<'_>) {
        match event {
            HttpDownloadEvent::Progress(p) => {
                self.meter
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .item_done = p.downloaded;
                self.emit(Phase::Downloading, None);
            }
            HttpDownloadEvent::VerificationStarted => self.emit(Phase::Verifying, None),
            HttpDownloadEvent::VerificationCompleted => {}
        }
    }

    /// Downloads one catalog item into its `.partial` with the automatic
    /// retries; `Ok(false)` when cancelled.
    async fn fetch(&self, item: &'static Item) -> Result<bool, SetupError> {
        self.start_item(item);
        let partial = catalog::partial_path(&self.app_data, &self.hub, item);
        if let Some(dir) = partial.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| SetupError::new(ErrorKind::Other, e.to_string()))?;
        }
        let mut network_retries = 0;
        let mut hash_retries = 0;
        loop {
            self.emit(Phase::Downloading, None);
            let result = match install::resolve_redirects(item.url).await {
                Ok(url) => ModelManager::download_http_resumable_with_events(
                    item.id,
                    &url,
                    &partial,
                    Some(item.size),
                    Some(item.sha256),
                    &self.token,
                    &|event| self.on_download_event(event),
                )
                .await
                .map_err(|e| SetupError::new(install::classify(&e), format!("{e:#}"))),
                Err(e) => Err(e),
            };
            match result {
                Ok(HttpDownloadOutcome::Completed) => return Ok(true),
                Ok(HttpDownloadOutcome::Cancelled) => return Ok(false),
                Err(e)
                    if e.kind == ErrorKind::Network
                        && network_retries < RETRY_DELAYS_SECS.len() =>
                {
                    let secs = RETRY_DELAYS_SECS[network_retries];
                    network_retries += 1;
                    warn!(
                        "Translate setup: {} failed ({}), retry {network_retries} in {secs} s",
                        item.id, e.message
                    );
                    self.emit(Phase::Retrying, Some((secs, network_retries as u32)));
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_secs(secs.into())) => {}
                        _ = self.token.cancelled() => return Ok(false),
                    }
                }
                Err(e) if e.kind == ErrorKind::Hash && hash_retries < HASH_RETRIES => {
                    hash_retries += 1;
                    warn!("Translate setup: {} is damaged, downloading again", item.id);
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Downloads and installs one engine; returns the devices it sees, or
    /// `None` when cancelled.
    async fn engine(&self, flavor: EngineFlavor) -> Result<Option<Vec<String>>, SetupError> {
        for id in flavor.items() {
            let item = catalog::item(id).expect("catalog engine item");
            if !self.fetch(item).await? {
                return Ok(None);
            }
            self.finish_item();
        }
        self.emit(Phase::Extracting, None);
        let checking = self.snapshot(Phase::CheckingEngine, None);
        let app = self.app.clone();
        let app_data = self.app_data.clone();
        let devices = tauri::async_runtime::spawn_blocking(move || {
            install_engine(
                &app_data,
                flavor,
                || publish(&app, checking),
                || stop_server_using(&app, &catalog::engine_dir(&app_data, flavor)),
            )
        })
        .await
        .map_err(|e| SetupError::new(ErrorKind::Other, e.to_string()))??;
        Ok(Some(devices))
    }
}

/// Stops llama-server when it runs from `dir` (an engine being replaced).
fn stop_server_using(app: &AppHandle, dir: &Path) {
    let current = PathBuf::from(settings::load(app).llama_server_path);
    if current.starts_with(dir) {
        if let Some(server) = app.try_state::<Arc<LlamaServer>>() {
            server.stop();
        }
    }
}

/// Unpacks the verified zips of `flavor` into a temporary folder next to the
/// engines, checks the tree and the engine, then moves it into place.
fn install_engine(
    app_data: &Path,
    flavor: EngineFlavor,
    on_check: impl FnOnce(),
    stop_old: impl FnOnce(),
) -> Result<Vec<String>, SetupError> {
    let engines = catalog::engines_dir(app_data);
    let io = |e: std::io::Error| SetupError::new(ErrorKind::Other, e.to_string());
    std::fs::create_dir_all(&engines).map_err(io)?;
    let tmp = engines.join(format!(".tmp-{}", install::random_suffix()));
    let result = (|| {
        for id in flavor.items() {
            let item = catalog::item(id).expect("catalog engine item");
            install::unpack(&catalog::zip_partial(app_data, item), &tmp)?;
        }
        install::check_tree(&tmp).map_err(|e| SetupError::new(ErrorKind::Tar, e))?;
        let dir = install::find_server_dir(&tmp).map_err(|e| SetupError::new(ErrorKind::Tar, e))?;
        on_check();
        let devices = install::engine_check(&dir.join(SERVER_EXE))?;
        info!("Translate setup: {flavor:?} engine sees {devices:?}");
        if flavor == EngineFlavor::Cuda && !devices.iter().any(|d| d.starts_with("CUDA")) {
            return Err(SetupError::new(
                ErrorKind::CudaFailed,
                format!("no CUDA device in --list-devices ({devices:?})"),
            ));
        }
        let target = catalog::engine_dir(app_data, flavor);
        if target.exists() {
            stop_old();
            let old = engines.join(format!(".old-{}", install::random_suffix()));
            std::fs::rename(&target, &old).map_err(io)?;
            std::fs::rename(&dir, &target).map_err(io)?;
            let _ = std::fs::remove_dir_all(&old);
        } else {
            std::fs::rename(&dir, &target).map_err(io)?;
        }
        Ok(devices)
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    // A CUDA that does not start is the driver's doing: keeping the 550 MB of
    // zips for a Retry would only show a "paused" download on the card.
    if result.is_ok() || matches!(&result, Err(e) if e.kind == ErrorKind::CudaFailed) {
        for id in flavor.items() {
            let _ = std::fs::remove_file(catalog::zip_partial(
                app_data,
                catalog::item(id).expect("catalog engine item"),
            ));
        }
    }
    result
}

/// Moves a verified model from `blobs\<sha>.partial` into the snapshot of its
/// commit and writes `refs\main`, as hf-hub lays the cache out.
fn finalize_model(hub: &Path, item: &Item) -> Result<PathBuf, SetupError> {
    let io = |e: std::io::Error| SetupError::new(ErrorKind::Other, e.to_string());
    let partial = catalog::model_partial(hub, item).expect("a model");
    let file = catalog::model_file(hub, item).expect("a model");
    let reference = catalog::model_ref(hub, item).expect("a model");
    for dir in [file.parent(), reference.parent()].into_iter().flatten() {
        std::fs::create_dir_all(dir).map_err(io)?;
    }
    std::fs::rename(&partial, &file).map_err(io)?;
    std::fs::write(&reference, item.repo.expect("a model").commit).map_err(io)?;
    Ok(file)
}

fn roots(app: &AppHandle) -> Result<(PathBuf, PathBuf), String> {
    settings::load(app);
    settings::setup_roots().ok_or_else(|| "The app data folder is unavailable".to_string())
}

/// The whole run; `Ok(None)` when cancelled. Errors name the item.
async fn run(
    ctx: &Ctx,
    mut queue: Vec<&'static Item>,
    pick: &TierPick,
    ready_before: bool,
) -> Result<Option<SetupDone>, (String, SetupError)> {
    let mut result = RunResult::default();
    let mut done = SetupDone {
        installed: Vec::new(),
        gpu_layers: 0,
        engine_sees_gpu: None,
        skipped: Vec::new(),
        notices: Vec::new(),
    };

    if let Some(flavor) = queue.iter().find_map(|i| flavor_of(i)) {
        let first = flavor.items()[0];
        let devices = match ctx.engine(flavor).await {
            Ok(Some(devices)) => devices,
            Ok(None) => return Ok(None),
            Err(e)
                if e.kind == ErrorKind::CudaFailed
                    && !Path::new(&settings::load(&ctx.app).llama_server_path).is_file() =>
            {
                // "Only CUDA" from the first card: Vulkan instead (F.10).
                warn!(
                    "Translate setup: CUDA failed ({}), installing Vulkan",
                    e.message
                );
                done.notices.push(Notice::CudaFailed);
                let vulkan = catalog::item(catalog::ENGINE_VULKAN).expect("catalog");
                ctx.add_total(vulkan.size);
                match ctx
                    .engine(EngineFlavor::Vulkan)
                    .await
                    .map_err(tag(vulkan.id))?
                {
                    Some(devices) => {
                        result.engine_exe =
                            Some(catalog::engine_exe(&ctx.app_data, EngineFlavor::Vulkan));
                        done.installed.push(vulkan.id.into());
                        devices
                    }
                    None => return Ok(None),
                }
            }
            Err(e) => return Err((first.to_string(), e)),
        };
        if result.engine_exe.is_none() {
            result.engine_exe = Some(catalog::engine_exe(&ctx.app_data, flavor));
            done.installed
                .extend(flavor.items().iter().map(|s| s.to_string()));
        }
        let sees = !devices.is_empty();
        result.engine_sees_gpu = Some(sees);
        if pick.tier.uses_gpu() && !sees {
            if let Some(pos) = queue.iter().position(|i| i.id == catalog::MODEL_QUALITY) {
                let quality = queue.remove(pos);
                done.skipped.push(quality.id.into());
                let rest =
                    quality
                        .size
                        .saturating_sub(partial_len(&ctx.app_data, &ctx.hub, quality));
                ctx.meter.lock().unwrap_or_else(|e| e.into_inner()).total -= rest;
            }
            done.notices.push(Notice::NoGpu);
        }
    }

    for item in queue.iter().filter(|i| i.kind == ItemKind::Model) {
        let file = if model_cached(&ctx.hub, item) {
            catalog::model_file(&ctx.hub, item).expect("a model")
        } else {
            if !ctx.fetch(item).await.map_err(tag(item.id))? {
                return Ok(None);
            }
            ctx.finish_item();
            finalize_model(&ctx.hub, item).map_err(tag(item.id))?
        };
        done.installed.push(item.id.into());
        match item.id {
            catalog::MODEL_FAST => result.fast = Some(file),
            _ => result.quality = Some(file),
        }
    }

    let mut s = settings::load(&ctx.app);
    apply_result(&mut s, &result, pick.tier, ready_before);
    settings::save(&ctx.app, &s)
        .map_err(|e| (String::new(), SetupError::new(ErrorKind::Other, e)))?;
    let app = ctx.app.clone();
    let _ = tauri::async_runtime::spawn_blocking(move || super::stop_server(&app)).await;
    done.gpu_layers = s.gpu_layers;
    done.engine_sees_gpu = result.engine_sees_gpu;
    Ok(Some(done))
}

#[tauri::command]
#[specta::specta]
pub async fn translate_setup_plan(app: AppHandle) -> Result<SetupPlan, String> {
    let (app_data, hub) = roots(&app)?;
    let s = settings::load(&app);
    let pick = tier::platform_pick();
    let items = plan_items(&app_data, &hub, &s, &pick);
    Ok(SetupPlan {
        tier: pick.tier,
        adapter_name: pick.adapter.as_ref().map(|a| a.name.clone()),
        dedicated_mib: pick.adapter.as_ref().map_or(0, |a| a.dedicated_mib),
        nvidia: tier::adapters()
            .iter()
            .any(|a| a.is_nvidia() && !a.software),
        unified_memory: tier::unified_memory(),
        items,
        models_volume: install::volume_of(&hub),
        engine_volume: install::volume_of(&app_data),
        free_models_bytes: install::free_bytes(&hub),
        free_engine_bytes: install::free_bytes(&app_data),
        ready: s.setup_ready(),
        has_fast: s.fast_available(),
        has_quality: s.quality_available(),
        engine: engine_info(&app_data, Path::new(&s.llama_server_path)),
        models_deletable_bytes: deletable_model_dirs(&hub, &s)
            .iter()
            .map(|dir| dir_size(dir))
            .sum(),
        state: app.state::<SetupState>().view(),
    })
}

#[tauri::command]
#[specta::specta]
pub fn translate_setup_status(app: AppHandle) -> SetupStateView {
    app.state::<SetupState>().view()
}

/// The CUDA engine only for NVIDIA video cards (F.10).
fn check_cuda_allowed(queue: &[&Item], nvidia: bool) -> Result<(), String> {
    if !nvidia && queue.iter().any(|i| i.id == catalog::ENGINE_CUDA) {
        return Err("CUDA works only with NVIDIA video cards".into());
    }
    Ok(())
}

/// Why an engine switch did not happen; the previous engine stays.
#[derive(Debug, PartialEq)]
enum SwitchRefusal {
    NotDownloaded,
    NoNvidia,
    Failed(SetupError),
}

/// The settings after a switch to an engine that is on disk already (F.10):
/// the engine must start, CUDA must show a CUDA device. Only the engine path
/// changes (and `gpu_layers` when the engine does not see the card); slots,
/// models and the rest stay.
fn switched_settings(
    s: &SelTranslateSettings,
    app_data: &Path,
    flavor: EngineFlavor,
    nvidia: bool,
    check: impl FnOnce(&Path) -> Result<Vec<String>, SetupError>,
) -> Result<SelTranslateSettings, SwitchRefusal> {
    let exe = catalog::engine_exe(app_data, flavor);
    if !exe.is_file() {
        return Err(SwitchRefusal::NotDownloaded);
    }
    if flavor == EngineFlavor::Cuda && !nvidia {
        return Err(SwitchRefusal::NoNvidia);
    }
    let devices = check(&exe).map_err(SwitchRefusal::Failed)?;
    if flavor == EngineFlavor::Cuda && !devices.iter().any(|d| d.starts_with("CUDA")) {
        return Err(SwitchRefusal::Failed(SetupError::new(
            ErrorKind::CudaFailed,
            format!("no CUDA device in --list-devices ({devices:?})"),
        )));
    }
    let mut next = s.clone();
    let result = RunResult {
        engine_exe: Some(exe),
        engine_sees_gpu: Some(!devices.is_empty()),
        ..Default::default()
    };
    apply_result(&mut next, &result, Tier::Both, true);
    Ok(next)
}

/// Switch to an engine the setup downloaded before (no download). `Ok(None)`:
/// switched; `Ok(Some(kind))`: the engine did not start, the previous one
/// stays; `Err`: could not even try (a translation is running, not on disk).
#[tauri::command]
#[specta::specta]
pub async fn translate_setup_use_engine(
    app: AppHandle,
    flavor: EngineFlavor,
) -> Result<Option<ErrorKind>, String> {
    if app.state::<SetupState>().is_running() {
        return Err("Wait for the download to finish".into());
    }
    // No translation starts while the engine is checked and replaced.
    let (Some(selection), Some(file)) = (super::action::hold_busy(), super::file::hold_running())
    else {
        return Err("A translation is running".into());
    };
    let (app_data, _) = roots(&app)?;
    let nvidia = tier::adapters().iter().any(|a| a.is_nvidia());
    let s = settings::load(&app);
    tauri::async_runtime::spawn_blocking(move || {
        let _held = (selection, file);
        match switched_settings(&s, &app_data, flavor, nvidia, install::engine_check) {
            Ok(next) => {
                settings::save(&app, &next)?;
                super::stop_server(&app);
                info!("Translate setup: switched to the {flavor:?} engine");
                Ok(None)
            }
            Err(SwitchRefusal::Failed(e)) => {
                warn!(
                    "Translate setup: {flavor:?} engine refused: {:?}: {}",
                    e.kind, e.message
                );
                Ok(Some(e.kind))
            }
            Err(SwitchRefusal::NotDownloaded) => Err("The engine is not downloaded".into()),
            Err(SwitchRefusal::NoNvidia) => Err("CUDA works only with NVIDIA video cards".into()),
        }
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Whether a translation (hotkey or file) is running: the engine cannot be
/// switched or the models deleted meanwhile. The card polls it.
#[tauri::command]
#[specta::specta]
pub fn translate_setup_busy() -> bool {
    super::action::is_busy() || super::file::is_running()
}

/// Deletes the downloaded engines that `llama_server_path` is not in.
#[tauri::command]
#[specta::specta]
pub fn translate_setup_delete_unused_engine(app: AppHandle) -> Result<(), String> {
    if app.state::<SetupState>().is_running() {
        return Err("Wait for the download to finish".into());
    }
    let (app_data, _) = roots(&app)?;
    let s = settings::load(&app);
    for flavor in engine_info(&app_data, Path::new(&s.llama_server_path)).unused {
        let dir = catalog::engine_dir(&app_data, flavor);
        info!(
            "Translate setup: deleting the unused engine {}",
            dir.display()
        );
        std::fs::remove_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    Ok(())
}

/// Cache folders of our models that the settings point into (the files the
/// translation uses, or their half-downloaded parts): what "Delete translation
/// models" removes. A model at a path of the user's own stays.
fn deletable_model_dirs(hub: &Path, s: &SelTranslateSettings) -> Vec<PathBuf> {
    [catalog::MODEL_FAST, catalog::MODEL_QUALITY]
        .into_iter()
        .filter_map(|id| {
            let dir = hub.join(catalog::item(id)?.repo?.cache_dir_name());
            let used = [&s.fast_model_path, &s.model_path]
                .iter()
                .any(|p| !p.is_empty() && Path::new(p.as_str()).starts_with(&dir));
            (used && dir.is_dir()).then_some(dir)
        })
        .collect()
}

fn dir_size(dir: &Path) -> u64 {
    std::fs::read_dir(dir).map_or(0, |entries| {
        entries
            .flatten()
            .filter_map(|e| e.path().symlink_metadata().ok().map(|m| (e.path(), m)))
            .map(|(path, m)| if m.is_dir() { dir_size(&path) } else { m.len() })
            .sum()
    })
}

/// Deletes the translation models from the Hugging Face cache (the folders the
/// settings use). Not while a translation or a download runs.
#[tauri::command]
#[specta::specta]
pub async fn translate_setup_delete_models(app: AppHandle) -> Result<(), String> {
    if app.state::<SetupState>().is_running() {
        return Err("Wait for the download to finish".into());
    }
    let (Some(selection), Some(file)) = (super::action::hold_busy(), super::file::hold_running())
    else {
        return Err("A translation is running".into());
    };
    let (_, hub) = roots(&app)?;
    let dirs = deletable_model_dirs(&hub, &settings::load(&app));
    tauri::async_runtime::spawn_blocking(move || {
        let _held = (selection, file);
        // The server holds the model file open.
        super::stop_server(&app);
        for dir in dirs {
            info!("Translate setup: deleting {}", dir.display());
            std::fs::remove_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Starts the download of `items` (ids of the catalog; `engine-cuda` brings
/// its runtime). Returns at once; the card follows the events.
#[tauri::command]
#[specta::specta]
pub fn translate_setup_start(
    app: AppHandle,
    items: Vec<String>,
    accept_gemma_terms: bool,
) -> Result<(), String> {
    let (app_data, hub) = roots(&app)?;
    let s = settings::load(&app);
    let pick = tier::platform_pick();
    let queue: Vec<&'static Item> = expand_queue(&items)?
        .into_iter()
        .filter(|item| match flavor_of(item) {
            Some(flavor) => !catalog::engine_exe(&app_data, flavor).is_file(),
            None => true,
        })
        .collect();
    if queue.is_empty() {
        return Err("Nothing to download".into());
    }
    if queue.iter().any(|i| i.kind == ItemKind::Model) && !accept_gemma_terms {
        return Err("Accept the Gemma Terms of Use first".into());
    }
    check_cuda_allowed(&queue, tier::adapters().iter().any(|a| a.is_nvidia()))?;
    let to_fetch: Vec<&Item> = queue
        .iter()
        .copied()
        .filter(|i| i.kind == ItemKind::Engine || !model_cached(&hub, i))
        .collect();
    check_space(
        needed_bytes(&app_data, &hub, &to_fetch),
        (&install::volume_of(&hub), &install::volume_of(&app_data)),
        (install::free_bytes(&hub), install::free_bytes(&app_data)),
    )
    .map_err(|e| e.message)?;

    let state = app.state::<SetupState>();
    if state.running.swap(true, Ordering::SeqCst) {
        return Err("The setup is already running".into());
    }
    let token = CancellationToken::new();
    *state.token.lock().unwrap_or_else(|e| e.into_inner()) = Some(token.clone());
    let total = to_fetch.iter().map(|i| i.size).sum();
    info!(
        "Translate setup: start {:?} ({} to download, tier {:?}, llama.cpp {})",
        queue.iter().map(|i| i.id).collect::<Vec<_>>(),
        gb(total),
        pick.tier,
        catalog::ENGINE_BUILD
    );
    let ctx = Ctx {
        app: app.clone(),
        token,
        app_data,
        hub,
        meter: Mutex::new(Meter {
            done_before: 0,
            total,
            item: queue[0].id,
            item_total: queue[0].size,
            item_done: 0,
            samples: VecDeque::new(),
        }),
    };
    ctx.emit(Phase::Downloading, None);
    let ready_before = s.setup_ready();
    tauri::async_runtime::spawn(async move {
        let outcome = run(&ctx, queue, &pick, ready_before).await;
        let state = ctx.app.state::<SetupState>();
        match outcome {
            Ok(Some(done)) => {
                info!("Translate setup: done {done:?}");
                state.set_view(SetupStateView::Idle);
                let _ = ctx.app.emit("translate-setup-done", done);
            }
            Ok(None) => {
                info!("Translate setup: cancelled");
                state.set_view(SetupStateView::Idle);
                let _ = ctx.app.emit("translate-setup-cancelled", ());
            }
            Err((item, e)) => {
                warn!("Translate setup: {item}: {:?}: {}", e.kind, e.message);
                state.set_view(SetupStateView::Error {
                    item: item.clone(),
                    kind: e.kind,
                    message: e.message.clone(),
                });
                let _ = ctx.app.emit(
                    "translate-setup-error",
                    SetupErrorEvent {
                        item,
                        kind: e.kind,
                        message: e.message,
                    },
                );
            }
        }
        *state.token.lock().unwrap_or_else(|e| e.into_inner()) = None;
        state.running.store(false, Ordering::SeqCst);
    });
    Ok(())
}

/// Clears the error shown on the card (a refused CUDA, a failed add).
#[tauri::command]
#[specta::specta]
pub fn translate_setup_dismiss_error(app: AppHandle) {
    let state = app.state::<SetupState>();
    if !state.is_running() {
        state.set_view(SetupStateView::Idle);
    }
}

/// Cancel: the `.partial` files stay (Resume continues them).
#[tauri::command]
#[specta::specta]
pub fn translate_setup_cancel(app: AppHandle) {
    if let Some(token) = app
        .state::<SetupState>()
        .token
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
    {
        token.cancel();
    }
}

/// Discard: deletes the half-downloaded files and leftover temporary folders.
#[tauri::command]
#[specta::specta]
pub fn translate_setup_discard(app: AppHandle) -> Result<(), String> {
    let state = app.state::<SetupState>();
    if state.is_running() {
        return Err("Cancel the download first".into());
    }
    let (app_data, hub) = roots(&app)?;
    discard_leftovers(&app_data, &hub);
    state.set_view(SetupStateView::Idle);
    Ok(())
}

fn discard_leftovers(app_data: &Path, hub: &Path) {
    for item in &catalog::CATALOG {
        let _ = std::fs::remove_file(catalog::partial_path(app_data, hub, item));
    }
    if let Ok(entries) = std::fs::read_dir(catalog::engines_dir(app_data)) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with(".tmp-") || name.starts_with(".old-") {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }
}

/// The popup's "Open settings": the main window on the Translation section.
#[tauri::command]
#[specta::specta]
pub fn translate_open_settings(app: AppHandle) {
    crate::show_main_window(&app);
    let _ = app.emit("seltranslate-open-section", "translate");
}

#[cfg(test)]
mod tests {
    use super::*;
    use settings::{SelectionSlot, Slot};

    fn touch(path: &Path, len: usize) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, vec![0u8; len]).unwrap();
    }

    #[cfg(windows)]
    fn both() -> TierPick {
        tier::pick_tier(&[tier::Adapter {
            name: "GPU".into(),
            vendor_id: tier::VENDOR_NVIDIA,
            dedicated_mib: 16 * 1024,
            software: false,
        }])
    }

    fn blank() -> SelTranslateSettings {
        SelTranslateSettings {
            llama_server_path: String::new(),
            model_path: String::new(),
            fast_model_path: String::new(),
            ..Default::default()
        }
    }

    #[test]
    #[cfg(windows)]
    fn the_plan_reads_the_disk() {
        let dir = tempfile::tempdir().unwrap();
        let (app_data, hub) = (dir.path().join("app"), dir.path().join("hub"));
        let by_id =
            |items: &[PlanItem], id: &str| items.iter().find(|i| i.id == id).unwrap().clone();

        let items = plan_items(&app_data, &hub, &blank(), &both());
        assert!(items.iter().all(|i| !i.installed && i.partial_bytes == 0));
        assert!(by_id(&items, catalog::ENGINE_VULKAN).recommended);
        assert!(!by_id(&items, catalog::ENGINE_CUDA).recommended);
        assert!(by_id(&items, catalog::MODEL_QUALITY).recommended, "16 GB");

        let quality = catalog::item(catalog::MODEL_QUALITY).unwrap();
        touch(&catalog::model_partial(&hub, quality).unwrap(), 1000);
        let vulkan = catalog::item(catalog::ENGINE_VULKAN).unwrap();
        touch(&catalog::engine_exe(&app_data, EngineFlavor::Vulkan), 1);
        let fast = catalog::item(catalog::MODEL_FAST).unwrap();
        touch(&catalog::model_file(&hub, fast).unwrap(), 1);
        let items = plan_items(&app_data, &hub, &blank(), &both());
        assert_eq!(by_id(&items, catalog::MODEL_QUALITY).partial_bytes, 1000);
        assert!(!by_id(&items, catalog::MODEL_QUALITY).installed);
        assert!(by_id(&items, catalog::ENGINE_VULKAN).installed);
        assert!(!by_id(&items, catalog::ENGINE_CUDA).installed);
        assert!(by_id(&items, catalog::MODEL_FAST).installed, "in the cache");

        // Space: the rest of the partial, the engine unpacked, Fast is there.
        let (models, engine) = needed_bytes(&app_data, &hub, &[vulkan, quality]);
        assert_eq!(models, quality.size - 1000);
        assert_eq!(
            engine,
            vulkan.size + catalog::unpacked_engine_bytes(EngineFlavor::Vulkan)
        );

        // A user's own files count as installed.
        let own = dir.path().join("own.gguf");
        touch(&own, 1);
        let s = SelTranslateSettings {
            model_path: own.display().to_string(),
            ..blank()
        };
        assert!(
            by_id(
                &plan_items(&app_data, &hub, &s, &both()),
                catalog::MODEL_QUALITY
            )
            .installed
        );
    }

    #[test]
    fn space_is_checked_per_volume_or_summed_on_one() {
        let gib = 1u64 << 30;
        assert!(check_space((5 * gib, gib), ("C:", "C:"), (Some(7 * gib), Some(7 * gib))).is_ok());
        let e =
            check_space((5 * gib, gib), ("C:", "C:"), (Some(6 * gib), Some(6 * gib))).unwrap_err();
        assert_eq!(e.kind, ErrorKind::DiskFull);
        assert!(e.message.starts_with("C:"), "{}", e.message);
        assert!(check_space((5 * gib, gib), ("D:", "C:"), (Some(6 * gib), Some(2 * gib))).is_ok());
        let e = check_space((5 * gib, gib), ("D:", "C:"), (Some(9 * gib), Some(gib))).unwrap_err();
        assert!(e.message.starts_with("C:"));
        assert!(
            check_space((5 * gib, 0), ("D:", "C:"), (None, None)).is_ok(),
            "unknown free space"
        );
        assert!(check_space((0, 0), ("C:", "C:"), (Some(0), Some(0))).is_ok());
    }

    #[test]
    #[cfg(windows)]
    fn the_queue_is_engine_fast_quality() {
        let ids = |q: Vec<&Item>| q.iter().map(|i| i.id).collect::<Vec<_>>();
        let q = expand_queue(&[
            "model-quality".into(),
            "engine-cuda".into(),
            "model-fast".into(),
        ])
        .unwrap();
        assert_eq!(
            ids(q),
            [
                "engine-cuda",
                "engine-cudart",
                "model-fast",
                "model-quality"
            ]
        );
        assert_eq!(
            ids(expand_queue(&["engine-vulkan".into()]).unwrap()),
            ["engine-vulkan"]
        );
        assert!(expand_queue(&["engine-vulkan".into(), "engine-cuda".into()]).is_err());
        assert!(expand_queue(&["engine-cpu".into()]).is_err());
    }

    #[test]
    #[cfg(windows)]
    fn the_engine_is_told_by_its_dll() {
        let dir = tempfile::tempdir().unwrap();
        let app_data = dir.path().join("app");
        let vulkan = catalog::engine_exe(&app_data, EngineFlavor::Vulkan);
        touch(&vulkan, 1);
        touch(&vulkan.with_file_name("ggml-vulkan.dll"), 1);
        let own = dir.path().join("tools").join("llama").join(SERVER_EXE);
        touch(&own, 1);
        touch(&own.with_file_name("ggml-cuda.dll"), 1);
        let cpu = dir.path().join("cpu").join(SERVER_EXE);
        touch(&cpu, 1);

        let info = engine_info(&app_data, &vulkan);
        assert_eq!((info.current, info.own_files), (EngineKind::Vulkan, false));
        assert!(info.vulkan_downloaded && !info.cuda_downloaded);
        assert!(info.unused.is_empty(), "the running engine is not unused");
        // The user's CUDA build in C:\SelTranslate-tools: "CUDA (your own files)".
        let info = engine_info(&app_data, &own);
        assert_eq!((info.current, info.own_files), (EngineKind::Cuda, true));
        assert_eq!(info.unused, [EngineFlavor::Vulkan], "ours, not in use");
        assert_eq!(engine_kind(&cpu), EngineKind::Cpu);
        assert_eq!(
            engine_kind(&dir.path().join("none.exe")),
            EngineKind::Missing
        );
    }

    fn with_engines(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
        let app_data = dir.join("app");
        let vulkan = catalog::engine_exe(&app_data, EngineFlavor::Vulkan);
        let cuda = catalog::engine_exe(&app_data, EngineFlavor::Cuda);
        touch(&vulkan, 1);
        touch(&cuda, 1);
        (app_data, vulkan, cuda)
    }

    fn running_on(exe: &Path) -> SelTranslateSettings {
        SelTranslateSettings {
            llama_server_path: exe.display().to_string(),
            model_path: "q.gguf".into(),
            fast_model_path: "f.gguf".into(),
            selection_slot: settings::SelectionSlot::Quality,
            file_slot: settings::Slot::Fast,
            gpu_layers: 20,
            ctx_size: 8192,
            ..Default::default()
        }
    }

    #[test]
    fn a_switch_to_a_downloaded_engine_changes_only_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let (app_data, vulkan, cuda) = with_engines(dir.path());
        let s = running_on(&vulkan);
        let next = switched_settings(&s, &app_data, EngineFlavor::Cuda, true, |exe| {
            assert_eq!(exe, cuda, "the downloaded engine is checked, not fetched");
            Ok(vec!["CUDA0".into()])
        })
        .unwrap();
        assert_eq!(
            next,
            SelTranslateSettings {
                llama_server_path: cuda.display().to_string(),
                ..s.clone()
            }
        );
        // Back to Vulkan: the video card is seen, the layers stay.
        let back = switched_settings(&next, &app_data, EngineFlavor::Vulkan, true, |_| {
            Ok(vec!["Vulkan0".into()])
        })
        .unwrap();
        assert_eq!(back, s);
        // An engine that does not see the card means the processor.
        let blind = switched_settings(&s, &app_data, EngineFlavor::Cuda, true, |_| {
            Ok(vec!["CUDA0".into()])
        })
        .and_then(|cuda_s| {
            switched_settings(&cuda_s, &app_data, EngineFlavor::Vulkan, true, |_| {
                Ok(vec![])
            })
        })
        .unwrap();
        assert_eq!(blind.gpu_layers, 0);
        assert_eq!(blind.llama_server_path, vulkan.display().to_string());
    }

    #[test]
    fn a_failed_switch_keeps_the_previous_engine() {
        let dir = tempfile::tempdir().unwrap();
        let (app_data, vulkan, _) = with_engines(dir.path());
        let s = running_on(&vulkan);
        // CUDA starts but shows no CUDA device (a driver that is too old).
        let refused = switched_settings(&s, &app_data, EngineFlavor::Cuda, true, |_| {
            Ok(vec!["Vulkan0".into()])
        });
        assert!(
            matches!(&refused, Err(SwitchRefusal::Failed(e)) if e.kind == ErrorKind::CudaFailed),
            "{refused:?}"
        );
        let broken = switched_settings(&s, &app_data, EngineFlavor::Cuda, true, |_| {
            Err(SetupError::new(ErrorKind::EngineCheck, "exit 1"))
        });
        assert!(
            matches!(&broken, Err(SwitchRefusal::Failed(e)) if e.kind == ErrorKind::EngineCheck)
        );
        // Not NVIDIA: not even started.
        let no_card = switched_settings(&s, &app_data, EngineFlavor::Cuda, false, |_| {
            panic!("the engine must not run")
        });
        assert_eq!(no_card, Err(SwitchRefusal::NoNvidia));
        // Not on disk: no download here.
        let empty = tempfile::tempdir().unwrap();
        let missing = switched_settings(&s, empty.path(), EngineFlavor::Vulkan, true, |_| {
            panic!("the engine must not run")
        });
        assert_eq!(missing, Err(SwitchRefusal::NotDownloaded));
    }

    #[test]
    #[cfg(windows)]
    fn cuda_is_for_nvidia_only() {
        let cuda = expand_queue(&["engine-cuda".into()]).unwrap();
        assert!(check_cuda_allowed(&cuda, true).is_ok());
        assert!(check_cuda_allowed(&cuda, false).is_err());
        let vulkan = expand_queue(&["engine-vulkan".into()]).unwrap();
        assert!(check_cuda_allowed(&vulkan, false).is_ok());
    }

    #[test]
    fn only_the_models_the_settings_use_can_be_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let hub = dir.path().join("hub");
        let fast = catalog::item(catalog::MODEL_FAST).unwrap();
        let quality = catalog::item(catalog::MODEL_QUALITY).unwrap();
        let fast_file = catalog::model_file(&hub, fast).unwrap();
        let quality_file = catalog::model_file(&hub, quality).unwrap();
        touch(&fast_file, 100);
        touch(&quality_file, 1000);
        touch(&catalog::model_partial(&hub, quality).unwrap(), 10);
        let own = dir.path().join("own.gguf");
        touch(&own, 5);
        let by = |fast: &Path, quality: &Path| SelTranslateSettings {
            fast_model_path: fast.display().to_string(),
            model_path: quality.display().to_string(),
            ..blank()
        };

        let dirs = deletable_model_dirs(&hub, &by(&fast_file, &quality_file));
        assert_eq!(dirs.len(), 2);
        assert_eq!(dirs.iter().map(|d| dir_size(d)).sum::<u64>(), 1110);
        // The user's own files elsewhere: nothing of ours is in use, nothing goes.
        assert!(deletable_model_dirs(&hub, &by(&own, &own)).is_empty());
        // One model in the cache, the other the user's own.
        let one = deletable_model_dirs(&hub, &by(&own, &quality_file));
        assert_eq!(one, [hub.join(quality.repo.unwrap().cache_dir_name())]);
        // No cache folder yet: nothing.
        assert!(
            deletable_model_dirs(&dir.path().join("none"), &by(&fast_file, &quality_file))
                .is_empty()
        );
    }

    fn files(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
        let exe = dir.join("e").join(SERVER_EXE);
        let fast = dir.join("f.gguf");
        let quality = dir.join("q.gguf");
        for p in [&exe, &fast, &quality] {
            touch(p, 1);
        }
        (exe, fast, quality)
    }

    #[test]
    fn a_full_setup_writes_paths_slots_and_layers_only() {
        let dir = tempfile::tempdir().unwrap();
        let (exe, fast, quality) = files(dir.path());
        let mut s = SelTranslateSettings {
            output_mode: settings::OutputMode::Replace,
            double_ctrl_enabled: false,
            idle_unload_minutes: 0,
            ctx_size: 2048,
            ..blank()
        };
        let before = s.clone();
        let r = RunResult {
            engine_exe: Some(exe.clone()),
            engine_sees_gpu: Some(true),
            fast: Some(fast.clone()),
            quality: Some(quality.clone()),
        };
        apply_result(&mut s, &r, Tier::Both, false);
        assert_eq!(s.llama_server_path, exe.display().to_string());
        assert_eq!(
            (s.selection_slot, s.file_slot, s.gpu_layers),
            (SelectionSlot::Auto, Slot::Quality, 99)
        );
        assert!(s.setup_ready());
        let untouched = SelTranslateSettings {
            llama_server_path: s.llama_server_path.clone(),
            model_path: s.model_path.clone(),
            fast_model_path: s.fast_model_path.clone(),
            selection_slot: s.selection_slot,
            file_slot: s.file_slot,
            gpu_layers: s.gpu_layers,
            ..before
        };
        assert_eq!(s, untouched, "fields outside the setup stay");
    }

    #[test]
    fn fast_only_and_no_gpu_setups() {
        let dir = tempfile::tempdir().unwrap();
        let (exe, fast, _) = files(dir.path());
        let mut s = SelTranslateSettings {
            model_path: dir.path().join("missing.gguf").display().to_string(),
            ..blank()
        };
        let r = RunResult {
            engine_exe: Some(exe),
            engine_sees_gpu: Some(false),
            fast: Some(fast),
            quality: None,
        };
        apply_result(&mut s, &r, Tier::Both, false);
        assert_eq!(s.model_path, "", "Fast only: Quality switched off");
        assert_eq!(
            (s.selection_slot, s.file_slot, s.gpu_layers),
            (SelectionSlot::Fast, Slot::Fast, 0)
        );

        // Cpu tier: the processor even without an engine check.
        let mut s = blank();
        let (_, fast, _) = files(dir.path());
        apply_result(
            &mut s,
            &RunResult {
                fast: Some(fast),
                ..Default::default()
            },
            Tier::Cpu,
            false,
        );
        assert_eq!(s.gpu_layers, 0);
    }

    #[test]
    #[cfg(windows)]
    fn adding_quality_later_switches_to_auto_and_keeps_the_layers() {
        let dir = tempfile::tempdir().unwrap();
        let (exe, fast, quality) = files(dir.path());
        let mut s = SelTranslateSettings {
            llama_server_path: exe.display().to_string(),
            fast_model_path: fast.display().to_string(),
            selection_slot: SelectionSlot::Fast,
            file_slot: Slot::Fast,
            gpu_layers: 20,
            ..blank()
        };
        let r = RunResult {
            quality: Some(quality),
            ..Default::default()
        };
        apply_result(&mut s, &r, Tier::Partial, true);
        assert_eq!(
            (s.selection_slot, s.file_slot, s.gpu_layers),
            (SelectionSlot::Auto, Slot::Quality, 20)
        );
    }

    #[test]
    fn an_engine_switch_keeps_slots_and_layers() {
        let dir = tempfile::tempdir().unwrap();
        let (exe, fast, quality) = files(dir.path());
        let ready = SelTranslateSettings {
            llama_server_path: "old.exe".into(),
            fast_model_path: fast.display().to_string(),
            model_path: quality.display().to_string(),
            selection_slot: SelectionSlot::Quality,
            gpu_layers: 0,
            ..blank()
        };
        let mut s = ready.clone();
        let r = RunResult {
            engine_exe: Some(exe.clone()),
            engine_sees_gpu: Some(true),
            ..Default::default()
        };
        apply_result(&mut s, &r, Tier::Both, true);
        assert_eq!(
            s,
            SelTranslateSettings {
                llama_server_path: exe.display().to_string(),
                ..ready.clone()
            }
        );
        let mut s = SelTranslateSettings {
            gpu_layers: 99,
            ..ready
        };
        let r = RunResult {
            engine_sees_gpu: Some(false),
            ..r
        };
        apply_result(&mut s, &r, Tier::Both, true);
        assert_eq!(s.gpu_layers, 0, "the new engine does not see the card");
    }

    #[test]
    fn a_model_is_moved_into_its_snapshot_with_refs_main() {
        let dir = tempfile::tempdir().unwrap();
        let hub = dir.path().join("hub");
        let fast = catalog::item(catalog::MODEL_FAST).unwrap();
        touch(&catalog::model_partial(&hub, fast).unwrap(), 3);
        let file = finalize_model(&hub, fast).unwrap();
        assert_eq!(file, catalog::model_file(&hub, fast).unwrap());
        assert!(file.is_file() && !catalog::model_partial(&hub, fast).unwrap().exists());
        assert_eq!(
            std::fs::read_to_string(catalog::model_ref(&hub, fast).unwrap()).unwrap(),
            fast.repo.unwrap().commit
        );
        assert!(model_cached(&hub, fast));
    }

    #[test]
    fn hf_home_moves_the_model_paths() {
        let dir = tempfile::tempdir().unwrap();
        let old = std::env::var_os("HF_HOME");
        std::env::set_var("HF_HOME", dir.path());
        let hub = hf_hub::Cache::from_env().path().clone();
        match old {
            Some(v) => std::env::set_var("HF_HOME", v),
            None => std::env::remove_var("HF_HOME"),
        }
        assert_eq!(hub, dir.path().join("hub"));
        let quality = catalog::item(catalog::MODEL_QUALITY).unwrap();
        assert!(catalog::model_file(&hub, quality)
            .unwrap()
            .starts_with(dir.path()));
    }

    #[test]
    fn discard_removes_partials_and_temporary_folders() {
        let dir = tempfile::tempdir().unwrap();
        let (app_data, hub) = (dir.path().join("app"), dir.path().join("hub"));
        let items: Vec<PathBuf> = catalog::CATALOG
            .iter()
            .map(|i| catalog::partial_path(&app_data, &hub, i))
            .collect();
        for p in &items {
            touch(p, 1);
        }
        let engines = catalog::engines_dir(&app_data);
        touch(&engines.join(".tmp-1234").join("x"), 1);
        touch(&engines.join(".old-1234").join("x"), 1);
        let kept = catalog::engine_exe(&app_data, EngineFlavor::Vulkan);
        touch(&kept, 1);
        discard_leftovers(&app_data, &hub);
        assert!(items.iter().all(|p| !p.exists()));
        assert!(!engines.join(".tmp-1234").exists() && !engines.join(".old-1234").exists());
        assert!(kept.is_file(), "an installed engine stays");
    }

    /// F2 ignore test 1 (network: 30.7 MB from github.com; only with the
    /// user's yes): the real Vulkan zip through the setup's own path —
    /// redirects, transport with hash, tar, tree, `--version`, `--list-devices`.
    /// `SELTRANSLATE_F2_DIR` is the scratch folder (outside the project).
    #[tokio::test]
    #[ignore]
    async fn real_vulkan_engine_downloads_and_starts() {
        let root =
            PathBuf::from(std::env::var("SELTRANSLATE_F2_DIR").expect("SELTRANSLATE_F2_DIR"));
        let app_data = root.join("app");
        let item = catalog::item(catalog::ENGINE_VULKAN).unwrap();
        let partial = catalog::zip_partial(&app_data, item);
        std::fs::create_dir_all(partial.parent().unwrap()).unwrap();
        let url = install::resolve_redirects(item.url).await.unwrap();
        eprintln!(
            "final host: {:?}",
            reqwest::Url::parse(&url).unwrap().host_str()
        );
        let started = Instant::now();
        let outcome = ModelManager::download_http_resumable_with_events(
            item.id,
            &url,
            &partial,
            Some(item.size),
            Some(item.sha256),
            &CancellationToken::new(),
            &|_| {},
        )
        .await
        .unwrap();
        assert!(matches!(outcome, HttpDownloadOutcome::Completed));
        eprintln!(
            "downloaded and verified in {:.1} s",
            started.elapsed().as_secs_f64()
        );
        let devices = install_engine(&app_data, EngineFlavor::Vulkan, || {}, || {}).unwrap();
        let exe = catalog::engine_exe(&app_data, EngineFlavor::Vulkan);
        let raw = std::process::Command::new(&exe)
            .arg("--list-devices")
            .output()
            .unwrap();
        eprintln!(
            "--list-devices:\n{}{}",
            String::from_utf8_lossy(&raw.stdout),
            String::from_utf8_lossy(&raw.stderr)
        );
        let unpacked: u64 = std::fs::read_dir(exe.parent().unwrap())
            .unwrap()
            .flatten()
            .map(|e| e.metadata().map_or(0, |m| m.len()))
            .sum();
        eprintln!("devices {devices:?}, unpacked {} MB", unpacked >> 20);
        assert_eq!(engine_kind(&exe), EngineKind::Vulkan);
        assert!(!partial.exists(), "the zip is deleted after the install");
        assert!(
            !devices.is_empty(),
            "the RTX 5070 Ti is seen through Vulkan"
        );
    }

    /// F2d ignore test (no network): a switch with the real engines. Vulkan is
    /// the unpacked engine of ignore test 1 (`SELTRANSLATE_F2_DIRpp`); CUDA is
    /// the user's own build, which a junction in a scratch `app` folder shows as
    /// the downloaded CUDA engine (nothing is copied or changed there).
    #[test]
    #[ignore]
    #[cfg(windows)]
    fn real_engines_switch_back_and_forth() {
        let root =
            PathBuf::from(std::env::var("SELTRANSLATE_F2_DIR").expect("SELTRANSLATE_F2_DIR"));
        let vulkan_app = root.join("app");
        let app_data = root.join("switch-app");
        let cuda_dir = catalog::engine_dir(&app_data, EngineFlavor::Cuda);
        let vulkan_dir = catalog::engine_dir(&app_data, EngineFlavor::Vulkan);
        std::fs::create_dir_all(cuda_dir.parent().unwrap()).unwrap();
        let junction = |link: &Path, target: &Path| {
            if !link.exists() {
                let status = std::process::Command::new("cmd")
                    .args(["/c", "mklink", "/J"])
                    .arg(link)
                    .arg(target)
                    .status()
                    .unwrap();
                assert!(status.success());
            }
        };
        junction(&cuda_dir, Path::new(r"C:\SelTranslate-tools\llama"));
        junction(
            &vulkan_dir,
            &catalog::engine_dir(&vulkan_app, EngineFlavor::Vulkan),
        );

        let s = SelTranslateSettings {
            llama_server_path: String::new(),
            gpu_layers: 99,
            ..Default::default()
        };
        let on_cuda = switched_settings(
            &s,
            &app_data,
            EngineFlavor::Cuda,
            true,
            install::engine_check,
        )
        .unwrap();
        assert_eq!(
            engine_kind(Path::new(&on_cuda.llama_server_path)),
            EngineKind::Cuda
        );
        let on_vulkan = switched_settings(
            &on_cuda,
            &app_data,
            EngineFlavor::Vulkan,
            true,
            install::engine_check,
        )
        .unwrap();
        assert_eq!(
            engine_kind(Path::new(&on_vulkan.llama_server_path)),
            EngineKind::Vulkan
        );
        assert_eq!(on_vulkan.gpu_layers, 99, "both engines see the card");
        let info = engine_info(&app_data, Path::new(&on_vulkan.llama_server_path));
        assert_eq!(info.unused, [EngineFlavor::Cuda]);
        // CUDA is refused when the machine has no NVIDIA card; the engine
        // never runs for that.
        assert_eq!(
            switched_settings(
                &on_vulkan,
                &app_data,
                EngineFlavor::Cuda,
                false,
                |_| panic!()
            ),
            Err(SwitchRefusal::NoNvidia)
        );
        // Remove the junctions (not what they point to).
        std::fs::remove_dir(&cuda_dir).unwrap();
        std::fs::remove_dir(&vulkan_dir).unwrap();
    }

    /// F2 ignore test 2 (no network, ~1 min): the models in
    /// `C:\SelTranslate-tools\models` are byte for byte the catalog files.
    #[test]
    #[ignore]
    fn local_models_match_the_catalog() {
        use sha2::{Digest, Sha256};
        use std::io::Read;
        for id in [catalog::MODEL_FAST, catalog::MODEL_QUALITY] {
            let item = catalog::item(id).unwrap();
            let path = Path::new(r"C:\SelTranslate-tools\models").join(item.file_name);
            let mut file = std::fs::File::open(&path).unwrap();
            assert_eq!(file.metadata().unwrap().len(), item.size, "{id}");
            let mut hasher = Sha256::new();
            let mut buf = vec![0u8; 1 << 20];
            loop {
                let n = file.read(&mut buf).unwrap();
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
            }
            assert_eq!(format!("{:x}", hasher.finalize()), item.sha256, "{id}");
        }
    }

    #[test]
    #[cfg(windows)]
    fn a_failed_engine_check_leaves_nothing_behind() {
        // A zip with a fake llama-server.exe (a text file): unpacking works,
        // the engine check fails, the old engine and the zip stay.
        let dir = tempfile::tempdir().unwrap();
        let app_data = dir.path().join("app");
        let src = dir.path().join("src");
        touch(&src.join(SERVER_EXE), 3);
        let vulkan = catalog::item(catalog::ENGINE_VULKAN).unwrap();
        let zip = catalog::zip_partial(&app_data, vulkan);
        std::fs::create_dir_all(zip.parent().unwrap()).unwrap();
        let status = std::process::Command::new(
            PathBuf::from(std::env::var_os("SystemRoot").unwrap()).join(r"System32\tar.exe"),
        )
        .arg("-a")
        .arg("-cf")
        .arg(dir.path().join("e.zip"))
        .arg("-C")
        .arg(&src)
        .arg(SERVER_EXE)
        .status()
        .unwrap();
        assert!(status.success());
        std::fs::rename(dir.path().join("e.zip"), &zip).unwrap();
        let old = catalog::engine_exe(&app_data, EngineFlavor::Vulkan);
        touch(&old, 7);
        let err = install_engine(
            &app_data,
            EngineFlavor::Vulkan,
            || {},
            || panic!("not replaced"),
        )
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::EngineCheck, "{}", err.message);
        assert_eq!(
            std::fs::metadata(&old).unwrap().len(),
            7,
            "the old engine stays"
        );
        assert!(zip.is_file(), "the verified zip stays for Retry");
        let leftovers: Vec<_> = std::fs::read_dir(catalog::engines_dir(&app_data))
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert_eq!(
            leftovers,
            [std::ffi::OsString::from(EngineFlavor::Vulkan.dir_name())]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_mac_plan_offers_metal_and_the_models() {
        let dir = tempfile::tempdir().unwrap();
        let (app_data, hub) = (dir.path().join("app"), dir.path().join("hub"));
        let pick = tier::pick_tier_unified("Apple M3 Pro", 18 * 1024);
        let items = plan_items(&app_data, &hub, &blank(), &pick);
        let ids: Vec<&str> = items.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(ids, ["engine-metal", "model-fast", "model-quality"]);
        let rec = |id: &str| items.iter().find(|i| i.id == id).unwrap().recommended;
        assert!(rec("engine-metal") && rec("model-fast") && !rec("model-quality"));

        let exe = catalog::engine_exe(&app_data, EngineFlavor::Metal);
        touch(&exe, 1);
        let items = plan_items(&app_data, &hub, &blank(), &pick);
        assert!(
            items
                .iter()
                .find(|i| i.id == "engine-metal")
                .unwrap()
                .installed
        );
        assert_eq!(engine_kind(&exe), EngineKind::Metal);
        let info = engine_info(&app_data, &exe);
        assert!(info.unused.is_empty() && !info.own_files);

        let metal = catalog::item(catalog::ENGINE_METAL).unwrap();
        let (_, engine) = needed_bytes(&app_data, &hub, &[metal]);
        assert_eq!(
            engine,
            metal.size + catalog::unpacked_engine_bytes(EngineFlavor::Metal)
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_mac_queue_takes_no_windows_engine() {
        let ids = |q: Vec<&Item>| q.iter().map(|i| i.id).collect::<Vec<_>>();
        let q = expand_queue(&[
            "model-quality".into(),
            "engine-metal".into(),
            "model-fast".into(),
        ])
        .unwrap();
        assert_eq!(ids(q), ["engine-metal", "model-fast", "model-quality"]);
        assert!(expand_queue(&["engine-vulkan".into()]).is_err());
        assert!(expand_queue(&["engine-cuda".into()]).is_err());
    }
}
