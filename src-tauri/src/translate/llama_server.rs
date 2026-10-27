//! Lifecycle of the local `llama-server` process that runs the translation model.
//!
//! Started hidden on first use, kept in a Job Object with KILL_ON_JOB_CLOSE (on unix a watchdog process stands in for it) (so it
//! dies with Handy even on a crash), stopped after `idle_unload_ms` without use
//! and explicitly on exit. Nothing here depends on Tauri, so it is testable
//! without an `AppHandle`; the Tauri glue lives in `translate/mod.rs`.

use std::fs::File;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Once};
use std::time::{Duration, Instant};

use log::{info, warn};
use serde::Serialize;
use specta::Type;

const HEALTH_POLL: Duration = Duration::from_millis(250);
const START_TIMEOUT: Duration = Duration::from_secs(90);
const IDLE_TICK: Duration = Duration::from_secs(15);
const LOG_TAIL_LINES: usize = 20;
const PROBE_CONNECT_TIMEOUT: Duration = Duration::from_millis(500);
const PROBE_IO_TIMEOUT: Duration = Duration::from_secs(2);

/// What the server is started with. A change of any field restarts it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerConfig {
    pub exe_path: PathBuf,
    pub model_path: PathBuf,
    pub gpu_layers: u32,
    pub ctx_size: u32,
    pub log_path: PathBuf,
    pub chat_format: ChatFormat,
}

/// How the model's chat is formatted (I). TranslateGemma needs the plain
/// Gemma 3 template: its own jinja template expects a structured content.
/// Every other model gets its own template from the GGUF, with reasoning off.
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum ChatFormat {
    TranslateGemma,
    Chat,
}

/// The format of a model file, by its name (no manual switch, I).
pub fn chat_format_for(model_path: &Path) -> ChatFormat {
    let name = model_path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    if name.contains("translategemma") {
        ChatFormat::TranslateGemma
    } else {
        ChatFormat::Chat
    }
}

/// `gpu_layers` of "Video card (fast)": as many layers as fit, chosen by
/// llama.cpp at every start from the free video memory (`-ngl auto`, F.1).
pub const GPU_LAYERS_AUTO: u32 = 99;

fn ngl_arg(gpu_layers: u32) -> String {
    if gpu_layers == GPU_LAYERS_AUTO {
        "auto".into()
    } else {
        gpu_layers.to_string()
    }
}

/// Command line verified in B1 (DESIGN B.2). `--no-jinja` is required: without
/// it b11623 reads `--chat-template gemma` as a literal jinja template "gemma".
/// A chat model (I) gets `--jinja --reasoning off` in their place.
/// `--temp 0 -n 1024` set the sampling defaults, since `llm_client` sends neither.
/// `--api-key`: only Handy may use the server (`/health` stays public).
pub fn build_args(config: &ServerConfig, port: u16, api_key: &str) -> Vec<String> {
    let template: &[&str] = match config.chat_format {
        ChatFormat::TranslateGemma => &["--no-jinja", "--chat-template", "gemma"],
        ChatFormat::Chat => &["--jinja", "--reasoning", "off"],
    };
    let mut args: Vec<String> = vec![
        "-m".into(),
        config.model_path.to_string_lossy().into_owned(),
    ];
    args.extend(
        [
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "-ngl",
            &ngl_arg(config.gpu_layers),
            "-c",
            &config.ctx_size.to_string(),
            "-np",
            "1",
            "-fa",
            "auto",
            "--no-webui",
        ]
        .map(String::from),
    );
    args.extend(template.iter().map(|a| a.to_string()));
    args.extend(["--temp", "0", "-n", "1024", "--api-key", api_key].map(String::from));
    args
}

/// 128 random bits as hex, new for every start. Without a key any local
/// program — or a web page probing localhost ports — could use the server.
/// `RandomState` is keyed from the OS CSPRNG, so its SipHash output is
/// unpredictable; no extra crate needed.
pub fn random_api_key() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    (0..2u64)
        .map(|i| {
            let mut hasher = RandomState::new().build_hasher();
            hasher.write_u64(i);
            format!("{:016x}", hasher.finish())
        })
        .collect()
}

/// OpenAI-compatible base URL for `llm_client`.
pub fn base_url_for_port(port: u16) -> String {
    format!("http://127.0.0.1:{port}/v1")
}

/// `NO_PROXY` value that also exempts the local server, or `None` if it already does.
pub fn no_proxy_with_localhost(current: Option<&str>) -> Option<String> {
    let current = current.unwrap_or("").trim();
    let listed = current
        .split(',')
        .any(|entry| entry.trim() == "127.0.0.1" || entry.trim() == "*");
    if listed {
        return None;
    }
    Some(if current.is_empty() {
        "127.0.0.1,localhost".to_string()
    } else {
        format!("{current},127.0.0.1,localhost")
    })
}

/// reqwest (inside `llm_client`) honours the Windows system proxy, and its
/// reading of `ProxyOverride` does not match 127.0.0.1 (`<local>`, `127.*`), so
/// with a system proxy on, requests to the local server would go through it.
/// `NO_PROXY` takes precedence over the registry; set once per process.
fn bypass_proxy_for_localhost() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let current = std::env::var("NO_PROXY")
            .or_else(|_| std::env::var("no_proxy"))
            .ok();
        if let Some(value) = no_proxy_with_localhost(current.as_deref()) {
            info!("NO_PROXY set to '{value}' for the local llama-server");
            std::env::set_var("NO_PROXY", value);
        }
    });
}

/// A port that was free a moment ago (the listener is closed again, so another
/// process may grab it first — a failed start is retried once with a new port).
pub fn pick_free_port() -> io::Result<u16> {
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Health {
    Ready,
    /// 503: the model is still loading.
    Loading,
    Unexpected(u16),
}

/// Status code from the first line of an HTTP response.
pub fn parse_health_status(response: &[u8]) -> Option<Health> {
    let line = response.split(|&b| b == b'\n').next()?;
    let line = std::str::from_utf8(line).ok()?;
    let mut parts = line.split_whitespace();
    if !parts.next()?.starts_with("HTTP/") {
        return None;
    }
    let code: u16 = parts.next()?.parse().ok()?;
    Some(match code {
        200 => Health::Ready,
        503 => Health::Loading,
        other => Health::Unexpected(other),
    })
}

/// `GET /health` over a plain socket (no async runtime needed on the caller's thread).
pub fn probe_health(port: u16) -> io::Result<Health> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&addr, PROBE_CONNECT_TIMEOUT)?;
    stream.set_read_timeout(Some(PROBE_IO_TIMEOUT))?;
    stream.set_write_timeout(Some(PROBE_IO_TIMEOUT))?;
    // One write: a request split over segments may be answered before it is read whole.
    let request =
        format!("GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes())?;
    let mut response = Vec::with_capacity(256);
    let mut chunk = [0u8; 256];
    while !response.contains(&b'\n') && response.len() < 4096 {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        response.extend_from_slice(&chunk[..n]);
    }
    parse_health_status(&response)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "not an HTTP response"))
}

/// A model file larger than this does not share the video card with the
/// speech model: Gemma 4 26B takes 15.7 of 16 GB (I.5). TranslateGemma 12B
/// (6.8 GB), Qwen3.5 9B and 4B stay under it and fit next to Whisper.
pub const BIG_MODEL_BYTES: u64 = 10 * 1024 * 1024 * 1024;

/// Dictation rule (I.5): stop the server when it holds a model over
/// `threshold` bytes and no request is in flight (never mid-answer: then the
/// answer finishes and the server stays).
pub fn should_yield_to_dictation(
    model_bytes: Option<u64>,
    threshold: u64,
    in_flight: usize,
) -> bool {
    in_flight == 0 && model_bytes.is_some_and(|bytes| bytes > threshold)
}

/// Idle rule: `idle_ms = 0` never stops; a request in flight always keeps it.
pub fn should_stop_idle(now_ms: u64, last_used_ms: u64, idle_ms: u64, in_flight: usize) -> bool {
    idle_ms > 0 && in_flight == 0 && now_ms.saturating_sub(last_used_ms) >= idle_ms
}

/// Milliseconds on a monotonic clock; replaced by a fake one in tests.
pub trait Clock: Send + Sync {
    fn now_ms(&self) -> u64;
}

struct MonotonicClock(Instant);

impl Clock for MonotonicClock {
    fn now_ms(&self) -> u64 {
        self.0.elapsed().as_millis() as u64
    }
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "lowercase")]
pub enum ServerState {
    Stopped,
    Starting,
    Running,
}

// `idle_secs`: since the last use; `last_start_secs`: spawn → /health 200 of the last start.
#[derive(Serialize, Debug, Clone, PartialEq, Type)]
pub struct ServerStatus {
    pub state: ServerState,
    pub port: Option<u16>,
    pub base_url: Option<String>,
    pub idle_secs: Option<u64>,
    pub last_start_secs: Option<f64>,
}

#[derive(Debug)]
enum StartError {
    /// The process exited during start-up (bad flags, port taken, out of VRAM …).
    Exited(String),
    Other(String),
}

impl StartError {
    fn into_message(self) -> String {
        match self {
            StartError::Exited(m) | StartError::Other(m) => m,
        }
    }
}

struct Running {
    child: Child,
    port: u16,
    api_key: String,
    config: ServerConfig,
    // Dropped after the child is killed; closing it kills the process anyway.
    _job: Option<job::Job>,
}

impl Running {
    fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    fn kill(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Unix: a `Running` that is dropped without `kill()` (its `LlamaServer` goes
/// away) must not leave the server behind: the watchdog is removed with the
/// `Job` on a normal drop, so nothing else would stop it. On Windows the Job
/// Object does this.
#[cfg(unix)]
impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct LlamaServer {
    running: Mutex<Option<Running>>,
    /// A start is in progress (`running` is locked meanwhile).
    starting: AtomicBool,
    /// Set by `stop()` to abort a start that is waiting for `/health`.
    cancel_start: AtomicBool,
    last_used_ms: AtomicU64,
    in_flight: AtomicUsize,
    idle_unload_ms: AtomicU64,
    /// Duration of the last successful start in ms, 0 = none yet.
    last_start_ms: AtomicU64,
    clock: Box<dyn Clock>,
    idle_watcher: Once,
}

impl Default for LlamaServer {
    fn default() -> Self {
        Self::with_clock(Box::new(MonotonicClock(Instant::now())))
    }
}

/// Keeps the server from being stopped for idleness while a request runs.
pub struct UseGuard<'a>(&'a LlamaServer);

impl Drop for UseGuard<'_> {
    fn drop(&mut self) {
        self.0.touch();
        self.0.in_flight.fetch_sub(1, Ordering::AcqRel);
    }
}

impl LlamaServer {
    pub fn with_clock(clock: Box<dyn Clock>) -> Self {
        let now = clock.now_ms();
        Self {
            running: Mutex::new(None),
            starting: AtomicBool::new(false),
            cancel_start: AtomicBool::new(false),
            last_used_ms: AtomicU64::new(now),
            in_flight: AtomicUsize::new(0),
            idle_unload_ms: AtomicU64::new(5 * 60_000),
            last_start_ms: AtomicU64::new(0),
            clock,
            idle_watcher: Once::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Option<Running>> {
        self.running.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn touch(&self) {
        self.last_used_ms
            .store(self.clock.now_ms(), Ordering::Release);
    }

    /// Hold for the whole translation (touches before and after).
    pub fn begin_use(&self) -> UseGuard<'_> {
        self.in_flight.fetch_add(1, Ordering::AcqRel);
        self.touch();
        UseGuard(self)
    }

    pub fn set_idle_unload_ms(&self, ms: u64) {
        self.idle_unload_ms.store(ms, Ordering::Release);
    }

    /// Returns the port and API key of a healthy server, starting (or
    /// restarting with the new config) one if needed. Blocks up to 90 s per attempt.
    pub fn ensure_running(&self, config: &ServerConfig) -> Result<(u16, String), String> {
        bypass_proxy_for_localhost();
        self.touch();
        let mut guard = self.lock();
        if let Some(running) = guard.as_mut() {
            if running.config == *config && running.is_alive() {
                let port = running.port;
                match probe_health(port) {
                    Ok(Health::Ready) => return Ok((port, running.api_key.clone())),
                    Ok(Health::Loading) => {
                        warn!("llama-server on port {port} reloading, restarting")
                    }
                    Ok(Health::Unexpected(code)) => {
                        warn!("llama-server on port {port} answered HTTP {code}, restarting")
                    }
                    Err(e) => warn!("llama-server on port {port} not answering ({e}), restarting"),
                }
            } else if running.config != *config {
                info!("llama-server settings changed, restarting");
            } else {
                warn!("llama-server exited on its own, restarting");
            }
            if let Some(old) = guard.take() {
                old.kill();
            }
        }

        if !config.exe_path.is_file() {
            return Err(format!(
                "llama-server not found: {}",
                config.exe_path.display()
            ));
        }
        if !config.model_path.is_file() {
            return Err(format!(
                "Model file not found: {}",
                config.model_path.display()
            ));
        }

        self.starting.store(true, Ordering::Release);
        let result = match self.start_once(config) {
            Err(StartError::Exited(first)) => {
                warn!("llama-server start failed, retrying with a new port: {first}");
                self.start_once(config)
            }
            other => other,
        };
        self.starting.store(false, Ordering::Release);

        let (running, took) = result.map_err(StartError::into_message)?;
        let endpoint = (running.port, running.api_key.clone());
        info!(
            "llama-server ready on port {} in {:.2} s",
            running.port,
            took.as_secs_f64()
        );
        self.last_start_ms
            .store((took.as_millis() as u64).max(1), Ordering::Release);
        *guard = Some(running);
        self.touch();
        Ok(endpoint)
    }

    fn start_once(&self, config: &ServerConfig) -> Result<(Running, Duration), StartError> {
        let port =
            pick_free_port().map_err(|e| StartError::Other(format!("No free local port: {e}")))?;
        if let Some(dir) = config.log_path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let log = File::create(&config.log_path).map_err(|e| {
            StartError::Other(format!("Cannot create {}: {e}", config.log_path.display()))
        })?;
        let log_err = log
            .try_clone()
            .map_err(|e| StartError::Other(e.to_string()))?;

        let api_key = random_api_key();
        let mut command = Command::new(&config.exe_path);
        command
            .args(build_args(config, port, &api_key))
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(log_err));
        if let Some(dir) = config.exe_path.parent() {
            command.current_dir(dir);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        // Handy sets this for its own process (it cures a crash when Handy exits);
        // the server does not need it.
        #[cfg(target_os = "macos")]
        command.env_remove("GGML_METAL_NO_RESIDENCY");

        info!(
            "Starting llama-server on port {port}: {}",
            config.model_path.display()
        );
        let started = Instant::now();
        let child = command.spawn().map_err(|e| {
            StartError::Other(format!(
                "Failed to start {}: {e}",
                config.exe_path.display()
            ))
        })?;
        let job = match job::Job::kill_on_close().and_then(|job| job.assign(&child).map(|_| job)) {
            Ok(job) => Some(job),
            Err(e) => {
                warn!("llama-server is not in a Job Object (it may outlive a crash): {e}");
                None
            }
        };
        let mut running = Running {
            child,
            port,
            api_key,
            config: config.clone(),
            _job: job,
        };

        loop {
            if self.cancel_start.load(Ordering::Acquire) {
                running.kill();
                return Err(StartError::Other("llama-server start cancelled".into()));
            }
            if let Ok(Some(status)) = running.child.try_wait() {
                return Err(StartError::Exited(format!(
                    "llama-server exited during start-up ({status}). Last log lines:\n{}",
                    log_tail(&config.log_path, LOG_TAIL_LINES)
                )));
            }
            if let Ok(Health::Ready) = probe_health(port) {
                return Ok((running, started.elapsed()));
            }
            if started.elapsed() > START_TIMEOUT {
                running.kill();
                return Err(StartError::Other(format!(
                    "llama-server did not become ready in {} s. Last log lines:\n{}",
                    START_TIMEOUT.as_secs(),
                    log_tail(&config.log_path, LOG_TAIL_LINES)
                )));
            }
            std::thread::sleep(HEALTH_POLL);
        }
    }

    /// Kills the server (aborting a start in progress). Returns whether one was running.
    pub fn stop(&self) -> bool {
        self.cancel_start.store(true, Ordering::Release);
        let taken = self.lock().take();
        self.cancel_start.store(false, Ordering::Release);
        match taken {
            Some(running) => {
                info!("Stopping llama-server (port {})", running.port);
                running.kill();
                true
            }
            None => false,
        }
    }

    pub fn status(&self) -> ServerStatus {
        let last_start_secs = match self.last_start_ms.load(Ordering::Acquire) {
            0 => None,
            ms => Some(ms as f64 / 1000.0),
        };
        let mut status = ServerStatus {
            state: ServerState::Stopped,
            port: None,
            base_url: None,
            idle_secs: None,
            last_start_secs,
        };
        if self.starting.load(Ordering::Acquire) {
            status.state = ServerState::Starting;
            return status;
        }
        let mut guard = self.lock();
        if let Some(running) = guard.as_mut() {
            if running.is_alive() {
                let idle_ms = self
                    .clock
                    .now_ms()
                    .saturating_sub(self.last_used_ms.load(Ordering::Acquire));
                status.state = ServerState::Running;
                status.port = Some(running.port);
                status.base_url = Some(base_url_for_port(running.port));
                status.idle_secs = Some(idle_ms / 1000);
            } else {
                warn!("llama-server exited on its own");
                guard.take();
            }
        }
        status
    }

    /// The model file of the running server, if one is up and not just being
    /// started (G2: tells whether a slot change needs a restart).
    pub fn running_model(&self) -> Option<PathBuf> {
        if self.starting.load(Ordering::Acquire) {
            return None;
        }
        self.running
            .try_lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|r| r.config.model_path.clone()))
    }

    /// Dictation starts: frees the video memory of a big model before the
    /// speech model loads (I.5). Returns whether the server was stopped.
    pub fn yield_to_dictation(&self) -> bool {
        self.yield_to_dictation_over(BIG_MODEL_BYTES)
    }

    fn yield_to_dictation_over(&self, threshold: u64) -> bool {
        if self.in_flight.load(Ordering::Acquire) > 0 {
            info!("Dictation starts while a translation runs: llama-server stays");
            return false;
        }
        // Busy lock = a start or a health check: a translation is on its way.
        let Ok(mut guard) = self.running.try_lock() else {
            return false;
        };
        let bytes = guard
            .as_ref()
            .and_then(|r| std::fs::metadata(&r.config.model_path).ok())
            .map(|m| m.len());
        // Re-checked under the lock: a request takes `begin_use` before `ensure_running`.
        if !should_yield_to_dictation(bytes, threshold, self.in_flight.load(Ordering::Acquire)) {
            return false;
        }
        if let Some(running) = guard.take() {
            info!(
                "Dictation starts: stopping llama-server with a big model ({:.1} GB) to free the video memory (port {})",
                bytes.unwrap_or(0) as f64 / 1e9,
                running.port
            );
            running.kill();
        }
        true
    }

    /// One idle-watcher tick. Returns whether the server was stopped.
    pub fn check_idle(&self) -> bool {
        let idle_ms = self.idle_unload_ms.load(Ordering::Acquire);
        let idle_now = || {
            should_stop_idle(
                self.clock.now_ms(),
                self.last_used_ms.load(Ordering::Acquire),
                idle_ms,
                self.in_flight.load(Ordering::Acquire),
            )
        };
        if !idle_now() {
            return false;
        }
        // Busy lock = a start or a health check is running: not idle.
        let Ok(mut guard) = self.running.try_lock() else {
            return false;
        };
        // Re-check under the lock: a request takes `begin_use` before `ensure_running`.
        if guard.is_none() || !idle_now() {
            return false;
        }
        if let Some(running) = guard.take() {
            info!(
                "llama-server idle for {} min, stopping (port {})",
                idle_ms / 60_000,
                running.port
            );
            running.kill();
        }
        true
    }

    /// Starts the idle watcher (once per process); it ends when the server is dropped.
    pub fn spawn_idle_watcher(self: &Arc<Self>) {
        self.idle_watcher.call_once(|| {
            let weak = Arc::downgrade(self);
            let spawned = std::thread::Builder::new()
                .name("llama-server-idle".into())
                .spawn(move || loop {
                    std::thread::sleep(IDLE_TICK);
                    match weak.upgrade() {
                        Some(server) => {
                            server.check_idle();
                        }
                        None => break,
                    }
                });
            if let Err(e) = spawned {
                warn!("Failed to start the llama-server idle watcher: {e}");
            }
        });
    }
}

/// Last `lines` lines of the server log (lossy UTF-8).
fn log_tail(path: &Path, lines: usize) -> String {
    let Ok(bytes) = std::fs::read(path) else {
        return "(log not readable)".into();
    };
    let text = String::from_utf8_lossy(&bytes);
    let all: Vec<&str> = text.lines().collect();
    all[all.len().saturating_sub(lines)..].join("\n")
}

#[cfg(windows)]
mod job {
    use std::ffi::c_void;
    use std::io;
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;

    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, JobObjectExtendedLimitInformation, SetInformationJobObject,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    // The windows crate gates CreateJobObjectW behind `Win32_Security` (only for
    // its SECURITY_ATTRIBUTES parameter); declared here so Cargo.toml needs just
    // `Win32_System_JobObjects`. Null attributes = handle not inheritable, so the
    // server never holds its own job open.
    #[link(name = "kernel32")]
    extern "system" {
        fn CreateJobObjectW(attributes: *const c_void, name: *const u16) -> HANDLE;
    }

    pub struct Job(HANDLE);

    // A kernel handle; usable from any thread.
    unsafe impl Send for Job {}

    impl Job {
        pub fn kill_on_close() -> io::Result<Self> {
            let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if handle.0.is_null() {
                return Err(io::Error::last_os_error());
            }
            let job = Job(handle);
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            unsafe {
                SetInformationJobObject(
                    job.0,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const c_void,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            }
            .map_err(io::Error::other)?;
            Ok(job)
        }

        pub fn assign(&self, child: &Child) -> io::Result<()> {
            let process = HANDLE(child.as_raw_handle());
            unsafe { AssignProcessToJobObject(self.0, process) }.map_err(io::Error::other)
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }
}

/// macOS / Linux have no Job Object. A tiny watchdog process stands in for it
/// (DESIGN J.8): `sh` waits on a pipe whose other end only SelTranslate holds.
/// When SelTranslate ends, however it ends (`kill -9` included), the kernel
/// closes the pipe, `read` returns and the watchdog stops the server: TERM,
/// then KILL after 3 s.
#[cfg(unix)]
mod job {
    use std::io;
    use std::process::{Child, ChildStdin, Command, Stdio};
    use std::sync::Mutex;

    const SCRIPT: &str = r#"read _; kill -TERM "$1"; sleep 3; kill -KILL "$1""#;

    struct Watchdog {
        process: Child,
        pipe: Option<ChildStdin>,
    }

    pub struct Job(Mutex<Option<Watchdog>>);

    impl Job {
        pub fn kill_on_close() -> io::Result<Self> {
            Ok(Job(Mutex::new(None)))
        }

        pub fn assign(&self, child: &Child) -> io::Result<()> {
            let mut process = Command::new("/bin/sh")
                .args(["-c", SCRIPT, "sh", &child.id().to_string()])
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()?;
            let pipe = process.stdin.take();
            *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(Watchdog { process, pipe });
            Ok(())
        }

        /// Test only: what the death of SelTranslate does.
        #[cfg(test)]
        pub fn close_pipe(&self) {
            if let Some(w) = self.0.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
                w.pipe = None;
            }
        }

        #[cfg(test)]
        pub fn watchdog_pid(&self) -> Option<u32> {
            self.0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .map(|w| w.process.id())
        }
    }

    impl Drop for Job {
        /// A normal stop: the watchdog goes first, then the pipe closes, so it
        /// can never fire at a pid the server has already given back.
        fn drop(&mut self) {
            let watchdog = self.0.get_mut().unwrap_or_else(|e| e.into_inner()).take();
            if let Some(mut watchdog) = watchdog {
                let _ = watchdog.process.kill();
                let _ = watchdog.process.wait();
                drop(watchdog.pipe.take());
            }
        }
    }
}

#[cfg(not(any(windows, unix)))]
mod job {
    use std::io;
    use std::process::Child;

    pub struct Job;

    impl Job {
        pub fn kill_on_close() -> io::Result<Self> {
            Ok(Job)
        }

        pub fn assign(&self, _child: &Child) -> io::Result<()> {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    struct FakeClock(Arc<AtomicU64>);

    impl Clock for FakeClock {
        fn now_ms(&self) -> u64 {
            self.0.load(Ordering::SeqCst)
        }
    }

    fn fake_server() -> (LlamaServer, Arc<AtomicU64>) {
        let now = Arc::new(AtomicU64::new(1_000_000));
        (
            LlamaServer::with_clock(Box::new(FakeClock(now.clone()))),
            now,
        )
    }

    fn config(exe: &str, model: &str, log: PathBuf) -> ServerConfig {
        ServerConfig {
            exe_path: exe.into(),
            model_path: model.into(),
            gpu_layers: 99,
            ctx_size: 4096,
            log_path: log,
            // The B1 tests use any file name; the B1 command line is TranslateGemma's.
            chat_format: ChatFormat::TranslateGemma,
        }
    }

    #[test]
    fn the_chat_format_comes_from_the_file_name() {
        for name in [
            "/m/translategemma-12b-it.Q4_K_M.gguf",
            "/m/TranslateGemma-4B-it.Q8_0.gguf",
            "translategemma.gguf",
        ] {
            assert_eq!(
                chat_format_for(Path::new(name)),
                ChatFormat::TranslateGemma,
                "{name}"
            );
        }
        for name in [
            "/m/Qwen3.5-9B-Q4_K_M.gguf",
            "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
            "/translategemma/other.gguf",
            "",
        ] {
            assert_eq!(chat_format_for(Path::new(name)), ChatFormat::Chat, "{name}");
        }
    }

    #[test]
    fn a_chat_model_gets_its_own_template_without_reasoning() {
        let model = "/models/Qwen3.5-9B-Q4_K_M.gguf";
        let cfg = ServerConfig {
            chat_format: chat_format_for(Path::new(model)),
            ..config(r"C:\llama\llama-server.exe", model, "x.log".into())
        };
        assert_eq!(cfg.chat_format, ChatFormat::Chat);
        let expected = "-m /models/Qwen3.5-9B-Q4_K_M.gguf --host 127.0.0.1 --port 8123 -ngl auto \
                        -c 4096 -np 1 -fa auto --no-webui --jinja --reasoning off \
                        --temp 0 -n 1024 --api-key k3y";
        assert_eq!(build_args(&cfg, 8123, "k3y").join(" "), expected);
        assert!(!build_args(&cfg, 1, "k").contains(&"--no-jinja".to_string()));
    }

    #[test]
    fn dictation_pushes_out_only_a_big_idle_model() {
        let big = 11 * 1024 * 1024 * 1024;
        assert!(should_yield_to_dictation(Some(big), BIG_MODEL_BYTES, 0));
        assert!(
            !should_yield_to_dictation(Some(big), BIG_MODEL_BYTES, 1),
            "mid-answer"
        );
        assert!(
            !should_yield_to_dictation(Some(7_300_794_112), BIG_MODEL_BYTES, 0),
            "12B"
        );
        assert!(
            !should_yield_to_dictation(None, BIG_MODEL_BYTES, 0),
            "no server"
        );
        // Gemma 4 26B is over the line, every other model on disk under it.
        const _: () = assert!(16_947_541_728 > BIG_MODEL_BYTES && 7_300_794_112 < BIG_MODEL_BYTES);
    }

    #[test]
    fn dictation_without_a_server_does_nothing() {
        let (server, _) = fake_server();
        assert!(!server.yield_to_dictation_over(0));
    }

    #[test]
    fn build_args_is_the_b1_command_line() {
        let cfg = config(
            r"C:\llama\llama-server.exe",
            r"C:\models\m.gguf",
            PathBuf::from("x.log"),
        );
        // 99 ("Video card (fast)") is `-ngl auto` since F2: as many layers as fit.
        let expected = "-m C:\\models\\m.gguf --host 127.0.0.1 --port 8123 -ngl auto -c 4096 \
                        -np 1 -fa auto --no-webui --no-jinja --chat-template gemma \
                        --temp 0 -n 1024 --api-key k3y";
        assert_eq!(build_args(&cfg, 8123, "k3y").join(" "), expected);

        let small = ServerConfig {
            gpu_layers: 20,
            ctx_size: 2048,
            ..cfg
        };
        let args = build_args(&small, 1, "k");
        let after = |flag: &str| &args[args.iter().position(|a| a == flag).unwrap() + 1];
        assert_eq!(after("-ngl"), "20");
        assert_eq!(after("-c"), "2048");
        assert!(args.contains(&"--no-jinja".to_string()));
        let cpu = build_args(
            &ServerConfig {
                gpu_layers: 0,
                ..small
            },
            1,
            "k",
        );
        assert_eq!(cpu[cpu.iter().position(|a| a == "-ngl").unwrap() + 1], "0");
    }

    #[test]
    fn base_url_points_at_the_v1_api() {
        assert_eq!(base_url_for_port(5000), "http://127.0.0.1:5000/v1");
    }

    #[test]
    fn api_keys_are_random_128_bit_hex() {
        let a = random_api_key();
        let b = random_api_key();
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
        assert_ne!(a[..16], a[16..], "two independent halves");
    }

    #[test]
    fn no_proxy_gets_the_local_server() {
        assert_eq!(
            no_proxy_with_localhost(None).as_deref(),
            Some("127.0.0.1,localhost")
        );
        assert_eq!(
            no_proxy_with_localhost(Some("corp.example")).as_deref(),
            Some("corp.example,127.0.0.1,localhost")
        );
        assert_eq!(no_proxy_with_localhost(Some("a, 127.0.0.1")), None);
        assert_eq!(no_proxy_with_localhost(Some("*")), None);
    }

    #[test]
    fn picked_port_is_free_and_nonzero() {
        // Another test may pick the same ephemeral port in between (macOS
        // hands them out again at once), so one retry round is allowed.
        let bindable = (0..5).any(|_| {
            let port = pick_free_port().unwrap();
            assert_ne!(port, 0);
            TcpListener::bind(("127.0.0.1", port)).is_ok()
        });
        assert!(bindable, "a picked port should be bindable");
    }

    #[test]
    fn parses_health_status_lines() {
        let ok = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\":\"ok\"}";
        assert_eq!(parse_health_status(ok), Some(Health::Ready));
        assert_eq!(
            parse_health_status(b"HTTP/1.1 503 Service Unavailable\r\n"),
            Some(Health::Loading)
        );
        assert_eq!(
            parse_health_status(b"HTTP/1.0 404 Not Found\r\n"),
            Some(Health::Unexpected(404))
        );
        assert_eq!(parse_health_status(b"SSH-2.0-OpenSSH\r\n"), None);
        assert_eq!(parse_health_status(b""), None);
        assert_eq!(parse_health_status(b"HTTP/1.1 abc\r\n"), None);
    }

    /// One-shot fake HTTP server answering every connection with `reply`.
    fn fake_http(reply: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                // Read the whole request: closing with unread data sends a reset.
                let mut request = Vec::new();
                let mut buf = [0u8; 512];
                while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                    match stream.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => request.extend_from_slice(&buf[..n]),
                    }
                }
                let _ = stream.write_all(reply.as_bytes());
            }
        });
        port
    }

    #[test]
    fn probes_health_over_tcp() {
        let ready = fake_http("HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}");
        assert_eq!(probe_health(ready).unwrap(), Health::Ready);
        let loading = fake_http("HTTP/1.1 503 Service Unavailable\r\n\r\n");
        assert_eq!(probe_health(loading).unwrap(), Health::Loading);
        let closed = pick_free_port().unwrap();
        assert!(probe_health(closed).is_err());
    }

    #[test]
    fn idle_rule() {
        let min5 = 5 * 60_000;
        assert!(!should_stop_idle(1_000, 0, min5, 0));
        assert!(should_stop_idle(min5, 0, min5, 0));
        assert!(
            !should_stop_idle(min5 * 10, 0, min5, 1),
            "request in flight"
        );
        assert!(!should_stop_idle(u64::MAX, 0, 0, 0), "0 = never");
        assert!(!should_stop_idle(0, 10, min5, 0), "clock behind last use");
    }

    #[test]
    fn idle_check_without_server_does_nothing() {
        let (server, now) = fake_server();
        server.set_idle_unload_ms(1_000);
        now.fetch_add(10_000, Ordering::SeqCst);
        assert!(!server.check_idle());
        assert_eq!(server.status().state, ServerState::Stopped);
        assert!(!server.stop());
    }

    #[test]
    fn use_guard_touches_and_tracks_requests() {
        let (server, now) = fake_server();
        now.fetch_add(60_000, Ordering::SeqCst);
        {
            let _use = server.begin_use();
            assert_eq!(server.in_flight.load(Ordering::SeqCst), 1);
            assert_eq!(server.last_used_ms.load(Ordering::SeqCst), 1_060_000);
            now.fetch_add(30_000, Ordering::SeqCst);
        }
        assert_eq!(server.in_flight.load(Ordering::SeqCst), 0);
        assert_eq!(server.last_used_ms.load(Ordering::SeqCst), 1_090_000);
    }

    #[test]
    fn missing_files_are_reported_before_spawning() {
        let (server, _) = fake_server();
        let log = PathBuf::from("never-created.log");
        let err = server
            .ensure_running(&config(
                r"Z:\nope\llama-server.exe",
                "Cargo.toml",
                log.clone(),
            ))
            .unwrap_err();
        assert!(err.starts_with("llama-server not found"), "{err}");
        let err = server
            .ensure_running(&config(
                &std::env::current_exe().unwrap().to_string_lossy(),
                r"Z:\nope\m.gguf",
                log,
            ))
            .unwrap_err();
        assert!(err.starts_with("Model file not found"), "{err}");
    }

    /// A long-lived stand-in process (ping waits ~1 s per echo).
    #[cfg(windows)]
    fn dummy_child() -> Child {
        use std::os::windows::process::CommandExt;
        Command::new("ping")
            .args(["-n", "120", "127.0.0.1"])
            .stdout(Stdio::null())
            .creation_flags(0x0800_0000)
            .spawn()
            .unwrap()
    }

    #[cfg(unix)]
    fn sleeper() -> Child {
        Command::new("sleep").arg("60").spawn().unwrap()
    }

    #[cfg(unix)]
    fn wait_for_exit(child: &mut Child, within: Duration) -> bool {
        let deadline = Instant::now() + within;
        while Instant::now() < deadline {
            if child.try_wait().unwrap().is_some() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        child.try_wait().unwrap().is_some()
    }

    #[cfg(unix)]
    #[test]
    fn the_watchdog_stops_the_server_when_the_pipe_closes() {
        let mut child = sleeper();
        let job = job::Job::kill_on_close().unwrap();
        job.assign(&child).unwrap();
        assert!(child.try_wait().unwrap().is_none());
        // SelTranslate dies: the kernel closes its end of the pipe.
        job.close_pipe();
        assert!(wait_for_exit(&mut child, Duration::from_secs(1)));
    }

    #[cfg(unix)]
    #[test]
    fn dropping_a_running_server_stops_it_and_its_watchdog() {
        let child = sleeper();
        let pid = child.id();
        let job = job::Job::kill_on_close().unwrap();
        job.assign(&child).unwrap();
        let watchdog = job.watchdog_pid().unwrap();
        let running = Running {
            child,
            port: 1,
            api_key: String::new(),
            config: config("x", "y", PathBuf::new()),
            _job: Some(job),
        };
        drop(running);
        // SAFETY: signal 0 only asks whether the process exists.
        let exists = |pid: u32| unsafe { libc::kill(pid as i32, 0) == 0 };
        assert!(!exists(pid), "the server is gone");
        assert!(!exists(watchdog), "the watchdog is gone");
    }

    #[cfg(unix)]
    #[test]
    fn a_normal_drop_removes_the_watchdog_and_leaves_the_server() {
        let mut child = sleeper();
        let job = job::Job::kill_on_close().unwrap();
        job.assign(&child).unwrap();
        let watchdog = job.watchdog_pid().unwrap() as i32;
        drop(job);
        std::thread::sleep(Duration::from_millis(200));
        assert!(
            child.try_wait().unwrap().is_none(),
            "the server lives until Running::kill"
        );
        // SAFETY: signal 0 only asks whether the process exists.
        let alive = unsafe { libc::kill(watchdog, 0) } == 0;
        assert!(!alive, "the watchdog is gone");
        child.kill().unwrap();
        child.wait().unwrap();
    }

    #[cfg(windows)]
    fn install_dummy(server: &LlamaServer, job: bool) -> u32 {
        let child = dummy_child();
        let pid = child.id();
        let job = job.then(|| {
            let j = job::Job::kill_on_close().unwrap();
            j.assign(&child).unwrap();
            j
        });
        *server.lock() = Some(Running {
            child,
            port: 1,
            api_key: String::new(),
            config: config("x", "y", PathBuf::new()),
            _job: job,
        });
        pid
    }

    #[cfg(windows)]
    fn process_alive(pid: u32) -> bool {
        let out = Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).contains(&pid.to_string())
    }

    #[cfg(windows)]
    #[test]
    fn idle_watcher_stops_only_after_the_timeout_and_never_mid_request() {
        let (server, now) = fake_server();
        server.set_idle_unload_ms(5 * 60_000);
        let pid = install_dummy(&server, false);
        server.touch();
        assert_eq!(server.status().state, ServerState::Running);

        now.fetch_add(4 * 60_000, Ordering::SeqCst);
        assert!(!server.check_idle(), "4 min < 5 min");
        {
            let _use = server.begin_use();
            now.fetch_add(10 * 60_000, Ordering::SeqCst);
            assert!(!server.check_idle(), "request in flight");
        }
        now.fetch_add(4 * 60_000, Ordering::SeqCst);
        assert!(!server.check_idle(), "guard drop touched the clock");
        assert_eq!(server.status().idle_secs, Some(240));

        now.fetch_add(60_000, Ordering::SeqCst);
        assert!(server.check_idle());
        assert_eq!(server.status().state, ServerState::Stopped);
        assert!(!process_alive(pid));
    }

    #[cfg(windows)]
    #[test]
    fn dropping_the_job_kills_the_process() {
        let mut child = dummy_child();
        let pid = child.id();
        let job = job::Job::kill_on_close().unwrap();
        job.assign(&child).unwrap();
        assert!(process_alive(pid));
        drop(job);
        let deadline = Instant::now() + Duration::from_secs(3);
        while child.try_wait().unwrap().is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(child.try_wait().unwrap().is_some(), "KILL_ON_JOB_CLOSE");
        assert!(!process_alive(pid));
    }

    #[cfg(windows)]
    #[test]
    fn dictation_stops_a_big_model_but_not_a_small_one_or_a_busy_one() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("big.gguf");
        std::fs::write(&model, [0u8; 100]).unwrap();
        let (server, _) = fake_server();
        let pid = install_dummy(&server, true);
        server.lock().as_mut().unwrap().config.model_path = model.clone();
        assert!(!server.yield_to_dictation_over(100), "not over the line");
        {
            let _use = server.begin_use();
            assert!(!server.yield_to_dictation_over(10), "a request in flight");
        }
        assert!(process_alive(pid));
        assert!(server.yield_to_dictation_over(10));
        assert!(!process_alive(pid));
        assert_eq!(server.status().state, ServerState::Stopped);
    }

    #[cfg(windows)]
    #[test]
    fn stop_kills_and_reports() {
        let (server, _) = fake_server();
        let pid = install_dummy(&server, true);
        assert_eq!(
            server.status().base_url.as_deref(),
            Some("http://127.0.0.1:1/v1")
        );
        assert!(server.stop());
        assert!(!process_alive(pid));
        assert_eq!(server.status().base_url, None);
        assert!(!server.stop());
    }

    /// A process that exits at once: start fails after one retry, with the log tail.
    #[cfg(windows)]
    #[test]
    fn early_exit_is_retried_once_and_reports_the_log_tail() {
        let (server, _) = fake_server();
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("llama-server.log");
        // where.exe does not understand the llama flags: it prints a message and exits.
        let cfg = config(r"C:\Windows\System32\where.exe", "Cargo.toml", log);
        let started = Instant::now();
        let err = server.ensure_running(&cfg).unwrap_err();
        assert!(err.contains("exited during start-up"), "{err}");
        let tail = err.split("Last log lines:\n").nth(1).unwrap_or("");
        assert!(!tail.trim().is_empty(), "log tail expected: {err}");
        assert!(started.elapsed() < Duration::from_secs(10));
        assert_eq!(server.status().state, ServerState::Stopped);
    }

    #[test]
    fn log_tail_keeps_the_last_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("l.log");
        let text: String = (1..=30).map(|i| format!("line {i}\n")).collect();
        std::fs::write(&path, text).unwrap();
        let tail = log_tail(&path, 20);
        assert!(tail.starts_with("line 11\n"));
        assert!(tail.ends_with("line 30"));
        assert_eq!(log_tail(&dir.path().join("none"), 20), "(log not readable)");
    }

    /// Live (unix): the app "dies" with `kill -9` and the watchdog takes the
    /// real server with it (DESIGN J.8). Needs SELTRANSLATE_LLAMA_EXE and
    /// SELTRANSLATE_MODEL; the helper below runs as a child process.
    #[cfg(unix)]
    #[test]
    #[ignore]
    fn live_helper_server_that_waits() {
        let exe = std::env::var("SELTRANSLATE_LLAMA_EXE").expect("SELTRANSLATE_LLAMA_EXE");
        let model = std::env::var("SELTRANSLATE_MODEL").expect("SELTRANSLATE_MODEL");
        let dir = tempfile::tempdir().unwrap();
        let server = LlamaServer::default();
        let cfg = ServerConfig {
            chat_format: chat_format_for(Path::new(&model)),
            ..config(&exe, &model, dir.path().join("llama-server.log"))
        };
        server.ensure_running(&cfg).expect("server start");
        let pid = server.lock().as_ref().expect("running").child.id();
        println!("READY {pid}");
        std::thread::sleep(Duration::from_secs(300));
    }

    #[cfg(unix)]
    fn process_exists(pid: u32) -> bool {
        // SAFETY: signal 0 only asks whether the process exists.
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }

    #[cfg(unix)]
    #[test]
    #[ignore]
    fn live_kill9_of_the_app_leaves_no_server() {
        use std::io::{BufRead, BufReader};
        let mut app = Command::new(std::env::current_exe().unwrap())
            .args([
                "translate::llama_server::tests::live_helper_server_that_waits",
                "--exact",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut lines = BufReader::new(app.stdout.take().unwrap()).lines();
        let server_pid: u32 = loop {
            let line = lines.next().expect("the helper ended early").unwrap();
            // libtest prints "test <name> ... " before the helper's own line.
            if let Some((_, pid)) = line.split_once("READY ") {
                break pid.trim().parse().unwrap();
            }
        };
        assert!(process_exists(server_pid), "the server runs");
        let killed = Instant::now();
        app.kill().unwrap(); // SIGKILL
        app.wait().unwrap();
        while process_exists(server_pid) && killed.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(20));
        }
        let gone_after = killed.elapsed();
        println!("llama-server gone {gone_after:?} after kill -9 of the app");
        assert!(!process_exists(server_pid), "the server outlived the app");
        assert!(gone_after < Duration::from_millis(1500), "{gone_after:?}");
        // The watchdog itself ends after its 3 s grace.
        std::thread::sleep(Duration::from_secs(4));
        let leftover = Command::new("pgrep")
            .args([
                "-f",
                &format!("kill -KILL \"\\$1\".*{server_pid}|sh -c read _"),
            ])
            .output()
            .unwrap();
        assert!(leftover.stdout.is_empty(), "a watchdog is left");
    }

    /// Real server + `llm_client` (B2 acceptance). Paths default to the
    /// settings defaults; override with SELTRANSLATE_LLAMA_EXE / SELTRANSLATE_MODEL.
    #[test]
    #[ignore]
    fn real_server_translates_through_llm_client() {
        let defaults = crate::translate::settings::SelTranslateSettings::default();
        let exe = std::env::var("SELTRANSLATE_LLAMA_EXE").unwrap_or(defaults.llama_server_path);
        let model = std::env::var("SELTRANSLATE_MODEL").unwrap_or(defaults.model_path);
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("llama-server.log");
        let server = Arc::new(LlamaServer::default());
        let cfg = config(&exe, &model, log);

        let (port, api_key) = server.ensure_running(&cfg).expect("server start");
        let status = server.status();
        println!(
            "llama-server ready on port {port} in {:?} s",
            status.last_start_secs
        );
        assert_eq!(status.state, ServerState::Running);
        assert_eq!(
            server.ensure_running(&cfg).unwrap(),
            (port, api_key.clone()),
            "reuses the server"
        );

        let provider = crate::settings::PostProcessProvider {
            id: "seltranslate-local".into(),
            label: "Local".into(),
            base_url: base_url_for_port(port),
            allow_base_url_edit: false,
            models_endpoint: None,
            supports_structured_output: false,
        };
        // DESIGN B.3, ru → en.
        let prompt = "You are a professional Russian (ru) to English (en) translator. Your goal is \
                      to accurately convey the meaning and nuances of the original Russian text \
                      while adhering to English grammar, vocabulary, and cultural sensitivities.\n\
                      Produce only the English translation, without any additional explanations \
                      or commentary. Please translate the following Russian text into English:\n\n\n\
                      Привет, мир";
        let chat = |key: String| {
            tauri::async_runtime::block_on(crate::llm_client::send_chat_completion(
                &provider,
                key,
                "translategemma",
                prompt.to_string(),
                false,
            ))
        };
        let refused = chat(String::new()).expect_err("no key must be refused");
        assert!(refused.contains("401"), "{refused}");
        let refused = chat("wrong".into()).expect_err("wrong key must be refused");
        assert!(refused.contains("401"), "{refused}");

        let answer = {
            let _use = server.begin_use();
            let t = Instant::now();
            let answer = tauri::async_runtime::block_on(crate::llm_client::send_chat_completion(
                &provider,
                api_key,
                "translategemma",
                prompt.to_string(),
                false,
            ))
            .expect("chat completion")
            .expect("content");
            println!("answer in {:.2} s: {answer:?}", t.elapsed().as_secs_f64());
            answer
        };
        let lower = answer.trim().to_lowercase();
        assert!(
            lower.contains("hello") && lower.contains("world"),
            "{answer:?}"
        );
        assert!(
            lower.len() < 40,
            "only the translation expected: {answer:?}"
        );

        assert!(server.stop());
        assert!(probe_health(port).is_err(), "port closed after stop");
    }
}
