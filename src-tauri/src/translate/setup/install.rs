//! The safety checks and file work of the setup (DESIGN F.7): allowed hosts
//! and redirects, free space, the engine archive and the engine check.

use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use reqwest::Url;
use serde::Serialize;
use specta::Type;

use super::catalog::SERVER_EXE;

/// What went wrong, for the card's text (DESIGN F.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    Network,
    DiskFull,
    Hash,
    SourceGone,
    Host,
    EngineCheck,
    /// CUDA did not start (old driver): the previous engine stays (F.10).
    CudaFailed,
    Tar,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupError {
    pub kind: ErrorKind,
    pub message: String,
}

impl SetupError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

/// The kind of a transport error (`download_http_resumable_with_events`
/// reports everything as `anyhow`): a full disk, a damaged file, a file gone
/// from the source, or a network problem worth retrying.
pub fn classify(err: &anyhow::Error) -> ErrorKind {
    for cause in err.chain() {
        if let Some(io) = cause.downcast_ref::<std::io::Error>() {
            // ERROR_HANDLE_DISK_FULL (39), ERROR_DISK_FULL (112).
            if io.kind() == std::io::ErrorKind::StorageFull
                || matches!(io.raw_os_error(), Some(39 | 112))
            {
                return ErrorKind::DiskFull;
            }
        }
        if cause.downcast_ref::<reqwest::Error>().is_some() {
            return ErrorKind::Network;
        }
    }
    let text = err.to_string();
    if text.contains("verification failed")
        || text.contains("Failed to verify")
        || text.contains("more than the expected")
    {
        return ErrorKind::Hash;
    }
    if let Some(code) = http_status_in(&text) {
        return status_kind(code);
    }
    if text.contains("advertises") || text.contains("HTTP 416") {
        return ErrorKind::SourceGone;
    }
    if text.contains("stalled") || text.contains("no response") || text.contains("incomplete") {
        return ErrorKind::Network;
    }
    ErrorKind::Other
}

/// `server returned HTTP 404 Not Found` → 404.
fn http_status_in(text: &str) -> Option<u16> {
    let rest = text.split("HTTP ").nth(1)?;
    rest.get(..3)?.parse().ok()
}

/// 4xx means the file is not there (no retry); 408/429 and 5xx are retried.
fn status_kind(code: u16) -> ErrorKind {
    match code {
        408 | 429 => ErrorKind::Network,
        400..=499 => ErrorKind::SourceGone,
        _ => ErrorKind::Network,
    }
}

/// Hosts the downloads may come from, exactly or as a parent domain.
const ALLOWED_HOSTS: [&str; 4] = [
    "github.com",
    "githubusercontent.com",
    "huggingface.co",
    "hf.co",
];
/// Of these only subdomains are allowed (not the bare domain).
const SUBDOMAIN_ONLY: [&str; 2] = ["githubusercontent.com", "hf.co"];

/// Whether `url` is https on an allowed host (`github.com`,
/// `*.githubusercontent.com`, `huggingface.co`, `*.huggingface.co`, `*.hf.co`).
pub fn host_allowed(url: &str) -> bool {
    let Ok(url) = Url::parse(url) else {
        return false;
    };
    if url.scheme() != "https" {
        return false;
    }
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host.to_ascii_lowercase();
    ALLOWED_HOSTS.iter().any(|allowed| {
        let sub = host
            .strip_suffix(allowed)
            .is_some_and(|rest| rest.ends_with('.') && rest.len() > 1);
        sub || (host == *allowed && !SUBDOMAIN_ONLY.contains(allowed))
    })
}

const MAX_REDIRECTS: usize = 5;
const PROBE_TIMEOUT: Duration = Duration::from_secs(30);

/// Follows the redirects of `url` itself (HEAD, no automatic following),
/// checking every hop before anything is downloaded, and returns the final
/// address for the transport (which then follows nothing new).
pub async fn resolve_redirects(url: &str) -> Result<String, SetupError> {
    resolve_redirects_with(url, &host_allowed).await
}

async fn resolve_redirects_with(
    url: &str,
    allowed: &(dyn Fn(&str) -> bool + Sync),
) -> Result<String, SetupError> {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(15))
        .timeout(PROBE_TIMEOUT)
        .build()
        .map_err(|e| SetupError::new(ErrorKind::Other, e.to_string()))?;
    let mut current = url.to_string();
    for _ in 0..=MAX_REDIRECTS {
        if !allowed(&current) {
            return Err(SetupError::new(ErrorKind::Host, host_of(&current)));
        }
        let mut response = client
            .head(&current)
            .send()
            .await
            .map_err(|e| SetupError::new(ErrorKind::Network, e.to_string()))?;
        if response.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED {
            // The body is never read: dropping the response closes it.
            response = client
                .get(&current)
                .send()
                .await
                .map_err(|e| SetupError::new(ErrorKind::Network, e.to_string()))?;
        }
        let status = response.status();
        if status.is_redirection() {
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| {
                    SetupError::new(ErrorKind::Other, format!("HTTP {status} without Location"))
                })?;
            let next = Url::parse(&current)
                .and_then(|base| base.join(location))
                .map_err(|e| SetupError::new(ErrorKind::Host, e.to_string()))?;
            current = next.to_string();
            continue;
        }
        if status.is_success() {
            return Ok(current);
        }
        return Err(SetupError::new(
            status_kind(status.as_u16()),
            format!("HTTP {}", status.as_u16()),
        ));
    }
    Err(SetupError::new(
        ErrorKind::Host,
        format!("more than {MAX_REDIRECTS} redirects"),
    ))
}

fn host_of(url: &str) -> String {
    Url::parse(url)
        .ok()
        .map(|u| format!("{}://{}", u.scheme(), u.host_str().unwrap_or("")))
        .unwrap_or_else(|| url.to_string())
}

/// Free bytes for the caller on the volume of `path` (the nearest existing
/// folder above it). `None` when Windows cannot tell.
#[cfg(windows)]
pub fn free_bytes(path: &Path) -> Option<u64> {
    use windows::core::HSTRING;
    use windows::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    let existing = path.ancestors().find(|p| p.is_dir())?;
    let mut free = 0u64;
    unsafe {
        GetDiskFreeSpaceExW(
            &HSTRING::from(existing.as_os_str()),
            Some(&mut free),
            None,
            None,
        )
    }
    .ok()?;
    Some(free)
}

#[cfg(unix)]
pub fn free_bytes(path: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let existing = path.ancestors().find(|p| p.is_dir())?;
    let c_path = std::ffi::CString::new(existing.as_os_str().as_bytes()).ok()?;
    // SAFETY: a NUL-terminated path and a zeroed out-struct of the right type.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    Some(u64::from(stat.f_bavail) * stat.f_frsize)
}

#[cfg(not(any(windows, unix)))]
pub fn free_bytes(_path: &Path) -> Option<u64> {
    None
}

/// The volume of a path for the card ("C:"), and to tell whether the models
/// and the engine share one.
pub fn volume_of(path: &Path) -> String {
    match path.components().next() {
        Some(Component::Prefix(prefix)) => prefix
            .as_os_str()
            .to_string_lossy()
            .trim_end_matches('\\')
            .to_uppercase(),
        _ => String::new(),
    }
}

/// Checks every name of `tar -tf` before anything is unpacked: no absolute
/// path, drive, `..`, `:` (an NTFS stream) or control character.
pub fn check_archive_names(listing: &str) -> Result<(), String> {
    let mut any = false;
    for name in listing.lines() {
        let name = name.trim_end_matches('\r');
        if name.is_empty() {
            continue;
        }
        any = true;
        let bad = name.starts_with('/')
            || name.starts_with('\\')
            || name.contains(':')
            || name.chars().any(char::is_control)
            || name.split(['/', '\\']).any(|part| part == "..");
        if bad {
            return Err(format!("unsafe name in the archive: {name:?}"));
        }
    }
    if any {
        Ok(())
    } else {
        Err("the archive is empty".into())
    }
}

/// Runs a short helper process without a window and kills it on timeout.
pub fn run_with_timeout(mut command: Command, timeout: Duration) -> Result<Output, String> {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    let read_all = |pipe: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut buf);
            }
            buf
        })
    };
    let out = read_all(
        child
            .stdout
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    );
    let err = read_all(
        child
            .stderr
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    );
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => break status,
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("no answer within {} s", timeout.as_secs()));
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    };
    Ok(Output {
        status,
        stdout: out.join().unwrap_or_default(),
        stderr: err.join().unwrap_or_default(),
    })
}

/// `/usr/bin/tar` (bsdtar, understands .tar.gz) on a Mac.
#[cfg(unix)]
fn tar_exe() -> PathBuf {
    PathBuf::from("/usr/bin/tar")
}

/// `tar.exe` by its full path (never from PATH).
#[cfg(not(unix))]
fn tar_exe() -> PathBuf {
    let root = std::env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into());
    PathBuf::from(root).join("System32").join("tar.exe")
}

const TAR_TIMEOUT: Duration = Duration::from_secs(300);

fn run_tar(args: &[&std::ffi::OsStr]) -> Result<String, SetupError> {
    let mut command = Command::new(tar_exe());
    command.args(args);
    let output =
        run_with_timeout(command, TAR_TIMEOUT).map_err(|e| SetupError::new(ErrorKind::Tar, e))?;
    if !output.status.success() {
        return Err(SetupError::new(
            ErrorKind::Tar,
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Lists the zip, checks the names, then unpacks it into `dest`.
pub fn unpack(zip: &Path, dest: &Path) -> Result<(), SetupError> {
    let listing = run_tar(&["-tf".as_ref(), zip.as_os_str()])?;
    check_archive_names(&listing).map_err(|e| SetupError::new(ErrorKind::Tar, e))?;
    std::fs::create_dir_all(dest).map_err(|e| SetupError::new(ErrorKind::Other, e.to_string()))?;
    run_tar(&[
        "-xf".as_ref(),
        zip.as_os_str(),
        "-C".as_ref(),
        dest.as_os_str(),
    ])?;
    Ok(())
}

#[cfg(windows)]
fn is_reparse_point(meta: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
    meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(not(windows))]
fn is_reparse_point(_meta: &std::fs::Metadata) -> bool {
    false
}

/// Whether a symbolic link at `link` (inside `root`) pointing at `target` stays
/// inside `root`: the target is relative and, resolved lexically from the
/// folder of the link, never climbs above the root. Pure; the macOS builds link
/// `libllama.0.dylib -> libllama.0.6.0.dylib` and the like.
#[cfg_attr(not(unix), allow(dead_code))]
pub fn link_stays_inside(root: &Path, link: &Path, target: &Path) -> bool {
    use std::path::Component;
    if target.is_absolute() || target.as_os_str().is_empty() {
        return false;
    }
    let Ok(relative) = link.parent().unwrap_or(link).strip_prefix(root) else {
        return false;
    };
    let mut depth = relative.components().count() as i64;
    for part in target.components() {
        match part {
            Component::Normal(_) => depth += 1,
            Component::CurDir => {}
            Component::ParentDir => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}

/// Walks the unpacked tree without following links. On Windows any link or
/// junction fails the install; elsewhere a symbolic link is fine if it stays
/// inside the tree (`link_stays_inside`).
pub fn check_tree(root: &Path) -> Result<(), String> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).map_err(|e| e.to_string())? {
            let path = entry.map_err(|e| e.to_string())?.path();
            let meta = std::fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
            if meta.file_type().is_symlink() || is_reparse_point(&meta) {
                #[cfg(unix)]
                if meta.file_type().is_symlink() {
                    let target = std::fs::read_link(&path).map_err(|e| e.to_string())?;
                    if link_stays_inside(root, &path, &target) {
                        continue;
                    }
                }
                return Err(format!("a link in the archive: {}", path.display()));
            }
            if meta.is_dir() {
                stack.push(path);
            }
        }
    }
    Ok(())
}

/// The folder with the server executable: the root, or the only folder at most
/// two levels down that has one (`llama-b11623/` on a Mac, `build/bin/` in some
/// builds).
pub fn find_server_dir(root: &Path) -> Result<PathBuf, String> {
    fn search(dir: &Path, depth: u32, found: &mut Vec<PathBuf>) -> Result<(), String> {
        if dir.join(SERVER_EXE).is_file() {
            found.push(dir.to_path_buf());
            return Ok(());
        }
        if depth == 0 {
            return Ok(());
        }
        for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.is_dir() {
                search(&path, depth - 1, found)?;
            }
        }
        Ok(())
    }
    let mut found = Vec::new();
    search(root, 2, &mut found)?;
    match found.as_slice() {
        [one] => Ok(one.clone()),
        [] => Err(format!("{SERVER_EXE} is not in the archive")),
        _ => Err(format!("{SERVER_EXE} is in several folders of the archive")),
    }
}

/// Device names of `llama-server --list-devices` (`CUDA0`, `Vulkan0`): the
/// lines `  <Dev>: <name> (<total> MiB, <free> MiB free)` after
/// `Available devices:`.
pub fn parse_devices(output: &str) -> Vec<String> {
    let mut devices = Vec::new();
    let mut listing = false;
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Available devices") {
            listing = true;
            continue;
        }
        if !listing || trimmed.is_empty() {
            continue;
        }
        let Some((dev, rest)) = trimmed.split_once(": ") else {
            continue;
        };
        // "BLAS: Accelerate (0 MiB, 0 MiB free)" on a Mac is no video memory.
        let well_formed = !dev.is_empty()
            && dev.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && rest.contains("MiB")
            && !rest.contains("(0 MiB");
        if well_formed {
            devices.push(dev.to_string());
        }
    }
    devices
}

const VERSION_TIMEOUT: Duration = Duration::from_secs(15);
const DEVICES_TIMEOUT: Duration = Duration::from_secs(30);

/// Runs the engine from its (checked) folder: `--version` must name the
/// pinned build, `--list-devices` tells which video cards it sees.
pub fn engine_check(exe: &Path) -> Result<Vec<String>, SetupError> {
    let run = |arg: &str, timeout| {
        let mut command = Command::new(exe);
        command.arg(arg);
        if let Some(dir) = exe.parent() {
            command.current_dir(dir);
        }
        let output = run_with_timeout(command, timeout)
            .map_err(|e| SetupError::new(ErrorKind::EngineCheck, e))?;
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        if !output.status.success() {
            return Err(SetupError::new(
                ErrorKind::EngineCheck,
                format!("{arg}: {} {}", output.status, text.trim()),
            ));
        }
        Ok(text)
    };
    let version = run("--version", VERSION_TIMEOUT)?;
    if !version.contains(super::catalog::ENGINE_VERSION_MARK) {
        return Err(SetupError::new(
            ErrorKind::EngineCheck,
            format!("unexpected version: {}", version.trim()),
        ));
    }
    Ok(parse_devices(&run("--list-devices", DEVICES_TIMEOUT)?))
}

/// A short random suffix for temporary folders.
pub fn random_suffix() -> String {
    super::super::llama_server::random_api_key()[..8].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn only_https_on_the_listed_hosts() {
        for ok in [
            "https://github.com/ggml-org/llama.cpp/releases/download/b11623/x.zip",
            "https://release-assets.githubusercontent.com/github-production/x",
            "https://huggingface.co/mradermacher/x/resolve/abc/x.gguf",
            "https://cdn-lfs.huggingface.co/x",
            "https://us.aws.cdn.hf.co/xet-bridge-us/x",
            "https://cas-bridge.xethub.hf.co/x",
            "https://HuggingFace.co/x",
        ] {
            assert!(host_allowed(ok), "{ok}");
        }
        for bad in [
            "http://github.com/x",
            "https://evilhf.co/x",
            "https://huggingface.co.evil.com/x",
            "https://evilhuggingface.co/x",
            "https://hf.co/x",
            "https://githubusercontent.com/x",
            "https://.hf.co/x",
            "ftp://github.com/x",
            "not a url",
            "https://127.0.0.1/x",
        ] {
            assert!(!host_allowed(bad), "{bad}");
        }
    }

    /// A local server answering each connection by its path: `/<n>` redirects
    /// to `/<n-1>` (absolute or relative), `/0` is the file, `/evil` points
    /// off-site, `/gone` is 404, `/nohead` answers HEAD with 405.
    async fn redirect_server() -> (String, tokio::task::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let base_for_task = base.clone();
        let handle = tokio::spawn(async move {
            let mut seen = Vec::new();
            loop {
                let accepted =
                    tokio::time::timeout(Duration::from_millis(1500), listener.accept()).await;
                let Ok(Ok((mut sock, _))) = accepted else {
                    break;
                };
                let mut buf = Vec::new();
                let mut byte = [0u8; 1];
                while !buf.ends_with(b"\r\n\r\n") {
                    if sock.read_exact(&mut byte).await.is_err() {
                        break;
                    }
                    buf.push(byte[0]);
                }
                let head = String::from_utf8_lossy(&buf).into_owned();
                let mut parts = head.split_whitespace();
                let method = parts.next().unwrap_or("").to_string();
                let path = parts.next().unwrap_or("").to_string();
                seen.push(format!("{method} {path}"));
                let reply = match path.as_str() {
                    "/0" => "200 OK".to_string(),
                    "/gone" => "404 Not Found".to_string(),
                    "/evil" => "302 Found\r\nLocation: https://evil.example/x".to_string(),
                    "/plain" => "302 Found\r\nLocation: http://github.com/x".to_string(),
                    "/nohead" if method == "HEAD" => "405 Method Not Allowed".to_string(),
                    "/nohead" => "302 Found\r\nLocation: /0".to_string(),
                    p => {
                        let n: usize = p.trim_start_matches('/').parse().unwrap_or(1);
                        if n.is_multiple_of(2) {
                            format!("302 Found\r\nLocation: {base_for_task}/{}", n - 1)
                        } else {
                            format!("301 Moved\r\nLocation: /{}", n - 1)
                        }
                    }
                };
                let response =
                    format!("HTTP/1.1 {reply}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n");
                let _ = sock.write_all(response.as_bytes()).await;
                let _ = sock.shutdown().await;
            }
            seen
        });
        (base, handle)
    }

    fn local_or_listed(base: &str) -> impl Fn(&str) -> bool + Sync + '_ {
        move |url: &str| url.starts_with(base) || host_allowed(url)
    }

    #[tokio::test]
    async fn redirects_are_followed_and_checked_hop_by_hop() {
        let (base, server) = redirect_server().await;
        let allowed = local_or_listed(&base);
        // Absolute and relative hops: /3 → /2 → /1 → /0.
        assert_eq!(
            resolve_redirects_with(&format!("{base}/3"), &allowed)
                .await
                .unwrap(),
            format!("{base}/0")
        );
        // HEAD refused: GET is used for that hop.
        assert_eq!(
            resolve_redirects_with(&format!("{base}/nohead"), &allowed)
                .await
                .unwrap(),
            format!("{base}/0")
        );
        let e = resolve_redirects_with(&format!("{base}/evil"), &allowed)
            .await
            .unwrap_err();
        assert_eq!(e.kind, ErrorKind::Host);
        assert!(e.message.contains("evil.example"), "{}", e.message);
        let e = resolve_redirects_with(&format!("{base}/plain"), &allowed)
            .await
            .unwrap_err();
        assert_eq!(e.kind, ErrorKind::Host, "http");
        let e = resolve_redirects_with(&format!("{base}/gone"), &allowed)
            .await
            .unwrap_err();
        assert_eq!(e.kind, ErrorKind::SourceGone);
        let e = resolve_redirects_with(&format!("{base}/6"), &allowed)
            .await
            .unwrap_err();
        assert_eq!(e.kind, ErrorKind::Host, "more than 5 hops");
        assert!(e.message.contains("redirects"));
        // Exactly 5 hops are fine.
        assert!(resolve_redirects_with(&format!("{base}/5"), &allowed)
            .await
            .is_ok());

        let seen = server.await.unwrap();
        // The off-site Location was refused before a request (and no retry).
        assert_eq!(seen.iter().filter(|r| r.contains("evil")).count(), 1);
        assert!(seen.contains(&"GET /nohead".to_string()));
        assert!(seen.contains(&"HEAD /3".to_string()));
    }

    #[tokio::test]
    async fn a_disallowed_start_is_never_requested() {
        let e = resolve_redirects("http://github.com/x").await.unwrap_err();
        assert_eq!(e.kind, ErrorKind::Host);
        let e = resolve_redirects("https://huggingface.co.evil.com/x")
            .await
            .unwrap_err();
        assert_eq!(e.kind, ErrorKind::Host);
    }

    #[test]
    fn archive_names_are_checked() {
        assert!(check_archive_names("llama-server.exe\nggml-vulkan.dll\nsub/dir/a.dll\n").is_ok());
        assert!(check_archive_names("bin\\llama-server.exe\r\nbin/\r\n").is_ok());
        assert!(
            check_archive_names("a..b.dll\n").is_ok(),
            "dots inside a name"
        );
        for bad in [
            "../x",
            "ok.dll\n../x",
            "/x",
            "\\x",
            "C:\\x",
            "C:x",
            "a\\..\\b",
            "a/../b",
            "a:stream",
            "a\u{1}b",
            "",
        ] {
            assert!(check_archive_names(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_server_folder_is_the_root_or_its_only_subfolder() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        assert!(find_server_dir(root).is_err(), "none");
        std::fs::create_dir(root.join("a")).unwrap();
        std::fs::write(root.join("a").join(SERVER_EXE), b"x").unwrap();
        assert_eq!(find_server_dir(root).unwrap(), root.join("a"));
        std::fs::create_dir(root.join("b")).unwrap();
        std::fs::write(root.join("b").join(SERVER_EXE), b"x").unwrap();
        assert!(find_server_dir(root).is_err(), "two");
        std::fs::write(root.join(SERVER_EXE), b"x").unwrap();
        assert_eq!(find_server_dir(root).unwrap(), root, "the root wins");
        assert!(check_tree(root).is_ok());
    }

    #[test]
    #[cfg(windows)]
    fn a_junction_in_the_tree_fails() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("root");
        std::fs::create_dir_all(root.join("sub")).unwrap();
        let target = dir.path().join("outside");
        std::fs::create_dir(&target).unwrap();
        let status = Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(root.join("sub").join("link"))
            .arg(&target)
            .stdout(Stdio::null())
            .status()
            .unwrap();
        assert!(status.success());
        let err = check_tree(&root).unwrap_err();
        assert!(err.contains("link"), "{err}");
    }

    const B1_DEVICES: &str =
        "Available devices:\n  CUDA0: NVIDIA GeForce RTX 5070 Ti (16275 MiB, 15037 MiB free)\n";

    #[test]
    fn devices_are_parsed_from_the_listing() {
        assert_eq!(parse_devices(B1_DEVICES), ["CUDA0"]);
        // The Vulkan b11623 build on the RTX 5070 Ti (F2 ignore test 1).
        let vulkan = "Available devices:\n  Vulkan0: NVIDIA GeForce RTX 5070 Ti (15995 MiB, 15227 MiB free)\n\n";
        assert_eq!(parse_devices(vulkan), ["Vulkan0"]);
        let two = "load_backend: loaded Vulkan backend\nAvailable devices:\n  Vulkan0: AMD Radeon RX 6600 (8176 MiB, 7900 MiB free)\n  Vulkan1: Intel(R) UHD Graphics (16000 MiB, 15000 MiB free)\n";
        assert_eq!(parse_devices(two), ["Vulkan0", "Vulkan1"]);
        assert!(parse_devices("Available devices:\n").is_empty());
        assert!(parse_devices("Available devices:\n\n").is_empty());
        assert!(parse_devices("garbage\nCUDA0: x (1 MiB, 1 MiB free)").is_empty());
        assert!(parse_devices("Available devices:\n  error: something failed\n").is_empty());
    }

    #[test]
    fn transport_errors_map_to_kinds() {
        let kind = |text: &str| classify(&anyhow::anyhow!(text.to_string()));
        assert_eq!(
            kind("server returned HTTP 404 Not Found"),
            ErrorKind::SourceGone
        );
        assert_eq!(
            kind("server returned HTTP 403 Forbidden"),
            ErrorKind::SourceGone
        );
        assert_eq!(
            kind("server returned HTTP 503 Service Unavailable"),
            ErrorKind::Network
        );
        assert_eq!(
            kind("server returned HTTP 429 Too Many Requests"),
            ErrorKind::Network
        );
        assert_eq!(
            kind("Download verification failed for model x: file is corrupt. Please retry."),
            ErrorKind::Hash
        );
        assert_eq!(
            kind("server sent more than the expected 5 bytes"),
            ErrorKind::Hash
        );
        assert_eq!(
            kind("transfer stalled: no data for 60s"),
            ErrorKind::Network
        );
        assert_eq!(
            kind("download incomplete: expected 5 bytes, got 3"),
            ErrorKind::Network
        );
        assert_eq!(
            kind("server advertises 3 bytes, expected 5"),
            ErrorKind::SourceGone
        );
        assert_eq!(kind("something else"), ErrorKind::Other);
        let full = anyhow::Error::from(std::io::Error::from(std::io::ErrorKind::StorageFull));
        assert_eq!(classify(&full), ErrorKind::DiskFull);
        let full112 = anyhow::Error::from(std::io::Error::from_raw_os_error(112));
        assert_eq!(classify(&full112), ErrorKind::DiskFull);
    }

    #[test]
    fn volumes_and_free_space() {
        assert_eq!(volume_of(Path::new("relative")), "");
        if cfg!(windows) {
            assert_eq!(volume_of(Path::new(r"C:\Users\u\.cache")), "C:");
            assert_eq!(volume_of(Path::new(r"d:\x")), "D:");
            let dir = tempfile::tempdir().unwrap();
            assert!(free_bytes(&dir.path().join("not").join("yet")).unwrap() > 0);
        }
    }

    #[test]
    #[cfg(windows)]
    fn a_hung_helper_is_killed() {
        let mut command = Command::new("cmd");
        command.args(["/C", "ping", "-n", "30", "127.0.0.1"]);
        let started = Instant::now();
        let err = run_with_timeout(command, Duration::from_millis(500)).unwrap_err();
        assert!(err.contains("no answer"));
        assert!(started.elapsed() < Duration::from_secs(10));
        let mut ok = Command::new("cmd");
        ok.args(["/C", "echo", "hi"]);
        let out = run_with_timeout(ok, Duration::from_secs(10)).unwrap();
        assert!(String::from_utf8_lossy(&out.stdout).contains("hi"));
    }

    #[test]
    #[cfg(windows)]
    fn tar_unpacks_a_checked_zip() {
        // Builds a tiny zip with tar.exe itself (bsdtar writes zip with -a).
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        std::fs::create_dir(&src).unwrap();
        std::fs::write(src.join(SERVER_EXE), b"exe").unwrap();
        std::fs::write(src.join("ggml-vulkan.dll"), b"dll").unwrap();
        let zip = dir.path().join("e.zip");
        run_tar(&[
            "-a".as_ref(),
            "-cf".as_ref(),
            zip.as_os_str(),
            "-C".as_ref(),
            src.as_os_str(),
            SERVER_EXE.as_ref(),
            "ggml-vulkan.dll".as_ref(),
        ])
        .unwrap();
        let dest = dir.path().join("out");
        unpack(&zip, &dest).unwrap();
        assert_eq!(std::fs::read(dest.join(SERVER_EXE)).unwrap(), b"exe");
        assert_eq!(find_server_dir(&dest).unwrap(), dest);
        let broken = dir.path().join("broken.zip");
        std::fs::write(&broken, b"not a zip").unwrap();
        assert_eq!(
            unpack(&broken, &dir.path().join("o2")).unwrap_err().kind,
            ErrorKind::Tar
        );
    }

    // The real `--list-devices` and archive listing of b11623 for macOS arm64 (J2.1).
    const MAC_DEVICES: &str =
        "Available devices:\n  MTL0: Apple M3 Pro (13639 MiB, 13639 MiB free)\n  BLAS: Accelerate (0 MiB, 0 MiB free)\n";

    #[test]
    fn the_metal_device_is_found_and_blas_is_not_a_card() {
        assert_eq!(parse_devices(MAC_DEVICES), ["MTL0"]);
    }

    #[test]
    fn the_names_of_the_mac_archive_are_safe() {
        let listing = "llama-b11623/\nllama-b11623/llama-server\nllama-b11623/libllama.0.dylib\nllama-b11623/libggml-metal.0.26.0.dylib\nllama-b11623/LICENSE\n";
        assert!(check_archive_names(listing).is_ok());
    }

    #[test]
    fn a_link_must_stay_inside_the_tree() {
        let root = Path::new("/tmp/x");
        let ok =
            |link: &str, target: &str| link_stays_inside(root, &root.join(link), Path::new(target));
        assert!(ok("a/libllama.0.dylib", "libllama.0.6.0.dylib"));
        assert!(ok("a/b/l", "../c"));
        assert!(ok("a/b/l", "./c/d"));
        assert!(!ok("a/l", "../../etc/passwd"));
        assert!(!ok("l", "../outside"));
        assert!(!ok("a/l", "/etc/passwd"));
        assert!(!ok("a/l", ""));
        assert!(!link_stays_inside(
            root,
            Path::new("/other/l"),
            Path::new("x")
        ));
    }

    #[cfg(unix)]
    #[test]
    fn check_tree_allows_links_inside_and_refuses_the_rest() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("llama")).unwrap();
        std::fs::write(root.join("llama").join("libllama.0.6.0.dylib"), b"x").unwrap();
        symlink(
            "libllama.0.6.0.dylib",
            root.join("llama").join("libllama.0.dylib"),
        )
        .unwrap();
        assert!(check_tree(root).is_ok(), "a link to a neighbour");
        symlink("../../outside", root.join("llama").join("up")).unwrap();
        assert!(check_tree(root).is_err(), "a link that climbs out");
        std::fs::remove_file(root.join("llama").join("up")).unwrap();
        symlink("/etc/hosts", root.join("llama").join("abs")).unwrap();
        assert!(check_tree(root).is_err(), "an absolute link");
    }

    #[test]
    fn the_server_is_found_two_levels_down_but_not_three() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let touch = |p: PathBuf| {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, b"x").unwrap();
        };
        touch(root.join("a").join("b").join("c").join(SERVER_EXE));
        assert!(find_server_dir(root).is_err(), "three levels down");
        touch(root.join("llama-b11623").join(SERVER_EXE));
        assert_eq!(find_server_dir(root).unwrap(), root.join("llama-b11623"));
        std::fs::remove_dir_all(root.join("llama-b11623")).unwrap();
        std::fs::remove_dir_all(root.join("a")).unwrap();
        touch(root.join("build").join("bin").join(SERVER_EXE));
        assert_eq!(
            find_server_dir(root).unwrap(),
            root.join("build").join("bin")
        );
        touch(root.join("other").join(SERVER_EXE));
        assert!(find_server_dir(root).is_err(), "two folders");
    }

    #[cfg(unix)]
    #[test]
    fn free_space_is_read_from_the_nearest_existing_folder() {
        let dir = tempfile::tempdir().unwrap();
        let free = free_bytes(&dir.path().join("not").join("yet")).unwrap();
        assert!(free > 0);
        assert_eq!(volume_of(dir.path()), "");
    }
}
