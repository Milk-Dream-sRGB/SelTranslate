//! SelTranslate: transcript library of the Audio file tab (stage H).
//!
//! Plain `.md` files in one folder, no database and no index: the list is a
//! scan of the folder. A file is named `YYYY-MM-DD HH-mm Title.md` (local time
//! of the end of the transcription) and starts with `# Title`, a source line
//! and the text in paragraphs. Commands take a file name, never a path, and
//! check that it stays inside the library folder.

use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use chrono::{DateTime, Local, NaiveDateTime};
use log::warn;
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::{AppHandle, Emitter};
use tauri_plugin_store::StoreExt;

pub const EVENT_CHANGED: &str = "library-changed";

/// Own key in Handy's settings store, next to the translation's `seltranslate`.
const STORE_KEY: &str = "seltranslate_library";
const DEFAULT_SUBDIR: &str = "library";

/// Leading filler words skipped by the auto title (compared in lower case,
/// without trailing punctuation), at most `MAX_FILLERS` of them.
const FILLERS: &[&str] = &[
    "ну",
    "так",
    "э",
    "ээ",
    "эээ",
    "эм",
    "значит",
    "ладно",
    "well",
    "so",
    "um",
    "uh",
    "okay",
    "ok",
];
const MAX_FILLERS: usize = 3;
const TITLE_WORDS: usize = 8;
const TITLE_CHARS: usize = 60;
/// A renamed title; the whole name (date, title, ` (99)`, `.md`) stays ≤ 150.
const MAX_TITLE_CHARS: usize = 120;
/// Chunker pieces (20–30 s each) per paragraph.
const CHUNKS_PER_PARAGRAPH: usize = 4;
const DATE_FORMAT: &str = "%Y-%m-%d %H-%M";
/// `YYYY-MM-DD HH-mm ` before the title.
const DATE_PREFIX_LEN: usize = 17;
const FALLBACK_TITLE: &str = "Transcript";

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq, Type)]
#[serde(default)]
pub struct LibrarySettings {
    /// Off by default: nothing is written until the user turns it on.
    pub enabled: bool,
    /// Empty = `library` in the app data folder.
    pub dir: String,
}

/// The settings plus the folder they resolve to (for the UI).
#[derive(Serialize, Debug, Clone, Type)]
pub struct LibraryInfo {
    pub enabled: bool,
    pub dir: String,
    pub resolved_dir: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Type)]
pub struct LibraryEntry {
    pub file_name: String,
    pub title: String,
    /// Local time without a zone, `YYYY-MM-DDTHH:mm:00` (the name's date, or
    /// the file's modification time for a name without one).
    pub date: String,
    pub size: u64,
}

pub fn load_settings(app: &AppHandle) -> LibrarySettings {
    let Ok(store) = app.store(crate::portable::store_path(
        crate::settings::SETTINGS_STORE_PATH,
    )) else {
        return LibrarySettings::default();
    };
    store
        .get(STORE_KEY)
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default()
}

pub fn save_settings(app: &AppHandle, settings: &LibrarySettings) -> Result<(), String> {
    let store = app
        .store(crate::portable::store_path(
            crate::settings::SETTINGS_STORE_PATH,
        ))
        .map_err(|e| format!("Failed to open the settings store: {e}"))?;
    let value = serde_json::to_value(settings).map_err(|e| e.to_string())?;
    store.set(STORE_KEY, value);
    Ok(())
}

/// The library folder; not created here (only on the first save).
pub fn resolve_dir(app: &AppHandle, settings: &LibrarySettings) -> Result<PathBuf, String> {
    if !settings.dir.trim().is_empty() {
        return Ok(PathBuf::from(settings.dir.trim()));
    }
    crate::portable::app_data_dir(app)
        .map(|d| d.join(DEFAULT_SUBDIR))
        .map_err(|e| format!("Failed to get the app data folder: {e}"))
}

pub fn info(app: &AppHandle) -> LibraryInfo {
    let settings = load_settings(app);
    let resolved_dir = resolve_dir(app, &settings)
        .map(|d| d.display().to_string())
        .unwrap_or_default();
    LibraryInfo {
        enabled: settings.enabled,
        dir: settings.dir,
        resolved_dir,
    }
}

pub fn notify_changed(app: &AppHandle) {
    let _ = app.emit(EVENT_CHANGED, ());
}

/// What saving a finished transcript gave: the file name, or the error.
pub struct SaveResult {
    pub saved_as: Option<String>,
    pub error: Option<String>,
}

/// Saves a finished transcript when the library is on. `chunks`: the
/// non-empty chunk texts. Never panics; a failure is only reported.
pub fn save_transcript(
    app: &AppHandle,
    source: &Path,
    audio_secs: f64,
    chunks: &[String],
) -> SaveResult {
    let settings = load_settings(app);
    if !settings.enabled || chunks.iter().all(|c| c.trim().is_empty()) {
        return SaveResult {
            saved_as: None,
            error: None,
        };
    }
    let result = resolve_dir(app, &settings).and_then(|dir| {
        write_transcript(&dir, source, audio_secs, chunks, Local::now().naive_local())
    });
    match result {
        Ok(name) => {
            log::info!("Library: saved {name}");
            notify_changed(app);
            SaveResult {
                saved_as: Some(name),
                error: None,
            }
        }
        Err(e) => {
            warn!("Library: {e}");
            SaveResult {
                saved_as: None,
                error: Some(e),
            }
        }
    }
}

fn write_transcript(
    dir: &Path,
    source: &Path,
    audio_secs: f64,
    chunks: &[String],
    now: NaiveDateTime,
) -> Result<String, String> {
    let source_name = source
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let source_stem = source
        .file_stem()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let title = auto_title(&chunks.join(" "), &source_stem);
    let content = render(&title, &source_name, audio_secs, now, chunks);
    fs::create_dir_all(dir)
        .map_err(|e| format!("Failed to create the library folder {}: {e}", dir.display()))?;
    create_unique(dir, &file_name(now, &title), content.as_bytes())
}

/// Creates `name` (or `name (2)`, `(3)`, …) without overwriting anything.
fn create_unique(dir: &Path, name: &str, content: &[u8]) -> Result<String, String> {
    for n in 1..=99 {
        let candidate = numbered(name, n);
        let path = dir.join(&candidate);
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                if let Err(e) = file.write_all(content).and_then(|_| file.sync_all()) {
                    drop(file);
                    let _ = fs::remove_file(&path);
                    return Err(format!("Failed to write {}: {e}", path.display()));
                }
                return Ok(candidate);
            }
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("Failed to create {}: {e}", path.display())),
        }
    }
    Err(format!("Too many files named {name}"))
}

/// `a.md` → `a.md`, `a (2).md`, `a (3).md`…
fn numbered(name: &str, n: u32) -> String {
    if n <= 1 {
        return name.to_string();
    }
    match name.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem} ({n}).{ext}"),
        None => format!("{name} ({n})"),
    }
}

/// Title from the first words: leading fillers skipped (at most three, and
/// only if words remain), up to eight words and 60 characters cut at a word
/// boundary, trailing punctuation removed. Empty → the audio file name.
pub fn auto_title(text: &str, source_stem: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let fillers = words
        .iter()
        .take(MAX_FILLERS)
        .take_while(|w| is_filler(w))
        .count();
    let rest = if fillers < words.len() {
        &words[fillers..]
    } else {
        &words[..]
    };
    let mut title = String::new();
    for word in rest.iter().take(TITLE_WORDS) {
        let next = if title.is_empty() {
            word.to_string()
        } else {
            format!("{title} {word}")
        };
        if next.chars().count() > TITLE_CHARS {
            if title.is_empty() {
                title = word.chars().take(TITLE_CHARS).collect();
            }
            break;
        }
        title = next;
    }
    let title = sanitize_title(trim_end_punct(&title));
    if !title.is_empty() {
        return title;
    }
    let fallback = sanitize_title(source_stem);
    if fallback.is_empty() {
        FALLBACK_TITLE.to_string()
    } else {
        fallback
    }
}

fn is_filler(word: &str) -> bool {
    let bare = trim_end_punct(word).to_lowercase();
    FILLERS.contains(&bare.as_str())
}

fn trim_end_punct(s: &str) -> &str {
    s.trim_end_matches(|c: char| {
        c.is_ascii_punctuation() || "…—–«»“”„".contains(c) || c.is_whitespace()
    })
}

/// Makes a title safe for a Windows file name: forbidden `<>:"/\|?*` and
/// control characters become spaces, runs of dots become `…` (so a name never
/// holds `..`), spaces are collapsed, trailing dots and spaces dropped, the
/// length is capped and a reserved device name gets a leading `_`. Unicode is kept.
pub fn sanitize_title(s: &str) -> String {
    let replaced: String = s
        .chars()
        .map(|c| {
            if c.is_control() || "<>:\"/\\|?*".contains(c) {
                ' '
            } else {
                c
            }
        })
        .collect();
    let mut out = String::with_capacity(replaced.len());
    let collapsed = replaced.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = collapsed.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '.' && chars.peek() == Some(&'.') {
            while chars.next_if_eq(&'.').is_some() {}
            out.push('…');
        } else {
            out.push(c);
        }
    }
    let out: String = out.chars().take(MAX_TITLE_CHARS).collect();
    let out = out.trim().trim_end_matches(['.', ' ']).to_string();
    if is_reserved(&out) {
        format!("_{out}")
    } else {
        out
    }
}

fn is_reserved(name: &str) -> bool {
    let base = name.split('.').next().unwrap_or("").trim().to_uppercase();
    matches!(base.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || ((base.starts_with("COM") || base.starts_with("LPT"))
            && base.len() == 4
            && matches!(base.as_bytes()[3], b'1'..=b'9'))
}

pub fn file_name(date: NaiveDateTime, title: &str) -> String {
    format!("{} {title}.md", date.format(DATE_FORMAT))
}

/// `YYYY-MM-DD HH-mm Title.md` → (date, title); `None` for other names.
pub fn parse_file_name(name: &str) -> Option<(NaiveDateTime, String)> {
    let stem = strip_md(name)?;
    let prefix = stem.get(..DATE_PREFIX_LEN)?;
    let date = NaiveDateTime::parse_from_str(prefix.trim_end(), DATE_FORMAT).ok()?;
    if !prefix.ends_with(' ') {
        return None;
    }
    let title = stem[DATE_PREFIX_LEN..].trim();
    (!title.is_empty()).then(|| (date, title.to_string()))
}

fn strip_md(name: &str) -> Option<&str> {
    let split = name.len().checked_sub(3)?;
    let ext = name.get(split..)?;
    ext.eq_ignore_ascii_case(".md").then(|| &name[..split])
}

/// Paragraphs of `CHUNKS_PER_PARAGRAPH` chunks, separated by a blank line.
pub fn paragraphs(chunks: &[String]) -> String {
    let texts: Vec<&str> = chunks
        .iter()
        .map(|c| c.trim())
        .filter(|c| !c.is_empty())
        .collect();
    texts
        .chunks(CHUNKS_PER_PARAGRAPH)
        .map(|p| p.join(" "))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn format_duration(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    let (h, m, s) = (s / 3600, (s % 3600) / 60, s % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

pub fn render(
    title: &str,
    source_name: &str,
    audio_secs: f64,
    date: NaiveDateTime,
    chunks: &[String],
) -> String {
    format!(
        "# {title}\n\n*Source: {source_name} · {} · {}*\n\n{}\n",
        format_duration(audio_secs),
        date.format("%Y-%m-%d %H:%M"),
        paragraphs(chunks)
    )
}

/// Checks a file name from the UI: a bare `.md` name, no separators, no
/// `..`, nothing that could point outside the library folder.
pub fn validate_file_name(name: &str) -> Result<(), String> {
    let bad = name.is_empty()
        || name.contains(['/', '\\', ':', '\0'])
        || name.contains("..")
        || name.trim() != name
        || strip_md(name).is_none_or(|stem| stem.trim().is_empty());
    if bad {
        Err(format!("Invalid library file name: {name}"))
    } else {
        Ok(())
    }
}

/// Full path of an existing library file, checked to lie in `dir`.
pub fn existing_path(dir: &Path, name: &str) -> Result<PathBuf, String> {
    validate_file_name(name)?;
    let path = dir.join(name);
    let (Ok(canon), Ok(canon_dir)) = (path.canonicalize(), dir.canonicalize()) else {
        return Err(format!("File not found in the library: {name}"));
    };
    if canon.parent() != Some(canon_dir.as_path()) || !canon.is_file() {
        return Err(format!("File not found in the library: {name}"));
    }
    Ok(path)
}

/// `.md` files of `dir`, newest first; a missing folder is an empty list.
pub fn list(dir: &Path) -> Vec<LibraryEntry> {
    let Ok(read) = fs::read_dir(dir) else {
        return Vec::new();
    };
    // (date of the name, modification time, entry): files of the same minute
    // (`Title.md`, `Title (2).md`) go newest first by their modification time.
    let mut entries: Vec<(NaiveDateTime, Option<SystemTime>, LibraryEntry)> = read
        .flatten()
        .filter_map(|e| {
            let meta = e.metadata().ok().filter(|m| m.is_file())?;
            let name = e.file_name().to_str()?.to_string();
            validate_file_name(&name).ok()?;
            let modified = meta.modified().ok();
            let (date, title) = parse_file_name(&name).unwrap_or_else(|| {
                let date = modified
                    .map(|t| DateTime::<Local>::from(t).naive_local())
                    .unwrap_or_default();
                (date, strip_md(&name).unwrap_or(&name).to_string())
            });
            Some((
                date,
                modified,
                LibraryEntry {
                    date: date.format("%Y-%m-%dT%H:%M:00").to_string(),
                    file_name: name,
                    title,
                    size: meta.len(),
                },
            ))
        })
        .collect();
    entries.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| b.2.file_name.cmp(&a.2.file_name))
    });
    entries.into_iter().map(|(_, _, e)| e).collect()
}

/// New file name for a rename: the date stays, the title changes.
fn renamed(name: &str, title: &str) -> String {
    match parse_file_name(name) {
        Some((date, _)) => file_name(date, title),
        None => format!("{title}.md"),
    }
}

/// The content with `# old` as the first line replaced by `# new`; `None`
/// when the first line is anything else (the user may have edited the file).
pub fn retitle_content(content: &str, old: &str, new: &str) -> Option<String> {
    let body = content.strip_prefix('\u{feff}').unwrap_or(content);
    let (first, rest) = match body.find('\n') {
        Some(i) => (&body[..i], &body[i..]),
        None => (body, ""),
    };
    (first.trim_end_matches('\r') == format!("# {old}")).then(|| {
        let cr = if first.ends_with('\r') { "\r" } else { "" };
        format!("# {new}{cr}{rest}")
    })
}

/// Renames a library file to a new title; returns the new file name.
pub fn rename(dir: &Path, name: &str, new_title: &str) -> Result<String, String> {
    let path = existing_path(dir, name)?;
    let title = sanitize_title(new_title);
    if title.is_empty() {
        return Err("The title is empty".to_string());
    }
    let old_title = parse_file_name(name)
        .map(|(_, t)| t)
        .unwrap_or_else(|| strip_md(name).unwrap_or(name).to_string());
    let target = renamed(name, &title);

    // The first line follows the title only while it still is `# <old title>`.
    if let Ok(content) = fs::read_to_string(&path) {
        if let Some(updated) = retitle_content(&content, &old_title, &title) {
            fs::write(&path, updated).map_err(|e| format!("Failed to update {name}: {e}"))?;
        }
    }
    if target == name {
        return Ok(target);
    }
    // A change of letter case only is the same file on Windows: no suffix.
    let new_name = if target.to_lowercase() == name.to_lowercase() {
        target
    } else {
        (1..=99)
            .map(|n| numbered(&target, n))
            .find(|c| !dir.join(c).exists())
            .ok_or_else(|| format!("Too many files named {target}"))?
    };
    fs::rename(&path, dir.join(&new_name)).map_err(|e| format!("Failed to rename {name}: {e}"))?;
    Ok(new_name)
}

pub fn delete(dir: &Path, name: &str) -> Result<(), String> {
    let path = existing_path(dir, name)?;
    fs::remove_file(&path).map_err(|e| format!("Failed to delete {name}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn at(h: u32, m: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(2026, 10, 17)
            .unwrap()
            .and_hms_opt(h, m, 0)
            .unwrap()
    }

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn auto_title_takes_first_eight_words() {
        let t = auto_title(
            "Созвон по релизу версии два, обсуждаем сроки и риски проекта сегодня",
            "call",
        );
        assert_eq!(t, "Созвон по релизу версии два, обсуждаем сроки и");
    }

    #[test]
    fn auto_title_skips_up_to_three_leading_fillers() {
        assert_eq!(
            auto_title("Ну, так, значит, начнём планёрку.", "x"),
            "начнём планёрку"
        );
        assert_eq!(auto_title("Um, okay so let us start.", "x"), "let us start");
        // Only three are skipped, the fourth stays.
        assert_eq!(auto_title("ну ну ну ну вот", "x"), "ну вот");
    }

    #[test]
    fn auto_title_of_fillers_only_keeps_them() {
        assert_eq!(auto_title("Ну, так.", "x"), "Ну, так");
    }

    #[test]
    fn auto_title_of_empty_text_is_the_source_name() {
        assert_eq!(auto_title("  ", "meeting: 17.10"), "meeting 17.10");
        assert_eq!(auto_title("…", ""), "Transcript");
    }

    #[test]
    fn auto_title_cuts_at_a_word_boundary_and_a_long_word_hard() {
        let t = auto_title(&"слово ".repeat(20), "x");
        assert_eq!(t, "слово слово слово слово слово слово слово слово");
        let t = auto_title(&"длинноеслово ".repeat(8), "x");
        assert!(t.chars().count() <= 60, "{t}");
        assert!(!t.ends_with(' '));
        let long = "а".repeat(100);
        assert_eq!(auto_title(&long, "x").chars().count(), 60);
    }

    #[test]
    fn auto_title_keeps_unicode_and_first_letter_case() {
        assert_eq!(auto_title("привет 👋 мир!", "x"), "привет 👋 мир");
    }

    #[test]
    fn sanitizer_replaces_forbidden_characters() {
        assert_eq!(
            sanitize_title("a<b>c:d\"e/f\\g|h?i*j"),
            "a b c d e f g h i j"
        );
        assert_eq!(sanitize_title("tab\there\nline"), "tab here line");
        assert_eq!(sanitize_title("  many   spaces  "), "many spaces");
    }

    #[test]
    fn sanitizer_drops_trailing_dots_and_dot_runs() {
        assert_eq!(sanitize_title("end. . ."), "end");
        assert_eq!(sanitize_title("wait... what"), "wait… what");
        assert!(!sanitize_title("a..b").contains(".."));
    }

    #[test]
    fn sanitizer_avoids_reserved_names() {
        for name in ["CON", "prn", "Aux", "NUL", "COM1", "lpt9", "con.txt"] {
            assert!(!is_reserved(&sanitize_title(name)), "{name}");
        }
        assert_eq!(sanitize_title("COM0"), "COM0");
        assert_eq!(sanitize_title("Console"), "Console");
    }

    #[test]
    fn sanitizer_caps_the_length() {
        let t = sanitize_title(&"я".repeat(500));
        assert_eq!(t.chars().count(), MAX_TITLE_CHARS);
        let name = numbered(&file_name(at(1, 2), &t), 99);
        assert!(name.chars().count() <= 150, "{}", name.chars().count());
    }

    #[test]
    fn file_name_round_trips() {
        let name = file_name(at(14, 32), "Созвон по релизу");
        assert_eq!(name, "2026-10-17 14-32 Созвон по релизу.md");
        assert_eq!(
            parse_file_name(&name),
            Some((at(14, 32), "Созвон по релизу".to_string()))
        );
        assert_eq!(
            parse_file_name("2026-10-17 14-32 Title (2).MD"),
            Some((at(14, 32), "Title (2)".to_string()))
        );
    }

    #[test]
    fn foreign_names_do_not_parse() {
        for name in [
            "notes.md",
            "2026-10-17 14-32 .md",
            "2026-10-17 14-32.md",
            "2026-13-17 14-32 x.md",
            "2026-10-17 14-32 x.txt",
            "2026-10-17_14-32 x.md",
            "я.md",
        ] {
            assert_eq!(parse_file_name(name), None, "{name}");
        }
    }

    #[test]
    fn validation_rejects_paths_and_other_extensions() {
        for bad in [
            "",
            "..",
            "../x.md",
            "a/b.md",
            "a\\b.md",
            "C:x.md",
            "a..b.md",
            "x.txt",
            "x.md.txt",
            ".md",
            " x.md",
            "x.md:stream",
        ] {
            assert!(validate_file_name(bad).is_err(), "{bad:?}");
        }
        assert!(validate_file_name("2026-10-17 14-32 Созвон.md").is_ok());
        assert!(validate_file_name("notes.MD").is_ok());
    }

    #[test]
    fn paragraphs_group_four_chunks() {
        let chunks = strings(&["a", "b", " ", "c", "d", "e", "f", "g", "h", "i"]);
        assert_eq!(paragraphs(&chunks), "a b c d\n\ne f g h\n\ni");
        assert_eq!(paragraphs(&[]), "");
    }

    #[test]
    fn render_has_heading_source_and_text() {
        let text = render(
            "T",
            "call.mp3",
            5025.0,
            at(14, 32),
            &strings(&["one", "two"]),
        );
        assert_eq!(
            text,
            "# T\n\n*Source: call.mp3 · 1:23:45 · 2026-10-17 14:32*\n\none two\n"
        );
        assert!(render("T", "a.wav", 65.4, at(1, 2), &[]).contains("· 1:05 ·"));
    }

    #[test]
    fn saving_never_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        let lib = dir.path().join("library");
        let chunks = strings(&["Созвон по релизу"]);
        let src = Path::new("C:/rec/call.mp3");
        let a = write_transcript(&lib, src, 10.0, &chunks, at(14, 32)).unwrap();
        let b = write_transcript(&lib, src, 10.0, &chunks, at(14, 32)).unwrap();
        let c = write_transcript(&lib, src, 10.0, &chunks, at(14, 32)).unwrap();
        assert_eq!(a, "2026-10-17 14-32 Созвон по релизу.md");
        assert_eq!(b, "2026-10-17 14-32 Созвон по релизу (2).md");
        assert_eq!(c, "2026-10-17 14-32 Созвон по релизу (3).md");
        let content = fs::read_to_string(lib.join(&a)).unwrap();
        assert!(content.starts_with("# Созвон по релизу\n\n*Source: call.mp3 · 0:10 · "));
    }

    #[test]
    fn saving_to_an_unusable_folder_reports_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("file");
        fs::write(&file, "x").unwrap();
        let chunks = strings(&["x"]);
        let result = write_transcript(
            &file.join("lib"),
            Path::new("a.mp3"),
            1.0,
            &chunks,
            at(1, 1),
        );
        assert!(result.is_err());
    }

    #[test]
    fn list_is_newest_first_and_only_md() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "2026-10-17 09-00 Morning.md",
            "2026-10-17 14-32 Afternoon.md",
            "2026-10-16 23-59 Yesterday.md",
            "notes.md",
            "desktop.ini",
            "audio.txt",
        ] {
            fs::write(dir.path().join(name), "x").unwrap();
        }
        fs::create_dir(dir.path().join("folder.md")).unwrap();
        let entries = list(dir.path());
        let names: Vec<&str> = entries.iter().map(|e| e.file_name.as_str()).collect();
        // notes.md is dated by its modification time (now): the newest.
        assert_eq!(
            names,
            [
                "notes.md",
                "2026-10-17 14-32 Afternoon.md",
                "2026-10-17 09-00 Morning.md",
                "2026-10-16 23-59 Yesterday.md"
            ]
        );
        assert_eq!(entries[0].title, "notes");
        assert_eq!(entries[1].title, "Afternoon");
        assert_eq!(entries[1].date, "2026-10-17T14:32:00");
        assert_eq!(entries[1].size, 1);
        assert!(list(&dir.path().join("missing")).is_empty());
    }

    #[test]
    fn same_minute_files_are_newest_first() {
        let dir = tempfile::tempdir().unwrap();
        let chunks = strings(&["Title"]);
        let src = Path::new("a.mp3");
        let first = write_transcript(dir.path(), src, 1.0, &chunks, at(14, 32)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        let second = write_transcript(dir.path(), src, 1.0, &chunks, at(14, 32)).unwrap();
        let names: Vec<String> = list(dir.path()).into_iter().map(|e| e.file_name).collect();
        assert_eq!(names, [second, first]);
    }

    #[test]
    fn retitle_touches_only_a_matching_first_line() {
        assert_eq!(
            retitle_content("# Old\n\ntext", "Old", "New").as_deref(),
            Some("# New\n\ntext")
        );
        assert_eq!(
            retitle_content("# Old\r\n\r\ntext", "Old", "New").as_deref(),
            Some("# New\r\n\r\ntext")
        );
        assert_eq!(
            retitle_content("\u{feff}# Old", "Old", "New").as_deref(),
            Some("# New")
        );
        assert_eq!(retitle_content("# Old, edited\n", "Old", "New"), None);
        assert_eq!(retitle_content("Old\n", "Old", "New"), None);
    }

    #[test]
    fn rename_changes_title_and_heading_but_keeps_the_date() {
        let dir = tempfile::tempdir().unwrap();
        let name = "2026-10-17 14-32 Old title.md";
        fs::write(dir.path().join(name), "# Old title\n\nbody\n").unwrap();
        let new = rename(dir.path(), name, "New: title?").unwrap();
        assert_eq!(new, "2026-10-17 14-32 New title.md");
        assert!(!dir.path().join(name).exists());
        assert_eq!(
            fs::read_to_string(dir.path().join(&new)).unwrap(),
            "# New title\n\nbody\n"
        );
    }

    #[test]
    fn rename_keeps_an_edited_file_and_avoids_collisions() {
        let dir = tempfile::tempdir().unwrap();
        let a = "2026-10-17 14-32 A.md";
        fs::write(dir.path().join(a), "# My own heading\n").unwrap();
        fs::write(dir.path().join("2026-10-17 14-32 B.md"), "# B\n").unwrap();
        let new = rename(dir.path(), a, "B").unwrap();
        assert_eq!(new, "2026-10-17 14-32 B (2).md");
        assert_eq!(
            fs::read_to_string(dir.path().join(&new)).unwrap(),
            "# My own heading\n"
        );
        assert_eq!(
            fs::read_to_string(dir.path().join("2026-10-17 14-32 B.md")).unwrap(),
            "# B\n"
        );
    }

    #[test]
    fn rename_of_case_only_and_foreign_names() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("2026-10-17 14-32 abc.md"), "# abc\n").unwrap();
        let new = rename(dir.path(), "2026-10-17 14-32 abc.md", "ABC").unwrap();
        assert_eq!(new, "2026-10-17 14-32 ABC.md");
        assert_eq!(
            fs::read_to_string(dir.path().join(&new)).unwrap(),
            "# ABC\n"
        );

        fs::write(dir.path().join("notes.md"), "# notes\n").unwrap();
        assert_eq!(rename(dir.path(), "notes.md", "Ideas").unwrap(), "Ideas.md");
    }

    #[test]
    fn rename_and_delete_reject_bad_input() {
        let dir = tempfile::tempdir().unwrap();
        let name = "2026-10-17 14-32 A.md";
        fs::write(dir.path().join(name), "# A\n").unwrap();
        assert!(rename(dir.path(), name, " ?* ").is_err());
        assert!(rename(dir.path(), "missing.md", "B").is_err());
        assert!(delete(dir.path(), "../A.md").is_err());
        assert!(delete(dir.path(), "missing.md").is_err());
        delete(dir.path(), name).unwrap();
        assert!(!dir.path().join(name).exists());
    }

    #[test]
    fn existing_path_stays_inside_the_folder() {
        let root = tempfile::tempdir().unwrap();
        let lib = root.path().join("lib");
        fs::create_dir(&lib).unwrap();
        fs::write(root.path().join("outside.md"), "x").unwrap();
        fs::write(lib.join("inside.md"), "x").unwrap();
        assert!(existing_path(&lib, "inside.md").is_ok());
        assert!(existing_path(&lib, "outside.md").is_err());
        assert!(existing_path(&lib, "..\\outside.md").is_err());
    }
}
