//! The model list of the Fast / Quality slots (I): GGUF files next to the
//! current models, the setup's models in the Hugging Face cache and the files
//! picked before. Pure file-system logic, no Tauri (the command is in `mod.rs`).

use std::path::{Path, PathBuf};

use serde::Serialize;
use specta::Type;

use super::llama_server::{chat_format_for, ChatFormat};

/// Short names of the models we know (file name in lower case → name).
const KNOWN_MODELS: &[(&str, &str)] = &[
    ("translategemma-4b-it.q4_k_m.gguf", "TranslateGemma 4B"),
    ("translategemma-12b-it.q4_k_m.gguf", "TranslateGemma 12B"),
    ("qwen3.5-9b-q4_k_m.gguf", "Qwen3.5 9B"),
    ("gemma-4-26b-a4b-it-ud-q4_k_m.gguf", "Gemma 4 26B"),
];

#[derive(Serialize, Debug, Clone, PartialEq, Type)]
pub struct ModelEntry {
    pub path: String,
    /// Short name for the list and the popup.
    pub name: String,
    /// f64: a file over 4 GB does not fit the u32 that TypeScript bindings
    /// would get from specta for u64.
    pub size_bytes: f64,
    pub format: ChatFormat,
}

/// The name shown for a model file: the known short name, or the file name
/// without `.gguf`.
pub fn display_name(model_path: &str) -> String {
    let file = Path::new(model_path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let lower = file.to_lowercase();
    if let Some((_, name)) = KNOWN_MODELS.iter().find(|(f, _)| *f == lower) {
        return (*name).to_string();
    }
    match lower.strip_suffix(".gguf") {
        Some(_) => file[..file.len() - ".gguf".len()].to_string(),
        None => file,
    }
}

/// A file the list may offer: a `.gguf` that is not a vision projector or a
/// download in progress.
fn is_model_file(path: &Path) -> bool {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    name.ends_with(".gguf") && !name.starts_with("mmproj") && path.is_file()
}

/// The `.gguf` files directly in `dir` (no subfolders), sorted by name.
fn gguf_in(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| is_model_file(p))
        .collect();
    files.sort();
    files
}

/// The model list: files in the folders of `slot_paths`, then the `extra`
/// files (catalog models in the HF cache, earlier picks) that exist. No
/// repeats (canonical paths), no `mmproj-*`, no `.partial`, nothing missing.
pub fn list_models(slot_paths: &[&str], extra: &[PathBuf]) -> Vec<ModelEntry> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    for path in slot_paths.iter().filter(|p| !p.is_empty()) {
        let path = Path::new(path);
        if is_model_file(path) {
            candidates.push(path.to_path_buf());
        }
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            candidates.extend(gguf_in(dir));
        }
    }
    candidates.extend(extra.iter().filter(|p| is_model_file(p)).cloned());

    let mut seen: Vec<PathBuf> = Vec::new();
    let mut out = Vec::new();
    for path in candidates {
        let canonical = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
        if seen.contains(&canonical) {
            continue;
        }
        seen.push(canonical);
        let text = path.display().to_string();
        out.push(ModelEntry {
            name: display_name(&text),
            size_bytes: std::fs::metadata(&path).map_or(0, |m| m.len()) as f64,
            format: chat_format_for(&path),
            path: text,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(dir: &Path, name: &str, bytes: usize) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, vec![0u8; bytes]).unwrap();
        p
    }

    #[test]
    fn known_models_get_short_names_and_others_their_file_name() {
        assert_eq!(
            display_name("/models/translategemma-4b-it.Q4_K_M.gguf"),
            "TranslateGemma 4B"
        );
        assert_eq!(display_name("/m/Qwen3.5-9B-Q4_K_M.gguf"), "Qwen3.5 9B");
        assert_eq!(
            display_name("/m/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"),
            "Gemma 4 26B"
        );
        assert_eq!(
            display_name("/m/translategemma-12b-it.Q4_K_M.gguf"),
            "TranslateGemma 12B"
        );
        assert_eq!(
            display_name("/m/Mistral-Small-3.2.Q5_K_M.GGUF"),
            "Mistral-Small-3.2.Q5_K_M"
        );
        assert_eq!(display_name("/m/model.bin"), "model.bin");
        assert_eq!(display_name(""), "");
    }

    #[test]
    fn the_list_has_the_slot_folders_and_the_extra_files_once_each() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let fast = touch(a.path(), "translategemma-4b-it.Q4_K_M.gguf", 10);
        let qwen = touch(a.path(), "Qwen3.5-9B-Q4_K_M.gguf", 20);
        touch(a.path(), "mmproj-Qwen3.5-9B-F16.gguf", 5);
        touch(a.path(), "abc.gguf.partial", 5);
        touch(a.path(), "notes.txt", 5);
        std::fs::create_dir(a.path().join("sub")).unwrap();
        touch(&a.path().join("sub"), "deep.gguf", 5);
        let other = touch(b.path(), "custom.gguf", 30);
        let missing = b.path().join("gone.gguf");

        let fast_text = fast.display().to_string();
        let list = list_models(
            &[&fast_text, &fast_text, ""],
            &[other.clone(), qwen.clone(), missing, fast.clone()],
        );
        let names: Vec<&str> = list.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(
            names,
            ["TranslateGemma 4B", "Qwen3.5 9B", "custom"],
            "{list:?}"
        );
        assert_eq!(list[0].format, ChatFormat::TranslateGemma);
        assert_eq!(list[1].format, ChatFormat::Chat);
        assert_eq!(list[1].size_bytes, 20.0);
        assert_eq!(list[2].path, other.display().to_string());
    }

    #[test]
    fn a_missing_slot_file_still_lists_its_folder() {
        let a = tempfile::tempdir().unwrap();
        touch(a.path(), "x.gguf", 1);
        let gone = a.path().join("gone.gguf").display().to_string();
        let list = list_models(&[&gone], &[]);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "x");
        assert!(list_models(&["", "relative-and-missing.gguf"], &[]).is_empty());
    }
}
