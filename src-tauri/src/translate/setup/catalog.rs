//! The pinned catalog of what the setup downloads (DESIGN F.8): the llama.cpp
//! engine of build b11623 and the TranslateGemma GGUF files at a fixed commit.
//! Sizes and SHA-256 are the trust anchor; nothing here is fetched at run time.

use std::path::{Path, PathBuf};

use serde::Serialize;
use specta::Type;

/// The llama.cpp build the translation was verified with (B1–G3).
pub const ENGINE_BUILD: &str = "b11623";
/// `llama-server --version` prints this for the pinned build.
pub const ENGINE_VERSION_MARK: &str = "build 11623";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ItemKind {
    Engine,
    Model,
}

/// Which engine a zip belongs to; the CUDA engine needs two zips.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum EngineFlavor {
    Vulkan,
    Cuda,
    /// macOS, Apple Silicon: Metal is built into the official build.
    Metal,
}

impl EngineFlavor {
    /// Folder under `translate\engine\` the engine is unpacked into.
    pub fn dir_name(self) -> &'static str {
        match self {
            EngineFlavor::Vulkan => "vulkan-b11623",
            EngineFlavor::Cuda => "cuda13-b11623",
            EngineFlavor::Metal => "metal-b11623",
        }
    }

    /// The catalog items (zips) of this engine, in unpack order.
    pub fn items(self) -> &'static [&'static str] {
        match self {
            EngineFlavor::Vulkan => &[ENGINE_VULKAN],
            EngineFlavor::Cuda => &[ENGINE_CUDA, ENGINE_CUDART],
            EngineFlavor::Metal => &[ENGINE_METAL],
        }
    }
}

pub const ENGINE_VULKAN: &str = "engine-vulkan";
pub const ENGINE_CUDA: &str = "engine-cuda";
pub const ENGINE_CUDART: &str = "engine-cudart";
pub const ENGINE_METAL: &str = "engine-metal";
pub const MODEL_FAST: &str = "model-fast";
pub const MODEL_QUALITY: &str = "model-quality";

/// A Hugging Face model repository at a pinned commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HfRepo {
    pub org: &'static str,
    pub name: &'static str,
    pub commit: &'static str,
}

impl HfRepo {
    /// Folder of the repository in the Hugging Face cache, as hf-hub names it.
    pub fn cache_dir_name(&self) -> String {
        format!("models--{}--{}", self.org, self.name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Item {
    pub id: &'static str,
    pub kind: ItemKind,
    pub file_name: &'static str,
    pub url: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    /// Models only: where the file lives in the Hugging Face cache.
    pub repo: Option<HfRepo>,
    pub license: &'static str,
}

impl Item {
    /// Host shown on the card next to the size.
    pub fn host(&self) -> &'static str {
        match self.kind {
            ItemKind::Engine => "github.com",
            ItemKind::Model => "huggingface.co",
        }
    }
}

const FAST_REPO: HfRepo = HfRepo {
    org: "mradermacher",
    name: "translategemma-4b-it-GGUF",
    commit: "d37868f759d2f817a43d28dd80cf9f982ffa62fd",
};

const QUALITY_REPO: HfRepo = HfRepo {
    org: "mradermacher",
    name: "translategemma-12b-it-GGUF",
    commit: "41d7c8aa650b26791225ad9d0597e674e680da96",
};

/// The engine the setup installs by default (the card's "recommended").
pub const DEFAULT_ENGINE: EngineFlavor = if cfg!(target_os = "macos") {
    EngineFlavor::Metal
} else {
    EngineFlavor::Vulkan
};

/// Every item the app knows. Which of them the card offers on this OS is
/// `offered()`: the Windows zips on Windows, the Metal archive on macOS.
pub const CATALOG: [Item; 6] = [
    Item {
        id: ENGINE_VULKAN,
        kind: ItemKind::Engine,
        file_name: "llama-b11623-bin-win-vulkan-x64.zip",
        url: "https://github.com/ggml-org/llama.cpp/releases/download/b11623/llama-b11623-bin-win-vulkan-x64.zip",
        size: 32_185_125,
        sha256: "b02f7c2016e61c141a24263a628ca8a6d83a5d81f98f80ef488adb3207633115",
        repo: None,
        license: "MIT",
    },
    Item {
        id: ENGINE_CUDA,
        kind: ItemKind::Engine,
        file_name: "llama-b11623-bin-win-cuda-13.4-x64.zip",
        url: "https://github.com/ggml-org/llama.cpp/releases/download/b11623/llama-b11623-bin-win-cuda-13.4-x64.zip",
        size: 153_522_247,
        sha256: "124fe40fe43780f7124a573cef18b3edc1b14f968a5fca22e2b4143c9fa2778e",
        repo: None,
        license: "MIT",
    },
    Item {
        id: ENGINE_CUDART,
        kind: ItemKind::Engine,
        file_name: "cudart-llama-bin-win-cuda-13.4-x64.zip",
        url: "https://github.com/ggml-org/llama.cpp/releases/download/b11623/cudart-llama-bin-win-cuda-13.4-x64.zip",
        size: 423_535_356,
        sha256: "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
        repo: None,
        license: "NVIDIA CUDA EULA",
    },
    Item {
        id: ENGINE_METAL,
        kind: ItemKind::Engine,
        file_name: "llama-b11623-bin-macos-arm64.tar.gz",
        url: "https://github.com/ggml-org/llama.cpp/releases/download/b11623/llama-b11623-bin-macos-arm64.tar.gz",
        size: 12_162_074,
        sha256: "391d4eeb19fb21cb408b946deb835a444cf6f81d6d03c4ad01f24877c96989ca",
        repo: None,
        license: "MIT",
    },
    Item {
        id: MODEL_FAST,
        kind: ItemKind::Model,
        file_name: "translategemma-4b-it.Q4_K_M.gguf",
        url: "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/d37868f759d2f817a43d28dd80cf9f982ffa62fd/translategemma-4b-it.Q4_K_M.gguf",
        size: 2_489_909_760,
        sha256: "81200d03e843d2ec1ece6eeafe7d13cb6e5211e1fcd336ade55790b683a08330",
        repo: Some(FAST_REPO),
        license: "Gemma Terms of Use",
    },
    Item {
        id: MODEL_QUALITY,
        kind: ItemKind::Model,
        file_name: "translategemma-12b-it.Q4_K_M.gguf",
        url: "https://huggingface.co/mradermacher/translategemma-12b-it-GGUF/resolve/41d7c8aa650b26791225ad9d0597e674e680da96/translategemma-12b-it.Q4_K_M.gguf",
        size: 7_300_794_112,
        sha256: "b7aac4b4be7ab0c49b6556c29c4467e74313df7f1e95d9f9676bb2adf0afa528",
        repo: Some(QUALITY_REPO),
        license: "Gemma Terms of Use",
    },
];

pub fn item(id: &str) -> Option<&'static Item> {
    CATALOG.iter().find(|i| i.id == id)
}

/// Whether the card offers the item on this OS.
fn is_offered(id: &str) -> bool {
    let windows_engine = [ENGINE_VULKAN, ENGINE_CUDA, ENGINE_CUDART].contains(&id);
    if cfg!(target_os = "macos") {
        !windows_engine
    } else {
        id != ENGINE_METAL
    }
}

/// The catalog as this OS offers it.
pub fn offered() -> impl Iterator<Item = &'static Item> {
    CATALOG.iter().filter(|i| is_offered(i.id))
}

/// Space an unpacked engine takes besides its zip (DESIGN F.6; measured in F2).
pub fn unpacked_engine_bytes(flavor: EngineFlavor) -> u64 {
    match flavor {
        EngineFlavor::Vulkan => 100 << 20,
        EngineFlavor::Cuda => 1300 << 20,
        // 31 MB measured on b11623 (J2.1).
        EngineFlavor::Metal => 32 << 20,
    }
}

/// `translate\` in the app data folder: the engines and their zips.
pub fn translate_dir(app_data: &Path) -> PathBuf {
    app_data.join("translate")
}

pub fn engines_dir(app_data: &Path) -> PathBuf {
    translate_dir(app_data).join("engine")
}

pub fn engine_dir(app_data: &Path, flavor: EngineFlavor) -> PathBuf {
    engines_dir(app_data).join(flavor.dir_name())
}

pub fn engine_exe(app_data: &Path, flavor: EngineFlavor) -> PathBuf {
    engine_dir(app_data, flavor).join(SERVER_EXE)
}

/// `.partial` of an engine zip (the zip itself is deleted after unpacking).
pub fn zip_partial(app_data: &Path, item: &Item) -> PathBuf {
    translate_dir(app_data)
        .join("downloads")
        .join(format!("{}.partial", item.file_name))
}

pub const SERVER_EXE: &str = if cfg!(unix) {
    "llama-server"
} else {
    "llama-server.exe"
};

/// Folder of a model's repository in the Hugging Face cache rooted at `hub`.
fn repo_dir(hub: &Path, item: &Item) -> Option<(PathBuf, HfRepo)> {
    let repo = item.repo?;
    Some((hub.join(repo.cache_dir_name()), repo))
}

/// The model file in the cache, where hf-hub puts it on Windows (a real file
/// in the snapshot of the pinned commit).
pub fn model_file(hub: &Path, item: &Item) -> Option<PathBuf> {
    let (dir, repo) = repo_dir(hub, item)?;
    Some(dir.join("snapshots").join(repo.commit).join(item.file_name))
}

/// `.partial` of a model: in `blobs\` of the same repository folder, so the
/// final rename stays on one volume even when `HF_HOME` is elsewhere.
pub fn model_partial(hub: &Path, item: &Item) -> Option<PathBuf> {
    let (dir, _) = repo_dir(hub, item)?;
    Some(dir.join("blobs").join(format!("{}.partial", item.sha256)))
}

/// `refs\main` of the repository (holds the commit, as hf-hub writes it).
pub fn model_ref(hub: &Path, item: &Item) -> Option<PathBuf> {
    let (dir, _) = repo_dir(hub, item)?;
    Some(dir.join("refs").join("main"))
}

/// Where a catalog item's `.partial` is while it downloads.
pub fn partial_path(app_data: &Path, hub: &Path, item: &Item) -> PathBuf {
    match item.kind {
        ItemKind::Engine => zip_partial(app_data, item),
        ItemKind::Model => model_partial(hub, item).expect("a model has a repo"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_go_into_the_hugging_face_cache_layout() {
        let hub = Path::new(r"D:\hf\hub");
        let quality = item(MODEL_QUALITY).unwrap();
        assert_eq!(
            model_file(hub, quality).unwrap(),
            hub.join("models--mradermacher--translategemma-12b-it-GGUF")
                .join("snapshots")
                .join("41d7c8aa650b26791225ad9d0597e674e680da96")
                .join("translategemma-12b-it.Q4_K_M.gguf")
        );
        let partial = model_partial(hub, quality).unwrap();
        assert_eq!(
            partial.parent().unwrap(),
            hub.join("models--mradermacher--translategemma-12b-it-GGUF")
                .join("blobs")
        );
        assert_eq!(
            partial.file_name().unwrap().to_str().unwrap(),
            format!("{}.partial", quality.sha256)
        );
        assert!(model_ref(hub, quality)
            .unwrap()
            .ends_with(Path::new("refs").join("main")));
        assert!(model_file(hub, item(ENGINE_VULKAN).unwrap()).is_none());

        // Models never go into the app models folder Handy scans.
        let app_data = Path::new(r"C:\Users\u\AppData\Roaming\com.seltranslate.app");
        for id in [MODEL_FAST, MODEL_QUALITY] {
            let it = item(id).unwrap();
            for p in [
                model_file(hub, it).unwrap(),
                partial_path(app_data, hub, it),
            ] {
                assert!(p.starts_with(hub) && !p.starts_with(app_data.join("models")));
            }
        }
        assert_eq!(
            engine_exe(app_data, EngineFlavor::Vulkan),
            app_data
                .join("translate")
                .join("engine")
                .join("vulkan-b11623")
                .join(SERVER_EXE)
        );
        assert_eq!(
            partial_path(app_data, hub, item(ENGINE_CUDART).unwrap()),
            app_data
                .join("translate")
                .join("downloads")
                .join("cudart-llama-bin-win-cuda-13.4-x64.zip.partial")
        );
    }

    #[test]
    fn every_item_is_pinned_and_well_formed() {
        for item in &CATALOG {
            assert!(item.url.starts_with("https://"), "{}", item.id);
            assert!(item.url.ends_with(item.file_name), "{}", item.id);
            assert!(item.size > 0, "{}", item.id);
            assert_eq!(item.sha256.len(), 64, "{}", item.id);
            assert!(
                item.sha256
                    .chars()
                    .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
                "{}: lowercase hex",
                item.id
            );
            assert!(
                super::super::install::host_allowed(item.url),
                "{}: host",
                item.id
            );
            match (item.kind, item.repo) {
                (ItemKind::Model, Some(repo)) => {
                    assert_eq!(repo.commit.len(), 40, "{}", item.id);
                    let expected = format!(
                        "https://huggingface.co/{}/{}/resolve/{}/{}",
                        repo.org, repo.name, repo.commit, item.file_name
                    );
                    assert_eq!(item.url, expected, "pinned to the commit, not main");
                    assert!(!item.url.contains("/main/"));
                }
                (ItemKind::Engine, None) => {
                    assert!(item.url.contains(&format!("/download/{ENGINE_BUILD}/")));
                }
                other => panic!("{}: kind and repo disagree: {other:?}", item.id),
            }
        }
        let mut ids: Vec<_> = CATALOG.iter().map(|i| i.id).collect();
        ids.dedup();
        assert_eq!(ids.len(), CATALOG.len(), "unique ids");
    }

    #[test]
    fn engines_list_their_zips() {
        assert_eq!(EngineFlavor::Vulkan.items(), &[ENGINE_VULKAN]);
        assert_eq!(EngineFlavor::Cuda.items(), &[ENGINE_CUDA, ENGINE_CUDART]);
        for flavor in [EngineFlavor::Vulkan, EngineFlavor::Cuda] {
            for id in flavor.items() {
                assert_eq!(item(id).unwrap().kind, ItemKind::Engine);
            }
            assert!(flavor.dir_name().ends_with(ENGINE_BUILD));
        }
        assert_eq!(
            item(MODEL_QUALITY).unwrap().repo.unwrap().cache_dir_name(),
            "models--mradermacher--translategemma-12b-it-GGUF"
        );
        assert!(item("engine-cpu").is_none(), "the CPU zip is not offered");
    }

    #[test]
    fn each_os_is_offered_its_own_engine() {
        let ids: Vec<&str> = offered().map(|i| i.id).collect();
        if cfg!(target_os = "macos") {
            assert_eq!(ids, [ENGINE_METAL, MODEL_FAST, MODEL_QUALITY]);
            assert_eq!(DEFAULT_ENGINE, EngineFlavor::Metal);
        } else {
            assert_eq!(
                ids,
                [
                    ENGINE_VULKAN,
                    ENGINE_CUDA,
                    ENGINE_CUDART,
                    MODEL_FAST,
                    MODEL_QUALITY
                ]
            );
            assert_eq!(DEFAULT_ENGINE, EngineFlavor::Vulkan);
        }
    }

    #[test]
    fn the_metal_engine_is_the_pinned_macos_archive() {
        let metal = item(ENGINE_METAL).unwrap();
        assert_eq!(metal.file_name, "llama-b11623-bin-macos-arm64.tar.gz");
        assert_eq!(metal.size, 12_162_074);
        assert_eq!(
            metal.sha256,
            "391d4eeb19fb21cb408b946deb835a444cf6f81d6d03c4ad01f24877c96989ca"
        );
        assert_eq!(EngineFlavor::Metal.items(), [ENGINE_METAL]);
        let app = Path::new("/app");
        assert_eq!(
            engine_exe(app, EngineFlavor::Metal),
            app.join("translate")
                .join("engine")
                .join("metal-b11623")
                .join(SERVER_EXE)
        );
        assert_eq!(
            SERVER_EXE,
            if cfg!(unix) {
                "llama-server"
            } else {
                "llama-server.exe"
            }
        );
    }
}
