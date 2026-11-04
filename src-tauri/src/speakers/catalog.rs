//! The Sortformer diarizer files the user can pick from: one repository at a
//! pinned commit, three precisions of the same weights. Sizes and SHA-256 are
//! the trust anchor, as in the translation catalog; the files go into the
//! Hugging Face cache in the same layout (`translate::setup::catalog`).

use std::path::{Path, PathBuf};

use crate::translate::setup::catalog::{self, HfRepo, Item, ItemKind};

/// handy-computer's GGUF conversion of nvidia/diar_streaming_sortformer_4spk-v2.1.
const REPO: HfRepo = HfRepo {
    org: "handy-computer",
    name: "diar_streaming_sortformer_4spk-v2.1-gguf",
    commit: "60ebab3ef7cae9869be0abcde13360347e96ba03",
};

const LICENSE: &str = "NVIDIA Open Model License";

pub const Q8_0: &str = "q8_0";
pub const F16: &str = "f16";
pub const F32: &str = "f32";

/// The variant used until the user picks another: the smallest, with almost
/// the same accuracy (DER on AMI IHM test: Q8_0 14.73 %, F16 14.23 %,
/// F32 14.59 %, from the model card).
pub const DEFAULT_VARIANT: &str = Q8_0;

/// The variants, smallest first; `Item::id` is the variant id.
pub const VARIANTS: [Item; 3] = [
    Item {
        id: Q8_0,
        kind: ItemKind::Model,
        file_name: "diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf",
        url: "https://huggingface.co/handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf/resolve/60ebab3ef7cae9869be0abcde13360347e96ba03/diar_streaming_sortformer_4spk-v2.1-Q8_0.gguf",
        size: 139_310_336,
        sha256: "a5dacdc650790266c7a362e54e6bf51952015487edaa606c4e11632bc32442a9",
        repo: Some(REPO),
        license: LICENSE,
    },
    Item {
        id: F16,
        kind: ItemKind::Model,
        file_name: "diar_streaming_sortformer_4spk-v2.1-F16.gguf",
        url: "https://huggingface.co/handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf/resolve/60ebab3ef7cae9869be0abcde13360347e96ba03/diar_streaming_sortformer_4spk-v2.1-F16.gguf",
        size: 236_606_560,
        sha256: "62faec7b99ad23e323087597604b50728abe85089b6364970b019a845547bf99",
        repo: Some(REPO),
        license: LICENSE,
    },
    Item {
        id: F32,
        kind: ItemKind::Model,
        file_name: "diar_streaming_sortformer_4spk-v2.1-F32.gguf",
        url: "https://huggingface.co/handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf/resolve/60ebab3ef7cae9869be0abcde13360347e96ba03/diar_streaming_sortformer_4spk-v2.1-F32.gguf",
        size: 470_910_560,
        sha256: "61597ce9447539ca32ce6fefb923551ab6afbdc068223c5d19929296ad2aa380",
        repo: Some(REPO),
        license: LICENSE,
    },
];

pub fn variant(id: &str) -> Option<&'static Item> {
    VARIANTS.iter().find(|v| v.id == id)
}

/// The variant's file in the cache rooted at `hub`, if it is there whole.
pub fn downloaded(hub: &Path, item: &Item) -> Option<PathBuf> {
    catalog::model_file(hub, item)
        .filter(|p| std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.len() == item.size))
}

/// Moves a verified `.partial` into the snapshot and writes `refs/main`, as
/// hf-hub would have.
pub fn finalize(hub: &Path, item: &Item) -> std::io::Result<PathBuf> {
    let partial = catalog::model_partial(hub, item).expect("a variant has a repo");
    let file = catalog::model_file(hub, item).expect("a variant has a repo");
    let reference = catalog::model_ref(hub, item).expect("a variant has a repo");
    for dir in [file.parent(), reference.parent()].into_iter().flatten() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::rename(&partial, &file)?;
    std::fs::write(&reference, REPO.commit)?;
    Ok(file)
}

/// Deletes the variant's file and any `.partial` of it; the repository folder
/// goes too once nothing of ours is left in it.
pub fn delete(hub: &Path, item: &Item) -> std::io::Result<()> {
    for path in [
        catalog::model_file(hub, item),
        catalog::model_partial(hub, item),
    ]
    .into_iter()
    .flatten()
    {
        match std::fs::remove_file(&path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e),
            _ => {}
        }
    }
    let any_left = VARIANTS.iter().any(|v| downloaded(hub, v).is_some());
    if !any_left {
        let dir = hub.join(REPO.cache_dir_name());
        if dir.exists() {
            std::fs::remove_dir_all(dir)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_is_pinned_to_the_repository_commit() {
        for v in &VARIANTS {
            assert!(v.url.contains(REPO.commit), "{}", v.id);
            assert!(v.url.ends_with(v.file_name), "{}", v.id);
            assert_eq!(v.sha256.len(), 64, "{}", v.id);
            assert!(crate::translate::setup::install::host_allowed(v.url));
        }
        assert!(variant(DEFAULT_VARIANT).is_some());
    }

    #[test]
    fn a_finalized_file_counts_as_downloaded_and_delete_clears_the_folder() {
        let hub = tempfile::tempdir().unwrap();
        let item = variant(Q8_0).unwrap();
        assert!(downloaded(hub.path(), item).is_none());

        let partial = catalog::model_partial(hub.path(), item).unwrap();
        std::fs::create_dir_all(partial.parent().unwrap()).unwrap();
        let file = std::fs::File::create(&partial).unwrap();
        file.set_len(item.size).unwrap();
        drop(file);

        let path = finalize(hub.path(), item).unwrap();
        assert_eq!(downloaded(hub.path(), item), Some(path));
        assert!(downloaded(hub.path(), variant(F16).unwrap()).is_none());

        delete(hub.path(), item).unwrap();
        assert!(downloaded(hub.path(), item).is_none());
        assert!(!hub.path().join(REPO.cache_dir_name()).exists());
    }
}
