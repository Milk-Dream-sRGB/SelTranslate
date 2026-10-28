//! Which models the setup offers, by the video memory of the best card
//! (DESIGN F.2). The memory comes from DXGI before anything is downloaded;
//! `llama-server --list-devices` only checks afterwards that the engine sees it.

// The DXGI tiers (`pick_tier` and its limits) are tested everywhere but used
// only off macOS, where `pick_tier_unified` takes over.
#![cfg_attr(target_os = "macos", allow(dead_code))]

use serde::Serialize;
use specta::Type;

use super::super::llama_server::GPU_LAYERS_AUTO;
use super::super::settings::{SelectionSlot, Slot};

/// One display adapter as DXGI reports it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Type)]
pub struct Adapter {
    pub name: String,
    pub vendor_id: u32,
    pub dedicated_mib: u64,
    pub software: bool,
}

pub const VENDOR_NVIDIA: u32 = 0x10DE;
/// Microsoft Basic Render Driver (and WARP).
const VENDOR_MICROSOFT: u32 = 0x1414;

impl Adapter {
    pub fn is_nvidia(&self) -> bool {
        self.vendor_id == VENDOR_NVIDIA
    }
}

/// Both models fit on the card whole: Quality 8.1 GB + fit margin + desktop.
pub const BOTH_MIN_MIB: u64 = 9728;
/// On a Partial card Quality is ticked by default from 8 GB up.
pub const PARTIAL_QUALITY_MIB: u64 = 7680;
pub const PARTIAL_MIN_MIB: u64 = 5632;
/// Below this the card is integrated graphics: the processor is used.
pub const SMALL_MIN_MIB: u64 = 2048;

/// Apple Silicon (DESIGN J.9): the model, Whisper, the system and the apps share
/// one memory. Below 12 GiB only Fast is offered; from 32 GiB Quality fits whole.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub const UNIFIED_PARTIAL_MIN_MIB: u64 = 12 * 1024;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub const UNIFIED_BOTH_MIN_MIB: u64 = 32 * 1024;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const VENDOR_APPLE: u32 = 0x106B;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Both,
    Partial,
    Small,
    Cpu,
}

impl Tier {
    /// Whether the card is used at all (`gpu_layers` auto rather than 0).
    pub fn uses_gpu(self) -> bool {
        self != Tier::Cpu
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierPick {
    pub tier: Tier,
    /// The adapter the tier is based on (`None`: no usable card).
    pub adapter: Option<Adapter>,
    /// Whether the Quality checkbox starts ticked.
    pub quality_default: bool,
}

/// The tier of the adapter with the most dedicated memory; software adapters
/// are ignored.
pub fn pick_tier(adapters: &[Adapter]) -> TierPick {
    let best = adapters
        .iter()
        .filter(|a| !a.software && a.vendor_id != VENDOR_MICROSOFT)
        .max_by_key(|a| a.dedicated_mib)
        .cloned();
    let mib = best.as_ref().map_or(0, |a| a.dedicated_mib);
    let tier = match mib {
        m if m >= BOTH_MIN_MIB => Tier::Both,
        m if m >= PARTIAL_MIN_MIB => Tier::Partial,
        m if m >= SMALL_MIN_MIB => Tier::Small,
        _ => Tier::Cpu,
    };
    TierPick {
        tier,
        adapter: best.filter(|_| tier.uses_gpu()),
        quality_default: tier == Tier::Both
            || (tier == Tier::Partial && mib >= PARTIAL_QUALITY_MIB),
    }
}

/// The tier of a Mac by its unified memory. Quality is never ticked below
/// 32 GiB: it takes about 8 GB that the other apps then lack.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub fn pick_tier_unified(name: &str, ram_mib: u64) -> TierPick {
    let tier = match ram_mib {
        m if m >= UNIFIED_BOTH_MIN_MIB => Tier::Both,
        m if m >= UNIFIED_PARTIAL_MIN_MIB => Tier::Partial,
        _ => Tier::Small,
    };
    TierPick {
        tier,
        adapter: Some(Adapter {
            name: name.to_string(),
            vendor_id: VENDOR_APPLE,
            dedicated_mib: ram_mib,
            software: false,
        }),
        quality_default: tier == Tier::Both,
    }
}

/// The tier of this computer: DXGI adapters on Windows, the unified memory on
/// a Mac, the processor elsewhere.
pub fn platform_pick() -> TierPick {
    #[cfg(target_os = "macos")]
    {
        use super::super::macos::memory;
        pick_tier_unified(
            &memory::chip_name().unwrap_or_else(|| "Apple Silicon".into()),
            memory::memory_mib().unwrap_or(0),
        )
    }
    #[cfg(not(target_os = "macos"))]
    pick_tier(&adapters())
}

/// Whether the memory is shared with the graphics (a Mac).
pub fn unified_memory() -> bool {
    cfg!(target_os = "macos")
}

/// What the setup writes to the settings (besides the paths).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetupSettings {
    pub gpu_layers: u32,
    pub selection_slot: SelectionSlot,
    pub file_slot: Slot,
}

/// Slots and GPU layers for the models present after the setup. A card the
/// engine cannot see means the processor (`gpu_layers` 0).
pub fn settings_for(tier: Tier, fast: bool, quality: bool, engine_sees_gpu: bool) -> SetupSettings {
    let gpu_layers = if tier.uses_gpu() && engine_sees_gpu {
        GPU_LAYERS_AUTO
    } else {
        0
    };
    // On a Mac below 32 GiB Quality is for Better translation only (J.9).
    let (selection_slot, file_slot) =
        slots_for(fast, quality, unified_memory() && tier != Tier::Both);
    SetupSettings {
        gpu_layers,
        selection_slot,
        file_slot,
    }
}

/// Slots for the models present; `prefer_fast` keeps Fast for the hotkey and the
/// file even when Quality is there too.
pub fn slots_for(fast: bool, quality: bool, prefer_fast: bool) -> (SelectionSlot, Slot) {
    match (fast, quality) {
        (true, true) if prefer_fast => (SelectionSlot::Fast, Slot::Fast),
        (true, true) => (SelectionSlot::Auto, Slot::Quality),
        (false, true) => (SelectionSlot::Quality, Slot::Quality),
        _ => (SelectionSlot::Fast, Slot::Fast),
    }
}

/// Adapters of this computer (empty when DXGI fails: the processor tier).
#[cfg(windows)]
pub fn adapters() -> Vec<Adapter> {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE,
    };
    let factory = match unsafe { CreateDXGIFactory1::<IDXGIFactory1>() } {
        Ok(factory) => factory,
        Err(e) => {
            log::warn!("Translate setup: DXGI unavailable: {e}");
            return Vec::new();
        }
    };
    let mut found = Vec::new();
    // EnumAdapters1 fails with DXGI_ERROR_NOT_FOUND past the last adapter.
    for index in 0..16 {
        let Ok(adapter) = (unsafe { factory.EnumAdapters1(index) }) else {
            break;
        };
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
            continue;
        };
        let len = desc
            .Description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(desc.Description.len());
        found.push(Adapter {
            name: String::from_utf16_lossy(&desc.Description[..len])
                .trim()
                .to_string(),
            vendor_id: desc.VendorId,
            dedicated_mib: (desc.DedicatedVideoMemory as u64) >> 20,
            software: desc.Flags & (DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0,
        });
    }
    found
}

#[cfg(not(windows))]
pub fn adapters() -> Vec<Adapter> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(name: &str, vendor_id: u32, dedicated_mib: u64) -> Adapter {
        Adapter {
            name: name.into(),
            vendor_id,
            dedicated_mib,
            software: false,
        }
    }

    fn tier_of(mib: u64) -> (Tier, bool) {
        let pick = pick_tier(&[card("GPU", VENDOR_NVIDIA, mib)]);
        (pick.tier, pick.quality_default)
    }

    #[test]
    fn tiers_follow_the_table() {
        for gb in [16, 12, 10] {
            assert_eq!(tier_of(gb * 1024), (Tier::Both, true), "{gb} GB");
        }
        assert_eq!(tier_of(8 * 1024), (Tier::Partial, true));
        assert_eq!(tier_of(6 * 1024), (Tier::Partial, false));
        assert_eq!(tier_of(4 * 1024).0, Tier::Small);
        assert_eq!(tier_of(2 * 1024).0, Tier::Small);
        assert_eq!(tier_of(128), (Tier::Cpu, false));
        let none = pick_tier(&[]);
        assert_eq!((none.tier, none.adapter), (Tier::Cpu, None));
    }

    #[test]
    fn tier_boundaries() {
        assert_eq!(tier_of(9727).0, Tier::Partial);
        assert_eq!(tier_of(9728).0, Tier::Both);
        assert_eq!(tier_of(7679), (Tier::Partial, false));
        assert_eq!(tier_of(7680), (Tier::Partial, true));
        assert_eq!(tier_of(5631).0, Tier::Small);
        assert_eq!(tier_of(5632).0, Tier::Partial);
        assert_eq!(tier_of(2047).0, Tier::Cpu);
        assert_eq!(tier_of(2048).0, Tier::Small);
    }

    #[test]
    fn software_adapters_are_ignored_and_the_biggest_card_wins() {
        let basic = Adapter {
            software: true,
            ..card("Microsoft Basic Render Driver", VENDOR_MICROSOFT, 16 * 1024)
        };
        let warp = card("WARP", VENDOR_MICROSOFT, 32 * 1024);
        assert_eq!(pick_tier(&[basic.clone(), warp]).tier, Tier::Cpu);

        let intel = card("Intel UHD", 0x8086, 128);
        let rtx = card("NVIDIA GeForce RTX 3060", VENDOR_NVIDIA, 12 * 1024);
        let pick = pick_tier(&[intel.clone(), basic, rtx.clone()]);
        assert_eq!(pick.tier, Tier::Both);
        assert_eq!(pick.adapter, Some(rtx));
        assert_eq!(pick_tier(&[intel]).adapter, None, "no card on Cpu");
    }

    #[test]
    #[cfg(windows)]
    fn settings_follow_the_tier_and_the_models() {
        use SelectionSlot as Sel;
        let s = |tier, fast, quality, sees| {
            let r = settings_for(tier, fast, quality, sees);
            (r.gpu_layers, r.selection_slot, r.file_slot)
        };
        assert_eq!(
            s(Tier::Both, true, true, true),
            (99, Sel::Auto, Slot::Quality)
        );
        assert_eq!(
            s(Tier::Partial, true, true, true),
            (99, Sel::Auto, Slot::Quality)
        );
        assert_eq!(
            s(Tier::Partial, true, false, true),
            (99, Sel::Fast, Slot::Fast)
        );
        assert_eq!(
            s(Tier::Small, true, false, true),
            (99, Sel::Fast, Slot::Fast)
        );
        assert_eq!(s(Tier::Cpu, true, false, true), (0, Sel::Fast, Slot::Fast));
        assert_eq!(
            s(Tier::Both, false, true, true),
            (99, Sel::Quality, Slot::Quality),
            "only Quality"
        );
        // The engine does not see the card: the processor.
        assert_eq!(
            s(Tier::Both, true, false, false),
            (0, Sel::Fast, Slot::Fast)
        );
        assert_eq!(GPU_LAYERS_AUTO, 99);
    }

    #[test]
    fn nvidia_is_told_by_the_vendor_id() {
        assert!(card("x", VENDOR_NVIDIA, 1).is_nvidia());
        assert!(!card("x", 0x1002, 1).is_nvidia());
    }

    /// Prints what DXGI reports here (no assertions on the hardware).
    #[test]
    #[cfg(windows)]
    fn dxgi_lists_adapters() {
        let found = adapters();
        eprintln!("adapters: {found:?} -> {:?}", pick_tier(&found).tier);
    }

    #[test]
    fn a_mac_is_tiered_by_its_unified_memory() {
        let gib = |n: u64| n * 1024;
        let t = |ram| {
            let pick = pick_tier_unified("Apple M3 Pro", ram);
            (pick.tier, pick.quality_default)
        };
        assert_eq!(t(gib(8)), (Tier::Small, false));
        assert_eq!(t(gib(12) - 1), (Tier::Small, false));
        assert_eq!(t(gib(12)), (Tier::Partial, false));
        assert_eq!(t(gib(16)), (Tier::Partial, false));
        assert_eq!(t(gib(18)), (Tier::Partial, false));
        assert_eq!(t(gib(24)), (Tier::Partial, false));
        assert_eq!(t(gib(32) - 1), (Tier::Partial, false));
        assert_eq!(t(gib(32)), (Tier::Both, true));
        assert_eq!(t(gib(64)), (Tier::Both, true));
        // The memory and the chip are what the card shows.
        let pick = pick_tier_unified("Apple M3 Pro", gib(18));
        let adapter = pick.adapter.unwrap();
        assert_eq!(
            (adapter.name.as_str(), adapter.dedicated_mib),
            ("Apple M3 Pro", gib(18))
        );
    }

    #[test]
    fn fast_is_kept_for_the_hotkey_when_asked() {
        use SelectionSlot as Sel;
        assert_eq!(slots_for(true, true, true), (Sel::Fast, Slot::Fast));
        assert_eq!(slots_for(true, true, false), (Sel::Auto, Slot::Quality));
        assert_eq!(slots_for(false, true, true), (Sel::Quality, Slot::Quality));
        assert_eq!(slots_for(true, false, true), (Sel::Fast, Slot::Fast));
        assert_eq!(slots_for(false, false, false), (Sel::Fast, Slot::Fast));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn on_a_mac_quality_is_for_better_translation_below_32_gib() {
        use SelectionSlot as Sel;
        let s = |tier, fast, quality| {
            let r = settings_for(tier, fast, quality, true);
            (r.gpu_layers, r.selection_slot, r.file_slot)
        };
        assert_eq!(
            s(Tier::Partial, true, true),
            (GPU_LAYERS_AUTO, Sel::Fast, Slot::Fast)
        );
        assert_eq!(
            s(Tier::Small, true, true),
            (GPU_LAYERS_AUTO, Sel::Fast, Slot::Fast)
        );
        assert_eq!(
            s(Tier::Both, true, true),
            (GPU_LAYERS_AUTO, Sel::Auto, Slot::Quality)
        );
        assert_eq!(
            s(Tier::Partial, true, false),
            (GPU_LAYERS_AUTO, Sel::Fast, Slot::Fast)
        );
    }
}
