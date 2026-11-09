//! What to do when llama-server crashes on start because of its video card
//! engine (an NVIDIA driver too old for the CUDA build is the usual case:
//! "the provided PTX was compiled with an unsupported toolchain").
//!
//! CUDA → Vulkan when the setup's Vulkan engine is there (kept in the
//! settings: the next launch starts on Vulkan right away), otherwise → the
//! processor (`--device none`) until the app restarts, so a driver update
//! gets the video card back without a visit to the settings. Nothing here
//! depends on Tauri; `translate::ensure_server` applies it.

use super::setup::EngineKind;

/// The line `ensure_running` starts a crash during start-up with.
const EXITED_ON_START: &str = "exited during start-up";

/// Lines of the log tail that only a broken GPU backend writes. A crash
/// without one (a damaged model, no memory) is not the engine's doing, and
/// switching the engine would not help.
const CUDA_SIGNS: &[&str] = &[
    "CUDA error",
    "ggml_cuda_init: failed",
    "no kernel image is available",
    "CUDA driver version is insufficient",
    "unsupported toolchain",
];
const VULKAN_SIGNS: &[&str] = &[
    "ggml_vulkan: Error",
    "ggml_vulkan: error",
    "vk::DeviceLostError",
    "ErrorDeviceLost",
    "VK_ERROR_",
];
const METAL_SIGNS: &[&str] = &["ggml_metal_init: error", "ggml_metal_device_init: error"];

/// Whether the start error is the video card engine crashing.
pub fn gpu_start_crash(kind: EngineKind, message: &str) -> bool {
    if !message.contains(EXITED_ON_START) {
        return false;
    }
    let signs = match kind {
        EngineKind::Cuda => CUDA_SIGNS,
        EngineKind::Vulkan => VULKAN_SIGNS,
        EngineKind::Metal => METAL_SIGNS,
        EngineKind::Cpu | EngineKind::Missing => return false,
    };
    signs.iter().any(|sign| message.contains(sign))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Switch `llama_server_path` to the setup's Vulkan engine (kept).
    UseVulkan,
    /// Start without a video card until the app restarts.
    UseCpu,
}

/// The next thing to try after a GPU start crash; `None` when there is nothing
/// left (already on the processor).
pub fn next_step(kind: EngineKind, vulkan_installed: bool, cpu_only: bool) -> Option<Step> {
    if cpu_only {
        return None;
    }
    match kind {
        EngineKind::Cuda if vulkan_installed => Some(Step::UseVulkan),
        EngineKind::Cuda | EngineKind::Vulkan | EngineKind::Metal => Some(Step::UseCpu),
        EngineKind::Cpu | EngineKind::Missing => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The log of the crash on an RTX 2060 with an old driver (CUDA 13 build).
    const PTX_CRASH: &str = r"llama-server exited during start-up (exit code: 0xc0000409). Last log lines:
0.00.180.390 I srv    load_model: loading model 'C:\Users\Admin\translategemma-4b-it.Q4_K_M.gguf'
0.03.553.197 I cmn          init: llama threadpool init, n_threads = 12
D:\a\llama.cpp\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:109: CUDA error
0.03.584.157 E CUDA error: the provided PTX was compiled with an unsupported toolchain.
0.03.584.167 E   current device: 0, in function ggml_cuda_kernel_can_use_pdl";

    #[test]
    fn the_ptx_crash_is_a_cuda_start_crash() {
        assert!(gpu_start_crash(EngineKind::Cuda, PTX_CRASH));
    }

    #[test]
    fn a_crash_without_a_gpu_sign_is_not_the_engines_doing() {
        let damaged = "llama-server exited during start-up (exit code: 1). Last log lines:\n\
                       llama_model_load: error loading model: tensor data is not within the file bounds";
        assert!(!gpu_start_crash(EngineKind::Cuda, damaged));
        assert!(!gpu_start_crash(EngineKind::Vulkan, damaged));
    }

    #[test]
    fn only_a_crash_during_start_up_counts() {
        let timeout = "llama-server did not become ready in 90 s. Last log lines:\nCUDA error";
        assert!(!gpu_start_crash(EngineKind::Cuda, timeout));
    }

    #[test]
    fn the_sign_must_match_the_engine() {
        assert!(!gpu_start_crash(EngineKind::Vulkan, PTX_CRASH));
        assert!(!gpu_start_crash(EngineKind::Cpu, PTX_CRASH));
        let lost = "llama-server exited during start-up (exit code: 0xc0000409).\n\
                    terminate called after throwing an instance of 'vk::DeviceLostError'";
        assert!(gpu_start_crash(EngineKind::Vulkan, lost));
    }

    #[test]
    fn cuda_goes_to_vulkan_when_it_is_there_and_to_the_processor_otherwise() {
        assert_eq!(
            next_step(EngineKind::Cuda, true, false),
            Some(Step::UseVulkan)
        );
        assert_eq!(
            next_step(EngineKind::Cuda, false, false),
            Some(Step::UseCpu)
        );
        assert_eq!(
            next_step(EngineKind::Vulkan, true, false),
            Some(Step::UseCpu)
        );
        assert_eq!(
            next_step(EngineKind::Metal, false, false),
            Some(Step::UseCpu)
        );
    }

    #[test]
    fn nothing_is_left_after_the_processor() {
        assert_eq!(next_step(EngineKind::Cuda, true, true), None);
        assert_eq!(next_step(EngineKind::Cpu, true, false), None);
    }
}
