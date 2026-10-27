//! The chip name and the unified memory of this Mac (DESIGN J.9), for the
//! tier of the setup card.

use std::ffi::{c_void, CString};

/// A sysctl value of `name` as raw bytes.
fn sysctl_bytes(name: &str) -> Option<Vec<u8>> {
    let name = CString::new(name).ok()?;
    let mut len = 0usize;
    // SAFETY: a size query (null buffer) for a NUL-terminated name.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || len == 0 {
        return None;
    }
    let mut buf = vec![0u8; len];
    // SAFETY: `buf` has `len` bytes, as the size query said.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            buf.as_mut_ptr() as *mut c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0).then(|| {
        buf.truncate(len);
        buf
    })
}

/// Installed memory in MiB (`hw.memsize`; the GPU shares it).
pub fn memory_mib() -> Option<u64> {
    let bytes: [u8; 8] = sysctl_bytes("hw.memsize")?.try_into().ok()?;
    Some(u64::from_ne_bytes(bytes) >> 20)
}

/// "Apple M3 Pro" (`machdep.cpu.brand_string`).
pub fn chip_name() -> Option<String> {
    let bytes = sysctl_bytes("machdep.cpu.brand_string")?;
    let text = String::from_utf8_lossy(&bytes);
    let name = text.trim_end_matches('\0').trim();
    (!name.is_empty()).then(|| name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn this_mac_reports_its_memory_and_chip() {
        let mib = memory_mib().expect("hw.memsize");
        assert!(mib >= 4096, "{mib} MiB");
        assert!(!chip_name().expect("brand string").is_empty());
    }
}
