//! Streaming audio-file decoder: symphonia packet by packet → mono f32 →
//! `FrameResampler` → fixed 16 kHz frames. Memory use is one packet plus one
//! resampler chunk, independent of the file length.

use std::fs::File;
use std::path::Path;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use symphonia::core::audio::{SampleBuffer, SignalSpec};
use symphonia::core::codecs::{Decoder, DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::audio_toolkit::audio::FrameResampler;
use crate::audio_toolkit::constants::WHISPER_SAMPLE_RATE;

/// Output frame length: 30 ms at 16 kHz, the Silero VAD frame size.
pub const FRAME_SAMPLES: usize = 480;
const FRAME_DURATION: Duration = Duration::from_millis(30);

/// Shown when a container (e.g. a video) has no track we can decode. The
/// UI matches this prefix to show a translated message.
pub const NO_AUDIO_TRACK: &str = "No supported audio track. Extract audio with ffmpeg first.";

pub struct StreamingDecoder {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
    track_id: u32,
    /// Native sample rate (from the container, or the first decoded packet).
    pub sample_rate: u32,
    pub channels: usize,
    /// Total length from the container; `None` when unknown (e.g. VBR mp3
    /// without a Xing/VBRI header).
    pub duration_secs: Option<f64>,
    /// Short codec name, e.g. "mp3", "aac", "flac".
    pub codec: String,
    resampler: Option<(u32, FrameResampler)>,
    sample_buf: Option<SampleBuffer<f32>>,
    mono: Vec<f32>,
    /// Native-rate frames decoded so far (for progress).
    decoded_frames: u64,
    decoded_secs: f64,
    finished: bool,
}

impl StreamingDecoder {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("Cannot open {}", path.display()))?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        let format_opts = FormatOptions {
            enable_gapless: true,
            ..Default::default()
        };
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &MetadataOptions::default())
            .map_err(|e| anyhow!("Unsupported or unreadable audio file: {e}"))?;
        let format = probed.format;

        // First track we can decode — not `default_track()`, which in a video
        // container may be the video. isomp4 reports video/subtitle tracks
        // as CODEC_TYPE_NULL; codecs without a decoder here (e.g. Opus) are
        // skipped too.
        let codecs = symphonia::default::get_codecs();
        let track = format
            .tracks()
            .iter()
            .find(|t| {
                t.codec_params.codec != CODEC_TYPE_NULL
                    && codecs.get_codec(t.codec_params.codec).is_some()
            })
            .ok_or_else(|| anyhow!(NO_AUDIO_TRACK))?;
        let params = &track.codec_params;

        let codec = codecs
            .get_codec(params.codec)
            .map(|d| d.short_name.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let decoder = codecs
            .make(params, &DecoderOptions::default())
            .map_err(|e| anyhow!("{NO_AUDIO_TRACK} ({codec}: {e})"))?;

        let sample_rate = params.sample_rate.unwrap_or(0);
        let channels = params.channels.map(|c| c.count()).unwrap_or(0);
        let duration_secs = match (params.n_frames, params.time_base) {
            (Some(n), Some(tb)) => {
                let t = tb.calc_time(n);
                Some(t.seconds as f64 + t.frac)
            }
            (Some(n), None) if sample_rate > 0 => Some(n as f64 / sample_rate as f64),
            _ => None,
        }
        .filter(|d| *d > 0.0);

        Ok(Self {
            track_id: track.id,
            format,
            decoder,
            sample_rate,
            channels,
            duration_secs,
            codec,
            resampler: None,
            sample_buf: None,
            mono: Vec::new(),
            decoded_frames: 0,
            decoded_secs: 0.0,
            finished: false,
        })
    }

    /// Seconds of source audio decoded so far.
    pub fn decoded_secs(&self) -> f64 {
        self.decoded_secs
    }

    /// Decode the next packet and emit any completed 16 kHz frames of
    /// `FRAME_SAMPLES`. Returns `false` once the stream is exhausted (the
    /// resampler tail has then been flushed, the last frame zero-padded).
    pub fn next_frames(&mut self, mut emit: impl FnMut(&[f32])) -> Result<bool> {
        if self.finished {
            return Ok(false);
        }

        loop {
            let packet = match self.format.next_packet() {
                Ok(p) => p,
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return self.finish(emit);
                }
                Err(SymphoniaError::ResetRequired) => {
                    // Chained streams (e.g. concatenated Ogg) — rare for
                    // recordings; stop at the first stream's end.
                    log::warn!("Audio file: stream reset required, stopping early");
                    return self.finish(emit);
                }
                Err(e) => return Err(anyhow!("Audio file read error: {e}")),
            };
            if packet.track_id() != self.track_id {
                continue;
            }

            let decoded = match self.decoder.decode(&packet) {
                Ok(d) => d,
                Err(SymphoniaError::DecodeError(msg)) => {
                    log::warn!("Audio file: skipping corrupt packet: {msg}");
                    continue;
                }
                Err(SymphoniaError::IoError(e)) => {
                    log::warn!("Audio file: skipping unreadable packet: {e}");
                    continue;
                }
                Err(e) => return Err(anyhow!("Audio decode error: {e}")),
            };

            let spec: SignalSpec = *decoded.spec();
            let ch = spec.channels.count();
            let frames = decoded.frames();
            if frames == 0 || ch == 0 {
                continue;
            }

            let needed = decoded.capacity() * ch;
            if self
                .sample_buf
                .as_ref()
                .is_none_or(|b| b.capacity() < needed)
            {
                self.sample_buf = Some(SampleBuffer::new(decoded.capacity() as u64, spec));
            }
            let sb = self.sample_buf.as_mut().unwrap();
            sb.copy_interleaved_ref(decoded);

            self.mono.clear();
            if ch == 1 {
                self.mono.extend_from_slice(sb.samples());
            } else {
                let inv = 1.0 / ch as f32;
                self.mono.extend(
                    sb.samples()
                        .chunks_exact(ch)
                        .map(|f| f.iter().sum::<f32>() * inv),
                );
            }

            self.sample_rate = spec.rate;
            self.channels = ch;
            self.decoded_frames += frames as u64;
            self.decoded_secs += frames as f64 / spec.rate as f64;

            // A mid-stream rate change (rare) gets a fresh resampler; the old
            // one is flushed first so no audio is lost.
            if self
                .resampler
                .as_ref()
                .is_some_and(|(hz, _)| *hz != spec.rate)
            {
                log::warn!("Audio file: sample rate changed to {} Hz", spec.rate);
                if let Some((_, mut old)) = self.resampler.take() {
                    old.finish(&mut emit);
                }
            }
            let (_, rs) = self.resampler.get_or_insert_with(|| {
                (
                    spec.rate,
                    FrameResampler::new(
                        spec.rate as usize,
                        WHISPER_SAMPLE_RATE as usize,
                        FRAME_DURATION,
                    ),
                )
            });
            rs.push(&self.mono, &mut emit);
            return Ok(true);
        }
    }

    fn finish(&mut self, mut emit: impl FnMut(&[f32])) -> Result<bool> {
        self.finished = true;
        if let Some((_, rs)) = self.resampler.as_mut() {
            rs.finish(&mut emit);
        }
        Ok(false)
    }

    /// Native-rate frames decoded so far.
    #[cfg(test)]
    pub fn decoded_frames(&self) -> u64 {
        self.decoded_frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn write_wav(path: &Path, rate: u32, channels: u16, secs: f64) {
        let spec = hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        let n = (rate as f64 * secs).round() as usize;
        for i in 0..n {
            let s = (2.0 * std::f64::consts::PI * 440.0 * i as f64 / rate as f64).sin() * 0.5;
            for _ in 0..channels {
                w.write_sample((s * i16::MAX as f64) as i16).unwrap();
            }
        }
        w.finalize().unwrap();
    }

    struct Decoded {
        samples_16k: usize,
        rms: f64,
        duration: Option<f64>,
        decoded_secs: f64,
    }

    fn decode_all(path: &Path) -> Decoded {
        let mut d = StreamingDecoder::open(path).unwrap();
        let duration = d.duration_secs;
        let mut n = 0usize;
        let mut sq = 0f64;
        let mut bad_frame = false;
        while d
            .next_frames(|f| {
                bad_frame |= f.len() != FRAME_SAMPLES;
                n += f.len();
                sq += f.iter().map(|s| (*s as f64).powi(2)).sum::<f64>();
            })
            .unwrap()
        {}
        assert!(!bad_frame, "every frame must be {FRAME_SAMPLES} samples");
        assert!(d.decoded_frames() > 0);
        Decoded {
            samples_16k: n,
            rms: (sq / n.max(1) as f64).sqrt(),
            duration,
            decoded_secs: d.decoded_secs(),
        }
    }

    /// ±1%. Output is quantized to 30 ms frames and starts with the
    /// resampler's delay line (≈15 ms at 48 kHz, ≈130 ms at 8 kHz), so very
    /// short files get an absolute floor of `min_tol_secs` instead.
    fn assert_close(actual: usize, expected_secs: f64, min_tol_secs: f64, what: &str) {
        let expected = expected_secs * WHISPER_SAMPLE_RATE as f64;
        let tol = (expected * 0.01).max(min_tol_secs * WHISPER_SAMPLE_RATE as f64);
        assert!(
            (actual as f64 - expected).abs() <= tol,
            "{what}: got {actual} samples, expected {expected:.0} ± {tol:.0}"
        );
    }

    #[test]
    fn decodes_generated_wavs_to_16k_frames() {
        let dir = tempfile::tempdir().unwrap();
        // (rate, channels, seconds, absolute tolerance floor)
        for (rate, ch, secs, min_tol) in [
            (44_100, 2, 20.0, 0.0),
            (48_000, 1, 15.0, 0.0),
            (16_000, 1, 15.3, 0.0),
            (8_000, 2, 20.0, 0.0),
            (22_050, 1, 0.2, 0.15),
        ] {
            let path = dir.path().join(format!("t_{rate}_{ch}.wav"));
            write_wav(&path, rate, ch, secs);
            let d = decode_all(&path);
            let what = format!("{rate} Hz x{ch}");
            let dur = d.duration.expect("wav has a known duration");
            assert!((dur - secs).abs() < 1e-3, "{what}: duration {dur}");
            assert!((d.decoded_secs - secs).abs() < 1e-3, "{what}");
            assert_close(d.samples_16k, secs, min_tol, &what);
            // 0.5-amplitude sine → RMS ≈ 0.354 (padding lowers it slightly).
            if secs >= 1.0 {
                assert!(d.rms > 0.3 && d.rms < 0.4, "{what}: rms {}", d.rms);
            }
        }
    }

    #[test]
    fn rejects_non_audio_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not_audio.mp3");
        std::fs::write(&path, b"definitely not an mp3 file").unwrap();
        assert!(StreamingDecoder::open(&path).is_err());
    }

    /// `<project>/temp/audio/` (outside the repo), or `SELTRANSLATE_AUDIO_DIR`.
    fn audio_dir() -> PathBuf {
        std::env::var_os("SELTRANSLATE_AUDIO_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../temp/audio"))
    }

    /// Length encoded in a test file name, e.g. `clip_10min.mp3` → 600 s.
    fn name_hint_secs(path: &Path) -> Option<f64> {
        let stem = path.file_stem()?.to_str()?;
        let mins = stem.rsplit('_').next()?.strip_suffix("min")?;
        mins.parse::<f64>().ok().map(|m| m * 60.0)
    }

    /// Real files from `temp/audio/` — the short clips and `clip_10min.mp3`;
    /// `long_*` (75 min) are for the manual A5 run only. Each must decode to
    /// the container's duration (or the one in its name) ± 1%, ± 0.15 s for
    /// short clips. Skipped when the folder is absent.
    #[test]
    fn decodes_files_from_temp_audio() {
        let dir = audio_dir();
        let Ok(entries) = std::fs::read_dir(&dir) else {
            eprintln!("skipping: {} not found", dir.display());
            return;
        };
        let mut checked = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if !path.is_file() || name.starts_with("long_") {
                continue;
            }
            let started = std::time::Instant::now();
            let d = decode_all(&path);
            let what = path.display().to_string();
            eprintln!(
                "{name}: container {:?}s, decoded {:.3}s, {} samples @16k, {:?}",
                d.duration,
                d.decoded_secs,
                d.samples_16k,
                started.elapsed()
            );
            let Some(expected) = d.duration.or_else(|| name_hint_secs(&path)) else {
                eprintln!("{name}: no known duration, only checked that it decodes");
                continue;
            };
            assert!(
                (d.decoded_secs - expected).abs() <= (expected * 0.01).max(0.06),
                "{what}: decoded {:.3}s vs expected {expected:.3}s",
                d.decoded_secs
            );
            assert_close(d.samples_16k, expected, 0.15, &what);
            checked += 1;
        }
        eprintln!("checked {checked} file(s) in {}", dir.display());
    }

    /// Garbage in the middle of an mp3 and a cut-off end: corrupt packets are
    /// skipped, the decode finishes without an error or a panic.
    #[test]
    fn survives_corrupt_and_truncated_mp3() {
        let src = audio_dir().join("clip_10min.mp3");
        let Ok(bytes) = std::fs::read(&src) else {
            eprintln!("skipping: {} not found", src.display());
            return;
        };
        assert!(bytes.len() > 1_600_000, "clip_10min.mp3 is too short");
        let mut bytes = bytes[..bytes.len().min(2_000_000)].to_vec();
        let mut x: u32 = 12345;
        for at in [300_000, 900_000, 1_500_000] {
            for b in &mut bytes[at..at + 8_192] {
                x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                *b = (x >> 16) as u8;
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt.mp3");
        std::fs::write(&path, &bytes).unwrap();

        let mut d = StreamingDecoder::open(&path).unwrap();
        let mut n = 0usize;
        while d
            .next_frames(|f| n += f.len())
            .expect("decode must not fail")
        {}
        eprintln!(
            "corrupt.mp3: decoded {:.1}s, {n} samples @16k",
            d.decoded_secs()
        );
        assert!(
            d.decoded_secs() > 60.0,
            "decoded only {:.1}s",
            d.decoded_secs()
        );
    }

    /// Audio track of a video file. Run by hand:
    /// `SELTRANSLATE_VIDEO=C:\path\movie.mp4 cargo test --lib video -- --ignored --nocapture`
    /// (`SELTRANSLATE_VIDEO_SECS` limits the decoded length, default 600 s).
    #[test]
    #[ignore]
    fn decodes_audio_track_of_video() {
        let path =
            PathBuf::from(std::env::var_os("SELTRANSLATE_VIDEO").expect("SELTRANSLATE_VIDEO"));
        let limit: f64 = std::env::var("SELTRANSLATE_VIDEO_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(600.0);
        let started = std::time::Instant::now();
        let mut d = StreamingDecoder::open(&path).unwrap();
        eprintln!(
            "{}: codec {}, {} Hz x{}, container {:?}s",
            path.display(),
            d.codec,
            d.sample_rate,
            d.channels,
            d.duration_secs
        );
        let (mut n, mut sq, mut bad_frame) = (0usize, 0f64, false);
        while d.decoded_secs() < limit
            && d.next_frames(|f| {
                bad_frame |= f.len() != FRAME_SAMPLES;
                n += f.len();
                sq += f.iter().map(|s| (*s as f64).powi(2)).sum::<f64>();
            })
            .unwrap()
        {}
        let rms = (sq / n.max(1) as f64).sqrt();
        eprintln!(
            "decoded {:.1}s → {n} samples @16k, rms {rms:.4}, {:?}",
            d.decoded_secs(),
            started.elapsed()
        );
        assert!(!bad_frame);
        assert!(rms > 1e-3, "audio track is silent");
        // Stopped mid-stream: no resampler flush yet, so allow its delay.
        assert_close(n, d.decoded_secs(), 0.15, "video audio");
    }
}
