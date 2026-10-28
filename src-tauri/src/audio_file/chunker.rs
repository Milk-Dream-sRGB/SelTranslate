//! Cuts a stream of 30 ms / 480-sample frames into 20–30 s chunks at pauses,
//! using a raw (unsmoothed) Silero VAD decision per frame.
//!
//! - chunk ≥ 20 s and a pause ≥ 300 ms ends → cut in the middle of the pause;
//! - chunk reaches 30 s → cut in the middle of the longest pause that starts
//!   its middle ≥ 10 s into the chunk (any length, including one still
//!   running); the tail moves to the next chunk; no pause at all → hard cut;
//! - chunks with < 0.5 s of speech, or < 2 s of speech scattered at < 35 %
//!   density over its span (a jingle, noise bursts), are dropped (Whisper
//!   hallucinates on them); a leading silence > 2 s with no speech before it is dropped
//!   too, keeping 1 s before the speech.

use std::path::Path;

use anyhow::Result;

use super::decoder::FRAME_SAMPLES;
use crate::audio_toolkit::{SileroVad, VoiceActivityDetector};

const FRAME_MS: usize = 30;
const fn frames(ms: usize) -> usize {
    ms.div_ceil(FRAME_MS)
}

pub const MIN_CHUNK_FRAMES: usize = frames(20_000);
pub const MAX_CHUNK_FRAMES: usize = frames(30_000);
pub const MIN_SILENCE_CUT_FRAMES: usize = frames(300);
pub const MIN_SPEECH_FRAMES: usize = frames(500);
/// A chunk with less speech than this *and* a sparse speech span is treated
/// as non-speech sound (jingle, noise bursts): Whisper hallucinates on it.
pub const SPARSE_SPEECH_FRAMES: usize = frames(2_000);
/// Speech share (percent) of the span between the first and the last speech
/// frame below which a chunk with < `SPARSE_SPEECH_FRAMES` is dropped.
pub const SPARSE_SPAN_DENSITY_PCT: usize = 35;
/// Fallback cuts are only considered this far into a chunk, so a forced cut
/// never leaves a tiny chunk behind.
const CANDIDATE_START_FRAMES: usize = frames(10_000);
/// Silence kept before the first speech when leading silence is trimmed.
const LEAD_KEEP_FRAMES: usize = frames(1_000);

/// Raw Silero speech-probability threshold (no smoothing — offline file).
pub const VAD_THRESHOLD: f32 = 0.5;

#[derive(Debug)]
pub struct Chunk {
    /// 16 kHz mono samples, at most 30 s.
    pub samples: Vec<f32>,
    /// Offset of the first sample from the start of the file.
    pub start_secs: f64,
    pub speech_frames: usize,
}

impl Chunk {
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / 16_000.0
    }
}

/// Pure cutting logic over (frame, is_speech) pairs.
#[derive(Default)]
pub struct Chunker {
    samples: Vec<f32>,
    speech: Vec<bool>,
    /// Trailing non-speech frames of the current chunk.
    silence_run: usize,
    /// Frames already handed out (or dropped) — start of the current chunk.
    consumed_frames: usize,
}

impl Chunker {
    pub fn push(&mut self, frame: &[f32], is_speech: bool) -> Option<Chunk> {
        debug_assert_eq!(frame.len(), FRAME_SAMPLES);
        let mut out = None;

        if is_speech && self.silence_run >= MIN_SILENCE_CUT_FRAMES {
            // A pause just ended at frame `n`; cut in its middle if the
            // chunk is long enough by then.
            let n = self.speech.len();
            let cut = n - self.silence_run / 2;
            if cut >= MIN_CHUNK_FRAMES {
                out = self.cut_at(cut);
            } else if self.silence_run > 2 * LEAD_KEEP_FRAMES && !self.speech.contains(&true) {
                // Long leading silence: drop it, keep ≤ 1 s before speech.
                out = self.cut_at(n - LEAD_KEEP_FRAMES);
                debug_assert!(out.is_none());
            }
        }

        self.samples.extend_from_slice(frame);
        self.speech.push(is_speech);
        self.silence_run = if is_speech { 0 } else { self.silence_run + 1 };

        if self.speech.len() >= MAX_CHUNK_FRAMES {
            let cut = best_cut(&self.speech).unwrap_or(self.speech.len());
            // At most one cut per pushed frame: the mid-pause cut above
            // leaves < MAX frames behind, so both can't fire together.
            debug_assert!(out.is_none());
            out = self.cut_at(cut);
        }
        out
    }

    /// Hands out the remainder at end of stream.
    pub fn finish(&mut self) -> Option<Chunk> {
        let n = self.speech.len();
        if n == 0 {
            return None;
        }
        self.cut_at(n)
    }

    /// Emits frames `[0, cut)` as a chunk (or drops them if they hold too
    /// little speech) and keeps the rest as the start of the next chunk.
    fn cut_at(&mut self, cut: usize) -> Option<Chunk> {
        let tail_samples = self.samples.split_off(cut * FRAME_SAMPLES);
        let tail_speech = self.speech.split_off(cut);
        let samples = std::mem::replace(&mut self.samples, tail_samples);
        let speech = std::mem::replace(&mut self.speech, tail_speech);
        self.silence_run = self.speech.iter().rev().take_while(|s| !**s).count();

        let start_frame = self.consumed_frames;
        self.consumed_frames += cut;
        let speech_frames = speech.iter().filter(|s| **s).count();
        let start_secs = (start_frame * FRAME_MS) as f64 / 1000.0;
        let sparse = match (
            speech.iter().position(|s| *s),
            speech.iter().rposition(|s| *s),
        ) {
            (Some(first), Some(last)) => {
                let span = last - first + 1;
                speech_frames < SPARSE_SPEECH_FRAMES
                    && speech_frames * 100 < span * SPARSE_SPAN_DENSITY_PCT
            }
            _ => false,
        };
        if speech_frames < MIN_SPEECH_FRAMES || sparse {
            log::debug!(
                "Audio file: dropping {:.1}s chunk at {:.1}s ({} speech frames{})",
                cut as f64 * FRAME_MS as f64 / 1000.0,
                start_secs,
                speech_frames,
                if sparse { ", sparse" } else { "" }
            );
            return None;
        }
        Some(Chunk {
            samples,
            start_secs,
            speech_frames,
        })
    }
}

/// Middle of the longest non-speech run whose middle lies at or after
/// `CANDIDATE_START_FRAMES` (a run touching the end of the buffer counts).
fn best_cut(speech: &[bool]) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None; // (run length, cut)
    let mut run_start = None;
    for i in 0..=speech.len() {
        let silent = i < speech.len() && !speech[i];
        match (silent, run_start) {
            (true, None) => run_start = Some(i),
            (false, Some(s)) => {
                let len = i - s;
                let cut = i - len / 2;
                // Ties go to the later pause → longer chunks.
                if cut >= CANDIDATE_START_FRAMES && best.is_none_or(|(l, _)| len >= l) {
                    best = Some((len, cut));
                }
                run_start = None;
            }
            _ => {}
        }
    }
    best.map(|(_, cut)| cut)
}

/// `Chunker` fed by a raw Silero VAD.
pub struct VadChunker {
    vad: Box<dyn VoiceActivityDetector>,
    chunker: Chunker,
    vad_errors: usize,
}

impl VadChunker {
    pub fn new(vad_model: &Path) -> Result<Self> {
        let vad = SileroVad::new(vad_model, VAD_THRESHOLD)?;
        Ok(Self::with_vad(Box::new(vad)))
    }

    pub fn with_vad(mut vad: Box<dyn VoiceActivityDetector>) -> Self {
        vad.reset();
        Self {
            vad,
            chunker: Chunker::default(),
            vad_errors: 0,
        }
    }

    pub fn push_frame(&mut self, frame: &[f32]) -> Option<Chunk> {
        let is_speech = match self.vad.is_voice(frame) {
            Ok(v) => v,
            Err(e) => {
                // Treat as speech: never drop audio because the VAD failed.
                if self.vad_errors == 0 {
                    log::warn!("Audio file: VAD error, treating frame as speech: {e}");
                }
                self.vad_errors += 1;
                true
            }
        };
        self.chunker.push(frame, is_speech)
    }

    pub fn finish(&mut self) -> Option<Chunk> {
        self.chunker.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEECH: f32 = 0.5;

    /// Feeds a pattern of (seconds, is_speech) segments; each frame's first
    /// sample encodes its global index so cuts can be checked exactly.
    fn run(pattern: &[(f64, bool)]) -> (Vec<Chunk>, usize) {
        let mut c = Chunker::default();
        let mut out = Vec::new();
        let mut idx = 0usize;
        for &(secs, speech) in pattern {
            let n = (secs * 1000.0 / FRAME_MS as f64).round() as usize;
            for _ in 0..n {
                let mut f = vec![if speech { SPEECH } else { 0.0 }; FRAME_SAMPLES];
                f[0] = idx as f32;
                idx += 1;
                out.extend(c.push(&f, speech));
            }
        }
        out.extend(c.finish());
        (out, idx)
    }

    fn frame_index(chunk: &Chunk, frame: usize) -> usize {
        chunk.samples[frame * FRAME_SAMPLES] as usize
    }

    fn assert_contiguous(chunks: &[Chunk]) {
        for w in chunks.windows(2) {
            let end = frame_index(&w[0], w[0].samples.len() / FRAME_SAMPLES - 1);
            assert_eq!(frame_index(&w[1], 0), end + 1, "chunks must be gap-free");
        }
        for c in chunks {
            assert_eq!(c.samples.len() % FRAME_SAMPLES, 0);
            assert!(c.samples.len() <= MAX_CHUNK_FRAMES * FRAME_SAMPLES);
            let first = frame_index(c, 0);
            assert!((c.start_secs - first as f64 * 0.03).abs() < 1e-9);
        }
    }

    #[test]
    fn constants_in_frames() {
        assert_eq!(MIN_CHUNK_FRAMES, 667);
        assert_eq!(MAX_CHUNK_FRAMES, 1000);
        assert_eq!(MIN_SILENCE_CUT_FRAMES, 10);
        assert_eq!(MIN_SPEECH_FRAMES, 17);
        assert_eq!(SPARSE_SPEECH_FRAMES, 67);
    }

    #[test]
    fn short_file_is_one_chunk() {
        let (chunks, total) = run(&[(0.3, false), (5.0, true), (0.6, false)]);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].samples.len(), total * FRAME_SAMPLES);
        assert_eq!(chunks[0].start_secs, 0.0);
    }

    #[test]
    fn cuts_in_middle_of_pause_after_20s() {
        // Pause at 15 s is too early; the 0.6 s pause at 22 s is taken.
        let (chunks, _) = run(&[
            (15.0, true),
            (0.6, false),
            (6.4, true),
            (0.6, false),
            (10.0, true),
        ]);
        assert_eq!(chunks.len(), 2);
        assert_contiguous(&chunks);
        // Pause frames 733..753 → cut at 753 - 10 = 743 (22.29 s).
        assert_eq!(chunks[0].samples.len() / FRAME_SAMPLES, 743);
        assert!((chunks[1].start_secs - 22.29).abs() < 1e-9);
    }

    #[test]
    fn short_pauses_are_not_cut_before_30s() {
        // 0.21 s pauses (7 frames) don't qualify for the 20 s rule.
        let mut p = Vec::new();
        for _ in 0..8 {
            p.push((4.0, true));
            p.push((0.21, false));
        }
        let (chunks, _) = run(&p);
        assert!(chunks.len() >= 2);
        assert_contiguous(&chunks);
        // First cut falls back to the best pause before 30 s, i.e. in the
        // middle of one of the 7-frame pauses, not a hard cut.
        let first = &chunks[0];
        let n = first.samples.len() / FRAME_SAMPLES;
        assert!(n < MAX_CHUNK_FRAMES, "expected a pause cut, got {n}");
        let last = first.samples[(n - 1) * FRAME_SAMPLES + 1];
        let next = chunks[1].samples[1];
        assert_eq!((last, next), (0.0, 0.0), "cut must fall inside a pause");
    }

    #[test]
    fn continuous_speech_is_hard_cut_at_30s() {
        let (chunks, total) = run(&[(75.0, true)]);
        assert_eq!(total, 2500);
        let lens: Vec<_> = chunks
            .iter()
            .map(|c| c.samples.len() / FRAME_SAMPLES)
            .collect();
        assert_eq!(lens, vec![1000, 1000, 500]);
        assert_contiguous(&chunks);
    }

    #[test]
    fn longest_pause_wins_the_30s_fallback() {
        // Pauses of 0.15 s at 12 s and 0.27 s at 25 s, both < 300 ms.
        let (chunks, _) = run(&[
            (12.0, true),
            (0.15, false),
            (13.0, true),
            (0.27, false),
            (10.0, true),
        ]);
        assert_contiguous(&chunks);
        // 0.27 s pause = frames 838..847 (9 frames) → cut 847 - 4 = 843.
        assert_eq!(chunks[0].samples.len() / FRAME_SAMPLES, 843);
    }

    #[test]
    fn pause_before_10s_is_ignored_by_fallback() {
        let (chunks, _) = run(&[(5.0, true), (0.27, false), (40.0, true)]);
        assert_eq!(chunks[0].samples.len() / FRAME_SAMPLES, MAX_CHUNK_FRAMES);
    }

    #[test]
    fn silence_only_chunks_are_dropped_but_time_advances() {
        let (chunks, _) = run(&[(65.0, false), (8.0, true), (0.6, false)]);
        // Leading silence is trimmed to 1 s before the speech at 65 s.
        assert_eq!(chunks.len(), 1);
        assert!((chunks[0].start_secs - 63.99).abs() < 1e-9);
        assert!(chunks[0].speech_frames >= 260);
        let first = frame_index(&chunks[0], 0);
        assert!((chunks[0].start_secs - first as f64 * 0.03).abs() < 1e-9);
    }

    #[test]
    fn little_speech_is_dropped() {
        let (chunks, _) = run(&[(3.0, false), (0.45, true), (3.0, false)]);
        assert!(chunks.is_empty());
        let (chunks, _) = run(&[(3.0, false), (0.51, true), (3.0, false)]);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn scattered_bursts_are_dropped_but_short_dense_speech_is_kept() {
        // A jingle: 0.3 s bursts over ~4 s (0.9 s of "speech", ~22 %) and
        // then quiet — the 16 s chunk at the start of clip_10min.mp3.
        let mut p = Vec::new();
        for _ in 0..3 {
            p.push((0.3, true));
            p.push((1.0, false));
        }
        p.push((12.0, false));
        let (chunks, _) = run(&p);
        assert!(chunks.is_empty());

        // One short word is dense in its own span and stays.
        let (chunks, _) = run(&[(10.0, false), (0.6, true), (10.0, false)]);
        assert_eq!(chunks.len(), 1);

        // The same burst pattern next to 2+ s of speech is kept.
        let mut p = vec![(1.0, false)];
        for _ in 0..3 {
            p.push((0.3, true));
            p.push((1.0, false));
        }
        p.push((2.5, true));
        let (chunks, _) = run(&p);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn long_recording_keeps_bounded_chunks() {
        // 10 minutes: 7 s of speech, 0.4 s pause, repeated.
        let mut p = Vec::new();
        while p.len() < 2 * 81 {
            p.push((7.0, true));
            p.push((0.4, false));
        }
        let (chunks, total) = run(&p);
        assert_contiguous(&chunks);
        let covered: usize = chunks.iter().map(|c| c.samples.len() / FRAME_SAMPLES).sum();
        assert_eq!(covered, total);
        for c in &chunks[..chunks.len() - 1] {
            let n = c.samples.len() / FRAME_SAMPLES;
            assert!((MIN_CHUNK_FRAMES..=MAX_CHUNK_FRAMES).contains(&n), "{n}");
        }
    }

    #[test]
    fn silero_drops_pure_silence() {
        let model =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/models/silero_vad_v4.onnx");
        let mut c = VadChunker::new(&model).unwrap();
        let silence = vec![0.0f32; FRAME_SAMPLES];
        let mut chunks = 0;
        for _ in 0..(45_000 / FRAME_MS) {
            chunks += c.push_frame(&silence).is_some() as usize;
        }
        chunks += c.finish().is_some() as usize;
        assert_eq!(chunks, 0);
    }

    /// Decoder + Silero + chunker on a real file, no recognition model.
    /// Run by hand:
    /// `SELTRANSLATE_FILE=C:\path\a.m4a cargo test --lib chunks_real_file -- --ignored --nocapture`
    /// (`SELTRANSLATE_FILE_SECS` limits the decoded length; default: whole file).
    #[test]
    #[ignore]
    fn chunks_real_file() {
        use crate::audio_file::decoder::StreamingDecoder;

        let path = std::path::PathBuf::from(
            std::env::var_os("SELTRANSLATE_FILE").expect("SELTRANSLATE_FILE"),
        );
        let limit: f64 = std::env::var("SELTRANSLATE_FILE_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(f64::MAX);
        let model =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/models/silero_vad_v4.onnx");
        let started = std::time::Instant::now();
        let mut decoder = StreamingDecoder::open(&path).unwrap();
        let threshold: f32 = std::env::var("SELTRANSLATE_VAD_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(VAD_THRESHOLD);
        let mut chunker =
            VadChunker::with_vad(Box::new(SileroVad::new(&model, threshold).unwrap()));
        let gain: f32 = std::env::var("SELTRANSLATE_GAIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        eprintln!("VAD threshold {threshold}, gain {gain}");
        eprintln!(
            "{}: {} {} Hz x{}, container {:?}s",
            path.display(),
            decoder.codec,
            decoder.sample_rate,
            decoder.channels,
            decoder.duration_secs
        );

        // (start, duration) of every chunk handed to recognition. The samples
        // are dropped at once: memory must not depend on the file length.
        let mut spans: Vec<(f64, f64)> = Vec::new();
        // Mean square per 10 s of audio, to judge dropped stretches by ear-less means.
        let mut sq: Vec<(f64, usize)> = Vec::new();
        let mut frames_seen = 0usize;
        while decoder.decoded_secs() < limit {
            let mut ready = Vec::new();
            let more = decoder
                .next_frames(|f| {
                    let bin = frames_seen / 333;
                    if sq.len() <= bin {
                        sq.resize(bin + 1, (0.0, 0));
                    }
                    sq[bin].0 += f.iter().map(|s| (*s as f64).powi(2)).sum::<f64>();
                    sq[bin].1 += f.len();
                    frames_seen += 1;
                    // Quiet-recording simulation (SELTRANSLATE_GAIN, e.g. 0.05).
                    let quiet: Vec<f32> = f.iter().map(|s| s * gain).collect();
                    ready.extend(chunker.push_frame(&quiet));
                })
                .unwrap();
            spans.extend(ready.iter().map(|c| (c.start_secs, c.duration_secs())));
            if !more {
                spans.extend(chunker.finish().map(|c| (c.start_secs, c.duration_secs())));
                break;
            }
        }

        let audio = decoder.decoded_secs();
        let sent: f64 = spans.iter().map(|s| s.1).sum();
        let (min, max) = spans
            .iter()
            .fold((f64::MAX, 0.0f64), |(a, b), s| (a.min(s.1), b.max(s.1)));
        let over_30 = spans.iter().filter(|s| s.1 > 30.0 + 1e-6).count();
        // Every chunk but the last is cut at >= 20 s.
        let under_20 = spans.iter().filter(|s| s.1 < 20.0).count();
        eprintln!(
            "audio {audio:.1}s, {} chunks, sent {sent:.1}s ({:.1}%), min {min:.2}s max {max:.2}s, \
             <20s: {under_20}, >30s: {over_30}, took {:.1}s",
            spans.len(),
            100.0 * sent / audio.max(1e-9),
            started.elapsed().as_secs_f64()
        );
        // Stretches that never reached recognition (no speech > 0.5 s).
        let mut prev_end = 0.0;
        let mut dropped: Vec<(f64, f64)> = Vec::new();
        for (start, dur) in &spans {
            if start - prev_end > 0.05 {
                dropped.push((prev_end, start - prev_end));
            }
            prev_end = start + dur;
        }
        if audio - prev_end > 0.05 {
            dropped.push((prev_end, audio - prev_end));
        }
        let dropped_secs: f64 = dropped.iter().map(|d| d.1).sum();
        let rms: Vec<String> = sq
            .iter()
            .map(|(s, n)| format!("{:.3}", (s / (*n).max(1) as f64).sqrt()))
            .collect();
        for (i, row) in rms.chunks(12).enumerate() {
            eprintln!("rms/10s from {:>5}s: {}", i * 120, row.join(" "));
        }
        eprintln!(
            "dropped: {} stretches, {dropped_secs:.1}s; first: {:?}",
            dropped.len(),
            dropped.iter().take(12).collect::<Vec<_>>()
        );
        assert_eq!(over_30, 0);
    }
}
