//! Sortformer through transcribe.cpp, in this process on CPU, like Handy's
//! language detector (`language_detection::detector`): no GPU driver code,
//! and the speech model in the engine worker is left alone. It is loaded for
//! one run and dropped right after; at ~76x real time on CPU (model card,
//! Apple M4) the load is a small part of a run.

use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, Result};
use log::info;
use transcribe_cpp::{
    Backend, CancelToken, DiarizeExtension, DiarizeOptions, Error, Model, ModelOptions,
    SortformerDiarizeOptions, SortformerPreset,
};

use super::turns::Segment;
use crate::audio_toolkit::cpu_engine::init_cpu_backends;

/// What a run ended with.
pub enum Outcome {
    Segments(Vec<Segment>),
    Cancelled,
}

/// Speaker segments of 16 kHz mono `pcm`, with the model at `path`.
/// `cancel` aborts a run in progress from another thread.
pub fn diarize(path: &Path, pcm: &[f32], cancel: &CancelToken) -> Result<Outcome> {
    init_cpu_backends()?;
    let started = Instant::now();
    let model = Model::load_with(
        path,
        &ModelOptions {
            backend: Backend::Cpu,
            device: None,
        },
    )
    .map_err(|e| anyhow!("Failed to load the speaker model: {e}"))?;
    let mut session = model
        .diarize_session()
        .map_err(|e| anyhow!("The file is not a speaker model: {e}"))?;
    session.set_cancel_token(cancel);
    let options = DiarizeOptions {
        // The offline-file operating point (~30 s lookahead): the most
        // accurate one, and what the model card's DER is measured at.
        family: Some(DiarizeExtension::Sortformer(SortformerDiarizeOptions {
            preset: Some(SortformerPreset::VeryHighLatency),
        })),
    };
    let segments = match session.run(pcm, &options) {
        Ok(segments) => segments,
        Err(Error::Aborted { .. }) => return Ok(Outcome::Cancelled),
        Err(e) => return Err(anyhow!("Speaker detection failed: {e}")),
    };
    info!(
        "Speakers: {} segments in {:.1} s of audio, {} ms",
        segments.len(),
        pcm.len() as f64 / 16_000.0,
        started.elapsed().as_millis()
    );
    Ok(Outcome::Segments(
        segments
            .into_iter()
            .map(|s| Segment {
                start_ms: s.t0_ms,
                end_ms: s.t1_ms,
                speaker: s.speaker_id,
            })
            .collect(),
    ))
}
