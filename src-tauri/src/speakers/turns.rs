//! From Sortformer's speaker segments to the turns we transcribe one by one.
//! Pure logic, no models.
//!
//! Sortformer returns segments grouped by speaker; segments of different
//! speakers may overlap, and a short "mm" inside someone's sentence is its own
//! segment. A transcription needs a single track of turns that do not
//! overlap and are long enough to carry words, so:
//!
//! 1. With a speaker count from the user, the speakers who talk least beyond
//!    that count are folded into the nearest kept speaker in time (they are
//!    most often the diarizer splitting one voice).
//! 2. Overlaps are resolved: a segment inside another one is dropped (its
//!    audio is in the outer turn anyway), a partial overlap is cut in the
//!    middle.
//! 3. Same-speaker neighbours closer than [`MERGE_GAP_MS`] are joined, and
//!    turns shorter than [`MIN_TURN_MS`] are absorbed by a neighbour.
//! 4. Speakers are numbered 1, 2, … in order of their first turn.

use std::collections::BTreeMap;

/// Same-speaker turns closer than this become one turn.
pub const MERGE_GAP_MS: i64 = 1_500;
/// Shorter turns are too short for a model to transcribe on their own.
pub const MIN_TURN_MS: i64 = 600;
/// Audio taken around a turn so its first and last words are not clipped;
/// never past the middle of the gap to a neighbouring turn.
pub const PAD_MS: i64 = 250;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    /// 1-based, as the diarizer numbers them.
    pub speaker: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Turn {
    pub start_ms: i64,
    pub end_ms: i64,
    /// 1-based, in order of first appearance.
    pub speaker: u32,
}

impl Turn {
    fn len(&self) -> i64 {
        self.end_ms - self.start_ms
    }
}

/// The turns of a recording `audio_ms` long, from the diarizer's `segments`.
/// `speakers`: how many people the user says are talking, if they said.
pub fn plan(segments: &[Segment], speakers: Option<u32>, audio_ms: i64) -> Vec<Turn> {
    let mut segs: Vec<Segment> = segments
        .iter()
        .map(|s| Segment {
            start_ms: s.start_ms.clamp(0, audio_ms),
            end_ms: s.end_ms.clamp(0, audio_ms),
            speaker: s.speaker,
        })
        .filter(|s| s.end_ms > s.start_ms && s.speaker > 0)
        .collect();
    segs.sort_by_key(|s| (s.start_ms, s.end_ms));
    if let Some(limit) = speakers.filter(|&n| n > 0) {
        fold_extra_speakers(&mut segs, limit as usize);
    }

    let mut turns = resolve_overlaps(&segs);
    merge_same_speaker(&mut turns);
    absorb_short(&mut turns);
    merge_same_speaker(&mut turns);
    renumber(&mut turns);
    turns
}

/// The audio range of turn `i` with padding, in ms.
pub fn padded_range(turns: &[Turn], i: usize, audio_ms: i64) -> (i64, i64) {
    let turn = turns[i];
    let floor = match i.checked_sub(1).map(|p| turns[p]) {
        Some(prev) => (prev.end_ms + turn.start_ms) / 2,
        None => 0,
    };
    let ceil = match turns.get(i + 1) {
        Some(next) => (turn.end_ms + next.start_ms) / 2,
        None => audio_ms,
    };
    (
        (turn.start_ms - PAD_MS).max(floor.min(turn.start_ms)),
        (turn.end_ms + PAD_MS).min(ceil.max(turn.end_ms)),
    )
}

/// Relabels the segments of all but the `limit` most talkative speakers with
/// the speaker of the nearest kept segment.
fn fold_extra_speakers(segs: &mut [Segment], limit: usize) {
    let mut talk: BTreeMap<i32, i64> = BTreeMap::new();
    for s in segs.iter() {
        *talk.entry(s.speaker).or_default() += s.end_ms - s.start_ms;
    }
    if talk.len() <= limit {
        return;
    }
    let mut ranked: Vec<(i32, i64)> = talk.into_iter().collect();
    ranked.sort_by_key(|&(speaker, ms)| (std::cmp::Reverse(ms), speaker));
    let kept: Vec<i32> = ranked.iter().take(limit).map(|&(s, _)| s).collect();
    let anchors: Vec<Segment> = segs
        .iter()
        .copied()
        .filter(|s| kept.contains(&s.speaker))
        .collect();
    for s in segs.iter_mut().filter(|s| !kept.contains(&s.speaker)) {
        let distance = |a: &Segment| {
            if a.end_ms < s.start_ms {
                s.start_ms - a.end_ms
            } else if a.start_ms > s.end_ms {
                a.start_ms - s.end_ms
            } else {
                0
            }
        };
        if let Some(nearest) = anchors.iter().min_by_key(|a| distance(a)) {
            s.speaker = nearest.speaker;
        }
    }
}

/// One track from time-sorted segments that may overlap.
fn resolve_overlaps(segs: &[Segment]) -> Vec<Turn> {
    let mut turns: Vec<Turn> = Vec::new();
    for s in segs {
        let mut turn = Turn {
            start_ms: s.start_ms,
            end_ms: s.end_ms,
            speaker: s.speaker as u32,
        };
        if let Some(last) = turns.last_mut() {
            if turn.start_ms < last.end_ms {
                if turn.end_ms <= last.end_ms {
                    continue;
                }
                if turn.speaker == last.speaker {
                    last.end_ms = turn.end_ms;
                    continue;
                }
                let middle = (turn.start_ms + last.end_ms) / 2;
                last.end_ms = middle;
                turn.start_ms = middle;
            }
        }
        turns.push(turn);
    }
    turns.retain(|t| t.len() > 0);
    turns
}

fn merge_same_speaker(turns: &mut Vec<Turn>) {
    let mut out: Vec<Turn> = Vec::with_capacity(turns.len());
    for &turn in turns.iter() {
        match out.last_mut() {
            Some(last)
                if last.speaker == turn.speaker && turn.start_ms - last.end_ms < MERGE_GAP_MS =>
            {
                last.end_ms = last.end_ms.max(turn.end_ms);
            }
            _ => out.push(turn),
        }
    }
    *turns = out;
}

/// A turn shorter than [`MIN_TURN_MS`] joins the neighbour it is closer to
/// (the previous one on a tie), which takes over its audio.
fn absorb_short(turns: &mut Vec<Turn>) {
    let mut i = 0;
    while turns.len() > 1 && i < turns.len() {
        if turns[i].len() >= MIN_TURN_MS {
            i += 1;
            continue;
        }
        let gap_before = i
            .checked_sub(1)
            .map(|p| turns[i].start_ms - turns[p].end_ms);
        let gap_after = turns.get(i + 1).map(|n| n.start_ms - turns[i].end_ms);
        let short = turns.remove(i);
        match (gap_before, gap_after) {
            (Some(before), Some(after)) if after < before => {
                turns[i].start_ms = short.start_ms;
            }
            (Some(_), _) => {
                turns[i - 1].end_ms = short.end_ms;
                i -= 1;
            }
            (None, _) => {
                turns[i].start_ms = short.start_ms;
            }
        }
    }
}

fn renumber(turns: &mut [Turn]) {
    let mut order: Vec<u32> = Vec::new();
    for t in turns.iter_mut() {
        let n = match order.iter().position(|&s| s == t.speaker) {
            Some(p) => p,
            None => {
                order.push(t.speaker);
                order.len() - 1
            }
        };
        t.speaker = n as u32 + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start_ms: i64, end_ms: i64, speaker: i32) -> Segment {
        Segment {
            start_ms,
            end_ms,
            speaker,
        }
    }

    fn turn(start_ms: i64, end_ms: i64, speaker: u32) -> Turn {
        Turn {
            start_ms,
            end_ms,
            speaker,
        }
    }

    #[test]
    fn a_plain_dialogue_stays_as_it_is_and_is_numbered_by_arrival() {
        let segs = [
            seg(5_000, 9_000, 1),
            seg(0, 4_000, 2),
            seg(10_000, 12_000, 2),
        ];
        assert_eq!(
            plan(&segs, None, 20_000),
            [
                turn(0, 4_000, 1),
                turn(5_000, 9_000, 2),
                turn(10_000, 12_000, 1)
            ]
        );
    }

    #[test]
    fn a_backchannel_inside_a_turn_is_dropped_and_a_partial_overlap_is_cut_in_the_middle() {
        let segs = [
            seg(0, 10_000, 1),
            seg(3_000, 3_400, 2),
            seg(9_000, 15_000, 2),
        ];
        assert_eq!(
            plan(&segs, None, 20_000),
            [turn(0, 9_500, 1), turn(9_500, 15_000, 2)]
        );
    }

    #[test]
    fn close_same_speaker_turns_join_and_short_turns_are_absorbed() {
        let segs = [
            seg(0, 3_000, 1),
            seg(3_800, 6_000, 1),
            // 300 ms "yes" of someone else, closer to the next turn.
            seg(9_000, 9_300, 2),
            seg(9_500, 12_000, 1),
        ];
        // The 3 s pause before the "yes" still separates two paragraphs.
        assert_eq!(
            plan(&segs, None, 20_000),
            [turn(0, 6_000, 1), turn(9_000, 12_000, 1)]
        );
    }

    #[test]
    fn a_speaker_count_folds_the_quietest_speakers_into_their_neighbours() {
        let segs = [
            seg(0, 5_000, 1),
            seg(6_000, 11_000, 2),
            // A third "speaker" for 2 s right after speaker 2: the same voice.
            seg(12_000, 14_000, 3),
            seg(20_000, 25_000, 1),
        ];
        assert_eq!(plan(&segs, None, 30_000).len(), 4);
        assert_eq!(
            plan(&segs, Some(2), 30_000),
            [
                turn(0, 5_000, 1),
                turn(6_000, 14_000, 2),
                turn(20_000, 25_000, 1)
            ]
        );
    }

    #[test]
    fn segments_are_clamped_to_the_audio_and_empty_ones_dropped() {
        let segs = [
            seg(-500, 2_000, 1),
            seg(3_000, 3_000, 2),
            seg(2_500, 99_000, 0),
        ];
        assert_eq!(plan(&segs, None, 10_000), [turn(0, 2_000, 1)]);
        assert!(plan(&[], Some(2), 10_000).is_empty());
    }

    #[test]
    fn padding_stops_halfway_to_the_neighbours() {
        let turns = [
            turn(100, 2_000, 1),
            turn(2_200, 5_000, 2),
            turn(8_000, 9_900, 1),
        ];
        assert_eq!(padded_range(&turns, 0, 10_000), (0, 2_100));
        assert_eq!(padded_range(&turns, 1, 10_000), (2_100, 5_250));
        assert_eq!(padded_range(&turns, 2, 10_000), (7_750, 10_000));
    }
}
