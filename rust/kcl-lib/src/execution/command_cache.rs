//! Content-addressed matching of a previous run's command log against a new
//! run's, so unchanged commands can be skipped (not re-sent to the engine)
//! and only genuinely new or superseded ones are acted on.
//!
//! Nothing calls into this module yet - it's the matching algorithm on its
//! own, validated by its own tests, ahead of the separate work of wiring it
//! into real `ModelingCmd` sending (see the surgical-delete follow-up plan).
//! Hence the module-wide `dead_code` allow below.
//!
//! This module is deliberately generic over what a "command" actually is
//! (see [`CommandLog`]'s `Payload` type parameter): it implements only the
//! matching *algorithm* - given a sequence of content ids from a previous
//! run and a new sequence arriving one at a time, decide for each new one
//! whether it's a hit (reuse the old result, nothing to send) or a miss
//! (send it for real), and which old entries turned out to have no
//! counterpart in the new run at all (delete them). Turning an actual
//! `ModelingCmd`/response pair into a [`ContentId`], and acting on the
//! decisions this module returns, is the job of its caller.
//!
//! # The matching algorithm
//!
//! A [`CommandLog`] holds the previous run's entries in their original
//! order, plus an index from content id to the (ascending) queue of
//! not-yet-consumed positions that have it. Feeding it the new run's content
//! ids one at a time, in order, via [`CommandLog::match_next`]:
//!
//! - If the next new content id matches the entry the log is currently
//!   positioned at (`cursor`), that's a direct hit: nothing was deleted,
//!   reuse that entry, advance.
//! - Otherwise, look the new content id up in the index. If it's found at
//!   some later position, every old entry strictly between the cursor and
//!   that position has no counterpart in the new run - they're superseded
//!   (to be deleted) - and the found position is the hit.
//! - If the content id isn't in the index at all, it's genuinely new (a
//!   miss): the caller must send it for real. Whatever entry the log was
//!   positioned at, if any, had no counterpart either and is superseded.
//!
//! Every position the log advances past is removed from the index as it's
//! consumed, so a later new content id can never incorrectly match a
//! position that's already been assigned a disposition (reused or deleted).
//!
//! This module does not know about ids being shared between two logically
//! different objects (e.g. `extrude()` reusing its input's engine id) - a
//! caller that cares about that must inspect what [`MatchDecision::Hit`]
//! reuses and what it supersedes, and override a hit into a forced miss if
//! something it's about to delete is also what a hit is relying on.
//!
//! # A known, deliberate limitation: insertions before an unchanged tail
//!
//! This is a true single-pass streaming match: new content ids arrive one
//! at a time, and a decision is made immediately, without buffering or
//! looking ahead into new content that hasn't been computed yet (doing so
//! would mean running the interpreter further ahead speculatively, which
//! isn't free the way looking ahead into the already-fully-known old log
//! is). So when a new content id doesn't match `old_log[cursor]` and isn't
//! found anywhere later in the old log either, this is reported as a miss
//! that supersedes `old_log[cursor]` - there's no way, at that point, to
//! tell "this is genuinely new content, unrelated to anything, and
//! `old_log[cursor]` will turn out to still be needed a moment later"
//! (a pure insertion) apart from "this is what replaced `old_log[cursor]`"
//! (the case the worked example in the module's design doc is about) -
//! both look identical from this algorithm's point of view at the moment
//! of the mismatch. The practical effect: an item inserted just before an
//! otherwise-unchanged tail causes the one shifted item right after it to
//! be needlessly deleted and recreated (with a fresh id) a step later,
//! instead of being left alone. This costs one avoidable round trip; it
//! does not produce an incorrect final result, since the recreated item's
//! content is identical to what was there before.

#![allow(dead_code)]

use std::collections::HashMap;
use std::collections::VecDeque;

use sha2::Digest as _;
use sha2::Sha256;

/// A content-addressed identity for one logged entry. Two entries (from the
/// same run or across two different runs) with the same `ContentId` are
/// guaranteed to produce the same result, so the second one doesn't need to
/// be redone - see the module docs for how a [`CommandLog`] uses this.
pub(crate) type ContentId = [u8; 32];

/// Hash arbitrary bytes into a [`ContentId`]. Callers building a real
/// `ContentId` for a command should hash its own fields together with the
/// `ContentId` (not the real engine id, which isn't stable across runs) of
/// anything it references, so a change anywhere upstream changes every
/// `ContentId` downstream of it.
pub(crate) fn content_id_from_bytes(data: &[u8]) -> ContentId {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// One entry in a [`CommandLog`], at the position it was created.
#[derive(Debug, Clone)]
pub(crate) struct LogEntry<Payload> {
    pub content_id: ContentId,
    pub payload: Payload,
}

/// The result of matching one new content id against a [`CommandLog`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MatchDecision {
    /// The new content id matches the entry at `old_index`; reuse it, no
    /// need to send anything for it. Every index in `superseded` (always in
    /// ascending order, all less than `old_index`) was skipped over to find
    /// this match - those old entries have no counterpart in the new run
    /// and should be deleted.
    Hit { old_index: usize, superseded: Vec<usize> },
    /// Nothing in the log matches this content id; the caller must send it
    /// for real. `superseded`, if present, is the old entry the log was
    /// positioned at - it has no counterpart in the new run either and
    /// should be deleted.
    Miss { superseded: Option<usize> },
}

/// The previous run's ordered command log, with an index for matching new
/// content ids against it in O(1) instead of scanning. See the module docs
/// for the matching algorithm.
#[derive(Debug)]
pub(crate) struct CommandLog<Payload> {
    entries: Vec<LogEntry<Payload>>,
    /// For each content id, the ascending queue of positions in `entries`
    /// that have it and haven't been consumed (assigned a disposition) yet.
    index: HashMap<ContentId, VecDeque<usize>>,
    /// The next not-yet-consumed position in `entries`, if any.
    cursor: usize,
}

impl<Payload> CommandLog<Payload> {
    /// Build a log from a previous run's entries, in their original order.
    pub(crate) fn new(entries: Vec<LogEntry<Payload>>) -> Self {
        let mut index: HashMap<ContentId, VecDeque<usize>> = HashMap::new();
        for (position, entry) in entries.iter().enumerate() {
            index.entry(entry.content_id).or_default().push_back(position);
        }
        Self {
            entries,
            index,
            cursor: 0,
        }
    }

    /// The payload at a given position. Only ever called by a caller with a
    /// position a [`MatchDecision`] just returned, so this never needs to
    /// handle an invalid position.
    pub(crate) fn payload(&self, position: usize) -> &Payload {
        &self.entries[position].payload
    }

    /// Match the next content id from the new run against this log. See the
    /// module docs for what each outcome means and the caller's
    /// responsibilities (sending a miss for real, deleting what's
    /// superseded).
    pub(crate) fn match_next(&mut self, content_id: ContentId) -> MatchDecision {
        if self.cursor < self.entries.len() && self.entries[self.cursor].content_id == content_id {
            let old_index = self.cursor;
            self.consume(old_index);
            self.cursor += 1;
            return MatchDecision::Hit {
                old_index,
                superseded: Vec::new(),
            };
        }

        if let Some(found_index) = self.take_earliest(&content_id) {
            // Invariant: nothing before `self.cursor` remains in the index
            // (every position the cursor has passed was consumed), and the
            // direct-hit check above already ruled out `found_index ==
            // self.cursor`, so `found_index > self.cursor` here.
            let superseded: Vec<usize> = (self.cursor..found_index).collect();
            for &position in &superseded {
                self.consume(position);
            }
            self.cursor = found_index + 1;
            return MatchDecision::Hit {
                old_index: found_index,
                superseded,
            };
        }

        let superseded = if self.cursor < self.entries.len() {
            let position = self.cursor;
            self.consume(position);
            self.cursor += 1;
            Some(position)
        } else {
            None
        };
        MatchDecision::Miss { superseded }
    }

    /// Pop and return the earliest not-yet-consumed position with this
    /// content id, if any.
    fn take_earliest(&mut self, content_id: &ContentId) -> Option<usize> {
        let queue = self.index.get_mut(content_id)?;
        let position = queue.pop_front();
        if queue.is_empty() {
            self.index.remove(content_id);
        }
        position
    }

    /// Remove a specific position from its content id's queue, marking it
    /// as having been assigned a disposition (reused or deleted) so it can
    /// never be matched again.
    fn consume(&mut self, position: usize) {
        let content_id = self.entries[position].content_id;
        let Some(queue) = self.index.get_mut(&content_id) else {
            return;
        };
        if let Some(queue_position) = queue.iter().position(|&p| p == position) {
            queue.remove(queue_position);
        }
        if queue.is_empty() {
            self.index.remove(&content_id);
        }
    }
}

/// Spike: accumulates one real run's content-addressed id decisions, both to
/// make this run's own decisions (via [`Self::content_id_of`], for
/// substituting an already-decided id's content identity into a later
/// command that references it) and to become the next run's [`CommandLog`]
/// (via [`Self::into_log`]).
///
/// This specializes [`CommandLog`]'s payload to `uuid::Uuid`: for this
/// spike, all that needs to survive between runs for each logged command is
/// the real engine id it was (or would have been) sent with.
#[derive(Debug, Default)]
pub(crate) struct CommandCacheRound {
    entries: Vec<LogEntry<uuid::Uuid>>,
    id_map: HashMap<uuid::Uuid, ContentId>,
    pub(crate) hits: usize,
    pub(crate) misses: usize,
}

impl CommandCacheRound {
    /// Decide the real id for a new entity whose command content hashes to
    /// `content_id`: reuse an old real id (a hit - the caller must skip
    /// sending) if `old_log`'s match finds one, or mint fresh via
    /// `mint_fresh` (a miss - the caller must send for real) otherwise.
    /// Either way, record the decision so later calls this run can look up
    /// this real id's content id (see [`Self::content_id_of`]), and so it
    /// becomes part of the log the next run matches against.
    pub(crate) fn id_for_content(
        &mut self,
        old_log: Option<&mut CommandLog<uuid::Uuid>>,
        content_id: ContentId,
        mint_fresh: impl FnOnce() -> uuid::Uuid,
    ) -> (uuid::Uuid, bool) {
        let (real_id, is_hit) = match old_log {
            Some(log) => match log.match_next(content_id) {
                MatchDecision::Hit { old_index, .. } => (*log.payload(old_index), true),
                MatchDecision::Miss { .. } => (mint_fresh(), false),
            },
            None => (mint_fresh(), false),
        };
        self.record(content_id, real_id);
        (real_id, is_hit)
    }

    /// Look up a previously-decided real id's content id, for substituting
    /// into a later command's own content hash (see the module docs on why
    /// a raw real id must never be hashed directly).
    pub(crate) fn content_id_of(&self, real_id: uuid::Uuid) -> Option<ContentId> {
        self.id_map.get(&real_id).copied()
    }

    fn record(&mut self, content_id: ContentId, real_id: uuid::Uuid) {
        self.entries.push(LogEntry {
            content_id,
            payload: real_id,
        });
        self.id_map.insert(real_id, content_id);
    }

    /// Finalize this run's decisions into the log the next run will match
    /// against.
    pub(crate) fn into_log(self) -> CommandLog<uuid::Uuid> {
        CommandLog::new(self.entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(label: &str) -> ContentId {
        content_id_from_bytes(label.as_bytes())
    }

    fn log_of(labels: &[&str]) -> CommandLog<String> {
        CommandLog::new(
            labels
                .iter()
                .map(|label| LogEntry {
                    content_id: id(label),
                    payload: label.to_string(),
                })
                .collect(),
        )
    }

    #[test]
    fn identical_content_hashes_the_same_across_two_independently_built_ids() {
        assert_eq!(id("a"), id("a"));
        assert_ne!(id("a"), id("b"));
    }

    #[test]
    fn everything_matches_is_all_hits_no_deletes() {
        let mut log = log_of(&["a", "b", "c"]);
        for label in ["a", "b", "c"] {
            let decision = log.match_next(id(label));
            assert_eq!(
                decision,
                MatchDecision::Hit {
                    old_index: ["a", "b", "c"].iter().position(|&l| l == label).unwrap(),
                    superseded: vec![],
                }
            );
        }
    }

    #[test]
    fn pure_trailing_append_is_misses_with_nothing_superseded() {
        let mut log = log_of(&["a", "b"]);
        assert_eq!(
            log.match_next(id("a")),
            MatchDecision::Hit {
                old_index: 0,
                superseded: vec![]
            }
        );
        assert_eq!(
            log.match_next(id("b")),
            MatchDecision::Hit {
                old_index: 1,
                superseded: vec![]
            }
        );
        // Nothing left in the old log: a pure miss, nothing superseded.
        assert_eq!(log.match_next(id("c")), MatchDecision::Miss { superseded: None });
    }

    #[test]
    fn pure_trailing_delete_supersedes_the_tail_one_at_a_time() {
        // Old log has a trailing "b" that the new run never produces.
        let mut log = log_of(&["a", "b"]);
        assert_eq!(
            log.match_next(id("a")),
            MatchDecision::Hit {
                old_index: 0,
                superseded: vec![]
            }
        );
        // Nothing more arrives from the new run; the caller is responsible
        // for deciding "b" (never matched) is superseded once execution
        // ends, same as today's trailing-delete case - this module only
        // reports supersession discovered *during* matching (point of the
        // worked example below), not log entries nobody ever asked about.
    }

    #[test]
    fn worked_example_mid_log_replacement_is_detected_before_the_new_create() {
        // Mirrors the task's worked example: ten entries, the first three
        // unchanged, the fourth changed in place (still a "miss" at that
        // position, nothing superseded since nothing later matches it
        // either), and the fifth replaced by brand new content that never
        // appears later in the old log.
        let mut log = log_of(&["1", "2", "3", "4", "5.1", "6", "7", "8", "9", "10"]);

        for label in ["1", "2", "3"] {
            assert!(
                matches!(log.match_next(id(label)), MatchDecision::Hit { superseded, .. } if superseded.is_empty())
            );
        }

        // Position 4's content changed; "4-new" appears nowhere else in the
        // old log, so it's a miss, and the old "4" is superseded.
        assert_eq!(log.match_next(id("4-new")), MatchDecision::Miss { superseded: Some(3) });

        // Position 5: "5.2" replaces "5.1". "5.1" never recurs later in the
        // old log, so this is detected right here, as a miss superseding
        // position 4 ("5.1") - before any create for "5.2" would be sent,
        // matching the task's requirement.
        assert_eq!(log.match_next(id("5.2")), MatchDecision::Miss { superseded: Some(4) });

        // Everything from "6" onward is unchanged and still matches by
        // direct position.
        for (expected_index, label) in [(5, "6"), (6, "7"), (7, "8"), (8, "9"), (9, "10")] {
            assert_eq!(
                log.match_next(id(label)),
                MatchDecision::Hit {
                    old_index: expected_index,
                    superseded: vec![],
                }
            );
        }
    }

    #[test]
    fn deletion_cluster_in_the_middle_is_found_via_the_index_not_a_scan() {
        // "b" and "c" are deleted together; "a" and "d" survive unchanged.
        let mut log = log_of(&["a", "b", "c", "d"]);
        assert_eq!(
            log.match_next(id("a")),
            MatchDecision::Hit {
                old_index: 0,
                superseded: vec![]
            }
        );
        assert_eq!(
            log.match_next(id("d")),
            MatchDecision::Hit {
                old_index: 3,
                superseded: vec![1, 2],
            }
        );
    }

    #[test]
    fn insertion_before_an_unchanged_tail_costs_one_avoidable_recreate_not_correctness() {
        // "x" is inserted between "a" and "b". This is the one honest
        // limitation of a true single-pass streaming match: when "x"
        // doesn't match `old[cursor]` ("b") and isn't found anywhere else
        // in the (fully known) old log either, there's no way to tell
        // "b" apart from a genuine replacement without looking ahead into
        // new content that hasn't been computed yet - which a streaming
        // match can't do. So "b" is conservatively treated as superseded
        // here, then recreated (with a fresh id) a step later when it's
        // seen again. The end state is still correct; this just costs one
        // avoidable delete+recreate for "b" instead of leaving it alone.
        let mut log = log_of(&["a", "b"]);
        assert_eq!(
            log.match_next(id("a")),
            MatchDecision::Hit {
                old_index: 0,
                superseded: vec![]
            }
        );
        assert_eq!(log.match_next(id("x")), MatchDecision::Miss { superseded: Some(1) });
        assert_eq!(log.match_next(id("b")), MatchDecision::Miss { superseded: None });
    }

    #[test]
    fn duplicate_content_pairs_with_earliest_unconsumed_occurrence_first() {
        // Two separately-created objects that happen to be identical. Two
        // new occurrences of that same content should pair with the old
        // ones in order, not both match the same old position.
        let mut log = log_of(&["x", "x"]);
        assert_eq!(
            log.match_next(id("x")),
            MatchDecision::Hit {
                old_index: 0,
                superseded: vec![],
            }
        );
        assert_eq!(
            log.match_next(id("x")),
            MatchDecision::Hit {
                old_index: 1,
                superseded: vec![],
            }
        );
    }

    #[test]
    fn a_consumed_position_is_never_matched_again() {
        // Regression guard for the bookkeeping this module exists to get
        // right: once a position has been assigned a disposition, nothing
        // later can accidentally match it again.
        let mut log = log_of(&["x", "y"]);
        let _ = log.match_next(id("x")); // consumes position 0
        // A later, unrelated miss must not accidentally resurrect position
        // 0 just because some future content happens to hash the same.
        let _ = log.match_next(id("z")); // miss, supersedes position 1 ("y")
        assert_eq!(log.match_next(id("x")), MatchDecision::Miss { superseded: None });
    }
}
