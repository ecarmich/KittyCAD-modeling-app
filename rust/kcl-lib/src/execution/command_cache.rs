//! Content-addressed matching of a previous run's command log against a new
//! run's, so unchanged commands can be skipped (not re-sent to the engine)
//! and only genuinely new or superseded ones are acted on.
//!
//! Nothing calls into this module yet - it's the matching algorithm on its
//! own, validated by its own tests, ahead of the separate work of wiring it
//! into real `ModelingCmd` sending (see the batch architecture plan at
//! `.claude/plans/command-cache-batch-architecture.md`). Hence the
//! module-wide `dead_code` allow below.
//!
//! This module is deliberately generic over what a "command" actually is
//! (see [`CommandLog`]'s `Payload` type parameter): it implements only the
//! matching *algorithm* - given a sequence of content ids from a previous
//! run and a new sequence from this run, decide for each new one whether
//! it's a hit (reuse the old result, nothing to send) or a miss (send it
//! for real), and which old entries turned out to have no counterpart in
//! the new run at all (delete them). Turning an actual `ModelingCmd`/
//! response pair into a [`ContentId`], and acting on the decisions this
//! module returns, is the job of its caller.
//!
//! # The matching algorithm
//!
//! A [`CommandLog`] holds the previous run's entries, indexed by content id
//! (an ascending queue of positions per id, so duplicates pair up in the
//! order they originally appeared - see [`CommandLog::match_next`]'s docs).
//! Matching is **index-based, not position-based**: feed it this round's
//! content ids, in any order (program order is the natural choice, but
//! nothing here depends on it), via [`CommandLog::match_next`]. Each call
//! looks the content id up in the index directly - no cursor, no notion of
//! "where the log currently is" - and either finds and consumes the
//! earliest not-yet-claimed old entry with that content id (a hit) or
//! finds nothing (a miss, genuinely new content).
//!
//! Supersession is decided separately, by **exclusion once every new
//! content id for the round has been fed in** ([`CommandLog::superseded`]):
//! an old entry is superseded if and only if nothing in the complete new
//! sequence ever claimed it. This is deliberately not decided eagerly at
//! each mismatch - see the next section for why that distinction matters.
//!
//! This module does not know about ids being shared between two logically
//! different objects (e.g. `extrude()` reusing its input's engine id) - a
//! caller that cares about that must inspect what [`MatchDecision::Hit`]
//! reuses and cross-check it against [`CommandLog::superseded`]'s eventual
//! result, and override a hit into a forced miss if something it's about
//! to delete is also what a hit is relying on.
//!
//! # Why index-based beats position-based, and what it still doesn't fix
//!
//! An earlier version of this module matched by cursor position, falling
//! back to an index lookup only on a mismatch, and decided supersession
//! eagerly at that mismatch rather than by exclusion at the end. That
//! turned out to be a real bug, not just a missed optimization: replaying
//! `[a, b]` against a new sequence `[a, x, b]` (an unrelated item `x`
//! inserted before an otherwise-unchanged `b`) would match `a` directly,
//! then see `x` match nothing at the cursor *or* anywhere later in the old
//! log, and - having no way to look ahead into the new sequence, which
//! hadn't been computed yet - conservatively treat old `b` as superseded
//! right then. A step later, `b` would reappear and be reported as a fresh
//! miss: a needless delete-and-recreate for something that never actually
//! changed. Precomputing the new sequence ahead of time did not fix this on
//! its own, because the old algorithm never looked ahead into it regardless
//! of whether the caller already had it - the bug was in *when* supersession
//! got decided, not in whether matching was streamed or batched.
//!
//! The fix above removes this entirely: matching `[a, x, b]` against old
//! `[a, b]` by index, then excluding only what's left unclaimed at the end,
//! correctly hits both `a` and `b` and reports nothing superseded. It also
//! makes a pure reorder free (`[a, b, c]` against old `[c, a, b]` hits all
//! three), which the old algorithm could not do either.
//!
//! **What [`CommandLog`] alone does not fix**: two old entries with
//! *identical* creation content that have since diverged via a later
//! mutation (e.g. ten identical `MakePlane`s, exactly one of which later
//! got a `PlaneSetColor`) are still indistinguishable to it, because it
//! only ever compares one command's content at a time - it has no notion
//! of an entity's accumulated history. Content-identical siblings pair up
//! by FIFO queue order within its index (see the
//! `duplicate_content_pairs_with_earliest_unconsumed_occurrence_first`
//! test below), which is safe only as long as nothing has actually told
//! them apart yet.
//!
//! [`EntityChain`] and [`ChainLog`] fix this: content identity becomes a
//! rolling, per-entity history (a Merkle-style chain), not an independent
//! hash per command, so two entities that start out identical but later
//! diverge are matched correctly regardless of position - see
//! [`match_round`]'s docs and the
//! `duplicate_content_with_a_divergent_mutation_disambiguates_by_chain_not_position`
//! test. **Not yet wired into anything real** - see [`match_round`]'s own
//! doc comment for exactly why generalizing past the two hand-written
//! call sites in `std/planes.rs` needs more than this module alone.

#![allow(dead_code)]

use std::collections::HashMap;
use std::collections::VecDeque;

use kittycad_modeling_cmds::ModelingCmd;
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
/// Neither variant carries supersession information - that's decided
/// separately, once, by [`CommandLog::superseded`] after every new content
/// id for the round has gone through [`CommandLog::match_next`]. See the
/// module docs for why eagerly deciding supersession per-call (an earlier
/// version of this type did) was the actual bug, not just an API wrinkle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatchDecision {
    /// The new content id matched the entry at `old_index`; reuse it, no
    /// need to send anything for it.
    Hit { old_index: usize },
    /// Nothing in the log matches this content id; the caller must send it
    /// for real.
    Miss,
}

/// The previous run's command log, indexed by content id for O(1) matching
/// instead of scanning. See the module docs for the matching algorithm.
#[derive(Debug)]
pub(crate) struct CommandLog<Payload> {
    entries: Vec<LogEntry<Payload>>,
    /// For each content id, the ascending queue of positions in `entries`
    /// that have it and haven't been claimed by anything in this round's
    /// sequence yet. Shrinks as `match_next` consumes positions; whatever
    /// remains once the whole round has been fed in has no counterpart in
    /// the new run - see [`Self::superseded`].
    index: HashMap<ContentId, VecDeque<usize>>,
}

impl<Payload> CommandLog<Payload> {
    /// Build a log from a previous run's entries, in their original order.
    pub(crate) fn new(entries: Vec<LogEntry<Payload>>) -> Self {
        let mut index: HashMap<ContentId, VecDeque<usize>> = HashMap::new();
        for (position, entry) in entries.iter().enumerate() {
            index.entry(entry.content_id).or_default().push_back(position);
        }
        Self { entries, index }
    }

    /// The payload at a given position. Only ever called by a caller with a
    /// position a [`MatchDecision`] just returned, so this never needs to
    /// handle an invalid position.
    pub(crate) fn payload(&self, position: usize) -> &Payload {
        &self.entries[position].payload
    }

    /// Match one new content id against this log by index lookup alone -
    /// no cursor, no position dependence. On a hit, consumes the earliest
    /// not-yet-claimed old entry with this content id (so two new
    /// occurrences of the same content pair with the old ones in the order
    /// they originally appeared, not both matching the same position - see
    /// the `duplicate_content_...` test below).
    pub(crate) fn match_next(&mut self, content_id: ContentId) -> MatchDecision {
        match self.take_earliest(&content_id) {
            Some(old_index) => MatchDecision::Hit { old_index },
            None => MatchDecision::Miss,
        }
    }

    /// Every old position that no content id in this round's sequence ever
    /// claimed, in ascending order. Call this only after every new content
    /// id for the round has already gone through [`Self::match_next`] -
    /// calling it mid-round would report positions as superseded that
    /// simply haven't been matched *yet*. These have no counterpart in the
    /// new run and should be deleted.
    pub(crate) fn superseded(&self) -> Vec<usize> {
        let mut positions: Vec<usize> = self.index.values().flatten().copied().collect();
        positions.sort_unstable();
        positions
    }

    /// Pop and return the earliest not-yet-claimed position with this
    /// content id, if any.
    fn take_earliest(&mut self, content_id: &ContentId) -> Option<usize> {
        let queue = self.index.get_mut(content_id)?;
        let position = queue.pop_front();
        if queue.is_empty() {
            self.index.remove(content_id);
        }
        position
    }
}

/// One entity's accumulated content history: `levels[0]` is its creation
/// content, and every later element is a subsequent in-place mutation's
/// content chained onto whatever came before it (a Merkle-style history,
/// not an independent hash per command). Two entities that start out
/// identical (same creation content, so the same `levels[0]`) diverge the
/// moment one of them receives a mutation the other doesn't - which is
/// exactly the case [`CommandLog`] alone cannot disambiguate (see the
/// module docs' section on what index-based matching still doesn't fix).
#[derive(Debug, Clone)]
pub(crate) struct EntityChain {
    levels: Vec<ContentId>,
}

impl EntityChain {
    /// Start a new chain from an entity's creation content.
    pub(crate) fn new(creation_content: ContentId) -> Self {
        Self {
            levels: vec![creation_content],
        }
    }

    /// Roll the chain forward by one in-place mutation.
    /// `mutation_content` should already have any *other* entity's
    /// reference fields substituted by *their* current chain content (the
    /// same rule [`CommandCacheRound::content_id_of`] exists for today) -
    /// this only threads in *this* entity's own prior history
    /// automatically.
    pub(crate) fn extend(&mut self, mutation_content: &[u8]) {
        let mut bytes = Vec::with_capacity(self.current().len() + mutation_content.len());
        bytes.extend_from_slice(&self.current());
        bytes.extend_from_slice(mutation_content);
        self.levels.push(content_id_from_bytes(&bytes));
    }

    /// The current (deepest) level - substitute this, not a raw real id,
    /// when a later command's own content hash references this entity.
    pub(crate) fn current(&self) -> ContentId {
        *self
            .levels
            .last()
            .expect("a chain always has at least its creation level")
    }

    /// How many mutations (not counting creation) this chain has seen so
    /// far. [`match_round`] resolves deeper chains before shallower ones,
    /// since depth is exactly what makes a chain more likely to be
    /// globally unique rather than shared with untouched siblings.
    pub(crate) fn depth(&self) -> usize {
        self.levels.len() - 1
    }
}

/// The result of matching one entity's [`EntityChain`] against a
/// [`ChainLog`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChainMatch {
    /// The entity's *entire* chain, through its deepest level, matches an
    /// old entity's: reuse its payload, and skip sending every command
    /// that contributed to the chain, not just the latest one.
    FullHit { old_index: usize },
    /// Only the first `matched_depth + 1` levels match (`matched_depth ==
    /// 0` means just the creation level matched); the old entity is still
    /// the same underlying object to reuse, but whatever this entity's
    /// chain does beyond `matched_depth` must be (re)sent - e.g. a
    /// mutation whose *value* changed (red last round, blue this round)
    /// reuses the base object and resends only that one mutation, instead
    /// of recreating the whole thing.
    PartialHit { old_index: usize, matched_depth: usize },
    /// Not even this chain's creation level matches anything in the old
    /// log: genuinely new.
    Miss,
}

/// The previous round's entity chains, indexed at every level - not just
/// each entity's deepest one - so a new entity's chain can match an old
/// one at whatever depth they still agree on; see [`ChainMatch::PartialHit`].
#[derive(Debug)]
pub(crate) struct ChainLog<Payload> {
    entities: Vec<(EntityChain, Payload)>,
    /// For each level's content id (across every entity, every depth), the
    /// ascending queue of entity indices that have it *somewhere* in their
    /// chain. An index can appear in several queues (one per level of its
    /// own chain); `claimed` - not removal from these queues - is what
    /// actually prevents an entity from being reused twice, since removing
    /// a claimed entity from every queue it appears in would cost more
    /// than just skipping it lazily on the next pop.
    index: HashMap<ContentId, VecDeque<usize>>,
    claimed: std::collections::HashSet<usize>,
}

impl<Payload> ChainLog<Payload> {
    /// Build a log from the previous round's entities and their full
    /// chains.
    pub(crate) fn new(entities: Vec<(EntityChain, Payload)>) -> Self {
        let mut index: HashMap<ContentId, VecDeque<usize>> = HashMap::new();
        for (entity_index, (chain, _)) in entities.iter().enumerate() {
            for &level_content in &chain.levels {
                index.entry(level_content).or_default().push_back(entity_index);
            }
        }
        Self {
            entities,
            index,
            claimed: Default::default(),
        }
    }

    /// The payload at a given old entity index. Only ever called with an
    /// index a [`ChainMatch`] just returned, so this never needs to handle
    /// an invalid one.
    pub(crate) fn payload(&self, old_index: usize) -> &Payload {
        &self.entities[old_index].1
    }

    /// Match one entity's chain against this log - deepest level first,
    /// falling back to shallower levels so a change partway through an
    /// entity's history still reuses its identity instead of recreating it
    /// outright. Prefer [`match_round`] over calling this directly for more
    /// than one entity: matching several entities one at a time in the
    /// wrong order (shallow before deep) is exactly how a degenerate
    /// sibling can steal a distinguished entity's one true match.
    pub(crate) fn match_chain(&mut self, chain: &EntityChain) -> ChainMatch {
        for depth in (0..chain.levels.len()).rev() {
            if let Some(old_index) = self.take_earliest_unclaimed(chain.levels[depth]) {
                self.claimed.insert(old_index);
                return if depth == chain.levels.len() - 1 {
                    ChainMatch::FullHit { old_index }
                } else {
                    ChainMatch::PartialHit {
                        old_index,
                        matched_depth: depth,
                    }
                };
            }
        }
        ChainMatch::Miss
    }

    /// Pop and return the earliest not-yet-claimed entity index with this
    /// content id at some level, discarding any already-claimed ones found
    /// along the way (they can never be matched again regardless).
    fn take_earliest_unclaimed(&mut self, content_id: ContentId) -> Option<usize> {
        let queue = self.index.get_mut(&content_id)?;
        while let Some(candidate) = queue.pop_front() {
            if !self.claimed.contains(&candidate) {
                if queue.is_empty() {
                    self.index.remove(&content_id);
                }
                return Some(candidate);
            }
        }
        self.index.remove(&content_id);
        None
    }
}

/// Match every entity produced this round against the previous round's
/// chains in one pass, ordered so entities with a deeper (more specific,
/// so more likely globally unique) chain are matched before shallower
/// ones. This is the actual fix for the duplicate-content-with-divergent-
/// mutation case (see the module docs and this file's
/// `duplicate_content_with_a_divergent_mutation_...` test): without this
/// ordering, a degenerate sibling processed first could claim the one old
/// entity a distinguished entity was actually supposed to reuse, before
/// the distinguished entity ever got a chance to ask for it.
///
/// `new_chains` should be in the order entities were produced this round;
/// ties (equal depth) keep that relative order, which is what makes
/// genuinely-interchangeable siblings pair up FIFO rather than arbitrarily,
/// the same way [`ChainLog::match_chain`] already does within one content
/// id.
///
/// Not yet called from anywhere real: generalizing past the two
/// hand-written call sites in `std/planes.rs` needs every entity's full
/// chain to be known *before* any real id is decided, which in turn needs
/// interpreter ids decoupled from real ids (the design note's SS2) - not
/// yet implemented, only designed. Calling this against eagerly-decided
/// real ids, the way `std/planes.rs` decides them today, would reproduce
/// the exact bug it exists to fix (see the design note's revision to SS3,
/// "before further implementation").
pub(crate) fn match_round<Payload>(old: &mut ChainLog<Payload>, new_chains: &[EntityChain]) -> Vec<ChainMatch> {
    let mut order: Vec<usize> = (0..new_chains.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(new_chains[i].depth()));
    let mut results = vec![ChainMatch::Miss; new_chains.len()];
    for i in order {
        results[i] = old.match_chain(&new_chains[i]);
    }
    results
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
                MatchDecision::Hit { old_index } => (*log.payload(old_index), true),
                MatchDecision::Miss => (mint_fresh(), false),
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

/// Declared once per `ModelingCmd` variant: every `Uuid`-shaped field that
/// is a reference to another entity's interpreter id, as opposed to a
/// value that merely happens to be shaped like a `Uuid` (a color, a
/// session token, ...). See the design note
/// (`.claude/plans/command-cache-batch-architecture.md`, SS3.3-3.4) for why
/// this has to be declared once per command *type* here, rather than
/// derived from field types (most reference fields are plain `Uuid`,
/// indistinguishable by type from an incidental one) or handled at each of
/// the ~150 call sites that construct a `ModelingCmd`.
pub(crate) trait IdRefs {
    /// Call `visit` once for every id-reference field this command variant
    /// carries, with a mutable reference so a translator (not built yet -
    /// this trait is standalone and unwired, same as the rest of this
    /// module) can overwrite it in place. The default covers every variant
    /// with no such field.
    fn for_each_id_ref_mut(&mut self, visit: &mut dyn FnMut(&mut uuid::Uuid)) {
        let _ = visit;
    }
}

impl IdRefs for ModelingCmd {
    /// One match arm per variant that has at least one id-reference field,
    /// derived mechanically from `kittycad-modeling-cmds = "0.2.250"`'s
    /// `def_enum.rs` (167 total `ModelingCmd` variants; 110 carry a
    /// `Uuid`/`Option<Uuid>`/`Vec<Uuid>`/`Option<Vec<Uuid>>`/
    /// `ModelingCmdId`/`Option<ModelingCmdId>` field). Every other variant
    /// falls through to the trait's no-op default.
    ///
    /// Re-derive this list mechanically against whatever crate version is
    /// pinned when this is actually wired in - don't hand-maintain it by
    /// eye - and confirm each field's semantics against the engine's own
    /// command docs, not just its name, before relying on it. `CreateRegion`'s
    /// `segment`/`intersection_segment` below are included because treating
    /// a real reference as inert is the dangerous direction (it would hash
    /// a run-to-run-unstable real id into a content id and break matching
    /// silently), but their exact semantics haven't been confirmed against
    /// engine docs - see SS3.5.
    fn for_each_id_ref_mut(&mut self, visit: &mut dyn FnMut(&mut uuid::Uuid)) {
        match self {
            ModelingCmd::MovePathPen(c) => visit(&mut c.path.0),
            ModelingCmd::ExtendPath(c) => visit(&mut c.path.0),
            ModelingCmd::Extrude(c) => {
                if let Some(id) = &mut c.target {
                    visit(&mut id.0);
                }
            }
            ModelingCmd::ExtrudeToReference(c) => {
                if let Some(id) = &mut c.target {
                    visit(&mut id.0);
                }
            }
            ModelingCmd::TwistExtrude(c) => visit(&mut c.target.0),
            ModelingCmd::Sweep(c) => {
                visit(&mut c.target.0);
                visit(&mut c.trajectory.0);
            }
            ModelingCmd::Revolve(c) => visit(&mut c.target.0),
            ModelingCmd::Solid3dShellFace(c) => {
                visit(&mut c.object_id);
                for id in &mut c.face_ids {
                    visit(id);
                }
            }
            ModelingCmd::Solid3dJoin(c) => visit(&mut c.object_id),
            ModelingCmd::Solid3dMultiJoin(c) => {
                for id in &mut c.object_ids {
                    visit(id);
                }
            }
            ModelingCmd::Solid3dGetEdgeUuid(c) => visit(&mut c.object_id),
            ModelingCmd::Solid3dGetFaceUuid(c) => visit(&mut c.object_id),
            ModelingCmd::Solid3dGetBodyType(c) => visit(&mut c.object_id),
            ModelingCmd::RevolveAboutEdge(c) => {
                visit(&mut c.target.0);
                if let Some(id) = &mut c.edge_id {
                    visit(id);
                }
            }
            ModelingCmd::Loft(c) => {
                for id in &mut c.section_ids {
                    visit(id);
                }
            }
            ModelingCmd::ClosePath(c) => visit(&mut c.path_id),
            ModelingCmd::Export2d(c) => {
                for id in &mut c.entity_ids {
                    visit(id);
                }
            }
            ModelingCmd::Export3d(c) => {
                for id in &mut c.entity_ids {
                    visit(id);
                }
            }
            ModelingCmd::Export(c) => {
                for id in &mut c.entity_ids {
                    visit(id);
                }
            }
            ModelingCmd::EntityGetParentId(c) => visit(&mut c.entity_id),
            ModelingCmd::EntityGetNumChildren(c) => visit(&mut c.entity_id),
            ModelingCmd::EntityGetChildUuid(c) => visit(&mut c.entity_id),
            ModelingCmd::EntityGetIndex(c) => visit(&mut c.entity_id),
            ModelingCmd::EntityGetPrimitiveIndex(c) => visit(&mut c.entity_id),
            ModelingCmd::EntityDeleteChildren(c) => visit(&mut c.entity_id),
            ModelingCmd::EntityGetAllChildUuids(c) => visit(&mut c.entity_id),
            ModelingCmd::EntityGetSketchPaths(c) => visit(&mut c.entity_id),
            ModelingCmd::EntityGetDistance(c) => {
                visit(&mut c.entity_id1);
                visit(&mut c.entity_id2);
            }
            ModelingCmd::EdgeGetLength(c) => visit(&mut c.edge_id),
            ModelingCmd::EntityClone(c) => visit(&mut c.entity_id),
            ModelingCmd::EntityLinearPatternTransform(c) => visit(&mut c.entity_id),
            ModelingCmd::EntityLinearPattern(c) => visit(&mut c.entity_id),
            ModelingCmd::EntityCircularPattern(c) => visit(&mut c.entity_id),
            ModelingCmd::EntityMakeHelix(c) => visit(&mut c.cylinder_id),
            ModelingCmd::EntityMakeHelixFromEdge(c) => {
                if let Some(id) = &mut c.edge_id {
                    visit(id);
                }
            }
            ModelingCmd::EntityMirrorAcross(c) => {
                for id in &mut c.ids {
                    visit(id);
                }
            }
            ModelingCmd::EntityMirror(c) => {
                for id in &mut c.ids {
                    visit(id);
                }
            }
            ModelingCmd::EntityMirrorAcrossEdge(c) => {
                for id in &mut c.ids {
                    visit(id);
                }
                if let Some(id) = &mut c.edge_id {
                    visit(id);
                }
            }
            ModelingCmd::QueryEntityType(c) => visit(&mut c.entity_id),
            ModelingCmd::SelectAdd(c) => {
                for id in &mut c.entities {
                    visit(id);
                }
            }
            ModelingCmd::SelectRemove(c) => {
                for id in &mut c.entities {
                    visit(id);
                }
            }
            ModelingCmd::SelectReplace(c) => {
                for id in &mut c.entities {
                    visit(id);
                }
            }
            ModelingCmd::HighlightSetEntities(c) => {
                for id in &mut c.entities {
                    visit(id);
                }
            }
            ModelingCmd::UpdateAnnotation(c) => visit(&mut c.annotation_id),
            ModelingCmd::ObjectVisible(c) => visit(&mut c.object_id),
            ModelingCmd::ObjectBringToFront(c) => visit(&mut c.object_id),
            ModelingCmd::ObjectSetMaterialParamsPbr(c) => visit(&mut c.object_id),
            ModelingCmd::ObjectSetName(c) => visit(&mut c.object_id),
            ModelingCmd::GetEntityType(c) => visit(&mut c.entity_id),
            ModelingCmd::Solid3dGetAllEdgeFaces(c) => {
                visit(&mut c.object_id);
                visit(&mut c.edge_id);
            }
            ModelingCmd::Solid3dFlip(c) => visit(&mut c.object_id),
            ModelingCmd::Solid3dFlipFace(c) => {
                visit(&mut c.object_id);
                visit(&mut c.face_id);
            }
            ModelingCmd::Solid2dAddHole(c) => {
                visit(&mut c.object_id);
                visit(&mut c.hole_id);
            }
            ModelingCmd::Solid3dGetAllOppositeEdges(c) => {
                visit(&mut c.object_id);
                visit(&mut c.edge_id);
            }
            ModelingCmd::Solid3dGetOppositeEdge(c) => {
                visit(&mut c.object_id);
                visit(&mut c.edge_id);
                visit(&mut c.face_id);
            }
            ModelingCmd::Solid3dGetNextAdjacentEdge(c) => {
                visit(&mut c.object_id);
                visit(&mut c.edge_id);
                visit(&mut c.face_id);
            }
            ModelingCmd::Solid3dGetPrevAdjacentEdge(c) => {
                visit(&mut c.object_id);
                visit(&mut c.edge_id);
                visit(&mut c.face_id);
            }
            ModelingCmd::Solid3dGetCommonEdge(c) => visit(&mut c.object_id),
            ModelingCmd::Solid3dFilletEdge(c) => {
                visit(&mut c.object_id);
                if let Some(id) = &mut c.edge_id {
                    visit(id);
                }
                for id in &mut c.edge_ids {
                    visit(id);
                }
                for id in &mut c.extra_face_ids {
                    visit(id);
                }
            }
            ModelingCmd::Solid3dCutEdgeReferences(c) => {
                visit(&mut c.object_id);
                for id in &mut c.extra_face_ids {
                    visit(id);
                }
            }
            ModelingCmd::Solid3dCutEdges(c) => {
                visit(&mut c.object_id);
                for id in &mut c.edge_ids {
                    visit(id);
                }
                for id in &mut c.extra_face_ids {
                    visit(id);
                }
            }
            ModelingCmd::FaceIsPlanar(c) => visit(&mut c.object_id),
            ModelingCmd::FaceGetPosition(c) => visit(&mut c.object_id),
            ModelingCmd::FaceGetCenter(c) => visit(&mut c.object_id),
            ModelingCmd::FaceGetGradient(c) => visit(&mut c.object_id),
            ModelingCmd::SendObject(c) => visit(&mut c.object_id),
            ModelingCmd::EntitySetOpacity(c) => visit(&mut c.entity_id),
            ModelingCmd::EntityFade(c) => visit(&mut c.entity_id),
            ModelingCmd::PlaneSetColor(c) => visit(&mut c.plane_id),
            ModelingCmd::CurveSetConstraint(c) => visit(&mut c.object_id),
            ModelingCmd::EnableSketchMode(c) => visit(&mut c.entity_id),
            ModelingCmd::CurveGetType(c) => visit(&mut c.curve_id),
            ModelingCmd::CurveGetControlPoints(c) => visit(&mut c.curve_id),
            ModelingCmd::ProjectEntityToPlane(c) => {
                if let Some(id) = &mut c.entity_id {
                    visit(id);
                }
                visit(&mut c.plane_id);
            }
            ModelingCmd::ProjectPointsToPlane(c) => visit(&mut c.plane_id),
            ModelingCmd::PathGetInfo(c) => visit(&mut c.path_id),
            ModelingCmd::PathGetCurveUuidsForVertices(c) => {
                visit(&mut c.path_id);
                for id in &mut c.vertex_ids {
                    visit(id);
                }
            }
            ModelingCmd::PathGetCurveUuid(c) => visit(&mut c.path_id),
            ModelingCmd::PathGetVertexUuids(c) => visit(&mut c.path_id),
            ModelingCmd::PathGetSketchTargetUuid(c) => visit(&mut c.path_id),
            ModelingCmd::PlaneIntersectAndProject(c) => visit(&mut c.plane_id),
            ModelingCmd::CurveGetEndPoints(c) => visit(&mut c.curve_id),
            ModelingCmd::Mass(c) => {
                for id in &mut c.entity_ids {
                    visit(id);
                }
            }
            ModelingCmd::Density(c) => {
                for id in &mut c.entity_ids {
                    visit(id);
                }
            }
            ModelingCmd::Volume(c) => {
                for id in &mut c.entity_ids {
                    visit(id);
                }
            }
            ModelingCmd::CenterOfMass(c) => {
                for id in &mut c.entity_ids {
                    visit(id);
                }
            }
            ModelingCmd::SurfaceArea(c) => {
                for id in &mut c.entity_ids {
                    visit(id);
                }
            }
            ModelingCmd::PhysicalProperties(c) => {
                for id in &mut c.entity_ids {
                    visit(id);
                }
            }
            ModelingCmd::DefaultCameraFocusOn(c) => visit(&mut c.uuid),
            ModelingCmd::ZoomToFit(c) => {
                for id in &mut c.object_ids {
                    visit(id);
                }
            }
            ModelingCmd::OrientToFace(c) => visit(&mut c.face_id),
            ModelingCmd::Solid3dGetExtrusionFaceInfo(c) => {
                visit(&mut c.object_id);
                if let Some(id) = &mut c.edge_id {
                    visit(id);
                }
            }
            ModelingCmd::Solid3dGetAdjacencyInfo(c) => {
                visit(&mut c.object_id);
                if let Some(id) = &mut c.edge_id {
                    visit(id);
                }
            }
            ModelingCmd::SetObjectTransform(c) => visit(&mut c.object_id),
            ModelingCmd::BooleanUnion(c) => {
                for id in &mut c.solid_ids {
                    visit(id);
                }
            }
            ModelingCmd::BooleanIntersection(c) => {
                for id in &mut c.solid_ids {
                    visit(id);
                }
            }
            ModelingCmd::BooleanSubtract(c) => {
                for id in &mut c.target_ids {
                    visit(id);
                }
                for id in &mut c.tool_ids {
                    visit(id);
                }
            }
            ModelingCmd::BooleanImprint(c) => {
                for id in &mut c.body_ids {
                    visit(id);
                }
                if let Some(ids) = &mut c.tool_ids {
                    for id in ids {
                        visit(id);
                    }
                }
            }
            ModelingCmd::MakeOffsetPath(c) => {
                visit(&mut c.object_id);
                if let Some(id) = &mut c.face_id {
                    visit(id);
                }
            }
            ModelingCmd::AddHoleFromOffset(c) => visit(&mut c.object_id),
            ModelingCmd::SetGridReferencePlane(c) => {
                visit(&mut c.grid_id);
                visit(&mut c.reference_id);
            }
            // `segment`/`intersection_segment`: see this fn's doc comment -
            // included conservatively, not yet confirmed against engine docs.
            ModelingCmd::CreateRegion(c) => {
                visit(&mut c.object_id);
                visit(&mut c.segment);
                visit(&mut c.intersection_segment);
            }
            ModelingCmd::CreatePlanarSurface(c) => {
                for id in &mut c.curve_ids {
                    visit(id);
                }
            }
            ModelingCmd::RegionGetResolvableIntersectionInfo(c) => visit(&mut c.region_id),
            ModelingCmd::CreateRegionFromQueryPoint(c) => visit(&mut c.object_id),
            ModelingCmd::RegionGetQueryPoint(c) => visit(&mut c.region_id),
            ModelingCmd::BoundingBox(c) => {
                for id in &mut c.entity_ids {
                    visit(id);
                }
            }
            ModelingCmd::OffsetSurface(c) => visit(&mut c.surface_id),
            ModelingCmd::ClosestEdge(c) => {
                if let Some(id) = &mut c.object_id {
                    visit(id);
                }
            }
            ModelingCmd::SketchGetInfo(c) => visit(&mut c.path_id.0),
            // Every other variant has no id-reference field.
            _ => {}
        }
    }
}

/// Whether a `ModelingCmd` variant's own outgoing id is the lasting
/// identity of a persistent engine entity - so `Cache::id_for_content`
/// should decide it before sending, as `MakePlane` already does for the
/// spike - or a disposable correlation token thrown away once its response
/// arrives.
///
/// Classified for a representative subset only, picked to cover the cases
/// already traced in the design note: an owning creation command
/// (`MakePlane`), a mutation that references but doesn't own
/// (`PlaneSetColor`), and the query-style commands already wired into
/// `std/` today (`extrude.rs`, `edge.rs`, `csg.rs`, etc. - see SS1 of the
/// design note). The rest deliberately return `None` rather than a guessed
/// answer; widening this table needs a real check against engine behavior
/// per variant (see SS3.5), not inference from naming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EntityIdentity {
    OwnsEntity,
    Correlation,
}

pub(crate) fn entity_identity(cmd: &ModelingCmd) -> Option<EntityIdentity> {
    match cmd {
        ModelingCmd::MakePlane(_) => Some(EntityIdentity::OwnsEntity),
        ModelingCmd::PlaneSetColor(_)
        | ModelingCmd::Solid3dGetExtrusionFaceInfo(_)
        | ModelingCmd::Solid3dGetAdjacencyInfo(_)
        | ModelingCmd::Solid3dGetAllEdgeFaces(_)
        | ModelingCmd::Solid3dGetOppositeEdge(_)
        | ModelingCmd::Solid3dGetNextAdjacentEdge(_)
        | ModelingCmd::Solid3dGetPrevAdjacentEdge(_)
        | ModelingCmd::Solid3dGetCommonEdge(_)
        | ModelingCmd::QueryEntityType(_)
        | ModelingCmd::BoundingBox(_) => Some(EntityIdentity::Correlation),
        _ => None,
    }
}

/// Whether a `ModelingCmd` variant creates a new entity, mutates an
/// existing one in place (extending its content-chain per the design
/// note's SS3.2), is a pure read, or - as with `Solid3dFilletEdge`, which
/// mutates `object_id` in place *and* introduces new face ids - more than
/// one of these on the same command. Same representative-subset caveat as
/// [`EntityIdentity`]/[`entity_identity`]: `None` means not yet classified,
/// not "doesn't apply."
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommandEffect {
    Creates,
    MutatesInPlace,
    Query,
    /// Both at once - see `Solid3dFilletEdge` above.
    MutatesAndCreates,
}

pub(crate) fn command_effect(cmd: &ModelingCmd) -> Option<CommandEffect> {
    match cmd {
        ModelingCmd::MakePlane(_)
        | ModelingCmd::BooleanUnion(_)
        | ModelingCmd::BooleanIntersection(_)
        | ModelingCmd::BooleanSubtract(_)
        | ModelingCmd::BooleanImprint(_) => Some(CommandEffect::Creates),
        ModelingCmd::PlaneSetColor(_) => Some(CommandEffect::MutatesInPlace),
        ModelingCmd::Solid3dFilletEdge(_) => Some(CommandEffect::MutatesAndCreates),
        ModelingCmd::Solid3dGetExtrusionFaceInfo(_)
        | ModelingCmd::Solid3dGetAdjacencyInfo(_)
        | ModelingCmd::Solid3dGetAllEdgeFaces(_)
        | ModelingCmd::Solid3dGetOppositeEdge(_)
        | ModelingCmd::Solid3dGetNextAdjacentEdge(_)
        | ModelingCmd::Solid3dGetPrevAdjacentEdge(_)
        | ModelingCmd::Solid3dGetCommonEdge(_)
        | ModelingCmd::QueryEntityType(_)
        | ModelingCmd::BoundingBox(_) => Some(CommandEffect::Query),
        _ => None,
    }
}

#[cfg(test)]
mod id_refs_tests {
    use kittycad_modeling_cmds::each_cmd::BooleanImprint;
    use kittycad_modeling_cmds::each_cmd::ClosePath;
    use kittycad_modeling_cmds::each_cmd::MakePlane;
    use kittycad_modeling_cmds::each_cmd::PlaneSetColor;
    use kittycad_modeling_cmds::each_cmd::SelectAdd;
    use kittycad_modeling_cmds::each_cmd::SketchGetInfo;
    use kittycad_modeling_cmds::id::ModelingCmdId;
    use kittycad_modeling_cmds::length_unit::LengthUnit;
    use kittycad_modeling_cmds::shared::Color;
    use uuid::Uuid;

    use super::*;

    // Every per-command struct in `kittycad-modeling-cmds` is marked
    // `#[non_exhaustive]` outside the `unstable_exhaustive` feature, so a
    // plain struct literal (`MakePlane { .. }`) doesn't compile here - the
    // crate's own `bon::Builder` derive is the sanctioned constructor, same
    // as `std/planes.rs`'s real (non-test) usage already relies on.

    fn make_plane_cmd() -> ModelingCmd {
        ModelingCmd::MakePlane(
            MakePlane::builder()
                .clobber(false)
                .origin(Default::default())
                .x_axis(Default::default())
                .y_axis(Default::default())
                .size(LengthUnit(100.0))
                .build(),
        )
    }

    fn visited(cmd: &mut ModelingCmd) -> Vec<Uuid> {
        let mut seen = Vec::new();
        cmd.for_each_id_ref_mut(&mut |id| seen.push(*id));
        seen
    }

    #[test]
    fn a_command_with_no_id_reference_fields_visits_nothing() {
        let mut cmd = make_plane_cmd();
        assert_eq!(visited(&mut cmd), Vec::<Uuid>::new());
    }

    #[test]
    fn a_bare_uuid_reference_field_is_visited() {
        let plane_id = Uuid::new_v4();
        let mut cmd = ModelingCmd::PlaneSetColor(
            PlaneSetColor::builder()
                .plane_id(plane_id)
                .color(Color::from_rgba(1.0, 0.0, 0.0, 1.0))
                .build(),
        );
        assert_eq!(visited(&mut cmd), vec![plane_id]);
    }

    #[test]
    fn a_modeling_cmd_id_reference_field_is_visited_by_its_inner_uuid() {
        let path_id = Uuid::new_v4();
        let mut cmd = ModelingCmd::SketchGetInfo(SketchGetInfo::builder().path_id(ModelingCmdId(path_id)).build());
        assert_eq!(visited(&mut cmd), vec![path_id]);
    }

    #[test]
    fn a_vec_uuid_reference_field_visits_every_element_in_order() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let mut cmd = ModelingCmd::SelectAdd(SelectAdd::builder().entities(vec![a, b]).build());
        assert_eq!(visited(&mut cmd), vec![a, b]);
    }

    #[test]
    fn an_option_vec_uuid_reference_field_is_visited_only_when_present() {
        let body = Uuid::new_v4();
        let mut with_tools = ModelingCmd::BooleanImprint(
            BooleanImprint::builder()
                .body_ids(vec![body])
                .tool_ids(vec![Uuid::new_v4()])
                .separate_bodies(false)
                .use_legacy(false)
                .keep_tools(false)
                .tolerance(LengthUnit(0.01))
                .build(),
        );
        assert_eq!(visited(&mut with_tools).len(), 2);

        let mut without_tools = ModelingCmd::BooleanImprint(
            BooleanImprint::builder()
                .body_ids(vec![body])
                .separate_bodies(false)
                .use_legacy(false)
                .keep_tools(false)
                .tolerance(LengthUnit(0.01))
                .build(),
        );
        assert_eq!(visited(&mut without_tools), vec![body]);
    }

    #[test]
    fn visiting_can_overwrite_ids_in_place() {
        let old_id = Uuid::new_v4();
        let new_id = Uuid::new_v4();
        let mut cmd = ModelingCmd::ClosePath(ClosePath::builder().path_id(old_id).build());
        cmd.for_each_id_ref_mut(&mut |id| *id = new_id);
        let ModelingCmd::ClosePath(c) = cmd else {
            panic!("expected ClosePath");
        };
        assert_eq!(c.path_id, new_id);
    }

    #[test]
    fn entity_identity_and_effect_are_classified_for_the_representative_subset_only() {
        let make_plane = make_plane_cmd();
        assert_eq!(entity_identity(&make_plane), Some(EntityIdentity::OwnsEntity));
        assert_eq!(command_effect(&make_plane), Some(CommandEffect::Creates));

        let close_path = ModelingCmd::ClosePath(ClosePath::builder().path_id(Uuid::new_v4()).build());
        assert_eq!(entity_identity(&close_path), None);
        assert_eq!(command_effect(&close_path), None);
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
                }
            );
        }
        assert_eq!(log.superseded(), Vec::<usize>::new());
    }

    #[test]
    fn pure_trailing_append_is_a_miss_with_nothing_superseded() {
        let mut log = log_of(&["a", "b"]);
        assert_eq!(log.match_next(id("a")), MatchDecision::Hit { old_index: 0 });
        assert_eq!(log.match_next(id("b")), MatchDecision::Hit { old_index: 1 });
        // Nothing left in the old log: a pure miss.
        assert_eq!(log.match_next(id("c")), MatchDecision::Miss);
        // Both old entries were claimed; nothing superseded.
        assert_eq!(log.superseded(), Vec::<usize>::new());
    }

    #[test]
    fn pure_trailing_delete_is_superseded_once_the_round_ends() {
        // Old log has a trailing "b" that the new run never produces.
        let mut log = log_of(&["a", "b"]);
        assert_eq!(log.match_next(id("a")), MatchDecision::Hit { old_index: 0 });
        // Nothing more arrives from the new run. "b" was never claimed, so
        // it shows up as superseded once the caller asks, after feeding in
        // everything this round actually produced.
        assert_eq!(log.superseded(), vec![1]);
    }

    #[test]
    fn worked_example_mid_log_replacement_is_detected_by_the_time_the_round_ends() {
        // Mirrors the task's worked example: ten entries, the first three
        // unchanged, the fourth changed in place, and the fifth replaced by
        // brand new content that never appears anywhere else in the old log.
        let mut log = log_of(&["1", "2", "3", "4", "5.1", "6", "7", "8", "9", "10"]);

        for label in ["1", "2", "3"] {
            assert!(matches!(log.match_next(id(label)), MatchDecision::Hit { .. }));
        }

        // "4-new" and "5.2" appear nowhere in the old log: both misses.
        assert_eq!(log.match_next(id("4-new")), MatchDecision::Miss);
        assert_eq!(log.match_next(id("5.2")), MatchDecision::Miss);

        // Everything from "6" onward is unchanged and still matches by
        // content, found directly via the index.
        for (expected_index, label) in [(5, "6"), (6, "7"), (7, "8"), (8, "9"), (9, "10")] {
            assert_eq!(
                log.match_next(id(label)),
                MatchDecision::Hit {
                    old_index: expected_index
                }
            );
        }

        // Old "4" (position 3) and old "5.1" (position 4) were never
        // claimed by anything in the new sequence - superseded, in order,
        // once the whole round is in.
        assert_eq!(log.superseded(), vec![3, 4]);
    }

    #[test]
    fn deletion_cluster_in_the_middle_is_found_via_the_index_not_a_scan() {
        // "b" and "c" are deleted together; "a" and "d" survive unchanged.
        let mut log = log_of(&["a", "b", "c", "d"]);
        assert_eq!(log.match_next(id("a")), MatchDecision::Hit { old_index: 0 });
        assert_eq!(log.match_next(id("d")), MatchDecision::Hit { old_index: 3 });
        assert_eq!(log.superseded(), vec![1, 2]);
    }

    #[test]
    fn insertion_before_an_unchanged_tail_no_longer_forces_a_recreate() {
        // "x" is inserted between "a" and "b". This is exactly the case an
        // earlier, position-based version of this matcher got wrong: it
        // would see "x" fail to match at the cursor (where "b" was) and,
        // having no way to look ahead into content it hadn't been fed yet,
        // conservatively treat "b" as superseded right then - a needless
        // delete-and-recreate for something that never actually changed.
        // Matching by index instead of position fixes this outright: "b"'s
        // content still exists in the old log regardless of where the
        // cursor would have been, so it's found and reused directly.
        let mut log = log_of(&["a", "b"]);
        assert_eq!(log.match_next(id("a")), MatchDecision::Hit { old_index: 0 });
        assert_eq!(log.match_next(id("x")), MatchDecision::Miss); // genuinely new
        assert_eq!(log.match_next(id("b")), MatchDecision::Hit { old_index: 1 }); // reused, not recreated
        assert_eq!(log.superseded(), Vec::<usize>::new()); // nothing deleted
    }

    #[test]
    fn pure_reorder_reuses_everything() {
        // A stronger case than insertion: every old entry survives, just in
        // a different order. Index-based matching doesn't care about order
        // at all, so all three are hits and nothing is superseded.
        let mut log = log_of(&["a", "b", "c"]);
        assert_eq!(log.match_next(id("c")), MatchDecision::Hit { old_index: 2 });
        assert_eq!(log.match_next(id("a")), MatchDecision::Hit { old_index: 0 });
        assert_eq!(log.match_next(id("b")), MatchDecision::Hit { old_index: 1 });
        assert_eq!(log.superseded(), Vec::<usize>::new());
    }

    #[test]
    fn duplicate_content_pairs_with_earliest_unconsumed_occurrence_first() {
        // Two separately-created objects that happen to be identical. Two
        // new occurrences of that same content should pair with the old
        // ones in order, not both match the same old position. This is the
        // one thing index-based matching does *not* fix on its own - see
        // the module docs' section on what's still unresolved until entity
        // chaining (design note SS3.2) exists: this is only safe as long as
        // nothing has mutated one of the duplicates differently from the
        // others.
        let mut log = log_of(&["x", "x"]);
        assert_eq!(log.match_next(id("x")), MatchDecision::Hit { old_index: 0 });
        assert_eq!(log.match_next(id("x")), MatchDecision::Hit { old_index: 1 });
        assert_eq!(log.superseded(), Vec::<usize>::new());
    }

    #[test]
    fn a_consumed_position_is_never_matched_again() {
        // Regression guard for the bookkeeping this module exists to get
        // right: once a position has been claimed, nothing later can
        // accidentally match it again.
        let mut log = log_of(&["x", "y"]);
        let _ = log.match_next(id("x")); // consumes position 0
        // A later, unrelated miss must not accidentally resurrect position
        // 0 just because some future content happens to hash the same.
        let _ = log.match_next(id("z")); // miss; "y" (position 1) stays unclaimed
        assert_eq!(log.match_next(id("x")), MatchDecision::Miss);
        assert_eq!(log.superseded(), vec![1]);
    }
}

#[cfg(test)]
mod chain_tests {
    use super::*;

    fn id(label: &str) -> ContentId {
        content_id_from_bytes(label.as_bytes())
    }

    /// Build a chain from a creation label plus zero or more mutation
    /// labels, applied in order - a test-only stand-in for real content
    /// bytes, same spirit as `tests::log_of`'s use of plain strings.
    fn chain_of(levels: &[&str]) -> EntityChain {
        let mut chain = EntityChain::new(id(levels[0]));
        for level in &levels[1..] {
            chain.extend(level.as_bytes());
        }
        chain
    }

    #[test]
    fn two_chains_with_the_same_creation_content_are_equal_until_one_is_extended() {
        let a = chain_of(&["plane"]);
        let b = chain_of(&["plane"]);
        assert_eq!(a.current(), b.current());
        let mut b = b;
        b.extend(b"red");
        assert_ne!(a.current(), b.current());
        assert_eq!(a.depth(), 0);
        assert_eq!(b.depth(), 1);
    }

    #[test]
    fn an_unmutated_entity_is_a_full_hit_at_depth_zero() {
        let mut old = ChainLog::new(vec![(chain_of(&["plane"]), "R1")]);
        assert_eq!(
            old.match_chain(&chain_of(&["plane"])),
            ChainMatch::FullHit { old_index: 0 }
        );
    }

    #[test]
    fn genuinely_new_content_is_a_miss() {
        let mut old = ChainLog::new(vec![(chain_of(&["plane"]), "R1")]);
        assert_eq!(old.match_chain(&chain_of(&["sketch"])), ChainMatch::Miss);
    }

    #[test]
    fn a_changed_mutation_value_falls_back_to_reusing_the_base_object() {
        // Colored red last round, blue this round: the deepest level
        // doesn't match anything, but the creation level still does - so
        // the base object is reused and only the color needs resending,
        // not a full recreate.
        let mut old = ChainLog::new(vec![(chain_of(&["plane", "red"]), "R1")]);
        assert_eq!(
            old.match_chain(&chain_of(&["plane", "blue"])),
            ChainMatch::PartialHit {
                old_index: 0,
                matched_depth: 0
            }
        );
    }

    #[test]
    fn duplicate_content_with_a_divergent_mutation_disambiguates_by_chain_not_position() {
        // The exact scenario that broke independent per-command content
        // ids: ten identical planes: old entity #6 (0-indexed 5) is the
        // only one that was ever colored. This is the worked example from
        // the design note's SS3.1.
        let old_entities: Vec<(EntityChain, String)> = (0..10)
            .map(|i| {
                let chain = if i == 5 {
                    chain_of(&["plane", "red"])
                } else {
                    chain_of(&["plane"])
                };
                (chain, format!("R{i}"))
            })
            .collect();
        let mut old = ChainLog::new(old_entities);

        // This round: the same ten planes, but an unrelated edit means the
        // colored one is no longer in the same position - put it *last*,
        // nowhere near position 5, to prove position plays no role.
        let mut new_chains: Vec<EntityChain> = (0..9).map(|_| chain_of(&["plane"])).collect();
        new_chains.push(chain_of(&["plane", "red"]));

        let results = match_round(&mut old, &new_chains);

        // The colored entity must land on old "R5" specifically (payload
        // index 5), the one old entity that actually has this history -
        // not an arbitrary one of the ten, despite arriving last.
        assert_eq!(results[9], ChainMatch::FullHit { old_index: 5 });
        assert_eq!(old.payload(5), "R5");

        // Every uncolored new entity gets *some* uncolored old real id -
        // any one of the remaining nine, but never the colored one, which
        // is already claimed.
        for result in &results[..9] {
            let ChainMatch::FullHit { old_index } = result else {
                panic!("expected a hit, got {result:?}");
            };
            assert_ne!(*old_index, 5, "must never reuse the colored plane for an uncolored one");
        }

        // And every one of the ten old entities was claimed by exactly one
        // new entity - no double-claims, nothing left over.
        let mut claimed: Vec<usize> = results
            .iter()
            .map(|r| match r {
                ChainMatch::FullHit { old_index } => *old_index,
                other => panic!("expected a hit, got {other:?}"),
            })
            .collect();
        claimed.sort_unstable();
        assert_eq!(claimed, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn a_claimed_entity_is_never_matched_again_even_at_a_shallower_level() {
        // Regression guard mirroring `tests::a_consumed_position_is_never_matched_again`:
        // once an entity has been claimed via its deepest level, it must
        // not resurface as a candidate for a *different* new entity that
        // only reaches its shallower (creation) level.
        let mut old = ChainLog::new(vec![(chain_of(&["plane", "red"]), "R1")]);
        assert_eq!(
            old.match_chain(&chain_of(&["plane", "red"])),
            ChainMatch::FullHit { old_index: 0 }
        );
        // A second, unrelated entity that only matches at the creation
        // level must not be handed the already-claimed old entity.
        assert_eq!(old.match_chain(&chain_of(&["plane"])), ChainMatch::Miss);
    }
}
