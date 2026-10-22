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
//!
//! # Response caching (query results), separate from id reuse
//!
//! Everything above decides whether to *send* a command. [`query_content_id`]
//! and [`ResponseCache`] are a separate, narrower piece: whether a
//! query-style command (one that reads state back rather than creating or
//! mutating anything) even needs to go to the engine at all, when its
//! target's chain proves nothing about it changed this round. See
//! [`query_content_id`]'s doc comment for the correctness argument and
//! [`ResponseCache`]'s for why this, too, is not yet wired into anything
//! real.

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

/// One entity registered this round, not yet (or not only) resolved: its
/// accumulated [`EntityChain`] and the interpreter id of every step that
/// contributed to it. `steps[0]` is always the entity's own creation step -
/// by construction the same id the caller registered the entity itself
/// under (see [`CommandCacheRound::register_creation`]) - and `steps[n]`
/// for `n >= 1` is the `n`th mutation's own step id (see
/// [`CommandCacheRound::register_mutation`]), used only to key that one
/// step's hit/miss decision, never to identify the entity itself.
#[derive(Debug, Clone)]
struct PendingEntity {
    chain: EntityChain,
    steps: Vec<uuid::Uuid>,
}

/// Accumulates one real run's entity chains and the real-id decisions
/// resolved from them so far, both to make this run's own decisions (via
/// [`Self::real_id_of`], for substituting an already-resolved entity's real
/// id into a later command that references it) and to become the next
/// run's [`ChainLog`] (via [`Self::into_log`]).
///
/// Entities are registered as soon as their commands are produced
/// ([`Self::register_creation`], [`Self::register_mutation`]) but not
/// resolved to a real id immediately - [`Self::resolve_pending`] does that,
/// in a batch, for every entity registered since the last call. This is
/// the split described in the design note's SS2: an entity's *interpreter*
/// id (what KCL code and other commands reference, assigned immediately,
/// never rewritten) is decoupled from its *real* engine id (decided only
/// once enough of the round is known to disambiguate correctly against
/// content-identical siblings via chaining - see SS3).
///
/// This specializes [`ChainLog`]'s payload to `uuid::Uuid`: all that needs
/// to survive between runs for each logged entity is the real engine id it
/// was (or would have been) sent with.
#[derive(Debug, Default)]
pub(crate) struct CommandCacheRound {
    entities: Vec<PendingEntity>,
    /// Entity (creation step) id -> index into `entities`, so
    /// `register_mutation` can find the right chain to extend.
    entity_index: HashMap<uuid::Uuid, usize>,
    /// Indices into `entities` not yet resolved - drained by
    /// `resolve_pending`.
    pending: Vec<usize>,
    /// Entity id -> its resolved real engine id, for every entity resolved
    /// so far this round (by any call to `resolve_pending`, not just the
    /// most recent one) - read by [`Self::real_id_of`].
    entity_real_id: HashMap<uuid::Uuid, uuid::Uuid>,
    pub(crate) hits: usize,
    pub(crate) misses: usize,
}

impl CommandCacheRound {
    /// Register a new entity's creation. `entity_id` is both the entity's
    /// identity (what a later reference field should resolve to via
    /// [`Self::real_id_of`]) and this creation step's own id (what
    /// [`Self::resolve_pending`]'s decision map keys the hit/miss decision
    /// on) - the same value, because there is nothing "before" a creation
    /// in its chain to tell the two apart.
    pub(crate) fn register_creation(&mut self, entity_id: uuid::Uuid, content_id: ContentId) {
        let index = self.entities.len();
        self.entities.push(PendingEntity {
            chain: EntityChain::new(content_id),
            steps: vec![entity_id],
        });
        self.entity_index.insert(entity_id, index);
        self.pending.push(index);
    }

    /// Register a mutation against an already-registered entity.
    /// `mutation_step_id` is this *mutation's own* id (e.g. a
    /// `PlaneSetColor` command's own `cmd_id`, never reused as any
    /// identity - only used to key this one step's own hit/miss decision);
    /// `entity_id` stays the entity's (e.g. the plane's), used for
    /// reference-field translation.
    pub(crate) fn register_mutation(
        &mut self,
        entity_id: uuid::Uuid,
        mutation_step_id: uuid::Uuid,
        mutation_content: &[u8],
    ) {
        let &index = self
            .entity_index
            .get(&entity_id)
            .expect("register_mutation called before register_creation for this entity");
        self.entities[index].chain.extend(mutation_content);
        self.entities[index].steps.push(mutation_step_id);
        if !self.pending.contains(&index) {
            self.pending.push(index);
        }
    }

    /// Resolve every entity registered (or re-registered via a mutation)
    /// since the last call, against `old_log`. Returns a decision per
    /// *step* id (every id passed to `register_creation` or
    /// `register_mutation` since the last call): `(real_id, is_hit)` - the
    /// caller must skip sending a hit, and must rewrite a miss's own id (and
    /// any reference field elsewhere pointing at the entity, via
    /// [`Self::real_id_of`]) to `real_id` before sending.
    pub(crate) fn resolve_pending(
        &mut self,
        old_log: Option<&mut ChainLog<uuid::Uuid>>,
    ) -> HashMap<uuid::Uuid, (uuid::Uuid, bool)> {
        let pending = std::mem::take(&mut self.pending);
        if pending.is_empty() {
            return HashMap::new();
        }

        let chains: Vec<EntityChain> = pending.iter().map(|&i| self.entities[i].chain.clone()).collect();
        let mut decisions = HashMap::new();

        match old_log {
            Some(log) => {
                // `log` is reborrowed mutably by `match_round` and then
                // immutably by `log.payload(...)` below - sequential, not
                // overlapping, so this is an ordinary reborrow, not a
                // conflict.
                let matches = match_round(log, &chains);
                for (&index, chain_match) in pending.iter().zip(matches) {
                    let (real_id, matched_depth) = match chain_match {
                        ChainMatch::FullHit { old_index } => {
                            self.hits += 1;
                            (*log.payload(old_index), Some(self.entities[index].chain.depth()))
                        }
                        ChainMatch::PartialHit {
                            old_index,
                            matched_depth,
                        } => {
                            self.hits += 1;
                            (*log.payload(old_index), Some(matched_depth))
                        }
                        ChainMatch::Miss => {
                            self.misses += 1;
                            // No history to resolve a real id from - reuse
                            // the entity's own creation (interpreter) id
                            // deterministically, rather than minting a
                            // fresh random one. See the `None` arm below
                            // for why this matters far beyond this one
                            // round.
                            (self.entities[index].steps[0], None)
                        }
                    };
                    let entity = &self.entities[index];
                    self.entity_real_id.insert(entity.steps[0], real_id);
                    for (depth, &step_id) in entity.steps.iter().enumerate() {
                        let is_hit = matched_depth.is_some_and(|matched| depth <= matched);
                        decisions.insert(step_id, (real_id, is_hit));
                    }
                }
            }
            None => {
                // No previous round to match against at all (the very
                // first run): every entity is a miss. Resolves to its own
                // creation (interpreter) id, deterministically, same as
                // the `ChainMatch::Miss` arm above - not a fresh random
                // id. This is what keeps ids reproducible across two
                // entirely independent fresh executions of the same
                // program (no shared `Cache`, so always this branch): the
                // same pre-existing guarantee this codebase relied on
                // before this mechanism existed, when every entity's wire
                // id came directly from `next_uuid()`'s deterministic
                // sequence with nothing in between. A random id here would
                // silently break that guarantee for every entity this
                // mechanism covers - not hypothetically: generalizing it
                // to default planes (`std/sketch.rs`'s
                // `ensure_sketch_plane_in_engine`) is exactly what
                // surfaced this, via `execution::tests::mock_has_stable_ids`.
                for &index in &pending {
                    self.misses += 1;
                    let entity = &self.entities[index];
                    let real_id = entity.steps[0];
                    self.entity_real_id.insert(entity.steps[0], real_id);
                    for &step_id in &entity.steps {
                        decisions.insert(step_id, (real_id, false));
                    }
                }
            }
        }
        decisions
    }

    /// Look up an already-resolved entity's real id, for substituting into
    /// a later command's *reference* field (see the module docs on why a
    /// raw interpreter id must never be sent directly). Only meaningful
    /// once [`Self::resolve_pending`] has resolved this entity - true for
    /// any entity this round has resolved so far, by any call.
    pub(crate) fn real_id_of(&self, entity_id: uuid::Uuid) -> Option<uuid::Uuid> {
        self.entity_real_id.get(&entity_id).copied()
    }

    /// A snapshot of every entity resolved so far this round, id -> real
    /// id - for a caller that needs the whole map at once (e.g. to
    /// translate a batch of queued commands synchronously) rather than one
    /// [`Self::real_id_of`] lookup at a time.
    pub(crate) fn resolved_entity_ids(&self) -> HashMap<uuid::Uuid, uuid::Uuid> {
        self.entity_real_id.clone()
    }

    /// Finalize this run's decisions into the log the next run will match
    /// against. Resolves anything still pending first, so nothing
    /// registered but never explicitly flushed is silently dropped from
    /// the next run's log.
    pub(crate) fn into_log(mut self, old_log: Option<&mut ChainLog<uuid::Uuid>>) -> ChainLog<uuid::Uuid> {
        self.resolve_pending(old_log);
        let entries = self
            .entities
            .into_iter()
            .map(|entity| {
                let real_id = *self
                    .entity_real_id
                    .get(&entity.steps[0])
                    .expect("every entity was just resolved above");
                (entity.chain, real_id)
            })
            .collect();
        ChainLog::new(entries)
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

/// Content id for a query-style (`CommandEffect::Query`) command's own
/// result, so an unchanged query against an unchanged target can reuse last
/// round's response instead of asking the engine again - see the design
/// note's SS4 for the correctness argument this implements directly: this is
/// safe only because a target whose chain is byte-identical to last round's
/// has, by construction, had nothing sent for it this round (a `FullHit`,
/// per [`ChainMatch::FullHit`]) - it is the same "nothing was touched" fact
/// id reuse already relies on, not a new assumption that the engine is
/// deterministic.
///
/// `target_chain_value` must be the query's target entity's *current*
/// (fully resolved) chain value - [`EntityChain::current`] - folded in
/// exactly the same way a mutation's content id is folded into the chain by
/// [`EntityChain::extend`]. This is the only thing that makes the safety
/// property automatic rather than a separately-tracked invariant: if the
/// target's chain differs at all from last round's (anything short of a
/// `FullHit` - a genuine miss, or a `PartialHit` where something upstream of
/// this query's target *did* change), the folded-in chain value differs too,
/// so the resulting content id differs, so [`CommandLog::match_next`]
/// reports a miss and the caller is forced to actually query the engine -
/// never a stale reuse by accident. A query against two different entities,
/// or two different queries against the same entity (e.g.
/// `Solid3dGetAllEdgeFaces` for two different edges of the same solid), stay
/// distinguishable because `query_fields` - the query's own literal fields,
/// not counting the fields that are themselves entity references (those
/// should already have been substituted by their *own* current chain value,
/// same rule as any other content id - see the module docs) - is folded in
/// too.
pub(crate) fn query_content_id(target_chain_value: ContentId, query_fields: &[u8]) -> ContentId {
    let mut bytes = Vec::with_capacity(target_chain_value.len() + query_fields.len());
    bytes.extend_from_slice(&target_chain_value);
    bytes.extend_from_slice(query_fields);
    content_id_from_bytes(&bytes)
}

/// A previous round's cached query responses, content-addressed exactly the
/// way [`CommandLog`] addresses any other content, via [`query_content_id`].
/// Deliberately just [`CommandLog`] itself under a self-documenting name,
/// not a new type with new matching logic: a query response's safety
/// property (never reusing one whose target actually changed) falls
/// entirely out of how its content id is built, as [`query_content_id`]'s
/// doc comment argues, not out of anything [`CommandLog`] would need to do
/// differently for it. `superseded` is meaningless for this use (a response
/// that goes unclaimed this round was simply a query nothing asked again -
/// not a deletion to act on) and callers caching responses should ignore it.
///
/// **Not yet wired into anything real** - same standalone, tested-in-
/// isolation scoping as the rest of this module's not-yet-wired pieces (see
/// the module docs). Unlike those, this one has a hard prerequisite beyond
/// its own wiring: `target_chain_value` only exists for an entity that was
/// actually registered via [`CommandCacheRound::register_creation`]/
/// [`CommandCacheRound::register_mutation`], and today that's only
/// `std/planes.rs`'s two call sites (see [`match_round`]'s doc comment on
/// why generalizing past them isn't safe yet). None of the real query call
/// sites this is meant for (`extrude.rs`, `edge.rs`, `csg.rs`, etc. - see the
/// design note's SS1) query an entity the chain mechanism knows about at
/// all yet, so there is currently nothing real to wire this into - doing so
/// before that prerequisite lands would mean computing `target_chain_value`
/// from something other than an actual resolved chain, which is exactly the
/// kind of "looks generalized but still decides eagerly" trap the design
/// note's SS3.2 already warns against for the matcher itself.
pub(crate) type ResponseCache<Response> = CommandLog<Response>;

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

#[cfg(test)]
mod response_cache_tests {
    use super::*;

    fn id(label: &str) -> ContentId {
        content_id_from_bytes(label.as_bytes())
    }

    fn chain_of(levels: &[&str]) -> EntityChain {
        let mut chain = EntityChain::new(id(levels[0]));
        for level in &levels[1..] {
            chain.extend(level.as_bytes());
        }
        chain
    }

    /// The core worked example: a query against an entity whose chain this
    /// round is byte-identical to last round's (a `FullHit`, nothing sent
    /// for it) reuses last round's cached response instead of needing to
    /// query the engine again.
    #[test]
    fn a_query_against_an_unchanged_entity_reuses_last_rounds_response() {
        let old_chain = chain_of(&["plane"]);
        let old_query_id = query_content_id(old_chain.current(), b"edge_a");
        let mut old_responses: ResponseCache<&str> = CommandLog::new(vec![LogEntry {
            content_id: old_query_id,
            payload: "faces=[1,2,3]",
        }]);

        // This round: same entity, nothing mutated it, so its chain is
        // byte-identical to last round's - exactly what a `FullHit` means.
        let new_chain = chain_of(&["plane"]);
        let new_query_id = query_content_id(new_chain.current(), b"edge_a");
        assert_eq!(
            new_query_id, old_query_id,
            "an unchanged entity's chain value must fold into an identical content id"
        );

        assert_eq!(
            old_responses.match_next(new_query_id),
            MatchDecision::Hit { old_index: 0 }
        );
        assert_eq!(*old_responses.payload(0), "faces=[1,2,3]");
    }

    /// The safety property this is actually for: if the entity being
    /// queried was mutated (or recreated) this round - anything short of a
    /// `FullHit` - its chain value differs, so the content id differs, so
    /// the cache reports a miss and the caller is forced to ask the engine
    /// for real. A stale response can never be reused by accident, because
    /// nothing here ever reuses by anything other than exact content match.
    #[test]
    fn a_query_against_a_mutated_entity_never_reuses_the_stale_response() {
        let old_chain = chain_of(&["plane"]);
        let old_query_id = query_content_id(old_chain.current(), b"edge_a");
        let mut old_responses: ResponseCache<&str> = CommandLog::new(vec![LogEntry {
            content_id: old_query_id,
            payload: "faces=[1,2,3]",
        }]);

        // This round: the same entity, but it picked up a mutation (e.g. a
        // color change) since last round - its chain value is no longer
        // byte-identical, even though the query's own fields didn't change.
        let mutated_chain = chain_of(&["plane", "red"]);
        let mutated_query_id = query_content_id(mutated_chain.current(), b"edge_a");
        assert_ne!(mutated_query_id, old_query_id);
        assert_eq!(old_responses.match_next(mutated_query_id), MatchDecision::Miss);
    }

    /// Two distinct queries against the very same unchanged entity (e.g.
    /// `Solid3dGetAllEdgeFaces` for two different edges of one solid) must
    /// not collide with each other just because they share a target - the
    /// query's own fields are folded in too.
    #[test]
    fn two_different_queries_against_the_same_entity_are_cached_independently() {
        let chain = chain_of(&["plane"]);
        let query_a = query_content_id(chain.current(), b"edge_a");
        let query_b = query_content_id(chain.current(), b"edge_b");
        assert_ne!(query_a, query_b);
    }
}
