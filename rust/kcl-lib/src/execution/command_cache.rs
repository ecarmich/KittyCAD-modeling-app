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
