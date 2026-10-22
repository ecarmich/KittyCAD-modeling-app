use std::collections::HashMap;

use ahash::AHashSet;
use kcmc::ModelingCmd;
use kittycad_modeling_cmds::websocket::ModelingCmdReq;
use kittycad_modeling_cmds::websocket::OkWebSocketResponseData;
use kittycad_modeling_cmds::{self as kcmc};
use uuid::Uuid;

use crate::ExecState;
use crate::ExecutorContext;
use crate::KclError;
use crate::SourceRange;
use crate::errors::KclErrorDetails;
use crate::exec::ArtifactCommand;
use crate::exec::IdGenerator;
use crate::exec::KclValue;
use crate::execution::EntityCloneInfo;
use crate::execution::FaceParentSolid;
use crate::execution::Solid;
use crate::execution::command_cache::IdRefs;
use crate::std::Args;

/// Developer toggle for the KCL 3.0 behavior where edge cuts (fillets and
/// chamfers) are sent to the engine immediately, in program order with other
/// modeling commands, instead of being deferred to the end of the batch.
///
/// Set this to `false` to restore the deferred behavior for every KCL version
/// while the immediate behavior is being evaluated. The KCL version gate in
/// [`ExecState::edge_cuts_are_immediate`] still applies when this is `true`.
pub(crate) const IMMEDIATE_EDGE_CUTS_IN_V3: bool = true;

/// Context and metadata needed to send a single modeling command.
///
/// Many functions consume Self so that the command ID isn't accidentally reused
/// among multiple modeling commands.
pub(crate) struct ModelingCmdMeta<'a> {
    /// The executor context, which contains the engine.
    pub ctx: &'a ExecutorContext,
    /// The source range of the command, used for error reporting.
    pub source_range: SourceRange,
    /// The id of the command, if it has been set by the caller or generated.
    id: Option<Uuid>,
}

impl<'a> ModelingCmdMeta<'a> {
    pub fn new(exec_state: &ExecState, ctx: &'a ExecutorContext, range: SourceRange) -> Self {
        ModelingCmdMeta {
            ctx,
            source_range: exec_state.mod_local.stdlib_entry_source_range.unwrap_or(range),
            id: None,
        }
    }

    pub fn with_id(exec_state: &ExecState, ctx: &'a ExecutorContext, range: SourceRange, id: Uuid) -> Self {
        ModelingCmdMeta {
            ctx,
            source_range: exec_state.mod_local.stdlib_entry_source_range.unwrap_or(range),
            id: Some(id),
        }
    }

    pub fn from_args(exec_state: &ExecState, args: &'a Args) -> Self {
        ModelingCmdMeta {
            ctx: &args.ctx,
            source_range: exec_state
                .mod_local
                .stdlib_entry_source_range
                .unwrap_or(args.source_range),
            id: None,
        }
    }

    pub fn from_args_id(exec_state: &ExecState, args: &'a Args, id: Uuid) -> Self {
        ModelingCmdMeta {
            ctx: &args.ctx,
            source_range: exec_state
                .mod_local
                .stdlib_entry_source_range
                .unwrap_or(args.source_range),
            id: Some(id),
        }
    }

    pub fn id(&mut self, id_generator: &mut IdGenerator) -> Uuid {
        if let Some(id) = self.id {
            return id;
        }
        let id = id_generator.next_uuid();
        self.id = Some(id);
        id
    }
}

impl ExecState {
    /// Add a modeling command to the batch but don't fire it right away.
    pub(crate) async fn batch_modeling_cmd(
        &mut self,
        meta: ModelingCmdMeta<'_>,
        cmd: ModelingCmd,
    ) -> Result<(), KclError> {
        self.batch_modeling_cmd_with_entity_clone_info(meta, cmd, None).await
    }

    pub(crate) async fn batch_modeling_cmd_with_entity_clone_info(
        &mut self,
        mut meta: ModelingCmdMeta<'_>,
        cmd: ModelingCmd,
        entity_clone_info: Option<EntityCloneInfo>,
    ) -> Result<(), KclError> {
        if self.is_in_sketch_block() {
            return Err(no_modeling_in_sketch_block_error(meta.source_range));
        }
        let id = meta.id(self.id_generator());
        self.push_command(ArtifactCommand {
            cmd_id: id,
            range: meta.source_range,
            command: cmd.clone(),
            entity_clone_info,
            omit_from_graph: false,
        });
        meta.ctx
            .engine
            .batch_modeling_cmd(&meta.ctx.engine_batch, id, meta.source_range, &cmd)
            .await
    }

    /// Add multiple modeling commands to the batch but don't fire them right
    /// away.
    pub(crate) async fn batch_modeling_cmds(
        &mut self,
        meta: ModelingCmdMeta<'_>,
        cmds: &[ModelingCmdReq],
    ) -> Result<(), KclError> {
        if self.is_in_sketch_block() {
            return Err(no_modeling_in_sketch_block_error(meta.source_range));
        }
        for cmd_req in cmds {
            self.push_command(ArtifactCommand {
                cmd_id: *cmd_req.cmd_id.as_ref(),
                range: meta.source_range,
                command: cmd_req.cmd.clone(),
                entity_clone_info: None,
                omit_from_graph: false,
            });
        }
        meta.ctx
            .engine
            .batch_modeling_cmds(&meta.ctx.engine_batch, meta.source_range, cmds)
            .await
    }

    /// Whether edge cuts (fillets and chamfers) are sent to the engine in
    /// program order, like any other modeling command.
    ///
    /// KCL 3.0: edge cuts execute immediately. Earlier versions defer them to
    /// the end of the batch; see [`Self::batch_end_cmd`]. This never varies
    /// within a single execution because it is keyed on the entry point's
    /// declared version.
    pub(crate) fn edge_cuts_are_immediate(&self) -> bool {
        IMMEDIATE_EDGE_CUTS_IN_V3 && self.entry_point_version_is_v3_or_higher()
    }

    /// Add an edge cut (fillet or chamfer) modeling command to the batch.
    ///
    /// KCL 3.0: the command is batched in order with other modeling commands,
    /// so it executes before whatever the program does next. The engine
    /// replaces a cut edge with a new face, so a later reference to that edge
    /// (for example `getOppositeEdge`) fails, matching the order the user
    /// wrote.
    ///
    /// Before KCL 3.0, the command is deferred to the end of the batch so that
    /// later references to the cut edge still resolve; see
    /// [`Self::batch_end_cmd`].
    pub(crate) async fn batch_edge_cut_cmd(
        &mut self,
        meta: ModelingCmdMeta<'_>,
        cmd: ModelingCmd,
    ) -> Result<(), KclError> {
        if self.edge_cuts_are_immediate() {
            self.batch_modeling_cmd(meta, cmd).await
        } else {
            self.batch_end_cmd(meta, cmd).await
        }
    }

    /// Add a modeling command to the batch that gets executed at the end of the
    /// file. This is good for something like fillet or chamfer where the engine
    /// would eat the path id if we executed it right away.
    pub(crate) async fn batch_end_cmd(
        &mut self,
        mut meta: ModelingCmdMeta<'_>,
        cmd: ModelingCmd,
    ) -> Result<(), KclError> {
        if self.is_in_sketch_block() {
            return Err(no_modeling_in_sketch_block_error(meta.source_range));
        }
        let id = meta.id(self.id_generator());
        // TODO: The order of the tracking of these doesn't match the order that
        // they're sent to the engine.
        self.push_command(ArtifactCommand {
            cmd_id: id,
            range: meta.source_range,
            command: cmd.clone(),
            entity_clone_info: None,
            omit_from_graph: false,
        });
        meta.ctx
            .engine
            .batch_end_cmd(&meta.ctx.engine_batch, id, meta.source_range, &cmd)
            .await
    }

    /// Send the modeling cmd and wait for the response.
    pub(crate) async fn send_modeling_cmd(
        &mut self,
        mut meta: ModelingCmdMeta<'_>,
        mut cmd: ModelingCmd,
    ) -> Result<OkWebSocketResponseData, KclError> {
        if self.is_in_sketch_block() {
            return Err(no_modeling_in_sketch_block_error(meta.source_range));
        }
        // A blocking send bypasses the queue entirely (this command was
        // never in it), so the queued-batch rewrite inside
        // `resolve_and_translate_pending` never sees it - translate its own
        // reference fields here too, using the same map, before anything
        // below builds on it. See that function's doc comment.
        let entity_real_ids = self.resolve_and_translate_pending(meta.ctx).await;
        if !entity_real_ids.is_empty() {
            cmd.for_each_id_ref_mut(&mut |rid| {
                if let Some(real) = entity_real_ids.get(rid) {
                    *rid = *real;
                }
            });
        }
        let id = meta.id(self.id_generator());
        self.push_command(ArtifactCommand {
            cmd_id: id,
            range: meta.source_range,
            command: cmd.clone(),
            entity_clone_info: None,
            omit_from_graph: false,
        });
        meta.ctx
            .engine
            .send_modeling_cmd(&meta.ctx.engine_batch, id, meta.source_range, &cmd)
            .await
    }

    /// Send a query-only modeling command that is recorded in command snapshots
    /// but omitted from the semantic artifact graph.
    pub(crate) async fn send_untracked_modeling_cmd(
        &mut self,
        mut meta: ModelingCmdMeta<'_>,
        mut cmd: ModelingCmd,
    ) -> Result<OkWebSocketResponseData, KclError> {
        if self.is_in_sketch_block() {
            return Err(no_modeling_in_sketch_block_error(meta.source_range));
        }
        // See `send_modeling_cmd`'s matching comment.
        let entity_real_ids = self.resolve_and_translate_pending(meta.ctx).await;
        if !entity_real_ids.is_empty() {
            cmd.for_each_id_ref_mut(&mut |rid| {
                if let Some(real) = entity_real_ids.get(rid) {
                    *rid = *real;
                }
            });
        }
        let id = meta.id(self.id_generator());
        self.push_command(ArtifactCommand {
            cmd_id: id,
            range: meta.source_range,
            command: cmd.clone(),
            entity_clone_info: None,
            omit_from_graph: true,
        });
        meta.ctx
            .engine
            .send_modeling_cmd(&meta.ctx.engine_batch, id, meta.source_range, &cmd)
            .await
    }

    /// Send the modeling cmd async and don't wait for the response.
    /// Add it to our list of async commands.
    pub(crate) async fn async_modeling_cmd(
        &mut self,
        mut meta: ModelingCmdMeta<'_>,
        cmd: &ModelingCmd,
    ) -> Result<(), KclError> {
        if self.is_in_sketch_block() {
            return Err(no_modeling_in_sketch_block_error(meta.source_range));
        }
        // See `send_modeling_cmd`'s matching comment.
        let entity_real_ids = self.resolve_and_translate_pending(meta.ctx).await;
        let mut cmd = cmd.clone();
        if !entity_real_ids.is_empty() {
            cmd.for_each_id_ref_mut(&mut |rid| {
                if let Some(real) = entity_real_ids.get(rid) {
                    *rid = *real;
                }
            });
        }
        let id = meta.id(self.id_generator());
        let tracked_command = match &cmd {
            ModelingCmd::ImportFiles(import_files) => {
                // Graph construction does not read imported file bytes. Retain the file paths
                // and format in execution state without duplicating the full input payload.
                let mut tracked_import = import_files.clone();
                for file in &mut tracked_import.files {
                    file.data.clear();
                }
                ModelingCmd::ImportFiles(tracked_import)
            }
            _ => cmd.clone(),
        };
        self.push_command(ArtifactCommand {
            cmd_id: id,
            range: meta.source_range,
            command: tracked_command,
            entity_clone_info: None,
            omit_from_graph: false,
        });
        meta.ctx.engine.async_modeling_cmd(id, meta.source_range, &cmd).await
    }

    /// Resolve anything the deferred content-cache mechanism (see
    /// `execution::command_cache`) has registered since the last call, and
    /// translate/filter the currently-queued-but-unsent batch in place
    /// before it's actually sent: a cache hit is dropped entirely, and
    /// *every* surviving command gets any id-reference field `IdRefs` finds
    /// rewritten from interpreter id to real id - not just a command whose
    /// own id happens to be one of this round's registered steps. That
    /// distinction matters: a command can reference a registered entity
    /// (e.g. `ObjectBringToFront` naming a solid whose creation went
    /// through the mechanism) without being registered itself, and an
    /// earlier version of this function only rewrote the former kind,
    /// silently leaving the latter's references pointing at an interpreter
    /// id the engine has never heard of.
    ///
    /// Returns the round's full interpreter-id -> real-id map (empty if
    /// nothing has ever been registered this round), so a caller about to
    /// send one *more* command directly - bypassing the queue entirely, as
    /// every blocking call below does - can translate that command's own
    /// reference fields the same way before handing it to the engine; the
    /// queued-batch rewrite above never sees that command, since it was
    /// never queued. Always re-reads the full accumulated map (not just
    /// what became newly resolved on this call) so a reference added in a
    /// *later*, separate call to an entity resolved in an *earlier* one
    /// still gets fixed, not just the common case of registration and
    /// reference happening in the same window.
    ///
    /// A cheap no-op (immediately returns an empty map, touches nothing)
    /// for every call that doesn't involve the mechanism at all - today,
    /// every call except the ones `std/sketch.rs`'s/`std/planes.rs`'s plane
    /// creation and `std/extrude.rs`'s edge-surface extrude queue into. See
    /// the design note's SS2 and SS4.
    pub(crate) async fn resolve_and_translate_pending(&self, ctx: &ExecutorContext) -> HashMap<Uuid, Uuid> {
        let decisions = ctx.cache.resolve_pending().await;
        let entity_real_ids = ctx.cache.resolved_entity_ids().await;
        if entity_real_ids.is_empty() {
            return entity_real_ids;
        }
        ctx.engine_batch
            .with_pending_batch(|batch| {
                batch
                    .into_iter()
                    .filter_map(|(mut req, range)| {
                        let kcmc::websocket::WebSocketRequest::ModelingCmdReq(req_inner) = &mut req else {
                            return Some((req, range));
                        };
                        // A hit: drop it, nothing to send.
                        if let Some((_, true)) = decisions.get(&req_inner.cmd_id.0) {
                            return None;
                        }
                        // A miss: rewrite this command's own id.
                        if let Some((real_id, false)) = decisions.get(&req_inner.cmd_id.0) {
                            req_inner.cmd_id = (*real_id).into();
                        }
                        // Whether or not this command is itself one of this
                        // round's registered steps, translate every
                        // reference field `IdRefs` finds - a no-op for any
                        // id that isn't in `entity_real_ids`.
                        req_inner.cmd.for_each_id_ref_mut(&mut |rid| {
                            if let Some(real) = entity_real_ids.get(rid) {
                                *rid = *real;
                            }
                        });
                        Some((req, range))
                    })
                    .collect()
            })
            .await;
        entity_real_ids
    }

    /// Force flush the batch queue.
    pub(crate) async fn flush_batch(
        &mut self,
        meta: ModelingCmdMeta<'_>,
        // Whether or not to flush the end commands as well.
        // We only do this at the very end of the file.
        batch_end: bool,
    ) -> Result<OkWebSocketResponseData, KclError> {
        if self.is_in_sketch_block() {
            return Err(no_modeling_in_sketch_block_error(meta.source_range));
        }

        self.resolve_and_translate_pending(meta.ctx).await;

        meta.ctx
            .engine
            .flush_batch(&meta.ctx.engine_batch, batch_end, meta.source_range)
            .await
    }

    /// Flush just the fillets and chamfers for this specific SolidSet.
    pub(crate) async fn flush_batch_for_solids(
        &mut self,
        meta: ModelingCmdMeta<'_>,
        solids: &[Solid],
    ) -> Result<(), KclError> {
        if self.is_in_sketch_block() {
            return Err(no_modeling_in_sketch_block_error(meta.source_range));
        }
        // Make sure we don't traverse sketches more than once.
        let mut traversed_sketches = AHashSet::new();

        // Collect all the fillet/chamfer ids for the solids.
        let mut ids = Vec::new();
        for solid in solids {
            // We need to traverse the solids that share the same sketch.
            let sketch_id = solid.sketch_id().unwrap_or(solid.id);
            if !traversed_sketches.contains(&sketch_id) {
                // Find all the solids on the same shared sketch.
                ids.extend(
                    self.stack()
                        .walk_call_stack_with(|value| match value {
                            KclValue::Solid { value } if value.sketch_id().unwrap_or(value.id) == sketch_id => {
                                Some(value.get_all_edge_cut_ids().collect::<Vec<_>>())
                            }
                            _ => None,
                        })?
                        .into_iter()
                        .flatten(),
                );
                traversed_sketches.insert(sketch_id);
            }

            ids.extend(solid.get_all_edge_cut_ids());
        }

        self.flush_batch_for_edge_cut_ids(meta, ids).await
    }

    /// Flush just the fillets and chamfers for the parent solids of face-backed sketches.
    pub(crate) async fn flush_batch_for_face_parent_solids(
        &mut self,
        meta: ModelingCmdMeta<'_>,
        solids: &[FaceParentSolid],
    ) -> Result<(), KclError> {
        if self.is_in_sketch_block() {
            return Err(no_modeling_in_sketch_block_error(meta.source_range));
        }
        // Make sure we don't traverse sketches more than once.
        let mut traversed_sketches = AHashSet::new();

        // Collect all the fillet/chamfer ids for the solids.
        let mut ids = Vec::new();
        for solid in solids {
            // We need to traverse the solids that share the same sketch.
            let sketch_id = solid.sketch_or_solid_id();
            if !traversed_sketches.contains(&sketch_id) {
                // Find all the solids on the same shared sketch.
                ids.extend(
                    self.stack()
                        .walk_call_stack_with(|value| match value {
                            KclValue::Solid { value } if value.sketch_id().unwrap_or(value.id) == sketch_id => {
                                Some(value.get_all_edge_cut_ids().collect::<Vec<_>>())
                            }
                            _ => None,
                        })?
                        .into_iter()
                        .flatten(),
                );
                traversed_sketches.insert(sketch_id);
            }

            ids.extend(solid.edge_cut_ids.iter().copied());
        }

        self.flush_batch_for_edge_cut_ids(meta, ids).await
    }

    async fn flush_batch_for_edge_cut_ids(
        &mut self,
        meta: ModelingCmdMeta<'_>,
        ids: Vec<Uuid>,
    ) -> Result<(), KclError> {
        // We can return early if there are no fillets or chamfers.
        if ids.is_empty() {
            return Ok(());
        }

        // We want to move these fillets and chamfers from batch_end to batch so they get executed
        // before whatever we call next.
        meta.ctx.engine_batch.move_batch_end_to_batch(ids).await;

        // Run flush.
        // Yes, we do need to actually flush the batch here, or references will fail later.
        self.flush_batch(meta, false).await?;

        Ok(())
    }
}

fn no_modeling_in_sketch_block_error(range: SourceRange) -> KclError {
    KclError::new_invalid_expression(KclErrorDetails::new(
        "Modeling commands communicating with the engine cannot be used inside a sketch block".to_owned(),
        vec![range],
    ))
}
