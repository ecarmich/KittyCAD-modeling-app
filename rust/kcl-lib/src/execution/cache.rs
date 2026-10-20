//! Functions for helping with caching an ast and finding the parts the changed.

use std::collections::HashSet;
use std::sync::Arc;

use indexmap::IndexMap;
use itertools::EitherOrBoth;
use itertools::Itertools;
use kcl_api::Object;
use kcl_api::ObjectId;
use kittycad_modeling_cmds::KclVersion;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::ExecOutcome;
use crate::ExecutorContext;
use crate::SourceRange;
use crate::errors::KclError;
use crate::execution::ConstraintKey;
use crate::execution::ConstraintState;
use crate::execution::EnvironmentRef;
use crate::execution::ExecutorSettings;
use crate::execution::KclValue;
use crate::execution::KclValueView;
use crate::execution::annotations;
#[cfg(test)]
use crate::execution::computed_kcl_version;
use crate::execution::memory::Stack;
use crate::execution::state::ModuleInfoMap;
use crate::execution::state::NotYetAdded;
use crate::execution::state::{self as exec_state};
use crate::modules::ModuleId;
use crate::modules::ModulePath;
use crate::modules::ModuleSource;
use crate::parsing::ast::types::Annotation;
use crate::parsing::ast::types::BodyItem;
use crate::parsing::ast::types::Node;
use crate::parsing::ast::types::Program;
use crate::walk::Node as WalkNode;

#[derive(Debug, Default)]
struct CacheInner {
    /// Last successful execution state for the cache.
    pub old_ast: Option<GlobalState>,
    // The last successful run's memory. Not cleared after an unsuccessful run.
    pub prev_memory: Option<SketchModeState>,
}

/// Cached execution state for one engine session and its associated mock execution.
///
/// Clones share the same state. A default cache is empty and independent of all
/// other caches. Callers must serialize executions that share an engine session;
/// this cache only locks individual state accesses.
#[derive(Debug, Clone)]
pub struct Cache(Arc<dyn CacheStorage>);

// Hide the cache's concrete state graph from Send/Sync proofs in callers.
trait CacheStorage: std::fmt::Debug + Send + Sync {
    fn inner(&self) -> &RwLock<CacheInner>;
}

impl CacheStorage for RwLock<CacheInner> {
    fn inner(&self) -> &RwLock<CacheInner> {
        self
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(CacheInner::default())))
    }
}

impl Cache {
    /// Read the old AST state from the cache.
    pub(super) async fn read_old_ast(&self) -> Option<GlobalState> {
        self.0.inner().read().await.old_ast.clone()
    }

    pub(super) async fn write_old_ast(&self, old_state: GlobalState) {
        let mut inner = self.0.inner().write().await;
        inner.old_ast = Some(old_state);
    }

    pub(crate) async fn read_old_memory(&self) -> Option<SketchModeState> {
        self.0.inner().read().await.prev_memory.clone()
    }

    pub(crate) async fn write_old_memory(&self, mem: SketchModeState) {
        let mut inner = self.0.inner().write().await;
        inner.prev_memory = Some(mem);
    }

    /// Invalidate live execution state while preserving usable sketch-mode memory.
    pub async fn bust_cache(&self) {
        let mut inner = self.0.inner().write().await;
        inner.old_ast = None;
    }

    /// Clear the memory reused by mock execution.
    pub async fn clear_mem_cache(&self) {
        let mut inner = self.0.inner().write().await;
        inner.prev_memory = None;
    }

    /// Read a named value from the previous sketch-mode execution.
    pub async fn read_old_memory_var(&self, name: &str) -> Option<KclValue> {
        self.read_old_memory()
            .await
            .and_then(|memory| memory.stack.get(name, SourceRange::default()).ok())
    }
}

/// Information for the caching an AST and smartly re-executing it if we can.
#[derive(Debug, Clone)]
pub struct CacheInformation<'a> {
    pub ast: &'a Node<Program>,
    pub settings: &'a ExecutorSettings,
    /// The KCL version of this execution.
    pub kcl_version: KclVersion,
}

impl<'a> CacheInformation<'a> {
    #[cfg(test)]
    fn new(ast: &'a Node<Program>, settings: &'a ExecutorSettings) -> Self {
        // If the version couldn't be parsed, the default is used.
        let kcl_version = computed_kcl_version(ast);
        Self {
            ast,
            settings,
            kcl_version,
        }
    }
}

/// The cached state of the whole program.
#[derive(Debug, Clone)]
pub(super) struct GlobalState {
    pub(super) main: ModuleState,
    /// The exec state.
    pub(super) exec_state: exec_state::GlobalState,
    /// The last settings used for execution.
    pub(super) settings: ExecutorSettings,
    /// The KCL version of this execution.
    pub kcl_version: KclVersion,
}

impl GlobalState {
    pub fn new(
        state: exec_state::ExecState,
        settings: ExecutorSettings,
        kcl_version: KclVersion,
        ast: Node<Program>,
        result_env: EnvironmentRef,
    ) -> Self {
        Self {
            main: ModuleState {
                ast,
                exec_state: state.mod_local,
                result_env,
            },
            exec_state: state.global,
            settings,
            kcl_version,
        }
    }

    pub fn with_settings(mut self, settings: ExecutorSettings) -> GlobalState {
        self.settings = settings;
        self
    }

    pub fn reconstitute_exec_state(&self, ctx: &ExecutorContext) -> exec_state::ExecState {
        exec_state::ExecState {
            execution_callbacks: ctx.execution_callbacks.clone(),
            global: self.exec_state.clone(),
            mod_local: self.main.exec_state.clone(),
        }
    }

    pub async fn into_exec_outcome(self, ctx: &ExecutorContext) -> Result<ExecOutcome, KclError> {
        // Fields are opt-in so that we don't accidentally leak private internal
        // state when we add more to ExecState.
        let variables = self.main.exec_state.variables(self.main.result_env)?;
        #[cfg(test)]
        let test_program_memory = variables.clone();
        let variables = variables
            .into_iter()
            .map(|(key, value)| (key, KclValueView::from(value)))
            .collect();
        Ok(ExecOutcome {
            variables,
            filenames: self.exec_state.filenames(),
            from_cache: false,
            operations: self.exec_state.operations_by_module(),
            artifact_graph: self.exec_state.artifacts.graph,
            scene_objects: self.exec_state.root_module_artifacts.scene_objects,
            source_range_to_object: self.exec_state.root_module_artifacts.source_range_to_object,
            var_solutions: self.exec_state.root_module_artifacts.var_solutions,
            refactor_metadata: self.exec_state.root_module_artifacts.refactor_metadata.clone(),
            issues: self.exec_state.issues,
            source_files: self.exec_state.id_to_source,
            default_planes: ctx.engine.get_default_planes().read().await.clone(),
            #[cfg(test)]
            test_program_memory,
        })
    }

    pub fn mock_memory_state(&self) -> Result<SketchModeState, KclError> {
        let mut stack = self.main.exec_state.stack.deep_clone()?;
        stack.restore_env(self.main.result_env)?;

        Ok(SketchModeState {
            stack,
            module_infos: self.exec_state.module_infos.clone(),
            path_to_source_id: self.exec_state.path_to_source_id.clone(),
            id_to_source: self.exec_state.id_to_source.clone(),
            never_type_ranges: self.exec_state.never_type_ranges.clone(),
            constraint_state: self.main.exec_state.constraint_state.clone(),
            scene_objects: self.exec_state.root_module_artifacts.scene_objects.clone(),
            std_not_yet_added: self.exec_state.std_not_yet_added.clone(),
            kcl_version: self.exec_state.entry_point_kcl_version.unwrap_or_default(),
        })
    }
}

/// Per-module cached state
#[derive(Debug, Clone)]
pub(super) struct ModuleState {
    /// The AST of the module.
    pub(super) ast: Node<Program>,
    /// The ExecState of the module.
    pub(super) exec_state: exec_state::ModuleState,
    /// The memory env for the module.
    pub(super) result_env: EnvironmentRef,
}

/// Cached state for sketch mode.
#[derive(Debug, Clone)]
pub(crate) struct SketchModeState {
    /// The stack of the main module.
    pub stack: Stack,
    /// The module info map.
    pub module_infos: ModuleInfoMap,
    /// Map from source file path to module ID.
    pub path_to_source_id: IndexMap<ModulePath, ModuleId>,
    /// Map from module ID to source file contents.
    pub id_to_source: IndexMap<ModuleId, ModuleSource>,
    /// Deferred `never` type uses in imported local modules.
    pub never_type_ranges: IndexMap<ModuleId, Vec<SourceRange>>,
    /// Sticky per-constraint state persisted across sketch-mode mock solves.
    pub constraint_state: IndexMap<ObjectId, IndexMap<ConstraintKey, ConstraintState>>,
    /// The scene objects.
    pub scene_objects: Vec<Object>,
    /// See `GlobalState::std_not_yet_added`. Restored because a run reusing
    /// this memory skips the prelude.
    pub std_not_yet_added: IndexMap<String, NotYetAdded>,
    /// The effective kclVersion (declared, or the default) of the program that
    /// wrote this memory; see [`Self::reusable_for`].
    pub kcl_version: KclVersion,
}

impl SketchModeState {
    /// Whether a program with the effective `kcl_version` may reuse this
    /// memory. Memory from another version keeps bindings, module outcomes,
    /// and a prelude this program must not see, and the LSP worker reuses
    /// memory with no other invalidation. Only the version counts: after other
    /// settings changes the frontend may execute a single sketch, which needs
    /// this memory and cannot rebuild it.
    pub(crate) fn reusable_for(&self, kcl_version: KclVersion) -> bool {
        self.kcl_version == kcl_version
    }
}

#[cfg(test)]
impl SketchModeState {
    pub(crate) fn new_for_tests() -> Self {
        Self {
            stack: Stack::new_for_tests(),
            module_infos: ModuleInfoMap::default(),
            path_to_source_id: Default::default(),
            id_to_source: Default::default(),
            never_type_ranges: Default::default(),
            constraint_state: Default::default(),
            scene_objects: Vec::new(),
            std_not_yet_added: Default::default(),
            kcl_version: KclVersion::default(),
        }
    }
}

/// The result of a cache check.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub(super) enum CacheResult {
    ReExecute {
        /// Should we clear the scene and start over?
        clear_scene: bool,
        /// Do we need to reapply settings?
        reapply_settings: bool,
        /// The program that needs to be executed.
        program: Node<Program>,
    },
    /// Check only the imports, and not the main program.
    /// Before sending this we already checked the main program and it is the same.
    /// And we made sure the import statements > 0.
    CheckImportsOnly {
        /// Argument is whether we need to reapply settings.
        reapply_settings: bool,
        /// The ast of the main file, which did not change.
        ast: Node<Program>,
    },
    /// Argument is whether we need to reapply settings.
    NoAction(bool),
    /// Statements were removed from the trailing end of the program, and every
    /// removed statement's value (if any) was a geometry kind whose engine
    /// object ids we can identify from the last successful run. Rather than
    /// clearing and re-executing the whole scene, delete just those engine
    /// objects. Nothing needs to be re-executed, since nothing after the
    /// deletion point exists in the new program.
    SurgicalDelete {
        /// The engine object ids to remove. May be empty, if the deleted
        /// statements didn't produce any scene geometry (e.g. they only bound
        /// plain values).
        object_ids: HashSet<Uuid>,
        /// Do we need to reapply settings?
        reapply_settings: bool,
    },
}

/// Given an old ast, old program memory and new ast, find the parts of the code that need to be
/// re-executed.
///
/// `old_vars` lazily returns the top-level variable bindings from the end of the last successful
/// execution (i.e. the values `old.ast` produced), or `None` if they're not available (e.g. there
/// was no previous successful execution). It's used to recognize a "pure delete" of statements
/// from the end of the program without re-executing or clearing the scene; a pure delete falls
/// back to a full cache bust, same as before this was supported, if it returns `None`. It's a
/// closure, rather than the bindings themselves, because computing them snapshots (and so clones)
/// every top-level value, which would be wasted work on every other kind of cache hit - it's only
/// called if we actually find a pure trailing delete to classify.
///
/// This function should never error, because in the case of any internal error, we should just pop
/// the cache.
pub(super) async fn get_changed_program(
    old: CacheInformation<'_>,
    new: CacheInformation<'_>,
    old_vars: impl FnOnce() -> Option<IndexMap<String, KclValue>>,
) -> CacheResult {
    let mut reapply_settings = false;

    if old.kcl_version != new.kcl_version {
        // If the KCL version differs, we need to send the version and
        // re-execute.
        return CacheResult::ReExecute {
            clear_scene: true,
            reapply_settings: true,
            program: new.ast.clone(),
        };
    }

    // If the settings are different we might need to bust the cache.
    // We specifically do this before checking if they are the exact same.
    if old.settings != new.settings {
        // If anything else is different we may not need to re-execute, but rather just
        // run the settings again.
        reapply_settings = true;
    }

    // If the ASTs are the EXACT same we return None.
    // We don't even need to waste time computing the digests.
    if old.ast == new.ast {
        // First we need to make sure an imported file didn't change it's ast.
        // We know they have the same imports because the ast is the same.
        // If we have no imports, we can skip this.
        if !old.ast.has_import_statements() {
            return CacheResult::NoAction(reapply_settings);
        }

        // Tell the CacheResult we need to check all the imports, but the main ast is the same.
        return CacheResult::CheckImportsOnly {
            reapply_settings,
            ast: old.ast.clone(),
        };
    }

    // We have to clone just because the digests are stored inline :-(
    let mut old_ast = old.ast.clone();
    let mut new_ast = new.ast.clone();

    // The digests should already be computed, but just in case we don't
    // want to compare against none.
    old_ast.compute_digest();
    new_ast.compute_digest();

    // Check if the digest is the same.
    if old_ast.digest == new_ast.digest {
        // First we need to make sure an imported file didn't change it's ast.
        // We know they have the same imports because the ast is the same.
        // If we have no imports, we can skip this.
        if !old.ast.has_import_statements() {
            return CacheResult::NoAction(reapply_settings);
        }

        // Tell the CacheResult we need to check all the imports, but the main ast is the same.
        return CacheResult::CheckImportsOnly {
            reapply_settings,
            ast: old.ast.clone(),
        };
    }

    // Check if the block annotations like @settings() are different.
    if !old_ast
        .inner_attrs
        .iter()
        .filter(annotations::is_significant)
        .zip_longest(new_ast.inner_attrs.iter().filter(annotations::is_significant))
        .all(|pair| {
            match pair {
                EitherOrBoth::Both(old, new) => {
                    // Compare annotations, ignoring source ranges.  Digests must
                    // have been computed before this.
                    let Annotation { name, properties, .. } = &old.inner;
                    let Annotation {
                        name: new_name,
                        properties: new_properties,
                        ..
                    } = &new.inner;

                    name.as_ref().map(|n| n.digest) == new_name.as_ref().map(|n| n.digest)
                        && properties
                            .as_ref()
                            .map(|props| props.iter().map(|p| p.digest).collect::<Vec<_>>())
                            == new_properties
                                .as_ref()
                                .map(|props| props.iter().map(|p| p.digest).collect::<Vec<_>>())
                }
                _ => false,
            }
        })
    {
        // If any of the annotations are different at the beginning of the
        // program, it's likely the settings, and we have to bust the cache and
        // re-execute the whole thing.
        return CacheResult::ReExecute {
            clear_scene: true,
            reapply_settings: true,
            program: new.ast.clone(),
        };
    }

    // Check if the changes were only to Non-code areas, like comments or whitespace.
    generate_changed_program(old_ast, new_ast, reapply_settings, old_vars)
}

/// Given a slice of top-level statements that were deleted (from the trailing
/// end of a program) and the variable bindings in effect at the end of the
/// last successful execution of the program they came from, return the set of
/// engine object ids those statements are safely known to have produced, or
/// `None` if any deleted statement isn't safe to treat as a pure delete.
///
/// A deleted statement is safe only if it's a single-name variable
/// declaration (the only kind KCL has) whose bound value is entirely made of
/// solids (or a homogeneous array of solids) - see [`kcl_value_solid_ids`] for
/// why sketches, planes, helices, imported geometry and GD&T annotations are
/// deliberately *not* included, even though the `delete()` builtin handles
/// all of those. Anything else (a bare expression statement, a
/// type/import/return statement, or a value we can't classify) makes the
/// whole deletion unsafe, so the caller falls back to a full cache bust
/// rather than risk leaving stale geometry in the scene.
///
/// Even once a candidate set of ids is found, it's rejected (and the caller
/// falls back to a full cache bust) if any of those ids are *also* reachable
/// from a binding that is NOT being deleted - see [`kcl_value_referenced_ids`].
/// A deleted statement's own binding being the only thing pointing at an id is
/// what makes it safe to delete; two different names can easily resolve to
/// the exact same engine entity (e.g. `b = a`, or two sketches started on the
/// same already-existing face), and deleting it out from under a surviving
/// binding would desync the scene from the code instead of fixing that.
fn ids_for_deleted_statements(deleted: &[BodyItem], old_vars: &IndexMap<String, KclValue>) -> Option<HashSet<Uuid>> {
    let mut deleted_names = std::collections::HashSet::new();
    let mut object_ids = HashSet::new();
    for item in deleted {
        let BodyItem::VariableDeclaration(decl) = item else {
            return None;
        };
        let name = &decl.declaration.id.name;
        let value = old_vars.get(name)?;
        let ids = kcl_value_solid_ids(value)?;
        object_ids.extend(ids);
        deleted_names.insert(name.as_str());
    }

    for (name, value) in old_vars {
        if deleted_names.contains(name.as_str()) {
            continue;
        }
        let mut surviving_ids = HashSet::new();
        kcl_value_referenced_ids(value, &mut surviving_ids);
        if !object_ids.is_disjoint(&surviving_ids) {
            return None;
        }
    }

    Some(object_ids)
}

/// Returns the engine object ids a [`KclValue`] refers to, if it's entirely
/// made of solids (or a homogeneous array of solids): `Some(ids)` where
/// `ids` is empty for a value that can't possibly reference scene geometry
/// (numbers, strings, tags, etc.), or `None` if the value isn't one we can
/// safely reason about (so the caller must be conservative and assume it
/// might), including every other geometry kind.
///
/// Solids are the only kind included here, even though [`kcl_value_referenced_ids`]
/// (and the `delete()` builtin) also handle sketches, planes, helices,
/// imported geometry and GD&T annotations. A `Sketch`'s or `Plane`'s id is
/// often not an engine entity exclusively "owned" by the statement that
/// bound it: sketching on a plane or an existing face reuses that surface's
/// id, including default planes (`XY`/`XZ`/`YZ`) that are shared globally,
/// not created per-statement. Deleting one of those ids out from under a
/// deleted statement's binding can delete a surface something else (another
/// sketch, a later `startSketchOn`, another default-plane use) still relies
/// on, since nothing about that sharing is visible as a reference to the
/// deleted statement's *name* - it only shows up as the same *id* appearing
/// elsewhere, which only the solid-only whitelist here plus the overlap
/// check in [`ids_for_deleted_statements`] can't fully rule out for those
/// kinds. A solid's own id isn't shared the same way: it changes as
/// operations are applied to that one solid, but isn't reused as some other,
/// unrelated object's id the way a plane or face can be.
fn kcl_value_solid_ids(value: &KclValue) -> Option<Vec<Uuid>> {
    match value {
        KclValue::Solid { value } => Some(vec![value.id]),
        KclValue::HomArray { value, .. } => {
            let mut ids = Vec::new();
            for element in value {
                ids.extend(kcl_value_solid_ids(element)?);
            }
            Some(ids)
        }
        KclValue::Uuid { .. }
        | KclValue::Bool { .. }
        | KclValue::Number { .. }
        | KclValue::String { .. }
        | KclValue::Enum { .. }
        | KclValue::TagIdentifier(_)
        | KclValue::TagDeclarator(_)
        | KclValue::KclNone { .. } => Some(Vec::new()),
        // Every other variant (including Sketch/Plane/Helix/ImportedGeometry/
        // GdtAnnotation - see above) is deliberately not whitelisted: new
        // variants, or any value we're not certain about, should fail closed
        // (fall back to a full cache bust) rather than silently miss an id
        // to delete, or delete one that's shared.
        KclValue::SketchVar { .. }
        | KclValue::SketchConstraint { .. }
        | KclValue::Tuple { .. }
        | KclValue::Object { .. }
        | KclValue::Face { .. }
        | KclValue::BoundedEdge { .. }
        | KclValue::Segment { .. }
        | KclValue::CameraView { .. }
        | KclValue::NamedView { .. }
        | KclValue::Function { .. }
        | KclValue::Module { .. }
        | KclValue::Type { .. }
        | KclValue::Sketch { .. }
        | KclValue::Plane { .. }
        | KclValue::Helix { .. }
        | KclValue::ImportedGeometry(_)
        | KclValue::GdtAnnotation { .. } => None,
    }
}

/// Best-effort, deliberately broad collection of every engine object id
/// reachable from a [`KclValue`], recursing into every container-like
/// variant (arrays, tuples, object fields). Used only as a safety net (see
/// [`ids_for_deleted_statements`]) to check whether an id a deleted statement
/// bound is *also* reachable from a surviving binding - so unlike
/// [`kcl_value_solid_ids`], erring on the side of collecting more ids here is
/// the safe direction, not the risky one: a false positive just means an
/// extra cache bust, while a false negative (missing a real alias) is the
/// kind of bug this function exists to catch.
fn kcl_value_referenced_ids(value: &KclValue, ids: &mut HashSet<Uuid>) {
    match value {
        KclValue::Solid { value } => {
            ids.insert(value.id);
        }
        KclValue::Sketch { value } => {
            ids.insert(value.id);
        }
        KclValue::Plane { value } => {
            ids.insert(value.id);
        }
        KclValue::Helix { value } => {
            ids.insert(value.value);
        }
        KclValue::GdtAnnotation { value } => {
            ids.insert(value.id);
        }
        KclValue::ImportedGeometry(value) => {
            ids.insert(value.id);
        }
        KclValue::Uuid { value, .. } => {
            ids.insert(*value);
        }
        KclValue::HomArray { value, .. } | KclValue::Tuple { value, .. } => {
            for element in value {
                kcl_value_referenced_ids(element, ids);
            }
        }
        KclValue::Object { value, .. } => {
            // Order doesn't matter: every value just gets unioned into `ids`.
            #[allow(clippy::iter_over_hash_type)]
            for element in value.values() {
                kcl_value_referenced_ids(element, ids);
            }
        }
        KclValue::Bool { .. }
        | KclValue::Number { .. }
        | KclValue::String { .. }
        | KclValue::Enum { .. }
        | KclValue::TagIdentifier(_)
        | KclValue::TagDeclarator(_)
        | KclValue::KclNone { .. }
        | KclValue::SketchVar { .. }
        | KclValue::SketchConstraint { .. }
        | KclValue::Face { .. }
        | KclValue::BoundedEdge { .. }
        | KclValue::Segment { .. }
        | KclValue::CameraView { .. }
        | KclValue::NamedView { .. }
        | KclValue::Function { .. }
        | KclValue::Module { .. }
        | KclValue::Type { .. } => {}
    }
}

/// Force-generate a new CacheResult, even if one shouldn't be made. The
/// way in which this gets invoked should always be through
/// [get_changed_program]. This is purely to contain the logic on
/// how we construct a new [CacheResult].
///
/// A CacheResult's program may be a *diff* of only the parts that need
/// to be executed (only in the case of "pure additions" at time of writing.).
/// This diff-based AST should not be persisted or used anywhere beyond the execution flow,
/// as it will be incomplete.
///
/// Digests *must* be computed before calling this.
fn generate_changed_program(
    old_ast: Node<Program>,
    mut new_ast: Node<Program>,
    reapply_settings: bool,
    old_vars: impl FnOnce() -> Option<IndexMap<String, KclValue>>,
) -> CacheResult {
    if !old_ast.body.iter().zip(new_ast.body.iter()).all(|(old, new)| {
        let old_node: WalkNode = old.into();
        let new_node: WalkNode = new.into();
        old_node.digest() == new_node.digest()
    }) {
        // If any of the nodes are different in the stretch of body that
        // overlaps, we have to bust cache and rebuild the scene. This
        // means a single insertion or deletion will result in a cache
        // bust.

        return CacheResult::ReExecute {
            clear_scene: true,
            reapply_settings,
            program: new_ast,
        };
    }

    // otherwise the overlapping section of the ast bodies matches.
    // Let's see what the rest of the slice looks like.

    match new_ast.body.len().cmp(&old_ast.body.len()) {
        std::cmp::Ordering::Less => {
            // The new AST is shorter than the old AST: statements were
            // removed from the end of the program (the overlapping prefix
            // matched above). If we know what each deleted statement bound
            // and it's all engine geometry (or nothing), we can delete just
            // those engine objects instead of busting the whole scene.
            let deleted = &old_ast.body[new_ast.body.len()..];
            match old_vars().and_then(|old_vars| ids_for_deleted_statements(deleted, &old_vars)) {
                Some(object_ids) => CacheResult::SurgicalDelete {
                    object_ids,
                    reapply_settings,
                },
                None => CacheResult::ReExecute {
                    clear_scene: true,
                    reapply_settings,
                    program: new_ast,
                },
            }
        }
        std::cmp::Ordering::Greater => {
            // the new AST is longer than the old AST, which means
            // statements were added to the new code we haven't previously
            // seen.
            //
            // Statements up until now are the same, which means this
            // is a "pure addition" of the remaining slice.

            new_ast.body = new_ast.body[old_ast.body.len()..].to_owned();

            CacheResult::ReExecute {
                clear_scene: false,
                reapply_settings,
                program: new_ast,
            }
        }
        std::cmp::Ordering::Equal => {
            // currently unreachable, but let's pretend like the code
            // above can do something meaningful here for when we get
            // to diffing and yanking chunks of the program apart.

            // We don't actually want to do anything here; so we're going
            // to not clear and do nothing. Is this wrong? I don't think
            // so but i think many things. This def needs to change
            // when the code above changes.

            CacheResult::NoAction(reapply_settings)
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::execution::ExecTestResults;
    use crate::execution::parse_execute;
    use crate::execution::parse_execute_with_project_dir;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_changed_program_same_code() {
        let new = r#"// Remove the end face for the extrusion.
firstSketch = startSketchOn(XY)
  |> startProfile(at = [-12, 12])
  |> line(end = [24, 0])
  |> line(end = [0, -24])
  |> line(end = [-24, 0])
  |> close()
  |> extrude(length = 6)

// Remove the end face for the extrusion.
shell(firstSketch, faces = [END], thickness = 0.25)"#;

        let ExecTestResults { program, exec_ctxt, .. } = parse_execute(new).await.unwrap();

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            || None,
        )
        .await;

        assert_eq!(result, CacheResult::NoAction(false));
        exec_ctxt.close().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_changed_program_same_code_changed_whitespace() {
        let old = r#" // Remove the end face for the extrusion.
firstSketch = startSketchOn(XY)
  |> startProfile(at = [-12, 12])
  |> line(end = [24, 0])
  |> line(end = [0, -24])
  |> line(end = [-24, 0])
  |> close()
  |> extrude(length = 6)

// Remove the end face for the extrusion.
shell(firstSketch, faces = [END], thickness = 0.25) "#;

        let new = r#"// Remove the end face for the extrusion.
firstSketch = startSketchOn(XY)
  |> startProfile(at = [-12, 12])
  |> line(end = [24, 0])
  |> line(end = [0, -24])
  |> line(end = [-24, 0])
  |> close()
  |> extrude(length = 6)

// Remove the end face for the extrusion.
shell(firstSketch, faces = [END], thickness = 0.25)"#;

        let ExecTestResults { program, exec_ctxt, .. } = parse_execute(old).await.unwrap();

        let program_new = crate::Program::parse_no_errs(new).unwrap();

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            CacheInformation::new(&program_new.ast, &exec_ctxt.settings),
            || None,
        )
        .await;

        assert_eq!(result, CacheResult::NoAction(false));
        exec_ctxt.close().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_changed_program_same_code_changed_code_comment_start_of_program() {
        let old = r#" // Removed the end face for the extrusion.
firstSketch = startSketchOn(XY)
  |> startProfile(at = [-12, 12])
  |> line(end = [24, 0])
  |> line(end = [0, -24])
  |> line(end = [-24, 0])
  |> close()
  |> extrude(length = 6)

// Remove the end face for the extrusion.
shell(firstSketch, faces = [END], thickness = 0.25) "#;

        let new = r#"// Remove the end face for the extrusion.
firstSketch = startSketchOn(XY)
  |> startProfile(at = [-12, 12])
  |> line(end = [24, 0])
  |> line(end = [0, -24])
  |> line(end = [-24, 0])
  |> close()
  |> extrude(length = 6)

// Remove the end face for the extrusion.
shell(firstSketch, faces = [END], thickness = 0.25)"#;

        let ExecTestResults { program, exec_ctxt, .. } = parse_execute(old).await.unwrap();

        let program_new = crate::Program::parse_no_errs(new).unwrap();

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            CacheInformation::new(&program_new.ast, &exec_ctxt.settings),
            || None,
        )
        .await;

        assert_eq!(result, CacheResult::NoAction(false));
        exec_ctxt.close().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_changed_program_same_code_changed_code_comments_attrs() {
        let old = r#"@foo(whatever = whatever)
@bar
// Removed the end face for the extrusion.
firstSketch = startSketchOn(XY)
  |> startProfile(at = [-12, 12])
  |> line(end = [24, 0])
  |> line(end = [0, -24])
  |> line(end = [-24, 0]) // my thing
  |> close()
  |> extrude(length = 6)

// Remove the end face for the extrusion.
shell(firstSketch, faces = [END], thickness = 0.25) "#;

        let new = r#"@foo(whatever = 42)
@baz
// Remove the end face for the extrusion.
firstSketch = startSketchOn(XY)
  |> startProfile(at = [-12, 12])
  |> line(end = [24, 0])
  |> line(end = [0, -24])
  |> line(end = [-24, 0])
  |> close()
  |> extrude(length = 6)

// Remove the end face for the extrusion.
shell(firstSketch, faces = [END], thickness = 0.25)"#;

        let ExecTestResults { program, exec_ctxt, .. } = parse_execute(old).await.unwrap();

        let program_new = crate::Program::parse_no_errs(new).unwrap();

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            CacheInformation::new(&program_new.ast, &exec_ctxt.settings),
            || None,
        )
        .await;

        assert_eq!(result, CacheResult::NoAction(false));
        exec_ctxt.close().await;
    }

    // Changing the grid settings with the exact same file should NOT bust the cache.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_changed_program_same_code_but_different_grid_setting() {
        let new = r#"// Remove the end face for the extrusion.
firstSketch = startSketchOn(XY)
  |> startProfile(at = [-12, 12])
  |> line(end = [24, 0])
  |> line(end = [0, -24])
  |> line(end = [-24, 0])
  |> close()
  |> extrude(length = 6)

// Remove the end face for the extrusion.
shell(firstSketch, faces = [END], thickness = 0.25)"#;

        let ExecTestResults {
            program, mut exec_ctxt, ..
        } = parse_execute(new).await.unwrap();

        // Change the settings.
        exec_ctxt.settings.show_grid = !exec_ctxt.settings.show_grid;

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &Default::default()),
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            || None,
        )
        .await;

        assert_eq!(result, CacheResult::NoAction(true));
        exec_ctxt.close().await;
    }

    // Changing the edge visibility settings with the exact same file should NOT bust the cache.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_changed_program_same_code_but_different_edge_visibility_setting() {
        let new = r#"// Remove the end face for the extrusion.
firstSketch = startSketchOn(XY)
  |> startProfile(at = [-12, 12])
  |> line(end = [24, 0])
  |> line(end = [0, -24])
  |> line(end = [-24, 0])
  |> close()
  |> extrude(length = 6)

// Remove the end face for the extrusion.
shell(firstSketch, faces = [END], thickness = 0.25)"#;

        let ExecTestResults {
            program, mut exec_ctxt, ..
        } = parse_execute(new).await.unwrap();

        // Change the settings.
        exec_ctxt.settings.highlight_edges = !exec_ctxt.settings.highlight_edges;

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &Default::default()),
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            || None,
        )
        .await;

        assert_eq!(result, CacheResult::NoAction(true));

        // Change the settings back.
        let old_settings = exec_ctxt.settings.clone();
        exec_ctxt.settings.highlight_edges = !exec_ctxt.settings.highlight_edges;

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &old_settings),
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            || None,
        )
        .await;

        assert_eq!(result, CacheResult::NoAction(true));

        // Change the settings back.
        let old_settings = exec_ctxt.settings.clone();
        exec_ctxt.settings.highlight_edges = !exec_ctxt.settings.highlight_edges;

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &old_settings),
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            || None,
        )
        .await;

        assert_eq!(result, CacheResult::NoAction(true));
        exec_ctxt.close().await;
    }

    // Changing only the KCL version must invalidate the cache and re-execute the whole program.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_changed_program_same_code_but_different_kcl_version_using_annotation() {
        for (old_version, new_version) in [
            ("2.0", "\"3.0-preview\""),
            ("2.0", "3.0"),
            ("3.0", "2.0"),
            ("\"3.0-preview\"", "3.0"),
            ("3.0", "\"3.0-preview\""),
        ] {
            let old_code = format!("@settings(kclVersion = {old_version})\nx = 1\ny = x + 1\n");
            let new_code = format!("@settings(kclVersion = {new_version})\nx = 1\ny = x + 1\n");
            let mut program = crate::Program::parse_no_errs(&old_code).unwrap();
            program.compute_digest();
            let mut new_program = crate::Program::parse_no_errs(&new_code).unwrap();
            new_program.compute_digest();
            let settings = ExecutorSettings::default();

            let result = get_changed_program(
                CacheInformation::new(&program.ast, &settings),
                CacheInformation::new(&new_program.ast, &settings),
                || None,
            )
            .await;

            assert_eq!(
                result,
                CacheResult::ReExecute {
                    clear_scene: true,
                    reapply_settings: true,
                    program: new_program.ast,
                },
                "{old_version} -> {new_version}"
            );
        }
    }

    // Changing the units settings using an annotation with the exact same file
    // should bust the cache.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_changed_program_same_code_but_different_unit_setting_using_annotation() {
        let old_code = r#"@settings(defaultLengthUnit = in)
startSketchOn(XY)
"#;
        let new_code = r#"@settings(defaultLengthUnit = mm)
startSketchOn(XY)
"#;

        let ExecTestResults { program, exec_ctxt, .. } = parse_execute(old_code).await.unwrap();

        let mut new_program = crate::Program::parse_no_errs(new_code).unwrap();
        new_program.compute_digest();

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            CacheInformation::new(&new_program.ast, &exec_ctxt.settings),
            || None,
        )
        .await;

        assert_eq!(
            result,
            CacheResult::ReExecute {
                clear_scene: true,
                reapply_settings: true,
                program: new_program.ast,
            }
        );
        exec_ctxt.close().await;
    }

    // Removing the units settings using an annotation, when it was non-default
    // units, with the exact same file should bust the cache.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_changed_program_same_code_but_removed_unit_setting_using_annotation() {
        let old_code = r#"@settings(defaultLengthUnit = in)
startSketchOn(XY)
"#;
        let new_code = r#"
startSketchOn(XY)
"#;

        let ExecTestResults { program, exec_ctxt, .. } = parse_execute(old_code).await.unwrap();

        let mut new_program = crate::Program::parse_no_errs(new_code).unwrap();
        new_program.compute_digest();

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            CacheInformation::new(&new_program.ast, &exec_ctxt.settings),
            || None,
        )
        .await;

        assert_eq!(
            result,
            CacheResult::ReExecute {
                clear_scene: true,
                reapply_settings: true,
                program: new_program.ast,
            }
        );
        exec_ctxt.close().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_multi_file_no_changes_does_not_reexecute() {
        let code = r#"import "toBeImported.kcl" as importedCube

importedCube

sketch001 = startSketchOn(XZ)
profile001 = startProfile(sketch001, at = [-134.53, -56.17])
  |> angledLine(angle = 0, length = 79.05, tag = $rectangleSegmentA001)
  |> angledLine(angle = segAng(rectangleSegmentA001) - 90, length = 76.28)
  |> angledLine(angle = segAng(rectangleSegmentA001), length = -segLen(rectangleSegmentA001), tag = $seg01)
  |> line(endAbsolute = [profileStartX(%), profileStartY(%)], tag = $seg02)
  |> close()
extrude001 = extrude(profile001, length = 100)
sketch003 = startSketchOn(extrude001, face = seg02)
sketch002 = startSketchOn(extrude001, face = seg01)
"#;

        let other_file = (
            std::path::PathBuf::from("toBeImported.kcl"),
            r#"sketch001 = startSketchOn(XZ)
profile001 = startProfile(sketch001, at = [281.54, 305.81])
  |> angledLine(angle = 0, length = 123.43, tag = $rectangleSegmentA001)
  |> angledLine(angle = segAng(rectangleSegmentA001) - 90, length = 85.99)
  |> angledLine(angle = segAng(rectangleSegmentA001), length = -segLen(rectangleSegmentA001))
  |> line(endAbsolute = [profileStartX(%), profileStartY(%)])
  |> close()
extrude(profile001, length = 100)"#
                .to_string(),
        );

        let tmp_dir = std::env::temp_dir();
        let tmp_dir = tmp_dir.join(uuid::Uuid::new_v4().to_string());

        // Create a temporary file for each of the other files.
        let tmp_file = tmp_dir.join(other_file.0);
        std::fs::create_dir_all(tmp_file.parent().unwrap()).unwrap();
        std::fs::write(tmp_file, other_file.1).unwrap();

        let ExecTestResults { program, exec_ctxt, .. } =
            parse_execute_with_project_dir(code, Some(crate::TypedPath(tmp_dir)))
                .await
                .unwrap();

        let mut new_program = crate::Program::parse_no_errs(code).unwrap();
        new_program.compute_digest();

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            CacheInformation::new(&new_program.ast, &exec_ctxt.settings),
            || None,
        )
        .await;

        let CacheResult::CheckImportsOnly { reapply_settings, .. } = result else {
            panic!("Expected CheckImportsOnly, got {result:?}");
        };

        assert_eq!(reapply_settings, false);
        exec_ctxt.close().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_cache_multi_file_only_other_file_changes_should_reexecute() {
        let code = r#"import "toBeImported.kcl" as importedCube

importedCube

sketch001 = startSketchOn(XZ)
profile001 = startProfile(sketch001, at = [-134.53, -56.17])
  |> angledLine(angle = 0, length = 79.05, tag = $rectangleSegmentA001)
  |> angledLine(angle = segAng(rectangleSegmentA001) - 90, length = 76.28)
  |> angledLine(angle = segAng(rectangleSegmentA001), length = -segLen(rectangleSegmentA001), tag = $seg01)
  |> line(endAbsolute = [profileStartX(%), profileStartY(%)], tag = $seg02)
  |> close()
extrude001 = extrude(profile001, length = 100)
sketch003 = startSketchOn(extrude001, face = seg02)
sketch002 = startSketchOn(extrude001, face = seg01)
"#;

        let other_file = (
            std::path::PathBuf::from("toBeImported.kcl"),
            r#"sketch001 = startSketchOn(XZ)
profile001 = startProfile(sketch001, at = [281.54, 305.81])
  |> angledLine(angle = 0, length = 123.43, tag = $rectangleSegmentA001)
  |> angledLine(angle = segAng(rectangleSegmentA001) - 90, length = 85.99)
  |> angledLine(angle = segAng(rectangleSegmentA001), length = -segLen(rectangleSegmentA001))
  |> line(endAbsolute = [profileStartX(%), profileStartY(%)])
  |> close()
extrude(profile001, length = 100)"#
                .to_string(),
        );

        let other_file2 = (
            std::path::PathBuf::from("toBeImported.kcl"),
            r#"sketch001 = startSketchOn(XZ)
profile001 = startProfile(sketch001, at = [281.54, 305.81])
  |> angledLine(angle = 0, length = 123.43, tag = $rectangleSegmentA001)
  |> angledLine(angle = segAng(rectangleSegmentA001) - 90, length = 85.99)
  |> angledLine(angle = segAng(rectangleSegmentA001), length = -segLen(rectangleSegmentA001))
  |> line(endAbsolute = [profileStartX(%), profileStartY(%)])
  |> close()
extrude(profile001, length = 100)
|> translate(z=100) 
"#
            .to_string(),
        );

        let tmp_dir = std::env::temp_dir();
        let tmp_dir = tmp_dir.join(uuid::Uuid::new_v4().to_string());

        // Create a temporary file for each of the other files.
        let tmp_file = tmp_dir.join(other_file.0);
        std::fs::create_dir_all(tmp_file.parent().unwrap()).unwrap();
        std::fs::write(&tmp_file, other_file.1).unwrap();

        let ExecTestResults { program, exec_ctxt, .. } =
            parse_execute_with_project_dir(code, Some(crate::TypedPath(tmp_dir)))
                .await
                .unwrap();

        // Change the other file.
        std::fs::write(tmp_file, other_file2.1).unwrap();

        let mut new_program = crate::Program::parse_no_errs(code).unwrap();
        new_program.compute_digest();

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            CacheInformation::new(&new_program.ast, &exec_ctxt.settings),
            || None,
        )
        .await;

        let CacheResult::CheckImportsOnly { reapply_settings, .. } = result else {
            panic!("Expected CheckImportsOnly, got {result:?}");
        };

        assert_eq!(reapply_settings, false);
        exec_ctxt.close().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_changed_program_added_outer_attribute() {
        let old_code = r#"import "tests/inputs/cube.step"
"#;
        let new_code = r#"@(coords = opengl)
import "tests/inputs/cube.step"
"#;

        let ExecTestResults { program, exec_ctxt, .. } = parse_execute(old_code).await.unwrap();

        let mut new_program = crate::Program::parse_no_errs(new_code).unwrap();
        new_program.compute_digest();

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            CacheInformation::new(&new_program.ast, &exec_ctxt.settings),
            || None,
        )
        .await;

        assert_eq!(
            result,
            CacheResult::ReExecute {
                clear_scene: true,
                reapply_settings: false,
                program: new_program.ast,
            }
        );
        exec_ctxt.close().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_changed_program_different_outer_attribute() {
        let old_code = r#"@(coords = vulkan)
import "tests/inputs/cube.step"
"#;
        let new_code = r#"@(coords = opengl)
import "tests/inputs/cube.step"
"#;

        let ExecTestResults { program, exec_ctxt, .. } = parse_execute(old_code).await.unwrap();

        let mut new_program = crate::Program::parse_no_errs(new_code).unwrap();
        new_program.compute_digest();

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            CacheInformation::new(&new_program.ast, &exec_ctxt.settings),
            || None,
        )
        .await;

        assert_eq!(
            result,
            CacheResult::ReExecute {
                clear_scene: true,
                reapply_settings: false,
                program: new_program.ast,
            }
        );
        exec_ctxt.close().await;
    }

    // A single statement that both sketches and extrudes, bound to one name
    // and referencing nothing else - so nothing else in a program using this
    // can alias its solid's id.
    const SOLID_PROGRAM: &str = r#"solid = startSketchOn(XZ)
  |> circle(center = [0, 0], radius = 4mm)
  |> extrude(length = 4mm)
"#;

    // A sketch-then-extrude program where the profile is kept bound after
    // the extrude - the realistic shape of code that keeps a sketch around
    // for a later `startSketchOn(face)`, etc.
    const EXTRUDE_FROM_NAMED_PROFILE_PROGRAM: &str = r#"@settings(kclVersion = 3.0)
profile = sketch(on = XZ) {
  bottom = line(start = [-4mm, -4mm], end = [4mm, -4mm])
  right = line(start = [4mm, -4mm], end = [4mm, 4mm])
  top = line(start = [4mm, 4mm], end = [-4mm, 4mm])
  left = line(start = [-4mm, 4mm], end = [-4mm, -4mm])
}
profileRegion = region(segments = [profile.bottom, profile.right, profile.top, profile.left])
"#;

    // Deleting a trailing `solid = ...` statement whose solid doesn't alias
    // any surviving binding should be recognized as a pure delete: no
    // re-execution, just remove the solid's engine object.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_changed_program_trailing_delete_of_unaliased_solid_is_surgical() {
        let old_code = SOLID_PROGRAM;
        let new_code = "";

        let ExecTestResults {
            program,
            exec_ctxt,
            exec_state,
            mem_env,
        } = parse_execute(old_code).await.unwrap();
        let old_vars = exec_state.mod_local.variables(mem_env).unwrap();
        let solid_id = match old_vars.get("solid").unwrap() {
            KclValue::Solid { value } => value.id,
            other => panic!("expected a solid, got {other:?}"),
        };

        let mut new_program = crate::Program::parse_no_errs(new_code).unwrap();
        new_program.compute_digest();

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            CacheInformation::new(&new_program.ast, &exec_ctxt.settings),
            move || Some(old_vars),
        )
        .await;

        assert_eq!(
            result,
            CacheResult::SurgicalDelete {
                object_ids: HashSet::from([solid_id]),
                reapply_settings: false,
            }
        );
        exec_ctxt.close().await;
    }

    // Regression test: `extrude()` reuses the extruded sketch/region's own id
    // as the resulting solid's id (see `kcl_lib::std::extrude::inner_extrude`,
    // `let id = sketch.id;`). So if the profile that was extruded is *also*
    // still bound to a surviving variable (a realistic pattern - keeping a
    // profile around to sketch on one of the resulting faces, etc.), deleting
    // just the trailing `solid = extrude(...)` statement would delete an
    // engine object the surviving `profileRegion` binding still points at.
    // That must NOT be treated as a safe surgical delete.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_changed_program_trailing_delete_of_solid_aliasing_surviving_profile_falls_back() {
        let old_code = format!("{EXTRUDE_FROM_NAMED_PROFILE_PROGRAM}solid = extrude(profileRegion, length = 4mm)\n");
        let new_code = EXTRUDE_FROM_NAMED_PROFILE_PROGRAM;

        let ExecTestResults {
            program,
            exec_ctxt,
            exec_state,
            mem_env,
        } = parse_execute(&old_code).await.unwrap();
        let old_vars = exec_state.mod_local.variables(mem_env).unwrap();
        // Confirm the premise: the solid really does alias the surviving
        // profileRegion's id, which is *why* this must fall back.
        let solid_id = match old_vars.get("solid").unwrap() {
            KclValue::Solid { value } => value.id,
            other => panic!("expected a solid, got {other:?}"),
        };
        let mut profile_region_ids = HashSet::new();
        kcl_value_referenced_ids(old_vars.get("profileRegion").unwrap(), &mut profile_region_ids);
        assert!(
            profile_region_ids.contains(&solid_id),
            "test premise failed: expected the solid's id to alias profileRegion's"
        );

        let mut new_program = crate::Program::parse_no_errs(new_code).unwrap();
        new_program.compute_digest();

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            CacheInformation::new(&new_program.ast, &exec_ctxt.settings),
            move || Some(old_vars),
        )
        .await;

        assert_eq!(
            result,
            CacheResult::ReExecute {
                clear_scene: true,
                reapply_settings: false,
                program: new_program.ast,
            }
        );
        exec_ctxt.close().await;
    }

    // Regression test: a deleted sketch/plane-valued statement must never be
    // treated as a safe surgical delete, even though `delete()` (the KCL
    // builtin) handles sketches and planes just fine. A `Sketch`'s or
    // `Plane`'s id is routinely the id of a shared surface (a default plane,
    // or another solid's face) rather than something exclusively created by
    // the statement that bound it - deleting it can corrupt a surface
    // unrelated code still relies on, with no reference to the deleted
    // statement's *name* anywhere nearby to warn us.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_changed_program_trailing_delete_of_plane_falls_back_to_clear_scene() {
        let old_code = r#"sketch001 = startSketchOn(XY)
"#;
        let new_code = "";

        let ExecTestResults {
            program,
            exec_ctxt,
            exec_state,
            mem_env,
        } = parse_execute(old_code).await.unwrap();
        let old_vars = exec_state.mod_local.variables(mem_env).unwrap();
        assert!(matches!(old_vars.get("sketch001").unwrap(), KclValue::Plane { .. }));

        let mut new_program = crate::Program::parse_no_errs(new_code).unwrap();
        new_program.compute_digest();

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            CacheInformation::new(&new_program.ast, &exec_ctxt.settings),
            move || Some(old_vars),
        )
        .await;

        assert_eq!(
            result,
            CacheResult::ReExecute {
                clear_scene: true,
                reapply_settings: false,
                program: new_program.ast,
            }
        );
        exec_ctxt.close().await;
    }

    // Deleting a trailing statement that only binds a plain value (nothing
    // that maps to engine geometry) is also a safe pure delete, just with no
    // engine objects to remove.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_changed_program_trailing_delete_of_plain_value_is_surgical_with_no_ids() {
        let old_code = format!("{SOLID_PROGRAM}x = 5\n");
        let new_code = SOLID_PROGRAM;

        let ExecTestResults {
            program,
            exec_ctxt,
            exec_state,
            mem_env,
        } = parse_execute(&old_code).await.unwrap();
        let old_vars = exec_state.mod_local.variables(mem_env).unwrap();

        let mut new_program = crate::Program::parse_no_errs(new_code).unwrap();
        new_program.compute_digest();

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            CacheInformation::new(&new_program.ast, &exec_ctxt.settings),
            move || Some(old_vars),
        )
        .await;

        assert_eq!(
            result,
            CacheResult::SurgicalDelete {
                object_ids: HashSet::new(),
                reapply_settings: false,
            }
        );
        exec_ctxt.close().await;
    }

    // Deleting a trailing statement that doesn't bind a name at all (a bare
    // expression statement) isn't something we can safely classify, so it
    // must fall back to a full cache bust, same as before surgical deletes
    // were supported.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_get_changed_program_trailing_delete_of_bare_expression_falls_back_to_clear_scene() {
        let old_code = format!("{SOLID_PROGRAM}solid\n");
        let new_code = SOLID_PROGRAM;

        let ExecTestResults {
            program,
            exec_ctxt,
            exec_state,
            mem_env,
        } = parse_execute(&old_code).await.unwrap();
        let old_vars = exec_state.mod_local.variables(mem_env).unwrap();

        let mut new_program = crate::Program::parse_no_errs(new_code).unwrap();
        new_program.compute_digest();

        let result = get_changed_program(
            CacheInformation::new(&program.ast, &exec_ctxt.settings),
            CacheInformation::new(&new_program.ast, &exec_ctxt.settings),
            move || Some(old_vars),
        )
        .await;

        assert_eq!(
            result,
            CacheResult::ReExecute {
                clear_scene: true,
                reapply_settings: false,
                program: new_program.ast,
            }
        );
        exec_ctxt.close().await;
    }
}
