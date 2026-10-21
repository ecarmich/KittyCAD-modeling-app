# Content-addressed command cache: batch-at-flush architecture

Status: design, not yet implemented. Recreated from scratch (the session that
tracked this before stored its notes outside the repo and that machine is
gone). Picks up from the state of branch `surgical-cache-deletes` at commit
`872aeb4` ("Spike: content-addressed id reuse for offsetPlane's engine
commands"). Read `rust/kcl-lib/src/execution/command_cache.rs`'s module docs
and `rust/kcl-lib/src/execution/cache.rs` before this; this doc assumes both.

## 1. Decision: windowed batch, not whole-file batch

**Adopt the interpreter-id/real-id split now.** It is strictly good: it
removes the retroactive-rewrite hazard of the earlier (already-rejected)
"redraw" design, and the spike already uses an informal version of it for
`MakePlane`.

**Do not build a single flush-at-end-of-file batch.** Instead, keep batching
scoped to the *existing* flush windows the engine wiring already has, and make
the matcher operate on a whole window at once instead of one command at a
time. This section explains why, because "batch" was underspecified in the
task that kicked this off and the two readings have very different risk
profiles.

### What a "window" already is

The engine batch (`EngineManager`/`ExecState::batch_modeling_cmd`,
`rust/kcl-lib/src/execution/modeling.rs`) already queues commands and only
sends on an explicit flush. Flushes are not rare or only-at-EOF; they are
forced constantly, every time a stdlib function needs to *read* something
back from the engine to keep interpreting:

- `flush_batch_for_solids` / `flush_batch_for_face_parent_solids` (fillet and
  chamfer ordering) — called from `shell.rs`, `clone.rs`, `transform.rs`,
  `csg.rs` (booleans), `mirror.rs`, `patterns.rs`, `surfaces.rs`, `planes.rs`.
- Direct, unbatched `async_modeling_cmd` calls that block on a real response
  because the interpreter needs the engine's answer *right there* to keep
  going: `extrude.rs` (`Solid3dGetExtrusionFaceInfo`, `EntityGetAllChildUuids`
  — face ids get bound to tags the KCL program may reference next statement),
  `edge.rs` (`Solid3dGetAllEdgeFaces`, `Solid3dGetAdjacencyInfo`,
  `Solid3dGetOppositeEdge`, `Solid3dGetNextAdjacentEdge`,
  `Solid3dGetPrevAdjacentEdge`, `Solid3dGetCommonEdge`, `QueryEntityType`),
  `csg.rs` (`BooleanUnion`/`BooleanIntersection`/`BooleanSubtract`/
  `BooleanImprint` — the resulting solid's real id comes back *in* the
  response, it is not chosen by the caller), `patterns.rs`
  (`EntityLinearPatternTransform`, `EntityCircularPattern`), `mirror.rs`
  (`EntityMirror`/`EntityMirrorAcross`/`EntityMirrorAcrossEdge`), `clone.rs`
  (`EntityGetAllChildUuids`), `sketch.rs` (`CreateRegion`/
  `CreateRegionFromQueryPoint`), `operation.rs` (`BoundingBox`), `ids.rs`
  (`Solid3dGetFaceUuid`/`Solid3dGetEdgeUuid`/`ClosestEdge`).

That is not a short list of edge cases — it is most of the surface area that
makes KCL's geometry model useful (every boolean, every fillet/chamfer,
every pattern, every mirror, every extrude that anything downstream
references by face/edge). A true whole-file batch would have to either stop
needing these responses mid-file (not possible without changing what KCL
*is*) or fabricate them from a cache instead of asking the engine (response
caching — see §4). Response caching is a real, separable project with its
own correctness assumption that is strictly stronger than anything the
existing mechanism needs; coupling the id-translation work to it is what the
task background flagged as the risk, and investigation confirms it: these
two should not ship as one change.

### What windowing buys instead

The documented limitation ("an insertion right before an unchanged tail
forces a needless recreate") is a property of *streaming* matching, not of
small windows. If a window's full content-id sequence is known before any
hit/miss decision is committed for that window, the limitation disappears
*within* that window, with no change to `command_cache.rs`'s matching
algorithm at all — see §3. Since windows are already bounded by
response-dependent ops rather than by individual statements, most real edits
(insert a statement, delete a statement, reorder two independent chains) keep
their surrounding unchanged content inside one window, so this captures the
common case the worked example in `command_cache.rs` is about, without
building response caching.

**Residual limitation, explicitly accepted:** an edit whose insertion and its
affected unchanged tail straddle two different flush windows (e.g. a new
statement right before a `fillet()` call whose *own* edge-cut ids are what
gets recreated) still hits the old streaming-only behavior. This is strictly
smaller than today's status quo and does not need to be fixed to ship this.

### A framing point worth stating plainly, because it is easy to lose

`cache.rs`'s `generate_changed_program` is unchanged by any of this and still
decides *whether a full re-execution happens at all*. Any single-statement
edit that is not a pure trailing append/delete still produces
`CacheResult::ReExecute { clear_scene: true, .. }` today, and will continue
to after this work — nothing proposed here touches that decision. The
command cache's job is strictly: *given that `cache.rs` already decided a
full re-execution must happen, reduce how many commands that re-execution
actually sends to the engine.* It does not avoid re-execution, and it does
not (by itself) change when `clear_scene` fires. Framing it as "batching the
file" invites the (wrong) expectation that this makes edits elsewhere in a
long file cheap to re-run; it only makes them cheap to re-*send*. The
CPU/WASM cost of re-running the KCL interpreter itself is untouched.

## 2. The interpreter-id / real-id split

Today, `ExecState::next_uuid()` (`state.rs:998`, backed by
`IdGenerator::next_uuid`) mints the one and only id space: it is called in
program order, deterministically for a given program, and its output is sent
*directly* as the outgoing command's id or an id-reference field. That is
why an insertion earlier in the file "shifts" every later id — it is the same
counter both logical layers currently share.

The split: keep calling `next_uuid()` exactly as today, in the same places,
in the same order, for the same purpose — call this the **interpreter id**.
It is bound into `KclValue::Solid`/`Sketch`/`Plane`/etc. exactly as today and
*never rewritten*, which is what avoids the redraw design's
`Stack::overwrite` hazard: nothing a KCL variable already closed over ever
changes under it.

Introduce a second id space, the **real id**, decided only at the moment a
command is actually serialized onto the wire (or, on a hit, reused from the
previous round's log — exactly `Cache::id_for_content`'s existing contract).
A per-round bidirectional table (generalizing `CommandCacheRound`'s existing
`id_map`) maps `interpreter_id <-> (content_id, real_id)`. Two ways an entry
gets into it:

- **Decided before send** (today's spike path, `id_for_content`): for a
  command whose own id *is* the identity of the entity it creates (see §3.1
  on `OwnsEntity`), content-hash it, consult the old round's log, get back
  `(real_id, is_hit)`. This is what `MakePlane` does today.
- **Recorded after a response** (new, needed for every response-dependent
  command in the §1 list): the interpreter still mints an interpreter id via
  `next_uuid()` for whatever new entity a response just revealed (a face,
  an edge, a boolean's result solid), exactly as it effectively does today by
  storing the response's id directly — the difference is that id now goes
  into the table as a `real_id` paired with a *freshly minted* interpreter id
  standing in for it internally, rather than being used as both. Any later
  command that references that entity references it by interpreter id, same
  as everything else.

This table, not raw `Uuid`s, is what every `*_id`/`*_ids` field in a queued
`ModelingCmd` holds while it sits in a window. Translation to real ids
happens once, at flush time (§3.3).

## 3. Per-command-type reference declaration

### 3.1 What actually needs declaring, and the real scope

Checked against the installed `kittycad-modeling-cmds = 0.2.250` crate
(`define_modeling_cmd_enum!` in `def_enum.rs`, which generates `ModelingCmd`'s
163 variants): **106 of the 163 variants** declare at least one field typed
`Uuid`, `Option<Uuid>`, `Vec<Uuid>`, `ModelingCmdId`, or `Option<ModelingCmdId>`.
That is the real size of the "declare once per command type" surface — more
than double the ~20-40 guessed before looking, and a reminder that the number
needs re-confirming against the actual crate version in use whenever this is
implemented, not assumed from memory.

Representative, individually verified entries (full list needs a dedicated
audit pass, see §3.4):

| Variant | id-reference fields | notes |
|---|---|---|
| `MakePlane` | *(none)* | its own cmd_id **is** the plane's real id |
| `PlaneSetColor` | `plane_id: Uuid` | references, doesn't own |
| `ExtendPath` / `MovePathPen` | `path: ModelingCmdId` | uses the dedicated id-ref newtype |
| `Extrude` / `ExtrudeToReference` | `target: Option<ModelingCmdId>` | |
| `Sweep` | `target: ModelingCmdId`, `trajectory: ModelingCmdId` | two references |
| `RevolveAboutEdge` | `target: ModelingCmdId`, `edge_id: Option<Uuid>` | mixed typing in one variant |
| `Solid3dGetExtrusionFaceInfo` | `object_id: Uuid`, `edge_id: Option<Uuid>` | query, not a creation |
| `EntityLinearPatternTransform`, `EntityMirror`, `EntityCircularPattern` | `entity_id`/`ids: Vec<Uuid>` | response carries the *new* ids |
| `Solid3dFilletEdge` | `object_id`, `edge_id: Option<Uuid>`, `edge_ids: Vec<Uuid>`, `extra_face_ids: Vec<Uuid>` | several reference fields on one command |

Two typing conventions coexist (`ModelingCmdId` newtype vs. plain `Uuid`) and
neither is reliable alone: most reference fields are plain `Uuid`, which
cannot be distinguished from a non-reference `Uuid` by type (e.g. a literal
color or session token would look identical to the compiler). So this cannot
be derived mechanically from field types — it needs a human-reviewed
declaration, once, not per call site. That one-time cost is the actual "fewer
places to change" win the task description was pointing at (vs. 146 existing
`next_uuid()` call sites across `src/`, counted directly in this repo).

### 3.2 The declaration itself

```rust
/// Declared once per `ModelingCmd` variant (see the per-variant audit in
/// `command_cache::id_refs`): every `Uuid`-shaped field that is a reference
/// to another entity's interpreter id, as opposed to a value that merely
/// happens to be shaped like a `Uuid`.
trait IdRefs {
    /// Call `visit` once for every id-reference field this command variant
    /// carries. A no-op default covers the majority of variants that
    /// reference nothing.
    fn for_each_id_ref_mut(&mut self, visit: &mut dyn FnMut(&mut Uuid)) {}
}
```

Implemented as a single `match` over `ModelingCmd` in
`command_cache.rs` (the trait is local, so `impl IdRefs for ModelingCmd` is a
normal inherent-looking impl despite `ModelingCmd` being a foreign type —
no newtype needed). One `match` arm per one of the 106 variants that has at
least one reference field; everything else falls through to the default.
This is the one place that needs updating when the engine adds a command with
a new reference field — not 106 call sites, and not the 146 `next_uuid()`
call sites either (those are unaffected; they keep minting interpreter ids
exactly as today).

Separately, declare which variants' **own cmd_id is the identity of a
persistent entity** (`MakePlane` today; by inspection, also e.g. `Extrude`'s
`ModelingCmdId` for the created solid, `Solid3dFilletEdge`'s own id for the
replacement face) versus which are disposable correlation tokens (every pure
query: `Solid3dGetExtrusionFaceInfo`, `QueryEntityType`, `BoundingBox`, etc. —
their *response* carries the ids that matter, their own cmd_id is thrown
away). Only the first kind goes through `Cache::id_for_content`'s
before-send decision (§2); the second kind always sends as a correlation id
(no caching needed on that command's own identity — the response-dependent
flush already forces it to go out — but its reference *fields* still need
translation, per §3.3). This mirrors `ModelingCmdId`'s own doc comment
("all commands have unique IDs... these should be randomly generated") not
distinguishing the two uses at the type level, so this also needs the
same kind of one-time, human-reviewed table, likely co-located with the
`IdRefs` match.

### 3.3 Flush-time batch-diff algorithm

Per window (§1):

1. **Queue, unchanged.** Every `batch_modeling_cmd` call mints its
   interpreter id via `next_uuid()` exactly as today and pushes the command,
   addressed entirely in interpreter ids, into the window's queue. No call
   site changes.
2. **Content-id pass, in program order, as commands are queued (not
   deferred to flush).** For each queued command, using `IdRefs` to find its
   reference fields, look up each referenced interpreter id's already-decided
   `content_id` (resolved either from an earlier window via the persisted
   log, or from earlier in *this* window — always available by construction,
   since a command can only reference something that was created before it)
   and hash the command's own fields with those substituted in, exactly as
   `content_id_of`/`id_for_content` already do. Record
   `interpreter_id -> content_id` in the round table. This is the same work
   the spike already does per-call-site for `MakePlane`/`PlaneSetColor`; the
   only change is doing it uniformly via `IdRefs` instead of by hand at two
   call sites.
3. **Match pass, once per window, over the whole queued content-id
   sequence.** This is `command_cache::CommandLog::match_next`, called
   repeatedly — *unchanged* — but now fed the window's full sequence after
   it is completely known, rather than interleaved with command production.
   This is the entire fix for the documented limitation: by the time
   matching starts, an inserted item's displaced successor is already in the
   sequence being matched, so the matcher's own "look for this content id
   later in the old log" branch (`command_cache.rs:167`) finds the
   successor's unaffected old entry directly, instead of the streaming
   version's forced one-step-late recreate. No change to
   `command_cache.rs` itself — only to when its caller invokes it.
4. **Translate and send.** For each `Hit`, take the old log's stored real
   id, record it against the interpreter id, skip sending. For each `Miss`,
   mint a fresh real id (a separate counter from interpreter ids — it does
   not need to be deterministic the way interpreter ids do, since nothing
   KCL-visible is keyed on it), rewrite the command's own id and every
   `IdRefs`-declared field from interpreter id to real id (all resolvable by
   now, since every reference predates this point by construction), and
   send. For each `superseded` index, emit a delete for that old entry's real
   id, generalizing the existing surgical-delete path
   (`cache.rs::ids_for_deleted_statements`) from "whole trailing statements"
   to "whole superseded log entries."
5. **Finalize.** Same as today's `finish_command_cache_round`, after the
   window — not the whole file — so the next window within the same run
   already benefits, and the persisted log reflects every window's decisions
   by end of execution.

### 3.4 Before implementing

- Re-derive the 106-variant list mechanically (don't hand-maintain it from
  this doc) against whatever `kittycad-modeling-cmds` version is pinned at
  implementation time, and confirm each reference field's semantics against
  the engine's own command docs, not just its field name — a few of the
  `Uuid` fields found (e.g. `CreateRegion`'s `segment`/`intersection_segment`)
  read as transient/computed rather than referencing a cacheable persistent
  entity, and need a real answer, not a guess, before being included.
- Confirm the `OwnsEntity` vs. correlation-token classification per variant
  against engine behavior, not naming convention alone.

## 4. Response caching (explicitly out of scope for this change)

Flagged by the task background as coupled to reordering-robustness; it is
coupled, but it should not be built alongside the id-translation work,
because it needs a materially stronger assumption than anything else here.

Everything in §§2-3 only ever decides *whether to resend a command whose
result, if sent, provably doesn't change anything new* — on a hit, the old
real id is still attached to the same persistent engine object; nothing about
that object was touched, so of course querying it again would return what it
returned before. That's not an assumption about engine determinism, it's a
fact about not having touched something.

General response caching is different in kind: it would need to assume that
sending the *same input content* to the engine again produces the *same
response content* — including things the engine itself chooses (exact face
ids from an extrude's wall count, the particular id a boolean result gets),
not just "nothing changed." That's a stronger claim than `command_cache.rs`'s
own module doc currently needs to defend, and it degrades badly: a wrong
response is fabricated geometry data silently propagated into further KCL
evaluation, not a merely-suboptimal extra round trip.

**If this is ever built, scope it to the provably-safe case only:** a query
command (`Solid3dGetExtrusionFaceInfo`, `Solid3dGetAllEdgeFaces`, etc.)
against an object whose *entire creation chain this round* was a hit (i.e.
nothing along the way actually sent a command) — there, caching the old
round's response is safe for the same "nothing was touched" reason as id
reuse, not because the engine is assumed deterministic. Do not extend this to
caching the output of a command that creates or mutates content from changed
or even unchanged-but-freshly-resent inputs.

## 5. The output-boundary translation problem

Flagged by the task background as "completely undesigned." Narrower than
feared once traced:

- `kcl_api::Object` (`scene_objects`, `source_range_to_object`) does **not**
  carry a real engine id at all — `Object.id: ObjectId` is a local `usize`
  bookkeeping id and `Object.artifact_id: ArtifactId` is a third, separate id
  space (`rust/kcl-api/src/front.rs:27`). Nothing to translate there.
- `ArtifactGraph` is already built as a single end-of-execution batch pass
  over the full `ArtifactCommand` log plus flattened engine responses
  (`state.rs::build_artifact_graph` ~1428, `artifact.rs::build_artifact_graph`
  ~454) — not incrementally during interpretation. That is one well-defined
  hook: feed that pass the interpreter-id-addressed `ArtifactCommand` log
  (unchanged) plus the round table, and have it resolve to real ids (or,
  more likely, leave the graph itself keyed on interpreter ids internally
  and only translate at serialization to the frontend, if anything downstream
  of it needs a real id at all — needs checking against what the frontend
  actually consumes from artifact graph nodes).
- `ExecOutcome.variables: IndexMap<String, KclValueView>` is the real
  surface: every `Solid`/`Sketch`/`Plane`/`Helix`/`ImportedGeometry`/
  `GdtAnnotation`'s `.id` embedded in program memory, exposed directly to the
  frontend. This needs a recursive rewrite pass over every `KclValueView`
  before `into_exec_outcome` (`cache.rs:232`) returns — and
  `cache.rs::kcl_value_referenced_ids` (`cache.rs:634`, already written for
  the surgical-delete alias check) is *already* exactly this traversal
  shape, just collecting instead of mutating. Reuse it (or a `KclValueView`-
  typed sibling of it) rather than writing a second one.

Net: the output-boundary problem is two call sites (`build_artifact_graph`'s
inputs, and a rewrite pass before `into_exec_outcome`), not a sprawling
audit — contingent on confirming `ArtifactGraph`'s own node payloads don't
embed raw engine `Uuid`s anywhere outside what `build_artifact_graph`
already controls, which needs a direct read of `artifact.rs`'s graph node
types before implementation, not assumed from this doc.

## 6. Suggested implementation order

1. `IdRefs` trait + the per-variant match (§3.2), tested standalone against
   synthetic `ModelingCmd` values — no wiring into execution yet, same spirit
   as `command_cache.rs`'s own introduction.
2. Generalize `CommandCacheRound`'s before-send path (today: `MakePlane` only)
   to use `IdRefs` instead of the two hand-written call sites in
   `std/planes.rs`, as a direct refactor of the existing spike — proves the
   declaration mechanism against the one case already working before
   widening scope.
3. Window-batching of the content-id + match passes (§3.3) for a *second*
   real call site that has more than one command per statement and shares a
   flush window with its neighbors (a good candidate: `offsetPlane` plus a
   following `startSketchOn`/`extrude`, which is exactly the shape of the
   spike's own "insertion before an unchanged tail" test) — confirm the
   limitation is actually gone within a window before generalizing further.
4. Output-boundary rewrite (§5) only once something upstream actually
   produces interpreter-id-addressed values that need it — no point building
   the translation pass before there's anything to translate.
5. Response caching (§4): separate follow-up, not a prerequisite for 1-4.

Each step should land as its own reviewable commit with its own tests, same
as the three commits already on this branch — this doc is deliberately not
a license to jump straight to a big-bang rewrite of `std/`.
