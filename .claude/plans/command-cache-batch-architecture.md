# Content-addressed command cache: batch-at-flush architecture

Status: design, not yet implemented. Recreated from scratch (the session that
tracked this before stored its notes outside the repo and that machine is
gone). Picks up from the state of branch `surgical-cache-deletes` at commit
`872aeb4` ("Spike: content-addressed id reuse for offsetPlane's engine
commands"). Read `rust/kcl-lib/src/execution/command_cache.rs`'s module docs
and `rust/kcl-lib/src/execution/cache.rs` before this; this doc assumes both.

**Revision note:** the first version of this doc claimed that re-invoking
`command_cache.rs`'s existing streaming matcher once per window, instead of
once per command, would by itself fix the documented "insertion before
unchanged tail" limitation, "with no change to the matcher itself." That
claim is wrong — traced by hand, replaying the same cursor-based algorithm
over an already-known sequence produces the identical result as streaming
it, because the algorithm never looks ahead into the new sequence regardless
of whether the caller already has it. A real-world duplicate-content case
(ten identical objects, one of them later mutated) also showed a sharper
problem the first version didn't address at all: content-addressing
independent commands can silently reattach a *stateful* mutation's effects
to the wrong one of several content-identical siblings. §3 below replaces
the flawed matching design with a corrected one. §1, §4 and §5 are
unaffected and still hold.

## 1. Decision: windowed sends, not whole-file batching

**Adopt the interpreter-id/real-id split now.** It is strictly good: it
removes the retroactive-rewrite hazard of the earlier (already-rejected)
"redraw" design, and the spike already uses an informal version of it for
`MakePlane`.

**Do not build a single flush-at-end-of-file batch for *sending*.** Keep the
decision of *when a command must actually go out* scoped to the engine
wiring's existing flush windows. The *matching* algorithm itself (§3) does
not need this restriction and should operate over the whole run's
accumulated content, not per-window — windowing only matters for the
send/flush timing question in this section, not for deciding hits and
misses.

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
references by face/edge). A true whole-file batch *for sending* would have
to either stop needing these responses mid-file (not possible without
changing what KCL *is*) or fabricate them from a cache instead of asking the
engine (response caching — see §4). Response caching is a real, separable
project with its own correctness assumption that is strictly stronger than
anything the existing mechanism needs; coupling the id-translation work to
it is what the task background flagged as the risk, and investigation
confirms it: these two should not ship as one change.

**What stays true regardless of §3's correction:** since matching (deciding
hit/miss/supersede) is now index-based rather than position-based (§3), it
does not actually need a window at all to be correct — it can and should run
against the full previous round's log no matter how many flush windows this
round ends up having. Windowing in this section is purely about *forced
send timing* for response-dependent commands, not about matching
correctness.

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
  command whose own id *is* the identity of the entity it creates (see §3.2
  on classifying commands), content-hash it, consult the old round's log,
  get back `(real_id, is_hit)`. This is what `MakePlane` does today.
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
`ModelingCmd` holds while it sits in the send queue. Translation to real ids
happens once, at flush time (§3.4).

## 3. Content identity and matching, corrected

### 3.1 The duplicate-content-with-divergent-mutation problem

Worked example (this is the scenario that broke §3's first draft): a script
creates ten identical planes (`MakePlane` with identical params, so all ten
hash to the same content id `C`), then sets a custom color on plane #6 only
(`PlaneSetColor`, content id `hash(C, color)`, since today's hashing scheme
keys a mutation's content on its *target's content id*, which is degenerate
— shared by all ten siblings — not on the target's specific identity).

Round 1: ten real ids `R1..R10` get minted in creation order; `R6` is the one
that actually receives the color command in the engine.

Round 2: an unrelated edit elsewhere disturbs the correspondence between
"this round's Nth occurrence of `C`" and "last round's Nth occurrence of
`C`" — this does not even require an insertion; swapping the statement order
of plane #6 and plane #9, with nothing else touched, is enough. Position- or
FIFO-based disambiguation among the ten now pairs them up differently than
before. Suppose this round's "plane #5" ends up matched (by position) to old
`R6`.

This is not merely wasted work. `R6` still physically carries the custom
color in the real engine session — matching only ever decides whether to
*send* a command, it never decides to *reset* state on an object it chooses
to reuse. So this round's "plane #5" (which the current script never colors)
comes back visibly colored, and nothing in a content-addressed-by-
independent-command scheme notices, because `PlaneSetColor`'s own content
hash never encoded *which* real object it actually landed on last time —
only the degenerate content id of "a plane shaped like `C`," shared by all
ten.

Meanwhile the *target* of `PlaneSetColor` resolves correctly this round
(whichever KCL value is logically "plane #6" this round does get the color
command re-sent against whatever real id it resolves to, since that part is
driven by the interpreter's own variable binding, not by the cache), the
*reused, uncolored* sibling that got silently paired with the actually-
colored old object is the real bug: stale, divergent state leaking onto the
wrong logical entity, invisible to the matcher.

### 3.2 Root cause: two separate problems, one fix each

**(a) Position-based matching can misalign even for globally unique
content.** This is the originally-documented streaming limitation
(`command_cache.rs`'s own module docs), and it does not actually need
duplicate content to manifest — `[a, b]` becoming `[a, x, b]` already breaks
it, with `a`, `x`, `b` all distinct. The existing algorithm commits a
decision per new item based only on cursor position plus an eager index
lookup; it never looks ahead into the new sequence, so feeding it an
already-fully-known sequence (as the first draft of this doc proposed)
changes nothing — confirmed by hand-tracing `match_next` over both the
streamed and the "pre-known" versions of `[a, x, b]` against old `[a, b]`:
identical result, `b` needlessly superseded and recreated either way.

**Fix:** match by index lookup only, with no cursor and no "everything
between cursor and the found position is superseded." Decide supersession
by *exclusion*, once, after every new content id has been looked up: an old
entry is superseded if and only if nothing in the complete new sequence ever
claimed it. Retracing `[a, b]` → `[a, x, b]`: `a` hits `Ra`, `x` misses
(genuinely new, correctly), `b` hits `Rb` — nothing is superseded. This is
simpler than the current algorithm, not more complex, and it also makes
pure reordering free (`[a, b, c]` → `[c, a, b]` reuses all three), which the
cursor-based version cannot do.

**(b) Content-addressing independent commands cannot disambiguate siblings
that only diverge via later mutation.** This is the new problem in §3.1.
Looking up `PlaneSetColor`'s content id by itself can correctly find *that
one* old `PlaneSetColor` entry (its content, `hash(C, color)`, really is
unique among the window's commands if only one plane was ever colored) — but
by the time we're deciding *which of the ten MakePlane entities this round's
color-setting statement is actually touching*, the creation-level lookup
alone is still degenerate and must fall back to position, which is exactly
what's unsafe.

**Fix:** content identity must be per-*entity* and cumulative, not
per-independent-command — a rolling hash chain, not unlike a Merkle chain or
event-sourcing aggregate hash:

- At creation: `chain(entity) = content_id_from_bytes(creation_command_bytes)`
  — same as today's `id_for_content` input.
- For every later command that **mutates** `entity` in place (rather than
  creating something new) — `PlaneSetColor`, `ObjectSetName`,
  `EntitySetOpacity`, and (see the caveat below) partially,
  `Solid3dFilletEdge` — roll forward:
  `chain(entity) := content_id_from_bytes(chain(entity) || this_mutation's own
  fields, with any *other* entity it references substituted by *their*
  current chain value)`.

Two entities that start identical (same creation content) but have since
received different mutations now have different chain values, full stop —
no position reasoning needed to tell them apart once they've diverged.

**Matching procedure, most-specific first — implemented.** `EntityChain`,
`ChainLog`, `ChainMatch`, and `match_round` all exist now in
`command_cache.rs`, standalone and tested, same scoping as everything else
in this module (nothing calls into them yet — see the caveat below on why
not). Within the set of entities being matched, `match_round` resolves
entities with a longer (more mutated, hence more likely globally unique)
chain before entities whose chain is still at the bare creation level. For
each, `ChainLog::match_chain` looks up its current chain value — falling
back to shallower levels if the deepest doesn't match — against the old
log's index; on a hit, the whole old entity is claimed (every level of its
history, not just the matched one), so it can never be handed to a
different new entity afterward. Once every distinguishable entity is
resolved this way, whatever's left on both sides is, by construction,
mutually indistinguishable — pairing the remainder in any fixed order is
safe, because nothing in either round's history ever told them apart.

Verified against the worked example directly: the
`duplicate_content_with_a_divergent_mutation_disambiguates_by_chain_not_position`
test builds ten old entities (one with a `["plane", "red"]` chain, nine with
bare `["plane"]`), moves the colored one to a different position in the new
round's sequence (nowhere near where it was), and asserts `match_round`
still resolves it to the correct old entity specifically — never one of the
nine uncolored ones — while the nine genuinely-interchangeable siblings
pair up with whatever's left, and every one of the ten claims exactly one
old entity (no double-claims, nothing orphaned).

This also recovers the mechanism's original purpose for the case where a
mutation's *value* changes rather than its presence: the
`a_changed_mutation_value_falls_back_to_reusing_the_base_object` test
confirms that a plane colored red last round and blue this round produces
`ChainMatch::PartialHit { matched_depth: 0 }` — the deepest level misses (no
old entity has `hash(C, blue)`), so matching falls back to the shallower,
creation-only level (`C`, which still matches), reusing the plane while
leaving only the changed `PlaneSetColor` needing to be resent, not a full
recreate.

**Discovered while implementing, not anticipated when this section was
first written:** generalizing past the two hand-written call sites in
`std/planes.rs` is *not* safe yet, and `match_round` is deliberately left
unwired rather than papering over this. The chain-based fix requires every
entity's *complete* chain (including mutations that come *later* in program
order, like `PlaneSetColor` after `MakePlane`) to be known before any real
id is decided — otherwise the `MakePlane` step still has to decide eagerly,
exactly as `std/planes.rs` does today, and the bug reproduces identically
regardless of how good the matcher is. That requires interpreter ids
decoupled from real ids (§2) to be actually *implemented*, not merely
designed — real ids deferred until enough of the round is known to rank
entities by chain depth first. Until §2 lands in code, `std/planes.rs`'s
existing two call sites should stay exactly as they are; replacing them with
something that merely *looks* generalized while still deciding eagerly would
be worse than not touching them, since it would silently reintroduce this
section's own bug under a more convincing-looking abstraction.

**Caveat, not yet resolved:** classifying a command as *creates* / *mutates
an existing entity in place* / *pure query* is a third dimension on top of
§3.4 (formerly numbered differently) below's `IdRefs`/`OwnsEntity` work, and
it is not always clean. `Solid3dFilletEdge` both mutates `object_id` in
place *and* produces new entities (`extra_face_ids`) — it needs both
treatments at once, on one command. This needs a real per-variant pass
before implementation, not an assumption that every command is purely one or
the other.

**Caveat, not yet resolved:** treating a pure statement reorder as fully
reusable (nothing superseded, as in `[a,b,c] -> [c,a,b]`) is a *stronger*
claim than today's behavior makes. Need to verify nothing downstream depends
on creation order for a reason other than engine content — e.g. feature-tree
display order, z-ordering of overlapping geometry — before relying on this
for anything beyond the send-reduction the cache already exists to do.

### 3.3 Per-command-type reference declaration

Checked against the installed `kittycad-modeling-cmds = 0.2.250` crate
(`define_modeling_cmd_enum!` in `def_enum.rs`, which generates `ModelingCmd`'s
167 variants, confirmed by brace-depth-aware parsing of `def_enum.rs` rather
than a line-oriented grep — the first pass undercounted at 163): **110 of
the 167 variants** declare at least one field typed `Uuid`, `Option<Uuid>`,
`Vec<Uuid>`, `Option<Vec<Uuid>>`, `ModelingCmdId`, or `Option<ModelingCmdId>`.
That is the real size of the "declare once per command type" surface — more
than double the ~20-40 guessed before looking, and a reminder that the number
needs re-confirming against the actual crate version in use whenever this is
implemented, not assumed from memory. The full, mechanically-derived list is
implemented as `IdRefs` in `command_cache.rs` (see §3.4) rather than
hand-maintained here.

Representative, individually verified entries (full list is now in
`command_cache.rs`'s `impl IdRefs for ModelingCmd`, not just this table):

| Variant | id-reference fields | notes |
|---|---|---|
| `MakePlane` | *(none)* | its own cmd_id **is** the plane's real id |
| `PlaneSetColor` | `plane_id: Uuid` | references, doesn't own; mutates in place |
| `ExtendPath` / `MovePathPen` | `path: ModelingCmdId` | uses the dedicated id-ref newtype |
| `Extrude` / `ExtrudeToReference` | `target: Option<ModelingCmdId>` | |
| `Sweep` | `target: ModelingCmdId`, `trajectory: ModelingCmdId` | two references |
| `RevolveAboutEdge` | `target: ModelingCmdId`, `edge_id: Option<Uuid>` | mixed typing in one variant |
| `Solid3dGetExtrusionFaceInfo` | `object_id: Uuid`, `edge_id: Option<Uuid>` | query, not a creation |
| `EntityLinearPatternTransform`, `EntityMirror`, `EntityCircularPattern` | `entity_id`/`ids: Vec<Uuid>` | response carries the *new* ids |
| `Solid3dFilletEdge` | `object_id`, `edge_id: Option<Uuid>`, `edge_ids: Vec<Uuid>`, `extra_face_ids: Vec<Uuid>` | mutates in place *and* creates — see §3.2's caveat |

Two typing conventions coexist (`ModelingCmdId` newtype vs. plain `Uuid`) and
neither is reliable alone: most reference fields are plain `Uuid`, which
cannot be distinguished from a non-reference `Uuid` by type (e.g. a literal
color or session token would look identical to the compiler). So this cannot
be derived mechanically from field types — it needs a human-reviewed
declaration, once, not per call site. That one-time cost is the actual "fewer
places to change" win the task description was pointing at (vs. 146 existing
`next_uuid()` call sites across `src/`, counted directly in this repo).

### 3.4 The declaration itself — implemented

This is no longer just designed: `IdRefs`, its `impl for ModelingCmd` (110
match arms, mechanically derived from `def_enum.rs` as described in §3.3),
and the `EntityIdentity`/`CommandEffect` classification enums all exist now
in `command_cache.rs`, with unit tests (`id_refs_tests`) covering each field
shape (bare `Uuid`, `ModelingCmdId`, `Vec<Uuid>`, `Option<Vec<Uuid>>`,
multi-field variants, and the no-op default). Still nothing calls any of it
— same "prove the piece in isolation" scoping as the rest of this module.

```rust
pub(crate) trait IdRefs {
    fn for_each_id_ref_mut(&mut self, visit: &mut dyn FnMut(&mut Uuid)) {}
}
```

(The trait is local, so `impl IdRefs for ModelingCmd` is a normal
inherent-looking impl despite `ModelingCmd` being a foreign type — no
newtype needed.)

Per §3.2, a command's role is classified along two further axes, not just
id-references — also implemented now, but only for a **representative
subset**, deliberately not the full 167:

- **Identity** (`EntityIdentity`): does its own cmd_id *own* a persistent
  entity (`MakePlane`), or is it a disposable correlation token (every
  query-style command already wired into `std/` today — see §1's list)?
- **Effect** (`CommandEffect`): does it *create*, *mutate in place*
  (`PlaneSetColor`), *query*, or both at once (`Solid3dFilletEdge`,
  `MutatesAndCreates`)?

`entity_identity`/`command_effect` return `Option`, and `None` means "not
yet classified," not "doesn't apply" — intentionally, since neither axis is
derivable from `ModelingCmdId` vs. `Uuid` typing, and guessing the rest for
167 variants without the engine's own docs would be fabricating semantics,
not designing. Widening these two tables is step 3's job (§6), gated on the
duplicate-content test below existing first.

### 3.5 Before further implementation

- The 110-variant `IdRefs` list is mechanically derived and in the code now
  (not hand-maintained in this doc) — but still needs re-deriving if
  `kittycad-modeling-cmds` is upgraded, and the handful of fields flagged
  uncertain (`CreateRegion`'s `segment`/`intersection_segment` — included
  conservatively, since treating a real reference as inert is the dangerous
  direction) still need confirming against the engine's own command docs,
  not just field names.
- `entity_identity`/`command_effect` cover only `MakePlane`, `PlaneSetColor`,
  the Boolean family, `Solid3dFilletEdge`, and the query-style commands
  already listed in §1 — confirm each against actual engine behavior before
  relying on it, and before adding more entries from naming convention
  alone.
- Write the duplicate-content-with-divergent-mutation scenario (§3.1) as an
  actual test *before* generalizing past `MakePlane`/`PlaneSetColor` — it is
  the sharpest known failure mode and should gate any further widening of
  scope, the same way the spike's own insertion test gated this doc's first
  draft.

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
feared once traced, and now **done**:

- `kcl_api::Object` (`scene_objects`, `source_range_to_object`) does **not**
  carry a real engine id at all — `Object.id: ObjectId` is a local `usize`
  bookkeeping id and `Object.artifact_id: ArtifactId` is a third, separate id
  space (`rust/kcl-api/src/front.rs:27`). Nothing to translate there.
  Unchanged by anything below.
- `ArtifactGraph` is already built as a single end-of-execution batch pass
  over the full `ArtifactCommand` log plus flattened engine responses
  (`state.rs::build_artifact_graph` ~1428, `artifact.rs::build_artifact_graph`
  ~454) — not incrementally during interpretation. **Done.** Confirmed by
  direct trace (`src/lib/selectSketchPlane.ts`,
  `src/hooks/useEngineConnectionSubscriptions.ts`) that the frontend does
  look plane artifact-graph nodes up by the id the engine reports back on
  selection in the 3D viewport — the real id, not the interpreter id — so
  this was a live gap, not a theoretical one. Fixed by translating
  `new_commands` (each `ArtifactCommand`'s `cmd_id`, every id-reference field
  via `IdRefs::for_each_id_ref_mut`, and `EntityCloneInfo`'s three
  `ArtifactId` fields) through `Cache::resolved_entity_ids()` right before
  they're handed to `artifact::build_artifact_graph` — same translate-in-place
  shape as `ExecOutcome.variables` below, reusing the already-built resolved-id
  map rather than threading a new one through. A no-op whenever the map is
  empty, i.e. for every execution that doesn't touch the deferred mechanism.
  Verified end-to-end with a new test
  (`command_cache_spike_artifact_graph_exposes_the_real_id_not_the_interpreter_id`)
  that runs an `offsetPlane`, confirms interpreter id and real id actually
  differ (so the assertion is meaningful), then asserts the `Plane` node in
  `exec_state.global.artifacts.graph` is keyed by the real id.
- `ExecOutcome.variables: IndexMap<String, KclValueView>` **— done.** Every
  `Solid`/`Sketch`/`Plane`/`Helix`/`ImportedGeometry`/`GdtAnnotation`'s `.id`
  (and `.artifact_id` for `Plane`/`Helix`, which the established codebase
  convention - see `std/sketch.rs`'s `ensure_sketch_plane_in_engine`, which
  always sets `artifact_id = id.into()` - ties to the same value as `.id`,
  so translating one without the other would have desynced them) is now
  rewritten in place before either of the two `into_exec_outcome`
  implementations returns (`state.rs`'s, which is the one most real callers
  use, and `cache.rs`'s `GlobalState` version). `cache.rs::kcl_value_referenced_ids`
  (already written for the surgical-delete alias check) turned out to be
  exactly the right traversal shape to copy, as predicted - its new sibling
  `translate_kcl_value_ids` mutates in place instead of collecting, with a
  `translate: impl Fn(Uuid) -> Uuid` that's the identity function for
  anything not part of the deferred mechanism (everything except
  `offsetPlane`'s planes today), fed from `Cache::resolved_entity_ids`.
  Verified with a new end-to-end test
  (`command_cache_spike_exec_outcome_exposes_the_real_id_not_the_interpreter_id`)
  that exercises `ExecState::into_exec_outcome` itself, not just the
  pre-translation state the other two spike tests read directly.

Net: the output-boundary problem was two call sites, as predicted, and both
are now done: `ExecOutcome.variables` and `build_artifact_graph`'s inputs
both translate interpreter ids to real ids before anything outside kcl-lib
sees them, closing the gap that would otherwise have shown up as a mismatch
between an offsetPlane's (now-correct) real-id-addressed `KclValueView` and
its (previously still interpreter-addressed) artifact-graph node.

## 6. Suggested implementation order

1. **Done.** `IdRefs` trait + the per-variant match (§3.4), tested standalone
   against synthetic `ModelingCmd` values — no wiring into execution yet,
   same spirit as `command_cache.rs`'s own introduction.
2. **Done.** The corrected matcher (§3.2): index-based hit/miss,
   exclusion-based supersession via a new `CommandLog::superseded`, no
   entity chaining yet. `MatchDecision` dropped its per-call `superseded`
   fields entirely (supersession is decided once, at the end, not per
   match) and `CommandLog` dropped `cursor`/`consume` - the corrected
   algorithm is simpler than the one it replaced, not more complex. The
   pre-existing `insertion_before_an_unchanged_tail_...` test now asserts
   the opposite of what it did before (reused, not recreated) and is
   renamed accordingly; a new `pure_reorder_reuses_everything` test covers
   the strictly-stronger case index-based matching also gets for free.
   `duplicate_content_pairs_with_earliest_unconsumed_occurrence_first`
   stays green but its doc comment now points at what it *doesn't* cover
   (divergent mutation among duplicates - step 3's job).
3. **Done**, completed by step 4 below. Entity-chained content identity
   (`EntityChain`, `ChainLog`, `ChainMatch`, `match_round`) and the
   duplicate-content test from §3.5 were implemented and passing in
   `command_cache.rs`, standalone, before step 4 existed - at that point
   generalizing past the two hand-written call sites in `std/planes.rs`
   was explicitly *not* safe, because it needed real ids deferred until a
   whole round's chains are known (§2's split, not yet implemented in code
   at the time). Step 4 built that split for real and `std/planes.rs` now
   uses the chain mechanism directly - see step 4.
4. **Done.** §2's interpreter-id/real-id split, implemented for real, not
   just designed - this is what actually unblocked step 3's chain matcher
   for live use rather than only its own standalone tests. Concretely:
   - `CommandCacheRound` reshaped around `register_creation`/
     `register_mutation` (record immediately, decide nothing yet) and
     `resolve_pending` (decide a batch of real ids at once, right before a
     flush is about to send anything - see below).
   - `EngineBatchContext::with_pending_batch` (`engine/mod.rs`): a small,
     fully generic, purely additive hook - take the queued-but-unsent
     batch, let the caller transform it, put it back. Zero behavior change
     for any caller that doesn't use it (everyone except the one below).
   - `ExecState::flush_batch` (`execution/modeling.rs`): the actual
     translation step, now the single common chokepoint for all of this.
     Calls `resolve_pending`, and - only when it returns something
     non-empty, a cheap no-op otherwise - rewrites each queued command's
     own id and reference fields (via `IdRefs`) from interpreter id to
     real id, or drops it outright on a hit. Every flush that isn't part
     of this mechanism (effectively every flush in the entire app except
     the two below) passes through completely untouched.
   - `std/planes.rs`'s `make_offset_plane_in_engine`: now mints a plain
     interpreter id immediately (reverting the eager, pre-step-4 decision)
     and registers creation/mutation content instead of hand-computing and
     resolving a real id inline. The two existing end-to-end tests in
     `execution/mod.rs` were rewritten to check the *real* id
     (`Cache::real_id_of`) rather than `plane.id` (now just the
     deterministic interpreter id, which would make a same-id assertion
     vacuous) - and the second one, previously asserting the known
     streaming-matcher limitation, now asserts the fix instead: inserting
     an unrelated statement before `plane001` no longer forces a recreate,
     verified end-to-end through the real call sites and the flush-time
     translation step, not just at the unit level.
   - Verified with no regressions: stashed this branch's changes and ran
     the identical targeted test sweep (`execution::`, `std::planes::`,
     `engine::`, `command_cache`) against the unmodified baseline - the
     same 44 tests fail on both (all pre-existing, needing `ZOO_API_TOKEN`/
     network access this environment doesn't have, confirmed unrelated by
     inspecting their panics), and the one test this step actually changes
     the behavior of flips from failing to passing only on this branch.
5. **Done.** Output-boundary rewrite (§5): `ExecOutcome.variables` is
   fixed - both `into_exec_outcome` implementations now translate
   interpreter ids back to real ids via the new `translate_kcl_value_ids`,
   verified end-to-end (see §5 above). `ArtifactGraph` is also fixed -
   `ExecState::build_artifact_graph` now translates each `ArtifactCommand`
   (via `IdRefs` and `Cache::resolved_entity_ids()`) before building the
   graph, so an offsetPlane's `Plane` node is addressed by the real id, same
   as the plane's `KclValueView`. Confirmed this mattered for real (not just
   in theory) by tracing the frontend's plane-selection path
   (`selectSketchPlane.ts` → `kclManager.artifactGraph.get(planeOrFaceId)`,
   fed by the engine's own selection event, which reports the real id) before
   implementing the fix. Verified end-to-end with a new test
   (`command_cache_spike_artifact_graph_exposes_the_real_id_not_the_interpreter_id`).
6. Response caching (§4): separate follow-up, not a prerequisite for 1-5.

Each step should land as its own reviewable commit with its own tests, same
as the three commits already on this branch — this doc is deliberately not
a license to jump straight to a big-bang rewrite of `std/`.
