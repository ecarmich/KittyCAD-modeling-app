@AGENTS.md

## Running the app locally to test changes

Do **not** use `npm run tron:start` (electron-forge dev mode) to manually test a change end to end.
It launches against the non-production `dev.zoo.dev` infrastructure, which most accounts
(anyone outside Zoo) cannot sign in to - the sign-in flow gets stuck / CORS-walled there. This
matches CONTRIBUTING.md's own note that `tron:start` "isn't accessible to everyone."

Instead, build and launch the packaged **production** app, which is the path CONTRIBUTING.md
documents as working for external contributors:

```bash
npm run build:wasm        # after any change under rust/ (kcl-lib, etc.) - rebuilds the wasm layer
npm run tronb:package:prod
```

This produces a runnable app in `out/` (e.g. `Zoo Design Studio.app` on macOS) that points at
production zoo.dev and uses the normal sign-in flow. Re-run both steps after every change under
`rust/` you want to see reflected in the running app - `build:wasm` is not automatic and
`tronb:package:prod` packages whatever wasm is already sitting in `public/`.

If `~/.cargo/bin` or `~/.asdf/shims` aren't already on `PATH` in the shell you're running these
in, `wasm-pack`/`cargo` (cargo-installed) or `node`/`npm` (asdf-managed) will fail with
"command not found" even though they're installed; add them to `PATH` for the command rather than
assuming they're missing.

Note that the `tronb:package:prod` npm script itself passes `electron-builder ... --publish always`.
Before running it, confirm no publish credentials (`GH_TOKEN`, `CSC_LINK`, Apple notarization vars,
etc.) are set in the environment, so a local test build can't actually publish anywhere; without
them electron-builder just builds locally and skips/warns on the publish step.

This is also a genuinely slow build (full release wasm compile + vite bundling + electron-builder
packaging, several minutes total): run it with Bash's `run_in_background: true` and no trailing
`&` in the command itself (a trailing `&` makes the backgrounded shell call return immediately,
so the completion notification fires for that instant return, not for the real build - you'll
need to separately wait on the actual process to get a correct "it's done" signal).
