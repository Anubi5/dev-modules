# EVA modules

Reusable local model, generation and agent modules for an EVA-compatible Linux runner. This public repository owns module implementations and contracts; the private [Scripts platform](https://github.com/Anubi5/Scripts) owns Windows/WSL provisioning, policy, release selection and recovery. Platform access is required for `dev` commands; this repository is not a standalone workstation installer.

## Catalogue

- [llama-swap](llama-swap/README.md): local OpenAI-compatible model serving and image questions.
- [comfyui](comfyui/README.md): image/edit/pose, video and textured 3D generation; reviewed model families.
- [agent-hermes](agent-hermes/README.md): pinned Hermes runtime and selected local profile.
- [blender](blender/README.md): Blender, prepared official MCP integration and rendering.
- [reverse-tools](reverse-tools/README.md): static-analysis skill with tools acquired on request.
- [rig](rig/README.md): experimental humanoid rigging; visual quality remains deferred.
- [character](character/README.md): finite file-based pipelines and attended animation handoff.

## Install and update

```powershell
dev mod pin
dev mod add comfyui --plan --json
dev mod add comfyui
dev mod test comfyui
```

The platform selects an exact reviewed module revision. Pinning does not install or approve weights. Missing eligible model files need source, byte size, license and SHA-256 plus explicit Windows download approval. Discovery/runtime launch never grants acquisition. Offline installation requires all needed source/dependencies cached; a retained runtime working does not imply every repeat installer is offline-safe. Review module-specific limitations before updating.

Inside Linux, `ai mod` exposes lifecycle operations; `ai mod add --from <directory>` selects explicit trusted local source for development. Run `ai mod --help` for exact options. Do not bypass approval by invoking a model fetcher directly.

## Module contract and development

Copy [_template](_template/README.md). Each module supplies `README.md`, `install.sh` and a meaningful `test.sh`; optional `remove.sh`, `check.sh`, `bin/`, `skills/`, `*.service` and `state.txt` extend it. `bin/ai-NAME` becomes `ai NAME`; skill directories are linked to supported agent homes. User services start on demand. `state.txt` declares home-relative private data to preserve, excluding weights, programs and caches.

Installers run as the working user and should be repeatable. Write application code/environments inside `$MOD`, results under `$WORK_DIR`, and only explicitly declared integration/state paths elsewhere. Do not edit the platform or another module. Shared UID ownership is an engineering convention, not adversarial isolation. Review source and dependencies before installation.

The runner supplies `MOD`, `MOD_NAME`, `WORK_DIR`, `MODELS_MOUNT`, `MODELS_RW`, `LLM_PORT`, `CUDA_ARCH` and `MOE_CPU`. Normal models are read-only; the writable mount is granted only for approved acquisition. Preserve existing weights/owner files. Model serving uses an OpenAI-style API and `POST /api/models/unload`. Heavy GPU work must cooperate with the shared lease and restore prior service state; hard-crash recovery needs explicit ownership checks.

Modules managing weights provide `model-plan.sh`, emitting schema-1 JSON `files` with path, bytes, source, licence, sha256, eligible and optional reason. The hook only reads declarations: no network discovery, downloads, installs or state writes. The platform binds approval/execution to the same clean source revision. Restricted licenses and unknown sizes remain manual, never implicitly accepted.

Tests run from this repository root. `bash _tests/contract_check.sh` checks module obligations; `.github/workflows/check.yml` defines focused suites. Module commands answer their own `--help`; the platform forwards it and keeps no copy, so parser changes need no export. Real GPU acceptance is separate from offline fixtures. Module READMEs identify additional dependency, contract and recovery requirements.
