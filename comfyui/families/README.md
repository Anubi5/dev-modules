# Reviewed model families

Each directory contains one `package.json` (`local-ai.family/1`): model parts
with byte counts/SHA-256, defaults, capability flags, licence/source references,
and a ComfyUI API workflow with declared recipe bindings. Krea 2 Turbo is the
first independently added family. It reuses the existing recipe engine; no
database, daemon, service, dependency environment or executable plugin is added.

These packages are trusted module source. Model files under `/models` are data;
copying a weight there never installs or authorizes a builder. Metadata discovery
never imports a package's Python files. New architectures express their native
ComfyUI graph in a reviewed package; existing complex families retain narrow
adapters to preserve their workflows.

To add an image family, add a directory and its manifest, validate every bound
input against the pinned `schema.json`, and test its required files and actual
workflow. No edit to `gen.py`, `profiles.json`, or another family is required.
Removing the package leaves its weight files untouched and removes its runtime
selection. Unrecognized remaining safetensors appear as `needs-something` in
the read-only inventory; copied LoRAs and GGUFs appear as `untested`.

```bash
ai gen image --model krea2 --seed 19 --json -- 'a ceramic teapot'
ai models list --json
ai models show krea2
ai models verify krea2  # Explicitly reads every component; never changes it.
```

`ready` means declared files/dependencies are present, not visual/GPU acceptance.
`verify` checks content against known pins, or fingerprints unpinned user files;
its result explicitly distinguishes those cases. Discovery never hashes entire
files, downloads, installs, loads a model, starts a service, or modifies weights.

Krea's [v1 licence](https://www.krea.ai/krea-2-licensing) grants worldwide use;
no EU/Germany exclusion is stated. It incorporates an
[acceptable-use policy](https://www.krea.ai/krea-2-use-policy) and requires
appropriate filtering or review. This experimental package is for attended
personal use: the operator must review prompts and outputs before use/sharing.
The published terms also include attribution/distribution requirements and
commercial revenue conditions. Weights are supplied separately on D:.
