---
name: gen-edit
description: Read this BEFORE changing an existing picture by instruction (recolour, replace, remove, restyle, put a person into another scene) on the local GPU with "ai gen edit", up to four reference images, and the img-check step that proves the result.
---

# Edit a picture

## When to Use
The user wants an existing picture changed, or a person or object from one picture placed into another.

## Procedure
1. The picture to change and any extra references must be .png/.jpg/.webp inside `~/work` (usually `~/work/inputs/`).
   The original is never touched; the result is a new file.
2. Pick the edit model from the inventory (`ai models list --json`): models whose description says they edit.
   FLUX 3 is the default. Fibo 2.0 (`--model fibo2`, non-commercial licence) is the second: refer to the pictures as
   <image_1>, <image_2> and describe the FINAL state of what changes - do not write "keep X unchanged".
3. Describe the change; everything else stays. To change the setting, say so ("change the background completely").
   FLUX 3 uses the instruction as typed; `--improve-prompt` first expands it with the local chat model.
4. Optional check without the GPU: `ai gen edit "<change>" --image <file> --name <name> --check`.
5. Run ONE command and wait: `ai gen edit "<what to change>" --image <file> [--image <extra reference>] --name <name>`.
   If one edit model gets a pose or detail wrong, try the other.
6. Report the files after `OUTPUT` and the seed; before you report success run `img-check <file>` and quote its line,
   and compare the result with the request if your vision tool works.

## Rules
- One `ai gen` at a time; never call the ComfyUI API directly. `GPU busy`: wait, retry once. `FAILED`: report it.
- Never put a real person into sexual or degrading images; only edit people who agreed or fictional characters.
