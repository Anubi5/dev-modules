---
name: gen-video
description: Read this BEFORE making a short video clip on the local GPU with "ai gen video" - prompt, length limits, run time, and the img-check step that proves the result.
---

# Generate a short video

## When to Use
The user asks for a short animated clip (a few seconds).

## Procedure
1. Write what happens and how the camera moves: subject, action, setting, light ("smooth camera tracking").
   Run a model whose `ai models list` stage is `accepted` unless the owner names another for this run; `verified` or
   `installed` models are not proven on this PC yet - say so if you use one.
2. Choose the length: `--frames 33` (~1.4 s, ~2-6 min on the GPU) up to `--frames 121` (~5 s); lengths are 4n+1.
3. Optional check without the GPU: `ai gen video "<prompt>" --name <name> --check`.
4. Run ONE command and wait for it: `ai gen video "<prompt>" --name <name> [--frames 33] [--seed N]`.
   Image options (`--pose`, `--face`, `--fix-*`, `--hires`, `--lora`) are refused for video.
5. The file after `OUTPUT` is an animated WebP in `~/work/outputs/comfy/<date>/`; report it with the seed. Before you
   report success run `img-check <file>` and quote its line.

## Rules
- One `ai gen` at a time; never call the ComfyUI API directly. `GPU busy`: wait, retry once. `FAILED`: report it.
- Limits: up to 121 frames, up to 60 steps. Do not send chat requests while a job runs.

Timeout overrides use `--timeout N`, with N from 60 to 7200 seconds. Options belonging to another operation are rejected before execution.
