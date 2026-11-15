---
name: gen-3d
description: Read this BEFORE making a textured 3D model (GLB) from one picture, or from up to four views, on the local GPU with "ai gen 3d" - model choice, quality, the Blender renders and mesh numbers that prove the result.
---

# Make a 3D model from a picture

## When to Use
The user wants a 3D model (GLB) of one object or character, from a picture or a turnaround.

## Procedure
1. The picture must be inside `~/work`: one object, plain background works best. Without one, make it first with
   `ai gen image` (see the gen-image skill). The back of the object is invented by the model.
2. Choose the model from the inventory (`ai models list --json`, capability 3d): the default follows the photo closely;
   a multi-view model takes up to four views in the order its description gives (front, left, back, right).
   Run a model whose `ai models list` stage is `accepted` unless the owner names another for this run; `verified` or
   `installed` models are not proven on this PC yet - say so if you use one.
3. Optional check without the GPU: `ai gen 3d --image <picture> --name <name> --check`.
4. Run ONE command and wait: `ai gen 3d --image <picture> [--image <next view> ...] --name <name>`;
   `--quality high` only when asked (slower, less GPU headroom); `--faces N` for a lighter or denser mesh;
   `--keep-background` when the picture is already cut out.
5. The output lists the .glb, four Blender renders (`*_view_front.png` ...) and a record with triangles, open edges and
   loose parts. Look at the renders (img-ask or your vision tool) before saying the model is good; `CHECK FAILED` or
   `NOT CHECKED` means a required check did not pass - report it.

## Rules
- One `ai gen` at a time; never call the ComfyUI API directly. `GPU busy`: wait, retry once. `FAILED`: report it.
- The 3D job holds the GPU through the Blender check; do not start Blender yourself meanwhile.

Timeout overrides use `--timeout N`, with N from 60 to 7200 seconds. Options belonging to another operation are rejected before execution.
