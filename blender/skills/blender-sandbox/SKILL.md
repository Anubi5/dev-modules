---
name: blender-sandbox
description: Read this BEFORE any Blender work in this sandbox - what is already set up for the official Blender MCP tools (Blender, add-on, a private display with screenshots), where files go, and how to prove a result.
---

# Blender in this sandbox

Work in Blender through the **official Blender Lab MCP tools** (Hermes plugin `blender`; its skill explains the tools).
Everything they need is already installed and pinned here - do not download or install Blender or the add-on.

## What is set up
- `blender` 5.2 on PATH, the Blender Lab MCP add-on v1.0.3, and the MCP server (prepared, works offline).
- A private virtual display (`DISPLAY=:99`, service `blender-display`): open Blender normally with
  `blender --online-mode` (detached). It gets a real window inside the sandbox - **screenshots and viewport renders
  work** - and nothing appears on the owner's Windows desktop. Use the window mode, not `--background`.

## How to work
1. Read the scene first (`get_objects_summary`), then make changes in small steps.
2. **Look after each change**: take a window screenshot or a viewport render and check it against the request
   (is the object in view, the right colour, the right size?). Fix what you see before going on.
3. Save `.blend`, `.glb`/`.fbx` and final renders into `~/work/projects/<name>/output/`.
4. Before you report done: run `img-check <final image>` (pixels: not blank, colours) and
   `img-ask <final image> "<is what was asked visible?>"` (a vision model says what it sees), quote both lines, and
   `ls -l` the saved files. Say honestly if something is off.

## Rules
- One heavy GPU job at a time: run `ai free` before long Cycles renders.
- Close the Blender you opened when the task is finished (it keeps memory and the bridge port 9876).
