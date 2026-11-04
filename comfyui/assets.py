#!/usr/bin/env python3
"""!
@file assets.py
@brief What is in the model folders and what is known about it, for ai gen list/show/check/note/verify.
@details Nothing is indexed: every call lists the folders as they are now (D:\\AI\\models\\comfy and the module's own
         folder), so files added, renamed or removed show up at once. Headers are read only when a file is shown or
         used. What cannot be read from a file (trigger words, strength, settings from a LoRA's card) lives in
         ~/work/comfyui/notes.json, keyed by the file's SHA-256 (survives renames, ignores a changed file). LoRA + model
         pairs that ran are recorded in ~/work/comfyui/tested.json, keyed by both hashes. Model files are never written.
         Embedded metadata is untrusted text from the file: only short, length-limited fields are shown.
"""
import datetime
import hashlib
import json
import os
import re
import struct

ROOTS = [os.path.join(os.environ.get("MODELS_MOUNT", "/models"), "comfy"),
         os.path.expanduser("~/.local/share/modules/comfyui/models")]
EXT = (".safetensors", ".gguf", ".pt", ".pth", ".bin", ".ckpt", ".onnx")
DATA = os.path.expanduser("~/work/comfyui")
NOTES, TESTED = os.path.join(DATA, "notes.json"), os.path.join(DATA, "tested.json")
CACHE = os.path.expanduser("~/.cache/ai-gen/hashes.json")    # path -> [size, mtime, sha256]; disposable
NOTE_KEYS = {"trigger": str, "strength": float, "cfg": float, "negative": str, "base": str, "about": str, "source": str,
             "sampler": str, "scheduler": str, "steps": int, "arch": str, "text_encoder": str, "vae": str,
             "shift": float, "edit_sampler": str, "style": str, "pose_strength": float}
# the keys a note shares with a profiles.json entry: a note is a local profile (profiles.json = the shipped ones)
PROFILE_KEYS = ("arch", "sampler", "scheduler", "steps", "cfg", "text_encoder", "vae", "shift", "edit_sampler", "style",
                "pose_strength")
MAX_HEADER = 100 << 20


def files(folder=None):
    """! @brief Model files now on disk. @param folder  e.g. "loras" (None: all) @return [{name, folder, path, size}] """
    out, seen = [], set()
    for root in ROOTS:
        for top in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            if folder and top != folder or not os.path.isdir(os.path.join(root, top)):
                continue
            for r, _, fs in os.walk(os.path.join(root, top)):
                for f in sorted(fs):
                    if f.lower().endswith(EXT):
                        p = os.path.join(r, f)
                        name = os.path.relpath(p, os.path.join(root, top))
                        if (top, name) not in seen:
                            seen.add((top, name))
                            out.append({"name": name, "folder": top, "path": p, "size": os.path.getsize(p)})
    return out


def header(path):
    """!
    @brief Reads a .safetensors header (no tensors). @param path  File.
    @return (tensors {name: info}, metadata {str: str}, complete) - complete is False while a copy is still running
            (file shorter than the data the header declares); (None, {}, None) when it is not a readable safetensors file.
    """
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if not 2 <= n <= MAX_HEADER:
                return None, {}, None
            h = json.loads(f.read(n))
    except (OSError, ValueError, struct.error):
        return None, {}, None
    meta = h.pop("__metadata__", {}) or {}
    end = max((v.get("data_offsets", [0, 0])[1] for v in h.values() if isinstance(v, dict)), default=0)
    return h, meta, os.path.getsize(path) >= 8 + n + end


def sha256(path):
    """! @brief SHA-256 of a file; cached by path, size and modified time (ai gen verify re-reads everything). """
    st = os.stat(path)
    cache = _load(CACHE)
    c = cache.get(path)
    if c and c[0] == st.st_size and c[1] == int(st.st_mtime):
        return c[2]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    cache[path] = [st.st_size, int(st.st_mtime), h.hexdigest()]
    _save(CACHE, cache)
    return h.hexdigest()


def _load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return {}


def _save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def summary(meta):
    """! @brief The useful, bounded part of embedded metadata (untrusted text; each value cut to 200 characters). """
    keep = {}
    for k in ("modelspec.title", "modelspec.architecture", "modelspec.description", "modelspec.trigger_phrase",
              "modelspec.usage_hint", "modelspec.resolution", "modelspec.license", "ss_base_model_version",
              "ss_sd_model_name", "ss_output_name", "ss_network_dim", "ss_network_alpha", "ss_resolution"):
        if meta.get(k):
            keep[k] = str(meta[k])[:200]
    try:                                                     # most frequent training tags: what the LoRA was taught
        freq = {}
        for tags in json.loads(meta.get("ss_tag_frequency", "{}")).values():
            for t, c in tags.items():
                freq[t] = freq.get(t, 0) + c
        if freq:
            keep["top_tags"] = ", ".join(sorted(freq, key=freq.get, reverse=True)[:8])[:200]
    except (ValueError, AttributeError):
        pass
    return keep


TE = ("lora_te", "te_", "te1_", "te2_", "text_encoder", "text_encoders", "clip_", "lora_clip", "conditioner")


def _module(k, lora):
    """! @brief A tensor name reduced to the layer it belongs to, comparable between LoRA and model files. """
    if lora:
        for cut in (".lora_", ".hada_", ".lokr_", ".alpha", ".dora_scale", ".diff"):
            k = k.split(cut)[0]
    else:
        for suf in (".weight", ".bias"):
            if k.endswith(suf):
                k = k[: -len(suf)]
    for pre in ("lora_unet_", "base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.", "unet.",
                "model.", "lora_transformer_"):
        if k.startswith(pre):
            k = k[len(pre):]
    return k.replace(".", "_")


def _fused(m):
    """! @brief Names a separate q/k/v projection has when the model stores them fused (ComfyUI splits the LoRA over it). """
    out = []
    for pat, rep in ((r"(^|_)(q|k|v)$", r"\1qkv"), (r"(^|_)(k|v)$", r"\1kv"), (r"(^|_)(q|k|v)_proj$", r"\1qkv_proj"),
                     (r"(^|_)to_(q|k|v)$", r"\1to_qkv")):
        f = re.sub(pat, rep, m)
        if f != m:
            out.append(f)
    return out


def lora_fit(lora_path, model_path):
    """!
    @brief Structural fit: how many of the LoRA's layers exist in the model (headers only, no GPU, no loading).
    @return {"status": fits|partial|mismatch|unchecked, "matched": n, "layers": n, "why": text}. "unchecked" means
            the key format cannot be compared here (it may still work); only "mismatch" is a proven conflict.
    """
    lt, _, lok = header(lora_path)
    mt, _, mok = header(model_path)
    if lt is None or mt is None:
        return {"status": "unchecked", "matched": 0, "layers": 0, "why": "not a safetensors file (GGUF, .pt ...)"}
    if lok is False or mok is False:
        return {"status": "unchecked", "matched": 0, "layers": 0, "why": "file incomplete (still copying?)"}
    mods = {_module(k, True) for k in lt if not k.lower().startswith(TE)}
    mods.discard("")
    if not mods:
        return {"status": "unchecked", "matched": 0, "layers": 0, "why": "only text-encoder layers"}
    have = {_module(k, False) for k in mt}
    hit = sum(1 for m in mods if m in have or any(f in have for f in _fused(m)))
    if hit == 0 and any(s in m for m in mods for s in ("down_blocks", "up_blocks", "mid_block")):
        return {"status": "unchecked", "matched": 0, "layers": len(mods), "why": "diffusers-style names, not compared"}
    share = hit / len(mods)
    status = "fits" if share >= 0.9 else "partial" if hit else "mismatch"
    return {"status": status, "matched": hit, "layers": len(mods),
            "why": f"{hit} of {len(mods)} LoRA layers found in the model"}


def notes():
    """! @brief All notes: {sha256: {name, trigger, strength, cfg, negative, base, about, source}} """
    return _load(NOTES)


def note_for(path):
    """! @brief The note for a file (by content) or {}. """
    return notes().get(sha256(path), {})


def set_note(path, values):
    """! @brief Adds or changes a note (values typed by NOTE_KEYS; "" removes a field). @return the note """
    all_, key = notes(), sha256(path)
    n = all_.get(key, {})
    n["name"] = os.path.basename(path)
    for k, v in values.items():
        if v == "" and k != "negative":
            n.pop(k, None)
        else:
            n[k] = NOTE_KEYS[k](v)
    all_[key] = n
    _save(NOTES, all_)
    return n


def tested(lora_path=None):
    """! @brief Recorded successful runs {lora sha: {model sha: {"model": file, "lora": file, "date": d}}} (one LoRA's when given). """
    t = _load(TESTED)
    return t.get(sha256(lora_path), {}) if lora_path else t


def record_tested(lora_path, model_path):
    """! @brief Remembers that this LoRA ran with this model (exact contents, not the architecture). """
    t, lk, mk = _load(TESTED), sha256(lora_path), sha256(model_path)
    t.setdefault(lk, {})[mk] = {"lora": os.path.basename(lora_path), "model": os.path.basename(model_path),
                                "date": datetime.date.today().isoformat()}
    _save(TESTED, t)


def verify():
    """!
    @brief Re-reads every file whose hash was used for a decision (all files in the hash cache), even when size and time
           look unchanged. @return [(file, problem)]
    """
    known = set(notes()) | set(_load(TESTED)) | {m for v in _load(TESTED).values() for m in v}
    bad = []
    for path, (_, _, old) in sorted(_load(CACHE).items()):
        cache = _load(CACHE)
        cache.pop(path, None)
        _save(CACHE, cache)
        if not os.path.exists(path):
            continue
        new = sha256(path)
        if new != old:
            bad.append((path, "content changed under the same name" +
                        (": its notes and test results no longer apply" if old in known else "")))
    return bad
