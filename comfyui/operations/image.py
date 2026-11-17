"""!
@file image.py
@brief ai gen image (and the shared part of ai gen edit): one picture from a model profile or file, with its options,
       LoRAs and reference images checked before the GPU is touched. @return of prepare: the job for the router.
"""
import re
import sys

import runtime as rt
from generation import builders
from generation.graph import apply_loras, use_skeleton
from generation.models import PROFILES, negative_for


OPTIONS = {"model", "checkpoint", "style", "negative", "count", "steps", "size", "upscale", "hires",
           "pose", "pose_skeleton", "pose_strength", "face", "face_weight", "cfg", "force_lora", "lora",
           "fix_faces", "fix_hands", "improve_prompt", "no_expand"}


def size(a, example):
    """! @brief --size WxH rounded down to multiples of 64 and within the per-job limits. @return (width, height) """
    m = re.fullmatch(r"(\d{3,4})x(\d{3,4})", a.size)
    if not m:
        sys.exit(f"--size must look like {example}")
    w, h = (int(v) // 64 * 64 for v in m.groups())
    if not (512 <= w <= rt.MAX_SIDE and 512 <= h <= rt.MAX_SIDE):
        sys.exit(f"--size: each side must be 512-{rt.MAX_SIDE}")
    return w, h


FEATURES = ("pose", "face", "fix_faces", "fix_hands", "hires", "upscale", "improve_prompt")


def supported(capability):
    """! @brief The options a family declares it supports, as typed on the command line. """
    return [f for f in FEATURES if f in capability["features"] or (f == "upscale" and capability["upscale"])
            or (f == "improve_prompt" and capability["expand"])]


def unsupported(prof, capability, missing):
    """!
    @brief The refusal for options a family does not support (before any GPU work): names the family, what it
           supports and the image models that support every requested option - all from the family declarations.
    """
    flags = lambda names: ", ".join("--" + n.replace("_", "-") for n in names) or "none of the optional features"
    others = []
    for name, p in PROFILES["models"].items():
        if p.get("kind", "image") != "image" or name == prof["name"]:
            continue
        try:
            have = supported(builders.capabilities(dict(p, name=name)))
        except ValueError:
            continue
        if all(m in have for m in missing):
            others.append(name)
    return (f"--model {prof['name']} (family {prof.get('family') or prof['arch']}) does not support {flags(missing)}; "
            f"it supports {flags(supported(capability))}. "
            + (f"Models that support {flags(missing)}: {', '.join(others)}" if others else f"No installed family supports {flags(missing)}"))


def prepare(a, seed, prefix, edit=None):
    """!
    @brief The workflow and record details of one image job. @param edit  the edit operation (its own checks), or None
    @return (workflow, uploads, info, time limit, profile)
    """
    w, h = size(a, "1024x1024")
    if a.model and a.checkpoint:
        sys.exit("use --model or --checkpoint, not both")
    if edit:
        edit.before(a)
    selection_reason = 'explicit model' if a.model or a.checkpoint else 'general default'
    if not a.model and not a.checkpoint:
        choice = next(((flag, PROFILES[key]) for flag, key in PROFILES.get('feature_defaults', {}).items()
                       if getattr(a, flag, False)), None)
        if choice:
            flag, a.model = choice
            selection_reason = f"--{flag.replace('_', '-')} feature default (general default: {PROFILES['default']})"
    log0 = lambda t: print(f"  {t}", flush=True)
    prof = rt.find_checkpoint(a.checkpoint, log0) if a.checkpoint else rt.resolve_model(a.model or PROFILES["default"], log0)
    log0(f"Selected model {prof['name']} (family {prof.get('family') or prof['arch']}): {selection_reason}")
    if not getattr(a, 'check', False):
        builders.registry.check_parts(prof, rt.MODELS)
    arch, style = prof["arch"], a.style or prof.get("style", "photo")
    capability = builders.capabilities(prof)
    allowed = capability["features"]
    used = [o for o in ("pose", "face", "fix_faces", "fix_hands", "hires") if getattr(a, o)]
    missing = [o for o in used if o not in allowed]
    if missing:
        sys.exit(unsupported(prof, capability, missing))
    if edit:
        edit.check(capability)
    if a.image and not edit:
        sys.exit("--image is for edit")
    pose_strength = a.pose_strength if a.pose_strength is not None else prof.get("pose_strength", 0.65)
    if not (0.3 <= pose_strength <= 1.0 and 0.3 <= a.face_weight <= 1.2):
        sys.exit("--pose-strength must be 0.3-1.0 and --face-weight 0.3-1.2")
    neg = a.negative if a.negative is not None else negative_for(prof, style)
    steps = a.steps or prof["steps"]
    loras, prompt, neg, prof, lora_checks = rt.resolve_loras(a, prof, neg, log0)
    uploads = [(node, rt.check_input(getattr(a, option))) for option, node in capability['uploads'].items()
               if getattr(a, option)]
    refs = [rt.check_input(f) for f in a.image]
    uploads += [(f"2{i}0", f) for i, f in enumerate(refs)]
    if a.improve_prompt and not capability['expand']:
        sys.exit(unsupported(prof, capability, ["improve_prompt"]))
    if a.improve_prompt:
        if getattr(a, "check", False):           # --check starts no service: only say what the run will need
            a.needs = dict(getattr(a, "needs", {}), **{"llm.chat": "llama-swap"})
        else:
            lacking=rt.missing_capabilities({'llm.chat':'llama-swap'})
            if lacking:sys.exit('Prompt improvement needs '+', '.join(lacking))
            prompt = rt.expand_prompt(a.prompt, "edit" if refs else "t2i", w, h, refs, log=log0, templates=capability.get('prompt_templates'))
    if not capability['negative']: neg = ""
    if a.upscale and not capability['upscale']:
        sys.exit(unsupported(prof, capability, ["upscale"]))
    wf = builders.build(prof, dict(prompt=prompt, negative=neg, seed=seed, steps=steps,
        width=w, height=h, prefix=prefix, pose_strength=pose_strength), a)
    if a.pose and rt.is_skeleton(a.pose, a.pose_skeleton): use_skeleton(wf)
    apply_loras(wf, loras, arch)
    info = {"kind": a.kind, "prompt": a.prompt, "expanded_prompt": prompt if prompt != a.prompt else None,
            "negative": neg, "seed": seed, "model": prof["name"], "arch": arch,
            "model_selection_reason": selection_reason,
            "model_file": prof["file"], "sampler": prof["sampler"], "scheduler": prof["scheduler"], "cfg": prof["cfg"],
            "steps": steps, "size": f"{w}x{h}", "style": style,
            "options": {k: getattr(a, k) for k in ("upscale", "hires", "fix_faces", "fix_hands")},
            "pose": a.pose and {"file": a.pose, "strength": pose_strength},
            "face": a.face and {"file": a.face, "weight": a.face_weight}, "images": a.image or None,
            "loras": lora_checks or None,
            "face_refiner": None}
    if capability.get('report'):
        info.update(capability['report'](prof, wf, a))
    limit = a.timeout or (1800 if a.hires or a.fix_hands or a.fix_faces else capability['timeout'])
    return wf, uploads, info, limit, prof
