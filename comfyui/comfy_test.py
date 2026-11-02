#!/usr/bin/env python3
"""!
@file comfy_test.py
@brief End-to-end ComfyUI test through the GPU hand-off (gen.py): SDXL image + 4x upscale, a short Wan 2.2 video, then
       the control features on a generated portrait: its pose and face reused, faces/hands repaired, detailed upscale;
       ai gen pose (photo + skeleton from a description) and an image made from that saved skeleton.

Uses the same fixed workflows, lock, unload/free and output folder as "ai gen", so it tests exactly what Hermes uses.
Outputs go to COMFY_OUTPUT/<date>/selftest-*. Exit code 0 only if every job produced its files there.
"""
import datetime
import os
import sys

import gen


def main():
    """!
    @brief Runs both test jobs. @return 0 if both passed, else 1.
    """
    day = f"{datetime.date.today():%Y-%m-%d}"
    jobs = [
        ("SDXL image + 4x upscale", gen.sdxl("a lighthouse on a rocky coast at sunset, dramatic clouds, photograph, highly detailed",
                                            gen.NEGATIVE, 42, 30, 1024, 1024, True, f"{day}/selftest-sdxl"), 900),
        ("Wan 2.2 5B video (GGUF, 33 frames)", gen.wan("a red fox running through fresh snow in a forest, cinematic, smooth camera tracking",
                                                      "static, blurry, distorted, low quality", 7, 20, 33, f"{day}/selftest-wan"), 1800),
    ]
    results = []
    portrait = None
    for name, wf, limit in jobs:
        ok, files, err = gen.generate(wf, limit, log=lambda s: print(f"    {s}", flush=True))
        print(f"  {name}: {'OK' if ok else 'FAILED'}  {', '.join(files) or '-'}{'  error: ' + err if err else ''}")
        results.append(ok and all(f.startswith(gen.OUTPUT + "/") for f in files))
    # pose control, consistent face, repair and tiled upscale: a portrait first, then reuse its pose and face
    log = lambda s: print(f"    {s}", flush=True)
    ok, files, err = gen.generate(gen.sdxl("RAW photo, portrait of a woman standing, front view, face clearly visible, arms relaxed, soft window light, 85mm, natural skin texture",
                                           gen.NEGATIVE, 11, 30, 896, 1152, False, f"{day}/selftest-ref"), 900, log=log)
    portrait = files[0] if ok and files else None
    print(f"  reference portrait: {'OK' if portrait else 'FAILED'}  {portrait or err}")
    results.append(bool(portrait))
    if portrait:
        wf = gen.sdxl("RAW photo of a woman in a dark red wool coat in a park, overcast autumn day, 50mm, natural skin texture, film grain",
                      gen.NEGATIVE, 12, 30, 896, 1152,
                      False, f"{day}/selftest-control", pose="pending", face="pending", fix_faces=True, fix_hands=True, hires=True)
        ok, files, err = gen.generate(wf, 1800, log=log, uploads=[("30", portrait), ("21", portrait)])
        print(f"  pose + face + face/hand repair + tiled upscale: {'OK' if ok else 'FAILED'}  {', '.join(files) or '-'}"
              f"{'  error: ' + err if err else ''}")
        results.append(ok and len(files) == 4)          # pose skeleton, base image, repaired image, detailed 2x copy
    # newer models: one real job each (FLUX 3 only when its files are there - it is optional)
    def ready(name):
        p = gen.profile(name)
        roots = [os.path.join(gen.env("MODELS_MOUNT", "/models"), "comfy"), os.path.expanduser("~/.local/share/modules/comfyui/models")]
        files = [("diffusion_models", p["file"]), ("text_encoders", p["text_encoder"]), ("vae", p["vae"])]
        return all(any(os.path.exists(os.path.join(r, d, f)) for r in roots) for d, f in files)
    newer = [("Z-Image Turbo", "zimage", lambda p: gen.zimage("RAW photo of an old fisherman mending a net on a pier, overcast, 35mm",
                                                              5, p["steps"], 1024, 1024, f"{day}/selftest-zimage", p)),
             ("Anima (+ pose from the portrait)", "anima",
              lambda p: gen.anima("1girl, solo, standing, red coat, autumn park, detailed illustration", p["negative"]["art"], 6,
                                  p["steps"], 896, 1152, f"{day}/selftest-anima", p, pose="pending" if portrait else None))]
    if ready("flux3"):
        newer.append(("FLUX 3 Klein (caption expanded by the local model)", "flux3",
                      lambda p: gen.flux3(gen.expand_prompt("a cozy bookshop window at dusk with a sign reading \"OPEN\"", "t2i",
                                                            log=log), 7, p["steps"], 1024, 1024, f"{day}/selftest-flux3", p)))
    else:
        print("  FLUX 3 Klein: not installed (optional) - skipped")
    for label, name, build in newer:
        if name != "flux3" and not ready(name):
            print(f"  {label}: FAILED  model files missing"); results.append(False); continue
        p = gen.profile(name)
        ups = [("30", portrait)] if name == "anima" and portrait else []
        ok, files, err = gen.generate(build(p), 1200, log=log, uploads=ups)
        if name == "flux3" and not gen.EXPANDED["ok"]:
            ok, err = False, (err + "; " if err else "") + "prompt expansion did not work (see CHECK above)"
        print(f"  {label}: {'OK' if ok else 'FAILED'}  {', '.join(files) or '-'}{'  error: ' + err if err else ''}")
        results.append(ok)
    if ready("flux3") and portrait:
        p = gen.profile("flux3")
        cap = gen.expand_prompt("put this woman in a yellow raincoat on a rainy city street, keep her face", "edit", images=[portrait], log=log)
        ok, files, err = gen.generate(gen.flux3(cap, 8, p["steps"], 1024, 1024, f"{day}/selftest-flux3-edit", p, refs=["pending"]),
                                      1200, log=log, uploads=[("200", portrait)])
        if not gen.EXPANDED["ok"]:
            ok, err = False, (err + "; " if err else "") + "prompt expansion did not work (see CHECK above)"
        print(f"  FLUX 3 edit (reference image): {'OK' if ok else 'FAILED'}  {', '.join(files) or '-'}{'  error: ' + err if err else ''}")
        results.append(ok)
    # ai gen pose: a pose from a description (photo + skeleton), then that skeleton reused directly with --pose
    ok, files, err = gen.generate(gen.pose_wf("seen from behind, standing, both hands on the hips", 21, 896, 1152,
                                              f"{day}/selftest-posegen"), 900, log=log)
    skel = next((f for f in files if gen.is_skeleton(f)), None)
    print(f"  pose from a description (photo + skeleton): {'OK' if ok and skel and len(files) == 2 else 'FAILED'}  "
          f"{', '.join(files) or '-'}{'  error: ' + err if err else ''}")
    results.append(ok and bool(skel) and len(files) == 2)
    if skel:
        wf = gen.use_skeleton(gen.sdxl("RAW photo of a man in a grey suit seen from behind, hands on his hips, city rooftop, 50mm",
                                       gen.NEGATIVE, 22, 30, 896, 1152, False, f"{day}/selftest-skeleton", pose="pending"))
        ok, files, err = gen.generate(wf, 900, log=log, uploads=[("30", skel)])
        print(f"  image from a saved skeleton (--pose skeleton): {'OK' if ok and files else 'FAILED'}  {', '.join(files) or '-'}"
              f"{'  error: ' + err if err else ''}")
        results.append(ok and len(files) == 1)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
