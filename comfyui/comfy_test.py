#!/usr/bin/env python3
"""!
@file comfy_test.py
@brief End-to-end ComfyUI test through the GPU hand-off (gen.py): SDXL image + 4x upscale, a short Wan 2.2 video, then
       the control features on a generated portrait: its pose and face reused, faces/hands repaired, detailed upscale.

Uses the same fixed workflows, lock, unload/free and output folder as "ai gen", so it tests exactly what Hermes uses.
Outputs go to COMFY_OUTPUT/<date>/selftest-*. Exit code 0 only if every job produced its files there.
"""
import datetime
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
        results.append(ok and len(files) == 3)          # pose skeleton, image, detailed 2x copy
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
