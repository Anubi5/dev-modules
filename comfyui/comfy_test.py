#!/usr/bin/env python3
"""!
@file comfy_test.py
@brief End-to-end ComfyUI test through the GPU hand-off (gen.py): SDXL image + 4x upscale, then a short Wan 2.2 video.

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
                                            "blurry, low quality, watermark", 42, 25, 1024, 1024, True, f"{day}/selftest-sdxl"), 900),
        ("Wan 2.2 5B video (GGUF, 33 frames)", gen.wan("a red fox running through fresh snow in a forest, cinematic, smooth camera tracking",
                                                      "static, blurry, distorted, low quality", 7, 20, 33, f"{day}/selftest-wan"), 1800),
    ]
    results = []
    for name, wf, limit in jobs:
        ok, files, err = gen.generate(wf, limit, log=lambda s: print(f"    {s}", flush=True))
        print(f"  {name}: {'OK' if ok else 'FAILED'}  {', '.join(files) or '-'}{'  error: ' + err if err else ''}")
        results.append(ok and all(f.startswith(gen.OUTPUT + "/") for f in files))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
