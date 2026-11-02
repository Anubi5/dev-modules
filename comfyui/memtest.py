#!/usr/bin/env python3
"""!
@file memtest.py
@brief Before/after test for ComfyUI's memory setting: runs the job that failed with "HostBuffer.read_file_slice failed"
       (SDXL + pose + face + face/hand repair + tiled upscale - more than the 12 GB GPU holds) N times with pinned memory
       ON (ComfyUI's default) and N times OFF (--disable-pinned-memory, this module's default), and prints failures and
       times. ComfyUI is restarted for each setting; afterwards the module default is back.
Usage: python3 ~/.local/share/modules/comfyui/memtest.py [runs per setting, default 3]
"""
import datetime
import os
import subprocess
import sys
import time

import gen

OVR = os.path.expanduser("~/.config/comfyui/override.env")


def restart(mem):
    """! @brief Restarts ComfyUI with COMFY_MEM=mem (None = module default) and waits until it answers. """
    if mem is None:
        if os.path.exists(OVR):
            os.remove(OVR)
    else:
        os.makedirs(os.path.dirname(OVR), exist_ok=True)
        open(OVR, "w").write(f"COMFY_MEM={mem}\n")
    subprocess.run(["systemctl", "--user", "restart", "comfyui"], check=False)
    for _ in range(90):
        if gen.comfy_up():
            return True
        time.sleep(2)
    return False


def main():
    """! @brief Runs both settings and prints the comparison. @return 0 if the module default had no failure """
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    day = f"{datetime.date.today():%Y-%m-%d}"
    log = lambda s: None
    results = {}
    try:
        for label, mem in (("pinned memory ON (ComfyUI default)", ""), ("pinned memory OFF (module default)", "--disable-pinned-memory")):
            print(f"== {label}", flush=True)
            if not restart(mem):
                print("  ComfyUI did not start"); results[label] = (n, n, []); continue
            ok, files, err = gen.generate(gen.sdxl("RAW photo, portrait of a woman standing, front view, face clearly visible, arms relaxed, soft window light, 85mm",
                                                   gen.NEGATIVE, 11, 30, 896, 1152, False, f"{day}/memtest-ref"), 900, log=log)
            portrait = files[0] if ok and files else None
            if not portrait:
                print(f"  reference portrait failed: {err}"); results[label] = (n, n, []); continue
            fails, times = 0, []
            for i in range(n):
                wf = gen.sdxl("RAW photo of a woman in a dark red wool coat in a park, overcast autumn day, 50mm", gen.NEGATIVE, 12 + i,
                              30, 896, 1152, False, f"{day}/memtest-control", pose="pending", face="pending",
                              fix_faces=True, fix_hands=True, hires=True)
                t0 = time.time()
                ok, files, err = gen.generate(wf, 1800, log=log, uploads=[("30", portrait), ("21", portrait)])
                dt = time.time() - t0
                ok = ok and len(files) == 4
                fails += 0 if ok else 1
                times.append(dt)
                print(f"  run {i + 1}: {'OK' if ok else 'FAILED'} in {dt:.0f} s{'  (' + err[:80] + ')' if err and not ok else ''}", flush=True)
            results[label] = (fails, n, times)
    finally:
        restart(None)                                   # back to the module default
    print("\n| setting | failures | average time |\n|---|---|---|")
    for label, (fails, n, times) in results.items():
        print(f"| {label} | {fails} of {n} | {sum(times) / len(times):.0f} s |" if times else f"| {label} | {fails} of {n} | - |")
    off = results.get("pinned memory OFF (module default)", (1, 1, []))
    return 0 if off[0] == 0 and off[2] else 1


if __name__ == "__main__":
    sys.exit(main())
