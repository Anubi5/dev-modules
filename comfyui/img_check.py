#!/usr/bin/env python3
"""!
@file img_check.py
@brief "img-check <image>...": facts about an image an agent can quote instead of guessing - size, contrast, main
       colours, and a clear BLANK verdict for flat/empty renders. Exit code 1 if any image is blank or unreadable.
"""
import sys

import numpy as np
from PIL import Image

NAMES = {"red": (200, 40, 40), "orange": (230, 140, 40), "yellow": (220, 210, 60), "green": (60, 170, 70),
         "cyan": (60, 190, 200), "blue": (50, 80, 200), "purple": (140, 60, 170), "pink": (230, 130, 170),
         "brown": (120, 80, 50), "white": (240, 240, 240), "light grey": (180, 180, 180), "grey": (120, 120, 120),
         "dark grey": (60, 60, 60), "black": (10, 10, 10)}


def colours(a):
    """! @brief Share of pixels nearest to each named colour. @param a HxWx3 int array @return [(name, share)] """
    px = a.reshape(-1, 3)[:: max(1, a.shape[0] * a.shape[1] // 40000)]
    ref = np.array(list(NAMES.values()))
    idx = np.argmin(((px[:, None, :] - ref[None]) ** 2).sum(-1), axis=1)
    share = np.bincount(idx, minlength=len(NAMES)) / len(idx)
    return sorted(((n, s) for n, s in zip(NAMES, share) if s >= 0.03), key=lambda x: -x[1])


def measure(img):
    """!
    @brief What is visible and how much it varies. Visible: with an alpha channel, the pixels that are not transparent;
           without, the pixels that differ clearly from the image's most common colour (its background). Contrast is
           the largest per-channel spread over the visible pixels, so one flat colour, of any hue, measures 0.
    @return (rgb HxWx3 int array, visible share 0-1, contrast, visible pixels Nx3)
    """
    rgba = np.asarray(img.convert("RGBA")).astype(int)
    a, has_alpha = rgba[..., :3], img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
    if has_alpha:
        mask = rgba[..., 3] >= 128
    else:
        q = (a // 16).reshape(-1, 3)
        keys, counts = np.unique(q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2], return_counts=True)
        k = keys[np.argmax(counts)]
        bg = np.array([k // 256, k // 16 % 16, k % 16]) * 16 + 8
        mask = (np.abs(a - bg) > 24).any(-1)
    vis = a[mask] if has_alpha else a.reshape(-1, 3)
    contrast = float(vis.std(axis=0).max()) if len(vis) else 0.0
    return a, float(mask.mean()), contrast, vis


def main():
    """! @brief Checks every file given; prints one line each. --min-coverage F: also fail an image whose visible
           content covers less than F (0-1) of it. @return exit code """
    args, cover = sys.argv[1:], 0.0
    if "--min-coverage" in args:
        i = args.index("--min-coverage")
        cover = float(args[i + 1]); del args[i:i + 2]
    bad = 0
    for f in args or ["--help"]:
        if f in ("-h", "--help"):
            print(__doc__.split("@brief")[1].split("\n\"")[0].strip()); return 0
        try:
            a, share, contrast, vis = measure(Image.open(f))
        except Exception as e:
            print(f"{f}: UNREADABLE ({e})"); bad += 1; continue
        blank = share < 0.005 or contrast < 5
        small = not blank and share < cover
        cs = ", ".join(f"{n} {s:.0%}" for n, s in colours(vis.reshape(-1, 1, 3))) if len(vis) else "none"
        print(f"{f}: {a.shape[1]}x{a.shape[0]}, content {share:.0%}, contrast {contrast:.0f}, colours: {cs}"
              + ("  -> BLANK (nothing visible, or one flat colour)" if blank else "")
              + (f"  -> TOO SMALL (content under {cover:.0%})" if small else ""))
        bad += blank or small
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
