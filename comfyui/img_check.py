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


def main():
    """! @brief Checks every file given; prints one line each. @return exit code """
    bad = 0
    for f in sys.argv[1:] or ["--help"]:
        if f in ("-h", "--help"):
            print(__doc__.split("@brief")[1].split("\n\"")[0].strip()); return 0
        try:
            a = np.asarray(Image.open(f).convert("RGB")).astype(int)
        except Exception as e:
            print(f"{f}: UNREADABLE ({e})"); bad += 1; continue
        std = float(a.std())
        blank = std < 5
        cs = ", ".join(f"{n} {s:.0%}" for n, s in colours(a))
        print(f"{f}: {a.shape[1]}x{a.shape[0]}, contrast {std:.0f}, colours: {cs}"
              + ("  -> BLANK (one flat colour - nothing visible was rendered)" if blank else ""))
        bad += blank
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
