"""Spatial pixel-content validation, independent of Blender and NumPy."""
import math
from collections.abc import Iterable, Sequence


def render_metrics(pixels: Iterable[Sequence[float]]) -> dict:
    """Measure visible coverage and maximum per-channel spatial standard deviation.

    Input is normalized RGBA. Existing alpha (.05), coverage (.02), and variation
    (.005) thresholds are retained. A difference between red/green/blue values at
    one pixel is not spatial detail. Invalid or empty pixels fail closed.
    """
    total = visible = 0
    means = [0., 0., 0.]
    squared = [0., 0., 0.]
    for row in pixels:
        if len(row) != 4 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in row):
            return dict(coverage=0., rgb_std=0., ok=False)
        total += 1
        if row[3] <= .05:
            continue
        visible += 1
        for channel in range(3):
            value = float(row[channel])
            delta = value - means[channel]
            means[channel] += delta / visible
            squared[channel] += delta * (value - means[channel])
    coverage = visible / total if total else 0.
    spatial_std = math.sqrt(max(0., max(squared)) / visible) if visible else 0.
    return dict(coverage=coverage, rgb_std=spatial_std,
                ok=coverage > .02 and spatial_std > .005)
