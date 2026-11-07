"""Orientation checks for the limited UniRig heel-ended reference skeleton.

This backend's terminal foot segment points toward the heel, not the toes.
The declared input facing must agree with both segments; ambiguous rigs fail.
"""
import math
from collections.abc import Sequence


FACING = {"-y": (0., -1., 0.), "+y": (0., 1., 0.),
          "-x": (-1., 0., 0.), "+x": (1., 0., 0.)}
Vector3 = Sequence[float]


def unit_xy(vector: Vector3) -> tuple[float, float, float]:
    """Return a finite horizontal unit vector, refusing collapsed evidence."""
    if len(vector) != 3 or not all(math.isfinite(v) for v in vector):
        raise ValueError("Orientation evidence contains non-finite coordinates")
    length = math.hypot(vector[0], vector[1])
    if length <= 1e-8:
        raise ValueError("Ambiguous orientation: missing horizontal foot/hip separation")
    return (vector[0] / length, vector[1] / length, 0.)


def verify_facing(heel_segments: Sequence[Vector3], hips: Sequence[Vector3],
                  determinants: Sequence[float], facing: str = "-y",
                  min_alignment: float = .9) -> dict:
    """Verify declared facing and return the yaw that normalizes it to -Y.

    Coordinates are Blender world space, Z up. The .9 alignment is the existing
    humanoid contract threshold. No claim is made for toe-ended skeletons.
    """
    if facing not in FACING or not .9 <= min_alignment <= 1:
        raise ValueError("Unsupported facing/alignment; use --facing=-y, +x, +y or -x")
    if not determinants or any(not math.isfinite(d) or d <= 0 for d in determinants):
        raise ValueError("Mirrored or singular transforms are unsupported; fix the source asset")
    if len(heel_segments) != 2 or len(hips) != 2:
        raise ValueError("Orientation needs exactly two heel-ended leg chains")
    forward = FACING[facing]
    measured = [tuple(-v for v in unit_xy(segment)) for segment in heel_segments]
    alignments = [sum(a*b for a, b in zip(axis, forward)) for axis in measured]
    if min(alignments) < min_alignment:
        raise ValueError("Declared facing disagrees with the UniRig heel reference; "
                         "verify the asset and set --facing explicitly, or reject this skeleton")
    lateral = unit_xy(tuple(b-a for a, b in zip(hips[0], hips[1])))
    right = (forward[1], -forward[0], 0.)
    alignment = abs(sum(a*b for a, b in zip(lateral, right)))
    if alignment < min_alignment:
        raise ValueError("Ambiguous left/right: hip separation disagrees with declared facing")
    return dict(reference="unirig-heel-v1", input_facing=facing,
                measured_forward_axes=measured, facing_alignments=alignments,
                lateral_alignment=alignment, minimum_alignment=min_alignment,
                yaw_radians=-math.pi/2-math.atan2(forward[1], forward[0]),
                output_facing=[0., -1., 0.])


def verify_named_sides(positions: dict[str, Vector3]) -> None:
    """Verify semantic labels in canonical Blender -Y-facing coordinates."""
    centre = positions["pelvis"][0]
    for stem in ("hand", "thigh"):
        left, right = positions[stem+"_l"], positions[stem+"_r"]
        if not all(math.isfinite(v) for p in (left, right) for v in p):
            raise ValueError("Non-finite named bone positions")
        if not right[0] < centre < left[0]:
            raise ValueError("Semantic left/right labels disagree with the canonical facing")
