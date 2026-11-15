"""!
@file edit.py
@brief ai gen edit: change a picture by instruction (up to 4 references) with an edit-capable model; the rest of the
       job is the image operation's.
"""
import sys

from generation.models import PROFILES
from operations import image


def before(a):
    """! @brief Edit's own inputs, checked before the model is resolved; the default edit model when none is given. """
    if not a.image or len(a.image) > 4:
        sys.exit("edit needs --image <file> (up to 4 with extra references)")
    a.model = a.model or PROFILES.get("edit_model", "flux3")


def check(capability):
    """! @brief The resolved model must be able to edit. """
    if not capability["edit"]:
        sys.exit("edit needs a FLUX 3 or Fibo 2.0 model (--model flux3 | fibo2)")


def prepare(a, seed, prefix):
    """! @return (workflow, uploads, info, time limit, profile), as image.prepare """
    return image.prepare(a, seed, prefix, edit=sys.modules[__name__])
