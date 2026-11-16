"""!
@file edit.py
@brief ai gen edit: change a picture by instruction (up to 4 references) with an edit-capable model; the rest of the
       job is the image operation's.
"""
import sys

from generation.models import PROFILES
from generation import builders
from operations import image


OPTIONS = {'model', 'image', 'steps', 'cfg', 'upscale', 'negative', 'no_expand', 'force_lora', 'lora', 'improve_prompt'}


def before(a):
    """! @brief Edit's own inputs, checked before the model is resolved; the default edit model when none is given. """
    if not a.image or len(a.image) > 4:
        sys.exit("edit needs --image <file> (up to 4 with extra references)")
    a.model = a.model or PROFILES["edit_model"]


def check(capability):
    """! @brief The resolved model must be able to edit. """
    if not capability["edit"]:
        available=[]
        for name,prof in PROFILES['models'].items():
            if prof.get('kind','image')!='image':continue
            if builders.capabilities(dict(prof,name=name))['edit']:available.append(name)
        sys.exit('edit needs a family declaring editing support (--model ' + ' | '.join(available) + ')')


def prepare(a, seed, prefix):
    """! @return (workflow, uploads, info, time limit, profile), as image.prepare """
    return image.prepare(a, seed, prefix, edit=sys.modules[__name__])
