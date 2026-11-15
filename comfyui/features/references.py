"""--image for edit: reference pictures, uploaded to nodes 200, 210, 220, 230 (first = the picture to change)."""


def load(wf, refs):
    """! @brief One LoadImage per reference. @return their image links, in order """
    links = []
    for i, name in enumerate(refs):
        wf[f"2{i}0"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        links.append([f"2{i}0", 0])
    return links
