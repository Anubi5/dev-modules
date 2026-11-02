#!/usr/bin/env python3
"""!
@file img_ask.py
@brief "img-ask <image> "<question>"": asks the local vision-check model (WeVisOne-VL 2B behind llama-swap) about an
       image and prints its short answer. Agents use it to check renders and images by what they show, not only by
       pixel numbers (img-check). Exit 0 = answered, 1 = could not ask (server, model or file problem), 2 = usage.
       --expect WORD: exit 3 when the answer does not contain WORD (e.g. img-ask render.png "Is there a cube?" --expect yes).
"""
import base64
import json
import os
import sys
import urllib.request


def port():
    """! @brief LLM_PORT from sandbox.env (default 8080). """
    try:
        for line in open("/opt/ai/setup/sandbox.env"):
            if line.startswith("LLM_PORT="):
                return line.split("=", 1)[1].split("#")[0].strip().strip('"')
    except OSError:
        pass
    return "8080"


IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
WORK = os.path.realpath(os.path.expanduser("~/work"))


def check_image(path):
    """!
    @brief Same rule as ai gen: a picture inside ~/work (after following links), .png/.jpg/.webp, at most 30 MB.
    @return (real path, None) or (None, reason).
    """
    real = os.path.realpath(os.path.expanduser(path))
    if not (real.startswith(WORK + os.sep) and os.path.isfile(real)):
        return None, f"{path}: must be an existing file inside ~/work"
    if os.path.splitext(real)[1].lower() not in IMAGE_TYPES or os.path.getsize(real) > 30 << 20:
        return None, f"{path}: must be a .png/.jpg/.webp image of at most 30 MB"
    return real, None


def ask(image, question, model="vision-check", timeout=300):
    """!
    @brief Sends one image and question to the model server.
    @return The answer text.
    """
    mime = IMAGE_TYPES[os.path.splitext(image)[1].lower()]
    data = base64.b64encode(open(image, "rb").read()).decode()
    body = json.dumps({"model": model, "max_tokens": 200, "temperature": 0,
                       "messages": [{"role": "user", "content": [
                           {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                           {"type": "text", "text": question + " Answer briefly."}]}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port()}/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return (json.load(r)["choices"][0]["message"].get("content") or "").strip()


def main(argv):
    """! @brief Command-line entry point. @return exit code """
    expect = None
    if "--expect" in argv:
        i = argv.index("--expect")
        if i + 1 >= len(argv):
            print('usage: img-ask <image> "<question>" [--expect WORD]'); return 2
        expect = argv[i + 1].lower()
        argv = argv[:i] + argv[i + 2:]
    if len(argv) != 2:
        print('usage: img-ask <image> "<question>" [--expect WORD]'); return 2
    image, question = argv
    image, why = check_image(image)
    if why:
        print(f"img-ask: {why}"); return 1
    try:
        answer = ask(image, question)
    except Exception as e:
        print(f"img-ask: could not ask the vision-check model ({e}) - is llama-swap running? (ai start llama-swap)"); return 1
    print(f"img-ask: {os.path.basename(image)}: {question} -> {answer}")
    if expect is not None and expect not in answer.lower():
        print(f"img-ask: expected '{expect}' in the answer - NOT confirmed"); return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
