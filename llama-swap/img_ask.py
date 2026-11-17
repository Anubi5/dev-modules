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
import errno
import re
import sys
import urllib.error
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


class InvalidAnswer(ValueError):
    """The backend replied, but its text cannot be used as verification evidence."""


def answer_text(response, question):
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
        reason = choice["finish_reason"]
    except (KeyError, IndexError, TypeError):
        raise InvalidAnswer("malformed inference response") from None
    if reason != "stop":
        raise InvalidAnswer("incomplete inference response (" + str(reason) + ")")
    if not isinstance(content, str) or not content.strip():
        raise InvalidAnswer("empty answer")
    text = content.strip()
    # WeVisOne may return one grounded caption. Strip only its complete known
    # envelope; never turn arbitrary markup or a partial generation into a pass.
    grounded = re.fullmatch(r"<ref>([^<>]+)</ref><box>(?:<\d{1,4}>){4}</box>", text)
    if grounded:
        text = grounded[1].strip()
    if not text or len(text) > 600 or any(c in text for c in "<>{}"):
        raise InvalidAnswer("empty, oversized or malformed answer")
    words = re.findall(r"\w+", text.lower())
    asked = re.findall(r"\w+", question.lower())
    if not words or len(words) > 100 or words in (asked, asked + ['answer', 'briefly']):
        raise InvalidAnswer("empty, oversized or echoed question")
    for width in range(1, min(8, len(words)//3) + 1):
        for start in range(len(words) - width*3 + 1):
            if words[start:start+width] == words[start+width:start+2*width] == words[start+2*width:start+3*width]:
                raise InvalidAnswer("repeated tokens in answer")
    return text


def ask(image, question, model="vision-check", timeout=300):
    """!
    @brief Sends one image and question to the model server.
    @return The answer text.
    """
    mime = IMAGE_TYPES[os.path.splitext(image)[1].lower()]
    with open(image, "rb") as stream:
        data = base64.b64encode(stream.read()).decode()
    body = json.dumps({"model": model, "max_tokens": 128, "temperature": 0,
                       "messages": [{"role": "user", "content": [
                           {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                           {"type": "text", "text": question + " Answer briefly."}]}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port()}/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = r.read(65537)
        if len(payload) > 65536:
            raise InvalidAnswer("oversized inference response")
        try:
            response = json.loads(payload)
        except (ValueError, UnicodeError):
            raise InvalidAnswer("malformed inference response") from None
        return answer_text(response, question)


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
    except InvalidAnswer as error:
        print(f"img-ask: vision-check returned unusable output ({error}); NOT verified"); return 1
    except urllib.error.URLError as error:
        reason = error.reason
        if isinstance(reason, ConnectionRefusedError) or getattr(reason, "errno", None) == errno.ECONNREFUSED:
            print("img-ask: llama-swap is not listening; NOT verified. Start the existing service with: ai start llama-swap")
        else:
            print(f"img-ask: vision-check request failed ({error}); NOT verified")
        return 1
    except (OSError, ValueError) as error:
        print(f"img-ask: could not ask vision-check ({error}); NOT verified"); return 1
    print(f"img-ask: {os.path.basename(image)}: {question} -> {answer}")
    if expect is not None and expect not in answer.lower():
        print(f"img-ask: expected '{expect}' in the answer - NOT confirmed"); return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
