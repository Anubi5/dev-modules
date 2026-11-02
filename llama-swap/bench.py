#!/usr/bin/env python3
"""!
@file bench.py
@brief Answers, speed and vision check for every model behind llama-swap; writes a Markdown report.

Models flagged 'speed_test_only' (autocomplete models) only get the speed test.
Models flagged 'long_prompt_test' in models.json also get a ~12K-token prompt (agents are prompt-heavy);
models flagged 'vision' must name the colour of a red test image ('vision_only': only that test).
Models flagged 'embedding' get a search test instead (a related sentence must be closer than an unrelated one).
Models flagged 'candidate' (being evaluated) are reported like the others but never fail the run.
Exit code 1 if a configured model is missing, any model errors or gives a wrong answer (vision included), or nothing was tested.

Usage: bench.py [model-id ...]   (no ids = all models)
"""
import base64
import re
import datetime
import json
import os
import struct
import sys
import time
import urllib.request
import zlib

SETUP = "/opt/ai/setup"
MOD = os.environ.get("MOD", os.path.expanduser("~/.local/share/modules/llama-swap"))
OUT = f"{MOD}/bench-{datetime.datetime.now():%Y%m%d-%H%M}.md"
TESTS = [
    ("arithmetic", "What is 17 * 23? Answer with the number only.", lambda t: re.fullmatch(r"\W*391\W*", t.strip()) is not None),
    ("instruction", "List exactly three fruits, comma separated, nothing else.", lambda t: t.count(",") == 2),
    ("speed", "Write a 200-word paragraph about the history of the bicycle.", lambda t: len(t.split()) > 120),
]


def env(key, default):
    """!
    @brief Reads one KEY="value" setting from sandbox.env.
    @param key      Setting name.
    @param default  Value if the file or key is missing.
    @return The value as a string.
    """
    try:
        for line in open(f"{SETUP}/sandbox.env"):
            if line.startswith(key + "="):
                return line.split("=", 1)[1].split("#")[0].strip().strip('"')
    except OSError:
        pass
    return default


API = f"http://127.0.0.1:{env('LLM_PORT', '8080')}"


def chat(model, content, max_tokens=400, timeout=900):
    """!
    @brief Sends one chat request (thinking off) to llama-swap.
    @param model       Model id.
    @param content     Message content (text or a list of content parts).
    @param max_tokens  Answer length limit.
    @param timeout     Seconds to wait (includes model loading).
    @return (response JSON, wall-clock seconds).
    """
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "temperature": 0.2,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.time()
    req = urllib.request.Request(f"{API}/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.time() - t0


def embed(model, texts, timeout=300):
    """!
    @brief Embeds texts with an embedding model behind llama-swap (OpenAI-style /v1/embeddings).
    @return (list of vectors, wall-clock seconds).
    """
    body = json.dumps({"model": model, "input": texts}).encode()
    t0 = time.time()
    req = urllib.request.Request(f"{API}/v1/embeddings", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return [d["embedding"] for d in sorted(json.load(r)["data"], key=lambda d: d["index"])], time.time() - t0


def cosine(a, b):
    """! @brief Cosine similarity of two vectors. """
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5) or 1)


def red_png():
    """!
    @brief Builds a 64x64 solid red PNG without extra libraries.
    @return The PNG as base64 text.
    """
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * 64 for _ in range(64))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()


def run(models, specs):
    """!
    @brief Runs all tests and returns the report rows.
    @param models  Model ids to test.
    @param specs   models.json "chat" entries by id (for the long-prompt and vision flags).
    @return List of Markdown table rows.
    """
    rows = ["| Model | Test | Pass | Tokens/s | Seconds (incl. load) |", "| --- | --- | --- | --- | --- |"]

    cand = {"on": False}

    def add(row):
        if cand["on"]:                                   # candidates: marked, never fail the run
            row = row.replace(" |", " (candidate) |", 1)
        rows.append(row)
        print(row, flush=True)

    for m in models:
        spec = specs.get(m, {})
        cand["on"] = bool(spec.get("candidate"))
        if spec.get("embedding"):
            try:
                q, near, far = "How do I render a Blender scene to a PNG file?", \
                    "Save the rendered Blender image as a PNG picture.", "The recipe needs two eggs and a cup of flour."
                v, wall = embed(m, [q, near, far])
                ok = cosine(v[0], v[1]) > cosine(v[0], v[2])
                add(f"| {m} | search (related closer than unrelated, {len(v[0])} dims) | {'yes' if ok else 'NO'} | - | {wall:.1f} |")
            except Exception as e:
                add(f"| {m} | search | ERROR | - | {str(e)[:60]} |")
            continue
        tests = [t for t in TESTS if t[0] == "speed"] if spec.get("speed_test_only") else TESTS   # e.g. autocomplete models
        if spec.get("vision_only"):                     # image-checking models: only the image test means anything
            tests = []
        for name, prompt, check in tests:
            try:
                d, wall = chat(m, prompt)
                text = d["choices"][0]["message"].get("content") or ""
                tps = (d.get("timings") or {}).get("predicted_per_second") or 0
                add(f"| {m} | {name} | {'yes' if check(text) else 'NO'} | {tps:.1f} | {wall:.1f} |")
            except Exception as e:
                add(f"| {m} | {name} | ERROR | - | {str(e)[:60]} |")
        if spec.get("long_prompt_test"):
            text = "The quick brown fox jumps over the lazy dog while the sun sets behind the hills. " * 700
            try:
                d, wall = chat(m, text + "\nHow many times does the word fox appear? Answer briefly.", 20, 1800)
                t = d.get("timings") or {}
                add(f"| {m} | long prompt ({t.get('prompt_n', '?')} tokens) | prefill {t.get('prompt_per_second', 0):.0f} tok/s"
                    f" | {t.get('predicted_per_second', 0):.1f} | {wall:.1f} |")
            except Exception as e:
                add(f"| {m} | long prompt | ERROR | - | {str(e)[:60]} |")
        if spec.get("vision"):
            try:
                d, _ = chat(m, [{"type": "text", "text": "What colour is this image? One word."},
                                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + red_png()}}], 20)
                ans = d["choices"][0]["message"].get("content") or ""
                add(f"| {m} | vision (red image) | {'yes' if 'red' in ans.lower() else 'NO'} | - | {ans.strip()[:30]} |")
            except Exception as e:
                add(f"| {m} | vision | ERROR | - | {str(e)[:60]} |")
    return rows


def main():
    """!
    @brief Command-line entry point: tests, writes the report, sets the exit code.
    """
    specs = {s["id"]: s for s in json.load(open(f"{MOD}/models.json"))["chat"]}
    models = [m["id"] for m in json.load(urllib.request.urlopen(f"{API}/v1/models"))["data"]]
    wanted = sys.argv[1:] or list(specs)
    missing = [m for m in wanted if m not in models]   # every configured (or requested) model must be served
    rows = run([m for m in models if m in wanted], specs)
    rows += [f"| {m} | present on the server | ERROR | - | not in /v1/models |" for m in missing]
    open(OUT, "w").write(f"# Local AI benchmark {datetime.datetime.now():%Y-%m-%d %H:%M}\n\n" + "\n".join(rows) + "\n")
    print(f"\nReport: {OUT}")
    tested = len(rows) - 2   # minus the table header
    sys.exit(1 if tested == 0 or any(("| ERROR |" in r or "| NO |" in r) and "(candidate) |" not in r for r in rows) else 0)


if __name__ == "__main__":
    main()
