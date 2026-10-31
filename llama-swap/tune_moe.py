#!/usr/bin/env python3
"""!
@file tune_moe.py
@brief Finds the fastest expert-offload value (--n-cpu-moe) for the MoE model on this GPU.

Fewer expert layers on the CPU is faster, until VRAM runs out (the server then fails to start or slows
down). Tries the candidates listed in models.json ('moe_tune'), writes the best to $MOD/moe.conf and
rewrites the llama-swap config. Run as root with the model server stopped.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

AI = os.environ.get("MOD", os.path.expanduser("~/.local/share/modules/llama-swap"))
PORT = 9099  # private port, so a running model server is not disturbed


def tokens_per_second(model_file, args, n):
    """!
    @brief Starts llama-server with one offload value and measures generation speed.
    @param model_file  GGUF file of the MoE model.
    @param args        The model's llama-server args from models.json ('{moe}' placeholder included).
    @param n           Offload value to test.
    @return Tokens per second, or None if the server did not start (out of VRAM).
    """
    cmd = [f"{AI}/app/bin/llama-server", "--host", "127.0.0.1", "--port", str(PORT), "-m", model_file,
           "--jinja", "-ngl", "99"] + args.replace("{moe}", str(n)).split()
    with open(f"{AI}/tune-last.log", "w") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(300):
                if p.poll() is not None:
                    return None
                try:
                    if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                        break
                except Exception:
                    time.sleep(2)
            else:
                return None
            body = json.dumps({"messages": [{"role": "user", "content": "Write 250 words about rivers."}],
                               "max_tokens": 300, "chat_template_kwargs": {"enable_thinking": False}}).encode()
            r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                                              {"Content-Type": "application/json"}), timeout=600)
            return json.load(r).get("timings", {}).get("predicted_per_second")
        finally:
            p.terminate()
            p.wait()


def main():
    """!
    @brief Tunes every model that has 'moe_tune' candidates (normally one) and stores the winner.
    """
    cfg = json.load(open(f"{AI}/models.json"))
    files = json.load(open(f"{AI}/models/models.json"))
    spec = next((s for s in cfg["chat"] if s.get("moe_tune") and s["id"] in files), None)
    if not spec:
        sys.exit("No MoE model with 'moe_tune' candidates is installed.")
    best_n, best_tps = None, 0.0
    for n in spec["moe_tune"]:
        tps = tokens_per_second(files[spec["id"]]["file"], spec["args"], n)
        print(f"--n-cpu-moe {n}: {'failed (see $MOD/tune-last.log)' if tps is None else f'{tps:.1f} tokens/s'}", flush=True)
        if tps is None or (best_tps and tps < best_tps * 0.7):
            break  # out of VRAM or spilling: stop
        if tps > best_tps:
            best_n, best_tps = n, tps
    if best_n is None:
        sys.exit("No working value found - check the log.")
    open(f"{AI}/moe.conf", "w").write(str(best_n))
    print(f"Best: --n-cpu-moe {best_n} at {best_tps:.1f} tokens/s")
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_config.py")], stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
