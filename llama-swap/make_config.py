#!/usr/bin/env python3
"""!
@file make_config.py
@brief Writes the llama-swap config ($MOD/llama-swap.yaml) from models.json and the downloaded files.

Every chat model that was downloaded gets an entry; nothing is preloaded (llama-swap loads a model on
first use and unloads it after 'ttl' idle seconds). '{moe}' in a model's args is replaced by the tuned
expert-offload value from $MOD/moe.conf (written by tune_moe.py), or by the model's 'moe_default'.
'fit': true lets llama.cpp place layers and experts between GPU and RAM itself (--fit on) instead of all on the GPU.

Usage: make_config.py [--config $MOD/models.json]   ($MOD = this module's folder)
"""
import argparse
import json
import os

AI = os.environ.get("MOD", os.path.expanduser("~/.local/share/modules/llama-swap"))


def moe_value(spec):
    """!
    @brief Returns the expert-offload value for a MoE model: tuned value if present, else its default.
    @param spec  One models.json "chat" entry.
    @return The value as a string.
    """
    path = f"{AI}/moe.conf"
    if os.path.exists(path):
        return open(path).read().strip()
    return str(spec.get("moe_default", 36))


def build(cfg, files):
    """!
    @brief Builds the llama-swap YAML text.
    @param cfg    Parsed models.json.
    @param files  Parsed $MOD/models/models.json (id -> {file, mmproj}).
    @return YAML as a string.
    """
    server = f"{AI}/app/bin/llama-server --host 127.0.0.1 --port ${{PORT}} --jinja"
    lines = ["healthCheckTimeout: 600", "logLevel: info", "models:"]
    by_id = {spec['id']: spec for spec in cfg['chat']}
    for spec in cfg["chat"]:
        if spec["id"] not in files:
            continue
        if spec.get('weights_from'):
            base = {key: value for key, value in by_id[spec['weights_from']].items() if key != 'alias'}
            spec = dict(base, **spec)  # Inherit hardware/vision settings, never steal the daily model's aliases.
        local = files[spec["id"]]
        place = "--fit on" if spec.get("fit") else "-ngl 99"   # fit: llama.cpp splits GPU/RAM itself (untuned models)
        cmd = f"{server} {place} -m {local['file']} {spec.get('args', '').replace('{moe}', moe_value(spec))}"
        if spec.get("vision") and local.get("mmproj"):
            cmd += f" --mmproj {local['mmproj']}"
        lines += [f'  "{spec["id"]}":', f'    name: "{spec["name"]}"', f"    cmd: {cmd}", f"    ttl: {spec.get('ttl', 300)}"]
        if spec.get("alias"):
            lines += ["    aliases:", f'      - "{spec["alias"]}"']
    return "\n".join(lines) + "\n"


def main():
    """!
    @brief Command-line entry point: writes the config and prints it.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=f"{AI}/models.json")
    a = ap.parse_args()
    yaml = build(json.load(open(a.config)), json.load(open(f"{AI}/models/models.json")))
    open(f"{AI}/llama-swap.yaml", "w").write(yaml)
    print(yaml)


if __name__ == "__main__":
    main()
