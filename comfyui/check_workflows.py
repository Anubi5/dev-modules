#!/usr/bin/env python3
"""!
@file check_workflows.py
@brief Checks every ai gen workflow variant against schema.json (node input definitions captured from the pinned ComfyUI
       with "ai gen schema"): node classes exist, required inputs are set, no unknown inputs, links point to existing
       nodes, choice values (samplers, schedulers, modes ...) are allowed. Runs without a GPU or ComfyUI (CI).
@return exit code 0 when all variants and self-tests pass. A missing schema.json or a node class not in it is a
       warning locally and a failure in CI (CI set).
"""
import contextlib
import itertools
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen  # noqa: E402


def check(wf, schema, label, unknown):
    """! @brief Problems found in one workflow; node classes not in the schema yet are collected in unknown. @return list of text """
    bad = []
    for nid, node in wf.items():
        c = node["class_type"]
        if c not in schema:
            unknown.add(c); continue
        spec = schema[c]
        if spec is None:
            bad.append(f"{label}: node {nid} {c} is not known to the pinned ComfyUI"); continue
        for k, v in node["inputs"].items():
            base = k.split(".")[0]                      # dynamic inputs: images.image0, resolution.megapixels
            if base not in spec:
                bad.append(f"{label}: {c}.{k} is not an input of {c}")
            elif isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                if v[0] not in wf:
                    bad.append(f"{label}: {c}.{k} links to missing node {v[0]}")
            elif k == base and spec[k]["values"] is not None and v not in spec[k]["values"]:
                bad.append(f"{label}: {c}.{k} = {v!r} is not one of {spec[k]['values'][:8]}")
        for k, d in spec.items():
            if d["type"] == "COMFY_AUTOGROW_V3":         # a growing list (images.image0 ...): zero entries is allowed
                continue
            if d["part"] == "required" and k not in node["inputs"] and not any(i.startswith(k + ".") for i in node["inputs"]):
                bad.append(f"{label}: {c} is missing required input {k}")
    return bad


def self_tests():
    """!
    @brief Checks of ai gen's own logic that need no ComfyUI: an SDXL checkpoint without a profile gets the SDXL default
           settings (not the overall default, which may be another architecture); the preflight finds a file ComfyUI
           does not list. @return list of problems
    """
    import struct
    import tempfile
    bad = []
    with tempfile.TemporaryDirectory() as d:
        head = json.dumps({"conditioner.embedders.1.model.x": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
        os.makedirs(os.path.join(d, "comfy", "checkpoints"))
        ck = os.path.join(d, "comfy", "checkpoints", "myOwnSDXL.safetensors")
        with open(ck, "wb") as f:
            f.write(struct.pack("<Q", len(head)) + head + b"\0\0")
        roots, gen.assets.ROOTS[:] = gen.assets.ROOTS[:], [os.path.join(d, "comfy")]
        saved = (gen.assets.NOTES, gen.assets.CACHE)
        gen.assets.NOTES, gen.assets.CACHE = os.path.join(d, "n.json"), os.path.join(d, "c.json")
        try:
            quiet = lambda t: None
            p = gen.find_checkpoint("myown", quiet)
            if p["arch"] != "sdxl" or p["file"] != "myOwnSDXL.safetensors":
                bad.append(f"self-test: --checkpoint with an unprofiled SDXL file got arch {p['arch']} (needs sdxl)")
            gen.assets.set_note(ck,
                                {"sampler": "euler", "scheduler": "sgm_uniform", "steps": "8", "cfg": "2", "negative": ""})
            p = gen.find_checkpoint("myown", quiet)
            if (p["sampler"], p["scheduler"], p["steps"], p["cfg"], gen.negative_for(p, "photo")) != ("euler", "sgm_uniform", 8, 2.0, ""):
                bad.append(f"self-test: checkpoint note not applied: {p['sampler']} {p['scheduler']} {p['steps']} {p['cfg']}")
        finally:
            gen.assets.ROOTS[:] = roots
            gen.assets.NOTES, gen.assets.CACHE = saved
    api = gen.api
    gen.api = lambda path, *a, **k: {"UNETLoader": {"input": {"required": {"unet_name": [["have.safetensors"]]}}}}
    try:
        wf = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "missing.safetensors"}}}
        ok = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "have.safetensors"}}}
        if gen.missing_files(wf) != ["UNETLoader.unet_name = missing.safetensors"] or gen.missing_files(ok):
            bad.append("self-test: the model-file preflight did not tell a missing file from a present one")
    finally:
        gen.api = api
    bad += lora_rule_tests() + model_rule_tests() + recipe_tests() + outcome_tests() + failure_tests()
    return bad


def outcome_tests():
    """!
    @brief Outcomes: a 3D model only counts as succeeded when every required check passed (Blender exit code, its result,
           4 renders, img-check); a failed job without outputs still gets a record; rerun refuses a workflow that
           does not match its record. @return list of problems
    """
    import tempfile
    bad, good, mesh = [], {"valid": True, "triangles": 10}, {"ok": True}
    cases = [(({"valid": False}, True, 0, mesh, 4, 0), "failed"), ((good, False), "check_skipped"),
             ((good, True, 1, mesh, 4, 0), "check_failed"), ((good, True, 0, None, 4, 0), "check_failed"),
             ((good, True, 0, {"ok": False}, 4, 0), "check_failed"), ((good, True, 0, mesh, 3, 0), "check_failed"),
             ((good, True, 0, mesh, 4, 1), "check_failed"), ((good, True, 0, mesh, 4, 0), "succeeded")]
    for args, want in cases:
        if gen.check_status(*args) != want:
            bad.append(f"self-test: 3D verdict {args[1:]} gave {gen.check_status(*args)}, needs {want}")
    if [gen.EXIT[s] for s in ("succeeded", "failed", "check_failed", "check_skipped")] != [0, 1, 2, 3]:
        bad.append("self-test: 3D exit codes changed")
    saved = (gen.OUTPUT, gen.WORK, gen.assets.CACHE, gen.JOBS)
    with tempfile.TemporaryDirectory() as d:
        gen.OUTPUT, gen.WORK, gen.assets.CACHE, gen.JOBS = d, d, os.path.join(d, "c.json"), os.path.join(d, "jobs")
        try:
            wf = {"1": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x"}}}
            rec = gen.write_record([], {"kind": "image"}, wf, "failed", "boom", "2026-01-01/t", lambda t: None)
            r = json.load(open(rec)) if rec else {}
            if r.get("status") != "failed" or r.get("error") != "boom" or not os.path.exists(rec[:-5] + ".workflow.json"):
                bad.append("self-test: a failed job without outputs got no proper record")
            w = rec[:-5] + ".workflow.json"
            json.dump({"1": {"class_type": "SaveImage", "inputs": {"filename_prefix": "changed"}}}, open(w, "w"))
            try:
                gen.rerun([w]); bad.append("self-test: rerun ran a workflow that no longer matches its record")
            except SystemExit as e:
                if "does not match" not in str(e):
                    bad.append(f"self-test: rerun refused a changed workflow for the wrong reason: {e}")
            # the job file's final state comes from the checks, not from the generation
            gen.JOBS = os.path.join(d, "jobs")
            real_gen, real_chk, real_lease = gen.generate, gen.check_3d, gen.gpu_lease

            def fake_gen(*a, **k):
                gen.JOB.pop("ledger", None); gen.job_state("validating")
                return True, [os.path.join(d, "m.glb")], ""
            gen.generate, gen.check_3d = fake_gen, lambda glb, info, log=None, lock=None: ("check_failed", [])
            gen.gpu_lease = lambda wait=600: contextlib.nullcontext(None)
            try:
                st, _, _ = gen.job_3d({}, [], {}, 10, lambda t: None)
                j = json.load(open(os.path.join(gen.JOBS, gen.JOB["ledger"]["id"] + ".json")))
                if (st, j["state"]) != ("check_failed", "check_failed"):
                    bad.append(f"self-test: 3D job file ends as {j['state']} after a failed check (needs check_failed)")
            finally:
                gen.generate, gen.check_3d, gen.gpu_lease = real_gen, real_chk, real_lease
            # one GPU job at a time: a second lease cannot be taken while one is held
            lk = gen.LOCK
            gen.LOCK = os.path.join(d, "gpu.lock")
            try:
                with gen.gpu_lease(0):
                    try:
                        with gen.gpu_lease(0):
                            bad.append("self-test: two GPU leases were held at once")
                    except gen.GpuBusy:
                        pass
            finally:
                gen.LOCK = lk
            # a chat server that does not stop blocks the job (fail closed)
            run = gen.subprocess.run
            gen.subprocess.run = lambda cmd, *a, **k: type("R", (), {"returncode": 0})()   # "still active", always
            try:
                gen.chat_pause(lambda t: None); bad.append("self-test: a chat server that did not stop was taken as paused")
            except RuntimeError:
                pass
            finally:
                gen.subprocess.run = run
            # strict hashes see a change that kept size and time; the cache alone does not
            p = os.path.join(d, "w.bin")
            open(p, "wb").write(b"a" * 64); st0 = os.stat(p)
            h1 = gen.assets.sha256(p)
            open(p, "wb").write(b"b" * 64); os.utime(p, (st0.st_atime, st0.st_mtime))
            if gen.assets.sha256(p, fresh=True) == h1:
                bad.append("self-test: a strict hash missed a change that kept size and time")
        finally:
            gen.OUTPUT, gen.WORK, gen.assets.CACHE, gen.JOBS = saved
    return bad


def glb(path, body=None, total=None, raw=None):
    """! @brief Writes a small GLB for the tests: one triangle, or the given JSON / raw bytes. """
    import struct
    j = raw if raw is not None else json.dumps(body or {"accessors": [{"count": 3}, {"count": 3}],
                                                     "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}]}).encode()
    j += b" " * (-len(j) % 4)
    data = b"glTF" + struct.pack("<II", 2, total or 12 + 8 + len(j)) + struct.pack("<I", len(j)) + b"JSON" + j
    open(path, "wb").write(data)
    return path


def failure_tests():
    """!
    @brief What goes wrong must end as a recorded failure: a recipe reduced to the chosen model names no file of the
           other; a broken GLB is reported, not raised; out of GPU memory, a crashed ComfyUI connection, a crashed
           Blender and an exception in the checks each end the job with the right state, a record and the chat
           server given back. @return list of problems
    """
    import signal
    import subprocess
    import tempfile
    import time
    bad, quiet = [], lambda t: None
    saved_term = signal.getsignal(signal.SIGTERM)
    # 1. a multi-model recipe is reduced to the chosen model
    threed = {n: p for n, p in gen.PROFILES["models"].items() if p.get("kind") == "3d"}
    for name, p in threed.items():
        r = gen.load_recipe(p["workflow"])
        w = gen.apply_recipe(r, dict(p.get("set", {}), image="x.png", prefix="t"))
        names = {v for n in w.values() for v in n["inputs"].values() if isinstance(v, str)}
        others = {q["file"] for m, q in threed.items() if m != name and q["file"] != p["file"]}
        if p["file"] not in names or names & others:
            bad.append(f"self-test: 3D model {name}: reduced workflow names {sorted(names & others)} of another model"
                       f" or misses its own {p['file']}")
        for inp in ("image", "prefix"):
            if r["inputs"][inp]["set"][0][0] not in w:
                bad.append(f"self-test: reducing the {name} workflow dropped its {inp} node")
        if any(n["class_type"] == "ComfySwitchNode" and not isinstance(n["inputs"].get("switch"), list) for n in w.values()):
            bad.append(f"self-test: a fixed switch stayed in the {name} workflow")
    toy = {"1": {"class_type": "Calc", "inputs": {}}, "2": {"class_type": "A", "inputs": {}}, "3": {"class_type": "B", "inputs": {}},
           "4": {"class_type": "ComfySwitchNode", "inputs": {"switch": ["1", 0], "on_true": ["2", 0], "on_false": ["3", 0]}},
           "5": {"class_type": "Save", "inputs": {"x": ["4", 0]}}}
    if set(gen.prune(json.loads(json.dumps(toy)))) != set(toy):
        bad.append("self-test: a switch decided at run time was removed")
    # 2. broken GLBs are reported, never raised
    with tempfile.TemporaryDirectory() as d:
        ok = gen.glb_stats(glb(os.path.join(d, "ok.glb")))
        if not ok.get("valid") or ok.get("triangles") != 1:
            bad.append(f"self-test: a valid one-triangle GLB read as {ok}")
        cases = {"truncated": dict(total=10 ** 6), "bad JSON": dict(raw=b"{not json"),
                 "index out of range": dict(body={"accessors": [{"count": 3}], "meshes": [{"primitives": [{"attributes": {"POSITION": 5}}]}]}),
                 "negative index": dict(body={"accessors": [{"count": 3}], "meshes": [{"primitives": [{"attributes": {"POSITION": -1}}]}]}),
                 "wrong types": dict(body={"meshes": "x"})}
        for why, kw in cases.items():
            try:
                r = gen.glb_stats(glb(os.path.join(d, "b.glb"), **kw))
                if r.get("valid") or not r.get("error"):
                    bad.append(f"self-test: a GLB with {why} read as valid: {r}")
            except Exception as e:
                bad.append(f"self-test: a GLB with {why} raised {type(e).__name__}")
        if gen.glb_stats(os.path.join(d, "missing.glb")).get("valid") is not False:
            bad.append("self-test: a missing GLB was not reported as invalid")
        # 3. stand-in failures through the real job_3d / check_3d / report_3d
        saved = (gen.OUTPUT, gen.WORK, gen.assets.CACHE, gen.JOBS, gen.generate, gen.gpu_lease, gen.check_3d,
                 gen.chat_pause, gen.chat_resume, gen.shutil.which, gen.subprocess.run)
        gen.OUTPUT, gen.WORK, gen.assets.CACHE, gen.JOBS = d, d, os.path.join(d, "c.json"), os.path.join(d, "jobs")
        gen.gpu_lease = lambda wait=600: contextlib.nullcontext(None)
        resumed = []
        gen.chat_pause, gen.chat_resume = (lambda log=None: "paused"), (lambda p, log=None: resumed.append(p))
        good, broken = glb(os.path.join(d, "m.glb")), glb(os.path.join(d, "t.glb"), total=10 ** 6)

        def gen_with(result=None, exc=None):
            def fake(*a, **k):
                gen.JOB.pop("ledger", None); gen.job_state("running")
                if exc:
                    raise exc
                if result[0]:
                    gen.job_state("validating")
                return result
            return fake

        def run(label, want, **fake):
            gen.generate = gen_with(**fake)
            info = {"kind": "3d"}
            try:
                st, files, err = gen.job_3d({}, [], info, 10, quiet)
                j = json.load(open(os.path.join(gen.JOBS, gen.JOB["ledger"]["id"] + ".json")))
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = gen.report_3d(st, files, info, {"1": {"class_type": "X", "inputs": {}}}, err, "2026-01-01/" + label, 1, quiet)
                rec = re.findall(r"record: (\S+)", out.getvalue())
                r = json.load(open(rec[-1])) if rec else {}
                if (st, j["state"], r.get("status"), code) != (want, want, want, gen.EXIT[want]):
                    bad.append(f"self-test: {label}: job {st}, job file {j['state']}, record {r.get('status')}, exit {code}"
                               f" (needs {want} everywhere)")
            except Exception as e:
                bad.append(f"self-test: {label} escaped as {type(e).__name__}: {e}")
        try:
            run("oom", "failed", result=(False, [], "CUDA error: out of memory"))
            run("comfyui-gone", "failed", exc=ConnectionError("connection refused"))
            run("broken-glb", "failed", result=(True, [broken], ""))
            gen.check_3d = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("checker bug"))
            run("check-raises", "check_failed", result=(True, [good], ""))
            gen.check_3d = saved[6]
            gen.shutil.which = lambda n: "/stand-in/blender" if n == "blender" else saved[9](n)
            gen.subprocess.run = lambda *a, **k: type("R", (), {"returncode": 139, "stdout": "", "stderr": "Segmentation fault"})()
            resumed.clear()
            run("blender-crash", "check_failed", result=(True, [good], ""))
            if resumed != ["paused"]:
                bad.append("self-test: the chat server was not given back after Blender crashed")
            # an old Blender wrapper (no lock hand-over) fails the check at once instead of waiting for the held lock
            gen.subprocess.run = saved[10]
            old_wrapper = os.path.join(d, "old-blender")
            open(old_wrapper, "w").write("#!/bin/sh\nexec 9>/tmp/ai-gpu.lock; flock 9\n")
            gen.shutil.which = lambda n: old_wrapper if n == "blender" else saved[9](n)
            called = []
            gen.subprocess.run = lambda *a, **k: called.append(a) or type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
            info = {}
            with open(os.path.join(d, "lock"), "w") as lk:
                st, _ = gen.check_3d(good, info, quiet, lk)
            if st != "check_failed" or called or "too old" not in str(info.get("blender")):
                bad.append(f"self-test: an old Blender wrapper gave {st} (ran: {bool(called)}), needs an immediate check_failed")
            gen.subprocess.run = saved[10]
            # a stop at any point leaves the job "interrupted"; a job whose process is gone reads as abandoned
            def stop_while_preparing():
                gen.JOB.pop("ledger", None); gen.job_state("preparing")
                gen.JOB["pending"] = ({"kind": "image"}, {"1": {"class_type": "SaveImage", "inputs": {}}}, "2026-01-01/stop")
                os.kill(os.getpid(), signal.SIGTERM); time.sleep(5)
            with contextlib.redirect_stderr(io.StringIO()):
                code = gen.entry(stop_while_preparing)
            j = json.load(open(os.path.join(gen.JOBS, gen.JOB["ledger"]["id"] + ".json")))
            if code != 130 or j["state"] != "interrupted":
                bad.append(f"self-test: a job stopped (SIGTERM) while preparing ended as {j['state']}, exit {code} (needs interrupted, 130)")
            rec = os.path.join(d, "2026-01-01", f"stop_interrupted_{j['id']}.json")
            if not os.path.exists(rec) or json.load(open(rec)).get("status") != "interrupted":
                bad.append("self-test: a stopped job got no record (needs <prefix>_interrupted_<job>.json)")
            if signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL and signal.getsignal(signal.SIGTERM) != saved_term:
                bad.append("self-test: entry() left its signal handler installed")
            # stopped while ComfyUI runs our job: the job is cancelled there before the lock is given back
            api, peak = gen.api, gen.Peak
            calls = []

            def fake_api(path, data=None, *a, **k):
                calls.append(path)
                if path == "/prompt":
                    return {"prompt_id": "p1"}
                if path == "/queue":
                    return {"queue_running": [[0, "p1"]], "queue_pending": []}
                if path.startswith("/history"):
                    os.kill(os.getpid(), signal.SIGTERM); time.sleep(5)
                return {}
            gen.api, gen.Peak = fake_api, lambda: type("P", (), {"start": lambda s: None, "peak": 0, "go": True})()
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    gen.entry(lambda: (gen.JOB.pop("ledger", None), gen.run_job({}, 60, quiet)))
            finally:
                gen.api, gen.Peak = api, peak
            if "/interrupt" not in calls:
                bad.append("self-test: a job stopped while ComfyUI ran it was not cancelled there")
            dead = subprocess.Popen(["true"]); dead.wait()
            for state, pid, want in (("running", dead.pid, "abandoned"), ("running", os.getpid(), "running"),
                                     ("succeeded", dead.pid, "succeeded")):
                v = gen.job_view({"id": "x", "state": state, "pid": pid, "started": gen.proc_start(pid) or 1})
                if v["state"] != want:
                    bad.append(f"self-test: a {state} job of a {'dead' if pid != os.getpid() else 'live'} process reads as {v['state']} (needs {want})")
        finally:
            (gen.OUTPUT, gen.WORK, gen.assets.CACHE, gen.JOBS, gen.generate, gen.gpu_lease, gen.check_3d,
             gen.chat_pause, gen.chat_resume, gen.shutil.which, gen.subprocess.run) = saved
    return bad


def recipe_tests():
    """! @brief Recipes: declared inputs are set (seeds keep their per-stage offsets), undeclared values refused,
           3D models refused on the image path. @return list of problems """
    bad = []
    for name, p in gen.PROFILES["models"].items():
        if p.get("kind") != "3d":
            continue
        r = gen.load_recipe(p["workflow"])
        w = gen.apply_recipe(r, dict(p.get("set", {}), seed=7, faces=123456, image="a.png", prefix="d/x"))
        for inp, val in (("faces", 123456), ("image", "a.png"), ("prefix", "d/x")):
            node, key = r["inputs"][inp]["set"][0][:2]
            if w[node]["inputs"][key] != val:
                bad.append(f"self-test: recipe {p['workflow']} input {inp} not set")
        for node, key, *off in r["inputs"]["seed"]["set"]:
            if w[node]["inputs"][key] != 7 + (off[0] if off else 0):
                bad.append(f"self-test: recipe seed offset lost on node {node}")
        try:
            gen.apply_recipe(r, {"texture": 3000}); bad.append("self-test: a texture size outside the recipe's choices was used")
        except SystemExit:
            pass
        try:
            gen.resolve_model(name, lambda t: None); bad.append(f"self-test: 3D model {name} accepted for images")
        except SystemExit as e:
            if "ai gen 3d" not in str(e):
                bad.append(f"self-test: 3D model refused for images with the wrong message: {e}")
    return bad


def model_rule_tests():
    """!
    @brief One rule for every model: a diffusion-model file gets its architecture from its note and starts from that
           architecture's profile; without a note it is refused with how to fix it; a profile's own file takes its note
           too; a file in the wrong folder is refused; every result saves the exact workflow next to its record.
    @return list of problems
    """
    import struct
    import tempfile
    bad, quiet = [], (lambda t: None)

    def st(p, keys):
        h = json.dumps({k: {"dtype": "F16", "shape": [1], "data_offsets": [2 * i, 2 * i + 2]} for i, k in enumerate(keys)}).encode()
        with open(p, "wb") as f:
            f.write(struct.pack("<Q", len(h)) + h + b"\0" * (2 * len(keys)))

    with tempfile.TemporaryDirectory() as d:
        for sub in ("checkpoints", "diffusion_models"):
            os.makedirs(os.path.join(d, "comfy", sub))
        ft = os.path.join(d, "comfy/diffusion_models/my_zimage_finetune.safetensors")
        st(ft, ["layers.0.attention.qkv.weight"])
        zfile = gen.profile("zimage")["file"]
        zp = os.path.join(d, "comfy/diffusion_models", zfile)
        st(zp, ["layers.0.x.weight", "layers.1.y.weight"])
        wrong = os.path.join(d, "comfy/diffusion_models/sdxl_in_wrong_place.safetensors")
        st(wrong, ["conditioner.embedders.1.model.x"])
        saved = (gen.assets.ROOTS[:], gen.assets.NOTES, gen.assets.CACHE)
        gen.assets.ROOTS[:] = [os.path.join(d, "comfy")]
        gen.assets.NOTES, gen.assets.CACHE = os.path.join(d, "n.json"), os.path.join(d, "c.json")
        try:
            unknown = os.path.join(d, "comfy/diffusion_models/strange_model.safetensors")
            st(unknown, ["blocks.0.x.weight"])
            try:
                gen.resolve_model("strange_model", quiet); bad.append("self-test: a file of unknown kind was used")
            except SystemExit as e:
                if "--arch" not in str(e):
                    bad.append(f"self-test: unknown kind refused without the fix: {e}")
            if gen.resolve_model("my_zimage_finetune", quiet)["arch"] != "zimage":
                bad.append("self-test: a detected Z-Image file did not get the zimage workflow")
            gen.assets.set_note(ft, {"steps": "6"})
            p = gen.resolve_model("my_zimage_finetune", quiet)
            z = gen.profile("zimage")
            if (p["arch"], p["file"], p["steps"], p["text_encoder"], p["vae"]) != ("zimage", "my_zimage_finetune.safetensors", 6,
                                                                                z["text_encoder"], z["vae"]):
                bad.append(f"self-test: diffusion-model file with a note resolved wrong: {p}")
            gen.assets.set_note(zp, {"steps": "5"})
            if gen.resolve_model("zimage", quiet)["steps"] != 5:
                bad.append("self-test: a profile's own file did not take its note")
            gen.assets.set_note(wrong, {"arch": "sdxl"})
            try:
                gen.resolve_model("sdxl_in_wrong_place", quiet); bad.append("self-test: an SDXL file in diffusion_models was used")
            except SystemExit as e:
                if "move it" not in str(e):
                    bad.append(f"self-test: wrong folder refused for the wrong reason: {e}")
            out = os.path.join(d, "x_00001_.png")
            open(out, "wb").close()
            wf = gen.zimage("p", 1, 8, 1024, 1024, "t", z)
            rec = gen.write_record([out], {"kind": "image"}, wf)
            if json.load(open(rec)).get("workflow") != "x_00001_.workflow.json" or json.load(open(os.path.join(d, "x_00001_.workflow.json"))) != wf:
                bad.append("self-test: the exact workflow was not saved next to the record")
        finally:
            gen.assets.ROOTS[:], gen.assets.NOTES, gen.assets.CACHE = saved
    return bad


def lora_rule_tests():
    """!
    @brief The LoRA rules on dummy files: structural fit, a mismatch refused before anything starts, the LoRA's note
           used for strength / trigger / CFG, typed values win over the note, conflicting notes stop the job,
           an incomplete (still copying) file is not judged. @return list of problems
    """
    import argparse
    import struct
    import tempfile
    bad = []

    def st(p, keys, trunc=False):
        h = json.dumps({k: {"dtype": "F16", "shape": [1], "data_offsets": [2 * i, 2 * i + 2]} for i, k in enumerate(keys)}).encode()
        with open(p, "wb") as f:
            f.write(struct.pack("<Q", len(h)) + h + b"\0" * (2 * len(keys) - (2 if trunc else 0)))

    with tempfile.TemporaryDirectory() as d:
        L = ["input_blocks.4.1.proj_in", "middle_block.1.proj_out", "output_blocks.2.1.proj_in"]
        for sub in ("checkpoints", "loras"):
            os.makedirs(os.path.join(d, "comfy", sub))
        st(os.path.join(d, "comfy/checkpoints/juggernautXL_ragnarok.safetensors"), ["model.diffusion_model." + k + ".weight" for k in L])
        st(os.path.join(d, "comfy/loras/sdxl_a.safetensors"), [f"lora_unet_{k.replace('.', '_')}.lora_down.weight" for k in L])
        st(os.path.join(d, "comfy/loras/sdxl_b.safetensors"), [f"lora_unet_{k.replace('.', '_')}.lora_up.weight" for k in L])
        st(os.path.join(d, "comfy/loras/flux_c.safetensors"), ["transformer.single_blocks.0.linear1.lora_A.weight"])
        st(os.path.join(d, "comfy/loras/part_d.safetensors"), [f"lora_unet_{L[0].replace('.', '_')}.lora_down.weight"], trunc=True)
        saved = (gen.LORA_DIRS, gen.assets.ROOTS[:], gen.assets.NOTES, gen.assets.TESTED, gen.assets.CACHE)
        gen.LORA_DIRS, gen.assets.ROOTS[:] = [os.path.join(d, "comfy/loras")], [os.path.join(d, "comfy")]
        gen.assets.NOTES, gen.assets.TESTED, gen.assets.CACHE = (os.path.join(d, n) for n in ("n.json", "t.json", "c.json"))
        quiet = lambda t: None
        prof = gen.profile("juggernaut")

        def run(loras, **kw):
            a = argparse.Namespace(prompt="a harbour", lora=loras, cfg=None, negative=None, force_lora=False)
            for k, v in kw.items():
                setattr(a, k, v)
            return gen.resolve_loras(a, prof, "blurry", quiet)
        try:
            lp = os.path.join(d, "comfy/loras/sdxl_a.safetensors")
            gen.assets.set_note(lp, {"trigger": "wtrclr", "strength": "0.6", "cfg": "5"})
            loras, prompt, neg, p, checks = run(["sdxl_a"])
            if (loras, prompt, p["cfg"], checks[0]["fit"]) != ([("sdxl_a.safetensors", 0.6)], "wtrclr, a harbour", 5.0, "fits"):
                bad.append(f"self-test: LoRA note not applied: {loras} {prompt!r} cfg {p['cfg']} {checks[0]['fit']}")
            loras, _, _, p, _ = run(["sdxl_a:0.9"], cfg=3.5)
            if loras[0][1] != 0.9 or p["cfg"] != 3.5:
                bad.append("self-test: typed strength / --cfg did not win over the LoRA note")
            gen.assets.set_note(os.path.join(d, "comfy/loras/sdxl_b.safetensors"), {"cfg": "7"})
            for case, ls, want in (("mismatch", ["flux_c"], "does not fit"),
                                   ("conflicting notes", ["sdxl_a", "sdxl_b"], "disagree")):
                try:
                    run(ls); bad.append(f"self-test: {case} was not refused")
                except SystemExit as e:
                    if want not in str(e):
                        bad.append(f"self-test: {case} refused for the wrong reason: {e}")
            st(os.path.join(d, "comfy/checkpoints/fused.safetensors"), ["layers.0.proj.qkv.weight", "layers.0.proj.o.weight"])
            st(os.path.join(d, "comfy/loras/sep.safetensors"), [f"diffusion_model.layers.0.proj.{x}.lora_A.weight" for x in "qkvo"])
            f = gen.assets.lora_fit(os.path.join(d, "comfy/loras/sep.safetensors"), os.path.join(d, "comfy/checkpoints/fused.safetensors"))
            if f["status"] != "fits":
                bad.append(f"self-test: separate q/k/v LoRA layers on a fused qkv model: {f['status']} ({f['why']})")
            if run(["part_d"])[4][0]["fit"] != "unchecked":
                bad.append("self-test: a still-copying LoRA was judged")
            if run(["flux_c"], force_lora=True)[4][0]["fit"] != "mismatch":
                bad.append("self-test: --force-lora did not record the mismatch")
        finally:
            gen.LORA_DIRS, gen.assets.ROOTS[:], gen.assets.NOTES, gen.assets.TESTED, gen.assets.CACHE = saved
    return bad


def main():
    """! @brief Builds all variants and checks them. @return 0 if clean """
    ci = bool(os.environ.get("CI"))     # GitHub Actions sets CI: there a missing schema or unknown node fails
    # CI has no ComfyUI: its model detection (detect.py) is replaced by a stand-in that names files by their layers;
    # on the PC, test.sh runs the real one against every profile file
    gen.detect_classes = lambda paths: {p: ("SDXL" if gen.is_sdxl(p) else "ZImage" if "zimage" in os.path.basename(p)
                                            else None) for p in paths}
    gen.class_arch = lambda: {"SDXL": "sdxl", "ZImage": "zimage"}
    bad = self_tests()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")
    if not os.path.exists(path):
        print("schema.json not captured yet (run: ai gen schema)" + ("" if ci else " - workflow check skipped"))
        return 1 if ci or bad else 0
    schema, n, unknown = json.load(open(path)), 0, set()
    for i, wf in enumerate(gen.all_workflows()):
        bad += check(wf, schema, f"variant {i} ({wf[next(iter(wf))]['class_type']}...)", unknown); n += 1
    if unknown:   # new nodes: recapture with "ai gen schema" after installing; locally a warning, in CI a failure
        msg = f"not in schema.json yet (recapture with: ai gen schema): {', '.join(sorted(unknown))}"
        if ci:
            bad.append(msg)
        else:
            print("WARNING " + msg)
    for b in sorted(set(bad)):
        print(b)
    print(f"{n} workflow variants checked: {'OK' if not bad else f'{len(set(bad))} problem(s)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
