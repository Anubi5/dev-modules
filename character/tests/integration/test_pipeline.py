import copy
import json
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import contracts
import pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        image = self.root/"front.png"; image.write_bytes(b"synthetic image input")
        contracts.write(self.root/"input.asset.json", "image", [("front", image)])
        self.plan = dict(schema=pipeline.SCHEMA, stages=[
            dict(id="mesh", operation="mesh", inputs=[dict(manifest="input.asset.json")]),
            dict(id="rig", operation="rig", inputs=[dict(stage="mesh")], parameters=dict(facing="-y")),
            dict(id="animation", operation="animation", inputs=[dict(stage="rig")])])
        self.calls = []

    def execute(self, args, log, environment, timeout, started):
        self.calls.append(args); started(os.getpid())
        operation = "rig" if args[1] == "rig" else "mesh"
        output = self.root/"producer-work"/operation; output.mkdir(parents=True, exist_ok=True)
        file = output/("model.fbx" if operation == "rig" else "model.glb"); file.write_bytes(b"synthetic output")
        record = output/"record.json"
        record.write_text(json.dumps(dict(status="succeeded", outputs={file.name: {}} if operation == "rig" else [str(file)],
            validation_results=dict(fbx=dict(ok=True), glb=dict(ok=True)), glb=dict(valid=True))))
        log.write_text("progress only; never parsed for record paths")
        log.with_suffix(".json").write_text(json.dumps(dict(schema="local-ai.producer/1", status="succeeded",
                                                          exit_code=0, record=str(record))))

    def test_stages_resume_without_repeating_producers_and_handoff_is_hashed(self):
        with patch.object(pipeline, "execute", self.execute):
            state = pipeline.run(self.plan, self.root)
            self.assertEqual(state["status"], "awaiting_host")
            self.assertEqual(len(self.calls), 2)
            self.assertIn("--facing=-y", self.calls[1])
            self.assertTrue(all("--json" in args for args in self.calls))
            rig = contracts.read(self.root/"rig.asset.json")
            self.assertTrue(all(row["path"].endswith(".fbx") for row in rig["files"]))
            state = pipeline.run(self.plan, self.root, resume=True)
            self.assertEqual(len(self.calls), 2)
            request = self.root/"animation.unreal-request.json"
            clip = self.root/"walk.fbx"; clip.write_bytes(b"synthetic host response")
            contracts.write(self.root/"animation.asset.json", "animation", [("walk", clip)],
                            dict(units="cm", up="+Z", forward="+X"),
                            source=dict(request_sha256=contracts.digest(request)))
            self.assertEqual(pipeline.run(self.plan, self.root, resume=True)["status"], "succeeded")
            self.assertEqual(len(self.calls), 2)

    def test_completed_artifact_or_plan_drift_is_refused(self):
        with patch.object(pipeline, "execute", self.execute): pipeline.run(self.plan, self.root)
        changed = copy.deepcopy(self.plan); changed["stages"][0]["parameters"] = dict(seed=2)
        with self.assertRaisesRegex(ValueError, "plan or state"):
            pipeline.run(changed, self.root, resume=True)
        manifest = contracts.read(self.root/"mesh.asset.json")
        contracts.inside(self.root, manifest["files"][0]["path"]).write_bytes(b"drift")
        with self.assertRaisesRegex(ValueError, "changed artifact"):
            pipeline.run(self.plan, self.root, resume=True)

    def test_producer_failure_retains_prior_stage_and_can_retry_explicitly(self):
        def fail(args, log, environment, timeout, started):
            if args[1] == "rig": raise RuntimeError("controlled failure")
            self.execute(args, log, environment, timeout, started)
        with patch.object(pipeline, "execute", fail), self.assertRaises(RuntimeError):
            pipeline.run(self.plan, self.root)
        self.assertTrue((self.root/"mesh.asset.json").exists())
        with patch.object(pipeline, "execute", self.execute):
            pipeline.run(self.plan, self.root, resume=True)
        self.assertEqual(len(self.calls), 2)

    def test_plan_rejects_arbitrary_commands_and_forward_references(self):
        for stage in (dict(id="x", operation="shell", parameters=dict(command="anything")),
                      dict(id="x", operation="rig", inputs=[dict(stage="later")])):
            with self.assertRaises(ValueError): pipeline.validate_plan(dict(schema=pipeline.SCHEMA, stages=[stage]))

    def test_prompt_is_passed_literally_after_option_terminator(self):
        stage = dict(id="image", operation="image", parameters=dict(prompt="--literal prompt", model="fixture"))
        self.assertEqual(pipeline.command(stage, [])[-2:], ["--", "--literal prompt"])

    def test_log_scraping_and_multiple_json_results_are_refused(self):
        log = self.root/"producer.log"
        log.write_text('  record: /some/record.json\nRIG {"record":"/some/record.json"}')
        with self.assertRaises(FileNotFoundError): pipeline.producer_outputs(log, "mesh", self.root)
        log.with_suffix(".json").write_text('{}\n{}\n')
        with self.assertRaises(json.JSONDecodeError): pipeline.producer_outputs(log, "mesh", self.root)

    def test_abandoned_running_stage_never_replays_automatically(self):
        class ControllerLost(BaseException): pass
        def lost(args, log, environment, timeout, started):
            started(os.getpid()); raise ControllerLost()
        with patch.object(pipeline, "execute", lost), self.assertRaises(ControllerLost):
            pipeline.run(self.plan, self.root)
        for _ in range(2):
            with patch.object(pipeline, "execute") as execute, self.assertRaises(pipeline.UnfinishedStage):
                pipeline.run(self.plan, self.root, resume=True)
            execute.assert_not_called()

    def test_timeout_stops_the_owned_cpu_process(self):
        pid = []
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            pipeline.execute([sys.executable, "-c", "import time;time.sleep(30)"],
                             self.root/"timeout.log", dict(os.environ), .1, pid.append)
        with self.assertRaises(ProcessLookupError): os.kill(pid[0], 0)


if __name__ == "__main__": unittest.main()
