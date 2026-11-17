"""Offline compatibility checks for extracted generation capabilities."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import gen
import runtime
from generation import image, models


class GenerationTests(unittest.TestCase):
    def test_existing_workflow_graphs_are_identical(self):
        baseline = json.loads((Path(__file__).resolve().parents[2] / 'tests/fixtures').joinpath("workflow-baseline.json").read_text())
        actual = [hashlib.sha256(json.dumps(workflow, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                  for workflow in runtime.all_workflows(include_packages=False)]
        self.assertEqual(actual, baseline["workflows"])

    def test_capabilities_import_without_cli_gpu_or_service_runtime(self):
        code = ("import sys; import generation.image,generation.video,generation.mesh; "
                "assert 'gen' not in sys.modules; assert 'torch' not in sys.modules; "
                "assert 'assets' not in sys.modules")
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]), PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([sys.executable, "-B", "-c", code], env=environment, check=True, timeout=10)

    def test_profile_override_does_not_change_the_catalogue(self):
        profile = models.profile()
        old = models.profile()["file"]
        profile["file"] = "caller-override.safetensors"
        self.assertEqual(models.profile()["file"], old)
        self.assertIs(gen.PROFILES, models.PROFILES)

    def test_image_adapter_rejects_a_different_architecture(self):
        with self.assertRaisesRegex(ValueError, "needs an SDXL model"):
            image.sdxl("prompt", "negative", 1, 20, 1024, 1024, False, "test",
                       prof={"arch": "another", "name": "fixture"})

    def test_recipe_rejects_an_unsupported_choice(self):
        recipe = {"workflow": {}, "inputs": {"texture": {"choices": [1024, 2048], "set": []}}}
        with self.assertRaisesRegex(SystemExit, "choose one of"):
            runtime.apply_recipe(recipe, {"texture": 3000})

    def test_packages_separate_capabilities_and_copy_nested_settings(self):
        self.assertEqual(models.package(kind="video").architecture, "wan22")
        self.assertEqual(models.package(kind="3d").kind, "3d")
        first = models.package("realvis")
        first.settings["file"] = "changed"
        self.assertNotEqual(models.package("realvis").settings["file"], "changed")
        with self.assertRaisesRegex(ValueError, "capability mismatch"):
            models.package(gen.PROFILES["default_3d"], kind="image")

    def test_incomplete_video_package_fails_before_workflow_construction(self):
        from unittest.mock import patch
        with patch.dict(models.PROFILES["video_models"], {"broken": {"kind": "video", "arch": "wan22"}}):
            with self.assertRaisesRegex(ValueError, "Incomplete model package"):
                models.package("broken", kind="video")

    def test_video_model_package_is_data_and_cannot_select_python_code(self):
        from unittest.mock import patch
        from generation.video import wan
        profile = models.package(kind="video").settings
        profile["file"] = "local-alternative.gguf"
        with patch.dict(models.PROFILES["video_models"], {"alternative": profile}):
            workflow = wan("prompt", "negative", 1, 20, 33, "test", model="alternative")
            self.assertEqual(workflow["1"]["inputs"]["unet_name"], "local-alternative.gguf")
            profile["arch"] = "arbitrary_python_module"
            with self.assertRaisesRegex(ValueError, "needs architecture wan22"):
                wan("prompt", "negative", 1, 20, 33, "test", model="alternative")

    def test_video_cli_uses_the_selected_package_without_running_a_service(self):
        import contextlib
        import io
        import tempfile
        from unittest.mock import patch
        selected = models.package(kind="video").settings
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(models.PROFILES["video_models"], {"fixture": selected}), \
             patch.object(sys, "argv", ["gen.py", "video", "a scene", "--model", "fixture", "--seed", "7"]), \
             patch.object(runtime, "OUTPUT", directory), patch.dict(gen.JOB, {}, clear=True), \
             patch.object(runtime.builders.registry, "check_parts"), \
             patch.object(runtime, "generate", return_value=(True, ["clip.webp"], None)) as execute, \
             patch.object(runtime, "write_record", return_value=None) as record, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen.main(), 0)
            self.assertEqual(execute.call_args.args[0]["1"]["inputs"]["unet_name"], selected["file"])
            self.assertEqual(record.call_args.args[1]["model"], "fixture")

    def test_video_image_conditioning_metadata_and_input_validation(self):
        from unittest.mock import patch
        from operations import video
        args=gen.generation_parser().parse_args(['video','a wave','--image','frame.png','--frames','33','--check'])
        with patch.object(runtime,'check_input',return_value='/fixture/frame.png') as check:
            graph,uploads,info,limit,profile=video.prepare(args,7,'test')
        check.assert_called_once_with('frame.png')
        self.assertEqual(uploads,[('11','/fixture/frame.png')])
        self.assertEqual(graph['7']['inputs']['start_image'],['11',0])
        self.assertEqual(graph['11']['class_type'],'LoadImage')
        self.assertEqual(info['fps'],graph['10']['inputs']['fps'])
        self.assertEqual(info['duration_seconds'],33/info['fps'])
        args.image=[]
        graph,*_=video.prepare(args,7,'test')
        self.assertNotIn('11',graph)
        self.assertNotIn('start_image',graph['7']['inputs'])
        args.image=['missing.png']
        with patch.object(runtime,'check_input',side_effect=ValueError('missing image')), self.assertRaisesRegex(ValueError,'missing image'):
            video.prepare(args,7,'test')
        args.image=['one.png','two.png']
        with self.assertRaisesRegex(SystemExit,'at most one'): video.validate(args)

    def test_feature_default_is_explained_and_explicit_model_is_preserved(self):
        import contextlib,io
        from unittest.mock import patch
        from operations import image as operation
        for requested in (None,'realvis'):
            argv=['image','a character','--pose','pose.png','--size','1024x1024','--check']
            if requested: argv+=['--model',requested]
            args=gen.generation_parser().parse_args(argv)
            args.face_weight=.65  # Applied by gen.main before operation dispatch.
            with patch.object(runtime,'resolve_model',side_effect=lambda name,log:models.profile(name)), \
                 patch.object(runtime,'check_input',return_value='/fixture/pose.png'), \
                 patch.object(runtime,'is_skeleton',return_value=True), \
                 patch.object(runtime,'resolve_loras',side_effect=lambda a,p,n,log:([],a.prompt,n,p,[])), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                _,_,info,_,_=operation.prepare(args,7,'test')
            expected=requested or models.PROFILES[models.PROFILES['feature_defaults']['pose']]
            self.assertEqual(info['model'],expected)
            self.assertIn('Selected model '+expected,output.getvalue())
            self.assertIn('explicit model' if requested else '--pose feature default',info['model_selection_reason'])

    def test_generated_pose_skeleton_uses_validated_model_canvas(self):
        from features.pose.generate import pose_wf
        graph=pose_wf('walking',7,896,1152,'test')
        self.assertEqual(graph['35']['inputs']['images'],['34',0])
        self.assertEqual(graph['34']['inputs'],dict(image=['31',0],upscale_method='nearest-exact',width=896,height=1152,crop='disabled'))
        reference=pose_wf('',7,896,1152,'test',src='photo.png')
        self.assertNotIn('34',reference)  # Reference extraction keeps its own canvas.

    def test_mp4_export_decodes_all_frames_and_preserves_existing_files(self):
        import tempfile
        from unittest.mock import patch
        from PIL import Image
        from operations import video
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'clip.webp'
            frames=[Image.new('RGB',(8,8),colour) for colour in ('red','green','blue')]
            frames[0].save(source,save_all=True,append_images=frames[1:],duration=42,lossless=True)
            before=source.read_bytes()
            captured={}
            def encode(command,**kwargs):
                captured['command']=command;captured['raw']=kwargs['stdin'].read()
                Path(command[-1]).write_bytes(b'encoded fixture')
            with patch.object(video.subprocess,'run',side_effect=encode):
                files=video.export_mp4([str(source)],24)
            self.assertEqual(len(captured['raw']),3*8*8*3)
            self.assertIn('-n',captured['command'])
            self.assertIn('rawvideo',captured['command'])
            self.assertEqual(files,[str(source),str(source.with_suffix('.mp4'))])
            self.assertEqual(source.read_bytes(),before)
            with self.assertRaisesRegex(ValueError,'overwrite'): video.export_mp4([str(source)],24)
            args=gen.generation_parser().parse_args(['video','a scene','--mp4'])
            with patch.object(video.shutil,'which',return_value=None), self.assertRaisesRegex(SystemExit,'existing ffmpeg'):
                video.validate(args)


if __name__ == "__main__":
    unittest.main()
