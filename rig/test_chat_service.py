"""No systemd or GPU calls: exercise the real lease/stop/restore wiring."""
import ast
from contextlib import contextmanager
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import chat_service


class ChatTests(unittest.TestCase):
    def test_stop_is_verified_and_bounded(self):
        with patch.object(chat_service.subprocess, "run") as run:
            run.return_value = SimpleNamespace(stdout="active\n")
            with self.assertRaisesRegex(RuntimeError, "did not stop"):
                chat_service.stop_chat()
            self.assertEqual(run.call_args_list[0].args[0][2], "stop")
            self.assertTrue(all(c.kwargs["timeout"] == 30 for c in run.call_args_list))

    def test_restart_is_verified(self):
        with patch.object(chat_service.subprocess, "run", return_value=SimpleNamespace(stdout="inactive\n")):
            with self.assertRaisesRegex(RuntimeError, "did not restart"):
                chat_service.start_chat()

    def test_transition_and_systemd_failure_refuse_work(self):
        with patch.object(chat_service.subprocess, "run", return_value=SimpleNamespace(stdout="activating\n")):
            with self.assertRaises(RuntimeError): chat_service.chat_state()
        with patch.object(chat_service.subprocess, "run", side_effect=subprocess.TimeoutExpired("systemctl", 30)):
            with self.assertRaises(subprocess.TimeoutExpired): chat_service.stop_chat()

    def exercise_wiring(self, initially_active, failure):
        # Execute the actual restoration callback and acquisition prefix from cli.py.
        # Cutting before inference keeps this test independent of installed assets.
        tree = ast.parse(Path(__file__).with_name("cli.py").read_text())
        execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
        restore = next(n for n in execute.body if isinstance(n, ast.FunctionDef) and n.name == "restore_chat")
        lease = next(n for n in ast.walk(execute) if isinstance(n, ast.With)
                     and isinstance(n.items[0].context_expr, ast.Call)
                     and getattr(n.items[0].context_expr.func, "id", "") == "gpu_lease")
        prefix = []
        for node in lease.body:
            prefix.append(node)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "step":
                break
        self.assertTrue(any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "stop_chat"
                            for s in prefix for n in ast.walk(s)), "original bug: chat is never stopped")
        lease.body = prefix
        events = []; record = {}; state = ["active" if initially_active else "inactive"]
        def stop(): state[0] = "inactive"; events.append("stop")
        def start(): state[0] = "active"; events.append("start")
        def step(*args):
            self.assertEqual(state[0], "inactive", "chat may reload during GPU work")
            events.append("work")
            if failure: raise failure
        @contextmanager
        def lock(callback):
            events.append("lock")
            try: yield 7
            finally: callback(); events.append("unlock")
        monitor = SimpleNamespace(stop=SimpleNamespace(set=lambda: None),
                                  thread=SimpleNamespace(is_alive=lambda: False))
        scope = dict(record=record, monitor=monitor, was_active=False, env={}, save=lambda: None,
                     chat_state=lambda: state[0], stop_chat=stop, start_chat=start, step=step, gpu_lease=lock)
        code = compile(ast.Module(body=[restore, lease], type_ignores=[]), "cli.py", "exec")
        if failure:
            with self.assertRaises(type(failure)): exec(code, scope)
        else: exec(code, scope)
        self.assertEqual(events, ["lock", "stop", "work"] + (["start"] if initially_active else []) + ["unlock"])

    def test_normal_failure_and_cancellation_restore_inside_lease(self):
        for active in (False, True):
            for failure in (None, RuntimeError("inference failed"), InterruptedError("cancelled")):
                with self.subTest(active=active, failure=failure): self.exercise_wiring(active, failure)


if __name__ == "__main__":
    unittest.main()
