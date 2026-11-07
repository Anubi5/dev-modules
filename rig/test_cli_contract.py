"""Keep existing argument and exit-code dispatch while adding safety checks."""
import unittest
from unittest.mock import patch

import cli


class CliContractTests(unittest.TestCase):
    def test_existing_command_defaults_and_options_remain_accepted(self):
        args = cli.parser().parse_args(["model.glb", "--name", "character", "--height", "1.7",
                                       "--seed", "12345", "--triangles", "100000", "--smooth-iterations", "256"])
        self.assertEqual(args.facing, "-y")
        self.assertEqual(args.seed, 12345)
        self.assertEqual(args.triangles, 100000)

    def test_exit_code_dispatch_is_unchanged(self):
        for status in (0, 1, 2, 130):
            with self.subTest(status=status), patch.object(cli.sys, "argv", ["ai-rig", "model.glb"]), \
                    patch.object(cli, "execute", return_value=status), patch.object(cli.signal, "signal"):
                self.assertEqual(cli.main(), status)


if __name__ == "__main__":
    unittest.main()
