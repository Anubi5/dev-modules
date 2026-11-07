"""Human-operated file contracts and finite character pipeline commands."""
import argparse
import json
from pathlib import Path
import signal

import contracts


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ai character", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest", help="Create a hashed artifact manifest; does not validate geometry")
    manifest.add_argument("kind", choices=sorted(contracts.KINDS))
    manifest.add_argument("--file", action="append", required=True, metavar="ROLE=PATH")
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--units", choices=["m", "cm"])
    manifest.add_argument("--forward", choices=["-Y", "+X"], default="-Y")
    inspect = commands.add_parser("inspect", help="Validate the manifest and every file hash")
    inspect.add_argument("manifest", type=Path)
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("pipeline", type=Path)
        if name == "run":
            command.add_argument("--work-dir", type=Path, required=True)
            command.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "manifest":
            files = [tuple(value.split("=", 1)) for value in args.file]
            if any(len(value) != 2 for value in files): raise ValueError("Use --file ROLE=PATH")
            coordinates = dict(units=args.units, up="+Z", forward=args.forward) if args.units else None
            result = contracts.write(args.output, args.kind, files, coordinates)
        elif args.command == "inspect":
            result = contracts.read(args.manifest)
        else:
            from pipeline import run, validate_plan
            data = validate_plan(json.loads(args.pipeline.read_text(encoding="utf-8")))
            if args.command == "plan": result = dict(status="planned", pipeline=data, execution="not_run")
            else:
                def interrupted(signum, frame): raise InterruptedError("Signal " + str(signum))
                signal.signal(signal.SIGTERM, interrupted)
                result = run(data, args.work_dir, args.resume)
        print("CHARACTER " + json.dumps(result))
        return 0
    except (KeyboardInterrupt, InterruptedError):
        print('CHARACTER {"status":"interrupted"}')
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print("CHARACTER " + json.dumps(dict(status="failed", error=str(error))))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
