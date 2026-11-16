# reverse-tools

Static reverse engineering: agents install REA/Ghidra and ILSpy on first use to inspect files.

Skill only: tells agents how to install [REA](https://github.com/morluto/rea) with Ghidra, ILSpy (.NET / Unity) and other free tools in
their own folders (no root) when asked to inspect or decompile a file. Nothing is installed until then.


## Local regression tests

Use `python3 -B tests/run.py --module reverse-tools --suite fast` from the repository root; use `--suite integration` explicitly for disposable multi-component fixtures. See the [shared test policy](../tests/README.md) for categories, prerequisites and skips. The module-root `test.sh` remains an explicit installer health check, outside the default runner.
