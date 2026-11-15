# Local checks

From this checkout run `git config --local core.hooksPath .githooks` once.
The same command applies to the modules checkout. Only this repository's hook
configuration changes. Git for Windows and WSL both run the versioned hook.

The hook checks staged whitespace, Python, Bash, PowerShell and JSON syntax,
changed module manifests, the release record and selected small offline contracts.
Checks use the index, including partially staged files. The subprocess budget is
28 seconds; usual commits should take under 10 seconds. Missing optional parsers
or schema packages warn and skip that check. Nothing installs or downloads tools.
Set `EVA_HOOK_PYTHON` to an existing Python executable when it is not on PATH;
alternatively save its path in this checkout with `git config --local eva.hookPython <path>`.
add existing native PowerShell to PATH for its parser. Bash on Windows is Git's
Bash: the WSL launch alias is never used.

Normal commits use these local checks. Bug fixes run their specific regression
manually. GPU, installation, recovery and end-to-end acceptance remain explicit
manual operations. GitHub Actions has been discontinued and is not a merge gate.
