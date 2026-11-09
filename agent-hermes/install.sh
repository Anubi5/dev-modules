#!/usr/bin/env bash
## @brief Installs or adopts the pinned Hermes and applies the reviewed local profile (safe to run again). Runs as dev.
set -euo pipefail
SETUP=${EVA_SETUP:-/opt/ai/setup}   # override only for tests
INSTALLER_SHA256=f4cdb0e20f1aca2c3871a76b962fd4db61510f549cf973759c753ac13349beb5   # scripts/install.sh at the pin
SRC="${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
for f in hermes_runtime.py hermes_local_setup.py hermes_profile_check.py; do
  [ -f "$SETUP/$f" ] || { echo "  CHECK $SETUP/$f missing - update the core first (dev update)"; exit 1; }; done
PIN=$(python3 -B -c "import sys; sys.path.insert(0, '$SETUP'); import hermes_runtime as r; print(r.PINNED_REVISION)")
export HERMES_DISABLE_LAZY_INSTALLS=1 PATH="$HOME/.local/bin:$PATH"

if python3 -B "$SETUP/hermes_runtime.py" check >/dev/null 2>&1; then
  echo "  Hermes source is the pinned ${PIN:0:7}: adopting the existing installation (data untouched)"
elif [ -e "$SRC" ]; then
  echo "  CHECK $SRC exists but is not the clean pinned Hermes ${PIN:0:7} (other commit, local edits or no launcher)."
  echo "        Nothing was changed. Review it; to reinstall, move that folder aside yourself and run this again."
  exit 1
else
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  echo "  Hermes ${PIN:0:7}: official installer from that exact commit"
  curl -fsSL "https://raw.githubusercontent.com/NousResearch/hermes-agent/$PIN/scripts/install.sh" -o "$tmp/install.sh"
  echo "$INSTALLER_SHA256  $tmp/install.sh" | sha256sum -c --quiet \
    || { echo "  CHECK installer hash differs from the reviewed one - nothing installed"; exit 1; }
  bash "$tmp/install.sh" --commit "$PIN" --non-interactive --skip-browser --skip-computer-use </dev/null
fi
python3 -B "$SETUP/hermes_runtime.py" install
python3 -B "$SETUP/hermes_local_setup.py" --apply --reviewed-revision "$PIN" >/dev/null \
  || { echo "  CHECK Hermes profile could not be applied (config.yaml restored from its backup)"; exit 1; }
soul="${HERMES_HOME:-$HOME/.hermes}/SOUL.md"   # the reviewed local section is appended once; your own text stays
[ ! -f "$SETUP/hermes-local-SOUL.md" ] || grep -qF "$(sed -n 1p "$SETUP/hermes-local-SOUL.md")" "$soul" 2>/dev/null \
  || { printf '\n' >> "$soul"; cat "$SETUP/hermes-local-SOUL.md" >> "$soul"; echo "  local profile section added to SOUL.md"; }
python3 -B "$SETUP/hermes_profile_check.py"
echo "Hermes ready ($(hermes --version 2>/dev/null | head -1))"
