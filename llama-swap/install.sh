#!/usr/bin/env bash
## @brief Builds llama.cpp (CUDA) and fetches llama-swap into $MOD/app, gets the models in models.json, writes llama-swap.yaml.
set -euo pipefail
cd "$MOD"; A=$MOD/app; mkdir -p "$A/bin" "$MOD/models"
export PATH=/usr/local/cuda/bin:$PATH
if ! "$A/bin/llama-server" --version >/dev/null 2>&1; then   # missing, or moved binaries that no longer start
  echo "  building llama.cpp with CUDA - 15-30 min, no output meanwhile"
  [ -d "$A/src/llama.cpp" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$A/src/llama.cpp"
  cmake -S "$A/src/llama.cpp" -B "$A/src/llama.cpp/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-89}" \
        -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null   # static: no library paths that break when moved
  cmake --build "$A/src/llama.cpp/build" -j "$(nproc)" --target llama-server llama-bench >/dev/null
  rm -f "$A"/bin/lib*.so*                                       # old shared-library build
  cp -a "$A/src/llama.cpp/build/bin/llama-server" "$A/src/llama.cpp/build/bin/llama-bench" "$A/bin/"
  rm -rf "$A/src/llama.cpp/build"
fi
"$A/bin/llama-server" --version >/dev/null 2>&1 || { echo "llama-server does not start:"; "$A/bin/llama-server" --version 2>&1 | tail -5; exit 1; }
if [ ! -x "$A/bin/llama-swap" ]; then
  url=$(curl -s https://api.github.com/repos/mostlygeek/llama-swap/releases/latest | jq -r '.assets[].browser_download_url' \
        | grep -i linux | grep -Ei 'amd64|x86_64' | grep -E '\.tar\.gz$' | head -1)
  curl -sL "$url" | tar -xz -C "$A/bin" llama-swap
fi
[ -x "$A/venv/bin/python" ] || python3 -m venv "$A/venv"
"$A/venv/bin/pip" install -q -U huggingface_hub requests
"$A/venv/bin/python" resolve_models.py --config models.json --out "$MOD/models" --seed "${MODELS_MOUNT:-/models}/llm"
[ -n "${MOE_CPU:-}" ] && [ ! -f "$MOD/moe.conf" ] && echo "$MOE_CPU" > "$MOD/moe.conf"
"$A/venv/bin/python" make_config.py >/dev/null
echo "llama-swap ready: $(grep -c 'cmd:' "$MOD/llama-swap.yaml") models"
