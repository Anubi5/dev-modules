#!/usr/bin/env bash
## @brief Builds llama.cpp (CUDA) and fetches llama-swap into $MOD/app, gets the models in models.json (on D:), writes llama-swap.yaml.
set -euo pipefail
cd "$MOD"; A=$MOD/app; mkdir -p "$A/bin" "$MOD/models"
export PATH=/usr/local/cuda/bin:$PATH
LLAMA_CPP_TAG=b12040        # pinned llama.cpp release (2 Nov 2026: EmbeddingGemma 2, LFM2.5, Nemotron 3.5 Bolt + vision); raise deliberately
if ! "$A/bin/llama-server" --version >/dev/null 2>&1 \
   || [ "$(cat "$A/llama.cpp.tag" 2>/dev/null)" != "$LLAMA_CPP_TAG" ]; then   # missing, broken, or a new pinned release
  echo "  building llama.cpp $LLAMA_CPP_TAG with CUDA - 15-30 min, no output meanwhile"
  rm -rf "$A/src/llama.cpp"
  git clone -q --depth 1 --branch "$LLAMA_CPP_TAG" https://github.com/ggml-org/llama.cpp "$A/src/llama.cpp"
  cmake -S "$A/src/llama.cpp" -B "$A/src/llama.cpp/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-89}" \
        -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null   # static: no library paths that break when moved
  cmake --build "$A/src/llama.cpp/build" -j "$(nproc)" --target llama-server llama-bench >/dev/null
  rm -f "$A"/bin/lib*.so*                                       # old shared-library build
  cp -a "$A/src/llama.cpp/build/bin/llama-server" "$A/src/llama.cpp/build/bin/llama-bench" "$A/bin/"
  rm -rf "$A/src/llama.cpp/build"
  echo "$LLAMA_CPP_TAG" > "$A/llama.cpp.tag"
fi
"$A/bin/llama-server" --version >/dev/null 2>&1 || { echo "llama-server does not start:"; "$A/bin/llama-server" --version 2>&1 | tail -5; exit 1; }
if [ ! -x "$A/bin/llama-swap" ]; then
  url=$(curl -s https://api.github.com/repos/mostlygeek/llama-swap/releases/latest | jq -r '.assets[].browser_download_url' \
        | grep -i linux | grep -Ei 'amd64|x86_64' | grep -E '\.tar\.gz$' | head -1)
  curl -sL "$url" | tar -xz -C "$A/bin" llama-swap
fi
[ -x "$A/venv/bin/python" ] || python3 -m venv "$A/venv"
"$A/venv/bin/pip" install -q -U huggingface_hub requests
# model files live on D: (read-only at /models); D: is writable at /models-rw only while Windows runs "dev mod add|update"
"$A/venv/bin/python" resolve_models.py --config models.json --state "$MOD/models" \
  --ro "${MODELS_MOUNT:-/models}/llm" --rw "${MODELS_RW:-/models-rw}/llm"
[ -n "${MOE_CPU:-}" ] && [ ! -f "$MOD/moe.conf" ] && echo "$MOE_CPU" > "$MOD/moe.conf"
"$A/venv/bin/python" make_config.py >/dev/null
echo "llama-swap ready: $(grep -c 'cmd:' "$MOD/llama-swap.yaml") models"
