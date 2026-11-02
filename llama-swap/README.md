# llama-swap (model serving)
Provides the core's **model-serving contract**: OpenAI-style API on 127.0.0.1:8080, `POST /api/models/unload`.
llama.cpp (CUDA) behind [llama-swap](https://github.com/mostlygeek/llama-swap); each model loads on first use and
unloads when idle. Chat page: http://127.0.0.1:8080/upstream/qwen3.5-9b/ (`dev chat`).

- Models: `models.json` (repo or search, quantisation, llama-server args). Add a model = add an entry, push, `dev mod update llama-swap`.
- Model files: `D:\AI\models\llm\<id>\` only (read-only at `/models/llm`); `$MOD/models/models.json` records which files
  each model uses. Downloads need D: writable, so run `dev mod add|update llama-swap` from Windows.
  Exception: `"local_copy": true` (the 35B) also keeps a copy inside the distro - it reads its MoE experts from the file
  while answering, ~2.5x slower from D:.
- MoE offload: `MOE_CPU` from config.ps1, or tune: `cd ~/.local/share/modules/llama-swap && app/venv/bin/python tune_moe.py` (service stopped).
- `test.sh --full` = answers, speed and vision of every model (report in `$MOD/bench-*.md`).
- llama.cpp is built from a pinned release (`LLAMA_CPP_TAG` in install.sh); raising it rebuilds on the next update (15-30 min).
- `"fit": true` lets llama.cpp split a model between GPU and RAM itself (`--fit on`) - for models not tuned yet.

## Search and candidate models (added 2 Nov 2026)
| id | What | Notes |
|---|---|---|
| `embed-gemma2` | EmbeddingGemma 2 (Q8_0, ~310 MB): text embeddings at `POST /v1/embeddings` for local search | Apache 2.0; bench checks a related sentence ranks above an unrelated one |
| `lfm2.5-24b` | LFM2.5-24B-A2B (Q4_K_M, 14.5 GB, 2.45B active): strong tool use and instruction following, text only | candidate for Hermes; licence "lfm1.0" (read its LICENSE file before any commercial use) |
| `nemotron-bolt` | Nemotron 3.5 Bolt 74B-A9B, native 2-bit (22 GB + 1.7 GB vision) | candidate for Hermes with vision; NVIDIA model licence; tight in 24 GB RAM and read from D: (slow first load) |

Candidates appear in the bench report as "(candidate)" and never fail `dev test`; they are compared with the 35B on the
Hermes evaluation before one replaces it (`AGENT_MODEL` in sandbox.env).

## Checking images: `img-ask` (vision-check model)
`img-ask <image> "<question>" [--expect WORD]` asks WeVisOne-VL 2B (Tencent, Apache 2.0, ~2.6 GB with its vision part)
what an image shows: "Is there a red cube?", "How many people?", "Is the text readable?". Agents use it next to
`img-check` (pixel numbers) to prove a render shows what was asked. `--expect yes` exits 3 when the answer
does not contain the word. The model loads in seconds and unloads after 2 idle minutes.
