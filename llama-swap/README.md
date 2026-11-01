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
