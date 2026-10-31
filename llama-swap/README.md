# llama-swap (model serving)
Provides the core's **model-serving contract**: OpenAI-style API on 127.0.0.1:8080, `POST /api/models/unload`.
llama.cpp (CUDA) behind [llama-swap](https://github.com/mostlygeek/llama-swap); each model loads on first use and
unloads when idle. Chat page: http://127.0.0.1:8080/upstream/qwen3.5-9b/ (`dev chat`).

- Models: `models.json` (repo or search, quantisation, llama-server args). Add a model = add an entry, push, `dev mod update llama-swap`.
- Model files: `$MOD/models` (a copy also stays on D:\AI\models\llm for reinstalls).
- MoE offload: `MOE_CPU` from config.ps1, or tune: `cd ~/.local/share/modules/llama-swap && app/venv/bin/python tune_moe.py` (service stopped).
- `test.sh --full` = answers, speed and vision of every model (report in `$MOD/bench-*.md`).
