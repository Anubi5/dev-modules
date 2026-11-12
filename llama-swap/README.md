# llama-swap

Local model serving through llama.cpp and llama-swap. Provides the OpenAI-style API and `POST /api/models/unload` contract used by cooperating modules. Models load on first use and unload when idle; query the running server's `/v1/models` for actual IDs.

Model declarations and server profiles live in `models.json`; `LLAMA_CPP_TAG` in `install.sh` pins the backend build. Use the [module workflow](../README.md) for reviewed updates/acquisition. Weights are stored under `$MODELS_MOUNT/llm/<id>` with file records under `$MOD/models/models.json`. Profile retirement does not delete weights or let stale cached records re-enable a profile.

Daily and agent profiles can share underlying files while retaining separate context/settings. Default declarations avoid local weight copies; do not enable `local_copy` merely to troubleshoot a service. Any verified duplicate cleanup must retain the external original. `MOE_CPU` tunes expert offload; `fit` delegates placement to the backend. Inspect available RAM/VRAM and stop the service before deliberate tuning with `tune_moe.py`.

`test.sh --full` exercises configured answers, speed and vision and writes a benchmark report. Short warm samples do not establish long-context performance. The embeddings profile, when present, serves `/v1/embeddings`; a chat model is not automatically an embedding model.

```bash
img-ask "$HOME/work/inputs/render.png" 'Is the red teapot visible?' --expect yes
```

`img-ask` uses the configured local vision model. `--expect` returns 3 if its word check fails; semantic answers remain fallible and do not replace visual inspection. Keep API services local rather than treating a placeholder client API key as authentication.
