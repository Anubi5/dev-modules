#!/usr/bin/env bash
## @brief Stops the model server (the runner removes the rest).
systemctl --user stop llama-swap 2>/dev/null; true
