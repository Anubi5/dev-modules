"""Bounded chat-service operations; callers own the GPU lease and restoration."""
import subprocess


def chat_state() -> str:
    """Read a stable service state, refusing transitions or unavailable systemd."""
    result = subprocess.run(
        ["systemctl", "--user", "show", "llama-swap", "-p", "ActiveState", "--value"],
        capture_output=True, text=True, check=True, timeout=30,
    )
    state = result.stdout.strip()
    if state not in ("active", "inactive"):
        raise RuntimeError(f"Chat state is {state!r}; inspect ai logs llama-swap before retrying")
    return state


def stop_chat() -> None:
    """Stop and verify the server so HTTP requests cannot reload its weights."""
    subprocess.run(["systemctl", "--user", "stop", "llama-swap"], check=True, timeout=30)
    if chat_state() != "inactive":
        raise RuntimeError("Chat did not stop; refusing GPU inference; inspect ai logs llama-swap")


def start_chat() -> None:
    """Restore a previously active server; failure must reach the job record."""
    subprocess.run(["systemctl", "--user", "start", "llama-swap"], check=True, timeout=30)
    if chat_state() != "active":
        raise RuntimeError("Chat did not restart; run ai start llama-swap and inspect its logs")
