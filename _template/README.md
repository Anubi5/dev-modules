# <name>
What it does, in one sentence. Upstream: <link>.

Use: `<command>` ...  Results go to `~/work/outputs/<name>/`.
Needs: GPU no/yes (through `ai gen`-style lock) · internet at install / at runtime · port (if a service).

Optional `state.txt`: one home-relative path per line for user data that `dev state save` must keep when the
distro is rebuilt (never model weights, programs or caches).
