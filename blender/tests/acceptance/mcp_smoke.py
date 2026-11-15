#!/usr/bin/env python3
"""Check the prepared official Blender MCP path without resolving dependencies.

Usage: mcp_smoke.py <prepared-command>

The caller provides the module-owned prepared executable. The smoke reads the
scene and takes a screenshot; it requires a separately started Blender bridge.
"""
import json
import subprocess
import sys
import time
from typing import Any


def main() -> int:
    """Return zero only after initialization, scene read and screenshot pass."""
    p = subprocess.Popen([sys.argv[1], "--transport", "stdio"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                         env={**__import__("os").environ, "BLENDER_MCP_HOST": "127.0.0.1", "BLENDER_MCP_PORT": "9876"})
    assert p.stdin is not None and p.stdout is not None
    n = [0]

    def call(method: str, params: dict[str, Any] | None = None, notify: bool = False) -> dict[str, Any]:
        """Send one protocol request and return an error if the child exits."""
        n[0] += 1
        msg = {"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {})}
        if not notify:
            msg["id"] = n[0]
        try:
            p.stdin.write(json.dumps(msg) + "\n"); p.stdin.flush()
        except BrokenPipeError:
            return {"error": "MCP process closed its input; see child stderr"}
        if notify:
            return {}
        end = time.time() + 120
        while time.time() < end:
            line = p.stdout.readline()
            if not line:
                break
            r = json.loads(line)
            if r.get("id") == n[0]:
                return r
        return {"error": "no answer"}

    ok = True
    try:
        r = call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "dev-test", "version": "1"}})
        print(f"  MCP initialize: {'OK' if 'result' in r else 'FAILED ' + str(r)[:200]}"); ok &= "result" in r
        if not ok:
            return 1
        call("notifications/initialized", notify=True)
        tools = [t["name"] for t in call("tools/list").get("result", {}).get("tools", [])]
        print(f"  MCP tools: {len(tools)} ({', '.join(t for t in tools if 'summary' in t or 'screenshot' in t)})"); ok &= len(tools) > 10
        r = call("tools/call", {"name": "get_objects_summary", "arguments": {}})
        txt = json.dumps(r.get("result", r))[:160]
        good = "result" in r and not r["result"].get("isError")
        print(f"  scene read (get_objects_summary): {'OK' if good else 'FAILED'} {txt}"); ok &= good
        shot = "get_screenshot_of_window_as_image" if "get_screenshot_of_window_as_image" in tools else None   # no arguments needed
        if not shot:
            print("  screenshot: FAILED (get_screenshot_of_window_as_image not offered)"); ok = False
        if shot:
            r = call("tools/call", {"name": shot, "arguments": {}})
            res = r.get("result", {})
            img = any(c.get("type") == "image" for c in res.get("content", [])) and not res.get("isError")
            print(f"  screenshot ({shot}): {'OK' if img else 'FAILED ' + json.dumps(res)[:200]}"); ok &= img
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
