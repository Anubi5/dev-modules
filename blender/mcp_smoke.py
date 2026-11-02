#!/usr/bin/env python3
"""!
@file mcp_smoke.py
@brief End-to-end check of the official Blender MCP path, the way an agent uses it: starts the pinned Blender Lab MCP
       server (stdio), initializes, lists the tools, reads the scene and takes a window screenshot (needs the GUI
       Blender on the virtual display). Prints one OK/FAILED line per step; exit 0 only if all passed.
Usage: mcp_smoke.py <uv tool source>
"""
import json
import subprocess
import sys
import time


def main():
    """! @brief Runs the steps. @return exit code """
    p = subprocess.Popen(["uv", "tool", "run", "--from", sys.argv[1], "blender-mcp", "--transport", "stdio"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                         env={**__import__("os").environ, "BLENDER_MCP_HOST": "127.0.0.1", "BLENDER_MCP_PORT": "9876"})
    n = [0]

    def call(method, params=None, notify=False):
        n[0] += 1
        msg = {"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {})}
        if not notify:
            msg["id"] = n[0]
        p.stdin.write(json.dumps(msg) + "\n"); p.stdin.flush()
        if notify:
            return None
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
        call("notifications/initialized", notify=True)
        print(f"  MCP initialize: {'OK' if 'result' in r else 'FAILED ' + str(r)[:200]}"); ok &= "result" in r
        tools = [t["name"] for t in call("tools/list").get("result", {}).get("tools", [])]
        print(f"  MCP tools: {len(tools)} ({', '.join(t for t in tools if 'summary' in t or 'screenshot' in t)})"); ok &= len(tools) > 10
        r = call("tools/call", {"name": "get_objects_summary", "arguments": {}})
        txt = json.dumps(r.get("result", r))[:160]
        good = "result" in r and not r["result"].get("isError")
        print(f"  scene read (get_objects_summary): {'OK' if good else 'FAILED'} {txt}"); ok &= good
        shot = next((t for t in tools if "screenshot" in t), None)
        if shot:
            r = call("tools/call", {"name": shot, "arguments": {}})
            res = r.get("result", {})
            img = any(c.get("type") == "image" for c in res.get("content", [])) and not res.get("isError")
            print(f"  screenshot ({shot}): {'OK' if img else 'FAILED ' + json.dumps(res)[:200]}"); ok &= img
    finally:
        p.terminate()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
