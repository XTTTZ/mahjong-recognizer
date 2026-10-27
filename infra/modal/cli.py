"""Run the installed Modal CLI, optionally bypassing API proxies for this process.

Usage: python infra/modal/cli.py [--direct] app list --json
The default path respects the existing environment and macOS system proxy.
"""
import os
import runpy
import sys


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--direct":
        os.environ["MODAL_DISABLE_API_PROXY"] = "1"
        del sys.argv[1]
    runpy.run_module("modal", run_name="__main__", alter_sys=True)
