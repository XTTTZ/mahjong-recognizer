"""Run a saved PowerShell script over the existing trusted Windows SSH alias.

Example: python infra/windows/ssh_ps.py infra/windows/setup.ps1
No credentials are copied or host trust checks bypassed.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path)
    parser.add_argument("--host", default="win")
    args = parser.parse_args()
    script = "$ProgressPreference='SilentlyContinue';\n" + args.script.read_text(encoding="utf-8-sig")
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8",
         args.host, "powershell", "-NoProfile", "-EncodedCommand", encoded],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
