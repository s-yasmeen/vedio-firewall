"""Standalone launcher used by the PyInstaller Falling Walls build.

The executable runs Streamlit in-process. This avoids recursively invoking the
frozen executable via ``sys.executable -m streamlit``.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

from streamlit.web import bootstrap


def resource_root() -> Path:
    """Return repository root in source mode or PyInstaller extraction root."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def choose_port() -> int:
    requested = os.environ.get("TAPF_DEMO_PORT", "").strip()
    if requested:
        port = int(requested)
        if not 1 <= port <= 65535:
            raise ValueError("TAPF_DEMO_PORT must be between 1 and 65535")
        return port
    return free_port()


def main() -> int:
    root = resource_root()
    app = root / "demo" / "falling_walls_demo.py"
    if not app.exists():
        raise FileNotFoundError(f"Demo application not found: {app}")

    port = choose_port()
    url = f"http://127.0.0.1:{port}"

    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    if os.environ.get("TAPF_DEMO_NO_BROWSER", "0") != "1":
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()

    flag_options = {
        "server.headless": True,
        "server.address": "127.0.0.1",
        "server.port": port,
        "browser.gatherUsageStats": False,
    }

    # Blocks until the user closes the app/process.
    bootstrap.run(str(app), False, [], flag_options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
