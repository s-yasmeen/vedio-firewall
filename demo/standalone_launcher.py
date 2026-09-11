"""Standalone launcher used by the PyInstaller Falling Walls build."""
from __future__ import annotations
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def resource_root() -> Path:
    return Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def main() -> int:
    root = resource_root()
    app = root / 'demo' / 'falling_walls_demo.py'
    if not app.exists():
        raise FileNotFoundError(f'Demo application not found: {app}')
    port = free_port()
    env = os.environ.copy()
    env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', str(app),
        '--server.headless=true',
        f'--server.port={port}',
        '--server.address=127.0.0.1',
        '--browser.gatherUsageStats=false',
    ]
    proc = subprocess.Popen(cmd, cwd=str(root), env=env)
    url = f'http://127.0.0.1:{port}'
    time.sleep(2.0)
    webbrowser.open(url)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
