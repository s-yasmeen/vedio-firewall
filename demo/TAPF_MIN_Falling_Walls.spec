# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

REPO_ROOT = Path.cwd().resolve()
ENTRY = REPO_ROOT / 'demo' / 'standalone_web_demo.py'

a = Analysis(
    [str(ENTRY)],
    pathex=[str(REPO_ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=['tapf.formal_privacy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='TAPF-MIN-Falling-Walls',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
