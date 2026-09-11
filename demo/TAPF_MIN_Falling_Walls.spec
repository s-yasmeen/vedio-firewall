# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

REPO_ROOT = Path.cwd().resolve()
SPEC_DIR = REPO_ROOT / 'demo'

streamlit_datas, streamlit_binaries, streamlit_hidden = collect_all('streamlit')
np_datas, np_binaries, np_hidden = collect_all('numpy')

datas = streamlit_datas + np_datas + [
    (str(SPEC_DIR / 'falling_walls_demo.py'), 'demo'),
    (str(REPO_ROOT / 'tapf'), 'tapf'),
]
binaries = streamlit_binaries + np_binaries
hiddenimports = streamlit_hidden + np_hidden + [
    'tapf.formal_privacy',
]

a = Analysis(
    [str(SPEC_DIR / 'standalone_launcher.py')],
    pathex=[str(REPO_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
