# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('streamlit')
np_datas, np_binaries, np_hidden = collect_all('numpy')
datas += np_datas
binaries += np_binaries
hiddenimports += np_hidden

a = Analysis(
    ['demo/standalone_launcher.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas + [('demo/falling_walls_demo.py', 'demo')],
    hiddenimports=hiddenimports + ['tapf.formal_privacy'],
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
