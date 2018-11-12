# -*- mode: python -*-

block_cipher = None


a = Analysis(['planetary_system_LRGB_aligner.py'],
             pathex=['D:\\SW-Development\\Python\\PlanetarySystemLRGBAligner\\Source'],
             binaries=[],
             datas=[( 'C:\\Python35\\Lib\\site-packages\\PyQt5\\Qt\\plugins\\platforms', 'platforms' )
                    ],
             hiddenimports=['PyQt5.sip'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='planetary_system_LRGB_aligner',
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='planetary_system_LRGB_aligner')