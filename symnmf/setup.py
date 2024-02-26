from setuptools import Extension, setup

module = Extension("mysymnmf",
                  sources=['matrix.c', 'symnmf.c','symnmfmodule.c'])
setup(name='mysymnmf',
     version='1.0',
     description='Python wrapper for custom C extension',
     ext_modules=[module])