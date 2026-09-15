"""Backend TTS-дашборда.

Импорт `config` здесь гарантирует, что лимиты потоков (OMP/OpenBLAS/MKL/VecLib)
и `PYTORCH_ENABLE_MPS_FALLBACK` выставлены до первого `import numpy`/`torch`/
`librosa` в любом подмодуле — независимо от порядка импортов внутри них.
"""

from . import config  # noqa: F401
