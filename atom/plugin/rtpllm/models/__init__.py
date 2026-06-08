# Models are registered lazily in base_model_wrapper.py via _try_register()
# to avoid import errors when dependencies are missing.
from . import base_model_wrapper as _bm  # noqa: F401 — triggers registration

__all__ = []
