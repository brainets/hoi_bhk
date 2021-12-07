"""Wrapping functions."""
from frites.io import is_numba_installed


USE_NUMBA = True

###############################################################################
# Numba wrapper

if USE_NUMBA:
    import numba
    def jit(signature=None, nopython=True, nogil=True, fastmath=True,  # noqa
            cache=True, **kwargs):
        return numba.jit(signature_or_function=signature, cache=cache,
                         nogil=nogil, fastmath=fastmath, nopython=nopython,
                         **kwargs)
else:
    def jit(*args, **kwargs):  # noqa
        def _jit(func):
            return func
        return _jit
