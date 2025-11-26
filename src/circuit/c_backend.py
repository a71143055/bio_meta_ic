import numpy as np
from cffi import FFI
import os

class FastBackend:
    def __init__(self):
        self.ffi = FFI()
        self.ffi.cdef("""
            void fast_update(float* state, float* inputs, float* leak, float* bias, float dt, int n);
        """)
        libpath = os.path.join(os.path.dirname(__file__), "..", "..", "csrc", "libfast_update.so")
        if not os.path.exists(libpath):
            raise RuntimeError("C backend not built. Run `make -C csrc`.")
        self.lib = self.ffi.dlopen(libpath)

    def fast_update(self, state, inputs, leak, bias, dt):
        n = state.shape[0]
        out = np.array(state, dtype=np.float32, copy=True)
        self.lib.fast_update(
            self.ffi.cast("float *", out.ctypes.data),
            self.ffi.cast("float *", inputs.astype(np.float32).ctypes.data),
            self.ffi.cast("float *", leak.astype(np.float32).ctypes.data),
            self.ffi.cast("float *", bias.astype(np.float32).ctypes.data),
            float(dt),
            int(n)
        )
        return out
