import numpy as np
from numba import types

ENABLE_CACHE = True
ENABLE_precompiled = True

# type_mode = "32"#有精度问题,不建议用
type_mode = "64"

type_dict = {
    "int_32": np.int32,
    "float_32": np.float32,
    "int_64": np.int64,
    "float_64": np.float64,
}

np_int = type_dict[f"int_{type_mode}"]
np_float = type_dict[f"float_{type_mode}"]


numba_dict = {
    "int_32": types.int32,
    "float_32": types.float32,
    "int_64": types.int64,
    "float_64": types.float64,
}


numba_int = numba_dict[f"int_{type_mode}"]
numba_float = numba_dict[f"float_{type_mode}"]
