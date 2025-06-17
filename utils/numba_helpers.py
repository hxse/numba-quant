from numba import jit, njit, cuda
import time
from functools import wraps
from .constants import ENABLE_CACHE


# 工厂函数
def numba_factory(mode, signature=None):
    def decorator(func):
        if mode == "jit":
            if signature:
                return jit(signature, nopython=True, cache=ENABLE_CACHE)(func)
            return jit(nopython=True, cache=ENABLE_CACHE)(func)
        elif mode == "njit":
            if signature:
                return njit(signature, cache=ENABLE_CACHE)(func)
            return njit(cache=ENABLE_CACHE)(func)
        elif mode == "cuda":
            if signature:
                return cuda.jit(signature, device=True, cache=ENABLE_CACHE)(func)
            return cuda.jit(device=True, cache=ENABLE_CACHE)(func)
        else:
            return func

    return decorator


# 计时装饰器
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"函数 '{func.__name__}' 执行时间: {elapsed_time:.6f} 秒")
        return result

    return wrapper
