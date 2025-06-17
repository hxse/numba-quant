# 可以在 Numba CUDA 中使用的 math 模块函数 (通常是标量函数)：
  * 根据 Numba 的 CUDA 文档和社区经验，以下 math 模块的函数通常可以在 GPU 设备代码（@cuda.jit(device=True) 和 CUDA kernel 中）中使用：
```
math.exp()
math.log()
math.log10()
math.pow()
math.sqrt()
math.sin()
math.cos()
math.tan()
math.asin()
math.acos()
math.atan()
math.atan2()
math.floor()
math.ceil()
math.trunc()
math.fabs()
math.fmod()
math.degrees()
math.radians()
math.hypot()
常量: math.pi, math.e
```
# 可以在 Numba CUDA 中使用的 numpy 模块函数 (通常是 ufuncs 或规约函数)：
  * 根据 Numba 的 CUDA 文档，以下 numpy 模块的函数通常可以在 GPU 设备代码和 CUDA kernel 中作为 ufuncs (element-wise) 或规约函数使用：
```
Ufuncs (element-wise)：
np.exp()
np.log()
np.log10()
np.power()
np.sqrt()
np.sin()
np.cos()
np.tan()
np.arcsin()
np.arccos()
np.arctan()
np.arctan2()
np.floor()
np.ceil()
np.trunc()
np.fabs() (np.abs() 也可以)
np.fmod()
算术运算符 (+, -, *, /, //, %, **)
比较运算符 (==, !=, <, >, <=, >=)
逻辑运算符 (&, |, ^, ~)
规约函数：
np.sum()
np.prod()
np.min()
np.max()
np.all()
np.any()
np.mean() (需要注意，Numba CUDA 对 mean 的支持可能有一些限制，尤其是在原子操作不适用的情况下)
np.std()
np.var()
```
# 注意：
  * 这个列表并非详尽无遗，Numba 对 NumPy 的 CUDA 支持在不断发展。
  * 对于 NumPy 的规约函数，Numba CUDA 通常需要你明确指定轴 (axis) 参数，并且并非所有规约操作都得到完全优化。
  * NumPy 中创建数组、改变形状等操作通常在主机端（CPU）进行，然后再将数组转移到 GPU 上。
