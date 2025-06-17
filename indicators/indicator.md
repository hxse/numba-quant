# 添加指标守则
* 要遵守文件结构,不可以添加函数,也不可以添加装饰器,也不可以改变函数参数,要保证文件结构和其他指标文件一模一样, 写法参考indicators\bbands.py和indicators\rsi.py
* 当前的文件结构已经是能够被我的numba工厂函数编译的文件结构了,并且可以支持jit/njit/cuda,不需要你去添加什么cuda装饰器,不需要你去更改文件结构,你只需要保持和其他文件一模一样的文件结构就好
* 语法上要符合cuda的严格写法, 不能用乱七八糟的numpy方法
* np.zero之类的创建数组方法都不能用,直接预定义数组后缀是_result,然后在函数结尾处返回这些_result数组
* np.isnan不能在cuda中使用, 用代替写法, ==或!=
* 类似于bbands可以分离出sma,rsi可以分离出rma,添加新指标也要考虑分离,分离后是新文件,并且文件结构都要保持一样
* 添加好指标后,更改indicators\indicator_selector.py,注意要保持好原有写法范式
* 然后在Test\test_indicators文件夹内部添加指标的测试文件,写法参考Test\test_indicators\test_bbands.py和Test\test_indicators\test_rsi.py
* 然后在Test\test_config\test_config_accuracy.py文件中更新config和_test_indicator_config_accuracy
