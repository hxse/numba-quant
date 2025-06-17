import os
import ast
import importlib.util
import sys
import numpy as np
from numba import cuda, jit, njit, prange
import astunparse


def generate_dynamic_file_ast(
    origin_template_path, output_path, output_filename, all_decorator_configs
):
    if not os.path.exists(origin_template_path):
        print(f"[ERROR] 模板文件路径 '{origin_template_path}' 不存在")
        raise FileNotFoundError(f"模板文件 '{origin_template_path}' 不存在")

    with open(origin_template_path, "r", encoding="utf8") as f:
        content = f.read()

    print(f"[INFO] 解析模板文件: {origin_template_path}")
    tree = ast.parse(content)
    import_nodes = [node for node in tree.body if not isinstance(node, ast.FunctionDef)]
    original_functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    print(f"[INFO] 发现原始函数: {list(original_functions.keys())}")

    generated_functions = []
    mode_suffixes = {"jit": "_jit", "njit": "_njit", "cuda.jit": "_cuda_jit"}

    for original_func_name, original_func in original_functions.items():
        for mode, configs in all_decorator_configs.items():
            for config in configs:
                if original_func_name in config:
                    params = config[original_func_name]
                    signature = params.get("signature", "")
                    cache = params.get("cache", "")
                    nopython = params.get("nopython", "")
                    parallel = params.get("parallel", "")

                    decorator_args_pos = []
                    decorator_keywords = []
                    if signature:
                        decorator_args_pos.append(ast.Constant(value=signature))
                    if cache:
                        decorator_keywords.append(
                            ast.keyword(
                                arg="cache",
                                value=ast.Constant(value=cache.lower() == "true"),
                            )
                        )
                    if nopython:
                        decorator_keywords.append(
                            ast.keyword(arg="nopython", value=ast.Constant(value=True))
                        )
                    if parallel:
                        decorator_keywords.append(
                            ast.keyword(arg="parallel", value=ast.Constant(value=True))
                        )

                    decorator_id = ast.Name(
                        id=mode if mode != "cuda.jit" else "jit", ctx=ast.Load()
                    )
                    decorator_value = None
                    if mode == "cuda.jit":
                        cuda_decorator_keywords = (
                            [ast.keyword(arg="device", value=ast.Constant(value=True))]
                            if original_func_name in ["helper_func", "math_ops", "core"]
                            else []
                        ) + decorator_keywords
                        decorator_value = ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="cuda", ctx=ast.Load()),
                                attr="jit",
                                ctx=ast.Load(),
                            ),
                            args=decorator_args_pos,
                            keywords=cuda_decorator_keywords,
                        )
                    elif mode in ["jit", "njit"]:
                        decorator_value = ast.Call(
                            func=decorator_id,
                            args=decorator_args_pos,
                            keywords=decorator_keywords,
                        )

                    if decorator_value:
                        suffix = mode_suffixes.get(mode, f"_{mode.replace('.', '_')}")
                        new_func_name = original_func_name + (
                            suffix if not original_func_name.endswith(suffix) else ""
                        )
                        new_func = ast.FunctionDef(
                            name=new_func_name,
                            args=original_func.args,
                            body=list(original_func.body),
                            decorator_list=[decorator_value],
                            returns=original_func.returns,
                            type_comment=original_func.type_comment,
                        )

                        # 构建 rename_map，包含所有模板中的函数
                        rename_map = {
                            name: name + suffix for name in original_functions
                        }

                        print(f"[AST Rewrite] 处理函数: {new_func_name} (模式: {mode})")
                        print(f"[AST Rewrite] 重命名映射: {rename_map}")

                        # 记录原始函数体中的调用
                        print(f"[AST Rewrite] 扫描原始函数体中的调用:")
                        original_calls = []
                        for node in ast.walk(original_func):
                            if isinstance(node, ast.Call) and isinstance(
                                node.func, ast.Name
                            ):
                                original_calls.append(node.func.id)
                        print(f"[AST Rewrite]   发现调用: {original_calls}")

                        # 执行函数调用替换
                        print(f"[AST Rewrite] 执行函数调用替换:")
                        for node in ast.walk(new_func):
                            if isinstance(node, ast.Call) and isinstance(
                                node.func, ast.Name
                            ):
                                old_name = node.func.id
                                # 替换原始函数名为带后缀的版本
                                if old_name in rename_map:
                                    new_name = rename_map[old_name]
                                    print(
                                        f"[AST Rewrite]   将调用 '{old_name}' 替换为 '{new_name}'"
                                    )
                                    node.func.id = new_name
                                # 处理已替换的函数名，确保正确模式
                                elif any(
                                    old_name.endswith(s) for s in mode_suffixes.values()
                                ):
                                    base_name = old_name
                                    for s in mode_suffixes.values():
                                        if old_name.endswith(s):
                                            base_name = old_name[: -len(s)]
                                            break
                                    if base_name in original_functions:
                                        new_name = base_name + suffix
                                        print(
                                            f"[AST Rewrite]   修正调用 '{old_name}' 为 '{new_name}'"
                                        )
                                        node.func.id = new_name
                                    else:
                                        print(
                                            f"[AST Rewrite]   忽略非模板函数调用: '{old_name}'"
                                        )
                                else:
                                    print(
                                        f"[AST Rewrite]   忽略非模板函数调用: '{old_name}'"
                                    )

                        # 验证替换结果
                        print(f"[AST Rewrite] 验证替换结果:")
                        remaining_calls = []
                        for node in ast.walk(new_func):
                            if (
                                isinstance(node, ast.Call)
                                and isinstance(node.func, ast.Name)
                                and node.func.id in original_functions
                                and node.func.id != original_func_name
                            ):
                                remaining_calls.append(node.func.id)
                        if remaining_calls:
                            print(
                                f"[AST Rewrite]   错误: 仍存在未替换的调用: {remaining_calls}"
                            )
                        else:
                            print(f"[AST Rewrite]   成功: 所有调用已正确替换")

                        generated_functions.append(new_func)

    final_module_body = import_nodes + generated_functions

    processed_original = set()
    for mode, configs in all_decorator_configs.items():
        for config in configs:
            processed_original.update(config.keys())

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name not in processed_original:
            final_module_body.append(node)

    main_block = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    ]
    final_module_body.extend(main_block)
    final_module_body = [
        node for node in final_module_body if node not in main_block
    ] + main_block

    new_tree = ast.Module(body=final_module_body, type_ignores=[])
    print(f"[INFO] 生成 AST，包含 {len(final_module_body)} 个节点")
    modified_code = astunparse.unparse(new_tree)

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, output_filename)
    with open(output_file, "w", encoding="utf8") as f:
        f.write(modified_code)

    print(f"[AST Rewrite] 生成文件: {output_file}")
    return output_file


def run_dynamic_file(file_path, all_modes, delete_after_run=False):
    print(f"[INFO] 运行动态文件: {file_path}")
    try:
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"无法加载模块 '{file_path}'。")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if hasattr(module, "main") and callable(module.main):
            print("\n[运行] 执行动态模块中的 main 函数")
            module.main()
        else:
            print("\n[运行] 未找到可调用的 main 函数")

    except UnicodeDecodeError as e:
        print(f"[错误] 解码失败: 无法以 UTF-8 读取 '{file_path}'，错误: {e}")
        raise
    except Exception as e:
        print(f"[错误] 运行动态文件失败: {e}")
        raise
    finally:
        if delete_after_run and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"[清理] 已删除临时文件: {file_path}")
            except Exception as e:
                print(f"[清理] 删除临时文件 '{file_path}' 失败: {e}")


def main_script(
    origin_template_path="template.py",
    output_path="./",
    output_filename="temp_ast.py",
    all_decorator_configs=None,
    delete_after_run=False,
):
    if all_decorator_configs is None:
        all_decorator_configs = {
            "jit": [
                {"helper_func": {"signature": "float64(float64)", "cache": "True"}},
                {"math_ops": {"signature": "float32(float32)", "cache": "True"}},
                {"core": {"signature": "float32(float32)", "cache": "True"}},
                {"cpu_parallel_calc": {"nopython": "True", "cache": "False"}},
                {"run_cpu_jit": {}},
                {"run_cpu_njit": {}},
                {"run_gpu_device": {}},
            ],
            "njit": [
                {"helper_func": {"signature": "float64(float64)", "cache": "True"}},
                {"math_ops": {"signature": "float32(float32)", "cache": "True"}},
                {"core": {"signature": "float32(float32)", "cache": "True"}},
                {"cpu_parallel_calc": {"parallel": "True", "cache": "False"}},
                {"run_cpu_jit": {}},
                {"run_cpu_njit": {}},
                {"run_gpu_device": {}},
            ],
            "cuda.jit": [
                {"gpu_kernel_device": {}},
                {"helper_func": {"signature": "float32(float32)"}},
                {"math_ops": {"signature": "float32(float32)"}},
                {"core": {"signature": "float32(float32)"}},
                {"run_cpu_jit": {}},
                {"run_cpu_njit": {}},
                {"run_gpu_device": {}},
            ],
        }

    try:
        generated_file = generate_dynamic_file_ast(
            origin_template_path, output_path, output_filename, all_decorator_configs
        )
        modes_to_run = list(all_decorator_configs.keys())
        run_dynamic_file(generated_file, modes_to_run, delete_after_run)

    except Exception as e:
        print(f"[错误] 执行失败: {e}")
        raise


if __name__ == "__main__":
    all_configs = {
        "jit": [
            {"helper_func": {"signature": "float64(float64)", "cache": "True"}},
            {"math_ops": {"signature": "float32(float32)", "cache": "True"}},
            {"core": {"signature": "float32(float32)", "cache": "True"}},
            {"cpu_parallel_calc": {"nopython": "True", "cache": "False"}},
            {"run_cpu_jit": {}},
            {"run_cpu_njit": {}},
            {"run_gpu_device": {}},
        ],
        "njit": [
            {"helper_func": {"signature": "float64(float64)", "cache": "True"}},
            {"math_ops": {"signature": "float32(float32)", "cache": "True"}},
            {"core": {"signature": "float32(float32)", "cache": "True"}},
            {"cpu_parallel_calc": {"parallel": "True", "cache": "False"}},
            {"run_cpu_jit": {}},
            {"run_cpu_njit": {}},
            {"run_gpu_device": {}},
        ],
        "cuda.jit": [
            {"gpu_kernel_device": {}},
            {"helper_func": {"signature": "float32(float32)"}},
            {"math_ops": {"signature": "float32(float32)"}},
            {"core": {"signature": "float32(float32)"}},
            {"run_cpu_jit": {}},
            {"run_cpu_njit": {}},
            {"run_gpu_device": {}},
        ],
    }

    main_script(
        origin_template_path="template.py",
        output_path="./",
        output_filename="temp_ast.py",
        all_decorator_configs=all_configs,
        delete_after_run=False,
    )
