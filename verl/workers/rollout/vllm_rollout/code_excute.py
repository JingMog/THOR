import os
import io
import regex
import pickle
import traceback
import copy
import datetime
import dateutil.relativedelta
import multiprocess
from multiprocess import Pool
from typing import Any, Dict, Optional
from pebble import ProcessPool
from tqdm import tqdm
from concurrent.futures import TimeoutError
from functools import partial
from timeout_decorator import timeout
from contextlib import redirect_stdout, redirect_stderr
import re
import warnings


CODE_BLOCK_PATTERN = re.compile(r"```python(.*?)```", re.DOTALL)  # 使用 re.DOTALL 标志 [[1]]

class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        if regex.search(r"(\s|^)?input\(", code_piece) or regex.search(
            r"(\s|^)?os.system\(", code_piece
        ):
            raise RuntimeError()
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)

    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v

    @property
    def answer(self):
        return self._global_vars["answer"]


class DateRuntime(GenericRuntime):
    GLOBAL_DICT = {
        "datetime": datetime.datetime,
        "timedelta": dateutil.relativedelta.relativedelta,
        "relativedelta": dateutil.relativedelta.relativedelta,
    }


class CustomDict(dict):
    def __iter__(self):
        return list(super().__iter__()).__iter__()


class ColorObjectRuntime(GenericRuntime):
    GLOBAL_DICT = {"dict": CustomDict}


class PythonExecutor:
    def __init__(
        self,
        runtime: Optional[Any] = None,
        get_answer_symbol: Optional[str] = None,
        get_answer_expr: Optional[str] = None,
        get_answer_from_stdout: bool = True,
        timeout_length: int = 10,
    ) -> None:
        self.runtime = runtime if runtime else GenericRuntime()
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.pool = Pool(multiprocess.cpu_count())
        self.timeout_length = timeout_length

    def process_generation_to_code(self, gens: str):
        return [g.split("\n") for g in gens]

    @staticmethod
    def execute(
        code,
        get_answer_from_stdout=True,
        runtime=None,
        answer_symbol=None,
        answer_expr=None,
        timeout_length=10,
    ):
        try:
            # 屏蔽警告
            warnings.filterwarnings("ignore")

            # 屏蔽标准错误输出
            # error_io = io.StringIO()

            if get_answer_from_stdout:
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    timeout(timeout_length)(runtime.exec_code)("\n".join(code))
                program_io.seek(0)
                result = program_io.read()
            if answer_symbol:
                timeout(timeout_length)(runtime.exec_code)(code)
                result = runtime._global_vars[answer_symbol]
            if answer_expr:
                timeout(timeout_length)(runtime.exec_code)(code)
                result = timeout(timeout_length)(runtime.eval_code)(answer_expr)
            if result == "":
                timeout(timeout_length)(runtime.exec_code)(code[:-1])
                result = timeout(timeout_length)(runtime.eval_code)(code[-1])
            report = "Done"
            str(result)
            pickle.dumps(result)  # serialization check
        except:
            result = ""
            report = traceback.format_exc().split("\n")[-2]
        return result, report

    def apply(self, code):
        return self.batch_apply([code])[0]

    @staticmethod
    def truncate(s, max_length=400):
        half = max_length // 2
        if len(s) > max_length:
            s = s[:half] + "..." + s[-half:]
        return s

    def batch_apply(self, batch_code):
        all_code_snippets = self.process_generation_to_code(batch_code)

        timeout_cnt = 0
        all_exec_results = []
        max_workers = min(len(all_code_snippets), 8)
        with ProcessPool(
            #max_workers=min(len(all_code_snippets), os.cpu_count())
            max_workers=max_workers
        ) as pool:
            executor = partial(
                self.execute,
                get_answer_from_stdout=self.get_answer_from_stdout,
                runtime=self.runtime,
                answer_symbol=self.answer_symbol,
                answer_expr=self.answer_expr,
                timeout_length=self.timeout_length,  # this timeout not work
            )
            future = pool.map(executor, all_code_snippets, timeout=self.timeout_length)
            iterator = future.result()

            # if len(all_code_snippets) > 10000:
            #     progress_bar = tqdm(total=len(all_code_snippets), desc="Execute")
            # else:
            progress_bar = None

            while True:
                try:
                    result = next(iterator)
                    all_exec_results.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    all_exec_results.append(("", "Timeout Error"))
                    timeout_cnt += 1
                except Exception as error:
                    exit()
                if progress_bar is not None:
                    progress_bar.update(1)

            if progress_bar is not None:
                progress_bar.close()

        batch_results = []
        for code, (res, report) in zip(all_code_snippets, all_exec_results):
            # post processing
            res, report = str(res).strip(), str(report).strip()
            res, report = self.truncate(res), self.truncate(report)
            batch_results.append((res, report))
        return batch_results


def _test():
    # batch_code = []
    batch_code = [
        """import sympy as sp

# Define the variables
V, S = sp.symbols('V S')

# Define the equation
equation = V + S - 108

# Solve the equation
solution = sp.solve(equation, (V, S))
print(solution)""",
        "\nimport sympy as sp\n\n# Define the variable\na = sp.symbols(\'a\')\n\n# Define the function f(a) = (3a^4 + 1) / (2a^2)\nf = (3 * a**4 + 1) / (2 * a**2)\n\n# Take the derivative of f with respect to a\nf_prime = sp.diff(f, a)\n\n# Solve for the critical points\ncritical_points = sp.solve(f_prime, a)\n\n# Calculate the value of f at the critical points to find the minimum value\nmin_value = min(f.subs(a, cp) for cp in critical_points if cp.is_real and cp > 0)\n\nprint(min_value)\n",
        """
        \nfrom math import comb\n\nn = 8\nk = 6\nbinom = comb(n, k)\na = (3/5)**(n - k)\nb = (-1/2)**k\ncoefficient = binom * a * b\nprint(coefficient)\n
        """,
        """def find_largest_NPMPP():
    # Loop through possible values of M (highest to lowest to find the largest possible one)
    for M in range(6, 0, -1):
        product = 1111 * M * M
        if 10000 <= product <= 99999:  # Ensure product is a 5-digit number
            str_product = str(product)
            if str_product[-1] == str(M) and str_product[-2] == str(M):  # Check if last two digits are the same as M
                # Check if the format is NPMPP
                N = str_product[0]
                P = str_product[1]
                if len(str_product) == 5:
                    return int(str_product)
    return None

largest_NPMPP = find_largest_NPMPP()
print(largest_NPMPP)"""
    ]   


    executor = PythonExecutor()
    predictions = executor.batch_apply(batch_code)
    print(predictions)

# executor = PythonExecutor()
'''
def detect_code(pred):
    if "```python" not in pred:
        return False
    elif "```" not in pred.split("```python")[1]:
        return False
    code = pred.split("```python")[1].split("```")[0]
    code = code.strip()
    return code != ''
'''


def detect_code(pred):
    # 使用预编译的正则表达式进行匹配
    match = CODE_BLOCK_PATTERN.search(pred)
    
    if match:
        # 提取匹配到的代码内容，并去除首尾空白字符
        code = match.group(1).strip()
        return code != ''  # 返回 True 如果代码不为空，否则返回 False
    return False  # 如果没有匹配到代码块，返回 False


def extract_code(preds):
    codes = []
    for pred in preds:
        if "```python" not in pred:
            code = ''
        else:
            code = pred.split("```python")[1].split("```")[0]
        codes.append(code)
    return codes

def extract_code_sole(pred):

    if "```python" not in pred:
        code = ''
    else:
        code = pred.split("```python")[1].split("```")[0]
    return code


def excute_code(preds, executor: PythonExecutor):
    codes = preds
    no_code_idx = []
    for i, code in enumerate(codes):
        if code == '':
            no_code_idx.append(i)
    batch_results = executor.batch_apply(batch_code=codes)
    return batch_results, no_code_idx


def process_string(input_str):
    # 找到第一个 "```python" 的位置
    start_index = input_str.find("```python")
    
    if start_index == -1:
        # 如果没有找到 "```python"，直接返回原字符串
        return input_str
    
    # 从 "```python" 之后开始找最近的 "```"
    end_index = input_str.find("```", start_index + len("```python"))
    
    if end_index == -1:
        # 如果没有找到结束的 "```"，直接返回原字符串
        return input_str
    
    # 截取到 "```" 为止（包括 "```"）
    result = input_str[:end_index + len("```")]
    return result




if __name__ == "__main__":
    _test()
