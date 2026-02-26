import signal
import re
from contextlib import contextmanager
import importlib.util
from math_verify import parse, verify
class TimeoutException(Exception):
    pass


SUBSTITUTIONS = [
    ("an ", ""),
    # ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
    ("\\left", ""),
    ("\\right", ""),
    ("∶", ":"),
    ("，", ","),
    ("$",  ""),
    ("\\approx", "="),
    ("\\simeq", "="),
    ("\\sim", "="),
    ("^\\prime", "'"),
    ("^{\\prime}", "'"),
    ("\\dfrac", "\\frac"),
    ("\\tfrac", "\\frac"),
    ("^\\circ", ""),
    ("%", ""),
    ("\u221a", "\\sqrt"),
    ("\u221e", "\\infty"),
    ("\u222a", "\\cup"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    # "ft", #this is dangerous, infty, left will be damaged!
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]



def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.
    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    # final_answer = final_answer.split("=")[-1]
    final_answer=final_answer.strip()
    if final_answer[:2]=="\\(" or final_answer[:2]=='\\[':
        final_answer=final_answer[2:]
    if final_answer[-2:]=='\\)' or final_answer[-2:]=='\\]':
        final_answer=final_answer[:-2]
    
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    if final_answer[:2]=="\\(" or final_answer[:2]=='\\[':
        final_answer=final_answer[2:]
    if final_answer[-2:]=='\\)' or final_answer[-2:]=='\\]':
        final_answer=final_answer[:-2]
    return final_answer.strip()



@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

timeout_seconds=2
chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
english_pattern = re.compile(r'[a-zA-Z]')
boxed_pattern = re.compile(r"\\boxed\{((?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{[^{}]*\}))*\}))*\}))*\})")
valid_char_pattern = re.compile(r'[a-zA-Z0-9\s\.,!?"\'\(\)\{\}\[\]_\-+=<>/@#$%^&*\\|:;~`\u2200-\u22FF]')
repeat_pattern = re.compile(r'(.{5,}?)\1{4,}')

def check_mixed_languages(text):
    chinese_chars = len(chinese_pattern.findall(text))
    english_chars = len(english_pattern.findall(text))
    return chinese_chars >= 20 and english_chars >= 20

def undesired_format(text):
    if "<|endoftext|>" not in text: return True
    else: return False


def check_garbled_characters(text):
    valid_chars = valid_char_pattern.sub('', text)
    if not text: 
        return False
    invalid_ratio = len(valid_chars) / len(text)
    return invalid_ratio > 0.3

def has_repeated_patterns(text):
    return bool(repeat_pattern.search(text))
    
def correctness_score_default(response, gt):
    matches = boxed_pattern.findall(response)
    if not matches: return 0.0
    pred = matches[-1][:-1]
    return 1.0 if is_equiv(pred, gt) else 0.0


def correctness_score_v2(response, gt):
    matches = boxed_pattern.findall(response)
    if not matches: return 0.0
    pred = matches[-1][:-1]
    return 1.0 if is_equiv(pred, gt) else -0.5



def code_format_reward(predict_str: str):
    if "```python" in predict_str and "```output" in predict_str:
        code_blocks = re.findall(r"```python(.*?)```.*```output(.*?)```", predict_str, re.DOTALL)
        if len(code_blocks) > 0 and len(code_blocks[-1]) > 1:
            code_blocks = code_blocks[-1]
        else:
            code_blocks = ""
    # elif "```python" in predict_str:
    #     code_blocks = re.findall(r"```python(.*?)```", predict_str, re.DOTALL)
    #     if len(code_blocks) > 0:
    #         code_blocks = code_blocks[-1]
    #     else:
    #         code_blocks = ""
    else:
        code_blocks = ""

    if code_blocks == "":
        format_score = 0.0
    else:
        format_score = 1.0

    return format_score



def check_repetition(completion: str, repetition_ngram_size=50, repetition_max_num=20) -> bool:
    """
    检测当前回复是否出现重复解码

    Returns:
        flags: True/False

    """
    def get_repetition_penalty(str_: str, ngram_size: int) -> float:
        n_garm_counter = dict()
        max_ = 0
        most_freq_n_gram = ''
        tokens = str_
        for i in range(0, len(tokens) - ngram_size):
            tmp_n_gram = tokens[i: i+ngram_size]
            if n_garm_counter.get(tmp_n_gram) is None:
                n_garm_counter[tmp_n_gram] = 0
            n_garm_counter[tmp_n_gram] += 1
            if max_ < n_garm_counter[tmp_n_gram]:
                max_ = n_garm_counter[tmp_n_gram]
                most_freq_n_gram = tmp_n_gram
        return max_

    max_repetition_num = get_repetition_penalty(completion, repetition_ngram_size)

    if max_repetition_num > repetition_max_num:
        # exist repetition
        return True
    else:
        # not exist repetition
        return False



def compute_score(solution_str: str, ground_truth: str, reward_type='default') -> float:   

    # # 匹配是否生成了 ```python(.*)```, ```output(.*)```
    # code_format_score = code_format_reward(solution_str)
    # if code_format_score == 0.0:
    #     return 0.0

    # 重复解码检测
    # is_repetition = check_repetition(completion=solution_str)
    # if is_repetition:
    #     return 0.0

    
    if reward_type=='default':
        try:
            # if undesired_format(solution_str): return 0.0
            return correctness_score_default(solution_str, ground_truth)            
        except TimeoutException:
            return 0.0
        except Exception as e:
            return 0.0
    elif reward_type=="v2.wformat":
        try:
            return correctness_score_v2(solution_str, ground_truth)
        except TimeoutException:
            return 0.0
        except Exception as e:
            return 0.0
    else:
        try:     
            # if undesired_format(solution_str): return 0.0
            return correctness_score_default(solution_str, ground_truth)            
        except TimeoutException:
            return 0.0
        except Exception as e:
            return 0.0




# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False
    if str1.strip().lower() == str2.strip().lower(): return True
    try:
        str1=normalize_final_answer(str1)
        str2=normalize_final_answer(str2)
        str1=parse(str1)
        str2=parse(str2)
        return verify(str1, str2)
    except:
        pass

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1==ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


if __name__ == "__main__":
    response="To determine which digit appears in the 534th place after the decimal point in the decimal representation of $\\frac{5}{13}$, we need to first find the repeating decimal sequence of $\\frac{5}{13}$. \n\nLet's start by calculating the decimal representation of $\\frac{5}{13}$.\n```python\nfrom decimal import Decimal, getcontext\r\n\r\n# Set the precision high enough to see the repeating pattern clearly\r\ngetcontext().prec = 1000\r\n\r\n# Calculate the decimal representation of 5/13\r\ndecimal_rep = Decimal(5) / Decimal(13)\r\nprint(str(decimal_rep))\n```\n```output\n0.\n```\nThe decimal representation of $\\frac{5}{13}$ is $0.\\overline{384615}$. This means the repeating sequence is \"384615\" and it has a length of 6 digits.\n\nTo find the digit in the 534th place after the decimal point, we need to determine the position within the repeating sequence. Since the sequence repeats every 6 digits, we can find the position by calculating the remainder when 534 is divided by 6.\n\nLet's calculate this.\n```python\n# Length of the repeating sequence\r\nrepeating_sequence = \"384615\"\r\nsequence_length = len(repeating_sequence)\r\n\r\n# Find the position within the repeating sequence\r\nposition = (534 - 1) % sequence_length  # -1 because indexing starts from 0\r\n\r\n# Get the digit at that position\r\ndigit_in_534th_place = repeating_sequence[position]\r\nprint(digit_in_534th_place)\n```\n```output\n6\n```\nThe digit in the 534th place after the decimal point in the decimal representation of $\\frac{5}{13}$ is $\\boxed{6}$. <|endoftext|>"
    answer="6"
    res = compute_score(response, answer)
    print(res)