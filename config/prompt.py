import os



R1_template = 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer>, the final answer should be put within \\boxed{} tags, respectively. i.e., <think> reasoning process here </think> <answer> answer here, \\boxed{final answer} </answer>'


qwen_math_template = "Please reason step by step, and put your final answer within \\boxed{}.\n[PROMPT]\n"


qwen_math_template_cn = "请一步步推理，并将你的最终答案放在 \\boxed{}."


no_template = "You are a helpful assistant. Please reason step by step, and put your final answer within \\boxed{}."


r1_distill_qwen_prompt = """
Please reason step by step, and put your final answer within \boxed{}.\n[PROMPT]
"""


math_tool_interaction_prompt_en = """
You are a scientist skilled in math and computer. I will give you a math problem. You can output Python code at any point to assist your reasoning—e.g., for solving equations, verifying assumptions, computing intermediate contents or anything other. Once you have finished outputting the python code, you need to stop outputting and wait for me to execute the code. I will return the results (wrapped in <|box_start|>output<|box_end|>) for you. Continue reasoning with the returned results until you reach the final answer.
Note:
1. Each output code should be able to execute independently and completely to output the result you want.
2. The output length of the code should not be too long. If it is too long, responses will exceed the context limit.

Code Format:
Each code snippet is wrapped with
<code>
```python
code snippet
</code>

Answer Format:
The final answer should be wrapped with \\boxed{}.

Question: 
"""


qwen_math_interaction_prompt_en = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\nUser: Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.\n[PROMPT]\n"

qwen3_tir_interaction_prompt_en = """
You are a scientist skilled in mathematics and computer science. Please integrate natural language reasoning and Python code to solve mathematical problems. You can use Python code during the thinking process for numerical calculations, equation solving, hypothesis verification, data processing, and any other steps to assist your reasoning. The final answer should be placed within \\boxed{}.
Note:
1. Independent Code Blocks: Each of your code blocks is independent, can run on its own, does not depend on any previous variables, and imports any required libraries independently.
2. Multiple Code Blocks Supported: You can output multiple code blocks as needed to assist different computational steps in the reasoning process.
3. Careful Thinking: Each piece of code should be carefully considered and go beyond simple arithmetic operations such as addition, subtraction, multiplication, or division. Each code block should include a `print()` function to display its output.

Code Format:
```python
code blocks
```
```output
code output
```

Question: [PROMPT]
"""



reasoning_step_to_python_code_zh = """你是一个精通计算机和数学的科学家。我将给你一个强大模型针对数学问题的详细思考过程（思维链），你的任务是对这个思考过程进行二次修改。
修正目标：
在不改变原始模型推理思路和方法的前提下，识别思维链中任何可以通过代码辅助进行数值计算、方程求解、假设验证、数据处理等的步骤，并将这些自然语言描述的计算过程替换为相应的代码执行和结果预测。

你需要完成以下步骤：
1. 识别可代码化部分：仔细分析原始思维链，找出任何涉及具体数值计算、代数运算、集合操作、逻辑验证等，可以使用Python代码精确执行的部分。
2. 撰写 Python 代码：针对识别出的部分，编写清晰、简洁且能够完成相应计算任务的 Python 代码块, 你的每一段代码都应深思熟虑, 而非做简单的加减乘除。
3. 预测代码输出：提供你所撰写代码的预期运行结果。
4. 嵌入思维链：将 Python 代码和执行结果嵌入到思维链中，保持修改之后的思维过程逻辑正确，完善，未作修改的部分需要保持完全一致！同时只修改原始思维链中包含的部分, 思维链未包含的部分不要随意延申和继续解题。
注意：
1. 多段代码支持：你可以根据需要输出多段代码块来辅助推理过程中的不同计算环节。
2. 代码块保持独立: 你的每个代码块之间相互独立, 可以独立运行, 不依赖任何之前的变量, 并独立导入任何需要的库.
3. 保持思路一致：不要修改原始思维链的核心推理逻辑、步骤顺序或基本方法。你的任务仅仅是将自然语言描述的具体计算和验证过程转换为等效的代码执行和输出。
4. 精确性：确保你提供的代码是正确的，并且预测的输出是准确无误的。
5. 禁止解题：你需要做的仅仅是二次修改思维链，不需要修改的部分需要保持不变并输出。

如果可以使用 Python 代码辅助求解，请严格遵循以下格式，确保你的修正内容易于机器解析和人类理解：
origna reasoning steps, ...
```python
python block 1
```
```output
code output 1
```
original reasoning steps, ...
```python
python block 2
```
```output
code output 2
```
continue reasoning...



## 样例输入:
好的，让我来试着解决这个篮球得分问题。题目说在上一场比赛后，球队得分的\(\frac{1}{4}\)由阿列克谢得到，\(\frac{2}{7}\)由布列塔尼得到，切尔西得了15分，并且其他7名队员得分都不超过2分。我们需要求出这其他7名队员的总得分。\n首先，让我逐步分析这些信息。我们用\(T\)表示球队的总得分。那么，阿列克谢的得分是\(\frac{1}{4}T\)，布列塔尼的得分是\(\frac{2}{7}T\)，切尔西得了15分，剩下7名队员的得分设为\(R\)。所以，我们可以列出等式：
总得分 = 阿列克谢的得分 + 布列塔尼的得分 + 切尔西的得分 + 其他7名队员的得分
\(T = \frac{1}{4}T + \frac{2}{7}T + 15 + R\)
我们的目标是求出\(R\)。
但在这之前，也许我应该先把分数合并一下。让我来计算\(\frac{1}{4}T + \frac{2}{7}T\) 。要把这两个分数相加，我需要一个公分母。这两个分母分别是4和7，所以最小公分母是28 。
那么转换一下：
\(\frac{1}{4}T = \frac{7}{28}T\)
\(\frac{2}{7}T = \frac{8}{28}T\)
把它们加起来：\(\frac{7}{28}T + \frac{8}{28}T = \frac{15}{28}T\)
所以，阿列克谢和布列塔尼一共得到了总得分的\(\frac{15}{28}\) 。
把这个代回到总得分的等式中：
\(T = \frac{15}{28}T + 15 + R\)
让我从等式两边减去\(\frac{15}{28}T\)来求解\(R\)：
\(T - \frac{15}{28}T = 15 + R\)
计算左边：
\((\frac{28}{28}T - \frac{15}{28}T) = \frac{13}{28}T\)
所以\(\frac{13}{28}T = 15 + R\)
或者\(R = \frac{13}{28}T - 15\)
题目中说这7名队员得分都不超过2分。所以，他们每人最多得2分，这意味着他们的总得分最多是\(7×2 = 14\)分。不过，因为“没有……得分超过2分”，所以有可能有些人得分更少。但重要的是，\(R\)必须是一个介于0到14（包含0和14 ）之间的整数。
但同时，因为\(R = \frac{13}{28}T - 15\)，那么\(\frac{13}{28}T - 15\)必须是一个介于0到14之间的整数。此外，\(T\)必须是一个整数，因为在篮球比赛中不能得到分数的小数部分。而且，阿列克谢的得分\(\frac{1}{4}T\)和布列塔尼的得分\(\frac{2}{7}T\)也都必须是整数。所以，\(T\)必须是4和7的倍数，对吧？因为\(\frac{1}{4}T\)必须是整数，所以\(T\)能被4整除。同样，\(\frac{2}{7}T\)必须是整数，所以\(T\)能被7整除。因此，\(T\)必须是4和7的公倍数。4和7的最小公倍数是28 。所以，\(T\)必须是\(28k\) ，其中\(k\)是一个正整数。
所以我们写成\(T = 28k\) ，其中\(k\)是大于等于1的整数 。
然后，把它代回到\(R\)的表达式中：
\(R = \frac{13}{28}×28k - 15 = 13k - 15\)
但\(R\)必须满足\(0 ≤ R ≤ 14\) 。
那么：
第一个不等式：\(0 ≤ 13k - 15\) ，则\(13k ≥ 15\) ，\(k ≥ \frac{15}{13} ≈ 1.15\) 。因为\(k\)必须是整数，所以\(k ≥ 2\) 。
第二个不等式：\(13k - 15 ≤ 14\) ，则\(13k ≤ 29\) ，\(k ≤ \frac{29}{13} ≈ 2.23\) 。因为\(k\)必须是整数，所以\(k ≤ 2\) 。
因此，综合这两个不等式，\(k\)只能是2 。因为\(k ≥ 2\)且\(k ≤ 2\) ，所以\(k = 2\) 。
所以，\(T = 28×2 = 56\)分 。
总得分是56分。现在让我检查一下这样是否正确。
阿列克谢的得分：\(\frac{1}{4}×56 = 14\) 。
布列塔尼的得分：\(\frac{2}{7}×56 = 16\) 。
切尔西的得分：15 。
其他7名队员：\(R = 13×2 - 15 = 26 - 15 = 11\) 。
那么总得分：\(14 + 16 + 15 + 11 = 14 + 16 = 30\)；\(30 + 15 = 45\)；\(45 + 11 = 56\) 。很好，加起来正好是56分。看起来是正确的。
但我们还需要检查一下其他7名队员每人得分是否不超过2分。因为他们的总得分是11分，一共有7名队员，我们来看看是否有可能每人最多得2分。
7名队员每人最多得2分，那么总得分最多是14分，而11分小于14分，所以是有可能的。例如，5名队员每人得1分，1名队员得2分，但这里有7名队员……等等，7名队员。如果7名队员得分都是1分或者2分，加起来是11分。比如，4名队员得2分（总共8分），3名队员得1分（总共3分），这样总共就是11分。这是可行的。或者其他组合方式。所以这是合理的。
所以，\(R = 11\) 。所以答案是11 。


## 样例输出:
让我们来解决这个篮球得分问题。题目说在上一场比赛后，球队得分的$\frac{1}{4}$由阿列克谢得到，$\frac{2}{7}$由布列塔尼得到，切尔西得了15分，并且其他7名队员得分都不超过2分。我们需要求出这其他7名队员的总得分。
首先，我们用$T$表示球队的总得分。那么，阿列克谢的得分是$\frac{1}{4}T$，布列塔尼的得分是$\frac{2}{7}T$，切尔西得了15分，剩下7名队员的得分设为$R$。所以我们可以列出等式：
$$
T = \frac{1}{4}T + \frac{2}{7}T + 15 + R
$$
接下来合并分数项：
$$
\frac{1}{4}T + \frac{2}{7}T = \frac{7}{28}T + \frac{8}{28}T = \frac{15}{28}T
$$
代入原方程得：
$$
T = \frac{15}{28}T + 15 + R
$$
从两边减去$\frac{15}{28}T$得：
$$
T - \frac{15}{28}T = 15 + R \Rightarrow \frac{13}{28}T = 15 + R \Rightarrow R = \frac{13}{28}T - 15
$$
由于其他7人每人最多得2分，因此他们总分最多为 $7 \times 2 = 14$ 分，即 $0 \leq R \leq 14$。同时，$T$必须是4和7的公倍数，因为阿列克谢和布列塔尼的得分都必须是整数。最小公倍数为28，因此设 $T = 28k$。
代入得：
$$
R = \frac{13}{28} \cdot 28k - 15 = 13k - 15
$$
我们可以通过编写Python代码枚举可能的$k$值，找到满足条件的解：
```python
# 枚举k的可能值，找到满足条件的k
for k in range(1, 10):  # 假设k最大不会超过10
    T = 28 * k
    R = 13 * k - 15
    if 0 <= R <= 14:
        print(f"k = {k}, T = {T}, R = {R}")
```
```output
k = 2, T = 56, R = 11
```
只有当 k=2 时，R=11 满足 0≤R≤14 的条件。因此，总得分 T=56 ，其他7名队员总得分为 R=11 。
现在验证各部分得分加起来是否等于56：
```python
# 验证总分是否正确
aleksey = 1/4 * 56
brittany = 2/7 * 56
chelsea = 15
rest = 11

total = aleksey + brittany + chelsea + rest
total
```
```output
56.0
```
确实，各项得分相加正好是56分，说明我们的计算是正确的。
最后检查其他7名队员的得分是否合理：他们的总得分为11分，人数为7人，每人最多得2分。例如，可以有4人得2分（共8分），3人得1分（共3分），刚好凑成11分，符合题意。
综上所述，其他7名队员的总得分为 11分 。


原始思维链：
[long_cot]
"""



reasoning_step_to_python_code_en = """
You are a scientist proficient in computer science and mathematics. I will provide you with a detailed thought process (chain of thought) from a powerful model for a mathematical problem. Your task is to revise this thought process.

## Revision Goal:
Without altering the original model's reasoning flow and methods, identify any steps in the chain of thought that can be assisted by code for numerical calculations, equation solving, hypothesis testing, data processing, etc., and replace these natural language descriptions of computational processes with corresponding code execution and predicted results.

### You need to complete the following steps:
1. Identify Codable Parts: Carefully analyze the original chain of thought to pinpoint any parts involving specific numerical calculations, algebraic operations, set operations, logical verifications, etc., that can be precisely executed using Python code.
2. Write Python Code: For the identified parts, write clear, concise Python code blocks that can accomplish the corresponding computational tasks. Each piece of your code should be carefully considered, not just performing simple arithmetic operations like addition, subtraction, multiplication, or division. Every code block should return the output with `print()` function.
3. Predict Code Output: Provide the expected output of the Python code you have written.
4. Embed in Chain of Thought: Embed the Python code and its execution results into the chain of thought, ensuring that the revised thought process remains logically correct and complete. Keep the parts that were not modified exactly as they were! Also, only modify the parts included in the original chain of thought; do not extend or continue solving parts not covered by the chain of thought.

### Note:
1. Multiple Code Blocks Supported: You can output multiple code blocks as needed to assist different computational steps in the reasoning process.
2. Independent Code Blocks: Each of your code blocks is independent, can run on its own, does not depend on any previous variables, and imports any required libraries independently.
3. Maintain Consistency of Thought: Do not alter the core reasoning logic, step order, or basic methods of the original chain of thought. Your task is solely to convert specific computational and verification processes described in natural language into equivalent code execution and output.
4. Accuracy: Ensure that the code you provide is correct and that the predicted output is accurate.
5. No Problem Solving: Your task is merely to revise the chain of thought. Parts that do not need modification should remain unchanged and be outputted as such.

If Python code can be used to assist in solving, please strictly follow the format below to ensure your revised content is easily parsable by machines and understandable by humans:

original reasoning steps, ...
```python
python block 1
```
```output
code output 1
```
original reasoning steps, ...
```python
python block 2
```
```output
code output 2
```
continue reasoning...



## Sample Input:
Okay, let me try to figure out this basketball points problem. So, the question says that after the last game, 1/4 of the team’s points were scored by Alexa, 2/7 by Brittany, Chelsea scored 15 points, and none of the other 7 team members scored more than 2 points. We need to find the total number of points scored by those other 7 players.
First, let me parse the information step by step. Let me denote the total points scored by the team as T. Then, Alexa scored (1/4)T, Brittany scored (2/7)T, Chelsea scored 15, and the remaining 7 players scored some amount, let's say R. So, we can write the equation:
Total points = Alexa's points + Brittany's points + Chelsea's points + Other 7 players' points
T = (1/4)T + (2/7)T + 15 + R
Our goal is to find R.
But before that, maybe I should combine the fractions first. Let me compute (1/4)T + (2/7)T. To add these fractions, I need a common denominator. The denominators are 4 and 7, so the least common denominator is 28.
So converting:
(1/4)T = 7/28 T
(2/7)T = 8/28 T
Adding them together: 7/28 T + 8/28 T = 15/28 T
Therefore, Alexa and Brittany together scored 15/28 of the total points.
So plugging back into the equation for total points:
T = 15/28 T + 15 + R
Let me subtract 15/28 T from both sides to solve for R:
T - 15/28 T = 15 + R
Calculating the left side:
(28/28 T - 15/28 T) = 13/28 T
So 13/28 T = 15 + R
or R = 13/28 T - 15
The problem states that none of these 7 players scored more than 2 points. Therefore, each of them scored at most 2 points, which means the maximum total they could have scored is 7*2=14 points. However, since "none... scored more than 2," it's possible that some scored less. But importantly, R must be an integer between 0 and 14, inclusive.
But also, since R = 13/28 T - 15, then 13/28 T - 15 must be an integer between 0 and 14. Moreover, T must be a whole number because you can't score a fraction of a point in basketball. Also, Alexa's points (1/4 T) and Brittany's points (2/7 T) must also be whole numbers. Therefore, T must be a multiple of 4 and 7 right? Because 1/4 T must be integer, so T divisible by 4. Similarly, 2/7 T must be integer, so T must be divisible by 7. Therefore, T must be a common multiple of 4 and 7. The least common multiple of 4 and 7 is 28. Therefore, T must be 28k, where k is a positive integer.
So let me write T = 28k, where k is integer ≥ 1.
Then, substituting back into R:
R = 13/28 * 28k - 15 = 13k - 15
But R must satisfy 0 ≤ R ≤ 14.
Inequality:
0 ≤ 13k - 15 ≤ 14
15 ≤ 13k ≤ 29
15/13 ≤ k ≤ 29/13
1.15 ≤ k ≤ 2.23
Since k must be integer, k ≤ 2.
Therefore, combining both inequalities, k must be 2. Because k ≥ 2 and k ≤ 2. So k=2.
Therefore, T = 28*2 = 56 points.
So total points is 56. Now let me check if this works.
Alexa's points: 1/4 * 56 = 14.
Brittany's points: 2/7 * 56 = 16.
Chelsea's points: 15.
Other 7 players: R = 13*2 - 15 = 26 -15 = 11.
So total points: 14 + 16 + 15 + 11 = 14+16=30; 30+15=45; 45+11=56. Perfect, that adds up to 56. So that seems correct.



## Sample Output:
Okay, let me try to figure out this basketball points problem. So, the question says that after the last game, 1/4 of the team’s points were scored by Alexa, 2/7 by Brittany, Chelsea scored 15 points, and none of the other 7 team members scored more than 2 points. We need to find the total number of points scored by those other 7 players.
First, let me parse the information step by step. Let me denote the total points scored by the team as T. Then, Alexa scored (1/4)T, Brittany scored (2/7)T, Chelsea scored 15, and the remaining 7 players scored some amount, let's say R. So, we can write the equation:
Total points = Alexa's points + Brittany's points + Chelsea's points + Other 7 players' points
T = (1/4)T + (2/7)T + 15 + R
Our goal is to find R.
But before that, maybe I should combine the fractions first. Let me compute (1/4)T + (2/7)T. To add these fractions, I need a common denominator. The denominators are 4 and 7, so the least common denominator is 28.
So converting:
(1/4)T = 7/28 T
(2/7)T = 8/28 T
Adding them together: 7/28 T + 8/28 T = 15/28 T
Therefore, Alexa and Brittany together scored 15/28 of the total points.
So plugging back into the equation for total points:
T = 15/28 T + 15 + R
Let me subtract 15/28 T from both sides to solve for R:
T - 15/28 T = 15 + R
Calculating the left side:
(28/28 T - 15/28 T) = 13/28 T
So 13/28 T = 15 + R
or R = 13/28 T - 15
The problem states that none of these 7 players scored more than 2 points. Therefore, each of them scored at most 2 points, which means the maximum total they could have scored is 7*2=14 points. However, since "none... scored more than 2," it's possible that some scored less. But importantly, R must be an integer between 0 and 14, inclusive.
But also, since R = 13/28 T - 15, then 13/28 T - 15 must be an integer between 0 and 14. Moreover, T must be a whole number because you can't score a fraction of a point in basketball. Also, Alexa's points (1/4 T) and Brittany's points (2/7 T) must also be whole numbers. Therefore, T must be a multiple of 4 and 7 right? Because 1/4 T must be integer, so T divisible by 4. Similarly, 2/7 T must be integer, so T must be divisible by 7. Therefore, T must be a common multiple of 4 and 7. The least common multiple of 4 and 7 is 28. Therefore, T must be 28k, where k is a positive integer.
We can now use Python code to enumerate possible values of $k$ and find the solution that satisfies the conditions:
```python
# Enumerate possible values of k to find the one that satisfies the condition
for k in range(1, 10):  # Assume k will not exceed 10
    T = 28 * k
    R = 13 * k - 15
    if 0 <= R <= 14:
        print(f"k = {k}, T = {T}, R = {R}")
```
```output
k = 2, T = 56, R = 11
```
Only when $k=2$ does $R=11$ satisfy the condition $0 \leq R \leq 14$. Therefore, the total score $T=56$, and the total score for the other 7 players is $R=11$.
Now, let's verify if the sum of all parts of the score equals 56:
```python
# Verify if the total score is correct
aleksey = 1/4 * 56
brittany = 2/7 * 56
chelsea = 15
rest = 11

total = aleksey + brittany + chelsea + rest
print(total)
```
```output
56.0
```
Indeed, the sum of all scores is exactly 56, which confirms our calculation is correct.



Original Chain-of-Thought:
[long_cot]
"""




policy_prompt = "Please Reason step by step and put your final answer within \\boxed{}. \n[question]"










if __name__ == '__main__':
    from transformers import AutoTokenizer
    qwen_path = './pretrained_models/Qwen2.5-Math-7B/'
    tokenizer = AutoTokenizer.from_pretrained(qwen_path)
    input_ids = tokenizer.tokenize(math_tool_interaction_prompt_en)
    print(input_ids)


