# 🚀 THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning 🚀


[![Paper](https://img.shields.io/badge/arXiv-2509.13761-b31b1b.svg)](https://arxiv.org/abs/2509.13761)
[![HF Daily Papers](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/papers/2509.13761)

![Pipeline](assets/introduction.png)

This is the official implementation of our paper **THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning**.


## :fire: News:

TODO:
- [x] Update arXiv preprint.
- [ ] Update inference code.
- [ ] Update training code.
- [ ] Update the cold start dataset.

## 🔍 Overview
Large Language Models (LLMs) have made remarkable progress in mathematical reasoning, but still continue to struggle with high-precision tasks like numerical computation and formal symbolic manipulation. Integrating external tools has emerged as a promising approach to bridge this gap. Despite recent advances, existing methods struggle with three key challenges: constructing tool-integrated reasoning data, performing fine-grained optimization, and enhancing inference. To overcome these limitations, we propose THOR (Tool-Integrated Hierarchical Optimization via RL). First, we introduce TIRGen, a multi-agent actor-critic-based pipeline for constructing high-quality datasets of tool-integrated reasoning paths, aligning with the policy and generalizing well across diverse models. Second, to perform fine-grained hierarchical optimization, we introduce an RL strategy that jointly optimizes for both trajectory-level problem solving and step-level code generation. This is motivated by our key insight that the success of an intermediate tool call is a strong predictor of the final answer's correctness. Finally, THOR incorporates a self-correction mechanism that leverages immediate tool feedback to dynamically revise erroneous reasoning paths during inference. Our approach demonstrates strong generalization across diverse models, performing effectively in both reasoning and non-reasoning models. It further achieves state-of-the-art performance for models of a similar scale on multiple mathematical benchmarks, while also delivering consistent improvements on code benchmarks.

## ✨ Key Contributions
1. 🛠 TIRGen Pipeline – Generates policy-aligned tool-integrated reasoning data.
2. 🎯 Hierarchical RL – Combines trajectory-level optimization with step-level correction.
3. 🔄 Self-Correction Inference – Dynamically fixes reasoning errors during inference.
4. 📊 Broad Generalization – Effective across reasoning and non-reasoning models.

## ⚙️ Method
Our method, THOR, enhances tool-integrated reasoning with a three-stage pipeline:

1️⃣ TIRGen: Tool-Integrated Data Construction
- Actor generates natural language reasoning steps.
- Critic evaluates whether parts of the reasoning can be executed as code.
- Identified steps are transformed into tool-augmented reasoning paths.
- Multi-stage filtering ensures policy alignment, code quality, and difficulty balance.

![TIRGen](assets/data_construction.png)

2️⃣ Hierarchical Reinforcement Learning
- Trajectory-level RL: Optimizes overall correctness of the final answer using GRPO.
- Step-level RL: Focuses on error-prone code generation steps, using execution results as fine-grained rewards.
- Joint optimization addresses sparse reward issues in long reasoning chains.

![THOR](assets/optimization.png)

3️⃣ Self-Correction During Inference
- During inference, if a tool call fails, the model backtracks to the reasoning step.
- It regenerates a new suffix and revised action, guided by tool feedback.
- This enables online error correction with minimal overhead.

## 📥 Installation
```python
# TODO
```

## 🚀 Usage
Inference Example
```python
# TODO
```


## 🙌 Acknowledgements
We thank the open-source community from [Qwen](https://github.com/QwenLM/Qwen), [verl](https://github.com/volcengine/verl) and [SandboxFusion](https://github.com/bytedance/SandboxFusion).


## 🖊️ Citation
If you find our work helpful, please consider giving us a ⭐ and citing our paper:
```
@article{THOR,
  title={THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning},
  author = {Chang, Qikai and Zhang, Zhenrong and Hu, Pengfei and Ma, Jiefeng and Pan, Yicheng and Zhang, Jianshu and Du, Jun and Liu, Quan and Gao, Jianqing},
  journal={arXiv preprint arXiv:2509.13761},
  year={2025}
}
```







