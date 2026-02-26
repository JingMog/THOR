# 🚀 THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning 🚀

<p align="center">
  <a href="https://arxiv.org/abs/2509.13761"><img src="https://img.shields.io/badge/arXiv-2509.13761-b31b1b.svg"></a>
  <a href="https://papers.cool/arxiv/2509.13761"><img src="https://img.shields.io/badge/CoolPaper-Paper-blue"></a>
  <a href="https://huggingface.co/papers/2509.13761"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Data-blue"></a>
</p>


![Pipeline](assets/introduction.png)

This is the official implementation of our paper **THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning**.

## :fire: News:

- 🎉🎉🎉 Our paper has been selected for the [🤗 Hugging Face Daily Papers](https://huggingface.co/papers/2509.13761)! Thanks to the community for the recognition and support 🚀
- 🎉🎉🎉Congratulations! Our paper has been accepted by ICLR 2026.



TODO:

- [x] Update arXiv preprint.
- [x] Update inference code.
- [x] Update TIRGen code.
- [ ] Update training code.
- [ ] Update the TIRGen dataset.

## 🔍 Overview
Large Language Models (LLMs) have advanced in mathematical reasoning but still struggle with precise computation and symbolic manipulation. THOR (Tool-Integrated Hierarchical Optimization via RL) addresses this by:

1. TIRGen – an actor–critic pipeline to construct high-quality tool-integrated reasoning data.
2. Hierarchical RL – jointly optimizing trajectory-level reasoning and step-level code generation.
3. Self-Correction – leveraging tool feedback to fix reasoning errors during inference.

THOR achieves state-of-the-art performance on multiple mathematical benchmarks and shows consistent improvements on code generation tasks, generalizing well across both reasoning and non-reasoning models.

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

## 📊 Results

### Comparison With State-of-the-Art Methods
![SOTA_result](assets/exp_sota_result.png)

### Effectiveness of TIRGen
![effectiveness_of_TIRGen](assets/exp_effectiveness_of_TIRGen.png)

### Ablation Study
![Ablation_Study](assets/exp_ablation_study.png)


## 📥 Installation
Step1. Install SandboxFusion
```bash
# install sandboxfusion to support code execution
conda create -n sandbox -y python=3.12
conda activate sandbox
poetry install
# to build the real docs, run `cd docs && npm ci && npm run build`
mkdir -p docs/build
make run-online
```

Step2. Install THOR environment
```bash
conda create -n THOR -y python=3.10
pip install -r requirements.txt
```

## 🚀 Usage

### 1. TIRGen: TIR data construction pipeline
```bash
cd TIRGen
bash construct_dataset_main.sh

# multi_stage_filter
bash filter.sh
```

### 2. TIR Inference
```bash
cd inference
bash submit_bon_policy.sh
```


### 3. cold start
Our cold start is based on swift
```bash
cd swift
bash sft_demo.sh
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







