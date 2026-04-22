# IA2: Alignment with ICL Activations improves Supervised Fine-Tuning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Supervised Fine-Tuning (SFT) is used to specialize model behavior by training weights to produce intended target responses for queries. In contrast, In-Context Learning (ICL) adapts models during inference with instructions or demonstrations in the prompt. 
ICL can offer better generalizability and more calibrated responses compared to SFT in data scarce settings, at the cost of more inference compute. In this work, we ask the question: \textit{Can ICL's internal computations be used to improve the qualities of SFT?} We first show that ICL and SFT produce distinct activation patterns, indicating that the two methods achieve adaptation through different functional mechanisms. Motivated by this observation and to use ICL's rich functionality, we introduce \textbf{I}CL \textbf{A}ctivation \textbf{A}lignment (\act), a self-distillation technique which aims to replicate ICL's activation patterns in SFT models and incentivizes ICL-like internal reasoning. Performing \act as a priming step before SFT significantly improves the accuracy and calibration of model outputs, as shown by our extensive empirical results on 12 popular benchmarks and two model families. This finding is not only practically useful, but also offers a conceptual window into the inner mechanics of model adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a novel method called IA2 (Instruction Alignment with In-Context Learning Activations) to improve the performance of supervised fine-tuning for large language models (LLMs). The key idea is to leverage the activations from in-context learning (ICL) to guide the supervised fine-tuning process, thereby enhancing the model's ability to follow instructions and generalize to unseen tasks.

### Strengths
Novel Conceptual Framework: The paper introduces a novel idea of aligning in-context learning (ICL) activations with supervised fine-tuning. The concept of using ICL activations as a regularization mechanism during fine-tuning is a creative combination of existing ideas (ICL and supervised learning) that addresses a specific limitation in current fine-tuning methods.

Problem Formulation: The authors identify a gap in the existing literature: the inconsistency between the behaviors of models during ICL and supervised fine-tuning. By framing the problem as an alignment issue, they provide a new perspective on how to improve instruction-following capabilities in large language models (LLMs).

Application to a New Domain: The paper demonstrates the effectiveness of IA2 in the context of instruction-following tasks, which is a critical domain for LLMs. This application is particularly relevant given the growing importance of LLMs in real-world scenarios where instruction following is essential.

### Weaknesses
- This aligns with the intermediate hidden states of knowledge distillation, logits alignment, and attention alignment, which seem to make little difference, limited only to different inputs?

- The experiments are conducted on a limited set of datasets and models. This may restrict the understanding of how IA2 performs across different model architectures and scales (<= 4B). For instruction-following tasks, larger and more challenging datasets such as SNI and P3 can be used.

- Compare IA2 with other alignment techniques such as prompt tuning, continuous prompt tuning, or other regularization methods like dropout or weight decay. This will help in understanding the unique benefits and trade-offs of IA2.

- The IA2 method involves additional computational overhead due to the alignment process. The paper does not discuss the computational efficiency or resource requirements of implementing IA2, which could be a significant limitation for practical applications.

- typos: line 257 `upto`

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposed IA2 (ICL Activation Alignment), which employ a two stage training method to align the models activation (self-attention output) as when using ICL. The proposed method is based on the observation of the huge gap between models activation during SFT and ICL. The experiment results shows consistent improvement of the proposed IA2 over baseline (simple SFT) on a wide range of models and benchmarks.

### Strengths
1. The paper is well organized and easy to follow.
2. The observation of the gap of models activation between ICL and SFT is interesting, demonstrating different internal mechanisms of models when performing ICL and SFT.
3. The proposed IA2 method is simple and novel.

### Weaknesses
1. How the methods works on some of the SOTA open source model is unclear. While the authors show improvement on pre-trained models such as qwen3-4b-base, whether it can generalize to those post-training language models (eg. qwen3-4b) is unclear.
2. The authors tested the method on tasks with direct prediction without reasoning. While the performance on these tasks are good, recent LMs paradigm have been solving tasks such as gsm8k with reasoning process. It's unclear whether the IA2 method can help with this paradigm. One experiment I'll like to see is on gsm8k, compare sft and IA2 models performance when using reasoning during inference on qwen-4b-base model, to see if this method also help with models when performing reasoning.

### Questions
1. When IA2 aligns the activation (self-attention output) between ICL and SFT, I'm curious about the engineering choice. For example, it also makes sense to align attention weight instead of the attention output. Is there any specific reason that you choose to align the activations?
2. While IA2 shows performance improvement on some tasks that generally benefit from in-context examples, do you think it will work well on reasoning tasks that often does not benefit from in-context examples?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a two-stage fine-tuning method for low-resource settings to bridge the gap between task-specific fine-tuning and in-context learning: the method first performs a self-distillation step by matching the activation patterns in LLMs between the ICL prompt and the supervised fine-tuning (SFT) prompt, and then performs an SFT step upon the warm-started model. Experiments across multiple benchmarks show that the two-stage method outperforms SFT only and ICL only in low-data settings, with not only better accuracy, but also better calibration.

### Strengths
1. The method is well-motivated by the misalignment of the inner mechanism of transformers between ICL and SFT, and the proposed method is in principle well-designed to address such a misalignment by explicitly optimizing the exact objective.

2. The paper conducts solid experiments across multiple tasks, spanning both tasks with single-token outputs and multiple-token outputs, and demonstrates the empirical strength of the proposed method, not only in terms of accuracy, but also calibration.

3. The paper presents in-depth analyses in the discussion section and provides ablation results of different components of the proposed method.

### Weaknesses
1. Baselines from the context distillation literature are missing from the experiments. As the authors mentioned in the related work section, the proposed method is similar to context distillation methods in the literature, where models are trained to distill information in the context.  However, the proposed method is not compared against any appropriate context distillation methods, such as ICL distillation (Yukun Huang et al., 2022) and MetaICL (Sewon Min et al., 2022). The experimental results can be strengthened by including such baselines.

2. There is a potential unfair comparison in terms of data when comparing IA2 with ICL only and SFT only baselines. Given that IA2 only needs to run two passes of inference, one over the query-only data example and one over the query + demonstration data example, this gives IA2 effectively 2 times the data seen during training compared with ICL only and SFT only (ICL only sees the query+demonstration data example, and SFT only sees the query-only data example). Moreover, the paper focuses on the low-data regime, where such a misalignment of data sizes can lead to a large difference in performance. Thus, in my opinion, the proposed method needs to be compared with the baselines where the number of data examples is better aligned.

3. The proposed method does not perform consistently well on multi-token tasks, especially GSM8K and HMathA. Particularly in Table 2, SFT or IA2 is worse than the ICL-only baseline with 4 training examples. These results undermine the generalizability of the proposed method.

### Questions
See the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper IA2 first aligns a model’s internal activations to those it produces during in-context learning, then runs standard supervised fine-tuning—yielding SFT models that better mimic ICL and show improved accuracy and calibration across tasks in low-data settings.

### Strengths
- IA2 offers a simple, data-efficient, label-free priming signal (activation matching) that can precede ordinary SFT.

- This paper has a broad empirical sweep with many tasks (SST-2/FinPhrase/PoemSent, AGNews/BBCN, StrategyQA, SciQ, GSM8K, MATH-Alg), two model families, varied #examples; separate 500-example eval sets held out from all adaptation methods.

### Weaknesses
1. LoRA edits only WQ, WK, WO (not WV), rank fixed at 8; authors note selections “subject to improvement,” but this restriction could cap IA2 capacity and bias results vs SFT-only. Ablations on rank/targets would clarify. 

2. Did the authors consider direct comparisons to context distillation or representation-distillation baselines that imitate ICL outputs (e.g., teacher-student on logits/hidden states) under the same setup? I didn't see any results reported in the main results; that would better isolate IA2’s contribution beyond output-level distillation. (Related work, e.g., [1]-[4] is cited but not reproduced head-to-head.)

---

[1] Gustavo Aguilar et al., Knowledge distillation from internal representations, 2020.

[2] Tong Chen et al., Demonstration distillation for efficient in-context learning, 2024.

[3] Lucas Caccia et al., Training plug-n-play knowledge modules with deep context distillation, 2025.

[4] Brian K. Chen et al., Exact conversion of in-context learning to model weights in linearized-attention transformers, 2024.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
