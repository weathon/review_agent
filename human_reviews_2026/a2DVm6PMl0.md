# SID: Multi-LLM Debate Driven by Self Signals

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Large Language Models (LLMs) have exhibited impressive capabilities across diverse application domains. Recent work has explored Multi-LLM Agent Debate (MAD) as a way to enhance performance by enabling multiple LLMs to discuss and refine responses iteratively. Nevertheless, existing MAD methods predominantly focus on utilizing external structures, such as debate graphs, using LLM-as-a-Judge, while neglecting the application of self signals, such as token logits and attention, that arise during generation. This omission leads to redundant computation and potential performance degradation. In this paper, we shift the focus to the self signals of multi-LLM debate and introduce a Self-Signals Driven Multi-LLM Debate (SID), which leverages two types of self-signals: model-level confidence and token-level semantic focus, to adaptively guide the debate process. Our approach enables high-confidence agents to exit early at the model level and compress the redundant debate contents based on the attention mechanism. We evaluate our method on various LLMs and Multimodal LLMs across multiple challenging benchmarks. Experimental results demonstrate that our method not only outperforms existing MAD techniques in accuracy but also reduces token consumption, highlighting the effectiveness of utilizing self signals in enhancing both the performance and efficiency of multi-agent debate systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposed a multi-agent debate method that relies on internal signals such as token probabilities (or equivalently, entropy) as well as attention scores with respect to a predefined instruction prompt. It differs from most prior related works that only relies on external signals. Experiments show the effectiveness and efficiency of their proposed method.

### Strengths
The authors draw inspiration from the internal signals of LLMs and extend these ideas to the context of multi-agent debate, which, to the best of my knowledge, is a novel perspective.

They further conduct experiments on both LLMs and MLLMs to demonstrate the effectiveness and efficiency of the proposed method.

### Weaknesses
1. The writing of the paper is fairly problematic. For example, the format of citation (the use of \citep, \citet) leaves a bad first impression on the evaluation of the paper. Moreover, while the presentation is generally understandable, it lacks clarity and polish, which affects the readability and professionalism of the paper.

2. Intuitively, the proposed method seems primarily designed to improve efficiency. For example, the early-exit mechanism aims to avoid redundant generation when the model is sufficiently confident. However, it remains unclear why this mechanism, along with the attention-based compression approach, would lead to substantial performance improvements. The authors should provide deeper analysis or more convincing explanations to clarify how these efficiency-oriented components jointly enhance the multi-agent system's overall performance. This should be a key point of this paper.

3. Some closely related studies are missing. For example, [1] discusses the limitations and applicability of multi-agent debate systems. Including such references and comparing with them would strengthen the context and completeness of the paper.

[1] If multi-agent debate is the answer, what is the question, arXiv preprint arXiv:2502.08788

### Questions
See Weakness 2. The intuition why these efficiency-oriented methods can improve the overall performance is confusing. For example, if you do not use the early-exit mechanism (and only use the compression method), will you achieve a worse performance? If not, what are the potential reasons?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Self-Signals–driven multi-LLM (SID) debate, a framework to leverage MAD with LLM internal signals: (1) model-level confidence, derived from token entropy; and (2) token-level semantic focus, from attention patterns conditioned on disagreement-oriented prompts. Experiments show that SID improves accuracy over MAD/DMAD while reducing token usage.

### Strengths
1. The paper presents another method to use self-signals for LLM multi-agent debate. 
2. The author clearly described their method and it is easy to understand for the general reader. 
3. The experiments show competitive performance.

### Weaknesses
1. The authors present this method as more token-efficient than other baselines, but we only see a comparison with MAD. How are the other methods comparing with it? How much of this efficiency is coming from the early-exit mechanism?
2. Selection of MAD/DMAD baselines seem reasonable, but fail to compare with a state-of-the-art LLM judge. 
3. The authors fail to identify relevant related work in the are of LLM confidence/uncertainty in MAD research. And therefore, the use of self-signals cannot be claimed as novel. The authors make this claim in the first point of their contributions. They should rephrase as another method using self-signals in MAD. For instance, [ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs](https://arxiv.org/abs/2309.13007) or [DebUnc: Improving Large Language Model Agent Communication With Uncertainty Metrics](https://arxiv.org/abs/2407.06426)
4. If I am not mistaken, the paper goes over the 9-page limit.

### Questions
1. How much of this efficiency is coming from the early-exit mechanism?
2. The authors present using self-signals in MAD as a novel technique, but as indicated in weaknesses other papers have already investigated the use of self-reported confidence and logit-based metrics as metrics for MAD. I'd suggest re-writing this contribution as another method of self uncertainty/confidence in MAD. 
3. Given that the experiments are run on OOS models, why are confidence intervals not included?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SID, a self-signals–driven framework for multi-LLM debate that replaces external controllers (e.g., LLM-as-a-judge) with two internal cues from the models themselves. The first cue is a model-level confidence score derived from token-probability statistics (entropy/NLL) that triggers an early-exit gate so confident agents stop debating; the second is a token-level “semantic focus” signal extracted from attention maps to compress debate histories into disagreement-relevant spans with a semantic-preservation heuristic. The early-exit mechanism can be implemented with a vocabulary-adaptive threshold or a lightweight calibrated classifier, with similar empirical performance and no training needed for the thresholded version. SID is evaluated on both LLMs and MLLMs, consistently improving accuracy over MAD/DMAD baselines while cutting token usage by up to ~30–40%.

### Strengths
* SID is well motivated for improving MAD from several critical perspectives
* The evaluation is comprehensive, covering a wide range of LLMs/MLLMs and different benchmarks
* The proposed components are novel, and be validated through fine-grained balation studies

### Weaknesses
* Missing baselines. There are several previous works with related confidence mechanisms, e.g., EoT [1] also incorporated a model-level confidence measurement. However, this work is not discussed or compared in this paper


[1] Exchange-of-Thought: Enhancing Large Language Model Capabilities through Cross-Model Communication

### Questions
* "Prompt-conditioned Attention Extraction" helps LLMs focus on the points of disagreement within a debate. Can we have the model do this by itself, rather than relying on external attention computations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Existing Multi‑Agent Debate (MAD) methods often rely on external structures and ignore self‑signals, which leads to redundant discussion and extra token consumption. The authors leverage entropy‑based model‑level confidence and token‑level semantic focus for early stopping and adaptive compression.

### Strengths
1. Experiments show the algorithm reduces token consumption (up to ~30%).
2. It also yields some performance improvement.
3. The paper is generally well written.
4. Although the use of logits for early stopping or attention to identify disputed segments can be found in the broad ML literature (see references below),  this paper is the first to apply them systematically within MAD. 

References

Schuster, T., Fisch, A., Gupta, J., et al. Confident Adaptive Language Modeling. NeurIPS, 2022.

Laaouach, Y. HALT‑CoT: Model‑agnostic early stopping for chain‑of‑thought reasoning via answer entropy. Muslims in ML Workshop @ ICML 2025, 2025.

Yang, C., Si, Q., Duan, Y., et al. Dynamic Early Exit in Reasoning Models. arXiv:2504.15895, 2025.

Corallo, G., Papotti, P. FINCH: Prompt‑guided key‑value cache compression for large language models. Transactions of the Association for Computational Linguistics (TACL), 2024.	

Fu, Q., Cho, M., Merth, T., et al. LazyLLM: Dynamic token pruning for efficient long context LLM inference. arXiv:2407.14057, 2024.

### Weaknesses
1. The core contribution is algorithmic, but the code is not open‑sourced.
2. The algorithm appears to constrain debate within homogeneous models, and calibrating confidence‑gating across different models seems challenging. If heterogeneous multi‑model debate is not supported, it underuses the potential of the MAD paradigm.
3. The evaluated benchmarks are somewhat limited (MMLU‑Pro, MATH, ScienceQA, MMStar); consider expanding to more benchmarks.
4.cBecause the method requires access to attention/logits, it is readily applicable only to open‑source/white‑box models; this is a limitation for MAD deployment scenarios.

### Questions
Zhang et al. report that, in many settings, Self‑Consistency (SC) can be both cheaper and more accurate than MAD. In your paper, however, SC seems unusually weak (e.g., GPT‑OSS‑20B on MMLU‑Pro: ~57). Could you explain why?

Zhang, H., Cui, Z., Wang, X., et al. If Multi‑Agent Debate Is the Answer, What Is the Question. arXiv:2502.08788, 2025.

### Soundness
2

### Presentation
3

### Contribution
2
