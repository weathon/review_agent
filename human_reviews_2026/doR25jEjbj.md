# Think Just Enough: Sequence-Level Entropy as a Confidence Signal for LLM Reasoning

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 4

## Abstract
We introduce a simple, yet novel entropy-based framework to drive token efficiency in large language models during reasoning tasks. Our approach uses Shannon entropy from token-level logprobs as a confidence signal to enable early stopping, achieving 25-50% computational savings while maintaining task accuracy. Crucially, we demonstrate that entropy-based confidence calibration represents an emergent property of advanced post-training optimization present in modern reasoning models but notably absent in standard instruction-tuned and pre-trained models (Llama 3.3 70B). We show that the entropy threshold to stop reasoning varies from model to model but can be calculated easily in one shot using only a few examples from existing reasoning datasets. Our results indicate that advanced reasoning models often know that they’ve gotten a correct answer early on, and that this emergent confidence awareness can be exploited to save tokens and reduce latency. The framework demonstrates consistent performance across reasoning-optimized model families with 25-50% computational cost reduction while preserving accuracy, revealing that confidence mechanisms represent a distinguishing characteristic of modern post-trained reasoning systems versus their predecessors.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a training-free framework for improving the computational efficiency of LLM reasoning. The core idea is to use the shannon entropy of token-level log-probabilities as a confidence signal for early stopping. If the entropy of the initial reasoning step is below a model-specific threshold, the model is considered confident, and further reasoning is halted, saving tokens. The authors claim this method can achieve 25-50% computational savings without any loss in task accuracy.

### Strengths
* **Insight on emergent confidence calibration**: The paper empirically finds that entropy-based confidence is an emergent property of advanced post-trained models. It's a good insight for the community.

* **Simplicity and applicability**: The proposed method is training-free, model-agnostic, and straightforward to implement.

* **Comprehensive evaluation**: The paper proposes four distinct, mathematically-derived thresholding methods and validates them with appropriate statistical measures (Cohen's d, t-tests, confidence intervals), demonstrating robust performance on challenging reasoning benchmarks (AIME, GPQA).

### Weaknesses
* **Incremental novelty**: While the emergent property analysis is novel, the core idea of using token entropy as a confidence signal for adaptive computation is not new. Works like HALT-CoT [1] and AdaDec [2] have explored similar concepts.

* **Simple modeling**: The paper defines "extended reasoning" as a fixed, 4-step sequential process. Modern advanced reasoning often involves more complex structures, such as iterative self-correction loops. The paper fails to investigate or discuss how to apply to these more practical frameworks.

* **Lack of other baselines**: The evaluation only compares the early-stopping performance against the full 4-step baseline of the same model. It does not compare against other potential confidence heuristics or other methods for improving inference efficiency.

[1] HALT-CoT: Model-Agnostic Early Stopping for Chain-of-Thought Reasoning via Answer Entropy. ICML 2025 workshop

[2] AdaDec: Uncertainty-Guided Adaptive Decoding for LLM-based Code Generation. Arxiv 2506

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper tackles test-time overcomputation in multi-step reasoning LLMs by asking if confidence after the first reasoning step is already sufficient to stop. They compute sequence-level entropy over the top-k (k=20) token probabilities for that step, average over all tokens, and compare to a calibrated threshold: low entropy means halt, high entropy means run the full schedule. They introduce four thresholding schemes (mean, “information-theoretic,” Bayesian, scale-invariant) and claim each can be calibrated from a small labeled set. On AIME-2024/2025 (30 problems each) and GPQA-Diamond (~200), and on the post-trained/reasoning models they use (GPT-OSS, Qwen3), correct first-step traces show lower entropy than incorrect ones, yielding ~25–50% token savings with little or no accuracy loss. They also report a negative result on Llama-3.3-70B, where this separation vanishes, and read it as evidence that “entropy as confidence” is tied to post-trained reasoning models.

However, the contribution is incremental: it stays within the standard entropy/confidence halting paradigm and mostly shifts the probe to “step-1 entropy + τ,” without head-to-head comparisons against the most natural existing halting methods. The evaluation is narrow and curated (tiny, clean, math/science datasets; no messy agent/tool/code/multi-hop settings), and most comparisons are intra-paper plus “full 4-step” rather than against strong prior baselines. Claims like “first comprehensive study … across mathematical and scientific reasoning” should be toned down given the small tests used.

### Strengths
- Clear, well-motivated problem
- Very simple, training-free mechanism, computing sequence-level entropy after the first reasoning step and gate with a single threshold
- Nice empirical observation: on reasoning-tuned models (DeepSeek/Qwen-style) correct step-1 traces have noticeably lower entropy than incorrect ones.
- Honest negative result on vanilla Llama 3.3 70B, showing the signal isn’t magic and seems tied to post-trained “thinking” models.

### Weaknesses
- Incremental beause it is within the entropy/confidence-based halting paradigm; mainly shifts the probe to “step-1 sequence entropy + calibrated τ” but does not compare to the closest existing methods.
- Evaluation is narrow and curated: tiny, clean, math/science datasets (AIME-24/25: 30 items each; GPQA-Diamond: ~200) and no tests on messier traces (tool-augmented agents, codegen, multi-hop QA, chattier models), so the “universal/model-agnostic” claim isn’t supported. Work in this area contains at least one messy dataset as I mentioned in the summary.
- Empirical comparisons are weak: mostly their own threshold variants vs vanilla full 4-step decoding, with no strong baselines from prior halting/entropy work, so real progress is hard to judge.
- They present a general budgeted scheme (A.2) and prove total calls = α, but never ablate α/δ/γ to show actual reallocations. Albations are important to assess the robustness.
- The paper says “information-theoretic / Bayesian / scale-invariant” thresholds, but the appendix relies on hand clamps (e.g. max(0, 1 − σc/µc), log(1 + |d|)), which is less principled than the main-text tone and should be toned down.

### Questions
Refer to weaknesses please

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a simple method using Shannon entropy to measure the confidence of reasoning trajectory, and perform early stopping based on the the confidence measure. The evaluation shows 25-50% token saving compared to basic baselines while preserving accuracy.

### Strengths
1. The paper explores the Shannon-entropy based method to measure sequence-level entropy as a confidence measure. The method is relatively sound. 
2. The writing is easy to follow.

### Weaknesses
1. Novelty. Entropy-based early-exit methods has been quite extensively studied in prior works such as the following. The paper needs to make a clearer distinction between the method and the prior works (in the related work). Just listing a few as addition to the current related work section:
- https://arxiv.org/abs/2502.12067
- https://arxiv.org/abs/2412.21187
- https://arxiv.org/abs/2504.01296
- https://arxiv.org/abs/2508.15260
- https://arxiv.org/abs/2412.20993
- https://arxiv.org/abs/2412.18547
- https://arxiv.org/abs/2207.05221

2. Lack of baseline. The paper doesn't seem to compare against the state of the arts to show the token saving compare to these methods. Therefore, the evaluation is considered not as convincing.

### Questions
1. Related works as stated in the weakness.
2. Lack of baseline as stated in the weakness.
3. Confidence threshold. In section 3.5 the authors mentioned the hyperparameter threshold to choose from. How to choose this value? Any ablation to support your claim?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents an Shannon Entropy based reasoning model early exiting and budget control. It computes the entropy with few shot reasoning sequence, and stop the thinking process once the entropy is higher than a threshold. The author tested four different types of threshold and show that with few shot examples (at most 20 data point), the framework can reduce 25-50% tokens to be generated, while maintaining the same accuracy.

### Strengths
- The idea of using entropy to control the reasoning budget is clean and effective.
- The authors design comprehensive experiments to show the power of using entropy to reduce computation cost.

### Weaknesses
- What is the source of validation dataset? The threshold's universality among different dataset is unclear. Based on Table 1, even the same model on different math problem dataset can have varied entropy range. Adding more details on the threshold value for each dataset and the the validation data can help improve the paper's contribution.
- The core concept "reasoning step" is not well defined, making the paper's soundness not satisfying. What is a reasoning step, how to define the start and end of such a step, and how to find the start and the end during the runtime?
- The experiment lacks comparison with the latest research on the same topic, such as [1][2][3].


[1] Chen, Xingyu, et al. "Do not think that much for 2+ 3=? on the overthinking of o1-like llms."

[2] Fu, Yichao, et al. "Reasoning without self-doubt: More efficient chain-of-thought through certainty probing."

[3] Zhang, Anqi, et al. "Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification."

### Questions
- Why is the number in Figure 4(a) exactly the same for the first two bars, and the same for the last two bars

### Soundness
2

### Presentation
1

### Contribution
3
