# InferSpec: Adaptive Inference-Time Compute with Ensemble Verifier-Guided Speculative Decoding for Efficient Reasoning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Large language models (LLMs) are effective at multistep reasoning, but suffer from high inference costs, making efficient deployment challenging. Although speculative decoding (SD) offers latency reductions by letting a lightweight draft propose tokens that a stronger target verifies, yet its token-centric nature admits subtle flaws in intermediate steps to propagate, ultimately producing incorrect final output.
The existing literature, such as reward-guided SD, rely on external pre-trained reward models, which increase latency and limit generalizability. To overcome this limitation, we propose InferSpec, a mathematically grounded, verification-aware framework for adaptive inference-time compute allocation.
At each step, InferSpec samples multiple draft candidates and applies a self-consistency selector to choose a representative one. It then evaluates the selected step using two model-internal criteria: (i) Attention-Based Grounding Verification (ABGV), which computes grounding scores from attention rollout matrices to ensure attribution to inputs or prior steps, and (ii) Log-Probability-Based Verification (LPBV), which bounds token-level confidence. 
These signals form a weighted ensemble score with formal guarantees that only grounded, high-confidence steps are accepted; uncertain steps escalate to the target model, allocating compute selectively.
Experiments on MATH500, GSM8K, Gaokao-2023-En, and OlympiadBench show that InferSpec improves accuracy by up to 3.6% while reducing latency by 11%, consistently outperforming both standard SD and reward-guided SD.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes INFERSPEC, a speculative decoding framework for multi-step reasoning that aims to replace expensive external reward models. It uses a lightweight "ensemble verifier" based on two model-internal signals: Attention-Based Grounding Verification (ABGV) and Log-Probability-Based Verification (LPBV). This verifier adaptively accepts draft steps or escalates to a stronger target model.

### Strengths
1.	Important Problem: The paper tackles the critical problem of high inference cost in LLM reasoning.

2.	Clear Motivation: The goal of replacing heavy, external Process Reward Models (PRMs) with lightweight, model-internal signals is excellent and practical.

### Weaknesses
1.	(Major Weakness) Lack of Foundational Validation: The paper fails to provide any direct evidence (e.g., correlation studies) that its two core signals, ABGV and LPBV, are meaningful proxies for step-wise reasoning correctness. The entire proposal is built on an unproven premise.

2.	Lack of Generality: The paper's claims of efficiency and accuracy are only supported by experiments on mathematical and formal reasoning datasets (MATH500, GSM8K, etc.) . It is completely unclear if this method has any utility in open-ended or creative generative tasks.

3.	Missing Critical Ablation: The method combines the new verifier (ABGV+LPBV) with a Self-Consistency Selector. The paper does not ablate the effect of the selector, making it impossible to know if the small accuracy gains come from the novel verifier or just from applying the known self-consistency technique.

### Questions
1.	Can the authors provide a direct quantitative analysis (e.g., a correlation) to demonstrate that the proposed ABGV and LPBV scores actually correlate with ground-truth step-wise correctness?

2.	What is the performance of a baseline that uses only the Self-Consistency Selector to choose from k draft candidates, but uses a simpler verification rule (e.g., standard SD or just LPBV)? This is critical to isolate the true contribution of the ABGV verifier.

### Soundness
2

### Presentation
3

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
This paper proposes InferSpec, a framework that enhances speculative decoding for LLM reasoning tasks through adaptive inference-time compute allocation. Different from Standard speculative decoding (SD) and reward-guided SD (RSD), InferSpec replaces external verifiers with two model-internal verification signals: (i) Attention-Based Grounding Verification (ABGV), which uses attention rollout matrices to ensure outputs are grounded in inputs or validated prior steps, and (ii) Log-Probability-Based Verification (LPBV), which enforces token-level confidence. At each reasoning step, InferSpec samples multiple draft candidates, applies a self-consistency selector to identify the most representative one, then evaluates it using a weighted ensemble of ABGV and LPBV scores. Steps passing the threshold are accepted; others trigger target model recomputation. Experiments on MATH500, GSM8K, Gaokao-2023-En, and OlympiadBench show InferSpec improves accuracy by up to 3.6% while reducing latency by ~11% compared to RSD and standard SD.

### Strengths
1. InferSpec introduces complementary internal signals (ABGV via attention rollout, LPBV via log-probabilities) that eliminate external PRM overhead while maintaining generalizability. 
2. Thorough evaluation across four reasoning benchmarks, three model families, and multiple baselines (majority voting, Best-of-N, beam search, SD, RSD). Systematic ablations examine design choices, and qualitative analysis effectively illustrates PRM failure modes.
3. InferSpec consistently achieves higher accuracy with lower latency than RSD (e.g., 95.8% in 34 minutes vs. 88.7% in 41 minutes on GSM8K). Complexity analysis confirms practical efficiency gains from avoiding external verifiers.

### Weaknesses
1. The approach exhibit limited domain generalizability. All experiments focus exclusively on mathematical reasoning. No evidence demonstrates that ABGV and LPBV generalize to code generation, commonsense reasoning, or other domains where attention patterns and confidence distributions may differ substantially.
2. The parameter setting of InferSpec seems to be crucial and determined empirically. Self-consistency selector overhead are not compared against simpler strategies. 
3. Authors should consider comparing the InferSpec to more recent baselines such as SpecReason: https://arxiv.org/abs/2504.07891 by Rui Pan, Yinwei Dai, Zhihao Zhang, etc.

### Questions
1. Have authors evaluated InferSpec on non-mathematical reasoning domains (e.g., code generation, commonsense reasoning, multi-hop QA) to validate whether ABGV and LPBV generalize beyond math-specific attention and confidence patterns?
2. ow does InferSpec compare against more recent speculative reasoning methods such as SpecReason (Pan et al., 2025)?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on efficient generation by incorporating SLM and LLM. Traditional speculative decoding (SD) method requires a strict distribution alighnment between LLM and SLM, and its token-centric nature admits subtle flaws in intermediate steps to propagate. A recent variant, RSD, relies on external reward, introducing latency.

In this paper, the authors propose InferSpec that samples multiple steps with SLM, selects a reprentative one with semantic similarity, then evaluates it with two model-internal criteria (attention-based grounding verification and log-probability based verification). If the selected step is rejected, the LLM will sample multple steps, chooses the representative one. And the decoding is switch again to SLM. 

With theoretical justification and extensive experiments, InferSpec shows comparable or better accuracy and efficiency than SD and RSD. Detailed ablation studies also justify the dsign choice.

### Strengths
1. The paper is well-written with a clear motivation, i.e. using model-internal signal to verify the step instead of external reward.
2. The theoretical justification supports the proposed method.
3. Extensive experiments show the accuray and efficiency benefits from InferSpec.

### Weaknesses
1. Lack of baseline. Since InferSpec applies the target model to sample multiple steps, it's suggested to include the majority vorting results from the target model only. 
2. Unfair runtime comparison. According to Table 1, it seems RSD is originally designed without majority voting. With majority voting, RSD performs much worse. When comparing runtime with RSD in Figure 2, I don't see the reason to compare with RSD majority.
3. Potential more memory consumption. According to L147, we need to save the attention matrices from each layer to compute R. This saving becomes more memory-consumed for larger model and longer output.

### Questions
1. Why do you still use the target model to generate multiple steps? What is the difference bwteen generating only one step and multiple steps with the target model?

### Suggestion
1. Table 1: highlight the best results for easy observation.

### Soundness
3

### Presentation
3

### Contribution
3
