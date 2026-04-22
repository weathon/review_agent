# Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding

- Avg Score: 5.00
- Decision: Accept (Oral)
- Scores: 6, 0, 8, 6

## Abstract
Verification is a key bottleneck in improving inference speed while maintaining distribution fidelity in Speculative Decoding. Recent work has shown that sequence-level verification leads to a higher number of accepted tokens compared to token-wise verification. However, existing solutions often rely on surrogate approximations or are constrained by partial information, struggling with joint intractability. In this work, we propose \emph{Hierarchical Speculative Decoding (HSD)}, a provably lossless verification method that significantly boosts the expected number of accepted tokens and overcomes joint intractability by balancing excess and deficient probability mass across accessible branches. Our extensive large-scale experiments demonstrate that HSD yields consistent improvements in acceptance rates across diverse model families and benchmarks. Moreover, its strong explainability and generality make it readily integrable into a wide range of speculative decoding frameworks. Notably, integrating HSD into EAGLE-3 yields over a 12\% performance gain, establishing state-of-the-art decoding efficiency without compromising distribution fidelity. Code is available at https://github.com/ZhouYuxuanYX/Hierarchical-Speculative-Decoding.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Hierarchical Speculative Decoding (HSD), a verification method for speculative decoding that seeks to be provably lossless while accepting longer draft prefixes than token-wise or blockwise verification.   The key idea is to view verification as hierarchical branch resampling with a single resampling step using a capped, branch-local resampling distribution. Several experiments on the Qwen series model demonstrate the effectiveness of the method.

### Strengths
- Across three benchmarks and multiple target sizes, HSD shows consistent  improvements in block efficiency and throughput over token-wise and block-wise verification
- This paper is overall well-written and easy to follow.
- The appendices provide solid derivations and decomposition lemmas.

### Weaknesses
- Results only use Qwen models and three academic datasets; evaluation on modern LLMs (e.g., Llama-2/3 families) and diverse tasks (long-context, multilingual, tool-use) would strengthen external validity.
- HSD’s acceptance and resampling probabilities depend on capped branch divergences computed over the vocabulary for each position. The authors may want to provide a clear complexity/latency breakdown of these additional reductions relative to blockwise verification (e.g., per-step FLOPs, memory accesses, kernel counts).

### Questions
please refer to weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The paper proposes a method termed Hierarchical Speculative Decoding for accelerating inference in LLMs by improving the verification stage of speculative decoding. The claim is that existing methods struggle with “joint intractability” when trying to verify a draft sequence at once, so HSD uses a hierarchical branching/verification approach to boost the number of accepted tokens (i.e., less rollback) in a “lossless” manner.

### Strengths
The topic: Accelerating generative inference in LLMs is a highly important problem

### Weaknesses
1 The flow of the paper is not smooth. The theoretical derivation of “joint intractability” and then the transition to the hierarchical method is abrupt. It is not always clear how the high-level algorithm ties into the low-level proofs and experiments.
2 While the paper describes “hierarchical speculative decoding”, the exact steps 
ie, branch generation, verification hierarchy, mass-balancing, acceptance criteria, are somewhat buried in dense math and less in intuitive explanation or pseudo-code. Readers may struggle to follow the pipeline end-to-end.

### Questions
It is hard to identify in one glance “Given draft model → generate K branches → verify hierarchically → accept or rollback”

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Hierarchical Speculative Decoding (HSD), a novel approach to improving the verification process in speculative decoding. The paper addresses verification bottleneck by proposing HSD, a lossless verification method that overcomes joint intractability by resampling portions of the target distribution in a hierarchical manner. Extensive experiments show that HSD significantly improves the expected number of accepted tokens, particularly when dealing with longer draft sequences. The method is also proven to work effectively across different model sizes and tasks, making it a promising tool for accelerating LLM inference.

### Strengths
- The proposed hierarchical branch resampling strategy is a novel and creative approach to addressing joint intractability in speculative decoding. 

- The theoretical analysis is rigorous, and the experimental validation is comprehensive, showing consistent improvements across various benchmarks.

- The paper is clearly written, and complex ideas are explained clearly. The use of figures and equations aids in understanding the methodology and its underlying theory.

### Weaknesses
- It’s unclear whether the backward scan of HSD introduces any additional computational overhead.

### Questions
- Could the author provide a more intuitive example or explanation of why joint verification leads to a higher expected number of accepted tokens? While the simulation using a toy example is convincing, it still isn't entirely clear to me why this happens.

- For Algorithm 2, at line 15, should it still sample from the target distribution until the sequence reaches $\gamma$?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Hierarchical Speculative Decoding (HSD), a novel approach to address the "joint intractability" problem in speculative decoding for large language model (LLM) inference. The authors identify a fundamental limitation in existing speculative decoding methods (tokenwise and blockwise) related to how acceptance probabilities are calculated for draft sequences. HSD solves this by introducing a hierarchical approach to acceptance probability calculation that better approximates the ideal case while maintaining the lossless property (preserving the target distribution exactly). The paper provides rigorous theoretical analysis proving HSD's correctness and demonstrates consistent improvements over baselines across multiple benchmarks (GSM8K for mathematical reasoning, HumanEval for code generation, and CNN/DailyMail for summarization) using the Qwen2.5 model suite with various scale combinations (0.5B draft model with 14B, 32B, and 72B target models).

### Strengths
1. Exceptionally rigorous theoretical analysis with detailed proofs establishing the lossless property of HSD. The paper clearly demonstrates how HSD correctly recovers the target distribution through careful handling of branch divergence and capped ratios.
2. Significant conceptual contribution by identifying and solving the "joint intractability" problem in speculative decoding - how existing methods miscalculate acceptance probabilities for multi-token sequences, leading to suboptimal performance.
3. Elegant hierarchical approach to acceptance probability calculation that provides a theoretically sound solution while remaining practically implementable. The capped ratio concept and unique capping indices are particularly insightful innovations.

### Weaknesses
1. Limited experimental scope - the paper only evaluates HSD using Qwen2.5 models across three benchmarks. A broader evaluation with multiple model families and additional tasks would strengthen the empirical validation.
2. Insufficient comparisons with state-of-the-art speculative decoding frameworks. While thoroughly comparing against tokenwise and blockwise methods, the paper omits systematic comparisons with more advanced approachesr.
3. Limited ablation studies to understand the contribution of different HSD components. For instance, how sensitive is performance to the identification of "unique capping indices" or the specific hierarchical structure?

### Questions
1. How does the actual computational overhead of HSD (for hierarchical acceptance probability calculations) compare to simpler methods, particularly for smaller draft lengths? Could this overhead offset some gains in block efficiency?
2. The paper mentions HSD can be integrated with multi-draft frameworks but provides limited evaluation. Could you share more comprehensive results comparing HSD Multi-draft against Tokenwise Multi-draft across more model combinations?
4. Have you observed any tasks or model configurations where HSD performs worse than existing methods? Understanding the limitations would help practitioners decide when to use this approach.
5. Could you provide more intuitive examples showing how HSD's hierarchical acceptance mechanism works in practice for concrete generation examples, perhaps with visualizations of the capped ratios and unique capping indices?

### Soundness
3

### Presentation
3

### Contribution
3
