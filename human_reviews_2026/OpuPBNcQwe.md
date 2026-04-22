# Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 8, 4, 4

## Abstract
Large Language Models (LLMs), despite advances in instruction tuning, often fail to follow complex user instructions. Activation steering techniques aim to mitigate this by manipulating model internals, but have a potential risk of oversteering, where excessive emphasis on the instruction degrades task accuracy and overall text quality. To address this, we introduce DIRECTER (Dynamic rejection steering), a novel steering method that dynamically modulates steering strength by scaling the KV cache without extra dataset. DIRECTER couples steering with a plausibility-guided decoding loop, which adaptively adjusts steering strength at each step by comparing the steered output distribution to the original. If the steered output is deemed implausible, steering strength is progressively weakened. This strength modulation is guided by a lightweight, one-time attention sensitivity analysis that ranks layers by their influence on model representations. Extensive evaluations show that DIRECTER significantly enhances instruction-following capabilities across diverse benchmarks, improving accuracy by up to 6.5% over baselines without the common trade-offs in generation quality or task fidelity. The proposed dynamic, plausibility-guided control during activation steering further demonstrates its potential as a general mechanism for mitigating oversteering that is compatible with existing baselines.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work introduces an attention-based steering mechanism for improved instruction following that maintains output quality and improves performance.

### Strengths
This work is well-written and scoped, and provides solid empirical evidence for their proposed control methodology. DIRECTER avoids the problem of "over-steering," which is present in other steering methods applied broadly to the input. The authors show that their method is successful across different benchmarks, model scales, and can be used successfully in conjunction with many different architectures. Importantly, they show that DIRECTER is efficient and performs well against simpler interventions such as plain prompting.

### Weaknesses
* Fixed plausibility threshold, while effective, is applied globally based on a hyperparameter sweep on a subset of the data, but it's unclear in the paper whether this generalizes across models or data domain types. A per-task adaptive value or assessment could help validate the applicability of this parameter.

* The LLM-based evaluation without human validation is a limitation, given the claims of text quality. Reliability of this as a metric should have a subset assessed manually to validate whether the text quality is maintained.

### Questions
1. How sensitive is the threshold to different domains or different models?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes DIRECTER, a dynamic activation-steering method that improves instruction-following of LLMs by adaptively adjusting during decoding. The main idea is a plausibility-guided decoding loop that compares steered and raw token distributions at each step. If the steered token is deemed implausible, steering strength is reduced by halving the set of “steered layers,” ranked by a precomputed attention sensitivity score.

### Strengths
1. The plausibility-guided decoding loop is conceptually simple yet effective. Its dynamic halving mechanism mimics reinforcement learning–style adaptive control but without extra training.
2. The method introduces minimal memory overhead and modest throughput reduction (≈ 16%)
3. The extensive ablation and robustness analyses (Fig. 2–3) convincingly demonstrate the benefits of dynamic steering and the effectiveness of layer ranking.

### Weaknesses
1. The attention sensitivity metric (Eq. 3–4) is ad-hoc and lacks theoretical justification. The “direct” and “propagated” effects are computed via cosine distance differences, but no intuition or derivation is provided. Why does summing cosine-distance deviations across layers accurately capture “influence”?
2. The method section is hard to follow due to poor narrative flow, cross references and undefined notation. For example, Readers are told what each symbol means after it’s used — e.g., $𝐿_{cand,𝑡}$ appears in the plausibility rule before explaining how it’s initialized or updated. The paper didn't provides a clear figure or algorithm before diving into formulas. It is better to summarize their method in a pseudocode algorithm. 
3. Eq. (2) assumes top-1 plausibility comparison, but this is sensitive to token-level noise. It could be better if they evaluate with multiple thresholds $\beta \in [0.3, 0.5, 0.7]$ to show sensitivity (only $\beta = 0.5$ is used).

### Questions
* Have you tried applying DIRECTER to other forms of controlled generation (e.g., factuality correction, refusal control)? A small-scale results would demonstrate the method’s broader applicability and strengthen the paper’s impact claim that DIRECTER is a “general mechanism” for mitigating oversteering.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces DIRECTER , a novel inference-time activation steering method that enhances the instruction-following ability of large language models . Unlike existing static steering methods which risk oversteering，DIRECTER adaptively adjusts the steering strength at each decoding step through a plausibility-guided decoding loop. It scales the key-value  cache layers dynamically based on attention sensitivity analysis, ensuring plausible outputs while mitigating instruction misalignment. Experiments on IFEval, LIFBench, and GSM8K-Format show consistent improvements (up to +6.5%) in instruction accuracy without sacrificing fluency or reasoning performance. The method is model-agnostic and incurs modest computational overhead (~16% latency).

### Strengths
- The paper accurately identifies the "oversteering" problem in existing activation steering techniques. The proposed DIRECTER provides a dynamic, adaptive control mechanism rather than relying on static, manually-tuned hyperparameters.
- The results strongly demonstrate the method's key advantage: it significantly improves instruction following without sacrificing core task accuracy or text quality.
- The evaluation is thorough, using diverse benchmarks including IFEval (strict instructions), LIFBench (long context), and the newly-created GSM8K-Format.

### Weaknesses
- RQ2 requires  the generalization of DIRECTER across different architectures and model sizes. To make the comparison more complete, Table 2 should also include results of other steering-based methods such as PASTA and SpotLight on more LLM architecture (e.g., Qwen) .
- The choice of the probability-ratio–based plausibility criterion (Eq. 2) is currently mainly empirical. However, commonly used measures for comparing probability distributions include KL divergence or JS divergence. The authors should compare these alternatives to justify Eq. (2). 
- The method proposed by the authors shows some improvements in text quality and task performance. However, from an efficiency perspective, the overhead is prohibitive. In particular, the time required to generate the first token is roughly five times of SpotLight.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
