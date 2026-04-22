# ReST-KV: Robust KV Cache Eviction with Layer-wise Output Reconstruction and Spatial-Temporal Smoothing

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6, 6

## Abstract
Large language models (LLMs) face growing challenges in efficient generative inference due to the increasing memory demands of Key-Value (KV) caches, especially for long sequences.
Existing eviction methods typically retain KV pairs with high attention weights but overlook the impact of attention redistribution caused by token removal, as well as the spatial-temporal dynamics in KV selection.
In this paper, we propose ReST-KV, a robust KV eviction method that combines layer-wise output **Re**construction and **S**patial-**T**emporal smoothing to provide a more comprehensive perspective for the KV cache eviction task. 
Specifically, ReST-KV formulates KV cache eviction as an optimization problem that minimizes output discrepancies through efficient layer-wise reconstruction. By directly modeling how each token’s removal affects the model output, our method naturally captures attention redistribution effects, going beyond simplistic reliance on raw attention weights.
To further enhance robustness, we design exponential moving average smoothing to handle temporal variations and an adaptive window-based mechanism to capture spatial patterns.
Our method, ReST-KV, significantly advances performance on long-context benchmarks. It surpasses state-of-the-art baselines by 2.58\% on LongBench and 15.2\% on RULER. Additionally, ReST-KV consistently outperforms existing methods on Needle-in-a-Haystack and InfiniteBench, all while achieving a remarkable 10.61$\times$ reduction in decoding latency at 128k context length. The code is included in the supplementary material and is designed for easy reproduction.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
To address the limitations of existing KV cache eviction method, which overlook attention redistribution effects and spatial-temporal dynamics in token importance, this paper proposes ReST-KV, a robust eviction framework for long-context LLMs. ReST-KV reformulates eviction as a layer-wise output reconstruction optimization problem: it quantifies KV importance by the reconstruction loss caused by token removal and enhances robustness via two smoothing mechanisms. Experiments on 5 LLMs across 4 long-context benchmarks (LongBench, RULER, Needle-in-a-Haystack, InfiniteBench) show  accuracy improvement. It also integrates seamlessly with budget allocation (PyramidKV/AdaKV) and quantization (KIVI/KVQuant) techniques.

### Strengths
1. Unlike baselines (SnapKV/H2O) that rely solely on raw attention weights, ReST-KV’s eviction indicator is derived from MHA output reconstruction loss. This formulation explicitly models two critical effects: (1) Nonlinear attention reweighting amplifies high-competition KV pairs; (2) Redistribution sensitivity measures how well remaining tokens compensate for the removed KV pair. Ablation shows this indicator outperforms attention-weight-only methods.
2.  Assigns higher weights to recent KV pairs mitigates short-term fluctuations in token importance. Ablation confirms it outperforms mean/inv-EMA smoothing by 1.84% accuracy.

### Weaknesses
1. ReST-KV is only tested up to 128k context. For 256k+ tokens, two issues arise: (1) The fixed observation window size may fail to capture long-range spatial dynamics (e.g., "vertical-slash" patterns spanning 1k+ tokens); (2) Cumulative errors in EMA smoothing could distort temporal importance estimation (older tokens are overly dampened). The paper does not report performance at 256k+ or adjust hyperparameters for extreme lengths.
2.  Computing the layer-wise reconstruction indicator requires extra MHA forward passes for each token removal (theoretically O(L) per eviction step). For 128k context, this adds non-negligible prefill overhead, though the paper reports TTFT comparable to full cache (0.97×), it does not quantify the breakdown of reconstruction calculation time. This could become a bottleneck for low-latency applications (e.g., real-time dialogue).

### Questions
1. How does ReST-KV perform at 256k/512k context lengths? It is better providing latency/accuracy data for 256k or longer context on benchmarks and analyze cumulative smoothing errors.
2. What fraction of prefill time is spent computing the layer-wise reconstruction indicator? Can lightweight approximations (e.g., sampling 10% of tokens to estimate reconstruction loss) reduce overhead without accuracy degradation?
3. ReST-KV is evaluated on autoregressive LLMs, does its design generalize to non-autoregressive (NAR) or diffusion-based LLMs (e.g., LLaDA)? NAR models have different attention dynamics, do the reconstruction indicator and spatial-temporal smoothing need modifications?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors argue that existing eviction methods, which primarily retain tokens based on high raw attention weights, are suboptimal. In light of the core flaw they identify is that these methods ignore the effect of attention redistribution, they propose ReST-KV, a novel KV cache eviction method that explicitly models attention redistribution and spatial-temporal dynamics, significantly improving LLM efficiency for long-sequence generation.

### Strengths
1. Reframing eviction as an output reconstruction problem rather than a key-query similarity problem is a more principled and robust approach.

2. The experiments are thorough. The method is tested on multiple models om multiple benchmarks and under various cache budgets.

3. Ablation study shows the effectiveness of the proposed method.

### Weaknesses
1. Does ReST-KV only evict tokens from the prompt during prefill? Or does it also evict tokens from the generated context during decoding?

2. The autghors justifies that the greedy, one-token-at-a-time removal heuristic by citing local linearity assumptions.This is a very strong assumption, as the softmax operation is highly non-linear, and removing one token can cause drastic, non-local redistribution. While the greedy approach is pragmatic, its justification is weak. A brief discussion of why this approximation holds in practice would be beneficial.

3. The Adaptive Window-Based Spatial Smoothing is complex and need more explanation. It computes the average index of the top-B tokens in two halves of a window to get a "shift," which then defines a new window size ($W_s$) and shift ($\gamma_{shift}$). This feels like an overly-engineered heuristic. A simple and intuitive explanation is needed here. And how sensitive is the model to the hyperparameter $\beta$?

4. authors may need to compare the most recent baseline[1]

References:

[1] LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models, ICML'25

### Questions
see weaknesses

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
4

### Summary
The paper proposes ReST-KV, a KV cache eviction method for efficient long-context LLM inference. It reformulates KV eviction as a layer-wise output reconstruction problem to capture attention redistribution effects and introduces spatial-temporal smoothing (EMA and adaptive windowing) to stabilize KV selection.

### Strengths
- The paper correctly identifies the attention redistribution problem ignored by prior eviction methods and offers a principled reconstruction-based indicator.

- Experiments are extensive, covering multiple long-context benchmarks and backbone models.

### Weaknesses
- Limited novelty: The layer-wise reconstruction objective largely follows standard Taylor-based pruning and reconstruction heuristics, and the smoothing mechanisms resemble temporal averaging used in previous cache compression work. Conceptually, the step from SnapKV to ReST-KV is incremental.

- The approach is not compatible with prefix caching frameworks such as SGLang. Because ReST-KV relies on layer-wise recomputation and global reconstruction loss, it cannot be easily merged into multi-request serving or cross-session KV reuse. This seriously limits real-world deployment.

### Questions
- How can ReST-KV be made compatible with prefix cache reuse or shared KV pools in modern inference systems such as SGLang or vLLM?

- What is the computational cost of evaluating Eq. (7) in real-time decoding, and how is it amortized?

- How would the approach behave under multi-turn interactive scenarios where KV eviction and reuse interleave dynamically?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a KV cache eviction method ReST-KV for LLM inference to improve robustness and efficiency under limited cache budgets. The key idea is to reformulate the eviction process as a layer-wise output reconstruction problem, estimating token importance by measuring output discrepancies when a KV pair is removed. The method further incorporates spatial-temporal smoothing through EMA and adaptive windowing to stabilize token selection over time. Experiments show improvements in accuracy and latency over prior eviction baselines.

### Strengths
- The authors clearly articulate the practical motivation behind efficient KV cache management.

- The proposed method is evaluated across multiple backbone models, and comprehensive ablation studies are conducted for each component.

- I have reviewed and reproduced the code in Appendix; it is well-organized and easy to follow.

### Weaknesses
- Measuring token importance by reconstruction error is essentially equivalent to attention-weight reweighting already explored in H2O, which models the same relationship between attention redistribution and output sensitivity.

- The EMA and adaptive window strategies are simple post-processing heuristics. Also, the authors do not include any hyperparameter sensitivity analysis.

- The evaluation focuses on retrieval-oriented long-context benchmarks, without including instruction-following or multi-turn dialogue scenarios.

### Questions
please see weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes ReST-KV, a novel KV cache eviction method that reformulates token importance as a layer-wise output reconstruction problem, explicitly accounting for attention redistribution upon token removal. It further enhances robustness via spatial-temporal smoothing: exponential moving average (EMA) for temporal dynamics and an adaptive window mechanism for spatial shifts. Evaluated across LongBench, RULER, Needle-in-a-Haystack, and InfiniteBench, ReST-KV consistently outperforms SOTA methods—e.g., +2.58% on LongBench, +15.2% on RULER—and achieves a 10.61× decoding speedup at 128K context with minimal accuracy loss.

### Strengths
* Principled formulation: The reconstruction-based importance metric (Eq. 7) is theoretically grounded and captures attention redistribution, a key limitation of prior attention-weight-only methods.
* Strong empirical results: Consistent gains across diverse benchmarks, models (Llama2/3, Mistral, Qwen, Gemma), and cache budgets (64L–1024L).
* Robust design: Spatial-temporal smoothing significantly improves stability, as validated by ablation (Table 3).
* Practical impact: 10×+ speedup and 36% memory reduction enable real-world deployment of long-context LLMs.
Compatibility: Integrates seamlessly with budget allocation (PyramidKV, AdaKV) and prefill acceleration (FlexPrefill).

### Weaknesses
* Computational overhead: Computing reconstruction error requires per-token MHA re-evaluation during prefill, which may slow down the prefill phase (though decoding benefits).
* Hyperparameter tuning: EMA factor α and spatial scale β require tuning, though sensitivity analysis shows robustness.
* Limited theoretical analysis: While empirically sound, formal guarantees on reconstruction error vs. downstream task performance are missing.
* Focus on prefill-only eviction: No dynamic eviction during decoding, which may limit adaptability in streaming settings.

### Questions
* What is the prefill-time overhead of computing the reconstruction-based indicator compared to SnapKV or H2O?
* Could the reconstruction loss be approximated more efficiently (e.g., via Jacobian or first-order Taylor expansion)?
* How does ReST-KV perform in multi-turn dialogue where context evolves incrementally over many turns?
Is the method compatible with sparse attention patterns (e.g., Longformer, BigBird) during prefill?

### Soundness
3

### Presentation
3

### Contribution
3
