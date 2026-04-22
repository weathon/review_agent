# Top-Theta Attention: Sparsifying Transformers by Compensated Thresholding

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 8

## Abstract
We present Top-Theta (Top-θ) Attention, a training-free method for sparsifying transformer attention during inference. Our key insight is that static, per-head thresholds can be calibrated to retain the desired constant number of significant elements per attention row. This approach enables content-based sparsity without retraining, and it remains robust across data domains. We further introduce compensation techniques to preserve accuracy under aggressive sparsification, establishing attention thresholding as a practical and principled alternative to top-k attention. We provide extensive evaluation on natural language processing tasks, showing that Top-θ achieves 3-10x reduction in $V$-cache usage and up to 10x fewer attention elements during inference while degrading no more than 1% in accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a novel method called Top-$\theta$ attention, which can approximate top-$k$ attention computation without using sorting algorithms or other approximation variants. This approach demonstrates a 3- to 10-fold reduction in V-cache usage and up to 10 times fewer attention elements during inference, while degrading accuracy by no more than 1%.

### Strengths
* **High Efficiency**: Reduces **V-cache usage by 3–10×** and attention elements by **up to 10×** with <1% accuracy loss.

* **Robust to Domain Shift**: Thresholds are **model-intrinsic** — calibrated once and work across tasks and datasets.

* **Better Than Top-\(k\)**: Replaces expensive top-\(k\) search with **constant-time thresholding**, removing row-wise dependencies.

### Weaknesses
* This paper lacks a comprehensive review of the field of sparse attention, which has already been popular to study for a long time.
* This paper lacks a comprehensive experimental comparison with current popular methods; it is not clear how this method outperforms other sparsity-based attention approximation algorithms.
* The authors present a new method with promising performance in this work, but after checking the whole paper, it is still not clear what this paper wants to solve. If they just provide a new method that approximates attention efficiently without losing too much accuracy, they should do a comprehensive experimental comparison with current popular methods; otherwise, the community cannot tell why this work is better than others and why people should use this method rather than others.
* This paper lacks a comparison of the efficiency among the top-$\theta$ attention and other methods, especially when it involves additional inference; the additional computational cost should be considered, too.

### Questions
* Please see Weaknesses
* Line 124-125 "We conjecture that the top-k search algorithms....": Could you explain why this conjecture holds?

### Soundness
2

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
This paper proposes Top-Theta Attention, a training-free method for sparsifying transformer attention during inference. The key idea is that static, per-head thresholds can be calibrated to retain the desired constant number of significant elements per attention row.

### Strengths
+ The proposed method does not require training.
+ The proposed method shows effectiveness on the state-of-the-art open-source LLM models.

### Weaknesses
- Limited novelty. Sparse attention has been extensively studied, and many state-of-the-art methods already exist. The proposed Top-Theta or threshold-based sparsification appears to be a minor variation of the well-known top-k attention, which has been applied both in standard transformer architectures and in large language models. The contribution, therefore, seems incremental rather than fundamentally new.

- Unclear motivation. The paper does not clearly explain why a content-based attention mechanism is necessary or how it addresses specific limitations of existing methods. The proposed approach essentially performs pruning by comparing attention scores against a threshold, but it remains unclear why this thresholding scheme is conceptually or practically superior.

- Unjustified design choices and missing ablations. Several components of the proposed method are introduced without sufficient justification or explanation. Moreover, the paper lacks ablation studies to demonstrate the contribution of each component or to support the design decisions made.

- Insufficient experimental validation. The experimental section lacks comprehensive comparisons with existing sparse-attention methods and does not include evaluations on modern large-scale LLMs. As a result, it is challenging to evaluate the generality and practical effectiveness of the proposed technique.

### Questions
See the Weaknesses.

### Soundness
2

### Presentation
2

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
The paper proposes Top-Theta Attention, a training-free sparsification method for transformer attention at inference. Key idea: instead of doing per-row top-k search, the authors pre-calibrate a static per-head, per-row threshold so that, on average, each attention row keeps ≈k important entries. At inference, attention scores are just compared to this threshold — a pure elementwise op — so no row-wise dependency, and it works with tiled / distributed kernels. To offset the loss from dropping entries, they add two compensations: Softmax Denominator Compensation (SDC) to better approximate post-softmax sparsity, and V-Mean Compensation (VMC) to add back the average contribution of discarded tokens. On LLaMA-2, LLaMA-3 and LLaMA-3.1 (7B–70B), across ARC-C/E, HellaSwag, HumanEval, and LongBench, they report 3–10× fewer V-rows / attention elements with ≤1% accuracy loss, sometimes even small gains, and the thresholds calibrated on ARC-C transfer to code and long-context tasks, suggesting the thresholds are model-not-data specific. Calibration needs only a few hundred samples and is one-time per model.

### Strengths
1. **Training-free & model-centric.** Calibrate once per model, a few hundred samples, then reuse across domains (ARC-C → HumanEval/LongBench) — this is much cheaper than retrain/fine-tune-based sparsity.
2. **Tile- and kernel-friendly formulation.** Pure elementwise thresholding; no per-row top-k that breaks tiling. This is exactly where existing top-k attention hurts.
3. **Strong empirical coverage.** LLaMA-2, LLaMA-3, LLaMA-3.1; 7B→70B; prefill (QA) and decoding (HumanEval, LongBench); GQA case analyzed.
4. **Storage overhead negligible**. Thresholds are tiny (~12MB for LLaMA-3-70B).

### Weaknesses
1. **No wall-clock / kernel-level evaluation.** The main selling point is “better for tiled / distributed kernels than top-k,” but the paper does not show an actual implementation or runtime vs. FlashAttention+Top-k baselines on GPU.
2. **Calibration cost vs. model/library changes not fully discussed.** If KV layout or rotary settings change (common in serving stacks), do we need to recalibrate?

### Questions
1. **On SDC choice.** You present three SDC estimators (offline, exp-threshold, exact). For an actual GPU kernel, which one do you expect to be used, and what’s the real compute/memory overhead relative to plain Top-j? Please quantify for LLaMA-3-8B decode (one token).
2. **GQA union strategy.** You mention two alternatives (discard low-support V-rows; or union across heads). Did you test either? If not, what made you decide on per-head selections only?
3. **Per-row vs parametric thresholds.** You mention fitting a parametric function to rows to reduce storage. Did you actually try it, and what is the accuracy drop vs. full per-row table?

### Soundness
4

### Presentation
4

### Contribution
3
