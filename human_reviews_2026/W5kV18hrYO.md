# AFORA: Activation-aware Factorization with Optimal Rank Allocation for Training-free LLM Compression

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 6, 2, 2

## Abstract
Large language models are challenging to deploy because of their extreme size and compute demands. In this work, we propose AFORA (Activation-aware Factorization with Optimal Rank Allocation for Training-free LLM Compression), a simple and hardware-friendly framework that directly reduces the number of parameters through low-rank factorization of weight matrices. AFORA consists of two core components: (1) Activation-aware Weight Factorization (AWF), a closed-form low-rank approximation that accounts for the input activation distribution to preserve task-relevant directions and ensure numerical stability; and (2) Optimal Rank Allocation (ORA), a global rank allocation strategy that assigns heterogeneous ranks across layers to minimize activation distortion under a given budget. Evaluations across multiple large-scale language models show that our framework consistently outperforms existing approaches at the same compression ratios, while additionally reducing model size, saving memory, and decreasing computation with hardware-friendly layer dimensions. It also requires only a short runtime to perform compression, and offers a principled mathematical interpretation. These results demonstrate that activation-aware, globally optimized low-rank compression offers a practical and theoretically grounded path to efficient LLM deployment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces AFORA, a training-free compression framework for LLMs. The method combines two key ideas: (1) Activation-aware Weight Factorization (AWF), which performs low-rank decomposition by accounting for input activation, and (2) Optimal Rank Allocation (ORA), a global optimization strategy that allocates ranks across layers. The paper provides strong theoretical grounding, deriving the method from loss approximations under the K-FAC assumption, and proving optimality of the global allocation. Experiments on LLaMA-2-7B demonstrate that AFORA outperforms existing training-free compression baselines.

### Strengths
- The paper is technically rigorous, with clear mathematical derivations and proofs connecting activation-aware factorization to second-order loss approximations.

- The paper is well-written, conceptually coherent.

- Empirical performance is strong, albeit demonstrated on a single model.

### Weaknesses
- The evaluation is limited to a single model (LLaMA-2-7B), leaving open the question of generalization to other model families or scales. As a compression technique is meant to be model-agnostic, demonstrating consistency across architectures would significantly strengthen the empirical claims.

- Although the method is described as hardware-friendly, there is no profiling on diverse hardware or latency benchmarks.

- The method’s behavior under extreme compression ratios (<5%) or its compatibility with quantization/pruning hybrids is not explored.

### Questions
- How sensitive is AFORA to the quality or size of the calibration dataset used to estimate Σ_in? Would smaller or domain-shifted calibration streams degrade performance?

- Can the authors provide results on other model families to demonstrate architectural generality?

- What is the runtime or memory overhead of computing Σ_in and performing SVDs at large scales ?

- How does AFORA interact with quantization or pruning if applied sequentially ?

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
This paper introduces AFORA, a training-free LLM compression framework that uses low-rank factorization to reduce model size. AFORA has two main components:
- Activation-aware Weight Factorization (AWF): A closed-form method that approximates weight matrices by minimizing output error, considering the input activation distribution. Unlike standard SVD, which focuses on weight reconstruction, AWF preserves task-relevant information.
- Optimal Rank Allocation (ORA): A global strategy that assigns different ranks to each layer to minimize overall error under a fixed parameter budget. It uses a water-filling algorithm to find the globally optimal rank distribution.

On LLaMA-2-7B, AFORA outperforms baselines like SVD and ASVD at various compression ratios. At 15% compression, it significantly lowers perplexity, reduces compression time by up to 9.6×, and achieves a 1.21× inference speedup while maintaining competitive zero-shot accuracy.

### Strengths
- AFORA's key innovation is its principled combination of activation-aware factorization (AWF) and globally optimal rank allocation (ORA). This moves beyond heuristic approaches by providing a closed-form, loss-aware solution for factorization and a theoretically grounded method for budget allocation.
- ORA exhibits greater elegance than the rank searching method in ASVD, owing to its robust theoretical background.
- AFORA addresses the need for efficient LLM deployment. Its training-free nature avoids costly fine-tuning, and its hardware-friendly output (dense matrices) ensures broad compatibility. It demonstrates an good balance between performance and efficiency.

### Weaknesses
- Experiments are confined to LLaMA-2-7B. The paper lacks evidence of AFORA's effectiveness on other model architectures (e.g., Qwen), scales, or fine-tuned variants (e.g.,Chat version), which limits the claims of its general applicability.
- AFORA achieves the same average zero-shot accuracy but much lower ppl as ASVD at 15% compression. The paper does not analyze why these differences occur, making it difficult to understand the unique benefits of AFORA's design.

### Questions
- AWF vs. SVD-LLM: Could you provide a quantitative comparison between AWF and SVD-LLM on LLaMA-2-7B, including perplexity, compression time, and numerical stability? This would clarify AWF's advantages.
- Zero-Shot Parity with ASVD: Can you provide a deeper analysis of the zero-shot results? For instance, why does AFORA excel on certain tasks (e.g., CB) but lag on others (e.g., GSM8K) compared to ASVD? Does this relate to layer-specific error patterns?
- ORA vs. ASVD Rank Allocation: Could you visualize the rank distributions produced by ORA and ASVD across layers? Do both methods prioritize the same layers (e.g., attention vs. MLP)? Explaining any differences would help validate ORA's claimed optimality.

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
3

### Summary
The paper introduces a training-free low-rank compression method (AFORA) for LLMs, built upon two ideas.
1. Activation-aware weight factorization (AWF): Instead of applying SVD directly to each weight matrix, the method uses the empirical covariance of real activations collected from a calibration dataset during the weight factorization and low-rank approximation.
2. Optimal rank allocation (ORA): Under a fixed total storage budget, the method assigns different numbers of ranks to different layers via a water-filling procedure.

### Strengths
In terms of originality, this paper proposes a new method (AFORA) that compresses the LLMs, which incorporates the covariance information of the input activations into a closed-form low-rank approximation of the weight matrices.

In terms of quality, the proposed method is mathematically grounded. Both components (AWF and ORA) follow directly from minimizing a clear objective rather than relying on heuristics.

In terms of clarity, the motivations, mathematical derivations, and method descriptions are clear and easy to follow.

In terms of significance, the evaluations show better or similar results than other low-rank approximation methods in both perplexity and zero-shot benchmark accuracies. The method is also efficient and fast to run.

### Weaknesses
My main concern is the evaluations, which are very confusing to me.

1. The definition of compression ratio seems inconsistent across the paper.

- The compression ratio is defined in Line 357 as the fraction of parameters retained relative to the dense model.
The ratio columns in Tables 1, 5, and 6 show 85%, 95%, and 90%, with the dense model being 100%, which is consistent with this definition.
However, the captions of Tables 1, 5, and 6, the ratio columns of Tables 3 and 4, the footnotes of Tables 1 and 2, as well as everywhere mentioning the ratio numbers in the main text (including on Line 357 right in front of the definition) use 5%, 10%, and 15%, which is in fact one minus the compression ratio.

- Only the attention projection layers (query, key, value, and output, as described on Line 323) are compressed.
The feed-forward projection layers (gate, up, and down), which take about $\frac{4096 \times 11008 \times 3}{4096 \times 11008 \times 3 + 4096 \times 4096 \times 4} \approx 66.8\\%$ of weights in LLaMA-2-7B, are preserved.
The embeddings, prediction head, and layer norms are also preserved.
The authors should specify whether the compression ratio is computed with respect to the whole model or only the attention layers.
If it is computed only with respect to the attention layers, removing 15% (the maximum ratio tested in the paper) of the attention weights will result in only 5% weight deduction for the whole model, which is not significant.

- I do not understand the last sentence on Line 323: "the same compression ratio is applied across the full model using our global allocator".
It is clear that different layers have different compression ratios using the global rank allocator.

2. Table 4 is not clear.
On Line 101, the reduction in the parameter count and the FLOPs are proportional, while the number of FLOPs reduces much faster than the parameters in Table 4.
The author should fix the definition of compression ratio and better define the FLOP counts, like whether the FLOPs include the attention calculations that increase quadratically with respect to the sequence length.
The calculation method of the speedup is not specified in the paper.
The authors should also mention the input size (batch size and sequence length), as the speedup is generally very sensitive to it.

3. The method seems to be restricted to square weight matrices in practice, while the theoretical part does not impose this restriction.
The evaluations are only done on a small and old LLaMA-2-7B model that has four square weight matrices (the attention projections) in each block.
Note that in many recent LLMs, e.g., Llama-3 series, only query and output layers have square weights (key and value layers become non-square due to head sharing).

4. The baseline evaluations are limited.
The results are primarily compared against other SVD-based methods rather than the more widely used LLM compression techniques, such as the quantization and pruning methods described in Section 2.3, making it difficult to assess the method's relative advantages in practical deployment scenarios.

### Questions
1. Algorithm 1 is not referred to in the paper.
The tie-fill step is not explained in the paper.
The description is not clear.
What does "increament the $m$" mean?

2. Why is there a bf16 mark for the dense model in Tables 1, 5, and 6? Are the low-rank methods not using the same data type as the dense model? Also, the LLaMA-2-7B uses float16 (FP16) as the default datatype, not bfloat16 (BF16).

3. How is the perplexity reduction of 5.24% on Line 473 calculated? I do not find the corresponding numbers in Table 1 that generate this ratio.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes AFORA, a training-free LLM compression method that (i) performs activation-aware low-rank factorization (AWF) by doing SVD and a whitened operator $ W\Sigma_{in}^{1/2} $, and (ii) allocates ranks globally across layers via an optimal rank allocation (ORA) scheme derived from a budgeted utility maximization. Experiments on LLaMA-2-7B show modest improvements over baselines at the same nominal compression ratio.

### Strengths
1. Clear and principled derivation from activation-aware objective, to whitening, to closed-form factorization.
2. Rank allocation is formulated cleanly and is computationally light.
3. It is fully training-free, which makes it easy to deploy. Uses only activation statistics from a calibration set.
4. Empirically, AFORA is at least competitive with strong SVD-style baselines.

### Weaknesses
1. Evaluation is on a single, older model (LLaMA-2-7B). No results are available for newer or structurally different models.
2. Gains over ASVD in zero-shot are small; in several budgets ASVD $ \ge $ AWF, so the incremental benefit of AWF is unclear.
3. No variance / CI reported, making close comparisons hard to trust. Experiments seem to be single-seed.
4. A Calibration size / source sensitivity ablation could have improved the results.
5. Reported compression “ratios” are inconsistently defined (text = retention, tables = reduction) and lead to an error. i.e., in the text and table caption, the ratio is defined as retention, and the results in the 5% table are stronger than those in the 15% table!
6. The statement in §2.3 that ultra-low-precision quantization “rarely” yields actual inference gains is too strong given recent work (e.g., QuTLASS).

### Questions
1. Can you report AFORA on more recent models and at least one more model family (Mistral/Mixtral, Qwen) and different sizes?

2. Can you add the missing ASVD + ORA baseline in the ablations to isolate whether the gain truly comes from AWF?

3. Please clarify the compression-ratio definition and fix the 5%/10%/15% inconsistency in the tables.

4. Can you report mean ± std (or at least 3-seed averages) for perplexity and zero-shot tasks where gaps are small?

5. Your naming in the TL;DR uses TACO. Please fix.

### Soundness
3

### Presentation
3

### Contribution
2
