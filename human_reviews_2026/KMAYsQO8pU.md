# SAES-SVD: Self-Adaptive Suppression of Accumulated and Local Errors for SVD-based LLM Compression

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
The rapid growth in the parameter scale of large language models (LLMs) has created a high demand for efficient compression techniques. 
As a hardware-agnostic and highly compatible technique, low-rank compression has been widely adopted. However, existing methods typically compress each layer independently by minimizing per-layer reconstruction error, overlooking a critical limitation: the reconstruction error propagates and accumulates through the network, which leads to amplified global deviations from the full-precision baseline.
To address this, we propose **Self-Adaptive Error Suppression SVD (SAES-SVD)**, a LLMs compression framework that jointly optimizes intra-layer reconstruction and inter-layer error compensation.
SAES-SVD is composed of two novel components:
**Cumulative Error-Aware Layer Compression (CEALC),**  which formulates the compression objective as a combination of local reconstruction and weighted cumulative error compensation. Based on it, we derive a closed-form low-rank solution relied on second-order activation statistics, which explicitly aligns each layer's output with its full-precision counterpart to compensate for accumulated errors.
\ding{183} Adaptive Collaborative Error Suppression (ACES), which automatically adjusts the weighting coefficient to enhance the low-rank structure of the compression objective in CELAC. Specifically, the coefficient is optimized to maximize the ratio between the Frobenius norm of the compressed layer's output and that of the compression objective under a fixed rank, thus ensuring that the rank budget is utilized effectively.
Extensive experiments across multiple LLM architectures and tasks show that, without fine-tuning or additional tricks, SAES-SVD consistently improves post-compression performance. For example, at a 0.2 compression ratio on LLaMA-7B, existing methods exhibit an average accuracy drop exceeding 0.05, whereas SAES-SVD restricts the drop to only 0.02. These improvements underscore the potential of SAES-SVD to effectively narrow the gap between compressed models and their full-precision counterparts, paving the way for more reliable compression of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the SAES-SVD framework, which leverages two key modules - Cumulative Error-Aware Layer Compression (CEALC) and Adaptive Collaborative Error Suppression (ACES) - to overcome the limitations of prior SVD compression methods that only independently minimize reconstruction error for individual layers while ignoring the layer-wise propagation and accumulation of compression errors throughout the model network. By dynamically optimizing weighting coefficients for each layer, the framework maximizes retained energy under fixed rank budgets, thereby providing a more effective LLM compression strategy.

### Strengths
- The proposed SAES-SVD framework is well-motivated and novel, addressing a fundamental yet previously overlooked limitation in prior SVD-based compression methods.
- This paper demonstrates exceptional mathematical rigor, providing comprehensive theoretical derivations for both CEALC and ACES components, which elevates the work from empirical observation to a theoretically grounded advancement.
- The framework achieves strong performance without any additional training, showcasing its straightforward and effective design.

### Weaknesses
- The proposed method is only evaluated on the LLaMA family of models, raising concerns about its generalizability and effectiveness across other popular LLM architectures such as Qwen.
- The paper only compares with SVD-based methods, lacking comparison with other compression approaches like structured pruning and quantization, which limits understanding of its overall effectiveness.

### Questions
1. Have the authors tested SAES-SVD on other popular LLM architectures beyond the LLaMA family?
2. How does SAES-SVD perform compared to other major compression paradigms such as structured pruning and quantization methods?
3. There appears to be an inconsistency between Figure 3 and the "Comparison on larger-scale models" section in Section 5: while the text claims Dip-SVD achieves 6.64 perplexity on LLaMA-30B, Figure 3 shows this result missing for LLaMA-30B and only displays 6.64 for LLaMA-13B. Could the authors clarify this discrepancy? On a related note concerning the same figure, could the authors also explain the anomalous trend for the ASVD method, where its performance degrades significantly on the larger LLaMA-30B model, contrary to the behavior of other methods?

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
3

### Summary
This paper proposes SAES-SVD (Self-Adaptive Error Suppression SVD), a framework that jointly optimizes for local reconstruction fidelity and global error compensation. It consists of two core componets: CEALC actively compensates for upstream accumulated errors by aligning each layer’s output with its full-precision counterpart; ACES automatically tunes the error compensation strength for each layer to enhance the low-rank structure of the objective. These components together mitigate the critical issue of error propagation inherent in layer-wise compression. Experiments show that SAES-SVD significantly outperforms existing methods, substantially narrowing the performance gap with the original model without requiring any post-compression fine-tuning.

### Strengths
- Adequate experiments and compelling results:
  - The experiments are comprehensive, covering multiple models and scales. The results are compelling, consistently outperforming strong SVD baselines (even those requiring fine-tuning), which effectively highlights the method's superiority.
- Systematic and interpretable methods:
  - The proposed method is theoretically sound and well-motivated. It systematically addresses the error accumulation problem, and the ACES component provides an elegant, closed-form solution for adaptive tuning, making the framework interpretable and efficient.

### Weaknesses
- Lack of computational complexity and time analysis:
 The paper does not provide a detailed evaluation of the computational overhead during the compression process. The time cost of statistics collection and ACES optimization, relative to baseline methods, is not quantified precisely.
- Limited comparison beyond SVD-based approaches:
 The evaluation focuses only on SVD-based baselines. It remains unclear whether the proposed method would still outperform non–SVD-based compression methods under the same compression ratio.

### Questions
- Detailed time breakdown:
 Could you provide a comprehensive breakdown of the total compression time (including statistics collection and ACES optimization) and compare it with baseline methods?
- Combination with quantization:
 Have you considered integrating SAES-SVD with quantization techniques such as GPTQ or AWQ? Since these methods address orthogonal types of redundancy (structural vs. numerical), such a combination could potentially achieve even higher compression efficiency and better performance.

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
3

### Summary
This paper proposes SAES-SVD, motivated by the observation that SVD-based compression for large language models leads to cross-layer error propagation and accumulation. Implementation-wise, the authors gather streaming second-order statistics by running two forward passes on the same mini-batch, thereby avoiding activation caching and keeping overhead manageable. Experiments span LLaMA-7B/13B/30B and LLaMA-3-8B across multiple compression ratios. Under a unified protocol that requires no finetuning and no mixed-rank assignment, the method consistently reduces perplexity, maintains or improves zero-shot accuracy, and delivers ~1.29×–3.79× inference speedups, demonstrating strong practicality and scalability.

### Strengths
1. Compelling motivation on cumulative error. The paper identifies a real pain point in SVD compression for LLMs, cumulative cross-layer error during inference—and directly targets it. The motivation is well supported by the empirical evidence in Figure 1, which demonstrates the phenomenon clearly.

2. Solid theoretical underpinnings (CEALC & ACES). Both components—CEALC and ACES—come with clear formulations and derivations. The overall approach is coherent: the objective design is principled, and the analysis provides a sound theoretical basis for the proposed procedure.

3. Thorough and convincing experimentation. The method is validated across multiple models and datasets, generally achieving better accuracy and lower perplexity while also reducing end-to-end latency. The breadth of settings and the consistency of gains add credibility to the claims.

### Weaknesses
1. Theoretical limitations and missing robustness analyses.
The fixed-subspace approximation used by ACES may break down under small spectral gaps or large perturbations. The current mitigation (β caps and shrinkage) is largely engineering-based. The paper would benefit from robustness curves bucketed by spectral gap, as well as a deeper theoretical justification for using RER and an explicit discussion of how RER improvements translate to final PPL.


2. Limited architectural diversity in experiments.
Evaluations focus primarily on the LLaMA family. Results on other architectures (e.g., Qwen) are missing, which leaves open questions about generality across model designs.


3. No combination with mixed-rank strategies.
Although the paper claims to outperform mixed-rank baselines under a uniform-rank setting, it does not explore combining SAES-SVD with mixed-rank schemes (e.g., ASVD, Dobi-SVD). Whether such combinations could further improve performance remains unaddressed.

### Questions
1. please refer weaknesses.
2. Please explain the differences and advantages of this article compared to the method in "AA-SVD: Anchored and Adaptive SVD for Large Model Compression", which is also submitted to ICLR 2026.

I'm willing to raise my score if my concern is resolved.

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
This paper proposes SAES-SVD (Self-Adaptive Error Suppression SVD), a framework for compressing large language models (LLMs) using low-rank decomposition. The key contribution addresses a critical limitation in existing layer-wise compression methods: the accumulation and propagation of reconstruction errors across network layers.

### Strengths
1. **Theoretically grounded approach**: The derivation of closed-form solutions based on second-order activation statistics provides a principled mathematical foundation. The formulation that combines local reconstruction with weighted cumulative error compensation is elegant and well-motivated.

2. **Adaptive mechanism**: The ACES component that automatically adjusts weighting coefficients is a practical contribution that removes the need for manual hyperparameter tuning across different layers and models.

3. **Consistent improvements**: The reported results show consistent improvements over baseline methods on the evaluated benchmarks, with the accuracy drop on LLaMA-7B at 0.2 compression ratio (0.02 vs 0.05) being noteworthy.

### Weaknesses
1. **Outdated evaluation benchmarks**: The datasets used for evaluation appear to be somewhat dated. Modern LLM compression research should include more challenging and recent benchmarks that better reflect current application demands and model capabilities.
2. **Limited model coverage**: The experiments focus primarily on medium-sized models like LLaMA-7B. To demonstrate the method's generalizability and practical value,
3. **Insufficient baseline comparisons**: The paper should compare against a broader range of compression techniques including recent quantization methods (e.g., GPTQ, AWQ, SmoothQuant).

### Questions
1. **Model scaling**: How does SAES-SVD perform on more recent and larger models? Specifically:
   - Can you provide results on LLaMA 3.1 (8B, 70B) and Qwen 2.5 series?
2. **Task diversity**: Can you evaluate on more challenging and diverse tasks? Choose 2 of this gourp please.
   - Long-context reasoning tasks (>8K tokens)
   - Code generation
   - Mathematical reasoning
3. **Practical integration**: 
   - Have you tested integration with popular inference frameworks?
   - What modifications are needed to existing serving infrastructure?

### Soundness
2

### Presentation
2

### Contribution
2
