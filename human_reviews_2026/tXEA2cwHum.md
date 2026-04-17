# Enhancing Delta Compression in LLMs via SVD-based Quantization Error Minimization

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Fine-tuning is a crucial process for adapting large language models (LLMs) to diverse applications. In certain scenarios, like multi-tenant serving, a large number of LLMs finetuned from the same base model are deployed to meet complex requirements for users. Recent works explore delta-compression approaches to quantize and compress the delta weights between the customized LLM and the corresponding base model. However, they exhibit inadequate performance at high compression ratios due to their empirical nature. In this work, we introduce DeltaMix, an adaptive mixed-precision delta-compression framework designed to minimize quantization error in the singular value decomposition (SVD) space without imposing additional assumptions. DeltaMix provides a theoretical justification for the necessity of mixed-precision compression and presents a practical quantization solution that involves solving a 0/1 linear integer programming problem alongside a reconstruction target correction method. Experimental results across multiple models and benchmarks illustrate that DeltaMix consistently outperforms all baseline methods. Notably, on tasks such as AIME2024 and GQA, DeltaMix exceeds the performance of the best baseline, Delta-CoMe, by 22.3\% and 6.1\% for 7B parameter models, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes an SVD-guided approach for low-bit mixed-precision compression of weight deltas (differences between original and fine-tuned models). The authors first provide a theoretical derivation showing that mixed-precision quantization is beneficial for the V matrix but less useful for the U matrix in SVD decomposition. To compensate for the error incurred by quantization of the V matrix during the U quantization step, the authors introduce Reconstruction Target Correction, which shifts U to a new optimal value. The analysis is accompanied by empirical evidence. Based on this foundation, the authors formulate the optimal compression as an integer programming problem and leverage a dedicated solver. The proposed method is validated on several large language and vision models and compared with prior work on delta compression.

### Strengths
* The SVD-guided search for mixed-precision quantization appears to be novel in the context of model compression.
* Delta-Mix noticeably outperforms baselines in terms of final accuracy for the same compression target.

### Weaknesses
* While theoretically and practically sound, the proposed method—formulated as compression of a model delta of the same size as the original model—still seems less appealing than established PEFT techniques [1, 2, 3, 4]. LoRA adapters can be one or two orders of magnitude smaller than the total number of model parameters, yet remain competitive with full fine-tuning when properly tuned [5]. The learning rate adopted in the experiments (4e-5) may not be optimal for LoRA.

---
References

[1] Hu, Edward J., et al. "Lora: Low-rank adaptation of large language models." ICLR 1.2 (2022): 3.

[2] Liu, Shih-Yang, et al. "Dora: Weight-decomposed low-rank adaptation." Forty-first International Conference on Machine Learning. 2024.

[3] Zhang, Qingru, et al. "Adalora: Adaptive budget allocation for parameter-efficient fine-tuning." arXiv preprint arXiv:2303.10512 (2023).

[4] Kopiczko, Dawid J., Tijmen Blankevoort, and Yuki M. Asano. "Vera: Vector-based random matrix adaptation." arXiv preprint 
arXiv:2310.11454 (2023).

[5] https://thinkingmachines.ai/blog/lora/

### Questions
* Can the proposed method be applied to LoRA adapters? This would potentially enable even higher compression rates and the possibility of serving a large number of fine-tuned versions of a given model simultaneously.


* The proposed method seems to be quantized representation-agnostic. Can Delta-Mix be combined with a vector quantization scheme [1, 2, 3] to achieve even higher compression rates with minimal performance degradation?


* LoRA may sometimes lack sufficient expressiveness to fully capture the difference between two models when the difference is substantial. How well do sparse + low-rank adapters [4] perform in this context?

---
References

[1] Van Baalen, Mart, et al. "Gptvq: The blessing of dimensionality for llm quantization." arXiv preprint arXiv:2402.15319 (2024).

[2] Egiazarian, Vage, et al. "Extreme compression of large language models via additive quantization." arXiv preprint arXiv:2401.06118 (2024).

[3] Chee, Jerry, et al. "Quip: 2-bit quantization of large language models with guarantees." Advances in Neural Information Processing Systems 36 (2023): 4396-4429.

[4] Nikdan, Mahdi, et al. "Rosa: Accurate parameter-efficient fine-tuning via robust adaptation." arXiv preprint arXiv:2401.04679 (2024).

### Soundness
3

### Presentation
2

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
This paper presents DeltaMix, an adaptive mixed-precision delta-compression framework for fine-tuned large language models.
The method decomposes delta weights using SVD and formulates quantization as minimizing layer-wise reconstruction error.
Unlike prior empirical approaches, DeltaMix derives a mathematically grounded formulation showing why mixed-precision is necessary and models bit allocation as a 0/1 integer-linear program with an additional RTC step.
Extensive experiments on reasoning, math, code, and multimodal benchmarks across diverse models demonstrate consistent performance gains, while also reducing GPU memory usage.

### Strengths
**1. Strong theoretical foundation.** The work formalizes SVD-based delta-compression as an explicit quantization-error-minimization problem and proves the necessity of mixed-precision allocation, advancing the theoretical rigor of delta-compression research.

**2. Comprehensive empirical validation.** Evaluations on 7B and 14B LLMs across four domains (reasoning, math, code, vision-language) show clear and reproducible gains over Delta-CoMe, BitDelta, and low-rank baselines.

**3. Practical deployment benefits with thorough system analysis.** The paper provides valuable end-to-end evaluation showing 6× memory savings and superior scaling properties, enabling deployment of different models for uncompressed approaches. 
The analysis of prefill time, generation speed, and varying arrival rates demonstrates real-world applicability beyond just accuracy metrics.

### Weaknesses
**1. Limited scalability analysis.** While integer-linear optimization is solved once per model, reported solving times (≈ 30 min for 7B) may become impractical for larger or frequent model updates. 
Discussion on scaling to 70B+ models is missing.

**2. Ablation study.** Although four task types are covered, the paper lacks ablation on calibration-set size, bit-budget sensitivity, or robustness under distribution shift, which are important for real-world deployment.

**3. Computational overhead.** Table 10 shows DeltaMix requires 3× more time than Delta-CoMe (per block), totaling 1-3 hours for full models. 
While the paper dismisses this as "acceptable since quantization is performed only once," this represents significant overhead for practitioners, especially for larger models or when iterating on model development.

### Questions
**1. Complexity and scalability.** How does the integer-program’s solving time and memory footprint scale with layer size and number of candidate bit-widths? 
Could approximate or heuristic solvers yield near-optimal results faster?

**2. Sensitivity towards calibration data.** How robust is the bit-allocation when the calibration set is small or domain-mismatched?
Does performance degrade significantly with limited calibration data, and how does this compare to baselines that may be less calibration-dependent?

**3. Generalization to other compression forms.** Could the same error-minimization principle be adapted for pruning or hybrid pruning-quantization pipelines?

**Justification for Rating.**

The paper presents a novel and theoretically motivated approach to delta compression.
However, the experimental section lacks sufficient analysis of scalability, sensitivity, and efficiency trade-offs, which limits the practical completeness of the proposed framework.
I am open to raising the score if these concerns are adequately addressed in the rebuttal.

### Soundness
3

### Presentation
3

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
The paper proposes DELTAMIX, a delta-compression framework that works in the SVD space of the fine-tuned-minus-base weight matrix $(W = U\Sigma V)$. The key analytical step decomposes the per-row quantization error for (V) into a fixed “scaling” term $(\Sigma_{ii}^2)$ and a data-dependent “difference” term $(\Delta V_i X X^\top \Delta V_i^\top)$. This yields a rationale for row-wise mixed precision on (V) under a global bit-budget. The bit allocation is cast as a 0/1 integer linear program, and the method introduces a Reconstruction Target Correction (RTC) to reduce bias when later quantizing (U). Experiments across reasoning, math, code, and multimodal tasks claim consistent gains over SVD low-rank, BitDelta, and Delta-CoMe at (\alpha=1/16), including large margins on AIME2024 (e.g., +22.3% over Delta-CoMe for 7B) and improved memory/speed scaling when hosting many deltas. Reported quantization overhead is higher than Delta-CoMe but presented as a one-time cost.

### Strengths
* Principled objective: Explicitly minimizes a reconstruction-error surrogate in SVD space, yielding a clear justification for row-wise mixed precision of (V) under a bit budget. The $(\Sigma_{ii}^2)$ scaling vs. difference decomposition is intuitive and actionable. 
* Concrete optimization: Bit allocation via 0/1 ILP provides a crisp mechanism to trade off error and storage, with constraints for budget and a cap $(f_{\max})$ on distinct bitwidths. 
* RTC mechanism: The Reconstruction Target Correction before quantizing (U) reduces deviation induced by using $(\hat{V})$ as the target, with measurable gains in harder regimes. 
* Empirical coverage: Multi-task evaluation (math/reasoning/code/VLM) across 7B and 13–14B backbones; large improvements are shown where $(\lVert \Delta W \rVert)$ is big (e.g., AIME2024, some multimodal). 
* Serving relevance: Memory and latency scaling when hosting many fine-tuned variants is compelling; DELTAMIX supports more concurrent models than baselines in the reported setup.

### Weaknesses
1. Inconsistency with “no singular-value assumptions.” The method claims to avoid empirical reliance on singular values, yet Section D.1 discards the last (k) ranks by singular-value magnitude to accelerate quantization, explicitly invoking the “larger singular values are more important” heuristic that the paper earlier critiques. This weakens the methodological positioning and may bias comparisons. 
2. Fair-budget accounting is under-specified. Results are reported at $(\alpha = 1/16)$, but the paper does not precisely tabulate end-to-end storage (including $(U, \Sigma, V)$, any indices/masks, solver-driven zero-bit ranks, and calibration metadata) vs. baselines. Without an apples-to-apples byte breakdown, it’s hard to assess dominance beyond accuracy. 
3. Selective gains; some regressions. While DELTAMIX shines on AIME2024 and certain VLM settings, elsewhere it’s only on par or slightly worse (e.g., 13–14B Math500 in Table 2). The average gains (~2–3%) are modest and may not outweigh extra complexity in production settings. 
4. Calibration sensitivity not analyzed. The difference term depends on calibration activations (X). The paper doesn’t study how sample size, domain shift, or layer-wise weighting affect EV estimates and allocations, nor robustness across seeds. 
5. Optimization overhead and practicality. The ILP solve is reported as ~29.4 minutes for a 7B variant; quantization takes ~1.2 h for 7B and ~2.4 h for 14B, versus ~0.4 h / 0.8 h for Delta-CoMe, all on a single GPU. This overhead may be non-trivial at scale, especially if per-task calibrations are required. 
6. Baselines and scope. Comparisons omit some strong PTQ/structured baselines relevant to error control (e.g., SPQR/SPQR-like sparse-quant, SVD-LLM variants in comparable regimes) or server-side delta systems beyond Delta-CoMe. This leaves open whether the observed gains are specific to the chosen set. 
7. Missing systems details. The paper does not clearly state how $(\Sigma)$ is stored/quantized, nor the runtime cost of reconstructing (W) vs. baseline delta formats. The serving experiment is helpful but still abstracts away some operator-level costs. 
8. Ablations are narrow. The $(f_{\max})$ study covers one model/task slice; RTC ablation is limited in breadth. A per-layer bit allocation analysis vs. error/outliers is mentioned, but stronger causal links to downstream accuracy would help.

### Questions
1. Budget parity: Provide a byte-accurate storage table for every method and model (including $(\Sigma)$, indices, zero-bit ranks, any metadata). Confirm that $(\alpha=1/16)$ implies comparable on-disk and in-memory footprints across methods. 
2. Singular-value reliance: Reconcile the claim to “eschew reliance on singular values” with D.1 rank truncation by $(\sigma)$. Can you replicate results without this heuristic, or with a heuristic-free pruning guided solely by EV? 
3. Calibration robustness: How many calibration samples are used per layer, how are they selected, and what is the variance of EV and the ILP allocation across seeds/domains? Show accuracy vs. calibration-set size curves. 
4. RTC cost/benefit: Quantify the computational overhead of RTC and analyze when it helps most. Could a joint optimization of $(U,V)$ under the same objective remove the need for RTC? 
5. $\Sigma$ handling: Are singular values stored in full precision? If quantized, to what precision, and how does that trade off with accuracy vs. bits? 
6. Operators/runtime: Provide end-to-end operator-level latency of reconstructing (W) or applying $(U\Sigma V)$ directly vs. baselines for single- and multi-tenant serving, including prefill and decode breakdowns. 
7. Broader baselines: Add comparisons to SPQR-style sparse-quant and SVD-LLM truncation-aware variants under matching storage, plus recent delta/tuning hybrids, to strengthen the empirical case.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
- The paper proposes EDC (Enhancing Delta Compression), a framework for improving delta-based model compression of large language models. It focuses on compressing fine-tuned deltas between a base model and its adapted version (e.g., instruction-tuned or domain-specific variants). The core idea is to enhance representational compactness by combining adaptive low-rank decomposition, residual quantization, and layer-wise scaling reweighting of delta tensors.

### Strengths
- Addresses a practical problem in model distribution and storage: delta checkpoint compression for multi-task or multi-domain fine-tuned models.

### Weaknesses
- The core idea—combining low-rank and quantized residual compression, is well explored in prior works such as QLoRA, AdaLoRA, and CompAdapter. The proposed “layer-wise scaling reweighting” is a small variant of norm-based importance metrics used in parameter-efficient tuning.

- The method is entirely empirical. The paper lacks mathematical justification or analysis on how the scaling or residual quantization improves representational fidelity beyond heuristic intuition.

- Experiments are restricted to a few fine-tuning tasks and medium-size models (≤13B). No results are provided for large instruction-tuned LLMs (>70B) or multi-domain deltas where compression instability typically arises.

- The paper reports storage reduction but not latency, energy, or end-to-end loading improvements. For real-world LLM deployment, I/O and kernel fusion dominate runtime, which EDC does not address.

### Questions
- How does EDC interact with prefix caching or parameter sharing in serving systems? Can deltas be applied incrementally without full reconstruction?

- What is the compression–accuracy trade-off compared to existing adapter compression frameworks like CompAdapter or LoRA-Prune under identical settings?

### Soundness
2

### Presentation
2

### Contribution
2
