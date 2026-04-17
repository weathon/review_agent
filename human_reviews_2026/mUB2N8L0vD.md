# Taming Massive Activations and Preconditioning Weights: GSR-Guided Quantization for W4A4

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Large language model inference is constrained by memory and latency. Uniform low‑bit quantization would help, but recent evidence shows massive activations—rare, extremely large, and largely input‑invariant per‑token scalars—rather than generic channel‑wise outliers. Methods that “smooth” activation outliers by migrating scale into weights are therefore less effective under this phenomenon. We address this by explicitly rotating activations and preconditioning weights so that both become easy to quantize.

We first identify that the \textbf{grid-to-standard-deviation ratio (GSR)}, 
$
\rho^X_\text{g} = \frac{\Delta_\text{g}}{\operatorname{std}(X_{\text{c}})},
$
is a useful proxy for quantization sensitivity, as it measures the relative coarseness of quantization steps compared to the intrinsic variability of activations. Building on this insight, we introduce \textbf{Flattened Rotation TSVD Quantization(FRTQ)}, a post-training quantization framework tailored for ultra-low-bit settings (e.g., W4A4). For activations (per-token), FRTQ learns orthogonal rotations at function-invariant points to contract GSR and stabilize quantization. For weights (per-channel), FRTQ fits a rank-$r$ truncated-SVD component to capture dominant directions, quantizes the residual, and realizes the correction via a fused low-rank path. All rotations are folded into adjacent weights, with only a single lightweight on-the-fly rotation required at the FFN down-projection.

By explicitly minimizing GSR, FRTQ aligns its updates with quantization error reduction. The method is purely post-training, requires only a small calibration set, and avoids gradient-based fine-tuning. Its alternating updates are simple, scalable, and kernel-friendly. Experiments across standard LLM backbones show that FRTQ consistently reduces GSR and improves W4A4 accuracy compared to smoothing-only or rotation-only baselines. On LLaMA-2 70B, FRTQ lowers $\rho$ of activation by 28.69\% compared to DFrot, and improves W4A4KV4 zero-shot accuracy by 1.25\%, matching higher-bit baselines while incurring negligible runtime overhead.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a post-training quantization method called FRTQ (Flattened Rotation tSVD Quantization) specifically to address the quantization challenges of large language models in the W4A4 setting. The authors note that traditional methods have limited effectiveness when dealing with "massive activations" (i.e., rare but extremely large activation values). Therefore, they propose: Grid-to-Standard-Deviation Ratio (GSR) as a proxy for quantization sensitivity; On the activation side, GSR is reduced through orthogonal rotation (DFRot + Uniform Preconditioning); On the weight side, a tuned truncated SVD (tSVD) is used to absorb the dominant direction and quantize the residual; All transformations are fused back into the weights, resulting in virtually no additional overhead during inference.
Experiments show that FRTQ significantly outperforms existing PTQ methods on multiple LLaMA models, even approaching the performance of QAT methods, without requiring end-to-end training.

### Strengths
Highly Innovative: This paper proposes GSR as a unified and scale-invariant quantization difficulty metric, with solid theoretical analysis.

Practical: This method effectively addresses the difficulties of activation and weight quantization by combining rotation and low-rank decomposition, without requiring any training.

Experimentally Sound: This method is validated on multiple LLaMA models and multiple evaluation datasets, with convincing results.

Efficient: This method requires only a small amount of calibration data and does not require gradient backpropagation, making it suitable for practical deployment.

Highly Reproducible: Detailed algorithm pseudocode and experimental setup are provided.

### Weaknesses
Strong theoretical assumptions: The linear relationship between GSR and quantization error relies on the Laplace distribution assumption. While experimentally validated, the theoretical generalizability requires further verification.

Limited model generalization: The experiments only cover the LLaMA family and do not test other architectures (such as encoder-decoder and vision-language architectures).

Inadequate runtime evaluation: While claiming "negligible runtime overhead," no actual latency or throughput data is provided.

Insufficient comparison with state-of-the-art methods: While comparing various PTQ methods, no comparison is made with recent, more advanced low-bit QAT or mixed-precision methods.

The analysis of the causes of "massive activations" is shallow: While noting their existence, the authors do not delve into their origins or whether they can be avoided through architectural design.

### Questions
Supplement experimental verification of other model families (such as BERT, T5, ViT, etc.);

Provide comparative data on actual inference speed and memory usage;

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper identify a metric named GSR. Empirical/theoretical analyses show that GSR is closely related to quantization error. Based on these findings, the authors further modify DFRot on the activation side, and update the low rank decomposition quantization method on the weight side. The experiment evaluations assess the proposed method to some extent.

### Strengths
1. The motivation of the paper is clear. The authors evaluate the heuristic metric GSR with both theoretical and empirical analyses, and then use the metric to improve existing quantization method.
2. The proposed method is backed up by the experiments.
3. The paper is well-structured overall.

### Weaknesses
1. The notation $R$ is confusing. In Sec. 4, it appears to represent two different meanings.
2. In Table 1, the authors states that "SpinQuant & OSTQuant use quantization-aware training to optimize $R_1$". First, $R_1$ is not defined elsewhere in the paper. Second, the paper should clarify what it means by quantization-aware training. Both SpinQuant & OSTQuant define themselves as Post-training Quantization (PTQ) method rather than QAT. **Given this, please justify the advantages the proposed method and explain the comparison when its performance is worse than SpinQuant/OSTQuant**
3. Recent baseline methods/models such as FlatQuant/Qwen are not included.
4. More benchmarks like MMLU should be included in addition to PPL and common-sense QA tasks.
5. The description of the proposed method is difficult to follow. It is not clearly presented. I could only understand the algorithm by the pseudo code in the Appendix. Also, the multi-paragraph abstract reads oddly and feels informal.
6. A runtime comparison with other baseline methods is required to demonstrate efficiency.

Minors:
1. In Line 193, 196 & 205, repeated "equation".
2. The text in Fig.1 is too small to read.

### Questions
See Weaknesses above. Please address Weakness 2 carefully and present the advantages of the proposed method over SpinQuant / OSTQuant / FlatQuant, especially where the proposed method underperforms.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces FRTQ, a post-training quantization method for W4A4 LLMs that explicitly reduces the grid-to-std ratio (GSR, Δ/σ) for both activations and weights. On activations, it performs Uniform Preconditioning (UP) to equalize per-row magnitudes before a DFRot refinement, contracting GSR without incurring quantization error. On weights, it adds a tuned SVD (tSVD) branch and row-wise ℓ∞ updates to depress per-row maxima of the INT4 residual, tightening the effective grid and thus lowering rounding error. Rotations are fused into weights (one lightweight rotation remains at FFN down-proj). Across LLaMA 7B–70B, FRTQ improves perplexity and zero-shot over QuaRot/DFRot and sometimes approaches QAT methods, while using tiny calibration and no backprop.

### Strengths
1. Simple and practical pipeline (UP+DFRot + tSVD), fully PTQ with minimal calibration; rotations fused, negligible runtime overhead.
2. Ablations & statistics (GSR/QErr tables) convincingly show how each component contributes and why it helps.

### Weaknesses
1. The font in Figure 1 is too small to read comfortably. Please increase the label and tick font sizes and/or provide a higher-resolution version in the main paper or the appendix.

2. Figure 1 aggregates pre-RMSNorm activations across layers but, on the weight side, uses only layer-0 query 𝑊𝑄. To support the “near-Laplace tails” and the error–GSR trend more broadly, please: (i) report CCDFs for several representative weight matrices beyond layer-0 𝑊𝑄; (ii) show activation CCDFs at the actual quantization insertion points (e.g., inputs to the main linear projections), not only pre-RMSNorm; and (iii) include the error-vs-𝑟 plots for a few deeper layers to verify the universality of the slope across the stack.

3. The paper would benefit from a side-by-side comparison with DuQuant [1] under matched settings—same checkpoints, KV precision, group sizes, calibration set size, evaluation harness/version, and decoding setup. Because FRTQ emphasizes tiny calibration (e.g., 1×2048 tokens for rotations, 128×2048 for weight calibration) and fused rotations with near-zero runtime overhead, an apples-to-apples table would clarify the accuracy/latency/compute trade-offs relative to DuQuant. If DuQuant requires materially different calibration or runtime transforms, please discuss how these differences affect the headline numbers.

[1] Lin et al. DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs. NeurIPS 2024.

### Questions
Please see the Weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Flattened Rotation tSVD Quantization (FRTQ), a post-training quantization (PTQ) framework designed for ultra-low-bit settings (e.g., W4A4). The key idea is to minimize a novel metric called the Grid-to-Standard-Deviation Ratio (GSR), defined as the quantization grid size Δ divided by the standard deviation σ. The authors theoretically and empirically show that quantization error scales linearly with GSR.

### Strengths
- The paper’s motivation is clear — it aims to provide a quantifiable measure (GSR) for characterizing quantization difficulty.
- The authors combine theoretical and empirical analysis to validate the GSR metric and apply it to guide improvements in quantization.
- The proposed approach is experimentally evaluated on multiple models, showing partial evidence of its effectiveness.

### Weaknesses
- The authors need to compare with related recent baselines, such as FlatQuant.
- The paper states negligible overhead, yet provides no runtime/memory comparisons. Numbers are needed to substantiate practicality.
- The study focuses on perplexity and a set of zero-shot commonsense tasks; broader benchmarks (e.g., MMLU/BBH) would better probe reasoning/generalization and could reveal trade-offs.
- What's the definition of $R_1$ in Table 1 caption?

### Questions
See weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2
