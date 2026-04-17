# CompSRT: Quantization and Pruning for Image Super Resolution Transformers

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Model compression has emerged as a way to reduce the cost of using image super resolution models by decreasing storage size and inference time. However, the gap between the best compressed models and the full precision model still remains large and a need for deeper understanding of compression theory on more performant models remains. Prior research on quantization of LLMs has shown that Hadamard transformations lead to weights and activations with reduced outlier, which leads to improved performance. We argue that while the Hadamard transform does reduce the effect of outliers, an empirical analysis on how the transform functions remains needed. By studying the distributions of weights and activations of SwinIR-light, we show with statistical analysis that lower errors is caused by the Hadamard transforms ability to reduce the ranges, and increase the proportion of values around $0$. Based on these findings, we introduce CompSRT, a more performant way to compress the image super resolution transformer network SwinIR-light. We perform Hadamard-based quantization, and we also perform scalar decomposition to introduce two additional trainable parameters. Our quantization performance statistically significantly surpasses the SOTA in metrics with gains as large as 1.53 dB, and visibly improves visual quality by reducing blurriness at all bitwidths. At 3-4 bits, to show our method is compatible with pruning for increased compression, we also prune 40% of weights and show that we can achieve 6.67-15% reduction in bits per parameter with comparable performance to SOTA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes two techniques, i.e., Hadamard transform before quantization and decomposition of quantization parameters. Compared to the state-of-the-art method, i.e., CondiQuant, the proposed method outperforms by a large margin, especially for low bit quantization cases.

### Strengths
This paper firstly shows that Hadamard transform is effective for Swin-IR in the SR domain.
The proposed decomposition is new and can reflect the actual quantization scale and bias as well as learnability depending on the distributions.
The proposed method outperforms SOTA by a large margin.

### Weaknesses
1. Hadamard transform is being widely used in may video coding and deep neural network compression (e.g., LLM, diffusion models). And the major performance gain of the proposed method comes from the Hadamard transform. Although This paper argues that Hadamard transform can concentrate energy to low frequency regions as make high frequency region sparse, this property of transforms (e.g., FFT, DCT, DST and DWHT) is well known and proved in many literature for natural images and W/A of deep neural networks [1-3]. This limits the novelty of this paper.

[1] Xinyu Wang, et al. 2025. HadaNorm: Normalization after Hadamard Incoherence for Activation Quantization. In Proceedings of the 42nd International Conference on Machine Learning (ICML 2025).

[2] Song Han, et al. 2024. QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks. In Proceedings of the 41st International Conference on Machine Learning (ICML 2024).

[3] Tim Dettmers, et al. 2024. QuaRot: Outlier-Free 4-bit Inference in Rotated LLMs. In Advances in Neural Information Processing Systems (NeurIPS 2024).


2. the proposed scalar decomposition can be viewed as a variant of LSQ which learns quantization parameters. To verify the effectiveness of the proposed hybrid (e.g., a uniform quant-based fixed scale with a learnable variable) method should be compared with LSQ. 

3. Nowdays, there are a lot of SR models based on diffusion models, Mamba [4, 5] etc.. This paper improves the performance of quantized Swin-IR models only. This makes the impact of this paper limited.

[3] Hang Guo, Jinmin Li, Tao Dai, Zhihao Ouyang, Xudong Ren, and Shu-Tao Xia. Mambair: A simple baseline for image restoration with state-space model. In European conference on computer vision, pp. 222–241. Springer, 2024a.  Xiangyu Guo, Kai Zhang, Jingyun 

[4] Liang, Yulun Wang, and Radu Timofte. Mambairv2: Hybrid mamba-transformer architecture for efficient image restoration. arXiv preprint arXiv:2501.01234, 2025.

### Questions
Please read the Weakness section and answer the questions.

### Soundness
3

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
The paper targets compressing SwinIR-light for image super-resolution via a Hadamard-guided PTQ pipeline, a lightweight reparameterization of the quantization scale and zero-point, and 40% weight pruning at 3–4 bits. The key empirical claim is that, for SwinIR-light, Hadamard transforms do not flatten distributions but do shrink ranges and increase near-zero mass, which reduces quantization error. Extensive benchmarks show consistent PSNR/SSIM gains over previous methods. With pruning, bits-per-parameter drop by 6.67–15% at comparable quality.

### Strengths
1. The paper provides a fairly comprehensive validation within SwinIR-light of the mechanism by which Hadamard transforms are effective.
2. By decoupling the quantization step size and zero-point into two learnable parameters, the method intuitively increases representational and optimization flexibility while remaining simple to implement and directly pluggable into existing PTQ pipelines.
3. An integrated practice combining pruning and quantization is presented.

### Weaknesses
1. The claim that “This operation has been said to flatten the matrices by distributing the magnitude of outliers and that is how the errors get reduced, but the exact mechanism has not been explored nor has the flatness or normality of distributions been tested.” is not fully accurate. QuaRot [A] demonstrates end-to-end 4-bit LLM inference using Hadamard-based rotations to mitigate outliers, with distributional visualizations supporting the “flattening” effect. SpinQuant [B] shows the mechanism is the rotation itself—random rotations vary widely, while learned rotations (via Cayley optimization) yield larger gains, directly explaining why errors decrease. HadaNorm [C] further analyzes when Hadamard mixing is weakened by per-channel mean/scale, and proposes centering+rescaling before Hadamard with quantitative improvements—a more systematic treatment of activation distributions. In light of these results, the paper’s statement is incomplete, and the corresponding contribution is consequently weakened.
References:
[A] Ashkboos et al., QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.
[B] Liu et al., SpinQuant: LLM Quantization with Learned Rotations.
[C] Federici et al., HadaNorm: Diffusion Transformer Quantization through Mean-Centered Transformations.
2. The paper lacks evaluations on real-world degradation benchmarks to verify robustness under non-ideal distribution shifts.
3. Experiments are currently limited to SwinIR-light. The paper lacks experiments on additional Transformer-based backbones to demonstrate architecture-agnostic effectiveness.
4. The paper lacks inference throughput results across varying input resolutions and multiple upscaling factors. Please report latency or FPS under representative settings.

### Questions
1. Please further explain why, under the 3-bit setting, modest gains appear in the 10–30% range. What drives this phenomenon? Does it persist across repeated runs with different random seeds?
2. The paper measures the effect of the Hadamard transform via the zero-neighborhood ratio with the threshold ε fixed at 0.05. How sensitive are your results to ε?
3. Why do the results in Table 3 show that the inference time and number of parameters for the pruned model do not decrease in roughly proportional terms?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes CompSRT, a post‑training compression pipeline for the SwinIR‑light super‑resolution model that combines (i) Hadamard‑guided quantization of weights and activations and (ii) scalar decomposition of the quantization scale and zero offset.

A central empirical claim is that Hadamard transforms do not “flatten” weight/activation distributions in SwinIR‑light; instead, they reduce value ranges and increase mass near zero, which the authors argue lowers quantization error.

Quantization is applied to attention/MLP components, and the method is optionally combined with 40% unstructured weight pruning. CompSRT outperforms prior PTQ baselines on five standard benchmarks. The authors also report a small per‑image runtime overhead from the extra Hadamard operations and a reduction in “bits per parameter” when pruning.

### Strengths
1) The paper provides consistent PSNR/SSIM improvements across five datasets and three scales at 2–4 bits. 

2) The experimental evaluation is very detailed and the results are supported by various statistical tests. The authors demonstrated that the Hadamard transformations reduce quantization errors in matrices by reducing the ranges of the values, and concentrating values around 0.
The paper goes beyond intuition, using paired tests and effect sizes to argue the Hadamard's benefit arises from range shrinkage and increased near‑zero mass.

3) The text is well writted; where the transforms are inserted and how scalar decomposition is optimized are clearly described.

4) The technique is lightweight conceptually, slots into PTQ, and composes with pruning (40%) to reduce effective bits per parameter.

5) The authors provided good ablation study. In particular, the study indicates that scalar decomposition contributes strongly; Hadamard on both weights and activations further helps, aligning with the distributional evidence. 

6) The paper includes a reproducibility statement and an anonymous code link.

### Weaknesses
1) Because CondiQuant code is unavailable, qualitative results use 2DQuant; this limits the strength of the visual SOTA claim. A controlled, code‑level visual comparison with CondiQuant (if possible) or an alternative strong visual baseline would help.

2) The paper reports a per‑image overhead due to extra FP Hadamard operations. There is no end‑to‑end profiling (e.g., batch throughput, latency on edge hardware, kernel fusion feasibility).

3) The 1‑bit dense mask plus storing the pruned tensor is simple, but practical implementations often require not standard sparse formats.

4) Results focus on SwinIR‑light; it is unclear whether the same gains hold for larger SwinIR variants or different SR architectures (EDSR/RDN) and for different activation/weight quantization granularities. The method likely generalizes, but evidence is limited to one backbone

5) The signed‑rank tests for superiority vs. CondiQuant are conducted over only five datasets per configuration. 

6) Robustness to \epsilon (set to 0.05) and to other orthonormal transforms (e.g., DCT, random orthogonal) is not explored, so it remains uncertain whether Hadamard is uniquely effective.

### Questions
1) Figures 3–4 show both Hadamard and inverse‑Hadamard nodes. Are these executed during inference for every forward pass? If so, what is the measured layer‑wise cost on typical hardware

2) How does your calibration set size and finetuning budget (up to 4k iters with Adam) compare to those used by 2DQuant/CondiQuant? Could differences partly explain the observed gains? Please clarify the exact data used during finetuning and whether early stopping was based on the validation set. 

3) How sensitive are the range‑reduction and near‑zero‑mass effects (and resulting PSNR) to the \epsilon threshold and to the choice of orthonormal transform (Hadamard vs. DCT vs. random orthogonal)? A small ablation would clarify whether the effect is Hadamard‑specific.

4) What granularity is used (per‑tensor vs. per‑channel/group) for activations and weights? Would scalar decomposition still help under finer granularity?

5) Have you tried the method on SwinIR‑base or on CNN SR backbones (EDSR/RDN)? Even a subset of experiments would bolster generality claims.

6) If CondiQuant visuals are unavailable, could you at least add more visual crops against a strong available baseline (e.g., 2DQuant with equal finetuning budget) on the same images and provide details of the experiment to ensure reproducibility?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper explores the use of Hadamard transformations for image super-resolution models, inspired by their demonstrated success in large language model (LLM) quantization.

### Strengths
* The paper targets an interesting direction.

### Weaknesses
The paper lacks a sufficiently comprehensive literature review and contains several inaccurate claims:
* At Line 73, the authors state that FlatQuant is a Hadamard-based method. This is incorrect — FlatQuant uses a learnable matrix instead of a fixed Hadamard transformation.
* The success of Hadamard or rotation-based transformations has already been extensively discussed in prior works [a, b, c]. These studies attribute the benefit to incoherent processing, as analyzed in QuIP[a]. However, the authors incorrectly claim that “the mechanism for Hadamard’s lowering of errors remains unexplored.”
* The concepts of learnable scale and offset have been explored in earlier works [d, e], so they cannot be considered novel contributions.
* The pruning component is overly simplistic, suggesting pruning parameters with values close to zero without discussing potential performance impacts or providing justification for the design choice. This section lacks insight and depth.

[a] Quip: 2-bit quantization of large language models with guarantees. NeurIPS 2023

[b] Quarot: Outlier-free 4-bit inference in rotated llms. NeurIPS 2024

[c] Spinquant: Llm quantization with learned rotations. ICLR 2025

[d] Learned step size quantization. ICLR 2020

[e] Lsq+: Improving low-bit quantization through learnable offsets and better initialization. CVPR 2020 workshop

### Questions
I strongly encourage the authors to conduct a more comprehensive and accurate literature review to better position this work within existing research.

### Soundness
1

### Presentation
1

### Contribution
1
