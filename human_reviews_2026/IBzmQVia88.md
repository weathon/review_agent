# Rethinking Expressivity and Degradation-Awareness in Attention for All-in-One Blind Image Restoration

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
All-in-one image restoration (IR) aims to recover high-quality images from diverse degradations, which in real-world settings are often mixed and unknown. Unlike single-task IR, this problem requires a model to approximate a family of heterogeneous inverse functions, making it fundamentally more challenging and practically important. Although recent focus has shifted toward large multimodal models, their robustness still depends on faithful low-level inputs, and the principles that govern effective restoration remain underexplored. We revisit attention mechanisms through the lens of all-in-one IR and identify two overlooked bottlenecks in widely adopted Restormer-style backbones: (i) the value path remains purely linear, restricting outputs to the span of inputs and weakening expressivity, and (ii) the absence of an explicit global slot prevents attention from encoding degradation context. To address these issues, we propose two minimal, backbone-agnostic primitives: a nonlinear value transform that upgrades attention from a selector to a selector–transformer, and a global spatial token that provides an explicit degradation-aware slot. Together, these additions improve restoration across synthetic, mixed, underwater, and medical benchmarks, with negligible overhead and consistent performance gains. Analyses with foundation model embeddings, spectral statistics, and separability measures further clarify their roles, positioning our study as a step toward rethinking attention primitives for robust all-in-one IR.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes ExDA, a lightweight and backbone-agnostic framework for all-in-one blind image restoration (IR). The authors identify two bottlenecks in Restormer-style architectures: (1) linear value paths that limit expressivity and (2) the absence of global degradation-aware context. To address these, they introduce two modules — a nonlinear value transform (NVT) and a global spatial token (GST) — which enhance expressivity and degradation-awareness without significant computational overhead. Extensive experiments on synthetic, mixed, real-world, and medical benchmarks demonstrate consistent improvements over recent methods such as PromptIR, AdaIR, and MoCE-IR.

### Strengths
* The design is simple, modular, and generalizable — the proposed primitives can be integrated into various backbones.
* The paper clearly articulates two structural limitations in popular IR architectures (linear value paths and missing global slots) and proposes minimal, interpretable solutions.

### Weaknesses
* The contribution, while practical, feels **architecturally incremental** rather than conceptually transformative. The nonlinear value transform is essentially a lightweight convolutional enhancement inserted into the attention value path, similar in spirit to prior nonlinear attention variants. Likewise, the global spatial token extends the CLS-token concept rather than introducing a fundamentally new paradigm for degradation modeling. The theoretical discussion on “expressivity expansion” and “degradation-awareness” is more intuitive than rigorously grounded.

* Although the comparisons are broad, the evaluated baselines are largely regression-based approaches, focusing on architectures. Incorporating comparisons with distribution-oriented models, e.g., DA-RCOT [1] or Defusion [2] would strengthen the paper's generality. 

* The related work can be enhanced with discussions of recent expressivity-enhanced attention variants and degradation-aware architectures, which are directly relevant to the claimed contributions.

[1] Tang et al., Degradation-aware residual-conditioned optimal transport for unified image restoration, TPAMI 2025.

[2] Luo et al., Visual-Instructed Degradation Diffusion for All-in-One Image Restoration. CVPR 2025.

### Questions
See weaknesses.

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
5

### Summary
The paper revisits Restormer-style channel-wise attention for All-in-One IR and pinpoints two bottlenecks: a purely linear value path that restricts outputs to the input span, and the absence of an explicit global slot to encode degradation context. It proposes two minimal, backbone-agnostic primitives: a nonlinear value transform applied before aggregation, and a Global Spatial Token that injects an explicit, content-adaptive global slot into attention. Through extensive experiments, ExDA reports consistent gains with negligible overhead.

### Strengths
1. The paper precisely connects linear-V and missing global slots to AiOIR failure modes, then remedies them with pre-aggregation nonlinear-V and GST—small, principled changes rather than wholesale redesigns. Both primitives are lightweight and easily inserted into Restormer-like stacks without destabilizing training.
2. Results span 3D/5D, compound (e.g., CDD11), adverse weather, underwater, and medical data, showing consistent improvements. 
3. Multiple FLOPs/parameters of ExDA size (e.g., the base, small and tiny models) help us evaluate deployment scenarios.

### Weaknesses
1. Additional controls comparing post-aggregation nonlinearity and stronger FFN/MLP capacity would better isolate the unique effect of pre-aggregation nonlinear-V.
2. t-SNE/UMAP plots with NMI/ARI, and GST attention maps for different degradations/strengths, would make “degradation-aware” behavior more concrete.
3. Provide resolution–quality/latency and model-size–quality/latency curves to guide engineering trade-offs across ExDA-Tiny/Small/Base.

### Questions
1. How do post-aggregation nonlinearity or beefed-up FFN/MLP compare, keeping compute similar? This would confirm the specific benefit of pre-aggregation nonlinear-V. 
2. Please add t-SNE/UMAP with NMI/ARI and visualize GST-driven attention for noise/blur/haze/rain and mixtures; analyze stride s beyond the current setting.
3. Provide resolution/model-size trade-off curves for ExDA-T/S/Base and recommended configs.
4. If a frequency-domain loss is used, how sensitive are results to its weight and which component (nonlinear-V or GST) benefits most? 
5. Have you tested on non-Restormer variants (e.g., U-former-style) or combined ExDA with frequency/prompt modules to show compatibility?
6. More recent works can be added for comparison such as Perceive-IR (TIP’25) and DFPIR (CVPR’25).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper revisits attention mechanisms for all-in-one blind image restoration and identifies two overlooked bottlenecks in Restormer-style architectures: limited expressivity from linear value aggregation and the absence of explicit global context. The authors propose two minimal, backbone-agnostic modules, a nonlinear value transform and a global spatial token, to address these issues. The method achieves consistent improvements across diverse benchmarks with negligible overhead. Overall, the work is well-written, experimentally solid, and offers a clear, though modest, conceptual contribution to the design of efficient all-in-one restoration models.

### Strengths
1. The paper targets a relevant and timely problem in all-in-one image restoration and presents clear motivation.

2. The proposed modifications are simple but meaningful, addressing expressivity and global context in a principled way.

3. Experiments cover diverse benchmarks and consistently show improvements, supported by ablation and diagnostic analysis.

4. The method is lightweight and practical, showing good trade-offs between performance and complexity.

5. The paper is clearly written and easy to follow.

### Weaknesses
1. The explanation about how the nonlinear value path improves expressivity is mostly intuitive, without quantitative or theoretical evidence.

2. All experiments are based on a Restormer-type backbone, so it is unclear if the same gains would appear on other architectures.

3. The claim of `negligible overhead` is not backed up by runtime or FLOPs comparisons.

4. The paper does not analyze what the proposed global tokens actually learn, which would help readers understand their role.

5. Most datasets are synthetic or composited; there is little discussion about generalization to real-world, uncontrolled degradations.

### Questions
1. Could you add a short explanation or table (in text form) that summarizes how the nonlinear value transform changes feature distribution or rank, even qualitatively?

2. Since the paper claims to be backbone-agnostic, it would help to briefly discuss how the same ideas might work on other architectures such as NAFNet or SwinIR (I believe at least 1 degradation type analysis should be doable).

3. Please include a small table with FLOPs, inference time, or memory to support the efficiency claim

4. For the global spatial tokens, if figures cannot be shown, a short textual description of how different tokens behave under different degradations would be informative.

5. It would also help to add a short discussion on how the proposed modules might generalize to unseen or mixed degradations in real-world conditions.

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
4

### Summary
This paper targets all-in-one blind image restoration (IR) and argues that the dominant Restormer-style channel attention is hurt by two under-studied limitations: (i) the value path is strictly linear, so the attention head can only select but never transform features, and (ii) no global slot is available to encode degradation context. The authors plug two light-weight, backbone-agnostic primitives into any Restormer-like block.

### Strengths
The paper explicitly diagnoses the “linear-value” bottleneck of channel-wise attention in IR and connects the absence of a CLS-like token to the difficulty of inferring unknown degradations. The proposed fixes are minimal and can be dropped into existing models without architectural surgery.

### Weaknesses
1. The theoretical justification is limited. While the authors quote universal-approximation arguments, no formal proof is given that the residual value transform enlarges the hypothesis space of the entire Restormer block, nor that GST tokens are minimal sufficient statistics for degradation type. 
2. The experiments are not sufficient. For example, all experiments are conducted on 128 × 128 or 256 × 256 crops. The paper does not report run-time or memory on >4 K images where Restormer is usually applied. Additionally, some latest all-in-one methods are not compared in the experiments. Please refer to the survey paper (A survey on all-in-one image restoration: Taxonomy, evaluation and future trends. TPAMI, 2025) for more competing methods.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
