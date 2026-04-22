# One-Step Residual Shifting Diffusion for Image Super-Resolution via Distillation

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Diffusion models for super-resolution (SR) produce high-quality visual results but require expensive computational costs. Despite the development of several methods to accelerate diffusion-based SR models, some (e.g., SinSR) fail to produce realistic perceptual details, while others (e.g., OSEDiff) may hallucinate non-existent structures. To overcome these issues, we present **RSD**, a new distillation method for ResShift. Our method is based on training the student network to produce images such that a new fake ResShift model trained on them will coincide with the teacher model. RSD achieves single-step restoration and outperforms the teacher by a noticeable margin in various perceptual metrics (LPIPS, CLIPIQA, MUSIQ, DISTS, NIQE, MANIQA). We show that our distillation method can surpass the other distillation-based method for ResShift - SinSR - making it on par with state-of-the-art diffusion-based SR distillation methods  with low computational costs in terms of perceptual quality. Compared to SR methods based on pre-trained text-to-image models, RSD produces competitive perceptual quality and requires fewer parameters, GPU memory, and training cost. We provide experimental results on various real-world and synthetic datasets, including RealSR, RealSet65, DRealSR, ImageNet, DIV2K, RealLR200 and RealLQ250.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Residual Shifting Distillation (RSD), a distillation framework for accelerating the ResShift diffusion-based super-resolution model to a single-step inference regime. RSD introduces a trainable “fake” ResShift model to align the joint trajectory distribution of the student-generated data with that of the teacher, in contrast to prior methods (e.g., VSD) that align only marginal distributions per timestep. The method is combined with LPIPS and GAN losses for improved perceptual quality. Extensive experiments on real-world (RealSR, DRealSR) and synthetic (ImageNet-Test, DIV2K) benchmarks demonstrate that RSD outperforms SinSR in perceptual metrics while maintaining competitive fidelity, and significantly reduces computational cost compared to T2I-based methods like OSEDiff and SUPIR.

### Strengths
1. Clear and principled formulation: The paper provides a well-motivated objective (Eq. 7–9) based on KL divergence over the full reverse trajectory, with a tractable surrogate derived via Proposition 3.1. The distinction between joint vs. marginal distribution alignment is clearly illustrated (Fig. 4, Appendix A).
2. Comprehensive experiments: The evaluation covers multiple datasets, degradation models, and SOTA baselines across GAN-based, diffusion-based, and T2I-based SR methods. Both quantitative (Tables 1–3, Appendix D–E) and qualitative results (Figs. 3, 5, 9–12) are thorough and convincing.
3. Strong practical impact: RSD achieves 1-step inference, 174M parameters, and <600MB GPU memory, making it highly deployable. Training is ~3× faster than SinSR due to simulation-free distillation.
4. Honest discussion of limitations: The authors acknowledge dependence on the ResShift teacher, failure cases on complex textures, and the gap with T2I models in hallucination-rich scenarios (Appendix H).
5. Excellent reproducibility: Pseudocode (Appendix B), training details (Appendix C), dataset licenses (Table 8), and codebase description are provided.

### Weaknesses
1. Incremental theoretical novelty: As acknowledged in Appendix A, RSD is closely related to IBMD (Gushchin et al., 2025) and can be viewed as its discrete, task-specific adaptation to ResShift. The core idea—using an auxiliary model to match joint distributions—has appeared in consistency distillation, diffusion bridges, and related works.
2. Missing comparison to very recent 1-step SR methods: For example, CCSR (Sun et al., 2024) and TSD-SR (Dong et al., 2024) are mentioned but not included in main tables.
3. Limited generalizability: The method is tightly coupled to the ResShift architecture (residual shifting, latent-space diffusion). It is unclear whether RSD can be applied to other diffusion frameworks (e.g., I2SB, LDM) without significant redesign.
4. Training-resolution mismatch in comparisons: RSD is trained on 256×256 crops, while OSEDiff uses 512×512. Although Appendix E shows RSD trained at 512×512, the main results are not fully comparable. This slightly favors T2I methods in perceptual metrics. 

For clarity, the following should be addressed.
The paper is generally well organized, but the abstract contains a 76-word sentence that should be split. Figure fonts are smaller than 8 pt and become illegible when printed. A symbol table summarizing latent-space vs pixel-space notation would help readers. Additionally, the phrase “diffusion-based SR models” appears four times within three lines; replacing subsequent instances with “such approaches” or “these methods” would improve fluency.

For the experiments, the following should be addressed.
1. The current teacher ResShift was pretrained on 256² crops, limiting its modelling of 512² high-frequency details and causing RSD to lag behind OSEDiff/SUPIR on high-resolution data. Higher-resolution teacher distillation is deferred to future work.
2. To enable a fairer assessment of the performance gap with T2I-based methods (e.g., OSEDiff), it would be better including in the main experiments an additional set of results where RSD is trained and fully fine-tuned at 512×512 resolution.
3. To strengthen the empirical evaluation and better position RSD within the landscape of state-of-the-art one-step super-resolution methods, it would be more convincing that the authors include a direct quantitative and qualitative comparison with CCSR (Sun et al., 2024) and TSD-SR (Dong et al., 2024) in the main results.

### Questions
1. Could RSD be applied to a non-ResShift diffusion model (e.g., standard DDPM or LDM)? What architectural or training changes would be needed?
2. In Appendix E, RSD trained on 512×512 still lags behind OSEDiff in CLIPIQA/MUSIQ. Do the authors believe this gap is due to the teacher (ResShift) or the distillation objective?

### Soundness
3

### Presentation
3

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
This paper presents RSD (Residual Shifting Distillation), a one-step distillation method for diffusion-based image super-resolution. The approach trains a student generator to produce images such that a "fake" ResShift model trained on these generated images matches the original teacher model. RSD achieves competitive results compared to the 15-step teacher ResShift and claims to outperform existing distillation methods. Experiments are conducted on RealSR, RealSet65, DRealSR, ImageNet, and DIV2K datasets.

### Strengths
1.	The paper combines a distillation objective that aligns joint distributions rather than marginal distributions at each timestep (as in VSD) with the ResShift architecture.
2.	Extensive quantitative results are provided across multiple benchmarks.
3.	Comprehensive related work discussion and comparison with VSD, SiD, FGM, and IBMD frameworks.

### Weaknesses
1.	Missing baseline method comparison. Appendix A.3 acknowledges that RSD is essentially a "discrete variant of IBMD," yet IBMD is never empirically compared. Given this close relationship, quantitative comparison is essential to validate the claimed improvements from task-specific adaptations. Equation 21 explicitly shows RSD is IBMD applied to ResShift.
2.	Limited Novelty. The main contribution is essentially combining ResShift with VSD/IBMD frameworks. Moreover, the authors also acknowledge the relationship to IBMD (Appendix A.3, Lines 1242-1295).
3.	Misleading Experimental Claims. The authors claim to "outperform the teacher by a large margin" and . However, in Table 1 (RealSR), RSD loses on fidelity metric PSNR. 
4.	The paper ignores recent state-of-the-art methods including:
[1] PiSASR (CVPR 2025)
[2] TSDSR (Dong et al., 2024)
[3] InvSR (CVPR 2025)
5.	These omissions weaken the comprehensiveness of the experimental comparison.
6.	FID is not reported, despite being widely used in diffusion-based SR papers such as StableSR and OSEDiff. This limits comparability with existing literature.
7.	Visual quality does not significantly outperform baseline methods like OSEDiff. For example, differences in Figure 10 (Lines 2295 and 2306) are minimal, which undermines claims of substantial perceptual improvements.

### Questions
See the weakness.

### Soundness
3

### Presentation
4

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
This paper proposes Residual Shifting Distillation (RSD), a novel one-step distillation framework for image super-resolution (SR) based on the ResShift diffusion model. Unlike previous knowledge distillation approaches such as SinSR, which require running the full teacher model through all diffusion steps, RSD introduces a “simulation-free” design by using an auxiliary fake ResShift model to estimate the teacher–student discrepancy through a theoretically derived loss. The method also combines perceptual and adversarial supervision (LPIPS + GAN losses) to further enhance visual realism. The appendices provide mathematical derivations, ablation studies, algorithmic details, proofs, and extensive visual comparisons.

### Strengths
1. The paper presents a clear derivation of the RSD loss from a probabilistic perspective, bridging diffusion-based knowledge distillation and joint distribution alignment. The inclusion of Proposition 3.1 and the equivalence to KL divergence (Eq. 9) adds strong theoretical grounding.

2. The proposed “simulation-free” training significantly reduces the computational overhead compared to SinSR, making one-step diffusion models more accessible for real-world scenes.

3. The paper is well-written and well-organized.

### Weaknesses
1. The paper does not compare with recent state-of-the-art methods such as AdcSR [1], PiSA-SR[2], CTMSR[3], and TSDSR [4]. This omission makes it difficult to fully evaluate the competitiveness of the proposed method in the context of the latest advances in this field.

2. Table 6 shows a significant decrease in the CLIPIQA score after introducing the GAN loss, but the authors have not discussed this in depth.

3. While the paper mentions the choice of $K=5$ for updating the fake model, it does not delve deeply into how hyperparameter sensitivity (e.g., $\lambda_1$, $\lambda_2$, $K$) affects training stability and final performance. More analysis of this sensitivity, including robustness to different configurations is needed.

4. The evaluation of the paper on real-world datasets is limited. Evaluating on larger and more diverse real-world datasets, such as RealLR200 in SeeSR or RealLQ250 in DreamClear, would strengthen the robustness and credibility of the experimental results.

[1] Adversarial Diffusion Compression for Real-World Image Super-Resolution. CVPR 2025

[2] Pixel-level and Semantic-level Adjustable Super-resolution: A Dual-LoRA Approach. CVPR 2025

[3] Consistency Trajectory Matching for One-Step Generative Super-Resolution. ICCV 2025

[4] TSD-SR: One-Step Diffusion with Target Score Distillation for Real-World Image Super-Resolution. CVPR 2025

### Questions
See the weaknesses above.

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
This paper introduces "RSD," a novel distillation method for the ResShift super-resolution (SR) model, aiming to reduce high computational costs while improving quality over existing acceleration methods like SinSR (unrealistic details) and OSEDiff (hallucinated structures).

The method involves training a student network to generate images, which are then used to train a "fake ResShift model." The objective is to make this "fake" model's performance "coincide with" the original teacher model.

### Strengths
The indirect training approach (training a student to generate data to train a proxy model) is a novel contribution to knowledge distillation for generative models.

### Weaknesses
The paper’s ambitious claims are not adequately supported by evidence. The core contribution remains vague: the notions of a “new fake ResShift model” and making it “coincide with” the teacher are abstract, lacking mathematical formulation or clear motivation. The statement that a distilled student “outperforms the teacher by a large margin” is implausible without thorough justification—this raises concerns about whether the teacher was properly optimized or if “outperform” is defined using non-standard metrics.

The abstract also includes unqualified claims (“surpass,” “on par”) without specifying the evaluation criteria (e.g., PSNR, LPIPS, FID). Given the method’s complexity, the authors should consider a simpler baseline—quantifying ResShift directly to obtain a lightweight variant for comparison. Without such analyses, the claimed advantages remain unconvincing.

### Questions
See the weekness secton. In summary, the abstract proposes an intriguing and timely solution to a significant problem. However, its credibility is undermined by a vague methodology and exceptionally strong claims that lack clear evidence. The assertion that a student model can dramatically outperform its teacher is extraordinary and demands rigorous proof and explanation.

### Soundness
2

### Presentation
2

### Contribution
2
