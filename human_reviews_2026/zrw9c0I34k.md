# FastFit: Accelerating Multi-Reference Virtual Try-On via Cacheable Diffusion Models

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Despite its great potential, virtual try-on technology is hindered from real-world application by two major challenges: the inability of current methods to support multi-reference outfit compositions (including garments and accessories), and their significant inefficiency caused by the redundant re-computation of reference features in each denoising step. To address these challenges, we propose FastFit, a high-speed multi-reference virtual try-on framework based on a novel cacheable diffusion architecture. By employing a Semi-Attention mechanism and substituting traditional timestep embeddings with class embeddings for reference items, our model fully decouples reference feature encoding from the denoising process with negligible parameter overhead. This allows reference features to be computed only once and losslessly reused across all steps, fundamentally breaking the efficiency bottleneck and achieving an average 3.5x speedup over comparable methods. Furthermore, to facilitate research on complex, multi-reference virtual try-on, we introduce DressCode-MR, a new large-scale dataset. It comprises 28,179 sets of high-quality, paired images covering five key categories (tops, bottoms, dresses, shoes, and bags), constructed through a pipeline of expert models and human feedback refinement. Extensive experiments on the VITON-HD, DressCode, and our DressCode-MR datasets show that FastFit surpasses state-of-the-art methods on key fidelity metrics while offering its significant advantage in inference efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents **FastFit**, a high-speed multi-reference virtual try-on framework that addresses current limitations in handling multiple outfit items and redundant feature recomputation. 

By introducing a cacheable diffusion architecture with a Semi-Attention mechanism and class embeddings for reference items, the method decouples feature encoding from denoising, achieving up to **3.5× faster inference** with minimal overhead. 

The authors also introduce **DressCode-MR**, a new dataset with over 28K curated image pairs, demonstrating that FastFit outperforms state-of-the-art methods in both fidelity and efficiency.

### Strengths
1. The paper provides extensive empirical results across multiple benchmarks (VITON-HD, DressCode, and DressCode-MR) to demonstrate the model’s performance.
2. The proposed FastFit framework effectively addresses real-world virtual try-on challenges by enabling multi-reference outfit compositions with improved efficiency.
3. The introduction of the large-scale DressCode-MR dataset enriches the field with diverse, high-quality samples for evaluating complex virtual try-on scenarios.

### Weaknesses
* The novelty appears limited, as the method largely reuses existing components. The proposed semi-attention with KV caching closely resembles techniques in **OminiControl2 (2025)** and **EasyControl (2025)**, while other modules (e.g., concatenation and cross-attention) are also standard design choices.
* Some recent and efficient baselines with strong detail preservation are missing from the comparison, such as **ITA-MDT (CVPR 2025)** and **IMAGDressing-v1 (arXiv:2407.12705)**.
* The paper’s clarity requires substantial improvement, as several explanations and figures are confusing or incomplete:

    (1) *Line 80*: The phrase “which avoids this redundancy” is unclear—what specific redundancy is being referred to, and how does reference-based modeling reduce it?

    (2) *Figure 4*: The relationship between sub-figures (a) and (b) is ambiguous. (a) shows seven inputs, whereas (b) only depicts four. The right part of (b) further introduces two small diagrams with inconsistent inputs (X with R1–R3 vs. X alone), making the overall data flow unclear.

    (3) Several notations (e.g., X′ in Figure 4) are unexplained. Important variables mentioned in the text should be explicitly labeled in the figure for coherence. Additionally, Figure 4(b) refers to a “Cacheable UNet Block,” but it is unclear how this integrates into the overview in (a) and how many such blocks are used. An ablation study analyzing this design would be valuable.

    (4) *Lines 268–269*: The explanation of timestep embeddings is inconsistent. The paper claims to explore timestep embeddings for enhanced control but then discusses only class embeddings. How exactly do class embeddings contribute to timestep control?

    (5) Baselines are listed without proper citations, making it difficult to trace their corresponding works. Please include references rather than assuming readers are familiar with each method by name.

    (6) The efficiency table lacks sufficient detail—specifically, the number of inference steps used for each baseline should be provided for fair comparison.

    (7) Please clarify Figure 4. Is that for training or inference? 

[1] ITA-MDT: Image-Timestep-Adaptive Masked Diffusion Transformer Framework for Image-Based Virtual Try-On, CVPR 2025

[2] IMAGDressing-v1: Customizable Virtual Dressing, arXiv:2407.12705

### Questions
Weakness part

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes FastFit, a diffusion-based framework designed to address two key challenges in virtual try-on (VTON) technology: the inability of existing methods to simultaneously compose multiple garments/accessories (e.g., tops, shoes, bags) and the computational redundancy caused by re-computing reference garment features at every denoising step. At the core of FastFit is a Cacheable UNet architecture, which decouples reference feature encoding from the iterative denoising process through two innovations: Reference Class Embedding and Semi-Attention Mechanism. This design enables a Reference KV Cache, where reference features are pre-computed once and reused across all denoising steps—delivering an average 3.5× speedup over comparable methods. To support multi-reference VTON research, the authors also introduce DressCode-MR, a large-scale dataset with 28,179 samples covering five categories (tops, bottoms, dresses, shoes, bags), curated via expert models (e.g., CatVTON) and human feedback.

### Strengths
- Most existing VTON methods (e.g., CatVTON, Chong et al. 2024; FitDiT, Jiang et al. 2024) only support single-garment try-on, requiring sequential inference for multi-item outfits (introducing error accumulation and latency). FastFit is among the first to enable simultaneous composition of garments and accessories (including shoes and bags), a critical capability for realistic outfit visualization. Qualitative results confirm it preserves fine details (e.g., text logos on T-shirts, sheer fabrics) across multiple references, outperforming baselines that produce blurred or misaligned components .
- Prior methods face a trade-off: ReferenceNet-based approaches (e.g., Xu et al. 2024) avoid redundancy but add massive parameter overhead (e.g., 1.7B params vs. FastFit’s 904M), while in-context learning methods (e.g., Huang et al. 2024a) re-compute features at each step (2–6× slower). FastFit’s decoupled design breaks this trade-off: ablations show disabling the Reference KV Cache increases latency by 1.66× (1.16s → 1.92s) with no quality loss, confirming its efficiency gain .
- Public VTON datasets (VITON-HD, DressCode) lack multi-reference pairs. DressCode-MR, built via expert model-based canonical image restoration and human feedback, provides the first large-scale benchmark for multi-reference VTON. This enables standardized evaluation of complex outfit composition, a previously under-explored direction

### Weaknesses
- The paper’s focus on multi-reference VTON overlaps with OmniTry (a previously published work on "try-on anything" that supports diverse garment/accessory categories). FastFit does not explicitly compare with OmniTry or demonstrate unique advantages in category coverage, generality, or composition flexibility—undermining its claim of advancing multi-reference VTON. This overlap reduces the work’s incremental contribution.
- The paper fails to cite key recent VTON research, such as VTON-HandFit (CVPR 2025), which addresses hand-garment interaction (a critical realism cue for accessories like bags or gloves). This omission limits the contextualization of FastFit’s contributions: VTON-HandFit’s techniques for handling fine-grained garment-object interactions could have informed FastFit’s multi-reference design, and ignoring it weakens the paper’s academic rigor.
- Training requires 8 NVIDIA H100 GPUs, and inference depends on high-end hardware (H100). No evaluation on consumer GPUs (e.g., RTX 4090) or edge devices limits real-world deployment—contrast with lightweight baselines like Leffa (1.8B params) that run on edge hardware , .
- FastFit cannot capture physical interactions (e.g., a tucked shirt, a jacket over a top) or fit variations (loose/tight clothing). Unlike OutfitAnyone (Sun et al. 2024) (which uses 3D modeling for layering), FastFit’s 2D diffusion approach produces flat, unrealistic composites—evident in qualitative results where accessories like bags lack depth relative to the body , .
- Evaluations focus on common categories (tops, dresses) but omit niche styles (e.g., sarees, kimonos) or challenging materials (e.g., leather, lace). The paper acknowledges this gap but provides no failure case analysis, limiting confidence in its applicability to diverse real-world scenarios .

### Questions
- How does FastFit differ from OmniTry in terms of category coverage, composition flexibility, and realism? What unique capabilities does FastFit offer that OmniTry lacks?
- Could integrating 3D garment priors (e.g., 3DMM) improve FastFit’s ability to model layering and physical interactions? Would this compromise its efficiency advantage?

### Soundness
2

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
4

### Summary
The authors identify two key limitations of current methods: their inability to support multi-reference outfit compositions and their significant computational inefficiency. To address these issues, they introduce a new large-scale dataset, DressCode-MR, and propose a Semi-Attention mechanism combined with cached features to accelerate the try-on process, reportedly saving approximately 40% inference time.

### Strengths
1. The writing is clear and the methodology is easy to understand.

2. The ablation studies thoroughly validate each component of the proposed approach—I appreciate the completeness of the ablation analysis.

3. In fact, the primary bottleneck preventing prior models from supporting multi-reference outfit composition has been the lack of suitable datasets. The most significant contribution of this paper is the introduction of the new multi-garment virtual try-on dataset, DressCode-MR.
However, the authors do not explicitly commit to open-sourcing DressCode-MR. If the authors clearly promise to fully release the DressCode-MR dataset (including both training and test sets), I would be happy to reconsider and potentially raise my rating.

### Weaknesses
1. The main concern is limited novelty. Multi-garment try-on has already been explored in prior works such as MMTryon [1] and AnyFit [2], which adopt similar strategies—e.g., spatially concatenating multiple garment references. In my view, using dedicated trainable branches for conditions versus sharing parameters with the denoising branch is primarily an engineering-level optimization rather than a fundamental academic distinction. One could interpret a single denoising branch handling both conditions and noise as an instance of shared trainable branches.

2. Although KV caching has not been explicitly framed as a novelty in prior try-on literature, it has already been implicitly applied in practice, which diminishes its perceived academic contribution. Moreover, the reported 40% speedup feels modest—almost trivial—for a paper whose title emphasizes acceleration. Notably, AnyFit [2] already employs KV caching or frozen condition features, yet the paper does not discuss or compare against AnyFit. I find this omission surprising and recommend that the authors add a dedicated discussion in the main body of the paper.

3. Additional evaluation metrics—such as DISTS [3]—would strengthen the experimental analysis.

4. It would also be beneficial to include visual results corresponding to the ablation studies.

Despite the above concerns, I believe the most valuable contribution of this work is the DressCode-MR dataset, which has the potential to significantly advance research in multi-reference virtual try-on and outweigh the methodological limitations.
Therefore, my final rating heavily depends on whether the authors commit to fully open-sourcing both the training and test sets of DressCode-MR.

References:

[1] MMTryon: Multi-Modal Multi-Reference Control for High-Quality Fashion Generation.

[2] AnyFit: Controllable Virtual Try-on for Any Combination of Attire Across Any Scenario.

[3] Image Quality Assessment: Unifying Structure and Texture Similarity.

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper primarily addresses two issues: first, it improves the efficiency of ReferenceNet-style garment swapping by caching features from reference images. Second, it introduces a dataset named DressCode-MR, which includes both garments and accessories, enabling simultaneous garment and accessory swapping.

### Strengths
It achieves more efficient injection of reference information by caching the features of the reference image, while also providing a dataset of garments and accessories with significant academic value. The paper is clearly written and easily understandable, with extensive experiments conducted.

### Weaknesses
The most significant issue with this paper is that its core algorithmic contribution lies in proposing a method to cache reference image information. However, such a technique is already widely used in both the research community and industrial applications. It appears more like a trick rather than a novel algorithm, which is my primary concern.

At the same time, compared to Anyfit [1], this paper does introduce an approach for simultaneous multi-garment replacement. The experimental results strike me as somewhat puzzling: why does the proposed method, with fewer parameters, outperform ReferenceNet? Caching reference image features should not enhance model performance—it only improves inference efficiency. The parameters in ReferenceNet enable it to better adapt to the features provided by the Denoising U-Net. In fact, in practical applications, we often precompute and store ReferenceNet's features. Doesn’t this approach closely resemble the method proposed in the paper? This similarity further undermines the core contribution of this work.

[1] Anyfit: Controllable virtual try-on for any combination of attire across any scenario

### Questions
Please refer to Weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
