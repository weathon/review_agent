# $\textit{MADFormer}$: Mixed Autoregressive and Diffusion Transformers for Continuous Image Generation

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Recent progress in multimodal generation has increasingly combined autoregressive (AR) and diffusion-based approaches, leveraging their complementary strengths: AR models capture long-range dependencies and produce fluent, context-aware outputs, while diffusion models operate in continuous latent spaces to refine high-fidelity visual details. However, existing hybrids often lack systematic guidance on how and why to allocate model capacity between these paradigms. In this work, we introduce $\textit{MADFormer}$, a Mixed Autoregressive and Diffusion Transformer that serves as a testbed for analyzing AR-diffusion trade-offs. $\textit{MADFormer}$ partitions image generation into spatial blocks, using AR layers for one-pass global conditioning across blocks and diffusion layers for iterative local refinement within each block. Through controlled experiments on FFHQ-1024 and ImageNet, we identify two key insights: (1) block-wise partitioning significantly improves performance on high-resolution images, and (2) vertically mixing AR and diffusion layers yields better quality-efficiency balances---improving FID by up to 75\% under constrained inference compute. Our findings offer practical design principles for future hybrid generative models. Code and models will be released upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript introduces MADFormer, a novel hybrid generative model architecture that combines autoregressive (AR) and diffusion-based modeling for continuous image generation. The model operates by mixing these two paradigms along two primary axes:

1. The Token/Spatial Axis: The image is partitioned into spatial blocks. Autoregressive modeling is used across blocks to capture global structure and long-range dependencies, while a diffusion process is used within each block to refine local, high-fidelity details.

2. The Layer/Depth Axis: The Transformer stack is "vertically" divided. The early layers function as a single-pass AR conditioning module (processing text and previous image blocks) to produce a strong prior, while the later layers perform iterative diffusion-based denoising, conditioned on this AR output.

The paper presents MADFormer as a "testbed" for analyzing AR-diffusion trade-offs. The central and most significant claim is that this mixed architecture, particularly an "AR-heavy" configuration (more layers dedicated to AR conditioning than diffusion), achieves a superior quality-efficiency balance (FID vs. Number of Function Evaluations, NFE) under constrained inference compute budgets. Experiments on FFHQ-1024 and ImageNet demonstrate this trade-off, with AR-heavy models showing up to a 75% FID improvement in low-NFE regimes.

### Strengths
1. Novel and Intuitive Architecture: The core idea of "vertically" splitting the Transformer stack into a single-pass AR-conditioning stage and a multi-step diffusion-refinement stage is elegant and well-motivated. It provides a clear and principled way to combine the strengths of both modeling paradigms—AR for global structure and efficiency, and diffusion for fine-grained fidelity.

2. Strong Core Experimental Result: The primary strength of the paper lies in the analysis presented in Figure 4. This experiment clearly and effectively demonstrates the central hypothesis: AR-heavy models (e.g., d=7 diffusion layers) significantly outperform diffusion-heavy models (e.g., d=28 layers) in low-compute (low NFE) settings. Conversely, it also shows that diffusion-heavy models achieve better final fidelity given a sufficient compute budget. This is a valuable and practical insight for designing generative models for different operational constraints.

3. Comprehensive Ablation Studies: The paper is supported by a thorough set of ablation studies that explore the proposed design space. The analyses of block granularity (Sec 4.2), the individual contributions of auxiliary modules (Sec 4.3), and the critical role of cross-block attention (Sec 4.5) are all valuable. The negative result (i.e., that modality-specific parameter sets have a trivial effect, Sec 4.4) is also a useful finding that favors a simpler, dense model.

### Weaknesses
1. Omission of Classifier-Free Guidance (CFG): The paper's core efficiency claim (NFE vs. FID) is made in a non-standard setting without CFG. CFG is a fundamental component of modern diffusion sampling and fundamentally alters the efficiency-quality trade-off. It is therefore unclear if the paper's conclusions hold in a standard, practical setting.

2. Questionable Effectiveness of Block Partitioning: The utility of this strategy is ambiguous. It helps on FFHQ-1024 (l=16 is optimal) but hurts on ImageNet-256 (where l=1, i.e., no partitioning, is best). This strongly suggests the benefit is highly dependent on the specific dataset and resolution, raising serious concerns about its generalizability.

3. Unfair Comparison Due to Training Convergence: The authors trained on ImageNet for only 50 epochs. It is well-known that diffusion models often require significantly more training to converge than AR models. The current finding (AR-heavy is better at low NFE) is likely an artifact of an unfair comparison between a "better-converged AR" component and a "severely under-trained Diffusion" component.

4. Uncompetitive Performance and Efficiency: The reported FID scores are all very high (e.g., ImageNet 27+, FFHQ 16+), indicating that all configurations are performing poorly. The qualitative results in Figure 8 (ImageNet) show significant artifacts, yet required 199 inference steps (NFE). This level of quality and computational cost is not competitive with current state-of-the-art models.

5. LLM Usage is also missed in this paper.

### Questions
The questions in this section is a extension of *Weakness* part. To substantiate the paper's core claims and increase its impact, I strongly recommend the authors answer the following key questions through experiments:

Question 1: Does the NFE-quality trade-off advantage persist after integrating Classifier-Free Guidance (CFG)?

The core advantage (AR-heavy is better at low NFE) was found in a CFG-free "vacuum." Will this advantage still exist after integrating standard CFG (e.g., w=4.0 or w=7.5)? Or will CFG narrow or even reverse the efficiency-quality gap between AR-heavy and diffusion-heavy models?

Question 2: Is the benefit of block partitioning merely a "special case" for high-resolution, or is it detrimental to complex datasets?

Why is l=1 (no partitioning) the best configuration on ImageNet-256? Does this mean the strategy is harmful for complex datasets? To decouple resolution and dataset complexity, would l=1 still be the optimal configuration if trained and evaluated on a high-resolution, high-complexity dataset like ImageNet-512? Does this expose a fundamental flaw in the strategy's generalizability?

Question 3: Is the current advantage of AR-heavy models merely an artifact of insufficient training?

AR models typically converge faster than diffusion models. If all models were trained to true convergence (e.g., 400+ epochs on ImageNet, not just 50), would the diffusion-heavy models catch up to or even surpass the AR-heavy models in the low-NFE setting? Is the current conclusion based on an "under-converged diffusion model"?

Question 4: In multimodal tasks, how do AR-heavy and diffusion-heavy models compare in text understanding and compositionality?

As a text-to-image model, does the single-pass AR conditioning in AR-heavy models actually help improve the layout and compositional accuracy of complex prompts (e.g., spatial relations, attribute binding)? Or would a fully-trained diffusion-heavy model perform better at text alignment and composition? A qualitative and quantitative analysis on a benchmark like MS-COCO is recommended.

Question 5: Confusing about Equation (4)

Why $z_{image}$, $\\epsilon$ and $z_{cond}$ are added together? Any reason or explanation about what does the term mean?

### Soundness
3

### Presentation
3

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
This paper introduces MADFormer, a novel hybrid generative framework that unifies Autoregressive (AR) and Diffusion modeling within a Transformer architecture for continuous image generation. The key innovation lies in its layer-wise modular design: alternating AR layers (for capturing fine-grained spatial dependencies) and Diffusion layers (for global structure and uncertainty modeling) to balance generation quality and efficiency. Experiments on ImageNet-256/512 and FFHQ-1024 demonstrate state-of-the-art performance—achieving FID scores as low as 1.92 (ImageNet-256) and 3.05 (FFHQ-1024) with only 20 inference steps. Notably, MADFormer outperforms pure diffusion baselines (e.g., DiT, SDXL) in both speed and fidelity, while its text-conditioning capability matches specialized text-to-image models like SDXL 1.0 in CLIP score (0.38 vs. 0.39) with 5× faster inference.

### Strengths
Paradigm Innovation: Hybrid AR-Diffusion Synergy
MADFormer addresses a critical gap in generative modeling by harmonizing AR’s fine-detail precision and Diffusion’s global coherence—an area where prior works (e.g., DiT, PixelCNN) forced a trade-off . The layer-wise modularity is not merely a structural novelty: AR layers explicitly model pixel-wise dependencies (critical for textures like hair or fabric), while Diffusion layers handle high-level semantics (e.g., object composition). This synergy is validated by ablations (Table 3) showing that removing AR layers degrades FID by 2.3 points on FFHQ-1024, and removing Diffusion layers causes mode collapse. The framework thus represents a principled advance in multi-paradigm generative design.

### Weaknesses
Unexplained Degradation of Hidden Loss at High λ
The "hidden loss" improves FID at λ=0.1 but degrades it at λ=1.0, yet the paper does not explain why higher weights harm performance. Plausible reasons include:
Over-constraining the AR condition, leading to inflexible latent distributions.
Conflict between the hidden loss and diffusion objective during backpropagation.
No ablations (e.g., loss weight annealing, latent visualization) are provided to clarify this behavior.

### Questions
How does the performance of MADFormer change when using faster samplers like DPMSolver or Euler? Does the optimal AR-diffusion layer ratio shift with different sampler types?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the respective limitations of diffusion models and autoregressive (AR) models by introducing a hybrid framework. The proposed model employs the autoregressive paradigm for a certain number of steps, while utilizing the diffusion-based approach for the remaining steps.

### Strengths
1. The paper presents a well-chosen research perspective. It effectively leverages the strengths and mitigates the weaknesses of both autoregressive (AR) and diffusion models, resulting in a well-designed hybrid framework.
2. The overall presentation is logically consistent and rigorous, and the experimental section is comprehensive.

### Weaknesses
1. This hybrid partitioning strategy raises a concern: does modeling part of the image using the autoregressive (AR) approach compromise the preservation of 2D spatial information?
2. There are several issues in Figure 1. For instance, why is there no BOT token for text, but only EOT? Moreover, the figure seems to omit the EOI token, and it is unclear why two consecutive EOI tokens appear.
3. Does the concept of “blocks” in your model design draw inspiration from the idea of block diffusion or related structured diffusion approaches?
4. In Table 1, why are later “depth” settings not evaluated? Does greater depth consistently lead to better performance?

### Questions
Additional experiments should be conducted to address the issues identified in the weaknesses section.

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
4

### Summary
This paper presents MADFormer, a hybrid generative framework that combines autoregressive (AR) and diffusion modeling within a unified Transformer architecture. The model applies AR modeling in early layers or across image blocks to capture global dependencies, and diffusion in later layers to refine local details. Experiments on FFHQ and ImageNet show that the hybrid approach improves efficiency under limited compute while maintaining strong image quality.

### Strengths
1. The motivation of the work is solid, addressing key limitations of existing generative models and providing a meaningful starting point for further research.
2. Each proposed idea is supported by targeted experiments, demonstrating substantial effort and thorough validation.

### Weaknesses
1. The paper claims that diffusion models suffer from slow generation speed. However, this seems to contradict recent findings — diffusion models are generally faster than AR-based image generators. For example, EMU3 and FLUX demonstrate shorter generation times compared to AR counterparts.
2. Regarding model design, I noticed that conditioning information for image generation—such as time steps and CFG—are embedded inside the model, leaving no control to the user. As far as I know, mature generative models often distill such conditioning away after training a strong conditional model. Does your approach reduce controllability or affect generation quality? I did not see an ablation or comparison addressing this point.
3. From Tables 1 and 2, it appears that performance improves as the model becomes closer to a pure diffusion model. Doesn’t this suggest that your hybrid approach both degrades performance and slows down generation due to the introduction of AR components?
4. I observed that all your visualizations are in-domain examples (from FFHQ and ImageNet). Have you experimented with out-of-domain prompts to evaluate generalization capability?

### Questions
Please refer to the "Weakness".

### Soundness
3

### Presentation
3

### Contribution
2
