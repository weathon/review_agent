# CineTrans: Learning to Generate Videos with Cinematic Transitions via Masked Diffusion Models

- Decision: Accept (Poster)
- Scores: 4, 6, 10

## Abstract
Despite significant advances in video synthesis, research into multi-shot video generation remains in its infancy. Even with scaled-up models and massive datasets, the shot transition capabilities remain rudimentary and unstable, largely confining generated videos to single-shot sequences. In this work, we introduce CineTrans, a novel framework for generating coherent multi-shot videos with cinematic, film-style transitions. To facilitate insights into the film editing style, we construct a multi-shot video-text dataset Cine250K with detailed shot annotations. Furthermore, our analysis of existing video diffusion models uncovers a correspondence between attention maps in the diffusion model and shot boundaries, which we leverage to design a mask-based control mechanism that enables transitions at arbitrary positions and transfers effectively in a training-free setting. After fine-tuning on our dataset with the mask mechanism, CineTrans produces cinematic multi-shot sequences while adhering to the film editing style, avoiding unstable transitions or naive concatenations. Finally, we propose specialized evaluation metrics for transition control, temporal consistency and overall quality, and demonstrate through extensive experiments that CineTrans significantly outperforms existing baselines across all criteria.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces CineTrans, a diffusion-based framework for generating coherent multi-shot videos with cinematic transitions, aiming to bridge the gap between single-shot and multi-shot video generation prevalent in recent works. To support this, the authors construct Cine250K, a large-scale, multi-shot video-text dataset with detailed shot annotations and hierarchical textual descriptions. CineTrans leverages an observed block-diagonal attention pattern for shot boundaries in diffusion models, and implements a mask-based control mechanism that enables frame-level control over cinematic transitions—even in a training-free regime. The approach is empirically validated against several strong baselines, showing superior transition control, temporal consistency, and overall quality via both standard and custom metrics.

### Strengths
The work introduces a novel mechanism that discovers and leverages a block-diagonal pattern in attention maps for transition control, directly inspiring an improved architecture; it further uses a mask-based strategy to align model internals with multi-shot video structure, enabling precise, shot-wise editing. A new large-scale dataset, Cine250K, fills a key gap with rich annotations (including frame-level labels and semantic stitching) and follows film-editing conventions. Extensive experiments provide thorough ablations and comparisons on transition control, intra- and inter-shot consistency, and aesthetic quality, complemented by insightful visual analyses. The approach transfers cleanly to customized or pre-trained diffusion backbones and improves both transition control and consistency in training-free and fine-tuned variants, with clear, intuitive examples and visual ablations highlighting advantages over existing methods.

### Weaknesses
1. Lack of Analysis on Limiting Scenarios: The paper demonstrates strong results on curated prompts and the Cine250K distribution, but does not critically examine or quantify limitations outside this scope, e.g., severe domain shifts, failure cases, or fundamental breakdowns of mask-based control when transition points are ambiguous or overlap.
2. Limited Theoretical Rigor or Insights: While the empirical demonstration of attention map patterns is clear (see Figure 4), the work lacks theoretical analysis or ablation to clarify why these attention correlations emerge (e.g., role of architecture, dataset, or objective). Are these properties universal, or dataset/model-dependent? This affects reproducibility and generalizability.

### Questions
1. Robustness to Ambiguous or Overlapping Shot Boundaries: How does CineTrans perform when shot boundaries are ambiguous or overlap semantically/visually (e.g., with crossfade-like transitions)? Is the mask mechanism brittle or robust under these settings?
2. Mask Relaxation Variants: Have the authors experimented with soft masks or probabilistic boundary definitions? Are there gains in aesthetic quality, or does this impair transition control?
3. Aesthetic Quality Decline After Fine-Tuning: What is the hypothesized cause of the decline in Aesthetic Quality (Table 2) after fine-tuning? Is it dataset-related or architectural?

### Soundness
2

### Presentation
2

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
This paper proposes a novel framework for generating coherent multi-shot videos with film-style transitions by introducing a block-diagonal attention mask and a “Visible-First-Frame” mechanism in video diffusion models, enabling precise shot boundaries and stable temporal consistency. Additionally, this paper constructs a 250K video–text dataset with frame-level shot labels to support video diffusion models in generating cinematic transitions and maintaining inter-shot consistency. This paper proposes a series of comprehensive metrics to evaluate the results of multi-shot video generation.

### Strengths
1. This paper contributes a large multi-shot dataset of 250K videos with frame-level shot boundaries and hierarchical captions.
2. The proposed method is simple and easy to follow.
3. The paper is well-structured and easy to read.

### Weaknesses
1. The method in this paper only adds a mask mechanism between shots to ensure content consistency, but this makes it difficult to maintain fine-grained consistency across different shots, especially for background regions, as shown on the left side of Figure 5. Table 1 (Intra-shot Consistency) also demonstrates that the improvement in consistency achieved by this method is limited compared to the baseline.
2. Although the paper claims to achieve cinematic transitions, the proposed method only supports hard cuts, which limits its range of applications.

### Questions
1. Using the mask strategy requires knowing between which frames a shot transition occurs. How does this paper determine the frame length of different shots when generating multi-shot videos? Is it manually set, or controlled by hyperparameters? If it is manually set, how do you ensure that a complete action can be generated within a single shot without being interrupted?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper proposes CineTrans, a masked-attention mechanism for controllable multi-shot, film-style transitions in video diffusion models, together with Cine250K, a curated multi-shot dataset. A simple block-diagonal mask added to attention logits enforces strong intra-shot and weak inter-shot correlations, aligning with observed attention patterns; the method works training-free and improves further with fine-tuning, achieving strong transition control and competitive quality on comprehensive metrics.

### Strengths
- Clear, well-motivated mechanism: The block-diagonal mask is explicitly defined and integrated into the attention logits, with principled alignment to measured intra- vs inter-shot attention structure.

- Strong empirical gains in control: Transition Control Score improves markedly over large T2V and multi-shot baselines while preserving quality.

- Thoughtful evaluation design: The paper evaluates transition control, intra-/inter-shot consistency, and aesthetic quality, including a novel Consistency Gap aligned to a film-edited reference set.

- Ablations that justify design: Impact of masking different layer ranges is analyzed for both UNet and DiT architectures, guiding the chosen masking strategy.

- **High-quality supplementary material: Appendix is detailed, code and logs are included, and an HTML project page with videos aids understanding and reproducibility.** Kudos on the hard work!

### Weaknesses
- Over-hard masking; missed opportunity for temporal scheduling: Equation (2) uses a binary mask with 0 on same-shot pairs and $-\infty$ across shots, which hard-zeros inter-shot attention in Equation (3). While effective, this may induce abrupt changes (also visible in the shared videos). A time-dependent or diffusion-step-dependent penalty could yield smoother transitions, e.g., replacing $-\infty$ by $-\alpha(t)$ that reaches $-\infty$ for a couple of time steps, this ramps near shot boundaries and across denoising steps, then saturates, before relaxing post-transition. This would preserve some cross-shot context pre-cut and reduce artifacts at boundaries. Please consider and, if possible, report a small ablation of annealed masks; if not possible, then consider it as a suggestion for future work.

- Related work gap: Prior work on long video generation with temporal control, e.g., VSTAR: Generative Temporal Nursing for Longer Dynamic Video Synthesis [ICLR’25], is not discussed. Please position CineTrans relative to these temporal-control strategies and clarify the conceptual differences between mask-based control and these schedules. Additionally, multiple prior works have considered Masked Attention for various applications (not just video generation). It would be a good idea to discuss a few of these related works as well, briefly.  

References:

[ICLR25] Li, Yumeng, et al. "VSTAR: Generative Temporal Nursing for Longer Dynamic Video Synthesis." The Thirteenth International Conference on Learning Representations.

### Questions
1. Layer selection sensitivity: Appendix E.2 suggests the best results with late layers for UNet and middle layers for DiT. Can you provide a brief heuristic or automated criterion for selecting mask layers on a new backbone, and quantify robustness to layer shifts?

2. Boundary robustness: At inference, how sensitive is control to small timestamp jitter in the provided shot boundaries, and do you recommend padding a few frames around boundaries to avoid failure cases like Figure 20?

3. Metrics specification: Appendix F mentions exact definitions. For completeness, would it be possible to include the precise formula for Transition Control Score and Consistency Gap in the main text or a boxed definition?

### Soundness
4

### Presentation
4

### Contribution
3
