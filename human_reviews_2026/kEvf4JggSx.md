# UniRA: Unified Representation Alignment for Diffusion Models via Local, Structural, and Global Constraints

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Diffusion models have achieved tremendous advancements in generative modeling generation, enabling appealing experiences in visual content generation. Yet, their conventional training objective focuses merely on predicting added noises, without any explicit consideration on the learning of intermediate features. This narrow focus might learn redundant representations that capture limited semantics and poor structural details, thus leading to suboptimal performance. To ameliorate this, this paper proposes a unified representation alignment (UniRA) paradigm that augments the diffusion objective with explicit constraints on enhancing intermediate features. Specifically, UniRA enforces three complementary forms of alignment: local semantic fidelity for discriminative patch-level features, structural consistency to preserve relational organization, and global coherence to match overall feature distributions with real data. Extensive results on the challenging ImageNet and text-to-image benchmarks show that UniRA consistently improves convergence speed and synthesis performance, gaining improved FID and precision/recall scores under the same compute budget with compared baselines. Moreover, ablative analysis demonstrate the efficacy of UniRA in reducing feature redundancy and strengthening semantic information, and improving structural organization, thereby promoting high-quality synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes an aligment method of diffusion internal representation to the represetntation space of an existing pre-trained vision encoder model. The method has three alignment losses - one is local (and is like an existing methods called REPA) where internal patch representations are made to be similar to their equivalent patches in the pre-trained vision model. The other is structural where the patch-wise similarity structure of the represtentation is made to be similar to the pre-trained encoder. Finally, an global loss is proposed by training a discriminator which tries to distinguish internal diffusion representations and the pre-trained represetnation, pooled globally across the image.

### Strengths
This well executed paper has several strengths.

* The core idea, especially the structural loss, are very interesting and cleverly use the known strengths of pre-trained encoders to shape the internal representation of the diffusion model.
* The paper is nicely written and well structured - an enjoyable read.
* experimental validation is very good - including baselines, ablations and qualitative analysis.

### Weaknesses
I think the main issue the paper suffers is its relative limited significance. As much as I enjoyed the paper, most of the improvement comes from the local loss which was already proposed in REPA. I actually think this is not a reason to reject the paper, but it does diminish the scope and impact of the paper.

Minor points:

* I would have loved to see more discussion why MAE and other models perform worse here than DINO v2. This is just visible in the supplementary but I think is an important question.
* Would we see the same levels of improvements with other image datasets, considering DINO was trained very much in light of ImageNet.

### Questions
See above.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents UniRA, a unified representation alignment framework for diffusion models that enhances intermediate feature representations through three complementary constraints: 1) Local semantic alignment with pretrained visual encoders (e.g., DINOv2), 2) Structural consistency via relational similarity matching, and 3) Global distributional coherence using a lightweight adversarial discriminator.
The method encourages the denoiser’s internal features to be semantically rich, spatially coherent, and distributionally well-structured instead of focusing on output-space noise prediction.

Experiments on ImageNet-256/512, MS-COCO text-to-image, and multiple DiT/SiT architectures show that UniRA:
- Improves FID and IS over REPA and base diffusion transformers
- Produces more expressive and less redundant representations

### Strengths
- UniRA unifies three alignment levels (local, structural, global) into a simple, modular framework applicable to any diffusion transformer.
- Consistent FID/IS improvements across resolutions (256 / 512) and architectures (DiT, SiT) with faster convergence
- Includes text-to-image (MMDiT), ablations on alignment components (Table 4), weight sensitivity (Table 5), and encoder types/sizes (Table A2)
- Correlates FID with probe accuracy, shows layer-wise semantic improvement, timestep robustness, and reduced feature redundancy (Fig. 6).
- Fig. 5 clearly visualize restored spatial organization and semantic locality; generated samples (Figs. 3–4, Appendix) are high quality.
- Improves both efficiency and fidelity while remaining architecture-agnostic

### Weaknesses
- Builds directly on REPA, extending from local to multi-level alignment rather than introducing a fundamentally new mechanism.
- Performance relies on DINOv2-like teachers; the method is less self-contained and may struggle when domain shift breaks encoder semantics.
- The adversarial (global) term is said to be "optional" yet quantitative analysis of its stability or cost is minimal.
- While feature quality improves, how this affects bias or semantic controllability isn’t explored.
- The paper motivates alignment intuitively but provides little formal connection between improved intermediate representations and diffusion ELBO.

### Questions
1. How sensitive is UniRA to the choice of alignment depth (e.g., 4th vs 8th layer) beyond Table A2?
2. Did you experiment with adaptive weighting for $\lambda$, $\beta$, $\gamma$ during training (e.g., curriculum)?
3. Could UniRA be combined with self-distilled encoders (no frozen teacher) to mitigate reliance on external pretrained models?
4. For global alignment: how is discriminator stability ensured, and does adversarial collapse ever occur?
5. Would alignment at multiple timesteps (not only t=0.5) further improve robustness?
6. Please fix a small typo on line 721 on Table A1, lr row (010001).
7. Please add runtime/compute analysis. Since efficiency is a major claim, show training wall-time or FLOPs comparison with REPA.
8. Could you please clarify failure cases or visual artifacts from over-alignment (loss of diversity)?

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
This paper introduces UniRA, a unified representation alignment paradigm for diffusion models that augments the standard denoising objective with three explicit constraints on intermediate representations: local semantic alignment (patch-level matching with pretrained encoders), structural consistency (similarity matrix matching to capture relational organization among patches), and global distributional coherence (adversarially aligning the pooled intermediate features). Experiments on ImageNet and text-to-image generation benchmarks show consistent improvements in sample quality metrics (FID, IS, precision/recall), with additional ablations and analyses demonstrating reduced feature redundancy and improved semantic fidelity over strong baselines such as REPA and SiT.

### Strengths
- Clarity & transparency. The paper is clearly written with transparent method details; the appendix enumerates objectives, hyperparameters, and ablations; comparisons to strong baselines (e.g., REPA) are careful.
- Consistent empirical gains.Across challenging settings the method improves standard metrics; e.g., on ImageNet-256 with SiT-L/2, FID drops from 10.0 (REPA) to 8.5 (UniRA), with similar gains on image and text-to-image benchmarks.
- Reproducibility. Implementation choices and hyperparameter ranges are documented (e.g., Table A1), facilitating fair evaluation and reproduction.

### Weaknesses
- Lack of novelty--“unified” story is under-justified. Each sub-objective has prior art; the main contribution is a combination/recipe. The paper lacks a compelling argument for why these three must be unified and why these specific instantiations are preferable to plausible alternatives (e.g., replacing structural alignment with multi-scale contrastive losses, or global adversarial matching with MMD/SWD).
- Weak theory for multi-objective trade-offs. The three losses may conflict; current tuning relies on grids/heuristics, without principled weighting, curriculum, or adaptive schemes.
- Diversity–fidelity trade-off possibly obscured. Strong representation alignment can suppress diversity; the paper lacks systematic precision–recall curves or coverage metrics, and FID/IS alone can be misleading.
- Teacher dependence and domain-mismatch risk. Reliance on a frozen external encoder (e.g., DINOv2) can propagate teacher biases; performance under teacher–data mismatch is unclear.

### Questions
- If the global adversarial term is replaced by MMD/SWD/CLIP-score alignment, or the structural term by InfoNCE/multi-scale contrastive objectives, how do results change? What evidence shows this triad is not interchangeable?
- Under noise/occlusion/style perturbations to the teacher—or when swapping to weaker or narrower-domain encoders—what are the degradation curves of the three alignment terms? Any observable signs of overfitting to teacher features?
- Have you measured pairwise gradient angles/alignment between the three losses? Do you observe phases where one term dominates and others yield negative marginal returns late in training?
- Under strict parity of training steps × FLOPs × memory, how do UniRA, REPA, SiT, and other distillation-style methods compare in final metrics and convergence speed? Please report wall-clock and GPU hours.

### Soundness
3

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
5

### Summary
The paper presents UNIRA, a method that enforces representation alignment while training diffusion transformers. On contrast to the previous work REPA, that introducted representation alignment, UNIRA performs alignment in terms of local semantic fidelity, structural cohernace and global coherence leading to better representation alignment between features of the diffusion transformer and the pretrained vision encoder leading to better intermediate features in the diffusion models that can generalize to discriminative tasks. Experiments show that UNIRA outperforms REPA for generation on ImageNet 1M dataset as well as the features potaying better performance for discriminative tasks like classification on ImageNet

### Strengths
1. The paper points out potential drawback in patch level similarity based alignment leading to loss of structural and distribution information loss
2. Extensive experiments are performed for generative discriminative tasks and extensive ablation studies are performed to show that the performance boosts by utilizing UniRA.
3. The PCA analysis shows the improvement in representation quality for semantic segmentation of object with respect to REPA
4. The paper is well written and easy to follow.

### Weaknesses
1. Aligning the distributions strongly with a stronger distribution alignment function seems like a natural design choice for boosting performance. From the methodological perspective, what is the difference of the approach from REPA other than additional loss functions in the latent features for better distribution alignment? 
2. Are there better regularizations one could utilize to obtain better results? How is the proposed regularization, the optimal distribution alignment ?
3. In Table 4, Can the authors provide the results in the present of cfg. Does the performance trend remain the same with the presence of cfg
4. Additionally, the authors claim that[Ln 399-400], without the local coherency global coherency becomes unreliable, Could the authors provide visualizations of PCA similar to Figure 5,  for each of the loss components and show that this is the case.

### Questions
Please refer weakness

### Soundness
2

### Presentation
3

### Contribution
2
