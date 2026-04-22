# Diverse Text-to-Image Generation via Contrastive Noise Optimization

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Text-to-image (T2I) diffusion models have demonstrated impressive performance in generating high-fidelity images, largely enabled by text-guided inference. However, this advantage often comes with a critical drawback: limited diversity, as outputs tend to collapse into similar modes under strong text guidance. Existing approaches typically optimize intermediate latents or text conditions during inference, but these methods deliver only modest gains or remain sensitive to hyperparameter tuning. In this work, we introduce Contrastive Noise Optimization, a simple yet effective method that addresses the diversity issue from a distinct perspective. Unlike prior techniques that adapt intermediate latents, our approach shapes the initial noise to promote diverse outputs. Specifically, we develop a contrastive loss defined in the Tweedie data space and optimize a batch of noise latents. Our contrastive optimization repels instances within the batch to maximize diversity while keeping them anchored to a reference sample to preserve fidelity. We further provide theoretical insights into the mechanism of this preprocessing to substantiate its effectiveness. Extensive experiments across multiple T2I backbones demonstrate that our approach achieves a superior quality-diversity Pareto frontier while remaining robust to hyperparameter choices.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses T2I diffusion models’ mode collapse (limited diversity under strong text guidance) via Contrastive Noise Optimization (CNO). It optimizes initial noise latents (before DDIM sampling) using a contrastive loss in Tweedie denoised space: repelling batch samples to boost diversity, while anchoring each to its unoptimized version to preserve fidelity. A γ coefficient balances repulsion/attraction for stability. It achieves a superior quality-diversity Pareto frontier, enabling creative applications (e.g., diverse concept generation) without complex tuning, advancing practical T2I usability.

### Strengths
1. It innovatively addresses T2I diversity by optimizing initial noise (not intermediate latents) via Contrastive Noise Optimization (CNO)—a contrastive loss in Tweedie denoised space (not latent space) with repulsion (batch sample separation) and attraction (anchoring to unoptimized noise). This differs from CADS (text embedding perturbation) and PG (intermediate latent repulsion), solving diversity at the source while avoiding hyperparameter fragility.

2. It sets a superior quality-diversity Pareto frontier, with 5% overhead vs. DDIM but outperforming costly baselines (DiversityPrompt). Enabling diverse, high-fidelity generation (e.g., creative concept design) without complex tuning, it advances practical T2I usability and guides future noise-optimization research.

3. Rigorous validation on SD1.5/SDXL/SD3 shows top Vendi Score (diversity) and PickScore (quality). From Fig.3, the CNO is outperforming DDIM/CADS/PG consistently. Such observations are also true after reading the supplementary.

### Weaknesses
1. Experiments only use MS-COCO validation prompts (daily scenes) but lack other-domain prompts. Maybe the authors could consider to include other kind of prompt datasets to broaden this setup.

2. The CNO is only tested on the SD series models, I wonder would it be applicable to other structure based T2I models? Flux, DeepFloyd, etc.

### Questions
refer to the above.

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
4

### Summary
This paper addresses the pervasive problem of limited diversity in text-to-image (T2I) diffusion models, particularly under strong text guidance. The authors propose Contrastive Noise Optimization (CNO), a pre-processing framework that directly optimizes the batch of initial noise latents to encourage diversity, applying a contrastive InfoNCE loss in the denoised Tweedie space. The approach balances a repulsion term (encouraging diversity) with an attraction to the original sample (preserving semantic fidelity), and introduces a coefficient $\gamma$ to stabilize this trade-off. Theoretical analysis and comprehensive experiments across multiple diffusion backbones demonstrate improvements in both image diversity and quality.

### Strengths
1. The paper addresses the diversity challenge at its root by optimizing initial noise, rather than intervening mid-sampling or in text embedding space. This leads to a simple, efficient method that is model-agnostic and does not require modifications to the diffusion backbone.
2. The authors extend the mutual information perspective of InfoNCE to simultaneously account for positive and negative pairs (Section 4.3), providing formal justification for how diversity and fidelity are balanced. They further detail how the $\gamma$ parameter scales this effect.
3. Substantial empirical evidence is provided, including Table 1 and Figure 3, benchmarking against DDIM, Particle Guidance (PG), CADS, and DiversityPrompt across three major diffusion models (SD 1.5, XL, 3). The proposed method consistently outperforms baselines in key diversity metrics (Vendi Score, MSS) and maintains or improves quality/fidelity (PickScore, CLIPScore).
4. The exposition is clear, diagrams such as Figure 2 provide an intuitive grasp of the architecture/mechanism, and the methodology is described in practical, replicable pseudocode (Algorithm 1, Appendix B.1).
5. The method shows low sensitivity to hyperparameter choices (Section 5, Table 2), making it appealing for real-world adoption.

### Weaknesses
1. While the theoretical analysis is a strength, there is a (potentially confusing) notational inconsistency when switching between $\mathcal{L}{\text{CNO}}$, $\mathcal{L}{\text{InfoNCE}}$, and $\mathcal{L}_{\text{CNO}}^\gamma$ across pages 5-6. The exact role of $\gamma$ in the contrastive loss numerator/denominator, and how gradients are stopped or propagated through each component, could be clarified further. For readers less familiar with contrastive objectives applied to image generation, a stepwise, worked example would reduce ambiguity.
2. Section 5.1 references an ablation in Figure 4 testing $w\in{4, 8, 16, 64}$, yet the impact on specific image types, prompt complexity, or semantic regions is not deeply probed. There is no discussion of whether aggressive downsampling disproportionately harms long-tail or compositional prompts.
3. While automated metrics (Vendi, CLIPScore, etc.) are robust and reflect recent trends, there is a notable absence of human evaluation for subjective diversity, creativity, or aesthetic ranking beyond PickScore. As text-to-image generation is ultimately a user-facing task, such a study would strongly solidify the real-world value of the method.

### Questions
1. Could the authors provide additional quantitative and qualitative results on how extreme values of $\gamma$ (especially $<<1$ and $>>1$) impact the fidelity/diversity trade-off? Are there prompts or settings where the method consistently fails or collapses into poor outputs?
2. What are the empirical impacts and best practices for choosing the batch size $B$ in real-world usage? Is there a threshold beyond which diversity improvements stagnate or degrade?
3. Is the optimization of the initial noise vectors consistently stable across different diffusion model architectures, or are there architectures/settings for which CNO performs suboptimally?

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
The paper presents Contrastive Noise Optimization (CNO), a training-free diversity-boosting wrapper for text-to-image diffusion models. By marrying Tweedie-space contrastive repulsion with anchor-guided fidelity control, CNO produces markedly varied outputs in a single, lightweight pre-processing stage. Comprehensive evaluations on SD 1.5, SDXL and SD3 reveal consistent gains in Vendi score with negligible CLIPScore degradation, surpassing the latest zero-shot samplers in MSS and human-preference metrics and establishing a new Pareto frontier for quality-diversity trade-offs.

### Strengths
1.	Diversity is an important and interesting topic. Many models tend to overlook diversity issues during pretraining. 
2.	With complex prompts like “A cow sits in a truck with hay barrels in it,” other methods either fail or repeat elements, but our method produces diverse and semantically accurate images.

### Weaknesses
1. The proposed method improves generation diversity by optimizing the initial latent, which is sampled from a Gaussian distribution. As the initial latent is determined by this distribution, adjusting the variance of the Gaussian may also affect diversity. It would be interesting to discuss whether such a change could further enhance diversity.
2. It might be worth exploring whether the proposed method can further improve the diversity of the few-step distilled models.
3. A user study could help verify, through human evaluation, whether the method achieves higher generative diversity than the baseline.
4. It would be helpful if the authors could report the additional computational resources required as B increases.

### Questions
The proposed method shows strong qualitative and quantitative performance across multiple models (SD1.5, SDXL, SD3). However, as highlighted in Diffusion2GAN and Loopfree, few-step models often exhibit more severe diversity degradation. It would be valuable to further analyze whether the proposed method remains effective under such few-step models.

Diffuison2GAN: Distilling Diffusion Models into  Conditional GANs \
Loopfree: One-Way Ticket : Time-Independent Unified Encoder for Distilling Text-to-Image Diffusion Models

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
5

### Summary
This paper addresses the persistent issue of limited diversity in text-to-image (T2I) diffusion models under strong text guidance. Instead of prior approaches that tinker with intermediate latents or guide text embeddings during or across inference steps, the authors introduce Contrastive Noise Optimization (CNO): a lightweight, pre-processing method that optimizes the initial noise latents before sampling. By defining a contrastive InfoNCE loss in the Tweedie denoised latent space—incorporating both attractive (anchoring) and repulsive (diversifying) forces—CNO aims to produce batches of noise that yield diverse yet faithful image outputs. The method is anchored in theoretical reformulation of InfoNCE to balance fidelity and diversity, is computationally efficient, and demonstrates robust gains across several quantitative and qualitative metrics on Stable Diffusion variants and benchmarks.

### Strengths
The paper is technically sound, mathematically rigorous in its core claims, and offers strong theoretical motivation for adopting a contrastive loss in the denoised Tweedie latent space. The derivations for the mutual information bounds (Propositions 1 & 2) and the use of a regularization parameter ($\gamma$) are helpful and clear (see Sections 4.2–4.3 and Appendix A).

Extensive experiments (see Table 1 and Figure 3) across SD1.5, SDXL, and SD3 show the method consistently outperforms recent strong baselines like CADS, PG, and DiversityPrompt, especially on diversity metrics (Vendi Score, MSS) while retaining high-quality image generation (PickScore, ImageReward).

### Weaknesses
The loss function as written in Section 4.1/Algorithm 1 and its implementation in Algorithm 2 (for KL regularization) are slightly inconsistent in the summation indices and argument ordering, which could lead to confusion for others seeking to re-implement the approach.

The diversity-boosting methods compared against cover PG, CADS, and DiversityPrompt; however, some recent methods involving multi-concept fusion, 3D-aware T2I generation, and counterfactual interventions (cf. TweedieMix (Kwon & Ye 2025), DiffSplat (Lin et al. 2025), CoT-lized Diffusion (Liu et al. 2025), and Pan & Bareinboim (2025)) are not considered or even discussed. These methods, while not identical, are highly relevant to the diversity and compositional generalization challenge, and their exclusion diminishes the scope of empirical claims.

The main conceptual innovation of optimizing the initial noise with a contrastive loss heavily draws from existing InfoNCE/contrastive techniques, which are increasingly used in generative and diffusion-based frameworks. The extension to the Tweedie latent space is a clever adaptation, though it may be seen as an incremental, albeit meaningful, refinement of existing paradigms (see Guo et al. 2024, Ahn et al. 2024). The contribution would have benefited from a more careful positioning, clarifying the distinction between novelty and adaptation.

This paper provides a practical contribution to advancing diversity in text-to-image generative models.  The empirical evaluation is thorough, with solid quantitative and qualitative evidence supporting the claims of improved Pareto frontiers.

However, the work is somewhat incremental in terms of conceptual novelty, with the main technical move (contrastive optimization of initial noise) being an adaptation of ideas familiar within the generative/representation learning community. There are also missing references to closely related state-of-the-art works on multi-concept and compositional generation, diversity quantification, and alternative diversity control strategies, some of which could serve as strong baselines. Additionally, the exposition could benefit from clarified notation and more critical discussion of potential failure corners.

### Questions
The loss function as written in Section 4.1/Algorithm 1 and its implementation in Algorithm 2 (for KL regularization) are slightly inconsistent in the summation indices and argument ordering, which could lead to confusion for others seeking to re-implement the approach.

The diversity-boosting methods compared against cover PG, CADS, and DiversityPrompt; however, some recent methods involving multi-concept fusion, 3D-aware T2I generation, and counterfactual interventions (cf. TweedieMix (Kwon & Ye 2025), DiffSplat (Lin et al. 2025), CoT-lized Diffusion (Liu et al. 2025), and Pan & Bareinboim (2025)) are not considered or even discussed. These methods, while not identical, are highly relevant to the diversity and compositional generalization challenge, and their exclusion diminishes the scope of empirical claims.

The main conceptual innovation of optimizing the initial noise with a contrastive loss heavily draws from existing InfoNCE/contrastive techniques, which are increasingly used in generative and diffusion-based frameworks. The extension to the Tweedie latent space is a clever adaptation, though it may be seen as an incremental, albeit meaningful, refinement of existing paradigms (see Guo et al. 2024, Ahn et al. 2024). The contribution would have benefited from a more careful positioning, clarifying the distinction between novelty and adaptation.

This paper provides a practical contribution to advancing diversity in text-to-image generative models.  The empirical evaluation is thorough, with solid quantitative and qualitative evidence supporting the claims of improved Pareto frontiers.

However, the work is somewhat incremental in terms of conceptual novelty, with the main technical move (contrastive optimization of initial noise) being an adaptation of ideas familiar within the generative/representation learning community. There are also missing references to closely related state-of-the-art works on multi-concept and compositional generation, diversity quantification, and alternative diversity control strategies, some of which could serve as strong baselines. Additionally, the exposition could benefit from clarified notation and more critical discussion of potential failure corners.

### Soundness
2

### Presentation
2

### Contribution
2
