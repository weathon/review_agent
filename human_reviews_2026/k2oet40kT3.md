# Generate the Forest before the Trees - A Hierarchical Diffusion model for Climate Downscaling

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Downscaling is essential for generating the high-resolution climate data needed for local planning, but traditional methods remain computationally demanding. Recent years have seen impressive results from AI downscaling models, particularly diffusion models, which have attracted attention due to their ability to generate ensembles and overcome the smoothing problem common in other AI methods. However, these models typically remain computationally intensive. We introduce a Hierarchical Diffusion Downscaling (HDD) model, which introduces an easily-extensible hierarchical sampling process to the diffusion framework. A coarse-to-fine hierarchy is imposed via a simple downsampling scheme. HDD achieves competitive accuracy on ERA5 reanalysis datasets and CMIP6 models, significantly reducing computational load by running on up to half as many pixels with competitive results. Additionally, a single model trained at 0.25° resolution transfers seamlessly across multiple CMIP6 models with much coarser resolution. HDD thus offers a lightweight alternative for probabilistic climate downscaling, facilitating affordable large-ensemble high-resolution climate projections. See a full code implementation at: https://github.com/HDD-Hierarchical-Diffusion-Downscaling/HDD-Hierarchical-Diffusion-Downscaling

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes HDD: a hierarchical diffusion downscaler that conditions the score network on a shape schedule, learning to denoise + un-coarsen jointly. Training and sampling embed each latent back to full resolution while alternating upsample → predict 𝜖 → downsample in the reverse process. The authors derive upper-bound compute savings via average active pixels, with up to ≈3× theoretical speedup for a linear shrink schedule; they evaluate on ERA5 and CMIP5 GCM transfer.

### Strengths
- Conceptual clarity and drop-in design. The paper cleanly formulates a hierarchical diffusion process as a strict generalization of EDM, compatible with existing denoisers. When the shape schedule is identity, it exactly recovers the vanilla EDM objective.

- Domain alignment. The coarse-to-fine structure is well motivated for spatial downscaling, naturally mirroring multi-scale atmospheric organization.

- Evaluation breadth. Benchmarks on ERA5 and several CMIP5 GCMs are thorough. Including RCM (CCAM) comparisons and CO₂ estimates is commendable.

### Weaknesses
- Theoretical compute saving not realized. Algorithm 1 explicitly upsamples 𝑧 back to full resolution before passing it to the UNet (fθ(U_t(z), t, (h_t,w_t))). Thus, all expensive convolutional operations still occur at full 𝐻×𝑊. The 3× saving derived from average active pixel area (α = 1/3) assumes the model operates natively at reduced resolutions, which it does not. The “3× pixel/FLOP saving” therefore represents a notional bound, not a real efficiency gain. Any empirical runtime saving would be far smaller.

- Underdeveloped novelty relative to prior hierarchical diffusion work. The coarse-to-fine generative idea overlaps with prior Cascaded Diffusion Models and Laplacian Pyramid GANs; the paper does not explicitly contrast HDD’s single-model training with these established cascades.

- Missing recent baselines. Omitted comparisons to strong and relevant lines: CorrDiff (corrective diffusion downscaling), STVD (Spatiotemporal Video Diffusion for precipitation), Neural operators (FNOs) for arbitrary-resolution downscaling, Foundation models like ClimaX used for resolution-agnostic transfer.

### Questions
- Score consistency: With upsampling and downsampling operators not being exact inverses, the score used at upsampled resolution approximates a different target than the true full-res score. Can you formalize the induced bias on the score (e.g., via a projection operator) and when it becomes negligible?

- Shape–noise coupling: You currently draw $\sigma_t$ and $s_t$ jointly but independently. Did you consider shape-dependent noise schedules (e.g., higher SNR at coarse scales to stabilize global structure)? Any ablation on coupling strategies (e.g., log-SNR that depends on shape)?

- Does Appendix I mirror any changes to the standard diffusion loss and training algorithm suggested in Algorithm 1?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a new methodology for training diffusion models. The method makes use of hierarchical structure, noising and denoising at different resolutions, and forcing scale consistency. The method can applied to different diffusion model, making it a plug-and-play with existing architectures. The method is specifically tailored for climate downscaling applications, and is evaluated on multiple benchmarks.

### Strengths
- Method is architecture-invariant, therefore, applicable to a broad range of existing diffusion architectures.
- Authors identify that weather exhibits similar patterns (power-law at different coarse/fine scales) to diffusion models (high vs low noise frequencies).
- The theoretical idea of going from coarse to fine, not just with noise but also with spatial resolution is an interesting research direction.
- Authors carry out an in-depth analysis on the theoretical speed up, justifying the use of scale changes in image resolution.
- Authors provide extensive ablations.

### Weaknesses
- Line 207: "blue arrows". All arrows are blue, it would be better to use different colors or make wordking clearer.
- Evaluation is only carried out for Australian extent. Could authors provide an intuition for how the model performs in other latitudes?
- Authors claim method is architecture-agnostic yet the use of their two additional scalar inputs implies having to retrain the models, thus, invalidating the plug-and-play claim.
- Figure 4 layout and explanation can be largely improved. It is not clear what authors want to convey.

### Questions
- Do authors explore variability/std of the outputs generated? Given the stochasticity of diffusion models, it would be interesting to see if authors sample multiple solutions for a better estimation of outputs uncertainty.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
### Goal
The paper discusses a method to train and sample from a multi-resolution diffusion model. The model is then applied to a transfer learning task, where it is trained on ERA5 (0.25°) conditionally on a down-sampled resolution of 1.5° and used to downscale samples from expensive general circulation models (GCMs), natively at resolutions ranging from 1.4° to 2.5°.

### Contributions
The contributions of this paper are limited. Many multi-resolution diffusion and flow matching models are present in the literature, but many of them (e.g. [1,2], the closely related cascaded diffusion models [3], and many others) are not even the cited. The few that are [4,5,6] are only cited *en passant*, and are not properly discussed or used as baselines in the experiments. This makes it very hard to decipher what is novel about the proposed Hierarchical Diffusion Downscaling model, beyond the fact that it is a multi-resolution diffusion model applied to a downscaling task. 


[1] Gu et al. (2023), f-DM: A multi-stage diffusion model via progressive signal transformation. 
[2] Teng et al. (2024), Relay diffusion: Unifying diffusion process across resolutions for image synthesis.
[3] Ho et al. (2022). Cascaded diffusion models for high fidelity image generation.
[4] Jin et al. (2024). Pyramidal flow matching for efficient video generative modeling.
[5] Zhang et al. (2022). Dimensionality-varying diffusion process.
[6] Campbell et al. (2023). Trans-dimensional generative modeling via jump diffusion models.

### Strengths
* Evaluation on an interesting transfer learning task: the paper shows that a model trained on ERA5 can be used to downscale simulations from GCMs.

### Weaknesses
* The paper lacks a comprehensive literature review, making it hard to understand where it is situated / what makes it different in the space of multi-resolution generative models.
* The experiment is not convincing. The only baseline is a standard EDM, and no other techniques for multi-resolution diffusion are considered. It is also unclear whether hyperparameters were properly tuned for both EDM and HDD, and the lack of confidence intervals / error bars makes it hard to understand the significance of the magnitude in improvement.
* Lack of details about the task and datasets being used (e.g. why is the downscaling at 0.5° if the model was trained to produce images at 0.25°? What is the number of samples in each split? etc.)
* While some of the hypotheses being raised in the paper are interesting, they are presented in a hand-wavy fashion, often resulting in unsubstantiated mathematical claims. For instance, take the claim: "more dimension-destruction steps is exactly parallel to more time steps in standard DDPM: it yields finer increments, lower local error, and a more faithful final reconstruction". No evidence is provided beyond the results of the experiment, which I personally find unconvincing.
* Typos (line 143, 207), messy figure ordering (Fig. 4 is the first being mentioned and the last being displayed), and inconsistent use of capital letters (lines 154, 356) make the paper less clear.

The paper could strongly benefit from:
* A thorough literature review.
* Comparisons with other methods for multi-resolution diffusion.
* Simple, toy experiments used to validate some of the hypotheses about the improvements in performance over fixed-resolution models.

### Questions
None.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a hierarchical diffusion model for probabilistic climate downscaling. The key idea is to impose an explicit coarse-to-fine hierarchy on the diffusion process by coupling Gaussian noise injection with progressive spatial downsampling and upsampling. This “dimension destruction” allows the model to learn multi-scale representations efficiently, thereby reducing computational cost. HDD achieves comparable or better accuracy than standard diffusion downscaling methods (e.g., CorrDiff) and Earth-ViT while operating on fewer pixels and generalizing across multiple GCMs with varying input resolutions.

### Strengths
1) The proposed coupling of noise scheduling and spatial scaling is elegant and well-motivated by both theoretical (spectral scaling in atmospheric data) and empirical (RAPSD) evidence.
2) The idea of treating spatial resolution as a conditioning variable in diffusion steps is intuitive and potentially impactful.
3) The model achieves competitive or superior results across ERA5 and CMIP5 benchmarks, reducing FLOPs and CO₂ footprint significantly.
4) The authors provide a detailed methodology and open-source implementation, which supports reproducibility.

### Weaknesses
1)	The paper assumes that hierarchical training (sampling one shape per iteration) leads to unbiased learning of the joint coarse-to-fine distribution, but this is not shown.
2)	There is no ablation on the effect of mismatch between σ_t and s_t sampling distributions. E.g.,, what happens when fine scales are learned early or skipped frequently?
3)	The model is described as “architecture-agnostic” but the experiments only use a U-Net backbone. It is unclear how the hierarchical conditioning would integrate with transformer-based or attention-heavy architectures where spatial scaling affects positional embeddings.
4)	The experimental comparison is limited. HDD is only compared against its own baselines (Base EDM, Earth-ViT) but not against other state-of-the-art downscaling methods that have been evaluated on ERA5 or similar reanalysis datasets. Examples of relevant SOTA baselines include:
[1] Mardani et. al., “Residual Corrective Diffusion Modeling for Km-scale Atmospheric Downscaling”, Communications Earth & Environment，2025.
[2] Zhong et. al.，“Debias Coarsely, Sample Conditionally: Statistical Downscaling through Optimal Transport and Probabilistic Diffusion Models”，NeurIPS 2023.
[3] Ignacio et. al., “Dynamical-generative downscaling of climate model ensembles”, PNAS, 2025.
[4] Robbie et. al. , “Generative Diffusion-based Downscaling for Climate”
[5] Declan et. al., “Resolution-Agnostic Transformer-based Climate Downscaling”
[6] Sebbar et. al., “Machine-Learning-Based Downscaling of Hourly ERA5-Land Data”, Atmosphere, 2023.
Minors:
1)	Line 279-280,  panguweather->PanguWeather
2)	Figure 4 needs to be better organized.

### Questions
1)	How is the shape schedule s_t chosen or optimized? Is it learned or hand-crafted?
2)	Can the hierarchical diffusion process cause aliasing artifacts during upsampling, and if so, how is this mitigated?
3)	Can the same trained model handle multiple target resolutions (e.g., 0.5°, 0.25°) without retraining?
4)	Why does the Pangu-based Earth-ViT baseline perform so poorly in the reported tables? The results show notably higher RMSE and lower spatial correlation than both HDD and even traditional RCMs. Could this be due to under-training, suboptimal hyperparameter choices, or the use of a configuration not well-suited for downscaling (e.g., lack of resolution-specific finetuning)? Clarifying this is important, as it affects the credibility of the baselines used for comparison.

### Soundness
2

### Presentation
2

### Contribution
2
