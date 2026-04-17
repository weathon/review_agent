# Discontinuity-Preserving Image Super-Resolution via MAP-Regularized One-Step Diffusion

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
We propose a real-world image super-resolution framework that leverages a pretrained text-to-image Stable Diffusion model optimized for single-step sampling. Unlike traditional multi-step diffusion-based methods, which are computationally intensive, our approach enables fast inference while preserving high perceptual quality. To this end, we integrate a lightweight image enhancement module trained jointly with the diffusion model under a Maximum A Posteriori (MAP) formulation. The optimization includes a compound Markov Random Field (MRF) prior, derived from the anticipated discontinuity line field energy, which functions as a structural regularizer to preserve fine image details and facilitate deblurring. Existing single-step diffusion approaches often rely on distillation or noise map estimation, which limits their ability to generate rich pixel-space details. In contrast, our method explicitly models high-frequency line field consistency between the low- and high-resolution domains, guiding the image enhancer to reconstruct sharp outputs. By preserving and enhancing structural features such as edges and textures, our framework effectively handles complex degradations commonly encountered in real-world scenarios. Experimental results demonstrate that our method achieves performance that is comparable to or exceeds that of state-of-the-art single-step and multi-step diffusion-based image super-resolution methods qualitatively, quantitatively, and computationally.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes DMAPSR, a real-time image super-resolution (Real-ISR) framework. It leverages a pre-trained Stable Diffusion (SD) model and optimizes it for single-step sampling to achieve extremely fast inference. To maintain high perceptual quality in a single step, the authors introduce a lightweight image enhancement module. The core of the method is its use of Maximum a Posteriori (MAP) estimation combined with a composite Markov Random Field (MRF) prior as a structural regularizer.

### Strengths
1. The method achieves single-step inference *without* distillation, making it significantly faster (0.11s) than multi-step methods (like StableSR, 11.50s) and other single-step approaches (like OSEDiff, 0.15s).
2. The core innovation, the MRF line-field energy prior, excels at preserving edges, textures, and other discontinuous details.
3. Despite being a single-step model, DMAPSR achieves SOTA or near-SOTA results on several perceptual metrics (LPIPS, DISTS, CLIPIQA, MANIQA).
4. As shown in Table 2, DMAPSR's trainable parameter count (33.51M) is much lower than many multi-step methods and comparable to InvSR, giving it an advantage for deployment.

### Weaknesses
1. The method performs poorly on pixel-level fidelity metrics (PSNR). As seen in Table 1, its PSNR is lower than ResShift-4 on all datasets, which the authors attribute to ResShift being trained end-to-end for pixel alignment.
2. DMAPSR falls behind multi-step methods like SeeSR-50 on the FID metric. The authors suggest this is due to the advantage of multi-step generation in capturing global content alignment, indicating DMAPSR may have shortcomings in global distributional realism.
3. The authors claim the method generates "high-fidelity" results, but there are no specific experiments or metrics to back this up.
4. The paper lacks an ablation study that isolates the performance impact of using a single-step method versus a multi-step one within their framework.

### Questions
1. Does the proposed MAP estimation framework have broader applicability? For instance, could it be integrated into multi-step diffusion models as well?
2. The paper claims "high-fidelity" results. Could you provide metrics specifically designed to measure this, such as the method from [1]? It would also be insightful to see comparisons against other models focused on reducing artifacts and improving reconstruction accuracy (e.g., [2], [3], [4]).
3. During inference, are the  $\mathcal{E}\_{MRF}$  and  $\mathcal{E}\_{patch}$ energies computed in real-time by the enhancement module $g\_{\phi}$, or are they only used for training? Is the 0.11s reported in Table 2 purely the forward-pass time of $g\_{\phi}$?
4. The final loss function depends on three hyperparameters ($\lambda_p, \lambda_r, \gamma$). Could you provide more detailed ablation studies on how sensitive the model's performance is to different values of these hyperparameters?

[1]: Details or artifacts: A locally discriminative learning approach to realistic image super-resolution

[2]: Pixel-level and Semantic-level Adjustable Super-resolution: A Dual-LoRA Approach

[3]: SAM-DiffSR: Structure-Modulated Diffusion Model for Image Super-Resolution

[4]: StructSR: Refuse Spurious Details in Real-World Image Super-Resolution

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces DMAPSR, a world image super-resolution framework that achieves high-quality reconstruction in a single diffusion step. It combines a frozen Stable Diffusion backbone with a lightweight image enhancer trained under a Maximum A Posteriori (MAP) formulation. A Markov Random Field (MRF) prior with discontinuity-preserving line-field energy serves as a structural regularizer to maintain edges and textures. Experiments on RealSR, DRealSR, and DIV2K show that DMAPSR matches state-of-the-art multi-step diffusion methods and surpasses state-of-the-art single-step diffusion methods.

### Strengths
+ Novel theoretical perspective. The paper introduces a new way to regularize image super-resolution by formulating it under a MAP framework with a discontinuity-preserving MRF prior. This provides a principled and theoretically grounded approach to integrate structural regularization into diffusion-based models, which is conceptually novel.
+ Empirical Validation. The proposed method is thoroughly evaluated on multiple standard benchmarks — RealSR, DRealSR, and DIV2K — demonstrating consistent improvements in both perceptual and quantitative metrics. These comprehensive experiments clearly verify the effectiveness and robustness of the approach.
+ Clear introduction. The introduction is well writte, clearly motivating the problem, identifying gaps in prior diffusion-based approaches.

### Weaknesses
1. Unclear guarantee of one-step super-resolution. The paper claims that the proposed framework enables single-step super-resolution, but the mechanism ensuring this is not sufficiently justified. In Equation (4), it mentions that “yields a consistent noise estimate under the frozen pretrained model \epsilon_\theta.” However, it remains unclear what this noise predictor refers to — for instance, whether \epsilon_\theta corresponds to SD v1.4 or other models. Since the pretrained \epsilon_\theta (x_0^\prime, T) is not inherently trained to estimate noise directly from T in a single step, it is questionable how the model guarantees faithful super-resolution in one step.
2. Outdated baseline selection. The experimental comparison omits several recent and competitive baselines from 2025, including: 

[a] Fine-Structure Preserved Real-world SR via Transfer VAE Training

[b] DiT4SR: Taming Diffusion Transformer for Real-World Image Super-Resolution

[c] Adversarial diffusion compression for real-world image super-resolution

[d] Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors

3. Limited ablation effectiveness. The reported performance differences among ablation settings in Table 3 are marginal, suggesting that the individual contributions of the proposed MRF-consistency and patch-energy terms may be minor. The results do not convincingly demonstrate that these regularization components significantly improve either fidelity or perceptual quality.
4. Inconsistent baseline reporting. The paper discusses InVSR in the method comparison section and includes it in Figure 3, but it is missing from Table 1 across the three benchmarks. 
5. Ambiguity in Equation (4). Equation (4) defines the correction as X_T-\sigma_T, yet earlier derivations imply the additive form X_T-\sigma_T. If the subtraction is intentional, the paper should explicitly explain its theoretical basis.

### Questions
My major concerns are included in the above weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes DMAPSR (Discontinuity-Preserving MAP-Optimized Image Super-Resolution) — a single-step diffusion-based framework for real-world image super-resolution.
Unlike traditional diffusion approaches that require many iterative denoising steps, DMAPSR performs fast, single-step generation while preserving structural details. The key innovation lies in combining a Maximum A Posteriori (MAP) formulation with a Markov Random Field (MRF) prior based on line-field discontinuity energy. This acts as a structural regularizer that explicitly models edge and texture consistency between low- and high-resolution images.

The method integrates a lightweight image enhancer trained jointly with a pretrained Stable Diffusion model, aligning residual noise predictions for one-step sampling.

### Strengths
1. The paper introduces a novel integration of MAP estimation and MRF regularization into the diffusion sampling process, which explicitly enforces discontinuity preservation.

2. The proposed one-step diffusion framework achieves meaningful magnitude of speedups while maintaining perceptual quality, addressing one of the most pressing issues in diffusion-based SR models (slow inference).

3. The manuscript is well-structured, and clearly motivated.

### Weaknesses
1. While the MRF–MAP integration is elegant, the single-step training formulation (Eq. 4) and frozen noise predictor resemble prior work in OSEDiff and SinSR, reducing novelty in the “one-step” inference aspect. The main contribution therefore lies more in the regularization design than in sampling methodology.

2. The MRF-based loss is theoretically grounded but practically introduced as a differentiable energy term. The empirical justification of why this surrogate matches the MAP–MRF formulation could be elaborated with more intuition or visualization (e.g., line-field maps before and after training).

3. Training primarily on LSDIR + FFHQ subsets limits the generalization to broader real-world degradations. Additional results on out-of-distribution or camera-specific degradation would strengthen the claim of real-world applicability.

4. Some comparisons (e.g., StableSR vs. DMAPSR) differ in training settings or use pretrained backbones with different capacities. Clarifying whether DMAPSR fine-tunes the SD encoder or not would improve reproducibility.

5. While ablation tables are quantitative, qualitative differences (e.g., with/without inter-channel energy) are not visualized.

### Questions
1. How is the KL-based MRF consistency (Eq. 7) computed in practice — is it approximated with Monte Carlo samples or replaced by a deterministic surrogate? Some clarification on computational feasibility would help.

2. Could the authors provide sensitivity analysis for λ_p and λ_r? The paper fixes both to 1 without justification.

3. During inference, is the MRF energy used explicitly, or is it only a training-time regularizer? Including an inference-time MAP refinement step could further justify the formulation.

### Soundness
2

### Presentation
2

### Contribution
2
