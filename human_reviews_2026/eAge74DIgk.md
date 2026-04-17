# LitExplorer: Training-Free Diffusion Guidance with Adaptive Exploration-Filtering Framework

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Diffusion models possess strong general generative capabilities, yet they remain insufficient when aligned with specific target objectives. Fine-tuning methods can enhance alignment but incur high training costs and face the risk of reward hacking. Consequently, training-free guidance mechanisms have emerged, which leverage external signals during inference to steer the generative distribution toward high-reward regions. However, existing training-free approaches encounter two key challenges: first, the guidance process tends to over-bias generation toward the target distribution, at the expense of excessively narrowing the pretrained model’s generative space; second, the guidance signals are mechanically imposed throughout inference, lacking mechanisms to identify and filter out ineffective or redundant signals. To mitigate these limitations, we propose \ourmethod{}. Regarding the first issue, we introduce exploratory guidance signals through \pos{} to prevent generation paths from prematurely converging to a single mode, while dynamically balancing the trade-off between exploration and stable generation based on denoising progress. This alleviates the excessive contraction of the generative space without deviating from the target distribution or the pretrained distribution. Regarding the second issue, to enable precise and efficient guidance, we incorporate an adjudication mechanism that evaluates the validity of guidance signals and adaptively eliminates ineffective or redundant ones. To demonstrate the generality of \ourmethod{}, we conduct extensive evaluations in both single-objective and multi-objective scenarios. Results show that \ourmethod{} achieves significant improvements over existing training-free baselines in terms of generative diversity, target alignment, and inference efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on training-free methods for aligning diffusion models with specific objectives while maintaining diversity, fidelity, and efficiency. The proposed method adopts an exploration technique to control the diversity-fidelity tradeoff and adaptive guidance to control the quality-efficiency tradeoff. Experiments are conducted with image generation with various objectives to compare with prior works.

### Strengths
1. The paper clearly identifies the problem of diversity degradation after alignment and the inefficiency of previous guidance methods.
2. Multiple metrics are incorporated to holistically evaluate preference, fidelity, diversity, and richness.
3. The proposed method improves preference, richness, and speed over the base model without sacrificing diversity or fidelity.

### Weaknesses
1. The method is mostly based on heuristics, and the design choices are not clearly justified. Though they provide ablation on diversity and diversity-fidelity tradeoff, it doesn't assess each component of the methods, thus their claims regarding each component lack evidence. (See Question 1, 2)
2. Quantitative results presented in Section 3 don't contain margin of error, hence it's hard to assess how much the proposed method outperforms prior methods.
3. The method neads to train a Control Network, so it's not totally training-free.

### Questions
1. How does exploration supplement variable help increase diversity? If the selection from the candidates is done randomly, it can increase diversity. However, the method uses a reward to select the highest reward sample, which actually decreases diversity as the number of candidates increases by always selecting the optimal candidate.
2. The motivation of 'inheritance' in inheritance-restart is unclear. The noise level, i.e., the magnitude of the exploration supplement variable, should be kept along with denoising to keep the sample on-manifold. Also, why would reusing the previous exploration supplement variable help improve?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a test-time alignment framework for text-to-image diffusion. The method aims to balance exploration and refinement while also improving efficiency during sampling. To achieve this, the approach combines reward-guided candidate sampling, a progression-aware control network, and screening and early-stop criteria to avoid unnecessary guidance. Experiments on SD v1.5 and SDXL with multiple reward models (PickScore, AES, ImageReward) show consistent improvements in reward-aligned image quality without requiring model fine-tuning.

### Strengths
1.	The framework is conceptually clear, separating exploration–exploitation and quality–efficiency in a straightforward manner.
2.	The method operates in a plug-and-play manner with respect to the diffusion backbone. This makes it easy to plug into existing pipelines without T2I diffusion model updates.
3.	The experiments shows consistent improvements across multiple reward signals.

### Weaknesses
1. While the paper emphasizes being "training-free", the framework relies on a learned network to estimate how close the current state is to the clean data sample. However, the training setup, dataset scale, and generalization capability of this network are not described in sufficient detail, and there is no widely recognized pretrained model that can be adopted for this role. This dependency weakens the claim that the method is truly "training-free", making it closer to a "test-time plugin" that still requires a separate trained module.

2. The paper presents a compelling conceptual framing through expolration-exploitation and quality-efficiency balancing, but the actual pipeline consists of multiple components and eight hyper-parameters. This makes the method less modular than suggested. Specifically, Table 4 does not fully reveal the isloated impact of eac hmodule, and the hyper-parameters settings in Table 5 lack sufficient justification for their chosen values. A more thorough study on component interactions and hyper-parameter sensitivity would strengthen the work, particularly since reward-guided generation is known to be sensitive to reward scale and optimization dynamics [1].

3. The paper does not compare against recent fine-tuning based alignment approaches such as [2] DRaFT, [3] Diffusion-RPO, and [4] DRTune. Adding these baselines would better contextualize the claimed advantages in quality and efficiency compared to fine-tuning based method.

(Minor)
1. The experiments do not present an execution time (or a wall-clock runtime) comparison on overhead breakdown for the full pipeline.

2. The experiments primarily focus on reward-model metrics without human preference studies or user evaluation. While these scores improve, it remains unclear whether the gains reflect real perceptual quality rather than reward-model alignment. Incorportaing even a small-scale human evaluation would make the empirical validation better.

3. The paper shows experimental results on U-Net based models(SDv15 and SD XL1.0), and it would be better to explore its applicability to recent Transformer-based models such as FLUX or SD3.

### Questions
Q1. Diffusion sampling assumes that intermediate states lie approximately on a perturbed noise manifold with respect to the forward process [5], [6]. Since the method introduces additional gradients and regularization signals during sampling, could these guidance terms push intermediate latents off the diffusion trajectory? How does the regularizations work within this method in terms of preverving manifold during denoising process.
Q2. [Typos] $x^{*}$ in Line 226 and $x^{⋆}$ in Equation (7) are not clealy distinguished. Also, $p_t$ in line 299 and $\alpha_t$ in  Figure 3 should be unified.

[1] Subject-driven Text-to-Image Generation via Preference-based Reinforcement Learning
[2] Directly Fine-Tuning Diffusion Models on Differentiable Rewards
[3] Diffusion-RPO: Aligning Diffusion Models through Relative Preference Optimization
[4] Deep Reward Supervisions for Tuning Text-to-Image Diffusion Models
[5] Diffusion Posterior Sampling for General Noisy Inverse Problems
[6] Manifold Preserving Guided Diffusion

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes LitExplorer, a training-free plugin to align diffusion models with target rewards while preserving diversity and reducing compute. It introduces an Inheritance-Restart exploration mechanism with an adaptive diversity-fidelity controller, and a Quality-Efficiency arbitration that screens ineffective guidance and triggers dynamic early stopping based on marginal reward gains. Experiments on SD v1.5 and SD-XL across Pick-a-Pic and HPSv2 show improved alignment preference, fidelity, and diversity, with faster sampling.

### Strengths
- Clear motivation; mitigates distribution narrowing and reward hacking while maintaining fidelity.
- Performance gains: top or top-2 on most of 12–13 metrics across datasets and backbones, improved diversity (LPIPS/TCE/NIQE), and generalizes to multiple reward objectives (PickScore, AES, ImageReward) as a drop-in plugin.
- Efficiency gains; reduces compute without degrading quality.

### Weaknesses
Empirical Support for Motivation
- DAS claims to avoid reward over-optimization while allowing reward guidance at inference time. However, it is unclear whether Fig. 1 includes results for other baselines like DAS beyond DyMO. It would strengthen the argument to include such comparisons.

Novelty
- The proposed _guidance screening_ module appears similar to early stopping. It seems to be an independent component that could be easily integrated into other methods rather than a fundamentally novel contribution. While tackling efficiency is valuable, the novelty of introducing a standalone early-stopping-like mechanism is questionable.


Experiment Setup
- The experiments only cover SD1.5 and SDXL; evaluations on more recent backbones would improve credibility. 
- Recent diffusion inference-time scaling baselines [1, 2] are missing. 
- Since the performance of sampling-based methods (DAS, [1], [2]) depends heavily on hyperparameters such as the number of particles, while their configurations used were missing. Experiments analyzing scaling with respect to nFE or similar metrics are also essential.

---

[1] Inference-time scaling for diffusion models beyond scaling denoising steps, CVPR, 2025

[2] A General Framework for Inference-time Scaling and Steering of Diffusion Models, ICML, 2025

### Questions
See weaknesses above

### Soundness
3

### Presentation
2

### Contribution
2
