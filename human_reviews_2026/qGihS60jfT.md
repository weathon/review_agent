# Consistent Noisy Latent Rewards for Trajectory Preference Optimization in Diffusion Models

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Recent advances in diffusion models for visual generation have sparked interest in human preference alignment, similar to developments in Large Language Models. While reward model (RM) based approaches enable trajectory-aware optimization by evaluating intermediate timesteps, they face two critical challenges: unreliable reward estimation on noisy latents due to pixel-level models' sensitivity to noise interference, and single-timestep preference evaluation across sampling trajectories where single-timestep evaluations can yield inconsistent preference rankings depending on the selected timestep.
To address these limitations, we propose a comprehensive framework with targeted solutions for each challenge. To achieve noise compatibility for reliable reward estimation, we introduce the Score-based Latent Reward Model (SLRM), which leverages the complete diffusion model as a preference discriminator with learnable task tokens and a score enhancement mechanism that explicitly preserves noise compatibility by augmenting preference logits with the denoising score function. To ensure consistent preference evaluation across trajectories, we develop Trajectory Advantages Preference Optimization (TAPO), which strategically performs Stochastic Differential Equations sampling and reward evaluation at multiple timesteps to dynamically capture trajectory advantages while identifying preference inconsistencies and preventing erroneous trajectory selection.
Extensive experiments on Text-to-Image and Text-to-Video generation tasks demonstrate significant improvements on noisy latent evaluation and alignment performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles two challenges in aligning diffusion models with human preferences using reward models: 
(1) Unreliable reward estimation on noisy latent representations, and (2) Inconsistent single-timestep preference evaluations across trajectories. 
To address these, the authors propose Score-based Latent Reward Model (SLRM), which incorporates task tokens and denoising score enhancement for noise-robust reward modeling, and Trajectory Advantages Preference Optimization (TAPO), which evaluates preferences across multiple timesteps to ensure consistent trajectory-level preference optimization. 
Experiments on text-to-image and text-to-video tasks demonstrate improved noisy latent evaluation, alignment, and generation quality compared to state-of-the-art baselines.

### Strengths
**Clarity:** 

The paper is clearly written, with motivating examples (Fig. 1) and architectural comparisons (Fig. 3). The separation between SLRM and TAPO contributions is explicit, making it easy to follow the technical flow.


**Quality:**

The improvements over baselines (e.g., LPO, Diffusion-DPO) are consistent across multiple preference and alignment metrics.

### Weaknesses
**W1. Score Enhancement Interpretability.** 

The denoising score augmentation mechanism is motivated theoretically, but its interpretability for human preference signals could be further explored. Does it merely stabilize noise handling, or does it bias preferences toward easier-to-denoise samples?

**W2. Lack of Baselines.** 

See Q3.

### Questions
1. Efficiency Trade-off. How does TAPO’s multi-timestep evaluation scale with longer trajectories or larger models?

2. Interpretability of Reward Scores. Can the authors provide more insights or visualizations into how denoising score enhancement affects reward distributions? For instance, does it correlate more strongly with human ratings than baseline models?

3. Baselines. More baselines are encouraged to compare with, such as SPPO [1] and RainbowPA [2].

[1] Bridging SFT and DPO for Diffusion Model Alignment with Self-Sampling Preference Optimization. arXiv:2410.05255, 2025.

[2] Diffusion-RainbowPA: Improvements Integrated Preference Alignment for Diffusion-based Text-to-Image Generation. Transactions on Machine Learning Research, 2025.

### Soundness
2

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
This paper introduces a novel framework to improve human preference alignment in diffusion models. It identifies two critical issues in existing reward-model-based alignment methods: (1) unreliable reward estimation on noisy latents, and (2) inconsistent preference evaluation across sampling trajectories. To address these, the authors propose:

1. SLRM (Score-based Latent Reward Model): integrates the diffusion model’s score function into the reward estimation process for noise-compatible evaluation.
2. TAPO (Trajectory Advantages Preference Optimization): performs multi-timestep evaluation to ensure consistent and trajectory-aware optimization.

Experiments on text-to-image (SD3.5) and text-to-video (Wan2.1) tasks show that this method outperforms prior works (LPO, SPO, Diffusion-DPO) on both alignment and preference metrics.

### Strengths
1. Clear motivation: The paper identifies a genuine gap in current diffusion preference optimization(noise robustness and trajectory consistency) and addresses both with well-justified solutions.

2. Technical novelty: The integration of the diffusion score function into reward estimation (SLRM) is original and theoretically grounded. TAPO’s multi-timestep evaluation design is practical and effective.

3. Strong empirical results: Extensive experiments on both T2I and T2V tasks, with detailed ablation studies and visual comparisons, show consistent superiority.

### Weaknesses
1. The diffusion-based step-wise reward model is interesting and makes sense to me; however, since the preference ground truth in the training data is still based on the final images, it may limit the generalization ability of the SLRM.
2. I didn’t observe a significant improvement of TAPO over LPO in the qualitative comparison presented in Table 5.

### Questions
1. How do the authors ensure that the SLRM trained on final-image-based preference labels can generalize to intermediate noisy latents, given that the ground-truth supervision does not explicitly cover those states?
2. Can the authors clarify under which conditions TAPO provides larger qualitative gains over LPO?

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
4

### Summary
This paper presents a framework for aligning diffusion models with human preference, addressing two challenges: unreliable reward scoring on noisy latents and inconsistent preference ranking from single-timestep evaluations. Specifically, the former is addressed by a Score-based Latent Reward Model (SLRM) that uses a diffusion backbone and a score enhancement mechanism to maintain noise compatibility. The latter is addressed using Trajectory Advantages Preference Optimization (TAPO), a trajectory-aware sampling strategy that evaluates rewards at multiple timesteps. Experiments show strong alignment performance on T2I and T2V tasks.

### Strengths
1. The proposed reward model, *i.e.,* SLRM, is innovative and efficient. It directly inherits the noise compatibility from the pretrained diffusion models while providing a more comprehensive evaluation of the text-image alignment, thanks to the self-attention mechanism at multiple semantic levels. Additionally, the solution to degraded noise compatibility is insightful: It is necessary to retain the noisy compatibility by incorporating denoising score matching objective into the reward model training.
2. The proposed TAPO strategy provides a practical way to account for the entire sampling trajectory of diffusion models.

### Weaknesses
1. The proposed latent-level reward model one inherent drawback. Unlike the pixel-level reward models that have access to high-frequency information, latent reward models may be less sensitive to finer details like textures. This also partial explains why SLRM underperform pixel-level score models, such as HPSv3 and PickScore, as the later stage of the sampling process (where $t\approx 0$) tends to generate finer details.

2. The paper fails to mention a highly-relevant prior work that addresses a similar core problem regarding sequential sampling trajectory. While this paper introduces "trajectory-aware optimization" as a new solution to "single-timestep preference evaluation," it omits the existing work by Yang et al. (2024) [1], which shares the same insight that "diffusion models focus on different dimensions at different timesteps" and introduces a dense reward perspective into DPO-style objectives.

3. Lack of computational cost comparison. The proposed TAPO method introduces significant computational overhead, requiring $n=8$ SDE steps and $P=4$ candidates, resulting in 64 reward evaluations per iteration. However, the training time or iteration cost for the baselines is not included in the paper for fair comparison.

4. Missing general quality metrics results. While the authors explicitly state that excessive exploration ($P=5$) leads to "reward hacking," there is no quantitative evidence against it for the default setting $P=4$. To see if preference scores improve without sacrificing underlying image fidelity, standard metrics such as FID would be helpful.

[1] Yang, Shentao, Tianqi Chen, and Mingyuan Zhou. "A Dense Reward View on Aligning Text-to-Image Diffusion with Preference." Forty-first International Conference on Machine Learning.

### Questions
1. Have the authors tried to combine both pixel-level reward models (like HPSv3, for high-frequency details at $t \approx 0$) and the latent-level SLRM (for noise compatibility at $t > 0$)? A hybrid reward based on the timestep seems necessary to capture all quality aspects.
2. Instead of augmenting the score logit with the multiplicative score distance (Eq. 8), is it possible to directly combine the denoising score objective (Eq. 6) as an additive regularization loss? How does that compare with the proposed enhancement approach?
3. Currently, TAPO samples $P=4$ candidates for both win and lose samples, and only utilizes the best and worst among the four 15. Have the authors considered random selection to avoid reward hacking? What if the algorithm is run on all possible pairwise combinations with a smaller $P$, such as $P=2$ or $P=3$?

### Soundness
3

### Presentation
4

### Contribution
3
