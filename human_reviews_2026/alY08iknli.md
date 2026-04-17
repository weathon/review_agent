# PCPO: Proportionate Credit Policy Optimization for Preference Alignment of Image Generation Models

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
While reinforcement learning has advanced the alignment of text-to-image (T2I) models, state-of-the-art policy gradient methods are still hampered by training instability and high variance, hindering convergence speed and compromising image quality. Our analysis identifies a key cause of this instability: disproportionate credit assignment, in which the mathematical structure of the generative sampler produces volatile and non-proportional feedback across timesteps. To address this, we introduce Proportionate Credit Policy Optimization (PCPO), a
framework that enforces proportional credit assignment through a stable objective reformulation and a principled reweighting of timesteps. This correction stabilizes the training process, leading to significantly accelerated convergence and superior image quality. The improvement in quality is a direct result of mitigating model collapse, a common failure mode in recursive training. PCPO substantially outperforms existing policy gradient baselines on all fronts, including the state-of-the-art DanceGRPO. Code is available at https://github.com/jaylee2000/pcpo/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Proportionate Credit Policy Optimization (PCPO), a reinforcement learning framework for aligning text-to-image (T2I) generation models. The authors identify disproportionate credit assignment across diffusion or flow timesteps as a core source of instability in PPO/GRPO-style methods. This issue leads to high-variance gradients, excessive clipping, and eventual model collapse. Empirical results across diffusion and flow models: Stable Diffusion 1.4/1.5 and FLUX, demonstrate: faster convergence, improved image fidelity, and reduced model collapse.

### Strengths
1. **Quality**

Ablations convincingly demonstrate the cumulative contribution of PCPO’s components. Detailed appendices cover proofs, hyperparameters, and resource requirements for reproducibility.


2. **Experimental Rigor**

The inclusion of Linear Mixed Model (LMM) statistical analyses adds robustness to the evaluation, while the human preference study further validates perceptual improvements.


3. **Clarity**

The paper is well-organized and clearly written, despite the technical density. Visualizations, e.g., Figures 2–4, effectively illustrate instability sources and PCPO’s corrective mechanism. Mathematical derivations are complete and logically consistent.

### Weaknesses
1. **Theoretical Intuition.**

While the proportionality argument is sound, the paper could better connect it to established temporal credit assignment or variance reduction principles in RL.


2. **Lack of baselines.**

It is encouraged to compare with other SOTA methods. See Q3.

While results on SD1.4/1.5 and FLUX are compelling, broader testing, e.g., on SDXL (Podell et al., 2023), is recommended to establish generality.

### Questions
1. **Computation vs. Speedup.** It reports per-epoch acceleration, but omits detailed wall-clock comparisons, leaving open questions about added per-step computation.

2. **Variance Quantification.**
Beyond the clipping fraction, could the paper include explicit variance or gradient norm distributions to quantify the stability gains?

3. **Baselines.** More baselines are encouraged to compare with, such as SPPO [1] and RainbowPA [2]. On the other hand, can PCPO be integrated with reward-model-free methods (e.g., self-play [3]) to further stabilize alignment training?

4. $\log \rho_{t} \approx \rho_{t} - 1$ only holds when $\rho_{t}$ is about $1$. This may fail in high-variance regions or off-policy updates, especially when sampling trajectories from older policies or noisy prompts. Could you please show the distribution of $\rho_{t}$?


[1] Bridging SFT and DPO for Diffusion Model Alignment with Self-Sampling Preference Optimization. arXiv:2410.05255, 2025.

[2] Diffusion-RainbowPA: Improvements Integrated Preference Alignment for Diffusion-based Text-to-Image Generation. Transactions on Machine Learning Research, 2025.

[3] Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation. NeurIPS 2024.

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
3

### Summary
This paper proposes PCPO (Proportionate Credit Preference Optimization), a new framework to address the credit assignment problem in diffusion model preference optimization. While previous methods (e.g., DPO, LPO, TAPO) apply uniform or stepwise weighting across diffusion steps, they fail to proportionally assign reward signals according to the relative contribution of each latent step to the final image quality.
PCPO introduces a proportionate credit weighting mechanism that computes latent-wise contributions via gradient-based sensitivity analysis, enabling consistent and efficient optimization across diffusion trajectories. The authors provide a theoretical justification linking PCPO to policy gradient credit assignment and demonstrate its empirical advantage on text-to-image (SDXL, SD3.5) and text-to-video (ModelScope) benchmarks, improving both alignment and fidelity metrics.

### Strengths
1. Novel problem formulation: The paper clearly identifies the overlooked issue of disproportionate credit assignment in trajectory-based diffusion alignment.
2. Technically novelty: The gradient-based proportional credit estimation is simple yet effective, bridging reinforcement learning and diffusion optimization perspectives.
3. Strong empirical validation: Experiments on multiple backbones (SDXL, SD3.5, ModelScope) show consistent improvements over DPO, LPO, and TAPO.

### Weaknesses
1. Missing comparisons on metrics such as HPS, CLIP, Pick-a-Pic(Not reward), etc. FID and IS is not enough to evaluate the model quality.
2.  Sensitivity & hyperparameters: Sensitivity plots exist but more systematic sweeps (schedules, batch sizes, guidance scales, LoRA ranks, SDE noise levels) can strengthen claims.

### Questions
1. Add more metrics?
2. Add more ablations?

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
3

### Summary
This paper reveals that disproportionate credit assignment is  a key reason of instability when training T2I models using reinforcement learning. Proportionate Credit Policy Optimization (PCPO) is therefore introduced to deal with this problem. PCPO improve training stability of RL in training T2I models, leading to convergence acceleration and superior image quality.

### Strengths
1. This paper proposes Proportionate Credit Policy Optimization to deal with  training instability of RL for T2I fine-tuning. PCPO not only enhances numerical stability, but also ensures proportional credit assignment. The author also provide detailed theoretical derivation.

2. Compared to DanceGRPO, PCPO introduces  convergence acceleration and superior image quality.

### Weaknesses
1. It would be beneficial to include an introductory overview of diffusion models and GRPO in the preliminary section, as their absence hinders the paper's accessibility for researchers new to the field.

2. FID can not measure the human preference score of the generated images and text-image consistency, CLIP Score, ImageReward, and HPSv2  should be included in the  Tables  for comparison

3. The text condition for image generation in Figure 13 and 14 is too simple,  I would like to see more complex qualitative comparisons in Appendix G, and text condition should be given so that I can observe text-image alignment.

4. The experiment is not conducted on widely used datasets, e.g., MSCOCO-2014 30K validation  dataset, MSCOCO-2017 5K validation  dataset, MJHQ-30K dataset. It is benefical to convince me via reporting the results on above datasets.

### Questions
Table 1 employs only a single target level for each reward. It is important to investigate whether the reported conclusions remain consistent across different target levels. A discussion on the robustness and generalizability of the results with respect to this parameter would strengthen the paper.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper identifies disproportionate credit assignment across timesteps as a major cause of instability in the alignment for text-to-image models. 
It proposes Proportionate Credit Policy Optimization (PCPO), which reformulates the PPO objective into a numerically stable log-hinge view and then enforces proportional credit over the sampling trajectory. 
For diffusion models, PCPO makes per-step weights uniform by improving the variance schedule; for flow models, it restores proportionality by reweighting the training loss by each step’s integration interval, avoiding invasive sampler changes.

Across DDPO and DanceGRPO, PCPO reduces clipping and variance, speeds convergence, and improves fidelity, as indicated by lower FIDs, while mitigating collapse. 
A small human-preference study favors PCPO over strong baselines. Overall, the method offers a simple fix that stabilizes training and yields samples of better fidelity at similar reward levels.

### Strengths
- Clarity. This paper is well written in terms of motivation and technical content. The paper clearly states the core problem, training instability in T2I alignment due to disproportionate credit across timesteps. The motivation and scope easy to follow. Figures helps understand the volatile and modified weighting clearly. 
- Technical contributions. Given the replace of $\rho_t-1$ with $\log \rho_t$, the motivation is clear. This work further provides fix to the weighting under time shift for diffusion and flow models.
- Improvements. PCPO offers convergence acceleration to the target reward levels from 24.6% to 41.2%, while keeping better fidelity, as indicated by FID scores.

### Weaknesses
- Metrics. FID provides a reasonable measurement of the shift from the base distribution, though it is less accurate for images of higher resolutions. It would be better to have modern metrics such as FD-DINOv2 for fidelity evaluation. 
- Regarding IS, it would be more scientific to claim that given similar FID levels, a higher IS indicates better quality; or given a similar IS score, a lower FID indicates less mode collapse.

To be clear, this paper is of good quality regarding a detailed question in the alignment of diffusion and flow models, and meets the bar of ICLR. The following questions and suggestions could improve the quality. 

Suggestions:
- Plot per‑timestep gradient contributions and clipping rates before/after PCPO; report variance reduction in those contributions.
- Quantify the Taylor error during training with a time‑series plot; show whether the stability gain comes primarily from the proportionality fix or from the log‑hinge.

Typos:
- model collapse -> mode collapse

### Questions
- Could you please report the $E[\|g_t\|]$, $Var(\|g_t\|)$, and clip rate w.r.t. t for basline and PCPO? Having two visualizations/reports in the early stage, and near convergence, would be better.
- Why does it cause a negative effect when over-weighting the higher noise region?

### Soundness
3

### Presentation
3

### Contribution
2
