# SIPO: Stabilized and Improved Preference Optimization for Aligning Diffusion Models

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Preference learning has garnered extensive attention as an effective technique for aligning diffusion models with human preferences in visual generation tasks. However, existing alignment approaches such as Diffusion-DPO suffer from two fundamental challenges: training instability caused by high gradient variances at various timesteps and high parameter sensitivities, and off-policy bias arising from the discrepancy between the optimization data and the policy model's distribution. Our first contribution is a systematical analysis of the diffusion trajectories across different timesteps and identify that the instability primarily originates from early timesteps with low importance weights. To address these issues, we propose SIPO, a Stabilized and Improved preference Optimization framework for aligning diffusion models with human preferences. Concretely, a key gradient, \emph{i.e.,} DPO-C\&M is introduced to facilitate stabilize training by clipping and masking uninformative timesteps. Followed by a timestep aware importance re-weighting paradigm to fully correct off-policy bias and emphasize informative updates throughout the alignment process. Extensive experiments on various baseline models, including image generation models on SD1.5, SDXL, and video generation models CogVideoX-2B, CogVideoX-5B, and Wan2.1-1.3B, demonstrate that our SIPO consistently promotes stabilized training and outperforms existing alignment methods, with meticulous adjustments on parameters.
Overall, these results highlight the importance of timestep-aware alignment and and provide valuable guidelines for improved preference optimization in diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Stabilized and Improved Preference Optimization (SIPO), a framework designed to address two fundamental challenges in applying Direct Preference Optimization (DPO) to diffusion models: training instability and off-policy bias. The authors first conduct a systematic analysis of the diffusion process, identifying that training instability is primarily caused by high gradient variances originating from early time steps that have low importance weights.

### Strengths
**Quality**

1. Stability and Robustness.
The method demonstrates high training quality by providing a smoother, more stable training loss curve (Figure 1b) and consistently improving test accuracy without the late-stage degradation (reward hacking) seen in Diffusion-DPO (Figure 1c, 8a, 8b). Crucially, SIPO shows remarkable robustness to the $\beta$ hyperparameter compared to the high sensitivity of Diffusion-DPO (Figure 1a)

2. Extensive Benchmarking.
The experiments are high-quality and comprehensive, validating SIPO across a diverse set of large-scale models and tasks, including both popular image generators (SDXL) and challenging video generation models (CogVideoX, Wan2.1-1.3B, FLUX-dev).


**Clarity**

1. Clear Problem Framing. The paper is clear in framing the work, articulating the two core challenges (instability and off-policy bias) and immediately tying them to the need for timestep-aware alignment.

2. Logical Flow. The introduction of the method is logically motivated by the preceding analysis on the reward dynamics and importance weights across different time steps (Figure 2 and 4), providing a foundation for the technical solution.

### Weaknesses
1. Missing Analysis on Late-Stage Instability.

The paper states that "early and late stages introduce instability" and that "preference signals are more informative in middle-to-late timesteps". However, the core analysis focuses heavily on early timesteps being problematic due to low importance weights (Figure 4, low weight up to $t \approx 63$). The paper is missing an explicit, corresponding analysis of why the very late timesteps, e.g., $t>900$ might also be unstable or uninformative, and how SIPO's mechanisms specifically address this tail-end instability.


2. Baselines. The baselines are not sufficient. More SOTA baselines are encouraged to compare with, such as SPPO [1] and RainbowPA [2].

[1] Bridging SFT and DPO for Diffusion Model Alignment with Self-Sampling Preference Optimization. arXiv:2410.05255, 2025.

[2] Diffusion-RainbowPA: Improvements Integrated Preference Alignment for Diffusion-based Text-to-Image Generation. Transactions on Machine Learning Research, 2025.

### Questions
1. Baselines. See W2.

2. Detailed Video Metrics.

The work explicitly addresses video generation challenges like maintaining temporal coherence. How about a breakdown of VBench results using metrics that specifically quantify temporal quality (e.g., motion smoothness, temporal consistency) in the main results section. This would provide stronger evidence for the successful alignment of video models compared to the general video quality metrics currently presented.

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
This paper addresses the instability observed in applying Direct Preference Optimization (DPO) to diffusion models. The authors propose two complementary improvements:
(1) DPO-C&M, which introduces timestep-dependent importance masking and gradient clipping to mitigate gradient explosion and overemphasis on uninformative steps; and
(2) SIPO, which modifies the DPO objective by applying clipped importance reweighting to the log-likelihood ratio term and reformulates the loss as KL minimization toward a reward-shaped target distribution. By skipping early diffusion steps and leveraging importance sampling, SIPO aims to improve training stability and convergence behavior.

### Strengths
oth modifications (C&M and SIPO) are lightweight yet principled, and can be easily integrated into existing preference-optimized diffusion frameworks.
The experiments demonstrate noticeable stability improvements across image and video generation tasks, suggesting the methods’ robustness in practice.

### Weaknesses
1. Inconsistent β values across methods. Table A1 shows that β differs between baselines and proposed methods (e.g., for video: DPO=2 vs. SIPO/DPO-C&M=0.02; for image: DPO=5 vs. SIPO/DPO-C&M=1). Since β directly controls the conservative/aggressive trade-off in DPO, this discrepancy could systematically favor the proposed variants. The authors should present comparisons under matched β settings or include grid-based sensitivity analyses with statistical significance.

2. Lack of human evaluation details. More information about the human annotation setup is needed—number of raters, cleaning or filtering criteria, and aggregation methods. Releasing minimal anonymized examples would further enhance reproducibility and transparency.

### Questions
see the weakness

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
The paper proposes SIPO (Stabilized and Improved Preference Optimization) for aligning diffusion models with human (or AI) preferences. The key ideas are: (1) identify that early timesteps in diffusion contribute high-variance, low-importance gradients; (2) introduce DPO-C&M (clipping & masking) using timestep-wise importance weights to stabilize training; and (3) further correct off-policy bias via importance-weighted DPO with clipped, timestep-aware weights. Experiments on SD1.5/SDXL for T2I and CogVideoX/Wan for T2V claim improved stability and accuracy over Diffusion-DPO and other baselines, with lower sensitivity to β and better human evals.

### Strengths
1. Clear empirical diagnosis of instability: The paper argues and shows that early timesteps have low importance weights and introduce noisy gradients; masking/clipping these improves stability.
2. Principled off-policy correction: Casting diffusion DPO with importance sampling and clipping is well-motivated by RL literature; the step-wise treatment is natural for diffusion.
3. Breadth of evaluation: Benchmarks across SD1.5/SDXL (T2I) and CogVideoX/Wan (T2V) with automatic metrics and human ranking; reports reduced β-sensitivity and smoother learning curves.

### Weaknesses
1. In line 361, you state “at 1000 steps, Diffusion-DPO collapses (67.28)” whereas Table 1 shows pretrained = 67.28 and Diffusion-DPO = 81.46. This contradiction undermines the claimed failure of Diffusion-DPO at longer training. Please reconcile.

2. Missing ablations: Choices like threshold 0.9 for pruning, the clipping range [1−ε,1+ε], and their per-dataset sensitivity lack rigorous ablations; Importance estimation per-timestep may add overhead; wall-clock comparisons to Diffusion-DPO/SPIN etc. are not reported.

### Questions
1. Please clarify the results in Table 1.

2. Please add more ablations such as sensitivity and justification for the 0.9 threshold and ε range across datasets.

### Soundness
3

### Presentation
3

### Contribution
3
