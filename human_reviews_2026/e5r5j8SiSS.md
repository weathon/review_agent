# E-DDPG: Dual-Objective Deep Deterministic Policy Gradient for MRI Acceleration and Disease Classification

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Long acquisition times remain a major challenge in clinical MRI, where a fundamental trade-off exists between the acceleration achieved through undersampling and the diagnostic utility of the reconstructed images. We cast the problem of acquiring MRI data within a fixed time budget as a discrete reinforcement learning (RL) task and propose an algorithm based on Deep Deterministic Policy Gradient, referred to as E-DDPG. E-DDPG jointly optimizes sampling patterns, image reconstruction quality, and diagnostic accuracy. We introduces three key innovations: (1) a composite reward function that simultaneously encourages cross-entropy reduction, structural similarity improvement, and decrease in predictive entropy; (2) a percentile-based replay buffer that diversifies learning by equally sampling low- and high-value transitions; and (3) integration of the Straight-Through Gumbel-Softmax mechanism to preserve end-to-end differentiability while enabling discrete action selection. We evaluate E-DDPG against state-of-the-art RL-based methods and ablation variants on the fastMRI/fastMRI+ knee datasets at acceleration factors of 4X, 8X, and 10X, demonstrating its superior performance and validating the effectiveness of each proposed component.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the long-standing challenge of accelerating MRI acquisition without compromising diagnostic quality. The authors frame k-space undersampling as a discrete reinforcement learning (RL) task and propose E-DDPG (Enhanced Deep Deterministic Policy Gradient) — a dual-objective RL framework that jointly optimizes sampling efficiency, image reconstruction fidelity, and disease classification accuracy.

E-DDPG extends the traditional DDPG algorithm with three innovations:

- A composite reward function combining classification accuracy (cross-entropy reduction), reconstruction quality (SSIM), and diagnostic confidence (entropy reduction).
- A percentile-based replay buffer (PRB) that balances sampling between high- and low-reward experiences for stable training.
- A Straight-Through Gumbel-Softmax (STGS) estimator that enables differentiable discrete k-space column selection while maintaining gradient flow.

The method is evaluated on the fastMRI and fastMRI+ knee datasets under 4×, 8×, and 10× acceleration. Quantitative metrics (SSIM, PSNR, AUC, Balanced Accuracy) and qualitative results show consistent superiority of E-DDPG over PPO- and DQL-based approaches.

The paper offers engineering and application-level novelty, not theoretical innovation. However, the adaptation of DDPG with STGS and percentile replay for a dual-objective, discrete-action medical imaging task represents a meaningful and creative contribution.

### Strengths
- Casting MRI acquisition as a dual-objective reinforcement learning problem is conceptually compelling and addresses a clinically important trade-off: scan speed versus diagnostic accuracy.
- Each enhancement (composite reward, PRB, STGS) directly addresses specific limitations of standard DDPG — i.e., single-scalar rewards, uniform replay bias, and non-differentiable discrete actions. The proposed design is methodologically coherent and practically effective.
- Comprehensive experimental validation: benchmarks on both fastMRI and fastMRI+ datasets with 4×–10× acceleration factors; four ablation variants (removing STGS, PRB, or reward components) clearly isolate each contribution; comparisons to leading RL approaches under controlled settings ensure fairness.
- Equations are precise, architectural details are complete, and training hyperparameters are fully listed. Figures are well-labeled and readable. The authors mention public code release, which strengthens reproducibility.

### Weaknesses
- SSIM, PSNR, and AUC improve modestly, and the clinical impact (e.g., lesion visibility, diagnostic confidence in radiologists) is unquantified. The numerical gains, though consistent, may be marginal in practice.
- Experiments focus only on 2D knee MRIs and binary classification. No tests on other anatomies, contrasts, or 3D data are presented. It is unclear whether the learned policies generalize beyond this specific domain.
- The work omits comparisons with modern differentiable or supervised sampling optimization techniques (e.g., [1][2]). This leaves unclear whether RL offers a genuine advantage over deterministic gradient-based optimization.

References:
[1] https://arxiv.org/pdf/1901.01960
[2] https://arxiv.org/pdf/2210.12548

### Questions
- Why was a single-step decision process used instead of a sequential MDP that models progressive acquisition? How does this affect exploration and long-term reward learning? Would a multi-step policy (e.g., sequentially adding k-space lines) improve adaptivity or efficiency?

### Soundness
3

### Presentation
3

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
The paper proposes E-DDPG, a DDPG-based reinforcement learning framework for accelerated MRI. The goal is to jointly optimize image reconstruction and disease classification. Experiments on the fastMRI knee dataset at 4x, 8x, and 10x acceleration demonstrate the performance improvements over recent RL-based baselines.

### Strengths
- The paper is well-written, and the problem formulation is clear. The proposed modules are well-motivated and technically sound.

- The paper provides comparisons with existing RL-based baselines.

- The authors explicitly discuss the limitations of their approach.

- The visualization of selected k-space trajectories and the reconstructed images across methods helps understanding.

### Weaknesses
- It is unclear how critical the use of the PromptMR model is to overall performance. Does the proposed method rely heavily on this specific reconstruction backbone?

- The experiments are limited to knee MRI and a binary meniscus tear classification task. The work would be stronger if evaluated on more anatomies or pathologies.

- For the competing baseline solutions, a negative weighted cross-entropy term was added to the original rewards. It remains unclear how these baselines perform without this additional term.

- The sensitivity of the model performance to the reward coefficients is not analyzed.

### Questions
- How much does the proposed approach depend on the PromptMR backbone? Would using another reconstruction model, e.g., U-Net, VarNet.., substantially change performance?

- Could the proposed method generalize to other anatomies or pathologies?

- How sensitive is E-DDPG to the choice of reward weights?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles the clinical MRI time–quality trade-off by formulating k-space acquisition under a fixed scan-time budget as a discrete RL problem and proposing E-DDPG, a DDPG-based agent that jointly optimizes sampling patterns, reconstruction quality, and downstream diagnostic accuracy. The method introduces three main components: a composite reward that balances cross-entropy reduction, SSIM improvement, and lower predictive entropy; a percentile-based replay buffer that balances learning from low- and high-value transitions; and a straight-through Gumbel-Softmax scheme that permits discrete top-K action selection while keeping gradients end-to-end. Experiments on fastMRI/fastMRI+ knee data show superior performance over prior RL baselines.

### Strengths
1.The paper tackles a problem with clear clinical value by jointly optimizing sampling patterns, reconstruction quality, and downstream diagnostic accuracy.
2. This paper adapts DDPG to MRI acquisition with a clean one-step formulation (“directly select K lines”), uses Straight-Through Gumbel-Softmax to handle the non-differentiable top-K action, and employs a percentile-based replay strategy to stabilize off-policy learning.
3. On fastMRI/fastMRI+ knee data, the approach improves both image quality metrics and downstream classification compared to strong RL baselines, supporting the practical promise of task-aware sampling.

### Weaknesses
1. The paper mainly ports a standard DDPG pipeline to MRI acquisition and combines it with known ingredients. This combination is more engineering-oriented and lacks practical innovation.
2. The reconstructor and classifier are frozen while the sampling policy changes. Different masks can interact with a fixed reconstructor’s generalization in unpredictable ways, so the reported quality may not reflect the true performance under the new sampling distribution. 
3. Choosing all K lines in a single step (non-sequential) foregoes the possibility of adapting later choices to earlier observations, which is central to active/greedy sampling and closer to how scanners could operate.
4. Experiments focus on a single anatomy and task (knee, binary classification). The method’s utility should be validated on additional anatomies and tasks (multi-class, detection/segmentation), as well as across datasets.
5. The action space is tailored to 1D phase-encode top-K selection. It does not naturally extend to 2D sampling patterns (e.g., random, variable-density, radial/spiral) or trajectory constraints. At minimum, the paper should discuss this limitation or provide preliminary extensions.

### Questions
1. Could you please either add a small sequential variant (choosing k lines per step) or at least provide a preliminary conceptual modification of your current framework to support sequential sampling?
2. Since the reconstructor and classifier are fixed, could a mismatch between the learned mask and the frozen models bias the results? Please clarify how you avoid this issue, or provide (i) results with retraining or brief fine-tuning under the learned mask, and (ii) robustness checks across different reconstructors and classifiers (e.g., UNet, ViT).
3. The action space targets 1D top-K columns. How does the approach extend to 2D Cartesian patterns or non-Cartesian trajectories (random, variable-density, radial/spiral) and scanner constraints? If not supported, please discuss this limitation.
4. Can you add at least one more anatomy/task or dataset and strengthen statistical evidence (e.g., p-values)?
5. How are ACS/central lines enforced in the action space, and how sensitive are results to ACS size? A brief clarification would help.

### Soundness
2

### Presentation
2

### Contribution
1
