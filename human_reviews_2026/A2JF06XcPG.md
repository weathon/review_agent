# Fine-tuning Diffusion Policies with Backpropagation Through Diffusion Timesteps

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Diffusion policies, widely adopted in decision-making scenarios such as robotics, gaming and autonomous driving, are capable of learning diverse skills from demonstration data due to their high representation power. However, the sub-optimal and limited coverage of demonstration data could lead to diffusion policies that generate sub-optimal trajectories and even catastrophic failures. While reinforcement learning (RL)-based fine-tuning has emerged as a promising solution to address these limitations, existing approaches struggle to effectively adapt Proximal Policy Optimization (PPO) to diffusion models. This challenge stems from the computational intractability of action likelihood estimation during the denoising process, which leads to complicated optimization objectives. In our experiments starting from randomly initialized policies, we find that online tuning of Diffusion Policies demonstrates much lower sample efficiency compared to directly applying PPO on MLP policies (MLP+PPO). To address these challenges, we introduce NCDPO, a novel framework that reformulates Diffusion Policy as a noise-conditioned deterministic policy. By treating each denoising step as a differentiable transformation conditioned on pre-sampled noise, NCDPO enables tractable likelihood evaluation and gradient backpropagation through all diffusion timesteps. This formulation enables direct optimization over the final denoised interactive actions without increasing MDP lengths. Our experiments demonstrate that NCDPO achieves sample efficiency comparable to MLP+PPO when training from scratch, outperforming existing methods in both sample efficiency and final performance across diverse benchmarks, including continuous robot control (with both state and vision inputs) and multi-agent coordination tasks. Furthermore, our experimental results show that our method is robust to the number denoising timesteps.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Noise-Conditioned Diffusion Policy Optimization (NCDPO), which treats the entire diffusion denoising process as a deterministic, noise-conditioned transformation. NCDPO reframes diffusion-policy fine-tuning as a differentiable stochastic control problem. Traditional DPPO optimizes each denoising step as if it were an RL sub-policy, which is inefficient and redundant. NCDPO instead treats diffusion as a deterministic function of noise, enabling backpropagation through time without explicitly expanding the MDP. It merges the advantages of diffusion’s expressivity and PPO’s stability, while removing redundant low-level decision horizons.

### Strengths
- It introduces backpropagation through diffusion timesteps with fixed noise conditioning, which was previously intractable. It also avoids multi-level MDP unrolling, and a shorter effective horizon improves both convergence speed and stability.
- It works across both continuous and discrete action spaces, and scales from simple locomotion to high-dimensional vision inputs.
- All derivations align with DDPM-style parameterizations and PPO gradient logic, demonstrating how a generative diffusion model can be differentiably integrated into RL objectives, without approximations or surrogate networks.

### Weaknesses
- The noise-fixing assumption makes the entire process differentiable but reduces stochastic exploration. It could lead to limited diversity in gradient updates and overfitting to specific noise realizations.
- While the chain rule derivation is correct, the paper lacks a formal exposition of why this yields the same fixed-point as PPO’s stochastic gradient.
- It remains unclear whether backpropagating through deterministic denoising leads to unbiased gradient estimates compared to true stochastic diffusion.

### Questions
- The conceptual connection between expectation over fixed noise and policy stochasticity could be clarified.
- Is the value function $V_{\phi}(s)$ trained marginalizing over the diffusion noise or conditioned on the fixed noise sequence $z_{1:K}$?
- Empirical success is clear, but there’s no formal convergence analysis or theoretical proof that BPDT guarantees unbiased gradients.
- Memory overhead requires caching all sampled noises and intermediate activations to compute BPDT gradients, which increases memory footprint and may limit scalability for very large diffusion horizons or long-horizon tasks. Could you add a discussion on it?

### Soundness
3

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
4

### Summary
The core contribution of this work is a reformulation of diffusion timesteps as noise-conditioned deterministic transformations, which unlocks efficient gradient computation and full backpropagation through the entire sampling process. This approach achieves sample efficiency comparable to established MLP-PPO methods while demonstrating significantly enhanced robustness and performance, as established through comprehensive evaluations in diverse RL settings such as robotic control and multi-agent games.

### Strengths
- Broad, diverse evaluation across continuous-control locomotion, multi-agent discrete environments, and manipulation (Robomimic), demonstrating robustness and applicability across domains
- Consistent gains across datasets of varying quality (medium-replay vs medium-expert), suggesting stability to data quality and strong offline-to-online transfer

### Weaknesses
- **Exploration and stability analysis, along with real-world validation, remain underdeveloped.** While DPPO is included as a key baseline, the study lacks visual diagnostics for exploration and stability—such as those presented in the original DPPO paper—and does not incorporate real-robot deployment to substantiate its practical applicability.

- **The multi-agent evaluation on Google Research Football is incomplete relative to the paper’s stated focus.** Given the core contribution revolves around RL optimization for diffusion policies, the GRF experiments should prioritize head-to-head comparisons against relevant diffusion-based baselines. Although several diffusion-RL methods are listed in the related work, they are not sufficiently included as comparators in the experiments.

- **Evidence for wall-clock training performance is incomplete.** While the paper claims strong performance and high sample efficiency, it does not provide  wall-clock learning curves, or time-to-threshold analyses, which are essential to rigorously support the efficiency and convergence claims.

### Questions
- Exploration/stability and hardware: Provide DPPO-like exploration and stability visualizations (state coverage, entropy, return variance); include at least one high-fidelity sim or real-robot deployment; report inference latency as denoising steps increase.
- Diffusion-based comparisons in GRF: Incorporate representative diffusion-RL baselines from the related work (e.g., diffusion+Q-learning, critic-guided denoising variants) under matched pretraining, data budgets, and compute to align evaluation with the paper’s stated focus

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
The paper presents the **NCDPO** algorithm, a novel diffusion policy fine-tuning framework that reformulates the **Diffusion Policy** as a **noise-conditioned deterministic policy**. NCDPO treats each denoising step as a **differential transformation** and fine-tunes the **fully denoised interactive actions**. Extensive experiments on **OpenAI Gym**, **Franka Kitchen**, **RoboMimic**, and **Google Research Football** demonstrate that **NCDPO** achieves superior performance compared to existing diffusion policy fine-tuning methods.

### Strengths
1. The paper tackles a **highly important and timely research question**, particularly as diffusion models are becoming increasingly influential in the domains of **imitation learning**, **reinforcement learning**, and **Vision-Language-Action (VLA)** modeling.

2. The paper is **well-written** and **clearly organized**, effectively presenting its motivation, methodology, and experimental results.

3. The proposed **NCDPO** algorithm demonstrates **strong performance** compared to other **Diffusion Policy online fine-tuning** methods.

### Weaknesses
### Major Weaknesses:

1. I am somewhat skeptical about the authors’ claim that the **low sample efficiency of DPPO** stems from a **lengthened MDP horizon**. More evidence is needed to substantiate this argument, as it is closely tied to the motivation of the paper. If a longer MDP horizon truly causes **DPPO** to be sample inefficient, then **fine-tuning DDIM** with fewer diffusion denoising steps, or fine-tuning only the final few denoising steps of DDPM, should yield sample efficiency improvements compared to fine-tuning all denoising steps with **DDPM**. Additionally, I find the experiments in **Section 4** somewhat unclear in purpose. The results appear self-evident, since (1) **DPPO** is not designed to train a randomly initialized Diffusion Policy from scratch, and (2) greater representational capacity does not necessarily translate to better performance, particularly in tasks where such capacity is not required.

2. The authors should clarify the **core contribution** of the proposed **NCDPO** algorithm. At present, it appears conceptually similar to directly fine-tuning a **Diffusion Policy** using **PPO** in a standard manner. It is generally acknowledged that **backpropagation through multiple denoising steps** can lead to unstable gradients. Therefore, the authors are encouraged to explain in greater detail how **reformulating Diffusion Policy as a deterministic policy** effectively mitigates this issue.

3. In the **Google Research Football** benchmark, the authors compare **NCDPO** with **MLP + MAPPO**. It is unclear what this comparison is intended to demonstrate. At a minimum, the authors should include **Diffusion Policy online fine-tuning methods** as baselines to provide a more meaningful and fair evaluation.

4. In several tasks, including **OpenAI Gym** and **Franka Kitchen**, **NCDPO** achieves higher final performance than **DPPO**. The authors are strongly encouraged to discuss why **NCDPO** not only improves **sample efficiency** but also enhances **final performance** in certain cases.

### Minor Weakness:

1. It is observed that **NCDPO** achieves significantly higher **sample efficiency** in the **RoboMimic Transport (visual)** setting compared to the **state-based** setting. The authors are encouraged to provide some discussion or analysis to explain this discrepancy.

2. The authors are encouraged to discuss recently proposed **Diffusion Policy fine-tuning methods** that do not require updating the parameters of the diffusion model itself. Including comparisons with such approaches would further strengthen the paper and is recommended for future work.

Wagenmaker, Andrew, et al. "Steering Your Diffusion Policy with Latent Space Reinforcement Learning." arXiv preprint arXiv:2506.15799 (2025).

Yuan, Xiu, et al. "Policy decorator: Model-agnostic online refinement for large policy model." arXiv preprint arXiv:2412.13630 (2024).

Ankile, Lars, et al. "From imitation to refinement-residual rl for precise assembly." 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025.

**I am more than willing to raise my scores if the authors adequately address my concerns.**

### Questions
1. **(Related to Major Weakness 1)** Why do the authors believe that a **lengthened MDP horizon** is the main cause of the **low sample efficiency** observed in **DPPO**? What are the key insights that the experiments in **Section 4** are intended to convey?

2. **(Related to Major Weakness 2)** What are the **main contributions and novelties** of this paper? How does **NCDPO** effectively mitigate the **unstable gradients** that arise from **backpropagation through multiple denoising steps**?

3. **(Related to Major Weakness 3)** What is the intended takeaway from the **Google Research Football** experiments? What do the authors aim to demonstrate with this comparison?

4. **(Related to Major Weakness 4)** Why does **NCDPO** not only improve **sample efficiency** but also achieve **higher final performance** compared to **DPPO**?

### Soundness
2

### Presentation
2

### Contribution
2
