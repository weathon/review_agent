# Enhanced DACER Algorithm with High Diffusion Efficiency

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Due to their expressive capacity, diffusion models have shown great promise in offline RL and imitation learning. Diffusion Actor-Critic with Entropy Regulator (DACER) extended this capability to online RL by using the reverse diffusion process as a policy approximator, achieving state-of-the-art performance. However, it still suffers from a core trade-off: more diffusion steps ensure high performance but reduce efficiency, while fewer steps degrade performance. This remains a major bottleneck for deploying diffusion policies in real-time online RL. To mitigate this, we propose DACERv2, which leverages a Q-gradient field objective with respect to action as an auxiliary optimization target to guide the denoising process at each diffusion step, thereby introducing intermediate supervisory signals that enhance the efficiency of single-step diffusion. Additionally, we observe that the independence of the Q-gradient field from the diffusion time step is inconsistent with the characteristics of the diffusion process. To address this issue, a temporal weighting mechanism is introduced, allowing the model to effectively eliminate large-scale noise during the early stages and refine its outputs in the later stages. Experimental results on OpenAI Gym benchmarks and multimodal tasks demonstrate that, compared with classical and diffusion-based online RL algorithms, DACERv2 achieves higher performance in most complex control environments with only **five diffusion steps** and shows greater multimodality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes DACERv2, an enhanced version of the diffusion-based online reinforcement learning algorithm, DACER. The authors identify a key limitation in DACER: a trade-off between performance and efficiency, where a large number of diffusion steps is required for high performance, leading to high computational cost.
To address this, DACERv2 introduces two main contributions:
1. **A Q-Gradient Field Objective ($\mathcal{L}_g$)**: This is an auxiliary loss term that provides intermediate supervision at each step $t$ of the denoising process. It is motivated by the connection between the optimal soft-Q policy (a Boltzmann distribution) and the score function via Langevin dynamics, aiming to align the policy's score function $S_{\theta}(s, a_t, t)$ with the normalized Q-gradient $\nabla_{a_t} Q(s, a_t)$.
2. **A Temporal Weighting Mechanism ($w(t)$)**: This mechanism modulates the strength of the Q-gradient objective based on the diffusion timestep $t$. This is designed to resolve the inconsistency between the time-independent Q-gradient field and the time-dependent nature of the diffusion denoising process.
The authors claim that this new combined objective ($\mathcal{L}_\pi = \mathcal{L}_q + \eta \mathcal{L}_g$) allows DACERv2 to achieve state-of-the-art (SOTA) performance on complex MuJoCo benchmarks using only 5 diffusion steps. This results in significant improvements in both training and inference efficiency.

### Strengths
1. **Clear and Significant Problem**: The paper addresses a critical, practical limitation of diffusion policies—their poor computational efficiency due to the high number of sampling steps.
2. **Strong Empirical Results**: The primary claim of achieving SOTA performance with only $T=5$ steps is backed by comprehensive experiments. The efficiency gains shown in Table 1 are dramatic and highly compelling (e.g., >3.5x faster inference than DACER). The ablations in Figure 4 clearly isolate the impact of the two key contributions ($\mathcal{L}_g$ and $w(t)$), empirically validating their necessity for the observed performance.

### Weaknesses
1. **Lack of Theoretical Novelty and Justification**: The paper's theoretical support is weak on two fronts.
    - **Heuristic Contribution**: The paper's primary novel contribution, the $\mathcal{L}_g$ auxiliary loss, is a pure heuristic. It is motivated by analogy but lacks any formal proof or analysis showing that the combined objective ($\mathcal{L}_\pi = \mathcal{L}_q + \eta \mathcal{L}_g$) leads to a better, faster, or more stable convergence to the optimal policy.
    - **Non-Novel Theorem**: The main theoretical result presented (Theorem 1 in Appendix A) appears to be a restatement of a standard, known result (i.e., that maximizing value under a global entropy constraint yields a Boltzmann policy with a global temperature). This theorem only justifies the baseline DACER objective and does not represent a novel contribution of this work.
2. **Potentially High Hyperparameter Sensitivity ($\eta$)**: The hyperparameter $\eta$ (the auxiliary loss weight) is clustered into two groups (Table 5), but the values are $1.0$ and $0.01$—a 100-fold difference. This implies that the algorithm's performance is highly sensitive to this choice. The paper offers no insight into what task properties (e.g., dimensionality) necessitate such a drastic change, which is crucial for applying this method to new environments.

### Questions
1. Regarding the hyperparameter $\eta$ in Table 5: The optimal value differs by 100x ($1.0$ vs. $0.01$) across tasks. What properties of the environment (e.g., dimensionality, task complexity) dictate this choice? How sensitive is the algorithm to this parameter?
2. The paper's main theoretical support, Theorem 1, appears to be a standard result for justifying the soft-Q objective. Given this, can the authors provide any novel theoretical analysis for their actual contribution, $\mathcal{L}_g$? For instance, can it be shown that $\mathcal{L}_g$ acts as a variance reduction term, or that the combined objective $\mathcal{L}_\pi$ has superior convergence properties compared to optimizing $\mathcal{L}_q$ alone?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces DACERv2, an improved version of DACER that learns a Q-gradient field for fast action denoising. Observing that the denoising process in diffusion relies on a time-dependent score function, DACER-v2 also introduces such dependency by scaling the Q-gradient with a crafted exponential decay function. Finally, DACER-v2 is evaluated on tasks from Gym-MuJoCo and demonstrates improved performance as compared to several diffusion policy baselines.

### Strengths
The overall idea is clearly presented and straightforward to implement. The proposed method is efficient both in terms of training and inference, which makes it preferable for deployment in embodied scenarios.

### Weaknesses
The idea of aligning the score networks with the gradient of Q-value functions has been extensively investigated in QSM [1], DAC [2], QGPO [3], iDEM [4], and [5]. One contribution of DACER-v2 seems to be the time-based weighting. However, this is purely heuristic and theoretically unjustified. On the other hand, QGPO, iDEM, and [5] also estimate the time-dependent score, and their estimations are exact in theory. Therefore, the novelty and insight of this paper are limited. 

Besides, this paper lacks a related work section to familiarize the readers with the frontier literature. For example, given that the proposed algorithm is termed DACER-v2, it is necessary to include detailed introductions about DACER and demonstrate how v2 improves the v1 algorithm. 

[1] Learning a Diffusion Model Policy from Rewards via Q-Score Matching.

[2] Diffusion Actor-Critic: Formulating Constrained Policy Iteration as Diffusion Noise Regression for Offline Reinforcement Learning. 

[3]: Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning. 

[4]: Iterated Denoising Energy Matching for Sampling from Boltzmann Densities. 

[5]: Sampling from Energy-based Policies using Diffusion.

### Questions
How many environment frames/steps does one iteration correspond to? 

The authors mentioned that the Q-gradient prediction is an auxiliary objective (line 208). However, I don’t see any further introduction about the actual objective of the diffusion policy in the paper. Could the authors make this clear?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DACERv2, an online RL algorithm built upon DACER and employs a new Q-gradient field matching objective in the policy learning loss. The proposed method enables action sampling with five diffusion steps and outperforms the included baselines on OpenAI Gym environments. The authors also introduce a temporal weighting function that adjusts gradient magnitude across diffusion timesteps.

### Strengths
1. The proposed algorithm achieves strong performance on state-based OpenAI Gym environments, with higher training and inference efficiency than most baselines.
2. The paper is well-written.

### Weaknesses
1. The score function in a standard diffusion SDE is the score function of the perturbed distribution $\int q_{t|0}(a_t|a_0) \frac{e^{\frac{1}{\alpha}Q(s, a_0)}}{Z(s)}da_0$ and is not in the form of Equation (9). Moreover, the non-annealed Langevin dynamics used in this paper may suffer from slow mixing, as shown in [1].
2. The method proposed in this paper is a straightforward combination of the QSM [2] policy training loss (with a newly introduced weighting function) and the DACER policy training loss, and the analysis is insufficient to explain why this combination boosts performance without gradient-conflict issues.
3. The argument in Lines 698-699 is not well supported. If the optimal action with the largest Q value is a single point, then maximizing the Q-value will result in a delta distribution, not a multimodal policy. The multimodal property is more likely due to the entropy regularization and the expressive capacity of diffusion models. 

[1] Song Y, Ermon S. Generative modeling by estimating gradients of the data distribution[J]. Advances in neural information processing systems, 2019, 32.

[2] Psenka, Michael, et al. Learning a diffusion model policy from rewards via Q-score matching. Proceedings of the 41st International Conference on Machine Learning. 2024.

### Questions
1. In the training curves in Figure 3, why does DACERv2 improve more slowly than most baselines in the early stage, especially on Ant-v3, HalfCheetah-v3, Walker2d-v3, and Swimmer-v3?
2. The argument in Lines 321-323 is not logically supported. Why does the auxiliary intermediate supervisory signal enable action sampling with fewer diffusion steps? A similar Q-gradient in QSM still requires 20 sampling steps.

If the authors can address the concerns above, I would be willing to increase the overall score.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces the **DACER v2** algorithm, an improved version of **DACER**, which incorporates a **Q-gradient field** and a **temporal weighting mechanism**. Experiments on **OpenAI Gym** benchmarks and **multi-modal tasks** demonstrate that **DACER v2** achieves superior performance and stronger multi-modality using only **five diffusion denoising steps**, outperforming other online diffusion RL algorithms.

### Strengths
1. The paper addresses a highly important and timely research question, especially as diffusion models are becoming increasingly dominant in the fields of **imitation learning**, **reinforcement learning**, and **Vision-Language-Action (VLA)** modeling.

2. The paper is **well-written** and **easy to follow**, presenting its ideas and methodologies clearly.

3. The **DACER v2** algorithm demonstrates **strong performance** compared to other **online diffusion RL** methods.

### Weaknesses
### Major Weaknesses:

1. The authors claim that the **DACER v2** algorithm focuses on improving the diffusion efficiency of the original **DACER**. Accordingly, one would expect **DACER v2** to achieve comparable performance with fewer diffusion denoising steps compared to **DACER** using the full number of steps. However, the experimental results show that **DACER v2** not only maintains efficiency but also exhibits **stronger multi-modality** and **better sample efficiency** with fewer denoising steps. The authors are encouraged to explain the source of this additional performance gain in more detail.

2. The main experimental figure (**Figure 3**) presents total average return plotted against **iterations**. Could the authors clarify what these iterations represent? Are they equivalent to **environment timesteps**? If not, please explain the rationale behind using this setting and consider including additional plots showing **total average return versus environment timesteps** for clearer comparison.

3. In **Section 4.2**, the authors claim that *“in real-time industrial control tasks, the inference time should be less than 1 millisecond to meet control requirements.”* It would be helpful to provide additional evidence or references to substantiate this statement. From my perspective, an inference time of **1.6 ms** (as achieved by the original **DACER** algorithm) already appears sufficient for most robotic control tasks, and further reducing it to **0.6 ms** may offer only marginal benefits. Since this point directly relates to the **motivation of the paper**, a clearer justification would strengthen the argument.

4. There exists a wide range of **diffusion acceleration methods**, such as **DDIM** and **Consistency Models**. In the introduction, the authors claim that these acceleration techniques trade performance for efficiency. It is highly recommended that the authors evaluate **DACER** combined with a diffusion acceleration method during inference and compare the results with **DACER v2**, to more clearly demonstrate the advantages of the proposed approach.

5. The authors are encouraged to include a discussion or experimental comparison with **Diffusion Policy Policy Optimization (DPPO)**, as it represents a closely related and widely used approach in online diffusion RL.


### Minor Weaknesses:

1. The authors are recommended to discuss some highly related works:

**Diffusion Acceleration:**

Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising diffusion implicit models." arXiv preprint arXiv:2010.02502 (2020).

Song, Yang, et al. "Consistency models." (2023).

**Online RL with Diffusion Policy:**

Yuan, Xiu, et al. "Policy decorator: Model-agnostic online refinement for large policy model." arXiv preprint arXiv:2412.13630 (2024).

Ankile, Lars, et al. "From imitation to refinement-residual rl for precise assembly." 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025.

Wagenmaker, Andrew, et al. "Steering Your Diffusion Policy with Latent Space Reinforcement Learning." arXiv preprint arXiv:2506.15799 (2025).

**Offline RL with Diffusion Policy:**

Hansen-Estruch, Philippe, et al. "Idql: Implicit q-learning as an actor-critic method with diffusion policies." arXiv preprint arXiv:2304.10573 (2023).

Park, Seohong, Qiyang Li, and Sergey Levine. "Flow q-learning." arXiv preprint arXiv:2502.02538 (2025).

**I am more than willing to raise my scores if the authors adequately address my concerns**

### Questions
1. (Related to Major Weakness 1) Where do the observed improvements in **multi-modality** and **sample efficiency** originate from? Is **DACER v2** a strictly superior algorithm compared to the original **DACER** ?

2. (Related to Major Weakness 2) What do the **iterations** in **Figure 3** represent?

3. (Related to Major Weakness 3) Why do the authors believe that an inference time of **less than 1 millisecond** is required to meet **real-time industrial control** requirements?

### Soundness
2

### Presentation
2

### Contribution
2
