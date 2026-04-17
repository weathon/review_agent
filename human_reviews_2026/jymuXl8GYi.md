# Reinforcing Diffusion Models by Direct Group Preference Optimization

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
While reinforcement learning methods such as Group Relative Preference Optimization (GRPO) have significantly enhanced Large Language Models, adapting them to diffusion models remains challenging. In particular, GRPO demands a stochastic policy, yet the most cost‑effective diffusion samplers are based on deterministic ODEs. Recent work addresses this issue by using inefficient SDE-based samplers to induce stochasticity, but this reliance on model-agnostic Gaussian noise leads to slow convergence. To resolve this conflict, we propose Direct Group Preference Optimization (DGPO), a new online RL algorithm that dispenses with the policy-gradient framework entirely. DGPO learns directly from group-level preferences, which utilize relative information of samples within groups. This design eliminates the need for inefficient stochastic policies, unlocking the use of efficient deterministic ODE samplers and faster training. Extensive results show that DGPO trains around 20 times faster than existing state-of-the-art methods and achieves superior performance on both in-domain and out-of-domain reward metrics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
- This paper proposes DGPO, a method that brings GRPO’s within-prompt, group-relative feedback into diffusion alignment, addressing the technical mismatch that has made GRPO difficult for diffusion models.
- The approach distills GRPO’s group information into a tractable objective inspired by diffusion-DPO’s density-ratio formulation, avoiding stochastic policy gradients and full-trajectory training.
- Empirically, DGPO achieves strong preference alignment and image quality compared to existing methods, while remaining computationally efficient.

Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Strengths
- The authors propose a novel online RL method for diffusion alignment that efficiently leverages rich group-level information.
- They reformulate the core idea of GRPO—relative, within-prompt group information—and make it applicable to diffusion models using a fast ODE sampler.
- Their DGPO achieves higher benchmark scores than GPT-4o and Flow-GRPO.
- Experiments show DGPO is roughly 15–20× faster than Flow-GRPO at comparable scores.
- They also examine the effectiveness of timestep clipping, ODE rollouts, and the online setting.

Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Weaknesses
- My concern is that the authors’ formulation relies on the Bradley–Terry model, which might alter core characteristics of the original GRPO strategy.
- The formulation explicitly partitions each within-prompt group into "good" and "bad" subsets—something the original GRPO did not do—which may change the objective’s behavior compared with the original approach (see my question).

Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Questions
- In Equation (17), the DGPO loss contains two separate summations over \(x_0 \in \mathcal{G}^+\) and \(x_0 \in \mathcal{G}^-\).  
  Can these be rewritten as a single summation, such as  
  $\sum_{x_0 \in \mathcal{G}} A(x_0) (L^{\theta}-L^{\theta_{\mathrm{ref}}})$,  
  by using Equation (14)?

- How would you incorporate PPO-style clipping into DGPO? The original GRPO introduced PPO clipping [1].

Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

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
4

### Summary
The paper presents DGPO, a clear, diffusion-native alternative to GRPO that removes the stochastic-policy requirement while exploiting group-relative preferences.

### Strengths
1. The proposed DGPO achieves significant training efficiency improvements and superior performance on benchmark metrics.
2. DGPO integrates the idea of group advantage estimation from GRPO into DPO, thereby avoiding the need for stochastic-policy and eliminating the intractable partition function through an advantage-based weight design.

### Weaknesses
1. The mathematical transition from the intermediate objective $\mathcal{L}(\theta)$ (Eq. 15) to the final DGPO training objective $\mathcal{L}_{DGPO}(\theta)$ (Eq. 17) is not explicitly detailed, as the steps involving Jensen's inequality and the convexity of $-\log \sigma$ are only summarized. Try to clarify it.
2. The definition of the term $\log \frac{p _ {\theta}(x _ {t-1}|x _ t,c)}{p _ {ref}(x _ {t-1}|x _ t,c)}$ in $\mathcal{L} _ {DGPO}(\theta)$ is related to the difference in the DSM losses, $L_{dsm}^{\theta}-L_{dsm}^{\theta_{ref}}$ (Eq. 17), but the exact derivation of this equivalence is omitted.
3. The hyperparameter $\beta$ is a critical part of the loss function (Eq. 15, Eq. 17), but its specific value for the main experiments is not provided in the Setup Details, and no ablation study is conducted on its effect, making it difficult to assess the stability and sensitivity of the method to this parameter.
4. The choice of $G=24$ samples per group is mentioned, but the impact of the group size $G$ on the training speed, final performance, and the robustness of the advantage normalization is not investigated.
5. DiffusionNFT [1] also addresses the same problem of Flow-GRPO and has publicly released its source code. It would be beneficial to include a comparison with this method and incorporate the evaluation metrics mentioned in their work.
6. Add related works on diffusion models integrated with online off-policy reinforcement learning, such as QSM [2], DACER [3], and DIME [4], to provide a more comprehensive discussion of recent advancements in this area.

[1] Zheng, Kaiwen, et al. "Diffusionnft: Online diffusion reinforcement with forward process." arXiv preprint arXiv:2509.16117 (2025).

[2] Psenka, Michael, et al. "Learning a diffusion model policy from rewards via q-score matching." arXiv preprint arXiv:2312.11752 (2023).

[3] Wang Y, Wang L, Jiang Y, et al. Diffusion actor-critic with entropy regulator[J]. Advances in Neural Information Processing Systems, 2024, 37: 54183-54204.

[4] Celik, Onur, et al. "Dime: Diffusion-based maximum entropy reinforcement learning." arXiv preprint arXiv:2502.02316 (2025).

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents an online reinforcement learning method (DGPO), by combining GRPO and DPO. It utilizes the fine-grained preference information from GRPO, but uses ODE-based samplers for efficient training. Once preference groups are constructed, DGPO directly learns from them. Experiments show that DGPO is 20x faster in training, compared with existing methods, and can achieve better performance.

### Strengths
- DGPO utilizes group-level preference information.
- DGPO uses ODE-based samplers and therefore trains faster.
- DGPO outperforms Flow-GRPO.

### Weaknesses
N/A

### Questions
- How to choose group size, and how does it affect the performance?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Direct Group Preference Optimization (DGPO), a reinforcement learning algorithm to fine-tune diffusion models more efficiently. Current methods are slow because they require inefficient stochastic samplers. DGPO avoids this requirement by learning directly from group-level preferences, which allows the use of efficient deterministic ODE samplers. As a result, DGPO is shown to train up to 30 times faster than existing methods like Flow-GRPO while achieving superior performance on compositional generation, text rendering, and human preference alignment tasks.

### Strengths
- Significant Efficiency Gains. The most compelling advantage of DGPO is its dramatic improvement in training efficiency. The claim of being ~20-30 times faster than Flow-GRPO is well-supported by the experiments shown in Figures 1 and 3. This is a substantial practical contribution that could make reinforcement learning-based fine-tuning of diffusion models much more accessible.
- Strong Empirical Performance. DGPO not only trains faster but also achieves state-of-the-art results. It surpasses the performance of the baseline SD3.5-M model and the strong Flow-GRPO competitor on the challenging GenEval benchmark, boosting the score from 63% to 97% (Table 1). The method also shows consistent improvements across visual text rendering and human preference alignment tasks (Table 2).
- Well-Motivated. The paper clearly identifies a key limitation in applying GRPO-style methods to diffusion models (the reliance on inefficient stochastic SDE samplers).
- Comprehensive Experimental Evaluation. The authors provide a thorough evaluation, including qualitative and quantitative comparisons on multiple benchmarks, ablation studies that validate key design choices like the "Timestep Clip Strategy" and the use of ODE rollouts over SDE rollouts, and evaluation on a range of out-of-domain metrics to check for reward hacking.

### Weaknesses
- Incremental Novelty. The core idea can be seen as a combination of two existing lines of work: Direct Preference Optimization (DPO) and Group Relative Preference Optimization (GRPO). The main novelty lies in successfully extending DPO to a group-wise setting for diffusion models, but it does not introduce a fundamentally new optimization paradigm. The resulting objective still relies on approximations like Jensen's inequality, making it less theoretically principled and elegant.
- Limited Discussion on Reward Quality. The success of any RL-based fine-tuning heavily relies on the quality of the reward function. The paper uses established reward models like GenEval, OCR accuracy, and PickScore. However, a brief discussion on the potential failure modes or biases of these reward models and how they might affect DGPO's output would strengthen the paper.

### Questions
- In "Timestep Clip Strategy," a minimum timestep $t_{\min}$ is used during training to avoid overfitting to artifacts from few-step generation. How sensitive is the final model's performance to the choice of $t_{\min}$? Is there a principled way to select this value, or is it found empirically through hyperparameter tuning?
- How do you cope with Classifier-Free Guidance (CFG) in training and sampling? As I understand, the diffusion loss cannot be applied to the CFG model directly. For example, [1] claims that they abandon CFG entirely during both training and sampling, while still achieving strong performance.

[1] DiffusionNFT: Online Diffusion Reinforcement with Forward Process

### Soundness
3

### Presentation
3

### Contribution
3
