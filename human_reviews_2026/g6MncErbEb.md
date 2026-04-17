# Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
Meta-reinforcement learning (Meta-RL) facilitates rapid adaptation to unseen tasks but faces challenges in long-horizon environments. Skill-based approaches tackle this by decomposing state-action sequences into reusable skills and employing hierarchical decision-making. However, these methods are highly susceptible to noisy offline demonstrations, leading to unstable skill learning and degraded performance. To address this, we propose Self-Improving Skill Learning (SISL), which performs self-guided skill refinement using decoupled high-level and skill improvement policies, while applying skill prioritization via maximum return relabeling to focus updates on task-relevant trajectories, resulting in robust and stable adaptation even under noisy and suboptimal data. By mitigating the effect of noise, SISL achieves reliable skill learning and consistently outperforms other skill-based meta-RL methods on diverse long-horizon tasks. Our code is available at https://github.com/epsilog/SISL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses a critical challenge in skill-based meta-RL: performance degradation when learning skills from noisy or suboptimal offline demonstration datasets.

The authors propose a new framework, SISL, designed to be robust to such data imperfections. SISL's novelty lies in two key contributions:
- Decoupled Skill Self-Improvement
- Skill Prioritization via Maximum Return Relabeling

### Strengths
- **Significance**: The paper tackles a highly significant and practical problem: the sensitivity of skill-based (meta-)RL algorithms to noisy offline demonstration data. As the field moves toward using large, imperfect, real-world datasets, methods that can "denoise" and "self-improve" upon this data are critical. This work provides a strong solution to this problem.

- **Clarity and Presentation**: The paper is a clear. It is well-written, logically structured, and supported by high-quality figures that provide both conceptual intuition (Fig. 1, 3) and compelling qualitative evidence (Fig. 7, F.2).

- **Empirical Rigor and Performance**: The experimental evaluation is good. The authors demonstrate SISL's superiority across four challenging long-horizon environments against a comprehensive set of relevant baselines.

### Weaknesses
- **Complexity and Computational Cost**: The SISL framework introduces several new components that add to the computational load: training the skill-improvement policy, the RND networks, and the reward model, plus periodically re-training the entire skill library.

- **Periodic Re-initialization**: The design choice to re-train the skill library and re-initialize the high-level policy is a key part of the algorithm. The ablation in G.4 shows that not re-initializing is catastrophic, which makes sense due to non-stationarity. However, this periodic "reset" might be disruptive, and the paper does not explore the sensitivity to the frequency itself.

### Questions
- **Sensitivity to $K_{iter}$**: The periodic re-training of the skill model and re-initialization of $\pi_h$ every $K_{iter}$ steps is a critical design choice. The ablation in G.4 proves re-initialization is necessary, but how sensitive is the algorithm to the value of $K_{iter}$? What happens if re-training is too frequent (e.g., $K_{iter}=100$) or too infrequent (e.g., $K_{iter}=5000$)?

- **Necessity of Decoupled Policies**: Why is the decoupling of $\pi_h$ and $\pi_{imp}$ strictly necessary? Could a single high-level policy not perform both roles?

### Soundness
3

### Presentation
3

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
The paper addresses the topic of skill-based meta-learning in reinforcement learning. It points out existing skill learning methods relying on offline demonstrations often suffer from suboptimal data quality. The proposed Self-Improving Skill Learning (SISL) framework shows robustness to noisy demonstrations by introducing a skill improvement policy and performing skill refinement through trajectory prioritization based on returns.

### Strengths
The paper adopts perturbations, a representative technique in meta-learning, where target tasks are unseen by the skill-based reinforcement learning agent. This approach constrains the learning process to remain close to the demonstration manifold, thereby facilitating effective skill acquisition.

### Weaknesses
The SISL framework appears to be quite complex and computationally expensive. For example, it involves maintaining multiple buffers that serve similar functions. The proposed method also does not seem to be specifically tailored to address the meta-learning problem. Baselines such as SPiRL and SiMPL learn skills solely from offline demonstrations, whereas SISL additionally collects online data, making the comparison unfair.

**Minor**

Appendix B should be moved to the main body of the paper. The figures and their captions are positioned too close to the main text, which reduces readability. Figure captions are brief and could be improved to provide clearer explanations of the content.

### Questions
(1) How does the method avoid overfitting to the source tasks when evaluating trajectories based on returns?

(2) Please compare the skills extracted by SISL with those obtained by other baseline methods.

(3) Consider including an additional experiment where Equation (3) is evaluated without the KLD term.

(4) Please report the success rate on the maze environments.

(5) Compare the computational cost across different ablation settings.

(6) Could you clarify why performance increases as $\sigma$ increases in some cases in Table 1?

**Minor**

In Figure 2, which algorithm is being illustrated?

How is the KLD term in Equation (3) computed?

How many offline trajectories generated by the noisy behavior policy are used to solve each task?

### Soundness
3

### Presentation
2

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
This paper targets the sensitivity of skill-based meta-RL methods to noisy offline demonstrations. The proposed method, Self-Improving Skill Learning (SISL), addresses this by dynamically refining the skill library during meta-training. Its core contributions are:

- Decoupled Skill Self-Improvement: A dedicated skill-improvement policy $\pi_{imp}$ is introduced alongside the standard high-level policy $\pi_h$. $\pi_{imp}$ explores near the offline data distribution to find higher-quality trajectories.
- Skill Prioritization via Maximum Return Relabeling: To fuse the noisy $B_{off}$ with the clean $B_{on}$ for skill refinement, SISL trains a reward model using only online data. This model assigns a hypothetical maximum to each offline trajectory. The skill update then samples from $B_{off}$ using a softmax prioritization based on $\hat{G}$, effectively filtering out low-quality data.

Experiments on long-horizon tasks show that SISL substantially outperforms baselines in noisy data regimes.

### Strengths
- This work focuses on a key practical problem in meta-RL, where offline demonstrations may be noisy.
- The paper provides strong empirical evidence demonstrating significant performance improvements over all baselines.
- The paper is well-structured. The method is presented logically.

### Weaknesses
- The paper claims only "16% more time per iteration", which seems surprisingly low. A SISL iteration appears to involve: (1) rollout with $\pi_h+\pi_l$, (2) rollout with $\pi_{imp}$, (3) $\pi_h$ update, (4) $\pi_{imp}$ update, (5) $\pi_l$ update, and (6) $\hat{R}$ update. In contrast, the baseline presumably only includes steps (1) and (3). It is unclear how the 16% figure was calculated. A more detailed breakdown and a comparison of total training time, not just per-iteration cost, would be more informative.
- The skill refinement interval, $K_{iter}$, is a critical new hyperparameter that balances the stability of the high-level policy $\pi_h$ against the speed of skill refinement for $\pi_l$. However, the paper provides no ablation study for $K_{iter}$, making it difficult to assess the algorithm's sensitivity to this important design choice.
- The "maximum return relabeling" mechanism hinges on a reward model $\hat{R}$ trained on online data. This implies a potential failure mode: in tasks with sparse or complex rewards, training $\hat{R}$ could become unstable, compromising the data prioritization mechanism and overall performance.
- Lack of Clarity in Pseudocode. For example:
    - In Algorithm 1 (Meta-Train), line 14 updates the skill parameters $\phi$ (including $q_{\phi}$). However, the algorithm does not show where the skill encoder $q_{\phi}$ is used for inference on trajectories.
    - In Algorithm 2 (Meta-Test), line 7 updates parameters $\theta$. It is ambiguous whether$\theta$also includes the task encoder $q_{e,\theta}$. If the task encoder is frozen during the meta-test phase (as is typical), it would be clearer to use different notation to distinguish its parameters from the policy/value function parameters being adapted.

### Questions
- Figure F.1 shows that the mixing coefficient $\beta$ quickly converges to high values, implying that the online-generated data quality significantly surpasses the offline data. This raises the question: could the offline data $B_{off}$ be discarded altogether? For instance, what if one first trained an expert policy (e.g.,$\pi_{imp}$ trained without the KL-divergence term, acting as a standard RL agent) to collect a new, clean dataset, and then used this dataset for learning $\pi_l$ and $\pi_h$? Would this not be a simpler and potentially lower-cost alternative to the complex relabeling and mixing process?
- Does the framework assume that only optimal trajectories are useful for training the low-level policy $\pi_l$? It is possible that certain suboptimal trajectories, while not useful for the current set of training tasks, might contain skills that are highly beneficial for generalizing to unseen test tasks. The "Maximum Return Relabeling" mechanism prioritizes trajectories based on their estimated maximum return on *training* tasks. Does this design inadvertently suppress these "suboptimal but generalizable" trajectories, thereby weakening the utilization of data that could be crucial for meta-generalization?
- See Weaknesses

### Soundness
2

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
4

### Summary
The authors propose SISL (Self-Improving Skill Learning), a skill-based meta-RL framework designed to handle noisy offline demonstrations in long-horizon tasks. SISL introduces two key mechanisms: (1) decoupled skill self-improvement, where a dedicated improvement policy explores near the offline data distribution to discover higher-quality rollouts that progressively refine the skill library, and (2) skill prioritization via maximum return relabeling, which uses a learned reward model to assign estimated returns to offline trajectories and reweight samples through softmax prioritization.

### Strengths
- The paper is well-motivated and very clearly presented, with very useful illustrations.
- Separating exploitation ($\pi_h$) and skill improvement ($\pi_{imp}$) with self-supervised guidance via prioritized buffers appears to be a novel and elegant contribution.
- The evaluation is sound and detailed, with 4 diverse environments with multiple noise levels, thorough baseline comparisons, and extensive ablations.

### Weaknesses
I believe the paper would benefit from comparing against a GCRL baseline with Hindsight Experience Replay (HER) or similar relabeling techniques. This would help demonstrate that SISL's approach to leveraging the offline dataset is superior to existing relabeling methods in both sample efficiency and final performance. 

Minor typo: 
- Line 279: "addtion"

### Questions
Shouldn't $\beta$ in Eq. 7 be task-dependent? I guess that sometimes, different tasks may benefit from different balances between $B_{\text{off}}$ and $B_{\text{on}}$. How sensitive is performance to using a global $\beta$ versus task-specific $\beta^i$?

### Soundness
4

### Presentation
4

### Contribution
3
