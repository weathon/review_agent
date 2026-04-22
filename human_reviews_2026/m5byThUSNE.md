# Optimistic Task Inference for Behavior Foundation Models

- Avg Score: 6.50
- Decision: Accept (Oral)
- Scores: 6, 6, 8, 6

## Abstract
Behavior Foundation Models (BFMs) are capable of retrieving high-performing policy for any reward function specified directly at test-time, commonly referred to as zero-shot reinforcement learning (RL). While this is a very efficient process in terms of compute, it can be less so in terms of data: as a standard assumption, BFMs require computing rewards over a non-negligible inference dataset, assuming either access to a functional form of rewards, or significant labeling efforts. To alleviate these limitations, we tackle the problem of task inference purely through interaction with the environment at test-time. We propose OpTI-BFM, an optimistic decision criterion that directly models uncertainty over reward functions and guides BFMs in data collection for task inference. Formally, we provide a regret bound for well- trained BFMs through a direct connection to upper-confidence algorithms for linear bandits. Empirically, we evaluate OpTI-BFM on established zero-shot benchmarks, and observe that it enables successor-features-based BFMs to identify and optimize an unseen reward function in a handful of episodes with minimal compute overhead.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an ICRL (Interactive or In-Context Reinforcement Learning) algorithm based on Behavior Foundation Models (BFMs). The method performs online task inference by estimating the task embedding from observed rewards during inference, enabling dynamic policy adjustment. The authors provide partial theoretical analysis and show that the proposed method can achieve near-oracle performance within only a few episodes. Moreover, the approach can be extended to handle non-stationary reward settings.

### Strengths
1. The method is grounded on a solid theoretical foundation.

2. It is innovative and computationally efficient.

3. The approach is extensible and may inspire further research in this direction.

### Weaknesses
1. The experimental section only compares against Oracle and LoLA; it should include comparisons with other ICRL methods and evaluate optimization speed under out-of-distribution (OOD) settings.

2. In Appendix A5.3, line 1036, equation (84), the variable _x_ should likely be _ψ_.

3. In Algorithm 1, the formula for updating the estimator should reference the corresponding equation number for clarity.

### Questions
1. How does the proposed estimation procedure perform in sparse-reward settings (e.g., when the reward is only given at the end of an episode)?

### Soundness
3

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
The paper adresses task inference for BFMs without relying on a labeled “task‑inference” dataset at test time. Instead of estimating a point task embedding offline, the agent actively interacts with the environment and updates a belief over reward parameters. The core idea is to view policy search over task embeddings as linear bandit optimization that comes from the insight that for USF-based BFMs, the relationship between successor features and returns is approximately linear. So the paper proposes OpTI‑BFM, which maintains a least‑squares estimate over the unknown reward weights, then selects the task embedding optimistically. It provides theoretical sublinear regret bounds $O(d\sqrt(n))$ demonstrates empirically that the method can identify tasks within 5 episodes on DMC benchmarks.

### Strengths
- One of the core original ideas is the new task‑space bandit formulation for BFMs. The paper formulates online task inference as linear bandit optimization in the task-embedding space: with well-trained USFs, the expected episode return is approximately linear in the successor features of the policy conditioned on a task embedding, i.e., $\mathbb{E}[\hat{G}_k \mid s_0, \pi_z] \approx \langle \psi(s_0, z), z_r \rangle$. This lets the agent choose $z$ using a rule over confidence sets on the unknown reward weights $z_r$ (Eqs.~(7)--(10), \S3.2). The ``two-context'' twist (estimating with features $\phi$ but acting with successor features $\psi$) is new in the BFM literature and differentiates the method from standard LinUCB.
- Another contribution shows that running least-squares on reward-level pairs $(\phi, r)$ yields tighter confidence sets than regressing on episode-level returns and empirical SFs $(\tilde{\psi}, \hat{G})$.
- The paper has efficient implementation details that are non-trivial. For example: the optimizer is computationally light and table 1 shows that the machinery is deployable in real-time control. 
- The paper is mathematically sound and strong.

### Weaknesses
- The theory assumes perfect USFs and that the policy conditioned on $z$ is (near) optimal for reward $z$ (A1), strictly linear rewards with sub-Gaussian noise (A2), and an optimization oracle for Eq.~(10) (A3) and are introduced before Algorithm1 on p.5. In practice, USFs are learned with function approximation and the acquisition is solved by random shooting. The paper notes ``we found OpTI-BFM to perform well even when [A1--A2] are violated'' but does not quantify robustness to misspecification. 
- Also the regret bounds are proven for a variant that only updates $z$ at episode starts (\S3.3), while the recommended/practical algorithm updates $z$ every step, which is empirically much better (Fig.4). So the bound doeen't exactly cover the method actually used. (p.5; Fig.4 p.8).

### Questions
- Add controlled experiments that systematically break assumptions. For example inject bounded bias into $\psi$ and report regret vs.\ SF error or create rewards with a tunable component orthogonal to $\phi$ or ablate sub-Gaussian noise level. Reporting regret/performance vs.\ the projection error $\|r - \phi^\top z\|$ would make the empirical section align with A2/A1. 

- Provide either 1) a theoretical extension to per-step updates, or 2) a head-to-head comparison that keeps the episodic-only variant as the default and shows the practical gap in other settings too, with a candid discussion of why the bound should still be true.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors tackle a key limitation of behavioural foundation models (BFMs) based on Universal Successor Functions (USFs): the need for a dataset of labeled (state, reward) pairs. For cases where such data may be impractical or prohibitively costly to acquire, an alternative framework based on actively collecting a smaller amount of data online during deployment is explored.

To navigate this new framework the authors propose OpTI-BFM(Optimistic Task Inference for Behavioural Foundation Models) which leverages the linear relationship between features and rewards to update its belief over the space of rewards (i.e., incrementally improves its estimate of the task embedding using observed rewards). OpTI-BFM maintains a confidence ellipsoid over possible task embeddings and selects actions optimistically to efficiently explore the reward space.

Leveraging the fact that policy search for well-trained USF-based BFMs reduced to online optimisation of a linear function, a regret bound for OpTI-BFM in an episodic setting is established - the expected regret over n episodes $R_n\leq \mathcal{O}(d\sqrt{n})$. OpTI-BFM is evaluated empirically using a common zero-shot RL benchmark (ExORL) consisting of Walker, Cheetah and Quadruped environments with four reward functions each. OpTI-BFM achieves oracle performance (upper bound) within five episodes (5k steps) on all tasks, outperforming LoLA and the “Random” (lower bound) baselines.

### Strengths
The authors introduce a new framework for task inference in BFMs without labeled offline (state, rewards) data. In this framework, the relationship between BFM policy search and linear bandits is exploited to develop, and prove a regret bound for, the OpTI-BFM algorithm for online task inference. OpTI-BFM is timely and tackles the problematic requirement for labeled data with implications for many real world applications. The empirical results support the efficacy of OpTI-BFM in three standard zero-shot tasks (Walker, Cheetah, Quadruped), outperforming LoLA and reaching Oracle-level performance within 5 episodes.

### Weaknesses
- Whilst the experiment section is quite strong already it could be improved further if the authors are able to show the performance of OpTI-BFM on an alternative environment to those of the DeepMind Control suite. e.g. an alternate task with pixel observations.

- There is not much discussion of how OpTI-BFM could be deployed for real-world use. The authors say that their method would enable BFMs to work “beyond domains in which rewards are readily available”, but it is not obvious to me how it would interact with a real environment to get immediate reward labels. It would be appreciated if this could be explained further by the authors.

### Questions
Could the authors comment on whether OpTI-BFM generalises beyond continuous control environments?

### Soundness
3

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
This paper proposes OpTI-BFM, an optimistic decision criterion that directly models uncertainty over reward functions and guides BFMs in data collection for task inference. Authors frame this online task inference problem as a linear bandit problem and maintain a probabilistic belief (a confidence ellipsoid) over the true task embedding z_r by performing real-time least-squares regression on reward-level feedback. Experiments on zero-shot benchmarks show that OpTI-BFM matches or surpasses offline reward inference methods with much less data.

### Strengths
- Tackles a significant and practical bottleneck. Online task inference with only a few active-interaction episodes is important for real-world applications.
  
- Solid theoretical guarantees via connections to linear bandit algorithms.
  
- Good empirical performance with additional experiments and analysis, e.g. episode-level updates and non-stationary rewards.

### Weaknesses
- The paper operates under assumptions of a perfect BFM / successor feature model. It is unclear what would happen for the theoretical guarantees or how the empirical results would change with approximation errors.
  
- The formal regret bound is proven for an episodic-update variant of OpTI-BFM. However, the experiments (Sec.5.3, Fig. 4) show that the step-update variant is empirically superior and converges much faster. While it is a positive result that the practical algorithm is even better, it means the theory doesn't formally cover the best-performing algorithm presented.
  
- The optimistic search for z is currently done by random shooting and may fail for larger, more complicated spaces.

### Questions
- How is the method's robustness to misspecification? For example, how does OpTI-BFM perform if the true reward function r(s) has a significant non-linear component, or if the pre-trained BFM is of lower quality (violating A1)? How does its degradation compare to the offline "Oracle" regression, which would also suffer from this misspecification?
  
- How does the random shooting for UCB optimization scale? Have you explored the sensitivity to this number of samples? Would a gradient-based approach be more robust or scalable?
  
- Could the theoretical guarantees be extended to the step-level update case?

### Soundness
3

### Presentation
4

### Contribution
3
