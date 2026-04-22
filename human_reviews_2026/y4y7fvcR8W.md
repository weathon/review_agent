# Reinforcement Learning with Verifiable Rewards: GRPO's Loss, Dynamics, and Success Amplification

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Group Relative Policy Optimization (GRPO) was introduced recently and used to train DeepSeek-R1 for promoting reasoning in LLMs under verifiable (binary) rewards. We show that the mean{+}variance calibration of these rewards induces a contrastive loss in which the contrastive samples are synthetic data drawn from the previous policy. While GRPO was originally paired with clipping to keep updates near the old policy, we analyze variants that differ in reward normalization (mean-only vs.\ mean{+}variance) and in how they regularize updates using KL divergence: either penalizing divergence from the previous model (\emph{mirror}), penalizing divergence from a fixed reference model $\pi_{\mathrm{ref}}$, or combining both forms of regularization. For each, the optimal policy $\pi_n$ admits an explicit form in terms of the binary reward and the first and second  order statistics of the reward under $\pi_{n-1}$, as well as the policies $\pi_{n-1}$ and $\pi_{\mathrm{ref}}$. Iterating results in a sequence  $\{\pi_n\}$ whose \emph{probability of success (PoS)} obeys a simple recurrence that converges to a fixed point determined by the reference PoS and the regularization strength. We further show that this fixed point exceeds the reference, demonstrating that GRPO amplifies the policy's probability of success.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper focuses on Reinforcement Learning with Verifiable Rewards and provides a theoretical analysis of several key properties of GRPO.

### Strengths
This paper provides a theoretical explanation of several key properties of GRPO and presents several recommendations for its use.

### Weaknesses
1. The abstract should not include citations.  
2. The related work analysis should be moved to a separate "Related Work" section.  
3. The structure of the abstract and introduction is somewhat unclear, making it difficult to identify the paper's motivation, i.e., the challenge it aims to address.  
4. Important experimental results should be included in the main text.  
5. Besides GRPO, are the theoretical results in this paper generalizable? Specifically, can they be applied to other SOTA RL methods in LLMs? If so, further experiments on additional RL methods should be conducted.

### Questions
Besides GRPO, are the theoretical results in this paper generalizable? Specifically, can they be applied to other SOTA RL methods in LLMs? If so, further experiments on additional RL methods should be conducted.

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
1

### Summary
This paper provides a theoretical analysis of GRPO for reinforcement learning with verifiable rewards.
It reformulates GRPO as a weighted contrastive loss and derives explicit success-rate dynamics, showing how the algorithm amplifies success probability and converges to a fixed point. The paper further studies variants such as Mirror GRPO and dual-KL GRPO, and experiments on GSM8K validate the predicted success amplification behavior.

### Strengths
1. **Clear theoretical formulation.**
The paper offers a clean and rigorous analysis of GRPO, reformulating it as a weighted contrastive learning problem and deriving explicit success-rate dynamics that explain its empirical behavior.

2. **Unifying perspective.**
It connects GRPO to broader RL principles such as mirror descent and KL-regularized optimization, providing a unified interpretation of several GRPO variants (Mirror GRPO, dual-KL GRPO, Dr.GRPO).

### Weaknesses
**Limited experimental scope and analysis.**
The experiments are limited to the GSM8K math reasoning dataset with experiements seems unrealted to the  (and place them in the appendix with fewer discussions related to it), leaving it unclear whether the theoretical findings and success amplification behavior generalize to other RLHF or verifiable reward tasks such as dialogue or code generation.

### Questions
Could the authors consider moving some of the appendix experiments into the main text and expanding the empirical discussion?
In particular, it would be valuable to include a deeper analysis of how different GRPO variants (e.g., Mirror GRPO, dual-KL GRPO, Dr.GRPO) behave empirically, and to discuss why their performance or convergence patterns differ in practice.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper provides a detailed theoretical analysis of GRPO, an RL method used to fine-tune LLMs using verifiable rewards. The authors show that GRPO can be viewed as an adaptive contrastive loss and derive mathematical recursive equations describing how the policy evolves over the course of training. Several variants are analyzed, including standard GRPO, Mirror-GRPO, and Dr. GRPO, with comparison of different normalization and regularization schemes. The results show that GRPO consistently increases the model’s probability of success. The proved theory predicts convergence properties for Mirror-GRPO. Additionally, an experiment on the GSM8K dataset shows that GRPO increases the success rate from 21% to 37.5%.

### Strengths
The paper provides mathematical grounding for GRPO, showing that different variants of the algorithm increase the probability of success of the learned policy. This is proven mathematically by the authors, showing that the probability of success converges to a fixed point (and the authors derive closed-form recursions). This offers theoretical insight into how the GRPO algorithm improves on stability and convergence. An experiment, although small, has consistent results with the theoretical predictions. Finally, the authors provide practical takeways that the readers may benefit from.

### Weaknesses
- The paper is very theory-focused, with limited experimental validation at small scale and without ablation studies. Besides, having some of these results in the main body of the paper would be benefitial, and more results with the different considered GRPO variants could improve the paper. Since almost any fine-tuning with verifiable rewards can raise the success rate, it's unclear whether the observed improvements come from GRPO itself or standard additional fine-tuning. 
- The paper could also benefit from a few intuitive examples of diagrams to illustrate the theory and why the various dynamics/variants help, in some sections, before diving into the equations. 
- At the end of the introduction, the paper could benefit from a higher-level explanation of the practical implications of the results shown in the paper; and at the end of the paper, of some mention of limitations, for instance regarding the assumptions made in the theory parts.

### Questions
* I am curious, as authors claim having improved success rate from 21% to 37.5% using GRPO, whether they have compared it to different fine-tuning methods such as PPO or DAPO? Do the authors have some idea about how GRPO improvement would translate in terms of sample efficiency or runtime, compared to other fine-tuning algorithms?
* How robust do the authors think the theoretical results would expand in the context of noisy imperfect binary rewards, which can be found in real-world LLM applications? What about continuous reward settings such as human feedback?
* Did the authors test the mirror GRPO version empirically, as the theory predicts a monotonic improvement of the success probability?

### Soundness
3

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
3

### Summary
The paper provides a formal analysis of GRPO-style RL for verifiable, 0/1 rewards. It derives closed-form policy updates and success-probability (PoS) dynamics for GRPO, Mirror-GRPO, and Dr.GRPO (with and without reference KL), showing that whitened advantages can be understood as an adaptive contrastive objective over successful vs failed trajectories. The main theoretical results characterize fixed points and monotonicity of PoS, with Mirror-GRPO shown to monotonically drive PoS to 1 under idealized assumptions.

### Strengths
1. Provides a clean, distribution-level view of GRPO that matches current practice and clarifies the roles of whitening and KL regularization.

2. The theory is tied back to practice with small but sensible experiments that qualitatively follow the predicted PoS amplification behavior.

### Weaknesses
1. The setting is highly idealized: single-step, 0/1 verifiable rewards and bandit-style updates, which sidesteps multi-turn reasoning or tool using cases.

2. Lack of experimental-level analysis.

3. The main analysis omits clipping and several stabilizing tricks that matter in large-scale GRPO, so it is unclear how close real training is to the derived dynamics.

4. Empirical evaluation is narrow and mostly illustrative; it does not stress-test where the theory breaks down (noisy or delayed rewards, strong model misspecification, heavy off-policy data).

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2
