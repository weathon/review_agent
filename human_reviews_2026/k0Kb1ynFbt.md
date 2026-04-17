# Exploratory Diffusion Model for Unsupervised Reinforcement Learning

- Decision: Accept (Oral)
- Scores: 6, 6, 6, 6

## Abstract
Unsupervised reinforcement learning (URL) pre-trains agents by exploring diverse states in reward-free environments, aiming to enable efficient adaptation to various downstream tasks. Without extrinsic rewards, prior methods rely on intrinsic objectives, but heterogeneous exploration data demand strong modeling capacity for both intrinsic reward design and policy learning. We introduce the **Ex**ploratory **D**iffusion **M**odel (**ExDM**), which leverages the expressive power of diffusion models to fit diverse replay-buffer distributions, thus providing accurate density estimates and a score-based intrinsic reward that drives exploration into under-visited regions. This mechanism substantially broadens state coverage and yields robust pre-trained policies. Beyond exploration, ExDM offers theoretical guarantees and practical algorithms for fine-tuning diffusion policies under limited interactions, overcoming instability and computational overhead from multi-step sampling. Extensive experiments on Maze2d and URLB show that ExDM achieves superior exploration and faster downstream adaptation, establishing new state-of-the-art results, particularly in environments with complex structure or cross-embodiment settings. The source code is provided at https://github.com/yingchengyang/ExDM.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes a novel approach called ExDM for unsupervised RL using a diffusion model. During pre-training, ExDM trains a diffusion model to model the state and action distributions of interactions with the environment and derives an intrinsic reward to encourage exploration that is inversely proportional with approximate probability of state visitations of the diffusion model. Using these intrinsic rewards, the approach trains a Gaussian policy that explores the environment. During fine-tuning, the Gaussian policy can be trained with task-specific rewards, or the diffusion policy trained during pre-training can be fine-tuned. To enable fine-tuning of the diffusion policy, a novel regularized training objective is being derived, similar to soft RL and using implicit Q-learning from the offline RL literature to avoid out-of-distribution actions being sampled. The new approach is shown to lead to more exhaustive exploration, as measured by state coverage, during pre-training, and the work shows that fine-tuning of the pre-trained Gaussian and diffusion policies lead to higher performance compared to alternative pre-training approaches.

### Strengths
## Originality
The approach proposed in this work appears original and novel. While diffusion policies are not new, and diffusion models have been used to express various data distributions, their application to URL is novel to the best of my knowledge. Furthermore, the theoretical contributions in Theorem 4.1 justifying the need for more expressive policies for unsupervised pre-training, and in deriving a novel algorithm for online fine-tuning of the diffusion policy are valuable to the community.

## Significance
As stated, I consider the theoretical contributions of this work significant and valuable to the community. Similarly, the empirical results indicate a small but consistent improvement of ExDM compared to the strongest URL baselines. Assuming these results were generated under fair hyperparameter tuning (see question 3), they demonstrate that ExDM is a significant contribution to the field.

One aspect that dampens my otherwise very positive impression of the significance of this work is the unclear benefits of the diffusion policy part of the ExDM algorithm. The diffusion policy is arguably the biggest and most complex novel contribution of this work, but it appears to not contribute meaningfully to the performance of the approach (see Weakness 1.). Without the diffusion policy, the ExDM algorithm could have also "just" been a diffusion model of the state distribution to derive a slightly novel intrinsic reward with a Gaussian policy. I would expect further discussion with reviewers and the authors to clarify that part of this work.

## Quality & Clarity
I find the writing and presentation of this work of a high quality. The empirical evaluation also appears to follow good practice, and provides further ablations and analyzes to shed more light on the learned components. There are few unclear or not well supported statements in this work that are listed below, but none of them are major issues or central to the work.

### Weaknesses
Below, I provide a list of weaknesses that should be addressed. I would consider all weaknesses stated as **major** important to address in order to justify acceptance.

1. **(Major)** The introduction of this work significantly leans into the motivation that typical URL approaches use policies that are not sufficiently expressive (often discrete or Gaussian policies) to properly explore the environment during pre-training. Theorem 4.1 as theoretical contribution of this work further supports this narrative and the pre-training and fine-tuning of the diffusion policy component of ExDM takes over large parts of Section 4. However, despite this motivation and more expressive diffusion policy, the environment interactions during pre-training are still done only using the Gaussian policy (as per line 8 in Algorithm 1). Furthermore, fine-tuning of the Gaussian policy of ExDM still leads to higher performance than fine-tuning the diffusion policy (see Figure 3 (a) vs (c)), a fact that is acknowledged by the authors in Section 5.4. 
   All this makes me question what the diffusion policy of ExDM truly adds to the method. It appears the benefits of ExDM are not from training a more expressive policy in the diffusion policy, but from the diffusion model of state distributions that appears to provide a more informative intrinsic reward to exhaustively explore the environment. Theorem 4.1 still motivates the use of more expressive policies but it appears that the challenges of training diffusion policies remain to outweigh the benefits of a more expressive policy. My questions here are
	1. Does the diffusion policy of ExDM provide some benefits, or does it contribute to any component of the approach in a way that I have not seen?
	2. Or am I correct in my assessment that it appears to provide no benefits over the Gaussian policy so far?
	3. Given Theorem 4.1, one might expect the more expressive diffusion policy of ExDM to provide clear benefits. In Section 5.4, it is stated that the diffusion policy might perform worse than the Gaussian policy "[...] due to limited interaction timesteps during fine-tuning". Did you fine-tune the diffusion policies with the same 2M steps as used for Gaussian policy fine-tuning? If not, how many steps were used to fine-tune diffusion policies and why was a different number of fine-tuning steps used for these policies? 
2. **Major:** Appendix C.3 states that "hyperparameters of baselines are taken from their implementations". Were these hyperparameters explicitly tuned for the Maze2D tasks evaluated in this work? How about ExDM hyperparameters listed in Table 2? Were these tuned in any form for the Maze2D tasks? I would expect comparable effort to be spent on tuning hyperparameters across all approaches to have confidence in the empirical results presented in this work, and this should be clarified.
3. The work makes the following two statements that are unclear, imprecise, or incorrect and should likely be expressed more precisely:
	1. "[...] the optimal policy of standard RL is a simple deterministic policy" (Section 4.1). It is unclear to me what "standard RL" means; to the best of my knowledge, the statement is only generally true in fully observable (single-agent) RL; partially observable or multi-agent environments might have stochastic optimal policies.
	2. "[...] the explored replay buffer is always diverse and heterogeneous, as the policy continuously changes to visit new states during pre-training. Consequently, URL requires capturing the heterogeneous distribution of collected data and obtaining policies with high diversity" -- I would argue that URL does not necessarily require to capture distribution of multiple policies to achieve its objective of learning a policy that maximizes state entropy (as stated in Section 4.1). It is merely the practical approach of using off-policy algorithms and a replay buffer of experiences that requires to capture such distributions.
	3. "The Gaussian behavior policy $\pi_g$ can then be trained using any RL algorithm" -- I would argue that this is not true since off-policy RL algorithms would be needed. The experience samples that $\pi_g$ is trained on, as per Algorithm 1, are from a replay buffer and, thus, off-policy with respect to the trained policy $\pi_g$.
4. The baselines visualized in Figure 2 appear to be mostly poor performing or middle of the pack when looking at Table 1. None of the strongest baselines (MEPOL, RE3, CIC) are included in Figure 2, supposedly to make the result of ExDM appear more impressive. I would appreciate Figure 2 would show the strongest 1-2 baselines in each family which appear to be R3 and MEPOL for exploration and CiC for skill discovery baselines. (Visualizations of all baselines are shown in Appendix C.4 but I would prefer for the most relevant ones to be shown in the main corpus of the paper)
5. The fine-tune box of Figure 1 appears confusing to me and I believe the policy titles should be flipped. The left half appears to show the fine-tuning of the Gaussian policy and the right half the fine-tuning of the diffusion policy (as per plot and legend) but the red titles above them are reversed. 
6. (Minor) I noticed that baseline algorithms do not have identical colors in Figure 3 (a) and (b) which makes it slightly harder to cross-reference these results at a glance.

### Questions
1. What are the benefits of the diffusion policy of ExDM, or does it not contribute to the performance of the system (given the fine-tuned Gaussian policy performs better)?
2. Would the authors be able to elaborate on their statement that the diffusion policy might perform worse than the Gaussian policy "[...] due to limited interaction timesteps during fine-tuning (Section 5.4). Were the diffusion policies fine-tuned with less than the 2M steps used for Gaussian policies? What other reasons might there be for the more expressive diffusion policies to perform worse after fine-tuning?
3. How were hyperparameters of all baselines and ExDM selected for the empirical Maze2D experiments?

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
The authors proposed an unsupervised RL algorithm called ExDM with diffusion action head.

### Strengths
1. The performance seems to be very strong compared to baselines
2. The presentation is clear and easy to follow

### Weaknesses
1. The motivation is somewhat weak. I'll put my questions in the following section.

### Questions
1. The results on URLB seems to be promising, but why APT and APS by Liu et al. were not included in the baseline?

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
3

### Summary
This paper introduce the Exploratory Diffusion Model, which leverages the expressive power of diffusion models to fit diverse replay-buffer distributions, thus providing accurate density estimates and a score-based intrinsic reward that drives exploration into under-visited regions. This mechanism substantially broadens state coverage and yields robust pre-trained policies. Beyond exploration, ExDM develops an efficient decoupled training scheme and a finetuning algorithm for adapting pre-trained diffusion components to downstream tasks under limited interaction, with theoretical guarantees of convergence and optimality.

### Strengths
- Empirical gains across multiple settings: The figure indicates consistent improvements over strong unsupervised exploration baselines in URL, in cross-embodiment transfer, and when initializing diffusion policies.
- Potentially general mechanism: A diffusion-based exploratory prior could be a broadly applicable way to induce diverse skills or state coverage that helps downstream RL fine-tuning and transfer.
- Sufficient theoretical proof.

### Weaknesses
None

### Questions
- How is the intrinsic reward designed, what is the underlying rationale, and were alternative design schemes considered?
- What is the difference between unsupervised reinforcement learning and Meta-RL, given that both seem to aim at rapidly solving new tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
-The paper introduces the Exploratory Diffusion Model (ExDM) to address the exploration bottleneck in Unsupervised Reinforcement Learning (URL).

-Unlike prior methods that use simple policies, ExDM leverages the superior expressive power of diffusion models to accurately model the complex and heterogeneous state distributions collected during exploration. A diffusion model ($\epsilon_{\theta^{\prime}}$) is trained on the replay buffer's state distribution. A novel score-based intrinsic reward ($R_{score}$) is calculated from this model's loss (its inability to fit a state), which guides the agent to under-visited regions. To ensure efficiency, a simple Gaussian behavior policy ($\pi_g$) is trained to maximize this $R_{score}$ and is used for fast data collection, avoiding slow diffusion sampling.
The pre-trained Gaussian policy ($\pi_g$) can be fine-tuned on downstream tasks using standard RL algorithms (like DDPG).

### Strengths
- The author addressed that this is the first work to successfully integrate diffusion models into the unsupervised exploration phase of RL. The concept of using the diffusion model's density estimation loss as the intrinsic reward is a significant contribution over prior reward mechanisms (like RND or ICM).
- It was impressed that the decoupled training scheme (fast Gaussian actor, slow diffusion reward-calculator) is a clever and practical solution to the primary obstacle of using generative models in online RL: slow sampling speed.
- I think that the paper provided a novel, non-trivial algorithm for fine-tuning the diffusion policy itself, complete with a formal proof of optimality (Theorem 4.2). This goes beyond just using the model as a static prior.
- Overall, the method's superior performance is not marginal. Its experiments dramatically outperform all baselines in complex exploration tasks (e.g., Fig. 2, where baselines get stuck and ExDM covers the entire maze) and shows consistent SOTA results across all aggregate metrics in URLB (Fig. 3).

### Weaknesses
- There is a limitation in terms of performance gap: The paper's own experiments (Fig. 3) show that fine-tuning the simple Gaussian policy ($\pi_g$) actually achieves better final performance than the proposed new, complex diffusion policy fine-tuning algorithm (Algorithm 2). The reason should be explained and analyzed intensively. Compared with Fig. 3(a) and (b), the expert normalized scores of the proposed algorithm in Fig. 3(c) were small. 
- The authors stated that the performance degradation may be due to limited interaction timesteps during fine-tuning. The advanced works to overcome this problem should be discussed further.
- While their new fine-tuning method (Algorithm 2) is a novel contribution, it is not yet fully optimized and is outperformed by a simpler, standard approach such as DDPG. 
Therefore, it is expected that the paper's primary strength lies in its pre-training exploration (which produces a superior Gaussian policy) rather than its diffusion policy fine-tuning performance. This mechanism could be considered in this discussion of this paper.

### Questions
Please, refer to the weakness and answer against my concerns.

### Soundness
3

### Presentation
2

### Contribution
3
