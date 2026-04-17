# Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
When there exists an unlearnable source of randomness (noisy-TV) in the environment, a naively intrinsic reward driven exploring agent gets stuck at that source of randomness and fails at exploration.
Intrinsic reward based on uncertainty estimation or distribution similarity, while eventually escapes noisy-TVs as time unfolds, suffers from poor sample efficiency and high computational cost. 
Inspired by recent findings from neuroscience that humans monitor their improvements during exploration, we propose a novel method for intrinsically-motivated exploration, named Learning Progress Monitoring (LPM).
During exploration, LPM rewards model improvements instead of prediction error or novelty, effectively rewards the agent for observing learnable transitions rather than the unlearnable transitions.
We introduce a dual-network design that uses an error model to predict the expected prediction error of the dynamics model in its previous iteration, and use the difference between the model errors of the current iteration and previous iteration to guide exploration.
We theoretically show that the intrinsic reward of LPM is zero-equivariant and a monotone indicator of Information Gain (IG), and that the error model is necessary to achieve monotonicity correspondence with IG.
We empirically compared LPM against state-of-the-art baselines in noisy environments based on MNIST, 3D maze with 160x120 RGB inputs, and Atari.
Results show that LPM's intrinsic reward converges faster, explores more states in the maze experiment, and achieves higher extrinsic reward in Atari.
This conceptually simple approach marks a shift-of-paradigm of noise-robust exploration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents the design of a new intrinsic reward based on maximizing a proxy for information gain under a learned model. The authors show that by rewarding the agent in consecutive iterations according to its capability to generate data which minimizes the error of the learned model, this leads to directionally optimizing for information gain under the model and to explorative agent behavior that achieves good performance. The main contribution is the theoretical proof which shows that the proposed intrinsic reward serves a directionally correct proxy for information gain under a learned model, which is a relevant and popular objective. The empirical evidence shows that the proposed method achieves comparable and sometimes superior performance than a subset of exploration baselines.

### Strengths
The motivation of the work and contribution (especially the theoretical contribution) is sound and rigorous. Information gain is a popular objecitve to optimize in model-based RL and particularly for enhanced exploration in intrinsically-motivated RL, and the presented intrinsic reward (which is very simple to understand and implement) is shown to be a correct directional estimator of the original and more complex information gain. 

The paper is clearly written, rigorous and easy to follow. The evaluation protocols are described in detail and a great number of independent seeds is used to draw statistically significant conclusions.

### Weaknesses
The main weakness of the paper is the empirical evidence provided. While the evaluation protocols are rigoroous in some regards (e.g. a great number of indpendent seeds is used to aggregate results), the choice of environments is non-standard, and other training protocols (e.g. RL algorithm used, training steps, and choice of baselines) is not adequate.

 I think the paper would benefit from the following points:

- First evaluation of the method in classical exploration environments where the properties of the method can be understood in a perfectly controlled environment (e.g. like a fully-observed MDP). These can be gridworlds, or other popular exploration tasks like DeepSea.

- Addition of tasks that truly test for exploration capabilities. At the current stage, I don't think the results in Atari are convincing or relevant. The paper describes SpaceInvaders and MsPacman as "hard exploration environments" but they are really not, since they both provide dense rewards and are fully-observed. Additionally, the agents are only trained for 1M environment, steps which is extremely far from the well-tested standards in Atari of 200M. The performance of the algorithm is unknown beyond 1M environment steps, which weakens the claims. I believe the suite of tasks from MiniGrid, ProcGen, or Crafter would represent much more interesting evaluation suites. 

- While there is an appropriate number of baselines used in the paper, it is unclear why those were chosen before others that are potentially more relevant and/or recent. e.g. E3B [1], Disagreement [2], NGU [3]. In general, the evaluation protocols of this work are not close or similar to other standards previously proposed in intrinsically-motivated RL [4].

- Related to the previous point, I think the authors could elaborate more in covering related work in intrinsically-motivated RL, especially recent algorithms that achieve great performance in noisy environments [5] and others based on active inference theory for RL [6,7,8,9], which is all about training agents to minimize the error/surprise of their internal world models (hence very relevant to this work), and is currently omitted in the paper. Since the authors cover curiosity and episodic methods, I would also suggest covering the combination of both [10,4].

With this, I think the paper presents an interesting and promising idea, and that the theoretical contribution is relevant. However, the poor evaluation and empirical evidence weakens the claims of the paper and should be improved and made more general to render this paper publishable.

[1] Henaff, Mikael, et al. "Exploration via elliptical episodic bonuses." Advances in Neural Information Processing Systems 35 (2022): 37631-37646.

[2] Pathak, Deepak, Dhiraj Gandhi, and Abhinav Gupta. "Self-supervised exploration via disagreement." International conference on machine learning. PMLR, 2019.

[3] Badia, Adrià Puigdomènech, et al. "Never give up: Learning directed exploration strategies." arXiv preprint arXiv:2002.06038 (2020).

[4] Yuan, Mingqi, et al. "Rlexplore: Accelerating research in intrinsically-motivated reinforcement learning." arXiv preprint arXiv:2405.19548 (2024).

[5] Saade, Alaa, et al. "Unlocking the power of representations in long-term novelty-based exploration." arXiv preprint arXiv:2305.01521 (2023).

[6] Berseth, Glen, et al. "SMiRL: Surprise minimizing RL in Entropic environments." (2019).

[7] Friston, Karl. "The free-energy principle: a unified brain theory?." Nature reviews neuroscience 11.2 (2010): 127-138.

[8] Rhinehart, Nicholas, et al. "Information is power: Intrinsic control via information capture." Advances in Neural Information Processing Systems 34 (2021): 10745-10758.

[9] Hugessen, Adriana, et al. "Surprise-Adaptive Intrinsic Motivation for Unsupervised Reinforcement Learning." arXiv preprint arXiv:2405.17243 (2024).

[10] Henaff, Mikael, Minqi Jiang, and Roberta Raileanu. "A study of global and episodic bonuses for exploration in contextual mdps." International Conference on Machine Learning. PMLR, 2023.

### Questions
Is the $g_{\phi}$ model reinitialized at each iteration to only fit the most recent model errors? or is it continually trained with the evolving world model?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies how intrinsic motivation methods in RL fail in stochastic environments, the classic “noisy TV” problem. It proposes Learning Progress Monitoring (LPM), which rewards how much a dynamics model improves rather than how much error it has. The key idea is to compare the current log-prediction error to an expected previous error estimated by a small network. The authors show that this bonus can be viewed as a monotone, noise-robust approximation of information gain. Experiments on Noisy-MNIST, MiniWorld with injected noise, and Atari with action-triggered random frames show that LPM is more stable and less distracted by unpredictable inputs than ICM, RND, and other curiosity methods.

### Strengths
- The motivation is clear and relevant. It is a known weakness of curiosity-based exploration.
- The method is simple and easy to add to existing RL frameworks.
- Good theoretical grounding: the authors link learning progress to information gain and show the expected-prior term is mathematically justified.
- The experiments are well aligned with the paper’s goal, with clear setups for both state and action noise.
- Consistent improvements over strong baselines on MiniWorld and Atari when noise is introduced

### Weaknesses
- The error-predictor network is central to the method, but there are no ablations replacing it with simpler options (e.g., EMA of past errors, or previous model prediction).
- No enough sensitivity analysis for key hyperparameters (queue size, update period, …).
- Evaluation is limited to two noise protocols (CIFAR injection, noisy wall). Other types such as temporal or occlusion noise are not explored.
- Experiments rely only on on-policy methods (A2C/PPO); no test with off-policy RL to check generality.
- Code is not yet released, which hinders reproducibility.
- The paper lacks some references and comparison with prior work. See references as examples.

[References]
Planning to Explore via Self-Supervised World Models — Sekar et al., ICML 2020
BYOL-Explore: Exploration by Bootstrapped Prediction — Guo et al., NeurIPS 2022
Fast and Slow Curiosity for High-Level Exploration in Reinforcement Learning — Nicolas Bougie and Ryutaro Ichise, Applied Intelligence 2021

### Questions
1. Add ablations replacing the error-predictor with simpler mechanisms (EMA, previous model) to confirm necessity.
2. Include more types of stochasticity (temporal, occlusion, heavy-tailed) to test robustness.
3. Add off-policy baselines or experiments to check method generality.
4. Release code or add details for reproducibility. 
5. Compare to additional noise-aware exploration baselines (e.g., Plan2Explore, Disagreement).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes" Learning Progress Monitoring (LPM)", an exploration method that aims at rewarding an agent for improving its predictions rather than for raw prediction error or novelty. By tracking changes in model error across iterations, LPM wants to focus on learnable transitions and avoids unlearnable noise (e.g., “noisy-TV”).

### Strengths
- The contribution tackles an important problem, in particular "(...) strategy is naturally robust to noisy-TV distractions and promotes efficient exploration by directing effort toward the most informative experiences.". 
- There is a theoretical connection to information gain to justify the approach and to the necessity of the error model.

### Weaknesses
A few elements do not seem to be fully detailed both on the methodology and the experiments.

### Questions
*Methodology*
- It is unclear how the approach deals with learning and overfitting in the model. In the context of the noisy TV, does the model fully overfits on the new transition before fitting the error model? How are the hyper-parameters chosen (e.g. training steps, NN architecture, etc.)

*Experiments*
- In Fig 4. the standard deviation of the extrinsic rewards that seems to be reported looks extremely small. Is there a reason? Can you clarify the setup?
-Space Invader, Ms PacMan and UpNDown (noisy) are not the hardest exploration games and are not very sparse in terms of rewards. Could you explain why you choose these games as opposed to e.g. Montezuma's revenge.
- Some visualisations of the intrinsic rewards and the estimated errors on examples (e.g. some specific trajectories of the agent) would help see the inner workings of the algorithm and check the suitability of the experiments.

*Other comments*
- Some typos (e.g. line 144,145, there is no dot between the sentences).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Learning Progress Monitoring(LPM), a novel intrinsic reward method to solve the noisy-TV problem where agents get stuck on unlearnable randomness . Instead of rewarding high prediction error, LPM rewards the improvement in prediction accuracy, effectively measuring the model's learning progress . It uses a dual-network to compare the current model's error against the expected error of the previous model. This learning progress reward is theoretically shown to be a monotone indicator of Information Gain (IG), uniquely dropping to zero for both fully learned states and unlearnable states. Empirical results show LPM achieves robust exploration in noisy environments, avoiding the catastrophic failures of other methods while maintaining SOTA performance in deterministic settings.

### Strengths
**Novel Reward Mechanism**: This paper proposes an innovative intrinsic reward mechanism based on "learning progress" (improvement in prediction error) rather than the magnitude of the error itself.

**Excellent Noise Robustness**: This draft empirically demonstrated to be unparalleled in handling random uncertainty (noisy TV), preventing the catastrophic performance degradation seen in baselines.

**Strong Theoretical Explanation**: This paper theoretically shows that the LPM reward is a monotonic measure of information gain (IG), mathematically justifying the ignorance of both boring and unlearnable states.

**High Performance in Deterministic Environments**: Unlike specialized methods that can sacrifice general performance, LPM competes with or surpasses state-of-the-art baselines in standard, noise-free environments.

**Fast Adaptation**: In the experiments, the intrinsic reward signal rapidly adapts, converging to zero in both learnable and unlearnable states, which prevents the agent from getting stuck.

### Weaknesses
**Not Sufficient Experimental Setup**: The experiments seem to be insufficient as they rely on synthetic noise rather than more realistic, naturally occurring stochasticity. This questions the method's applicability to real-world problems. Therefore, additional explanation and scenarios should be suggested.

**Insufficient and Limited Experiments**: The paper's claims are based on a very small set of environments. Compared with recent researches in RL, the case studies are not satisfactory. More complex and continuous state-based environments could be adopted to extract generalization from experiments. For example, this paper omits critical, hard-exploration benchmarks (like 'Montezuma's Revenge') and lacks testing in procedurally generated or continuous control (robotics) domains.

**Risk of Local Minima in Clean Environments**: In noisy-free settings, the reward mechanism focuses on "learnable" states. This could lead to the agent "getting stuck" by focusing too much on a complex, learnable area it has already found, rather than exploring new, simpler areas (as you pointed out). This intuitive question should be addressed.

### Questions
Please, consider concerns in Weakness.

### Soundness
3

### Presentation
1

### Contribution
2
