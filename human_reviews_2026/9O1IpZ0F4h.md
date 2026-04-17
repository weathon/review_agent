# NFPO: Stabilized Policy Optimization of Normalizing Flow for Robotic Policy Learning

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Deep Reinforcement Learning (DRL) has experienced significant advancements in recent years and has been widely used in many fields. In DRL-based robotic policy learning, however, current *de facto* policy parameterization is still multivariate Gaussian (with diagonal covariance matrix), which lacks the ability to model multi-modal distribution. In this work, we explore the adoption of a modern network architecture, i.e. Normalizing Flow (NF) as the policy parameterization for its ability of multi-modal modeling,  closed form of log probability and low computation and memory overhead. However, naively training NF in online Reinforcement Learning (RL) usually leads to training instability. We provide a detailed analysis for this phenomenon and successfully address it via simple but effective technique. With extensive experiments in multiple simulation environments, we show our method, NFPO could obtain robust and strong performance in widely used robotic learning tasks and successfully transfer into real-world robots.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes NFPO, an on-policy RL method that replaces the standard Gaussian policy with a normalizing flow (specifically, RealNVP) inside PPO. The authors experimentally analyze why naïvely plugging a flow into PPO is unstable, and proposed techniques to stabilize training by bounding the scale network with a tanh transform and a limit parameter l. Experiments span URG and MuJoCo Playground, and compare NFPO against PPO baselines. NFPO often matches or beats PPO on several locomotion/manipulation tasks; qualitative studies (gridworld and UR10 reaching) suggest more multi-modal behavior; and they show sim-to-real transfers to Unitree robots. Wall-clock cost increases by ~19% vs. PPO on g1.

### Strengths
- The proposed technique is simple and clearly presented, with writing that is easy to follow.
- The paper includes detailed studies on different design choices (e.g., different methods to stablize scaling function, different hyper-parameter choices).
- NFPO offers slight performance gains compared to PPO, especially for certain control tasks such as g1-joystick, h1-joystick, and G1JoyStickRoughTerrain.

### Weaknesses
- While the paper may be among the first to pair normalizing flows with on-policy RL, similar ideas have been explored extensively in off-policy settings, e.g., [1-3]. Without a clearer technical distinction or theoretical contribution beyond the training stabilizations, the novelty of this work seems modest.

- Reported gains are small and the comparisons are not run on widely accepted benchmarks with commonly adopted baselines (e.g., mixture Gaussian PPO variants, off-policy methods). As a result, it’s hard to assess the practical significance of the method.

- The sim-to-real experiments omit key operational specifics, e.g., task definitions, control frequency/latency, domain-randomization settings, and failure rates, making reproducibility and reliability difficult to judge.

---

**Minor Errors:**    
- “the $\exp(s_\theta (x_d))$ used in RealNVP apply …” → “the $\exp(s_\theta (x_d))$ used in RealNVP applies …” in Line 171.   
- “overfiting” → "overfitting"; appears in Lines 162-174.   
- “logprobability” → "log-probability" (hyphen) in Line 177.

### Questions
- Please provide the state/action space definitions and concise descriptions of each evaluated environment. Also explain why NFPO strongly outperforms PPO on G1JoyStickRoughTerrain yet underperforms on MJP-PandaOpenCabinet—what properties of these tasks favor or hinder flow policies?

- Please include learning curves on common MuJoCo benchmarks (e.g., Hopper, HalfCheetah, Walker2d, Ant, Humanoid) and compare against widely used baselines (e.g., SAC, TD3) to contextualize performance.

- The experiment presented in Fig. 8 appears to modify baseline defaults without retuning. Because online RL is sensitive to hyperparameters, fair comparisons should use the best settings found within a shared search space. How were the baseline hyperparameters chosen, and why is that selection representative?

- Section 5.5 claims NFPO supports deterministic action sampling, but for non-volume-preserving flows the mode (argmax density) is generally nontrivial [3]. How is deterministic sampling implemented for RealNVP in practice?

- Fig. 5 suggests NFPO and PPO achieve similar returns, while Table S2 reports ~19% sampling overhead for flows. Under what conditions (task characteristics, exploration regime, multimodality) are normalizing-flow policies clearly preferable than Gaussian policies?

- Beyond the reported settings, how sensitive is performance to flow depth (e.g., 2/6/8 layers) and hidden sizes? Please include a brief ablation.

- Did you evaluate other efficient multimodal policy families (e.g., consistency models [4,5] or related diffusion variants)? How do they compare to NFPO in stability, sample efficiency, and runtime?

---

**References:**

[1] Haarnoja et al. Latent space policies for hierarchical reinforcement learning. ICML 2018. \
[2] Mazoure et al. Leveraging Exploration in Off-policy Algorithms via Normalizing Flows. CoRL 2019. \
[3] Chao et al. Maximum Entropy Reinforcement Learning via Energy-Based Normalizing Flow. NeurIPS 2023. \
[4] Song et al. Consistency Models. ICML 2023.\
[5] Ding et al. Consistency Models as a Rich and Efficient Policy Class for Reinforcement Learning. ICLR 2024.

### Soundness
2

### Presentation
2

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
This paper aims to integrate the multi-modal modeling capabilities of Normalizing Flows (NF) into robotic PPO policies. The authors first identify the training instability of combining NF directly with PPO, attributing it to the exploding Jacobian determinant caused by the $exp(s)$ term in RealNVP.  To solve this, the authors propose NFPO, which uses an ${tanh}$ activation function to constrain the output of $s$, thereby stabilizing training. Experiments show that NFPO performs robustly on several simulation tasks, matching or exceeding  PPO 3, and was successfully transferred to a real-world robot.

### Strengths
Problem Diagnosis and Solution: The paper clearly diagnoses the root cause of instability when combining NF with PPO (exploding determinant) and proposes a simple, effective solution ($tanh$ activation) .

Solid Experimental Validation: The paper provides comprehensive benchmarks on 9 tasks across multiple simulators (IsaacGym, Mujoco Playground) 7and includes thorough ablation studies.

Real-World Deployment: The policy was successfully transferred from simulation to a real Unitree G1 robot, strongly demonstrating the algorithm's robustness and practical value.

### Weaknesses
Limited Innovation: The work is primarily an application and engineering-level adaptation of RealNVP for policy optimization, rather than a fundamental algorithmic innovation. The core stabilization technique ($s\_tanh$) is a known trick.

Ambiguous Multi-modal Advantage: Although multi-modality was shown in specific tasks (Sec 5.3), its direct link to the performance gains in the main benchmarks (Sec 5.2) is not clear.

### Questions
Regarding Real-World Deployment: You mentioned using a "deterministic version" in Sec 5.5.a) How was this "deterministic version" obtained (e.g., $z=0$)? Does this imply that the stochastic or multi-modal policy is unstable in the real world?

Regarding FPO/Meow Comparison: In Sec 5.4, NFPO outperformed FPO and Meow in robustness. Were FPO/Meow given the same level of hyperparameter tuning as NFPO, or were their default parameters used?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes NFPO algorithm, which parameterizes the policy-network with a Normalizing Flow to capture the multi-modal action distributions. The authors diagnose the instability in RealNVP-based flows inside PPO, and provide a simple but effective solution: normalizing the $s_θ(x)$ output to make it in a proper range. Extensive experiments on multiple robotics simulators and with multiple seeds show a stronger and more stable performance on several locomotion tasks, with mixed results on some manipulation tasks and sim-to-real deployments on Unitree hardware.

### Strengths
- The authors propose NFPO, a new framework that integrates Normalizing Flows (NF) into PPO for robotic multi-modal policy learning, and further analyze the causes of its training instability and introduce effective stabilization techniques. The authors provide a clear problem formulation for the multi-modal action distribution for on-policy control, and a simple and reproducible solution by swapping policy head and bounding the scale output of the flow.

- Comprehensive experiments are conducted on several widely used simulation environments. With the same configuration settings, NFPO achieves competitive performance compared with state-of-the-art Gaussian-based PPO implementations. Real-world validation also demonstrates that policies trained with NFPO can be successfully transferred to physical robots. These extensive experiments on multiple robotics simulators and deployments show the effectiveness of NFPO in capturing the multi-modal action distributions and stabilizing learning.

### Weaknesses
- Considering this is the ICLR submission, the theoretical analysis may be more important than the engineering implementation and results. But the theoretical analysis for the algorithm and mathematical proofs in this paper are limited, e.g., one may expcet to see the analysis on the stability of NFPO and the reason why adding entropy loss in NFPO does not bring a significant performance difference.

- In some tasks like  MJP-PandaOpenCabinet and MJP-Go1JoystickRoughTerrain, NFPO fails to learn a good policy, which shows the limitation on the generalization ability of NFPO.

- Runtime overhead is reported but not decomposed. And there is no complexity analysis vs. the action dimension or number of coupling layers.

### Questions
- As stated in weaknesses, the theoretical analysis and mathematical proofs in this paper are limited. Can the authors provide more theoretical analysis on why NFPO can work better? 

- Why tanh is more robust than clip in the authors' implementation? 

- Can the authors provide more experiments with different hyperparameter combinations to show the robustness of NFPO?

### Soundness
2

### Presentation
3

### Contribution
2
