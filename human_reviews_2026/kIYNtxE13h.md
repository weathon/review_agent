# Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Controlling high-dimensional biological and robotic systems is challenging due to expansive state–action spaces, where effective exploration is critical. Commonly used exploration strategies in reinforcement learning are largely undirected with sharp degradation as action dimensionality grows. Many existing methods resort to dimensionality reduction, which constrains policy expressiveness and forfeits system flexibility. We introduce Q-guided Flow Exploration (Qflex), a scalable reinforcement learning method that conducts exploration directly in the native high-dimensional action space. During training, Qflex traverses actions from a learnable source distribution along a probability flow induced by the learned value function, aligning exploration with task-relevant gradients rather than isotropic noise. Our proposed method substantially outperforms representative online reinforcement learning baselines across diverse high-dimensional continuous-control benchmarks. Qflex also successfully controls a whole-body human musculoskeletal model to perform agile, complex movements, demonstrating superior scalability and sample efficiency in very high-dimensional settings. Our results indicate that value-guided flows offer a principled and practical route to exploration at scale.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces QFLEX, a novel reinforcement learning method that uses a Q-guided probability flow for directed exploration in high-dimensional action spaces. The approach is motivated by the well-justified problem of vanishing exploration with undirected noise in over-actuated systems. The ideas are clearly presented, and the experimental evaluation is comprehensive, spanning from typical humanoids like Unitree H1 to a challenging 700-actuator musculoskeletal model.

### Strengths
1. The core idea of using a learned probability flow for value-directed exploration is novel and well-motivated. It effectively addresses a key limitation (vanishing signal) of isotropic exploration in high-dimensional spaces.
2. The paper is generally well-written. The main ideas are clear and easy to follow. 
3. The empirical validation is a major strength. The benchmarks are diverse and highly relevant, and the successful application to the full-body musculoskeletal model (MS-Human-700) provides strong evidence for the method's scalability and practical potential.

### Weaknesses
1. The method incurs a non-trivial computational cost. Obtaining an exploratory action $a^{(1)}$ requires calculating the gradient of $Q$ with respect to $a^{(\frac{n-1}{N})}$ for N times (Eq. 14).
Quantifying the wall-clock time compared to baselines would strengthen this analysis.
2. The comparison focuses on Gaussian-based and diffusion-based RL baselines. Including a comparison with other advanced exploration strategies for high-dimensional spaces (e.g., those based on intrinsic motivation or curiosity) could better situate QFLEX within the broader RL exploration literature and highlight its specific advantages.
3. The intuition behind learning a flow model, as opposed to simply taking a fixed number of gradient ascent steps from the current policy $\pi^{(0)}$, could be more thoroughly justified. An ablation study directly comparing the learned flow against a simple multi-step gradient ascent baseline would help isolate the benefit of the flow matching component.

### Questions
1. For the case analysis of vanishing exploration in high DoF settings, the result seems to depend on the choice of the link length $L/|\mathcal{A}|$. If we choose the length as $L/\sqrt{|\mathcal{A}|}$, we can not get the conclusion. Could the authors comment on the choice of $l_i=L/|\mathcal{A}|$.
2. The algorithm simultaneously trains the Q-function, the initial Gaussian policy $\pi^{(0)}$, and the flow model $v_w$. Since both the source and target of this flow are non-stationary, could the author provide more insight or empirical evidence on why this joint training is stable?

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
The paper introduces Qflex, a reinforcement learning algorithm designed for efficient exploration in high-dimensional continuous control. Exploration in such systems is challenging because Gaussian noise fails to explore efficiently, particularly in over-actuated settings (e.g. musculoskeletal control). Qflex addresses this by aligning exploration with value-function gradients through a probability flow formulation. The method directs exploration toward high-value regions by integrating a value-guided velocity field. Experiments on several high-dimensional benchmarks demonstrate improved sample efficiency and control performance over Gaussian and other exploration baselines.

### Strengths
- The idea of aligning exploration with the value function through a flow-based transformation is well grounded and connects flow matching with reinforcement learning in a principled way.
- The method achieves state-of-the-art performance on several high-dimensional control benchmarks, including tasks with >700 actuators.
- Qflex integrates easily into standard actor-critic frameworks such as SAC, without architectural modifications.

### Weaknesses
- While applying flow matching to online RL for exploration is interesting, closely related ideas appear in [1] and [2]. The main novelty lies in scaling to high-dimensional systems rather than a fundamentally new formulation.
- The paper does not analyze how value-guided flows improve exploration or in which regimes they outperform Gaussian or latent-space baselines.
- The qualitative figure contrasting Gaussian vs. Qflex exploration is illustrative but lacks quantitative evaluation or a controlled ablation demonstrating the behavioral difference.
- Metrics such as energy efficiency would help interpret whether exploration leads to more structured or efficient control.

[1]: McAllister, David, et al. "Flow matching policy gradients." *arXiv preprint arXiv:2507.21053* (2025).

[2]: Lv, Lei, et al. "Flow-Based Policy for Online Reinforcement Learning." *arXiv preprint arXiv:2506.12811* (2025).

### Questions
1. How does Qflex differ theoretically and empirically from [1] and [2]? Are the reported improvements due to the use of value-guided flows? A side-by-side comparison on shared benchmarks would clarify novelty.
2. Can the authors provide a small-scale toy controlled example (e.g., a 50-100 DOF) comparing Gaussian vs. Qflex noise? Visualizing trajectories and action-space evolution would concretely illustrate how Qflex produces more directed exploration. Additionally, can the authors discuss how much Qflex depends on the value function at the beginning of the training (as the critic is inaccurate) and when it transitions from random (Gaussian-like) to value-guided? How far into the future (short vs. long-term value) does the velocity field effectively look?    
3. Does Qflex noise exhibit the same correlation structure as the learned action covariance? Is it aligned with principal action subspaces or dependent on state geometry? This would help understanding whether Qflex adaptively focuses exploration on effective actuator subsets.
4. How does solving the flow ODE scale with action dimension |A|? Does QFlex incur additional runtime or convergence cost compared to SAC, or are gains due to more efficient sampling?
5. In over-actuated systems, exploration efficiency is often correlated with energy-efficient policies (as shown in [3]). Does Qflex yield more energy-efficient or smoother action trajectories?
6. Are all baselines trained under identical regimes (fully online)? How are comparisons to DEP-RL (which is an offline method) made? Would Qflex still outperform other exploration baseline if trained with PPO instead of SAC?

[1]: McAllister, David, et al. "Flow matching policy gradients." *arXiv preprint arXiv:2507.21053* (2025).

[2]: Lv, Lei, et al. "Flow-Based Policy for Online Reinforcement Learning." *arXiv preprint arXiv:2506.12811* (2025).

[3]: Chiappa, Alberto Silvio, et al. "Latent exploration for reinforcement learning." *Advances in Neural Information Processing Systems* 36 (2023): 56508-56530.

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
3

### Summary
The work addresses a critical challenge in high-dimensional continuous control: achieving efficient exploration in complex state–action spaces. Common practices such as adding Gaussian noise to policy actions often fail to promote meaningful exploration—an issue well recognized in the community and clearly demonstrated by the authors. In contrast, the proposed method leverages action samples guided by estimated Q-value improvements, enabling more directed and effective exploration.

The method outperforms Gaussian-based and diffusion-based RL baselines across several high-dimensional locomotion tasks and even achieves control of a 700-actuator full-body human musculoskeletal model, which is particularly impressive and highlights the practical advantages of the proposed approach.

### Strengths
The idea is well-motivated, clearly established, and novel. The problem it addresses is central to scaling reinforcement learning policies to more complex and realistic control tasks. Leveraging Q-guided flows to generate actions for directed exploration is both natural and insightful, with its benefits demonstrated across several high-dimensional tasks that are typically challenging for RL policies to learn from scratch. Moreover, the method introduces only minimal modifications to a standard actor–critic framework, further enhancing its practicality and applicability.

### Weaknesses
The primary weakness of the proposed method lies in its strong reliance on accurate Q-gradient estimation. During early training stages, or in scenarios where Q-gradients are unreliable, the resulting action updates may misguide exploration and ultimately hinder policy improvement. Although the paper discusses the use of batch normalization to stabilize Q-learning, a more thorough analysis of potential failure cases or mitigation strategies would further strengthen the work.

### Questions
The experiments are primarily conducted on tasks with dense reward signals, which are common in locomotion benchmarks. It might be unclear how the method would perform in sparse-reward environments, where accurate Q-function learning is more challenging. In such settings, would Q-guided flows still yield effective exploration, or would their reliance on precise Q-gradients diminish their advantages?

From Algorithm 2, the policy appears to always execute the final transported action from the Q-guided flow when interacting with the environment. This design seems heavily biased toward exploitation, as each exploratory step moves strictly in the direction of higher estimated value. Could this cause the exploration to collapse prematurely and converge to sub-optimal policies, especially when the Q-gradient is inaccurate early in training? Have you considered sampling actions from intermediate states along the flow trajectory—rather than only at the terminal point—to preserve exploration breadth? If so, what empirical effects and trade-offs did you observe?

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
3

### Summary
This paper proposes a scalable online reinforcement learning method （QFLEX）to efficiently explore high-dimensional continuous control systems. Through a case analysis, the paper demonstrates that the undirected stochasticity leads to vanishing exploration in high-dimensional continuous control. The proposed method, QFLEX, outperforms six other baselines across multiple high-dimensional control benchmarks. Additionally, QFLEX successfully controls a full-body musculoskeletal model with 700 actuators to perform complex tasks, such as walking and ballet dancing.

### Strengths
1.	The paper presents clear and concise expression.
2.	The paper presents thorough comparative experiments, testing QFLEX on six benchmark tasks and comparing it with eight baseline methods.
3.	The proposed QFLEX method efficiently addresses exploration in high-dimensional action spaces, overcoming the inefficiencies of undirected exploration.

### Weaknesses
1. Although QFLEX performs well in simulated tasks, the paper does not provide sufficient validation on real-world systems.

### Questions
In the Unitree H1–Run/Balance task, the results presented in the paper indicate that QFLEX performs notably better than the other methods. Whether Proximal Policy Optimization (PPO) was tested as well ? It may strengthen the paper's contributions to include a comparison with PPO.

### Soundness
3

### Presentation
4

### Contribution
3
