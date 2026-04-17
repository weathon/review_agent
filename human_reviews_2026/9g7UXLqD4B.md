# RobustVLA: Robustness-Aware Reinforcement Post-Training for Vision-Language-Action Models

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Vision-Language-Action (VLA) models have recently emerged as powerful general-purpose policies for robotic manipulation, benefiting from large-scale multi-modal pre-training. However, they often fail to generalize reliably in out-of-distribution deployments, where unavoidable disturbances such as observation noise, sensor errors, or actuation perturbations become prevalent. While recent Reinforcement Learning (RL)-based post-training provides a practical means to adapt pre-trained VLA models, existing methods mainly emphasize reward maximization and overlook robustness to environmental uncertainty. In this work, we introduce RobustVLA, a lightweight online RL post-training method designed to explicitly enhance the resilience of VLA models. Through a systematic robustness analysis, we identify two key regularizations: Jacobian regularization, which mitigates sensitivity to observation noise, and smoothness regularization, which stabilizes policies under action perturbations. Extensive experiments across diverse robotic environments demonstrate that RobustVLA significantly outperforms prior state-of-the-art methods in robustness and reliability. Our results highlight the importance of principled robustness-aware RL post-training as a key step toward improving the reliability and robustness of VLA models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces RobustVLA, a robustness-aware reinforcement post-training framework for VLA models. The authors identify two key sources of environmental disturbance—observation noise and action noise—and propose two corresponding regularizations: Jacobian regularization (to reduce input sensitivity) and smoothness regularization (to stabilize model updates).

### Strengths
* The paper is well-motivated and theoretically grounded, clearly linking robustness analysis with practical algorithmic design.
* The paper's focus on robust post-training of VLA models fills an important gap between pure reward maximization and real-world reliability.

### Weaknesses
* The experiments, though extensive, are all in simulation (LIBERO), so the claim of “real-world robustness” is not empirically validated on physical robots.
* The method’s computational overhead due to Jacobian regularization is also not clearly quantified.

### Questions
* How sensitive is the method to the choice of Jacobian and smoothness weights (α, β)? Is there a theoretical or empirical guideline for setting them across different tasks?
* I know this benchmark (https://github.com/sylvestf/LIBERO-plus?tab=readme-ov-file) is new, but could you test the method on this benchmark and report the performance?
* It seems like all models are added with a lot of noises during rollout, could you provide some data about what is the average noise level during rollout in reality?
* Will the loss term for enhancing robustness harm the model's general performance (i.e., the performance w/o noise)

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
2

### Summary
This paper investigates the robustness of VLA against observation space transformations and action space perturbations. The method introduces gradually adjusted noise into the training environment and proposes two regularization terms. Evaluations on LIBREO with varying noise levels show that the proposed method outperforms other baselines in success rate.

### Strengths
- This paper highlights the critical robustness issue in VLA models. 
- The method builds on a reliable, widely adopted VLA baseline. 
- The paper introduces five carefully designed observation perturbations in a common benchmark.

### Weaknesses
- The perturbation assumption may not be practical and seems inconsistent with the experimental setup. The state deviation is oversimplified by adding noise to the state space, and the dynamics are assumed to be Lipschitz continuous. 
- There are some issues in the proof of the Theorem 1, and it appears inconsistent with the algorithm. Please refer to questions for detail.
- While the study of robustness/generalization is an established topic in visual reinforcement learning [1,2,3], adding noise to the training environment to improve evaluation performance is a well-explored approach. The specific challenges of this problem in the VLA model are not clearly addressed.

[1] Yuan Z, Ma G, Mu Y, et al. Don’t touch what matters: Task-aware lipschitz data augmentationfor visual reinforcement learning[C]// IJCAI, 2022.

[2] Fan L, Wang G, Huang D A, et al. SECANT: Self-Expert Cloning for Zero-Shot Generalization of Visual Policies[C]//ICML, 2021.

[3] Hollenstein J, Auddy S, Saveriano M, et al. Action noise in off-policy deep reinforcement learning: Impact on exploration and performance[J]. arXiv preprint arXiv:2206.03787, 2022.

### Questions
- In the proof of Theorem 1 (Lines 781-783), the authors use a deterministic policy by substituting the action with the policy symbol directly. However, their method employs the PPO algorithm, which is a classical stochastic policy optimization method. Could the authors clarify this difference? 
- In the proof of Theorem 1 (Lines 781-782), the derivation to $\lambda \cdot \parallel \tilde{s}_t – s_t \parallel$ is unclear. Could the authors provide further clarification, and is there a risk that this results in too loose a bound? 
- What is the difference between the proposed Jacobian penalty and the widely used clipped gradient norm technique? 
- In the Introduction, the authors mention the out-of-distribution problem involving "unseen objects, novel environments, and different robot embodiments." However, in the experiments, they only inject simple noise into the observation and action space, along with image transformations. It would be helpful to provide a clearer and more consistent definition of the out-of-distribution scenario.

### Soundness
2

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
4

### Summary
This paper introduces RobustVLA, an online RL algorithm that improves the robustness of VLA models. The authors claim pre-trained VLA models fail in OOD settings due the the sensitivity to observation noise and action disturbance. To address this, they first applied theoretical analysis to bound the gap caused by noise, and then propose RobustVLA by introducing Jacobian regularization and policy smooth.

Their contributions are:

1. introduce RobustVLA, a robust RL method
2. conduct three analyses
3. experiment results show both components are effective.

### Strengths
1. The robustness of VLA is important.

2. Most of the idea makes sense.

3. Experiment results are good. And the authors include ablation studies.

Despite the flaws in section 4.1 (see weakness), this paper is still understandable, and the method is somewhat reasonable. I believe this paper is marginally above the acceptance threshold.

### Weaknesses
1. Section 4.1, the motivation to introduce the bounded Jacobian ($\left\lVert \nabla_s \pi_t(s) \right\rVert \leq \lambda$) and $\left\lVert \pi_t - \pi_{t-1} \right\rVert_\infty \leq \delta_t$ is not explained. And not all of the notations are defined.

2. Why do we need to introduce $\left\lVert \pi_t - \pi_{t-1} \right\rVert_\infty \leq \delta_t$? I believe $\left\lVert \nabla_s \pi_t(s) \right\rVert \leq \lambda$ is already sufficient to bound the gap even when actions are perturbed.

3. What is the connection between Theorem 2 and the design of $\mathcal{R}_{smooth}(\theta)$. They are telling completely different things. (1) In Theorem 2, t denotes the step in a trial. (2) But in the latter one, the comparison is between the old policy and the current policy.

Overall, these weaknesses are my major concerns. I would not mind if the paper is rejected.

### Questions
1. J = sum of rewards in H steps? You may want to explicitly define all the functions and variables in Section 4 (e.g. J, H, ...)
2. L437 `””` typo
3. How to interpret Figure 4 (b)(c)? All points look random.

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
3

### Summary
This paper addresses the critical issue of Vision-Language-Action (VLA) models' sensitivity to observational noise and actuation perturbations in real-world deployments. The authors propose RobustVLA, a method that enhances model robustness by incorporating Jacobian regularization and smoothness regularization into online reinforcement learning (RL) post-training. The work is grounded in theoretical robustness analysis, which rigorously examines the impact of perturbations on performance to motivate the design of the corresponding regularizers. Extensive experiments on the LIBERO simulation platform demonstrate the method's superior performance under a variety of perturbation settings.

### Strengths
1. High Practical Relevance: The work astutely targets a well-known and significant vulnerability of VLA models: their fragility to observational and action perturbations in out-of-distribution (OOD) scenarios. The focus on enhancing robustness has substantial practical value for real-world robotic deployment.

2. Solid Theoretical Foundation: The theoretical analysis is a key strength. The authors establish explicit upper bounds on the performance gap under observation perturbations, action perturbations, and their combination. This provides a principled foundation and clear motivation for the proposed regularization terms.

3. Comprehensive Experimental Evaluation: The empirical validation is thorough. Experiments are conducted across diverse task suites within the LIBERO benchmark, incorporating multiple types of observation and action perturbations. The comparison against a wide array of state-of-the-art offline and online baselines makes the results compelling and convincing.

### Weaknesses
1. The authors note that computing the Jacobian directly on high-dimensional pixel inputs is prohibitively expensive and instead calculate it on the low-dimensional embeddings from the Llama-2 encoder used by OpenVLA-OFT. However, the paper fails to specify precisely which layer's output is used as the surrogate for the states. This lack of detail makes it difficult to fully assess the implementation's validity.

2. All experiments rely on a single, fixed pre-trained Llama-2 encoding. The work does not validate the inherent robustness of this specific encoder itself, nor does it demonstrate that the proposed method remains effective when applied to VLA models built upon different visual encoders. This raises concerns about the generalizability of the approach beyond the specific architecture tested.

3. The action perturbation is modeled exclusively as additive white Gaussian noise. Real-world robotic systems often exhibit more complex structured perturbations, such as correlated noise, latency, systematic biases and so on. The analysis and methodology do not account for these more challenging and realistic disturbance types, limiting the claimed robustness's applicability.

4. Lack of Performance-Robustness Trade-off Analysis: A critical piece of analysis is missing: the performance of RobustVLA in a completely perturbation-free, nominal environment. Without comparing its success rates to the baselines under these ideal conditions, it is impossible to evaluate the potential performance trade-off incurred by the robustness enhancements.

### Questions
Please see the weaknesses section

### Soundness
3

### Presentation
2

### Contribution
2
