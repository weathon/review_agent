# Mitigating Distribution Shifts: Uncertainty-Aware Offline-to-Online Reinforcement Learning

- Avg Score: 4.40
- Decision: Reject
- Scores: 5, 3, 5, 3, 6

## Abstract
Deploying reinforcement learning (RL) policies in real-world scenarios, particularly through offline learning approaches, faces challenges due to distribution shifts from training environments. Past approaches have shown limitations such as poor generalization to out-of-distribution (OOD) variations or requiring extensive retraining on target domains. We propose Uncertainty-aware Adaptive RL, UARL, a novel offline RL pipeline that enhances OOD detection and policy generalization without directly training in OOD environments. UARL frames distribution shifts as OOD problems and incorporates a new OOD detection method to quantify uncertainty. This approach enables iterative policy fine-tuning, starting with offline training on a limited state space and progressively expanding to more diverse variations of the training environment through online interactions. We demonstrate the effectiveness and robustness of UARL through extensive experiments on continuous control tasks, showing reliability in OOD detection compared to existing method as well as improved performance and sample efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents a novel approach aimed at solving the OOD problem faced by deploying reinforcement learning strategies in real-world scenarios when the distribution of training environments is shifted. The proposed approach tackles this issue by adopting a new diversity ensemble Q-network approach to OOD detection. Furthermore, the method incorporate an iterative policy fine-tuning method that starts with offline training in the original environment and gradually scales up to more stochastic environments through online interactions. Experimental results show that this approach outperforms the Baseline algorithm in Mujoco environments with randomized environmental hyperparameter and typically requires fewer samples to converge.

### Strengths
1.This paper propose a novel OOD detection method and an iterative online policy fine-tuning training framework.
2.Good experimental results are obtained on Mujoco environments with randomized environmental hyperparameter, verifying the validity of the method.
3.The writing is good.

### Weaknesses
1. It should be further shown, either theoretically or experimentally, why the diversity term $\mathcal{L}^{\text{RL}}_{\text{div}} $ in Eq. 5 allows the ensemble Q network to learn the value of diversity. Intuitively, minimising $\text{exp}(-\Vert Q_i(s, a) - (r + Q_i(s^\prime, a^\prime)) \Vert^2/2\delta^2)$ will allow the Q network to not converge quickly to a certain value on the repulsive dataset, but it does not guarantee that the ensemble Q network learns diverse values.
2. Further clarification is needed for how to calculate $V_Q$ in Algorithm 2 and how to calculate critical variance in uncertainty estimation experiments.
3. The ablation experiments in Appendix B.3 are not detailed enough. The training curves for different parameter combinations should be differentiated to illustrate the algorithm's parameter sensitivity to $\lambda$ and $\delta$ during training.
4. For each $E_i$, the randomized environmental hyperparameter range is determined without a common metric but as a hyperparameter, which may require a lot of time for online tuning for complex scenarios.

### Questions
1. According to Eq. 5, $R(s, a)$ is not sampled from the dataset $\mathcal{D}$, but the general Q-function update for offline RL, as in Eq. 1, is to use the sampled $r$. Is this a mistake here?
2. Is there a performance or computational advantage of UARL over direct processing of $E_\omega$'s using the Robust RL algorithm or the algorithm with domain randomization techniques? Can this be illustrated experimentally?
3. Notice that in offline training (1st iteration), EDAC performs much worse than CQL and TD3+BC in many environments, which doesn't seem to match the experimental results in the EDAC article?
4. The experiments in this paper were all performed in Mujoco, how do we obtain the real-world demonstration dataset $\mathcal{D}_\omega$ in the simulation environment like Mujoco?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper deals with the distribution shift issue in reinforcement learning (RL). The authors introduce an approach called Uncertainty-aware Adaptive RL (UARL) that enhances policy generalization across diverse variations of a given environment. UARL views distribution shifts as OOD problems and integrates an OOD detection method to quantify uncertainty, i.e., Q-value variance. UARL realizes diversity in critics via the DENN method. The authors claim that UARL enables iterative policy fine-tuning, starting with offline training on a limited state space and progressively expanding to more diverse variations of the same environment through online interactions. The authors demonstrate the effectiveness of UARL through some experiments on continuous control tasks, showing improved performance and sample efficiency compared to existing methods.

### Strengths
## Pros

This paper enjoys the following advantages,

- This paper is well-written. The presentation of this paper is very good and of high quality. The figures are very nice and helpful. Some of the illustration figures significantly convey the idea and core design of UARL, e.g., Figure 1, Figure 2
- This paper is easy to read and easy to follow
- The authors provide open-source codes in the anonymous website, and I believe that the results reported in this paper are reproducible

### Weaknesses
## Cons

Despite the aforementioned advantages, this paper has the following flaws.

- **Offline-to-online RL or off-dynamics RL?** This paper claims that they focus on the offline-to-online setting, while it seems that they actually are dealing with off-dynamics RL [1]. The offline-to-online RL typically refers to training a policy offline from a static dataset and then fine-tuning it with some extra online interactions in the same environment. Instead, the authors conduct experiments by modifying the environmental parameters, which essentially constructs dynamics shifts. The experimental setting resembles that in off-dynamics RL [2,3,4]. It seems unclear to me whether it is suitable to name the paper *offline-to-online* or *off-dynamics*.
- **Insufficient related work and Limited novelty.** The authors emphasize that the proposed method can enhance the safety and robustness of RL, but it includes too few related works on safe offline/offline-to-online RL and robust offline/offline-to-online RL. Meanwhile, I have doubts about the novelty of this work. The authors progressively increase hyperparameter randomization of the environment (e.g., friction) when the variance between the Q-ensemble is large and terminates until the policy is safe enough to be deployed. Such an idea resembles [5], which progressively adapts its learned policy by modifying the parameters of the environment. Furthermore, I am a bit confused about the benefits of parameter randomization against domain randomization. If the user can adjust the parameters of the environment, then why not directly use domain randomization? I would expect reasonable justifications for the design of the tasks here. Furthermore, the diversity term is not novel, it is borrowed directly from the existing literature. These all together make the contribution of this paper somewhat limited.
- **Lacking baseline algorithms.** As commented above, this paper claims that it addresses the offline-to-online RL but actually focuses on the off-dynamics RL setting, the authors should include the following baselines,
  - baselines on off-dynamics RL, e.g., [2,3,4]. This is vital to show the effectiveness of the UARL in terms of policy generalization to the target domain
  - RLPD [6], which is a method specially designed for learning with offline data and online interactions with the environment. This baseline is important since it exhibits superior performance given the experimental setting described in this paper (offline data and online interactions). Based on my experience, RLPD can achieve quite strong performance even when there exists dynamics shifts between the offline data and the online environment. Involving this baseline can justify the necessity of the components adopted in UARL (otherwise, one can directly use RLPD for deployment)
  - baselines on safe RL and robust RL. The authors claim that UARL can enhance the safety and robustness of the executed actions, while they do not include any safe RL or robust RL methods for comparison, making it hard to see the rationality and effectiveness of UARL
  - baselines on offline-to-online RL. Unfortunately, this paper also does not include offline-to-online RL methods as valid baseline methods. It is hard to tell the true effectiveness of UARL without these methods, e.g., [7,8,9]
- (minor) **Lacking theoretical justifications.** There is no theoretical analysis of the UARL. I do not want to blame the authors too much on this point. I understand that this paper may set the focus mainly on the empirical side, but including some theoretical analysis can strengthen this paper.
- (minor) **Other issues.**
  - in Equation 5, you wrote $R(s,a)$ in the bellman error, while $r$ in the diversity term $\mathcal{L}_{div}^{RL}$. I think they should be identical, right?
  - the authors do not discuss the limitations of their method in the main text or the appendix. It is important to acknowledge both the advantages and the limitations of the proposed method. 
  - the performance improvement of UARL seems limited and incremental on some tasks (e.g., see Figure 3)
  - UARL can still suffer from performance degradation during the fine-tuning phase (e.g., see Figure 5)

Given the above concerns, I vote for rejection since I believe that this paper needs a significant revision before being accepted for possible publication.

[1] Off-dynamics reinforcement learning: Training for transfer with domain classifiers. ICLR

[2] When to trust your simulator: Dynamics-aware hybrid offline-and-online reinforcement learning. NeurIPS

[3] Cross-domain policy adaptation via value-guided data filtering. NeurIPS

[4] Cross-domain policy adaptation by capturing representation mismatch. ICML

[5] Revolver: Continuous evolutionary models for robot-to-robot policy transfer. ICML

[6] Efficient online reinforcement learning with offline data. ICML

[7] Offline-to-online reinforcement learning via balanced replay and pessimistic q-ensemble. CoRL

[8] Bayesian Design Principles for Offline-to-Online Reinforcement Learning. ICML

[9] Proto: Iterative policy regularized offline-to-online reinforcement learning. Arxiv

### Questions
Please c.f. the comments above. Besides, I have the following questions,

- In Lines 184-185, you wrote, *disagreement among ensemble models, particularly at the boundary of the training distribution*, what do you exactly mean by *at the boundary of the training distribution*?
- what are the advantages of the diversity term in Equation 5 compared to other diversity terms? (e.g., the diversity term used in the EDAC paper) The authors ought to justify the advantages of doing so rather than using some other methods.
- how can the authors tell that the uncertainty measurement provided in this paper is valid? It would be better to compare against some other uncertainty estimation methods and visualize the uncertainty measurement for a better comparison
- do you have any parameter study on the threshold parameter in Algorithm 2? How it can affect the performance of the agent? Do we need to tune this hyperparameter per task? How can we ensure that the policy is safe when $V_Q \le {\rm threshold}$?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces Uncertainty-aware Adaptive RL (UARL), an innovative framework to tackle distributional shifts and out-of-distribution (OOD) issues when deploying reinforcement learning (RL) policies in real-world environments. This is accomplished by implementing OOD detection to quantify policy uncertainty and iteratively refine high-uncertainty regions (of the state space), adapting the policy
for safe and effective performance deployment. UARL demonstrates several notable advancements,
- A method for quantifying policy uncertainty using OOD detection.
- An offline-to-online (O2O) adaptation strategy that balances online and offline data, utilizing a diverse ensemble of critics to better handle distributional shifts.
- Experiments on MuJoCo continuous control tasks that validate UARL’s effectiveness in terms of performance, robustness, and sample efficiency.

### Strengths
- UARL presents a compelling approach to address the challenges of deployment a policy in RL. The progressive expansion of state space via repulsive locations and a balanced replay buffer to manage data distribution shifts are novel and theoretically sound.
-The usage of an ensemble of diverse critics to perform OOD detection and policy refinement represents a robust methodology that has support from the experimental result

### Weaknesses
- The paper could better highlight its unique contributions compared to existing OOD and ensemble-based offline RL methods. A clearer differentiation of UARL's specific advancements would help underscore its novelty within the landscape of similar approaches.

- The experimental validation, limited to few environments such as the Ant-v4 and HalfCheetah-v4 environments, may not fully capture the method’s effectiveness across a diverse range of tasks. Extending the experiments to include more varied environments would provide a more comprehensive assessment and enhance the generalizability of the results.

- A comparison with recent state-of-the-art methods, such as PBRL[1], RORL[2], would strengthen the empirical evaluation. By benchmarking UARL against PBRL and similar approaches, the paper could provide a more robust validation of its improvements in uncertainty handling and performance stability.

[1] Pessimistic Bootstrapping for Uncertainty-Driven Offline Reinforcement Learning
[2] RORL: Robust Offline Reinforcement Learning via Conservative Smoothing

### Questions
1. Comparing the computational overhead of your method with that of baseline algorithms would strengthen your work. Could you include this information to provide a clearer understanding of its efficiency?

1. Does this algorithm fall within the scope of Offline Reinforcement Learning? If so, it would be helpful to clarify its placement within the Offline Reinforcement Learning landscape. Enhancing the abstract and introduction to better position the algorithm within this broader context would significantly improve the clarity and impact of your paper.

I am open to raising my score based on these improvements.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
In this paper, a novel RL pipeline, Uncertainty-aware Adaptive RL (UARL), has been proposed to enhance policy generalization across diverse variations of a given environment. UARL frames distribution shifts as OOD issues and integrates a new OOD detection method to quantify uncertainty. This method enables iterative policy fine-tuning, beginning with offline training on a limited state space and gradually expanding to more diverse variations of the same environment through online interactions.

### Strengths
The problem raised in this paper is important and the experiments are solid.

### Weaknesses
Weakness:

1) The core problem in this paper, i.e., distributional shift, and many important concepts are not discussed in detail. In the introduction, the authors discuss about the theoretical shortcomings of robust RL, safe RL, and the distributional shift in offline2online RL, but they ignore the discussion about the relationships between these concepts. For example, what is the relationship between the robust RL and the distributional shift in offline setting? Furthermore, what is the difference between the distributional shift problems in offline RL and offline2online RL settings? Why the proposed method could successfully solve the problem of distributional shift? Please answer this question from a high-level view.

2) In offline (to online) RL, the uncertainty quantifier is defined clearly as the upper bound of the error produced by the empirical Bellman operator (see [1], Eq.(4.1)). Then we may concern that whether the uncertainty defined in this paper's Eq.(5) has the relationship with the uncertainty quantifier as we have known in [1]? Does it a valid uncertainty quantifier theoretically? The author should discuss about this point.

[1] Jin. et al., Is Pessimism Provably Efficient for Offline RL.

3) In offline RL, there have been many uncertain-aware methods to deal with the distributional shift problem, such as [2] and [3]. In the list two works, they both penalize the OOD actions by the constructed uncertainty quantifiers. So in our view, the method in this work is not beyond the scope of these methods and lack of the sufficient discussion with the advantage over the existing uncertain-aware methods.

[2] Bai. et al., Pessimistic bootstrapping for uncertainty-driven offline reinforcement learning.
[3] Sun. et al., Model-Bellman Inconsistency for Model-based Offline Reinforcement Learning.

4) the convergence property of the proposed algorithm should be discussed, especially line 8 in Algorithm 2 - what if this condition is never voilated?

### Questions
see above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a novel pipeline to detect OOD environment variations and gradually fine-tunning the agent until high confidence safe deployment is possible.

### Strengths
1. The overall method is well-motivated and clearly stated.
2. Weighting samples in the normal dataset and repulsive dataset differently is intuitive and is demonstrated to be effective empirically.
3. Using a set of critiques and its variances as a measure of environmental uncertainty explore new possibilities from the existing DENN method.

### Weaknesses
1. Repulsive locations discussion could be written more formally and thus more concisely. The current version is a bit too dense verbally. I'm also a bit confused by Figure 2, while it's a nice visual, is it something derived from the experiments or is it just a conceptual illustration?
2. Lack of related works: changing environment parameters to achieve repulsive locations is quite related to the literature on curriculum learning. A good survey to start with is https://arxiv.org/pdf/2003.04960. Also, blindly varying the environmental parameters may lead to unexpected harmful environments dampening the agent training: https://openreview.net/forum?id=hp4yOjhwTs&noteId=vZMeHQbnJK  
I would suggest authors add a subsection for curriculum reinforcement learning in the related work for a more thorough introduction to the problem backgrounds.

### Questions
1. Typo: Algo 1 line 7, should it be "OOD" instead of "ODD"?
2. How do you define "progressively expanding the randomization range" for different environment parameters? More specifically, increasing friction by 1% and increasing the agent's mass by 1% may have vastly different impacts on the task difficulties. Could you discuss more on the relative impact of changing each parameter to the environment difficulty?

### Soundness
3

### Presentation
3

### Contribution
3
