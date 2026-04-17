# TD-M(PC)$^2$: Improving Temporal Difference MPC Through Policy Constraint

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 8, 4

## Abstract
Model-based reinforcement learning (MBRL) algorithms that integrate model predictive control with learned value or policy priors have shown great potential to solve complex continuous control problems. However, existing practice relies on online planning to collect high-quality data, resulting in value learning that is entirely dependent on off-policy experiences. Contrary to the belief that value learned from model-free policy iteration within this framework is sufficiently accurate and expressive, we found that severe value overestimation bias occurs, especially in high-dimensional tasks. Through both theoretical analysis and empirical evaluations, we identify that this overestimation stems from a structural policy mismatch: the divergence between the exploration policy induced by the model-based planner and the exploitation policy evaluated by the value prior.
To improve value learning, we emphasize conservatism that mitigates out-of-distribution queries. The proposed method, TD-M(PC)$^2$, addresses this by applying a soft-constrained policy update—a minimalist yet effective solution that can be seamlessly integrated into the existing plan-based MBRL pipeline without incurring additional computational overhead.  Extensive experiments demonstrate that the proposed approach improves performance over baselines by large margins, particularly in 61-DoF humanoid control tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
I believe that the authors propose an improvement to TD-MPC2 by adding a regularization term to the policy loss. 
The proposed regularizer is a KL-penalty to TD-MPC2's behavior policy instead of the maximum-entropy (or uniform policy) regularizer that we see in standard TD-MPC2.

The authors argue that this is fundamentally improves a bottleneck in value learning/ bootstrapping.
Meaning that the regularization to the behavior policy will make the value bootstraps more accurate to the current policy, and thus reduce the mismatch between the prior or maximum-entropy policy.
This makes sense intuitively, and the authors also present theoretical results to back this.

The experiment section is quite extensive and also includes a didactic result in figure 1, showing that value mismatch is improved by the authors' method.
Overall, the new method typically improves upon the baselines.

### Strengths
- The empirical results look very promising for such a simple change to the base TD-MPC2. I think it will be useful to other people that this finding gets advertised around so that anyone using TD-MPC2 at least considers the use of a behavior-policy regularizer over the maximum-entropy regularizer.
- Clear visualization of the methods' effect in figure 1.

### Weaknesses
- Although the theoretical contributions support the intuition of the policy-value mismatch and error accumulation, the results are detached from the methodological contribution that the authors make.

- The document often discusses the problem of value overestimation, however, reading more between the lines, and looking at the methodological contribution. The actual problem being discussed seems to be a value mismatch, but this gets wrongly advertised.

- The methods section is written somewhat murky, mixing in many details of TD-MPC2, making it difficult to isolate what the authors did/ contribute. Thus, I might have underreported the authors contribution; if so, correct me, but this should also be more apparent in the text.

- Most results reported only utilize 3 or 5 seeds per experiment and CIs. Some of the results are thus noisy and have overlapping confidence bands, see e.g., ablations in figure 4, or in the appendix figure 11. More seeds should also smoothen out the instable spikes in e.g., figure 2.

- I am confused why DreamerV3 is a baseline in figure 2? It is not a MPC-method. Of course its inclusion does not hurt, but I also don't see why you would bother if better baselines exist which do use some form of MPC. For example, stochastic Alpha-/MuZero (MCTS based planning instead of MPPI).

*Very minor comment:* The paper is decently written, but could use some minor spelling/ grammar revisions for improved flow (can use LLM for this).

*My summary/ verdict:*
Overall I think this is a nice paper with promising results that also align with intuition/ previous theoretical findings. 
However, the contribution (in my opinion) feels too much like a work-in-progress. 
As I mentioned, I base this on the detached theory with respect to the proposed method.
Furthermore, although the method gets advertised as reducing value overestimation, I believe it really is about reducing a mismatch in learning and using the policy and value neural networks, which is quite different from overestimation.
Still, I think this paper will probably have a good audience at a workshop, so I would advice the authors to consider that instead and revise/ expand where necessary based on this and other reviews.

### Questions
1. How is your method related to MPO? Since you substitute the uniform policy for the MPPI policy in the KL-regularizer, is your method still soft-optimal? Or will this regularizer slowly improve as learning continues, ultimately contracting to a stochastically optimal policy?

2. Related to 1), although the use of the regularizer to the MPPI policy improves the mismatch of value and policy learning, would it not have also made sense to use a better value estimator instead (e.g., Retrace where $\mu$ is used as the behavior policy term)? In this way, value learning is more closely aligned with the maximum-entropy objective. 

3. Taken 1) and 2) together, if I'm correct that your method is more like MPO rather than SAC (i.e., no longer tracking soft-optimality), how does this change in objective skew the results that you present? Is there a way to create a more fair comparison?

4. Is there a way to extend your theoretical results to modulate the regularization in Eq.10? This would create a strong contrast with the baseline TD-MPC2 while also validating your results.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a modification to the TD-MPC2 algorithm, aiming to reduce the off-policyness of its policy iteration updates. The authors identify that in TD-MPC2, the actor policy can diverge from the planner, i.e. the data-collection policy, leading to off-policy value estimation errors and overestimation bias. To address this, the paper introduces a simple regularization term that penalizes divergence between the actor policy and the the data-generating policy, effectively aligning the learned policy with past planner distributions. The resulting method, TD-M(PC)$^2$, is conceptually simple but empirically effective. It improves performance on several continuous-control benchmarks, including DMC and HumanoidBench, while adding minimal computational or implementation overhead.

### Strengths
- The proposed modification is algorithmically minimal but addresses a real and well-known issue in all of RL: overestimation errors in highly off-policy settings. 
- Across the tested domains, TD-M(PC)$^2$ consistently improves upon TD-MPC2 in both sample efficiency and stability. The magnitude of improvements is sometimes very large on the complex humanoid-bench environments. 
- The paper gives an intuitive argument connecting reduced off-policyness with less value overestimation. The introduction of a KL regularization term between the actor and stored planner policies is natural and easy to follow.

### Weaknesses
**Novelty**:

The biggest weakness of this contribution, in my view, is its novelty. The proposed regularization term closely resembles known conservative regularization terms from the offline RL literature. The behavior cloning (BC) penalty term suggested by Fujimoto et al. - a widely used trick in offline and off-policy RL - performs nearly on par with the shown algorithm. The conceptual novelty thus lies mainly in the application context (TD-MPC2) rather than in the underlying idea. 

**Experiments** 

It is dissatisfying that the authors use a different set of baseline algorithms in their experiments. While the shown curves do tend to correspond roughly to known results for these algorithms, the fairness of the experimental comparisons is somewhat unclear. For example, why is Dreamer-v3 not used as a baseline for the DMC suite, the main suite it has been optimized for (and is known to perform well on)? In contrast, the results of Dreamer-v3 seem to match roughly what is reported by Sferrazza et al. in the humanoid bench, but notably this is not the original implementation of Dreamer-v3 and it is unclear how well this model was tuned in the context of this benchmark paper. More generally, it would be empirically more valuable if all algorithms were tested on all benchmarks and underwent a comparable tuning pipeline. 

**Others**:

- The algorithm regularizes the current actor toward the planner that collected the data. This ensures conservative updates but effectively ties the learned policy to a historic dataset. This introduces the pathology that the algorithm can not solve problems optimally if the data-collection policy was suboptimal, even in the limit of infinite data. A natural alternative - which is not explored in this work - would be to regularize the current actor toward the current planner rather than historical ones. Algorithmically, this is more difficult because one would have to plan or find clever alternatives to acquire up-to-date properties of the planning policy, but I would find this alternative algorithm insightful. 

- While the paper is easy to follow conceptually, it has several language and stylistic issues that detract from readability. Minor grammatical errors and inconsistent notation appear throughout. 

Scott Fujimoto and Shixiang Shane Gu. A minimalist approach to offline reinforcement learning.
Advances in neural information processing systems, 34:20132–20145, 2021.

Carmelo Sferrazza, Dun-Ming Huang, Xingyu Lin, Youngwoon Lee, and Pieter Abbeel. Humanoid-
bench: Simulated humanoid benchmark for whole-body locomotion and manipulation. arXiv
preprint arXiv:2403.10506, 2024.

### Questions
- Could you discuss the potential of regularizing toward the current planner $\pi_{H,k}$ rather than the historical, data-collection planner? Would this be computationally feasible or beneficial? 
- Did you implement a hyperparameter tuning protocol and could you run a consistent set of baselines, tuned accordingly?

### Soundness
3

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
This paper identifies that persistent value overestimation caused by structural policy mismatch between the MPC planner's H-step lookahead policy and the nominal policy is a bottleneck in plan-based model-based reinforcement learning (MBRL), especially in high-dimensional tasks, and proposes the TD-M(PC)² algorithm which incorporates a distribution-constrained conservative policy update to mitigate out-of-distribution queries, reduce value overestimation, and significantly improve performance over baselines while being seamlessly integrable into existing plan-based MBRL pipelines with negligible additional computational overhead.

### Strengths
- The paper makes a distinct original contribution by identifying structural policy mismatch (between MPC planners’ exploration policy and nominal policies for value learning) as the root cause of persistent value overestimation in plan-based MBRL.

- The work demonstrates high methodological and empirical quality. Theoretically, it provides rigorous theorems (Theorems 3.1–3.3) to quantify value approximation error, performance gaps, and distribution shifts, with detailed proofs in appendices. Empirically, it validates on diverse benchmarks, compares against strong baselines to confirm the proposed components’ effectiveness.

- The paper is well-structured and clear. 

- The paper demonstrates strong logical coherence, effectively integrating theoretical analysis with experimental observations to present its core arguments and the research problems it aims to address.

### Weaknesses
- In the section of *Policy Iteration with Reduced OOD Query*, the approach adopted in this paper aligns more with existing design ideas of policy constraints from other directions, without obvious distinctive innovations. However, this part is not the core focus of the study and therefore has no adverse impact on the overall quality and validity of the research.

### Questions
- Due to space limitations, the introduction to the *ABLATION STUDY* is not detailed enough. It would be more comprehensive if the methods in Figure 4 could be introduced in the appendix.

- Could you please elaborate on the following points regarding the section *Appendix E: Conservative Policy Learning*? First, what types of policy constraint methods have been attempted? Second, has there been an analysis of the reasons why certain methods are not applicable under the current setting?

### Soundness
4

### Presentation
4

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
This work is in the area of planner-Based Model-based Reinforcement Learning. They look at a recently proposed method TD-MPC2 and propose that the performance of this method can suffer due to the overestimation bias of the value function. They show empirically and theoretically that this overestimation comes from the divergence of the planning policy that collects the data and the sampling policy that the value function evaluates.

The authors propose to alleviate this by adding a KL divergence term in the loss of the learned policy in order to keep the learned policy close to the behaviour policy induced by the replay buffer. Experiments show improvements in performance across most of the benchmark tasks.

### Strengths
I really liked the way the authors explained the problem, showed evidence for it, and proposed a simple approach to solve it. The writing was clear and flowed well from section to section. Figure 1 was a great illustration for their point and the effectiveness of their approach overall. The theoretical results were explained well, and supported the proposal of the paper. Experimental results also supported their points and showed improvement over the main baseline given the simple and practical solution proposed. Overall, this is a strong paper.

### Weaknesses
The paper’s proposed idea is very similar to (Wang et al. 2025), but this is not explained very well in the paper. The authors mention this in L934-936 but I believe their explanation is not complete: BMPC uses the planner’s data as the expert to perform imitation learning, meaning they use almost the same objective as the current paper, with the difference that they do not add the KL objective to the policy loss but perform it as a separate step (i.e. imitation learning), and they also filter the replay buffer data to keep it more up-to-date. I  believe these differences should be expanded on in the paper. I think BMPC should be added as a baseline as well.

I think the exposition of the approach itself needs a bit more space as well. Specifically the section starting at L293. The practical implementation of the objective in Eqn (8) is briefly talked about but the policy loss in Eqn (10) does not show how $\log \mu$ is lower-bounded. If some of the exposition in Appendix D were added here (e.g. L1043 - 1053) it would be a lot clearer. I understand that there are space limitations but this is a core part of the proposed approach. 

On a more minor side, some of the notation used can be confusing. 
- I think using $\pi_k$ and $\pi_{H,k}$ for the two different policies is too similar, perhaps $\Pi_{H,k}$ for the planner policy might make it more readable, or using $\pi_{\theta, k}$ for the learned policy to indicate there are parameters that are learned
- In Theorem 3.1, $\epsilon_k$ is the approximation error of the value function, but we also see $\epsilon_{m,k}$ and $\epsilon_{p,k}$. Perhaps it would be clearer if the approximation error of the value function was $\epsilon_{v,k}$ to make it clearer that k refers to the iteration. 
- L310: multi-variant Gaussian -> mixture of Gaussians? 
- L317: “s” for the threshold is immediately used to mean state in Eq (10) and elsewhere. 
- Sometimes the replay buffer is denoted as $\mathcal{D}$ and sometimes as $\mathcal{B}$ -- e.g. Eqn (8) vs. Eqn (10).
- L403 -- “up-to-date (UTD)” → “update-to-data (UTD)”

### Questions
- (L308) why do you consider reverse KL and not forward KL?
- In L311: what does $\bar{\pi}$ refer to?

### Soundness
3

### Presentation
4

### Contribution
3
