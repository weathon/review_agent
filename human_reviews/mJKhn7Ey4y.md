# DIPPER: Direct Preference Optimization for Primitive-Enabled Hierarchical Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 1, 5, 5, 6

## Abstract
Hierarchical reinforcement learning (HRL) is an elegant framework for learning efficient control policies to perform complex robotic tasks, especially in sparse reward settings. However, concurrently learning policies at multiple hierarchical levels often suffers from training instability due to non-stationary behavior of lower-level primitives. In this work, we introduce DIPPER, an efficient hierarchical framework that leverages Direct Preference Optimization (DPO) to mitigate non-stationarity at the higher level, while using reinforcement learning to train the corresponding primitives at the lower level. We observe that directly applying DPO to the higher level in HRL is ineffective and leads to infeasible subgoal generation issues. To address this, we develop a novel, principled framework based on lower-level primitive regularization of upper-level policy learning. We provide a theoretical justification for the proposed framework utilizing bi-level optimization. The application of DPO also necessitates the development of a novel reference policy formulation for feasible subgoal generation. To validate our approach, we conduct extensive experimental analyses on a variety of challenging, sparse-reward robotic navigation and manipulation tasks. Our results demonstrate that DIPPER shows impressive performance and demonstrates an improvement of up to 40% over the baselines in complex sparse robotic control tasks.

## Human Reviews

## Human Reviewer 1

### Rating
1

### Rating Number
1

### Confidence
4

### Summary
The authors present DIPPER, a hierarchical RL method that directly optimizes environment reward and human preferences. The specific objective is derived from DPO, and helps mitigate non-stationarity common to most HRL methods.

### Strengths
**Motivation:** Non-stationarity in HRL is a big issue and this paper presents a well-motivated solution for it.

**Comparisons:** The # of and relevancy of baselines is solid, this is a convincing set of comparisons.

**Experiments:** The experiments are performed on tasks well-suited for HRL, and the analysis on goal distance prediction against HIER and HAC demonstrates that HPO’s objective encourages sampling reachable goals for the lower-level policy.

**Clarity:** THe paper is overall quite clear and the walkthrough of how to obtain the objective was both interesting and easy to read.

### Weaknesses
**Novelty:** I am unable to accept this paper as I just reviewed another submission for ICLR that is essentially identical except for the inclusion of actual human preferences vs using environment-given human preferences. The experiments, ablations, and **even the final objective are completely identical.** These should be one paper, ablating the use of human preferences, instead of creating 2 nearly identical papers both claiming the same novelty. I am willing to change my mind on this if sufficiently convinced, but otherwise this paper is a strong reject.

**Experiments:** Why not compare HAC and HIER on the same graphs in Figs 3/4? It’s a little strange to pick each one individually for a separate comparison when they can be compared on the same things. 

**Major issues:**

- L331: Despite the subsequent claim in the paper, $\hat{m}(s_t)$ cannot be removed from the equation when optimizing for the higher level policy since it includes thye term $-\log \pi_U(g_t \mid s_t)$.
- L318-323: How the method actually **solves** with non-stationarity and infeasible subgoal is poorly explained. Please expand more on these sections with more clear explanation.

**Minor issues:**

- Line 279 uses $\pi^H$ instead of $\pi_U$
- L305: $\pi_{\pi_L}$

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper provides a hierarchical (goal-conditioned) RL method for solving long horizon, spare reward tasks. It addresses an important and common issue with HRL methods, namely, the instability due to the inherent non-stationarity at the high level caused by the changing low level policies or primitives. The paper proposes to expoit preference based learning, in praticular direct preference optimization, to disentangle the high-level form the low-level learning step, thereby circumventing the high-level non-stationarity problem. The proposed method is evaluated on four sparse reward tasks and performs favorably in most of them, when compared with a number of relevant HRL and DPO baselines.

### Strengths
Clear and thorough derivation, good number of experiments and reasonable baselines, good theoretical contribution towards more robust and general HRL.

### Weaknesses
Experimental results are not very convincing to me. The main result in Fig 2 shows that Dipper (the proposed method) seems to be very slow at learning the simple maze navigation task, while baselines solve it by what seems orders an magnitude faster. In the other environments, Dipper seems to perform better than existing baselines, but never seems to solve the task fully. In the "pick and place" and "push" task it only reaches a success rate of 0.3. Granted, the baselines perform even worse, but I find that result somewhat hard to believe. Perhaps the baselines were not configured adequately? It also seems like the non of the methods robustly converged to a certain level of performance.

### Questions
Are there related works reporting results on some of the environments used? For example, are there other HRL methods that solve the franka kitchen environment, as far as I know that is a fairly standard method environment for benchmarking HRL.

Why is dipper so slow to learn the maze navigation problem? 

The method requires eliciting human preference data. This introduces a real bottleneck. How many preference trajectory pairs does dipper need to perform well, for example on the franka kitchen environment?  

I am currently rating this paper a 5, but will raise my score if the concern regarding the performance of other baselines metrics is alleviated.

### Soundness
3

### Presentation
2

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
This paper proposes DIPPER, a novel Hierarchical Reinforcement Learning (HRL) method that can mitigate known issues in (goal-conditioned) HRL. These issues are the non-stationarity of the low-level policies preventing stable updates of the high-level policy and the proposition of invalid goals. DIPPER relies on preference optimization, more precisely on Direct Preference Optimization (DPO) which optimizes a policy based on preferences without a reward function.

### Strengths
- The paper proposes an interesting method that addresses important issues in Hierarchical Reinforcement Learning. 
- The paper is well-written and clearly states the existing limitations and the contributions. I particularly like the way how the limitations are defined in the beginning and are recalled throughout the whole paper

### Weaknesses
I appreciate the efforts to provide insightful experiments, however, I believe there is more need for experiments given that the baselines outperform or perform equally well on two out of four tasks. 

Some environments such as the Box pushing [1] environment could be additional and challenging environments that can provide some insights. 

- Li et al. Open The Black Box: Step-Based Policy Updates For Temporally-Correlated Episodic Reinforcement Learning (ICLR 2024)

### Questions
- The main text mentions that the reference policy is not available for complex tasks. Is this because of the lack of the Value function, or to what exactly is this statement referred to? Technically any policy could be chosen as the reference policy. 

- How could trust-region RL methods help with the presented limitations (L1 & L2)? To my understanding, the non-stationarity happens because of drastic changes in the higher-level and lower-level policies. Couldn't this be mitigated using trust region layers for reinforcement learning as presented in [2]? As a Value function is generally learned in RL, this would also mitigate Limitation 2 (infeasible goals).

Those Trust region-based updates have been proposed to stabilize learning in Hierarchical Reinforcement Learning before [3,4] and, therefore, need to be discussed in the work. 

[2] Otto et al. Differentiable Trust Region Layers for Deep Reinforcement Learning (ICLR 2021) 
[3] Daniel et al. Hierarchical Relative Entropy Policy Search (JMLR 2016) 
[4] Celik et al. Acquiring Diverse Skills using Curriculum Reinforcement Learning with Mixture of Experts (ICML 2024)

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors propose a hierarchical reinforcement learning (HRL) method that addresses two key problems in the literature: the non-stationary problem of training a high-level policy while a low-level policy is also changing, and the infeasible subgoal generation problem. The proposed method addresses these two issues by training the high-level policy with direct preference optimization (DPO) and setting the reference policy to be the low-level policy.

### Strengths
- Using the low-level policy as a reference policy for training the high-level policy with DPO and formulating this as a bi-level optimization problem to address the non-stationary issue is a neat and novel idea. The authors present sufficient evidence that their formulation somewhat relieves the problem of infeasible goal generation (Fig. 4). 

- The results demonstrate that the proposed approach performs better than baseline HRL and non-HRL methods on long-horizon sparse-reward tasks. The comparisons in Figure 2 are thorough and fair.

### Weaknesses
- Some aspects of the paper are poorly written. In particular, sections 4.1.1 - 4.1.3 were hard to follow. For example, it's not immediately obvious that equation 6 and the accompanying explanation is referring to equation 5 and this should be made explicit. The small paragraphs "Dealing with L1/L2" are written in a confusing way and make it sound like the solution for dealing with these problems is about to be described next, whereas in fact the authors most likely wanted to convey that their bi-level optimization formulation and solution that was previously described alleviates both the issues. Overall this section could be written better and maybe accompanied with a diagram explaining the hierarchical policy and bi-level optimization problem formulation as a visual aid. 

- Figure 4 only compares subgoal distances to HIER, even though the authors include other HRL baselines in other experiments. Why weren't those comparisons made here as well? This seems a rather weak experiment to me.

### Questions
Question was asked above in the weaknesses section.

### Soundness
3

### Presentation
1

### Contribution
3
