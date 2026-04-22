# Breaking Safety Paradox with Feasible Dual Policy Iteration

- Avg Score: 5.67
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8, 6, 4

## Abstract
Achieving zero constraint violations in safe reinforcement learning poses a significant challenge. We discover a key obstacle called the safety paradox, where improving policy safety reduces the frequency of constraint-violating samples, thereby impairing feasibility function estimation and ultimately undermining policy safety. We theoretically prove that the estimation error bound of the feasibility function increases as the proportion of violating samples decreases. To overcome the safety paradox, we propose an algorithm called feasible dual policy iteration (FDPI), which employs an additional policy to strategically maximize constraint violations while staying close to the original policy. Samples from both policies are combined for training, with data distribution corrected by importance sampling. Extensive experiments show FDPI's state-of-the-art performance on the Safety-Gymnasium benchmark, achieving the lowest violation and competitive-to-best return simultaneously.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
**DISCLAIMER: I was one of the reviewers of this paper at an earlier conference.**

The paper proposes "safety paradox" as a main obstacle towards zero constraint violation in safe RL, where the improvement in policy safety causes the estimation error of the feasibility function to increase and in turn harms policy safety. An algorithm is proposed to solve this issue. It works through introducing another policy that seeks to violate the constraint as much as possible, so as to provide more samples for the training of the feasibility function. Importance sampling and KL-divergence constraints are employed to mitigate the distributional shifts between the this newly introduced policy and the original learning policy. Empirical valuation shows the effectiveness of the proposed method.

### Strengths
- The flow of this paper is clear and easy to follow.
- Achieving zero constraint violations has great practical significance, and the estimation error of the feasibility function has been a key issue to this. In this regard, this work is well-motivated.
- I did not go through the proof, but the theoretical analysis makes sense to me, and can serve as a valid justification of the proposed safety paradox.
- The experiment results of the proposed method look quite promising.

### Weaknesses
- The theory (Theorem 1) only shows that the upper bound of the estimation error will become larger, not exactly the error itself. I know showing how the exact error relates to constraint violations may be too much to ask, so I recommend showing some empirical evidence about this, which would be a good supplement for the theory. (In the previous rebuttal phase, the authors gave empirical evidence demonstrating the relationship between constraint-violating samples and estimation error. I would appreciate it if the results can be added to the paper.)
- The proposed method solves the safety paradox by introducing another policy that aims to intentionally violate the constraint, which causes extra cost during training and thus may not be practical in the real world.
- There are some other existing baselines this paper is missing: IPO [1] and CRPO [2]. In the previous rebuttal phase, the authors showed their performance. I would appreciate it if the results can be added to the paper.
- The metric used in the empirical evaluation may be a problem. See Questions.

[1]: Liu et al., 2020, Ipo: Interior-point policy optimization under constraints.

[2]: Xu et al., 2021, Crpo: A new approach for safe reinforcement learning with convergence guarantee.

### Questions
My questions were resolved during the previous rebuttal phase of this paper, at an earlier conference. The only thing left: It would be nice if the additional results and discussions from the previous rebuttal can be added to the paper.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper points out a safety paradox in safe RL: as a policy gets safer, it visits constraint-violating states less, making learning a feasibility function harder. The authors formalize this by showing the estimation error of the constraint decay function grows when violating samples become rare. To break this, the authors propose to run a primal policy that tries to be safe with a dual policy that seeks violations but is kept KL-close, and mix them with a truncated importance sampling.

### Strengths
1. The paper proposes a novel algorithm that uses a second KL-coupled policy to actively seek violating rollouts.
2. It is useful in the context of safe RL to formalize the safety paradox.

### Weaknesses
1. The bound that "error grows when violations are rarer" depends on the chosen CDF function, which is not motivated enough, making the paradox narrower than claimed.
2. FDPI creates unsafe data via the dual policy, and the paper does not quantify how many unsafe transitions it generates, or how to cap it in non-simulation settings.
3. The main difficulty, the state-distribution shift, is handled with a single-trajectory, truncated IS ratio plus a KL constraint. However, there is no bias/variance analysis on truncation depth, despite this is the core technical contribution.
4. The paper can be benefited from a broader scope of experiments. Currently all tasks are from Sarety-Gym with similar constraint types

### Questions
Please see weaknesses.

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
The paper introduces a new theoretical problem in safe reinforcement learning (safe RL) called the safety paradox. This paradox arises because as an agent’s policy becomes safer—producing fewer constraint violations—it also generates fewer “unsafe” samples. However, those samples are crucial for accurately estimating the feasibility function, which measures whether future states will satisfy safety constraints. The result is a paradoxical loop: improving safety reduces the ability to learn safety accurately, which then harms the policy’s overall safety performance.

To address this issue, the authors propose a new algorithm called Feasible Dual Policy Iteration (FDPI). FDPI introduces an auxiliary “dual” policy that deliberately maximizes constraint violations while staying close to the original policy. This strategy increases the diversity and number of violating samples without requiring additional data collection. The algorithm then combines samples from both policies and corrects for the resulting distribution shift using importance sampling.

The authors conduct extensive experiments on the Safety-Gymnasium benchmark and show that FDPI achieves state-of-the-art results—producing the lowest constraint violation rates and competitive or best reward performance among all compared algorithms.

In essence, the paper both identifies a fundamental limitation in safe RL and provides a theoretically grounded, empirically validated solution to overcome it.

### Strengths
1.Novel Theoretical Insight — the “Safety Paradox”
The paper uncovers and rigorously explains a previously overlooked phenomenon in safe reinforcement learning: improving a policy’s safety reduces constraint-violating samples, which paradoxically increases the estimation error of the feasibility function and degrades future safety. This insight is both conceptually fresh and theoretically significant.

2.Principled and Effective Solution — Feasible Dual Policy Iteration (FDPI)
The proposed algorithm introduces a dual policy that deliberately generates controlled constraint violations to maintain balanced data for learning safety. With importance sampling and KL constraints ensuring stability, FDPI smartly breaks the paradox without altering the environment or reward structure.

3.Strong Empirical Validation
Extensive experiments on 14 Safety-Gymnasium environments demonstrate that FDPI achieves state-of-the-art safety performance (lowest violations) and competitive rewards. The empirical results clearly support the theoretical claims, showing reduced estimation error and consistent safety improvements across diverse tasks.

### Weaknesses
1.Too Many Theoretical Assumptions
The theoretical framework depends on several restrictive assumptions—such as smoothness of the distance-to-violation function, bounded state transitions, and weak covariance between trajectory segments. These simplify analysis but may not hold in high-dimensional or discontinuous environments, limiting the generality and real-world validity of the theory.

2.Limited Analysis of Hyperparameter Sensitivity
FDPI introduces new hyperparameters, including the dual activation threshold 
d, KL divergence bound 
δ, and feasibility tolerance 
ϵ. However, the paper fixes these values without investigating their sensitivity or adaptive tuning. Since these parameters directly affect the balance between exploration and safety, a lack of analysis leaves uncertainty about robustness across different tasks.

3.Computational and Implementation Overhead
The algorithm maintains two policy networks and two feasibility estimators while applying importance-sampling corrections, which significantly increase computational and implementation complexity. The paper does not report runtime or convergence cost, raising concerns about scalability to larger or real-time safety-critical systems.

### Questions
1. Could you further explain the intuition behind the assumptions introduced in Section 4 (e.g., continuity of the distance-to-violation function, bounded state transitions, and weak covariance conditions)? It would be helpful to understand how these assumptions can still hold or approximately apply in practical scenarios, such as high-dimensional control or discontinuous dynamics.


2. While the motivation for Feasible Dual Policy Iteration (FDPI) is strong and well-argued, the paper does not appear to include a theoretical bound or convergence analysis for the proposed method. Could you clarify whether there exists any theoretical guarantee—such as stability, safety improvement, or sample-efficiency bounds—for FDPI, and how it connects to the theoretical framework established earlier?


3. In real-world applications where safety-critical systems cannot deliberately generate unsafe behaviors, how can the dual policy—designed to intentionally cause constraint violations—be implemented or approximated safely? Is the concept of a “dual policy” operationally feasible in physical or safety-limited environments, and what mechanisms could ensure it remains practical?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper identifies and formalizes a **safety paradox**: as a policy becomes safer, **violation samples become sparse**, which makes the **feasibility function** (e.g., a CDF) harder to estimate and actually increases its error bound—degrading feasible-set identification and, paradoxically, safety. To break this loop, the authors propose **FDPI (Feasible Dual Policy Iteration)**: alongside the primal policy, a **dual policy** deliberately generates violation samples; data from both policies are **importance-sampled (IS)** to correct distribution shift, and a **(symmetric) KL constraint** stabilizes the IS ratios. On 14 Safety-Gymnasium tasks, FDPI achieves **the lowest violation cost** while maintaining **best or near-best return**.

### Strengths
* **Clear, broadly relevant problem framing**: Elevates the “sparse violations → higher feasibility error → less safety” feedback loop into a unifying paradox for feasibility-function–based safe RL.
* **Coherent method design**: The **dual policy** raises violation coverage without increasing total interactions; **truncated-trajectory IS** approximates the state-marginal ratio, and a **KL constraint** tames ratio underflow—practically implementable.
* **Compatible with common policy-improvement frameworks** (e.g., FPI/SAC) with concrete loss forms and Lagrangian updates; reproducibility looks feasible from the described components.
* **Empirical scope and metrics are solid**: 14 environments with cost–return trade-off plots showing **lowest cost** and **competitive returns**, plus targeted ablations that match the stated research questions.

### Weaknesses
* **Theory–practice gap**: Error bounds are developed under MC/CDF estimation, while training uses **TD learning with function approximation**; the bridge between them is under-substantiated.
* **Assumption generality**: Key results rely on continuity/neighboring-state bounds and other conditions whose validity in contact-rich or discontinuous dynamics is not fully justified.
* **IS approximation and robustness**: The state-marginal ratio is approximated along a single trajectory and truncated at first appearance, which can bias estimates; there’s no systematic **sensitivity study** for the **KL threshold, truncation length, or dual sampling fraction**.

### Questions
1. Can you provide an **error-propagation analysis under TD learning**, or at least a joint training trace that links **violation coverage → feasibility-estimation error → final cost**, to empirically validate the paradox-breaking story?
2. Please include **sensitivity analyses** for the **KL threshold**, **dual activation/sampling ratio**, and **IS truncation length**; also report the **distribution of IS ratios** (mean/variance/overflow/underflow rates) over training.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper identifies the “safety paradox” in safe reinforcement learning, where making a policy safer reduces constraint-violating samples and thus worsens safety estimation. It proposes Feasible Dual Policy Iteration (FDPI), which adds a dual policy that deliberately induces controlled violations to improve feasibility function accuracy using importance sampling and KL constraints. Empirical experiments show FDPI achieves strong safety and returns, breaking the paradox

### Strengths
* **Problem Significance:** The paper addresses the well-known and critical problem of learning from sparse cost signals in safe R. Accurately modeling safety boundaries, especially when violations are rare , is a key obstacle to deploying RL in the real world
* **Writing:** This paper is in general well written, barring some minor issues, see weaknesses and questions for a detailed list.
Intuitive Solution: The proposed mechanism of using a dedicated "dual policy" to actively generate the very data that is missing (i.e., constraint violations) is an intuitive and direct approach to solving the sparse data problem. 
* **Experimental Results:** The experimental results presented in Figure 1 and 2 show that FDPI (and its ablative variant, SAC-FPI) achieve a superior cost/return trade-off when compared against the chosen set of baselines.

### Weaknesses
1. **Missing Context of the "Safety Paradox.”** At the crux, this paradox is due to the sparsity of the safety constraint, which becomes more sparse as the policy becomes more sparse. The authors address this for one specific policy iteration algorithm, FPI, by using a dual policy to sample unsafe data. There have been existing works that tackle this problem from a different angle. For example “Feasibility Consistent Representation Learning for Safe Reinforcement Learning”[1] proposes using representation learning to extract the safety-related information from the raw state. Perhaps the authors could include a discussion or comparisons to such methods.  
2. **Baselines:** The current suite of baselines are limited and rather old, with the newest being from 2023\. Perhaps some newer baselines could be added, such as ESPO [2] and PCRPO [3], and the FCSRL[1] method mentioned above?  
3. **Ablations:** There are no ablations of the key design hyperparameters such as $\\epsilon$ and $d$.  
4. **Over usage of $d$ (Minor)** $d$ is used several times in the paper for different notations. For example, d is defined as a continuous function that measures the distance to a constraint violation in Assumption 2\. But then in lines 259-260, it is defined again as a dual threshold hyperparameter.   
5. **Intuitive Justification of Assumption 4 (Minor)** The intuitive justification for Assumption 4 is not particularly clear.

[1] Cen, Zhepeng, et al. "Feasibility consistent representation learning for safe reinforcement learning." Proceedings of the 41st International Conference on Machine Learning. 2024.

[2] Gu, Shangding, et al. "Enhancing efficiency of safe reinforcement learning via sample manipulation." Advances in Neural Information Processing Systems 37 (2024): 17247-17285.

[3] Gu, Shangding, et al. "Balance reward and safety optimization for safe reinforcement learning: A perspective of gradient manipulation." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 19. 2024.

### Questions
1. What is the impact of the sampling fraction from the dual policy? In the paper, it seems to be fixed at 0.5. In particular, if compute allows, an interesting ablation would be how altering both the dual threshold and sampling fraction would impact the final policy.  
2. In Eq 5, the optimization problem is formulated with a constraint that the expectation of the CDF is less than 0, ie $\\mathbb{E}\_{x\\sim d\_{init}}\\left\[F^{\\pi}(x)\\right\]\\leq 0$ However in equation 4, the CDF is defined as the expectation of $\\gamma^{N(\\tau)}$ where $\\gamma \\in (0,1)$, so how is constraint possible?  
3. Could the authors clarify the precise relationship with FPI? It appears the core primal policy update (Eq. 11\) is adopted from FPI, and the primary novelty is the dual-policy data-augmentation mechanism designed to fix FPI's data-starvation failure mode (the paradox). Is this an accurate characterization of the contribution?  
4. Are the proposed data centric solutions and existing representation centric solutions mutually exclusive? Could FDPI's dual-policy sampler be combined with a safety-aware embedding (like that from FCSRL) to potentially achieve even greater stability and performance?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies safe RL. It first introduces a problem termed as safety paradox, which describes a phenomenon that as a safe RL agent's policy improves, it encounters constraint-violating states less frequently. The consequence of this paradox is that an inaccurate feasibility function estimate leads to an incorrect identification of the feasible region. This biases the policy update and undermines policy safety. To address it, the paper proposes an algorithm called Feasible Dual Policy Iteration. The algorithm maintains and trains two policies in parallel. The dual policy is trained to maximize constraint violations by maximizing the dual feasibility function. This mechanism ensures that even as the primal policy becomes too safe, the dual policy continually supplies the necessary critical samples to maintain an accurate feasibility function estimate.

### Strengths
- It proposes several definitions to formulate an important problem in safe RL and also provides an algorithm to address it.
- It has both theoretical works and empirical experiments.

### Weaknesses
- There is a gap between the paper's theoretical foundation and its algorithmic implementation. Theorem 1 is built upon a Monte Carlo estimate of the feasibility function. However, the proposed FDPI algorithm is based on SAC and uses a TD estimate. 

- The paper frames the "safety paradox" as a fundamental and general obstacle in safe RL. However, the entire analysis is built upon a specific problem formulation. It is unclear whether this paradox, as defined, holds for other significant branches of safe RL.

- The proposed solution, FDPI, hinges on training a dual policy that is explicitly optimized to maximize constraint violations. This approach may be impractical and dangerous for many real-world, safety-critical applications.

- The IS ratio is defined as a sequential product of policy probabilities, which is prone to high variance. To address it, the algorithm has a KL divergence constraint. However, the hyperparameter chosen for the KL divergence is high (5.0), suggesting the algorithm could be unstable.

### Questions
How are the hyperparameters chosen, such as the feasibility threshold (0.1) and the dual threshold (0.95)?

### Soundness
3

### Presentation
3

### Contribution
3
