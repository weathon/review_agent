# State-wise Constrained Policy Optimization

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 8, 3

## Abstract
Reinforcement Learning (RL) algorithms have shown tremendous success in simulation environments, but their application to real-world problems faces significant challenges, with safety being a major concern. In particular, enforcing state-wise constraints is essential for many challenging tasks such as autonomous driving and robot manipulation. However, existing safe RL algorithms under the framework of Constrained Markov Decision Process (CMDP) do not consider state-wise constraints. To address this gap, we propose State-wise Constrained Policy Optimization (SCPO), the first general-purpose policy search algorithm for state-wise constrained reinforcement learning. SCPO provides guarantees for state-wise constraint satisfaction in expectation. In particular, we introduce the framework of Maximum Markov Decision Process, and prove that the worst-case safety violation is bounded under SCPO. We demonstrate the effectiveness of our approach on training neural network policies for extensive robot locomotion tasks, where the agent must satisfy a variety of state-wise safety constraints. Our results show that SCPO significantly outperforms existing methods and can handle state-wise constraints in high-dimensional robotics tasks.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a state-wise expectation-constrained policy optimization approach with a Maximum Markov Decision Process formulation.  The MMDP transforms the state-wise constraint into a CMDP-like safety constraint and then solves the safe RL problem with a TRPO-like algorithm. Experiments show that the proposed approach works better on several safety gym tasks.

### Strengths
Originality: the MMDP formulation seems novel to me. It transforms a state-wise expectation constraint into a safety constraint similar to the CMDP's. I am concerned about the claimed novelty on the state-wise constrained RL, which is the weakness below. 
Clarity: Overall it is easy to follow, despite some critical parts being a bit confusing to me. 
Significance: The experimental results in the paper look promising, compared to other CMDP approaches.

### Weaknesses
1. The claim of novelty on state-wise safety-constrained RL: The paper might miss a significant reference overview of recent state-wise constrained RL references. I am listing several of the recent works here, such as 
a) Wang, Yixuan, et al. "Enforcing hard constraints with soft barriers: Safe reinforcement learning in unknown stochastic environments." International Conference on Machine Learning. PMLR, 2023
b) Xiong, Nuoya. "Provably Safe Reinforcement Learning with Step-wise Violation Constraints." arXiv preprint arXiv:2302.06064 (2023).
c) Wachi, Akifumi, et al. "Safe Exploration in Reinforcement Learning: A Generalized Formulation and Algorithms." arXiv preprint arXiv:2310.03225 (2023).

2. Novelty: the novelty of this work might not be enough. Overall, it transforms the state-wise expectation constraint into a cumulative discount expectation constraint (which is fairly simple) and solves the safe RL problem in a TRPO way (which doesn't show novelty as well).

3. Clarity:  the MMDP writing is a bit confusing to me with unclear symbols which I will discuss in the questions.

### Questions
In the MMDP introduction on page 4, what's $M_1, M_2, M_m, M_{it}$? 
 
Is $D_i$ defined on $(\mathcal{S}, \mathcal{M}^m) \times \mathcal{A} \times \mathcal{S}$ or $(\mathcal{S}, \mathcal{M}^m) \times \mathcal{A} \times (\mathcal{S},  \mathcal{M}^m)$?

Why $M_{it} = \sum_{k=0}^{t-1}D_i(\hat{s}k, a_k, \hat{s}_{k+1})$?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a new MDP formulation, namely SCMDP, which requires the cost for every state action transition to satisfy a hard constraint, instead of the usual CMDP's cumulative cost constraint. Deriving from the SCMDP, the paper converts this problem to MMDP, which is further optimized using the proposed SCPO algorithm.

The proposed algorithm enjoys the theoretical guarantees, and the paper proposes three techniques for practical implementation. Results show the algorithm yields near-zero violation and empirically outperforms the baselines.

### Strengths
1. The paper is overall well-written, and the framework is easy to understand.

2. The conversion to MMDP looks interesting.

3. Theoretical guarantee is provided.

4. Empirical results are valid.

### Weaknesses
1. The policy optimization might be costly due to the constraint optimization formulation?

2. The tested environments are customized. As a result, the only source of soundness is the paper itself.

### Questions
1. How does the error between the estimation using $V_D$ and the ground-truth maximum cost increment evolve during training? Does it consistently increase? Does there exist some bootstrapping error?

2. How is $\epsilon_D^{\pi}$ evaluated in practice?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a variation of Constrained Policy Optimization tailored for finite-time MDPs with state-wise constraints. The proposed approach is specifically applied to enable model-free end-to-end control. The experimental validation is conducted using an extended version of the safety-gym environment suite, considering diverse robots and constraints.

The manuscript is recommended for rejection due to several key issues:
(1) the presentation of the new formalism MMDP lacks of clearity and precision, 
(2) the manuscript contains multiple chunks of text directly copied from prior works, 
(3) the evaluation of practical implementation tricks that differentiate it from prior research is partial,
(4) there are numerous unclear references in the related works section that primarily stem from the same author and seem irrelevant to the context.

### Strengths
### Originality
The paper demonstrates a degree of originality in its attempt to adapt prior works by Achiam et al. to finite-time constrained MDPs with state-wise constraints, introducing a novel formalism, MMDP. 

### Clarity:
The paper is generally well-written but many parts are too closely related to Achiam et al. (especially the proofs of main theorems reported as supplementary material). Also, the overall clarity might be improved. For example, the introduction of MMDP lacks a formal definition of the up-to-now state-wise costs in the augmented CMDP, leading to ambiguity in understanding the extended transition dynamics. Equations like Equation 11 that combine discounted state-distribution over infinite horizon and undiscounted state distribution over a finite horizon lack clear theoretical justification. In the experimental metrics, the absence of a clear definition of the cost rate also impacts the interpretability of the results in the figures.

### Significance:
The paper addresses an important area in the field by focusing on constrained MDPs with state-wise constraints. A better development, both in theory and practice, of the current approach might have potential significance for the community.

### Weaknesses
The paper builds upon previous work by Achiam et al., aiming to adapt it to constrained MDPs with state-wise constraints by introducing a new formalism, MMDP. However, the manuscript lacks clarity in justifying the necessity of this new formalism. 
Furthermore, the paper lacks a theoretical discussion on the solvability of MMDP in relation to existing formalisms, opting instead to employ approximate methods to demonstrate its efficacy in high-dimensional problems. 
To improve the theoretical grounding of this new formalism, I would expect a more in-depth presentation and discussion on its necessity, along with a theoretical characterization of its solvability.

Among the contributions that differentiate the current work from the existing methods, in the practical implementation section, the authors introduce a sub-sampling technique to train the maximum state-wise critic. The evaluation of it is confined to a very limited setting (specifically, a single experiment on Drones-3DHazard-8) without comparative reference performance from baseline methods (Figure 6). Although a comparison of performance can be inferred by checking the previous figure (Figure 4.d), the consistency of the proposed techniques across various experiments remains unclear. It would be beneficial to expand the evaluation to multiple experiments to ascertain the consistency and generalizability of the proposed sub-sampling technique.

### Questions
Other observations/questions that would improve the clarity of this work:
 - The paper initially claims optimization to fulfill hard state-wise cost constraints but defines the set of safe policies based on expected state-wise cost constraints, which are not "hard." This inconsistency is also acknowledged by the authors at the end of the paper.
- The augmented CMDP introduces up-to-now state-wise costs M_i in the state, yet lacks a formal definition of these costs and the extended transition dynamics for the M_i components.
- Equation 11 combines discounted state-distribution over an infinite horizon and undiscounted state distribution over a finite horizon within the same optimization problem. The rationale behind this mix and its theoretical justification remain unclear.
- Baseline algorithms are re-implemented by the authors, and the manual tuning of hyperparameters (as detailed in the Appendix) might not be adequate for a comprehensive comparison. Given the sensitivity of current RL algorithms and the significance of implementation details, employing stable implementations from RL libraries and automatic hyperparameter tuning is crucial.
- The absence of a clear definition of the cost rate diminishes the interpretability of the results in the Figures 1, 4. It seems that the range of the cost rate depends on the problem and does not converge to zero for any algorithm, leading to personal uncertainty in understanding the metric correctly.
- Some claims within the paper appear to be hasty conclusions. For instance, in section 6.2, the authors state "End-to-end safe RL algorithms fail since all methods rely on CMDP to minimize the discounted cumulative cost..." This might not entirely represent the situation, as numerous factors, such as tuning and algorithmic implementations, could significantly influence performance in this setting rather than solely attributing it to the CMDP framework.
- Figure 5 demonstrates the max state-wise cost for the proposed SCPO algorithm. It would be valuable to compare this with the max state-wise cost for the other baselines for a more comprehensive evaluation.
- Figure 6 exhibits the impact of sub-sampling for the Drone environment. As already said, it would be equitable to visualize the effects of this technique across all environments and tasks to provide a complete evaluation of practical implementation strategies.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
