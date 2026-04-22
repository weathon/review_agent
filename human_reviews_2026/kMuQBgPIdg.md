# Enhancing Generative Auto-bidding with Offline Reward Evaluation and Policy Search

- Avg Score: 6.00
- Decision: Accept (Oral)
- Scores: 4, 8, 6, 6

## Abstract
Auto-bidding is a critical tool for advertisers to improve advertising performance. Recent progress has demonstrated that AI-Generated Bidding (AIGB), which learns a conditional generative planner from offline data, achieves superior performance compared to typical offline reinforcement learning (RL)-based auto-bidding methods. However, existing AIGB methods still face a performance bottleneck due to their inherent inability to explore beyond the static dataset with feedback. To address this, we propose AIGB-Pearl (Planning with EvaluAtor via RL), a novel method that integrates generative planning and policy optimization. The core of AIGB-Pearl lies in constructing a trajectory evaluator to assess the quality of generated scores and designing a provably sound KL-Lipschitz-constrained score-maximization scheme to ensure safe and efficient exploration beyond the offline dataset. A practical algorithm that incorporates the synchronous coupling technique is further developed to ensure the model regularity required by the proposed scheme. Extensive experiments on both simulated and real-world advertising systems demonstrate the state-of-the-art performance of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies how to integrate an offline reinforcement learning (offline RL) approach and conditional generative planning for the auto-bidding task of commercial advertisement. The proposed approach is akin to a conservative model-based RL method, where the evaluator is first trained in a pessimistic way with a Lipschitz constraint and then, controller is optimized to maximize the evaluator's score under the KL and Lipschitz constraints. The theoretical analysis is provided to bound the sub-optimality, and the experiment demonstrates that the proposed method works better than existing offline RL baselines and generative planning baselines.

### Strengths
- The motivation for combining conditional generative planning and optimization (score maximization) is well-explained.

- The proposed approach appears reasonable given the motivation of maximizing the score as an RL problem. In experiments, the proposed method is compared to offline RL baselines and works better than the compared method. 

- A good ablation study is provided. Especially, qualitative analysis showing how the trajectory of the proposed method differs between w/ and w/o KL explains why Lipschitz constraints are not sufficient even under the Lipschitz condition, which answers one of the questions I had.

### Weaknesses
- I wonder where the conditional generative planning component is in the proposed method. While the proposed method works better than both traditional offline RL and generative planning-based baselines, given that the proposed method does not try to maximize the likelihood of demonstration, this looks more like a conservative model-based RL rather than a combination of planning and RL. The phrasing should be reconsidered, and in this sense, I wonder how impactful this paper is to the ICLR community (I acknowledge the contributions for the auto-bidding application, while in the offline RL context, Lipschitz constraints are another simple way to introduce conservativeness differently).

- Theorem 2 seems to lack a necessary assumption. In my understanding, the evaluator's bias cannot be measured without having an assumption of the evaluator's accuracy or how the evaluator is trained (e.g., the evaluator is trained to satisfy the Lipschitz condition). It seems that he necessary assumption is not provided in advance.

- Similarly, the paper claims that Theorem 1 is satisfied by "the evaluator's design" (i.e., by having $\root{T} R_m$ constraints in the regression). However, if this is the reason, the theorem should be about the Lipschitz continuity of $\hat{y}(\tau)$, not that of $y(\tau)$. There is no evidence presented regarding whether $y(\tau)$ satisfies the Lipschitz continuity. I believe this should be an "assumption", instead of a "theorem". Overall, the theoretical analysis does not seem rigorous, and these details should be carefully reviewed.

### Questions
- What is the sensitivity of the model like regarding the constraints (i.e., $\delta_K$ and $L_p$)? How are these values determined in the experiment, and how much effort is needed to adjust these hyperparameters?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes AIGB-Pearl (Planning with EvaluAtor via RL), a method that augments generative auto-bidding (AIGB) with a learned trajectory evaluator and a KL–Lipschitz constrained score-maximization procedure to enable safe exploration beyond an offline dataset. The work combines theoretical analysis (evaluator bias bounds and a sub-optimality gap under KL/Lipschitz constraints), a practical planner/evaluator architecture (synchronous coupling, Lipschitz regularization), and extensive simulated and real-world A/B experiments that report consistent GMV/ROI improvements over strong baselines.

### Strengths
1. Integrating an evaluator into a generative planner and constraining planning by KL + Lipschitz criteria to control evaluator bias is a principled and timely idea for offline auto-bidding.
2. The paper gives provable bounds (evaluator bias → performance gap; sub-optimality bound under the proposed constraints) that make the method more convincing.
3. Results include both simulated and large real-world A/B tests (TaoBao), showing consistent gains over state-of-the-art baselines and sensible ablations.
4. The synchronous coupling and the proposed surrogate for the Wasserstein term are well-motivated and seem implementable in practice.

### Weaknesses
1. In the problem statement the intrinsic impression value is introduced as v_i>0 (Section 2 / preliminaries). Later, in the experimental settings (Tables 5/6) the “value of impressions” is reported as ranging from 0 to 1. This inconsistency leaves readers unclear whether the theory (which uses v_i>0 and constants like R_m = max_i v_i/p_i) assumes arbitrary positive values or implicitly assumes normalized values in [0,1]. It also affects interpretability of constants such as R_m` Lipschitz constants that depend on reward magnitudes, and the hyperparameter choices.
2. In the simulated-experiment paragraph the manuscript text says: “The maximum and minimum budgets of advertisers are 1000 Yuan and 4000 Yuan, respectively.” This phrasing is inverted relative to Table 5, which lists Minimum budget = 1000 and Maximum budget = 4000. In the real-world experiment paragraph a similar inversion appears: “The maximum and minimum budgets of advertisers are 50 Yuan and 10,000 Yuan, respectively.” This contradicts Table 6 (Minimum = 50, Maximum = 10,000). Such directional swaps are minor editorial errors but can confuse reproducibility and readers’ understanding of budget regimes used in experiments. Given that the budget ranges are central to the task formulation, they should be unambiguous.

### Questions
See weakness

### Soundness
4

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
This paper proposes a novel method (AIGB-Pearl) to enhance generative auto-bidding (AIGB) by integrating
reinforcement learning (RL) into a trajectory-based generative modeling framework. The key idea is to
train a trajectory evaluator to score the quality of generated trajectories and use this score to guide the generative
planner to find higher-performance trajectories beyond the offline dataset, while ensuring safety and
generalization via KL and Lipschitz constraints. The method successfully leads to more stable training than
traditional offline RL, and is supported by theoretical guarantees on sub-optimality and generalization. The
authors conduct extensive experiments in simulated and real-world advertising environments to demonstrate
state-of-the-art performance.

### Strengths
1. Originality: The integration of RL into a generative auto-bidding framework is innovative and addresses
the inability to explore beyond the offline dataset of existing AIGB methods. The introduction of a
trajectory evaluator and the use of KL-Lipschitz constraints to ensure safe exploration are also novel
contributions.

2. Theoretical Rigor: The paper provides a thorough theoretical analysis, including bounds on evaluator
bias, sub-optimality and generalization. The use of Wasserstein distance and synchronous coupling for
Lipschitz regularization is well-motivated and technically sound.

3. Significance: The method is deployed in a real-world advertising platform and shows significant improvements
compared to previous methods, indicating high practical relevance.

4. Ablation Analysis: Ablation studies clearly demonstrate the importance of both KL and Lipschitz
constraints. The analysis enhances the rigor and completeness of the article’s theoretical framework.

5. Experimental Validation: Comprehensive and detailed experiments are conducted in both simulated
and real-world environments with large-scale datasets and multiple budget levels, demonstrating the
superior performance of the algorithm.

### Weaknesses
Theoretical Assumptions: Some theoretical results in this paper rely on strong Assumption 1, which can be hard to verify on real-world dataset.

### Questions
1. The paper considers the second-price bidding problem. How would the core formulation need to be adapted
for a first-price auction environment? Will the method still be effective under first-price settings?

2. The theoretical analysis assumes the offline dataset satisfies LSI. What is the performance of AIGBPearl
if the assumption does not hold? Are there any ways to relax the assumption?

3. The paper considers offline RL baselines. Can the framework be extended to online cases?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces AIGB-Pearl, a novel method designed to enhance AI-Generated Bidding (AIGB) by overcoming its limitation of imitating trajectories from a static offline dataset. AIGB-Pearl integrates generative planning with policy optimization by first constructing a trajectory evaluator through supervised learning to provides the necessary reward signal to guide the generative model. To ensure reliable utilization of this evaluator and to mitigate the Out-of-Distribution (OOD) issue, the method designs and enforces a provably sound KL-Lipschitz-constrained score maximization objective for the generative planner. AIGB-Pearl achieves greater training stability and demonstrates SOTA performance in extensive simulated and real-world advertising systems, consistently achieving improvement in GMV and other metrics compared to prior SOTA AIGB method

### Strengths
1. Novelty. AIGB-Pearl is the first work that integrates generative planning and policy optimization to address the performance bottleneck of existing AIGB methods. The integration of offline RL methods into the AIGB is important to guide the policy training and exploration of new trajectories
2. Technical Contribution. The paper provides provable theoretical guarantees to ensure the reliability and safety of the policy optimization, which is critical when exploring beyond the static offline dataset. The core innovation is the design of a provably sound KL-Lipschitz-constrained score maximization objective that help mitigates the OOD problem
3. Evaluation. In both the simulation and the real-world online A/B test, the methods achieve SOTA performance in the auto-bidding problem, out-performing the strong benchmark AIGB.

### Weaknesses
1. The paper provides details on the overall dataset sizes: the simulated experiment uses 5k trajectories generated by 20 advertisers, while the real-world experiment uses 200k trajectories from 10k advertisers. However, it doesn’t explicitly state how many of these trajectories are used to train the evaluator model, and 5k/200k appears relatively small for training a stable evaluator capable of mitigating OOD issues.

2. The paper states and assumes that the offline data distribution must satisfy the Logarithmic Sobolev Inequality (LSI) with a positive constant ρ, and this assumption could be satisfied through strategic adjustment of the trajectory sampling probability per condition y in the offline dataset D, the need to potentially alter the raw offline dataset distribution to ensure the theoretical bounds hold represents a limiting constraint or assumption on the data

### Questions
1. The effectiveness of AIGB-Pearl hinges on the reliability of the trajectory evaluator. The theoretical analysis bounds the gap between the planner's score and its true performance, where one of the core terms is the evaluator's bias on its training dataset. Could the authors provide the measured value of the evaluator's bias in both the simulated and real-world experiments?

2. Related to weakness 2, Regarding “this assumption could be satisfied through strategic adjustment of the trajectory sampling probability per condition y in the offline dataset D”, Could the authors detail the specific nature and extent of this strategic adjustment applied to the offline dataset (e.g., which trajectories or conditions were adjusted)?

### Soundness
3

### Presentation
3

### Contribution
3
