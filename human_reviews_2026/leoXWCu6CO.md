# Dynamic Drone-Assisted Pickup and Delivery Routing

- Decision: Reject
- Scores: 6, 4, 2, 4, 2

## Abstract
We investigate the dynamic drone-assisted pickup and delivery problem (DAPDP), which concerns real-time, on-demand routing decisions in scenarios where new paired orders arrive stochastically throughout the day. By leveraging a fleet of trucks each equipped with a drone, operators can split tasks between ground vehicles and aerial vehicles, aiming to minimize total travel costs while respecting constraints on time windows, capacity, and drone flight endurance. We propose a deep reinforcement learning (DRL) approach based on deep Q-learning, to decide dynamically which newly arrived orders to dispatch and how to integrate drone sorties effectively. Our experiments on a large, real-world-inspired dataset demonstrate substantial performance gains over greedy, random, and lazy dispatch baselines, yielding 10.6\%, 22.6\%, and 37.2\% savings, respectively, in total travel cost. 
Additionally, our value-based RL learns subset selection decisions that co-adapt with a paired sub-solver, yielding near-oracle performance and outperforming classical and PPO baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a Deep Q-Network (DQN) framework to address the Dynamic Drone-Assisted Pickup and Delivery Problem (DAPDP), where new paired orders arrive throughout the day. The system decides dynamically which requests to dispatch, coordinating a fleet of trucks and onboard drones. The authors model the problem as an MDP and train a value-based agent to select subsets of requests for dispatch. Experimental results on large-scale, real-world-inspired datasets show better performance over heuristic and RL baselines.

The problem setting is relevant and timely, and the combination of DRL with combinatorial routing optimization is technically interesting. However, despite numerical improvements, the presentation, clarity, and scientific rigor of the paper are significantly undermined by weak methodological exposition, poor appendix structure, and lack of qualitative visualization.

### Strengths
1.	Relevant and timely topic – The paper tackles a timely and complex problem at the intersection of dynamic logistics and heterogeneous fleet coordination (trucks and drones). The dynamic, rolling-horizon setting with stochastic order arrivals is highly relevant to real-world last-mile delivery. 
2.	Effective Co-Adaptation: A key insight is that the DRL agent learns to co-adapt with the static sub-solver, selecting request subsets that are easier to solve within the time budget, which is a non-trivial and valuable learned behavior.
3.	Empirical results – Demonstrated measurable cost reduction across realistic datasets, achieving near-oracle performance under certain settings.
4.	Comprehensive ablation studies – Includes sensitivity analysis on learning rate, discount factor, and solver time limits, providing valuable insights into training behavior.

### Weaknesses
1. Lack of qualitative visualization
Despite focusing on spatial routing, the paper contains no figures showing truck and drone trajectories, launch/rejoin points, or deferral maps. This omission makes it impossible to assess whether the learned policy demonstrates interpretable or practically meaningful behavior. The results remain purely numerical and fail to provide intuition about what the agent has actually learned.
2. Figure 3 – Conceptually unclear and unsupported
The explanation of Figure 3 (penalty landscape) is inadequate and purely illustrative. The figure provides no empirical content (no learning curves, convergence plots, or variance analysis). The authors claim the “dense signal proved more stable than a flat FAIL reward” but present no numerical evidence. A more useful figure would compare training stability or performance with vs. without this penalty structure.
3. Appendix quality and structure
•	The appendix contains numerous editorial and structural problems:
o	Self-referential typos such as “Equation equation 25.”
o	Over-fragmented subsections (e.g., C.3.1, C.3.2) that add little substance.
o	Redundant or verbose content that reads like documentation rather than a formal academic supplement.
4. State definition inconsistencies
•	The state representation is described twice — in Section 3.1 (compact features for DQN input) and Appendix C.1 (environment-level details) — but with inconsistent terminology and no mapping between them.
•	The paper never defines the state formally (e.g., st=(⋯ )s_t = (\cdots)st=(⋯)), nor specifies which variables are normalized or aggregated.
•	This ambiguity compromises reproducibility and obscures the MDP’s structure.
5. Incomplete reward specification
•	The reward is defined in pieces across Section 3.1 and Appendix C.2.3, without a single unified formula.
•	The interaction between the base reward and penalty terms is unclear.
•	Since the reward function defines the MDP, this version undermines theoretical completeness and reproducibility.
6. Unreferenced static model
•	Appendix A reproduces the standard static DAPDP without citing foundational sources.
•	Acknowledging prior work is necessary for academic integrity and to clarify what is new in this formulation.
7. Writing and style inconsistencies
•	Reinforcement learning is written in full in Section 3.1 but later abbreviated as reinforcement learning (RL) in Appendix C.5 — inconsistent usage.
•	Section C.5.1 introduces AlphaGo without prior context or explanation. The reference appears abrupt and unnecessary.
•	Several sentences require stylistic polishing for academic tone and precision.
•	Section 2 ('Positioning and Novelty') is overly dense and complex, which makes it difficult to follow. Consider simplifying the language and structure to improve readability.
•	The conclusion section should be polished for consistency in tone and tense.

### Questions
Visualization and interpretability
	Can the authors include visualizations (maps or trajectory diagrams) showing the spatial behavior of their learned policy?
	For example, how do drone launch points and truck paths differ between the DQN policy and baselines?

	Reward formulation
	What is the exact unified mathematical expression for the reward function used in training, including penalty terms?
	How is the penalty integrated into the Q-learning target update?

	State representation
	Please provide a formal definition of the MDP state space s_t and observation mapping.
	Which features are normalized or transformed before being input to the neural network?

	Figure 3
	Could the authors provide quantitative evidence (e.g., learning curves, variance reduction) to justify the claim that the dense penalty improves training stability?
	Otherwise, consider removing or replacing Figure 3 with empirical results.

	Static DAPDP references
	The static MILP model in Appendix A is standard in prior literature. Which works did the authors build upon or modify? Please cite them.

	Appendix structure
	Will the authors reorganize Appendix C to reduce redundancy and clarify environment vs. learning-level definitions?

	AlphaGo reference
	Why is AlphaGo mentioned? If used as an analogy, could the authors connect it more explicitly to the reinforcement learning discussion?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper analyzed the dispatching problem for the last-mile delivery using drone-truck integration. A MDP is formulated and experiment is conducted on a simulated data-set.

### Strengths
+ The problem is novel and interesting. It’s also more complicated compared to existing last-mile delivery problem.
+ The formulation of the problem is relatively complete and clear.

### Weaknesses
- The major contribution of the paper seems to be in the formulation, not sure it meets the expectation for a ICLR paper.
- The assumption that orders come randomly is not realistic. Last-mile orders, especially for food and grocery delivery, has strong spatial and temporal patterns.
- Grammar. “the agent must time dispatch decisions” on page 2.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper studies the dynamic drone-assisted pickup and delivery problem (DAPDP), where a fleet of trucks equipped with drones serves dynamically arriving paired pickup–delivery requests under time-window, capacity, and endurance constraints. The authors propose a deep Q-learning (DQN)–based approach to decide which new orders to dispatch and how to coordinate drone sorties. A paired ALNS (adaptive large neighborhood search) sub-solver is used for route construction. Experiments on a “real-world-inspired” dataset with up to 200 customers reportedly show that the method outperforms greedy, random, and PPO baselines and achieves performance close to a clairvoyant oracle.

### Strengths
1. The topic — dynamic pickup-and-delivery with drones — is practically relevant and fits the growing interest in combining reinforcement learning with combinatorial logistics optimization.
2. The empirical results, if valid, suggest that learning-based decision rules might yield efficiency gains in dynamic dispatching environments.
3. The integration of an RL agent with an optimization-based sub-solver is conceptually interesting and aligns with decision-focused learning paradigms.

### Weaknesses
1. Lack of technical novelty.
The proposed approach relies on a standard deep Q-learning framework with minor heuristic adjustments. There is no clear algorithmic or theoretical innovation beyond existing DQN formulations or hybrid RL–metaheuristic approaches.
2. Insufficient methodological rigor.
The MDP formulation and environment definition are vague and incomplete. Key components — such as state representation, transition dynamics, and reward specification — are not defined rigorously, making it difficult to assess reproducibility or correctness.
3. Unjustified design choices.
The integration of ALNS into the framework is insufficiently explained. It remains unclear why ALNS is chosen, how its neighborhoods are designed, what hyperparameters or termination criteria are used, or how it interacts with the learning policy. Without such detail, the claimed near-optimal performance is not verifiable.
4. Unrealistic or underspecified experimental setup.
Several assumptions are overly restrictive or inconsistent with realistic dynamic pickup-and-delivery settings — most notably the “all requests must be served” constraint. Furthermore, the dataset description lacks transparency: the origin and realism of the 200-customer scenario are unclear, and there is no evidence that the environment captures meaningful stochasticity or dynamism.
5. Questionable baselines and results.
The claim that the proposed DQN approach achieves performance within 1% of a clairvoyant baseline is implausibly strong and not supported by proper justification. Details on the oracle baseline, its computational budget, and its use of future information are missing. No ablation or sensitivity analysis is provided to understand why such high performance is achieved.
6. Limited generalization and insight.
The study provides no conceptual or methodological insights that would generalize beyond this specific problem. As a result, the contribution is primarily empirical and lacks depth expected for ICLR.

### Questions
1. What are the hyperparameters and termination criteria for the ALNS sub-solver, and how is its performance benchmarked?
2. How is the clairvoyant (oracle) baseline implemented, and what future information does it assume access to?
3. Could the authors provide details about the dataset generation process and justify why 200 customers represent a realistic operational scale?
4. Have the authors tested generalization across different levels of dynamism or uncertainty in request arrivals?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work investigates the dynamic drone-assisted pickup and delivery problem and proposes a deep reinforcement learning (DRL) approach based on deep Q-learning, to decide dynamically which newly arrived orders to dispatch and how to integrate drone sorties effectively.  However, it is not clear how to use the DQN to solve the challenges in the dynamic drone-assisted pickup and delivery problem, such as the dynamic orders and the cooperation of trucks and drones. Additionally, this work lacks the experiments of comparing with the baselines in the truck-drone delivery.

### Strengths
1. This work models the dynamic drone-truck collaborative delivery problem with time windows aligns with emerging research topics in the current logistics industry, boasting high application value and cutting-edge relevance.
2. This paper applies the standard DQN to learn dispatching decisions, while the complex routing problem is optimized by a dedicated traditional optimizer ALNS. This is a highly practical and effective choice, and the experimental results also demonstrate the superiority of this method.

### Weaknesses
1. The generalization experiments explore scenarios with different urban distributions. What is the model's generalization performance when the request scale of test instances (e.g., 300 requests) is much larger than that during training (100-200 requests)?
2. The experimental results do not include a comparison of inference time. Adding this comparison would enable a better understanding and evaluation of the method, as it is important to know whether ALNS optimization is time-consuming.
3. The addition of shaped rewards is mentioned in Section 4.5.1, but no specific details are provided.
4. Regarding the reward_for_decision in Section 3.2.2, the reward for one step is evenly distributed among each node, which seems unreasonable. For example, if A, B, and C are selected—where A is far from B and C, while B and C are close to each other—this single action returns a large negative reward. A, B, and C receive the same penalty, but in this case, the priority should be to avoid selecting A as much as possible.

### Questions
1. It is not clear how to use the DQN to solve the challenges in the dynamic drone-assisted pickup and delivery problem, such as the dynamic orders and the cooperation of trucks and drones.

2. Could you compare this method with the state-of-the-art method in the truck-drone delivery problems? This work lacks the experiments of comparing with the baselines in the truck-drone delivery.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes a deep Q learning approach for a dynamic drone-assisted pickup and delivery routing problem. The authors attempt to address some of the main challenges of this last-mile delivery problem, such as dynamic requests,  coordination of the ground vehicle and the drone, etc. The static version of the problem is formulated as a Markov Decision Process, with the state space, the action space being a binary variable (dispatch or defer), and the reward being reflective of the total distance as cost, along with the constraint violation penalty after solving the subproblem. The approach was compared against 7 other baseline methods. Parametric study, ablation studies, and hyperparameter sensitivity studies were also performed.

### Strengths
The problem addressed in this manuscript is both highly relevant and inherently challenging within the context of Last Mile Delivery (LMD) — a domain that plays a critical role in modern logistics and e-commerce operations. Efficiently optimizing routes and resource allocation in this stage directly impacts delivery speed, operational cost, and customer satisfaction, making it a central focus of contemporary research in operations research and transportation systems. The authors have effectively captured the complexity of this problem through a well-formulated optimization framework that accurately reflects the real-world constraints and dynamic nature of LMD scenarios. The mathematical formulation is rigorous and thoughtfully constructed, providing clear insights into the trade-offs and decision variables involved. Furthermore, the detailed problem description enhances the manuscript’s clarity and accessibility, allowing readers to fully appreciate the technical depth and practical significance of the proposed approach. Overall, the problem formulation and presentation demonstrate a strong alignment with real-world logistics challenges and contribute meaningfully to advancing optimization methodologies for last-mile delivery systems. The problem considered in this manuscript is a very relevant and difficult problem in Last Mile Delivery. The optimization formulation of the problem, along with the detailed description, is well appreciated.

### Weaknesses
1. Even though the practicality of this work is unquestionable, and the authors have formulated the static version of the problem as an MDP, there is no novelty in the methodology used (simple deep Q learning). Even though it can be argued that there is no need for a more novel, sophisticated approach, given the focus of ICLR, I feel  ICLR might not be the right platform for showcasing this work, and is more apt for an optimization-related platform.

2. The writing style of the manuscript can also be significantly improved. At present, the writing style resembles more of a project report than an academic paper. For example, the meaning of many terms in the state space is not clear. I encourage the authors to improve the clarity of writing there, and also in many other parts of the manuscript.

3. The current choice of baseline appears relatively weak and limits the strength of the comparative analysis. To provide a more convincing evaluation, the study would benefit from incorporating stronger and more representative baselines. In particular, since the Oracle method assumes complete knowledge of the system — an unrealistic assumption in practical settings — it would be more appropriate to replace or complement it with a Mixed Integer Linear Programming (MILP) formulation. MILP-based methods serve as a more rigorous and interpretable benchmark, offering a well-established optimization standard against which the proposed approach’s performance can be meaningfully compared. Including such a baseline would not only strengthen the empirical validation but also highlight the practical advantages and limitations of the proposed method under realistic conditions.

### Questions
1. What is 60s ALNS?

2. How is the sub-problem solved during every decision-making step? 

3. The experimental implementation details are not provided. Can the authors please provide how the environment is implemented using the dataset? For example, the programming platform, or other simulation environment, if any.

### Soundness
2

### Presentation
2

### Contribution
2
