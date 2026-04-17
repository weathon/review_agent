# Safe Multi-Objective Reinforcement Learning via Multi-Party Pareto Negotiation

- Decision: Reject
- Scores: 4, 6, 2, 4, 4

## Abstract
Safe multi-objective reinforcement learning (Safe MORL) seeks to optimize performance while satisfying safety constraints. Existing methods face two key challenges: (i) incorporating safety as additional objectives enlarges the objective space, requiring more solutions to uniformly cover the Pareto front and maintain adaptability under changing preferences; (ii) strictly enforcing safety constraints is feasible for single or compatible constraints, but conflicting constraints prevent flexible, preference-aware trade-offs.
To address these challenges, we cast Safe MORL within a multi-party negotiation framework that treats safety as an external regulatory perspective, enabling the search for a consensus-based multi-party Pareto-optimal set. We propose a multi-party Pareto negotiation (MPPN) strategy built on NSGA-II, which employs a negotiation threshold $\varepsilon$ to represent the acceptable solution range for each party. During evolutionary search, $\varepsilon$ is dynamically adjusted to maintain a sufficiently large negotiated solution set, progressively steering the population toward the $(\varepsilon_{\text{efficiency}}, \varepsilon_{\text{safety}})$-negotiated common Pareto set.
The framework preserves user preferences over conflicting safety constraints without introducing additional objectives and flexibly adapts to emergent scenarios through progressively guided $(\varepsilon_{\text{efficiency}}, \varepsilon_{\text{safety}})$. Experiments on a MuJoCo benchmark show that our approach outperforms state-of-the-art methods in both constrained and unconstrained MORL, as measured by multi-party hypervolume and sparsity metrics, while supporting preference-aware policy selection across stakeholders.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work addresses safe multi-objective reinforcement learning within a multi-party negotiation framework, rather than treating safety as an additional objective. This approach enables the search for a consensus-based, multi-party Pareto-optimal set without enlarging the objective space. Within this framework, the authors propose a multi-party Pareto negotiation (MPPN) strategy.

### Strengths
1. Formulating safe multi-objective reinforcement learning as a multi-party negotiation problem is, to the best of my knowledge, novel, interesting, and practically valuable.

2. The overall method design is generally reasonable and coherent.

### Weaknesses
1. Algorithm 2 appears to update policy parameters solely through DE mutation, without using policy gradients or value-based guidance. This raises questions about efficiency and whether the learning signal may be too sparse, so it is also unclear if this approach can still be considered “reinforcement learning.” The observed improvements may primarily result from the multi-party-specific evaluation metrics, whereas baseline methods are not designed for a multi-party setting.

2. It would be helpful to present the learned behaviors and analyze how they relate to multi-party Pareto optimality.

3. The policy indices in Figure 2 do not seem to correspond with the text description in lines 180–185.

### Questions
1. Does it update policy parameters solely through DE mutation?
2. How could policy gradients be incorporated to provide denser and more informative updates?
3. Does the method learn meaningful multi-party Pareto behaviors in complex environments, such as humanoid tasks?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper reconceptualizes Safe MORL as a multi-party negotiation problem, where the safety objectives and efficiency objectives are treated as separate multi-objective decision parties rather than as additional objectives in a single objective space.
Building on this idea, they develop a negotiation-driven evolutionary framework, MPPN-MORL, which integrates multi-party Pareto negotiation into policy search without increasing the dimension of the objective space. The algorithm incorporates an ε-dominance criterion to enable negotiation into evolutionary search. The idea offers a novel and well-motivated perspective on safe MORL.

### Strengths
The paper presents a novel conceptual formulation of safe MORL as a multi-party negotiation process, which provides a fresh perspective on balancing safety and efficiency.

The proposed MPPN-MORL framework is well-motivated, integrating negotiation principles with evolutionary policy search in a coherent way.

The use of an ε-dominance based negotiation rule and differential evolution operators is clearly described and logically connected to the goal of efficient compromise among objectives (see however below). 

It is possible that the algorithm respects user-specified preferences over both performance and safety, preserves diversity in the solution set, and promotes fairness across parties.

### Weaknesses
The paper claims adaptability, diversity, and fairness in the learned policy set, but these aspects are not directly analyzed or supported by quantitative experiments. Including such evidence would strengthen the empirical evaluation. 

The scalability and computational cost of negotiation among multiple parties are not extensively discussed, which may limit understanding of its practical applicability. 

Fig. 1 may be excellent for use in a talk, but in a collection containing several contributions on MORL, there is no need to start from this level.

Definition 3.1 does not define Pareto Dominance, but Pareto Dominance w.r.t a DM. This should be stated in the beginning in brackets.

In Table 1, the proposed method is referred to as “MPNN”, while the paper elsewhere uses “MPPN.” Please check whether this is a typo. 

Please clarify what the $x$- and $y$-coordinates in Figure 5 represent, including their units and scales? It is important as it is not clear from the caption or text how Figure 5 is obtained or what it is intended to show.

A problem is the need to choose two ε thresholds (for performance and safety) which is at odds with the idea of MORL where the  decision whether a criterion is more less strict is left for the user for after the optimization, while here a related decision is to be made before the start of MORL, so that it is questionable whether the MORL framework has to be used here in the first place or whether already a scalarization is sufficient. I understand that there is theoretical difference between the preference coefficients and the ε thresholds, but it will be difficult to explain this to any users.

### Questions
The paper notes that MPPN-MORL has certain limitations in terms of solution distribution. Could the authors elaborate on what causes this limitation, and whether it relates to the ε-dominance mechanism or the negotiation dynamics? 

How would you treat safety constraints that cannot be expressed as objectives?

Wouldn't in the case of a safety-critical application a hierarchical approach be useful? I.e. why should unsafe regions be explored at all? If this is in some cases justifiable, then such a justification needs to be discussed already here. 

Can you define a Multi-party Pareto Front?

### Soundness
3

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
The paper proposes a new framework for Safe Multi-Objective Reinforcement Learning that treats efficiency and safety as separate decision-making parties in a multi-party negotiation process rather than as combined objectives or hard constraints.

### Strengths
- Novel conceptual reformulation: Framing Safe MORL as a multi-party negotiation problem is original.

- Clear algorithmic description: The dynamic adjustment of ε to control exploration vs. exploitation is intuitive and aligns with practical safety–efficiency trade-offs.

### Weaknesses
I feel the authors want to discuss multiple things, which are entangled together: safety, different decision makers, adaptability. It would be more meaningful to separate these challenges, and discuss what is the key motivation and novelty.

The negotiation is just a selection of hyperparameters. Note that in the training process, there is no "negotiation" between different agents.

Simulations are quite limited to few-dimension simulations, while the performance of proposed techniques are unclear on high-dimensional Pareto front.

Little theoretical insights or guarantees are provided for this method.

For real-world safe RL, there are hard constraints which can never violate, which shall be discussed and compared to other approaches.

### Questions
- I think the first challenge raised by the paper, "incorporating safety as additional objectives enlarges the objective space, requiring more solutions to uniformly cover the Pareto front and maintain adaptability under changing preferences" is a quite mixed one. How to show the proposed method can achieve both coverness and adaptability, while achieving safety?

- Can the authors explain more about the "perspective" of Pareto front? Because in the Pareto front, it is already discussed about the different weighted combinations of preferences. Then why is that different from the different decision makers?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a negotiation-based framework for safe multi-objective reinforcement learning, where efficiency and safety are modeled as two decision parties.

### Strengths
1. The idea of viewing safety vs. performance as a negotiation problem is interesting and conceptually novel.

2. The paper is well-written and structured, with detailed experimental comparisons.

3. Includes ablation and multiple environments.

### Weaknesses
1. The paper is technically dense and hard to follow for readers unfamiliar with MORL or NSGA-II.

2. The theoretical justification for why ε-dominance negotiation leads to better Pareto solutions is weak.

3. The experiments, though numerous, are limited to simulated MuJoCo control tasks and do not test broader generality.

4. The innovation appears to be an incremental combination of existing techniques (Pareto negotiation + NSGA-II) rather than a fundamental new theory.

### Questions
1. How does the negotiation mechanism differ in practice from traditional Pareto dominance used in NSGA-II?

2. Is there any formal analysis or convergence guarantee that supports the ε-dominance negotiation mechanism?

3. Could the authors provide empirical evidence (e.g., ablation on ε decay rate or negotiation threshold) to show its quantitative effect?

4. How well would the proposed framework scale to higher-dimensional or discrete-action environments?

5. Could the authors clarify what is fundamentally new beyond integrating NSGA-II with multi-party negotiation?

6. Are there any new theoretical insights or properties that emerge uniquely from the proposed formulation?

7. Could the authors include statistical tests or confidence intervals to confirm the significance of improvements?

8. Why does the proposed method have weaker MPSP (diversity) performance, and how could that be improved?

9. Can the authors analyze trade-offs between global convergence and solution diversity more clearly?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper seeks to solve two key challenges in safe multi-objective reinforcement learning (Safe MORL): 1. how to find more solutions to cover the Pareto front uniformly and have adaptability when preferences change; 2. how to enforce safety constraint with conflicting constraints. The authors propose a multi-party Pareto negotiation (MPPN) strategy built on NSGA-II. To tackle the first challenge, the authors adjusted the negotiation threshold to maintain a sufficiently large negotiated solution set and steer the population toward the negotiated common Pareto set. To tackle the second challenge, MPPN is able to keep user preferences over conflicting safety constraints without introducing additional objectives. The authors conduct experiments with MuJoCo to demonstrate the superior performance over state-of-the-art methods in both constrained and unconstrained MORL.

### Strengths
I like that the authors are able to evaluate the model performances using multiple important metrics, hypervolume and sparsity, to better capture the overall performance in MORL scenarios.

### Weaknesses
1. While the authors cover some of the literatures in MORL domains, some state-of-art methods are missing. For example, 

Hairi, Fnu, et al. "Enabling Pareto-Stationarity Exploration in Multi-Objective Reinforcement Learning: A Multi-Objective Weighted-Chebyshev Actor-Critic Approach." IEEE Conference on Decision and Control (2025).

Zhou, Tianchen, et al. "Finite-time convergence and sample complexity of actor-critic multi-objective reinforcement learning." ICML 2024.

2. The Robot example in page 4 is a very good illustration of the idea of safe MORL. It would be great if the authors can run a toy experiment on this as well.

3. The current experiment focuses on MPMO continuous control benchmark, i.e., MPMO-MuJoCo. It would be better if the authors can consider more use cases in the experiments.

### Questions
1. I would like to see if the authors can cover more state-of-art methods such as those mentioned in weakness. 

2. I suggest the authors run a toy experiment on the Robot example to be coherent with those in page 4.

3. I think it would be better if the authors can consider more use cases in the experiments.

### Soundness
3

### Presentation
3

### Contribution
3
