## Summary
This paper proposes the Dynamics Feature Representation (DFR) framework for Reinforcement Learning (RL) in dynamic path planning. DFR hierarchically refines high-dimensional global traffic dynamics into a compact state representation using a pre-computed distance-based policy attention to sparsify the graph and an n-hop neighborhood method to localize features. The goal is to balance information completeness and computational efficiency for RL agents.

## Strengths
- **Clear and Well-Motivated Framework:** The paper clearly identifies the core trade-off between global and local dynamics in RL state representation for path planning and proposes a logically structured, three-level hierarchical refinement (global → task-related → node-related) to address it.
- **Thorough and Relevant Empirical Evaluation:** Experiments are conducted on multiple real-world urban road networks using several core RL algorithms (DQN, PPO, GCN+DQN). The evaluation includes key metrics like planning optimality gap, success rate, feature compactness, and planning time, providing a holistic view of the framework's benefits.
- **Informative Ablation Study:** A detailed analysis of the hyperparameters *k* (policy attention breadth) and *n* (neighborhood hops) provides practical insights into their effects on performance and offers sensible deployment recommendations (e.g., moderate *k*, smaller *n*).

## Weaknesses
- **Superficial Theoretical Grounding:** The connection to Predictive State Representations (PSR) is mentioned as a theoretical basis but is not developed rigorously. The claim that the refined state preserves policy optimality (Eq. 8) is asserted rather than proven or formally analyzed, missing an opportunity to strengthen the paper's theoretical contribution.
- **Strong and Unvalidated Assumption in Policy Attention:** The core "policy attention" mechanism relies on a pre-trained static shortest-path policy. This assumes that paths optimal under a static distance metric are a good proxy for the relevant subgraph under dynamic travel-time conditions. The paper does not analyze the consequences when this assumption breaks down (e.g., under severe, non-uniform congestion), which is a significant limitation of the proposed approach.
- **Incomplete Empirical Validation of Core Claims:** The paper claims DFR helps achieve a "Markovian state representation," but provides no empirical validation (e.g., by comparing performance using history versus the current DFR state). Furthermore, results are presented as averages without reporting variance measures (e.g., standard error over multiple runs), which is expected for rigorous evaluation at a venue like ICLR.

## Nice-to-Haves
- **Exploration of Adaptive Parameters:** As noted in the conclusion, a method to automatically adapt *k* and *n* based on graph properties or learned context would enhance the framework's practicality and scalability.
- **Extended Baseline Context:** While the paper's focus is internal to the RL paradigm, a comparison with a simple random subgraph sparsification baseline of comparable size could help isolate the benefit of "task-relevance" from mere dimensionality reduction.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **"Triangle visualization is unconventional and difficult to parse."** – This is a presentation/style nitpick; the underlying metrics (GAP, SR, CR) are clearly defined and reported.
- **"The term 'attention' is misleading for a static, pre-computed filter."** – While semantically debatable, the paper defines its mechanism clearly, so this criticism does not identify a factual error or substantive flaw.
- **"Lack of comparison to traditional dynamic planning algorithms (e.g., D* Lite)."** – The paper explicitly scopes its contribution to improving state representation *within* RL-based approaches, as stated in Section 5.1. Demanding comparisons outside this scope is unreasonable.
- **"Need for user studies or theoretical proofs."** – These are not standard expectations for an empirical systems paper in this domain.

## Novel Insights
The paper's core novel insight is the specific hierarchical refinement pipeline for state representation in graph-based RL: using a task-specific, pre-computed structural prior (distance-based policy attention) for coarse, global sparsification, followed by agent-centric localization (n-hop neighborhoods) for fine-grained feature extraction. This provides a practical blueprint for constructing compact, decision-relevant states in large-scale dynamic environments, balancing the often-conflicting goals of information sufficiency and computational efficiency.

## Suggestions
- **Strengthen the Theoretical Discussion:** Provide a more formal analysis or proof sketch under what conditions the DFR-compressed state preserves sufficient information for near-optimal decision-making, properly leveraging PSR concepts.
- **Add Variance Reporting:** Include standard deviations or confidence intervals for key results (e.g., Mean GAP, Planning Time) across multiple training runs to demonstrate statistical robustness.
- **Conduct a Sensitivity Analysis:** Add an experiment analyzing how the performance of DFR degrades as the optimal dynamic path deviates from the top-k static shortest paths, to better characterize the limitations of the policy attention assumption.