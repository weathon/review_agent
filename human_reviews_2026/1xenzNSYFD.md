# From Individual to Multi-Agent Algorithmic Recourse: Minimizing the Welfare Gap via Capacitated Bipartite Matching

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 0, 4, 6

## Abstract
Decision makers are increasingly relying on machine learning in sensitive situations. In such settings, algorithmic recourse aims to provide individuals with actionable and minimally costly steps to reverse unfavorable AI-driven decisions. While existing research predominantly focuses on single-individual (i.e., seeker) and single-model (i.e., provider) scenarios, real-world applications often involve multiple interacting stakeholders. Optimizing outcomes for seekers under an individual welfare approach overlooks the inherently multi-agent nature of real-world systems, where individuals interact and compete for limited resources. To address this, we introduce a novel framework for multi-agent algorithmic recourse that accounts for multiple recourse seekers and recourse providers. We model this many-to-many interaction as a capacitated weighted bipartite matching problem, where matches are guided by both recourse cost and provider capacity. Edge weights, reflecting recourse costs, are optimized for social welfare while quantifying the welfare gap between individual welfare and this collectively feasible outcome. We propose a three-layer optimization framework: (1) basic capacitated matching, (2) optimal capacity redistribution to minimize the welfare gap, and (3) cost-aware optimization balancing welfare maximization with capacity adjustment costs. Experimental validation on synthetic and real-world datasets demonstrates that our framework enables the many-to-many algorithmic recourse to achieve near-optimal welfare with minimum modification in system settings. This work extends algorithmic recourse from individual recommendations to system-level design, providing a tractable path toward higher social welfare while maintaining individual actionability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper studies the "many-to-many" recourse optimization problem where traditional model assumes a one-to-one structure. It models the system as a capacitated weighted bipartite matching problem. They introduce the problem as a mixed-integer linear programming with capacity constraints. Experiments show this final approach achieves near-optimal social welfare with minimal changes to the system.

### Strengths
1. The many-to-many model is well-motivated as a realistic scenario in real-world applications.

2. The three-layer optimization framework is well presented and experiments support the claim.

### Weaknesses
1. While the paper addresses an important and practical problem in recourse optimization, it lacks theoretical depth which is below the ICLR bar. The problem is formulated as a mixed-integer linear programming (eq. 1) and then is solved through existing approaches. The contribution of this work is more of a problem formulation, which is the main weakness.

2. The paper models the recourse problem as a single, static matching event. Real-world systems are dynamic: new seekers enter the pool, seekers re-apply, and providers retrain their models. The current framework does not address how the system would adapt to these changes.

### Questions
1. Have you considered fairness objectives? 

2. What is the computational cost of the solution? Does it scale with the market size?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper considers an algorithmic recourse problem: individuals that are rejected by a classification models are provided some recourse information that corresponds to a minimum-cost change in their input to the model that would change the classifier's output to be positive. The objective is to assign the individuals to the classifiers respecting the classifier capacities minimizing the total cost. They also consider a variant where the capacities of the classifiers are variable and have an associated cost to increase.

### Strengths
The authors state that the scenario with multiple individuals seeking to be matched to one of many classifiers has not been considered before. I am not familiar with the algorithmic recourse literature specifically, but this seems to be the main strength of their paper.

### Weaknesses
The problems the authors solve are too simple to be of interest to a wider audience. The authors seem to be under the impression that the problems they are solving are NP-Hard. See for example, line 176, footnote 2. This is not true, since the matching polytope is integral. Any simple LP/flow solver suffices to solve LP (1). The remainder of their problems are also not NP-Hard and can be easily solved exactly using simple known techniques, and do not require heavy machinery like Gurobi.

Algorithm 1 to solve LP (2) is also not very interesting, since it corresponds to the obvious policy of assigning each seeker to its most preferred (ie least-costly) provider. 

LP (3) is also solving a polynomial-time solvable problem. It can be reformulated as finding a min-cost flow. Each seeker can be represented as a node with an edge of capacity 1 from the source. Each seeker has edges to nodes corresponding to providers. The edge from seeker i to provider j has capacity 1 and cost -w_ij. The node corresponding to provider j has two edges to the sink. The first edge has cost 0 and capacity k_j. The second edge has infinite capacity and cost \beta_j. The min cost flow precisely solves the LP. 

The instances the authors solve experimentally are very small, perhaps because they are using complex solvers. But as I mentioned, this is not necessary.

### Questions
Capacity augmentation has seen some more work recently. I suggest the authors look up some of the literature on this or flexible capacities.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes extending algorithmic recourse from individual to multi-agent settings by framing the problem as a capacitated, weighted bipartite matching between recourse seekers and recourse providers. The authors define a three-layer optimization framework that (1) performs welfare-maximizing matching, (2) redistributes provider capacities to minimize the gap between individual and social welfare, and (3) regularizes capacity adjustments to balance welfare gains against deviation costs. The method is evaluated on synthetic and real-world data and is shown to improve overall social welfare under capacity constraints.

### Strengths
Addresses a meaningful and underexplored problem: multi-agent aspects of algorithmic recourse.

Formulates a clear optimization approach grounded in established matching theory.

The layered structure (matching → redistribution → penalized adjustment) is logically organized and tractable.

Empirical results are consistent with the theoretical claims and provide useful illustration of welfare trade-offs.

### Weaknesses
**Conceptual misframing and motivation**
The main weakness lies in the conceptual grounding of the work. The paper presents itself as advancing algorithmic recourse, but the formulation and experiments primarily concern resource allocation under capacity constraints. The usual meaning of recourse involves providing actionable interventions to reverse an algorithmic decision (e.g., a classifier output), but no such model or decision process appears here. The “providers” do not represent decision-making systems, and the “recourse costs” are exogenous edge weights. As a result, the setting is more naturally understood as a fair allocation or matching problem rather than an extension of algorithmic recourse. This isn't disqualifying in and of itself, but this connection should be explored. The connection to recourse theory, feature-space manipulation, or model-driven feedback is not clearly justified.

**Modeling incoherence**
Even within the proposed framework, several modeling assumptions are underspecified. The recourse costs $c_{ij}$ are treated as known, fixed, and comparable across seekers and providers, but it is unclear how they are derived or whether they depend on the decision model, individual features, or other agents’ actions. The “individual optimum” (where seekers independently choose the lowest-cost provider) is not well-defined behaviorally—it assumes perfect information and ignores strategic effects, making the “welfare gap” somewhat contrived. The subsequent “capacity redistribution” layers appear as planner-side postprocessing, not something interpretable as policy or learning dynamics.

**Limited engagement with relevant literature**
The paper primarily cites standard recourse and interpretability works (e.g., Karimi et al., 2021; O’Brien & Kim, 2022) but omits key literatures that directly address the welfare and allocation aspects central to this formulation. In particular, perhaps the most relevant work beyond the above is not cited:

Perello, N., Cousins, C., Zick, Y., & Grabowicz, P. (2025, June). Discrimination Induced by Algorithmic Recourse Objectives. In Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency (pp. 1653-1663).


Fair and welfare-aware allocation. Some also handle harms, which is crucial in the recourse setting:

Rea et al. (2021) — Unequal but fair: Incorporating distributive justice in operational allocation models.

Cousins, Viswanathan, & Zick (2023) — The good, the bad, and the submodular.



Welfare-Based and Allocation-Oriented Learning:

The paper discusses “social welfare,” but don’t connect to the established literature on welfare-based optimization or multi-agent fairness in learning. This literature should be discussed, starting with, e.g.,

Lily Hu and Yiling Chen. 2020. Fair classification and social welfare. In Proceedings of the 2020 ACM Conference on Fairness, Accountability and Transparency (FAccT). 

Binns, R. (2018). Fairness in machine learning: Lessons from political philosophy. Proceedings of the 2018 Conference on Fairness, Accountability, and Transparency (FAT*).

Heidari, H., Ferrari, C., Gummadi, K. P., & Krause, A. (2019). Fairness behind a veil of ignorance: A welfare analysis for automated decision making. NeurIPS 2019.

Kim, M. P., Ghorbani, A., & Zou, J. Y. (2019). Multiaccuracy: Black-box post-processing for fairness in classification. AAAI 2019. (Useful because it’s another welfare-related aggregation view on fairness.)

These connect directly to their welfare gap metric, and could justify that conceptual move more rigorously.

For the dynamics of interactive systems, other work explores sequential tasks, which also seem relevant here:

Cousins, Asadi, Lobo, & Littman (2024). On welfare-centric fair reinforcement learning (Reinforcement Learning Conference).
(Addresses recurrent decision settings where welfare and resource allocation interact dynamically.)




*Clarity and exposition*
The exposition could be improved. It is often unclear whether $c_{ij}$ values and recourse costs are known or estimated, and what their behavioral interpretation is. The figures do not clarify how the transition from one-to-one to many-to-many settings changes the mathematical problem. The optimization layers are described as algorithmic innovations, but they largely restate standard LP formulations.

Marginal technical novelty
The technical content—capacitated matching with welfare regularization—is standard in operations research and computational social choice. While applying it to “recourse” is a creative idea, the paper does not introduce new algorithms or theoretical results. The empirical results confirm expected monotonic trends (e.g., increasing capacity yields higher welfare), without surprising or interpretive insight.



Minor Issues:

043: Fix your quotation marks: a ”right to explanation.”

399: Please use math mode for math.

### Questions
Interpretation of “recourse”:
Could you clarify what constitutes recourse in your setting? Are the costs $c_{ij}$ derived from underlying model-based counterfactuals (as in standard recourse), or are they abstract utilities within a matching framework? If the latter, how does this remain within the conceptual scope of algorithmic recourse?

Knowledge and estimation of $c_{ij}$ and costs:
Do the seekers or providers know their corresponding $c_{ij}$ values, costs, or utilities? Are these assumed known to a central planner, or estimated empirically? How sensitive are your conclusions to uncertainty or estimation error in these quantities?

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
4

### Summary
The paper introduces the problem of multi-agent algorithm recourse, formalized as a matching problem between recourse seekers and providers (e.g., the bank). The authors provide three different “flavors” of the optimization problem by casting it as an MILP problem (Equations 1, 2, and 3), and by also providing a custom algorithm. Lastly, the authors performed some experimental evaluation of their approach, under a synthetic and a real-world setting, showing the effectiveness of their approach on the welfare gap and related metrics.

### Strengths
I think the paper provides an interesting and original contribution to the field of algorithmic recourse. The paper is well-written, and the problem (and related solutions) are described nicely. In particular, I think the angle presented in Section 3.1 is very interesting, since it shows how sometimes it is the provider that has to adjust its parameters to improve recourse chances, fairness, and social welfare (as underlined by some papers [1,2,3]).

[1] Barrainkua, A., De Toni, G., Lozano, J. A., & Quadrianto, N. (2025). Who Pays for Fairness? Rethinking Recourse under Social Burden. arXiv preprint arXiv:2509.04128.

[2] Alejandro Kuratomi, Evaggelia Pitoura, Panagiotis Papapetrou, Tony Lindgren, and Panayiotis Tsaparas. Measuring the burden of (un) fairness using counterfactuals. In Joint European conference on machine learning and knowledge discovery in databases, pages 402–417. Springer, 2022.

[3] Francesca ED Raimondi, Andrew R Lawrence, and Hana Chockler. Equality of effort via algorithmic recourse. arXiv preprint arXiv:2211.11892, 2022.

### Weaknesses
I feel the theoretical tractability of the (bipartite matching) recourse problem is somewhat not well-explored. Indeed, besides relying on Gurobi, it would have been nice to see some analysis on the complexity of each “flavor” and/or potential approximation results (e.g., about Algorithm 1). For example, Equation 2 can be solved by Gurobi (which is quite efficient already). Thus, the relevance of Algorithm 1 is not clear not me, since in practice I might as well use the solver directly. It would have been nice to see some analysis of the approximation factor we get on the welfare gap with respect to Gurobi.

### Questions
- Can you better describe the relevance and/or theoretical improvements of Algorithm 1 with respect to the solution provided by the solver? For example, a comparison between Algorithm 1 and Gurobi produced gaps in Figure 3c?

### Soundness
3

### Presentation
3

### Contribution
3
