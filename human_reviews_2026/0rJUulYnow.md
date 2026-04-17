# EvoMAS : Heuristics in the Loop—Evolving Smarter Agentic Workflows

- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
The rapid development of Large Language Models has driven Multi-Agent Systems (MAS) growth, but constructing efficient MAS requires labor-intensive manual design. Current automation methods generate templated agents, use monolithic optimization, and ignore task complexity gradients. This paper presents Evolutionary MAS (\textbf{EvoMAS}), a biologically-inspired framework that systematically addresses these limitations through three interconnected dimensions: (1) \textbf{dynamic and diverse evolutionary strategies} with six biologically-inspired operators (3 exploration, 3 exploitation) and adaptive strategy selection; (2) \textbf{role-level evolution} that dynamically optimizes agent specialization and collaboration patterns; and (3) \textbf{curriculum-guided evolution} partitioning tasks by difficulty levels and evolving sequentially from simple to complex with cross-stage stability constraints. Additionally, to resolve the contradiction between the inefficiency of pure evolutionary methods and the limited flexibility of manual design, we developed the \textbf{"Cyber Creator"}, a meta-control system combining dynamic rule formulation with reflective updates. Experimental evaluations demonstrate that EvoMAS consistently outperforms existing methods across multiple domains while maintaining cost efficiency, with agent roles dynamically evolving from homogeneous actors to specialized reasoning ensembles. Codes are available at \href{https://anonymous.4open.science/r/EvoMAS-DEF4}
{EvoMAS}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents **EvoMAS**, an evolutionary framework that formulates multi-agent workflow automation as a constrained single-objective optimization problem. It models the “variation–selection–reflection” cycle as a non-homogeneous Markov process and employs a meta-controller, **Cyber Creator**, to adaptively refine strategies and rules. The approach integrates six biologically inspired operators and demonstrates consistent performance gains across diverse benchmarks.

### Strengths
* Rigorous formalization bridging evolutionary optimization and strategy learning.
* Well-defined and operational evolutionary cycle (variation–selection–reflection).
* Rich operator set balancing exploration and convergence.
* Proven theoretical underpinnings ensuring stability and monotonic improvement.
* Comprehensive experiments showing consistent gains in performance and efficiency.
* Transparent reporting and open resources supporting reproducibility.

### Weaknesses
* **Incomplete experimental disclosure:** Missing detailed hyperparameters and multiple-run statistics.
  *Fix:* Add a complete configuration table with mean±std metrics.
* **Opaque curriculum mechanism:** Quantitative thresholds for difficulty staging are unclear.
  *Fix:* Define bucketing rules and add an ablation without curriculum.
* **Unverified theoretical assumptions:** No empirical monitoring of information gain or strategy dynamics.
  *Fix:* Log empirical statistics of these quantities and compare with theoretical expectations.
* **Limited comparison with strong graph-retrieval baselines:** End-to-end tests are lacking.
  *Fix:* Include controlled comparisons under unified corpora and retrieval quality metrics.

### Questions
* Does the policy distribution evolve consistently across curriculum stages?
* How is the reflection interval for Cyber Creator selected, and what is its impact on convergence and cost?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes EvoMAS, a biologically inspired framework to evolve multi-agent workflows along three coupled axes: role-level evolution, dynamic and diverse evolutionary strategies, and curriculum learning. With the developed meta-controller Cyber Creator to adapt rules and operator distributions, EvoMAS achieves SOTA performance across six benchmarks, while maintaining superior cost-efficiency, outperforming both manual designs and automated baselines.

### Strengths
1. This paper focuses on an interesting and important topic, MAS evolution, which is significant to drive the future development of AI systems.
2. The paper clearly classifies the three dimensions of MAS evolution, and novelly proposes a graph-based formulation for the evolution search of MAS. 
3. The experiments in the paper are abundant

### Weaknesses
1. The six operators in exploration and exploitation could be better presented mathematically.
2. The methodology does not detail discuss scheduling/halting (loop bounds, convergence, deadlock avoidance) for cycles in graph search.
3. The task difficulty could be improved by adding human evaluations instead of pure LLM-as-a-judge.
4. EvoMAS improves performance at a higher cost. A detailed cost analysis should be included to justify the significance of using EvoMAS.

### Questions
1. What are the concrete termination/halting rules for cycles, and how are deadlocks/oscillations detected?
2. Does the evolution process only occur in the training process? Or is it dynamically evolving in the inference process as well?
3. How's the performance deviation in multiple runs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
EvoMAS is a system that automatically builds better Multi-Agent Systems using ideas from biology. It evolves agents’ roles, teamwork, and learning stages to handle tasks from simple to complex. It employs six biologically inspired strategies: Diversity Expansion, Conceptual Recombination, Cross-domain Hybridization, Fine Optimization, Best Practice Synthesis, and Role Specialization.

### Strengths
Strong empirical results: Achieves top performance on five of six benchmarks, surpassing prior methods like AFlow and EvoFlow

Broad evaluation coverage: Tested on 8 datasets across diverse domains for robust generalization.

Cost-efficiency: Demonstrates favorable Pareto efficiency, i.e., strong performance gains with moderate computational cost

Ablations: Provides quantitative ablation studies showing which biologically inspired operators contribute most to performance

### Weaknesses
The meta-agent evaluation process involves multiple sources of randomness, including (1) LLM output variance, (2) error propagation in chained reasoning within agents, (3) sampling variability within the meta-agent, (4) stochasticity in evaluation results for the designed agents, and (5) trajectory-level divergence caused by differences in sampled agent chains and their evaluation scores. While the reported results are averaged over three runs, the overall variability remains higher than that of typical single-LLM evaluations, making performance comparisons across runs and methods less statistically stable.

The “Cyber Creator” label is somewhat exaggerated, may obscure rather than clarify its technical role.

### Questions
How do the contributions of the six biologically inspired strategies interact? Are there synergy effects or diminishing returns when multiple operators are combined?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a three-dimensional evolution (roles, strategies, curricula) for multi-agent systems. The intuition is clear which is from biological evolution.
The overall empirical is solid and extensive showing good performance.

### Strengths
This paper introduces a three-dimensional evolution (roles, strategies, curricula) for multi-agent systems. The intuition is clear which is from biological evolution.
The overall empirical is solid and extensive showing good performance.

### Weaknesses
The overall writing is poor and confusing. 
This paper relies heavily on metaphors (“cross-domain grafting,” “meta-rule induction”) with no algorithmic clarity or pseudocode. 
For ambiguous definition of “strategy evolution”, it’s unclear what exactly evolves, prompt templates? Or graph structure?
The core components (Cyber Creator, rule encoding, variation operators) are underspecified; It’s hard to understand how those components work.
For Scalability, this paper mention “scalability” for their proposed method, but only small-scale systems (≤10 agents) tested. Reflection and rule updates could explode computationally. Does the cost record those progress?
In AFlow paper it incurs only around $1 of token cost for their workflows (see their paper). However, in the present paper, the reported cost for the authors’ system is much higher (e.g., $20 or more). The authors should clearly explain why the cost difference is so large. I don’t think this is efficient (most MAS with optimization won’t incur so much high cost). $20 is sufficient to support hundreds of thousands of words!

### Questions
1. Strategy Evolution Mechanism:
What is the update rule for the “strategy probability distribution (A_t)”? 
2. For Cyber Creator, is this a separate LLM acting as meta-controller? How is “rule induction” implemented, prompt synthesis?

### Soundness
2

### Presentation
1

### Contribution
1
