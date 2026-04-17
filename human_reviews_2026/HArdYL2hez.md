# Mining Valuable Sub-Expressions for Symbolic Regression

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Symbolic Regression (SR) aims to discover mathematical expressions from data, but classical methods are hampered by an immense search space. This inefficiency stems from their tendency to construct expressions atom-by-atom using basic operators and variables, overlooking the power of reusing meaningful sub-expressions. To address this challenge, we introduce Mining Sub-Expression Symbolic Regression (MSSR), a novel framework that discovers and leverages valuable sub-expressions to efficiently search for the correct symbolic form. MSSR employs a cooperative multi-agent reinforcement learning framework, augmented with genetic programming, to intelligently sample sub-expressions from a dynamically evolving library, combining them into a mathematical expression. A pruning mechanism based on the coefficient of variation is utilized to remove redundant terms, promoting the discovery of the parsimonious expression. We conduct extensive experiments on the SRBench and fluid dynamics benchmarks. The results demonstrate that, compared to 24 baseline methods, MSSR recovers more ground-truth expressions and achieves a superior balance between predictive accuracy and model simplicity.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces MSSR, a novel framework for symbolic regression that leverages sub-expressions within a multi-agent reinforcement learning framework. While the idea of reusing sub-expressions is promising, the manuscript has several limitations in methodological justification, experimental rigor, and contextualization within the existing literature. Below are detailed comments and suggestions for improvement.

### Strengths
**Robust and Adaptive Library Management​**

​Dynamic Library Updates: The sub-expression library Lis not static; it evolves using GP based on mutual information (MI) with the target. This ensures the library remains relevant and enriched with components that have high predictive utility, especially under noisy conditions.

​Information-Theoretic Sub-expression Evaluation: Using MI to evaluate sub-expressions provides a noise-resistant measure of relevance, contributing to the method's robustness in noisy data scenarios.

### Weaknesses
**1. Limitations in Expression Form Assumption​**

The proposed method inherently assumes that target expressions can be decomposed into a weighted sum of sub-expressions. This design may favor expressions naturally adhering to this form (e.g., additive models) but could struggle with expressions that do not decompose additively. In such cases, MSSR might overcomplicate the solution by forcing a sum-of-terms structure, potentially leading to less parsimonious fits than methods without this structural bias.

**2. Insufficient Coverage of Related Work​**

The literature review lacks depth in several key areas:

​GP-based methods with similar ideas: Techniques like Genetic Programming with Automatically Defined Functions (GP-ADF) or modular GP explicitly evolve and reuse sub-expressions but are not discussed.

​RL-based symbolic regression methods: Prior works such as GP-RL or hierarchical RL approaches for expression construction are not adequately compared.

**3. Methodological Clarifications Needed​**

​Sub-expression selection via Mutual Information (MI)​: The manuscript mentions using MI to evaluate sub-expressions but omits critical details: How is MI computed between a sub-expression and the target? What is the exact formulation? Are continuous outputs discretized? Clarification is essential for reproducibility.

​Agent coordination and scalability: The framework employs 3 agents per sub-expression, implying 3×n agents for an expression with n terms. It is unclear how the search space scales with term count or how inter-term dependencies are handled. Since agents update independently, the approach may overlook synergies between terms, potentially hindering global optimization.

**​4. Experimental Comparisons and Baselines​**

​Comparison with state-of-the-art: The omission of PySR (a recently published and widely recognized SR tool) undermines the credibility of claimed advancements. PySR should be included in benchmarks.

​Ablation study limitations: The ablation experiments are limited to a few datasets, which may coincidentally align with the method’s structural assumptions. To demonstrate generalizability, ablations should cover all benchmark categories (PMLB, FSRB, Strogatz).

​Lack of comparison with GP-ADF and pretrained methods: GP-ADF directly addresses sub-expression reuse and should be included. Additionally, pretrained approaches like SNIP or NeSymReS, which excel in model complexity reduction, are not evaluated.

### Questions
The questions are described in the "Weakness" section.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes MSSR (Mining Sub-Expression Symbolic Regression) — a novel framework for symbolic regression that aims to reduce the massive combinatorial search space by reusing valuable sub-expressions rather than building expressions from atomic operators.

### Strengths
1. Clear and Innovative idea. The paper identifies an under-explored direction — reusing sub-expressions to shrink the symbolic search space. This idea is well justified both theoretically and empirically.

### Weaknesses
1. **Unclear modeling justification.**
   The rationale for modeling the problem as a *cooperative Multi-Agent Reinforcement Learning (MARL)* setup is unclear. There appears to be a significant conceptual gap between the motivation of *mining valuable sub-expressions* and the decision to formulate it as a MARL problem. A stronger conceptual or mathematical justification for this modeling choice is needed to make the approach more convincing.

2. **Lack of integration with other frameworks.**
   The proposed idea of reusing sub-expressions should, in principle, be applicable beyond GP—such as within Deep RL, MCTS, or LLM-based symbolic regression frameworks. However, the paper presents results only on a GP-style setup without demonstrating or even discussing how MSSR could be extended or adapted to these other paradigms. Including such discussions or preliminary experiments would substantially strengthen the contribution.

3. **Presentation and formatting issues.**

   * The paper lacks sufficient background on symbolic regression and (multi-agent) reinforcement learning. A concise overview of these topics—along with brief introductions to genetic programming and mutual information—should be added before Section 3.
   * The citation format is inconsistent. All instances of `\cite{}` should be replaced with `\citep{}` to ensure proper inline citation style.
   * Mathematical notation is inconsistent—for example, line 161 uses `\mathcal{R}` while line 163 uses `\mathbf{\mathcal{R}}`. This should be standardized throughout.
   * *Theorem 1* is mislabeled; it is not a formal theorem but rather a derived gradient equation. It should be presented as an equation or proposition with a derivation, **not** as a theorem accompanied by a proof.
   * In Table 5, the variable should be written as `$u_0$` (not `u0`). All mathematical operators should use LaTeX commands such as `$\sin$`, `$\log$`, etc., for proper typesetting consistency.

### Questions
Overall, the idea is novel and promising. However, I strongly recommend that the authors carefully proofread the entire manuscript during the rebuttal phase. Without first addressing the numerous writing and formatting issues, additional technical feedback will have limited impact on improving the overall quality of the paper.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes MSSR, a new symbolic regression algorithm that explicitly identifies and reuses sub-expressions. MSSR trains 3 cooperative agents, tasked with sampling the left sub-expression, operators and right sub-expression in a specific formulation they present in Eq. (1).

### Strengths
Novel engineering solution that utilizes a variety of techniques (i.e., information theory, evolutionary computation, reinforcement learning) in an appropriate manner that is competitive.

Paper is organized in an easy-to-read way and is intuitive.

Except for the main results in 4.1.1., Ablation results and PDE discovery case study is a nice addition and is acceptable as-is in my opinion.

### Weaknesses
What is the definition of “symbolic recovery rate” defined? There are different definitions in SR literature, it should be stated clearly in the paper. How are the constants treated for equality (i.e., is 0.999 treated effectively the same as 1 in an equation)?

Seem to be missing SR algorithms published post-2021. Can the paper comment on this?

The paper claims that other SR approaches tend to be “overlooking the power of reusing meaningful sub-expressions”. However, it is well-known that SR algorithms, especially evolutionary approaches, reuse meaningful sub-expressions via mechanisms like crossover. Thus, I think the discussion and definition of “overlooking the power of reusing meaningful sub-expressions” needs to be more specific and nuanced to accurately reflect current SR literature.

Contradictory/inconsistent results. Figure 2 and Figure 3 contradict each other. For example, in PMLB, MSSR has the best R2 test, and the model size is smaller than Operon. By the definition of Pareto-optimality, Operon cannot be Pareto optimal, yet it is Pareto optimal in Figure 3. This applies to other algorithms as well. I suspect the issue stems from plotting the average ranks instead, which is not the standard practice for Pareto fronts in general and has been proven to suffer from "rank inversion paradox" for SR benchmarking [1]. An easy fix would be to plot the absolute metrics on the axis instead of the ranks of the metrics.

[1] Fong, K. S., & Motani, M. Pareto-Optimal Fronts for Benchmarking Symbolic Regression Algorithms. In Forty-second International Conference on Machine Learning.

### Questions
Please address the weaknesses above. In addition to the weaknesses, below are some questions that could possibly justify a further increase in recommendation score.

Can the paper clarify what happens to the sub-expression library when there is an optimized constant in the sub-expression (e.g., x^2.5)? The examples given in the paper do not have constant in the sub-expression.

What algorithm is used to estimate MI (e.g., KSG [2])? Please discuss and cite the algorithm used along with the settings, because while the definition of MI is not ambiguous, there are many variants of MI estimators, each with potentially very different outputs. And, was the choice of MI estimator justified via experimentation?

[2] Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. Physical Review E, Statistical, Nonlinear, and Soft Matter Physics, 69(6), 066138.

Others:

Instead of MI, would another concept in information theory, unique relevance (UR), be more applicable? This is because 2 sub-expressions with high MI could have high overlap/redundancy.

### Soundness
2

### Presentation
2

### Contribution
2
