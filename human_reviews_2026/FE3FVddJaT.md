# CONFEX: Uncertainty-Aware Counterfactual Explanations with Conformal Guarantees

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Counterfactual explanations (CFXs) provide human-understandable justifications for model predictions, enabling actionable recourse and enhancing interpretability. To be reliable, CFXs must avoid regions of high predictive uncertainty, where explanations may be misleading or inapplicable. However, existing methods often neglect uncertainty or lack principled mechanisms for incorporating it with formal guarantees. We propose CONFEX, a novel method for generating uncertainty-aware counterfactual explanations using Conformal Prediction (CP) and Mixed-Integer Linear Programming (MILP). CONFEX explanations are designed to provide local coverage guarantees, addressing the issue that CFX generation violates exchangeability. To do so, we develop a novel localised CP procedure that enjoys an efficient MILP encoding by leveraging an offline tree-based partitioning of the input space. This way, CONFEX generates CFXs with rigorous guarantees on both predictive uncertainty and optimality. We evaluate CONFEX against state-of-the-art methods across diverse benchmarks and metrics, demonstrating that in many cases, our approach more robust and plausible explanations compared to competing uncertainty-aware generators.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces CONFEX, a novel method for generating counterfactual explanations (CFXs) that incorporates uncertainty quantification through Conformal Prediction (CP). The core motivation is that existing CFX methods often fail to account for predictive uncertainty, potentially leading to unreliable explanations that suggest changes in regions where the model's predictions are uncertain or unsupported by data. The authors identify a critical flaw in naively applying CP to CFX generation: the generated counterfactuals may not be exchangeable with the calibration data, violating CP's fundamental assumptions. To address this, they propose using localized CP (LCP) to enforce approximate conditional (test-conditional) coverage guarantees rather than just marginal guarantees. The paper presents three variants: CONFEX-Naive, CONFEX-LCP, and CONFEX-Tree. All methods use Mixed-Integer Linear Programming (MILP) to guarantee optimal solutions and enforce that CFXs yield singleton prediction regions for the target class, indicating high confidence predictions.

### Strengths
The paper demonstrates technical rigor in identifying and addressing a fundamental problem with applying conformal prediction to counterfactual explanation generation. The recognition that CFX search violates the exchangeability assumption which a cornerstone of CP's validity which represents genuine theoretical insight. The authors don't simply note this problem but provide a principled solution through localized CP that enforces approximate conditional guarantees. The mathematical formulation is clear and well-justified, with formal proofs of group-conditional coverage guarantees for CONFEX-Tree. This theoretical grounding distinguishes the work from heuristic approaches to uncertainty-aware CFX generation.The MILP encoding is another strength, ensuring both optimality of solutions and hard satisfaction of CP constraints. CONFEX provides provable guarantees on both the validity and minimality of explanations which is unlike gradient-based methods that may fail to find valid counterfactuals or converge to suboptimal solutions. The formulation integrates uncertainty constraints into the optimization framework and thus making uncertainty in the CFX generation process rather than an afterthought.

### Weaknesses
The novelty of this work is fundamentally incremental rather than groundbreaking. The paper essentially combines two existing techniques i.e., localized conformal prediction (Guan, 2023) and MILP-based counterfactual generation (Kanamori et al., 2020). in a principled way. While the combination is non-trivial and the application domain is new, neither the core CP methodology nor the optimization framework represents novel technical contributions. The localized CP procedure is directly borrowed from prior work, and the MILP encoding follows established patterns for constraint satisfaction problems. 

The tree-based quantile computation (CONFEX-Tree) is the most original technical contribution, but even this builds on well-known spatial data structures (KD-trees). The insight to use offline partitioning is clever engineering rather than algorithmic innovation. The connection to existing group-conditional CP methods further diminishes the novelty, as the approach can be viewed as a specific instantiation of known conditional CP frameworks with spatial grouping.

A critical limitation affecting the contribution's impact is the restrictive scope of applicable models. CONFEX is limited to models with MILP-encodable architectures such as linear models, MLPs with ReLU activations, and decision trees. This restriction severely limits the practical applicability of CONFEX to contemporary machine learning systems. The experiments are conducted only on tabular datasets with relatively simple models (50-unit MLPs, small random forests), which may not reflect real-world deployment scenarios where deeper, more complex models are standard.

The paper doesn't adequately address why formal coverage guarantees are necessary for CFX generation in practice. While theoretically appealing, the practical benefit of having a 90% coverage guarantee versus a heuristic that produces empirically plausible counterfactuals isn't convincingly demonstrated.

### Questions
1. What is the formal approximation quality of CONFEX-Tree's conditional coverage guarantees, and how does it degrade with bandwidth selection? 

2. How does computational cost scale with model complexity, and what is the practical runtime-quality tradeoff compared to gradient-based methods?

3. Can the method be extended to models with non-MILP-encodable components, and what theoretical guarantees would remain?

4. How should bandwidth be selected systematically, and what is the sensitivity of coverage guarantees and counterfactual quality to this choice?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper is concerned with finding counterfactual explanations that are from regions of high certainty of the model. Specifically, the paper argues for including the information about data support to avoid counterfactuals in regions without data. They utilize Mixed-Integer Programming (MIP) to find counterfactuals with high certainty according to the Localised Conformal Prediction (LCP). To improve scalability, they propose approximating the true LCP with a tree model used to split the input space into a more manageable number of strata, rather than expressing the full calibration set in the formulation.

### Strengths
The paper tackles an interesting question and reads okay. I believe the paper addresses an important problem with using CP for counterfactual generation. The authors also cleverly simplify the problem to enable better scalability.

### Weaknesses
There are multiple issues with the paper. In order of perceived importance:

- The paper seems to wrongly claim that the MIP formulation is linear. Algorithm 2 contains the multiplication of two variables ($in_i$ and $w_i$), specifically on lines 8 and 9. The paper should specify the exact and complete MIP formulation, as is otherwise common in MIP literature, at least in appendix. Especially, the formulation of CONFEX-Tree is essential. Algorithm 1 specifies only a general idea and no further explanation or the actual mathematical formulation is provided. This is promised on line 90, sa the first contribution.
- The comparison to the state of the art is dubious. On line 373, four datasets are mentioned, but results on two of them are postponed to the appendix, where they are presented without commentary. It is unclear why two out of four datasets were not included in the main body, though the poor plausibility results on the Adult dataset suggest potential selection bias. Furthermore, the comparison does not include any other method that considers plausibility, despite there being plenty of them, including the cited FACE (Poyiadzi et al., 2020) or some MILP-based, e.g., DACE using the LOF (Kanamori et al., 2020), or more recent LiCE (https://openreview.net/pdf?id=rGyi8NNqB0 - ICLR 2025). Alternatively, C-CHVAE (Pawelczyk et al., 2020) is implemented inside the Carla library that was used for experiments. A similar thing could be said about robustness, where one could look at e.g., PROPLACE (https://proceedings.mlr.press/v222/jiang24a/jiang24a.pdf). Despite this lack of a plausible CFX method, the authors claim to provide an extensive evaluation that demonstrates CONFEX provides more plausible and stable explanations than competing generators. Note that results on stability are not discussed, just presented in the appendix, and their interpretation is inconclusive. 
- The scalability and performance of the Tree variant in relation to the LCP variant should be evaluated. The evaluation of runtime with respect to calibration set size could be useful in deciding at which point the (assumed) decrease in performance is worth the improvement in scalability. Three minutes per instance for 100 instances seems to be computable.
- The results in Table 1 are non-trivial to compare, since the mean is reported over different sets of factuals (only those where each method was valid). 
- The definition of sensitivity and the particular distance used in Table 1 could be stated in the main body, and the Coverage Gap should also be defined more clearly. 

The issues I found led me to the recommendation to reject the contribution. I am certainly willing to be swayed if my concerns are alleviated.

Minor comments:
- Figure 1 should be better described.
- On line 178, a confusion might arise, as the f can be non-linear, while still being MILP-representable. I would recommend rewording this to avoid confusion.
- line 226-227, a confusingly worded section
- line 359 eq. (14) - A colon should follow the g, possibly?
- Distance result of the FOCUS method on the Credit datasets in Table 1 should probably be bold.

### Questions
Most pressing questions:
- Can the MIP formulation of LCP be linearized?
- Why were the results of stability not discussed and included in the main body? They seem to be the most relevant measure of certainty of the counterfactual.
- Why were results on Adult and GMSC omitted from the main body?
- Are the results statistically significant?
- How do the methods compare when taking the intersections of factuals for which all methods were successful?

Minor questions:
- Is the ratio (6) a standard score function?
- Why was CLUE not used in evaluations?
- The +- information and the bands in Figure 2 are the standard deviation? Over how many runs under what configuration?
- What does the (10^-1) by the Sensitivity mean?
- How exactly is the coverage gap defined?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Counterfactual explanations are a popular explainability technique that provides the minimum input perturbation/change required to flip a model's decision. This paper proposes a method for generating uncertainty-aware counterfactual explanations, bringing together techniques from Conformal Prediction and Mixed-Integer Linear Programming. They propose new optimization tricks to obtain these counterfactuals. The counterfactuals obtained are designed to have low predictive uncertainty. Experiments compare SOTA methods across diverse benchmarks and metrics.

### Strengths
-- I enjoyed reading this paper. The idea of bringing together ideas from conformal prediction and MILP for counterfactuals is interesting. Conformal prediction techniques give prediction sets that are guaranteed to contain the true (unknown) outcome with a given probability. MILP provides a framework for deriving CFXs as a constraint-solving problem. The technique is called CONFEX.

-- They also introduce CONFEX-Tree which is a more efficient way of computing the uncertainty-aware counterfactuals, drawing inspiration from KD trees.

-- Experiments include several baselines and standard tabular datasets. Several metrics have been considered.

### Weaknesses
-- The tabular datasets used are also used for gradient-based counterfactual generation techniques. Could you highlight the differences and benefits of this class of technique from gradient based methods?
There's this line in limitations that would be great to elaborate and clarify: gradient-based methods like Wachter and ECCCo are less prone to this problem, but they sacrifice guarantees on CFX validity.

-- How do you compute plausibility and validity in the experiments?

-- The experiments seem to show tradeoffs. Cost increases to do better in other metrics. Some tradeoff plots can also be good.

--  It would be great to write a complete problem statement to clearly have all the desirables that the algorithm is seeking in one place. This will also help to contrast with what other techniques are doing, why CP is necessary, what are the performance metrics of interest, etc.

-- The abstract uses the word "guarantees" but the guarantee is only on validity with a certain probability, right? Would some of the other properties provably hold or they just hold empirically, e.g., LOF? It would be nice to clarify. Again, going back to the problem statement with the desirable properties, then it will be nice to clarify which of those desirable properties provably hold, and which ones hold intuitively (empirically observed benefit).

-- It would be nice to consider including a theorem statement if there is any kind of provable guarantee on the counterfactuals meeting any of these criteria.

### Questions
Some of my questions are in the weakness section.

1. What is the advantage/disadvantage over gradient-based methods like Wachter?

2. More clarification on the problem statement, all the properties desired, and which of them are provably achieved (guaranteed), vs which properties are found to hold empirically? This makes the presentation more streamlined.

### Soundness
3

### Presentation
3

### Contribution
3
