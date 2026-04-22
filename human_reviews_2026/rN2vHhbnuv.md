# Prequential Evidence Pruning: Information-Theoretic Edge Selection for Ordering-Based Causal Discovery

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Ordering-based causal discovery reduces the complex problem of structure learning to parent selection given a candidate topological order. However, the pruning stage remains a critical bottleneck, as widely used procedures rely on marginal, additivity-constrained tests with manually tuned thresholds. These limitations often prevent the detection of non-additive interactions and hinder reproducibility. To address these challenges, we introduce *Prequential Evidence Pruning* (PEP), a framework that reformulates pruning as a local information-theoretic model selection problem. For each candidate edge, PEP computes the prequential log-evidence gain by evaluating the predictive density of a child node conditioned on its current co-parents using a sample-splitting strategy. An edge is retained if and only if this gain exceeds an adaptive Minimum Description Length (MDL) penalty that accounts for the sample size, the number of admissible parents, and the set size. Theoretically, we establish that the population target of the evidence gain corresponds to the Conditional Mutual Information (CMI). Furthermore, we prove that the statistic is stable under bounded log-loss regret and that prequential scoring provides finite-sample concentration guarantees. Empirically, instantiating PEP with a pre-trained tabular foundation model yields consistent improvements across diverse ordering backbones. Notably, our framework incorporates a hierarchical pruning strategy that enables scalability to higher-dimensional graphs, effectively elevating the pruning stage from marginal testing to scalable, context-aware evidence maximization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Prequential Evidence Pruning (PEP), a framework to improve ordering-based causal discovery. It replaces traditional pruning heuristics with a principled cost-benefit analysis, where the out-of-sample predictive evidence for an edge is weighed against a computable MDL complexity penalty. The method is shown to be a highly effective "plug-and-play" module that consistently improves the performance of ordering-based algorithms on small-scale data.

### Strengths
The paper’s key strength lies in its reframing of the pruning problem. The information-theoretic foundation connects the evidence metric to Conditional Mutual Information, and the resulting method addresses the known brittleness of marginal, assumption-heavy pruning techniques

### Weaknesses
The paper's impact is severely limited by its failure to address the critical challenge of scalability, which is a primary focus of the causal discovery community.
1. The primary weakness is that this work provides an enhancement to a paradigm (ordering-based search) that is fundamentally constrained by exponential time complexity. The current research frontier in causal discovery is focused on overcoming this exact limitation through scalable, gradient-based methods that reframe the problem for continuous optimization. By focusing on a search-based paradigm, this work feels out of step with the direction the field is heading to solve large-scale problems.
2.  It is a local improvement that does not resolve the global bottleneck of the search-based approach. The paper's own analysis confirms a pruning cost that is quadratic in the number of candidate parents, and the exponential complexity of the broader search remains the limiting factor. This restricts the method's practical applicability to the large-scale datasets where new causal discovery methods are most critically needed.
3. The experiments are confined to small graphs with 10 to 20 nodes. A significant contribution in causal discovery must demonstrate its relevance to larger, more challenging problems. There is no evidence or discussion of how this method would perform on graphs with 100 or 200 nodes. Without a comparison showing that a PEP-enhanced method can compete with leading gradient-based methods in terms of accuracy and time efficiency at a larger scale, the paper's claims of broad utility are unsubstantiated. It perfects a method within a specific niche without challenging the very paradigms developed to overcome that niche's limitations.

### Questions
1. Figure 5 shows that PEP's significant performance gains are realized almost exclusively when using the powerful, pre-trained TabPFN model, while its performance with standard learners like XGBoost is far less compelling. How can you disentangle the contribution of the PEP framework itself from the exceptional zero-shot performance of its predictive engine? Does this reliance on a large foundation model limit the practical accessibility of your method?
2. The experiments demonstrate strong performance on graphs with up to 20 nodes. However, the field is increasingly focused on scalable, gradient-based methods for larger problems. Given that PEP operates within the computationally expensive ordering-based paradigm, how do you justify its relevance for real-world applications where the number of variables is often in the hundreds or thousands? Can you provide results on larger scale data and real-world data?
3. Given the computational constraints, what do you see as the primary use case for PEP? Is it best suited for small, high-stakes problems where principled, auditable decisions are critical, or do you envision a path to making it viable for larger-scale exploratory causal analysis?

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
The authors introduce Prequential Evidence Pruning (PEP) as a plug-and-play technique to improve existing methods for topological ordering-based causal discovery. Briefly, PEP takes a topological order of a causal graph and iteratively refines it by pruning edges that provide an information gain smaller than a Minimum Description Length (MDL)-based threshold. Experiments illustrate that incorporating PEP into other methods leads to significant improvements with respect to standard metrics for causal discovery (SHD, SID, and F1).

Although the work provides an interesting perspective on a specific class of causal discovery algorithms, it employs a significantly heavy language that hinders my attempts to clearly understand the proposed method.  For example, it states that PEP captures “synergistic and non-additive interactions”, and that it differs from other methods by an “evidence semantics” through a score that “concentrates under cross-fitting” - with the meaning of these expressions remaining elusive throughout the manuscript.

### Strengths
1. Algorithm 1 provides a clear description of PEP.

2. Experimental results highlight PEP’s effectiveness.

### Weaknesses
My main concern with the work regards the excessive use of hand-waved expressions and formulas that are hard to understand. On top of the ones presented in the Summary section, I also have the following remarks. 

1. Expectation operators do not specify with respect to which distribution the expectations are being computed. 

2. Corollary 2 uses the term $P_{j}$, which is only introduced later in the text. This is also true for $\text{Pred}_{\pi}$.

3. Also, it is unclear what the authors mean by “marginal additivity-constrained methods" - and how PEP can circumvent this presumed issue. 

4. It is correspondingly confusing what the authors mean by “the price of order-aware combinatorics”. Could the authors elaborate on this?

5. Proposition 1 is used to support the claim that “the statistic remains well-behaved with imperfect predictors”. However, both the meanings of “statistic” (perhaps the authors are referring to $\delta$?) and the fact that it is well-behaved under imperfect predictors (which predictors?) are not clearly represented. (This is also connected to (1); expectation operators are not clearly described).

6. Corollary 1 talks about small regrets; how small, and how near is $\delta$ to $0$? 

7. Figure 4 is also difficult to parse: how is each axis measured and what is the baseline for the $\Delta \text{Area}$ calculations?

8. In Figure 7, how is “linearity” measured?

### Questions
Please refer to the questions above.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses a key bottleneck in "Ordering-based Causal Discovery" methods—the pruning stage—by proposing a novel, information-theoretic framework called "Prequential Evidence Pruning" (PEP). The authors point out that existing pruning methods (like CAM pruning) heavily rely on marginal tests, additivity assumptions, and manually-tuned thresholds. This causes them to fail in capturing non-additive interactions (such as synergies) and compromises reproducibility.The PEP framework reframes the pruning problem as a local, information-theoretic cost-benefit analysis. For each candidate edge $i \rightarrow j$, the "benefit" is quantified by a "Prequential Log-Evidence Gain," which is the improvement in the prequential (i.e., out-of-fold) predictive log-likelihood for the child $X_j$ when conditioning on $X_i$ in addition to its co-parents $S \setminus \{i\}$. The population target of this gain is equal to the Conditional Mutual Information (CMI). The "cost" is an adaptive code-length penalty computed according to the Minimum Description Length (MDL) principle, which adjusts to sample size, the number of admissible parents, and the current parent set size. An edge is retained only when this benefit (evidence gain) exceeds the cost (MDL penalty).

### Strengths
The paper accurately identifies a core pain point in ordering-based causal discovery methods: the pruning stage. The comparison in Figure 1 between the "Keyhole view" (marginal) and the "Panoramic view" (context-aware) very intuitively illustrates the limitations of existing methods.

The misspecification stress test in Figure 4 is very comprehensive. PEP's advantages are clearly validated, especially in the Post-Nonlinear (PNL) setting and in the robustness to functional form test (Figure 7), which directly confirms the paper's hypothesis.

The experiment using a random topological order in Table 2 effectively isolates the performance of the pruning stage. This demonstrates that PEP's advantage is not just "piggybacking" on a strong ordering algorithm, but stems from the superiority of its own local decision rule.

### Weaknesses
Although the paper presents the use of TabPFN as an advantage (zero-shot, well-calibrated), this is also its main weakness. The results in Figure 5 show that when PEP is paired with RF or XGBoost, its performance has no significant advantage over CAM pruning, and is even worse in some cases. This strongly suggests that the practical performance of the PEP framework is highly dependent on a powerful and well-calibrated density estimator like TabPFN. This weakens the paper's claim of being "model-class agnostic". In domains where TabPFN is not applicable or performs poorly (e.g., high-dimensional data, non-tabular data), the effectiveness of the PEP framework would be highly questionable.

All synthetic data experiments are limited to $d=10$ nodes (Sachs also has only $d=11$). This is a very small scale. Algorithm 1 implements greedy backward pruning. The computational analysis in Appendix H.1 shows that PEP's cost is on the order of $O(Knm_j^2\overline{\alpha})$, where $m_j$ is the number of candidate parents. This is in the same complexity class as CAM pruning's $O(B s^2 n m_j^2)$ (both are quadratic in $m_j$). When the graph density increases, or under the stress test of a random order (as in Table 2), $m_j$ could approach $O(d)$. This implies the pruning cost is at least $O(d^2)$ (per node), leading to a total complexity of $O(d^3)$ or even $O(d^4)$. The paper only reports runtime for $d=10$ (Table H.1), where its runtime is already noticeably higher than CAM pruning.

A core argument of the paper is replacing "manually-tuned thresholds". However, its MDL gate (Eq. 3) includes a fixed overhead $\kappa$. In Appendix G, the authors state that $\lambda=1$ and $\kappa=25$ is "calibrated once for the entire study". This feels like just swapping one "manually-tuned $\alpha$" for another "manually-tuned $\kappa$". Although the sensitivity analysis in Figure 6 shows that the computed MDL gate (marked with $\star$) lies within a flat, high-performance plateau, this analysis does not explore how the position of this $\star$ mark, and consequently the final SHD/SID, would be affected if $\kappa$ took different values.

### Questions
Given the quadratic dependency on $m_j$ (number of candidate parents) in the complexity analysis and the limitation of experiments to $d=10$, could the authors provide an experiment on scaling with the number of nodes $d$? 

The results in Figure 5 show that RF/XGB perform poorly , and the authors speculate this is because TabPFN provides "high-fidelity calibrated densities". This speculation is very reasonable. It is suggested that the authors add an experiment: when using XGBoost or RF as the predictor, add an extra "post-hoc calibration" step (e.g., Isotonic Regression or Platt Scaling) before feeding the outputs into the PEP framework.

Algorithm 1 uses Greedy Backward Elimination. What is the reason for choosing this strategy? At the beginning of backward elimination, the context $S$ used to evaluate $\delta_{i\rightarrow j}$ contains many irrelevant variables, which could be computationally expensive and interfere with the detection of true synergies.

### Soundness
2

### Presentation
3

### Contribution
3
