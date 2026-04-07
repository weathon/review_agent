## Summary
VISTA is a modular framework for large-scale causal structure learning that decomposes the global problem into local Markov Blanket subgraphs, aggregates them via a novel weighted voting mechanism that down-weights low-support edges, and enforces acyclicity with a Feedback Arc Set heuristic. The method is model-agnostic, comes with finite-sample error bounds and asymptotic consistency guarantees, and demonstrates improved accuracy and efficiency across a variety of base learners in synthetic experiments.

## Strengths
- **Genuinely model-agnostic and modular design.** The framework imposes no assumptions on the base learner, Markov Blanket estimator, or data distribution, acting as a plug-and-play wrapper. This is evidenced by consistent improvements across diverse base learners (NOTEARS, GOLEM, DAG-GNN, etc.) in Tables 1 and 2.
- **Rigorous theoretical grounding.** The paper provides finite-sample error bounds for the weighted voting scheme (Theorems 3.2, 3.4) and proves asymptotic consistency under mild conditions (Theorem 3.5), a significant contribution over heuristic merging methods.
- **Comprehensive and convincing empirical evaluation.** Experiments cover multiple graph families (ER, scale-free), sizes (up to 300 nodes), and data-generating models (linear, nonlinear), consistently showing reductions in False Discovery Rate (FDR) and Structural Hamming Distance (SHD) while often improving F1 score (Tables 1, 2, 9-14). Runtime improvements are substantial (Table 3).

## Weaknesses
- **Limited validation on large real-world datasets.** The only real-data experiment is on the small Sachs network (11 nodes). While synthetic scalability to 300 nodes is shown, demonstrating performance on larger, real-world benchmarks with hundreds/thousands of variables is needed to fully substantiate the practical scalability claim.
- **Performance trade-offs are not thoroughly discussed.** In some cases (e.g., NOTEARS+VISTA-WV on ER5 in Table 1), the weighted voting improves precision but reduces True Positive Rate (TPR) compared to the baseline. The paper would benefit from a clearer discussion of when and why this recall trade-off occurs and how it relates to the theoretical precision-recall trade-off controlled by λ.
- **Theoretical bounds rely on an idealized independence assumption.** The analysis assumes votes from different subgraphs are independent, which is acknowledged as an idealization since subgraphs overlap and share data. The paper does not quantify how violations of this assumption affect the practical validity of the bounds, leaving a gap between theory and practice.
- **Lacks concrete guidance for hyperparameter selection.** While Theorem 3.4 provides a feasible range for λ and Figure 4 shows sensitivity, the paper uses fixed values (λ=0.5, t=0.7) without a data-driven procedure for choosing them when ground truth is unavailable. A practical tuning strategy would strengthen usability.

## Nice-to-Haves
- **Runtime breakdown and parallel scaling analysis.** Reporting the time spent on Markov Blanket identification, local learning, and aggregation separately would clarify the source of speedups. Demonstrating strong scaling with more cores would better support the parallelization claim.
- **Comparison with a broader set of modular baselines.** The comparison with DCILP is valuable; including other recent divide-and-conquer methods (e.g., Shah et al. 2024) would further contextualize the contribution.
- **Inclusion of constraint-based base learners.** Testing with algorithms like PC or FCI would further validate the model-agnostic claim across fundamentally different learner families.
- **Visual case studies.** Side-by-side visualizations of true vs. recovered graphs for representative cases could intuitively show the types of errors VISTA corrects.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Criticism that the GreedyFAS acyclicity enforcement is "heuristic."** Using an efficient approximation for the NP-hard Feedback Arc Set problem is standard practice; the paper justifies its ordering (FAS before thresholding) and the method provides a valid DAG.
- **Demand for statistical significance testing.** Reporting mean and standard deviation over multiple runs is standard in the field; formal significance testing is not a universal requirement.
- **Criticism that hyperparameters are fixed "without justification."** The paper provides a theoretical admissible range (Theorem 3.4) and empirically explores the precision-recall trade-off (Figure 4), which constitutes reasonable justification for the chosen operating point.
- **Suggestion that the framework "sometimes harms recall" is an overstatement.** The overall results show consistent improvements in F1 and SHD; the occasional TPR drop is a recognized trade-off for substantially improved precision, which is discussed in the context of the weighted voting mechanism.

## Novel Insights
The paper’s core novel insight is the design of a weighted voting aggregation rule with an exponential decay term (1−e^{-λm}) that acts as a data-dependent pseudo-count, dynamically regularizing edges based on their frequency of appearance across subgraphs. This provides a principled, tunable mechanism to suppress low-support noise while preserving high-confidence signals, moving beyond simple majority voting. The accompanying theoretical analysis explicitly links the hyperparameter λ to a feasible operating range and to the precision-recall trade-off, offering a formal understanding of how the aggregation calibrates confidence.

## Suggestions
- **Supplement the real-data evaluation** with at least one larger-scale benchmark where a consensus causal structure or interventional data can serve as a proxy ground truth (e.g., a gene regulatory network dataset).
- **Add a brief discussion or empirical analysis** on how correlated votes (due to overlapping subgraphs) might affect the concentration bounds in practice, perhaps by estimating vote correlations in the synthetic experiments.
- **Provide a practical recommendation** for selecting λ and t in the absence of ground truth, such as using a score function (e.g., BIC) on a held-out validation set or proposing a default value based on graph sparsity.