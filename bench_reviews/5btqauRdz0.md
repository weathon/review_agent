## Summary

This paper introduces STAGE, a method for zero-shot generalization of GNNs to graphs with entirely different node attribute domains. The key insight is to transform raw node features (which are domain-specific) into representations of statistical dependencies between features (which can transfer across domains), specifically by constructing "STAGE-edge-graphs" from empirical conditional probability matrices. The approach is grounded in the theory of maximal invariants from statistical testing, and experiments demonstrate substantial improvements over baselines in link prediction (40–103% relative improvement in Hits@1) and node classification (~10% improvement in accuracy) when transferring to unseen attribute domains.

## Strengths

- **Novel problem formulation with principled solution:** The paper identifies a genuine gap—existing GNN foundation models cannot handle graphs with entirely different attribute spaces—and proposes a conceptually elegant solution: rather than learning feature values, learn their statistical dependencies, which can have analogous structure across domains. This is a meaningful shift from prior approaches that either ignore features, use LLM textification, or assume shared feature semantics.

- **Strong theoretical grounding:** The connection to maximal invariants (Bell 1964; Berk & Bickel 1968) provides principled justification for why rank-based dependency representations enable domain transfer. Theorem 3.4 establishes that STAGE is provably invariant to the specified class of domain transformations (COGGs), giving formal grounding for the approach.

- **Substantial and consistent empirical gains:** The improvements over baselines are large (40–103% in link prediction, ~10% in node classification) and consistent across six test domains. The robustness to the extreme H&M domain shift (a completely different data provider with different products, customers, and features) is particularly compelling evidence for the method's transfer capability.

- **Handles mixed feature types naturally:** The conditional probability definitions (Equation 2) explicitly accommodate both continuous and categorical features, addressing a practical challenge that standard normalization or embedding approaches struggle with.

## Weaknesses

- **Theory applies only to fixed-dimensional feature spaces, but experiments use variable dimensions.** Section 3 explicitly restricts theoretical results to "domains with a fixed number of features to simplify the proofs." Yet the core empirical contribution involves transfer across domains with different feature dimensions (e.g., smartphones with RAM, display vs. clothes with size, color). The gap between what is proved and what is demonstrated is significant and should be addressed directly—the paper should clarify whether the GNN's ability to handle variable-sized inputs provides the necessary extension, or whether the theoretical guarantees simply do not apply to the main use case.

- **Theoretical guarantees rely on "most-expressive" GNNs not used in practice.** Theorems 3.2 and 3.3 condition on maximally expressive GNN encoders, but practical implementations use standard message-passing networks (NBFNet, GINE) which have known expressivity limitations under the Weisfeiler-Leman hierarchy. The drop of feature-ID labels to achieve COGG-invariance (Section 3.2) is explicitly acknowledged to "sacrifice maximal expressivity." A more honest framing would clarify that the theorems establish what is *representable* rather than what will be *learned* in practice.

- **Computational cost is under-analyzed in the main text.** For each edge in the original graph, STAGE constructs a STAGE-edge-graph with 2d nodes and O(d²) edges. For a graph with |E| edges, this creates |E| subgraphs. For moderate d and large |E|, this is non-trivial. Appendix F reportedly contains complexity analysis, but scalability claims for large graphs (e.g., Friendster with millions of edges) need prominent discussion in the main text, including any approximations or efficiency considerations.

- **No ablation isolating the contribution of conditional dependencies.** The core claim is that modeling *statistical dependencies* (off-diagonal elements of S^{uv}) enables transfer. Yet the paper does not compare against a simpler baseline that uses only marginal probabilities (diagonal of S^{uv})—essentially rank-normalizing each feature independently without capturing cross-feature dependencies. Such an ablation would isolate whether the method succeeds because dependencies transfer, or simply because rank-based normalization removes domain-specific value scales.

- **Node classification evaluation is thin.** The node classification experiments use only one train–test pair (Friendster→Pokec) and one task (gender prediction). The age regression task is mentioned but results are apparently uninformative. With only one domain pair, it is unclear whether the 10% improvement generalizes beyond this specific setting. Additional node classification benchmarks would substantially strengthen this portion.

- **Missing connection to copula theory.** The empirical conditional CDF representations (p(x_i^u | x_j^v)) are mathematically related to empirical copulas, a well-established framework for modeling dependence structures independently of marginal distributions. The paper's theoretical framing would be strengthened by acknowledging this connection and situating the work relative to copula-based methods in statistics and machine learning.

## Nice-to-Haves

- **Explicit probability estimation protocol:** The paper should clarify how empirical conditional probabilities are computed for continuous features—is this done via empirical CDF, binning, or kernel density? This detail affects reproducibility and behavior on sparse data.

- **More seeds for statistical confidence:** While 3 seeds is not unusual, the variance in some baselines (e.g., std=0.025 for GINE-gaussian in Table 2) suggests that additional seeds would provide more robust conclusions, particularly for the node classification results.

- **Discussion of failure modes:** The method assumes analogous statistical dependencies exist across domains. If the dependency structure fundamentally differs (e.g., income correlates with price in train but not in test), how does the method degrade? Some analysis of this failure mode would be valuable.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Claim that NBFNet-raw achieving 0.0000 is an implementation bug:** This is not a bug—it correctly reflects that raw feature dimensions mismatch between train and test domains, so a model trained on one feature space cannot process a different feature space. This is the fundamental problem STAGE is designed to solve.

- **Criticisms about the supervised H&M baseline being weak:** The structural-supervised baseline serves its intended purpose: showing that zero-shot STAGE can compete with a supervised model that only uses graph structure. A stronger supervised model using H&M's own features would be an upper bound comparison, but the current baseline adequately contextualizes STAGE's capability.

- **Concern about 3 seeds being insufficient for statistical power:** While more seeds would strengthen the paper, 3 seeds with the magnitude of improvements shown (e.g., 0.47 vs 0.23 MRR) and the consistent variance patterns provides reasonable evidence for the main claims. This is within ICLR norms.

- **Demand for user studies or confidence intervals on large benchmarks:** The reviewer requested more rigorous statistical testing, but the experiments follow standard practices for the benchmarks and task types used.

- **Formatting and style nitpicks:** These were correctly filtered as not substantive.

- **Bipartite graph handling as a flaw:** The approach of adding edges between nodes of the same type is a reasonable engineering solution for bipartite graphs and does not undermine the method's contribution.

## Novel Insights

The key insight from synthesizing the reviews is that STAGE's fundamental contribution can be understood through the lens of *representation alignment without semantic correspondence*. Unlike domain adaptation methods that assume shared label spaces or feature alignment techniques that require corresponding features, STAGE transfers by identifying that certain *statistical patterns* (correlations, dependencies) recur across domains even when the underlying features have nothing in common. This is conceptually similar to how meta-learning finds "learning algorithms" that transfer, but applied to dependency discovery. The empirical finding that performance *improves* with more training domains (Figure 4)—unique to STAGE among all methods—suggests it is genuinely learning transferrable dependency patterns rather than overfitting to specific feature statistics. This positions STAGE as learning a "dependency discovery algorithm" that can be applied to any domain, which is a distinct conceptual contribution beyond the specific architecture.

## Suggestions

- **Add an ablation using only marginal probabilities:** Include a "STAGE-marginal" variant that uses only diagonal elements of S^{uv} (marginal CDFs without conditional dependencies). This would directly test whether capturing dependencies drives the improvements versus simple rank-based normalization.

- **Prominently discuss computational complexity in main text:** Move key complexity analysis from Appendix F to the main paper, including runtime/memory measurements on the experimental datasets. Discuss any approximations for large graphs.

- **Extend theory section to address the fixed-d limitation:** Either extend the proofs to variable dimensions (if straightforward), or add a discussion of how the theoretical guarantees might degrade or hold under dimension mismatch. Clarify whether the GNN's ability to handle variable-sized inputs provides a practical bridge.

- **Add at least one more node classification benchmark:** A second train–test domain pair would substantially strengthen confidence in the node classification results, which currently rest on a single dataset pair.