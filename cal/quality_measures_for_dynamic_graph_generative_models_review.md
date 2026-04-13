=== CALIBRATION EXAMPLE 3 ===

# Final Consolidated Review
## Summary

This paper proposes a novel evaluation metric for Continuous-Time Dynamic Graph (CTDG) generative models based on Johnson-Lindenstrauss random projections. The method encodes each node's temporal event sequence into a fixed-dimensional embedding via random projections, then aggregates across nodes to produce a unified scalar distance metric. The authors argue this overcomes limitations of existing snapshot-based metrics: i.i.d. assumptions between snapshots, lack of unified feature-topology sensitivity, and computational inefficiency from explicit snapshot instantiation.

## Strengths

- **Addresses a genuine methodological gap.** The paper correctly identifies that existing CTDG evaluation relies on discretizing continuous temporal data into snapshots, then applying i.i.d. assumptions that are violated for temporally dependent events. The proposed approach avoids snapshot instantiation entirely, which is both conceptually cleaner and computationally more efficient.

- **Captures feature-topology interactions uniquely.** As shown in Figure 1 and Table 1, the JL-Metric is the only method with measurable sensitivity to "Event Permutation" perturbations (median Spearman correlation 0.988), where event features are permuted while preserving topology and marginal feature distributions. This demonstrates genuine capability to capture dependencies between topology and features that classical metrics miss.

- **Computational efficiency demonstrated empirically.** Table 1 shows JL-Metric runs at 1.05 s/100 events compared to 8.41–11.99 s/100 events for snapshot-based topological metrics, confirming the theoretical complexity advantages of avoiding explicit snapshot construction.

- **Comprehensive empirical framework.** The evaluation adapts established protocols from image and static graph domains (fidelity, diversity, sample efficiency, computational efficiency) across five datasets with multiple perturbation types, providing a thorough assessment of metric behavior.

## Weaknesses

- **Variable-length projection may not satisfy JL distance-preservation guarantees.** The paper applies projection matrix $W_1^{M \times n}$ to node event sequences of different lengths by "ignoring unused rows of the matrix where necessary" (Section 3). This means nodes with fewer events are effectively projected by a different sub-matrix than nodes with more events. The JL lemma guarantees distance preservation for a *fixed* projection applied to all points in a set; using different effective projections for different nodes breaks this guarantee as stated. The claim that "JL embedding quality is agnostic to vector length" addresses comparing two vectors of equal dimension, not comparing embeddings derived from vectors of different dimensions via different projections. A formal argument or alternative formulation (e.g., consistent zero-padding) would strengthen the theoretical foundation.

- **Node ordering sensitivity limits general applicability.** The second projection $W_2^{Z \times o}$ requires aggregating per-node embeddings into a matrix, then computing Frobenius cosine similarity. For this to yield consistent distances, nodes must be ordered identically across compared graphs. The paper does not address how this works when generated graphs have different node sets or different node identifiers than the reference graph—common scenarios in generative modeling. The metric in its current form is not permutation-invariant to node relabeling.

- **No evaluation on actual generative model outputs.** All experiments use synthetic perturbations of real graphs (edge rewiring, time perturbation, event permutation, mode dropping/collapse). There is no experiment evaluating actual outputs from CTDG generative models (TagGen, TIGGER, Dymond, TG-GAN) against independent quality criteria such as downstream task performance or human evaluation. A metric that responds well to synthetic perturbations may still fail to distinguish real generative model failures if those failures don't align with the perturbation types tested.

- **Missing random temporal GNN baseline.** The paper motivates its approach by connecting random neural network feature extraction (Thompson et al., 2022) to JL projections, but does not compare against a random temporal GNN baseline. A natural comparison would apply random weights to an existing temporal GNN architecture (e.g., TGN) and use the resulting embeddings for distance computation. Without this baseline, it's unclear whether the advantage comes from the JL formulation specifically versus simply using any neural embedding approach.

- **Hyperparameter selection raises potential for data leakage.** Appendix D describes grid search over embedding dimensions $n$ and $o$. If this grid search uses the same datasets and perturbation experiments as the evaluation, the reported performance may be optimistic. For an "application-agnostic" metric, hyperparameters should either be fixed universally or selected on held-out data.

## Nice-to-Haves

- **Downstream task correlation:** Demonstrating that generated graphs ranked higher by the JL-Metric perform better on downstream tasks (e.g., link prediction, node classification) would strengthen the claim that the metric captures practically meaningful quality.

- **Ablation on projection dimensions:** Including sensitivity analysis for $n$ and $o$ in the main text would improve confidence in robustness without requiring dataset-specific tuning.

- **Handling graphs with different node counts:** A practical discussion of how the metric handles cases where generated graphs have vastly different numbers of nodes than reference graphs.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Likelihood-based methods dismissal:** The harsh critic notes the paper dismisses likelihood-based methods as intractable, suggesting diffusion-based or flow-based models may have tractable likelihoods. However, this is scope creep—the paper explicitly focuses on sample-based methods, which is a valid methodological choice.

- **Theoretical justification for why random GNNs work:** The harsh critic notes the paper discusses random GNNs but doesn't use them, calling this "unnecessary scaffolding." While valid as a writing critique, the GNN discussion serves as motivational context for why random projections might work, not as a claimed contribution. The actual method directly applies random projections to event sequences.

- **Single-sample distribution estimation:** The critic argues the single-sample approach cannot capture distributional diversity. However, the paper explicitly notes this is "common in the CTDG literature" due to stationarity assumptions, and the diversity experiments test sensitivity to structural perturbations within one graph—which is a valid design choice given field conventions.

- **Frobenius cosine not being a proper metric:** The triangle inequality concern is technically valid but not central to the paper's claims, since the metric is used for ranking rather than metric-space operations.

- **Multiple additional experiments requested by spark finder:** Many (trained embedding baselines, additional 2023-2024 baselines, differentiable loss function extension) are beyond the paper's stated scope of introducing a sample-based metric.

## Novel Insights

The connection between random neural network feature extraction (as in Inception Score, FID, and random GNN metrics) and the Johnson-Lindenstrauss lemma is an insightful theoretical contribution. While the extension to variable-length sequences breaks the formal guarantee, the core insight—that random projections can provide unified embeddings for dynamic graph data without training—deserves attention. The empirical finding that only the JL-Metric detects event permutation perturbations (where feature-topology relationships are altered but marginal statistics preserved) suggests that classical metrics fundamentally miss a class of generative failures that this approach captures.

## Suggestions

- Provide a formal specification (algorithm box) for handling variable-length sequences, clarifying whether zero-padding or sub-matrix selection is used and how this affects distance preservation properties.

- Evaluate the metric on samples from at least one actual CTDG generative model, comparing the metric's rankings against an independent quality measure (e.g., likelihood on held-out events or downstream task performance).

- Add a simple experiment using a random temporal GNN as a feature extractor to isolate whether the benefit comes from the JL projection specifically or from any random embedding approach.

- Clarify how node ordering is handled when comparing graphs with different node sets, or acknowledge this as a limitation for the specific use case of comparing generated graphs to reference graphs with different node identities.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept
