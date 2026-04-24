## Summary
This paper presents an empirical study applying topological data analysis (TDA), specifically persistent homology and Betti curve similarity (BCS), to analyze the functional graphs of convolutional neural networks trained on disjoint subsets of ImageNet. The pipeline reduces neuron activations via k-means++ clustering, constructs correlation-based distance matrices, and computes persistent homology to derive Betti curves for comparing architectures, epochs, and data subsets.

## Strengths
- **Novel application of Betti Curve Similarity to compare DNN internal representations**: The paper defines and operationalizes BCS (Equation 7, Section 2.5) as a scalar metric for topological comparison across models, epochs, and datasets. Figures 4 and 5 demonstrate that BCS tracks structural convergence during training and distinguishes AlexNet from VGG-16 representations.
- **Computationally tractable pipeline for large-scale TDA on neural activations**: The design combining Spearman correlation distance (Equation 1) with GPU-accelerated k-means++ reduction to 1,000 points makes persistent homology feasible on high-dimensional CNN activations, completing experiments in ~66 minutes (Section 2 introduction).
- **Honest acknowledgment of methodological limitations**: The paper explicitly reports that silhouette scores show poorly separated clusters (Section 2.3, line 79) and discusses the approximation error from dimensionality reduction (lines 83-85), rather than masking these constraints.
- **Clear visualization pipeline**: Figures 3-6 effectively communicate how persistence diagrams and Betti curves evolve across training epochs and model architectures, making TDA outputs interpretable for deep learning practitioners. Figures 7-9 provide concrete case studies linking BCS observations to accuracy trajectories on specific subsets.

## Weaknesses

### Fatal
None

### Major
- **No comparison to established representational similarity baselines**: The paper claims BCS provides a "nuanced understanding" complementary to accuracy but never demonstrates how BCS differs from or improves upon well-validated metrics like Centered Kernel Alignment (CKA), Representational Similarity Analysis (RSA), or standard TDA distances (Bottleneck/Wasserstein). Without such comparisons, readers cannot assess whether BCS captures signals already available through simpler methods. This is a fundamental gap for a paper whose central contribution is a new similarity metric.
  
- **No null experiments or statistical validation of BCS**: The paper draws strong interpretive claims (e.g., "models are converging towards the same global structure," "models are creating distinct internal representations") from visual heatmap trends without reporting statistical significance, variance across the 30 subsets, or ablation against null baselines such as randomly initialized networks or shuffled weights. Without a null distribution, it is impossible to determine whether observed topological differences exceed what one would expect from architectural depth/width alone or random correlation structure.

- **Training regime produces models with limited representational maturity**: Models are trained for 60 epochs with a static learning rate of 0.001, plateauing at ~40-45% test accuracy on 10-class ImageNet subsets (Figure 2). While the accuracy curves do plateau, this is well below what these architectures can achieve on such limited classification tasks. Claims about "convergence of the network's functional graph towards some global structure" (Section 3.1) become difficult to interpret when the underlying models have not converged to high-performance minima. The observed topological shifts may reflect optimization transients rather than stable learned representational geometry. This cannot be resolved without extended training experiments or analysis of whether topological features stabilize after accuracy plateaus.

### Minor
- **Dimensionality reduction quality directly impacts TDA input**: The paper's k-means++ reduction produces clusters with poor silhouette scores, which the authors acknowledge (line 79) but then argue is acceptable because "the local structure of the neuron activations is not as important as the global structure" (lines 83-85). This reasoning is asserted without evidence. No ablation is provided to show that PH outputs are stable across different clustering methods, seed initializations, or reduction targets (e.g., 500 vs. 2000 clusters). Given that Vietoris-Rips filtration is sensitive to point cloud construction, this approximation step's impact on the final Betti curves should be quantified.

- **The distance function $d_\rho$ is a pseudometric, not a metric**: The paper notes that $d_\rho$ violates strict positivity (Section 2.4) but does not address how pairs of neurons with equal correlation but different activation magnitudes should be handled, or whether this "collapses" distinct neurons into topological degeneracy. While pseudometrics are used in TDA practice, the implications for this specific application need brief discussion.

- **No quantitative clustering/classification evaluation of BCS**: The abstract states BCS "can distinguish between different DNN models across datasets," but no classification or clustering accuracy evaluation of BCS as a discriminative metric is presented. The heatmaps in Figures 4-6 show qualitative patterns but lack downstream quantitative validation (e.g., can a classifier using BCS features correctly identify architecture pairs?).

### Trivial
- The introduction's positioning of TDA as "a candidate tool for analyzing the global structure" (Section 1) somewhat undersells the contribution — the paper already delivers results, so framing could be more assertive about what was found rather than what "can be" done.
- Figure 3's persistence diagrams show H1 features with reasonable persistence, but the paper does not discuss why higher-dimensional features (H2, H3) appear predominantly near the diagonal (noise level) — whether this reflects genuine absence of higher-order structure or computational limitations of the reduction step.

## Nice-to-Haves
- Ablation experiments varying the k-means++ reduction size (e.g., 500, 2000 clusters) to demonstrate the stability of Betti curves across approximation levels.
- Extended training runs with learning rate schedules to verify whether topological convergence continues beyond epoch 60 or stabilizes alongside accuracy.
- Demonstration of BCS on a known perturbation (e.g., removing residual connections from ResNet, applying dropout) to show it detects changes that accuracy metrics miss.
- Comparison of BCS to Bottleneck/Wasserstein distances on the reduced point clouds to validate that the simpler L-infinity norm on Betti curves is an adequate proxy for more theoretically grounded TDA distances.

## Removed Points
These points are flagged to be removed; treat them with caution.

1. **"Structurally decouples topological features from neural geometry"** — The harsh critic claims the k-means++ reduction "structurally decouples the topological features from the actual neural representational geometry," invalidating the core claim. This overreads the paper: the paper itself acknowledges poor silhouette scores and argues the reduction is acceptable for capturing global structure (lines 83-85). While the reduction quality concern is valid (moved to Minor), the critic's claim that this fully invalidates the approach is not substantiated — TDA on reduced point clouds is a standard scaling strategy in the TDA literature, and the paper does not claim the reduction is perfect, only practical. The severity of this criticism is inflated.

2. **"Severely undertrained models"** — The harsh critic claims ~40-45% accuracy on 10-class tasks indicates "severe undertraining." While the accuracy is modest, the learning curves do plateau (Figure 2), suggesting the models have converged given the architecture and 64×64 input resolution constraints. This concern is valid (included in Major as "limited representational maturity") but the critic's framing of it as completely undermining convergence claims is overstated — the paper could be improved by extended training, but the current setup is not fundamentally broken.

3. **"Literature review incomplete"** — Removed per the hard rule on not flagging missing related works. The paper cites Corneanu et al. (2019), Edelsbrunner & Harer (2010), and several TDA references, which is sufficient scope for an empirical study.

4. **"Ad-hoc similarity metric lacks statistical validation" (critic version)** — Partially kept as Major (no null experiments, no statistical validation). What is removed: the critic's criticism of the ∞-norm itself as "ad-hoc" compared to Bottleneck/Wasserstein. The BCS is a standard approach — Betti curves are common TDA summaries, and computing distances between them via L-infinity is a natural choice. The real weakness is the lack of statistical grounding, not the choice of norm itself.

5. **"Pseudometric positivity violation"** — Included as Minor above but weakened significantly. The paper explicitly acknowledges this (Section 2.4) and notes it is discussed in prior work (López De Prado, 2016). This is a known property of correlation-based distances in TDA practice, not an overlooked error.

6. **"Overclaims in Abstract"** — The critic says no quantitative classification evaluation validates the abstract's claim that BCS "can distinguish between different DNN models." This is partially true (included as Minor), but the abstract is consistent with what the paper demonstrates qualitatively — it is not an overclaim of false results, merely a lack of supporting quantitative analysis.

## Novel Insights
The paper's most compelling contribution is the empirical observation that Betti curve similarity captures representational geometry that tracks both training progression and architectural differences independently of accuracy metrics. The case studies on subsets 11 and 27 (Sections 3.2, Figures 6-9) suggest that ResNet's residual connections induce topologically distinct functional graphs compared to VGG-16 and AlexNet on morphologically distinct image classes — a pattern not directly visible in accuracy curves alone. If validated with proper baselines and null controls, this could establish TDA as a complementary diagnostic for probing how architectural choices shape learned representations beyond what accuracy or loss can reveal.

## Suggestions
1. **Add null baselines**: Run the pipeline on randomly initialized networks and/or weight-shuffled networks to establish a null distribution for BCS. This is essential to demonstrate that the metric captures learned structure rather than architectural depth/width artifacts.
2. **Compare to CKA/RSA**: Include Centered Kernel Alignment or RSA on the same activation data as a baseline. Even a single dataset/architecture comparison would show whether BCS provides genuinely additional insight beyond existing representational similarity tools.
3. **Report variance and significance**: Add error bars or confidence intervals to the BCS heatmaps and report whether observed differences are statistically significant across the 30 subsets. Currently, the results are purely descriptive.
4. **Ablate k-means++ reduction**: Run the pipeline on 2-3 different reduction sizes (e.g., 500, 1000, 2000 points) and report the correlation or BCS between the resulting Betti curves to demonstrate stability of the topological features under approximation.

## Score and Decision

**Calibration anchors consulted:**

| Anchor Paper | Avg Score | Comparison |
|---|---|---|
| **High (≥6):** Q0zmmNNePz (topology-preserving segmentation, 7.50) | 7.50 | Well above — that paper has rigorous validation, a topology-aware loss, and strong empirical results. |
| **High (≥6):** mYgoNEsUDi (zigzag persistence for diffusion models, 6.33) | 6.33 | Above — that paper uses established TDA methods with clear novelty and solid experiments. |
| **Medium (≈5):** sq5gkjC9jv (topological expressivity of ReLU nets, 5.67) | 5.67 | Similar category — interesting theoretical idea but mixed reception due to limited practical experiments. This paper has more empirical breadth but weaker validation. |
| **Medium (≈5):** NiCSyYOfex (node-level TDA, 5.33) | 5.33 | Similar — interesting TDA idea with methodological gaps and missing validations. This paper is slightly weaker due to missing all baselines. |
| **Low (≤4):** BXMoS69LLR (blind baselines beat membership inference, 4.50) | 4.50 | Similar — makes an interesting empirical claim but with missing experimental details and limited scope. |
| **Low (≤4):** izDiFGXn9B (wCKA similarity metric, 3.50) | 3.50 | Slightly below — that paper had a promising weight similarity metric but was limited to MNIST and lacked large-scale validation. Similar pattern: novel metric, insufficient breadth. |
| **Low (≤4):** 7pIxS9m283 (WISE-GNN topological encoding, 3.00) | 3.00 | Above — that paper had limited originality and small datasets. This paper has more genuine novelty in its TDA application. |

This paper sits squarely between the low and medium anchors. It has genuine novelty in applying Betti curve similarity to compare CNN functional graphs across architectures and datasets, and the pipeline is coherent and reproducible. However, it is critically missing: (1) comparison to established representational similarity baselines, (2) null experiments to validate the metric, and (3) statistical grounding for the interpretive claims. These are not fixable with minor revisions — they require additional experiments. Compared to sq5gkjC9jv (5.67, accepted by one reviewer with an 8), this paper has weaker methodological rigor. Compared to izDiFGXn9B (3.50), it has more breadth and honest self-assessment. The closest anchor is BXMoS69LLR (4.50) — interesting empirical findings undermined by missing validation.

I position this paper slightly above the lowest anchors due to genuine novelty and clear pipeline, but below the medium anchors due to fundamental gaps in experimental validation.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>