=== CALIBRATION EXAMPLE 3 ===

# Final Consolidated Review
## Summary

ACSP introduces a pruning method for CNNs that automatically determines the pruning volume per layer without manual tuning. It constructs a graph space encoding each component's class-pair separability (via JM distance), selects complementary components using k-Medoids clustering with the Mean Simplified Silhouette index, and determines the subset size automatically via a knee-finding algorithm. Experiments across VGG, ResNet, DenseNet, and MobileNet on CIFAR and ImageNet show competitive accuracy retention with significant FLOP reductions.

## Strengths

- **Automatic pruning volume determination**: The combination of MSS scoring with knee-finding eliminates the need for manual pruning-ratio specification or sensitivity analysis—a genuine practical advance over methods requiring per-layer hyperparameter search. The automation is principled (data-driven knee point) rather than heuristic.
- **Complementary selection principle**: Formulating component selection as a graph-space diversity problem is a distinct and well-motivated departure from standard magnitude-based or reconstruction-error-based pruning. The idea that redundant separation capabilities should be minimized is intuitive and the MSS-based implementation is technically coherent.
- **Broad empirical coverage**: Evaluation across six architectures and three datasets (including ImageNet-1K) provides meaningful evidence of generalizability. The method achieves accuracy gains on several configurations while reducing FLOPs, which is non-trivial.

## Weaknesses

### Major:

- **FLOP reduction ≠ inference speedup, and the gap is severe.** The Abstract states ACSP "results in faster inference time… (e.g., 2.25× on ResNet-50)," but Table 2 reports only an 8.07% single-inference latency reduction for ResNet-50—not a 2.25× speedup. The 2.25× figure is a FLOP ratio (Table 1). Across all models, FLOP reductions of 1.5–2.6× translate to latency improvements of only 3–9%. The paper acknowledges in Section 4.5 that "hardware utilization is not perfectly linear with FLOP count," but a 2.25× FLOP reduction yielding ~8% latency improvement is far beyond typical non-linearity—it suggests the pruned architectures may have irregular channel counts that degrade hardware utilization (e.g., tensor core alignment). For a paper whose stated focus is "accelerating inference time" (Abstract), this discrepancy substantially undermines the core value proposition. The authors should analyze why latency gains are so modest (memory bandwidth bottleneck? irregular layer shapes?) and report latency as the primary efficiency metric.

### Minor:

- **No ablation validating complementary selection vs. magnitude-only selection.** The central claim is that complementary (diverse) component selection outperforms redundancy-agnostic selection. Yet no experiment compares ACSP against a "top-k by weight magnitude only" baseline using the same automatic knee-finding framework. Without this, it is unclear whether the graph construction, JM distance computation, and k-Medoids clustering contribute meaningfully beyond what simple weight-ranking within the same automatic volume-selection framework would achieve. This is a critical missing ablation for the paper's core contribution.

- **Computational overhead of graph construction and clustering is not analyzed.** For ImageNet (C=1000), constructing the separation matrix requires computing JM distances for ~500K class pairs per component. The paper states the Kneedle step takes <0.1s but does not report the time for graph construction or for running k-Medoids for every k ∈ [2, Nᵢ]. The Conclusion acknowledges the C² scaling as a limitation but provides no measurements on ImageNet to demonstrate tractability.

- **Layer-wise fine-tuning overhead is unquantified.** Algorithm 1 fine-tunes the model after pruning each layer. For a 50-layer ResNet, this means 50 sequential fine-tuning cycles (each 2–3 epochs on 25% of data). The cumulative cost relative to standard "prune-all-then-fine-tune" approaches is not reported, making it difficult to assess the total pruning pipeline cost.

- **Gaussian assumption underlying JM distance is not discussed.** The Bhattacharyya distance (Eq. 2) and thus the JM distance assume Gaussian class-conditional activation distributions. After ReLU, activations are non-negative and typically skewed or sparse, violating this assumption. The paper does not discuss the robustness of JM distance to distributional misspecification, nor whether this affects pruning decisions in practice.

- **k-Medoids stochasticity not addressed.** k-Medoids is not deterministic across runs. An "automatic" method that produces different pruning ratios per run raises reproducibility concerns. No variance or stability analysis across multiple runs is provided.

### Trivial:

- The terminology "Speed Up" in Table 1 is misleading when it refers to FLOP ratios rather than measured latency. Consistent labeling (e.g., "FLOP Reduction") would improve clarity.

## Nice-to-Haves

- An accuracy-vs.-measured-latency Pareto plot comparing ACSP with baselines would immediately clarify whether ACSP dominates on the metric that matters for deployment.
- Visualizing the automatically selected pruning ratios per layer against known manually-tuned profiles would reveal whether the knee-finding algorithm captures meaningful architectural structure.
- Integrating a latency model into the pruning objective (rather than optimizing FLOPs) could align the method more directly with its stated goal of inference-time efficiency.
- An ablation comparing JM, Hellinger, and Wasserstein distances with full results (not just a brief mention) would strengthen the metric-choice justification.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Missing 2025-2026 baselines"** — Removed per rule against criticizing missing related works; the paper's comparison set is already substantial.
- **"Restricted to supervised learning limits applicability"** — Removed (weakened); the paper explicitly scopes to supervised tasks in Section 3.2. Criticizing the absence of unsupervised support is scope creep.
- **"Algorithm 1 Line 12 contradicts Section 3.4.2"** — Removed; these are equivalent (k' clusters → one top-weight component per cluster → k' components total). The phrasing differs but the semantics are consistent.
- **"Fine-tuning protocol is too light / unfair comparison"** — If baselines used more fine-tuning and ACSP still matches them with less, the asymmetry favors the baselines (stronger competitor), not ACSP. Per the hard rule on unfair comparisons, this criticism is removed. If anything, achieving competitive accuracy with 3 epochs is a strength.
- **"Memory requirements for storing activations on ImageNet"** — Weakened to trivial and not listed separately; this is standard for any activation-based method and not unique to ACSP.

## Novel Insights

The FLOP-vs.-latency discrepancy reveals a subtle but important point: methods that prune channels based on class-separability criteria may inadvertently produce channel counts that are poorly aligned with hardware tensor-core requirements (e.g., multiples of 8 or 32). A channel count of 53 is functionally different from 64 on an RTX 6000, even if the FLOP difference is modest. This suggests that complementary separation pruning, while theoretically sound for maintaining accuracy, may produce architectures that are "semantically efficient but structurally inefficient"—a previously underappreciated tension in the pruning literature between preserving discriminative capability and maintaining hardware-friendly tensor shapes.

## Suggestions

- **Report latency as the primary metric.** Re-title Table 1's "Speed Up" column as "FLOP Reduction," and lead with Table 2 results in the Abstract and Introduction. If the latency gains are genuinely the goal, they should be front and center.
- **Add the critical ablation**: Compare ACSP against a "Weight-Only" variant that uses the same automatic knee-finding but selects top-weight components without the graph-space clustering step. This directly tests whether complementary selection matters.
- **Quantify total pruning pipeline cost**: Report wall-clock time for graph construction + clustering + fine-tuning on ImageNet, so readers can assess whether the automation overhead is acceptable.
- **Pad channel counts to hardware-friendly multiples** post-pruning (or incorporate alignment constraints into the selection), which could substantially close the FLOP-to-latency gap.

# Actual Human Scores
Individual reviewer scores: [2.0, 0.0, 2.0]
Average score: 1.3
Binary outcome: Reject
