Now I have all the information I need. Let me synthesize the final review.

## Summary

This paper proposes using manifold capacity theory—originally developed for the untangling hypothesis in neuroscience—as a representation-based framework to quantify feature learning in neural networks. The key contributions are: (1) theoretical and empirical evidence that manifold capacity tracks the degree of feature learning (Theorem 1 for 2-layer networks; empirical validation on VGG-11, ResNet-18, and RNNs), (2) geometric decomposition of capacity into radius, dimension, and alignment measures that reveal distinct "learning strategies" and "learning stages" beyond the lazy/rich dichotomy, and (3) applications to neuroscience (structural inductive biases in RNNs) and machine learning (geometric correlates of OOD generalization failure).

## Strengths

- **Novel and well-motivated application of manifold capacity theory to feature learning.** The connection between manifold capacity and feature learning is natural and previously unexplored. The representation-based approach directly addresses the neuroscience gap where synaptic weights cannot be directly observed (Section 1, Section 5.1). The theoretical link between capacity and geometric measures (Equation 1, Section 2.2) provides a principled decomposition that is more interpretable than raw capacity alone.

- **Capacity outperforms conventional measures in specific settings.** Figure 3a demonstrates that capacity correctly orders the degree of richness (controlled by scale parameter η̃) more clearly than weight changes, NTK-label alignment, or representation-label alignment. Figure 3b shows a concrete, non-trivial advantage: capacity correctly identifies differences in task-relevant features at initialization, while representation-label alignment yields the wrong ordering.

- **RNN experiment demonstrates subtypes invisible to the lazy/rich label.** Figure 5d shows that RNNs with different initial weight ranks achieve the same final capacity but with opposite geometric organizations (large radius/small dimension vs. small radius/large dimension). This is a clear demonstration that "rich learning" encompasses multiple mechanistically distinct subtypes that the dichotomy misses.

- **Learning stages reveal dynamics invisible to accuracy curves.** Figure 4c identifies four stages of geometric change in VGG-11 training on CIFAR-10, even after accuracy has saturated. This shows the geometric measures capture representational dynamics that accuracy alone cannot, providing genuine new information.

- **Architectural difference in OOD failure mode.** The finding that radius expansion explains OOD failure in VGG-11 while dimension increase explains it in ResNet-18 (Figure 21) is a nuanced, architecture-aware diagnosis that simpler measures would not provide.

## Weaknesses

### Fatal
None.

### Major

- **The "beyond the dichotomy" claim is asserted more strongly than the evidence supports.** The paper identifies different geometric trajectories (learning "strategies" and "stages") and shows they exist, but does not demonstrate that these subtypes have predictive power for any downstream outcome that the lazy/rich label alone fails to predict. The RNN experiment (Figure 5d) shows different geometries for the same capacity, but the paper does not establish whether these geometric differences have any functional consequence (e.g., different noise robustness, different memory properties, different transfer learning ability). Without at least one experiment where geometric subtype—rather than just capacity level—predicts a downstream outcome, the observation that "different conditions produce different geometric paths" extends the vocabulary of the dichotomy but not its explanatory power. The paper's own conclusion acknowledges this as future work (Section 6).

- **Capacity is not adequately compared against simpler representation-based baselines.** Throughout the paper, capacity and test accuracy track each other closely (Figures 2b, 3a, 6b). The paper argues capacity is superior to weight-change and alignment measures, but never establishes that capacity provides information beyond simple linear separability of representations—e.g., linear probe accuracy or SVM margin on the penultimate layer—which would capture a very similar notion of manifold "untangling" at far lower computational cost. While Figure 3b shows capacity outperforms one alignment measure at initialization, and Figure 4c shows geometry continues evolving after accuracy saturates, the absence of a linear separability baseline leaves open the possibility that capacity is essentially an expensive proxy for how linearly separable the representations are. The geometric decomposition adds interpretive vocabulary, but its independent explanatory power over simpler measures is unestablished.

### Minor

- **Theorem 1 covers only one gradient step in a restricted setting.** The theoretical result establishes a monotone relationship between capacity and feature learning degree for a single gradient step in a 2-layer network with fixed readout weights (Section 3.1, footnote 6). The paper is transparent about this limitation, and the empirical evidence extends well beyond this setting. However, the gap between the theorem (one step, toy model) and the paper's framing ("geometry insights in feature learning" broadly) is substantial. This does not invalidate the empirical findings but means the theoretical pillar supports a narrower claim than the paper's scope.

- **Learning stages are identified by visual inspection without formal criteria.** The four stages in Figure 4c (clustering, structuring, separating, stabilizing) are identified by visual inspection of normalized geometric measure trajectories. There is no formal criterion for stage boundaries, no test of robustness across architectures/datasets, and no validation against an external criterion. The naming imposes narrative structure on what could be continuous, architecture-dependent dynamics.

- **The OOD generalization analysis uses a mild distribution shift.** Section 5.2 treats CIFAR-100 as OOD relative to CIFAR-10 pretraining, but both are natural-image datasets from the same 80 Million Tiny Images corpus sharing low-level features and some semantic categories. The linear probe accuracy drop in the "ultra-rich" regime is more naturally interpreted as standard overfitting than as a novel geometric discovery, though the geometric decomposition does provide additional structural insight (e.g., the architectural difference between VGG-11 and ResNet-18 in which geometric measure drives the drop).

### Trivial
None.

## Nice-to-Haves

- **Regularization experiments based on geometric insights.** The paper identifies radius expansion and center-axis alignment as correlates of OOD failure (Section 5.2). Penalizing these quantities during training to test whether OOD performance improves would turn a descriptive finding into an actionable one. The authors acknowledge this as future work.

- **Causal manipulation of geometry.** The paper describes geometric changes that accompany capacity changes but does not investigate whether manipulating geometry (e.g., regularizing radius or dimension during training) causally affects feature learning or generalization.

- **Sensitivity analysis of geometric measures.** The geometric measures require solving a QP (Algorithm 1). Sensitivity to the number of random projections/dichotomies sampled and scaling with representation dimension and number of classes would strengthen the practical utility claims.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh critic: "The geometric decomposition is a principled accounting identity rather than an independent explanatory framework."** While technically accurate that the geometric measures are derived from anchor points solving the capacity optimization problem, this is by design—the decomposition is meant to explain *how* capacity changes, not provide independent evidence. The measures offer genuinely interpretable components (radius, dimension, alignment) that reveal distinct learning strategies and stages. Calling it an "accounting identity" understates its utility; it's more like a variance decomposition that reveals which factors drive changes. Moved to minor/reframed.

- **Harsh critic: "The OOD analysis is circular because the ultra-rich regime is defined post-hoc."** The ultra-rich regime is defined by the inverse scale factor η̃ ≈ 1.0, which is a pre-specified parameter of the training procedure, not defined post-hoc based on where OOD performance drops. The critic mischaracterizes the experimental design.

- **Harsh critic: "CIFAR-100 as OOD is just overfitting, not a novel discovery."** While the distribution shift is mild, the geometric decomposition does provide genuinely new information beyond "it overfits"—specifically, the architectural difference in which geometric measure drives the drop (radius in VGG-11 vs. dimension in ResNet-18). This is not something standard overfitting analysis reveals. However, the mild OOD test remains a minor concern.

- **Strength finder: "Rigorous theoretical justification that manifold capacity tracks feature learning."** The theorem is limited to one gradient step in a toy model. Calling this "rigorous" overstates the scope. The contribution is real but narrow; the strength has been rephrased to acknowledge the limitation.

- **Harsh critic: Request for t-SNE/UMAP visualizations of manifolds.** This is a presentation suggestion, not a substantive weakness. The contour plots and heatmaps already communicate the geometric changes effectively. Removed as trivial.

- **Harsh critic: Missing related works.** Per instructions, I do not flag missing related works.

- **Harsh critic: Reproducibility concerns about hyperparameters and implementation details.** Per instructions, these are removed as nitpicks about reproducibility.

## Novel Insights

The paper's most genuinely novel insight is that "rich learning" is not a monolith: the RNN experiment (Figure 5d) demonstrates that networks can arrive at the same capacity value through different geometric paths, resulting in distinct structural organizations even when performance is identical. This suggests that feature learning regimes have internal structure that current binary classifications miss. However, the critical open question—whether these geometric differences matter for anything beyond description—remains unanswered, and this gap between observation and functional consequence is the paper's most important limitation.

## Suggestions

- **Add a linear probe / SVM margin baseline.** Compute linear probe accuracy (or SVM margin) on the penultimate-layer representations alongside capacity in the main experiments. If capacity provides information that linear separability does not (as Figure 3b hints), this would substantially strengthen the case for capacity as a non-redundant measure.

- **Demonstrate at least one functional consequence of geometric subtypes.** The simplest test: in the RNN experiment, evaluate whether networks with different geometric organizations (same capacity, different radius/dimension tradeoffs) differ in noise robustness, memory retention, or transfer to new tasks. Even one such result would transform the "beyond the dichotomy" claim from descriptive to predictive.

- **Provide formal criteria for learning stages.** Define quantitative thresholds or change-point detection for stage boundaries, and validate their consistency across architectures and datasets.

## Evaluation

**Originality:** The application of manifold capacity theory to feature learning is novel and well-motivated. The "wealthy vs. poor" initialization distinction and the discovery of geometric subtypes are original contributions. However, the individual components (manifold capacity theory, lazy/rich regime) are not new.

**Importance of research question:** Understanding and quantifying feature learning is important for both ML and neuroscience. The representation-based approach addresses a genuine gap for neuroscience applications.

**Claims support:** The core claim that capacity quantifies feature learning is well-supported empirically but limited theoretically (one gradient step). The "beyond the dichotomy" claim is the least well-supported—descriptive but without demonstrated functional consequences.

**Experimental soundness:** Experiments are well-designed and cover multiple settings. The main gap is the missing comparison with simpler representation-based baselines.

**Clarity:** The paper is well-written with clear motivation and structure. The geometric measures are technically dense but well-explained.

**Community value:** The framework provides a useful analytical lens for both ML and neuroscience communities. The representation-based approach is particularly valuable for neuroscience.

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Decision | Comparison |
|-------|-----------|----------|------------|
| CRH / Canonical Representation Hypothesis (Njx1NjHIx4) | 7.50 | Accept (Spotlight) | Stronger: unifies multiple phenomena with theoretical and empirical support; the paper under review is less cohesive |
| Task structure and nonlinearity (k9t8dQ30kU) | 6.75 | Accept (Poster) | Similar: empirical study of representational geometry; more focused but deeper; the paper under review has broader scope with some theory |
| Neural Manifold Regularization (TVnkjz4MqV) | 5.50 | Reject | Weaker in theory but stronger in practical applicability; the paper under review has more novel theoretical grounding |
| Loss flatness and compressed representations (CtiFwPRMZX) | 5.00 | Reject | Weaker: similar type of contribution (connecting two perspectives) but with weaker theory and limited experiments |
| Manifold topology / expressive power (FE7PY7e4tr) | 5.25 | Reject | Similar topic but the paper under review has broader empirical scope and more settings |
| Ricci flow geometry (xA25Ib7H8U) | 2.33 | Reject | Much weaker: unclear theory with no experiments |

The paper under review sits between the 5.0–5.5 reject anchors and the 6.75 accept anchor. It has genuine contributions (novel framework, empirical findings about capacity vs. conventional measures, RNN structural biases, learning stages) but also real gaps (limited theory, no functional validation of subtypes, missing simpler baselines). It is stronger than the typical reject (5.0–5.5) but not as strong as the borderline accept (6.75). The main issue is that the central "beyond the dichotomy" claim outpaces the evidence—geometric subtypes are demonstrated descriptively but not shown to matter functionally.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>