## Summary

This paper introduces a framework based on manifold capacity theory to quantify and characterize feature learning in neural networks. The central idea is that task-relevant manifold untangling—measured by manifold capacity—serves as a representation-based signature of rich learning, and that associated geometric measures (radius, dimension, alignment) can reveal subtypes of feature learning beyond the lazy-versus-rich dichotomy. The authors provide a theoretical result connecting capacity to learning rate (richness) and prediction accuracy in a 2-layer one-step GD setting, demonstrate empirical advantages over conventional measures, and apply the framework to uncover geometric insights in RNNs trained on neuroscience tasks and OOD generalization in CNNs.

## Strengths

- **Novel representation-based framework with genuine utility for neuroscience**: Shifting the analysis from weight/NTK-based measures to manifold geometry of neural representations directly addresses a real gap—the neuroscience community cannot easily access synaptic weight changes. The framework is principled (built on existing MCT) and provides interpretable geometric descriptors rather than a single scalar. This is a meaningful contribution.

- **Superiority over conventional measures demonstrated**: Figure 3b shows a concrete case where representation-label alignment gives the *wrong* ordering of feature wealth at initialization while capacity correctly recovers it. This is a genuine and important advantage over simpler measures.

- **Geometric decomposition provides genuinely new granularity**: The finding that networks with same final capacity (Figure 5d) can have different geometric organizations (different radius/dimension trade-offs) is a compelling demonstration that the lazy-versus-rich dichotomy misses meaningful structure. This is a real insight enabled by the geometric decomposition.

- **Theoretical grounding, even if limited**: Theorem 1 provides formal justification that capacity monotonically tracks richness and is linked to prediction accuracy via an invertible function, establishing more than purely empirical correlations.

- **Interesting empirical phenomena**: The ultra-rich OOD degradation finding (Section 5.2)—where linear probe accuracy on CIFAR-100 drops while in-distribution performance continues improving—is a novel observation, and the geometric correlates (radius expansion, center-axis alignment) suggest mechanistic hypotheses worth pursuing.

## Weaknesses

### Major

- **Gap between theoretical guarantees and empirical scope**: Theorem 1 holds only for one gradient step in a 2-layer network with squared loss, fixed readout, and random Gaussian data in a teacher-student setting. The paper acknowledges this (footnote 6) but then extrapolates to multi-epoch deep networks with cross-entropy loss on natural images. There is no analysis of why or whether the monotone connection between capacity and richness extends beyond one step or beyond the proportional asymptotic regime. The core claim—*"capacity quantifies the degree of feature learning"*—is established rigorously only in this very restricted setting. The title and framing imply broader validity than is proven. This matters because all downstream interpretations (learning stages, strategies, OOD explanations) rely on capacity faithfully tracking meaningful feature learning.

- **"Learning stages" are identified qualitatively, not rigorously**: The four stages in Figure 4c (clustering, structuring, separating, stabilizing) are identified by visual inspection of normalized heatmap plots. There is no formal change-point detection, no statistical test, no operational criterion that would allow another researcher to independently identify these stages. The same concern applies to the "learning strategies" in Figures 4a,b (radius vs. dimension trade-offs), which are single-run trajectories without variance estimates or robustness checks across initializations. Given that this "beyond the dichotomy" finding is a headline contribution (explicitly stated in the title), the evidence should meet a higher bar.

- **Conceptual scope of what "task-relevant features" means**: Capacity measures average-case linear separability of class-conditional convex hulls. The paper repeatedly frames this as measuring "the amount of task-relevant features" (Abstract: *"capacity quantifies the degree of richness"*). However, linear separability of training-class manifolds can increase for reasons unrelated to robust task relevance—e.g., overfitting to spurious training-set features. The paper's own OOD experiment (Section 5.2) reveals exactly this tension: CIFAR-10 capacity rises monotonically with inverse scale factor while CIFAR-100 linear-probe accuracy drops in the ultra-rich regime. This means capacity can increase while *useful* feature learning for transfer degrades, which complicates the unqualified identification of capacity with task-relevant feature quality. The paper does not reconcile or even explicitly discuss this tension.

### Minor

- **Limited architecture and task diversity**: Empirical demonstrations use VGG-11, ResNet-18, and small 2-layer MLPs on synthetic data, CIFAR-10/100, and CIFAR-10C. No experiments with transformers, larger-scale datasets, or more complex tasks (e.g., language, sequential decision-making). The RNN experiments use only 300-unit single-layer ReLU RNNs on simple NeuroGym tasks. As noted in reviews of similar papers studying representational geometry ([k9t8dQ30kU, Reviewer 2]; [KJFyOwAnLR, Reviewer 1]), claims about universal "learning stages" based on limited architecture diversity are premature.

- **No comparison to simpler class-separation measures**: The paper compares capacity against weight changes, NTK-label alignment, and representation-label alignment, but does not compare against simpler representation-based class-separation metrics (e.g., linear discriminant analysis ratios, Fisher discriminant, inter-class distance metrics). It is unclear whether the added complexity of the MCT framework (quadratic programs, anchor points, convex hulls) yields substantially more insight than simpler alternatives, especially for the stage/strategy identification claims.

- **RNN structural inductive bias results lack functional validation**: Section 5.1 shows that RNNs with different initial weight ranks converge to similar final capacity but different geometric organizations. The paper calls these "structural inductive biases," but no functional consequences are demonstrated—these networks already achieve similar accuracy. Whether the geometric differences matter for robustness, learning speed on new tasks, or other functional properties is not tested. As noted in a closely related review of RNN rich/lazy work ([slSmYGc8ee.md, Reviewer 2]): *"the richness/laziness of learning dynamics by itself tells us very little about the representations a network learns"* without connecting to functional outcomes.

- **Convex hull assumption not validated**: Manifold capacity models each class as a convex hull, which is a strong geometric assumption. No empirical comparison is provided between geometric measures computed on convex hulls versus those computed directly on point clouds, leaving it unclear how sensitive the results are to this assumption.

### Trivial

- The inverse scale factor from Chizat et al. (2019) is used as ground truth for richness, which is a specific construction that may not capture all dimensions of feature learning. The paper is transparent about this, so it is not a hidden issue, but it is worth noting for completeness.

## Nice-to-Haves

- Formal change-point detection or statistical criteria for identifying "learning stages," which would make the stages reproducible across settings.
- Intervention experiments regularizing specific geometric measures (e.g., penalizing radius growth) to test whether geometric changes causally affect OOD generalization.
- Application to real neural recordings to validate the neuroscience relevance claimed in the introduction.
- Experiments on intermediate layers rather than only last-layer representations.
- Comparison to simpler linear separation metrics to establish the added value of the full MCT framework.
- Computational cost analysis for estimating manifold capacity and geometric measures at scale.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"No comparison to neural recording data"** (Harsh Critic): The paper explicitly scopes its neuroscience contribution to RNNs and states that applying to real neural data is future work (Section 6). Demanding real neural data is scope creep beyond the paper's stated contributions.

- **"Computational cost is not discussed"** (Neutral Reviewer, Spark): While true, this is a practical implementation detail. The paper provides Algorithm 1 and references Chou et al. (2024) for computational details. This is a nice-to-have, not a weakness.

- **"Theoretical result only covers one step"** as a *fatal* flaw: This is a real limitation (kept above as Major), but it is not fatal—the paper provides extensive empirical validation beyond one step and is transparent about the theorem's scope. The one-step result still provides an exact analytical anchor.

- **"No transformer experiments"** (Spark): This is a generic request for broader evaluation. The paper tests on multiple architectures (MLP, CNN, RNN). Demanding transformers is a nice-to-have, not a core flaw.

- **"OOD findings are only correlational, not causal"** as a *fatal* flaw (Harsh Critic): This is a valid concern but is overstated. The paper uses cautious language ("correlates," "geometric insights," "future direction") and does not make causal claims. The finding is presented as observational, which is appropriate for a first pass.

- **Demands for statistical tests, seeds, variance bars** (Harsh Critic, Spark): While more rigor would help, single-run visualization is standard practice in the representational geometry community (see Chung et al., 2018; Stephan et al., 2023). Requesting extensive statistical validation is disproportionate given the paper's exploratory nature.

## Novel Insights

The most interesting insight in this paper is the decomposition of feature learning into *geometric sub-strategies*—specifically, that networks can achieve similar capacity through different radius/dimension trade-offs (Figure 5d). This moves beyond the lazy-rich dichotomy not merely by adding shades of gray but by opening up a two-dimensional (or multi-dimensional) geometric landscape of feature learning paths. The OOD finding that ultra-rich learning degrades transfer performance while in-distribution manifold capacity continues to increase is also a genuinely novel and potentially important observation—it suggests that *too much* feature specialization (in the form of increased manifold packability for training classes) can be harmful, providing a geometric lens on the classic bias-variance trade-off in representation learning.

## Suggestions

- **Soften the core claim**: Replace "capacity quantifies the degree of feature learning" with "capacity provides a representation-based measure correlated with feature learning in specific settings, with known limitations." Acknowledge the OOD tension explicitly.
- **Add formal stage identification**: Use a simple change-point detection method on geometric measure trajectories to define stages objectively, and test whether these stages replicate across random seeds and architectures.
- **Connect geometry to function in RNNs**: Even a simple experiment testing whether radius-dominant vs. dimension-dominant representations differ in noise robustness or few-shot transfer would transform the "structural inductive bias" claim from descriptive to substantive.
- **Add a failure case analysis**: Show at least one setting where capacity does not track meaningful feature learning, or where it gives misleading results, to calibrate the reader's expectations.

## Score and Decision

**Calibration**: I compared against papers with similar profiles—studies of representational geometry and lazy/rich learning in neural networks. *How connectivity structure shapes rich and lazy learning in neural circuits* ([slSmYGc8ee]) received scores of 8, 6, 8, 5 (average ~6.75) and was accepted as a poster. *Task structure and nonlinearity jointly determine learned representational geometry* ([k9t8dQ30kU]) received scores of 5, 6, 8, 8 (average ~6.75) and was accepted as a poster. *Grokking as the transition from lazy to rich training dynamics* ([vt5mnLVIVo]) received scores of 8, 8, 3, 5 (average ~6) and was accepted. *Stagewise Development in Transformers* ([xEZiEhjTeq]), which had similar issues with qualitative stage identification and limited validation, received scores of 6, 5, 5, 6 (average 5.5) and was rejected.

This paper is comparable to slSmYGc8ee and k9t8dQ30kU in that it introduces a useful framework with genuine insights but with somewhat limited validation. It is somewhat stronger than xEZiEhjTeq because it has a theoretical result (even if restricted) and more systematic benchmarking. However, it overclaims somewhat in its title and framing. I place it at **6**, consistent with the accepted-but-not-strong borderline of similar papers.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>