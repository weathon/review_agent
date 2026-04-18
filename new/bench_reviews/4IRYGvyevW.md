Now I have enough calibration information. Let me synthesize my final review.

## Summary

This paper proposes using manifold capacity theory—a representation-based measure of the "packability" of task-relevant neural manifolds—to quantify the degree of feature learning (rich vs. lazy) in neural networks and to go beyond the lazy-versus-rich dichotomy by decomposing capacity changes into geometric measures (radius, dimension, alignment) that reveal distinct learning strategies and stages. The authors provide a theoretical result (Theorem 1) showing that manifold capacity monotonically tracks the degree of richness and prediction accuracy in a 2-layer teacher-student model after one gradient step, and then empirically demonstrate that capacity and its geometric decomposition reveal learning strategies, learning stages, structural inductive biases in RNNs, and geometric correlates of OOD generalization failure.

## Strengths

- **Principled representation-based framework.** The paper correctly identifies that weight-based or NTK-based measures of lazy/rich regimes are limited from a neuroscience perspective (where synaptic changes cannot be directly observed) and proposes manifold capacity as a representation-level alternative. This is a genuine contribution: the framework provides a quantitative measure with an analytical connection to interpretable geometric descriptors (radius, dimension, alignment), giving a vocabulary for describing *how* representations change, not just *whether* they do.

- **Theoretical grounding, though limited.** Theorem 1 establishes a formal monotone relationship between capacity, learning rate, and prediction accuracy in a concrete 2-layer teacher-student setting. Extending Ba et al. (2022) from regression to classification and connecting capacity to margin/accuracy is a non-trivial theoretical contribution, even if the scope is narrow.

- **Empirical demonstrations show genuine advantages over conventional measures.** Figure 3 shows that capacity correctly orders different degrees of feature learning (via the inverse scale factor knob) when weight changes and representation-label alignment give incorrect or ambiguous orderings. This is a concrete, empirical advantage over standard baselines.

- **The geometric decomposition yields novel qualitative observations.** The discovery that networks trading off radius vs. dimension compression in different regimes (Fig. 4a,b), that VGG-11 training exhibits four identifiable geometric stages (Fig. 4c), and that RNNs with different initial weight ranks reach the same final capacity via different geometric paths (Fig. 5d) are interesting observations that would not be accessible through scalar measures alone.

- **The OOD generalization finding is provocative.** The observation that ultra-rich training on CIFAR-10 degrades CIFAR-100 linear-probe performance, and that this correlates with radius expansion and center-axis alignment increase (Fig. 6c), identifies a concrete geometric signature of overfitting that goes beyond standard narratives.

## Weaknesses

### Fatal
None.

### Major

- **The central interpretive claim—equating manifold capacity with "task-relevant feature learning"—extends well beyond what the theory guarantees.** Theorem 1 holds only for a 2-layer network, one gradient step, squared loss, and a teacher-student model where the *only* discriminative direction is β*. In this setting, higher capacity is guaranteed to reflect more task-relevant information. But manifold capacity (Definition 1) measures the average separability of manifolds under random dichotomies and random projections—it is agnostic to the *task* labels. For real data with nuisance directions, spurious features, or complex label structure, higher capacity could reflect features that increase average separability but are misaligned with the true task. The paper acknowledges this disconnect nowhere. This does not invalidate the method, but it means the repeated equation of "capacity = task-relevant feature learning" (abstract, Sec. 1.1, Sec. 3 running text) is an overclaim. The empirical evidence shows capacity correlates with the inverse scale factor knob and with accuracy, but these are indirect validations of "task relevance"—they do not rule out that capacity is capturing generic separability rather than task alignment.

- **The learning "strategies" and "stages" (Sec. 4) are descriptive rather than established phenomena.** Figure 4a,b shows training trajectories on (R_mf, D_mf) contour plots, and the authors interpret segments as distinct strategies (e.g., "sacrificing radius for further dimension compression"). Figure 4c names four stages for VGG-11 on CIFAR-10. But: (i) there are no error bars or variability analysis across seeds, (ii) no principled segmentation criterion for the stages, (iii) no validation across architectures or datasets, and (iv) no demonstration that these stages correspond to distinct functional regimes (e.g., generalization vs. memorization). The capacity approximation formula α ≈ (1 + R^{-2})/D inherently builds in a radius–dimension tradeoff, so "discovering" that networks navigate this tradeoff is somewhat built into the framework. The stages and strategies are plausible hypotheses but not yet demonstrated as reproducible or meaningful phenomena.

- **The theoretical result is limited to one gradient step, yet the paper's claims cover full training.** The authors note in footnote 6 that the Gaussian equivalence may not hold beyond one step. However, the paper's framing ("we theoretically and empirically show that manifold capacity tracks the degree of feature learning in a wide range of settings") implies broader theoretical coverage than what exists. The theory guarantees only that capacity tracks the *initial tendency* toward feature learning. The gap between one-step theory and multi-epoch empirical demonstrations is substantial and should be more explicitly acknowledged as a limitation rather than treated as implicitly resolved.

### Minor

- **The OOD section (Sec. 5.2) is correlational and does not rule out simpler explanations.** The finding that radius expansion and center-axis alignment increase coincide with OOD degradation is interesting but observational. Standard overfitting to CIFAR-10 class structure could explain the same phenomenon without invoking manifold geometry. No intervention experiments (e.g., regularizing radius) are provided, and no comparison to simpler representation-level metrics (linear probe accuracy, CKA, spectral properties) is made. The geometric diagnosis is offered without a prescription.

- **Comparison with baselines in Sec. 3.2 is qualitative.** The paper claims capacity is "better" than weight changes or alignment measures at distinguishing learning regimes, but this is shown only through visual inspection of curves in Figure 3. No quantitative metric (e.g., rank-ordering accuracy, correlation with ground-truth richness) or statistical test is provided. The claim of superiority over conventional measures is therefore not rigorously established.

- **The "wealthy vs. poor" initialization framing risks circularity.** It is defined by capacity at initialization, which is itself the proposed measure. The interesting finding—that RNNs with different initial ranks reach the same final capacity via different geometric paths—does not require this terminology and would be clearer without it.

### Trivial
- The notation for some geometric measures (e.g., the radius definition involving ratios of projections) is dense and could benefit from more intuitive explanation in the main text.

## Nice-to-Haves

- Interventions on geometric properties (e.g., regularizing radius or dimension during training) to test whether the identified "learning strategies" are causally meaningful rather than merely correlational.
- Testing the framework on more modern architectures (transformers) or larger datasets.
- Validating the "learning stages" across architectures, optimizers, and random seeds with formal segmentation criteria.
- Comparison to other representation-level baselines (linear probe accuracy, SVCCA) to better establish capacity's unique value.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic's point about "capacity is agnostic to label function" implying it may capture non-task-relevant features.** While this is a valid concern in principle, I have incorporated a moderated version of this as a major weakness above. The harsh critic's version overstates the problem: the paper does show empirical correlations between capacity and task accuracy, and capacity is computed on *class* manifolds, so it is not entirely label-agnostic. The concern is real but not fatal—the claim "capacity = task-relevant feature learning" is an overclaim, but capacity is a reasonable *proxy* for it in the settings studied.

- **Harsh Critic's point about comparisons being qualitative only.** I have included this as a minor weakness. The comparison is qualitative, but the qualitative evidence is clear and consistent across settings, showing capacity correctly orders regimes where other measures fail. This is sufficient for an initial demonstration, even if a quantitative comparison would strengthen the paper.

- **Demands for experiments on transformers/larger datasets.** The paper scopes its experiments to models and datasets that are "large enough to see interesting phenomena, while the computational cost is still reasonable" (Sec. 2.3). Demanding larger-scale experiments goes beyond the paper's stated scope and is a nice-to-have rather than a weakness.

- **Demands for real neural data (electrophysiology/fMRI).** The paper is in the "applications to neuroscience" primary area and uses RNNs as a neuroscience model, which is standard practice. Demanding real data application is scope creep.

- **Spark's suggestion about "experiments on naturally trained networks without the scale factor manipulation."** The paper *does* include experiments on standard training without scale factor manipulation (VGG-11 on CIFAR-10 in Fig. 4c, RNNs in Sec. 5.1). This point misunderstands the paper's experimental design.

- **Harsh Critic's concern about "the actionable implications of geometric insights are underspecified."** The paper explicitly identifies this as future work in the conclusion. This is a fair critique but better framed as a nice-to-have than a major weakness, since the paper positions itself as providing a framework for understanding, not yet for intervention.

- **Spark's point about "direct visualizations of manifold structure (t-SNE/UMAP)."** This is a presentational suggestion, not a substantive weakness.

- **Spark's concern about convex hull assumptions.** The paper's methodological section (Sec. 2.1) discusses the convex hull model, which is standard in manifold capacity theory. While the approximation could break down for highly nonlinear manifolds, this is inherited from the framework (Chung et al., 2018; Chou et al., 2024) and not a novel concern for this paper.

- **Demand for formal/automated criteria for learning stages.** This is a methodological preference; visual identification of phases is common in the deep learning dynamics literature. Formal criteria would strengthen but not invalidate the observation.

- **Neutral Reviewer's concern about "wealthy vs. poor" being "circular."** I have included a moderated version as a minor weakness. The terminology is used as a shorthand for different initialization regimes, and the empirical findings (same final capacity, different geometric paths) stand independently of this label.

## Novel Insights

The paper's key novel insight—that representational geometry provides a vocabulary for distinguishing *subtypes* of rich learning (radius compression vs. dimension compression, different geometric paths to the same final capacity)—goes meaningfully beyond the lazy/rich dichotomy. The finding that RNNs with different initial weight ranks converge to the same capacity but via different geometric organizations is particularly interesting and suggests that "how much" feature learning occurs and "how" it occurs are orthogonal questions. However, these insights remain largely qualitative and their robustness and generality are not yet established.

## Suggestions

- **Moderate the central claim.** Replace "capacity quantifies the degree of task-relevant feature learning" with "capacity correlates with—and provides a representation-based proxy for—the degree of feature learning," and explicitly acknowledge that capacity measures average separability rather than task-aligned separability.

- **Acknowledge the one-step theoretical limitation more prominently.** State in the main text (not just in a footnote) that Theorem 1 applies only to a single gradient step in a specific model, and that the multi-epoch empirical findings are supported by experiments rather than theory.

- **Add variability analysis for the learning stages and strategies.** Show whether the four stages in Fig. 4c reproduce across random seeds and architectures, at minimum.

- **Provide quantitative comparisons with baselines.** Include a simple metric (e.g., rank-ordering accuracy or correlation with known richness parameter) to substantiate the claim that capacity outperforms weight changes and alignment measures.

## Score and Decision

**Calibration anchors:**

- **slSmYGc8ee** (How connectivity structure shapes rich and lazy learning): Accept poster, scores 8/6/8/5. This paper directly investigates lazy/rich regimes in RNNs using weight/kernel changes (the very measures the current paper argues are limited). The current paper extends this with a representation-based alternative, but with less causal depth in the neuroscience application.

- **vt5mnLVIVo** (Grokking as lazy-to-rich transition): Accept poster, scores 8/8/3/5. This paper has a more restrictive theoretical model but clearer causal mechanism. Reviewers noted limited theoretical results and simple experiments as weaknesses, yet it was accepted.

- **iEfdvDTcZg** (Optimization landscape of SGD across feature learning strength): Accept poster, scores 6/6/5/8. Similar in studying lazy/rich regimes with scale parameters. Slightly narrower scope but more systematic empirical investigation.

- **KJFyOwAnLR** (Emergent Geometry): Withdrawn/Reject, scores 3/6/3. Overclaimed geometric findings with insufficient experimental controls. This is clearly weaker than the current paper.

The current paper has genuine methodological novelty, a sound theoretical starting point, and interesting empirical observations. However, it overclaims on two fronts: (1) equating capacity with "task-relevant" feature learning beyond what the theory supports, and (2) treating qualitative observations of learning stages/strategies as established phenomena. These are significant but not fatal—the core framework is valuable even with more modest claims. Compared to accepted papers in this space (slSmYGc8ee, vt5mnLVIVo), the current paper offers a broader methodological toolkit but with less depth in any single application. The gap between theory and empirical claims is comparable to that in vt5mnLVIVo (which also had a one-step result extended to multi-epoch experiments).

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>