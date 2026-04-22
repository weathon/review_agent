Now let me do calibration searches to properly anchor the score.Now I have enough information to write the final consolidated review.

---

## Summary

This paper proposes manifold capacity—a representation-based measure from manifold capacity theory (MCT)—as a framework for studying feature learning in neural networks and neuroscience. The core claim is that task-relevant manifolds untangle during rich (feature-learning) training, and that capacity monotonically tracks the degree of richness. The paper provides a theoretical result (Theorem 1, restricted to one-step gradient descent in a two-layer Gaussian teacher-student setting), empirical validation across synthetic and real settings, and applications to RNN structural inductive biases and out-of-distribution (OOD) generalization. A secondary contribution is the use of geometric decompositions (radius, dimension, alignment) to reveal subtypes of feature learning beyond the binary lazy/rich dichotomy.

---

## Strengths

- **Theorem 1 is a technically non-trivial theoretical contribution** (Section 3.1): it extends Ba et al. (2022)'s Gaussian equivalence framework from regression to classification, deriving both monotone tracking of capacity with richness parameter η and an invertible function linking capacity to prediction accuracy. This required new tools (Montanari et al., 2019, Proposition 2) and is acknowledged as a meaningful technical extension.

- **Novel RNN structural inductive bias finding** (Section 5.1, Figure 5c/d): RNNs with different initial weight ranks converge to similar capacity at final epoch but via geometrically distinct manifold organizations (small rank → larger radius, smaller dimension; large rank → smaller radius, larger dimension). This is specific, non-obvious, and not visible from weight-matrix-based analysis alone.

- **Geometric decomposition beyond binary lazy/rich** (Section 4.1, Figures 4a–b): The distinct learning strategies (radius-dominant vs. dimension-dominant compression as richness varies) go meaningfully beyond the binary dichotomy and are illustrated compactly in the capacity contour plots.

- **Representation-based framework with neuroscience applicability** (Introduction, para. 3): Because the method operates on neural activity patterns rather than weight matrices or NTKs, it is genuinely accessible in neuroscience settings where synaptic weights cannot be tracked—a real gap addressed by the paper.

- **Consistent experimental protocol using scale factor η̄** (Section 2.3): The reuse of Chizat et al. (2019)'s inverse scale factor across all experiments enables controlled, reproducible comparisons between lazy and rich regimes.

---

## Weaknesses

### Fatal
None.

### Major

- **The superiority claim for capacity over conventional measures is validated only in synthetic 2-layer settings** (Section 3.2, Figure 3). The abstract and introduction state that "capacity is better than conventional measures in quantifying the degree of feature learning," but the quantitative comparison (against weight changes, NTK-label alignment, representation-label alignment) is performed exclusively on 2-layer networks trained on Gaussian clouds. It is not demonstrated in VGG-11, ResNet-18, or RNNs—precisely the settings where the broader readership cares most about the advantage. The Figure 3b result showing representation-label alignment incorrectly ordering wealthiness is a concrete demonstration of advantage, but only in a narrow, synthetic context the authors designed. Demonstrating the same advantage in DNNs would substantially strengthen this central claim.

- **The theoretical-empirical regime gap is real, though acknowledged** (Theorem 1, footnote 6): The theory (one-step gradient descent, two-layer network, Gaussian teacher-student, proportional asymptotic) and the empirics (VGG-11/ResNet-18 multi-epoch training, RNNs over 10,000 iterations) inhabit disjoint regimes. The paper does acknowledge this in footnote 6 ("the key Gaussian equivalence step might not hold for more steps"), which is appropriate transparency—but Theorem 1 is nonetheless presented as providing "theoretical justification" for the general manifold-untangling claim, which it can only partially do. The paper would benefit from a clearer separation of what the theorem establishes versus what is supported only empirically.

### Minor

- **The four learning stages** (Section 4.2, Figure 4c) are identified for a single run of VGG-11 on CIFAR-10, presented with four named stages (Clustering, Structuring, Separating, Stabilizing). The paper uses "an example" phrasing, which partially mitigates the concern, but no validation on ResNet-18, other datasets, or multiple seeds is provided. Without replication across architectures, these stages cannot be treated as general properties of feature learning—only as a characterization of one run.

- **Causal language in the OOD section** (Section 5.2, Figure 6 caption): The paper states that "the expansion of manifold radius and the increase of center-axis alignment *explain* the failure of OOD generalization." This is correlational—no intervention is performed, and no causal direction is established. The paper does acknowledge future work is needed to translate these insights into practical improvement, but "explain" should be replaced with observational language.

- **The VGG-11 vs. ResNet-18 OOD discrepancy is mentioned but not analyzed** (Section 5.2): "it is the increment in dimension in the ultra-rich regime explaining the drop of capacity in ResNet-18 (Figure 21)"—a different geometric mechanism from VGG-11 (radius expansion). Two architectures showing different geometric correlates of the same OOD failure phenomenon weakens the generalizability of the geometric diagnosis and deserves more than a single-sentence mention.

### Trivial

None identified beyond formatting artifacts (parser issues, not paper errors).

---

## Nice-to-Haves

- Replicate the learning stage analysis (Figure 4c) on ResNet-18 or at least a second architecture/dataset to test whether the four-stage structure is architecture-universal or VGG-specific.
- Validate the superiority of capacity over baselines (Section 3.2 comparison) in DNN settings (VGG-11/ResNet-18 on CIFAR-10), even as a supplementary figure.
- An intervention experiment connecting manifold geometry to OOD performance (e.g., regularizing manifold radius during training) would transform the OOD finding from a correlation into a mechanistic claim.
- Extending the RNN analysis to more than one neuroscience task (beyond perceptual decision making) would strengthen the generality of the structural inductive bias finding.

---

## Removed Points

These points are flagged to be removed, treat them with caution.

1. **Harsh critic, Section 2.1 — convex hull approximation concern**: The reviewer suggests the convex hull assumption is a "meaningful modeling choice, not a neutral one" with unstated consequences for real data. However, footnote 5 explicitly states: "In the context of linear classification, it is mathematically equivalent to study the convex hull of a manifold." This is a standard and well-justified choice in the MCT literature (Chung et al., 2018; Chou et al., 2024). The concern is not valid as a criticism of this paper.

2. **Harsh critic, "Paper does not acknowledge" the one-step gap**: The critic states "the paper does not acknowledge this gap." This is factually incorrect—footnote 6 explicitly reads: "Here we follow the convention in (Ba et al., 2022) and study only the first gradient step as the key Gaussian equivalence step might not hold for more steps as remarked in footnote 2 of (Ba et al., 2022)." The gap exists but is transparently acknowledged.

3. **Harsh critic, Section 2.3 — scale factor conflation**: The reviewer argues that η̄ conflates scaling of outputs with relative learning rates of intermediate vs. readout layers. The paper directly says (Section 2.3): "a larger η̄ indicates that the learning rate of intermediate layers is faster compared to that of the readout weights," and the appendix details this further. This is not a conflation—it is the paper's intended operationalization following Chizat et al. (2019).

4. **Strength Finder, "scale factor η̄ reused consistently — reproducibility strength"**: This is a procedural point, not a genuine scientific strength. Removed as generic.

---

## Novel Insights

The most genuinely novel observation in this synthesis is the geometry-capacity dissociation in the RNN experiment: networks with different structural priors (initial weight rank) can achieve nearly identical final capacity—and thus equivalent task separability—while reaching qualitatively different geometric organizations of their learned manifolds (opposing radius/dimension trade-offs). This suggests that capacity alone is insufficient to characterize the nature of learned representations in biological or artificial neural circuits, and that geometric decomposition is necessary to distinguish learning strategies that are otherwise capacity-equivalent. This finding is specific, non-obvious, and has direct implications for how structural inductive biases in neural circuits should be studied.

---

## Suggestions

1. **Move the capacity-vs.-baselines comparison into DNN settings** as a supplementary experiment. Even a single figure showing that capacity correctly orders richness (by η̄) better than CKA-based measures in VGG-11 would substantially back the superiority claim made in the abstract.
2. **Soften causal language in Section 5.2** ("explain → correlate with / are geometric signatures of") and frame the OOD finding explicitly as hypothesis-generating for future intervention-based work.
3. **Show stage analysis for at least one additional architecture** (e.g., ResNet-18 on CIFAR-10) in the supplementary material to test generality of the four-stage structure.
4. **Clarify the scope of Theorem 1** in the abstract and introduction by explicitly noting it is established for one gradient step in a two-layer model, while the empirical claims span multi-epoch training in deep networks.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Human Score | Comparison to this paper |
|---|---|---|---|
| On the Joint Interaction of Models, Data, and Features | ze7DOLi394.md | 7.50 (oral) | More conceptually original; interaction tensor framework more broadly applicable; cleaner theoretical-empirical integration than this paper |
| Training dynamics + Hessian alignment | MHjigVnI04.md | 7.67 (spotlight) | Stronger theoretical grounding and more rigorous experimental design |
| Task structure and nonlinearity (1-layer geometry) | k9t8dQ30kU.md | 6.75 (poster) | Similar scope (1-hidden-layer representational geometry); purely empirical; this paper has additional theorem and broader application but more overclaiming |
| Loss flatness to compressed representations | CtiFwPRMZX.md | 5.00 (reject) | Also connects manifold geometry to network behavior; rejected for limited empirical validation and ill-posed theory — similar validation gap but this paper's theory is more carefully scoped |
| Neural Manifold Regularization | TVnkjz4MqV.md | 5.50 (reject) | Also applies manifold geometry to neural dynamics; lacked clear advantage demonstration |
| Brain-inspired geometry constraint | RIaIpdUCPb.md | 3.00 (reject) | Insufficient justification of claims; weaker than this paper overall |

**Calibration reasoning**: This paper is clearly above the low-scoring cluster (3.0 range), which suffered from fundamental validity issues or near-zero novelty. It is competitive with k9t8dQ30kU.md (6.75, poster accept) in scope and better in theoretical content, but falls below the oral-level papers (7.5+) due to overclaiming and limited validation of key claims. The superiority-of-capacity claim — a central advertised contribution — resting only on 2-layer synthetic data is the primary reason this paper falls short of clean poster acceptance. The learning stages result (single run) is a secondary concern. These together position the paper below k9t8dQ30kU.md's 6.75 despite the additional theoretical content.

**Final assessment**: The paper makes real contributions — a non-trivial theorem extending Ba et al. (2022) to classification, a genuinely novel RNN geometry-capacity dissociation finding, and a useful representational geometry framework with neuroscience applicability. However, its central empirical superiority claim is inadequately validated beyond synthetic settings, its learning stage "discovery" rests on a single run, and its OOD findings use causal language for correlational results. These are fixable in a revision but are substantive enough that the current submission oversells what is demonstrated.

**Score: 5.5 | Decision: Reject**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>