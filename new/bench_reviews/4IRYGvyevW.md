Now let me read the relevant human review files for calibration.Now let me write the final consolidated review.

---

## Summary

This paper proposes using manifold capacity theory (MCT) as a representation-based framework to quantify and characterize feature learning in neural networks. The central insight is that task-relevant neural manifolds become more separable ("untangled") during rich learning, and manifold capacity—along with its associated geometric descriptors (radius, dimension, center alignment, axis alignment, center-axis alignment)—can track and decompose these changes. The paper offers three contributions: (1) theoretical and empirical justification that capacity quantifies the degree of richness; (2) identification of distinct learning "stages" and "strategies" through geometric trajectories; and (3) applications to structural inductive biases in RNNs and geometric correlates of OOD generalization failure.

---

## Claims and Support

**Claim 1: Manifold capacity quantifies the degree of feature learning / richness; task-relevant manifolds untangle during rich learning.**
- *Theoretical support*: Theorem 1 establishes that capacity monotonically tracks learning rate η and links to prediction accuracy in a 2-layer teacher-student model after one gradient step. The paper explicitly acknowledges in footnote 6 that this is limited to the first step. The theorem is mathematically sound but narrow.
- *Empirical support*: Figures 2b and 3a show capacity tracks the scale interpolation parameter $\bar\eta$ in both 2-layer synthetic and DNN settings, providing genuine empirical corroboration.
- **Verdict**: Partially supported. The theoretical backing is much narrower than the headline claim, but empirical evidence is consistent and well-presented.

**Claim 2: Manifold capacity is better than conventional measures for quantifying feature learning.**
- *Support*: Figure 3a (synthetic 2-layer setting) shows capacity separates scale parameters better; Figure 3b shows capacity gives the correct ordering at initialization while representation-label alignment gives the wrong ordering.
- **Verdict**: Partially supported but overclaimed. Evidence is restricted to narrow synthetic experiments; no systematic quantitative comparison (e.g., rank correlation with $\bar\eta$) is provided, and superiority is not demonstrated in the DNN settings where the framework is deployed.

**Claim 3: Manifold geometry reveals distinct learning stages and strategies.**
- *Support*: Figure 4a,b shows different geometric trajectories in 2-layer synthetic networks. Figure 4c identifies four stages in VGG-11 on CIFAR-10 from a single normalized trajectory.
- **Verdict**: Partially supported for "strategies" (geometric trade-offs are clearly visible). "Stages" are descriptive overlays from a single run, lacking formal criteria or cross-architecture replication.

**Claim 4: Different RNN connectivity structures produce different manifold geometry even at similar final capacity / feature-learning level.**
- *Support*: Figure 5 shows different initial weight ranks yield different final geometric measures while reaching similar final capacity. This is descriptively clear and consistent.
- **Verdict**: Supported for the descriptive claim. The framing as "structural inductive biases in neural circuits" is a minor overstatement (results are on artificial RNNs), but the paper explicitly positions this as a representation-based analog of prior weight-based analysis (Liu et al., 2024), which is fair.

**Claim 5: Geometric correlates explain OOD generalization failure in the ultra-rich regime.**
- *Support*: Figure 6c shows correlations between richness, radius expansion, center-axis alignment increase, and CIFAR-100 probe accuracy drop. The paper itself states this is correlational ("We leave it as a future direction").
- **Verdict**: Partially supported as a correlation/observation. "Explain" is too strong—the evidence is correlational and no isolation of geometric quantities is performed.

---

## Strengths

- **Novel representation-based lens on feature learning.** The paper is the first to apply MCT to the lazy-rich learning dichotomy in the neural network training literature. This is a principled departure from weight/NTK-based approaches and is genuinely valuable for neuroscience applications where synaptic weights are inaccessible.
- **Formal theoretical grounding.** Theorem 1, while narrow, provides a real mathematical connection between capacity, learning rate (richness), and accuracy in a well-studied model. The proof requires nontrivial extensions from Ba et al. (2022) (regression → classification via Montanari et al. tools).
- **Interpretable geometric decomposition.** By decomposing capacity into radius, dimension, and alignment components, the framework provides mechanistic descriptors, not just a scalar summary. This is a concrete advantage over single-number measures.
- **Informative empirical demonstrations.** The scale factor interpolation experiments (Chizat et al. protocol) in both 2-layer and DNN settings show capacity reliably tracks richness and yields new qualitative structure invisible to weight-change or NTK alignment measures.
- **Suggestive application findings.** The RNN structural bias and OOD analyses are genuinely interesting observations that open avenues for further investigation.

---

## Weaknesses

### Fatal
*None.* The paper's core idea is sound and the evidence, though imperfect, supports the central framework.

### Major

- **The comparative claim ("better than conventional measures") is insufficiently established.** The comparison in Figure 3 is restricted to 2-layer synthetic settings and is purely visual/qualitative. No quantitative metric (e.g., correlation or rank-correlation with the ground-truth scale factor $\bar\eta$) is reported; no comparison appears in the DNN or RNN experiments where the framework is applied. The claim as stated in the abstract and Section 3.2 overstates what the evidence supports.

- **Learning stages (Section 4.2) lack formal grounding and replication.** The four stages in Figure 4c are identified by visual inspection of a single normalized trajectory for a single architecture (VGG-11) and dataset (CIFAR-10). There are no objective criteria for stage boundaries (e.g., sign changes in geometric derivatives, changepoint detection), no variance estimates across random seeds, and no demonstration that these stages appear consistently across architectures or datasets. As presented, this is an interesting exploratory visualization, not an established taxonomy.

- **OOD geometric claims are correlational, presented as explanatory.** The language in Section 5.2 ("explain the failure") implies causal attribution. All geometric measures co-vary with the richness parameter $\bar\eta$, so the attribution of OOD failure specifically to radius expansion and center-axis alignment is not isolated from the richness axis itself. The paper acknowledges this gap ("We leave it as a future direction") but does not moderate the language in the main text accordingly.

### Minor

- **Theoretical scope mismatch.** Theorem 1 covers one gradient step in a 2-layer fixed-readout teacher-student model. The paper is transparent about this in footnote 6, but the main text (Abstract, Section 1.1, Section 3.1) sometimes uses language implying broader theoretical validation ("we theoretically and empirically show"). The caveat should be more prominent in the main text.

- **Last-layer analysis only.** Section 2.3 explicitly restricts analysis to the last layer, which is precisely where supervised classification optimizes class separation. This makes the framework appear particularly favorable. The paper does not discuss whether capacity is informative at intermediate layers or in settings where last-layer separability and feature richness can diverge.

- **No discussion of computational cost.** Computing mean-field manifold capacity requires solving a quadratic program per manifold per random dichotomy/projection. The paper never reports timing or scaling characteristics compared to simpler measures, which is relevant for practical adoption.

### Trivial

- Minor: the geometric approximation $\alpha_{mf} \approx (1 + R_{mf}^{-2})/D_{mf}$ is used in the contour plots of Figure 4. The accuracy of this approximation in the regimes shown is not reported, though the results are primarily illustrative.

---

## Nice-to-Haves

- **Quantitative comparison with conventional measures.** Even a simple metric—Spearman rank correlation between each measure and the ground-truth $\bar\eta$ across seeds and settings—would turn the qualitative visual comparison in Figure 3 into a rigorous result.
- **Objective criteria for learning stages.** Changepoint detection or derivative sign-change criteria applied to the geometric trajectories would allow reproducible stage identification and cross-architecture validation.
- **Validation on intermediate layers.** Showing capacity tracks richness in intermediate layers (not just the last layer) would substantially broaden the claimed scope.
- **Intervention experiment for OOD.** Regularizing against radius expansion or center-axis alignment growth during training and testing whether OOD probe accuracy improves would convert the correlational OOD finding into a genuine mechanistic insight.
- **Variance estimates / multi-seed analysis.** Error bars or seed-level variation for at least the main empirical results would improve confidence in the robustness of learning stages and strategies.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Biological relevance requires real neural data"** (Harsh Critic / Human Finder): The paper explicitly scopes itself to RNN simulations and neuroscience tasks, noting "current limitations in neuroscience technology for precisely tracking synaptic weight changes necessitate a framework based on neural representations." It does not claim to validate on actual neural recordings—this is prospective motivation, not a factual error. Requiring real neural data goes outside the stated scope.
- **"Superiority of MCT over SVCCA/CCA etc."** (Spark reviewer): The paper does not compare against these measures, but the claim "better than conventional measures" is already cited and criticized under Major weaknesses with the appropriate scope (the paper chooses specific baselines). Adding more baselines is a nice-to-have, not a missing validation demanded by the paper's own claims.
- **Demands for theoretical coverage of deep/multi-step dynamics**: The paper explicitly scopes the theory to one step and acknowledges the gap. Criticizing the absence of a full training-trajectory theorem for deep networks is outside the paper's stated scope; this is already captured under Minor: Theoretical scope mismatch.
- **Architecture diversity (transformers, ViTs)**: The conclusion explicitly lists this as future work. The current architecture coverage (VGG-11, ResNet-18, RNNs) is adequate for the stated framework demonstrations. This is a nice-to-have.

---

## Novel Insights

The paper's most genuinely novel observation is the **geometric decomposition of learning dynamics into interpretable trajectories**—showing that different richness levels and initialization regimes follow distinct paths through the (radius, dimension) space, with some regimes trading radius for dimension in different orderings. This goes beyond the binary lazy/rich framing and provides a richer language for describing representational change. The finding that RNNs with different initial weight ranks reach similar final capacity but via geometrically distinct routes (Section 5.1) is a concrete, interesting extension of Liu et al. (2024)'s weight-based analysis into representation space, and represents a type of analysis that is more tractable for neuroscience-motivated study. The OOD radius/center-axis alignment correlation, while correlational, suggests a concrete geometric signature worth investigating causally.

---

## Suggestions

1. **Moderate the "better than" comparative claim** by either (a) adding quantitative metrics to Figure 3 or (b) narrowing the claim to "capacity provides complementary information" rather than strictly superior information.
2. **Add formal stage criteria** (e.g., threshold on derivative sign changes) and replicate Figure 4c across at least three seeds and a second architecture to establish robustness.
3. **Change "explain" to "correlate with"** in Section 5.2 and explicitly note that causal validation is left to future work in the main text.
4. **Include a paragraph on computational cost** relative to the simpler measures (weight change, CKA).
5. **Acknowledge the last-layer favorability** explicitly in Section 2.3 as a designed choice with its attendant limitations.

---

## Score and Decision

**Calibration comparisons:**
- **slSmYGc8ee.md** (*How connectivity structure shapes rich and lazy learning in neural circuits*, Accepted, scores 8/6/8/5): This paper also studies lazy/rich regimes in RNNs with strong theory and broad empirics. It received an accept. The current paper is comparable in topic and somewhat comparable in quality, though its theoretical scope is narrower (one-step only vs. Theorem 1 + experimental sweep) and its comparative claims are thinner.
- **k9t8dQ30kU.md** (*Task structure and nonlinearity*, Accepted, scores 5/6/8/8): This paper studies representational geometry in similar settings, mostly empirically, and was accepted. Its core weakness ("mostly empirical without theoretical insights") parallels a minor concern here, though the current paper does provide a theorem.
- **vt5mnLVIVo.md** (*Grokking as lazy-to-rich transition*, Accepted, scores 8/8/3/5): Received highly variable scores but was accepted; notably, it makes similar broad claims about lazy-to-rich transitions with a restricted formal theory.
- **oKglS1cFdb.md** (*Feature Accompaniment / OOD*, Rejected, scores 6/6/5): Rejected; this paper also makes correlational OOD claims without actionable interventions. The current paper is stronger in that the OOD section is one application among several, not the central contribution.

The current paper brings a principled, formally-grounded (if narrowly so) representation-based method to an important problem in understanding feature learning. It makes genuine novel contributions and is well-motivated. Its weaknesses—overclaiming on comparative superiority, single-run learning stage identification, and correlational OOD language—are real but do not undermine the core contribution. The paper lands near the acceptance threshold, slightly above given the novelty of the approach and the non-trivial theorem. Calibrating against the above anchors, this is comparable to k9t8dQ30kU.md (Accepted, avg ~6.75) and marginally weaker than slSmYGc8ee.md.

**Overall assessment:** The paper is original, addresses an important research question, provides claims that are mostly (if imperfectly) supported, conducts sound experiments within its stated scope, and writes clearly. The value to the neuroscience-meets-machine-learning community is genuine. The main issues are scope of comparative evidence and the descriptive/correlational nature of some applications—both manageable with revision.

**Score: 6.0** — Marginally above acceptance threshold. The paper makes real contributions with a principled methodology, but the comparative claims and learning-stages analysis need stronger evidence to fully justify the broad narrative.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>