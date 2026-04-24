## Summary

This paper introduces DyAug, the first graph data augmentation framework specifically designed for discrete-time dynamic graphs (DTDG). DyAug addresses the limitations of static graph augmentation methods by proposing temporal-conditioned rationale-environment separation, where causal subgraphs (rationales) are extracted in a Markovian fashion across snapshots and augmented via three environment-replacement strategies in the latent space. The method is evaluated on five datasets, three backbones, and three regimes (standard performance, adversarial robustness, and out-of-distribution generalization).

## Strengths

- **Novel problem framing and method.** DyAug pioneers graph rationalization for dynamic graphs by extending the static rationale-environment paradigm with temporal conditioning (Eq. 2) and consistency regularization. The embedding-space augmentation design (Eq. 8–10) is an elegant way to avoid the structural disruption caused by per-snapshot topological perturbations.
- **Comprehensive empirical landscape.** The evaluation spans five real-world datasets, three DyGNN backbones (GCRN, DySAT, SEIGN), and three distinct evaluation regimes. Table 1 shows consistent performance improvements across all 15 backbone-dataset combinations, and Figure 5 demonstrates strong adversarial robustness gains.
- **Diagnostic evidence for temporal consistency disruption.** Figure 1 and Figure 4 provide concrete empirical evidence that rule-based static augmenters (DropEdge, DropNode, GraphMixup) skew the edge-timespan distribution toward short-lived edges, while DyAug and RGDA preserve the vanilla distribution more faithfully.

## Weaknesses

### Fatal
None.

### Major
- **Equation 6 is mathematically inverted as written.** The paper defines graph “similarity” as an L1 distance (`sim(G_t^R, G_p^R) = sum(|M_t^R - M_p^R|)`) and inserts it into an InfoNCE-style denominator (Eq. 6) intended to pull temporally nearby rationales together. Because L1 distance is inversely related to similarity, the loss as written would penalize proximity: minimizing it requires making positive pairs dissimilar and negative pairs similar—the exact opposite of “temporal consistency.” This is a serious notational error in a core equation. That said, the ablation study (Figure 6) shows that removing the consistency regularization (`w/o CR`) slightly degrades performance (80.10 vs. 81.40 AUC on clean data; 76.70 vs. 77.40 under structure attack), which strongly suggests the implementation uses a corrected form (e.g., negative distance) and that this is a typographical error rather than a fundamental methodological flaw. Nevertheless, the equation as printed undermines the formal claim and must be corrected.
- **Ablation study is too narrow to support broad claims.** The ablation in Figure 6 is restricted to a single dataset (ACT) and a single backbone (GCN). The paper makes strong “triple win” claims across performance, robustness, and OOD generalization for a framework pitched as generally applicable. Without ablations extended to adversarial and OOD settings—where the attribution of gains to specific components (rationale extraction, temporal conditioning, contrastive loss, or replacement augmentation) is most needed—the reader cannot determine which modules drive robustness and generalization.

### Minor
- **Motivational argument is correlational, not causal.** The paper shows that per-snapshot DropEdge/GraphMixup alter edge timespans and hurt performance, but it does not include a controlled baseline (e.g., dropping the *same* edges across consecutive snapshots) to isolate temporal consistency as the root cause rather than the general destructiveness of independent per-snapshot randomization. The paper’s actual claim is appropriately hedged (“not fully applicable”), but a controlled check would strengthen the argument.
- **OOD baseline comparison lacks methodological transparency.** Table 2 compares DyAug against IRM, DIDA, and DGIB-Bern under distribution shifts, but it is unclear whether these baselines are built on comparable DyGNN backbones (e.g., SEIGN, DySAT, GCRN) or use the same tuning budgets. This makes it difficult to assess whether the gaps reflect methodological superiority or implementation mismatch.
- **Causal identification is asserted but not empirically validated.** Section 3.3 presents a heuristic Structural Causal Model and assumes that the learned mask `M^R` isolates the true causal variable `C` and that `M-bar` captures the confounder `S`. No synthetic experiment or intervention is provided to validate that this identification occurs in practice.

### Trivial
- **Mislabeled backbone in Table 2.** The table lists “GCN (Seo et al., 2018),” but Seo et al., 2018 is GCRN (a GCN+GRU architecture), not GCN. The paper correctly identifies it as GCRN elsewhere.

## Nice-to-Haves
- Qualitative visualizations of rationale mask trajectories `M_t^R` across time for sample nodes/edges to empirically demonstrate smooth temporal evolution.
- Failure-mode case studies showing concrete examples where DyAug corrects errors made by vanilla DySAT or naive DropEdge.
- A mechanistic explanation of *why* environment-replacement augmentation specifically defends against Nettack or structure attacks (e.g., spurious-edge suppression vs. generic regularization).
- Discussion of whether the rationale-environment separation can be adapted to continuous-time dynamic graphs (CTDG), given that the current scope is limited to DTDG.

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **“Static methods are categorically unfit” / “inherently incompatible.”** The harsh critic attributes this claim to the paper, but the text actually states static methods are “not fully applicable” and acknowledges that RGDA (a static learning-based method) achieves “modest improvement” when adapted. This is a strawman.
- **Criticism that Eq. 4’s parameterization is unexplained.** The use of `ε ~ Uniform(0,1)` inside a sigmoid with temperature is the standard Binary Concrete / Gumbel-softmax relaxation for differentiable Bernoulli sampling; the paper’s explanation is standard in the graph rationalization literature.
- **Missing related works (DIR, GREA, JOAO, AIA).** The paper explicitly acknowledges and justifies these exclusions in Section 4.1.
- **“Progressive sparsification” undefined in main text.** This is a minor figure-to-text label mismatch; the environment subgraph is defined explicitly in Eq. 5.
- **Notation inconsistency (⊕ vs. ⊙).** While the paper switches symbols between element-wise operations, the meaning is recoverable from context (`M-bar = A - M`); this is a minor presentation issue.
- **Extension to CTDG as a weakness.** The paper explicitly scopes itself to DTDG; demanding CTDG experiments is scope creep.
- **Structure-space vs. embedding-space augmentation comparison.** This is a valid research direction but not a weakness of the current work.
- **Various typos, grammar, and formatting artifacts.** These are parser errors or trivial issues that carry no evaluative weight.

## Novel Insights

The most insightful observation across the reviews is the tension between the paper’s strong empirical showing and its formal under-specification. The embedding-space augmentation paradigm—replacing environment embeddings rather than graph edges—is a genuinely elegant solution to the temporal-consistency problem, and the comprehensive three-regime evaluation sets a good standard for dynamic graph augmentation papers. However, the paper would benefit enormously from tightening the formal presentation of Eq. 6 and expanding the ablation to match the breadth of its empirical claims.

## Suggestions

1. **Correct Eq. 6.** If the implementation uses negative L1 distance (or cosine similarity), update the equation and text to match. If the equation is indeed implemented as written, the effect should be carefully re-examined because the ablation suggests the opposite behavior.
2. **Expand the ablation study.** Report component ablations (w/o TC, w/o RA, w/o CR, w/o CL) on at least one additional dataset and under adversarial and OOD conditions to isolate which modules drive robustness and generalization.
3. **Clarify OOD baseline setup.** Explicitly state what DyGNN backbones (if any) IRM, DIDA, and DGIB-Bern use in Table 2, and ensure comparable tuning procedures.

## Score and Decision

**Calibration reasoning:**  
I compared this paper against several anchors from the human-review corpus:

- **`y5einmJ0Yx.md`** (avg 7.50, Accept Spotlight): GOLD for graph OOD detection. DyAug is below this anchor; GOLD had no obvious technical flaws, solid theory, and comprehensive experiments. DyAug’s central equation error and narrow ablation place it clearly below this tier.
- **`AJBkfwXh3u.md`** (avg 6.00, Accept Poster): Causality-inspired explanations for DyGNNs. This paper also had equation clarity issues but was accepted. DyAug has stronger and more comprehensive experiments (5 datasets, 3 backbones, 3 regimes), but its Eq. 6 error is more central to the core methodological claim. I place DyAug slightly below this anchor.
- **`rWHMe2O5VW.md`** (avg 5.00, Reject): Graph ODE with factorized prototypes. Had thorough experiments and theoretical analysis but weaknesses in motivation and related-work differentiation. DyAug is comparable: stronger empirical consistency but a more concrete formal flaw in Eq. 6.
- **`bCNYFOaWsy.md`** (avg 5.25, Reject): TOBE for class-imbalanced graphs. Had theoretical support but mixed experimental results. DyAug has more consistent wins but a more serious notational error.
- **`yCr55EjC1d.md`** (avg 3.50, Reject): NodeDup. A simple heuristic with limited novelty. DyAug is substantially stronger in contribution and empirical depth.

DyAug sits in the same band as the 5.0–5.25 anchors: it makes a real contribution (first dynamic-graph rationalization/augmentation framework) and backs it with extensive experiments, but the inverted Eq. 6 and the overly narrow ablation are significant enough to keep it below the acceptance threshold without author clarification.

**Score:** 5.0  
**Decision:** Reject

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>