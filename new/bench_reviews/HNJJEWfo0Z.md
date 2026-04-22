Good — these are very relevant anchors. Let me now synthesize the final review.

## Summary

The paper investigates why perturbation-based fidelity metrics for saliency maps (e.g., AOPC, AD%, IC%) produce inconsistent results, extending Tomsett et al.'s observation that they *are* inconsistent. The authors formalize two assumptions they claim underlie such metrics — [P1] that perturbing any pixel decreases output probability, and [P2] that pixel importance rankings should be invariant across perturbation types — and propose two conformity measures: DROP (quantifying [P1] violations) and PSim (quantifying [P2] violations). Large-scale experiments across 9 perturbation types, 5 models, 3 datasets, and ~75M predictions show DROP ≈ 0.5–0.6 and PSim well below 1, indicating systematic violations.

## Strengths

- **Important problem formulation**: The paper tackles a meaningful question — *why* fidelity metrics are inconsistent — going beyond Tomsett et al.'s observation that they *are* inconsistent. Identifying the root causes of metric inconsistency in XAI evaluation is a valuable direction (Section 1, p. 29–33).

- **Comprehensive empirical scope**: The evaluation spans 9 perturbation types, 5 models (including 2 adversarially trained), 3 datasets, and both pixel-wise and segment-wise perturbation schemes, with ~75M prediction calls (Section 4, Table 1). This scale goes well beyond prior work and establishes that inconsistency is a pervasive property rather than a narrow artifact.

- **Actionable finding about Gaussian Blur**: Figures 2–3 and 4–5 consistently show Gaussian Blur variants yield higher DROP scores and higher pairwise PSim (e.g., ResNet50 peaks near 1.0 in Figure 2; G3–G9, G3–G15, G9–G15 pairs in Figure 4). This provides concrete, immediately usable guidance for practitioners choosing perturbation types.

- **Adversarial training analysis**: Table 2 shows adversarially trained ResNet50 models have even lower PSim scores (0.18–0.33 for pixel-wise), extending the finding that inconsistency is fundamentally a property of model behavior under perturbation (Table 2, Section 5.1).

- **Clear algorithmic specification**: Algorithm 1 and Figure 1 together provide an unambiguous step-by-step procedure for computing DROP and PSim, making adoption straightforward.

## Weaknesses

### Fatal

None.

### Major

- **[P1] is stated too strongly as a universal requirement, weakening the diagnostic's relevance to fidelity metrics.** Eq. 2 states $p_0 > p_i^\phi \;\forall\; i, \phi$ — that *every* pixel perturbation must decrease output probability. But fidelity metrics like AOPC operate by comparing a saliency map's ranking against a perturbation-derived ranking, requiring only that the *ordering* of probability drops is meaningful. A pixel that is unimportant and causes probability to *increase* when perturbed is entirely compatible with a fidelity metric; the saliency map should simply assign it low importance. By computing DROP over *random* pixels (50 random pixels, Section 4.1) rather than the *saliency-map-ranked* pixels that fidelity metrics actually target, the paper's diagnostic doesn't align with how these metrics operate. A DROP of ~0.5 on random pixels is unsurprising and doesn't directly tell us about the reliability of fidelity metrics that focus on top-ranked pixels. — This matters because it undermines the paper's core claim that DROP diagnoses conformance to fidelity metrics' needs.

- **The gap between "rankings differ across perturbations" and "metrics are unreliable" is not closed.** PSim demonstrates that different perturbation types produce different importance rankings (Section 5.3, Figure 4), but the paper leaps to the conclusion that fidelity metrics are therefore "unreliable" (Abstract, Conclusion). This requires an additional premise: either that there exists a single ground-truth ranking which fidelity metrics should recover, or that cross-perturbation consistency is a precondition for validity. The paper provides no argument for either premise. If different perturbations legitimately capture different aspects of a model's behavior, a fidelity metric using perturbation φ could be perfectly valid *for that perturbation*, even if it disagrees with the metric using perturbation ψ. Low PSim re-describes the problem Tomsett et al. already identified rather than explaining it in terms the metrics themselves depend on. — This matters because it weakens the paper's central explanatory claim.

- **"Theoretically establish" in contributions is an overclaim.** Section 1.1 states "We first theoretically establish the scenarios under which such assumptions are violated" but Section 2.1 contains no theorem, condition, or analytical result establishing when assumptions hold or fail. It merely restates the assumptions in equation form (Eqs. 1–5). The formalization itself is reasonable, but calling it "theoretical establishment" overpromises. — This matters because readers expecting analytical conditions will find only notation.

### Minor

- **DROP's indicator function uses ≥0 while [P1] requires strict inequality.** Eq. 6 uses $[(p_0 - p_s^\phi) \geq 0]$, treating probability-stays-same as conforming, while [P1] (Eq. 2) requires $p_0 > p_i^\phi$. This subtle inconsistency means DROP overcounts conformity relative to the stated assumption.

- **50 random pixels is a very sparse sample (~0.1% of a 224×224 image).** The paper relies on the claim that "a subset of a ranked order list maintains ranking" (Appendix S2), but this only holds if the sample is representative of the full ranking's structure. For such a small fraction, PSim computed on 50 pixels may be dominated by noise, especially when pixel importance differences are small.

- **DROP's practical utility as a preconditional check is questionable.** The paper shows DROP is almost always ~0.5–0.6 for most perturbation types (Table 1, Figure 2). If DROP will almost always flag a problem, it acts as an uninformative gatekeeper rather than a useful diagnostic. Similarly, low PSim tells you rankings differ but doesn't guide *which* perturbation to trust (Section 6, Conclusion).

- **Notation is slightly tangled.** $R_\sigma$ is introduced in Section 2.1 but never used in subsequent equations. The distinction between the saliency-map-derived ranking $\mathfrak{R}$ (Eq. 1) and the perturbation-derived ranking $\mathfrak{R}(\phi)$ (Eq. 4) — which is what fidelity metrics actually measure agreement between — is never formally related.

### Trivial

- None.

## Nice-to-Haves

- Computing DROP specifically over *top-k pixels by saliency-map importance* (rather than random pixels) would directly test the conformance that fidelity metrics require, making the diagnostic more relevant.
- Showing that low DROP or PSim *predicts* the degree of inconsistency in fidelity metrics (e.g., that images with lower DROP produce more divergent AOPC scores) would demonstrate the diagnostic value of the conformity measures.
- Investigating *why* Gaussian Blur performs differently (e.g., whether it better preserves the local neighborhood assumption) would transform an observation into an explanation — this is the paper's most actionable finding.
- Analytical derivation of conditions under which [P1]-type and [P2]-type violations occur would constitute the theoretical contribution claimed.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Theoretical establishment overclaim" as Fatal**: The harsh critic treated this as a structural failure, but the formalization of assumptions (Eqs. 1–5) is useful even without proving conditions for violation. The overclaim is a presentation issue (downgraded to Major).
- **"No fidelity metric claims perturbation-invariance" as a standalone criticism**: The harsh critic argued [P2] is a straw man because no metric claims perturbation-invariance. However, the paper's point is that the metrics' *implicit* dependence on PIR consistency creates a vulnerability — even if this isn't explicitly claimed by metric designers, it's a logically necessary precondition for the metrics to agree across perturbations. This is partially valid but not a fatal straw man.
- **Formatting/presentation nitpicks**: The harsh critic noted tangled notation and figure labeling issues. These are minor but substantive enough to retain in simplified form under Minor.
- **Missing appendix references**: Several criticisms referenced proofs or details in the appendix. Since the parser strips appendices, these cannot be fully evaluated and are excluded from the main review.
- **"The paper doesn't leverage Gaussian Blur finding"**: This is a nice-to-have suggestion, not a weakness. The paper does present the finding clearly; deeper analysis would strengthen but doesn't undermine existing contributions.

## Novel Insights

The most insightful observation across the reviews is that there is a fundamental asymmetry between what DROP measures (probability behavior on *random* pixels) and what fidelity metrics actually need (probability behavior on *saliency-ranked* pixels). This mismatch is the clearest structural gap in the paper: if most random-pixel perturbations don't decrease probability but the top-saliency-ranked pixels do, fidelity metrics could still work correctly. The paper doesn't test this critical alternative, leaving its core diagnostic claim under-supported.

## Suggestions

- Compute DROP and PSim specifically over pixels/segments ranked by each saliency method being evaluated. If these conformity scores remain low even for top-ranked pixels, the paper's conclusions would be substantially stronger.
- Add a direct validation experiment: measure the correlation between DROP/PSim values and the actual inconsistency of fidelity metrics (e.g., the variance of AOPC scores across perturbation types) at the image level. This would establish whether the proposed measures have diagnostic value.
- Moderate the "theoretically establish" claim in the contribution statement to accurately reflect that the paper *formalizes* the assumptions rather than analytically deriving violation conditions.
- Investigate why Gaussian Blur behaves differently — this is the paper's most practical finding and deserves deeper analysis.

## Score and Decision

**Originality**: The paper identifies a worthwhile question but the formalization restates assumptions at a level that doesn't add significant analytical insight beyond what careful reading of Tomsett et al. would suggest. The two conformity measures are simple and reasonable but not deeply novel.

**Importance of research question**: High — understanding why evaluation metrics for XAI fail is important for the community.

**Claims well supported**: Partially. The empirical findings are well-supported at scale, but the central claim that DROP/PSim *explain* why fidelity metrics are inconsistent is not well-supported due to the mismatch between what DROP measures and what fidelity metrics need, and the unclosed gap between "rankings differ" and "metrics are unreliable."

**Soundness of experiments**: Thorough in scope but with a design limitation (random pixels vs. saliency-ranked pixels).

**Clarity of writing**: Adequate, with some notation issues.

**Value to community**: The Gaussian Blur finding is immediately actionable, and the recommendation to specify perturbation type when reporting fidelity scores is practical and valuable.

Comparing against calibration anchors:
- **High-scoring anchors** (>7): `PBjCTeDL6o` (avg 8, Oral — novel adaptive baseline method with strong theoretical+empirical contribution); `5IWJBStfU7` (avg 7, Accept — identifiability analysis with deep formal taxonomy and non-trivial experiments). This paper has weaker analytical depth and the diagnostic doesn't fully align with what it claims to diagnose.
- **Medium-scoring anchors** (4–6): `mKGXdsq7fD` (avg 4.33, Withdrawn — pixel-level saliency evaluation replicates known findings without significant new insight); `EwAGztBkJ6` (avg 4.0, Reject — formalizes generalization of saliency but questioned whether formalization proves anything meaningful); `ZYuiuxB7H4` (avg 4.0, Withdrawn — certifies attribution robustness under perturbation but questioned significance of assumptions). This paper is in a similar space: it formalizes assumptions underlying existing metrics and shows they're violated, but the formalization is shallow and the diagnostic doesn't perfectly align with its target.
- **Low-scoring anchors** (<3): `wJVZkUOUjh` (avg 2.0, Reject — EXAGREE with fundamentally flawed formalization and unsubstantiated claims). This paper is clearly better than these — the empirical work is genuinely thorough and the findings about DROP/PSim patterns are real, just not as diagnostic as claimed.

The paper sits in the medium band, similar to the saliency evaluation papers that formalize known problems without adding deep analytical insight. Its empirical scale and Gaussian Blur finding push it slightly above the weakest_medium anchors, but the structural gap between DROP's measurement and fidelity metrics' needs, plus the unclosed inferential gap for PSim, keep it from scoring higher.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>