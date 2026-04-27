Now let me do calibration searches to score this paper relative to anchors.Now I have sufficient calibration data. Let me compile the final review.

---

## Summary

KOALA (KL-L0 Adversarial Detector via Label Agreement) proposes a semantics-free adversarial detector that flags adversarial inputs when nearest-prototype classifiers using KL-divergence and L₀-based metrics disagree. The method requires only clean-image fine-tuning and provides a formal theorem (Theorem 1) guaranteeing detection when inter-class prototype separation is sufficiently large. Experiments on ResNet-18/CIFAR-10 and CLIP/Tiny-ImageNet show strong detection on theorem-compliant subsets and reasonable aggregate metrics.

---

## Strengths

- **Formal theorem with empirical validation**: Theorem 1 establishes explicit conditions for guaranteed detection. Table 1 directly validates this: on theorem-compliant subsets, KOALA achieves perfect precision, recall, and accuracy of 1.0 across both models and both perturbation budgets, providing genuine empirical support for the theoretical claim.

- **Empirically validated complementarity**: The ablation in Table 2 shows KL+L₀ achieves the highest F1 (0.87) on ResNet/CIFAR-10, outperforming L₀+Cosine (0.66), KL+Cosine (0.74), and the three-way combination (0.69). Table 3 further confirms that KL+L₀ fine-tuning yields 57.32%/54.60% adversarial accuracy under PGD at ε=2/255 and 4/255 respectively — substantially better than single-metric baselines. The complementarity argument is not just stated but demonstrated.

- **Clean-image-only training**: The method avoids adversarial examples or semantic priors during fine-tuning, making it architecture-agnostic and practically lightweight. This is a meaningful practical advantage over methods like Metzen et al. (2017) or semantics-driven detectors.

- **Honest and transparent analysis of failure modes**: Section 4.3 candidly identifies that the CLIP KL+L₀+Cosine combination achieves high detection by breaking the underlying classification (adversarial accuracy near 0), not by genuine robustness, and correctly distinguishes this from the preferred KL+L₀ combination.

- **Theorem-compliant vs. non-compliant partition**: Splitting the test set by Theorem 1 compliance (Table 1) is an honest experimental design that transparently communicates where the theory applies and where it does not, rather than reporting only aggregate numbers.

---

## Weaknesses

### Fatal
None that outright invalidate the existence of the method.

### Major

- **No adaptive attack evaluation.** The paper evaluates exclusively against PGD, CW, and AutoAttack — attacks designed to fool the classifier, not the detector. The detection mechanism is publicly specified: flag inputs when ŷ_KL ≠ ŷ_L₀. An adaptive adversary can directly minimize disagreement between the two prototype classifiers as an auxiliary loss term. Theorem 1's incompatibility argument applies to a *fixed* threshold τ and *fixed* prototypes; it does not account for an adversary who optimizes δ with knowledge of τ, the prototypes, and the disagreement condition. This is a non-negotiable requirement in the adversarial ML literature (as the KOALA paper itself acknowledges by citing Carlini & Wagner and Croce & Hein). Without it, the empirical detection claims are unverified against the adversary the system is designed to defend against. The comparable SPADE paper (a provable adversarial detector also accepted at the venue at avg score 5.5) was also criticized for this, but SPADE provides quantitative comparison against prior detectors which partially compensates; KOALA does neither.

- **No comparison to existing adversarial detectors.** The related work section cites Feature Squeezing, LID, MagNet, Mahalanobis distance, NIC, and CADet, but none appear as baselines in Tables 1 or 2. The ablation compares only within-KOALA metric combinations. The headline precision of 0.94 and recall of 0.81 on ResNet/CIFAR-10 are uninterpretable without a common-reference comparison to prior detectors on identical setups. This is a standard requirement for the field.

- **TP definition conflates attack failure with detection.** The evaluation in Section 4.2 defines:

  > TP := [a=1] ∧ [(â,ŷ)=(1,⊥) ∨ **(â,ŷ)=(0,y*)**]

  The second disjunct scores an adversarial example as "detected" if the model correctly classifies it despite the perturbation — even if the detector never flagged it (â=0). This makes the recall metric partially measure the backbone classifier's natural robustness rather than KOALA's detection capability. From Table 3, the unmodified ResNet-18 achieves ~45% adversarial accuracy under PGD ε=2/255 — all those correctly classified adversarial examples are counted as TPs for detection even with no detector. The reported recall of 0.81 cannot be cleanly interpreted as measuring the detector's standalone contribution, making cross-method comparison invalid.

- **Theorem 1's coverage is limited to ~10% of the harder evaluation setting.** For CLIP/Tiny-ImageNet at ε=2/255, only 510 out of 5000 test samples satisfy Theorem 1's conditions (10.2%). The paper acknowledges this is "due to the massive scale of CLIP's pre-training" but does not analyze what structural properties of the fine-tuning affect compliance rates, nor does it show whether fine-tuning increases compliance. The guarantee with perfect recall applies to ~10% of test inputs; the remaining 90% fall in the non-compliant regime with recall dropping to 0.84 (at ε=2/255) and F1 of 0.72. While the paper is honest about this, it substantially qualifies the claim that KOALA provides "a mathematical certainty" of detection.

### Minor

- **CLIP adversarial accuracy inconsistency.** Table 4 shows KL+L₀ on CLIP achieves only 26.50% adversarial accuracy under PGD ε=2/255, compared to 60.02% (KL-only) and 53.31% (L₀-only). The paper correctly criticizes KL+L₀+Cosine for "breaking underlying classification" in Section 4.3, but does not apply the same scrutiny to KL+L₀ itself on CLIP, which also underperforms individual metrics under AutoAttack and CW. This asymmetric treatment weakens the CLIP results.

- **Impact of Assumption A1 normalization not analyzed.** Assumption A1 requires softmax normalization of all feature embeddings to make them probability distributions. For ResNet-18, penultimate layer features are trained under cross-entropy without this constraint, so applying softmax normalization at detection time distorts the trained geometry. No experiment measures whether this normalization degrades clean accuracy or changes embedding structure relative to the backbone's training objectives.

### Trivial

- No learning curves or training dynamics are reported; convergence behavior of the composite loss is not discussed.

---

## Nice-to-Haves

- Embedding geometry visualizations (t-SNE or PCA) showing KL and L₀ decision regions before and after fine-tuning would ground the theoretical claims in observable structure.
- An analysis of what geometric properties of the embedding space determine theorem compliance (i.e., why ~67% of CIFAR-10 samples are compliant but only ~10% of Tiny-ImageNet samples).
- Reporting clean false positive rates separately from the mixed TP metric would allow assessment of false alarm behavior.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh critic — proof sketch's τ argument.** The critic argues that τ being "chosen in response to a particular attack" invalidates the incompatibility argument. However, the proof sketch (Section 3.2) is clear that τ is a model hyperparameter fixed before any input is seen, and the argument is about existence of such a τ. The full proof is in the appendix (stripped by the parser). This critique is partially speculation about the appendix proof, and is REMOVED.

- **Harsh critic — KL divergence as inner product.** The claim that softmax normalization "destroys angular and magnitude structure" is an assertion without checking whether the ResNet fine-tuning actually mitigates this via the training objective. This is speculative and too minor to retain as a standalone weakness given the paper's acknowledged design trade-offs. REMOVED as a standalone weakness (merged into the A1 minor point).

- **Harsh critic — CLIP baseline fragility.** The critic characterizes the result as "the detection mechanism compensating for a fundamentally broken classifier." This overstates the case — the paper explicitly scopes CLIP results to show detection metrics; it does not claim CLIP KL+L₀ is a complete robust system. The characterization misreads the paper's framing. REMOVED.

- **Strength Finder — "Differentiable L₀ surrogate enables gradient-based training."** This is a genuine technical contribution but is generic engineering (differentiable relaxations of discrete functions are standard). Retained only as implicit support of the training recipe, not as a standalone strength.

---

## Novel Insights

The key novel observation is that two complementary prototype classifiers — one sensitive to dense distributional shifts (KL) and one sensitive to sparse high-amplitude shifts (L₀) — create mutually exclusive prediction stability bands under energy-constrained perturbations, making disagreement a provable detector. This "dual-band" perspective on the geometry of adversarial perturbations is a clean and generalizable framing. The empirical observation (Table 2) that adding a third metric (Cosine) consistently degrades ResNet detection performance, while counterintuitively boosting CLIP "detection" by destroying the backbone, is a genuinely insightful finding about metric interference in prototype classifiers. However, the paper leaves unexplained why the proposed joint fine-tuning objective actually improves theorem compliance rather than asserting it by design.

---

## Suggestions

1. **Add an adaptive attack baseline**: Implement PGD with an auxiliary loss term penalizing disagreement between ŷ_KL and ŷ_L₀ (or maximizing agreement on the adversarial class). This is necessary to establish any empirical security claim.

2. **Fix the TP definition**: Use a strict definition where TP = adversarial example flagged (â=1), and FN = adversarial example not flagged regardless of classifier outcome. Report both the strict detection metric and the "system success" metric separately.

3. **Add detector baselines**: At minimum, run Feature Squeezing and Mahalanobis distance on the same ResNet-18/CIFAR-10 setup using identical adversarial attacks.

4. **Analyze compliance rates**: Report how fine-tuning changes the fraction of theorem-compliant samples, and what structural property (e.g., inter-class distance in prototype space) correlates with compliance.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison to KOALA |
|---|---|---|---|
| "Detecting Adversarial Examples" | `KAWlH5pfQu.md` | 3.00 | Similar topic (adversarial detection), no adaptive attacks, no prior baselines, but weaker: no formal theorem, partially incorrect proof |
| "Statistical Method for Attack-Agnostic Detection" | `kz78RIVL7G.md` | 2.60 | Similar topic, no adaptive attacks, no prior baselines, no formal theory, poor presentation |
| "Provably Safeguarding a Classifier from OOD and Adversarial Samples" (SPADE) | `kwCHcaeHrf.md` | 5.50 | Closest match: formal guarantee + adversarial detector, accepted as poster. Stronger: compares to state-of-the-art detectors on larger datasets. |
| "Adversarially Robust DL with OT-Regularized Divergences" | `ybavRGEmpw.md` | 4.50 | Adversarial robustness paper, medium quality, missing key experiments |
| "Data-Driven Lipschitz Continuity" | `GNOMC90vbl.md` | 4.80 | Adversarial defense with theoretical claims, borderline reject |

**Reasoning:** KOALA is clearly above the 2.6-3.0 cluster (KAWlH5pfQu, kz78RIVL7G): it has a genuine formal theorem, cleaner experiments, and honest analysis. However, it falls short of SPADE (5.50) — the most comparable accepted paper — in three critical ways: no comparison to prior detectors, a flawed TP metric that inflates results, and adaptive attacks missing. The three major weaknesses interact: the reported precision/recall numbers cannot be trusted due to the TP definition, cannot be contextualized without detector baselines, and cannot be stress-tested without adaptive attacks. The paper is closer to the 4.0–4.5 borderline range (ybavRGEmpw/GNOMC90vbl) than to SPADE.

**Score: 4.0 — Reject**

The paper has a genuinely novel core idea and a formal theorem, but the missing adaptive attack evaluation, absence of detector baselines, and inflated evaluation metric together prevent acceptance at this stage. These are not cosmetic issues — they concern the validity of the paper's central empirical claims.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>