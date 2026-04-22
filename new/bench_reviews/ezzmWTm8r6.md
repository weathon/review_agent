Now I have enough information to write the comprehensive final review. Let me synthesize everything.

---

## Summary
The paper proposes two logit-space loss functions for test-time adaptation (TTA): Sparse-CL, a positive pseudo-label loss derived by approximating the softmax probability as the raw exponentiated logit, and k-NL, a negative learning loss that selects the top-k hard-negative classes (excluding the closest confusable ones). The core motivation is that these losses yield constant L1 gradient norms, enabling stable training under high learning rates to counter pseudo-label noise. The method is applied as a drop-in loss replacement within the SAR framework, and evaluated on ImageNet-C across three challenging settings (normal, class-imbalanced, batch-size-1).

---

## Strengths

- **Constant gradient norm property, demonstrated empirically (Eqs. 11, 16; Figure 3):** The gradient analysis correctly shows that sparse-CL has L1 norm = 1 and k-NL has L1 norm = k regardless of logit values. Figure 3 validates this empirically, showing flat gradient norms for the proposed losses versus large fluctuations for entropy minimization and CE across hundreds of batches on two corruption types. This is the paper's best-grounded claim.

- **Figure 2 directly supports the stability claim:** Scaled-SAR (entropy minimization at the high LR used by the proposed method) collapses from ~55% to ~30% accuracy, while the proposed method holds at ~57%. This is concrete, direct evidence that the constant-norm property translates to stable adaptation—not just a theoretical assertion.

- **Consistent, substantial empirical improvements (Tables 1–3):** The method achieves +5.2% (imbalanced), +6.7% (batch-size-1), and +8.1% (normal) over SAR. Improvements are consistent across all 15 corruption types and particularly dramatic in collapse cases (e.g., Snow: SAR 39.1% → proposed 64.2%; Frost in batch-1: TENT 12.4% → proposed 64.3%).

- **Figure 4 provides mechanistic insight:** True pseudo-labels shift toward higher positive-logit regions and noisy labels toward negative-logit-dominant regions as the method is applied, with true sample counts increasing from ~9000 (EM) to ~14000 (full method). This is a clear, informative visualization independent of the derivation flaws.

- **Practical, well-scoped design:** The method modifies only the loss function with no architectural changes and is applicable at batch-size-1—a notoriously difficult TTA scenario.

---

## Weaknesses

### Fatal
None. The method is empirically valid even where the theoretical framing is wrong.

### Major

- **The derivation of sparse-CL rests on an unjustified approximation (Eq. 8).** The paper substitutes p_i ≈ exp(h_i) (i.e., treating the raw exponentiated logit as a probability) without any justification. This requires the softmax denominator ∑_j exp(h_j) ≈ 1, which is false for essentially all practical logit vectors—especially ImageNet's 1000-class setting. The paper writes "So we have p_i ≈ exp(h_i)" as a bare algebraic step with no qualification. The resulting loss L_sparse-CL = −h_k is a perfectly reasonable logit-maximization objective and can be justified directly (e.g., as margin maximization in logit space, or as the limit of label-smoothed CE), but the paper does not offer such alternatives. As currently presented, the theoretical framing—that sparse-CL is a principled cross-entropy surrogate—is formally invalid. This undermines the paper's claims to principled theoretical grounding.

- **The learning-rate and loss-function contributions are not fully isolated.** All three tables compare SAR† (entropy minimization at the original low LR) against SAR* + proposed loss (at a higher LR). Figure 2's scaled-SAR comparison shows that high LR with entropy minimization crashes, which is important evidence that the loss function—not just the LR—matters. However, the paper never shows sparse-CL at the same LR as SAR†, so it is impossible to determine how much of the 5–8% improvement comes from the loss function alone versus the interaction of a stable loss and a higher LR. The scaled-SAR experiment partially addresses this, but the full factorial design (loss × LR) is missing. For a paper whose central claim is "the new loss enables high-LR training," this ablation is important.

### Minor

- **The hardness-aware claim for k-NL (Section 3.3) is imprecise and technically wrong.** The paper states: "selecting k complementary labels in this way keeps the hardness-aware characteristic of the original negative loss unchanged." But Eq. 15 shows that ∇L_k-NL/∂h_i = 1 for selected negative classes, regardless of their probability—constant gradient, not probability-weighted. The hardness enters only in *selection* (which k classes are chosen), not in *gradient magnitude*, which is materially different from NL+'s gradient ∝ p_i. The claim as written is false, though the design motivation (focus updates on hard negatives) is reasonable and survives with a corrected description.

- **Evaluation is restricted to a single benchmark family.** All results are on ImageNet-C (severity 5) with three batch-size variants. These share the same corruption types and architecture, so they are not independent evaluations. No experiments on ImageNet-R, ImageNet-Sketch, CIFAR-10-C, or any non-corruption distribution shift are reported. For a paper claiming "diverse TTA experiments," this limits the generalizability of the conclusions.

- **No variance or confidence intervals reported.** In online TTA where sample ordering can matter, reporting variance across seeds or orderings would strengthen the empirical claims.

### Trivial

- The notation "SAR*" vs "SAR†" is not explicitly defined in the main body text (only extractable from context). A clear table footnote defining the two configurations would help readability.

---

## Nice-to-Haves
- Add an explicit derivation of sparse-CL that does not rely on the invalid p_i ≈ exp(h_i) step; the loss can be motivated as logit-space margin maximization with no approximation.
- Report pseudo-label accuracy over training steps to empirically verify the claim that fast adaptation (high LR) reduces memorization of incorrect labels.
- Add a logit-magnitude tracking plot to address the theoretical unboundedness concern (the method appears empirically stable, but the mechanism is unexplained).
- Evaluate on at least one additional benchmark (e.g., ImageNet-R or ImageNet-Sketch) to support broader generalization claims.

---

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Unbounded logit collapse concern (Critic Point 2):** While theoretically valid, Figure 2 shows empirical stability across hundreds of online batches. The paper operates within SAR's BN-only update framework, which in practice constrains how much logit magnitudes can shift. No empirical failure mode is demonstrated. Moved to trivial/nice-to-have rather than major.

- **"Unbounded noise ratio" motivation dropped in method (Critic):** This is scope-creep criticism. The paper identifies four challenges and addresses the ones within its design scope; not every stated problem needs a formal theoretical guarantee to produce a useful method.

- **CE contracting-gradient framing as "not a feature but a bug" (Critic):** The paper's argument that zero gradient for confident predictions is inefficient for fast adaptation is coherent within the paper's design goals. Whether this is a "feature" or a "bug" depends on the regime (noise tolerance vs. fast adaptation). The paper's framing is internally consistent.

- **Memorization argument is informal (Critic):** The connection between high LR and reduced memorization is indeed informal, but this is a motivating heuristic, not a core technical claim. Removing the memorization argument would not change the paper's method or results.

- **Figure 2 shows no logit-magnitude tracking (Critic):** This is a nice-to-have visualization, not a demonstrated failure. Moved to Nice-to-Haves.

- **Claim about three settings being "not independent":** This is accurate but a soft criticism; the three settings deliberately test distinct practically-motivated scenarios (label imbalance and batch-size stress).

---

## Novel Insights

The observation in Figure 4 that applying logit-space losses creates a cleaner geometric separation between true and noisy pseudo-labels—shifting true labels toward high positive-logit mass while pushing noisy labels into negative-logit-dominant regions—is a genuinely useful empirical observation that extends beyond this paper's specific method. The combination of positive logit maximization (to push the predicted class outward) and hard-negative logit minimization (to widen the decision gap) as complementary training signals in TTA is a relatively unexplored paradigm compared to entropy-minimization-dominated prior work. The skip parameter *s* in k-NL (discarding the *s* most confusable negatives) is a small but thoughtful design choice that could transfer to other noisy-label settings.

---

## Suggestions
1. **Replace the invalid p_i ≈ exp(h_i) derivation** with a direct justification of sparse-CL as logit-space margin maximization; this requires no approximation and gives the method a sound theoretical basis.
2. **Add a 2×2 ablation: {sparse-CL, EM} × {low LR, high LR}** to fully disentangle the contributions of the loss and learning rate. This is the single most important experiment missing from the paper.
3. **Correct Section 3.3's hardness-aware claim** to accurately describe selection-based hardness (choice of which k negatives) versus gradient-weighted hardness (NL+'s p_i-proportional updates).
4. **Add at least one non-ImageNet-C benchmark** (e.g., ImageNet-R or CIFAR-10-C) to support the generalization claim.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| "Entropy is not Enough for TTA: Disentangled Factors" | 9w3iw8wDuE.md | **7.0** (Spotlight) | Stronger theoretical foundation, novel confidence metric (PLPD), comprehensive experiments across biased/wild settings. Notably higher bar than this submission. |
| "Distribution Shift-Aware Prediction Refinement for TTA" | xqxG5WogN6.md | **5.67** (Reject) | TTA paper with clear practical contribution, rejected for limited scope and novelty concerns—comparable in scope to this submission. |
| "Adversarial Vulnerability of Label-Free TTA" | N0ETIi580T.md | **5.25** (Poster) | Accepted TTA paper with theoretical grounding and solid experiments; similar quality tier. |
| "Continual TTA with Source/EMA Prototypes" | eXrUdcxfCw.md | **4.8** (Reject) | Rejected for modest improvements and limited novelty; this paper's improvements are substantially larger. |
| "Active Test Time Prompt Learning in VLMs" | pdzHpQbGrn.md | **2.5** (Withdrawn) | Much weaker paper, withdrawn—useful only as low anchor. |

**Assessment against anchors:** This paper's empirical improvements (5–8%) are meaningfully larger than eXrUdcxfCw's sub-0.5% gains, clearly placing it above the 4.8 anchor. However, it falls short of the 7.0 Spotlight anchor (9w3iw8wDuE) due to: (1) the invalid theoretical derivation that undermines the claimed principled motivation, (2) the incorrect hardness-aware claim, and (3) the partially-confounded learning-rate ablation. The paper is closest to the 5.25–5.67 band: practical and empirically meaningful, but with substantive methodological gaps that prevent confident acceptance.

**Originality:** Moderate. Logit-space losses are known; combining positive and negative logit learning in TTA is relatively novel. The specific application to stable high-LR adaptation is a fresh angle.
**Importance:** Moderate-to-high. TTA under pseudo-label noise is practically important; the batch-size-1 setting is particularly underserved.
**Claims well-supported:** Partially. The empirical claims are well-supported; the theoretical claims contain errors.
**Soundness of experiments:** Adequate for the proposed scope, but missing the key LR-control ablation and broader benchmark evaluation.
**Clarity of writing:** Generally clear, but with the derivation flaw and incorrect hardness claim needing correction.
**Value to the community:** The practical contributions (Figure 4 insight, collapse avoidance, batch-size-1 robustness) have genuine value.

**Final Score: 5.0** — The paper is a borderline case. Strong empirical contributions with practically important results, but the theoretical framing requires correction in two specific places (the sparse-CL derivation and the hardness-aware claim), and the missing LR-isolation ablation is a real gap. The paper is better than the rejected anchors at 4.8 but not clearly above the 5.25–5.67 band. Score: 5.0 (marginal reject; revision recommended).

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>