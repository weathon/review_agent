Now let me read the calibration papers identified by the Human Finder reviewer.Now let me check the calibration papers further and the SAR*/SAR† distinction in the paper:Now I have enough information to write the consolidated final review. Let me synthesize all the inputs.

---

## Summary

The paper proposes two logit-space loss functions for pseudo-label-based test-time adaptation (TTA): **sparse cross-logit (sparse-CL)**, which replaces cross-entropy with a direct linear loss on logits to achieve a constant L1 gradient norm, enabling stable training at high learning rates; and **k-hardness negative learning (k-NL)**, which penalizes a selected subset of hard negative classes to sharpen decision boundaries. These are plugged into the SAR framework and evaluated on ImageNet-C across three settings (normal batch, imbalanced, and batch size = 1), achieving improvements of ~6–8% over SAR on average.

---

## Strengths

- **Consistent and substantial empirical gains**: Across all 15 corruption types and all three evaluation settings, the method improves meaningfully over SAR (sparse-CL alone: +6.4% normal, +3.8% imbalanced, +6.7% small-batch; further gains with k-NL). Collapse cases (snow, frost) that cause entropy minimization to fail are clearly resolved.
- **Simple, drop-in design**: sparse-CL and k-NL are computationally cheap, straightforward to implement, and plug into an existing backbone (SAR) with minimal changes—a practically useful property.
- **Motivated gradient analysis**: The observation that cross-entropy's gradient norm for the pseudo-class vanishes as confidence increases (Eq. 6: `L_grad^CE = 2(1 − p_k)`) is a valid and insightful structural critique of CE in the pseudo-label setting where we want to keep pushing confident predictions.
- **Stable-gradient intuition empirically supported**: Figure 3 does show that the proposed loss yields a much flatter L1 gradient norm profile across batches compared to CE and EM on both corruption types shown—consistent with the claimed stability benefit.
- **Focus on challenging TTA regimes**: Evaluating at batch size 1 and under class imbalance is valuable, as most TTA work focuses on larger, balanced batches.

---

## Weaknesses

### Fatal
None. The empirical results are real and consistent; no core result is falsified.

### Major

- **The derivation of sparse-CL rests on an unjustified approximation.** The critical step (Eq. 7→8) replaces `p_i = exp(h_i) / Σ_j exp(h_j)` with `p_i ≈ exp(h_i)`, silently discarding the softmax normalization. This approximation holds only when the denominator ≈ 1, i.e., when all logits are near zero—a condition that the paper itself undermines by advocating high learning rates, which amplify logit magnitudes. The paper provides no analysis of when this approximation breaks down, no error bound, and no discussion of how large logit values affect it. The resulting loss `−Σ ŷ_i h_i` is perfectly valid as a *direct* logit-margin objective, but the CE-derivation narrative is misleading. Because this derivation is the stated theoretical foundation of the entire paper, the gap is significant even though the loss itself works in practice.

- **Core scientific claims about memorization and confirmation bias are asserted but not measured.** The abstract and introduction prominently claim the method "mitigates the memorization effect" and "confirmation bias." However, neither of these phenomena is directly measured anywhere in the paper. The conclusion itself acknowledges this, saying only "hypothesizing that this rapid updating helps the model...effectively combat both memorization and confirmation bias problems." There is no trajectory analysis of pseudo-label accuracy over time, no error accumulation measurement, and no direct comparison of noise rates between methods. The paper shows an accuracy improvement on ImageNet-C but does not demonstrate that the *mechanism* it proposes—fast stable learning preventing memorization—is what causes it.

- **No ablation studies on the critical hyperparameters k, s, and α.** The k-NL loss introduces three key hyperparameters (k: number of complementary labels; s: number of classes to skip; α: balance weight). The skip parameter s is particularly central to the paper's claim that k-NL is "hardness-aware" and avoids noisy nearby classes—yet there is no experiment varying s to demonstrate that skipping actually helps vs. hurts, nor any guidance on choosing it. Similarly, no sensitivity analysis is provided for k or α. This makes the practical reliability of the method unclear and leaves the specific design choices unjustified.

- **Evaluation is confined to a single benchmark and severity level.** All main results are on ImageNet-C, severity level 5 only. No evaluation on CIFAR-10/100-C, ImageNet-R, ImageNet-A, or any non-synthetic distribution shift is provided. Given the paper's broad claims about pseudo-label TTA generally, this narrow evaluation scope limits confidence that the proposed losses generalize beyond ImageNet-C.

### Minor

- **SAR* vs. SAR† distinction is not explained in the main text.** Tables 1–3 list both "SAR†" and "SAR* + loss" rows with different numbers, but the main paper does not define what SAR* is or how it differs from SAR†. If SAR* introduces implementation differences beyond just swapping the loss, part of the gain may be attributable to those differences rather than the loss. This needs to be explicitly stated.

- **The "hardness-aware" label for k-NL is overstated.** The paper contrasts k-NL with NL+ by arguing k-NL is "hardness-aware" (Section 3.3), but by Eq. 15, the gradient of k-NL on selected classes is a constant ±1 regardless of class probability. Hardness enters only through *which* classes are selected (the top-k after skipping s), not through gradient magnitude. The comparison with NL+ (which has probability-proportional gradients) oversells k-NL's hardness sensitivity.

- **The gradient stability argument does not extend to parameter space.** The paper argues from zero L1 gradient norm variance in logit space to stable optimization dynamics. This argument holds at the logit level but does not formally extend to the gradients of the actual trainable parameters (BN affine weights in this case), which pass through deep nonlinear layers. Figure 3 provides empirical support for logit-space stability; an analogous figure for parameter gradients would strengthen the claim.

- **Figure 4's "absolute true labels" visualization requires ground-truth labels.** The paper explains that the second row of Figure 4 counts "true label samples minus false (noise) label samples." This computation requires oracle ground-truth access, which the method does not have at test time. This should be clearly stated in the figure caption; readers may otherwise interpret it as an unsupervised diagnostic.

### Trivial

- Table captions appear before Table 3 in the paper's presentation (Table 1 and Table 2 captions appear out of order relative to the table data), which could confuse readers.

---

## Nice-to-Haves

- **Learning rate sensitivity study**: A grid of (LR × loss function) performance would cleanly demonstrate that sparse-CL uniquely enables high-LR stable training rather than simply being a better loss at any LR.
- **Pseudo-label accuracy tracking over time**: Plotting pseudo-label accuracy vs. batch number for different losses would directly test the memorization/bias hypothesis rather than relying on the indirect accuracy metric.
- **Evaluation on natural domain shifts** (e.g., ImageNet-R, VisDA-C): These would test whether the method generalizes beyond synthetic corruptions.
- **Provide a direct derivation of sparse-CL** as a gradient-variance-minimizing logit objective, bypassing the flawed CE approximation. This would make the theoretical contribution cleaner and more honest.
- **Comparison with gradient clipping / learning rate scheduling as baselines**: These simple stabilizers for high-LR training would help isolate whether the specific loss formulation is necessary.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Missing related works (CoLA, CoTTA, RMT, etc.)**: Removed per hard rule — cannot verify existence of external works; including such criticisms risks fabricating references.
- **Reproducibility concerns about hyperparameter reporting**: Removed — falls under nitpick about trivial implementation details.
- **Computational efficiency / runtime reporting**: Removed — not a standard requirement for a loss-function contribution paper; the method is explicitly argued to be lightweight.
- **Novelty criticism ("just A+B")**: The gradient analysis and logit-space reframing constitute a genuine novel framing, not merely combining off-the-shelf components. The k-NL is incremental but valid. Removed as too dismissive.
- **"Unfair comparison" concern about SAR at high LR**: The whole point is to show that SAR collapses under high LR while the method does not. Per hard rule, asymmetry that disfavors the baseline (scaled-SAR) and favors the author's method is the intended point and is not unfair.
- **Harsh Critic concern that CE "pays more attention to hard negatives" is misleading in the pseudo-label context**: The critic argues this only matters if the pseudo-label is wrong. This is partially valid but the paper's general point—that CE's gradient on the pseudo-class vanishes when confidence is high—is mathematically correct and is a real design tension under pseudo-label reuse. Removed as a strawman.
- **Logit unboundedness concern (Spark reviewer)**: Logit growth is a concern in principle but is managed by the SAR framework's BN adaptation scope, which naturally constrains scale. Not enough evidence it causes problems; too speculative to include.

---

## Novel Insights

The most genuinely insightful observation from the combined reviews is the structural mismatch between what cross-entropy's gradient does under pseudo-label TTA and what a practitioner actually wants: when the model becomes confident in a (potentially correct) pseudo-label, CE stops reinforcing it, while simultaneously gradient mass shifts toward plausible competing classes. The paper's move to a constant-gradient logit objective is a practically motivated and empirically effective response to this mismatch, even if the CE-approximation derivation used to motivate it is formally weak. The collateral observation (Fig. 4) that the difference between positive and negative logit magnitudes is a useful signal for discriminating true from noisy pseudo-labels—analogous to how SAR uses entropy magnitude—points to a promising direction for online pseudo-label filtering.

---

## Suggestions

1. **Restate the sparse-CL motivation directly**: Motivate `L_sparse-CL = −Σ ŷ_i h_i` as a constant-gradient logit-margin objective *without* the p_i ≈ exp(h_i) step. The loss is well-motivated independently; the approximation derivation only weakens the paper.
2. **Add ablations on k, s, and α**: At minimum, a table showing accuracy at several values of each would establish robustness to hyperparameter choice.
3. **Define SAR* explicitly** in the main text with a clear statement of what differs from the original SAR (SAR†) implementation.
4. **Track pseudo-label accuracy over adaptation steps**: Even a single experiment showing whether sparse-CL maintains higher pseudo-label accuracy over time than CE/EM would provide direct evidence for the memorization narrative.
5. **Test on at least one additional benchmark** beyond ImageNet-C to support the generality claim.

---

## Score and Decision

**Calibration:**

| Paper | Decision | Scores |
|---|---|---|
| PROGRAM (x5LvBK43wg) — TTA with pseudo-labels, multiple benchmarks, graph-based PL, proper ablations | Accept (poster) | 5, 6, 6, 6, 8 |
| PASLE (3Z2flzXzBY) — selective label enhancement for TTA, multiple benchmarks, theory | Accept (poster) | 6, 6, 6, 8, 6 |
| CTA prototypes (eXrUdcxfCw) — continual TTA with EMA prototypes, limited contribution | Reject | 3, 5, 5, 6, 5 |

The paper under review sits **below PROGRAM and PASLE** on scientific rigor: it evaluates on a single benchmark at a single severity, has no ablation studies, and its central theoretical narrative (CE approximation derivation; memorization/bias mitigated) is either flawed or unverified. Its empirical gains (6–8%) are real and substantial—stronger than the rejected CTA prototype paper's <0.5% gains—but the method's scope and justification are materially weaker than the accepted pseudo-label TTA papers.

**Assessment axes:**
- *Originality*: Moderate — the gradient-stability framing is interesting, but the resulting losses (logit-margin + complementary label penalty) are simple.
- *Importance of research question*: Good — pseudo-label noise in online TTA is a well-motivated problem.
- *Claims supported*: Weak — the CE derivation is flawed and the core memorization/bias claims are unverified.
- *Soundness of experiments*: Below standard — single benchmark, no ablations, no sensitivity analysis.
- *Clarity of writing*: Reasonable, but the SAR*/SAR† distinction is left undefined.
- *Value to community*: Moderate — the method is useful as a practical baseline loss, but the paper overpromises on theoretical understanding.

Overall, this paper is below the acceptance bar in its current form. The empirical improvements are genuine and the practical insight is real, but insufficient experimental rigor (no ablations, one benchmark) combined with a flawed central derivation and unverified core claims make it a weak submission by ICLR standards.

**Final score: 4.0**

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>