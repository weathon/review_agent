Now let me read the calibration papers to establish score anchors.Now I have enough information to write the final consolidated review. Let me synthesize everything.

---

## Summary

The paper proposes two compute-efficient machine unlearning methods—*Unlearning via Activations* and *Unlearning via Examples*—built on the Discrete Key-Value Bottleneck (DKVB) architecture. By masking the key-value pairs most activated by the forget class, the method achieves complete output-level class forgetting without any gradient-based retraining. Experiments span CIFAR-10, CIFAR-100, LACUNA-100, and ImageNet-1k with two backbone families, showing competitive forget/retain accuracy trade-offs alongside ≥20× FLOP savings over SCRUB and other baselines.

---

## Strengths

- **Elegant core idea with clear motivation:** The insight that architectural sparsity in DKVB localizes class knowledge into disjoint key-value pairs, enabling trivially cheap forgetting, is well-articulated and intellectually satisfying. The connection between localized sparse representations and unlearning locality is clearly developed (Sec. 1, Sec. 4).

- **Strong empirical breadth:** Results across 4 datasets and 2 pre-trained backbone families (CLIP ViT-B/32 and ImageNet ResNet-50) consistently show complete output-level forget-class erasure with <0.5% average retain accuracy change. This goes well beyond the prior version of this work (TLBPjECC5D), which only covered 3 datasets and one backbone.

- **Compelling and concrete compute efficiency:** Table 2 provides a detailed FLOP breakdown for both forward and backward passes across all methods and datasets. The zero backward-FLOP cost of the proposed methods is a genuine practical advantage. The FLOP savings are orders of magnitude, not marginal.

- **Thorough hyperparameter analysis:** Figures 2 and 3 plot the full retain/forget accuracy frontier across a sweep of N_a and N_e values, providing practitioners meaningful guidance rather than a single operating point.

- **Honest limitations section:** Section 6 explicitly acknowledges dependence on pre-trained encoders, class-level-only evaluation, and the frozen-backbone caveat. Section 2 also candidly frames the method within the "weak unlearning" taxonomy.

---

## Weaknesses

### Fatal
*(None. The paper makes a real and honest contribution.)*

### Major

- **Architecture confound undermines comparative claims.** The proposed methods run on Backbone + DKVB; all baselines (SCRUB, finetuning, retraining, NegGrad+) run on Backbone + Linear Layer. Section 5.2.1 acknowledges this but frames it as a feature: "we compare…against several baseline methods, which are optimized for models without such a bottleneck." However, the abstract still claims the method "performs as well as, if not better than SCRUB," which is architecturally confounded—better retain-accuracy preservation may reflect DKVB's sparser class separation, not a superior unlearning *algorithm*. The legitimate contribution is that *DKVB as an architecture* enables dramatically cheaper class suppression; the paper should frame the comparison accordingly rather than positioning it as an apples-to-apples unlearning algorithm comparison. Applying SCRUB or finetuning directly to the DKVB model would cleanly disentangle the two effects.

- **Incomplete unlearning: frozen backbone retains forget-class information.** The paper acknowledges in Sec. 2 that "the pre-trained backbone might retain some information about the forget set" and that the method "affect[s] intermediate rather than final model activations." Yet the abstract and conclusion still use phrases like "erasing knowledge" and "highly effective means of unlearning the forget set." Since the entire encoder is frozen and pre-trained (in some cases on data that includes the forget class), a linear probe on frozen encoder features would likely still recover the forget class. This limits the privacy guarantee substantially. The method is best characterized as output-level class suppression / weak unlearning, not full knowledge erasure—the claims should reflect this throughout, not only in the limitations section.

- **MIA evaluation absent from main paper.** Section 5.1 notes that "we refer to Appendix A.7 for further discussion on MIAs." Given that the frozen backbone almost certainly retains forget-class representations, Membership Inference Attack results are critical to evaluating privacy claims, not merely a supplement. The discussion in the appendix (described as "discussion" rather than "experiments") does not substitute for empirical MIA results. For a paper motivated by GDPR and right-to-be-forgotten, this omission is significant.

### Minor

- **Forget class selection may inflate results.** The protocol selects "the class that is best learned by the respective DKVB models" as the forget class (Sec. 5.1), arguing this maximizes challenge. However, the best-learned class in DKVB may also be the most cleanly separated in key-value space, making it the *easiest* to mask without collateral damage. Averaging over multiple forget classes (or reporting variance across classes) would more robustly support the general claim.

- **No principled hyperparameter selection for N_a and N_e.** The paper sweeps these values empirically. In a real deployment, forget-set test accuracy is not available; there is no described mechanism to select the cutoff without circular access to held-out forget-set labels. At minimum, a heuristic based on activation frequency statistics would be needed for practical use.

- **"Zero-shot" label is imprecise.** Section 4 uses "zero-shot unlearning" to mean the method can operate with a proxy distribution instead of the exact training forget data. Both variants still require a forward pass over forget examples or cached activations; neither is zero-shot in the usual sense of requiring no task-specific data. This is minor but should be corrected to avoid misleading readers.

- **Stopping criterion for baselines underspecified.** Section 5.2.1 states baselines are stopped "when the forget set is completely unlearned or the forget set test accuracy has converged with minimal damage." Since compute and retain-accuracy both depend strongly on stopping point, a more objective criterion (e.g., fixed epoch count, fixed forget-set threshold) would allow cleaner comparison.

### Trivial

- Results are reported as relative changes without absolute base accuracies for DKVB vs. Linear in the main table (Table 1), making it harder to assess whether the architecture itself incurs a performance baseline cost.

---

## Nice-to-Haves

- **Apply SCRUB/finetuning directly to the DKVB model.** This single experiment would cleanly separate "DKVB as an architecture enables cheap forgetting" from "the masking algorithm is superior to gradient-based alternatives on the same architecture." This would substantially strengthen or clarify the paper's comparative claims.

- **Linear probe on frozen encoder post-unlearning.** Train a new probe head on frozen encoder features after the proposed unlearning and report its accuracy on the forget class. This would directly quantify how much forget-class information is retained in the backbone, providing honest upper-bound information for the privacy analysis.

- **Quantify key-value overlap between forget and retain classes.** A heatmap or overlap coefficient showing which key-value pairs are exclusively vs. jointly activated would explain *why* the method works and predict conditions where it might fail (e.g., fine-grained datasets with visually similar classes).

- **Brief experiment on instance-level unlearning.** Even a small pilot (e.g., unlearn a 10% subset of one class) would demonstrate whether the method generalizes beyond class-level settings, addressing the most practical privacy use case.

- **Discuss LLM / transformer applicability.** The introduction motivates DKVB in the context of LLMs, but no detail is given on how DKVB integrates into transformer architectures. A brief conceptual analysis would validate or scope the claim.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic #1 (claim that the paper fails to demonstrate machine unlearning "in any meaningful sense"):** Overstated. The paper honestly acknowledges in Sec. 2 that its method approaches "weak unlearning" and qualifies the frozen-backbone limitation in Sec. 6. The paper does contribute a meaningful, practically useful form of class suppression with genuine efficiency advantages. The overclaiming framing is a real but correctable issue, not grounds for dismissal as "not a paper." This criticism is subsumed into the Major weakness above at appropriate severity.

- **Harsh Critic #3 (compute efficiency claim "confounded" by omitting upfront DKVB training cost):** While the observation that the paper omits the amortized cost of DKVB training is technically valid, for the purpose of evaluating unlearning methods the relevant comparison *is* the marginal cost of the unlearning step. Whether DKVB or a linear head is the right architecture for a deployment is a separate design question. The FLOP comparison is honest and labeled correctly.

- **Harsh Critic (architecture as "not comparable"—baseline comparison should be removed as unfair):** Per the rules, "unfair comparison with other methods if the asymmetry favors the baseline" should be removed. Here, all baselines use the stronger (standard) architecture, so this asymmetry does NOT favor the proposed method—if anything it makes the comparison conservative for the authors. The comparison is kept (and acknowledged in the paper). The criticism about confounding is kept but softened.

- **Human Finder Weakness #5 / Spark "trivial novelty" claim:** The observation that DKVB for class unlearning is "quite literally the inverse" of class-incremental learning is worth noting, but applying an existing architecture to a new problem in a principled way with substantial experiments and compute analysis does constitute a real contribution. This is not removed from consideration but should not be listed as a major weakness.

---

## Novel Insights

The most important insight across all reviewers is that the paper's actual contribution and its framing are misaligned: what is demonstrated is that **DKVB as an architecture** enables near-zero-cost output-level class suppression, which is a legitimate and useful systems result; but the paper frames this as a superior *unlearning algorithm* relative to SCRUB via a confounded comparison, and simultaneously overclaims privacy-relevant "knowledge erasure" while acknowledging the frozen encoder retains forget information. Re-framing the paper around architecture-enabled class suppression with explicit weak-unlearning scope, and adding a linear-probe or MIA experiment to honestly bound the privacy guarantee, would make this a sharper and more credible contribution.

---

## Suggestions

1. **Re-frame abstract and conclusion** to explicitly scope claims to output-level class suppression / weak unlearning. Replace "erasing knowledge" with "suppressing class predictions" or "output-level class unlearning."
2. **Apply at least one gradient-based baseline (e.g., SCRUB) to the DKVB model** to decouple architecture effect from algorithm effect.
3. **Add MIA experimental results to the main paper** (not just appendix discussion), even for one dataset, to provide an honest privacy bound.
4. **Report accuracy variance across multiple forget-class choices** to demonstrate robustness beyond the best-learned class.
5. **Provide a practical heuristic for setting N_a or N_e** without access to forget-set test labels—e.g., a threshold based on activation frequency distribution.

---

## Score and Decision

**Calibration:**

- **TLBPjECC5D.md** ("Unlearning via Sparse Representations"): Human scores 5, 5, 6, 5 → ~5.25 average, **Rejected**. This is almost certainly a prior version of this paper, missing ImageNet, FLOP comparison, and ResNet experiments. The current submission addresses the main empirical gaps that drew scores below 6.

- **pUOesbrlw4.md** ("Deep Unlearning: Fast and Efficient Training-free Approach"): Scores 5, 5, 3, 8 → ~5.25 average, **Rejected**. Similar training-free class unlearning, similar limitation around class-level only evaluation. The outlier score of 8 reflects the one reviewer who found the method compelling; most found the evaluation incomplete.

- **NGF1wDDBMm.md**: Scores 5, 6, 6, 6 → ~5.75 average, comparable work on evaluation metrics for unlearning. That paper has stronger theoretical grounding; the present paper is more empirical.

**Assessment:** The current paper is meaningfully stronger than the prior TLBPjECC5D submission: the addition of ImageNet-1k and the FLOP table directly address the two most cited weaknesses in human reviews. However, the core confound (DKVB vs. Linear baseline comparison), weak unlearning concern with frozen encoder, and absent MIA experiments remain. These are not fatal—the contribution is real—but they prevent this from clearing the acceptance bar cleanly. The paper is at the borderline: incrementally above the prior rejected version but still not compelling for acceptance without revision.

**Score: 5.0** — marginally below acceptance threshold, consistent with TLBPjECC5D and pUOesbrlw4 human calibration. The addition of compute experiments and ImageNet nudges slightly upward from the prior version, but the fundamental framing and comparison issues are unresolved.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>