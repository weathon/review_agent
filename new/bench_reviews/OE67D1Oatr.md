Now I have all the information needed. Let me write the final consolidated review.

## Summary

Gradient Storm proposes an enhancement to the Sleeper Agent (SA) clean-label backdoor attack by distributing poison budget across R rounds of model retraining (claimed "expanded parameter space coverage") and running S independent optimization cycles with best-of-S selection. The paper's primary contribution is extending this framework to multi-trigger, multi-target attacks, demonstrating that 2–3 heterogeneous backdoor triggers (patches, blended patches, sinusoidal signals) can be injected simultaneously while maintaining high individual ASRs and benign accuracy.

## Strengths

- **Multi-trigger backdoor attacks are demonstrated as feasible.** Tables 5 and 6 show that 2–3 concurrent attacks with heterogeneous trigger types (patch, blended patch, sinusoidal) can coexist in a single poisoned model with per-trigger ASRs mostly above 90% and benign accuracy above ~89%. This is a non-trivial finding that extends the threat model beyond single-trigger attacks studied by prior work.

- **Strong single-trigger ASR.** Gradient Storm achieves 99.76% ASR on CIFAR-10 (Table 1) and 84.25% on GTSRB (Table 2), compared to the next-best SA at 89.73% and 58.19%, while maintaining comparable benign accuracy (~90% and ~94%). The improvement on GTSRB is particularly notable.

- **Broad defense evaluation.** Table 3 tests 8 defense mechanisms, providing a more comprehensive empirical picture than most attack papers. Several data-filtering defenses (Spectral Signatures, Activation Clustering, DeepKNN) are ineffective against GS, which is a useful data point for the community.

- **Cross-architecture transferability.** Table 4 shows that poisons crafted for ResNet18 transfer effectively to ResNet20 (97.06%), ResNet34 (99.8%), MobileNetV2 (95.9%), VGG11 (98.6%), and VGG16 (98.4%) ASR, validating the black-box threat model.

## Weaknesses

### Fatal
None.

### Major

- **Confounded comparison with Sleeper Agent due to best-of-S selection.** GS runs S=4 independent optimization cycles and selects the cycle with the highest cumulative ASR (Algorithm 1, line 22), while SA is run only once with 4 retraining periods (Section 4.1, line 190). The best-of-4 selection gives GS a substantial advantage—effectively equivalent to running SA four times and cherry-picking the best result. Without controlling for this by either (a) running SA 4 times with best-of-4 selection, or (b) reporting GS results for a single cycle, the headline improvement (99.76% vs. 89.73%) cannot be attributed to the proposed "expanded parameter space coverage" mechanism rather than to the selection procedure. This undermines the paper's central claim that GS is a fundamentally stronger attack.

- **Overclaimed robustness against defenses.** The abstract states GS demonstrates "robustness against eight different poisoning defense mechanisms," and the conclusion claims "strong resilience against a range of poisoning defense mechanisms." However, Table 3 shows that 3 of 8 defenses effectively neutralize the attack: ABL reduces ASR to 2.1%, DP-InstaHide to 6.47%, and Gradient Shaping to 8.9%. Claiming "robustness against eight" when nearly half the tested defenses reduce ASR below 10% is a direct misrepresentation of the experimental evidence. The abstract phrasing implies robustness against all eight, which the data contradicts.

- **No multi-trigger baseline comparison.** The paper's primary novelty is enabling multiple simultaneous backdoor triggers (Section 4.4), yet no comparison with any multi-trigger baseline is provided. The most natural baseline—sequentially applying SA for each trigger on the same dataset—is never evaluated. Without this, it is impossible to determine whether GS's specific mechanism (multi-round optimization with budget splitting) provides any advantage over simply applying an existing single-trigger attack multiple times. The multi-trigger results show feasibility but do not establish that the proposed method contributes anything beyond sequential application.

### Minor

- **Defense and transferability evaluations are limited to the single-trigger setting.** Tables 3 and 4 evaluate defense resistance and cross-architecture transferability only for single-trigger attacks, despite multi-trigger being the paper's main contribution. If multi-trigger attacks are less robust to defenses or less transferable, the practical threat model is narrower than claimed. This is a significant gap in evaluation scope but not a fatal flaw since single-trigger results already provide useful information.

- **No reported variance or statistical significance.** Source and target classes are "selected randomly" (Section 4.1) but no random seeds, standard deviations, or repeated runs are reported. Given that the improvement over SA is the paper's central empirical claim, reporting variance across multiple random class selections would strengthen confidence in the results.

- **Unclear differentiation from SA's existing multi-retraining approach.** SA already distributes optimization across 4 retraining periods (line 190). The paper claims GS disperses perturbations "over a wider region of the parameter space" (Section 3.2, lines 191–194), but this is never formalized—what constitutes a distinct "region" of parameter space, and how does GS's R=2 rounds differ substantively from SA's 4 retraining periods? Without a formal or empirical distinction, the claimed mechanism risks being a re-description of what SA already does.

### Trivial
None.

## Nice-to-Haves

- **Ablation on R and S.** The paper mentions an ablation study in Appendix A, but it would strengthen the main text to include results disentangling the contributions of R (multi-round budget splitting) and S (best-of-cycle selection) to the improvement over SA.

- **Scaling analysis beyond 3 triggers.** The paper stops at 3 concurrent attacks; testing with 5+ triggers would establish practical limits.

- **Analysis of trigger interference.** Some individual ASRs in the multi-trigger setting are notably lower (e.g., 78.8% ASR-2 in Table 6 row 4; 82.1% ASR-3 in Table 6 row 3). Understanding whether this is due to trigger type, class pair, or budget allocation would strengthen the paper.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Double the retraining steps" claim (Harsh Critic #1):** The critic claims GS runs "8 poison-craft-and-retrain steps per trigger" vs SA's 4, but this conflates total compute across all 4 cycles with per-cycle compute. Within the selected cycle, GS has only R=2 rounds of retraining, while SA has 4 retraining periods—SA actually uses more per-run steps. The real issue is the best-of-4 selection, not "double the retraining steps."

- **"Minimal modification" is vague (Harsh Critic, Abstract):** 500/50,000 = 1% poisoning rate is standard and reasonable for clean-label attacks; this is not a substantive weakness.

- **Sequential optimization doesn't match joint objective (Harsh Critic, Section 3.1):** The paper decomposes Equation 1 per-trigger (Equation 2) and solves sequentially with retraining between triggers. While the independence assumption isn't rigorously justified, the sequential retraining partially accounts for trigger interactions, and this is standard practice in bilevel optimization.

- **Missing ablations on R and S in main paper (Harsh Critic):** The paper explicitly references an ablation study in Appendix A. The appendix exists in the original submission but was stripped by the parser.

- **Defense results without discussing BA trade-offs (Harsh Critic):** Table 3 reports both ASR and BA for every defense. The data is available for readers to assess trade-offs; this doesn't require explicit discussion.

- **"SA already distributes poisons across multiple retraining checkpoints" (Harsh Critic, Section 3.2):** While SA does use multiple retraining periods, the specific budget-splitting approach in GS (dividing each trigger's budget across R rounds) is technically distinct. The weakness is better framed as "unclear differentiation" rather than "no differentiation."

## Novel Insights

The multi-trigger results in Tables 5 and 6 reveal an interesting asymmetry: sinusoidal signal triggers tend to achieve lower individual ASRs (78.8–90.7%) compared to patch-based triggers (95–100%) in multi-trigger settings, suggesting that trigger visual salience may affect interference in concurrent attacks. This pattern, if confirmed, would have practical implications for which trigger types are most effective for multi-trigger deployment.

## Suggestions

- Run SA with best-of-4 selection (4 independent runs, select the best) and report results alongside the current GS comparison to isolate the contribution of the multi-round budget splitting mechanism from the selection advantage.
- Add a sequential SA baseline for the multi-trigger setting: apply SA independently for each trigger on the same poisoned dataset, then evaluate all triggers simultaneously.
- Retract or substantially qualify the robustness claim: replace "robustness against eight" with specific language like "partial robustness—effective against 5 of 8 tested defenses" or similar.
- Evaluate at least one defense and one transferability experiment in the multi-trigger setting to validate the core contribution under relevant conditions.

## Score and Decision

**Calibration anchors compared:**

| Anchor | Path | Avg Score | Comparison |
|--------|------|-----------|------------|
| Influencer Backdoor (segmentation) | VmGRoNDQgJ | 7.50 | Clearly stronger: novel mechanism design, comprehensive experiments, fair comparisons. GS is below this. |
| Poisoned Forgery Face | 8iTpB4RNvP | 7.33 | Stronger: novel framework with well-supported claims. GS's confounded comparison puts it below. |
| Sharpness-Aware Poisoning (SAPA) | bxITGFPVWh | 5.83 | Comparable topic (retraining uncertainty). SAPA is more principled with comprehensive ablations; GS has more serious comparison issues. |
| Learnable Poisoning Selection | uDNP1q5aZq | 5.50 | Similar: extensive experiments but methodology questioned. GS's comparison fairness issue is somewhat more serious. |
| Transferable Availability Poisoning | XbLffB0T2z | 4.40 | Similar overclaiming pattern. GS has a stronger contribution (multi-trigger) but similar evaluation concerns. |
| FV-NeRV (unfair comparison) | hrXt6Fdl2P | 2.60 | GS is clearly above this — GS does compare against SA and has genuine multi-trigger novelty. |

Gradient Storm demonstrates a genuinely valuable finding (multi-trigger backdoor feasibility) but its central claims are undermined by a confounded comparison with SA (best-of-4 selection not controlled for), an overclaimed robustness narrative, and no multi-trigger baseline. These are substantive issues that a rebuttal cannot fully resolve without new experiments. The paper is below the SAPA anchor (5.83) due to weaker methodological rigor, and below the Learnable Selection anchor (5.50) due to the confounded comparison being more central to the paper's claims. It sits near the Transferable Availability Poisoning anchor (4.40), but GS's multi-trigger contribution is stronger than that paper's contribution, so it should be slightly above.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>