Now I have all the information I need. Let me write the final consolidated review.

## Summary

TrojanTO proposes the first action-level post-training backdoor attack against trajectory optimization (TO) models in offline RL. The paper first empirically demonstrates that reward manipulation—the dominant attack vector in traditional RL backdoors—is ineffective for TO models due to their sequence modeling nature, motivating a post-training paradigm. The attack combines trajectory filtering, batch poisoning (poisoning a single transition per batch for trigger consistency), and alternating training (bi-level co-optimization of trigger and model parameters). Experiments across 6 D4RL environments and 3 TO model architectures (DT, GDT, DC) show TrojanTO achieves an average CP of 0.701 with only 0.3% poisoning rate, substantially outperforming baselines Baffle and IMC.

## Strengths

- **Identifies a genuine and underexplored threat vector**: The paper is the first to systematically study action-level backdoors against TO models, which are architecturally distinct from traditional RL agents and increasingly deployed in practice. This fills a real gap in the literature.

- **Useful negative result on reward manipulation**: Section 4.3 and Figure 1 provide clear evidence that varying reward signals during backdoor training has negligible effect on ASR and BTP for TO models. This is a meaningful finding that distinguishes TO models from traditional RL agents and justifies the post-training attack paradigm.

- **Comprehensive evaluation scope**: Table 4 evaluates across 6 D4RL environments, 3 TO model architectures (DT, GDT, DC), and 3 target action types, providing broad coverage. The ablation study (Table 5) validates each component: removing batch poisoning drops ASR from 0.719 to 0.528, removing alternating training drops it to 0.507.

- **Well-designed batch poisoning strategy**: Poisoning only a single transition per batch (rather than the entire sequence) addresses a real distribution shift problem in transformer-based models, and the ablation confirms its importance for both ASR and BTP.

- **Honest characterization of persistent backdoor limitations**: Table 6 and the accompanying text clearly state that the backdoor duration is bounded by the TO model's context window (~20 steps), providing concrete limits rather than overclaiming.

## Weaknesses

### Fatal
None.

### Major

- **The ASR threshold ε is not specified in the main text, making the primary metric uninterpretable**: Equation 2 defines ASR such that an attack succeeds only if *every component* of the output action is within ε of the target action. For continuous action spaces with 6–17 dimensions in [-1,1], the value of ε determines whether the metric is meaningful or trivially loose. This value is not stated in the main text near the metric definition. While it may appear in the appendix, a metric's defining parameter should be specified where the metric is introduced. Without knowing ε, the reader cannot assess whether the reported ASR values (e.g., 0.719 average) reflect precise action manipulation or loose proximity.

- **Headline ASR is inflated by inclusion of the easy boundary target action '1' in the average**: Table 1 shows that the boundary target '1' yields near-perfect ASR (0.993–1.000 across environments), while interior targets like 'fixed random' (0.243–0.420) and 'arithmetic' (0.413–0.513) are dramatically lower. Table 4's headline ASR of 0.719 averages across all three, but this average is dominated by the trivially easy boundary target. The paper acknowledges target action sensitivity in Section 4.1 and provides per-target results in Appendix Table 24, but the main results table obscures the substantial performance gap. A reader examining only Table 4 would overestimate the attack's effectiveness on the more realistic interior targets. This matters because the paper itself argues that evaluating across diverse target actions is essential.

### Minor

- **Tension in the threat model regarding data access**: Section 3.3 states the adversary aims to implant a backdoor "without access to the original training dataset," but the method (Section 5) clearly requires trajectory data for post-training fine-tuning, and the 0.3% poisoning rate is defined relative to a dataset. If the adversary uses publicly available data (e.g., D4RL), the claim is technically consistent but misleading—D4RL data *is* the training data for these models. The paper should clarify whether the adversary requires the exact original training data or any compatible dataset, and how this affects the poisoning rate calculation.

- **Mathematical inconsistency between Eq. 6 and Eq. 7**: Equation 6 defines the overall objective as L = L_p + λL_c, while Equation 7's inner optimization for model parameters uses λL_p + (1−λ)L_c. With λ ∈ [0,1], these give different weightings (e.g., with λ=0.5, Eq. 6 gives L_p + 0.5L_c vs. Eq. 7 gives 0.5L_p + 0.5L_c). The paper does not explain this discrepancy. While Eq. 7 is a bi-level reformulation where different weightings may be intentional, the inconsistency should be explicitly addressed.

- **Trajectory filtering uses sequence length as a proxy for quality without justification**: Section 5.1 assumes "longer trajectories are more representative of successful behavior," but this heuristic is domain-dependent—short trajectories can be high-quality in sparse-reward settings, and long trajectories can be suboptimal. The threshold ϵ = 20 is stated without justification or sensitivity analysis.

- **Defense evaluation is thin for a security paper**: Section 6.5 devotes only a few sentences to defense evaluation, deferring all details to the appendix. The claim that fine-tuning is "the most effective defense" lacks characterization of its cost (how much clean data, what performance degradation during fine-tuning), which directly impacts the practical significance of the attack threat.

- **Conclusion overclaims that "trigger design" is the core**: The paper shows reward manipulation is ineffective and trigger design matters, but Table 1 shows target action selection has equally dramatic impact on ASR. Target action selection is not a trigger design matter. The conclusion should acknowledge both factors.

### Trivial
None.

## Nice-to-Haves

- Sensitivity analysis over the ε threshold would strengthen confidence in the ASR metric.
- Per-target-action ASR/BTP breakdowns in the main text (not just appendix) would improve transparency.
- Qualitative visualization of triggered vs. clean trajectories would make the attack's behavioral effect concrete.
- Characterization of fine-tuning defense cost (data, compute, performance trade-off) would better contextualize the practical threat.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"The comparison of 0.3% vs 10% poisoning rate is misleading because they refer to fundamentally different processes"**: The paper explicitly categorizes attacks by intervention stage (pre-training vs. post-training) in Section 3.3 and Section 2. Both rates represent the fraction of data the adversary needs to manipulate; the different processes are the point of the comparison, not a hidden confound. Removed because the paper is transparent about this distinction.

- **"BTP variance and confidence intervals not reported"**: The paper does report standard deviations for persistent backdoor results (Table 6, e.g., 0.847±0.012). Requesting confidence intervals for all metrics is a nice-to-have, not a substantive weakness, especially given that CP is computed per-run.

- **"Trigger dimension generalization leap from two environments"**: The paper tests dimensions on Half and Walk and fixes (1,2,3) for all experiments. This is a valid concern about generalization but is acknowledged and addressed further in Appendix F. It's minor rather than major.

- **"Convergence analysis of bi-level optimization missing"**: This is a standard request for any heuristic optimization but not expected in empirical security papers. Moved to nice-to-have.

- **"Missing related works"**: Per instructions, removed as I cannot verify existence of allegedly missing works.

## Novel Insights

The paper reveals a fundamental asymmetry in TO model security: the dominant attack vector for traditional RL (reward manipulation) is rendered ineffective by TO models' sequence modeling objective, yet this very architecture creates a new vulnerability—post-training backdoor injection that exploits the model's tendency to fit sequential patterns. The finding that boundary target actions are trivially achievable while interior targets are substantially harder suggests that the backdoor threat in continuous action spaces is highly non-uniform, and that the practical severity of such attacks depends critically on whether the adversary's goal aligns with boundary actions.

## Suggestions

- Report the ε value explicitly in Section 3.4 alongside the ASR definition, and include a sensitivity analysis showing how ASR changes across reasonable ε values.
- Present per-target-action results in the main text (at minimum as a supplementary table) rather than relegating them to the appendix, since the paper itself demonstrates target action choice swings ASR by 10×.
- Clarify the threat model: state explicitly whether the adversary requires the exact original training data or any compatible dataset, and discuss how the data source affects the poisoning rate interpretation.
- Reconcile or explain the difference between the weighting in Eq. 6 and Eq. 7.

## Score and Decision

**Calibration anchors used:**

1. **Z3SH1xlFs6** (avg 6.5, Accept Poster) — "Reward-Free Backdoor Attacks in RL via untrusted simulators" — similar topic (RL backdoor attacks, novel threat model), but that paper had formal proofs and real robot hardware experiments. TrojanTO has weaker formal grounding but comparable empirical scope.

2. **YM23GVqQqj** (avg 3.0, Reject) — "Stealthy Backdoor Attack in RL via Bi-level Optimization" — directly overlapping topic (RL backdoor via bi-level optimization) but rejected for outdated baselines and minimal contributions. TrojanTO is substantially stronger: it targets a genuinely new setting (TO models), provides useful negative results, and has much more comprehensive evaluation.

3. **QQdn8nNqgi** (avg 3.5, Reject) — "Clean-Action Backdoor Attacks on VLA Models" — similar niche (backdoor on new model class), rejected for limited architecture coverage and missing defense evaluation. TrojanTO is stronger in evaluation breadth but shares the thin defense evaluation concern.

4. **qqWPNiig1Q** (avg 4.67, Reject) — "BOBA batch-oriented backdoor attack" — similar pattern (novel attack vector, questionable threat model practicality, missing defense evaluation). TrojanTO has similar weaknesses but stronger empirical breadth.

5. **6R42MRRs50** (avg 4.0, Reject) — "Angel or Demon: backdoor threats in DRL" — large empirical study but overgeneralized conclusions. TrojanTO is comparable in empirical scope but has clearer novel contributions.

6. **PluWBC86I4** (avg 1.5, Reject) — "GCBD cross-attack backdoor detector" — low anchor, fundamentally flawed evaluation. TrojanTO is far above this.

TrojanTO is clearly above the rejected papers at 3.0–4.67 (it has a genuine novel contribution, extensive evaluation, and useful insights) but below the accepted paper at 6.5 (which had formal proofs and real hardware evaluation). The unspecified ε and the inflated headline ASR from averaging are the main factors keeping it from scoring higher. The paper sits in the borderline range, closer to the medium-scoring anchors. The core contribution is real but the evaluation presentation has substantive gaps that weaken confidence in the headline claims.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>