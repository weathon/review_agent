Now I have a thorough understanding of the paper. Let me compile the final review.

## Summary

TrojanTO proposes the first post-training, action-level backdoor attack against trajectory optimization (TO) models in offline RL. It identifies that reward manipulation—prior RL backdoor methods' primary attack vector—is ineffective for TO models, since they minimize reconstruction loss rather than maximize reward. The method uses three components: trajectory filtering (selecting high-quality trajectories), batch poisoning (poisoning one transition per batch for trigger consistency), and alternating training (jointly optimizing the trigger and model). Evaluated across six D4RL environments and three TO architectures (DT, GDT, DC), TrojanTO achieves average CP of 0.701 at 0.3% trajectory poisoning, substantially outperforming baselines Baffle (0.342 CP at 10% poisoning) and IMC (0.551 CP).

## Strengths

- **Identifies that reward manipulation is ineffective for TO model backdoors (Section 4.3, Figure 1).** This is a clear and valuable finding: varying the manipulated reward signal during backdoor training produces virtually no change in ASR or BTP, establishing that TO models' reconstruction-loss-based training objective fundamentally differs from Bellman-equation-based agents in their susceptibility to reward-based attacks. This insight motivates the entire approach and is well-supported empirically.

- **Systematic analysis of how target action type impacts ASR (Table 1).** The finding that boundary actions ('1', '-1') achieve near-perfect ASR while interior actions (e.g., '0' in Walk at 0.11) yield dramatically lower ASR is an important result for the community, revealing a key structural property of continuous-action backdoors that prior work overlooked.

- **Well-designed technical contributions validated by ablation (Table 5).** Each module contributes measurably: removing batch poisoning drops ASR from 0.719 to 0.528, removing alternating training drops it to 0.507, and trajectory filtering preserves BTP (0.914 vs 0.850 without it). The consensus poisoning strategy is a thoughtful solution to the teacher-forcing context mismatch specific to Transformer-based TO models.

- **Comprehensive evaluation across three TO architectures and six environments.** Testing on DT, GDT, and DC across locomotion, navigation, and manipulation tasks provides meaningful breadth and demonstrates the method is not tailored to a single architecture.

## Weaknesses

### Fatal
None.

### Major

- **The headline comparison emphasizing 0.3% vs 10% data budget is misleading because the threat models are incommensurable.** TrojanTO operates as a post-training attack where the adversary directly modifies the pretrained model (with fine-tuning access), while Baffle is a pre-training data poisoning method and IMC is adapted from image classification. The adversary in TrojanTO's threat model has fundamentally greater capabilities—direct model modification—so comparing footprint sizes across these different threat models conflates method efficiency with adversary power. The paper does categorize these in Section 3.3, but the abstract and conclusion repeatedly present the 0.3% budget as evidence of "superior stealth and attack efficiency," which is an apples-to-oranges claim. The comparison is informative (showing prior methods fail on TO models) but should be framed as demonstrating that existing attacks are ineffective on TO models, not that TrojanTO is more efficient per unit of data. This overclaims what the evidence supports.

- **Some BTP values contradict the "negligible impact on benign performance" claim.** The abstract states the backdoor has "a low attack budget (0.3% of trajectories)" and the conclusion claims "minimal impact"; however, DC-Kit achieves BTP=0.455 (55% performance degradation), DC-Ant BTP=0.302 (70% degradation), and DT-Walk BTP=0.486 (51% degradation). The harmonic mean CP metric masks these cases where high ASR comes with severely degraded normal functionality. A backdoor that causes the model to lose half or more of its task performance is not "negligible" by any standard, even if the averages look reasonable. The paper should more honestly characterize when and why the attack fails to preserve BTP.

- **The "effectiveness across diverse attack objectives" claim in the abstract is weakened by the target-action-dependent results.** Table 1 shows interior actions yield ASR as low as 0.11 (type '0' on Walk), 0.24–0.43 (type 'fixed random'), and 0.41–0.51 (type 'arithmetic'). The main Table 4 averages across three target actions including '1' (a boundary action achieving near-1.0 ASR), which inflates the aggregate. The paper itself identifies this important variation but the abstract and Section 6.1 do not qualify the claim. For many practical adversarial objectives where fine-grained manipulation matters (e.g., steering in a specific direction), the attack's effectiveness is substantially weaker than the headline numbers suggest.

### Minor

- **The ε threshold in the ASR definition (Equation 2) is not specified in the main text.** Since ASR requires *all* action components to fall within ε of the target, the value of this threshold directly determines how stringent the success criterion is. A loose ε in a [-1,1] action space would make almost any perturbation "successful," while a tight one makes success harder. This parameter should appear alongside the metric definition for the evaluation to be fully interpretable.

- **Trigger dimensions fixed to (1,2,3) correspond to the most salient state features.** While the paper shows a systematic exploration in Table 2 and the choice is well-motivated by performance, it should be noted that the first three dimensions often encode the most physically meaningful state variables (e.g., position, velocity). This could make triggers placed there more detectable in practice.

### Trivial

- Zero variance cells in Table 6 (e.g., Hopp k=0: 0.922±0.000) could benefit from brief explanation, though this may result from deterministic evaluation or very small per-run variance.

## Nice-to-Haves

- A matched-threat-model baseline—e.g., giving Baffle-style data poisoning the same fine-tuning access as TrojanTO—would clarify how much of the performance gain comes from the method versus the stronger threat model.
- Per-target-action-type breakdowns in Table 4 (not just the averages) would let readers assess effectiveness across interior vs. boundary actions directly.
- Discussion of trigger detectability via statistical tests or input preprocessing would strengthen the security analysis for a paper with a security motivation.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Unfair baseline comparison (incommensurable threat models).** Retained in major weaknesses above but reframed: the comparison exists and is informative, but the *framing* of 0.3% vs 10% as demonstrating "superior stealth" is the problem, not the existence of the comparison itself. The paper correctly categorizes threat models in Section 3.3; the issue is the abstract/conclusion language. Per rules, the comparison is kept (not removed), since the asymmetry here favors the baseline (Baffle has weaker capabilities), not the author's method—the author's method actually has *stronger* capabilities, which means the comparison could be seen as unfair *against* the baselines, not against TrojanTO. However, the overclaiming about "efficiency" remains a valid issue.

- **"Not yet released" / reproducibility concerns about Baffle or IMC.** Removed per hard rules—cited models/baselines are assumed to exist.

- **Defense section delegated to appendix.** Removed per rules—the parser strips appendices from all papers; the original submission includes them.

- **Missing variances in Tables 4 and 5.** Tables 6–7 include ±variance, and the paper states results are averaged over three seeds. While variance reporting is good practice, this is a minor presentation choice, not a methodological flaw. Removed to trivial.

- **ε=20 trajectory length threshold is arbitrary.** This is a standard hyperparameter in filtering; the paper explains the motivation (longer trajectories are more representative). Removed as it's a minor hyperparameter choice, not a fundamental flaw.

- **Post-training adversary capability makes the attack trivial.** Removed—the paper clearly addresses this (Section 3.3): the adversary must implant a backdoor that preserves normal behavior, which is a non-trivial constraint beyond just "fine-tuning the model."

- **Batch poisoning trains on sparse contexts—robustness not analyzed.** The ablation study (w/o BP, Table 5) validates the design works; whether it generalizes to arbitrary trigger positions during evaluation is a reasonable concern but addressed in the design section and left to persistent attack experiments (Table 6). Weakened to nice-to-have.

- **"Standard deviation" / missing related works.** Removed per rules—missing related works cannot be verified, and formatting/notation issues are parser artifacts.

## Novel Insights

The identification that reward manipulation—a primary vector for RL backdoors for over five years—is fundamentally ineffective for TO models because their reconstruction-loss training objective is disconnected from the reward signal is a genuinely novel and practically important finding. This reframes the attack surface for this growing model class and should inform future security research. Similarly, the stark contrast between boundary and interior action ASR (near-1.0 vs. as low as 0.11) reveals a structural property of continuous-action backdoors that has been overlooked in prior discrete-action work, where all target actions are similarly easy to achieve.

## Suggestions

- Reframe the comparison with Baffle and IMC explicitly around threat model differences. The strength of TrojanTO is not that it achieves more with less data *per se*, but that post-training attacks are the practical vector for TO models (given training costs) and prior in-training methods fail. Make this the narrative.
- Report per-target-action-type results in the main table, not just the average. This transparently shows the boundary vs. interior action gap and lets readers judge for themselves.
- Qualify the "negligible impact on benign performance" claim to acknowledge cases where BTP drops below 0.5 (DC-Kit, DC-Ant, DT-Walk), and discuss what task/environment properties make the attack less stealthy.

## Calibration

**Anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Inception attack on DRL (bounded reward backdoor) | NALkteEo9Q.md | 5.0 | Most topically similar: DRL backdoor attack with bounded perturbations. TrojanTO has more comprehensive evaluation and a cleaner novel insight (reward manipulation ineffectiveness), but shares the issue of overclaiming relative to threat model. TrojanTO is somewhat stronger. |
| Multi-vehicle backdoor on offline RL for driving | em0gAL8fbK.md | 4.0 | Similar domain (offline RL backdoor) with unclear threat model assumptions. TrojanTO has better methodology and a clearer contribution. |
| Backdoor in ViTs (unfair baseline comparison) | vdHSMJpBya.md | 3.5 | Shares the issue of questionable baseline comparisons but TrojanTO's comparisons are more defensible (relevant domain baselines exist). |
| Certified defense against poisoning in offline RL | X2x2DuGIbx.md | 6.75 | Strong theoretical contribution in same domain (offline RL security). TrojanTO has weaker theoretical footing but a different type of contribution (attack). |
| Backdoor trigger inversion (supply-chain) | Tw9wemV6cb.md | 7.5 | Much cleaner claims and stronger validated contribution. TrojanTO falls short of this level due to overclaiming. |

TrojanTO has genuine contributions (first action-level TO backdoor, reward-manipulation finding, well-designed method), but the overclaiming about "superior stealth" from an incommensurable threat model comparison and "negligible impact" contradicted by some BTP values puts it below the clearly accepted papers (6.75+). It is somewhat stronger than the rejected DRL backdoor papers (4–5 range) due to its more systematic methodology, clear technical contribution, and important empirical finding about reward manipulation ineffectiveness. The paper would benefit significantly from honest qualification of its claims.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>