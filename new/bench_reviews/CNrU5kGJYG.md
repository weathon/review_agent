Now I have a good understanding of the paper, the reviewers' concerns, and calibration papers. Let me synthesize my final review.

Key calibration notes:
- RL backdoor attack papers with weak evaluations or overclaimed threat models tend to score 3-5 (em0gAL8fbK, NALkteEo9Q, TCpJXzMnnp)
- Papers with strong evaluations and reasonable claims (BadEdit, Poisoned Forgery Face) can score 6-8
- Papers with limited technical novelty but decent empirical results tend to be in the 5-6 range

Now let me assess the paper's actual strengths and weaknesses by verifying the critical claims.

Looking at the harsh reviewer's points:
1. **ASR metric and task-relevance**: The ε threshold is indeed not specified in the main text, which is a gap. The concern that high ASR with high BTP means the attack isn't actually harmful is substantive.
2. **Baseline fairness**: This is a valid concern - comparing against methods designed for different threat models.
3. **Threat model**: The adversary needing trajectories despite "no access to training data" is contradictory. The fine-tuning defense point is important.
4. **Fragility to design choices**: The trigger dimensions, values, and target-action sensitivity are real concerns.
5. **CP metric obscuring tradeoffs**: Valid but not fatal.

Let me verify from the paper whether ε is mentioned anywhere...

From the paper: "ASR is calculated as the proportion of successfully launched attacks within evaluation episodes. An attack is considered successful if, at a single triggered step within an episode, all components of the model's output action π˜([a]i, [s]i + δ, [Rˆ]i) are within a threshold ε of the corresponding components of a predefined target action a†δ."

The paper states ε but does not specify its value in the main text. It likely appears in an appendix. This is a genuine gap.

For the target-action issue - Table 1 shows clear sensitivity, and the main Table 4 averages over target actions. This is a valid concern.

For the adversarial dataset source - Section 3.3 says "the adversary aims to implant a backdoor into the pretrained TO model without access to the original training dataset" but Section 5.1 says "the filtered trajectory set Fτ is then exclusively utilized for both the backdoor training process." This is contradictory - where does the attacker get these trajectories if they have no training data?

For fine-tuning defense - it's mentioned in Section 6.5 but details are in appendix.

Let me synthesize the final review.</think>## Summary

TrojanTO proposes the first action-level backdoor attack against trajectory optimization (TO) models in offline RL. Motivated by the ineffectiveness of traditional reward-based backdoor attacks against sequence-model-based policies, the paper introduces a post-training attack combining trajectory filtering, batch poisoning, and alternating (trigger/model) training to implant backdoors at 0.3% trajectory poisoning rates. Experiments span six D4RL tasks and three TO architectures (DT, GDT, DC), demonstrating high attack success rates (avg ASR 0.719) while maintaining benign task performance (avg BTP 0.914).

## Strengths

- **Well-motivated problem with useful empirical characterization (Section 4):** The systematic study showing that reward manipulation is ineffective for TO models (Figure 1), while trigger design and target action selection are critical (Tables 1–3), provides genuine insight for the RL security community. The finding that boundary actions yield near-perfect ASR while interior actions are far harder to enforce is consequential and non-obvious.

- **Practically relevant post-training threat model:** Shifting from training-time attacks (which require control over the entire training pipeline) to a post-training paradigm is well-aligned with the reality of large-scale model deployment via model zoos or supply-chain scenarios, making the attack setting more realistic than prior RL backdoor work.

- **Low poisoning budget with competitive performance:** Achieving avg CP of 0.701 with 0.3% trajectory poisoning is a meaningful improvement over Baffle's 0.342 CP at 10% poisoning rate, demonstrating attack efficiency.

- **Comprehensive evaluation scope:** Testing across 6 environments, 3 TO architectures, multiple target actions, persistent attacks, trigger perturbations, and defenses provides a reasonable breadth of evidence. The ablation study (Table 5) cleanly demonstrates each component's contribution.

## Weaknesses

### Major:

- **Evaluation metric (ASR) does not establish meaningful threat: the ε threshold is unspecified and ASR may not correspond to harmful behavior.** Equation 2 defines attack success as the model's output being within threshold ε of a target action, but the value of ε is never reported in the main text. In high-dimensional continuous action spaces, the choice of ε drastically affects ASR; a large ε can make "success" trivially easy. Moreover, the central motivation—"manipulate a single action at a critical moment to cause irreversible and catastrophic outcomes" (Section 3.2)—is never empirically verified. BTP ≈ 0.91 on average (often ≈ 1.0 for specific tasks) means that the backdoored agent achieves essentially the same returns as the clean agent, raising the question of whether the forced target actions actually cause any meaningful harm. Without linking ASR to downstream task failure or demonstrating episodes where trigger activation leads to catastrophic outcomes, the security narrative is weakened. The paper needs at minimum: (a) a clear statement of ε, (b) sensitivity analysis across ε values, and (c) some demonstration that target actions (especially beyond boundary actions) produce qualitatively different, harmful behavior.

- **Target-action sensitivity inflates average results and undermines generality claims.** Table 1 shows ASR ranges from ~1.0 for boundary actions ('1', '-1') down to 0.11 for interior actions ('0' on Walk). The main results in Table 4 average over three target actions including '1' (the easiest), which inflates the headline ASR/CP numbers. More concerning, the persistent backdoor evaluation (Table 6) and trigger perturbation analysis (Table 7) only use target type '1', eliminating any insight into whether the method works for more realistic, non-boundary targets. The claim of "broad applicability" across "diverse attack objectives" (Abstract) is not well supported when interior targets yield dramatically worse ASR.

- **Baseline comparison confounds threat models and tuning budgets.** Baffle is a pre-training data-poisoning method (10% poisoning rate, policy-level objective), while TrojanTO is a post-training model-editing method (0.3% trajectories, action-level objective). These differ in both threat model and objective. The 105% CP improvement over Baffle primarily shows that a method tailored to TO models outperforms one not designed for them, rather than establishing TrojanTO's inherent superiority. Additionally, there is no evidence that Baffle or IMC received equivalent tuning effort for the TO setting. A fairer comparison would either adapt baselines to the post-training setting or conduct controlled experiments at matched poisoning rates.

- **Contradictory threat model regarding data access.** Section 3.3 states the adversary has "no access to the original training dataset," yet Section 5.1 operates on a trajectory set {τ_i} filtered by length for backdoor training. The paper never clarifies whether these trajectories come from (a) the original training data (contradicting the threat model), (b) a separate dataset the attacker collects (requiring environment interaction, which contradicts the offline setting), or (c) publicly available datasets (which should be explicitly stated). This ambiguity is significant because the viability of post-training attacks depends on what data the attacker needs.

### Minor:

- **Trigger dimension selection is ad hoc.** Table 2 shows ASR varies from 0.000 to 0.915 depending on the three state dimensions chosen, yet the paper fixes dimensions (1,2,3) without a principled method. In new environments, the attacker cannot know which dimensions are effective without exhaustive search, limiting practical applicability.

- **Trajectory filtering heuristic is poorly justified.** Using length ≥ 20 as a proxy for trajectory quality is a heuristic that may not hold across all D4RL datasets (short trajectories can have high returns). No ablation or validation of this heuristic is provided.

- **Defense analysis is superficial.** Section 6.5 acknowledges that fine-tuning is effective but provides no quantitative results in the main text (details deferred to Appendix B.1). For a security paper, the ease with which the most natural defense removes the backdoor is a significant limitation that should be prominently discussed, not relegated to an appendix.

- **CP metric aggregates away important per-task and per-target variance.** Averaging CP across 6 tasks, 3 models, and 3 target actions obscures failure cases (e.g., DT-Hopp has CP 0.365; DT-Ant has CP 0.302). For a security evaluation, worst-case behavior matters more than averages.

### Trivial:

- The disconnect between the conceptual objective (Eq. 1) and the implemented loss (Eqs. 4-6) is minor but adds confusion about what exactly is being optimized.

## Nice-to-Haves

- **Principled trigger dimension selection:** An automated method (e.g., gradient-based sensitivity analysis) for identifying effective trigger dimensions would address the high variance in Table 2 and improve generalizability.

- **Matched-baseline comparisons:** Running Baffle and IMC at the same 0.3% poisoning rate (or TrojanTO at 10%) would clarify whether the efficiency gains come from the methodology or the budget difference.

- **Per-target-action breakdown in main results:** Reporting ASR/BTP for each target type separately in Table 4 would give readers an honest picture of where the method succeeds and fails.

- **Episode-level qualitative analysis:** Showing example trajectories where trigger activation leads to task-relevant consequences (collapses, failures) would substantiate the threat narrative.

## Removed Points

- **"No demonstration of catastrophic outcomes"** — partially retained in the ASR metric weakness (the lack of task-level harm demonstration is a valid concern), but the harsh reviewer's claim that BTP ≈ 1.0 "contradicts the motivational narrative" is slightly overstated; a backdoor can be a genuine security threat even if returns don't collapse, since the forced action could be harmful in safety-critical deployments (e.g., robotics) where specific actions at specific states matter independent of overall return.

- **"Reward manipulation overgeneralization in Introduction"** — The introduction states that prior RL backdoors "typically rely on reward manipulation throughout training," which is factually correct for the cited works. The paper's empirical demonstration in Section 4.3 specifically tests and validates this claim for TO models. Removed as an over-interpretation.

- **"Training DT/GDT/DC is not obviously so costly"** — This is speculative and contradicts the field's trend toward larger models. For TO models scaling to embodied AI applications (e.g., RT-2), retraining is indeed costly. Removed as an unsubstantiated claim by the reviewer.

- **"Equation 1 does not match the implemented loss"** — This is a minor presentation issue, not a substantive weakness. The conceptual objective (Eq. 1) and the practical implementation (Eqs. 4-6) serve different purposes. Moved to Trivial.

- **"Crude trajectory filtering heuristic"** — Retained under Minor (it is a valid but not fatal concern). The reviewer's suggestion to use returns instead of length is reasonable but doesn't invalidate the method.

- **"Incomplete comparison fairness"** — This overlaps significantly with the already-retained Major weakness on baseline comparison. The specific point about matching poisoning rates is retained as a Nice-to-Have (controlled comparison at matched rates), but the claim that this is a "structural" unfairness is softened since the different threat models reflect genuinely different attack paradigms and the paper's contribution is specifically in the post-training paradigm.

- **"Human finder's SHINE/BIRD defense comparison"** — The human finder suggests comparing against RL-specific backdoor defenses like SHINE. However, Section 6.5 already tests weight pruning, spectral analysis, activation clustering, provable defense, and fine-tuning. Adding SHINE (which is designed for online RL) or BIRD would be a nice addition but is not a critical omission since those methods target different RL settings.

- **"Trigger realism in physical robotics"** — The paper evaluates on D4RL MuJoCo tasks, not physical robots. While the Introduction mentions robotics and embodied intelligence as application domains, the paper explicitly studies TO models on standard benchmarks. Requesting physical-world trigger deployment is outside scope.

## Novel Insights

The paper's most important empirical finding—that reward manipulation, the primary attack vector for traditional RL backdoors, is essentially irrelevant for TO models because they optimize reconstruction loss rather than long-term returns—reframes how the community should think about securing sequence-model-based decision-making systems. This suggests that defenses designed around reward sanitization or reward-robust training are misplaced for this model class, and that trigger detection or input-verification mechanisms are the appropriate focus. However, this insight also cuts against the paper's own threat narrative: if the backdoor doesn't affect cumulative returns (BTP ≈ 1), the practical severity of action-level backdoors in TO models remains an open question.

## Suggestions

1. **Report ε value and conduct sensitivity analysis** across at least 2–3 ε thresholds to make ASR interpretable and establish that reported successes are not artifacts of a generous tolerance.
2. **Include per-target-action ASR/BTP breakdowns** in the main results table, and evaluate persistent attacks and perturbation analysis on at least one non-boundary target action.
3. **Clarify the data source** used by the adversary for trajectory filtering in the threat model, and discuss whether the attack remains viable under more restricted data access.
4. **Provide quantitative fine-tuning defense results in the main text** (minimum: number of steps, data fraction, resulting ASR/BTP degradation), since this represents the most natural barrier to the proposed attack.
5. **Conduct matched-baseline comparisons** (e.g., Baffle at 0.3% trajectory poisoning) to disentangle methodology contributions from budget advantages.

## Score and Decision

**Calibration:** I compared against several papers in related areas:
- RL backdoor papers with overclaimed threats or limited evaluations (em0gAL8fbK: scores 3-5, withdrawn; NALkteEo9Q: scores 3-6, rejected; TCpJXzMnnp: scores 3-6, withdrawn) — these have similar structural issues of mismatched threat models and incomplete evaluations.
- Post-training backdoor via model editing (ZyPRwskBli "Backdoor in Seconds": scores 3-6, rejected; duZANm2ABX "BadEdit": scores 3-8, accepted poster) — BadEdit shares TrojanTO's post-training paradigm and achieves strong ASR with minimal data, but BadEdit also faces criticism about practical deployability and fine-tuning nullifying the backdoor.
- Strong backdoor attack papers with well-established threat models (8iTpB4RNvP "Poisoned Forgery Face": scores 6-8, accepted spotlight) — these have clearer threat realizations and demonstrated consequences.

TrojanTO is stronger than the rejected RL backdoor papers (which often have unrealistic threat models or minimal evaluations) and makes a genuine contribution by identifying that reward manipulation doesn't work for TO models. However, it is weaker than BadEdit (which demonstrates robustness to fine-tuning) and significantly weaker than the strongest backdoor papers (which demonstrate concrete downstream harm). The unspecified ε, the target-action sensitivity, and the fine-tuning vulnerability are substantive gaps.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>