Now let me read some human reviews for calibration purposes.Now I have sufficient information to write the consolidated final review. Let me synthesize everything.

---

## Summary
This paper introduces Preference-based Planning (PbP), a synthetic embodied benchmark built on OmniGibson/Omniverse that defines a three-tiered hierarchical vocabulary of 290 preferences (action, option, and sequence levels) and generates 15,000 egocentric video demonstrations via a rule-based planner. The authors benchmark SOTA vision-language and language models on preference prediction and preference-conditioned planning, finding that (a) end-to-end models fail to infer preferences from raw visual demonstrations, (b) explicitly conditioning planning on predicted preference labels dramatically improves action sequence generation, and (c) symbol-based models generalize better across scenes while vision-based models overfit to visual context.

---

## Strengths

- **Timely and well-scoped problem.** Personalizing embodied agents to user-specific latent preferences rather than explicit instructions is a genuinely underexplored and practically important problem. The hierarchical taxonomy (action / option / sequence level) is a principled and intuitive decomposition, substantively more structured than prior single-task work (NeatNet, SAND).

- **Comprehensive model coverage.** The benchmark evaluates both fine-tuned video models (ViViT, LLaVA, EILEV) and prompted black-box models (GPT-4V, GPT-4), alongside symbolic DAG-based reasoning. This cross-modal evaluation surfaces the consistent finding that even the best video models struggle to extract abstract preferences from raw egocentric observations.

- **Insightful generalization analysis.** The direct vs. orig comparison in Table 4 and Figure 6 clearly demonstrates that vision-based models overfit to scene-specific visual cues while symbol-based models remain robust, which is a concrete and actionable finding for future model design.

- **Diagnostic value of the two-stage decomposition.** While the supervision design is methodologically problematic (see Weaknesses), the experiment is still informative as a diagnostic: it shows how large the gap is between what current models can do end-to-end versus what they could do if they had reliable preference representations, quantifying the opportunity for improvement.

---

## Weaknesses

### Fatal
None triggered. The paper has real contributions, though the methodological concerns are substantial.

### Major

- **Privileged supervision in the two-stage comparison undermines the headline claim.** Section 4.1 states that in the two-stage setting, "models are provided with explicit preference labels during training." The end-to-end models do not receive this. The headline conclusion — "incorporating preference as a key intermediate representation in planning can *significantly improve* planning" — is therefore partly tautological: providing the ground-truth latent variable (or a supervised predictor of it) unsurprisingly helps. The result does not show that preference is a *naturally arising* useful intermediate representation independent of this supervision advantage. To support the paper's claim, one would need to compare equally supervised alternatives or an oracle-preference baseline. The current design only demonstrates that knowing the preference helps, not that the proposed decomposition uniquely or efficiently captures this.

- **Action-level preferences (75 of 290, ~26% of the vocabulary) are entirely absent from all evaluation tables.** Tables 1, 2, 3, and 4 only report Option Level and Sequence Level results. The paper's benchmark coverage claim and "Overall" averages are therefore incomplete. For a benchmark paper, leaving out a full tier of evaluation is a significant gap that weakens the completeness of results.

- **Inconsistency between preference prediction accuracy and near-zero planning distance.** GPT-4 achieves 86.27% Option Level and 68.42% Sequence Level preference prediction accuracy (Table 2), yet its second-stage Levenshtein distances are 0.12 and 12.29 respectively (Table 1). For Option Level, the near-zero distance under imperfect preference accuracy is unexplained — either the planning stage is insensitive to which preference is predicted (undermining the claim that preference matters), or the Levenshtein metric is insensitive to preference-violating action substitutions (undermining the use of this metric). This inconsistency must be resolved; it directly affects the paper's core interpretation.

- **Sole reliance on Levenshtein distance is an inadequate planning metric.** The metric treats action sequences as flat edit-distance strings. It cannot verify (a) whether the generated plan satisfies the target preference, (b) whether semantically equivalent alternative orderings are penalized, or (c) whether the plan is executable in the environment. Section 5.2's interpretation that near-average-length Levenshtein distances mean models "do not understand the preferences implied in the demonstration videos" is too strong: it only shows failure to reproduce the reference string. Without a preference-satisfaction metric or execution-based evaluation, the planning improvement claim is metric-dependent and potentially misleading.

- **Only 3-shot evaluation, no scaling analysis.** The paper is framed around "few-shot learning capabilities" but only evaluates one shot count (N=3). Whether the gap between end-to-end and two-stage approaches shrinks with more demonstrations, or whether model performance scales with context examples at all, is entirely unknown. This significantly weakens the "few-shot" framing.

### Minor

- **No simple baselines for calibration.** The paper lacks random preference selection, majority-class prediction, or a preference-agnostic template planner. Without these, it is impossible to determine whether a Levenshtein distance of ~12–15 in the end-to-end setting reflects genuinely poor performance or an inherently hard metric on these sequence lengths.

- **No error propagation analysis.** The two-stage pipeline is only evaluated under (approximate) oracle stage-1 conditions. The paper never feeds stage-1 *mistakes* into stage-2 to measure how sensitive planning is to prediction error. This matters critically for understanding whether the two-stage design is robust in practice.

- **Standard deviations missing from Table 2.** Preference prediction accuracies in Table 2 are point estimates, making it impossible to assess statistical reliability. Table 1 includes ± values; Table 2 should too.

### Trivial

- The ablation in Table 3 (removing demonstrations) is a useful sanity check, but the interpretation that it "suggests that models do extract meaningful information for preference prediction" is mild; a stronger ablation would compare to a majority-class baseline.

---

## Nice-to-Haves

- A preference-satisfaction metric (binary: does the generated plan include the preference-relevant sub-action/option/sequence?) would complement Levenshtein distance and directly measure the paper's core goal.
- Scaling curves for N=1, 3, 5, 10 demonstrations would substantiate the "few-shot" framing and provide useful guidance for practitioners.
- Side-by-side qualitative examples showing ground-truth vs. predicted action sequences for correct vs. incorrect preference predictions would make the findings more intuitive.
- A per-preference difficulty analysis (which of the 290 preferences are systematically harder?) would help characterize the benchmark's structure.
- Discussion of architectural approaches to close the gap between symbol-based robustness and vision-based perception.

---

## Removed Points

*These points are flagged to be removed — treat them with caution.*

- **Harsh Critic Claim: Symbolic baselines solve a fundamentally easier problem, undermining modality comparisons.** The paper itself explicitly states in Section 4.2 that "The action input serves as a high-level abstraction of the egocentric video, reducing the complexity associated with visual data," and that symbol-based models are included "for ablative purposes." The paper does not present the modality comparison as apples-to-apples; it uses it to diagnose the difficulty of visual perception. The underlying concern is valid but the paper already frames it appropriately — this is weakened to a minor note rather than a structural criticism.

- **Harsh Critic Claim: The benchmark mainly tests memorization of preference-specific action templates.** This concern is raised without direct evidence that train/test splits are compromised. The paper uses 15,000 instances across 290 preferences with randomly assigned scenes and objects per instance. While the concern about template memorization is conceptually valid, it is speculative in the absence of an empirical memorization check. Removed as unsubstantiated.

- **Harsh Critic: The watch-and-help framing is inconsistent with the simulator-generated setting.** The paper says "A PbP task resembles a real-world watch-and-help setting" (Section 4.1). "Resembles" is not a factual claim that PbP *is* a real watch-and-help system. The motivation/analogy is reasonable for a benchmark paper. Removed as a strawman.

- **Human Finder Reviewer: Missing baselines from related embodied planners (NeatNet, SAND, etc.).** Per the DO NOT MENTION MISSING RELATED WORKS rule, this is removed. These methods are also from different task settings (rearrangement-only) and would not be directly comparable.

---

## Novel Insights

The most genuinely interesting finding in this paper is the *asymmetric generalization failure* of vision-based versus symbol-based models: vision models fail when scene or object identity changes even for the same underlying preference, while symbol-based models remain robust (Table 4, Figure 6). This suggests that current VLMs encode preferences as visual scene templates rather than abstract behavioral regularities — a concrete diagnostic that would inform future architectural choices. The finding that end-to-end models produce action sequences with Levenshtein distances near the *average sequence length* (effectively random output) while still being state-of-the-art VLMs is also a sobering result about the current gap in preference-aware embodied reasoning.

---

## Suggestions

1. **Report action-level results in all tables.** The 75 action-level preferences represent 26% of the vocabulary and must be included for the benchmark evaluation to be complete.
2. **Add a preference-satisfaction metric** (e.g., binary check: does the generated sequence include the preference-relevant action/option/sub-sequence?), which would directly measure planning quality without Levenshtein's shortcomings.
3. **Clarify the stage-2 evaluation protocol:** Is stage 2 evaluated with ground-truth or predicted preference labels? If predicted, explain the inconsistency between GPT-4's ~77% accuracy and near-zero Levenshtein distances.
4. **Include N=1 and N=5+ few-shot ablations** to substantiate the "few-shot" framing.
5. **Add simple baselines** (random preference, majority-class preference, preference-agnostic planner) to anchor what "hard" and "easy" look like on this benchmark.
6. **Run an error-propagation experiment**: feed stage-1 predictions (including errors) into stage-2 and measure the downstream planning impact.

---

## Score and Decision

**Calibration:**
- *PARTNR* (T5QLRRHyL1, 8,6,6,8 → Accepted Poster): 100K tasks, LLM-generated with verification loop, real human experiments, execution-based metrics. Substantially stronger than this paper.
- *RAPL* (CTlUHIKF71, 6,3,6,6 → Accepted Poster): Preference-based robot learning with real visual representations, also simulated but with a more rigorous evaluation loop. Similar scope.
- *DivScene* (G6DLQ40VVR, 6,5,6,8 → Rejected): Large diverse navigation benchmark rejected primarily for weak baselines and naive methodology. This paper has analogous issues.
- *7NHF4txacw* (3,3,3,6 → Rejected): Weaker paper overall with more fundamental methodological flaws. PbP is clearly above this.
- *pwKokorglv* (3,5,5,3 → Rejected): Rejected for unfair comparisons and lack of novelty. PbP's benchmark contribution is more solid.

The paper under review is closest in quality to the borderline range of CTlUHIKF71 (accepted) and G6DLQ40VVR (rejected). The benchmark itself has genuine value, but the major evaluation issues — missing action-level results (26% of vocabulary), supervision asymmetry in the headline comparison, sole reliance on an inadequate metric, GPT-4 accuracy/distance inconsistency, and no few-shot scaling — together represent a pattern of incomplete evaluation that prevents the central claims from being well-supported. This is sufficient to place it below the acceptance threshold in its current form.

**Originality:** Moderate. The preference hierarchy concept is novel, but the evaluation methodology is underdesigned.  
**Importance of research question:** High. Personalized preference learning for embodied AI is a critical problem.  
**Claims vs. support:** Weak. The headline claim about preference as an intermediate representation is supported by privileged supervision rather than a clean ablation.  
**Soundness of experiments:** Fair. The evaluation covers many models but has significant metric and design limitations.  
**Clarity of writing:** Good. The paper is readable and well-organized.  
**Value to the research community:** Moderate. The benchmark could be useful if the evaluation gaps are filled.

**Score: 4.5**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>