=== CALIBRATION EXAMPLE 1 ===

# Final Consolidated Review
## Summary
This paper introduces Preference-based Planning (PbP), a simulation benchmark built on NVIDIA Omniverse/OmniGibson for evaluating embodied agents' ability to infer and execute personalized preferences via few-shot demonstrations. The authors define a three-tier preference hierarchy (Action, Option, Sequence levels), construct 290 preferences across 50 scenes with 15,000 synthetic demonstrations, and benchmark a wide suite of vision-based and symbol-based models. The central finding is that a two-stage pipeline — explicit preference prediction followed by preference-conditioned planning — substantially outperforms end-to-end approaches, and that symbol-based LLMs significantly outperform vision-based models in both stages.

---

## Strengths

- **Specific modality gap finding:** Table 2 reveals a stark and reproducible gap between symbol-based models (GPT-4: 86.27% option-level accuracy) and the best vision-based model (GPT-4V: 48.48%), despite GPT-4V being the strongest multimodal model tested. This is not a generic observation — it precisely characterizes where the bottleneck lies (visual preference abstraction, not planning) and directly informs future research directions.

- **Demonstration necessity validated by ablation:** Table 3 shows that removing in-context demonstrations causes catastrophic degradation at the sequence level (GPT-4V: 37.50% → 0.00%; EILEV: 32.69% → 0.00%), while option-level performance is more retained. This disentangles prior knowledge from genuine in-context preference inference, providing non-trivial evidence that models are actually using the demonstrations rather than relying on commonsense priors alone.

- **Generalization experiment (Table 4) isolates a mechanistic insight:** Symbol-based GPT-4 is nearly unaffected by scene/object changes (86.27% vs. 86.32% option-level), while vision-based models degrade. Crucially, Figure 6 shows that vision-model failures in generalization and non-generalization cases do not strongly overlap, suggesting the models rely on contextual visual consistency rather than truly learning abstract preferences — a concrete and falsifiable claim.

- **Hierarchical preference taxonomy:** The three-level hierarchy (Action/Option/Sequence) is a principled decomposition that enables level-specific evaluation. The distinction between option-level (alternative sub-task methods) and sequence-level (sub-task ordering) preferences captures qualitatively different human decision-making dimensions, which is more granular than prior work on rearrangement-only personalization.

---

## Weaknesses

### Fatal
*None identified.*

### Major

- **Ambiguity about oracle labels in Stage 2 (core finding at risk):** Section 5.3 states models are provided with "the current preference label" and Figure 5's caption says "predicted preference labels," yet these characterizations may be inconsistent. GPT-4 achieves a Levenshtein distance of 0.12 ± 3.12 at option level in Stage 2, which is almost suspiciously low even given its 86.27% Stage 1 accuracy. If Stage 2 actually uses *ground-truth* labels (not Stage 1 outputs), then the two-stage improvement is near-tautological: given the correct preference token, planning reduces to a lookup. The paper must explicitly confirm whether Stage 2 uses predicted labels or ground-truth labels, and if the former, provide the full end-to-end chain evaluation (Stage 1 output → Stage 2). This is the most important clarification needed, as it affects the interpretation of the paper's central claim.

- **Action-level preferences defined but never evaluated:** The paper defines 75 action-level preferences and counts them toward "hundreds of diverse preferences," yet Tables 1–4 report only option-level and sequence-level results. No explanation is given for the omission. If action-level evaluation was skipped due to triviality or evaluation difficulty, this should be explicitly stated; otherwise, a third of the preference vocabulary is unaccounted for in the empirical results.

- **GPT-4's anomalously high zero-shot accuracy raises prior-knowledge concern:** Without demonstrations, GPT-4 retains 73.87% option-level accuracy (Table 3) — far above any other model (next best: Llama3 at 39.50%, GPT-4V at 29.42%). Since preferences are defined over Behavior-1K household activities, and GPT-4 was trained on broad internet text, this performance gap may partly reflect memorized commonsense knowledge about typical household task orderings rather than genuine preference inference. The paper acknowledges (correctly) that models do extract information from demonstrations via the sequence-level drop, but the source of GPT-4's outsized zero-shot performance is not analyzed. A control using unusual/counter-commonsense preferences would help distinguish prior-knowledge leakage from genuine few-shot learning.

### Minor

- **Levenshtein distance is an imperfect proxy metric.** Treating each action as a token and comparing sequences penalizes plans that satisfy the preference but differ in equivalent sub-orderings. A preference-aware accuracy metric (e.g., "does the generated sequence exhibit the stated preference?") would more directly measure the paper's goal and could reveal cases where low Levenshtein distance masks preference violation, or where high distance nonetheless satisfies the preference.

- **No task-success evaluation in the simulator.** The paper evaluates plans only by textual similarity to ground-truth sequences, not by whether the generated plans are physically executable or produce the intended outcome in the simulator. Since the simulator supports execution, reporting task success rates would strengthen claims about planning quality.

- **Incomplete failure analysis.** Figure 6's heatmap shows scene-dependent and model-dependent failure patterns, but the analysis is qualitative ("failure cases are not completely repeated"). Quantifying which scene types or preference classes drive failures, or showing specific confusion patterns among preferences (e.g., which option-level preferences are systematically confused), would make the analysis substantially more actionable.

- **Stage 2 results omit ViViT** (no second-stage row for ViViT in Table 1). It is not explained whether ViViT was excluded from the two-stage setting due to architectural constraints or simply not evaluated. This should be clarified.

### Tiny

- The paper states "models seem to lack this mode of thinking" (Section 5.3) without elaboration. This is vague and should be replaced with a more precise characterization.
- The Limitations section is very brief (four sentences). The implications of synthetic-only data for benchmark validity deserve more substantive engagement than the current treatment.

---

## Nice-to-Haves

- **Few-shot sensitivity ablation:** Varying the number of demonstrations (1, 2, 3, 5) to characterize the data-efficiency of preference learning would validate the "few-shot" framing and reveal whether 3 demonstrations is a meaningful choice.
- **Human performance baseline:** Even a small set of human annotators predicting the preference from 3 demonstrations would calibrate difficulty and contextualize model results.
- **Modality-controlled VLM baseline:** Providing VLMs with transcribed action text (rather than raw video) would isolate perceptual failure from reasoning failure, determining whether vision models fail because they cannot *see* the actions or because they cannot *abstract* the preference.
- **Noisy demonstration experiments:** Introducing small amounts of noise or minor inconsistencies in demonstrations would assess robustness to the kind of variability present in real human behavior — a useful stress test given the synthetic data's perfect consistency.
- **Closed-loop / task-success evaluation** (simulator execution) would be a natural next step and would make the benchmark substantially more compelling as a robotics resource.
- **Confusion matrix on preference prediction:** Showing which preferences are systematically confused (e.g., option-level preferences that share many visual signatures) would characterize benchmark difficulty structure.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"DAG-Opt is a weak and inappropriate baseline"** (Harsh Critic): The paper explicitly introduces DAG-Opt as an ablative baseline to examine dependency-based symbolic reasoning, not as a competitive method. Its poor performance is informative in context, and the paper appropriately notes the difference between dependency learning and next-token prediction. The baseline serves its stated purpose.
- **"GPT-4 is unfairly compared to Llama3-8B due to scale"** (Harsh Critic): The paper is benchmarking existing models, not proposing a new one. Size disparity among benchmarked models is expected and not a flaw in benchmark design; the scale difference is informative.
- **"Societal impact statement is dismissive"** (Harsh Critic): The preference-learning task scoped here is within private domestic scenarios, and the stated limitation is reasonable for this scope.
- **"Contributions relative to NeatNet/SAND are insufficiently differentiated"** (Harsh Critic): The paper clearly states that prior work is limited to single-task rearrangement, and the contribution is generalizing to diverse tasks across multiple scenes. The differentiation, while brief, is adequate.
- **"No related work X is missing"**: Per review policy, missing related work is not raised as we cannot confirm the existence of external references.
- **Generic strength: "The paper is well-written / topic is important"** (Balanced Review): Removed as non-specific.
- **"Extensive experiments"** (Balanced Review): Removed as a generic strength.

---

## Novel Insights

The most penetrating cross-review observation is the **perception-abstraction dissociation**: vision-based models fail not because they lack planning capability (they improve dramatically when given explicit preference labels, Table 1 Stage 2), but because they cannot abstract a preference from visual demonstrations. This distinguishes preference inference as a distinct capability bottleneck from both perception and planning. Notably, Figure 6's overlap analysis adds a further nuance: even when vision models get the right preference label in the *direct* (same-scene) case, failure patterns don't repeat in the generalization case, suggesting the model is not truly learning the preference but rather pattern-matching on scene-consistent cues. This implies the fundamental challenge is not labeled-data quantity but the absence of scene-invariant preference representation learning — a specific and actionable gap that is more precisely characterized here than in prior work.

---

## Suggestions

1. **Clarify Stage 2 label source explicitly**: Add one sentence in Table 1's caption and Section 5.3 specifying whether Stage 2 uses predicted labels from Stage 1 or ground-truth labels. If ground-truth, add an end-to-end chain evaluation (Stage 1 outputs feeding Stage 2) as the primary reported result, since the oracle-label case is best moved to an analysis of planning capability given perfect preferences.

2. **Add action-level results or explain omission**: Report results for action-level preferences (75 instances) or provide a clear explanation of why they are excluded (e.g., action-level preferences cannot be evaluated with Levenshtein on option sequences). If the level is not evaluable by current metrics, revise the benchmark description accordingly.

3. **Analyze GPT-4's zero-shot performance source**: Add a controlled experiment with counter-commonsense or reversed preferences (e.g., "cut first, then wash") to test whether GPT-4's zero-shot retention is due to prior knowledge about typical task orderings. This would directly address the contamination concern.

4. **Report a full end-to-end two-stage pipeline number**: In addition to the staged analysis, report performance when Stage 1 outputs are chained into Stage 2, to give practitioners a realistic expectation of the system's capability.

5. **Strengthen failure analysis in Section 5.4**: Quantify what fraction of failures are consistent across direct/gen conditions per model, and identify whether specific preference categories (e.g., fine-grained option choices vs. coarse sequence orderings) drive the fragility of vision models.

---

**Evaluation summary:**
- **Novelty:** Moderate — primarily a benchmark contribution with a well-designed preference taxonomy; no new algorithm or learning method is proposed.
- **Technical soundness:** Moderate — the experimental protocol has a meaningful ambiguity in the Stage 2 evaluation that must be resolved; the remaining methodology is straightforward.
- **Empirical support:** Mixed — the modality gap, demonstration ablation, and generalization findings are convincingly supported; the two-stage benefit is well-documented but its interpretation depends on the oracle-label question.
- **Significance:** Moderate-to-good — addresses a real and underexplored gap (personalized preference inference at scale) and provides a reusable benchmark, though the synthetic-only nature limits immediate real-world applicability.
- **Clarity:** Adequate overall, with one consequential ambiguity (Stage 2 label source) that should be fixed in revision.

# Actual Human Scores
Individual reviewer scores: [8.0, 5.0, 3.0, 3.0]
Average score: 4.8
Binary outcome: Reject
