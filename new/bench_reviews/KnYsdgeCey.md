Now let me read the calibration tools and search for comparable papers.Now I have enough to write a thorough, calibrated final review.

---

## Summary

This paper introduces **Preference-based Planning (PbP)**, an embodied benchmark built on NVIDIA Omniverse / OmniGibson that frames personalized household planning as a few-shot preference learning problem. The environment contains 290 hierarchical preferences across three levels (action, option, sequence), 50 simulated scenes, and 15,000 egocentric video instances. The paper evaluates a range of video-based (ViViT, LLaVA, EILEV, GPT-4V) and symbol-based (DAG-Opt, Llama3, GPT-4) models in end-to-end and two-stage settings, finding that explicitly using a predicted preference label as an intermediate representation substantially improves subsequent planning.

---

## Strengths

- **Hierarchical preference taxonomy with multimodal annotation**: The three-tier hierarchy (action/option/sequence levels; 75, 135, 80 preferences respectively) is principled and supports evaluation at different granularities. Each instance comes with synchronized egocentric video, bird's-eye map, and frame-level text annotations (Figure 4), making the dataset richer than prior single-modality rearrangement benchmarks (e.g., NeatNet, SAND).

- **Clean ablation isolating in-context learning from prior knowledge (Table 3)**: Removing demonstrations causes a sharp drop in preference prediction accuracy—GPT-4 falls from 68.42% to 9.42% at the sequence level—directly demonstrating that models genuinely extract preference signal from demonstrations rather than relying purely on prior knowledge. This is the most informative experiment in the paper.

- **Modality-specific generalization analysis (Table 4 + Figure 6)**: The comparison between "direct" (same scene/objects) and "orig" (generalized) conditions, combined with per-datapoint heatmaps, provides a nuanced finding: symbol-based models are robust to scene variation because preferences map cleanly to symbolic abstractions, while vision models overfit to contextual visual cues. This insight has clear implications for future work.

- **Scale and realism of the simulation infrastructure**: 50 scenes, 15,000 instances, physics-realistic rendering, and egocentric views designed to transfer to real head-worn device data represent a genuine infrastructure contribution.

---

## Weaknesses

### Fatal
None.

### Major

- **Action-level preferences are entirely absent from all evaluations.** The paper defines 75 action-level preferences (Section 3.2) as part of its core 290-preference vocabulary, yet Tables 1, 2, and 3—which form the basis of all empirical claims—contain results only for option-level and sequence-level. There is no explanation for this omission. For a benchmark paper, this is a significant gap: a substantial portion (26%) of the defined benchmark is never evaluated, leaving the coverage claim unverified and making it impossible to compare model behaviors across all three preference levels.

- **Second-stage evaluation does not clearly report joint pipeline performance.** Section 5.3 describes a two-stage pipeline where Stage 1 predicts preference labels and Stage 2 uses those predicted labels for planning. However, the paper reports Stage 1 accuracy (Table 2) and Stage 2 Levenshtein distance (Table 1) in complete isolation without ever computing the full pipeline's expected performance under propagated Stage 1 errors. It also does not include an oracle condition (Stage 2 with ground-truth labels) to establish an upper bound. Without this three-way table (end-to-end / predicted-labels pipeline / oracle pipeline), the headline claim that "preference as an intermediate representation significantly improves planning" cannot be properly evaluated in terms of real-world deployability.

### Minor

- **Levenshtein distance against a single rule-based reference is the only evaluation metric.** Since preferences are abstract (e.g., "wash fruit before eating"), multiple valid action sequences may satisfy a given preference depending on scene layout. Deviation from the single reference trajectory produced by the rule-based planner conflates *preference alignment* with *trajectory imitation*. At minimum, a preference-conditioned success rate—does the generated plan satisfy the declared preference regardless of exact ordering?—would complement the metric and strengthen the benchmark's construct validity.

- **No minimum calibration baselines for the end-to-end setting.** The claim that end-to-end models "predominantly predict inconsistent actions" (Section 5.2) is supported by Levenshtein distances close to average sequence length, but without a trivial baseline (e.g., predict the most frequent training sequence, or return the nearest-demonstration sequence), it is impossible to determine whether models are performing near-chance or meaningfully above it. This makes the reported distances hard to interpret.

### Trivial

- The paper includes ViViT as a "lower bound" but does not train or fine-tune any video model on the PbP training split. All video baselines are evaluated essentially zero-shot, so the experiment conflates model capability with dataset difficulty.

---

## Nice-to-Haves

- A user study or qualitative validation confirming that the 290 hand-designed preference types meaningfully correspond to real human behavioral variation would greatly strengthen the benchmark's claim to study "personalized human preferences." The paper acknowledges this gap in its limitations but does not propose a validation path (beyond mentioning ongoing real-world data collection).
- Qualitative failure analysis with predicted vs. ground-truth action sequences for representative preferences at each level would reveal whether errors are near-misses (wrong order) or fundamentally wrong preference attributions, helping future researchers diagnose model weaknesses more precisely.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Two-stage comparison is trivially true / reduces to instruction-following"** (Harsh Critic, Section 5.3): The critic argues that telling GPT-4 its preference label and asking it to generate actions is just instruction-following. While partially valid, the paper's actual contribution is revealing the *specific capability gap* between extracting preferences from raw observations vs. using explicit labels—not claiming that LLMs cannot follow instructions. The finding has diagnostic value. Moreover, based on Section 4.1, the second stage uses *predicted* labels (not oracle labels), so the result is not purely a trivial instruction-following demonstration. Removed as overclaimed.

- **"Levenshtein distance is a core methodological failure"** (Harsh Critic): Since preferences are implemented by a deterministic rule-based planner with canonical action sequences per preference + scene, there is a well-defined reference plan. Levenshtein distance against that reference is a reasonable first metric, even if imperfect. The concern is real but much weaker than "invalid"—downgraded to Minor above.

- **"Benchmark should require human demonstrations or a user study to be valid"** (Harsh Critic): The paper explicitly scopes itself as a simulation benchmark with synthetic demonstrations, acknowledges the gap, and frames real-world collection as future work. Requiring real human subjects to validate a simulation benchmark would go well beyond the paper's stated scope and community norms for this type of work. Removed as scope creep.

- **Strength: "Clear empirical demonstration that preference as intermediate representation improves planning"** (Strength Finder): Partially valid, but conflated with the unresolved oracle/predicted-label ambiguity. Kept in weaknesses (Major) rather than as a strength since it depends on the unresolved evaluation design issue.

---

## Novel Insights

The most genuinely novel diagnostic finding is the *asymmetric failure mode* revealed by combining Tables 2 and 3: vision-based models (GPT-4V) show moderate preference prediction accuracy (~48%) but near-zero Stage 2 Levenshtein when given explicit labels, while symbol-based models (GPT-4) show high prediction accuracy (~86%) and also near-zero Stage 2 distance. This jointly implies that the bottleneck for vision models is entirely in the perceptual/inferential front end—they cannot abstract preference from visual input—while symbol-based models have essentially solved the planning backend once the preference is known. This cross-modal diagnosis is more informative than aggregate scores and suggests that the benchmark could serve as a useful decomposition tool for evaluating embodied AI systems.

---

## Suggestions

1. **Evaluate action-level preferences** in Tables 1–3 to complete the benchmark coverage.
2. **Report the three-way pipeline comparison**: end-to-end, two-stage with predicted labels, and two-stage with oracle labels. This turns the current ambiguous result into a proper ablation.
3. **Add a simple baseline** (most-frequent training sequence, nearest-demo retrieval) to anchor the end-to-end Levenshtein numbers.
4. **Add a preference-satisfaction metric** alongside Levenshtein: for each preference type, code a binary checker (does the plan include the required action sub-sequence?) that is independent of trajectory ordering.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison to PbP |
|---|---|---|---|
| HAZARD Challenge | `n6mLhaBahJ.md` | **6.75** | New embodied simulation benchmark with cleaner evaluation, no major evaluation gaps |
| LoTa-Bench | `ADSxCpCu9s.md` | **6.0** | Benchmark for LLM planners in embodied settings, similar scope, no missing category evaluation |
| VisualAgentBench | `2snKOc7TVp.md` | **5.75** | Broader benchmark for VLMs as agents, accepted poster despite some presentation gaps |
| MOR Benchmark | `XDEWIMoiNK.md` | **4.0** | Embodied AI benchmark rejected for missing critical baselines; similar pattern of evaluation gaps |
| TeamCraft | `nE3flbe88p.md` | **3.25** | Embodied multi-agent benchmark withdrawn for weak experimental choices and inadequate benchmarking |

PbP has more novel environment infrastructure than LoTa-Bench (which reuses existing simulators) and a more principled preference hierarchy than VisualAgentBench. However, the missing action-level evaluation is a concrete, documented gap that the low-scoring MOR and TeamCraft papers were also penalized for in similar form (incomplete evaluation, missing baselines). The joint pipeline analysis gap and single metric concern further separate PbP from the 6+ cluster. The paper falls below LoTa-Bench (6.0) due to the missing evaluation category, and sits closer to the VisualAgentBench (5.75) / borderline range.

**Score: 5.0** — The paper makes real contributions (novel environment, hierarchical preferences, informative ablations) but has a substantive incomplete evaluation (missing action level) and an ambiguous experimental design for its central claim, placing it marginally below the acceptance threshold.

**Decision: Reject**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>