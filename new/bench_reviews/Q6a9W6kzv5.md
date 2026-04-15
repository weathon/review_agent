Now let me read some calibration review files to score appropriately.## Summary

PhysBench introduces a large-scale benchmark (10,002 interleaved video-image-text entries) for evaluating Vision-Language Models on physical world understanding, organized across four domains (physical object properties, relationships, scene understanding, physics-based dynamics), 19 subclasses, and 8 capability dimensions. Through evaluation of 75 VLMs, the paper documents a large gap between current models (~40% average, GPT-4o at 49.5%) and humans (95.87%), and proposes PhysAgent—a framework augmenting VLMs with vision foundation models (Depth Anything, SAM, GroundingDINO) and a knowledge memory module—achieving an 18.4% improvement on GPT-4o and showing downstream gains on five simulated embodied manipulation tasks.

---

## Claims and Support

**Claim 1: PhysBench is a comprehensive benchmark across four domains / 19 subclasses / mixed modalities.**
*Partially supported.* The breadth of tasks and modalities is real and well-documented in Tables 1–2 and Figures 2–3. However, "comprehensive" in the sense of measuring a coherent *physical world understanding* construct is not validated. There is no per-subtask human agreement, ambiguity analysis, or shortcut analysis to confirm questions require physical reasoning rather than difficult visual discrimination. Human accuracy is high overall (95.87%), which is reassuring, but limited validation detail is provided.

**Claim 2: Current VLMs have poor physical world understanding; best models are far from human level.**
*Well supported within groups, partially confounded across groups.* The claim that VLMs perform around 40% vs. human 95.87% is consistent throughout Table 3. However, Image VLMs and Video VLMs are evaluated on a subset (no interleaved data), while General VLMs see the full dataset. Cross-group comparisons therefore reflect both model capability and dataset difficulty differences.

**Claim 3: Closed-source models significantly outperform open-source models.**
*Mostly supported, with one numeric clarification.* The paper states "GPT-4 surpasses the best open-source model, LLaVA-interleave, by 20.7%." This is a relative improvement: (49.49–41.00)/41.00 ≈ 20.7%—mathematically correct, though the comparison mixes subsets. The broader trend is plausible but the protocol inconsistency (different model classes see different data) means the gap quantification is imprecise.

**Claim 4: VLM physical understanding does not scale with model size, data, or frames, likely due to lacking physical knowledge in training data.**
*Scaling trend: partially supported on a narrow set of model families. Causal explanation: not established.* The scaling analysis (Fig. 6) is suggestive but covers only 3 model families with uncontrolled architecture changes. The conclusion about training data deficiency is repeatedly hedged in the paper ("likely due to," "may be attributed to"), which is appropriate—but the body still frames it as a finding rather than a hypothesis. Alternative explanations (perception bottlenecks, architectural limits, MCQ artifacts) are not ruled out.

**Claim 5: Perceptual errors and knowledge gaps are primary failure causes.**
*Plausible but unvalidated rigorously.* The 500-question expert annotation is a reasonable exploratory analysis. No inter-annotator agreement is reported, making the precise percentages (37–45% perceptual, 23–35% knowledge) hard to interpret quantitatively. The analysis is directionally useful but lacks reliability statistics.

**Claim 6: PhysAgent improves GPT-4o by 18.4% without task-specific rigidity.**
*Performance gain: supported for GPT-4o. Generality and mechanism: not demonstrated.* The reported Relationships improvement from 61.8% to 84.2% (22.4 pp) is large and raises questions about whether the knowledge memory is shortcutting. The "18.4%" headline figure is inconsistent with the task-level numbers in Figure 9(a), which compute to roughly ~21–23% relative improvement depending on denominator; the exact subset/calculation is unspecified. No component ablation is provided to attribute the gain to specific modules. The framing as a "general" framework conflicts with explicit task-specific routing ("manually or automatically classifies the question and activates task-specific prompts").

**Claim 7: PhysAgent outperforms ContPhy because ContPhy is inflexible.**
*Partially supported as observation, unsupported as mechanism.* The empirical observation that the authors' ContPhy re-implementation underperforms GPT-4o baseline is documented. Whether this is due to ContPhy's rigidity or the reimplementation setup is not established.

**Claim 8: Physical understanding improvements help embodied agents (MOKA).**
*Directionally supported as a limited proof-of-concept.* Five simulation tasks show consistent improvement. No trial counts, variance, or significance analysis is reported (results in 0.1 increments suggesting ~10 trials per task). The tasks are simple single-step pick-and-place and aligned with benchmark categories—insufficient to support broad embodied deployment claims.

---

## Strengths

- **Scale and breadth**: At 10,002 entries spanning four physical domains, 19 sub-tasks, and mixed modalities (image, video, interleaved), PhysBench substantially exceeds comparable physics benchmarks (CLEVRER: 300K but one domain; PhyGenBench: 160 prompts; Physics-RW: narrow scope). Table 1 makes this comparison concrete.
- **Large-scale evaluation**: 75 VLMs tested is among the most extensive evaluations on any VQA benchmark, providing a meaningful community resource. The finding that performance stagnates across scaling dimensions within multiple model families is a valuable empirical result.
- **Error analysis motivates design**: The six-category expert analysis (Fig. 7), while lacking reliability statistics, provides a reasonable and actionable diagnosis (perception ≈37–45%, knowledge ≈23–35%) that directly motivates the PhysAgent architecture.
- **Downstream embodied validation**: Connecting benchmark performance to MOKA manipulation tasks is a differentiating contribution compared to prior physics benchmarks that stop at leaderboard results.
- **Dataset construction rigor**: The five-step pipeline with 4,000 hours of annotation, multi-round cleaning, and preservation of intermediate outputs (depth/reflectance maps, annotated physical principles) is a genuine contribution to the community.

---

## Weaknesses

### Fatal
*None.* The core finding—VLMs struggle substantially with physical world understanding—is adequately established even under the most charitable reading of the methodological concerns.

### Major

- **Non-uniform evaluation protocol undermines cross-group comparisons (Sec. 3.3, Table 3)**: Image VLMs and Video VLMs are tested on a data subset without interleaved QA, while General VLMs (including GPT-4o and Gemini) receive the full dataset. The aggregate Table 3 rankings therefore conflate two effects: model capability and test set difficulty. The paper draws broad conclusions ("closed-source models significantly outperform open-source ones") from a table that cannot cleanly support them. A common evaluation subset for all models, or explicit stratified reporting, is required to justify cross-group comparisons.

- **PhysAgent has no component ablation (Sec. 4.1, Fig. 9a)**: The framework combines task-specific prompt activation, three foundation models (Depth Anything, SAM, GroundingDINO), a knowledge memory, chain-of-thought reasoning, and self-verification. None of these are isolated. It is impossible to determine whether the 18.4% gain comes from retrieved knowledge rules, the foundation model outputs, the task routing logic, the self-verification step, or their combination. This is the minimum necessary to justify calling PhysAgent a principled framework rather than an engineered pipeline.

- **Task classification is "manually or automatically" with no reported accuracy or failure analysis (Sec. 4.1)**: Automatic question classification is a prerequisite for real-world deployment, yet the paper does not report classification accuracy or any results under automatic-only classification. If manual classification is used in the reported experiments, the 18.4% improvement figure is an oracle-assisted result, not a deployable system result.

- **The 18.4% improvement figure is inconsistent with the reported task-level numbers**: GPT-4o baselines in Figure 9(a) (Property 56.9, Relationships 61.8, Scene 30.1, Dynamics 46.0) yield an average of ~48.7%, and PhysAgent results (58.4, 84.2, 45.8, 51.2) yield ~59.9%—a ~23% relative improvement, not 18.4%. The subset, denominator, and weighting used to arrive at 18.4% are not specified anywhere in the main paper. The Relationships jump (61.8→84.2, +22.4 pp) is implausibly large relative to the other domains, warranting targeted error analysis.

### Minor

- **Knowledge memory is underspecified**: The knowledge memory is central to PhysAgent but its construction, size, coverage scope, and retrieval mechanism are only illustrated with a single example (light/shadow rules in Fig. 8). Characterizing the knowledge base and evaluating retrieval quality independently would substantially strengthen this contribution.

- **Embodied evaluation lacks statistical rigor (Sec. 4.2, Fig. 9c)**: Success rates are reported to one decimal place (e.g., 0.6, 0.7, 0.8) without trial counts, standard deviations, or confidence intervals. From the precision of reported values, the implied sample size is ~10 trials per task, which is insufficient to support statistical claims. Changes of 0.1–0.3 across conditions should be reported with appropriate uncertainty.

- **The 75-model claim vs. 39 in main paper**: Abstract and Sec. 3.3 claim "75 representative VLMs" but Table 3 shows 39 ("Evaluation results for 39 VLMs"). The remaining 36 are deferred to an appendix. While this does not invalidate the results, the gap between the headline claim and what is directly visible in the paper creates unnecessary confusion. The main text should clarify the selection logic for the 39 shown.

- **Error taxonomy lacks inter-annotator agreement**: The expert annotation of 500 mispredictions into six error categories (Fig. 7) underpins the PhysAgent design rationale, yet no inter-annotator agreement (Fleiss's κ or similar) is reported. Since categories like "perceptual error" vs. "lack of knowledge" vs. "reasoning error" involve judgment calls, the reliability of the proportions (37%, 34%, etc.) cannot be assessed.

- **Language/text-only baseline absent**: No text-only baseline is provided to verify that questions require visual input rather than exploiting language priors. For a four-option MCQ benchmark with answer choices sometimes containing linguistic cues, this is a standard sanity check. Similarly, answer option distribution balance across A/B/C/D is not reported.

### Trivial

- Human performance is reported only in aggregate (95.87%) without per-domain or per-subtask agreement, making it hard to identify ceiling effects or ambiguous subcategories.
- The knowledge transfer experiment (200 entries, Fig. 9b) is too small to generalize across 19 subcategories; presented conclusions are likely correct in direction but should be scoped appropriately.

---

## Nice-to-Haves

- **Automatic classification evaluation**: Report PhysAgent accuracy under fully automatic question classification and analyze failure modes of the classifier.
- **Ablation of PhysAgent components**: Isolate contributions of (a) knowledge memory, (b) foundation model outputs, (c) task routing, and (d) self-verification.
- **Shortcut analysis**: A text-only baseline (question + choices, no images) would confirm questions require visual input. Answer distribution balance across options should be reported.
- **Harder/more diverse embodied tasks**: Multi-step manipulation or real-world deployment would substantially strengthen the embodied claim.
- **Computational overhead reporting**: PhysAgent invokes multiple foundation models plus multiple VLM calls; inference cost vs. performance trade-off matters for practitioners.
- **Evaluate PhysAgent on existing benchmarks**: Demonstrating transfer to CLEVRER, Physion, or similar benchmarks would validate generality beyond the authors' own benchmark.

---

## Removed Points

*These points were flagged for removal. Treat them with caution.*

- **"GPT-4o vs LLaVA-interleave 20.7% is a calculation error" (Harsh Critic)**: Removed. The claim is mathematically correct as a relative improvement: (49.49 − 41.00) / 41.00 ≈ 20.7%. While potentially confusing relative to an absolute difference, this is not an error.

- **SuperCLEVR Table 1 coverage mismatch (Spark)**: Removed. This reviewer raises the possibility that SuperCLEVR (Wang et al., 2024g) in Table 1 is mislabeled as covering Temperature, Viewpoint, Light, Manipulation, and Fluid. However, since the paper cites this as a 2024 work (distinct from the original SuperCLEVR), and we cannot verify external dataset capabilities, this concern cannot be confirmed and is therefore excluded per the hard rules.

- **Missing related works**: Removed per policy.

- **Reproducibility nitpicks** (undisclosed hyperparameters, prompt templates in appendix): Removed per policy; the paper refers to appendices for full details.

- **"ContPhy comparison is unfair because ContPhy was not adapted faithfully" (Harsh Critic)**: Partially removed. The paper explicitly describes its ContPhy re-implementation in Appendix E.7. The concern that ContPhy may not have been adapted fairly is speculative without access to appendix details. The empirical observation stands; the mechanistic explanation is already noted in the main review as speculative.

- **Causal language "likely due to" as overclaiming (Harsh Critic)**: Weakened to a Minor concern. The paper consistently uses hedged language throughout Section 3.4 and the Conclusion ("likely," "may be attributed to"). The harsh critic's reading of this as a strong unsupported causal claim overstates the issue; the paper's hedged language is appropriate.

---

## Novel Insights

The most novel observation from the synthesis of reviewers is the anomalously large PhysAgent gain on the Relationships domain (GPT-4o: 61.8% → 84.2%, +22.4 pp) relative to other domains (+1.5 to +5.7 pp). This disparity is not discussed in the paper and warrants investigation: if the knowledge memory contains rules that effectively "answer" relationship questions rather than reason through them, the 18.4% headline improvement may partly reflect knowledge shortcutting rather than genuine physical reasoning enhancement. This could be a critical check for the paper's core claim that PhysAgent improves *physical understanding* rather than benchmark gaming. The scaling-doesn't-help finding is also genuinely interesting, and the embodied validation is a differentiating contribution that few benchmark papers attempt—even if the current execution is limited.

---

## Suggestions

1. **Unified evaluation subset**: Create a common evaluation subset that can be scored by image-only, video, and general VLMs under equivalent conditions, or at minimum provide stratified tables separating within-group from across-group comparisons.
2. **Component ablations for PhysAgent**: Add a table isolating the contribution of each module (knowledge memory, each foundation model, task routing, self-verification) on the full 10K test set.
3. **Automatic classification experiment**: Implement and report fully automatic task classification accuracy, and run PhysAgent with automatic-only classification. Disclose which mode was used in reported results.
4. **Investigate the Relationships anomaly**: Analyze why PhysAgent gains 22.4 pp on Relationships but only 1.5–5.1 pp on other domains—this could reveal whether the improvement is a knowledge retrieval shortcut.
5. **Statistical reporting for embodied experiments**: Report number of trials (at minimum), and consider confidence intervals. Even 30 trials per task would substantially strengthen the conclusions.
6. **Text-only baseline**: Add a condition where VLMs see only the question text and choices (no images/video) to quantify language-prior exploitation.
7. **Inter-annotator agreement**: Report κ or percent agreement for the 500-question error taxonomy to establish reliability.

---

## Score and Decision

**Calibration:**

| Paper | Topic | Scores | Decision |
|---|---|---|---|
| Physics-RW | Physical reasoning benchmark for world models | 3,5,5,5,3,5 (avg ~4.3) | Withdrawn/Reject |
| PhyGenBench | Physics benchmark for T2V generation, 160 prompts | 5,5,6,5,5 (avg ~5.2) | Reject |
| Sequential Visual Reasoning | VLM benchmark for predictive reasoning | 5,3,3,5 (avg ~4.0) | Withdrawn/Reject |
| MMBench | Comprehensive VLM benchmark | 6,3,6,6 (avg ~5.25) | Reject |
| VLB (Dynamic Eval) | Novel evaluation method for VLMs | 6,8,8,8 (avg ~7.5) | Accept (Oral) |

**Positioning**: PhysBench is clearly above the Physics-RW and PhyGenBench calibration papers in scope (10K entries vs. 160–300, 75 VLMs evaluated, embodied validation, proposed method). It has more empirical substance than the rejected MMBench (which also showed novel insights) but lacks the methodological elegance and novelty of VLB. The major weaknesses—non-uniform evaluation protocol, missing PhysAgent ablations, unresolved "automatic vs. manual" classification—are real and would normally require revision to resolve, but none are fatal to the core benchmark contribution.

The paper sits in a borderline zone. The benchmark itself is a genuine contribution that will be useful to the community. The PhysAgent contribution is substantially weaker due to missing ablations and the manual classification ambiguity. Weighing the solid benchmark contribution, extensive evaluation, and embodied validation against the methodological gaps in the analysis and method sections, this paper falls slightly below the ICLR acceptance threshold in its current form.

**Originality**: Above average — the benchmark scale and embodied validation are novel.  
**Importance of research question**: High — physical world understanding for VLMs is critical for embodied AI.  
**Claims well supported**: Moderate — the benchmark contribution is well-supported; the PhysAgent and causal claims are not.  
**Soundness of experiments**: Fair — within-group VLM comparisons are sound; cross-group and PhysAgent evaluations have issues.  
**Clarity of writing**: Good — the paper is readable and well-organized.  
**Value to research community**: Good — the benchmark and extensive evaluation will be useful as a resource.

**Score: 5.0** — Borderline reject. The benchmark is a valuable contribution, but the evaluation protocol inconsistency and absent PhysAgent ablations are substantive issues that prevent comfortable acceptance in this form. This places it above the weakest calibration examples (3–4) and consistent with rejected benchmarks of comparable strength (MMBench, PhyGenBench).

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>