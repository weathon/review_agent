I now have all the data I need. The critic's claim #3 is factually wrong — Mobile SR drops from ~28% to ~17% (not ~28%→~28%) when removing the unified action space. Let me write the consolidated review.

## Summary

OS-Atlas presents a foundation action model for generalist GUI agents, with two main contributions: (1) a multi-platform GUI grounding data synthesis toolkit producing the largest open-source cross-platform grounding corpus (13.58M elements across 5 platforms, including previously unavailable desktop data), and (2) a unified action space that resolves cross-platform naming conflicts during multi-task fine-tuning. The model is evaluated across six benchmarks spanning mobile, desktop, and web under both zero-shot OOD and supervised fine-tuning settings, demonstrating strong performance that the authors argue makes it a viable open-source alternative to commercial VLMs like GPT-4o.

## Strengths

- **Largest open-source cross-platform GUI grounding corpus with unique desktop coverage**: Table 1 shows OS-Atlas provides 13.58M elements across 2.3M screenshots, uniquely covering desktop (54K screenshots) absent from all prior open-source datasets. The ablation in Figure 4 proves this is critical: removing desktop data collapses mobile Icon/Widget performance from 58.52% to 10.04%.

- **Strong grounding accuracy improvements over prior SOTA**: OS-Atlas-Base-7B achieves 82.47% average grounding accuracy on ScreenSpot (Table 2, standard setting), an 8.3% absolute improvement over UGround-7B (74.15%). Even the 4B model (70.13%) is competitive with the 7B UGround.

- **Clear and significant zero-shot OOD gains over GPT-4o on SR**: Tables 4–5 show OS-Atlas-7B consistently outperforms GPT-4o on step success rate across all six OOD benchmarks — e.g., GUI-Odyssey SR: 26.96% vs 5.36%, GUI-Act-Web SR: 57.02% vs 41.84%, AndroidControl-Low SR: 50.94% vs 28.39%.

- **Unified action space yields substantial and measurable gains**: Section 3.3 standardizes conflicting actions (tap→click, press_home→home, type→input), reducing action types from 17 to 10. Figure 5 validates this: removing the unified action space drops mobile SR from ~28% to ~17% (11pp) and mobile grounding accuracy from ~51% to ~42% (9pp), confirming that resolving action naming conflicts genuinely benefits multi-platform training.

- **ScreenSpot-V2 benchmark correction**: Identifying and correcting 11.32% annotation errors in a widely-used benchmark (Section 4.1) is a valuable community contribution improving evaluation reliability.

- **Effective as a plug-in grounding module**: Table 3 shows replacing GPT-4o's internal grounding with OS-Atlas-Base-7B in OSWorld improves success rate from 5.03% to 14.63%, outperforming SeeClick (9.21%) by a clear margin.

- **Clear two-stage training pipeline with convincing ablation**: The separation into GUI Grounding Pre-training and Action Fine-tuning is well-motivated; Figure 5 shows removing pre-training collapses desktop SR from ~28% to ~10%, validating its necessity.

## Weaknesses

### Fatal
None.

### Major

- **Insufficient transparency about GPT-4o zero-shot OOD baseline setup**: The paper claims OS-Atlas "demonstrated superior capabilities in addressing unseen tasks across all six OOD evaluation datasets" compared to GPT-4o (Section 5.2). However, Section 5.1 provides no details about the GPT-4o prompt, output format specification, action space description, or retry/parsing mechanism. Since OS-Atlas is specifically trained to produce actions in a defined format while GPT-4o operates zero-shot, this comparison has a structural asymmetry: OS-Atlas benefits from format-specific training that GPT-4o does not receive. The claim of being "an open-source alternative to powerful commercial VLMs" is directionally supported by the SR results, but its strength is uncertain without demonstrating that GPT-4o was given equivalent formatting instructions. The paper should provide the exact GPT-4o prompt or acknowledge this confound explicitly.

- **Overclaimed "superior capabilities across all six" OOD datasets**: While OS-Atlas-7B wins on SR (the most comprehensive metric) across all six datasets, GPT-4o still outperforms on specific individual metrics — Type accuracy on AndroidControl-Low (74.33% vs 73.00%), Type on AndroidControl-High (63.06% vs 57.44%), and Grounding on OmniAct-Desktop (63.25% vs 62.87%). The blanket claim of "superior capabilities" without specifying it applies to SR only is imprecise and could mislead readers. The paper should qualify this claim to specify which metric shows superiority, or acknowledge the metrics where GPT-4o remains competitive.

### Minor

- **OS-Atlas-Pro evaluation includes training data overlap**: Section 5.4 introduces OS-Atlas-Pro trained on all 7 agent datasets and evaluates on the same benchmark domains. This means results on AndroidControl, GUI-Act-Web, OmniAct-Desktop, etc. are no longer OOD — they are in-sample. The paper does not adequately flag this distinction, and readers could easily misinterpret these results as comparable to the OOD evaluation of the base OS-Atlas model.

- **Web data filtering threshold lacks justification**: Section 3.2 restricts webpages to a maximum of 10 elements "to encourage diversity," but provides no rationale for this specific threshold. Real webpages often have many more actionable elements, and this cutoff biases the dataset toward simpler pages, potentially limiting grounding capability on complex interfaces.

- **Absolute performance levels temper practical viability claims**: On OSWorld (Table 3), the best OS-Atlas configuration achieves 14.63% success rate vs. 72.36% for humans. On multiple OOD benchmarks, SR is ~30–57%. While these represent improvements over baselines, the gap from human-level performance is large, and the paper's framing of OS-Atlas as a "foundation" for agents should acknowledge this gap more prominently.

- **Missing fine-tuning sample counts per dataset**: Section 5.1 uses three agent datasets (AMEX, AITZ, Mind2Web) but does not report how many samples come from each, making it impossible to assess data balance or the relative contribution of each dataset.

### Trivial
None.

## Nice-to-Haves

- A controlled ablation randomly reducing the action space from 17 to 10 types (not via semantic unification) would further isolate whether the benefit comes from resolving naming conflicts specifically or simply from having a smaller action vocabulary.

- Error analysis on zero-shot OOD tasks showing which action types or interface categories OS-Atlas fails on would strengthen understanding of generalization limits.

- Qualitative examples showing OS-Atlas success and failure cases on desktop (the platform with no action fine-tuning data) would illuminate whether the model truly understands desktop interfaces or transfers low-level visual patterns.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic Claim #3: "Negligible improvement from unified action space on mobile SR (~28% → ~28%)"** — This is factually wrong. Figure 5 data clearly shows Mobile SR drops from ~28% (OS-Atlas-4B) to ~17% (w/o Unify Action), an 11pp drop. The critic confused w/o Pre-training (mobile SR ~26%) with w/o Unify Action (mobile SR ~17%). The unified action space has its LARGEST effect on mobile, not the smallest. This error invalidates the critic's entire argument that the unified action space contribution is weak.

- **Harsh Critic Claim #3: "Desktop has only one dataset so conflicts are irrelevant"** — This misunderstands the experimental setup. The model is fine-tuned simultaneously on web AND mobile data (3 datasets total: AMEX, AITZ, Mind2Web). Cross-platform action naming conflicts exist between these training datasets, not within a single platform. The unified action space resolves conflicts across the training data, which benefits performance across all platforms including desktop at inference time.

- **Harsh Critic: "Unfair GPT-4o baseline invalidates the paper's most prominent claim"** — While the lack of prompt details is a legitimate concern (retained as a Major weakness), the claim that this "invalidates" the paper is overblown. The comparison shows a specialist fine-tuned model outperforming a general-purpose zero-shot model, which is the relevant comparison for practitioners deciding which tool to use. The concern warrants transparency, not invalidation.

- **Harsh Critic: "Comparison is confounded by format familiarity"** — This overstates the issue. The format asymmetry is a real concern but SR requires BOTH correct action type AND correct coordinates, so format familiarity alone cannot explain the large SR gaps (e.g., GUI-Odyssey: 26.96% vs 5.36%).

- **Harsh Critic: "UGround comparison deserves scrutiny / OS-Atlas-Base-4B underperforms UGround-7B in Grounding Mode"** — Comparing a 4B model against a 7B model and finding it underperforms is expected and not a weakness. The 7B variant outperforms UGround-7B on most metrics.

- **Harsh Critic: "14.63% success on OSWorld undermines foundation claims"** — This is a minor point, not a major one. The paper frames OS-Atlas as a foundation for building agents, not as a finished agent. Low absolute performance on a challenging benchmark is consistent with this framing and the paper already notes the gap vs. human performance.

- **Strength Finder: "Figure 4 proves desktop data critical — removing desktop data collapses mobile performance from 85.71%/58.52% to 10.04%/10.04%"** — This is a misread. Figure 4 shows that removing BOTH mobile AND desktop data (web-only pre-training) collapses mobile performance. Removing just desktop data is not separately shown for mobile. The more accurate claim is that cross-platform pre-training is essential.

- **Harsh Critic: "Data contamination between FineWeb pre-training and evaluation benchmarks"** — The paper already takes steps to avoid data contamination, noting in footnote 1: "We remove entries from ScreenSpot, Mind2Web, and Omniact to avoid data contamination in downstream evaluation." This concern is already addressed.

- **Harsh Critic: "Number of fine-tuning samples not reported"** — Retained as a minor point, but the demand for exact counts as a "significant omission" is overstated for a paper centered on data contributions where the pre-training corpus statistics are clearly reported.

## Novel Insights

The paper reveals an important asymmetry in GUI agent capabilities: grounding accuracy (predicting coordinates) improves dramatically with multi-platform pre-training data, but action type prediction (choosing the right action verb) is where commercial VLMs like GPT-4o retain an edge — likely because semantic understanding of task instructions requires different capabilities than spatial localization. The unified action space's largest effect is on mobile SR (11pp), suggesting that naming conflicts are most damaging when the model must bridge the largest conceptual gap between platforms (mobile actions having the most distinct naming conventions from desktop/web conventions). This finding hints that action vocabulary alignment may be as important as visual pre-training for cross-platform generalization.

## Suggestions

- Provide the exact GPT-4o prompt and format instructions used in zero-shot OOD evaluation, or explicitly acknowledge the structural asymmetry in the comparison and qualify the "alternative to GPT-4o" claim accordingly.

- Qualify the "superior capabilities across all six OOD datasets" claim to specify it applies to SR, and acknowledge metrics where GPT-4o remains competitive.

- Clearly flag OS-Atlas-Pro results as in-sample evaluation rather than OOD, to avoid confusion with the base model's OOD results.

## Calibration

| Anchor Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| UGround (GUI grounding, concurrent work) | kxnoqaisCT.md | 7.75 (Oral) | OS-Atlas is comparable in scope — similar grounding data contribution, but OS-Atlas covers more platforms (including desktop) and evaluates action fine-tuning. UGround has cleaner experimental framing. OS-Atlas is slightly below due to comparison transparency issues. |
| WebAgent (GUI agent) | 9JQtrumvg8.md | 7.25 (Oral) | OS-Atlas has a stronger data contribution and broader evaluation, but WebAgent has more novel methodology. Comparable quality. |
| GUI-World (GUI dataset/benchmark) | QarKTT5brZ.md | 6.25 (Poster) | OS-Atlas has stronger empirical results and a more actionable model contribution. Clearly above this. |
| Octopus (open-source VLM alternative to GPT-4) | VUA9LSmC2r.md | 4.0 (Reject) | Octopus was rejected for overclaimed results and weak baselines. OS-Atlas has much stronger empirical support and a genuine data contribution. Far above this. |
| OmniParser (GUI grounding for GPT-4V) | C6hUK6Q1Pi.md | 5.0 (Reject) | Rejected for limited novelty and insufficient analysis. OS-Atlas has a more substantial contribution. Clearly above this. |
| Harry Potter OOD (overclaimed) | 3ZdGSTxKuy.md | 2.0 (Withdrawn) | Grossly overclaimed with minimal contribution. OS-Atlas has real data and results. Not comparable. |

OS-Atlas sits between UGround (7.75) and GUI-World (6.25), closer to UGround. The data contribution is substantial and genuinely multi-platform, the evaluation is comprehensive, and the unified action space is well-motivated with strong ablation support. The main weakness is the GPT-4o comparison transparency, which is a real concern but does not invalidate the results. This warrants a score in the 6.5–7.0 range.

## Score and Decision

**Originality**: The multi-platform data synthesis toolkit and unified action space are solid contributions. The data synthesis infrastructure across 5 platforms (especially desktop) is novel. The unified action space is a practical but incremental methodological contribution.

**Importance of research question**: High — the gap between open-source and commercial VLMs for GUI agents is a significant barrier, and large-scale multi-platform grounding data is a real need.

**Claims support**: Mostly well-supported. The grounding results are strong. The zero-shot OOD comparison with GPT-4o would be more convincing with full prompt disclosure. The unified action space ablation provides clear evidence (11pp mobile SR drop), contrary to the harsh critic's misreading.

**Experimental soundness**: Comprehensive — six benchmarks, three platforms, two settings, clear ablations. The OS-Atlas-Pro evaluation overlapping with training data is a minor blemish.

**Clarity**: Well-structured with clear two-stage pipeline, three-mode design, and informative figures. Some claims could be more precisely stated.

**Community value**: High — the open-source toolkit, dataset, ScreenSpot-V2 correction, and model all provide tangible resources for the community.

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>