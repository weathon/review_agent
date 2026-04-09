## Summary
GUI-Spotlight introduces a 7B-scale multimodal model for GUI visual grounding that iteratively narrows its focus using specialized visual tools (crop, extract, find_color) coordinated via a modified Group Sequence Policy Optimization (GSPO) reinforcement learning algorithm. With only 18.5K training samples, it achieves 52.8% accuracy on ScreenSpot-Pro, surpassing 7B baselines trained on orders of magnitude more data, and approaches the performance of much larger 72B models.

## Strengths
- **Exceptional data efficiency:** The model achieves 52.8% on ScreenSpot-Pro using 18.5K curated samples (2,561 SFT trajectories + 12K RL + 4K high-resolution RL), compared to V2P-7B's 50.6% with 9.6M samples and GTA-1-7B's 50.1% with 1.56M samples. The efficiency gap is two orders of magnitude, which is practically meaningful for the field.
- **Stabilized RL training for multi-turn tool use:** The auxiliary cross-entropy loss term $J'(\theta)$ on format-valid and result-correct samples addresses a real and documented failure mode—vanilla GRPO/GSPO collapses around 300 steps due to tool-call syntax degradation (Figure 3, right panel). This is a genuine methodological contribution backed by empirical evidence, not just a minor tweak.
- **Transparent documentation of negative results:** Section 4 systematically reports what does not work (e.g., continuously updating the reference policy degrades accuracy; dense answer rewards underperform sparse ones; high-uncertainty prompt filtering hurts). This is unusually thorough for the area and provides genuine practical value for future work on agentic RL.

## Weaknesses

### Major:
- **Inference cost is completely unreported, yet central to practical viability:** The model performs multi-turn tool invocations (potentially 3–5+ rounds per query), each requiring full model forward passes and image processing. For GUI agents, latency is a first-class concern—users expect sub-second click responses. The paper never reports average number of tool calls per query, wall-clock time, token cost, or any latency comparison against single-pass baselines. Without this, the claim of a "practical" improvement (Abstract) is unsupported. The accuracy–latency trade-off may well favor single-shot models for real deployment.
- **Domain gap between web-centric training data and native-application benchmarks:** The training data is collected primarily via a Selenium-based headless browser crawling high-traffic websites (Appendix A.4: google.com, youtube.com, amazon.com, etc.). Yet the primary benchmark, ScreenSpot-Pro, evaluates on Creative tools, CAD software, Scientific applications, and Office platforms—domains with fundamentally different visual layouts, icon systems, and interaction patterns than web pages. The UGround dataset also originates from web and mobile sources. The paper claims the model "improves across all six domains" but provides no analysis of how web-trained tool policies transfer to native desktop UIs, nor any ablation separating web vs. desktop training data performance. This gap undermines the generalization claims.

### Minor:
- **No recovery mechanism when the spotlight loses the target:** If an early `extract` or `crop` operation removes the target element from the visible region, all subsequent iterations operate on a wrong sub-image with no possibility of correction. The pipeline (Algorithm 1) has no "zoom-out" or "reset" action. The paper does not quantify how often this cascading failure occurs or discuss it as a limitation.
- **Absence of failure mode analysis:** The paper claims "robustness" to dense, cluttered UIs but never characterizes when or why the method fails. What types of instructions (e.g., ordinal references like "the third icon from the left" requiring global context) or UI configurations cause the iterative approach to break? Without this, the robustness claim is unsupported.
- **Small accuracy margins without statistical significance testing:** The headline improvement of 52.8% vs. 50.6% (V2P-7B) is a 2.2-point gap on a benchmark. No confidence intervals, multiple seeds, or significance tests are reported. While single-run evaluation is common in the field, a gap this narrow could reflect evaluation variance rather than a real improvement.
- **Per-tool ablation missing:** Section 4.2 ablates reward weight ratios (Crop vs. Extract) but never ablates individual tools (e.g., removing `find_color` entirely). It is unclear whether all three tools are necessary or whether the model relies primarily on one. Tool usage frequency and coordination patterns are not analyzed.
- **Mixed evaluation protocols:** Some baseline numbers come from the ScreenSpot-Pro leaderboard, while others (UI-TARS-1.5-7B) are self-evaluated by the authors. Differences in prompts, post-processing, or evaluation code could confound comparisons. This is common practice but should be acknowledged.

### Trivial:
- The reward formula for $S_{BA}$ in Section 3.2.1 is garbled in the PDF, making it hard to verify the exact overlap metric used for bounding-box accuracy filtering.

## Nice-to-Haves
- Report average tool calls per query and accuracy as a function of iteration count to reveal whether gains come from iterative refinement or primarily from the first 1–2 steps.
- Compare against test-time compute baselines (e.g., multiple single-shot predictions with voting) to isolate the value of tool coordination from the value of additional compute.
- Include an SFT-only (no RL stages) ablation to quantify how much RL contributes beyond supervised tool-use warm-up.
- Analyze tool usage patterns (frequency, sequences) to validate that the model learns meaningful coordination rather than relying on one dominant tool.
- Evaluate on end-to-end GUI tasks (beyond single-step grounding) to test whether improved grounding translates to improved agent task completion.

## Removed Points
These points are flagged to be removed; treat them with caution.
- **"Table 3 does not contain a row for GUI-Spotlight"** — This is a PDF parser artifact; the paper's text explicitly references and discusses GUI-Spotlight's per-domain results in Section 5.1.
- **"18.5K training sample count is inconsistent or unclear"** — The stages sum to approximately 2,561 + 12,000 + 4,000 = 18,561, consistent with the stated 18.5K. The breakdown is available in Section 3.2.2.
- **"Qwen2.5-VL-72B dependency for data filtering limits reproducibility"** — This is a reproducibility nitpick about a standard practice (using large models for data curation); removed per hard rules on reproducibility nitpicks.
- **"Ethics statement should address malicious automated UI interaction"** — Scope creep; the paper is about visual grounding, not autonomous agent deployment. Removed per soft rules.
- **"Baseline fairness concern about V2P-7B's base model"** — The comparison already favors V2P-7B (9.6M training data vs. 18.5K); removed per hard rules on unfair comparison complaints where asymmetry favors the baseline.
- **"Demand for deeper analysis of why continuously updating the reference policy fails"** — The paper already documents this as a negative result. Demanding theoretical explanation for a discarded variant is scope creep.

## Novel Insights
The most revealing finding across the reviews is that the paper's core architectural idea—iterative tool-based refinement—and its RL training contribution are conflated in the experimental design. Section 5.4 compares against training-free iterative inference and shows gains, but there is no SFT-only (same tools, no RL) baseline. This means the paper cannot definitively attribute its gains to RL learning of tool coordination versus the inductive bias of the tool interface itself. The training-free comparison in Section 5.4 shows the untrained model has "virtually no multi-step reasoning capacity," suggesting RL is essential—but an SFT-only intermediate would cleanly separate these factors and is a surprising omission for an otherwise thorough empirical section.

## Suggestions
- Add a table reporting inference cost (average tool calls, wall-clock time, and total tokens per query) for GUI-Spotlight versus single-pass 7B baselines. This is the single most important missing piece for evaluating practical impact.
- Add a per-category breakdown of GUI-Spotlight results on ScreenSpot-Pro in the main text (not just the aggregate), and explicitly discuss the web-to-desktop transfer: which categories benefit most from iterative refinement, and does the model struggle on categories least represented in training data?
- Include at least 3–5 representative failure cases showing where the iterative spotlight diverges, with analysis of the root cause (e.g., wrong initial extract, loss of global context, color tool failure on low-contrast themes).