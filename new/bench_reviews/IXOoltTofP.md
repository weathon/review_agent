Now let me search for calibration papers in the review database.Now I have enough information to write the final review.

---

## Summary
3DAxisPrompt is a visual prompting method that overlays a 3D coordinate axis and SAM-generated masks onto multi-view renderings of point clouds to elicit 3D spatial grounding in GPT-4o — without fine-tuning. The paper investigates several visual prompt formats (depth images, multi-view vs. tri-view, 2D/3D marks) and evaluates on four datasets spanning indoor localization, outdoor localization, route planning, robot action prediction, and coarse object generation. The core claim is that this prompting scheme promotes latent 3D reasoning capabilities in GPT-4o.

---

## Strengths

- **Timely and interesting research question.** Whether prompt engineering alone can unlock 3D spatial reasoning in general-purpose MLLMs is an underexplored and practically relevant question, and the paper is one of the first to investigate it systematically.
- **Novel prompt-format investigation (Sec. 3.2–3.3).** The finding that *tri-view* prompting activates 3D reasoning in GPT-4o even *without* text-formatted point clouds, whereas multi-view alone fails (Fig. 2), is a concrete and interesting empirical finding. Similarly, the ablation showing that axis *ticks* are essential (removing them causes failure, while removing only labels causes a ~37% increase in to-bbx error; Fig. 7) is a useful empirical takeaway.
- **Multi-dataset coverage.** Testing across ShapeNet, ScanNet, FMB, and nuScenes and spanning localization, route planning, and robot manipulation exposes where the method works and where it breaks down.
- **Practical design guidance.** The comparison of contour-based marks, 3D edge points, AABB, OBB, and the result that mark + 2D contour outperforms 3D marks (Table 1) is actionable for practitioners.

---

## Weaknesses

### Fatal
*(None that fully invalidate every finding, but the combination of Major issues below effectively prevents the core quantitative claims from being trusted.)*

### Major

1. **Confounding of visual prompting with explicit text-format 3D coordinates.** The paper's headline claim is that 3DAxisPrompt "elicits 3D grounding" via visual prompting, but the main pipeline (Eq. 2) feeds `p^i` — the raw point cloud *in text format* — alongside visual images. The authors explicitly state: *"we consider the point cloud in text format to be an essential input for the model"* (Sec. 3.2). When explicit 3D coordinates are handed to the model as text, it is no longer clear that the visual axis prompt is the source of 3D reasoning; the model may primarily be parsing the coordinate text and the axis is secondary scaffolding. The paper never isolates: how much does performance drop without text-format point cloud? This is the single most important ablation and it is missing. Note that tri-view *does* work without text point clouds (Fig. 2), but tri-view is not the primary evaluated pipeline in Tables 1–2. The paper should restructure to clarify which results involve text point clouds and which do not, and provide the ablation.

2. **Evaluation scale of n=20 is far too small to support any quantitative conclusions.** All reported NRMSEs and success rates are from 20 manually-selected scenes per dataset. For route planning (Table 2), this means 10 trials per route type. There are no confidence intervals, no error bars, no repeated runs, and no control for GPT-4o's stochasticity. Claimed improvements like "7% decline in to-bbx distance" or "19% improvement on to-center distance" (CoT vs. base) are statistically uninterpretable at this sample size. With n=20, even a two-point swing in success rate represents a 10% absolute change. The paper should significantly expand the evaluation or restrict its claims to "exploratory."

3. **No baselines in quantitative tables — only self-comparisons.** Table 1 compares only variants of 3DAxisPrompt against each other; there is no GPT-4o without any visual prompt, no GPT-4o with text point cloud only (no axis), and no non-MLLM heuristic reference. Table 2 is the same. Figure 4 provides only a qualitative, cherry-picked comparison. Without these controls, it is impossible to know whether the axis prompt itself, the SAM contours, or simply the provision of point cloud text is responsible for the performance — a fundamental attribution problem.

4. **CoT condition adds non-standard extra information, confounding its interpretation.** The best-performing condition in Table 1 is "Mark+2D contour (colors) + CoT," but the CoT in this work adds *"the additional coordinate of a nearby object"* as part of the prompt (Sec. 4.2). This is additional supervision (a reference anchor), not standard chain-of-thought reasoning scaffolding. The 19% improvement therefore conflates prompt format with extra ground-truth-derived hints. This result cannot be used as evidence for the value of the visual prompt alone.

5. **Several tasks have vague or absent quantitative evaluation.** ShapeNet coarse object generation (Fig. 6) is presented entirely qualitatively with no metrics — it is impossible to assess whether the predictions are accurate. Robot action prediction success is judged by "whether the orders can complete the mission" (Sec. 4.2) without any simulator, no physical execution, and no operational definition of success. These sections inflate the apparent scope of the contribution without supplying commensurate evidence.

### Minor

6. **NRMSE metric is non-standard and has notation problems.** Eq. (3) normalizes by `max(x_i)`, where `x_i` is first used as a position vector in `D(x̂_i, x_i)` and then as a scalar in `max(x_i)`. Per-scene normalization by the maximum coordinate value makes errors incomparable across scenes of different scale. A fixed-distance threshold (e.g., % of predictions within 0.5 m) would be more interpretable and comparable across datasets.

7. **Experiments are run manually through the ChatGPT web interface** (Sec. 4.1: "exhaustively send 3DAxisPrompt-augmented images to the ChatGPT interface… opening a new chat window for each scene"). Model version, temperature, decoding settings, and exact prompt templates are not specified. This precludes reproducibility and makes it impossible to rule out session-to-session variation.

8. **Outdoor localization is restricted to two coarse, frequently occurring object types** (vehicles and vegetation). This is much easier than general outdoor localization and should not be generalized beyond its narrow scope.

### Trivial

9. **The "random selection" of 20 scenes is described simultaneously as aimed to "cover as many diverse scenes as possible"** (Sec. 4.1) — random selection and diversity-seeking are in tension, and the protocol is ambiguous.

---

## Nice-to-Haves

- Probing experiments on deliberately perturbed axes (offset origin, wrongly scaled ticks, rotated frame) would validate whether GPT-4o is genuinely reading the axis or using other visual cues.
- Testing on at least one open-source MLLM (e.g., LLaVA, Qwen-VL) would clarify whether the findings generalize beyond GPT-4o.
- Scatter plots of predicted vs. ground-truth 3D positions across scenes would reveal whether localization errors are systematic (e.g., depth underestimation) or random.
- Failure case analysis, particularly for dense-object scenarios, would provide more nuanced insights than just noting that the 70% success rate for "door to desk" is lower.

---

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **Harsh Critic: "Evaluation cannot support the claim about visual prompting because all performance comes from text-format point clouds."** — Partially removed/weakened. The paper is transparent that text point cloud is an essential input. More importantly, Figure 2 (case 4) demonstrates that tri-view *without* text point clouds does produce 3D coordinates, so the claim is not entirely vacuous. However, the missing ablation on the text point cloud contribution is retained as a Major weakness because the main quantitative results (Tables 1–2) do not clarify which pipeline was used.

- **Neutral Reviewer / Spark: Reproducibility concerns about proprietary API, undisclosed hyperparameters.** — Removed per hard rules on reproducibility nitpicks. The manual interface concern is kept only at the Minor level because it affects result validity, not just reproducibility.

- **Spark: "No comparison with any existing 3D grounding method (VoteNet, 3DETR)."** — Weakened. This paper is a prompting study on a zero-shot model, not a competition with specialized fine-tuned models. A comparison against methods that require full training is outside the paper's stated scope. However, a no-prompt GPT-4o baseline IS within scope and is kept as a Major weakness.

- **Human Finder: "SAM quality and occlusion issues not analyzed."** — Weakened to a nice-to-have. This is a secondary concern; the paper acknowledges occlusion issues in tri-view (Sec. 3.2) and pivots to multi-view with contours for the main method.

- **Human Finder / Neutral: "Cost and practicality concerns."** — Removed. Cost is not a scientific flaw and the paper's scope is exploratory proof-of-concept, not deployment.

---

## Novel Insights

The most genuinely novel empirical finding in this paper is the discrete behavioral threshold between multi-view and tri-view prompting: multi-view (arbitrary viewing angles) fails to elicit 3D coordinate reasoning in GPT-4o even with a 3D axis overlay, while tri-view (axis-aligned XY/ZX/YZ projections) succeeds even without text-format point cloud input. This suggests that GPT-4o has a strong inductive bias toward orthographic decomposition of 3D space, consistent with the tri-plane representations prevalent in its likely training data, and implies that prompt design for 3D MLLMs should be aligned with canonical coordinate frames rather than arbitrary views. The ablation showing axis *ticks* (not just axis lines or labels) are the functionally essential component supports the interpretation that GPT-4o uses the axis as a literal ruler rather than a directional reference.

---

## Suggestions

1. **Add the critical ablation**: present results for (a) text-format point cloud only, no visual prompt; (b) 3DAxisPrompt visual prompt only, no text point cloud; (c) the full method. Without this, the paper cannot support its central claim.
2. **Expand evaluation to at least 100 scenes per dataset**, ideally using the API (not manual interface) with fixed temperature and model version. Report standard deviations.
3. **Add a no-prompt GPT-4o baseline row in Table 1**, showing raw GPT-4o performance on the same 20 scenes.
4. **Disentangle CoT from anchor hint**: run the CoT condition both with and without the nearby-object coordinate hint to separate reasoning scaffolding from information augmentation.
5. **Provide quantitative evaluation for ShapeNet generation** (e.g., chamfer distance between predicted keypoint skeleton and ground-truth).
6. **Specify clearly in Table 2 captions** which results use text-format point cloud and which use tri-view without it.

---

## Score and Decision

**Calibration:**

| Paper | Decision | Scores | Relevant similarity |
|---|---|---|---|
| Coarse Correspondences (8ibaVk4mU8) | Reject (withdrawn) | 6, 5, 3 | Most similar — same GPT-4o + visual prompting for 3D, same task domain. But tested on established benchmarks (ScanQA, OpenEQA) with full-scale evaluation and open-source models. Substantially stronger. |
| CUBE-LLM (yaQbTAD2JJ) | Accept | 6, 6, 6, 6 | Similar topic (MLLM 3D understanding) but full training, large-scale, proper baselines. Far stronger. |
| GPT-4o vision evaluation (h3unlS2VWz) | Reject | 5, 5, 6, 6 | Also an exploratory evaluation paper on GPT-4o, but with multiple models, established datasets, and comprehensive coverage. Stronger. |
| GeVLM (7nWKBRQuLT) | Reject | 3, 5, 5, 5 | Similar score band — limited novelty, weak improvements, missing baselines. |
| Zoomer (SOVwGa0H2c) | Reject | 3, 6, 3, 3, 5 | Visual prompting method for MLLMs with limited novelty. |

This paper is *weaker* than Coarse Correspondences (which scored avg ~4.7 but had proper benchmarks and scale) and significantly weaker than the h3unlS2VWz GPT-4o evaluation paper (which at least had multiple models and full-scale evaluation). The combination of n=20 evaluation, no baselines, confounded main claim, and partially missing quantitative results puts it in the 3–3.5 range, consistent with the GeVLM and Zoomer cluster of rejections. The research question and tri-view insight give it some value over a purely negative result, placing it at **3.5** rather than 3.

**Assessment across axes:**
- *Originality*: Moderate — embedding a 3D coordinate axis is simple but the systematic investigation of visual prompt formats, especially the tri-view finding, is novel.
- *Importance of research question*: High — 3D spatial grounding in MLLMs matters.
- *Claims well-supported*: Poor — n=20, no baselines, confounded ablations.
- *Soundness of experiments*: Poor — manual interface, missing uncertainty, no proper controls.
- *Clarity of writing*: Adequate — the paper reads clearly but glosses over the text-point-cloud confound.
- *Value to community*: Low-to-moderate — the findings may inspire follow-up but are not reliable enough to build on as-is.

**Final verdict**: The ideas are interesting and the prompt-investigation aspect has merit, but the experimental evidence is too thin and too methodologically compromised to support publication at a top venue. The paper reads as a preliminary exploration that needs substantially more rigorous evaluation.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>