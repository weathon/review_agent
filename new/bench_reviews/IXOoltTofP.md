Now I have a good understanding of the calibration landscape. Let me synthesize my final review.

## Summary
This paper introduces 3DAxisPrompt, a visual prompting scheme that overlays 3D coordinate axes and SAM-generated masks with alphanumeric marks onto multi-view rendered images of point cloud scenes to elicit 3D spatial grounding and reasoning from GPT-4o without fine-tuning. The authors systematically investigate various visual prompt formats (depth maps, tri-view vs. multi-view rendering, 2D vs. 3D mark types) and evaluate on four datasets (ShapeNet, ScanNet, FMB, nuScenes) covering indoor/outdoor localization, route planning, and robot action prediction, claiming their approach extends MLLMs' 2D grounding capabilities to 3D scenarios.

## Strengths

- **Systematic investigation of prompt formats**: The paper provides a thorough empirical study of how different visual cues (3D axis, depth images, multi-view vs. tri-view images, 2D vs. 3D marks, axis ticks vs. labels) affect GPT-4o's 3D reasoning, yielding actionable insights such as the importance of axis ticks, object contours, and the failure of depth maps alone. This investigation is novel and useful for the community.

- **Novel prompting idea**: Embedding a 3D coordinate axis directly into rendered point cloud views to provide a geometric reference frame is an intuitive and training-free approach that avoids architectural changes. The finding that tri-view prompts can activate spatial reasoning without text point clouds (Section 3.2) is interesting.

- **Multi-task evaluation across diverse 3D scenarios**: Testing across four datasets covering indoor localization, outdoor localization, route planning, and robot action prediction demonstrates the breadth of applicability. The qualitative figures (especially Figures 2, 4, 5) effectively illustrate the behavioral differences between prompted and unprompted GPT-4o.

- **Clear and honest framing**: The authors acknowledge that their "objective is not to achieve perfect zero-shot performance" but to explore limitations and potential. The ablation on axis elements (ticks, labels) is informative and well-designed.

## Weaknesses

### Fatal

None.

### Major

- **Ambiguous definition of "3D grounding" that conflates coordinate reading with 3D reasoning**: The paper claims to elicit "3D grounding and reasoning" in GPT-4o, but a core mechanism is drawing a 3D axis with numeric tick labels into rendered images and asking the model to "read off" positions. Many outputs (e.g., Figure 1: "X = 0.512, Y = 0.608, Z = -0.447") read like estimates aligned to the axis grid. The paper does not cleanly distinguish between (a) the model interpreting a drawn ruler and outputting plausible coordinates in that frame (a 2D visual-reading ability), and (b) the model constructing a genuine 3D spatial representation from multi-view reasoning. The fact that tri-view images "work" without point cloud text input (Section 3.2) actually suggests the model may be matching 2D projections across views rather than performing 3D inference. If "3D grounding" is defined loosely enough that reading coordinate tick marks qualifies, the contribution is largely demonstrating that GPT-4o can do OCR on axis labels, which is a much weaker claim.

- **Very small evaluation scale with no strong baselines**: Only 20 randomly selected scenes per dataset are used (Section 4.1), with no confidence intervals, variance reporting, or statistical significance tests. More critically, there are no meaningful baselines: the paper compares only variants of its own prompt format (Table 1) but does not compare against any existing 3D prompting method (e.g., PIVOT, Coarse Correspondences), specialized 3D models (e.g., PointLLM), or even simple heuristics (e.g., always predicting scene centroid, or an oracle that reads ground-truth axis coordinates). Without such baselines, the NRMSE of ~0.27 for indoor localization and the 79% route-planning success rate are uninterpretable — we cannot tell whether the method actually outperforms trivial alternatives.

- **Missing ablation on the text-formatted point cloud input**: The paper states in Section 3.2 that "we consider the point cloud in text format to be an essential input for the model" and that point cloud text combined with 3D Axis elicits effective reasoning. Yet no quantitative ablation removes this component, making it impossible to determine whether the 3D axis visual prompt or the text point cloud drives performance. This is a critical gap for understanding the method's mechanism.

### Minor

- **Limited generalizability claims with single-model evaluation**: All experiments are conducted exclusively on GPT-4o. While the paper's scope is explicitly about this model, broader claims about "MLLMs for 3D understanding" in the abstract and conclusion are not supported. This should be framed as findings about GPT-4o specifically.

- **Under-specified evaluation for route planning and robot action prediction**: Route planning (Table 2) uses a binary success rate without precisely defining "success" (does reaching the destination by any path count, or must the path be collision-free?). Robot action prediction evaluates grasp/release on FMB with an unspecified number of trials and vague success criteria. These make the reported numbers hard to interpret.

- **NRMSE metric definition ambiguity**: The normalization term `max(x_i)` in Equation 3 is unclear — whether it is per-scene, per-axis, or global affects interpretation significantly, especially given the small sample size.

### Trivial

- The coarse object generation experiment on ShapeNet (Figure 6) is presented qualitatively only, with no quantitative evaluation. It adds little to the paper's core claims.

## Nice-to-Haves

- Test on additional open-source MLLMs (e.g., LLaVA, InternVL) to assess whether the prompting approach transfers, which would dramatically increase the community impact of the findings.
- Report scatter plots of predicted vs. ground-truth 3D coordinates and per-axis error breakdowns to reveal systematic biases.
- Add a simple heuristic baseline (e.g., scene centroid prediction) to contextualize the absolute NRMSE numbers.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Not yet released" or availability concerns about models/tools**: Several reviewer comments raised concerns about GPT-4o being proprietary/closed-source or API quota limits affecting reproducibility. Per the rules, we assume cited models and tools exist and are available. The ChatGPT interface usage (rather than API) is a legitimate methodological concern that I've kept in minor weaknesses, but the general "closed-source model" complaint is removed since the paper explicitly scopes around GPT-4o.

- **Demand for comparisons with specialized 3D models (PointLLM, 3D-LLM, SpatialVLM)**: These are fundamentally different approaches that require task-specific fine-tuning or architectural changes, while this paper explicitly proposes a training-free prompting method. Comparing against methods that require retraining would be an asymmetric comparison. The paper does fairly compare different prompt variants against each other.

- **Formatting and style nitpicks**: Removed per rules.

- **Claims about "overclaiming" that ignore the paper's own disclaimers**: The paper explicitly states "our objective is not to achieve perfect zero-shot performance with GPT-4o, but to explore its limitations and potential" (Section 1). While some wording in the abstract could be toned down ("effectively perceive" is strong), the paper does acknowledge its exploratory nature.

- **Reproducibility concerns about hyperparameters/API parameters**: The paper acknowledges using the ChatGPT interface and explains its divide-and-conquer strategy. While the lack of temperature/sampling parameters is a minor concern, demanding complete training logs for API-based experiments is impractical per the rules.

## Novel Insights

The most striking finding is the asymmetry between tri-view and multi-view prompting: tri-view images (XY, ZX, YZ projections) successfully provoke coordinate reasoning even without text point cloud input, while arbitrary multi-view angles fail (Figure 2). This suggests GPT-4o has some latent capacity to associate canonical planar projections with 3D coordinates, but cannot decompose arbitrary camera views into 3D structure — a meaningful characterization of the model's spatial reasoning boundary.

## Suggestions

- **Add a minimal baseline**: Compute the NRMSE for a "scene centroid predictor" or "nearest object center predictor" on the same 20 ScanNet scenes. This costs almost nothing and immediately contextualizes whether your ~0.27 NRMSE is meaningful.
- **Run an ablation without text point cloud input** on the localization task to quantify its contribution vs. the visual prompt alone. This is the most critical missing experiment.
- **Add a scatter plot** of predicted vs. ground-truth coordinates across all test instances, which would immediately reveal whether the model is systematically biased, scale-miscalibrated, or approximately calibrated near the origin.
- **Temper the abstract/conclusion claims**: Replace "can effectively perceive an object's 3D position" with "can produce approximate 3D coordinate estimates from annotated views" or similar, to more accurately reflect what the evidence shows.

## Score and Decision

**Calibration comparison**: I compared against several papers:

- **"On Inherent 3D Reasoning of VLMs"** (scores 3,3,5,5 → Reject): Similar topic of evaluating VLMs' 3D reasoning, but had a smaller evaluation (25 QAs per task), overclaimed 3D reasoning when it was actually 2D, and lacked strong baselines. This paper has similar issues (small evaluation, ambiguous definition of "3D grounding") but does offer a more systematic investigation of prompt designs.

- **"Coarse Correspondences"** (scores 6,5,3 → Withdrawn/Reject): A training-free prompting method for 3D understanding in MLLMs with actual quantitative improvements (+20.5% on ScanQA, +9.7% on OpenEQA). Despite being a similar prompting approach, it had more substantial baselines and quantitative results. This paper is weaker because it lacks comparable baselines and has a less rigorous evaluation.

- **"Does Spatial Cognition Emerge in Frontier Models?"** (scores 6,8,8,5 → Accept Poster): A comprehensive benchmark with systematic evaluation across many tasks, human baselines, and multiple models. Much more rigorous than this paper.

- **"Language-Image Models with 3D Understanding"** (scores 6,6,6,6 → Accept Poster): A trained model (CUBE-LLM) with actual SOTA comparisons (-21.3 points improvement). Much stronger methodology.

- **"How well does GPT-4o understand vision?"** (scores 5,5,6,6 → Withdrawn/Reject): Evaluates GPT-4o on vision tasks with prompt engineering. Similar pattern — evaluation of closed-source models with prompting — but with more tasks and baselines. Still received moderate-low scores.

This paper sits below "Coarse Correspondences" (which had real baselines and stronger results but still was withdrawn/rejected) and below "Does Spatial Cognition Emerge" (which had comprehensive benchmarking). It has some value as an exploratory study, but the combination of (1) very small evaluation sets, (2) no meaningful baselines, (3) the conceptual ambiguity about what "3D grounding" means, and (4) the missing text-point-cloud ablation significantly undermines the claims. The systematic prompt investigation is a genuine contribution, but it's not sufficiently supported by rigorous evaluation.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>