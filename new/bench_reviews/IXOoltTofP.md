The paper content is provided in the user message itself. Let me re-examine key sections carefully.

From the paper:
- Section 4.1: "Our method does not require model training. However, due to the limited and costly GPT-4o API quota, we must exhaustively send 3DAxisPrompt-augmented images to the ChatGPT interface."
- Section 3.1: The equation includes p^i (text-formatted point cloud) as an input alongside the visual prompts.
- Section 3.2: "we discovered that when GPT-4o is challenged with both the point cloud p^i provided in text format and visual prompts, it can recognize the text file as the point cloud p^i and reason about spatial positions" and "we consider the point cloud in text format to be an essential input for the model"
- The ablation study (Section 4.3) only ablates number of images and axis elements (ticks, labels). There is NO ablation for: (a) removing the 3D axis entirely, (b) removing the text-formatted point cloud, (c) comparing against GPT-4o without any prompts.
- Table 1 compares various prompt element combinations but doesn't have a "no prompt" or "no axis" baseline.
- Table 2 has no baseline comparison at all.

This confirms the harsh reviewer's concern about missing baselines is valid and substantive.

Now let me compose the final review.

## Calibration Summary

Papers with similar topic/methodology:
- **Coarse Correspondences** (8ibaVk4mU8): Visual prompting for 3D spacetime understanding in MLLMs. Scores: 6, 5, 3. Withdrawn → Reject. This paper had actual quantitative improvements (+20.5% on ScanQA, etc.) and compared against baselines, but was still rejected/withdrawn.

- **On Inherent 3D Reasoning of VLMs** (uBhqll8pw1): Evaluating VLMs on 3D spatial reasoning. Scores: 3, 3, 5, 5. Reject. Evaluated 5 models, but overclaimed 3D capabilities in 2D tasks.

- **GPT-4V geometry prompting** (0vKokoPKTo): Empirical study of what matters in prompting GPT-4V for geometry. Scores: 3, 5, 3, 3. Withdrawn → Reject. "Almost zero algorithmic/technical contribution."

- **GPT-4o CV tasks** (h3unlS2VWz): Prompt chaining for CV tasks with GPT-4o. Scores: 5, 5, 6, 6. Withdrawn → Reject. Similar prompting-for-capabilities evaluation pattern.

- **SPACE benchmark** (WK6K1FMEQ1): Spatial cognition benchmark for frontier models. Scores: 6, 8, 8, 5. Accept Poster. This paper had comprehensive evaluation, human baselines, and meaningful findings.

- **CUBE-LLM / Language-Image Models with 3D Understanding** (yaQbTAD2JJ): Fine-tuning LLMs for 3D with real baselines. Scores: 6, 6, 6, 6. Accept Poster.

The current paper is most similar to the "prompting GPT-4o for 3D" family that received scores in the 3-6 range. It has weaker evaluation than most of these (only 20 scenes, no baselines). I'll calibrate around 3-4.

Now let me write the consolidated review.</think>## Summary

The paper proposes 3DAxisPrompt, a visual prompting method that overlays 3D coordinate axes and SAM-generated object masks/contours onto multi-view rendered images of 3D scenes to elicit 3D grounding and reasoning capabilities from GPT-4o without fine-tuning. Through systematic investigation of prompt formats (axis types, depth images, multi-view vs. tri-view, 2D vs. 3D marks, chain-of-thought), the paper identifies key enabling factors—axis ticks and object contours—and evaluates across four datasets (ShapeNet, ScanNet, FMB, nuScenes) on localization, route planning, robot action prediction, and coarse object generation.

## Strengths

- **Novel and important question**: The paper targets a genuinely underexplored question—whether and how MLLMs can be prompted into 3D spatial reasoning without fine-tuning. This is a timely contribution as MLLMs are increasingly deployed in embodied and spatial settings.

- **Systematic design-space exploration**: The investigation across multiple prompt formats (Section 3.2–3.3)—axis vs. depth images, multi-view vs. tri-view, 2D vs. 3D marks, various bounding-box and contour types—provides useful empirical insights. The finding that tri-view images can provoke 3D reasoning while multi-view images alone cannot (Figure 2), and that depth images fail as a depth compensation strategy, are tangible contributions for practitioners designing spatial prompts.

- **Clear ablation findings**: The axis-element ablation (Section 4.3) showing that removing axis ticks causes complete failure and removing labels increases bbox error by 37% is a useful diagnostic result. The finding that 2D contours outperform 3D edge points underscores the importance of clear object boundaries.

- **Diverse task coverage**: Evaluating across indoor localization, outdoor localization, route planning, robot action prediction, and coarse object generation demonstrates breadth, even if the depth of evaluation is limited.

## Weaknesses

### Major:

- **No baseline comparison isolates the effect of 3DAxisPrompt.** The headline quantitative results (Tables 1 and 2) are presented entirely in isolation. There is no comparison against: (a) GPT-4o with the same rendered images but *without* 3D axis overlays; (b) GPT-4o with simpler 2D prompts (e.g., SoM, bounding boxes only); (c) trivial coordinate-reading baselines (e.g., a script that maps pixel positions to coordinates given the known rendering). Without any such baselines, it is impossible to determine whether 3DAxisPrompt itself is responsible for the reported performance or whether GPT-4o already achieves similar results from raw scene images. The paper states "no previous work has presented localization errors related to 3D spatial grounding" to justify the absence of baselines, but this does not excuse the absence of *within-paper* controls. This is a structural flaw in the empirical argument that directly undermines the core claim.

- **The role of text-formatted point clouds is insufficiently disentangled.** Equation 2 formally includes $p^i$ (text-formatted point cloud coordinates) as input alongside the visual prompts. Section 3.2 states that multi-view images "can only trigger 3D spatial grounding in GPT-4o when combined with text-formatted point clouds" and that point cloud text is "an essential input for the model." Yet no quantitative ablation isolating the contribution of $p^i$ versus the visual prompt is provided. This is critical: if GPT-4o is primarily reasoning from explicit text coordinates rather than from the visual axis overlays, the claim that 3DAxisPrompt "promotes 3D understanding" through visual prompting is substantially weakened, and the method's contribution scope narrows considerably.

- **Small evaluation scale with no variance reporting.** Only 20 scenes per dataset are evaluated "due to the limited and costly GPT-4o API quota." No error bars, confidence intervals, or per-scene breakdowns are reported. Given the known high variance of LLM outputs and the small sample size, the relative differences between prompt variants emphasized in the paper (e.g., "7% decline," "19% improvement") may be within noise. The route planning results in Table 2 are particularly thin—success rates for 5 hand-picked route types with no reported number of trials or variance.

- **The "reading a ruler" vs. "3D reasoning" ambiguity is not addressed.** The paper's own discussion acknowledges that "axis ticks unify the units of measurement" and "act as a ruler," and the ablation shows complete failure without ticks. This strongly suggests GPT-4o may be performing coordinate reading/interpolation from an explicitly encoded visual scale rather than genuine 3D geometric inference. The paper never includes a control to disentangle these possibilities—for example, testing whether GPT-4o can predict 3D positions of objects not aligned with visible axis markings, or testing with deliberately distorted axes. This is the most fundamental interpretive question for the work, and it remains unanswered.

### Minor:

- **Only tested on GPT-4o.** The title and abstract generalize to MLLMs, but all experiments use a single closed-source model. Whether the findings transfer to other models (e.g., Gemini, Claude, open-source VLMs) is unknown.

- **Route planning evaluation is coarse.** Success is binary (reach destination or not) across 5 hand-picked route types. There is no metric for path optimality, safety, or efficiency. The example in Figure 4 shows GPT-4o producing somewhat inconsistent coordinate reasoning even with 3DAxisPrompt, which undermines the narrative of effective spatial planning.

- **ShapeNet (coarse object generation) task is purely qualitative.** Figure 6 shows one example with no quantitative metric for keypoint accuracy or skeleton quality, making this contribution unsupported as a quantitative claim.

- **NRMSE metric normalization is unconventional.** Normalizing by $\max(x_i)$ per scene conflates error magnitude with coordinate system offset (a scene with coordinates far from the origin will artificially lower NRMSE). The paper does not report raw distances or per-axis error breakdowns that would make the errors physically interpretable.

### Trivial:

- The discussion section speculates about "where 3D spatial grounding comes from" and draws analogy to human perception, but humans infer 3D from parallax and shading rather than reading coordinate axes—this analogy is misleading given the setup.

## Nice-to-Haves

- **Add a no-axis, no-prompt baseline** on the same tasks to isolate the marginal contribution of 3DAxisPrompt. This is the single most important experiment for validating the core claim.

- **Ablate the text-formatted point cloud input** to determine how much performance depends on visual reasoning vs. explicit coordinate parsing.

- **Test on at least one additional MLLM** to assess generalizability of the visual prompt design.

- **Per-axis error analysis** (X vs. Y vs. Z) to reveal whether depth estimation is consistently worse, which the qualitative investigation hints at but never quantifies.

- **Failure case analysis** to characterize when and why 3DAxisPrompt fails (e.g., occlusion, dense clusters, axis orientation ambiguity).

## Removed Points

These points were flagged for removal:

- **"No comparison with specialized 3D methods (e.g., 3D-LLM, PointLLM, SpatialVLM)"**: Per hard rules, this is an unfair comparison demand. This paper evaluates prompt engineering for existing MLLMs without fine-tuning, while specialized 3D models require architectural changes and training. The paper explicitly scopes itself as a prompt-engineering study, so demanding comparison with trained 3D models is outside scope.

- **"Overclaimed novelty relative to PIVOT, COARSE CORRESPONDENCES, etc."**: The paper cites and distinguishes these works in Section 2, noting they focus on spatial relations rather than positional localization. This is a reasonable differentiation.

- **"Reproducibility / undisclosed hyperparameters"**: Per hard rules, nitpicks about implementation details of API-based prompting are removed.

- **"Missing related works"**: Per hard rules, I cannot verify whether unmentioned related works exist.

- **"Format/style nitpicks"**: Per hard rules, removed.

## Novel Insights

The finding that tri-view projections activate 3D reasoning while arbitrary multi-view images do not (unless paired with text point clouds) is a genuinely interesting empirical observation about MLLM spatial capabilities. It suggests these models may have implicit tri-plane representations that can be leveraged via specific visual prompts, which connects to tri-plane representations in 3D generation literature but emerges here as a discovery about MLLM behavior rather than an architectural design. However, this insight is currently buried in qualitative investigation rather than rigorously validated.

## Suggestions

1. **Run and report a no-prompt baseline**: For each table, add rows showing GPT-4o's performance on the raw rendered images without any axis/mask overlay, and optionally with only 2D prompts (SoM-style). This is the most critical missing experiment.

2. **Ablate text point cloud input**: Run the best-performing prompt configuration with and without $p^i$ to quantify the visual vs. text contribution. This directly addresses the ambiguity about what "3DAxisPrompt" actually does.

3. **Include a simple coordinate-reading baseline**: Given axis ticks with known scale, implement a deterministic mapping from marked pixel locations to coordinates to establish an upper bound for what is achievable via pure reading rather than reasoning.

4. **Report variance across multiple runs** and scale evaluation to more scenes if possible, to make the quantitative claims credible.

5. **Tone down claims**: Replace "effectively perceive an object's 3D position" with qualified language about "eliciting coordinate predictions" or "enabling approximate localization," since the current evidence does not establish genuine 3D understanding versus coordinate reading.

## Score and Decision

**Calibration**: I compared against papers with similar methodology and topic:
- **Coarse Correspondences** (visual prompting for 3D in MLLMs, GPT-4o experiments): Scores 6/5/3, Withdrawn→Reject. This paper had stronger quantitative results, compared against baselines, and evaluated on standard benchmarks, yet still received low scores.
- **GPT-4V Geometry Prompting** (empirical study of prompting GPT-4V): Scores 3/5/3/3, Withdrawn→Reject. Reviewer: "almost zero algorithmic/technical contribution... just a bunch of prompts."
- **On Inherent 3D Reasoning of VLMs**: Scores 3/3/5/5, Reject. Overclaimed 3D capabilities on what were essentially 2D tasks.
- **GPT-4o CV Tasks** (prompt chaining for standard CV): Scores 5/5/6/6, Withdrawn→Reject. Had more tasks and models but still withdrawn.
- **SPACE benchmark** (spatial cognition evaluation): Scores 6/8/8/5, Accept Poster. Comprehensive evaluation with human baselines, multiple models, rigorous methodology.
- **CUBE-LLM** (training a model for 3D understanding): Scores 6/6/6/6, Accept Poster. Real method with baselines and strong quantitative results.

The current paper shares the "prompting study" pattern of the lower-scored papers but has weaker quantitative methodology than any of them (no baselines, 20 scenes, no variance, single model). Its contribution is primarily empirical/observational, with the core claim undermined by the absence of critical controls. It is above papers that were purely evaluation (no method proposed), but below papers that had real baselines and stronger experimental design.

**Score: 3**

The paper proposes an interesting prompt engineering idea but the evaluation does not substantiate its core claims. The absence of no-prompt baselines, the uncontrolled role of text-formatted point clouds, and the inability to disentangle "reading coordinates" from "3D reasoning" are fundamental issues that together undermine the claimed contribution. The qualitative explorations and ablation design choices (ticks, contours) are useful observations, but they are presented as findings about "3D grounding and reasoning" when they may merely reflect GPT-4o's ability to read off explicitly encoded coordinate information. With proper baselines and ablations, this could be a useful empirical study, but in its current form the claims significantly outpace the evidence.

MY FINAL SCORE: <pineapple>3</pineapple>
MY FINAL DECISION: <orange>Reject</orange>