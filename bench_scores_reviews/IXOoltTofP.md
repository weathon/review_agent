## Summary
3DAxisPrompt proposes a visual prompting scheme that inserts 3D coordinate axes and SAM-generated segmentation masks into rendered point cloud images to elicit zero-shot 3D grounding and reasoning capabilities in GPT-4o, without any model fine-tuning. The paper provides a systematic investigation of visual prompt formats (depth, multiview, tri-view, various mark types) and evaluates across four datasets covering indoor/outdoor localization, route planning, robot action prediction, and coarse object generation. The central, explicit goal is not to achieve state-of-the-art 3D grounding performance but to probe and document the limits and potential of prompting-based 3D reasoning in existing MLLMs.

---

## Strengths

- **Concrete, actionable findings from prompt format investigation**: The paper systematically tests depth images, multiview images, tri-view projections, and various mark types (AABB, OBB, 3D edge points, 2D contours), and reports interpretable, differentiated findings. The discovery that tri-view projections onto the XY/YZ/ZX planes can elicit 3D coordinate prediction even without text point cloud input — while arbitrary multi-view images cannot — is a specific, non-obvious result that will be useful to practitioners designing 3D prompting pipelines.

- **Concrete axis element ablation with actionable design insight**: The ablation in Figure 7 specifically quantifies the contribution of axis ticks and labels. Removing ticks causes complete failure; removing labels increases to-bbox error by 37%. These are specific, reproducible design conclusions — not generic insights — that directly guide future prompt engineering work.

- **Informative negative results**: The paper's finding that depth images cause GPT-4o to predict 2D pixel coordinates rather than 3D world-space positions, and that generic multi-view sequences fail to activate 3D reasoning (Section 3.2), saves future researchers significant experimentation effort. The scope of negative results is one of the genuine contributions.

- **Multi-task evaluation breadth**: Few prompt-engineering papers systematically cover localization, spatial route planning, robot action prediction, and object generation across both indoor (ScanNet) and outdoor (nuScenes) environments. The breadth is appropriate for an exploratory paper probing the capability frontier.

---

## Weaknesses

### Fatal
None. The paper is an exploratory empirical study with a clearly modest goal, and no single issue fully invalidates its contribution. However, two major weaknesses together substantially weaken confidence in the claimed mechanism.

### Major

- **Missing ablation of text-formatted point cloud input (p^i) — core claim is unverified**: The method as formalized in Eq. (2) explicitly includes *both* the visual prompt 3DAxis(I) and the text-formatted point cloud p^i. Section 3.2 directly states: *"we consider the point cloud in text format to be an essential input for the model."* However, no experiment measures performance with text coordinates alone (no visual axis), nor with visual axis alone (no text coordinates). Without this ablation, it is impossible to determine whether GPT-4o is performing genuine visually-grounded 3D reasoning from the axis images, or is primarily doing coordinate lookup from the text point cloud with the visual prompt playing a secondary role. This directly undercuts the paper's framing as a "visual prompting method." This is the most critical missing experiment in the paper.

- **No baselines — NRMSE and success-rate numbers are uninterpretable**: There are zero comparisons to any external reference: no random baseline, no centroid prediction, no prior 2D grounding method applied to the same scenes. An NRMSE of 0.138 (best indoor to-bbx) is meaningless without knowing what a trivial predictor would achieve on the same 20-scene sample. The route planning 79% success rate is similarly unanchored. Even a random coordinate prediction or a scene-centroid baseline would allow readers to calibrate what the numbers mean.

- **NRMSE metric definition is ambiguous**: In Eq. (3), the normalization factor `max(x_i)` is never clearly defined. If `x_i` is a 3D position vector, `max(x_i)` is scalar-ambiguous — is it the maximum axis-aligned coordinate, the maximum scene extent, or something else? For the to-center metric, D(x̂_i, x_i) is a Euclidean distance (scalar), while `max(x_i)` appears to reference coordinates (vector), making the formula dimensionally confused. This metric is not standard and is not sufficiently explained to be reproduced or compared against.

- **CoT variant conflates reference coordinate injection with chain-of-thought**: The best-performing entry in Table 1 ("Mark+2D contour (colors) + CoT") appends *"the additional coordinate of a nearby object"* along with "let's think step by step." This is not pure chain-of-thought prompting — it provides privileged reference point information as input, which is a qualitatively different intervention. The improvement from 0.271 to 0.219 (to-center) and 0.138 to 0.115 (to-bbox) cannot be attributed to CoT alone, and this should be disentangled into separate ablation rows.

### Minor

- **Small sample size and no variance estimates**: 20 scenes per dataset due to API quota constraints is explicitly acknowledged, but there are no confidence intervals, bootstrapped estimates, or repeated trials across GPT-4o's stochastic outputs. Several of the performance differences between mark types in Table 1 are on the order of 0.01–0.02 NRMSE units over 20 scenes, which may not be distinguishable from noise. The paper should at minimum report variance across repeated queries on the same scenes.

- **Reproducibility gap from ChatGPT UI evaluation**: Experiments are conducted via the ChatGPT interface rather than a controlled API pipeline. This introduces uncontrolled variables: hidden system prompts, possible model version drift, decoding parameters, and UI-specific behavior. Exact prompts are not reported in the main paper. For the community to build on this work, a controlled API-based evaluation protocol with released prompts is required.

- **Route planning success rate is subjectively graded**: The criterion for "success" in route planning is not defined operationally — it is described as "whether the navigation successfully reaches the destination" without specifying who judges this, by what rubric, or whether it uses execution simulation vs. plan inspection. The "From door to chair" task appears twice in Table 2 (rows 1 and 5, with 80% and 60%) without explanation of whether these are two different configurations or scenes.

- **Only GPT-4o tested**: The paper's title and framing suggest findings about "MLLMs" in general, but all experiments use a single proprietary model. The identified prompt formats (3D axis, contours, tri-view) may or may not transfer to GPT-4V, Gemini, or open-source models. This limits the scope of generalizable conclusions.

- **Outdoor localization covers only two object classes**: Only vehicles and vegetation are evaluated on nuScenes. These two classes were presumably chosen because they worked reasonably well with the method. The scope of outdoor 3D grounding is much broader, and restricting evaluation to two classes without justification weakens this sub-evaluation.

### Tiny

- The term "extensive" in the abstract is not calibrated to the actual 20-scene evaluation protocol; more accurate language would help.
- The paper uses "elicit," "activate," and "provoke" interchangeably without tying them to measurable outcome definitions.
- Coarse object generation on ShapeNet has no quantitative evaluation and reads as a qualitative demo, which is fine as a demonstration but should not be listed as a primary evaluation task.

---

## Nice-to-Haves

- A comparison to at least one trained 3D-MLLM (e.g., PointLLM, SpatialVLM) on the same scenes — not to claim competitiveness but to contextualize what "zero-shot prompting" achieves relative to trained systems.
- Absolute localization error in meters alongside NRMSE to help readers calibrate practical utility for robotics/driving applications.
- Testing tri-view prompting on a larger sample or reporting its quantitative comparison to multiview prompting in a table (currently the distinction is described qualitatively with single examples).
- An axis perturbation experiment — shifting origin or mis-calibrating scale — to probe whether GPT-4o is doing true 3D spatial reasoning or visual tick-mark reading. This would validate the mechanistic hypothesis in Section 5.
- Systematic failure-case analysis (e.g., occlusion level, scene density, distance from axis origin) to better delineate when the method fails.
- A brief discussion of robustness to model version updates, since GPT-4o is proprietary and may change.

---

## Removed Points
*These points are flagged to be removed — treat them with caution.*

- **"Figure 2 armrest example shows degenerate coordinate reading" (Harsh Critic)**: Section 3.2 is explicitly framed as an investigation into what works and what doesn't. The armrest example is presented as an early finding that basic 3D axis prompting elicits some coordinates, motivating further work. The paper does not claim this is accurate localization. Removing this criticism as it misreads the section's intent.

- **Missing related works on spatial/embodied reasoning VLMs (Harsh Critic)**: Removed per instructions — we cannot confirm which works exist or are missing.

- **Unfair comparison to specialized 3D-MLLMs (Review 2)**: The paper makes no claim to outperform trained 3D models. The absence of such comparisons is not an unfairness to the author's method; it reflects the paper's scoped goal of probing zero-shot prompting capabilities. This is not a weakness within the stated scope.

- **"Claiming p^i text is readable is unsubstantiated beyond anecdote" (Harsh Critic)**: The paper appends point cloud coordinates as text and observes GPT-4o's responses. This is a reasonable qualitative demonstration for an exploratory study. A fully controlled experiment would be ideal but is not required to observe that the model responds to the input.

- **The paper relies on proprietary closed-source model (Review 2 weakness)**: This is a limitation of the field, not a flaw of this paper. Using GPT-4o is explicitly justified (Section 2.3). The paper is transparent about this.

---

## Novel Insights

The most genuinely novel analytical insight surfaced across all three reviews is the **visual-vs-text contribution confound**: the paper's method formally combines text-formatted point cloud coordinates (p^i) with visual axis prompts, and explicitly labels text point clouds as "essential," yet frames the contribution as a visual prompting framework. This creates an unresolved question about the actual mechanism: is GPT-4o performing visually-grounded 3D reasoning from the axis and contour images, or is it primarily performing spatial arithmetic on the text coordinates with the visual prompts serving as a secondary alignment cue? Resolving this would be a substantive scientific contribution beyond the paper's current results, and the p^i ablation experiment is the single experiment most critical to clarifying the paper's actual contribution.

---

## Suggestions

1. **Run the p^i ablation** — measure indoor localization performance with (a) text point cloud only, no visual axis; (b) visual axis only, no text coordinates; and (c) both combined. This is the highest-priority missing experiment and directly determines whether the paper's contribution is a visual prompting method or a text-in-the-loop hybrid method.
2. **Report a trivial baseline** (scene centroid, random uniform prediction within scene bounds) alongside NRMSE so readers can calibrate what the numbers mean.
3. **Separate the CoT + reference-coordinate entry** in Table 1 into two rows: pure CoT ("let's think step by step") and CoT + reference coordinate. These are qualitatively different interventions.
4. **Define success in route planning explicitly** — either use an automatic evaluator (e.g., check whether intermediate coordinates are collision-free given scene geometry) or publish the grading rubric, and explain the duplicate "From door to chair" rows.
5. **Clarify max(x_i) in Eq. (3)** — define it precisely (e.g., maximum scene-extent coordinate) and consider supplementing with absolute Euclidean error in meters for at least one dataset.
6. **Release evaluation protocol** — provide exact prompt templates, parsing code, and the specific 20 scene IDs for each dataset to enable reproducibility without API-based ChatGPT UI interaction.

---

## Evaluation on Key Axes

**Originality**: Moderate. Applying visual prompting to elicit 3D grounding in a 2D-pretrained MLLM without finetuning is a timely and distinguishable contribution from prior work, and the specific 3D-axis plus contour design is novel. However, the combination of text coordinate injection with visual prompts muddles the originality of the "visual prompting" framing.

**Importance of research question**: High. Whether off-the-shelf MLLMs can reason in 3D with lightweight prompting is a practically important question for robotics, embodied AI, and autonomous driving. The explicit exploratory framing is honest.

**Whether claims are well supported**: Weak. The central claim that 3DAxisPrompt "elicits 3D grounding" is not cleanly supported because the text-point-cloud component's independent contribution is never isolated. Quantitative results are on very small samples (20 scenes) with no statistical uncertainty.

**Soundness of experiments**: Below ICLR standard. The UI-based evaluation, absence of baselines, single-run single-sample results, and ambiguous metrics collectively limit the scientific trustworthiness of the quantitative conclusions. The qualitative investigation sections are more informative than the quantitative tables.

**Clarity of writing**: Adequate. The high-level idea and exploratory structure are easy to follow. Key methodological details (exact prompts, parsing logic, scene selection procedure) are missing from the main paper.

**Value to the research community**: Moderate. The prompt format investigation and negative results (depth images, arbitrary multiview) have genuine practical value. But without the critical p^i ablation, the paper's central design recommendation — use 3DAxisPrompt — cannot be confidently endorsed.

**Contextualization relative to prior work**: Adequate for the prompting literature. The discussion of PIVOT and SoM is appropriate. The absence of quantitative comparison to prior approaches is a gap, though partly justified by the exploratory framing.