## Summary
This paper studies whether GPT-4o can be induced to perform 3D grounding and reasoning without fine-tuning by using a prompt pipeline that overlays 3D axes, ticks, labels, and object marks/contours on rendered views of point clouds. It also reports an exploratory investigation of alternative prompt encodings and evaluates the resulting system on indoor/outdoor localization, route planning, robot action prediction, and a coarse object-generation demonstration.

## Strengths
- The paper tackles a timely and meaningful question: whether a general-purpose MLLM can be pushed toward instance-level 3D grounding and reasoning without model retraining. This is clearly stated in the introduction and is relevant to embodied and spatial reasoning settings.
- The prompt design is concrete and reasonably well motivated. Embedding a visible 3D axis with ticks/labels and highlighting object contours is a plausible way to expose geometric reference structure to a vision-language model.
- The paper includes a fairly broad exploratory study of prompt ingredients, including depth images, multi-view vs. tri-view, 2D vs. 3D marks, and axis-element ablations. Some findings are practically useful, e.g., the paper reports that axis ticks and labels matter, and that 2D contours perform well for localization.
- The task coverage is broad for an exploratory study: ScanNet indoor localization and route planning, nuScenes outdoor localization, FMB robot action prediction, and ShapeNet qualitative coarse object generation.
- The writing is generally understandable, and the paper is unusually explicit about some limitations, e.g., that “a single prompt engineering approach does not consistently achieve the best outcomes for all 3D tasks,” and that the goal is exploratory rather than perfect zero-shot performance.

## Weaknesses

###: Fatal
None.

### Major:
- **The paper’s central framing overstates what is actually demonstrated, because the method is not purely a visual prompting method.** The abstract and introduction repeatedly present 3DAxisPrompt as a visual prompting approach that extends 2D grounding into 3D, but Sec. 3.2 explicitly says that “the point cloud in text format [is] an essential input for the model,” and Eq. (2) formally defines the model input as \( \mathcal{F}(T^i, p^i, 3DAxis(I)) \). This is a substantive issue, not a wording nit: several results depend on supplying structured 3D geometry in text in addition to prompted images. As written, the paper does not cleanly isolate how much of the success comes from the axis/mark visual prompt versus from the text-formatted point cloud.
- **The evaluation protocol is too small and weakly controlled to support the paper’s broader empirical claims.** Sec. 4.1 states that the authors “randomly selected 20 scenes from each test dataset as validation data” due to API limits. That is very limited for making dataset-level claims such as “extensive quantitative experiments” and “demonstrate the effectiveness.” The paper also reports no repeated-run variability and relies on interactive ChatGPT sessions rather than a more controlled evaluation pipeline. Even if some amount of single-run evaluation is common for API-based systems, the present scale is too small for strong claims across several tasks and datasets.
- **Several key task evaluations are underspecified and do not convincingly substantiate the claimed planning/action capabilities.** For route planning, Table 2 gives success rates, but the paper does not clearly define what counts as a successful route beyond “whether the navigation successfully reaches the destination,” nor whether paths are checked for geometric validity or collision avoidance. The example in Figure 4 itself appears shaky, with repeated coordinates and an implausible distance estimate. For robot action prediction, the criterion is again simply “whether the orders can complete the mission,” with no clear simulator/executor or annotation procedure described. These evaluations are too coarse to strongly support claims about 3D planning or robot action competence.
- **There are important missing baselines/ablations needed to identify the source of improvement.** The paper compares prompt variants within its own pipeline, which is useful, but does not adequately isolate simpler alternatives such as text point-cloud input alone, image+marks without axis, or axis+images without point-cloud text on the main quantitative tasks. Since Sec. 3.2 itself says text-formatted point clouds are essential in many cases, this omission matters: the current results do not firmly establish that the proposed visual axis prompting is the main driver of performance rather than the additional structured 3D input.

### Minor
- **Some conclusions in Sec. 3 are stronger than the evidence shown.** Claims such as “multi-view visual prompting cannot directly activate the 3D reasoning capabilities of MLLMs, but tri-view prompting can” and that all mark formats “successfully elicit 3D spatial position reasoning” are drawn largely from limited exploratory evidence and qualitative examples. These are interesting observations, but the paper presents them too categorically.
- **The localization metric is not well justified.** Eq. (3) introduces an NRMSE-like quantity normalized by \(\max(x_i)\), but the paper does not justify why this normalization is the right choice, nor whether it yields fair comparability across scenes of differing scale.
- **The CoT comparison in Table 1 is somewhat confounded.** The “CoT” condition is not just reasoning-style prompting; the paper states it also provides “the additional coordinate of a nearby object.” That introduces extra reference information, so the gain should not be attributed to CoT alone.
- **Outdoor evaluation is narrow relative to the paper’s breadth of claims.** On nuScenes, the quantitative evaluation appears limited to two object types, vehicle and vegetation. That is understandable as an exploratory slice, but it is only modest evidence for broader outdoor 3D grounding claims.
- **Failure analysis is limited.** The paper mentions issues such as occlusion in tri-view and failures in dense scenes, but does not systematically characterize failure modes across tasks, which would be particularly valuable given the paper’s stated goal of probing both potential and limits.
- **Generality claims about “MLLMs” are broader than the actual evidence.** In practice, the experiments are on GPT-4o only. The paper does acknowledge this focus in places, but the title, abstract, and some body text still generalize beyond what is directly shown.

### Trivial
- The paper would benefit from more careful wording around “zero-shot.” “Training-free” or “no fine-tuning” is the more precise characterization, since the pipeline still uses substantial preprocessing and structured 3D inputs.

## Nice-to-Haves
- Evaluate at least one additional model, ideally an open model, to determine whether the observed prompt effects are GPT-4o-specific or more general.
- Add richer visual analysis such as predicted-vs-ground-truth scatter plots or per-axis error plots to show whether errors are dominated by depth, scale, or axis confusion.
- Report practical cost/latency of multi-view prompting, since the method requires multiple rendered images and interactive model queries.
- Expand the coarse object generation section into either a proper quantitative evaluation or keep it clearly as a qualitative proof-of-concept.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Pure reproducibility attacks based on GPT-4o being closed-source or subject to change.** The paper’s dependence on GPT-4o can still motivate a request for stronger evaluation control, but criticisms framed as doubting availability/verifiability of the cited model are removed per instruction.
- **Complaints about missing related work.** Not included, per instruction.
- **Pure formatting/style issues and typo comments.** The extracted text has parsing artifacts, and style nitpicks are not central.
- **Claims that the paper should compare to more baselines simply because they exist.** Kept only where the baseline is needed to isolate the paper’s own claimed mechanism; generic “add more baselines/models” requests were weakened or moved to nice-to-haves.
- **Overly strong novelty dismissal (“not even a paper,” “zero technical contribution”).** This is not supported by the manuscript. The work does contain a real, if modest, exploratory contribution in prompt design and empirical probing.

## Novel Insights
The most important synthesis is that the paper is strongest when read as an exploratory probe of GPT-4o’s sensitivity to explicit geometric scaffolding, and weakest when read as a clean demonstration of visual 3D grounding. The manuscript itself contains the seeds of this reinterpretation: Sec. 3.2 effectively shows that 3D grounding here often emerges from a hybrid representation—rendered geometric cues plus structured point-cloud text—rather than from visual prompting alone. Framed that way, the prompt studies and ablations are genuinely useful, but the broader claims about eliciting robust 3D grounding in MLLMs are currently ahead of the evidence.

## Suggestions
- Reframe the paper more honestly around **hybrid prompt-based 3D grounding** rather than purely visual prompting.
- Add the critical ablation matrix that isolates: visual prompt only, point-cloud text only, visual+text, marks without axis, axis without marks.
- Strengthen evaluation protocol: enlarge the sample, state exact denominators, and clarify how route/action success is judged.
- For route planning and robot action prediction, define success rigorously and, if possible, use executable or checkable criteria rather than subjective mission completion.
- Present failure cases systematically, especially for occlusion, dense object layouts, and sparse outdoor point clouds.
- Tone down categorical claims in Sec. 3 unless backed by controlled quantitative experiments.
- Clarify that the CoT improvement in Table 1 is partly due to added reference coordinates, not just chain-of-thought prompting.
- Narrow claims about “MLLMs” unless an additional model is tested.

## Score and Decision
**Assessment across axes:**  
- **Originality:** Moderate. The core ingredients are simple and largely compositional, but the specific 3D axis prompting formulation and exploratory study are still a legitimate contribution.  
- **Importance of the question:** High. Whether MLLMs can be induced to do 3D grounding without fine-tuning is an important question.  
- **Support for claims:** Weak-to-moderate. The strongest issue is the mismatch between the paper’s visual-prompt framing and its reliance on text-formatted point clouds, plus underspecified evaluation for planning/action tasks.  
- **Soundness of experiments:** Moderate for exploratory ablations, weak for broad capability claims.  
- **Clarity:** Reasonably clear overall, though some claims are stated more strongly than the evidence warrants.  
- **Value to the community:** Moderate as an exploratory study, but limited as a definitive empirical paper.

**Calibration against human-reviewed anchors:**  
- Compared to **“How well does GPT-4o understand vision?”** (`h3unlS2VWz.md`, scores 5/5/6/6), this submission is less convincing experimentally: that paper had broader, more controlled benchmarking and clearer evaluation protocols, even if it was also largely an empirical study.  
- Compared to **“Coarse Correspondences Boost 3D Spacetime Understanding in MLLMs”** (`8ibaVk4mU8.md`, scores 6/5/3), this paper is somewhat weaker on evidence because its central mechanism is more confounded by auxiliary structured input, and its evaluation is smaller and less well defined on key tasks.  
- Compared to **“On Inherent 3D Reasoning of VLMs…”** (`uBhqll8pw1.md`, scores 3/3/5/5), this paper has a more concrete method and somewhat more actionable findings, so it sits above the low end.  
- Compared to **the GPT-4V geometry prompting empirical study** (`0vKokoPKTo.md`, scores 3/5/3/3), this paper is in a similar family: interesting prompt-based probing, but limited technical depth and weaker-than-needed empirical support for the strongest claims.

Overall, this lands in the **borderline-below-threshold** range: there is a real exploratory contribution, but the confounded central claim and weak evaluation protocol make the current paper unconvincing for acceptance.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>