# GeoReasoning: Structured Semantic Reasoning for Image-to-Map Localization

- Decision: Reject
- Scores: 2, 2, 8, 2

## Abstract
We introduce *reasoning localization*, a new paradigm for self-localization that leverages multimodal large language models (MLLMs) to interpret spatial context from 2D maps and first-person images. Unlike traditional approaches that depend on LiDAR, odometry, or engineered markers, reasoning localization emulates how humans orient by aligning visual cues with map structure. To address this new self-localization problem, we present **GeoReasoning**, a zero-shot framework that decomposes image-to-map grounding into *structured semantic reasoning* followed by *geometric verification*. Instead of directly predicting coordinates, GeoReasoning (i) identifies map-visible landmarks, (ii) grounds them as anchors via promptable segmentation, (iii) estimates coarse distances through language-based reasoning, and (iv) solves a robust triangulation program to recover the pose. This design separates high-level semantic reasoning from metric optimization, yielding interpretable rationales, verifiable intermediate outputs, and resilience against map symmetries. To support this task, we release the first benchmark for reasoning localization, spanning diverse indoor maps, image-map pairs, and candidate poses, along with diagnostic metrics such as rationale consistency, mean/median localization error, and success@$r$ for $r\in{0.1,0.5,1,3}$ m. Experiments with state-of-the-art MLLMs demonstrate that GeoReasoning significantly improves localization accuracy over direct prediction baselines, while exposing open challenges in symmetry disambiguation and monocular scale estimation. Our results highlight structured reasoning--geometry integration as a promising path toward scalable, human-like localization in GPS-denied settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces GeoReasoning, a training-free framework for indoor reasoning localization that leverages large language models. The authors also present a novel indoor benchmark specifically designed for this task. The core of the proposed solution is a zero-shot framework that decomposes the complex localization problem into two stages: structured semantic reasoning followed by geometric verification. This approach explicitly models anchor (landmark) selection, verifies cross-view consistency to handle ambiguity, and uses robust trilateration to solve for the final pose. As demonstrated in the evaluations, the GeoReasoning framework significantly outperforms direct-prediction baselines, which often fail at this task.

### Strengths
1.	Overall, The paper is well-written, clearly articulating the reasoning localization concept. The proposed GeoReasoning framework is presented logically, and its two-stage (Reason & Ground, Constrain & Solve) methodology is easy to follow.
2.	A key contribution is the proposal of a new indoor localization benchmark, which is derived from an existing dataset. The experimental validation further confirms that the proposed method achieves better performance over the baselines

### Weaknesses
1. The claimed reasoning localization paradigm is said to move beyond geometry-first methods. However, it still appears to be closely related to retrieval-based localization. The system essentially searches for geometric landmarks, employs large language models and segmentation to identify potential landmarks, expands the candidate set, and then applies additional rule-based reasoning (e.g., object associations) to refine localization.
2. The authors argue that traditional SLAM-based localization pipelines, while effective, suffer from sensor noise, calibration drift, dynamic scenes, and appearance changes, and require careful tuning for long-term deployment. However, I do not find convincing evidence in this paper that the proposed method actually mitigates these issues. In fact, similar performance improvements can also be achieved by simply expanding the candidate set and introducing additional rules, such as re-ranking strategies.
3. The experiments only compare different language models within the proposed framework, without benchmarking against other representative localization approaches such as LalaLoc or SceneGraphLoc[1] and etc. As a result, it is difficult to substantiate the claimed advantages of this “new paradigm.”
4. The method is evaluated on a newly introduced dataset rather than on existing benchmarks. This raises the concern that the approach may rely on specific data conditions or annotations, which could limit its general applicability.
5. The authors further claim that the proposed method generalizes well to outdoor environments. I remain skeptical of this assertion, as no experimental evidence or quantitative validation is provided to support it.

[1] Scenegraphloc: Cross-modal coarse visual localization on 3d scene graphs. ECCV2024.

### Questions
please seek weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a novel localization paradigm, which leverages the reasoning power of MLLMs to handle indoor-localization tasks. Without training, it decomposes the indoor-localization into the semantic reasoning and geometric trilateration stages. A new benchmark is released for this reasoning localization paradigm.  Compared with the direct-prompt baseline, the proposed GeoReasoning gains obvious improvements in the zero-shot setting.

### Strengths
The proposed reasoning localization framework contains two stages. The first extract localization anchors (landmarks) through MLLM captioning and segmenter prompting, and it is more time&memory efficient compared to the sophisticated retrieval used in relocalization and SLAM systems. The second stage constrains the reasoning results with geometric trilateration, which prevents fragile matching and makes it more robust when facing scene clutter, repetition and symmetry.

### Weaknesses
The application of the proposed egocentric-floor localization is limited: indoor objects are often moved or replaced in daily life. The framework localizes itself by reasoning and triangulating objects both observed in the egocentric image and the floor plan,  which is unreliable compared with matching (contains more anchor points and excludes irrelevant parts from RANSAC). Moreover,  it is hard for users to update the floor plan without other mapping or rendering techniques, but users can easily update scenes by uploading new egocentric images in traditional localization pipelines. 

The proposed method lacks comparison with traditional localization and direct-prompt methods, partly due to the egocentric-floor localization settings, which hinder the demonstration of the contributions mentioned in Strengths.

### Questions
Distance reasoning from a single image is inaccurate for SOTA MLLMs, which would be the bottleneck of the localization error under S@1. The authors could leverage monocular depth estimators like Moge-2 or more related DepthLM to get more accurate depth(and camera model) priors.

It would be better to add more visual analysis for failure cases in Line 432-436, such as a pie chart.

Equations are not strictly labeled.  In equations under Line 283，$T$ and $R$ are not mentioned before, and $M$ is duplicated with the symbol of the RGB image.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a method for localization in a camera setting through its use of landmarks, done a bit differently from SLAM like approaches. They also claim to release a new dataset for indoor scenes to support the research effort. The procedure can be used as an addon to generic MLLMs and improve its performance. 

The main idea is to reconcile a camera view with landmarks obtained from a 'map' - a top down view of the scene. It is a two stage process, first using intelligent prompting to get an intermediate output and then refining it further through optimization.

The map can generate anchors of interest, prompted by a multimodal LLM - a SAM like apparatus. Once they are obtained, one sets up an optimization objective through a trilateralization procedure described. 

$ L = \sum w_k \varphi (||p - a_k|| - \rho_k) $

Here, we know $a_k$ from the landmarks (global solve), and $\rho_k$ is the distance in local camera. So from this we can get $p$ and thus localize. In essence we obtain p that minimizes the residual contained within.

A fairly convincing set of evaluations is provided. The gist of it is that the fittings - querying with a prompt for landmarks, and the optimization are able to significantly improve vanilla MLLM performance.

### Strengths
+ Intuitive way to connect indoor localization pieces 
+ Camera + map makes sense - also applicable in settings like autonomous driving with BEV. 
+ Principled constraint and prompting machinery. 
+ Results demonstrate the correctness of the approach. We see improved results with their fittings in nearly all the cases and models. To drill down, ID switches are reduced, localization accuracy is improved.

### Weaknesses
- Please take this with a grain of salt. I am pretty sure that similar ideas abound in localization (for instance, in my research in BEV modelling), where we can carry out the task given a map and images by correlating them (with attention, or other means). However, the main novelty in this work is to use it with multi-modal LLMs. So to this end, I (very weakly) question the novelty of the approach. 
- Temporal modelling. It would have been nice if the authors could have extended the analysis with temporal modelling. I would gather that this extension is straight forward given the methods we have today.
- More failure cases would be helpful.

### Questions
I am curious about what would happen if the map were erroneous. These happen in outdoor driving scenes like construction zones. 
On similar lines, I am curious about scaling errors, and generally about miscalibrated maps.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces GeoReasoning, a new zero-shot framework for self-localization that requires no sensors or environment-specific training. It uses multimodal large language models (MLLMs) to interpret maps and visual observations. The method works by first identifying landmarks with an MLLM, locating them in an image, estimating their distance through textual reasoning, and then using trilateration to find the user's position. The paper also introduces Loc-Bench, a new evaluation dataset, on which GeoReasoning outperformed direct MLLMs prompting methods.

### Strengths
The framework is interpretable and modular, separating semantic reasoning from geometric computation. Using semantic anchors followed by trilateration mirrors human reasoning and makes the multimodal localization process more transparent and analyzable.

### Weaknesses
Main Weakness:
1. The author states that they create Loc-Bench that differs from previous work by focusing on egocentric-to-allocentric transformation, while VSI-Bench[1] have included a 2D cognitive map and evaluates visual-spatial relationship from egocentric videos, espatially testing on relational reasoning
and egocentric-allocentric transformation.
2. The baselines used for comparison are limited. There are more SOTA open-sourced VLM that could be compared to, e.g. InternVL2.5 and LLaVA, which would better contextualize the improvements. I also suggest the authors to provide baselines for (1) always selecting the most frequent answer; (2) a random selection strategy, which would help quantify the task’s inherent difficulty. Besides, a human-level performance would establish an intuitive upper bound and clarify the practical significance of the results.
3. The ablation study does not evaluate each component's strength. Key design choices (e.g. anchor reasoning, verification and refinement) are not individually quantified, leaving it unclear which parts actually drive performance. A more fine-grained ablation would clarify whether the observed improvements arise from the overall pipeline design or a few dominant modules.



Minor:
1. Only one equation is indexed; all others are not. Please ensure consistent numbering.
2. In the equation on line 250, what does $\mathcal{D}$ represent? Is it a typo for $G$? Also, the use $\mathbb{R} > 0$ is not formal.
3. Some notations are undefined, e.g., the $L$ in Equation (1). Define all symbols on first use.


[1]Yang, Jihan, et al. "Thinking in space: How multimodal large language models see, remember, and recall spaces." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
In lines 220–238, how does the cross-view semantic verifier handle repeated object categories (e.g., multiple bathrooms, beds, or tables), given that prompt-based segmentation on $M$ may highlight only one instance among several plausible candidates?

### Soundness
1

### Presentation
2

### Contribution
2
