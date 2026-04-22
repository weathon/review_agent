# Towards Human-Like Event Boundary Detection in Unstructured Videos through Scene-Action Transition

- Avg Score: 4.40
- Decision: Reject
- Scores: 6, 4, 4, 8, 0

## Abstract
Humans segment continuous experience into episodes by detecting perceptual novelties and retrospectively consolidating them into coherent memories. Drawing inspiration from these cognitive principles, we introduce a two-level, backward-only event segmentation framework designed to structure continuous sensory input into stable episodic units. The goal is to identify meaningful situational shifts (e.g., transitions between activities or environments) that lie between coarse scene-change detection and the dense, motion-driven micro-boundaries targeted in Generic Event Boundary Detection (GEBD). At Level1, an error-driven novelty detector with semi-supervised adaptive thresholding identifies candidate transitions robust to noise, viewpoint shifts, and repeated micro-actions. At Level2, a retrospective boundary consolidation mechanism validates and merges these candidates using multimodal cues (scene graphs, captions, audio), producing stable, semantically grounded episodes without relying on future frames. Unlike prior GEBD approaches that depend on motion cut-points or heavy task-specific supervision, our method uses sparse labels only for threshold calibration, making it label-efficient, cognitively inspired, and broadly applicable. Experiments on Ego4D show state-of-the-art performance, with our semi-supervised model surpassing heavily supervised baselines. This work introduces episodic segmentation for embodied perception, taking conceptual inspiration from human memory research while focusing on scalable machine perception.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a two-stage framework to segment events in videos. (1) In the first stage, it uses pre-trained (frozen) deep neural networks as feature encoders to capture the semantic, perceptual and structural representations of sampled frames. It then computes frame-to-frame similarity between the current frame and several preceding frames, deriving statistical features such as mean, variance and short-term similarity scores. These features are used to train an adaptive threshold model that identifies the candidate event boundary. (2) In the second stage, the framework refines those initial candidates by merging or removing boundaries based on semantic, perceptual and linguistic cues.

### Strengths
(1) The adaptive threshold model is effective in identifying event boundaries, its performance is shown by the experiments.
(2) The paper is clearly written, with well-structured methodology and experimental sections that make it easy to follow.
(3) Since the framework only relies on frozen, pre-trained networks to extract various features, it can be easily adapted to new datasets or domains without expensive re-training these feature extractors.
(4) The second stage is effective in removing spurious boundaries.

### Weaknesses
(1) The framework’s heavy dependency on the frozen pre-trained feature extractor can limit its application if the extractor does not fit the application domain. There is no ablation study or sensitivity analysis to show how a noisy feature extract may affect the overall accuracy.
(2) The use of an adaptive threshold is conceptually straightforward and not particularly novel. It would be helpful if the authors could compare their model against alternative decision models which employ temporal models or probabilistic models, such as RNN, Transformers.

### Questions
(1) How sensitive is the proposed method to the choice of the pre-trained feature extractor?
(2) Why was an adaptive threshold model based on hand-picked statistical features chosen instead of other temporal models such as RNN, Transformers based directly on the features extracted from feature extractors?
(3) Has the method been evaluated on datasets with different video types (eg, sports, surveillance)?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method for event boundary detection in unstructured videos. The authors propose a two-level approach: Level 1 employs an error-driven novelty detector with a semi-supervised adaptive threshold to find candidate transitions, while Level 2 uses retrospective consolidation with multimodal cues (scene graphs, captions, audio) to validate and merge these boundaries into semantically coherent episodes. Experimental results on ADL-GEBD and Ego4D demonstrate that this framework achieves state-of-the-art performance.

### Strengths
- Outperforms unsupervised baselines on ADL-GEBD: The Level 1 detection method demonstrates strong segmentation accuracy in densely annotated videos, achieving an average F1 score of 0.885 outperforming all five tested unsupervised boundary detection methods across all evaluated distance thresholds.
- Outperforms supervised baselines on Ego4D
- The level 2 consolidation is robust, and extensive ablations are performed to validate the design choices.

### Weaknesses
- Two claims of the authors in the abstract are not substantiated in the paper: label-efficient, and broadly applicable. How is this approach label-efficient compared to other approaches? For the broad applicability claim, it would be preferable to include experiments that substantiate that.
- The authors report results on 5 models for boundary detection that they tested themselves. No issue with that, but I think other papers have self-reported results on this dataset (Kinetics-GEBD), and the authors should consider reporting these results. Some examples and their average F1 scores: EfficientGEBD [1] (90.8), End-to-end… [2] (0.865). I am sure a quick search will return plenty of others.
- Both of the benchmarks used by the authors are well-established in the literature; I think more baselines could be added.
[Styling] According to ICLR26’s guidelines, captions should be placed above the tables.


[1] Zheng, Ziwei, et al. "Rethinking the architecture design for efficient generic event boundary detection." Proceedings of the 32nd ACM International Conference on Multimedia. 2024.

[2] Li, Congcong, et al. "End-to-end compressed video representation learning for generic event boundary detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

### Questions
See weaknesses plus
- I think the community calls it the Kinetics-GEBD dataset instead of ADL-GEBD or is this a different dataset?
- When is the audio used and how? I find little mention of it in the paper. Also, could we have an ablation to determine whether it is necessary and how much better it makes the pipeline’s Level 2.
- For Table 6 and 7 and Sec. 5.3.3 why is Scene Graph + Caption tied? Could it be Scene Graph only or Captions only?
- Could the authors please clarify 1.5x real-time? 
- The system is explicitly stated to achieve an average F1 score of 0.885 in Table 2 and Table 4. This value is the primary evidence of the model's superior performance over fixed thresholds and unsupervised baselines. However, when studying the optimal context length for the adaptive threshold network (Section 5.3.2), the system using the chosen 3-frame context (labeled "Ours") reports an average F1 score of only 0.829. Is this a typo?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript introduces a cognitively-inspired, two-level framework for event boundary detection. The goal is to segment continuous video into semantically coherent episodes, distinct from traditional motion-driven Generic Event Boundary Detection (GEBD). Level 1 utilizes an error-driven novelty detector with a semi-supervised adaptive threshold. Level 2 employs an uncertainty-driven consolidation mechanism that retrospectively validates boundaries using multimodal cues. The approach is backward-only, mimicking episodic memory formation. The authors report state-of-the-art performance on the ADL-GEBD and Ego4D datasets. The paper claims cognitive grounding and practical implications for episodic memory modeling in embodied agents.

### Strengths
1. Novelty and Intuition of the Framework
The proposed two-level architecture (Level 1 Detection, Level 2 Consolidation) is intuitive, well-motivated by cognitive science, and provides a clear separation of concerns. The "backward-only" design, which operates exclusively on past context , is a crucial constraint for real-world cognitive agents and distinguishes the work from offline methods that use future frames.

2. Label-Efficient Adaptive Thresholding
The adaptive threshold network dynamically calibrates decision boundaries using retrospective statistics, which is technically sound. Table 4 empirically shows that adaptive thresholds outperform fixed cutoffs across 10 relative distance thresholds.

### Weaknesses
1. Limited Cognitive Evidence
Despite cognitive framing, there is no direct behavioral or empirical evidence that the model aligns with human segmentation patterns. The claim of “episodic segmentation mirroring human memory” (in Abstract; Sec. 1) remains qualitative.

2. Lack of Evaluation Scope
The two datasets (ADL-GEBD and Ego4D) used are both egocentric. I wonder why no tests on third-person domains (e.g., Kinetics-GEBD[1] or TAPOS[2]), which seem to be the most widely-used benchmarks for the GEBD domain.

3. Method Clarity and Writing
The manuscript provides too few details about the specific module designs of the whole framework in Fig. 2. What is the detailed architecture of the Semantic Encoder, Perceptual Encoder? I found it quite difficult to understand the whole method.


[1] Generic event boundary detection: A benchmark for event segmentation. ICCV 2021.
[2] Intra-and inter-action understanding via temporal action parsing. CVPR 2020.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a two-level, cognitively inspired event segmentation scheme for unstructured video, mimicking how humans detect and consolidate experiences into episodic memories. The first stage uses a semi-supervised adaptive thresholding novelty detector that filters out noise and repetitive micro-actions. The second stage retrospectively validates and merges boundaries using semantic, perceptual, and audio cues, yielding stable episodes grounded in meaning rather than transient visual changes. Unlike prior motion-cutpoint-driven GEBD systems, this framework leverages sparse supervision, multimodal fusion, and a backward-only processing window, showing state-of-the-art results on ADL-GEBD and Ego4D datasets.

### Strengths
The dual-level, backward-only pipeline closely aligns with established models of human event segmentation and episodic memory. This is different from current machine-centric approaches focused on frame-level novelty.

In addition, adaptive thresholding requires only light supervision for calibration. The method is shown to be scalable across domains with limited annotated data.

### Weaknesses
* It seems to me that dialogue-driven content is rudimentary and not deeply integrated. Boundary decisions sometimes defer to utterance completion, but there is limited fine-grained modelling of discourse transitions; this, I think, is critical in conversational or instructional videos.
* The explicit backward-only causality prevents the use of future context, so marking the boundaries is sensitive for transitions spread over several frames; under-segmentation is visible in fast, crowded scenes, diluting retrieval for granular action tasks.
* The method is susceptible at under-segmentation. Merging micro-actions seems to prioritise episode coherence. If I understood it correctly, this may miss subtle event boundaries, particularly in fast-paced or dense activity streams. This can reduce granularity in retrieval-oriented settings.

### Questions
*  How does the method address memory aging, task transfer, or handling ambiguous ground-truth in real time?
* The fusion weights for modality are hand-tuned; CLIP and DINOV2 are balanced at 0.2 and 0.3, but if the weight of CLIP increases or token similarity decreases, F1 decreases. Doesn’t this tuning make it dataset-specific and the model sensitive modalities are missing or imbalanced?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposes an event boundary detection method, where the goal is to avoid over-segmentation and detect event boundaries with more semantically coherent regions. The framework is based on two steps: 1) adaptive thresholding based on several multimodal encoders and 2) refinement of candidate boundaries within short window. Experimental results include comparison with unsupervised approaches in ADL-GEBD and Ego4D.

### Strengths
- The goal of segmenting semantically coherent episodes, rather than focusing on frame-level or micro transitions, is meaningful and aligns with real-world applications such as summarization and long-form video understanding.
- The paper have shown rich ablation studies on weight fusion, selection of multimodal encoders, threshold selection, and context lengths.

### Weaknesses
1. **Heuristic-heavy design and overclaimed “cognitive grounding”:** The overall pipeline relies on a collection of heuristics (e.g., similarity thresholds, merging rules, pretrained feature similarities), with limited principled learning or optimization. At the same time, the paper repeatedly emphasizes a “cognitively grounded paradigm,” yet none of the components are derived from or validated against cognitive models or human studies. Besides, the step 2 is named "uncertainty-driven" whereas there is no uncertainty-aware design at all.
2. **Unclear experimental validations:** The paper claims to reduce micro-level transitions and detect semantically coherent episodes, but the experiments do not convincingly support this. The evaluation primarily compares against unsupervised methods, without showing whether fully supervised models indeed over-segment relative to the proposed semi-supervised approach. As a result, it is unclear whether the claimed “semantic coherence” is achieved in practice. Besides, the abstract claims this episodic segmentation can help cognitive agents, whereas such validation is completely missing afterwards.
3. **Generic multimodal integration:** The use of CLIP, DINOv2, captioning, and scene graphs follows standard multimodal fusion practices widely adopted in recent works for video understanding and event boundary detection. There is little evidence that this integration is novel or provides a substantial advantage over existing methods.
4. **Writing and presentation issues:** The paper’s organization and clarity need major improvement. Many sections are hard to follow, and Figure 1 looks incomplete. For example, the figure incorrectly labels “Level 1” as “uncertainty-driven conolidation” with typo, which actually belongs to Level 2, causing confusion from the start. Similarly, Table 1’s “GEBD Focus vs. Our Task Focus” comparison is also misleading. The task remains the same as event boundary detection while the method is different.

### Questions
See weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1
