# GLaVE-Cap: Global-Local Aligned Video Captioning with Vision Expert Integration

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Video detailed captioning aims to generate comprehensive video descriptions to facilitate video understanding. Recently, most efforts in the video detailed captioning community have been made towards a local-to-global paradigm, which first generates local captions from video clips and then summarizes them into a global caption. However, we find this paradigm leads to less detailed and contextual-inconsistent captions, which can be attributed to (1) no mechanism to ensure fine-grained captions, and (2) weak interaction between local and global caption generation.
To remedy the above two issues, we propose **GLaVE-Cap**, a **G**lobal-**L**ocal **a**ligned framework with **V**ision **E**xpert integration for **Cap**tioning, which consists of two core modules: TrackFusion enables comprehensive local caption generation, by leveraging vision experts to acquire cross-frame visual prompts, coupled with a dual-stream structure; while CaptionBridge establishes a local-global interaction, by using global context to guide local captioning, and adaptively summarizing local captions into a coherent global caption. Besides, we construct **GLaVE-Bench**, a comprehensive video captioning benchmark featuring 5$\times$ more queries per video than existing benchmarks, covering diverse visual dimensions to facilitate reliable evaluation.
We further provide a training dataset **GLaVE-1.2M** containing 16K high-quality fine-grained video captions and 1.2M related question-answer pairs. Extensive experiments on four benchmarks show that our GLaVE-Cap achieves state-of-the-art performance. Besides, the ablation studies and student model analyses further validate the effectiveness of the proposed modules and the contribution of GLaVE-1.2M to the video understanding community. The source code, model weights, benchmark, and dataset will be open-sourced.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
GLaVE-Cap is a Global-Local aligned video captioning framework that generates fine-grained and contextually consistent descriptions by bridging local and global information. To support evaluation and training, the authors construct GLaVE-Bench, and release a large-scale dataset GLaVE-1.2M containing 16K detailed captions and 1.2M QA pairs. Experiments on multiple state-of-the-art video-language models, including GPT-4o and Qwen2.5-VL-72B, demonstrate that GLaVE-Cap achieves superior fine-grained understanding and strong generalization across diverse benchmarks.

### Strengths
1. The paper proposes GLaVE-Cap, which combines vision-expert tracking with semantic-level global–local caption alignment, representing a creative bridge between traditional CV modules and large vision-language models.
2. It contributes two large-scale datasets, GLaVE-1.2M and GLaVE-Bench, enabling systematic and fine-grained evaluation across multiple video captioning dimensions, with extensive experiments on strong VLM backbones showing consistent gains.
3. The paper is well written and supported by clear figures and ablations, making a complex multi-stage pipeline understandable and highlighting the empirical value of detailed video captioning.

### Weaknesses
1. The GLaVE-Cap pipeline is overly complex and relies on multi-stage, caption-level integration rather than end-to-end feature fusion. This design increases the system’s dependency on intermediate textual outputs, which may propagate or amplify errors from earlier stages, and limits the model’s ability to maintain visual consistency across frames. As a result, the framework sacrifices efficiency and holistic visual-text alignment.

### Questions
1. Case studies mainly involve examples with only one salient object in motion. How would the proposed method behave when multiple objects move simultaneously within the same scene, or when multiple events occur concurrently in overlapping spatial regions? Would the model remain stable and consistent under such conditions?

2. Why did the authors choose to separately integrate Grounding DINO and SAM 2 instead of directly using Grounded-SAM 2, which already provides joint grounding and segmentation capabilities?

3. The method assumes that the VLM can reliably align each overview sentence with a specific keyframe range and then uses this alignment to drive downstream scene segmentation. However, the model only sees sampled keyframes and may propagate hallucinated boundaries.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces GLaVE-Cap, a novel framework for fine-grained video captioning that addresses limitations of existing local-to-global captioning paradigms. The method integrates two core modules: TrackFusion, which uses vision experts and a dual-stream architecture to generate detailed and consistent local captions, and CaptionBridge, which aligns local and global captioning through context injection and adaptive scene-level summarization. The authors also contribute GLaVE-Bench, a comprehensive evaluation benchmark with significantly more queries per video than existing datasets, and GLaVE-1.2M, a large-scale training dataset with fine-grained captions and QA pairs. Extensive experiments demonstrate state-of-the-art performance across multiple benchmarks, and ablation studies validate the effectiveness of the proposed components.

### Strengths
1. GLaVE-Bench and GLaVE-1.2M are valuable contributions, offering multi-scene videos with dense annotations that support more reliable and fine-grained evaluation and training.
2. The method achieves SOTA results across multiple benchmarks and generalizes well across different VLM backbones, demonstrating high performance.

### Weaknesses
1. The core "local-to-global" paradigm itself is not new, and the individual components (using vision experts for grounding, dual-stream processing, global context injection) have been explored in related forms in prior image and video understanding works. The primary novelty lies in the specific integration and orchestration of these ideas into a cohesive pipeline for this task.
2. The reliance on multiple vision experts and a multi-stage pipeline (e.g., for object tracking, dual-stream captioning, and scene segmentation) is computationally expensive, which may limit real-time or scalable deployment. Furthermore, the marginal performance improvement appears to come at the cost of a significant increase in computational load. While the paper focuses on captioning quality, it does not address critical deployment metrics like computational cost, latency, throughput, or efficiency. The authors should compare these metrics with those of other methods.
3. While hallucination is briefly evaluated, a more thorough analysis of how hallucinations propagate through the multi-stage pipeline is lacking.
4. The framework still relies on keyframe-based processing and may miss important visual elements in cluttered scenes, as acknowledged in the limitations. A shift to object-centric captioning is suggested but not implemented.

### Questions
1. The dual-stream structure in TrackFusion is a key component. Did you experiment with merging the dynamic and static streams in a different order or through a more integrated, iterative process rather than a final merge? What was the rationale for the chosen sequential approach?
2. The pipeline heavily relies on external vision experts (Grounding DINO, SAM 2). How critical is the accuracy of these models to the final caption quality? Have you analyzed how noise or errors in the object masks or tracking propagate through your system?
3. The multi-stage pipeline seems computationally intensive. Could you provide a rough estimate of the inference time and cost compared to a simpler, single-pass VLM approach for a typical video in your benchmark?

### Soundness
3

### Presentation
1

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
The paper presents GLaVE-Cap, a global–local aligned video captioning framework that integrates vision experts through TrackFusion and summarizes local captions via CaptionBridge to improve contextual coherence and detail accuracy. It also introduces GLaVE-Bench and GLaVE-1.2M, two benchmarks for evaluating fine-grained captioning and question answering on multi-scene videos.

### Strengths
1. The paper identifies a meaningful gap in video captioning—the weak interaction between local and global descriptions—and introduces a novel global–local alignment framework to address it.
2. The proposed TrackFusion and CaptionBridge modules form a coherent architecture that links frame-level detail modeling with global summarization, while integrating “vision experts” for object-level reasoning. 
3. The introduction of GLaVE-Bench and GLaVE-1.2M provides valuable large-scale benchmarks for fine-grained captioning and QA evaluation on multi-scene videos, which could benefit the broader research community.

### Weaknesses
1.	Although TrackFusion integrates object tracking and keyframe-based prompts, it lacks explicit mechanisms for multi-scale temporal reasoning. The model remains clip-level and fixed-window, limiting its ability to adapt to videos with diverse temporal dynamics.
2.	The proposed GLaVE-Bench dataset contains only 55 evaluation videos, which may be insufficient to support claims of generality and scalability.
3.	The evaluation omits comparisons with recent large-scale Video-Language Models (e.g., Qwen2-VL, InternVL3), making it unclear whether the proposed approach remains competitive in the current multimodal LLM landscape.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes GLaVE-Cap, a novel global-local aligned video captioning framework integrating vision experts to improve fine-grained and contextually consistent video captions. It introduces two key modules:
- TrackFusion integrates Grounding DINO and SAM 2 for cross-frame object tracking and uses a dual-stream architecture (differential+detailed captions) to capture both static and dynamic details.
- CaptionBridge injects a global overview caption to guide local captioning and performs adaptive scene-level summarization for coherent video-level captions.

Additionally, the authors introduce:
- GLaVE-Bench, a new benchmark with ~5x more queries per video than existing datasets.
- GLaVE-1.2M, a large-scale dataset (16K videos, 1.2M QA pairs).

Experiments on GLaVE-Bench, Video-MME, VidCapBench, MVBench show state-of-the-art (SOTA) results. A student model (GLaVE-7B) fine-tuned on GLaVE-1.2M also outperforms baselines, validating dataset quality.

### Strengths
1. The paper has conceptual novelty by moving beyond the local-to-global paradigm by explicitly modeling bidirectional local-global interaction.

2. Integration of vision experts:
Clever combination of Grounding DINO+SAM2 for robust object tracking, plus dual-stream design effectively captures both dynamic and static details.

3. The paper proposes new benchmark and dataset.
GLaVE-Bench and GLaVE-1.2M fill an evaluation and data gap in fine-grained video captioning. 
The QA-per-video (118 vs. 16) and multi-scene structure make it far more comprehensive.

4. Evaluations on 4+ major benchmarks with both closed (GPT-4o) and open (Qwen2.5-VL-72B) models show consistent SOTA performance.

5. Detailed ablations quantify the contribution of each module. Appendix provides thorough prompt designs and reproducibility materials.

6. Both quantitative and qualitative results (including user study in video generation) convincingly demonstrate improved caption granularity and contextual consistency.

### Weaknesses
1. **Limited novelty in components:**
Though integration is well-engineered, both TrackFusion (based on Set-of-Mark+vision experts) and CaptionBridge (context injection+summarization) build on known elements rather than introducing fundamentally new algorithms.

2. **Dependence on powerful LLMs:**
Most evaluations rely on GPT-4o and Qwen2.5-VL-72B. It's unclear how much gain comes from the model scale versus method design. Smaller models (e.g., Qwen2.5-VL-7B) perform notably worse.

3. Since GLaVE-Bench and GLaVE-1.2M are built using GLaVE-Cap captions, potential data leakage or alignment bias could inflate results, even if mitigated by re-captioning and human verification.

4. The framework requires multiple calls to large VLMs and expert models (Grounding DINO, SAM2), making it computationally heavy and less practical for large-scale or real-time applications.

5. While video generation results are interesting, the paper could better relate fine-grained caption quality to downstream generation metrics quantitatively.

### Questions
1. How is "adaptive scene segmentation" quantitatively validated beyond accuracy gains---are the segmentation boundaries human-verified?

2. Were hallucination rates or factual alignment metrics (e.g., VideoHallucer subset) computed per-stage to confirm reduced propagation?

3. Can the dual-stream design be generalized to text-only summarization tasks?

4. How much of the performance gain remains if using smaller backbones ($\leq$7B), without GPT-4o supervision?

5. How does GLaVE-Cap handle videos with dense camera motion or overlapping actions (e.g., sports broadcasts)?

6. Finally, I suggest resolving my concerns in Weaknesses.

### Soundness
4

### Presentation
4

### Contribution
4
