# Connecting Where You Look With What You Understand: Trajectory-Driven Localized Understanding for Interactive Vision-Language Models

- Decision: Reject
- Scores: 6, 6, 4, 6

## Abstract
Recent Large Vision Language Models (LVLMs) demonstrate remarkable capabilities in image understanding and natural language generation. However, current approaches focus predominantly on global image understanding, struggling to simulate human visual attention trajectories and explain associations between generated descriptions and specific image regions. To address this challenge, we propose TraceVLM, a unified vision-language model that integrates trajectory-aware spatial understanding within an end-to-end framework. TraceVLM employs a Trajectory-aware Visual Perception (TVP) module for deep bidirectional fusion of visual features and trajectory information. We utilize a geometric simplification algorithm to extract semantic keypoints from raw trajectories and propose a three-stage training pipeline where trajectory information guides description generation and region localization. We further extend TraceVLM to attention trajectory-guided segmentation and video scene understanding tasks, enabling cross-frame trajectory tracking and temporal attention analysis. Based on large vision-language model reasoning capabilities, we construct the Reasoning-based Interactive Localized Narratives (RILN) dataset to enhance logical reasoning and interpretability. Extensive experiments on trajectory-guided captioning, text-guided trajectory prediction, image understanding, and segmentation demonstrate that TraceVLM achieves state-of-the-art performance, establishing a foundation for intuitive human-computer spatial interaction and interpretable visual understanding.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes TraceVLM, a unified LVLM that integrates trajectory-aware spatial reasoning into an end-to-end framework. The core idea is a Trajectory-aware Visual Perception (TVP) module that performs bidirectional fusion between visual tokens and continuous human attention trajectories via alternating cross-attention—enhancing visual features with trajectory context and refining trajectories with updated visual context. The model targets trajectory-conditioned captioning, text-guided trajectory prediction, referring localization/segmentation, and video understanding. The authors also construct RILN (320k), a reasoning-oriented dataset derived from Localized Narratives and related sources to support instruction-following spatial reasoning. Experiments report consistent gains vs. LVLM baselines on COCO-LN controlled caption/trajectory generation, regional captioning (VG/RefCOCOg/Ref-L4), and referring localization/segmentation, with a claimed 23% improvement in spatial reasoning when training on RILN.

### Strengths
1. Clear formulation of continuous attention as a first-class signal. The paper argues convincingly that static region prompts (boxes/masks/points) miss temporal continuity; modeling trajectories as dense signals of intent is well-motivated and novel within LVLMs. 

2. Architectural neatness. The TVP module’s two-stage, bidirectional cross-attention (trajectory→vision and vision→trajectory) is simple and compatible with standard LVLM stacks (QwenViT encoder + Qwen2.5-VL-7B LLM), enabling plug-and-play fusion without heavy custom heads.

3. Task coverage. One unified model handling trajectory-conditioned captioning, trajectory prediction, referring understanding, segmentation (via a special [SEG] token + lightweight decoder), and multi-frame video is ambitious and practically valuable for interactive systems.

### Weaknesses
1. Ambiguity around “semantic-guided” Douglas–Peucker. The method states that phrase-level semantic weights (from Qwen2.5-VL-72B) modulate local DP tolerance, but it’s unclear how these weights are computed, calibrated, and normalized across images/phrases, or how sensitive performance is to the weighting scheme vs. plain DP/uniform sampling. This is central to the trajectory tokenizer but currently under-specified for replication/ablation. 

2. Evaluation parity concerns. Several baselines (classic LVLMs) are not natively designed for trajectory inputs. It’s not fully spelled out how prompting/adapters were engineered so that comparisons on “controlled trajectory generation/captioning” are fair and equally optimized. The paper should detail adapters/prompts and ensure no hidden advantages from the new tokenizer/fusion. 

3. Segmentation details. The “segmentation codebook” and “lightweight decoder” are described at a high level. Training losses, resolution, codebook size, and how [SEG] tokens are interleaved with language tokens for mask generation should be clarified, especially since results are competitive with specialized systems.

### Questions
1. TVP design choice: Why alternating two cross-attention passes per block (Traj→Vis then Vis→Traj) rather than a single multi-query fusion? 

2. RILN contamination controls: Given that RILN uses outputs from frontier models, what checks guard against direct training-set leakage into evaluation sets (especially when LN/LNV images appear online)? Did you blacklist evaluation images/texts from synthetic generation prompts?

3. Generalization beyond LN style: LN trajectories may reflect annotator mouse traces during narration. How does TraceVLM perform on physiological gaze traces (eye-tracking) if available, or on touch trajectories gathered on mobile devices? Any zero-shot results?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces TraceVLM, a unified Large Vision-Language Model (LVLM) designed to address the limitations of existing LVLMs, which primarily focus on global image understanding and struggle to simulate human visual attention trajectories or explain the association between generated descriptions and specific image regions. Existing methods for localized understanding typically rely on static, discrete localization elements (such as bounding boxes or points), which fail to capture the continuity and temporal dynamics inherent in human visual attention. TraceVLM tackles this by directly predicting and interpreting human attention trajectories, treating them as fine-grained and temporally structured records of human focus.

### Strengths
1. First End-to-End Trajectory LVLM: TraceVLM is proposed as the first end-to-end Large Vision-Language Model (LVLM) designed to model human attention trajectories for bidirectional trajectory–language understanding. This directly addresses the limitation of previous models that rely on static, discrete localization elements, which struggle to capture the continuity and temporal dynamics of human visual attention.
2. Trajectory-aware Visual Perception (TVP) Module: The introduction of the TVP module is an original architectural contribution. This module enables deep bidirectional fusion of visual features and trajectory information through alternating enhancement and refinement stages, which captures the sequential patterns of irregular trajectories.
3. Semantic-Guided Geometric Simplification: To handle the inherent noise and redundancy in raw human trajectories, the authors innovated a pre-processing step using a semantic-guided variant of the Douglas-Peucker (DP) algorithm. This strategy effectively extracts semantically meaningful keypoints (achieving 91% compression from 410 points to 37 keypoints) while preserving geometric structures necessary for precise localization.
4. Novel Reasoning Dataset (RILN): The creation of the Reasoning-based Interactive Localized Narratives (RILN) dataset (320k samples) is a major original contribution. This dataset moves beyond the simple descriptive narratives of previous datasets (like Localized Narratives) by focusing on complex reasoning, instruction-following capabilities, and interactive trajectory reasoning QA, which is essential for advanced spatial understanding.

### Weaknesses
1. A core contribution is the Reasoning-based Interactive Localized Narratives (RILN) dataset (320k samples), designed to enhance logical reasoning and spatial understanding. However, this dataset is constructed using an advanced pipeline leveraging powerful Large Vision-Language Models (LVLMs) like GPT-4o, Qwen2.5VL-72B, and Gemini-2.5 Pro to automatically generate instructional samples, structured reasoning trees, and question prompts. While high-quality, this reliance on synthetic generation introduces potential risks of model bias. The reasoning patterns learned might reflect the biases or specific linguistic styles of the generating LLMs rather than reflecting the full diversity and unexpected complexity of real human-generated complex instructions or reasoning flows.
2. The raw human attention trajectories are formalized as continuous temporal point sequences T, where each point encodes spatial coordinates and temporal information $t_{j}$. To handle noise and redundancy, TraceVLM applies a complex two-stage preprocessing approach, featuring a semantic-guided variant of the Douglas-Peucker (DP) algorithm to extract "semantic keypoints". This process achieves significant compression (e.g., reducing 410 points to 37 keypoints, achieving 91% compression). Although efficient, relying on geometric simplification inherently involves a trade-off where some fine-grained, continuous information-especially the temporal dynamics of human visual exploration-may be lost or smoothed out.
3. The training regimen for TraceVLM is highly resource-intensive and structurally complex. The model is built on the Qwen2.5-VL-7B foundation model and uses a specific three-stage curriculum learning approach (Pretraining, Joint Training, and Instruction Fine-tuning). The reliance on a "Fixed Order" training strategy is explicitly validated as superior to a "Random Order" approach. This fixed, multi-stage process requires significant computational resources (approximately 3 days total on 16 A800 GPUs), making it potentially less accessible for smaller research groups and less flexible for adapting to new tasks without a complete architectural overhaul or extensive retraining.

### Questions
See weakness above, please.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents TraceVLM, a trajectory-aware large vision-language model (LVLM) designed to model human attention trajectories bidirectionally. The framework combines a Trajectory-aware Visual Perception (TVP) module and a large-scale Reasoning-based Interactive Localized Narratives (RILN) dataset. The model aims to capture the temporal and spatial continuity of gaze-like signals to improve spatial reasoning and interpretability in multimodal tasks such as trajectory-conditioned captioning, referring localization, and segmentation.

### Strengths
+ Clear motivation: The authors identify the discrete, static nature of existing region-based supervision as a limitation and propose a continuous trajectory-based approach to better simulate human attention.

+ Methodological coherence: The TVP module effectively integrates visual and trajectory representations through cross-attention and curriculum learning, as supported by the ablation studies

+ Various subtasks: Experiments span multiple subtasks (captioning, localization, segmentation) and demonstrate consistent quantitative improvements over prior baselines.

### Weaknesses
- Missing very closely related work
The paper does not cite or compare with [ref1], which already addressed interactive regional understanding in LVLMs. Both works share the same motivation—to move beyond global visual reasoning toward localized, human-interpretable grounding—and even rely on the same training data (Localized Narratives). Considering the nearly identical motivation and overlapping tasks, the paper’s novelty appears limited and should be more clearly distinguished from [ref1].

[ref1] Toward Interactive Regional Understanding in Vision-Large Language Models, NAACL 2024

- Unfair backbone comparison
The model’s reported gains over Qwen2.5-VL-7B are expected, since TraceVLM is trained directly on top of that same backbone with additional trajectory data. However, the proposed model is compared with other region understanding methods under different backbones. To make the comparison meaningful, the authors should either (1) reimplement other region-understanding methods (e.g., RegionVLM, PixelLLM) under the same backbone, or (2) train their model on different backbones to show consistent improvement regardless of the base model. Without such controlled experiments, the fairness and validity of the reported performance advantages remain questionable.

- Inconsistent and unexplained benchmark performance
According to Table 2, TraceVLM does not achieve state-of-the-art results on several benchmarks, and Table 3 shows only marginal improvements over baselines or even lower performance compared to certain prior models. However, the paper provides no analysis or discussion to explain these inconsistencies. The lack of qualitative or diagnostic analysis makes it unclear why the proposed method performs better in some tasks but fails to generalize consistently across benchmarks.

- Missing analysis of the bidirectional design
Although the paper repeatedly emphasizes the bidirectional nature of the proposed TVP module—where trajectory features refine visual representations and vice versa—there is no ablation directly examining this property. Table 4 only compares a simple “with vs. without TVP” setting, without disentangling the two directional attention flows. To validate the core claim, the authors should disable the visual-to-trajectory pathways and analyze their individual effects. Without such evidence, the purported advantage of bidirectional interaction remains unsubstantiated.

- No analysis of the 3-stage training procedure
The paper adopts a three-stage training pipeline, but never reports performance after each stage. Since the model jointly leverages pretraining, trajectory-conditioned tuning, and reasoning-based instruction learning, it is unclear which stage contributes most to the final improvement. Without such analysis, readers cannot tell whether the gain comes mainly from trajectory supervision or from later multimodal reasoning refinement. A simple comparison after each stage (e.g., Stage 1 → Stage 2 → Stage 3) would make the training strategy much more convincing.

### Questions
- Table 3 is partially obscured by Figure 5 due to layout overlap, making the numerical results invisible. Please fix the figure placement or adjust formatting so that both elements are clearly visible in the final version.

- How much additional inference time does the TVP module introduce compared to the baseline Qwen2.5-VL-7B?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors propose the first trajectory-aware VLM, TraceVLM, to handle the correspondence between the human attention trajectories and the linguistic representation of a video. To finetune the QwenVL-2.5, a dataset construction pipeline is created to generate high-quality reasoning-enabled training data. The experiments demonstrate the effectiveness of the model on both trajectory-aware tasks and related benchmarks.

### Strengths
1. The authors propose TraceVLM, the first VLM to achieve trajectory-aware video understanding/reasoning.
2. To facilitate the model training, the authors construct a data annotation pipeline to obtain the training data based on existing trajectory video data. And the experiments demonstrate the effectiveness of the dataset and training strategy.
3. The Geometric Simplification is efficient in reducing redundancy and noise.
4. Not only in trajectory-aware tasks, but also in other visual benchmarks, TraceVLM has comparable or superior performance compared with other VLMs.

### Weaknesses
Major:
1. In the introduction, line 42, the author claimed, “they often focus their attention on the primary regions of an image while neglecting surrounding contextual information, and may even be distracted by irrelevant areas”, and in Figure 1, the example depicts a man in a car holding a phone. The question is: (a) how to differentiate important context information and unimportant context information. Actually, in Figure 1, even humans will ignore the background because of the higher exposure and irrelevance of the main part of the image, and this is an exact example of NOT being distracted by irrelevant areas. (b) Is there any example showing “distracted by irrelevant areas”?
2. In terms of the motivation, can the authors add some discussion of the real-world application of such a technique? What is the point of developing trajectory-driven localized understanding? 
3. In Section 3.2.3, line 223, what is ϵ_base? And how is w_i determined for each segment?
4. In Section 3.2.5, what is the size of the segmentation codebook?
5. In the data generation pipeline, how to ensure the correctness of the reasoning generation? Does the predefined global-->object-->paragraph --> local reasoning chain (Figure 4) work for every case? Maybe when the question changes, the reasoning process will vary a lot, since the focus and the scene structure or the intrinsic logic will be totally different.
6. In the visualization of 4.5, it would be better if these cases also include the original Qwen2.5VL-7B, so that the attention map difference can be observed.

Minor:
1. Line 45, add a reference for this: “Research on human visual attention trajectories also plays an important role in domains such as virtual reality and autonomous driving.”
2. In Figure 4, Step 1, the text is unclear due to the color (light yellow/green part)
3. In Figure 4, Step 2-1, the Q1 and Q2 look incomplete.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
2
