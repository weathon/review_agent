# Analyzing Dynamic Surgical Workflows Through Multi-Scale Vision-Language Reasoning

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Analyzing surgical workflow is critical for understanding complex procedural dynamics of a surgery. Current works focus on surgical workflow recognition to classify video streams into predetermined workflow sequences, inadequately representing the adaptive nature of clinical surgeries that respond to patient variability and evolving circumstances. We introduce the \textit{dynamic surgical workflow reasoning} task, which eliminates fixed workflow constraints to address these limitations. Supporting this paradigm shift, we present DySurg (\textbf{Dy}namic \textbf{Surg}ical Workflow), a comprehensive dataset containing over 100 hours of long surgical videos in real-world clinical recordings with annotated dynamic workflows across 7 major surgical categories. To enhance reasoning capabilities for this analysis, we propose an commentary-aligned video reasoning framework that constructs top-down visual reasoning sequences modeling surgeons' cognitive processes. Our approach aligns visual embeddings from surgical videos with semantic information extracted from video title and expert commentary, thereby aligning explainability with the visual representations. During inference, our model maps video frames to the semantic space to generate appropriate workflows, without the need of expert commentary. Extensive evaluations on the DySurg dataset demonstrates that our approach significantly outperforms existing large vision-language models (e.g. QWen2.5-VL) and surgical-specific pre-trained models (e.g., SurgVLP) in recognizing dynamic surgical workflows. All code, data, and models will be publicly released after the review process concludes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Dynamic Surgical Workflow Reasoning, a new paradigm replacing fixed surgical step recognition. It introduces DySurg, a 100-hour dataset of annotated surgical videos with expert commentary. The authors develop MSVLR, a multi-scale vision-language framework that aligns video and text features to model surgeons’ reasoning. MSVLR segments videos and generates step descriptions without needing expert input. Experiments show it significantly outperforms existing models in both accuracy and description quality.

### Strengths
1. This work proposes a novel formulation of dynamic surgical workflow reasoning, moving beyond fixed-step recognition to model real-time procedural variability, which better reflects the adaptive nature of real clinical surgeries.

2. This work introduces the DySurg dataset, a large-scale and richly annotated collection of over 100 hours of surgical videos with synchronized expert commentary, providing an unprecedented resource for visual-language learning in surgery.

3. This work develops the MSVLR framework, which uniquely aligns multi-scale visual representations with textual commentary to mimic surgeons’ hierarchical reasoning, demonstrating strong performance gains and clinical feasibility across seven surgical categories.

### Weaknesses
1. The proposed MSVLR framework is only evaluated on the DySurg dataset, which is collected from publicly available surgical teaching videos. These data may not accurately represent real intraoperative conditions such as occlusion, instrument clutter, or non-standard workflows. It remains unclear whether the model generalizes to live OR data or videos from different institutions.

2. Lack of comparative analysis with reasoning-focused models: Although comparisons are made with TimeChat, Qwen2.5-VL, and SurgVLP, the paper omits evaluation against reasoning-oriented video-language models such as Video-LLM (Zhang et al., 2024) or GPT-4V-based frameworks.

3. Evaluate cross-hospital or real OR data generalization, possibly through fine-tuning or zero-shot testing on other datasets (e.g., Cholec80, EndoVis).

Ref:
[1] OphNet: A Large-Scale Video Benchmark for Ophthalmic Surgical Workflow Understanding

[2] OphCLIP: Hierarchical Retrieval-Augmented Learning for Ophthalmic Surgical Video-Language Pretraining

[3] Ophora: A Large-Scale Data-Driven Text-Guided Ophthalmic Surgical Video Generation Model

[4] Towards Dynamic 3D Reconstruction of Hand-Instrument Interaction in Ophthalmic Surgery

### Questions
Could the authors clarify how the commentary alignment module behaves under noisy or incomplete commentary? Since real-world ASR transcripts or live commentary can be error-prone, it would be valuable to know whether the alignment is robust to missing or inaccurate text and whether any data augmentation or filtering was applied during training.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces the problem of „dynamic surgical workflow recognition“, where videos need to be segmented into keysteps, and the keysteps need to be described with text captions (dense video captioning). The problem specifically aims at complex and unpredictable workflows for procedures that are less standardized than the typical cholecystectomy.
To address this problem, the study compiles a new benchmark dataset („DySurg“) of 470 surgical videos across 7 major surgical categories that were previously uploaded to the WebSurg educational platform. Notably, the spoken expert commentary for each video is transcribed into text, and the text transcriptions are included in the dataset.
In addition, the work presents a method for video segmentation and captioning („MSVLR“), which leverages the transcribed expert commentary as additional signal at training time for learning high- and mid-level video representations.

### Strengths
1. Important and timely topic, to be useful in clinical practice, automatic methods for surgical workflow recognition need to be able to handle workflows that are more complicated than the examples in the well-known Cholec80 benchmark. This study presents a commendable step into this direction.
2. The presented DySurg dataset has the potential to serve as a novel benchmark for surgical workflow recognition on realistic, challenging cases. A big plus is the inclusion of commentary in text form.
3. The idea to integrate the available expert commentary when learning visual representations is appealing and has the potential to advance the field of vision-language modeling for surgical applications.
4. The authors promise to release all code, data, and models in the future.

### Weaknesses
Major
1. The problem definition and the method description lack detail and rigor to an extent that limits comprehensibility.
1a. Problem definition: The notion of a „keystep“ remains undefined. It is also not clear what a „keyframe“ is and how it differs from a common frame in the video. In line 210/211, it is not clear how „video workflows“ are defined.
1b. Video segmentation: The inputs at the lowest level (e.g. f_{v_t} in line 304/5) are undefined.
1c. Video segmentation: It is unclear how the „key step indicators“ y and y’ (at the mid and low level) are translated into a segmentation of the video. It is also not specified how the segmentation would be parameterized: the ground truth s („true video keystep“) appears in equations (3) and (4) without explanation.
1d. Text generation: It is unclear how the two modules (cross-level and inter-level workflow text generation) interact or how their predictions are fused for a final result.
2. The presented method seems inadequate for solving the problem and lacks motivation.
2a. It seems that the method is not suitable for real-time analysis of the intraoperative video stream („online recognition“), where the complete video is not available yet. In particular the mid-level segmentation, where larger portions of the video are analyzed at once, seems to violate online requirements.
2b. The method seems to repeatedly compute attention between two sequences of length 1 (i.e. individual tokens or feature vectors). This does not make much sense as the result of an individual query vector that attends over an individual key/value vector is simply the value vector itself. In combination with a residual connection, this operation basically sums the query and the value vector. What is the point?
2c. When parsing the video, it is unclear how the different levels interact and benefit from each other. In what form does information flow from top to bottom?
2d. The results are discouraging, reaching scores <0.1 on the text generation task and limited overlap with the ground truth segmentations.
3. The paper states that the presented method „models surgical visual reasoning processes aligned with surgeons’ higher-level cognitive functions“ (line 98/99) but lacks proper justification for this claim.
4. Evaluation seems superficial:
4a. The evaluation metrics should be defined in more detail.
4b. It is not clear to the reviewer why „Recall@1“ was chosen as the only metric to measure the segmentation quality.
4c. Details are missing about how the baseline models for comparison are employed, finetuned and adjusted to be applicable to what is basically a dense video captioning task. In particular, it is unclear whether the baselines were finetuned on the DySurg dataset at all.
4d. The presented MSVLR model seems to be intended to segment and describe surgical workflows of any kind. Therefore, an evaluation on common benchmarks for surgical phase recognition (Cholec80, MultiBypass140) should be included.

Minor
5. References are missing, e.g. for Transformer/attention and Generalized IoU
6. More motivation could be provided for why a natural-language description of keysteps would be required as opposed to a pre-defined categorical label. Which clinical applications are intended to be implemented based on the generated segmentation and text?
7. More information on the DySurg dataset could be provided, such as the number and frequency of different surgical steps, the number of surgeons and hospitals involved, or the variability of workflows from the same surgical category.
Limitations of the dataset should be discussed, including the limited scale (less than 100 videos per surgical category), the relatively short average duration of videos, and potential differences from ordinary intraoperative recordings, which were not edited for educational purposes.

### Questions
Under which conditions can WebSurg videos be used for research and be re-distributed (e.g. in form of a curated benchmark dataset)? What about the copyright of the surgeons who created the educational videos?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces the novel task of dynamic surgical workflow reasoning, arguing that traditional fixed-step analysis is insufficient for real-world clinical variability. To support this new task, the authors present DySurg, a new 100+ hour dataset of surgical videos across 7 categories, annotated with dynamic workflows and aligned expert commentary. They also propose MSVLR, a multi-scale vision-language framework that aligns video features with semantic information (titles, commentary) to parse videos into procedure-specific steps. The method is shown to significantly outperform existing surgical-specific and general vision-language models on this new benchmark.

### Strengths
- The primary strength is the task definition. The shift from fixed to dynamic workflows is well-motivated, necessary, and highly relevant to advancing computer-assisted intervention systems for actual clinical practice.
- The DySurg dataset is a significant contribution to the field. Its size, multi-category nature, and especially the inclusion of temporally-aligned expert commentary provide a valuable resource for this new, more complex task.
- The MSVLR model's design is intuitive. The use of multi-scale reasoning (high, middle, low) combined with a commentary-alignment module to model a surgeon's cognitive process is a sound approach.

### Weaknesses
- The paper introduces a new dataset involving real clinical surgical videos and expert annotations. The authors state the data is from a public source, but they fail to provide a dedicated Ethics Statement addressing patient consent, data privacy, and anonymization procedures for their dataset creation and use. There is no mention of an IRB review for this new collection, which is a significant concern for any work dealing with clinical data. Additionally, a formal Reproducibility Statement detailing data, code, and model availability beyond a brief mention is absent.
- The model is trained and evaluated exclusively on the newly proposed DySurg dataset. It is unclear how this method would perform on other, existing surgical video datasets (even for fixed-phase recognition). This makes it difficult to distinguish how much of the performance gain is from the model architecture versus its specialization to the DySurg data structure.
- The method relies heavily on expert commentary during training to learn the alignment. While it is a strength that commentary is not needed for inference, acquiring such high-quality, temporally-aligned commentary for new procedures or institutions is a major practical bottleneck. This could limit the scalability and adoption of the method.
- The core idea of the MSVLR model fusing different granularities of semantic information (title, commentary) with visual features in a multi-scale manner, is a common technique in MLLM training, such as Qwen3-VL. The paper does not clearly articulate what specific architectural insights or novel components differentiate this approach from existing hierarchical fusion methods.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces DySurg, a >100-hour dataset of 470 real clinical surgical videos across 7 categories with dynamic, video-specific keystep annotations and aligned expert commentary. It formulates dynamic surgical workflow reasoning (vs. fixed-phase recognition) and proposes MSVLR, a multi-scale vision–language framework that (i) parses videos into keysteps via a two-stage visual parser, (ii) aligns visual features with title/commentary using cross-attention, and (iii) generates per-keystep textual descriptions via LoRA-tuned LLM heads (inter- and cross-level). On DySurg, MSVLR outperforms strong baselines (Qwen2.5-VL, UniVTG, TimeChat, SurgVLP/PeskaVLP) on keystep segmentation (R@0.5: 0.32) and description quality (BLEU-1: 0.088), with ablations showing the importance of commentary alignment.

### Strengths
Clear new task (dynamic, title-conditioned keysteps) and dataset; commentary-alignment at high/mid levels is a neat mechanistic prior for surgical reasoning.

Competitive baselines from both video-LLMs (TimeChat) and surgical VLP (SurgVLP/PeskaVLP) are included; MSVLR wins by a sizable margin on DySurg.

Method pipeline and losses are described with sufficient mathematical detail; the inference setting (no commentary) is explicit.

Addresses a limitation often noted in surveys—real surgeries are non-rigid and context-dependent—which existing fixed-phase systems underserve.

### Weaknesses
Training aligns frames to titles and transcribed commentary, while inference uses only titles. Without careful controls, descriptive overlap may inflate caption metrics. Recommend reporting results with title masking and commentary-drop during training and showing robustness.

DySurg is sourced from WebSurg narrated edits; real OR feeds (non-narrated, multi-view, noisy, privacy-constrained) differ substantially. Compare on non-WebSurg datasets (e.g., depth-camera OR or other phase corpora) to test transfer.

BLEU/ROUGE are low in absolute terms and may not reflect clinical usefulness; consider human evaluation (surgeons) for actionability and safety, and task-specific measures (temporal edit distance; coverage of critical safety steps).

While TimeChat/UniVTG provide temporal localization, dynamic procedure-conditioned keysteps could also be approximated by PeskaVLP/SurgVLP with retrieval+captioning; include such composed baselines to isolate the benefit of commentary alignment.

No confidence intervals or significance tests; limited per-category failure analysis (e.g., why Upper-GI underperforms). Provide CIs over multiple seeds and confusion analyses of over/under-segmentation.

### Questions
How were “dynamic keysteps” normalized across annotators to avoid proliferating synonyms? Any inter-rater agreement stats (κ) for boundaries and labels?

Whisper transcripts were QC’d—did you simulate residual noise or accents? How sensitive is alignment to WER perturbations?

What happens if titles are generic or misleading? Please report performance with obfuscated or shuffled titles to quantify reliance on the objective prompt.

Can the model handle multi-objective procedures (combined surgeries) or emergent events (complications)? Any plan to support branching workflows?

Any preliminary reader study with surgeons assessing whether generated keysteps are safe/complete enough for training or assistance?

### Soundness
3

### Presentation
3

### Contribution
3
