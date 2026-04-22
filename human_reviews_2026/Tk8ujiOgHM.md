# Unveiling Perceptual Artifacts: A Fine-Grained Benchmark for Interpretable AI-Generated Image Detection

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Current AI-Generated Image (AIGI) detection approaches predominantly rely on binary classification to distinguish real from synthetic images, often lacking interpretable or convincing evidence to substantiate their decisions. This limitation stems from existing AIGI detection benchmarks, which, despite featuring a broad collection of synthetic images, remain restricted in their coverage of artifact diversity and lack detailed, localized annotations. To bridge this gap, we introduce a fine-grained benchmark towards eXplainable AI-Generated image Detection, named X-AIGD, which provides pixel-level, categorized annotations of perceptual artifacts, spanning low-level distortions, high-level semantics, and cognitive-level counterfactuals. These comprehensive annotations facilitate fine-grained interpretability evaluation and deeper insight into model decision-making processes. Our extensive investigation using X-AIGD provides several key insights: (1) Existing AIGI detectors demonstrate negligible reliance on perceptual artifacts, even at the most basic distortion level. (2) While AIGI detectors can be trained to identify specific artifacts, they still substantially base their judgment on uninterpretable features. (3) Explicitly aligning model attention with artifact regions can increase the interpretability and generalization of detectors.
The data and code are available at: https://github.com/Coxy7/X-AIGD.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces X-AIGD, a benchmark for interpretable AI-generated image (AIGI) detection featuring pixel-level masks and a three-level taxonomy of perceptual artifacts (low-level distortions, high-level semantics, cognitive-level counterfactuals). Data are paired: each fake is generated from a caption of a real image using 13 modern generators; a subset is densely annotated via multi-round human labeling. Two tasks are specified: Authenticity Judgment (AJ) and Perceptual Artifact Detection (PAD) (category-aware and category-agnostic). Empirically, the authors show: (i) popular detectors largely do not align with human-perceived artifacts; (ii) PAD as an auxiliary task only marginally improves AJ; (iii) an attention-alignment loss that encourages ViT attention to overlap artifact masks,

### Strengths
1. The paper tackles a core need in AIGI detection: explainability grounded in evidence. X-AIGD’s large, human-effort annotation effort is itself a significant contribution—creating supervision that most existing benchmarks lack and that practitioners actually need to diagnose model behavior.


2. Thoughtful three-tier taxonomy (low → semantic → cognitive): The taxonomy organizes artifacts by perceptual level, which is both conceptually coherent and practically useful. It supports analyses across levels (e.g., low-level texture vs. semantic object integrity vs. cognitive/commonsense violations), and invites targeted model improvements rather than one-size-fits-all heuristics.

### Weaknesses
1. My own experience annotating AIGC artifacts suggests substantial inter-annotator variation—especially for semantic flaws and low-level distortion— that the paper does not adequately discuss this important issue. 

   * **Granularity ambiguity.** For the “wrong number of fingers” example in Fig. 1, some annotators box the entire palm while others isolate only the finger region. For the misspelled “McDonald’s,” some include the apostrophe while others do not. Both are defensible, but they produce inconsistent spatial supports and labels that are hard to reconcile.
   * **Diffuse low-level cues.** Low-level distortions (e.g., “over-smooth skin” to annotate a fake person) are inherently fuzzy; deciding “how much” of a face to include varies widely by annotator.
   * **Why it matters:** This variance directly affects the reliability of the proposed benchmark and any conclusions drawn from artifact-aligned training.

2. Insufficient design for precise and unified human annotation.  The value of X-AIGD hinges on label precision and consistency, yet the paper offers little methodology to *achieve* them.

   * **Missing protocol depth.** There is no clear calibration phase, no adjudication workflow for contentious cases, and no rubric that operationalizes the taxonomy (positive/negative exemplars, near-miss guidance).
   * **Coarse quality assessment.** Fig. 2 provides only a coarse human assessment; there is no formal inter-annotator reliability.
   * Without rigorous agreement analysis and a documented process to reduce disagreement, it is hard to trust the benchmark as a gold standard rather than a noisy training signal.

3. Marginal ablation gains for attention alignment (Table 4).  The reported improvements are small and could plausibly be explained by training noise or seed variance.

### Questions
1. Comprehensive assessment of X-AIGD: expand the dataset validation beyond the current coarse plots and include a side-by-side comparison with other annotated resources (e.g., Legion) along these dimensions.

2. Synthetic edit–driven annotations as a unification strategy: Human annotations of semantic/low-level flaws are noisy. Why not also leverage modern image-edit models (e.g., FLUX-kontext, GPT-4o) to produce programmatically controlled flaws with auto-derived pixel masks from edit operations? Please discuss pros/cons and provide an experiment comparing X-AIGD (human) vs Edit-AIGD (synthetic edits) vs Hybrid (human+synthetic).

3. Using X-AIGD to strengthen VLMs under fair conditions: Since X-AIGD offers pixel-level, typed supervision, it’s natural to test whether it measurably improves a strong VLM (e.g., Qwen2.5-VL) on AJ and PAD. Add a controlled finetune where only supervision differs (with/without artifact masks), holding training budget, data volume, and augmentations constant across VLM baselines.

### Soundness
3

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
This paper introduces an interpretable image forgery benchmark that includes pixel-level masks and artifact categories, featuring a fine-grained taxonomy of seven subcategories grouped into three major types, ranging from low-level distortions to high-level semantic or commonsense anomalies. Moreover, the authors demonstrate that existing methods make limited use of perceptible artifacts, relying primarily on high-level, imperceptible cues for discrimination. To address this issue, the authors design an **attention alignment loss** that explicitly guides the model to focus on perceptible artifact regions, achieving superior performance as a result.

### Strengths
1. The proposed interpretable benchmark with pixel-level artifact masks is of substantial value to the research community, promoting further progress in explainable image forgery detection and attribution.  
2. The paper reveals that current detectors make limited use of perceptible artifacts. By introducing an attention alignment loss, the authors encourage the model to make judgments from a human perceptual perspective, which adds credibility to the interpretability of the approach.

### Weaknesses
1. The paper lacks detailed information about whether and when the proposed benchmark will be publicly released. Clear disclosure of dataset availability is essential for reproducibility and community adoption.  
2. The claim that existing detectors rely on imperceptible artifacts is indeed intuitive. To further validate the effectiveness of the proposed attention alignment loss, more ablation studies are needed. For example:  
   - (1) What happens if the ground-truth mask is randomly assigned?  
   - (2) What if attention is forced to align with a normal (artifact-free) object region in the image (which can be segmented using tools such as SAM)?  
These experiments would help clarify whether the performance gains truly stem from aligning attention with perceptible artifact regions    rather than from generic spatial regularization effects.

### Questions
Please refer to the *Weakness* section for detailed explanations.  
If the authors can adequately address my concerns, I would be very willing to raise my score.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces X-AIGD, a high-quality, fine-grained benchmark for interpretable AI-generated image detection. It provides pixel-level, multi-level  artifact annotations on paired real-fake images generated from 13 state-of-the-art text-to-image models. Using this benchmark, the authors systematically reveal that existing AIGI detectors barely rely on human-perceptible artifacts, even at the lowest distortion level. They further explore artifact detection as an auxiliary task and attention alignment, showing that explicitly grounding model attention to annotated artifact regions significantly improves both interpretability and cross-model generalization.

### Strengths
- X-AIGD introduces the first benchmark with paired real-fake images and pixel-level annotations across 7 fine-grained artifact categories, spanning low-level distortions, high-level semantics, and cognitive-level counterfactuals, enabling grounded interpretability evaluation.

- Empirical analysis surprisingly reveals that SOTA detectors achieve good performance without relying on perceptual artifacts—contrary to common intuition that visible flaws should be primary cues, as evidenced by uncorrelated performance across fidelity metrics in Fig. 3 and severely misaligned heatmaps in Fig. 4.

- Attention alignment to ground-truth artifact regions significantly improves both interpretability and cross-generator generalization, providing a practical and effective method to make AIGI detection more human-trustworthy.

### Weaknesses
- No comparison or integration with MLLM-based explanation methods (e.g., GPT-4o, LLaVA) is provided, despite critiquing their lack of grounding—missing an opportunity to show whether X-AIGD enables better spatial reasoning in language models.

- Inter-annotator agreement (e.g., IoU-based Fleiss’ κ) is not reported, especially critical for subjective high-level and cognitive-level artifacts, potentially affecting reproducibility.

### Questions
Could the authors provide the results of MLLM-based explanation methods?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents X-AIGD, a benchmark for interpretable AI-generated image detection. The dataset pairs real and generated images and provides pixel-level masks of perceptual artifacts. Using this benchmark, the authors show that current detectors, despite strong real/fake accuracy, often do not actually focus on these artifact regions. To improve this, they add an artifact segmentation auxiliary task and an attention alignment loss. These changes yield more interpretable predictions and better cross-dataset generalization.

### Strengths
- The dataset contribution is concrete and timely, providing pixel-level artifact masks with semantic labels and paired real/fake images. This directly supports studying why a detector made its decision instead of treating the task as only real or fake.
- The analysis of existing detectors is systematic. The authors compare saliency and attention maps with ground-truth artifact masks and show that they often do not overlap, which raises doubts about how explainable current high-performing detectors actually are.
- The paper appropriately introduces perceptual artifact segmentation as an auxiliary task and constrains the model to focus on artifact regions through attention alignment. This approach leads to more stable F1 scores across datasets and more consistent visual explanations, directly addressing the limitations of prior detectors.
- The writing quality is strong, and the logic of the paper is very clear and easy to follow.

### Weaknesses
- There is a growing line of work that uses multimodal large language models (MLLMs) to "look at an image and generate textual explanations," and claims to identify "where it is fake and why it is fake." The authors briefly mention related efforts (e.g., using MLLMs to generate explanations), but a more explicit discussion of works such as [1,2,3] would make the connection to this direction clearer for readers and better contextualize the contribution.
- The authors report 12 human annotators, 3 annotation rounds, and subjective confidence scores, indicating high pixel-level quality. However, the full-resolution / region-level protocol is extremely expensive, and ultimately only ~3k forged samples received high-quality annotations. This cost is understandable, but it raises a concern: is the pipeline realistically reproducible or extensible for larger-scale data? The paper would be stronger with a discussion of scalability and reproducibility, and possible ways to reduce cost (e.g., using MLLMs to assist or pre-label forged regions).
- **Typos**: On line 52-53, “X-AIGD (eXplainable AI-**Genereted** image Detection)” should be changed to “X-AIGD (eXplainable AI-**Generated** Image Detection)”

[1] Heie: Mllm-based hierarchical explainable aigc image implausibility evaluator

[2] Spot the fake: Large multimodal model-based synthetic image detection with artifact explanation

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
