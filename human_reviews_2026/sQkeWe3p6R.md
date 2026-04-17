# NO DARK DATA REQUIRED: BRIDGING THE GAP BETWEEN NORMAL AND LOW-LIGHT DETECTION VIA RETINEX DECOMPOSITION

- Decision: Reject
- Scores: 2, 2, 2

## Abstract
Conventional low-light object detection approaches typically involve distinct image enhancement modules before the detection process. This can lead to compromised performance due to misaligned objectives and reduced robustness in challenging visual contexts. Many existing methodologies either do not optimize both tasks jointly or overlook significant latent features that are essential for accurate detection. To address this issue, a novel end-to-end framework was proposed that was exclusively trained on normal-light images, eliminating the need for low-light data during the training phase. This approach drew inspiration from the Retinex theory, which separated images into reflectance (representing scene structure) and illumination (indicating lighting conditions). The proposed framework approximates this decomposition within the feature space. The architecture utilises deep multi-scale feature aggregation along with a reflectance-guided fusion pathway, enabling the adaptive integration of illumination-aware representations through element-wise modulation. Despite being trained on normal-light images, the framework demonstrates effective generalisation to low-light and visibility compromised environments. Comprehensive experiments conducted on both synthetic datasets (Pascal VOC) and real-world benchmarks (ExDark, RTTS) indicate that this method achieves enhanced detection accuracy and robustness, particularly in adverse lighting conditions, and outperforms current state-of-the-art techniques.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an end-to-end object detection architecture inspired by Retinex theory. The design features a decomposition module that estimates reflectance and illumination in feature space, with a novel fusion approach to produce illumination-invariant features for object detection. The key distinguishing property is that the model is trained solely on normal-light data but tested robustly on both synthetic and real low-light/foggy datasets. The method is benchmarked on Pascal VOC and ExDark and RTTS.

### Strengths
1. The model is trained exclusively on normal-light images which is practically significant to reduce data curation burdens.
2. The decomposition of deep features into reflectance and illumination via Retinex-inspired, feature-level processing seems a distinctive integration.

### Weaknesses
1. The paper is difficult to read due to a combination of disorganized content and poor layout. The logical progression of ideas is unclear, and the unprofessional typesetting, evidenced by significant blank space on page 3, detracts from the work's credibility.
2. The mathematical descriptions in Section 3.2–3.3 (especially around the decomposition and fusion) are rather high level. Crucial details such as the explicit forms of the aggregation $\mathcal{A}(\cdot)$, sampling method for constructing $L(x, y)$ and $R(x, y)$, channel alignment techniques, normalization procedures, and whether the element-wise fusion is normalized or bounded, are unspecified.
3. The paper does not adequately differentiate its proposed methodology from existing Retinex decomposition/fusion techniques.
 - Deep Retinex Decomposition for Low-Light Enhancement, BMVC18
 - IniRetinex: Rethinking Retinex-type Low-Light Image Enhancer via Initialization Perspective, AAAI25

### Questions
1. Can the authors provide explicit mathematical details for the aggregation and fusion steps, particularly a precise functional form for $\mathcal{A}$ and the normalization/activation used in $F_{i}^{\text{fused}}$? Are there learned or fixed weights, or dynamic selection mechanisms in fusion?
2. What is the process and rationale for selecting RepNCSPELAN4 blocks, and do alternative feature processing blocks (e.g., C2f, ELAN) materially affect performance? Quantitative ablation here would strengthen the architectural justification.
3. What are the failure points under extreme conditions (e.g., mAP vs. fog/darkness level)? Is there a critical threshold below which the proposed method degrades significantly earlier or later than the SOTA?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This manuscript attempts to introduce Retinex theory into the YOLO framework, but in reality, it is merely a patchwork of pre-existing concepts and methods.

### Strengths
The title of the manuscript is decent.

### Weaknesses
1. There are significant formatting problems, including large blank spaces on several pages (e.g., pages 3, 4, and 6) and improperly scaled tables (e.g., Tables 1, 2, and 3).
2. The proposed method is an unjustified assembly of classic methods from the vision field (YOLO and Retinex) with virtually no original design contributions.
3. The contributions summarized in the introduction are all based on existing methods, lacking any original design from the authors, and are poorly written. The author describes the method as an "AI model." The core of the work is merely applying different processing to feature maps of different scales within YOLO and claiming this constitutes a Retinex decomposition. This claim is unsubstantiated, and the author fails to provide a clear explanation, instead just restating concepts from Retinex theory and YOLO object detection.
4. The author's writing suggests a lack of familiarity with standard academic terminology in this field. For instance, summarizing their method as "an Artificial Intelligence (AI) solution" is not a phrasing I have encountered in computer vision or related fields.
5. The experiments in this manuscript are primarily compared against baseline and older methods, failing to include comparisons with the latest state-of-the-art (SOTA) approaches. Furthermore, only limited results are presented, which is insufficient to validate the effectiveness of the proposed method.

### Questions
Please refer to Weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This manuscript proposes a new end-to-end framework based on normal-light images for low-light image detection. The proposed method separates images into reflectance and illumination, which approximates this decomposition within the feature space. A multi-scale feature aggregation is introduced to learn illumination-aware representation.

### Strengths
1. Compared to multiple baselines, the method exhibits enhanced generalization under challenging lighting and weather conditions.
2. The framework attains high inference speed, enabling effective support for real-time applications.

### Weaknesses
1. The work as a whole lacks a distinct innovative core. Its technical approach largely manifests as a direct integration of the YOLO model with Retinex theory, without presenting substantial original theoretical advancements, nor conducting in-depth exploratory research on key technical bottlenecks. 
2. The introduction fails to provide an explicit and structured summary of the study’s contributions. In academic writing, a clear statement of contributions serves as a "guide" for readers to quickly identify the work’s core value and differences from prior studies. 
3. The paper’s formatting is extremely poor. This lack of rigor in the submission attitude has raised doubts about the quality of the paper’s content and the authenticity of its experiments

### Questions
Poor readability of the paper
Poor formatting of the paper.

### Soundness
2

### Presentation
1

### Contribution
1
