# Text-Aware Image Restoration with Diffusion Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
While diffusion models have achieved remarkable success in natural image restoration, they often fail to faithfully recover textual regions, frequently producing plausible yet incorrect text-like patterns, a phenomenon we term text-image hallucination. To address this limitation, we propose Text-Aware Image Restoration (TAIR), a task requiring simultaneous recovery of visual content and textual fidelity. For this purpose, we introduce SA-Text, a large-scale benchmark of 100K high-quality scene images with dense annotations of diverse and complex text instances. We further present a multi-task diffusion framework, TeReDiff, which leverages internal features of diffusion models to jointly train a text-spotting module with the restoration module. This design allows intermediate text predictions from the text-spotting module to condition the diffusion-based restoration process during denoising, thereby enhancing text recovery. Extensive experiments demonstrate that our approach faithfully restores textual regions, outperforms existing diffusion-based methods, and achieves new state-of-the-art results on TextZoom, an STISR benchmark considered a subtask of TAIR. The code, weights, and dataset will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduced the Text-Aware Image Restoration (TAIR) task, which requires simultaneous recovery of visual content and textual fidelity. And proposed corresponding diffusion-based method TeReDiff, which leverages text-spotting task to supervise intermidiate features and guide the optimization process. Also the paper built high-quality scene images SA-Text benchmark with dense annotations of diverse and complex text instances. Extensive experiments demonstrate the effectiveness of the proposed method.

### Strengths
Strengthens:
1. Proposed an exhaustive pipeline for building the dataset and made the large-scale high-quality scene images SA-Text benchmark, contributing to the advancement of research community.
2. Introduced text-spotting module to supervision on intermediate features, explicitly guides optimization and empowers the model to implicitly perceive textual information from LQ inputs.
3. The proposed method achieves outstanding performance on TAIR, surpassing the existing Real-SR methods.

### Weaknesses
Weaknesses:
1. The novelty is limited. 
    - The proposed method integrates a plain ControlNet to inject LQ priors, without any specific design tailored for the text-focused nature of the task. 
    - The detail structure of Text-spotting module like optimization objective also has not been fully described, which undermines the reproducibility. 
    - This paper is not the first to define the TAIR task, as earlier work [1] have already proposed similar tasks.
    References:
    [1]Q. Hu, L. Fan, Y. Luo, Y. Yu, X. Guo, and Q. Fan, “Text-Aware Real-World Image Super-Resolution via Diffusion Model with Joint Segmentation Decoders,” CoRR, vol. abs/2506.04641, 2025, doi: 10.48550/ARXIV.2506.04641.
2. The experiments are insufficient. 
    - TAIR focuses on both visual content restoration and textual fidelity. However, the paper lacks comprehensive evaluations on real-world image super-resolution tasks. Moreover, the metrics individually assess either scene reconstruction quality or text restoration quality, and thus fail to reflect the balance between these two aspects. 
    - While the visual results in the paper are impressive, the examples contain only a single line of text with limited style variation, which makes it difficult to demonstrate the effectiveness of the proposed method in real-world scenarios with diverse text densities and styles.
3. The layout is unsatisfactory. The gap between the text and corresponding tables/figures is too large, making readers hard to follow the content, e.g. Figure 3, Table 2.

### Questions
1. Does the proposed method only support English text? How does it perform on non-Latin characters?

2. Would the optimization objectives of the TAIR task affect the restoration of images without text, for example, causing tree branches to be mistakenly reconstructed as characters?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the critical challenge of text-image hallucination in diffusion models for image restoration, proposing a novel task called Text-Aware Image Restoration (TAIR). To advance this field, the authors constructed a large-scale benchmark with dense text annotations named SA-Text dataset. The key innovation is the TeReDiff model, where the joint training of a text-spotting module with the restoration module enables text-aware supervision during restoration, achieved new state-of-the-art results on TextZoom and demonstrated superior super-resolution results.

### Strengths
1. Excellent originality: This paper purposefully introduces a new task (TAIR), along with a corresponding dataset and a SOTA model capable of performing both tasks simultaneously.
2. Good quality and performance: The paper constructs a large-scale, high-quality dataset for TAIR and designs a novel model architecture that leverages joint training for both text-spotting and restoration tasks, achieving outstanding performance across multiple benchmarks.
3. The paper is well-written and easy to follow.

### Weaknesses
1. While the paper emphasizes that the new TAIR task differs from previous models by focusing on text-image hallucination (text readability), it fails to sufficiently demonstrate the model's superiority in this core aspect. The evidence is limited to a few comparative images and existing metrics that are not fully relevant. This lack of qualitative comparison methods specific to the TAIR task undermines the credibility of its performance evaluation.
2. The comparison in Table 4 is insufficient to substantiate the claim of superior image quality, as it includes too few diffusion-based SR models. To properly validate performance, the evaluation should be expanded to include more competing methods, particularly established models like StableSR and ResShift that are already discussed in Table 2.
3. The provided figures in Table 2 and Table 3 contain multiple inaccuracies in labeling "Best results" and "second-best"， which is confusing.

### Questions
1. Can the authors provide qualitative, direct metrics or a feasible evaluation scheme? The core objective of the newly proposed task is text super-resolution guided by solving the text-image hallucination problem. However, the paper also states that no existing metrics can accurately measure the effectiveness in addressing this issue. The proposal of a task with no clear way to determine if it has been completed lacks meaningfulness.
2. Firstly, the comparison involves too few methods, which undermines the credibility of the claimed superiority in image quality. Secondly, it is unclear why most performance metrics for the "further trained DiffBIR" model are significantly lower than those of the original DiffBIR. Moreover, the fact that this model was not included in the comparisons of Tables 2/3。 This counterintuitive result requires clarification.

### Soundness
2

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
This paper presents Text-Aware Image Restoration (TAIR), designed to mitigate diffusion models’ tendency to hallucinate text in degraded images. The authors introduce SA-Text, a dataset of 100K images with detailed text annotations, and TeReDiff, a diffusion-based framework that integrates text spotting for joint training and text-conditioned denoising. Experimental results demonstrate improved text fidelity compared with existing approaches on the SA-Text and TextZoom benchmarks. However, the method’s dependence on synthetic degradations, limited validation on real-world data, and insufficient analysis of computational overhead reduce its practical significance.

### Strengths
1. The author clearly identifies a critical limitation in diffusion-based image restoration—its inability to accurately recover text regions. To address this, the author introduces Text-Aware Image Restoration (TAIR) as a novel task that jointly optimizes visual quality and text fidelity, offering strong potential for practical applications.

2. The author constructs a dataset of 100K high-resolution images derived from SA-1B, densely annotated with text polygons and transcriptions through a scalable VLM-based annotation pipeline.

### Weaknesses
1. The author primarily conducts training and evaluation on synthetic degradations (e.g., Real-ESRGAN). Moreover, while the Real-Text results (Table 3) demonstrate modest improvements (e.g., a +6% F1-score over FaithDiff), the absence of corresponding visual examples limits the clarity of qualitative gains.

2. The author provides no analysis of error propagation across the pipeline. Given that the method relies on accurate text spotting, recognition errors are likely to propagate and compromise restoration performance, but this crucial aspect remains unexplored.

3. Jointly training the text-spotting module with the diffusion backbone is conceptually straightforward. However, the ablation studies (Table 6) indicate that most of the observed improvements stem from the inclusion of the SA-Text dataset and the use of multi-stage training, rather than from the joint optimization itself.

4. The evaluation depends exclusively on text-spotting metrics (e.g., F1-score), which fail to reflect the semantic accuracy of restored text. Such metrics emphasize detection precision and recall but overlook whether the recovered text is contextually or linguistically correct, thereby limiting the reliability of the reported improvements.

5. Although SA-Text is described as a “language-agnostic” annotation pipeline (Section 3), the experiments and validations are conducted exclusively on English text. The absence of cross-lingual evaluations leaves the claimed language generality unverified.

### Questions
1. When the text-spotting module fails at early denoising steps (e.g., misreads "HELLO" as "H3LLO"), how does TeReDiff recover?

2. Ablations suggest SA-Text drives performance. If trained on TextOCR, would TeReDiff still outperform DiffBIR?

3. Why not test on non-English text? Can the VLM filtering in SA-Text handle languages with complex scripts?

### Soundness
2

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
5

### Summary
This paper proposes TeReDiff, a text-aware image restoration framework that effectively recovers readable text in severely degraded full-scene images while preserving overall visual quality. Unlike prior scene text image super-resolution (STISR) methods that focus on small, single-word crops, TeReDiff addresses the more challenging and general Text-Aware Image Restoration (TAIR) task. By incorporating multi-stage training and text-conditioned diffusion, the model mitigates text hallucination and achieves good performance on both text detection/recognition metrics and general image quality benchmarks.

### Strengths
- Introduces TAIR—a more realistic and general setting than existing STISR—by restoring full-scene images with multiple, diverse text instances.
- Significantly reduces text hallucination and improves readability through text-conditioned diffusion and multi-stage training.
- Achieves good performance on both text recognition metrics and standard image restoration benchmarks, demonstrating balanced fidelity and perceptual quality.

### Weaknesses
- The evaluation relies heavily on synthetic or curated datasets (e.g., SA-Text), which may not fully reflect the complexity and variability of real-world degraded images.
- Lacks in-depth discussion of scenarios where the method fails (e.g., extremely low-resolution or occluded text), reducing insight into limitations.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
