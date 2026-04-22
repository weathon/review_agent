# From Frames to Sequences: Temporally Consistent Human-Centric Dense Prediction

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 6, 6

## Abstract
In this work, we focus on the challenge of temporally consistent human-centric dense prediction across video sequences. While progress has been made in per-frame predictions of depth, surface normals, and segmentation, achieving stability under motion, occlusion, and illumination changes remains difficult. For this, we design a synthetic data pipeline that produces large-scale photorealistic human images and motion-aligned video sequences with high-fidelity annotations. Unlike prior static data synthetic pipelines, our pipeline provides both frame-level and sequence-level supervision, supporting the learning of spatial accuracy and temporal stability. Building on this, we introduce a model that integrates human-centric priors and temporal modules to jointly estimate temporally consistent segmentation, depth, and surface normals within a single framework. Our two-stage training strategy, combining static pretraining with dynamic sequence supervision, enables the model to first acquire robust spatial representations and then refine temporal consistency across motion-aligned sequences. Extensive experiments show that we achieve state-of-the-art performance on THuman2.1 and Hi4D and generalize effectively to in-the-wild videos.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper targets temporally consistent human-centric dense prediction (depth, normals, segmentation) across video.
Key obstacles are the lack of large-scale human video with paired dense labels, and coupling temporal stability with multi-task learning.
A synthetic data pipeline generates photorealistic human images and motion-aligned sequences with high-fidelity labels, enabling frame- and sequence-level supervision.
A ViT-based model integrates human-centric priors and temporal modules to jointly predict depth, normals, and segmentation.
Experiments report state-of-the-art results on THuman2.1 and Hi4D for both depth and surface normal estimation.

### Strengths
- Build a scalable data synthesis pipeline for human-centric frames and videos with pixel-accurate depth, normals, and segmentation. 
- Going beyond static-image training with video supervision improves temporal stability and generalization in natural scenes.

### Weaknesses
1. Points Requiring Clarification

(1) Channel Weight Adaptation (CWA): The manuscript does not explain how the module distinguishes channels dominated by texture and illumination from those that are geometry-related, nor how the reweighting is computed to downweight the former and upweight the latter (thereby weakening the influence of appearance on geometry prediction and maintaining the consistency of the global representation).

(2) Human Geometric Prior fusion: There is no description of how the human geometric prior is fused with the decoder features.

(3) 𝐿_{grad}  in Eq. (1): The definition of 𝐿_{grad} is missing.

2. Evaluation Completeness

(1) Although segmentation is one of the tasks, quantitative evaluation for segmentation is absent.

(2) The ablation study omits the following:\
- (a) a DPT head without the additional CNN branch,\
- (b) results for the full model configuration, and\
- (c) temporal-layer ablations.

(3) In Table 1, it would be more appropriate to compare against a video-specific depth estimation model (e.g., DepthCrafter) rather than DepthAnything.

3. Minor Suggestion

Line 348 reads awkwardly (“Let 𝐸𝑘 be a dilated edge map extracted from the current predicted depth M_{edge} = 1 − Dilate(Ek).") and likely nees revision.

### Questions
1. Regarding Weakness 1: Points Requiring Clarification, would it be possible to provide additional details?

2. Regarding Weakness 2: Evaluation Completeness, would it be possible to share the missing results?

### Soundness
2

### Presentation
1

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
The paper targets temporally consistent, human-centric dense prediction (segmentation, depth, surface normals) in videos by pairing a large synthetic data pipeline with a ViT-based model that injects human geometric priors. It first proposes a large-scale human-centric synthetic dataset (both images and videos). Then authors propose VIT-based architecture, built from DAViD, with a temporal DPT, human geometric prior, and weight adaptation for local geometry. Experiments are conducted on THuman2.1 and Hi4D datasets, outperforming previous approaches on depth and normal estimation tasks.

### Strengths
* Synthetic data pipeline: Builds a large human-centric data synthesis pipeline to generate diverse human-centric image and video data.
* Simple and straightforward design: Uses a ViT backbone for feature extraction with a temporal head (DPT-style) to enforce temporal consistency, while leveraging human priors and local geometry cues.
* Strong empirical results: competitive or superior performance across multiple benchmarks compared to prior methods (Sapiens, DAViD)

### Weaknesses
* **Insufficient overview of the synthetic data pipeline**. Since the dataset is a key contribution, the paper should include a clear figure of the data generation process and provide additional sample visualizations (e.g., in the supplement) to make the pipeline understandable and auditable.
* **Missing training-data ablations**. The model is trained on a mixture of SynthHuman and the proposed dataset, but there is no study isolating data effects (e.g., only SynthHuman vs. proposed vs. mixture). Such ablations are needed to quantify the data-driven gains.
* **Limited technical novelty**. Most components are adapted from prior work—ViT + local geometry from DAViD and the temporal DPT head from VideoDepthAnything—making the pipeline feel incremental rather than introducing a new technical idea.
* The paper mentions a foreground/background segmentation component at line 192, but provides no quantitative metrics or qualitative visuals. Please include results to substantiate the claim.

### Questions
Will the dataset be open-sourced?

**Conclusion**:
Overall, this work contributes a large-scale human-centric dataset alongside a simple, effective model for temporally consistent dense geometry. Given that the dataset is the paper’s most impactful contribution, I strongly encourage the authors to open-source the data (and generation pipeline) to maximize community benefit and reproducibility.

### Soundness
3

### Presentation
3

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
This paper presents a method that extends the DAViD framework from static frames to video sequences, introducing two key components, CSE and CWA, to enhance human-centric 4D reconstruction. The approach achieves significant performance improvements, often comparable to large-scale models, while maintaining reasonable efficiency.

### Strengths
1. The proposed method demonstrates substantial quantitative gains over existing approaches across multiple benchmarks. The improvements are consistent and, in some cases, even comparable to large-scale models such as Sapiens, which highlights the effectiveness and efficiency of the proposed design.
2. The paper successfully builds upon the previous observation from DAViD — that “a single high-fidelity dataset is sufficient to tackle multiple dense prediction tasks and achieve state-of-the-art accuracy.” Extending this principle from static frames to dynamic, temporal sequences is both meaningful and well-motivated, providing a clear conceptual bridge between frame-level and sequence-level human reconstruction tasks.
3. The overall architecture and proposed components (CSE and CWA) are logically structured and supported by clear intuitions. The approach balances model complexity and performance, offering a practical solution that integrates geometric priors and adaptive weighting in a coherent way.

### Weaknesses
1. The authors highlight a scalable data synthesis pipeline for human-centric frames and videos as one of their main contributions. However, although the pipeline is described in Section 3.1, it is not clear how novel or distinctive this approach is compared to existing dataset synthesis methods. The proposed pipeline appears fairly conventional and lacks clear justification or comparison to prior data generation frameworks.
2. Table 4 only includes ablation results for models using either CSE or CWA individually. Including a CSE + CWA combination would provide a clearer picture of their complementary effects and help isolate the contributions of each module more effectively.
3. The ablation studies are conducted only on the Hi4D dataset, where the performance gain is quite substantial. Additional results on the THuman2.1 dataset would verify whether CSE and CWA generalize across different data domains rather than being overfitted to Hi4D.
4. The paper does not include inference time or computational cost comparisons between the proposed method and the baselines listed in Table 2. Reporting runtime performance, memory consumption, or FLOPs would strengthen the evaluation by demonstrating the method’s practical applicability.
5. Although the proposed method is overall reasonable and well-structured, its main technical contributions—the channel reweighting module (CWA) and the human geometry prior (CSE)—are relatively standard. These components resemble existing techniques in feature reweighting and prior-based guidance, which limits the methodological novelty of the paper.

### Questions
1. Could the authors explain why the proposed method shows such a significant improvement on the Hi4D dataset—achieving results even comparable to the large-scale Sapiens model in Table 2?
2. The paper mentions that “the parameter size of Sapiens-0.3B is equivalent to that of large models of ViT-based methods.” Should we then assume that DaViD, Sapiens-0.3B, and Ours-L are approximately comparable in model size? A clear statement or table comparing parameter counts would be helpful.

### Soundness
3

### Presentation
3

### Contribution
2
