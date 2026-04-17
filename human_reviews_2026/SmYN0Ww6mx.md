# Multi-modality Image Fusion under Adverse Weather: Mask-Guided Feature Restoration and Interaction

- Decision: Reject
- Scores: 4, 6, 6

## Abstract
Multi-modality image fusion (MMIF) enhances scene representation by exploiting complementary cues from different modalities. Adverse weather, however, causes significant image degradation, disrupting feature representation and requiring simultaneous feature restoration and cross-modal complementarity. Existing methods often struggle with effective representation learning under such conditions, limiting their practical performance. To address these challenges, we propose a mask-guided MMIF method that integrates feature restoration and interaction. We first introduce "Pseudo Ground Truth" to simplify training, promoting faster and more effective feature learning. Then, we design a mask generation mechanism based on the mapping relationship between the fused result and the source images, quantifying the relative contribution of each modality during the fusion process. By incorporating the proposed mask-guided cross-modal cross-attention mechanism, the network is encouraged to selectively attend to informative features during modality interaction, mitigating the risk of overfitting to the static distribution of the "Pseudo Ground Truth". Additionally, we propose a mask-guided and a task-coupled degradation-aware strategy to balance feature restoration and interaction. Extensive experiments on synthetic and real-world datasets (rain, haze, and snow) demonstrate that our method surpasses state-of-the-art approaches in visual quality, quantitative metrics, and downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel multi-modality image fusion method designed to address image fusion challenges under adverse weather conditions. The proposed method integrates feature restoration with cross-modal interaction through a mask-guided mechanism, utilizing "Pseudo Ground Truth" to simplify the training process. This helps in recovering degraded features and enhancing the interaction between multi-modal features. Key innovations include the construction of a mask that quantifies each modality's contribution during the fusion process, enabling effective separation and interaction of features. Additionally, the method introduces a Mask-Guided Feature Extraction Module (MFEM), Mask-Guided Learning Strategy (MGLS), and Task-Coupled Degradation-Aware Learning Strategy (TDAS) to improve feature restoration and cross-modal interaction. Extensive experiments on both synthetic and real-world datasets under adverse weather conditions demonstrate that AMG-Fuse outperforms state-of-the-art methods in visual quality, quantitative metrics, and downstream tasks, while also showing strong adaptability in ideal conditions.

### Strengths
1.The AMG-Fuse integrates feature restoration and cross-modal interaction into a unified framework to tackle the challenges of image fusion under adverse weather conditions. The mask-guided mechanism combined with "Pseudo Ground Truth" allows the network to learn dynamic cross-modal information allocation, overcoming the limitations of traditional methods relying on static modality distribution.

2.The proposed method is thoroughly tested on both synthetic and real-world datasets under various weather conditions. The results demonstrate AMG-Fuse’s superiority in visual quality, quantitative metrics, and downstream tasks, proving its robustness and applicability across different environments.

3.While AMG-Fuse excels in handling adverse weather, it also performs well in ideal environments, showcasing its generalization ability and versatility in a variety of conditions.

### Weaknesses
1.The inclusion of the Histogram Transformer module and depthwise separable convolutions leads to high computational demands and a large number of parameters, which may limit its applicability in real-time scenarios with resource constraints.

2.The overall architecture of this article is similar to a simple stacking of modules, which also includes a frozen restoration model in the middle. The understanding of the essential relationship and differences between restoration and fusion tasks is not yet thorough enough.

3.While the method performs well in adverse weather conditions, it struggles with noise and other degradation types, as the mask fails to accurately capture the degradation distribution in such cases, leading to suboptimal performance.

4.The degradation targeted in this article mainly includes rain, snow, and haze, and the rain and snow part mainly uses artificial virtual data, which is not realistic enough in terms of visual effects. Therefore, it is difficult to reflect the value in practical applications. The method may require further optimization for handling other types of degradation, such as motion blur or strong lighting variations.

5.The method involves multiple strategies that require careful tuning of hyperparameters, which increases the complexity of the training process and experimental design.

### Questions
1.As shown in the Haze section of Fig.6, the restoration results of AdaIR are obviously not excellent enough. Can the author choose other restoration models for comparative experiments and prove that the proposed method is indeed more effective than the method of two models.

2.Some of the results in the ablation experiment in this article do not reflect the significance of this module, especially in Mask Guided Learning Strategy, where the contours of the characters are very blurry. The author needs to submit other figures to prove it.

3.In the author's comparison method, only Text-if and AWFusion are integrated restoration fusion methods. There have been many works in this area in the past year, and the author should provide more reference methods.

4.The author needs to provide performance on real-world datasets to validate the practical value of the method.

5.The author did not provide visual results for object detection, and relevant content needs to be supplemented.

6.Is it possible to use model compression or quantization techniques to reduce the computational overhead during inference and improve the feasibility of this method in real-time applications for high computational complexity problems?

7.If the degree of degradation changes, will the model's processing capability weaken and will the area of focus of Mask change?

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
5

### Summary
Authors proposed a mask-guided multi-modal image fusion method designed to handle degradations under adverse weather. The main contributions are the proposed mask-guided feature extraction module and task-coupled degradation-aware learning strategy, which alleviate the optimization difficulty of jointly learning cross-modal interactions and feature restoration in complex scenes. Experiments show that the proposed method consistently outperforms the state-of-the-art methods on both image fusion and downstream tasks.

### Strengths
(1) By explicitly separating modality-specific components within the fused image, the network dynamically learns how to allocate multimodal information. This differs from existing mask-based fusion methods and is both novel and effective.

(2) The proposed mask-guided feature extraction module leverages masks to recover degraded features while enhancing cross-modal interactions, offering a valuable new perspective for image fusion task in challenging scenarios. The proposed task-coupled degradation-aware learning strategy exploits restoration priors to encourage the fusion network to learn cleaner representations, which is a well-motivated supervision scheme.

### Weaknesses
**Major**

(1) In Fig. 2, constraining the network directly with clean multimodal source images seems to improve generalization. However, intuitively, models trained on harder examples often generalize better to easier ones. Authors should clarify why the opposite appears beneficial here, and provide theoretical insights or empirical evidence.

(2) The mask is constructed based on Eq. (3). Please discuss whether this formulation captures the composition of more general fusion tasks, and detail its assumptions and potential failure modes.

(3)The evaluation on downstream tasks is not sufficiently comprehensive. Currently, only detection on M3FD is reported. Please add semantic segmentation comparisons on the MSRS dataset.

**Minor**

(1) Please standardize notation (case, subscripts/superscripts) throughout, including symbols in equations. Also, keep the term “Pseudo Ground Truth” consistent.

(2) The qualitative results in Fig. 6 are relatively small. please add higher resolution visual comparisons for adverse-weather cases in the appendix.

### Questions
(1) How is the Pseudo Ground Truth generator trained? Does it use off-the-shelf weights, or is it re-trained/fine-tuned on AWMM-100k? Please provide more training details and data splits.

(2) Can the proposed method generalize to other image fusion tasks, such as CT–MRI fusion in medical imaging?

### Soundness
3

### Presentation
4

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
The paper proposes a mask-guided multi-modality image fusion (MMIF) framework that jointly performs feature restoration and cross-modal interaction under adverse weather. 
It introduces pseudo ground-truth to simplify training and a mask generation mechanism to measure each modality’s contribution during fusion. 
A mask-guided cross-modal attention enables the network to focus on informative features, while degradation-aware learning strategies balance restoration and interaction. 
Experiments on rain, haze, and snow datasets show that the proposed method achieves superior visual quality, quantitative performance, and downstream task results compared to state-of-the-art methods.

### Strengths
1. This work introduces a mask-guided framework (AMG-Fuse) that unifies feature restoration and modality interaction, addressing the limitations of two-stage “restoration + fusion” pipelines.

2. The Mask-Guided Feature Extraction Module (MFEM), Mask-Guided Learning Strategy (MGLS), and Task-Coupled Degradation-Aware Strategy (TDAS) are well-motivated and technically sound contributions that enhance cross-modal complementarity.

3. Extensive experiments on multiple adverse weather datasets (rain, haze, snow) and clean datasets demonstrate consistent improvements across quantitative and qualitative metrics.

4. The proposed method outperforms state-of-the-art methods in both degraded and ideal conditions, showing robustness and generalization ability.

5. The paper is well-organized, with detailed ablation studies, visualization analyses, and fair comparisons that support the claims effectively.

### Weaknesses
1. Although conceptually justified, the reliance on pseudo targets may still raise questions about the upper-bound constraint on model generalization and possible supervision bias not fully explored in the experiments. Though softened by decaying loss weights, the reliance on pseudo targets may introduce bias and constrain the model’s learning dynamics. Would this be addressed by theoretical or empirical analysis?

2. Could the authors provide a more extensive comparisons on FLOPs or runtime for efficiency analysis?

3. The method introduces Histogram Transformer and multi-head attention structures, yet the discussion of computational trade-offs versus the versions of lightweight fusion models remains underdeveloped.

4. The manuscript has neglected several related works, such as pseudo ground-truth in detection under adverse weather [1], image fusion under hazy condition [2], etc. More extensive literature review is recommended.

[1] Rethinking image restoration for object detection. 2022.
[2] Image dehazing by artificial multiple-exposure image fusion, 2018.

5. Some mathematical derivations (e.g., Equation 5 reformulation) and visualizations could be better contextualized for readers unfamiliar with modality-decoupled learning. The citation format of the paper should be revised.

The concerns are not vital and could be addressed before potentially accepted.

### Questions
Would the authors consider providing theoretical or empirical analysis to clarify whether the reliance on pseudo targets introduces supervision bias or constrains the model’s generalization capacity and learning dynamics?

### Soundness
3

### Presentation
3

### Contribution
3
