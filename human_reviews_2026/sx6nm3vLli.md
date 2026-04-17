# AutoAWG: Adverse Weather Generation with Adaptive Multi-Controls for Automotive Videos

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Perception robustness under adverse weather remains a critical challenge for autonomous driving, with the core bottleneck being the scarcity of real-world video data in adverse weather. Existing weather generation approaches struggle to balance visual quality and annotation reusability. We present \emph{AutoAWG}, a controllable \emph{A}dverse \emph{W}eather video \emph{G}eneration framework for \emph{Auto}nomous driving. Our method employs a semantics-guided adaptive fusion of multiple controls to balance strong weather stylization with high-fidelity preservation of safety-critical targets; leverages a vanishing-point--anchored temporal synthesis strategy to construct training sequences from static images, thereby reducing reliance on synthetic data; and adopts masked training to enhance long-horizon generation stability. On the nuScenes validation set, \emph{AutoAWG} significantly outperforms prior state-of-the-art methods: without first-frame conditioning, FID and FVD are relatively reduced by 50.0\% and 16.1\%; with first-frame conditioning, they are further reduced by 8.7\% and 7.2\%, respectively. Extensive qualitative and quantitative results demonstrate advantages in style fidelity, temporal consistency, and semantic--structural integrity, underscoring the practical value of \emph{AutoAWG} for improving downstream perception in autonomous driving.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents AutoAWG, a video diffusion model for adverse weather generation. To capture fine details, it introduces an Adaptive Fusion module that combines depth, lineart, and sketch maps as input conditions. The video diffusion model acts as a painter, filling in color information into the condition maps based on weather text descriptions. To address data scarcity, the authors propose the VP (Vanishing Point)-Anchored Temporal Synthesis strategy to generate video data from images.

### Strengths
1. The VP-Anchored Temporal Synthesis strategy offers an interesting approach to addressing data scarcity.
2. The video results demonstrate good visual quality.

### Weaknesses
1. The literature review is insufficient: Important recent works are not mentioned or compared: [DriveScape](https://arxiv.org/abs/2409.05463) and [MaskGWM](https://arxiv.org/abs/2502.11663). Although their condition settings differ, they should be included in the quantitative comparison table since they have been published very early and demonstrate better numerical performance than some methods mentioned in the paper.
2. Novelty Issue: While [UniMLVG](https://arxiv.org/abs/2412.04842) does not explicitly describe adverse weather generation as a task, Figure 4 demonstrates impressive weather change examples using text conditions. Notably, it can handle some failure cases shown in AutoAWG's Appendix without explicit weather-specific training. UniMLVG supports more diverse text control and naturally extends weather control to multi-view diffusion by incorporating larger datasets like OpenDV-YouTube, even though it is a single-view dataset. Given this comparison, the paper's contribution is insufficient for publication.

### Questions
1. Are the multiple conditions (depth, lineart, and sketch) required for every frame during the diffusion generation process?
2. The ablation study of the weighted loss does not include the case when $\alpha=0$. What are the results for this setting?

### Soundness
2

### Presentation
2

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
The paper proposes AutoAWG, a controllable framework for generating adverse weather video sequences for autonomous driving, addressing the challenge of data scarcity in adverse weather conditions. AutoAWG combines semantics-guided fusion of multiple control conditions to maintain high fidelity in safety-critical objects while applying strong weather stylization. It uses a vanishing point-anchored temporal synthesis strategy to generate pseudo-video sequences from static images, reducing reliance on synthetic data and improving temporal consistency. AutoAWG outperforms prior methods in terms of style fidelity, temporal consistency, and semantic structure preservation, showing promising improvements in downstream perception tasks for autonomous driving.

### Strengths
1. The paper introduces a method for generating video sequences under adverse weather conditions while maintaining both realistic weather effects and structural integrity of safety-critical objects.

2. AutoAWG ensures that the semantic and geometric properties of safety-critical objects are preserved across weather conditions, ensuring that the generated videos remain useful for perception tasks without re-annotation.

3. By generating realistic videos from static images and leveraging a vanishing point-anchored synthesis, AutoAWG mitigates the challenges posed by the lack of real-world adverse weather videos, offering an efficient way to augment training datasets.

### Weaknesses
1. Since the model does not utilize cross-attention between multiple views but rather relies on control conditions, it remains unclear how the model guarantees visual and foreground motion consistency across different camera viewpoints, which is crucial for multi-camera systems in autonomous driving.

2. The current implementation only simulates four weather types (rain, fog, snow, and night), which feels insufficient given the wide variety of possible weather conditions that could impact autonomous driving. This limits the generalization of the model’s practical applicability.

3. The use of a diffusion model for video weather style transfer may seem unnecessary, as similar video transfer tasks have already been tackled in the style transfer domain with traditional methods (e.g., using filters in video games like GTA5 to simulate weather effects), questioning the novelty of this approach.

4. The experimental comparison in Table 1 between AutoAWG and other models (which use different architectures like DiT vs. SD 1.5 or 2.0) lacks clarity on whether the performance improvement comes from the proposed algorithm or the underlying model's architecture and pretraining. The difference in model architectures makes the comparison somewhat unfair.

5. Missing Reference: "SimGen" (NeurIPS 2024).

### Questions
1. How are the fused control conditions evaluated for their effectiveness？Does it provide strong evidence that they accurately reflect deep semantic information? More rigorous evaluation metrics or comparisons with other fusion strategies are needed.

### Soundness
2

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
The paper proposes the AutoAWG framework, which aims to address the scarcity of real video data under adverse weather conditions in autonomous driving, as well as the difficulty of balancing visual quality and annotation reusability in existing weather generation methods. Its core designs include: a semantic-guided multi-control adaptive fusion strategy to balance weather stylization with the fidelity of safety-critical targets; a vanishing-point-anchored temporal synthesis strategy to construct training sequences from static images and a mask-based training scheme to enhance stability in long-sequence generation. The framework achieves excellent performance on benchmark datasets and demonstrates strong results in downstream tasks.

### Strengths
The proposed method in this paper is innovative. The authors conduct an in-depth analysis of the current challenges faced by real data synthesis under adverse weather conditions and introduce the AutoAWG framework. First, a semantic-guided multi-control adaptive fusion strategy is proposed, which innovatively introduces masks and applies different masking schemes for different control signals, achieving an optimal balance between weather stylization and target fidelity. To reduce dependence on synthetic data, a vanishing-point-anchored video synthesis strategy is proposed, improving data utilization. In addition, mask-based training is employed to enhance the stability of long-sequence generation. Extensive experiments are conducted to verify the effectiveness of the proposed method.

### Weaknesses
Although the paper proposes multiple strategies and conducts experimental validation, the ablation study section is insufficient and does not fully demonstrate the effectiveness of the proposed strategies. It is recommended to provide additional ablation experiments.

### Questions
- First, the paper proposes the ADAPTIVE FUSION OF MULTIPLE CONTROLS strategy. In what aspects does the “adaptive” property manifest? Does it mean that each control changes according to the mask? During the fusion of multiple controls after masking, has an adaptive strategy been considered? Does the quality of the mask affect the overall performance of the network?
- In the ablation study section, does the difference between the last two rows in Table 4 refer to the presence or absence of weight loss? If so, why does this loss lead to a decrease in the weather stylization metrics? In this loss function, is it necessary to differentiate between regions such as object and sky, similar to the multi-control fusion strategy? If not, is there any experimental evidence demonstrating the effectiveness of the weight loss?
- Has the paper conducted any tests regarding the inference efficiency of the model — for example, metrics such as GPU memory usage during inference or inference time?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a controllable diffusion-based framework for generating realistic automotive videos under adverse weather conditions. By analyzing the characteristics and effects of different control conditions, a simple yet effective multi-condition fusion method is proposed to achieve a balance between weather transformation and cirtical element preservation. It is further enhanced through dataset augmentation, loss function design, and training strategy adjustment. Evaluation results demonstrate the superior performance and effectiveness of the proposed method. And the improvement on downstream 3D object detection task validates the practical value of generated videos.

### Strengths
S1. This paper employs semantic-guided adaptive fusion of multiple control signals to balance stylization and fidelity, achieving high-quality video generation.

S2. The method proposed in this paper ensures robust controllability and scalability, supporting multi-camera and long-sequence scenarios.

S3. The proposed method in this paper can be applied to improve downstream task performance in autonomous driving, highlighting its practical value.

S4. This paper is easy to read and follow.

### Weaknesses
W1. The semantic masking design is overly simplified, relying mainly on a binary “object–sky” separation to guide control fusion. This neglects finer-grained semantic hierarchies (e.g., roads, buildings, pedestrians, vegetation), which may limit the model’s ability to maintain localized semantic consistency and structural harmony in complex scenes.

W2. The model overlooks the underlying physical dynamics of adverse weather (e.g., raindrop motion distribution, wind field perturbations, snow accumulation changes), relying mainly on visual texture matching rather than physically grounded simulation. This omission limits its ability to generate weather videos with realistic temporal and physical evolution characteristics.

### Questions
Q1. Evaluation on the temporal consistency. While visual examples and long-sequence results are shown, no quantitative metrics (e.g., temporal LPIPS, tOF/tFID, or temporal smoothness measures) are reported to assess inter-frame coherence and stability, making it difficult to gauge the model’s robustness over extended video sequences.

Q2. About dataset. Existing multi-weather driving video datasets such as BDD100K, which contain real-world sequences across diverse weather and lighting conditions, are not considered. The lack of comparison or validation on more datasets leaves the model’s generalization to real-world weather diversity uncertain.

Q3. About utility on downstream tasks. How much improvement can the videos generated by other methods bring to downstream tasks? Can the enhancement in video quality achieved by this method be reflected in corresponding gains on downstream tasks?

Q4. About ablation. What qualitative or quantitative improvements are brought respectively by the design of the constructed training data, loss functions, and training strategies?

### Soundness
2

### Presentation
3

### Contribution
2
