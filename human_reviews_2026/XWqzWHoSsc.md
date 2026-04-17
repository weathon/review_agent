# UniAG: Unified Anomaly Generation via Local Spatial-Texture Alignment Diffusion Model

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Few-shot Anomaly Generation (FSAG) aims to enhance anomaly detection by generating realistic and diverse anomalies from a limited set of anomalous examples, addressing the challenge of scarce anomalous data in real-world scenarios. However, existing FSAG methods require training separate models for different anomaly types, leading to low training and deployment efficiency. Most importantly, the lack of sufficient realism and diversity limits the performance of anomaly detectors trained on them. To overcome these limitations, we propose UniAG, a unified model capable of generating realistic and diverse anomalies across multiple categories, thereby improving both generation efficiency and anomaly detection performance. Specifically, we propose a deep copy–paste anomaly generation strategy in which a Spatial-Texture Alignment Diffusion model (STA-DM) learns to fill local region masks with anomaly textures corresponding to user-specified categories. We further propose a novel generation condition with explicit spatial–category guidance instead of text embeddings for diffusion models, enabling realistic and diverse generation.  Experimental results show that UniAG outperforms existing methods both in anomaly generation quality and in downstream anomaly detection performance. Notably, we achieve a new state-of-the-art anomaly localization AUROC/AP performance $\mathbf{99.2/81.0}$ with only $\mathbf{4}$ anomaly examples and $\mathbf{500}$ generated samples for each anomaly on the comprehensive MVTec AD dataset.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes UniAG, a unified diffusion-based model for few-shot anomaly generation. The authors design a Spatial–Texture Alignment Diffusion Model (STA-DM) that introduces a spatial-category embedding and an anomaly injection adapter to unify the generation of multi-class anomalies within a single model. The authors claim that this approach improves training efficiency, generation quality, and downstream anomaly detection performance compared to existing category-specific models.

### Strengths
1. The writing is clear, and the figures effectively illustrate the performance.
2. The idea of decoupling texture generation from background modeling via a localized strategy is intuitively appealing.

### Weaknesses
1.	The motivation of this paper is problematic. Although the scarcity of anomaly samples and the diversity of anomaly patterns pose a challenge to anomaly detection as the authors declared in the introduction, many unsupervised anomaly detection methods, especially reconstruction-based methods such as [a] and [b], achieve higher scores using the same dataset (MVTec AD) compared to the results reported in this paper. These methods only use normal samples to train the model and do not need any synthetic data to augment the existing training data. In this case, it is difficult to see any meaningful value to generate such anomaly images, especially for industrial applications. 

[a] One Dinomaly2 Detect Them All: A Unified Framework for Full-Spectrum Unsupervised Anomaly Detection (an extension of CVPR2025)

[b] INP-Former++: Advancing Universal Anomaly Detection via Intrinsic Normal Prototypes and Residual Learning (an extension of CVPR2025)

2.	The proposed UniAG framework largely integrates existing components from prior diffusion-based anomaly generation and inpainting literature. The ‘deep copy–paste’ strategy resembles prior local inpainting and patch-based augmentation methods. The ‘spatial–category embedding’ and ‘adapter’ modules are minor variations of established conditional diffusion techniques (e.g., ControlNet, structure-guided diffusion). The paper does not introduce a new learning principle or a fundamentally novel diffusion mechanism. Therefore, the contribution is incremental and mainly engineering-oriented. In my opinion, the paper primarily focuses on empirical gains on one benchmark without offering new insights into anomaly generation or diffusion theory. The work reads more like an application paper suited for an applied vision venue rather than a top-tier learning conference like ICLR.
3.	The paper lacks formal analysis or theoretical justification of why STA-DM improves over text-conditioned diffusion. The model description essentially reproduces standard conditional diffusion objectives with added embeddings.
4.	The comparison methods are limited and many current SOTA methods are not compared especially published in 2025.
5.	In the main text, the experiments are almost entirely limited to MVTec AD, a single industrial dataset. This is not convincing to support the main contribution of this paper because it is a very easy dataset for anomaly detection. Furthermore, only 4-shot is analyzed in this paper, lacking the validation of generalization of other numbers of few-shot.
6.	The architectural contributions are not clearly distinguished from existing conditional diffusion models such as ControlNet [c].

[c] Adding conditional control to text-to-image diffusion models. (CVPR2023)

### Questions
Except for the major concerns presented above in the Weaknesses, I have another two additional questions:

1.	In Equation (4) and Section 3.1, the latent variable z_a^t and the binary mask \hat{m} lie in different spaces (latent vs. pixel), with different spatial resolutions and channel dimensions. How is the mask’s spatial resolution and channel depth adapted to enable the described element-wise multiplication with the latent tensor?

2.	Regarding Figure 4, what is the exact category identifier used during training and inference for the bottle example? If "hazelnut cut" is used for a bottle, how does the model prevent generating semantically incorrect textures, and what is the rationale behind this design?

### Soundness
2

### Presentation
2

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
The paper proposes UniAG, a novel framework for Few-Shot Anomaly Generation (FSAG). Its core innovation is a unified model that can generate realistic and diverse anomalies for multiple categories, addressing a key inefficiency in prior work that required training a separate model for each anomaly type. The method combines a "deep copy-paste" strategy with a custom Spatial-Texture Alignment Diffusion Model (STA-DM) that uses explicit spatial-category embeddings instead of ambiguous text prompts. The results on the MVTec AD benchmark are impressive, demonstrating state-of-the-art performance in both generation quality and downstream anomaly detection tasks.

### Strengths
1. Motivation: The paper clearly identifies the limitations of existing FSAG methods: The need to train one model per anomaly type is computationally expensive and impractical for deployment.
2. Well-Motivated Methodology: Replacing text embeddings with a learned spatial-category embedding (e_a) is a key contribution. It provides more precise control over the generation process, leading to higher fidelity.
3. Experiments: Studied datasets are thourough. The ablation studies are thourough to validate the contribution of each component of the method. The scalability analysis (Figure 6) effectively demonstrates the practical advantages of UniAG as the number of categories and the scale of generated data increase.
4. Clarity and Reproducibility: The paper is well-structured and clearly written. The appendices provide substantial additional detail on datasets, implementation, mask augmentation, and fusion strategies, which is commendable for reproducibility.

### Weaknesses
1. Simplicity of "Direct Pasting": The paper finds that the simplest blending strategy, "direct pasting," performs best for downstream detection tasks. While this is efficient, it can create sharp, unrealistic edges between the anomalous patch and the normal background. The authors state this has "little effect", but this could be a drawback for anomaly types where the boundary itself is a key feature (e.g., subtle gradients or stains)
2. Clarification on contribution: The claim in the introduction to be the "first unified model to support multi-class anomaly generation"  requires more precise qualification. This claim is very strong, and other recent works have also proposed unified "one-for-all" generation models, such as [1] (unsupervised synthesis method) and [2][3] (zero-shot method using pre-trained diffusion models). While UniAG's contribution is unique, it should be clearly and explicitly limited to the Few-shot Anomaly Generation (FSAG) setting, where a single model is trained from a small set of real anomaly examples. This distinction from unsupervised or zero-shot paradigms should be clarified.
3. More visual results: While the paper provides extensive quantitative results on the composite dataset (VisA, BrainMRI, HeadCT) in Appendix C , it lacks corresponding visual examples. To fully substantiate the claims of generalizability, adding a few qualitative examples of generated anomalies from the VisA and medical datasets would strengthen the results.

[1] Zhang, Ximiao, Min Xu, and Xiuzhuang Zhou. "Realnet: A feature selection network with realistic synthetic anomaly for anomaly detection." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2024.
[2] Sun, Han, et al. "Unseen Visual Anomaly Generation." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.
[3] He, Shidan, et al. "Anomalycontrol: Learning cross-modal semantic features for controllable anomaly synthesis." arXiv preprint arXiv:2412.06510 (2024).

### Questions
1. A key question is: how does the model perform when a new anomaly category is introduced? Can the model be incrementally fine-tuned, or does it require retraining from scratch on all categories? A discussion on this "incremental learning" aspect would strengthen the paper's contribution to a truly deployment-friendly system. If retraining is required, the efficiency benefit of the unified model is significantly diminished over time, as it would be more costly than simply training one new light-weighted model for the one new class. A discussion on this incremental learning capability is a missing, yet critical, piece of the evaluation.
2. The paper tests UniAG's generalizability on a composite dataset that includes BrainMRI and HeadCT images. Are the compared works on BrainMRI and HeadCT the SOTA methods in the medical field?

### Soundness
3

### Presentation
4

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
This paper proposes UniAG, a unified model capable of generating realistic and diverse anomalies across multiple categories. Specifically, it introduces a deep copy-paste anomaly generation strategy, in which a Spatial-Texture Alignment Diffusion Model (STA-DM) learns to fill local region masks with anomaly textures corresponding to user-specified categories. The paper further proposes a novel conditioning mechanism with explicit spatial–category guidance, instead of relying on text embeddings, enabling more realistic and diverse anomaly synthesis. Experimental results on the MVTec AD dataset demonstrate the effectiveness of the method.

### Strengths
1. The method is designed to address several issues in prior work, and introduces a unified model.

2. The writing is relatively clear and fluent.

### Weaknesses
1. Table 2 results are not very competitive. Even under the 4-shot setting with both normal and anomalous samples, the performance is still noticeably below recent FSAD SOTA methods that use only 4 normal shots (e.g., AnomalyAny, PromptAD).

2. The dataset is limited, experiments are conducted only on MVTec AD.

3. The experimental pipeline is relatively complex and may have limited applicability. Training and synthesis rely on foreground masks; the paper uses foreground masks from Zhang et al. (2023a) to locate object foregrounds in normal images. In other domains, accurate masks may not be available.

4. The paper notes that DualAnoDiff is incompatible with FSN-style reconstruction detectors because it cannot provide a background-aligned normal counterpart; this should be flagged as a setup difference rather than evidence of inferiority.

5. Minor issue: there are several spelling errors in the paper (e.g., “performancew” in the caption of Figure 1).

### Questions
see weaknesses:

1. Table 2 results are not very competitive. Even under the 4-shot setting with both normal and anomalous samples, the performance is still noticeably below recent FSAD SOTA methods that use only 4 normal shots (e.g., AnomalyAny, PromptAD).

2. The dataset is limited, experiments are conducted only on MVTec AD.

3. The experimental pipeline is relatively complex and may have limited applicability. Training and synthesis rely on foreground masks; the paper uses foreground masks from Zhang et al. (2023a) to locate object foregrounds in normal images. In other domains, accurate masks may not be available.

4. The paper notes that DualAnoDiff is incompatible with FSN-style reconstruction detectors because it cannot provide a background-aligned normal counterpart; this should be flagged as a setup difference rather than evidence of inferiority.

5. Minor issue: there are several spelling errors in the paper (e.g., “performancew” in the caption of Figure 1).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents UniAG, a novel approach designed to address the issue of lacking anomaly samples in multi-class anomaly detection tasks. Traditional methods require training a separate model for each anomaly category, resulting in low training and deployment efficiency. Besides, GANs or copy-and-paste methods often generate anomalous samples that lack authenticity or diversity. UniAG is a diffusion-based few-shot anomaly generation framework that uses a learnable local “deep copy-paste” strategy plus explicit spatial–category conditioning to generate realistic, diverse multi-class anomalous patches from only a few examples, improving downstream detection, localization and classification.

Contributions:
1. Proposes UniAG, the first (practical) unified FSAG model that generates multiple anomaly categories with a single network (avoids one-model-per-class).
2. Introduces a deep copy–paste local generation pipeline: learnable anomalous patch synthesis + blending into normal backgrounds to preserve background fidelity and reliably synthesize small anomalies.

### Strengths
Strengths:
1. Unified Multi-Class Generation: Enables a single model to support up to 73 anomaly categories on MVTec-AD dataset, resolving the training and deployment burden of "one model per category," facilitating industrial applications and model maintenance.

2. Explicit Conditional Design: Spatial-Category Embedding. Replaces ambiguous textual conditions with explicit spatial + category conditions, providing the diffusion model with precise local location and category signals, thereby improving the generation and category consistency of small-region anomalies.

3. Anomaly Injection Adapter and Local Training Strategy: By injecting a trainable branch AIA and local crop/affine enhancements into UNet, the model can focus on learning local texture distributions, generating diverse and realistic anomaly patches, and stably blending them onto a normal background.

4. Scalability and Data Efficiency: The authors present experiments on scalability with increasing number of categories and data volume, showing that the model experiences minimal performance degradation and significant gains as the number of categories and generated samples increases, indicating that the method has good adaptability.

### Weaknesses
Weaknesses:

1. Still relies on a small number of real anomaly samples and their masks.

    I'm not saying that FSAG's task is unreasonable, but your method seems to heavily rely on anomalous images from a single category, failing to transfer anomalous data across categories. Especially for texture data, can scratches from wood be transferred to tiles? Can cracks from tiles be transferred to wood? I believe the core goal of FSAG isn't to generate a large number of anomalous images from a small number of anomalous images, because for newly designed, newly produced categories, we struggle to even obtain a single anomalous image. It seems that through cross-category transfer, we can generate some anomalous samples for model training.

2. Sensitive to mask/foreground positioning quality.

    SCE relies on accurate local masks; if the mask is biased (boundary misalignment, missing label), the generated texture may be misaligned or produce boundary artifacts, affecting downstream detection. I think what you absolutely must consider is the accuracy of the labels. Due to the nature of the FSAG task, only a very small number of anomalous images can be used. Therefore, if noisy labels appear in these anomalous images, the features learned by the model will be severely affected.
    Mitigation suggestions: Add mask pertubation augmentation, post-processing fusion (improved Poisson blending/seamless cloning), or add mask noise simulation during training.

3. Possible generation-detection bias.

    While UniAG prioritizes realism, synthetic samples may still exhibit micro-statistical differences from real anomalies, and the trained detector may be sensitive to these differences. In your qualitative experimental results, I can clearly see the lack of realism in the carpet class and some other categories. A clear boundary between the anomalous and normal regions may not be what we want. Cut-paste is the first step; how to make the final generated result smoother is a question that needs to be considered.

### Questions
1. Can scratches from wood be transferred to tiles? Can cracks from tiles be transferred to wood?
2. I want to see if, under the 4-shot task setting, the label of one image is not accurate enough, which would lead to a significant decline in the experimental results.
3. Can you resolve the issue of insufficient realism in the generated images, especially in the carpet category?

### Soundness
2

### Presentation
2

### Contribution
2
