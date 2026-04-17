# Toward Robust Image Manipulation Localization: A Novel Framework with VLMs and Weight-Aware Decoder

- Decision: Reject
- Scores: 6, 4, 8, 2

## Abstract
Image Manipulation Localization (IML) aims to identify and pinpoint regions within an image that have been forged or manipulated. Although some progress has been made in the task of IML, existing techniques still face several challenges. First, tampering techniques are diverse and complex, leaving various tampering artifacts in images. To effectively identify different types of tampered images, the model must extract comprehensive and highly discriminative tampering features. Second, some frameworks of IML use identical weights to fuse features from different scales during the decoding process, ignoring the varying sensitivity of different scales to the prediction results. To address these challenges, we propose a novel framework VLWA-Net, based on Vision-Language Models (VLMs). This framework leverages a VLMs-enhanced Artifact Extractor and a Multi-Domain Artifact Modulator to capture rich and discriminative tampering features, combining with traditional noise features as auxiliary cues. Next, we introduce a Weight-Aware Decoder (WAD) that comprehensively accounts for the sensitivity differences across scales and among feature points within the same scale. Additionally, the overall framework is trained using a Joint Information Supervision strategy, which enhances the model’s ability to capture and perceive the details of tampered regions. The experimental results demonstrate that the proposed framework significantly improves accuracy on multiple mainstream test datasets and exhibits strong robustness and generalization capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper, Toward Robust Image Manipulation Detection under Unknown Perturbations, investigates how current image manipulation detection (IMD) models fail when confronted with unseen perturbations such as noise, compression, and resizing. The authors systematically analyze this vulnerability and propose a robust training strategy that incorporates perturbation simulation and feature regularization to enhance generalization. The proposed method is benchmarked across several datasets (e.g., CASIA, Coverage, NIST) and perturbation types, demonstrating consistent performance improvements over existing IMD models. The study provides both theoretical insights and empirical evidence on the robustness gap between standard and perturbed test conditions.

### Strengths
The paper addresses an important and underexplored challenge in IMD—robustness to unseen perturbations—making it highly relevant to real-world forensic applications. The experimental design is comprehensive, covering diverse datasets and perturbation settings. The proposed method is simple yet effective, easily integrated into existing architectures without major architectural changes. Moreover, the empirical analysis is thorough, offering clear comparisons with strong baselines and detailed ablation studies. The paper is well-written, the motivation is clear, and the results convincingly show that robustness can be significantly improved through the proposed strategy.

### Weaknesses
While the contributions are solid, several limitations remain. The paper focuses primarily on low-level perturbations and does not explore more complex, real-world manipulations such as GAN-based editing or compositional attacks. The theoretical justification for why the proposed regularization improves generalization remains relatively shallow. Moreover, the model diversity in experiments is limited—only a few representative IMD architectures are evaluated. Including larger or open-source vision transformers and more commercial or closed-source detectors would make the claims more robust. Finally, the evaluation mainly considers static perturbations, leaving open questions about performance under sequential or compound distortions.

### Questions
1. Have the authors tested the robustness of the proposed method on semantic or GAN-based manipulations, beyond pixel-level perturbations?

2. How sensitive is the performance to the choice and intensity of simulated perturbations during training?

3. Could the authors include results for additional model families, such as ViT-based or multimodal forensic detectors, to assess generality across architectures?

4. Does the proposed regularization improve performance under combined or dynamically varying perturbations, as might occur in real-world social media compression pipelines?

5. Would the authors consider releasing a perturbation-robust benchmark or dataset split to facilitate standardized evaluation for future research?

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
4

### Summary
The paper proposes a multi-scale fusion network for forgery detection. While the problem is relevant, I have several major concerns about the paper's technical novelty, the significance of its proposed contributions, and the thoroughness of its experimental evaluation. The core methodology seems to be an incremental combination of existing techniques, and the justification for certain design choices is not convincing. Furthermore, the evaluation is incomplete and lacks critical analysis. For these reasons, I cannot recommend acceptance of the paper in its current form.

### Strengths
The topic is meaningful. This paper is well-structured.

### Weaknesses
1. The core motivation and technical approach of the paper， fusing multi-scale features from RGB and noise streams， lacks novelty. This paradigm has been extensively explored in many prior works [r1-r2]. In the current state of research (as of 2024/2025), simply applying this fusion strategy is no longer considered a novel contribution. The paper fails to differentiate itself sufficiently from the large body of existing literature that employs similar principles.

[r1] Pixel-Inconsistency Modeling for Image Manipulation Localization. TPAMI 25

[r2] MVSS-Net: Multi-View Multi-Scale Supervised Networks for Image Manipulation Detection. TPAMI 22

2. The authors position the "Weight-aware Aggregation Decoder" (WAD) as a key contribution for adaptively weighting features from different scales. However, I am concerned about its actual technical significance. If my understanding is correct, the goal of WAD is to learn the importance of different feature streams. This same objective can be readily achieved using simpler and well-established methods. For instance, one could concatenate the features from all views along the channel dimension and then apply a standard 1x1 or 3x3 convolution, which would inherently learn to weigh and fuse the channels. Alternatively, a channel attention mechanism like SENet could be employed to explicitly model channel-wise interdependencies. The fact that these simpler alternatives exist casts doubt on the necessity and novelty of the proposed WAD module.

3. The experimental evaluation appears to focus exclusively on pixel-level localization metrics (e.g., pixel-wise F1-score or IoU). However, the paper fails to report image-level detection results (i.e., the binary classification performance of discriminating between real and manipulated images). This is a critical omission, as many real-world applications require a fast, image-level decision before performing expensive pixel-level analysis. A comprehensive evaluation must include standard image-level metrics like AUC or accuracy.

4. The proposed method involves multiple image resizing operations. It is well-documented that resizing algorithms (e.g., bilinear, bicubic interpolation) introduce their own high-frequency artifacts. These artifacts could act as confounding signals, potentially being learned by the detector instead of the actual manipulation traces. This raises two questions: (1) Have the authors analyzed or considered the negative impact of these resizing-induced artifacts on the detection performance? (2) Could the authors specify which interpolation method was used for resizing and provide a justification for this choice? Different methods produce different artifacts, which could affect reproducibility and performance.

5. The paper reports F1-scores as a primary metric, but it is not specified what decision threshold was used to compute them. Was a fixed threshold of 0.5 used for the probability maps, or was an optimal threshold selected for each model/dataset based on a validation set? This information is crucial for fair comparison and reproducibility.

6. The results show that adding a noise stream causes a significant performance degradation for the VLWA-Net baseline. The paper acknowledges this but dismisses it with a very brief explanation. Such a counterintuitive and dramatic result warrants a much more rigorous and in-depth analysis. Why does a model supposedly designed for this domain fail so catastrophically when provided with what should be additional, useful information? A deeper investigation into this phenomenon is necessary to build confidence in the authors' experimental methodology and conclusions.

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper tackles the persistent challenges in Image Manipulation Localization (IML). To address these issues, the authors introduce VLWA-Net, a robust framework leveraging VLMs through a fine-tuned VAE and a MDAM to capture the diverse and complex artifacts. The framework further employs a Weight-Aware Decoder (WAD) that adaptively weights features across scales and within scales to enhance localization precision. Extensive experiments on multiple benchmarks show that VLWA-Net outperforms state-of-the-art models in accuracy and robustness.

### Strengths
1. The article provides an in-depth analysis of the current challenges in IML tasks, such as the diversity of artifacts.
2. Proposed combination of VAE and MDAM effectively extracts diverse tampering artifacts.
3. Proposed WAD effectively decodes multi-scale features through adaptive weighting.
4. The framework is evaluated across seven benchmark datasets, and robustness is tested under common distortions (JPEG, blur, noise).
5. The idea is straightforward and easy to implement.

### Weaknesses
1. Some technical details are not clearly explained, particularly regarding the MDAM and WAD modules.
2. The backbone among all the methods listed in Table.1 are different, making the comparison unfair.
3. Fail to compare with more effective models such as APSC-Net, or discuss them in related works, limiting the demonstrated effectiveness.
4. Could the authors provide the actual dimensionalities of the feature maps from GAES and NTS at each scale, and explain how alignment is ensured before fusion in DFFM? This is important because scale alignment across architectures with different receptive fields (e.g., ConvNeXt vs. ViT) can significantly impact fusion effectiveness.

[1] Qu C, Zhong Y, Liu C, et al. Towards modern image manipulation localization: A large-scale dataset and novel methods[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 10781-10790.

### Questions
Please provide a detailed explanation of the working mechanism of the SCFF module.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes VLWA-Net, a framework for image manipulation localization that allegedly leverages Vision-Language Models (VLMs) to extract tampering artifacts. The model integrates a “VLMs-enhanced Artifact Extractor,” a Multi-Domain Artifact Modulator (MDAM), and a Weight-Aware Decoder (WAD). The authors claim that by incorporating VLMs, the method achieves superior robustness and generalization across multiple benchmarks.

### Strengths
- The overall architecture is clearly described and experimentally validated.
- Experiments are comprehensive across several datasets.

### Weaknesses
## Main Issue
- The central concept is flawed: the paper repeatedly calls SAM a Vision-Language Model, but SAM is a vision-only segmentation model, not trained with textual input or multimodal supervision.
- Claims such as “VLMs exhibit superior universal feature extraction capabilities” (lines 50–51) are unsupported by evidence.
- The proposed “VLMs-enhanced Artifact Extractor” could be replaced by any visual backbone (e.g., DINOv3, ViT) without affecting the framework.
- The work’s framing around VLMs is misleading and self-inconsistent, as “language” plays no role in the model.

## Minor Issue
- Using VAE as a module name is highly inappropriate, as it conflicts with the well-established term Variational Autoencoder in ML literature.


In summary, I believe the core argument of this paper is invalid, and the discussions built around it fail to support the claimed contributions to the network and the task. The paper lacks solid theoretical and logical foundations; therefore, I **strongly recommend rejection.**

### Questions
As discussed in the Weaknesses section, the paper’s core argument is not well-founded.

### Soundness
1

### Presentation
3

### Contribution
1
