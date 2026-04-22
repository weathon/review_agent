# Modeling the Density of Pixel-level Self-supervised Embeddings for Unsupervised Pathology Segmentation in Medical CT

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Accurate detection of all pathological findings in 3D medical images remains a significant challenge, as supervised models are limited to detecting only the few pathology classes annotated in existing datasets. To address this, we frame pathology detection as an unsupervised visual anomaly segmentation (UVAS) problem, leveraging the inherent rarity of pathological patterns compared to healthy ones. We enhance the existing density-based UVAS framework with two key innovations: (1) dense self-supervised learning for feature extraction, eliminating the need for supervised pretraining, and (2) learned, masking-invariant dense features as conditioning variables, replacing hand-crafted positional encodings. Trained on over 30,000 unlabeled 3D CT volumes, our fully self-supervised model, Screener, outperforms existing UVAS methods on four large-scale test datasets comprising 1,820 scans with diverse pathologies. Furthermore, in a low-shot supervised fine-tuning setting, Screener surpasses existing self-supervised pretraining methods, establishing it as a state-of-the-art foundation for pathology segmentation. The code and pretrained models are available at https://github.com/mishgon/screener.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose Screener, an unsupervised visual anomaly segmentation (UVAS) method, which leverages dense self-supervised pre-training and masking-invariant dense feature conditioning variables as replacement for positional encodings. The authors train Screener on 30k unlabeled CT-volumes and outperform existing UVAS methods and self-supervised pretraining methods when fine-tuning their method.

### Strengths
The method the authors propose, is really nice, as it leverages dense self-supervised pretraining, which is a good choice given their goal of dense downstream applications and their addition of their masking-invariant dense feature conditioning variables as replacement for positional encoding is also well motivated and creative.
Additionally the clarity of the paper is also very high, with the authors explaining the different aspects very well.

### Weaknesses
The majority of my criticisms hinge around two key points of the authors paper.

1) The authors claim their method exceeds current self-supervised learning methods. However, the authors don't compare against MAE pre-training, which was shown to be the strongest SSL pre-training for 3D medical image computing in the medical domain in a recent benchmark [1]. In particular the chosen VoCo and SwinUNETR pre-training baselines were shown to be bad for segmentation in general, making this claim not substantiated.

2)  The evaluation protocol of using just 25 training cases is limited. If the authors claim their SSL method to be overall useful for segmentation I would like them to additionally evaluate their pre-training against other pre-training methods on a full-data regime. This is largely, because pre-training methods in general appear to yield performance benefits in small-data regimes, however when more data is available it may not do so anymore. Having this information is crucial for practitioners to know which SSL method to choose given their data availability.


If the authors 1) include MAE pre-training as an additional SSL baseline (when finetuning) and 2) include a finetuning experiment in a  full-data-regime and 3) include a full-length nnU-Net training as reference, I will raise my score further.


[1] Wald, Tassilo, et al. "An OpenMind for 3D medical vision self-supervised learning." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.

### Questions
Q1: I am not sure if I missed it but how do you get from the anomaly maps to hard predictions as needed for DSC measurement? Is there some thresholding and if so, how is the thresholding done in the unsupervised and supervised setting?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Screener, a self-supervised anomaly segmentation framework for 3D CT. The method combines dense self-supervised descriptor learning, masking-invariant conditioning embeddings, and density modeling to detect abnormal patterns. The model trains on over 30,000 unlabeled CT volumes. Screener is tested on four public CT benchmarks under both unsupervised and fine-tuning settings.

### Strengths
The problem is clearly motivated, as a supervised method cannot capture all distributions of pathologies.

The integration of dense SSL for descriptors and SSL-learned conditioning embeddings looks novel.

The proposed model is tested on four public CT benchmarks under both unsupervised and fine-tuning settings. Strong and consistent performance is observed.

### Weaknesses
While the method is well-executed, the contribution may appear incremental as the dense SSL for voxel embeddings and conditioning is a straightforward extension of VICReg and conditioning replacing sin-cos encodings is conceptually natural. The paper would benefit from clearer articulation of why condition embeddings are fundamentally more informative than APE or sin-cos encodings beyond empirical gains.

The proposed method looks closely related to an existing method [1]. It would be beneficial if the authors acknowledge and contrast with the prior work.

The method produces per-voxel anomaly scores. A threshold is necessary to perform the final segmentation. It is unclear how this threshold is determined.

References:

[1] Seince, Maxime, Loı̈c Le Folgoc, Luiz Facury De Souza, and Elsa Angelini. "Dense Self-Supervised Learning for Medical Image Segmentation." In Medical Imaging with Deep Learning, pp. 1371-1386. PMLR, 2024.

### Questions
1. Are there reasons why condition embeddings are fundamentally more informative than APE or sin-cos encodings?

2. How is the threshold determined to obtain the segmentation?

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
The paper presents Screener, a fully self-supervised framework for unsupervised pathology segmentation in 3D CT. It reframes pathology detection as unsupervised visual anomaly segmentation (UVAS), under the assumption that pathological regions are statistically rare. The method enhances density-based UVAS by (i) learning dense voxel-level self-supervised descriptors tailored to CT data, and (ii) introducing masking-invariant contextual embeddings as conditioning variables within a conditional density model. The models are trained on 30k unlabeled CT volumes. Screener achieves state-of-the-art results on four public CT datasets (LIDC, MIDRC, KiTS, LiTS). When distilled into a single UNet and fine-tuned with limited labeled data, it matches or surpasses medical SSL baselines.

### Strengths
Reviewer acknowledges the following contributions:

 **Addresses an important medical challenge**: The paper tackles the critical and realistic problem of detecting all pathological findings in 3D CT without requiring pixel-level annotation, which is a limitation that hinders clinical deployment of current supervised methods.

 **Interesting and well-motivated technical idea**: 

- The proposed use of self-supervised learning for both the descriptor and conditioning models is conceptually appealing. It enables the system to model normal anatomy and contextual expectations directly from large unlabeled data, thereby detecting deviations that correspond to abnormalities.

- Furthermore, the integration of dense SSL with masking-invariant conditioning is simple but effective. It allows a simple Gaussian density model to perform on par with more complex normalizing flows, indicating that meaningful contextual embeddings can simplify anomaly modeling.

**Broad evaluation**: Experiments across four diverse datasets demonstrate that Screener generalizes well across organs and pathologies. The unsupervised anomaly segmentation results are significant, with clear ablation studies supporting each component.

 **Clear presentation**: The paper is clearly written, logically organized, and easy to follow, making complex ideas accessible to both machine learning and medical imaging audiences.

### Weaknesses
I found the following weaknesses in the current manuscript:

- **Core conceptual clarity**: The motivation for why optimizing the **conditional density model** $q_{\theta}(y|c)$. helps detect abnormal regions, is not fully articulated. Since this is a central concept, further clarification and discussion would strengthen the theoretical grounding. For instance, authors need to provide a visualization of normal pixels near the abnormal regions and show how the heatmap behaves to provide insights.

- **Experimental organization**: The unsupervised experiments could be better structured to highlight the benefits of dense SSL. For instance, baselines should be grouped into (i) image-level SSL models (e.g., autoencoder) as well as adding SOTA models like LVM-Med [1] (developed for ResNet-50 and suitable with U-Net) and (ii) dense SSL models (the current author compared), to clearly demonstrate that dense SSL drives the improvements. 

- **Supervised performance gap** – While competitive, the fine-tuned Screener in Table 2 lags behind certain SOTA supervised or SSL-pretrained models. The authors could explore initializing the distilled UNet with a pretrained medical backbone (e.g., RadImageNet or LVM-Med to enhance feature representations.

- **Presentation detail** – Some result table formats (e.g., Tables 1–3) could be reformatted and polished for a more professional appearance and readability.

[1] Lvm-med: Learning large-scale self-supervised vision models for medical imaging via second-order graph matching, NeurIPS 2023.

[2] RadImageNet: an open radiologic deep learning research dataset for effective transfer learning." Radiology: Artificial Intelligence

### Questions
- In the self-supervised learning settings, besides the medical models, can authors examine the performance of the generalized model, for e.g., the Grounding Dino technique, which can retrieve objects given prompt input and apply some threshold techniques to filter noise [3]? Incorporating such experiments is interesting and highlights the benefit or potential weakness of the current strategy.

-  It would be great to see the advanced performance by applying the U-Net with pre-trained medical models during the distillation process, aiming to bridge the gap between current performance and other SOTA.


[3] Grounding Dino 1.5: Advance the" edge" of open-set object detection.

### Soundness
3

### Presentation
3

### Contribution
3
