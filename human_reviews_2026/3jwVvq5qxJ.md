# Quantifying Statistical Significance in Diffusion-Based Anomaly Localization via Selective Inference

- Decision: Reject
- Scores: 6, 2, 6

## Abstract
Anomaly localization in images—identifying regions that deviate from expected patterns—is vital in applications such as medical diagnosis and industrial inspection. A recent trend is the use of image generation models in anomaly localization, where these models generate normal-looking counterparts of anomalous images, thereby allowing flexible and adaptive anomaly localization. However, these methods inherit the uncertainty and bias implicitly embedded in the employed generative model, raising concerns about the reliability. To address this, we propose a statistical framework based on selective inference to quantify the significance of detected anomalous regions. Our method provides $p$-values to assess the false positive detection rates, providing a principled measure of reliability. As a proof of concept, we consider anomaly localization using a diffusion model and its applications to medical diagnoses and industrial inspections. The results indicate that the proposed method effectively controls the risk of false positive detection, supporting its use in high-stakes decision-making tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This study proposes a method to compute p-values for anomaly localization based on diffusion models (DAL). Because the anomalous regions are data-dependent, standard hypothesis testing results suffering from Type I error. Therefore, selective inference is necessary. So, this study proposes selective inference for diffusion models.

### Strengths
It has a long history and is widely used to use the reconstruction errors of deep generative models for anomaly detection. However, it has been difficult to assess what constitutes a meaningful difference. The proposed method addresses this question in a rigorous way.

There are many studies on selective inference and p-values, but this study addresses a highly unique setting, which gives it strong originality. The extensions needed to apply the framework to diffusion models are non-trivial and show a great contribution.

### Weaknesses
This study assumes a U-Net with ReLU. It may not apply to architectures that use other activation functions or layers. For example, more modern diffusion models heavily use attention layers and normalization layers. Can this method still be applied?

In the context of unsupervised anomaly segmentation, reconstructions by diffusion models are not necessarily the default, and for strong performance, feedforward methods appear more powerful and more common (for example, see the leaderboard of the VAND 3.0 Challenge at CVPR). In this sense, the practical utility of the proposed method may be limited.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a statistical framework based on Selective Inference (SI) to quantify the significance of anomalous regions detected by diffusion-based anomaly localization methods. The authors focus on Denoising Diffusion Probabilistic Models (DDPM) as a proof-of-concept, applying it to medical diagnosis (e.g., brain MRI) and industrial inspection tasks. The core idea is to compute valid p-values that control the false positive detection rate by conditioning on the data-driven selection event induced by the diffusion model's reconstruction

### Strengths
1. First to use Selective Inference in anomly detection: The application of Selective Inference to diffusion-based anomaly localization is highly novel and addresses a critical, often overlooked problem of reliability and statistical validity in this rapidly growing field. This is particularly relevant for high-stakes applications like medical diagnosis.
2. Solid Theoretical Foundation: The paper is well-grounded in selective inference theory. The derivation of the conditional sampling distribution and the proof for the validity of the proposed p-value are sound and clearly presented in the appendix.
3. Awareness of Limitations: The authors acknowledge the computational demands of their method, which is a significant and honest point for discussion regarding practical deployment.

### Weaknesses
1. Lack of Depth in SI Applications: The section lists citations but provides only high-level categorizations (e.g., "linear model features," "complex feature selection," "unsupervised learning tasks," "deep learning models"). It does not explain specific application scenarios, such as: What problems were solved in linear models?
2. Lack of novelty: The core framework essentially extends established SI techniques—originally developed for feature selection in linear models and later adapted to deep learning and unsupervised tasks—to the context of diffusion models like DDPM, without introducing fundamentally new methodological innovations or theoretical breakthroughs
3. Insufficient Discussion of Computational Complexity: The computational burden of the parametric programming approach is drastically understated. For high-resolution images, exhaustively searching the one-dimensional line to identify all truncation intervals Z is likely computationally prohibitive. The paper provides no data on runtimes or scalability, making it impossible for the reader to assess its practical feasibility.
4. Limited and Weak Baselines: 
	1. The experimental comparisons are insufficient to convincingly demonstrate the method's advantage. The chosen baselines are weak: the naive and permutation methods fail to control Type I error, making power comparisons with them uninformative. While the Bonferroni correction controls error rates, it is a notoriously conservative baseline, and outperforming it does not sufficiently prove the method's power. 
	2. The evaluation is limited to only Type I error and power at the image level. It does not provide segmentation-level metrics, such as AUROC or F1-score, which are standard in anomaly detection (AD). Consequently, the paper fails to demonstrate a tangible improvement for practical AD tasks.
	3. The description of the real-world datasets is unclear, omitting essential details for reproducibility, such as precise data splits and preprocessing steps.

### Questions
1. Computational Scalability: Could you provide concrete data on the runtime of your algorithm (e.g., for the largest image size n=4096 or the 128x128 MVTec images)? How does the computational cost scale with image size (n) and model complexity? This is critical for assessing the method's practical utility.
2. Baseline Comparison:** Would you consider adding a comparison with the Benjamini-Hochberg procedure to control the False Discovery Rate (FDR)? This is a more powerful and commonly used alternative to Bonferroni for multiple testing, and a comparison would better situate the performance of your method.
3. Parameter Sensitivity and Robustness: The anomaly region Mx​ is highly dependent on the threshold λ. How sensitive are the Type I error and power of the DAL-Test to the choice of λ?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this work, the authors propose a statistical framework to quantify the reliability of anomaly localization methods that use diffusion models. These generative models are used in domains like medical diagnosis and industrial inspection by reconstructing a "normal-looking" version of an input image; the difference between the input and reconstruction highlights potential anomalies. However, the authors note that inherent uncertainties and biases in these models can lead to inaccurate localizations, posing a critical risk.
To address this, the paper introduces the Diffusion-based Anomaly Localization (DAL) Test, which is based on the principles of Selective Inference (SI). The key problem with standard statistical tests is that the hypothesis (i.e., the specific anomalous region is selected using the same data that is used for testing. This "double-dipping" invalidates traditional p-values and leads to an inflated false positive (Type I error) rate. The proposed DAL-Test framework computes a valid p-value by performing the statistical test conditional on the selection event—the fact that the specific anomalous region was identified by the diffusion model. The authors formulate this as a two-sample test comparing the mean pixel values in the detected region between the test image and a reference image.
Technically, the framework conditions on a nuisance parameter, which reduces the problem to a one-dimensional search. The authors show that if the diffusion model's U-Net uses piecewise-linear activation functions (like ReLU), the entire reconstruction process is piecewise-linear. This crucial insight allows the conditional sampling distribution of the test statistic to be characterized as a truncated Gaussian distribution. The truncation intervals (the set of values that produce the same anomalous region) are identified analytically using parametric programming.
The authors validate their method on synthetic data and real-world datasets (BraTS brain tumors and MVTec industrial defects). Experiments show that the "naive" and "permutation" methods fail to control the Type I error rate, exhibiting high false positives (e.g., 0.46–0.88 on MVTec). In contrast, the proposed method successfully controls the Type I error rate at the desired significance level (e.g., $\alpha=0.05$) and demonstrates higher statistical power than the conservative Bonferroni correction. The authors conclude that their framework provides a principled way to assess the statistical reliability of diffusion-based anomaly detection.

### Strengths
The following are the positives of this work:

- The paper addresses a gap in the use of generative models for anomaly localization. It moves beyond merely generating reconstruction maps and provides a formal statistical framework, the DAL-Test, to quantify the risk of false positive detections by computing valid p-values.
- The authors propose using the selective inference (SI) framework for diffusion models. The key insight is the characterization of the U-Net-based reconstruction process as a piecewise-linear function , which enables the analytical derivation of the conditional sampling distribution as a truncated Gaussian.
- The experimental validation is well-conceived. It correctly focuses on demonstrating the failure of standard and permutation-based approaches to control the Type I error rate.

### Weaknesses
I have the following concerns regarding the paper:
- The entire tractability of the proposed selective inference framework, as detailed in Appendix B, revolves around U-Net and the entire reconstruction process $\mathcal{D}(X)$ being a piecewise-linear function of the input $X$. This is achieved by restricting the model to components like ReLU activations and pooling (line 703). This is a significant constraint, as many state-of-the-art diffusion models employ non-linearities such as SiLU/Swish or attention mechanisms, which are not piecewise-linear. This raises a critical question about a potential trade-off: does one have to sacrifice the generative performance of the diffusion model to gain statistical rigor? The paper does not explore this trade-off or discuss the method's applicability to non-piecewise-linear architectures.
- The method relies on parametric programming (Algorithm 2) to identify all truncation intervals $\mathcal{Z}$ along a 1D path. For a deep, piecewise-linear network, the number of linear regions (and thus potential intervals) can grow extremely large. The authors acknowledge this in the limitations ("growing the size of the diffusion model also leads to increased computational demands," line 483) but do not provide a formal complexity analysis or empirical runtime data. This makes it difficult to assess the practical feasibility of the DAL-Test for larger, higher-resolution images (e.g., $256 \times 256$ or $512 \times 512$) or deeper models, which are common in real-world applications.
- The paper formulates the statistical test as a two-sample comparison of the mean pixel value within the detected region $\mathcal{M}_x$ (Eq. 6, line 246). This test statistic (Eq. 7) is sensitive to changes in mean intensity but may lack statistical power for anomalies characterized by other features. For instance, subtle textural changes, fine scratches, or complex tissue abnormalities might not significantly alter the mean pixel value of a region but would be visually distinct. The framework's validity is not in question, but its ability to detect these more complex, non-mean-based anomalies could be limited.

### Questions
Please see weakness section above

### Soundness
3

### Presentation
2

### Contribution
3
