# Adaptive Gray: Reducing Color Dependency to Improve Generalization in Deepfake Detection

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Deepfake technology, powered by advanced generative models such as GANs and diffusion models, has raised serious ethical and security concerns due to its potential for creating realistic yet deceptive content. As these generative models become increasingly sophisticated, humans struggle to distinguish real images from synthetic ones, highlighting the need for reliable machine-based detection. However, current detection methods often generalize poorly, especially across different generative models (cross-generator) and diverse image domains (cross-dataset), which limits their reliability in real-world deployment. 

To address this issue, we observe that strong \emph{color dependency} can be unnecessary and may even impede deepfake detection. Building on this insight, we propose Adaptive Gray (AG), a lightweight, learnable RGB-to-grayscale module that compresses color channels before classification to improve generalization. We validate AG on DiffusionForensics (in-distribution and cross-generator) and GenImage (cross-dataset) benchmarks. On GenImage, AG improves mean ACC / AP / TPR@5\%FPR by over 15, 30, and 30 percentage points, respectively, compared to an RGB ResNet-50 baseline, while remaining competitive with or superior to strong state-of-the-art detectors. At the same time, AG introduces only three additional parameters and can reduce inference cost by up to \(10^{4}\times\) relative to reconstruction-based pipelines, making it both effective and highly efficient for practical deepfake detection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an adaptive and learnable method to compress the RGB color information in generated images, aiming to improve the generalization ability of classifiers for binary deepfake detection. Specifically, the authors introduce a mechanism that learns three scalar weights corresponding to each color channel, which are jointly optimized alongside the downstream deepfake classifier. Experiments are conducted on several established benchmarks, using DiffusionForensics for training and GenImage for evaluation.

### Strengths
- Paper Presentation: I previously reviewed this paper during the CVPR 2025 cycle. After comparing the current version with the earlier submission, I find that the presentation has significantly improved.

- The core idea proposed is simple but appears effective within the defined task setting, notably in improving detection accuracy and generalization. The authors also provide an empirical analysis section that helps clarify the functionality of the proposed adaptive grayscale method.

### Weaknesses
- W1: From a rather high-level perspective, the proposed task presents a rather simplified version of the deepfake detection problem, which is the binary classification task. There are more recent and advanced generative techniques that modify a small region or subpart of the images. This limits the practical contributions of the work as a pure empirical paper, which is also partially demonstrated through the reported 100% detection rate.

- W2: Missing important SOTA baseline: [a] from CVPR’25 seems to be a relevant baseline to discuss and include in this work, which is currently missing.

- W3: Rather limited experimental setup: Although the paper claims to evaluate cross-domain and cross-generator generalization, the current experiments are restricted to the “bedroom” subset of existing benchmarks. While the results are generally favorable and align with the paper’s claims, the narrow scope limits the ability to convincingly demonstrate generalization capabilities.

- W4: In the appendices, the authors perform the combination of the proposed AG and the existing detection method DIRE, and draw conclusions that these two techniques are not compatible. I wonder if this applies to other detection methods like [a], which does not involve reconstruction of the synthesized images. This ablation might help toprovide a more comprehensive understanding of the proposed AG method, as this is a rather empirical and heuristic method.

### Questions
Please see the Weakness for details, particularly W4.

### Soundness
2

### Presentation
3

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
This paper explores the role of color dependency in deepfake detection. The authors hypothesize that color cues may hinder cross-generator and cross-dataset generalization. They first provide empirical evidence supporting this claim via UMAP analyses and separability metrics (Density Overlap, MMD). Based on these findings, they propose Adaptive Gray, a lightweight module that learns optimal RGB-to-grayscale transformation weights through an alternating co-adaptive training procedure with a binary classifier (ResNet-50). The approach is evaluated on the DiffusionForensics and GenImage benchmarks, showing strong gains in generalization and inference efficiency compared to SOTA methods such as DIRE, DE-FAKE, and NPR.

### Strengths
- The paper identifies and systematically validates the overlooked issue of color dependency in deepfake detection. The analysis linking color variance to poor generalization is compelling.

- The Adaptive Gray layer is conceptually simple but effective, offering an interpretable preprocessing mechanism with almost no computational overhead.

- Results on both cross-generator and cross-dataset benchmarks show large, consistent improvements (+20% ACC/AP/TPR), even when compared to stronger and more complex SOTA methods.

- The AG transformation is nearly cost-free at inference, improving speed by several orders of magnitude while retaining accuracy.

Readable and well-structured: The paper is clearly written and logically organized; figures and tables effectively support the claims.

### Weaknesses
Limited ablation analysis:
While the paper includes comparisons between OG (fixed grayscale), AG, and AG+DIRE, it lacks finer ablations on:

1) Initialization of AG weights (wR, wG, wB);

2) impact of backbone architecture (e.g., ViT, ConvNeXt);

3) the training approach (the alternating manner chosen by the authors versus a simple jointly training of the two components).
The alternating optimization scheme is interesting but underexplained. The rationale for choosing it over joint training is not discussed, and the number of epochs per cycle is missing. Clarifying this would strengthen reproducibility.

Experimental setup:
 Although the authors do perform a cross-generator analysis, a broader range of generators should be included. To this end, an evaluation on UniversalFakeDetect Dataset would further prove the hypotheses presented in this work.

Other small changes that need to be addressed:
Fig 1 -> wrong caption: “top row shows original RGB, bottom row displays the same images after our proposed AG”. Actually, it’s the left column and right column. Also, it is not clear what the authors tried to show in the zoomed-in version.
Citations need to be fixed: “…Haliassos Haliassos et al. (2021)…”, “…Wang Wang & Deng (2021)…”.

### Questions
In section 3.2.2. it is not clear why the authors have chosen the presented training approach (alternating between optimizing the two components). Could this setup be more prone to instabilities during training?
The authors did not mention the number of epochs trained in each step from 3.2.2. How many epochs is allocated for each cycle? How many epochs is the classifier trained for? What about the Adaptive Grayscale?
In Figure 3 Step 2, there is a “frozen” sign on the “Real” label and a “unfrozen” (fire) sign on the “Fake” label. What does this represent? Is the model trained only using fake samples?
What is the shape of the AG parameters? In equation (2) it seems that these parameters take the form of a 1D vector of weights, one parameter for each color channel. In Figure 3 on the other hand, the AG component is represented as a set of 2D kernels, one for each color channel.
In Table 1 the results for baselines seem off. For example: DIRE trained on ADM from DiffusionForensics and tested in domain achieves an ACC of 95.1 and an AP of 99.5, but the results reported by the authors of DIRE on DiffusionForensics (trained on ADM and tested in domain) shows an ACC of 100/100. Is there a reason for this discrepancy? Did the authors use only a subset of the testing set?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents Adaptive Gray (AG), a novel deepfake detection method designed to enhance generalization across various generative models and datasets by reducing color dependency. AG processes RGB images into adaptive grayscale representations, improving the detection of subtle, texture-based artifacts inherent in synthetic images. Extensive experiments demonstrate AG’s superior performance, achieving significant improvements in accuracy, average precision, true positive rate, and inference efficiency. AG outperforms state-of-the-art methods, particularly in cross-dataset and cross-generator generalization.

### Strengths
-  AG offers a novel solution by identifying and mitigating the issue of color dependency in deepfake detection. The approach of transforming images into adaptive grayscale representations that emphasize texture-based artifacts rather than color details is innovative and addresses a critical gap in the field.
- AG demonstrates superior generalization performance across a variety of generative models and datasets. This ability to adapt to unseen models and datasets sets it apart from existing methods, which often struggle with generalization in diverse scenarios.
- In addition to improved performance, AG also excels in inference efficiency, offering a significant speedup over other SOTA methods, making it a practical choice for real-time deepfake detection applications.

### Weaknesses
- The paper does not include comparisons with other relevant methods like UniFD(CVPR2024), C2P-CLIP(AAAI2025), or Fatformer(CVPR2024), which would help better position AG’s effectiveness in the broader landscape of deepfake detection methods.
- The visual comparison in Figure 2 (a) and (b) fails to show a clear advantage for AG, especially in the DiffusionForensics dataset. The presentation of results across different datasets and generator could be enhanced for more compelling comparisons.
- Despite GenImage’s more extensive dataset (8 classes), the experiments only use 4 classes in testing, which limits the generalization assessment across all categories.
- The choice of DiffusionForensics as the training dataset is somewhat limited. Using more widely used datasets like GenImage could have better supported the claim of AG’s generalization performance across diverse image categories.

### Questions
- Why was the GenImage training dataset not utilized in the experiments? It would be insightful to see AG’s performance with this more comprehensive dataset.
- Could the authors provide a more detailed comparison with other state-of-the-art methods such as UniFD, C2P-CLIP, and Fatformer? This would provide a clearer understanding of AG’s position in the field.
- In Figure 2, the visual advantage of AG over other methods is not clear. Could the authors clarify the specific advantages that AG provides, especially in cross-dataset scenarios?
- Why were only 4 classes from the GenImage dataset used for testing, when the dataset offers 8 classes? What was the rationale behind this choice, and would testing across all classes provide more robust results?
- Can the authors provide a more detailed explanation of why texture artifacts are left behind during image generation? What specific limitations or shortcomings in the generative process lead to these artifacts? An explanation that incorporates the principles of the generative models would make this argument stronger.
- Could the authors elaborate on why the differences between the RGB channels are particularly effective at capturing texture-based forgery artifacts?
- I have concerns regarding the experimental setup and the fairness/completeness of the comparisons.

### Soundness
3

### Presentation
3

### Contribution
2
