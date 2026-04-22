# The Unanticipated Asymmetry Between Perceptual Optimization and Assessment

- Avg Score: 5.33
- Decision: Reject
- Scores: 4, 8, 4

## Abstract
Perceptual optimization is primarily driven by the fidelity objective, which enforces both semantic consistency and overall visual realism, while the adversarial objective provides complementary refinement by enhancing perceptual sharpness and fine-grained detail. Despite their central role, the correlation between their effectiveness as optimization objectives and their capability as image quality assessment (IQA) metrics remains underexplored. In this work, we conduct a systematic analysis and reveal an unanticipated asymmetry between perceptual optimization and assessment: fidelity metrics that excel in IQA are not necessarily effective for perceptual optimization, with this misalignment emerging more distinctly under adversarial training. In addition, while discriminators effectively suppress artifacts during optimization, their learned representations offer only limited benefits when reused as backbone initializations for IQA models. Beyond this asymmetry, our findings further demonstrate that discriminator design plays a decisive role in shaping optimization, with patch-level and convolutional architectures providing more faithful detail reconstruction than vanilla or Transformer-based alternatives. These insights advance the understanding of loss function design and its connection to IQA transferability, paving the way for more principled approaches to perceptual optimization. Code and models will be released publicly.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the relationship and potential asymmetry between perceptual optimization objectives and their effectiveness as image quality assessment (IQA) metrics in image super-resolution (SR) tasks. The authors construct a family of DISTS-style perceptual metrics with diverse backbone architectures, design an array of adversarial discriminators, and perform comprehensive experiments on SwinIR. They report that the best-performing IQA metrics do not necessarily yield superior perceptual optimization, and adversarially trained discriminators, while effective at artifact suppression, do not transfer as strong initializations for IQA models. The work highlights the critical influence of discriminator design, particularly patch-level convolutional architectures, on optimization results and training stability.

### Strengths
1. Comprehensive Experimental Evaluation: The paper presents an controlled empirical study, spanning multiple backbone architectures (VGG-16, ResNet-50, ConvNeXt, CLIP-ViT, Swin-T, and DINOv2) for both perceptual losses and discriminators. This breadth adds significant credibility to the claims.
2. Systematic Analysis of Loss Design: The explicit formulation and analysis of the DISTS-style perceptual metric and its alternative feature backbones is thorough, with concrete mathematical details provided.
3. Insight into Optimization-Assessment Asymmetry: The study surfaces a well-supported, thought-provoking observation that high IQA performance does not guarantee optimization efficacy.
4. Strong Visualizations: The qualitative figures are well-chosen and annotated, highlighting subtle texture and artifact differences that strengthen the analysis.

### Weaknesses
1. Limited Generalization Beyond SR and SwinIR: The study is conducted entirely within the SwinIR SR framework, with all optimization experiments targeting one architecture. There is no evidence showing similar asymmetries arise for other image-to-image or generative tasks.
2. Superficial Discussion of Optimization Failures: While the work documents the underwhelming performance of some metrics in optimization, it rarely delves into why this happens.
3. Insufficient Theoretical Context: While the empirical results are robust, the theoretical motivation for the observed asymmetry is shallow.
4. Overreliance on NR-IQA for Main Evaluation: Table 1 uses NR-IQA scores as the primary measure for comparing perceptual optimization effectiveness. While defensible given the issues noted for FR-IQA, this risks circularity: the choice of NR metric may bias the conclusions, especially if the NR metric is itself optimized using similar backbones.

### Questions
1. On Generalization: Can the observed asymmetry between optimization and assessment be demonstrated for tasks beyond SR or architectures beyond SwinIR?
2. Metric Structure and Failure Analysis: Have you conducted feature visualizations or layer-wise analyses to pinpoint why certain backbones or perceptual metrics perform poorly during optimization?
3. NR-IQA Metric Bias: Your evaluation heavily relies on a set of modern NR-IQA methods. How sensitive are your findings to this choice? Did you observe substantially different trends using alternative NR or FR-IQA metrics?
4. Explaining Training Instabilities (Figure 7): Do you have further diagnostics explaining why DINOv2-based discriminators degrade at higher GAN weights, while ResNet-50 remains robust?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper reveals a surprising mismatch between image quality assessment (IQA) metrics and their usefulness as training objectives: metrics that score highly for IQA don’t necessarily improve perceptual optimization, and this gap widens with adversarial training. It also finds that while GAN discriminators help suppress artifacts and boost realism, especially patch-level, convolutional designs, their learned features transfer poorly when reused to initialize IQA models. Overall, adversarial supervision tends to dominate outcomes once a reasonable perceptual loss is present, challenging the common practice of equating IQA strength with optimization effectiveness.

### Strengths
1.	This finding shows that high-scoring IQA metrics do not necessarily make better optimization objectives, overturning a long-held assumption and opening a new direction for perceptual loss design.
2.	The study presents a carefully controlled experimental framework spanning multiple loss compositions, discriminator architectures, and perceptual backbones, ensuring robust and reproducible findings.
3.	The paper is clearly written, with concise exposition and high readability.

### Weaknesses
1.	The analysis should be supplemented with results on additional architectures such as HAT and other models to verify whether the same properties persist beyond the SwinIR-based cases.
2.	The paper should include a user study to mitigate potential inaccuracies of NR-IQA scores and verify that the observed improvements align with human judgments.
3.	Figure 7 should include additional GAN loss weights (at least six) to better illustrate the effects.

### Questions
N.A.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper conducts systematic experiments to investigate the relationship between perceptual optimization and image quality assessment. Using image-super-resolution with SwinIR as the testbed, this work constructs and evaluates a family of DISTS-style perceptual metrics using various backbones and examines whether discriminator features learned during adversarial training can transfer to IQA tasks. Meanwhile, analyses on the choices of GAN-based initialization and architecture of discriminator are provided.

### Strengths
The study systematically evaluates multiple backbones, discriminator designs, and loss combinations, revealing that good IQA metrics are not necessarily effective for perceptual optimization.

### Weaknesses
1. The experiments are restricted to super-resolution using SwinIR. As this study focuses on revealing the effectiveness of different IQA metrics and discriminator designs, the choice of backbone architecture is necessary and should be considered.
2. Although the asymmetry is clearly demonstrated empirically, there is limited theoretical analysis explaining why high-performing IQA metrics fail in optimization. Meanwhile, the results in Sec 3.4 suggest that Transformer-based discriminators perform poorly and are more unstable. However, it lacks detailed analysis on attention behaviors or feature maps.
3. This work has limited practical guidance. For instance, if IQA metrics and optimization are misaligned, what practical heuristics should be followed when selecting losses? 
4. In Table 2, the extremely low performance of DINOv2 using ImageNet initialization on TQD is not explained.

### Questions
Please refer to the weakness part.

### Soundness
2

### Presentation
2

### Contribution
2
