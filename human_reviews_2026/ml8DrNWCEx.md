# Detecting Generated Images via Machine Unlearning

- Decision: Reject
- Scores: 4, 2, 8

## Abstract
Robust detection of generated images is critical to counter the misuse of generative models. Existing methods primarily depend on learning from human-annotated training datasets, limiting their generalization to unseen distributions. In contrast, large-scale vision models (LVMs) pre-trained on web-scale datasets exhibit exceptional generalization power through exposure to diverse distributions, offering a transformative paradigm for this task. However, our experimental results reveal that LVMs pre-trained exclusively on natural images effectively capture the features of both natural and generated images to achieve comparably low loss, thereby failing to distinguish both types of images. This prompts a key question: *When and how do LVMs exhibit different behaviors when capturing features of natural and generated images?* This investigation reveals an insight: during unlearning, LVMs exhibit disparate forgetting dynamics with feature degradation for generated images escalating faster than natural ones. Inspired by the disparate dynamics, we introduce two detection methods: 1) data-free detection, which prunes model parameters to induce unlearning without data access, and 2) data-driven detection, which optimizes LVMs to unlearn knowledge tied to generated images. Extensive experiments conducted on various benchmarks demonstrate that our unlearning-based approach outperforms conventional detection methods. By recasting the detection task as a problem of machine unlearning, our work establishes a new paradigm for generated image detection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper reframes synthetic-image detection as machine unlearning: selectively degrading a pre-trained vision model so that generated-image features fade faster than natural ones. A data-free detector prunes 90\% of the smallest weights and flags images whose cosine similarity between original and pruned features drops below a threshold; a data-driven variant enlarges the gap with LoRA fine-tuning. Across ImageNet, GenImage, DiffusionForensics, LSUN, Chameleon and unseen Sora clips the methods exceed 98\% AUROC, and stay stable under JPEG compression, blur and noise.

### Strengths
1. Novel paradigm: a new method that casts synthetic-image detection as machine unlearning, turning a limitation of pruning into a discriminative signal.

2. Strong generalization: excellent accuracy across diverse GAN, diffusion and autoregressive models, including black-box Sora footage, without retraining.

3. Lightweight and data-efficient: the data-free version needs no generated samples or GPU training.

### Weaknesses
1. Although the paper proposes a novel approach, a critical issue arises with the authors' central claim that Weight Pruning has a greater impact on OOD synthetic images. However, images from modern generative models (e.g., the Chameleon Dataset) are inherently OOD, yet the model's performance on these new generators degrades significantly (as shown in Table 2). This observation appears to contradict the stated insight and, to some extent, calls this claim into question. Further explanation from the authors is required.

2. Data-free unlearning still underperforms most training-based methods. Meanwhile, the data-driven approach essentially repurposes the model delta as a feature extractor. This falls short of the unlearning capability claimed by the authors. Its utility would be greater if it were explicitly employed as a feature extractor with traditional learning-based methods.

3. Ambiguous dataset names: ImageNet and LSUN, as they are real datasets which is not valid for synthetic image detection. Their precise constitution only becomes clear after reading the appendix. To prevent confusion, it would be beneficial to adopt names that directly reference the source. 

4. Some spelling mistakes: huors in Table 11, IMAGENNET in line 961. And incorrect citation format in line 59, line 79, and line 104.

### Questions
As described in Weaknesses above. I am deeply concerned about the theoretical soundness of the paper and the interpretability of results. Nevertheless, the methodological contribution is striking; after thorough discussion I am willing to raise my currently conservative score.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper reveals that LVMs pre-trained exclusively on natural images capture features of both natural and generated images, resulting in comparably low loss and thus failing to distinguish between the two. To address this, the authors employ unlearning to induce different behaviors in LVMs when capturing features of natural versus generated images. Extensive experiments demonstrate the effectiveness of the unlearning-based approach.

### Strengths
1. The motivation of the paper is appreciated, particularly the effort to adjust LVMs to make them better suited for the AIGI detection task.

2. It’s encouraging to see the method evaluated on the in-the-wild benchmark, Chameleon, achieving a 71% accuracy. However, there are concerns about the fairness of the comparisons.

### Weaknesses
1. Comparison Concerns: The comparison presented in the paper appears to be somewhat unfair. **The authors have intentionally placed non-output performance comparison tables in the appendix, while benchmarks like ImageNet (2009) and LSUN-BEDROOM (2015) are included in the main paper. Since GenImage (2023) and DRCT-2M (2024) are more recent and popular benchmarks that include advanced generative models, I recommend that these comparisons be moved to the main body of the paper.** What is even more concerning is that **some outperformed baselines are deliberately omitted from these benchmark comparison tables.** For instance, in Table 12 (comparison on DRCT-2M), the average performance of DRCT is 91.35 (which is equivalent to DRCT/UnivFD + SDv2), but I found that DRCT/UnivFD + SDv2 actually achieves an average accuracy of 96.55, which is higher than the unlearning-based method's 94.50. In Table 13, the comparison with SAFE [1], which achieves an accuracy of 95.6 on GenImage (much higher than the unlearning method's 89.7), is also missing.

2. Concerns on the Unlearning: The paper would benefit from a more detailed discussion comparing unlearning with other learning strategies, such as LoRA, full fine-tuning, and linear probing. Specifically, it would be helpful to highlight the computational overhead and the performance trade-offs associated with each approach. This comparison would provide a clearer justification for choosing unlearning over more direct fine-tuning methods.

[1] Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspective. KDD 2025.

### Questions
1. **Fairer Comparisons**: The paper would benefit from including more recent baselines, such as SAFE [1], AlignedForensics [2], BFree [3], and Co-SPY [4], to provide a more balanced comparison.

2. **Broader Comparison Benchmarks**: It would also be helpful to include a wider range of benchmarks, such as ForenSynths [5] and AIGCDetectionBenchmark [6], to further validate the method's performance.

3. **Computation-Performance Trade-Off**: A discussion on the computational performance trade-offs of the proposed method, particularly in comparison to other strategies like LoRA or full fine-tuning, would add valuable context and help readers understand the practical implications of the approach.

[1] SAFE: Aligned datasets improve detection of latent diffusion-generated images, ICLR 2025.

[2] AlignedForensics: Aligned datasets improve detection of latent diffusion-generated images, ICLR 2025.

[3] BFree: A bias-free training paradigm for more general AI-generated image detection, CVPR 2025.

[4] Co-SPY: Combining semantic and pixel features to detect synthetic images by AI, CVPR 2025.

[5] ForenSynths: CNN-generated images are surprisingly easy to spot... for now, CVPR 2020.

[6] AIGCDetectionBenchmark: Patchcraft: Exploring texture patches for efficient AI-generated image detection, arXiv 2023.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a novel method for detecting AI-generated images by unlearning knowledge of generated content in large pre-trained vision models. Standard vision models trained only on natural images yield similarly low loss on natural and generated images, but when the model is degraded using weight pruning, its features for generated images change much faster than those for natural images. The authors develop two detection schemes using this observation, a data-free method that simply prunes model weights with no training data needed, and a data-driven method that fine-tunes the model via parameter-efficient LoRA adapters. Detection is done by comparing feature similarity between the original and unlearned model where generated images cause a large shift in feature space. The authors conduct experiments across many benchmarks and unseen models and show that the unlearning-based approach outperforms prior detectors

### Strengths
1. The novelty contribution is substantial. The authors proposed a method that leverages model unlearning as a method to detect synthetic images, which to my knowledge, is a novel utilization of model unlearning. Furthermore, the concept is based on both theoretical and empirical justification. This concept opens brand new ways of doing content authentication.

2. The paper tests on many datasets and generative architectures to show a comprehensive evaluation and ablation study. 

3. The data-free variant requires no additional labeled data or training, which could be attractive in practice.

### Weaknesses
1. The approach assumes generated images are effectively out-of-distribution relative to the model’s training.  However, as synthetic data is becoming more common for large-scale model pretraining, this might not be the case in the future. This may cause limitations for future implementations (this is already noted in Limitations section). 

2. The method requires a threshold to be selected, which is currently done based on set of generated and natural images. However, this kind of threshold selection isn't robust and can vary widely depending on the image set and type. Furthermore, the sensitivity of model performance to the threshold can also be analyzed. Figure 7 also shows cosine similarity distribution when comparing feature similarity, which shows considerable overlap between the natural and AI-generated images. Thus, this method looks like it will be highly sensitive and output more false-positives and false-negatives. 

3. There is an increase in inference time with respect to existing methods, as this requires multiple forward passes.

### Questions
1) Did the authors do experiments that specifically look at model performance when that model is pretrained with generated synthetic image? 

2) How similar are the generated and original images used to show detection performance? If a generated image is generated using original image and using some inpainting techniques, I would imagine the performance will deteriorate. Did the authors do any experiments regarding this? 

3) Did the authors do any experiments showing the sensitivity of selecting the threshold?

### Soundness
3

### Presentation
3

### Contribution
4
