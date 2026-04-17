# Beyond Hard Supervised Fine-tuning: Enhancing Image-text Alignment of Strong Models with Weak Models

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Image–text alignment models, such as CLIP, are typically trained with large-scale contrastive learning: paired data are treated as positives, while all unpaired pairs are treated as negatives.
However, this hard supervision overlooks the fact that some unpaired pairs are semantically related rather than irrelevant, and penalising them as strict negatives introduces noise that limits model performance.
We propose Permute-then-Adapt (PTA), a weak-to-strong supervision framework that addresses this issue.
PTA comprises two key innovations: (1) a permutation-based thresholding that identifies and filters unreliable negatives by estimating a null distribution of similarities, and (2) a soft supervision strategy that leverages above-threshold similarities to provide extra training signals. Across benchmarks, PTA consistently improves the alignment ability of strong models on object recognition and cross-modal retrieval.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the problem of "false negatives" in CLIP. In standard training, all unpaired image-text combinations in a batch are treated as negatives, but many are semantically related, leading to noisy supervision that hinders model performance. The authors propose a "weak-to-strong" supervision framework. PTA uses a weaker model to guide the fine-tuning of a stronger model, with the process of Permute-then-Adapt. The method is evaluated on zero-shot classification, and two cross-modal retrieval tasks, demonstrating improvements over baselines.

### Strengths
- I think the idea of using permutation test to solve FP problem is novel. This is a improvement of traditional hard supervised fine-tuning. Moving beyond fixed, heuristic thresholds to a statistically adaptive one is a significant conceptual leap. The "weak-to-strong" framing is aptly applied to the multimodal domain.

- Strong empirical study: The paper provides thorough analysis of false negatives in CLIP, which is quite solid. 

- Good writing. The paper is quite easy to follow, which contains essential background for CLIP.

### Weaknesses
- Limited literature review: There exists quite a few works tackling false negatives in CLIP, including but not limited to [1], [2], [3]. More baselines should be included in your literature review.

- Limited baseline comparison. As mentioned above, more baselines should be evaluated to improve the completeness of this work.

- The pre-requisite for this method is laborious. You must shuffle datasets for many iterations before obtaining the distribution.

- The choice of $\alpha$ really matters in scalable contrastive learning. While the authors simply evaluate on COCO and F30K, more diverse datasets should be included to prove the generalization ability.

- Computational overhead: While the authors note a ~1.3x slowdown (Figure 6), this is a non-trivial cost, especially when considering large-scale pre-training. The requirement to compute the weak model's similarity for all in-batch negative pairs for every training step (to compute L_PTA_batch) adds a persistent overhead. A more detailed discussion on the scalability of this approach and potential optimizations (e.g., caching, using a much smaller proxy model as the "weak" model) would be beneficial.


- Sensitivity to weak model quality: The method's success hinges on the weak model providing a meaningful similarity distribution. The ablation in A.5.2 shows robustness between B/32 and B/16, but what happens if the weak model is very weak or poorly calibrated? A systematic analysis of how the performance gap and quality of the weak model affect the strong model's final performance would provide valuable insights into the limits of this approach.

- Clarity on loss formulation: The loss functions in Equations (12) and (13) could be clarified. The denominator sums over all B items in the batch, but the numerator only sums over those where l_km^PTA = 1. This seems to encourage the model to assign high probability to any of the soft positives, which is a form of multiple-positive learning. This is a valid approach, but the connection to and difference from other multiple-positive losses could be discussed more explicitly.



[1]. Mitigating Noisy Correspondence by Geometrical Structure Consistency Learning, CVPR 2024.


[2]. Discovering Clone Negatives via Adaptive Contrastive Learning for Image-text Matching, ICLR 2025.


[3]. Unlearning the Noisy Correspondence Makes CLIP More Robust, ICCV 2025.

### Questions
As mentioned in `Weaknesses`. I am looking forward to the authors' rebuttal, experiments, and the updated manuscript.

### Soundness
2

### Presentation
3

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
This paper proposes a weak-to-strong supervision framework to address the issue of false negatives in CLIP pretraining, which identifies and filters unreliable negatives based on permutation-based thresholding. Experiments on downstream tasks demonstrate the effectiveness of proposed method.

### Strengths
1. This paper proposes a weak-to-strong supervision framework to tackle the problem of false negatives in CLIP, which can filter unreliable negatives by estimating a null distribution of similarities through permutation-based thresholding and use weak models to provide training signals to guide strong models.
2. Experiments on downstream tasks including zero-shot classification demonstrate that proposed method can consistently improve the alignment ability of strong models.

### Weaknesses
1. While the paper dedicates significant space to its motivation in Sec. 3.1 (the false-negative problem in CLIP), this issue has been previously identified and explored by several existing works, such as PyramidCLIP [1], SoftCLIP [2], and CLIP-PSD [3]. The authors should properly include and discuss these relevant studies. Additionally, the "Related Work" section, currently placed in the appendix, should be integrated into the main body of the paper to ensure the completeness.
2. The fine-tuning dataset, COCO, has a limited data size. To further validate the scalability of the proposed method, it would be beneficial to include larger-scale public datasets such as CC3M [4], CC12M [5], YFCC [6], or LAION [7]. 
3. To provide a more comprehensive evaluation, it is recommended to include quantitative comparisons on the linear probing task. Furthermore, the addition of qualitative results—such as t-SNE visualizations, Grad-CAM heatmaps, or retrieval examples—would offer valuable intuitive insights into the model's behavior and strengths.
4. Regarding Eq. (15), what about the results when $\lambda >1$ ?

[1] Y. Gao, J. Liu, Z. Xu, J. Zhang, K. Li, and C. Shen. Pyramidclip: Hierarchical feature alignment for vision-language model pretraining. In NeurIPS, 2022.

[2] Y. Gao, J. Liu, Z. Xu, T. Wu, E. Zhang, W. Liu, J. Yang, K. Li, and X. Sun. SoftCLIP: Softer Cross-modal Alignment Makes CLIP Stronger. In AAAI, 2024.

[3] A. Andonian, S. Chen, and R. Hamid. Robust cross-modal representation learning with progressive selfdistillation. In CVPR, 2022.

[4] P. Sharma, N. Ding, S. Goodman, and R. Soricut. Conceptual captions: A cleaned, hypernymed, image alttext dataset for automatic image captioning. In ACL, 2018.

[5] S. Changpinyo, P. Sharma, N. Ding, and R. Soricut. Conceptual 12m: Pushing web-scale image-text
pre-training to recognize long-tail visual concepts. In CVPR, 2021.

[6] B. Thomee, D. A. Shamma, G. Friedland, B. Elizalde, K. Ni, D. Poland, D. Borth, and L.-J. Li. Yfcc100m:
The new data in multimedia research. Communications of the ACM, 2016.

[7] C. Schuhmann, R. Vencu, R. Beaumont, R. Kaczmarczyk, C. Mullis, A. Katta, T. Coombes, J. Jitsev, and
A. Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. In NeurIPS Workshop, 2021.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

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
This paper proposes a weak-to-strong supervision framework called Permute-then-Adapt (PTA) to improve image–text alignment in contrastive models such as CLIP. The key motivation is that standard contrastive learning treats all unpaired samples as negatives, introducing false negatives when semantically related pairs are penalized. PTA uses a weak model to estimate semantic relatedness and employs a permutation-based thresholding mechanism to statistically determine which unpaired pairs are likely semantically relevant. These pairs are then used as soft positives to fine-tune a strong model. The method enriches binary supervision with soft, statistically grounded signals. Experiments on multiple benchmarks (COCO, Flickr30K, CIFAR, ImageNet, EuroSAT, etc.) show moderate improvements in both zero-shot classification and cross-modal retrieval tasks, with controlled robustness to batch size, α-level, and λ-weighting.

### Strengths
- The introduction of permutation-based thresholding provides a statistically grounded way to separate noise from meaningful weak signals, improving over heuristic thresholding.
- Experiments are broad, covering several datasets and extensive ablations on α, λ, and fine-tuning depth.

### Weaknesses
- The methodological novelty is relatively limited. The framework essentially combines soft-label supervision and weak-to-strong transfer with a permutation-derived threshold; the conceptual and empirical increment over prior soft alignment or pseudo-labeling methods is minor.
- The issue of false negatives in contrastive learning has already been widely recognized and addressed in many recent works; therefore, the empirical analysis in Section 3.1 mainly reiterates an established observation rather than offering new insights. More critically, the paper does not sufficiently articulate how its proposed PTA framework fundamentally differs from, or improves upon, prior baselines that also tackle the same problem through soft-labeling, pseudo-labeling, or adaptive thresholding.
- Gains may largely stem from controlled partial fine-tuning (FT4) rather than the permutation mechanism itself.

### Questions
- How does the proposed permutation-based threshold compare to simpler percentile-based or temperature-calibrated thresholds under identical fine-tuning settings? Is the improvement statistically significant?
- How sensitive is PTA to the domain and quality of the weak model? Would it still work if the weak model is trained on a different distribution (e.g., non-COCO data)?

### Soundness
2

### Presentation
2

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
This paper introduces Diffusion Dataset Condensation (D2C), a framework of dataset distillation for diffusion models. The key idea is to construct a small yet information-rich synthetic sub-dataset that enables high-quality diffusion model training with only a fraction of the original data. D2C contains two phases: Select phase and Attach phase. The experiments show that only using 0.8% data the diffusion model can be trained from scratch and achieve a good performance.

### Strengths
1.  This is the first paper to formally study dataset condensation for diffusion models, whereas prior works (e.g., SRe2L, MTT, CAFE) targeted discriminative tasks like classification.

  2.  D2C achieves up to 233× faster training using only 0.8% of ImageNet data, while maintaining competitive FID (e.g., 4.3 at 40k steps).

### Weaknesses
1.  The proposed benchmark offers a controlled setting for studying weak supervision, but its novelty is limited since it builds on existing datasets rather than introducing new data. The weak signals derived from CLIP-B/32 may not reflect real-world noisy supervision, reducing external validity. Moreover, the benchmark and PTA framework are co-designed, which introduces potential self-validation bias and limits generalizability.

  2. The assumption here is that weak-model similarity scores adequately represent real-world weak supervision. In practice, models like CLIP-B/32 are already well-trained and produce relatively clean signals, which differ from the noisy, inconsistent annotations typically found in large-scale web data. As a result, the simulated weak supervision in this benchmark may underestimate the complexity and noise of real-world multimodal datasets, limiting the ecological validity of the findings.

 3. The experiments focus mainly on CLIP-style models (ViT-B/16 and ViT-B/32). It remains unclear whether the same improvements hold for other architectures or multimodal models (e.g., BLIP, ALIGN, or Qwen-VL), limiting generality. 

 4. The PTA framework requires an additional forward pass of the weak model for every batch, leading to extra computational cost and potentially reduced scalability for large datasets. Although statistically consistent, the reported performance gains (+1–2%) are relatively modest, raising questions about the practical significance of the improvement versus the added training cost (≈1.3–1.4× slowdown).

### Questions
1. How does your benchmark differ fundamentally from existing datasets like COCO or Flickr30k beyond the weak-supervision setup? 

2. Since the PTA framework and benchmark were co-designed, how do you ensure that the benchmark does not inherently favor your method over alternative weak supervision approaches?

The rest was stated in the weakness part.

### Soundness
2

### Presentation
3

### Contribution
2
