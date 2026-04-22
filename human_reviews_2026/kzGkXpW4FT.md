# Identifying Robust Neural Pathways: Few-Shot Adversarial Mask Tuning for Vision-Language Models

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 2

## Abstract
Recent vision-language models (VLMs), such as CLIP, have demonstrated remarkable transferability across a wide range of downstream tasks by effectively leveraging the joint text-image embedding space, even with only a few data samples. Despite their impressive performance, these models remain vulnerable to adversarial attacks, raising significant concerns about their security and reliability in practical deployments. To address this issue, we propose Adversarial Mask Tuning (AdvMask), a method that effectively enhances the robustness of VLMs without directly modifying their pre-trained weights. Instead, our AdvMask learns a set of binary masks that selectively deactivate model parameters vulnerable to adversarial perturbations. By identifying robust neural pathways within the vision encoder, AdvMask facilitates the generation of features and predictions that are resistant to adversarial attacks. Furthermore, we introduce a Layer-wise Adaptive Feature Alignment (LAFA) loss, specifically designed to optimize AdvMask in few-shot scenarios. The LAFA loss adaptively aligns intermediate-layer features from clean and adversarial samples across each transformer block, enhancing the representational robustness of the model. Experimental results across multiple benchmarks confirm that our AdvMask approach substantially outperforms existing adversarial tuning techniques for VLMs, especially in few-shot settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces AdvMask to improve the adversarial robustness of in few-shot settings. Instead of fine-tuning model weights, it learns binary masks to identify and activate robust neural pathways, preserving stable features under attack. The proposed LAFA loss further enhances robustness by aligning intermediate features between clean and adversarial samples.

### Strengths
1. The setting of the task of this work is clear. The convert from Mask Tuning to Adv Mask Tuning is interesting and theoretical.

2. The method framework is clearly introduced, while using visualization to help readers quickly understand the method.

3. The paper provides sufficient experimental evidence and further insight.

### Weaknesses
1. This work involves two-shot scenarios: few-shot training and zero-shot evaluation. The authors are advised to clearly explain and distinguish these in the introduction to facilitate understanding. For example, the phrase "overfitting in a few-shot setting" on line 54 and "zero-shot robustness" on line 55 may not be aligned settings, but their use together could easily lead to misunderstanding and confusion.

2. In the ablations Table 17, Table 18, etc., the 16-shot performances of "47.1" and "47.3" seem to be different from the "41.99" given in Table 10. How are they obtained?

3. If the author's training data uses 3.2% of the data, then is the training time also reduced to 3.2% compared to TGA-ZSR? I didn't see a direct comparison in the paper.

### Questions
1. Are there any visualization or statistical results that show the specific situation of the final mask, and can we summarize in a regular way which weights are more important for adversarial and which need to be ignored?

2. I'm curious if this few-shot approach would be applicable to the scenarios tuned for LVLM in FARE? Are there any challenges in doing so?

[1] Schlarmann C, Singh N D, Croce F, et al. Robust clip: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models[J]. arXiv preprint arXiv:2402.12336, 2024.

### Soundness
3

### Presentation
4

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
The paper investigates an interesting adversarial defense mechanism without explicitly tuning the model weights; instead, the proposed method optimizes a model weight-level mask across different layers to enhance robustness. In other words, the proposed method shields potentially vulnerable weights to adversarial robustness. Furthermore, the paper introduced a layer-wise adaptive feature alignment scheme to optimize such a model weight-level mask by aligning clean features and their adversarial counterparts in the few-shot setup. Experiments across diverse datasets and scenarios demonstrate the generalization capability of the proposed method. Further ablations justifies the efficacy of the design of the proposed pathway method.

### Strengths
1. The proposed idea is interesting. Instead of tuning the whole weight map of VLMs to improve robustness, the paper explores an alternative way by exploring the weight-level mask to remove some implicitly vulnerable model weights against adversarial attacks.
2. The paper is well-written and organized. A detailed recap of previous works and a background introduction are given.
3. Extensive experiments across diverse benchmarks and scenarios are provided. In addition, a series of ablation analyses is given to verify the effectiveness of each module. Insights in Figure 4 are also interesting.

### Weaknesses
1. The proposed method might not be novel in the context of adversarial learning (for both single-modal and multimodal architectures). [a] has already explored the model weights connected with adversarial robustness. Although [a] is based on a single-modal architecture, its idea can also be easily extended to a multimodal backbone.

2. The evaluated adversarial attacks are primarily low-intensity (low perturbation radius) attacks with eps=1/255. It's questionable that if the proposed method also exhibits robustness against stronger adversarial attacks with higher eps (e.g., 4/255, 8/255)

3. Can the mask be regarded as part of the weights of VLMs? If so, I think that finetuning the VLM weights can also achieve the same effect, the mask would be mostly like some scailing of the standard adversarial finetuning, which means the proposed mask is an indirect adversarial finetuning. Then, it should achieve similar performance compared with adversarial finetunin.

4. It seems that the mask is only for the image encoder. But the paper focuses on VLMs. In this case, it would be more appropriate to consider both branches, otherwise the work would be mostly similar to single-modal adversarial learning works.

[a] Improving Generalization of Adversarial Training via Robust Critical Fine-Tuning (ICCV 2023)

### Questions
1. Can the authors evaluate the text-level (BERT-Attack) or joint image-text-level attacks (CO-Attack) in addition to image-level attacks only?

2. In addition to image classification, VLMs are powerful in diverse visual-language tasks, e.g., image captioning, Visual question answering. Can the authors test some of them instead of classification?

3. Can the proposed mask be a soft format instead of the 0-1 style?

[b] BERT-ATTACK: Adversarial Attack Against BERT Using BERT (EMNLP-2020) 

[c] Towards Adversarial Attack on Vision-Language Pre-training Models (ACMMM 2022)

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes AdvMask, a method to enhance adversarial robustness of Vision-Language Models (VLMs) in few-shot settings. Instead of modifying pre-trained weights or learning prompts, AdvMask learns binary masks that selectively deactivate parameters vulnerable to adversarial perturbations, effectively identifying robust neural pathways within the vision encoder. The authors introduce a new loss (LAFA) that aligns intermediate features between clean and adversarial samples with confidence-based weighting. Experiments on 11 datasets show AdvMask outperforms prompt-based baselines (AdvVP, AdvVLP, AdvMaPLe, FAP) in few-shot adversarial robustness while maintaining competitive clean accuracy.

### Strengths
. The concept of finding robust neural pathways via binary masks is creative and well-motivated.

.The framing as deactivating vulnerable parameters rather than adding robust features is a fresh perspective on adversarial defense.

. The experimental evaluation is thorough.

. Substantial improvements over baselines across most datasets. 

.Clean accuracy recovers with more shots.

### Weaknesses
. The model is trained with only 2 steps of PGD at a very small noise level (ε = 1/255) and tested mostly at the same level. This is weak to prove robustness and may hide gradient masking. The authors should test with stronger attacks (for example ε = 4/255) to make sure the binary mask and straight-through estimator do not block gradients.

. The adaptive weight in LAFA uses the model’s own prediction confidence. At early stages, this confidence can be wrong, which may make the model ignore “hard but useful” samples. The authors could try using a teacher model or stop the gradient from this weight to prevent bias.

. The paper claims that certain layers or heads are more robust, but there is no clear visualization. It would help to show which layers or attention heads are most often masked and whether this pattern is consistent across datasets or random seeds.

. The paper argues that full fine-tuning overfits in few-shot cases, but this is not shown. A simple 16-shot full fine-tuning baseline would make this claim stronger.

. The paper says the method is efficient, but Table 19 shows higher inference memory than some baselines. A simple memory breakdown would help clarify this.

. Modern papers show that some defenses look strong until the attack is adapted to the defense itself. Here, the main defensive component is the mask and LAFA, so it would be good to test with attacks that target them directly.

### Questions
1. Can you show a baseline where the model is fully fine-tuned with 16 samples per class?

2. Which layers or heads are masked the most? Are these patterns stable across runs?

3. Why did you choose the mask initialization values?

4. Where does the method fail? For example, why is performance lower on Cars, Food101, and Aircraft datasets?

5. Why is the inference memory higher than other lightweight methods?

6. How does sparsity (number of masked weights) relate to robustness?

7. Can you add stronger attacks (ε = 4/255)?

8. Have you tested an adaptive attack that specifically targets the mask or LAFA?


---- Additional Suggestions

. Add a baseline where the mask is trained without adversarial samples to see if adversarial tuning is truly necessary.

. Show how the mask changes during training (sparsity per epoch).

. Include a small table showing scaling to a larger backbone such as ViT-L/14.

. Report per-class accuracy to show which categories benefit the most.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes an adversarial mask tuning (AdvMask) approach that searches for robust subnetwork within well-trained VLMs as a promising alternative which is highly parameter-efficient and is trained with Layer-wise Adaptive Feature Alignment (LAFA) loss. Experiments across various downstream datasets demonstrate that AdvMask consistently improves few-shot adversarial robustness over existing baselines.

### Strengths
The paper addresses an important problem of improving adversarial robustness in vision-language models (VLMs) with few-shot learning. The motivation is well-explained, and the paper is well-organized and clearly written. In figure 2, the authors provide a clear illustration of the proposed AdvMask method with different shots, which helps a lot in understanding the performance of the approach.

### Weaknesses
1. The proposed method is strongly related to adversarial model pruning, which has been extensively studied in the literature (arxiv.org/pdf/2409.01249). Therefore, the novelty of the proposed method is limited. The authors should also consider comparing with these adversarial pruning methods.

2. The improvement of the proposed method is not significant enough. From Table 4, adding AdvMask only improves around 0.7% robust accuracy and 1.5 % clean accuracy compared to directly using typical adversarial training.

3. The experimental settings are not strong enough. The authors should evaluate the performance of the proposed method with more adversarial settings, e.g., epsilon=2/155 or 4/255. See questions.

### Questions
1. Why is this method particularly effective for few-shot learning? What if using more training data?

2. Since the model parameters are changed after masking, does the author adopt adaptive attacks (e.g., PGD attack with knowledge of the mask) during evaluation? If not, the evaluation is not fair.

3. Can this method be better than existing zero-shot robust methods? e.g., simply adversarially train the entire model with one specific dataset like ImageNet?

4. Does this mask have any interpretability? For example, are they similar for different datasets?

### Soundness
3

### Presentation
3

### Contribution
2
