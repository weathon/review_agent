# Efficiently Disentangling CLIP for Multi-Object Perception

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Vision-language models like CLIP excel at recognizing the single, prominent object in a scene. However, they struggle in complex scenes containing multiple objects. We identify a fundamental reason for this limitation: VLM feature space exhibits excessive mutual feature information (MFI), where the features of one class contain substantial information about other, unrelated classes. This high MFI becomes evident during class-specific queries, as unrelated objects are activated alongside the queried class. To address this limitation, we propose DCLIP, an efficient framework that learns an optimal level of mutual information while adding only minimal learnable parameters to a frozen VLM. DCLIP uses two complementary losses: a novel MFI Loss that regulates class feature similarity to prevent excessive overlap while preserving necessary shared information, and the Asymmetric Loss (ASL) that aligns image features with the disentangled text features. Through this disentanglement, DCLIP reduces excessive inter-class similarity by 30\%. On multi-label recognition, DCLIP performs favorably over SOTA approaches on VOC2007 and COCO-14 while using 75\% fewer training parameters. For zero-shot semantic segmentation, it shows improved performance across six benchmark datasets. These results highlight the importance of feature disentanglement for multi-object perception in VLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DCLIP, a lightweight method to mitigate mutual feature information (MFI) among different object classes in vision-language models such as CLIP.
By introducing a simple MFI loss to decorrelate class-level text features and combining it with an asymmetric loss (ASL) for image-text alignment, the model achieves improved performance on multi-label recognition (MLR) and zero-shot semantic segmentation (ZS3) tasks.

### Strengths
* Performance on multi-object scenes is clearly improved. The proposed MFI suppression leads to more disentangled and precise class activations in complex images.

* The method is conceptually simple and lightweight, with minimal parameter overhead.

* The paper provides a clear explanation of why feature disentanglement benefits multi-label and zero-shot tasks.

### Weaknesses
* Limited multi-label recognition evaluation.
The experiments are restricted to VOC2007 and COCO-14.
For a fair evaluation of generalization in MLR, it would be valuable to include larger and more diverse datasets, such as NUS-WIDE, VG-256, or Open Images.

* Outdated or insufficient segmentation baselines.
The comparison for zero-shot semantic segmentation includes only CLIP-Surgery and CLIP-VV, both of which were published on arXiv in 2023 and accepted in 2025.
However, there have since been multiple self–self attention–based approaches that achieve significantly higher training-free zero-shot segmentation performance such as ClearCLIP and NACLIP.
The current baselines are thus not sufficiently representative of SOTA methods.

* Inconsistency in reported CLIP-VV / CLIP-Surgery results.
According to my understanding, CLIP-VV and CLIP-Surgery are same paper.
The paper, however, reports noticeably different results for them.
If they are indeed the same work, this discrepancy raises questions about experimental consistency and the fairness of comparison.
The authors should clarify this point.

### Questions
* How would DCLIP perform if a prompt tuning method such as DualCoOp were applied on top of it?
In other words, would the proposed MFI loss act as a complementary component to prompt tuning methods, or do they overlap in effect?

### Soundness
2

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
5

### Summary
This paper investigates the weakness of CLIP in understanding scenes with multiple objects and attributes the problem to excessive mutual feature information, meaning that class features are overly entangled. To address this issue, this work introduce DCLIP, a lightweight framework that freezes the original CLIP encoders while adding small projection layers trained with two complementary losses: an MFI loss derived from the Information Bottleneck principle to reduce inter-class correlations, and an Asymmetric Loss to preserve image-text alignment.

Experimental results show that DCLIP significantly decreases inter-class similarity and achieves superior performance in both multi-label recognition and zero-shot semantic segmentation, using far fewer trainable parameters than existing approaches. This paper provides an effective solution to improve the multi-target perception capability of CLIP.

### Strengths
1.The proposed method is simple and easy to understand, yet it achieves strong performance without relying on complex or specialized modules.

2.The experimental evaluation is comprehensive, demonstrating that the approach not only improves multi-label image classification but also generalizes effectively to zero-shot semantic segmentation.

3.The paper is well-written and provides detailed methodological explanations, and implementation settings, which make the approach easy to reproduce.

### Weaknesses
1.In terms of methodology, this paper aims to make text features as orthogonal as possible in the feature space to enhance their discriminability. However, this idea is not entirely new. In addition, the proposed approach is not specifically designed for multi-label problems, as it does not take into account the correlations between different labels.

2. For multi-label image classification, the competing methods are mainly based on CoOp or CLIP, but these approaches are designed for missing-label or zero-shot scenarios. Since this work also uses the CLIP model in a fully supervised setting, it would be reasonable to consider prompt-based fine-tuning of the visual and text encoders, or to incorporate a lightweight image feature decoupling module (such as ADD-GCN, ML-Decoder, or SSGRL, etc). Since the backbone remains frozen, the proposed method might not fully exploit the representational capacity of CLIP, potentially limiting further gains in fine-grained or domain-specific scenarios.

3. The paper acknowledges difficulty in distinguishing visually similar subcategories (e.g., dog breeds or bird species), but does not provide experiments or insights on how the approach might be extended to address this limitation.

### Questions
1. Why does this method choose to separate text features without considering aligning text and image patches?
2. Since the method is not explicitly designed to model label correlations, how might DCLIP handle classes that frequently co-occur in real-world scenes?

3. Given that the method operates in a fully supervised setting, have the authors compared DCLIP with prompt-based fine-tuning approaches (e.g., CoOp, CoCoOp) or lightweight feature decoupling models like ADD-GCN (SAM models ) or ML-Decoder (Decoder models )?

4. See more on Weaknesses

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
This paper investigates why vision-language models like CLIP struggle with complex, multi-object scenes and attributes the problem to excessive mutual feature information (MFI), where features of one class encode redundant information about others. To address this, the authors propose DCLIP, a lightweight framework that disentangles class features by learning an optimal level of mutual information with minimal additional parameters. DCLIP employs an MFI loss to regulate feature similarity and an asymmetric loss to align image and text features.A large number of experiments were conducted  to verify the effectiveness of DCLIP

### Strengths
1.The paper introduces the novel concept of mutual feature information to explain why models like CLIP perform poorly on multi-object perception tasks, which is an interesting and valuable idea.
2.The authors provide a clear explanation of the proposed framework and conduct extensive experiments to support their methodology.

### Weaknesses
1.Is the MFI only present in the text modality? The paper previously points out that there exists feature entanglement in CLIP’s representation space, but the proposed MFI loss is applied only to the text branch. How is disentanglement achieved for the image features? In the paper, it states: “To mitigate this, we project zi and ti into a new disentangled space using learnable projectors (hϕ : hϕ,img and hϕ,text), parameterized by weights ϕ. These projectors map zi and ti from the original space (d-dim) to a new disentangled space (d′-dim).” How should we understand that the image features are also projected into a disentangled space?
2.There is a point of confusion regarding the image-text alignment mechanism. The paper mentions that similarity is computed between patch-level image features and disentangled text features. Taking “a photo of a cat” as an example, each patch may contain only part of the cat. Would it be unreasonable to compute similarity between such local visual information and global textual information? The content of a patch does not represent a complete semantic concept ,it is a local representation, whereas the text feature represents global semantics. What is the theoretical basis for computing similarity between the two? For instance, suppose the cat has pure white fur, and during patch division, one patch contains only the fur, making it nearly a pure white image. When calculating the similarity between such a “pure white” patch and the full semantic representation “a photo of a cat,” the similarity intuitively seems low, even though this patch indeed belongs to the cat.
3.There is a point of confusion regarding the ablation experiments. The core hypothesis of the paper is that mutual feature information (MFI) negatively affects multi-object perception. However, as shown in Table 7, changing the pooling order has a significant impact on the results: on the COCO dataset, keeping the pooling operation yields 81.3 mAP, while removing pooling increases it to 85.6 mAP — an improvement of 4.3 mAP. In contrast, Table 4 shows that removing the MFI loss only decreases performance from 85.4 to 84.2 mAP, a drop of 1.2 mAP. This suggests that the pooling operation has a more substantial effect on accuracy, and the improvement brought by MFI is relatively minor. Moreover, when MFI is applied together with pooling, the performance is notably worse than other methods. Although the paper claims that MFI is the fundamental reason why CLIP performs poorly in multi-object perception, the experimental data indicate that removing global pooling , a known and non-novel operation  leads to a much larger performance gain (4.3 mAP) than the core contribution, MFI Loss (1.2 mAP).
4.There is an issue regarding the baselines used in the ZS3 task. The paper aims to highlight DCLIP’s zero-shot capability, but the number of baselines compared in the unlabeled setting is relatively limited ,only three models are included. It would be better to include more baseline models for comparison, or at least provide an explanation if additional comparisons are not feasible.

### Questions
1.Clarify whether MFI exists in both modalities and elaborate on how image-side disentanglement is achieved beyond applying MFI loss only to text.
2.Provide a clearer theoretical justification or empirical validation for computing similarity between local (patch-level) and global (text-level) features.
3.Revisit the ablation results to better isolate the contribution of MFI loss from other factors like pooling, and discuss why pooling affects performance more strongly.
4.Include more zero-shot baselines in the ZS3 evaluation or justify the limited comparison to strengthen the evidence for DCLIP’s zero-shot capability.
Attention：The conference requires a section in the appendix describing the use of LLMs, but I couldn’t find it in either the main text or the appendix.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the root cause of high inter-class feature entanglement in vision-language models (VLMs) like CLIP, particularly in multi-object scenarios. The authors propose DCLIP, a framework that freezes the original VLM and introduces lightweight image and text projectors. The core idea is to enforce orthogonality among class text embeddings using a Mutual Feature Information (MFI) Loss on the text side, and to align local visual features with these "disentangled" text features using Asymmetric-Similarity Loss (ASL) on the image side. The proposed framework achieves state-of-the-art or comparable performance on multi-label recognition (VOC2007, COCO-14) and zero-shot semantic segmentation (on six datasets). Overall, this paper presents a unified and computationally efficient solution for multi-object perception. The motivation is clear and the experiments are thorough, making it a solid contribution worthy of publication.

**Strengths**
1、The motivation is well-established. The authors clearly demonstrate the "cross-class activation" problem in CLIP by visualizing and calculating the pairwise cosine similarities of class text embeddings on VOC and COCO.
2、The method is efficient. By only training small projectors, the framework avoids expensive segmentation annotations or large-scale fine-tuning, resulting in low computational and parameter overhead.
3、The experimental evaluation is comprehensive, with systematic comparisons on both MLR and ZS3 tasks.
**Weaknesses**
1、There appears to be a leap in the mathematical reasoning in Section 3.3. The paper equates minimizing mutual information (MI) with optimizing the diagonal/off-diagonal elements of a correlation matrix under a Gaussian assumption. It then discards the diagonal terms by claiming that vector normalization implies an identity covariance matrix (Σ = I). This step, equating a unit L2-norm with unit variance and zero covariance for all dimensions, is not strictly valid. The authors should provide a more rigorous derivation, perhaps by connecting the loss to an upper or lower bound of mutual information.
2、The ZS3 evaluation setting could be described more clearly. The projectors are trained on COCO-14 (using image-level labels) and then used for inference on segmentation datasets. This is more accurately described as weakly-supervised transfer rather than a purely zero-shot setting. This should be clarified in the main text.
3、The paper reports the mean mAP over 5 runs in Table 1 but lacks standard deviations or confidence intervals. The reported improvement of DCLIP over DualCoOp++ on COCO-14 is only +0.5 mAP (85.6 vs. 85.1), which may not be statistically significant and could fall within random fluctuations.
4、Aggressively enforcing low inter-class similarity could disrupt the inherent semantic hierarchy between classes (e.g., different breeds of dogs or birds). The authors briefly mention this limitation but lack a quantitative analysis of this potential side effect. A supplementary analysis would strengthen the paper.
5、The terms "mutual information minimization" and "decorrelation" are used interchangeably throughout the text. It would be better to maintain consistent terminology.
6、The caption for Figure 3 ("Fixed Dual Prompts / Introduced in Sec 3.4") is not self-contained. It would be more helpful to explicitly state the template (e.g., "A photo of a {classname}") and explain the meaning of the positive/negative prompts directly in the caption.

### Strengths
This paper investigates the root cause of high inter-class feature entanglement in vision-language models (VLMs) like CLIP, particularly in multi-object scenarios. The authors propose DCLIP, a framework that freezes the original VLM and introduces lightweight image and text projectors. The core idea is to enforce orthogonality among class text embeddings using a Mutual Feature Information (MFI) Loss on the text side, and to align local visual features with these "disentangled" text features using Asymmetric-Similarity Loss (ASL) on the image side. The proposed framework achieves state-of-the-art or comparable performance on multi-label recognition (VOC2007, COCO-14) and zero-shot semantic segmentation (on six datasets). Overall, this paper presents a unified and computationally efficient solution for multi-object perception. The motivation is clear and the experiments are thorough, making it a solid contribution worthy of publication.

1、The motivation is well-established. The authors clearly demonstrate the "cross-class activation" problem in CLIP by visualizing and calculating the pairwise cosine similarities of class text embeddings on VOC and COCO.
2、The method is efficient. By only training small projectors, the framework avoids expensive segmentation annotations or large-scale fine-tuning, resulting in low computational and parameter overhead.
3、The experimental evaluation is comprehensive, with systematic comparisons on both MLR and ZS3 tasks.

### Weaknesses
1、The equivalence jump from Eq (1)→(2) rests on a questionable assumption. The paper implies that vector normalization leads to an identity covariance matrix (Σ=I), which allows diagonal terms to be treated as constants. This leap from a unit L2-norm to unit variance & zero covariance is not strictly valid, and the provided "Gaussian + BN" justification is insufficient to bridge this gap.

2、The core mechanism—that text-side orthogonalization improves image-side alignment—lacks theoretical grounding. The argument currently relies on empirical evidence (Fig. 7, Table 4). To make a stronger connection to the paper's IB motivation, a brief formal argument, perhaps via an upper-bound analysis, would be beneficial.

3、The ZS3 evaluation setting could be described more clearly. The projectors are trained on COCO-14 (using image-level labels) and then used for inference on segmentation datasets. This is more accurately described as weakly-supervised transfer rather than a purely zero-shot setting. This should be clarified in the main text.

### Questions
1、The paper reports the mean mAP over 5 runs in Table 1 but lacks standard deviations or confidence intervals. The reported improvement of DCLIP over DualCoOp++ on COCO-14 is only +0.5 mAP (85.6 vs. 85.1), which may not be statistically significant and could fall within random fluctuations.
2、Aggressively enforcing low inter-class similarity could disrupt the inherent semantic hierarchy between classes (e.g., different breeds of dogs or birds). The authors briefly mention this limitation but lack a quantitative analysis of this potential side effect. A supplementary analysis would strengthen the paper.
3、The terms "mutual information minimization" and "decorrelation" are used interchangeably throughout the text. It would be better to maintain consistent terminology.
4、The caption for Figure 3 ("Fixed Dual Prompts / Introduced in Sec 3.4") is not self-contained. It would be more helpful to explicitly state the template (e.g., "A photo of a {classname}") and explain the meaning of the positive/negative prompts directly in the caption.

### Soundness
3

### Presentation
2

### Contribution
2
