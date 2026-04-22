# Prompt Optimization Meets Subspace Representation Learning for Few-shot Out-of-Distribution Detection

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
The reliability of artificial intelligence (AI) systems in open-world settings depends heavily on their ability to flag out-of-distribution (OOD) inputs unseen during training. Recent advances in large-scale vision-language models (VLMs) have enabled promising few-shot OOD detection frameworks using only a handful of in-distribution (ID) samples. However, existing prompt learning-based OOD methods rely solely on softmax probabilities, overlooking the rich discriminative potential of the feature embeddings learned by VLMs trained on millions of samples. To address this limitation, we propose a novel context optimization (CoOp)-based framework that integrates subspace representation learning with prompt tuning. Our approach improves ID-OOD separability by projecting the ID features into a subspace spanned by prompt vectors, while projecting ID-irrelevant features into an orthogonal null space. To train such OOD detection framework, we design an easy-to-handle end-to-end learning criterion that ensures strong OOD detection performance as well as high ID classification accuracy. Experiments on real-world datasets showcase the effectiveness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper tackles the problem of few-shot semantic OOD detection. It extends the LoCoOP framework with an additional loss term $\mathcal{L}_\text{sub}$. Specifically, after obtaining the pseudo-labels of local features obtained by CLIP embeddings, it projects the ID-relevant features to the subspace of  learnable prompt matrix $W$, and projects the OOD features to the orthogonal complement space.  The proposed method is easy to implement and effective.

### Strengths
Strengths:

-  The paper is well-written and easy to follow. 
-  The proposed loss term is technically sound and intuitive. 
-  The experiments are extensive.

### Weaknesses
Weaknesses:
-   Several recent works are missing.  
    - Designing a scoring function has been a central topic in the community, it is a bit disrespectful not to cite recent works on this direction. Particularly, ViM [1], GEN[2], and NN-Guide[3] are three strong baselines. Meanwhile, it is not completely true that zero shot OOD detection "depends heavily on manually crafted prompts, where even slight variations (e.g., “a flower” vs. “a type of a flower”) can significantly impact performance". Indstead, TAG [4] takes the advantage of this phenomena to enhance the performance of OOD detection, and NegLabel [5] only requires OOD labels. Neither of them are sensitive to prompt templates or require fine-tuning. 
-   The experimental settings do not follow the standard OOD benchmark (https://zjysteven.github.io/OpenOOD/). 
     - The authors are expected to include the results on OpenImage-O.

- The experiments are only done for ViT-B/16. 
   - The authors are expected to show results on ResNet-50.

[1] ViM: Out-Of-Distribution with Virtual-logit Matching. CVPR, 2022.
[2] GEN: Pushing the Limits of Softmax-Based Out-of-Distribution Detection. CVPR, 2023.
[3] Nearest Neighbor Guidance for Out-of-Distribution Detection. ICCV, 2023.
[4] TAG: Text Prompt Augmentation for Zero-Shot Out-of-Distribution Detection. ECCV, 2024.
[5] Negative Label Guided OOD Detection with Pretrained Vision-Language Models. ICLR, 2024.

### Questions
See weaknesses.

### Soundness
3

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
4

### Summary
This paper focuses on enhancing the OOD detection capability of VLMs under the setting of few-shot training with only ID data. It proposes a new learning framework which projects the ID features into a subspace spanned by the prompt vectors, while simultaneously projecting ID-irrelevant features into the orthogonal null space. Experiments on ImageNet OOD benchmarks demonstrate that the proposed method outperforms state-of-the-art few-shot OOD detection approaches.

### Strengths
1. The proposed method is clearly presented and easy to understand.
2. Various ablation experiments are conducted to analyze the proposed method.

### Weaknesses
1.  **The motivation of the proposed method is logically weak and flawed**. The paper claims that existing prompt learning-based OOD methods overlook the geometry of the visual feature embeddings learned by VLMs, but the reviewer believes that the loss function of LoCoOp inherently incorporates the geometry of the visual features. To be specific, the cross-entropy loss is defined as:

$$
\mathcal{L}\_{CE}=-\log\frac{exp(sim(f^{in},g\_k)/ \tau)}{\sum\_{k'=1}^K exp(sim(f^{in},g\_{k'})/ \tau)}=-(\frac{sim(f^{in},g\_k)}{\tau}-\log(\sum\_{k'=1}^K \frac{exp(sim(f^{in},g\_{k'}))}{\tau}))
$$

where the ground truth label $y=k$ and $sim(f^{in},g_k)$ represents the cosine value of the angle between visual features $f^{in}$ and textual features $g_k$. Minimizing $\mathcal{L}_{CE}$ is basically equivalent to maximizing the cosine similarity between $f^{in}$ and $g_k$ and minimizing the cosine similarity between $f^{in}$ and the textual features of other ID classes, which indeed involves the geometry of the visual features. If we consider the subspace spanned by the textual features of all ID classes, i.e., $W=[g_1, g_2, \dots, g_K]$, the original framework of LoCoOp can also be seen as a form of subspace representation learning.

2. **The proposed method seems misaligned with the core design of CLIP**. To be specific, CLIP is pretrained to directly align the visual features (the output of the image encoder) and textual features (the output of the text encoder), rather than the visual features and the embedding of the input prompts (the input of the text encoder). However, the proposed method projects the visual features into the subspace spanned by the learnable prompt vectors. The reviewer suggests that the authors provide an in-depth theoretical analysis or discussion on this misalignment problem.

3. The reviewer **doubts whether it’s reasonable to choose the learnable prompt vectors as the basis of the ID subspace.** The learnable prompt vectors are originally designed to represent the template prompts, such as “This is a photo of”, which contains no ID information at all. The ID information of the textual features is supposed to lie in the ID classnames, rather than the template prompts.  Therefore, using the prompt vectors as the basis for the ID subspace lacks a theoretical justification.

4. **The contribution of this paper is rather empirical.** The authors provide no theoretical analysis or justification for the proposed methods, so extensive experiments are needed to demonstrate that the proposed method can significantly benefit existing methods. But the proposed method **shows marginal improvement on some benchmarks**, such as ImageNet-100 OOD benchmark and using ViT-B/32 and RN-50 as image encoder, with improvement on FPR95 less than 1. 

5. **This paper omits discussion on several important related works**, including ID-like [1], NegPrompt [2] and Local-Prompt [3], which are all prompt-tuning-based OOD detection methods. What’s more, ID-like and Local-Prompt both utilize the ID-irrelevant regions for OOD regularization, which are similar to the proposed method.

6. The introduction section presents the representative LoCoOp and SCT methods but **fails to clearly explain the core motivation** of the proposed method. The reviewer suggests that the authors add a more detailed explanation of the motivation in the “Our contributions” paragraph which starts at Line 92.

7. The reviewer suggests that the authors conduct experiments to **evaluate and compare the actual time cost** of a training iteration of LoCoOp and the proposed method, to demonstrate the computation efficiency of the proposed method.

8. Could the authors evaluate the proposed method on the **CIFAR-100 OOD benchmark** and the **hard OOD setting** with ImageNet-10 as ID and ImageNet-100 as OOD?

9. The statement about efficiently extracting more informative proxy-OOD supervision in Line 94 is **confusing and unclear**. Could the authors explain what “extracting more informative proxy-OOD supervision” means? After all, the proposed method uses the same OOD Local Features Extraction as LoCoOp.

10. **The organization of the experiment section could be improved for better clarity and readability.** For instance, the experiment setups, main experiments and ablation study can be presented as three subsections with second-level headings.

[1]. Yichen Bai, et al. ID-like Prompt Learning for Few-Shot Out-of-Distribution Detection

[2]. Tianqi Li, et al. Learning Transferable Negative Prompts for Out-of-Distribution Detection

[3]. Fanhu Zeng, et al. Local-Prompt: Extensible Local Prompts for Few-Shot Out-of-Distribution Detection

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes SubCoOp, a geometry-aware prompt optimization framework for few-shot OOD detection with CLIP. Prompt vectors are used as a basis for an ID subspace; ID features are encouraged to lie in the column space, while proxy-OOD (local irrelevant) features are pushed into the orthogonal null space via subspace regularization. The loss combines CE, entropy maximization, and two projection-based terms, weighted by model confidence. The method is lightweight, end-to-end, and integrates seamlessly with prior prompt-tuning OOD detectors.

### Strengths
- Casting prompts as subspace basis is principled, and parameter-efficient. Orthogonal projections rigorously separate ID and proxy-OOD signals in latent space, complementing softmax-based methods.

- The empirical evaluation confirms that the proposed methodology achieves SOTA performance, consistently outperforming established baselines. Furthermore, the ablation studies provide a clear validation for the contribution of each component within the objective function, justifying the design choices.

### Weaknesses
- The novelty of the proposed methodology is questionable. The 'OOD Local Features Extraction' module appears to be a direct application, if not identical to, the LoCoOP method. Concurrently, the reliance on subspace projection constitutes a standard approach that has been extensively explored in prior OOD research [1, 2, 3]. As such, the work feels derivative, and its unique contribution remains unclear.

- The analysis of the learned subspace is underdeveloped. The manuscript would be significantly strengthened by a more rigorous evaluation, including: (1) A sensitivity analysis or ablation study on the subspace dimensionality, $M$, to justify its choice. (2) A more compelling demonstration of the subspace's effectiveness. The empirical results show only a marginal performance gain over the baseline in the original feature space (94.07% vs 93.16% AUROC). This limited improvement raises questions about the practical utility and necessity of the projection component. The authors are encouraged to explore potential refinements, such as enforcing constraints (e.g., basis orthogonality) on the subspace, which might yield a more discriminative representation.

Reference:

[1]: NECO: NEural Collapse Based Out-of-distribution detection

[2]: Out-of-Distribution Detection Using Union of 1-Dimensional Subspaces

[3]: Out-of-distribution detection based on subspace projection of high-dimensional features output by the last convolutional layer

### Questions
Please see Weaknesses.

### Soundness
3

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
This paper presents a prompt optimization-based few-shot OOD detection method for pre-trained VLMs such as CLIP.

The proposed method builds upon the existing few-shot OOD detection framework called LoCoOp (Miyai et al., NeurIPS 2023), which extracts ID-relevant and ID-irrelevant regions from real ID images (foreground and background) and uses them as pseudo ID and OOD samples for prompt optimization; the goal of this work is to improve the discriminability between such regions.

To this end, the core idea is to introduce subspace regularization into the LoCoOp framework. Motivated by the observation that ID-relevant features tend to align with the subspace spanned by the learned prompt embeddings, the method optimizes the prompts so that pseudo ID regions lie within this subspace, while ID-irrelevant regions are projected onto its orthogonal complement.

Experimental results demonstrate that the proposed method achieves comparable or superior performance to LoCoOp-based baselines across multiple datasets.

### Strengths
**S1.** Although subspace-based approaches have been widely explored in the OOD detection literature, prior work within LoCoOp and its variants have not leveraged the prompt-spanned subspace and its orthogonal complement to separate ID-relevant and ID-irrelevant regions.

**S2.** The method is conceptually simple and well-motivated.

**S3.** The paper is clearly written and easy to follow.

### Weaknesses
**W1. Novlty**

Although the idea of leveraging subspace geometry for separating ID-relevant and ID-irrelevant regions is interesting, the contributions of this paper are somewhat limited in terms of novelty across idea, motivation, and method.

- *Idea*: While the introduction of subspace regularization is conceptually interesting, the use of subspace geometry for OOD detection has already been widely explored in prior work, e.g., [a–c].

- *Motivation*: Improving the quality and reliability of ID-relevant and ID-irrelevant regions in LoCoOp is a well-recognized objective. Indeed, similar goals have already been addressed by SCT (Yu et al., NeurIPS 2024) and OSPCoOp (Xu et al., CVPR 2025), both of which aim to address the same limitations of LoCoOp in selecting and diversifying ID-irrelevant regions.

- *Method*: The overall pipeline and components (e.g., local feature extraction and loss reweighting) remain almost identical to LoCoOp and SCT. Technically, the method can be regarded as adding a subspace regularization term on top of the SCT formulation.

Consequently, the proposed approach substantially depends on existing LoCoOp-based designs, and its contribution, though reasonable and well-motivated, appears relatively incremental.


**W2. Technical Soundness**

The proposed geometric regularization is reasonable and mathematically sound. However, the ID-irrelevant regions (proxy OOD regions) generated within the LoCoOp framework suffer from several inherent limitations:

- they are mostly confined to the background areas of ID images, resulting in limited diversity, especially in few-shot settings; and

- their selection is based on confidence scores, which can be affected by context bias; CLIP often assigns high confidence to background regions that frequently co-occur with foreground objects.

The proposed subspace regularization could potentially mitigate these issues by promoting better generalization, but this hypothesis is not thoroughly analyzed in the paper. 


**W3. Empirical Significance**

**W3-1.** The empirical advantage of the proposed method is marginal. Compared with OSPCoOp, it improves AUROC by only +0.1% and reduces FPR95 by −0.9%; compared with SCT, the gains are +0.9% in AUROC and −2.7% in FPR95. These differences are small relative to the reported error bars and may easily vanish or reverse under different hyperparameter settings (Fig. 6–8).

**W3-2**. Several recent methods, such as [d] and [e], are not included in the comparison. Unless there is a clear justification for their exclusion, it would be desirable to include them for a fair and up-to-date evaluation.

**W3-3.** Given the small performance margin, the method is likely sensitive to hyperparameter choices. However, the sensitivity analysis is limited in scope: Fig. 6 explores a narrow range of values, and the interaction between $\lambda_1$ - $\lambda_3$ and $C$ is not examined. A more comprehensive and systematic evaluation of these parameters would strengthen the empirical evidence.

-----
[a] Guan et al., Revisit PCA-based technique for Out-of-Distribution Detection, ICCV 2023.

[b] Behpour et al., GradOrth: A Simple yet Efficient Out-of-Distribution Detection with Orthogonal Projection of Gradients, NeurIPS 2023.

[c] Zaeemzadeh et al., Out-of-Distribution Detection Using Union of 1-Dimensional Subspaces, CVPR 2021.

[d] Zeng et al., Local-Prompt: Extensible Local Prompts for Few-Shot Out-of-Distribution Detection, ICLR 2025.

[e] Zhang et al., LAPT: Label-driven Automated Prompt Tuning for OOD Detection with Vision-Language Models, ECCV 2024.

### Questions
**Q1. (Related to W1)**

If there are any factual misunderstandings in W1, please clarify them in the authors' rebuttal. 

**Q2. (Related to W2)**

Could the authors provide objective evidence that the proposed subspace regularization mitigates the weaknesses of the LoCoOp framework? A more detailed investigation, such as qualitative visualization or correlation analysis between region selection and semantic context, would strengthen the technical claims.

**Q3. (Related to W3)**

Are there specific reasons why recent methods such as [d,e] were not included in the comparison? If not, could the authors provide results including these methods for a fair and up-to-date evaluation? Additionally, have the authors conducted broader or joint parameter sweeps to confirm the robustness of the reported results?

### Soundness
3

### Presentation
3

### Contribution
2
