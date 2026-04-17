# CLIP-TTA: Robust Test-Time Adaptation via Dual Regularization Beyond Optimal Transport

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Despite the remarkable zero-shot performance of vision-language models, such as Contrastive Language-Image Pretraining (CLIP), on many downstream tasks, their potential may be degraded under distributional shifts. Test-time adaptation (TTA) offers a solution by adapting the model to these shifts during inference, without requiring labeled data. Prior methods like CLIP-OT leverage optimal transport for pseudo-labeling. However, the quality of these labels can be unreliable, leading to suboptimal adaptation and error accumulation. To address this, we propose CLIP-DR, which introduces two extra key components: (1) a cosine similarity loss to align image features with textual prototypes, stabilizing the adaptation direction; and (2) an information maximization regularizer to promote confident and diverse predictions, preventing model collapse. Extensive evaluation on seven benchmarks (covering 15 corruption types and domain shifts, totaling $\sim$6000 trials) demonstrates that CLIP-DR consistently outperforms state-of-the-art methods while adding $\sim$0.01 seconds of computing time per batch (e.g., 4\% and 12\% higher than CLIP-OT and WATT-S on the TinyImageNet-C dataset with 1.98 second per batch).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel method, CLIP-TTA, which adapts CLIP models for downstream tasks without requiring labeled data. To leverage unlabeled data during testing, CLIP-TTA employs optimal transport for pseudo-labeling and incorporates two regularization losses to prevent pseudo-label collapse. Experimental results demonstrate that CLIP-TTA enhances the performance of CLIP under distribution shifts.

### Strengths
1. The problem that how to utilize unlabeled data is central to several areas of machine learning, such as unsupervised and semi-supervised learning. A common approach is self-training, which alternates between assigning pseudo-labels and training the model with confident data. In our view, this paper improves the self-training framework by integrating optimal transport into pseudo-labeling, which is an interesting and inspiring idea.

2. Building upon the improved self-training framework, the paper introduces two regularization losses which use the confidence (or entropy) of predicted sample to prevent collapse. These regularization methods are straightforward and conceptually sound.

3. Figure 3 (Left) shows that the proposed method is not only effective but also efficient.

### Weaknesses
1. In my opinion, this paper is somewhat incremental and similar with CLIP-OT [1]. While it adds two regularization losses to improve the pseudo-labeling process, the novelty feels reduced compared to CLIP-OT. I would appreciate a more detailed comparison to highlight the differences between this work and CLIP-OT.

2. In Figure 5, hyper-parameters $\lambda_1$ and $\lambda_2$  have minimal impact on the average accuracy of CIFAR-10-C and CIFAR-100-C. I suggest the authors provide further explanation on the effectiveness of the proposed regularization losses.

3. CLIP-TTA updates only the visual encoder $\theta$ during the adaptation process, assuming that distribution shifts affect only the images. This assumption limits the scope of application for this method.

4. If I understand correctly, both regularization losses are computed on the output logits. The additional lines in Figure 2 seem unnecessary and make the framework more complex and difficult to understand.

[1] Words Matter: Leveraging Individual Text Embeddings for Code Generation in CLIP Test-Time Adaptation, ArXiv 24

### Questions
See my questions in weakness.

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
3

### Summary
This paper proposes a new test-time adaptation method named CLIP-TTA for VLMs that addresses the unreliable pseudo-labels of prior work like CLIP-OT. The authors introduce two losses: a cosine similarity loss to align image logits with text prototypes and an information maximization regularizer to encourage confident and diverse predictions. Experiments show that CLIP-TTA improves robustness against corruptions and domain shifts over current methods.

### Strengths
1.	The method is presented clear and easy to understand. The component of CLIP-TTA is clearly organized by each section with correct and appropriate reference.

2.	The paper demonstrates consistent performance gains over the primary baseline CLIP-OT across a wide array of benchmarks.

3.	The authors provide detailed model analysis including ablation study, sensitivity of each different parameters and experimental settings.

### Weaknesses
1.	The paper's primary motivation is to solve the "over-confidence" problem (high ECE) of the CLIP-OT baseline. However, the proposed core components: a cosine similarity loss and an information maximization loss, lack a direct theoretical link to this stated goal. The direct objective of $\mathcal{L}\_{cos}$ is to align features, while $\mathcal{L}\_{IM}$ aims to promote confident and diverse predictions to prevent model collapse. The paper fails to clearly articulate the theoretical chain of reasoning for why "alignment" and "preventing collapse" directly solve the problem of over-confidence, making the connection feel indirect and insufficiently supported.

2.	The experimental validation omits several standard and challenging benchmarks. To better assess robustness, evaluation on ImageNet-C[1] is necessary. Furthermore, to test generalization on different data types, the paper would benefit from including fine-grained classification datasets from the CLIP zero-shot suite, such as the DTD[2] or EuroSAT[3].

3.	All experiments are conducted solely on the CLIP (ViT-B/32) backbone. To demonstrate the generalizability of the proposed dual-regularization approach, it should be tested on other vision-language model architectures, such as BLIP, to prove that the method is not just tailored to CLIP.

4.	The paper's core methodological contribution is arguably incremental. The problem formulation (Eq. 1) is standard, and the optimal transport mechanism (Eqs. 2-8) is adopted directly from the CLIP-OT baseline. The primary novelty lies in adding two existing loss functions,  constitutes a limited conceptual advance.

[1] Hendrycks, Dan, and Thomas Dietterich. "Benchmarking neural network robustness to common corruptions and perturbations." arXiv preprint arXiv:1903.12261 (2019).

[2] Cimpoi, Mircea, et al. "Describing textures in the wild." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.

[3] Helber, Patrick, et al. "Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification." IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 12.7 (2019): 2217-2226.

### Questions
We would like to draw your attention to the recent, highly relevant paper by Lafon et al. (2025), "Cliptta: Robust contrastive vision-language test-time adaptation" (arXiv:2507.14312). (Already Cited in Section 2 in original paper)

1.	The title "Cliptta" used by Lafon et al. is practically identical to your proposed "CLIP-TTA". Given this, are you concerned that this will create significant ambiguity and confusion for future researchers when citing and attempting to differentiate these two distinct methods?

2.	Lafon et al. argue that gradient-based TTA can "degrade learned knowledge," and for this reason, they propose a gradient-free solution. How does your dual regularization specifically prevent this degradation?

### Soundness
2

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
4

### Summary
This paper proposes CLIP-TTA, a test-time adaptation (TTA) method designed to address the issue of unreliable pseudo-labels generated by the previous CLIP-OT approach.  CLIP-TTA introduces two key components: (1) a cosine similarity loss to align image features with textual prototypes, ensuring stable adaptation;  and (2) an information maximization regularizer to encourage confident and diverse predictions, preventing model collapse.  Extensive experiments across 7 benchmarks demonstrate competitive performance.

### Strengths
- The paper is well-written and easy to understand.

### Weaknesses
- The contribution of the paper is limited, as the proposed method framework is largely similar to OT-CLIP, with the addition of only two extra losses. Furthermore, there is no theoretical evidence provided to support how these losses contribute to the reduction of ECE.​
- The effectiveness of L_cos relies on the assumption of highly distinguishable text prototypes. However, in many fine-grained tasks, text templates are unable to differentiate between subclasses, which may lead to pushing the model toward incorrect priors.
- Sensitivity to Hyperparameters. The method shows extreme sensitivity to hyperparameters, as shown in Figure 5. Different tasks exhibit strong dependence on hyperparameter settings, which undermines the robustness claimed by the paper.

### Questions
- The manuscript should include validation of CLIP-TTA on cross-dataset and cross-domain benchmarks, as this would make the method's claims more convincing.
- The method should be tested on a broader range of TTA techniques (e.g., TPT) to demonstrate its effectiveness in reducing ECE, rather than being evaluated solely on OT-CLIP.

### Soundness
3

### Presentation
3

### Contribution
2
