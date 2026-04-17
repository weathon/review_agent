# CLUE: Fine-Grained Self-Supervised Learning with Multi-Level Regularization

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Self-supervised learning (SSL) has achieved strong results on coarse-grained tasks
but often struggles with fine-grained recognition, where categories differ only by
subtle local cues. For strong downstream transfer, features must form compact
within-class clusters with large inter-class margins at the fine level. However,
standard SSL losses either over-separate visually similar subcategories by treating
all non-positives as equally negative, or overlook part-based evidence and thus
merge them under coarse prototypes. We propose a multi-level regularization
framework that improves clustering across granularities. At the global level, a soft
variant of InfoNCE reduces false negatives and enhances class separation. At the
part level, clustering on local descriptors preserves subtle intra-class distinctions.
At the instance level, semantic descriptions from vision–language models provide
attribute-level anchors. Together, these components yield representations with
balanced clustering across granularities. Experiments on CUB200-2011, Stanford
Cars, and FGVC-Aircraft show consistent improvements in both classification and
retrieval, validating the effectiveness of our approach for fine-grained SSL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper studies the problem of self-supervised fine-grained visual categorization. 

Starting from the over-dispersion and over-collapse challenges of the prior arts, this paper proposes a multi-level regularization framework CLUE.

Technically, it consists of:

- a soft-InfoNCE global term via Sinkhorn-based soft assignments, 

- a part-level loss using VLAD-style local descriptors, 

- a VLM-guided alignment using CLIP text embeddings.

The experiments are conducted on three standard FGVC datasets, namely, CUB-200, Cars, and Aircraft.

 CLUE improves the accuracy and retrieval metrics over existing self-supervised methods.

### Strengths
+ Overall this paper is clearly-presented and the idea is easy-to-follow.

+ The technical design is clearly-motivated and solid. 

+ When doing experiment analysis, the introduction of clustering metrics to show additive effects of global/local/text modules is new in the community.

### Weaknesses
- The proposed soft-InfoNCE is extremely close to ReSA [1], in which Sinkhorn-derived doubly-stochastic targets replacing the identity in InfoNCE. As a result, the technique novelties of the proposed method are not as strong as it claims.

[1] Clustering Properties of Self-Supervised Learning, ICML'2025.

- This paper significantly misses the comparison with state-of-the-art SSL methods and self-supervised FGVC methods, particularly after 2023. After a brief search, it at least misses the comparison with [2-5].

[2] Hu, Feiran, et al. "An asymmetric augmented self-supervised learning method for unsupervised fine-grained image hashing." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[3] Li, Chunyuan, et al. "Efficient Self-supervised Vision Transformers for Representation Learning." International Conference on Learning Representations, 2022.

[4] Shi, Jialu, et al. "LoDisc: Learning global-local discriminative features for self-supervised fine-grained visual recognition." IEEE Transactions on Circuits and Systems for Video Technology (2025).

[5] Bi, Qi, et al. "Cross-Level Multi-Instance Distillation for Self-Supervised Fine-Grained Visual Categorization." IEEE Transactions on Image Processing (2025).

- Besides, the idea to consider both local and global features in self-supervised FGVC is also not novel, as already studied in [3, 4, 5]. 

- Results are only on ResNet-50. Please also report the performance on ViT-B and Swin-Transformer image encoder, like the rest 2023–2025 SSL works do. Without them, claims of SOTA are not credible. 

- The performance improvement of more recent methods such as LoDisc and LDF is marginal. 

- Aligning images to CLIP text embeddings is common, which also negatively affects the novelties.

- Missing ablations on prompt templates, attribute extraction strategy, number/quality of anchors, and robustness to noisy text. 

- What if we use some simplier local descriptor alternatives, for example, attentive pooling, token-wise contrast, or other NetVLAD variants? 

- Typos and notation issues.

### Questions
Please refer to the weakness section, and address the questions point-by-point.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the limitations of self-supervised learning (SSL) for fine-grained visual recognition (FGVR). The authors posit that standard SSL losses suffer from a "granularity mismatch," which leads to two failure modes on fine-grained categories: "over-dispersion" or "over-collapse" . To address this, the paper proposes CLUE, a multi-level regularization framework. This framework integrates three key components: (1) a global-level soft InfoNCE loss , (2) a part-level clustering loss on local descriptors , and (3) instance-level semantic guidance from Vision-Language Models (VLMs) . Experimental results demonstrate that this method achieves state-of-the-art (SOTA) classification and retrieval performance on several fine-grained benchmarks, including CUB-200, Stanford Cars, and FGVC-Aircraft.

### Strengths
1. Clear Problem Diagnosis and Quantification: One of the paper's primary strengths is its in-depth diagnosis of the problem in Section 3. The authors move beyond the general statement that "SSL is unsuitable for fine-grained tasks" and identify two specific failure modes: "over-dispersion" and "over-collapse".
2. The three components of the proposed CLUE framework are logically sound and directly target the diagnosed problems.
3. Thorough experimental validation and analysis. The paper achieves SOTA results on three major FGVR datasets (Table 1).

### Weaknesses
1. Heavy Reliance on the Baseline (ReSA): From the ablation study (Table 2), the first significant performance jump (e.g., 62.29% to 65.82% on CUB) comes from replacing the InfoNCE baseline with Soft-InfoNCE (ReSA). 
2. Still using ResNet as the backbone, rather than other advanced backbones e.g, ViT.
3. The number of part clusters is a critical hyper-parameter that must be manually selected and tuned18181818. The paper does not propose a method for this value to be learned adaptively. This requires dataset-specific tuning (as suggested by the varying results in Table 3) and limits the method's automatic adaptability, potentially impacting its practical deployment on new datasets.

### Questions
1. Overhead of VLM Description Generation: Could the authors elaborate on the process and overhead of generating the VLM descriptions? Generating CoT descriptions for every single image represents a non-trivial inference cost. How long did this process take? Was any manual filtering or cleaning of the VLM-generated text required beyond the "light de-duplication" mentioned?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a new learning model for learning representations of fine-grained visual categories. The approach applies a self-supervised method at the image and part level. It also exploits semantic guidance from a pre-trained VLM. Experiments show that the approach induces better fine-grained visual representations, using three datasets.

The paper provides a clear and pedagogical introduction to the problem, and the approach is well presented. However, it has several significant shortcomings.

- The work lacks a clear positioning with respect to self-supervised approaches that learn hierarchical representations (e.g [1,2,3,4]). Adding a whole new paragraph in the related work section is required to clarify the novelty of the method. 
- The model is limited by its dependence on a pretrained VLM; in addition, it is unclear how centroids of parts are computed: are they pretrained as well ? Ablation experiments must be completed: it is unclear whether VLM guidance, without other learning modules, suffices to learn fine-grained visual representations.
- Pretraining on three small datasets is not sufficient to demonstrate the generality of the approach. A more standard experimental protocol in SSL is to pretrain the method on ImageNet. During evaluation, this allows clarifying how the model trades-off coarse and fine-grained category representations. Common evaluation protocols (e.g. BYOL) evaluate transfer learning performance on fine-grained datasets. More fine-grained datasets, such as OxfordPet or Flowers, are necessary to show that the approach capture diverse fine-grained features, and not only those related to cars, birds or aircrafts.

Minor detail: To the best of my knowledge, common practice does not use a momentum coefficient of 0.999, but decreases it from 0.996 to 0.999 over training.

[1] Tan, B., Wei, X. S., & Zhao, L. (2025). Prototype-based Contrastive Learning with Stage-wise Progressive Augmentation for Self-Supervised Fine-Grained Learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4125-4134).

[2] Saha, O., & Maji, S. (2023). Particle: Part discovery and contrastive learning for fine-grained recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 167-176).

[3] Xu, M., Guo, Y., Zhu, X., Li, J., Sun, Z., Tang, J., ... & Ni, B. (2022). HIRL: A General Framework for Hierarchical Image Representation Learning. arXiv e-prints, arXiv-2205.

[4] Manová, A., Durrant, A., & Leontidis, G. (2023). S-JEA: Stacked joint embedding architectures for self-supervised visual representation learning. arXiv preprint arXiv:2305.11701.

### Strengths
The work is well-presented

### Weaknesses
The novelty is unclear and the experiments are insufficient to demonstrate the advantages of the approach.

### Questions
Please, see above.

### Soundness
1

### Presentation
3

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
The paper investigates the representation capability of self-supervised learning (SSL) methods in fine-grained visual recognition (FGVR) tasks. The authors first empirically identify that standard SSL approaches often suffer from either over-dispersion or over-collapse when applied to FGVR. To address this issue, they propose a multi-level regularization framework called CLUstEring-aware regularization (CLUE), which incorporates three complementary levels: class-level, part-level, and instance-level regularization. In particular, at the instance level, vision–language models such as CLIP are leveraged as external experts to provide semantic guidance. Experiments on popular FGVR benchmarks—including CIFAR-100, Stanford Cars, CUB-200, and FGVC-Aircraft—demonstrate that CLUE consistently outperforms existing SSL methods.

### Strengths
1. Self-supervised representation learning in computer vision is a popular and important topic, but its fine-grained representation capability requires further research attention. This paper empirically and comprehensively studies this problem, which is commendable.

2. The metrics and empirical analyses presented in Section 3 are comprehensive and effectively motivate the research problem. The proposed multi-level granularity regularization framework (CLUE) provides a well-rounded solution to the challenges of applying SSL to FGVR.

3. Experimental results on four FGVR benchmarks—CIFAR-100, Stanford Cars, CUB-200, and FGVC-Aircraft—demonstrate the effectiveness and superiority of CLUE over existing methods.

### Weaknesses
1. The novelty of the proposed multi-level regularization in SSL is limited. Self-supervised learning has been extensively studied in the computer vision community, especially from 2020 to 2022. Contrastive learning–based methods have already been the mainstream approach, and multi-granularity regularization has been explored in prior works such as [A]. Therefore, the contribution in terms of multi-level regularization does not appear sufficiently novel in this paper.

2. The presentation of the experimental setup in Section 5 is confusing. Due to missing details of the pre-training stage, it is unclear whether the pre-training follows the standard SSL paradigm—i.e., training a model from scratch on a large unlabeled dataset such as ImageNet, and then evaluating the learned representation on various downstream datasets. In Appendix A.2, the authors seem to instead use supervised ImageNet-pretrained weights as initialization and perform “pre-training” directly on the four FGVR evaluation datasets, which contradicts the standard SSL setting.

3. The experimental evaluation is incomplete. Although the four FGVR benchmarks are representative for the fine-grained recognition scenario, it remains unclear how the pre-trained model performs on non-FGVR tasks or datasets. In addition, only ResNet-50 is considered, and no experiments are reported for transformer-based SSL backbones, which limits the generality of the conclusions.

[A] Mugs: A Multi-Granular Self-Supervised Learning Framework. NeurIPS 2022

### Questions
To address the weaknesses, I have the following additional suggestions:

1. Discussion or comparison with relevant SSL works to justify the novelty or key contribution of the paper to SSL.

2. In addition to the current empirical analysis, adding theoretical analysis on FGVR of SSL can strengthen the contribution.

3. Clarify the full experimental setting and provide full details of the pre-training stage. If not following the standard SSL setting, it should be justified why not.

4. Ensure the empirical evaluation of SSL is comprehensive in terms of both FGVR and other tasks.

### Soundness
3

### Presentation
2

### Contribution
2
