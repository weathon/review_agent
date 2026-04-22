# Efficient Cell Painting Image Representation Learning via Cross-Well Aligned Masked Siamese Network

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Computational models that predict cellular phenotypic responses to chemical and genetic perturbations can accelerate drug discovery by prioritizing therapeutic hypotheses and reducing costly wet-lab iteration. However, extracting biologically meaningful and batch-robust cell painting representations remains challenging. Conventional self-supervised and contrastive learning approaches often require a large-scale model and/or a huge amount of carefully curated data, still struggling with batch effects. We present Cross-Well Aligned Masked Siamese Network (CWA-MSN), a novel representation learning framework that aligns embeddings of cells subjected to the same perturbation across different wells, enforcing semantic consistency despite batch effects. Integrated into a masked siamese architecture, this alignment yields features that capture fine-grained morphology while remaining data- and parameter-efficient. For instance, in a gene-gene relationship retrieval benchmark, CWA-MSN outperforms the state-of-the-art publicly available self-supervised (OpenPhenom) and contrastive learning (CellCLIP) methods, improving the benchmark scores by +29\% and +9\%, respectively, while training on substantially fewer data (e.g., 0.2M images for CWA-MSN vs. 2.2M images for OpenPhenom) or smaller model size (e.g., 22M parameters for CWA-MSN vs. 1.48B parameters for CellCLIP). Extensive experiments demonstrate that CWA-MSN is a simple and effective way to learn cell image representation, enabling efficient phenotype modeling even under limited data and parameter budgets. The source code for CWA-MSN is available at [anonymous code link](https://anonymous.4open.science/r/cwa-msn-56D0DEZIMYNONA/README.md)

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a self-supervised contrastive learning framework for representation learning on Cell Painting images. The proposed approach achieves strong data efficiency and outperforms existing methods.

### Strengths
The paper addresses an important and timely problem in morphological profiling by proposing an efficient supervised contrastive learning framework for Cell Painting images. The proposed cross-well sampling strategy, which leverages plate and batch diversity as a form of data augmentation, is an intuitive and biologically grounded idea that improves robustness to batch effects. The method demonstrates strong performance in data-efficient regimes and shows potential for scalable representation learning in high-content imaging in RxRx3-core.

### Weaknesses
**Motivation & technical contribution**
* While the proposed cross-well sampling strategy across batches and plates is an interesting data augmentation approach for Cell Painting data, the overall proposed method relies heavily on existing architectures and previously established contrastive SSL, limiting methodological novelty.

* The authors argue that their approach provides a stronger proxy training signal for capturing perturbation effects than prior methods. However, since supervision is based primarily on the same pairs, it remains unclear why this setup would effectively distinguish true perturbation effects from false positives, perturbations that are visually similar to controls.

**Limited evaluation**
* The evaluation is relatively narrow in scope. Given that the proposed method is generalized for Cell Painting images, experiments beyond RxRx3-Core would strengthen the work. For instance, evaluating on CP-JUMP1 [1] or reporting performance on the Bray et al. dataset would better demonstrate generalizability.

* Since the method is presented as a contrastive self-supervised learning framework, it would be important to include baselines that leverage supervision in comparable ways (e.g., standard supervised or semi-supervised contrastive methods, or some other augmentation strategies [2,3]). Such comparisons would provide a fairer and more comprehensive evaluation of the proposed approach.

* Similarly, the paper does not provide sufficient detail or analysis on how the design choices, such as the criteria for selecting anchors and target pairs, affect the effectiveness of the proposed method. Since the formulation primarily emphasizes positive-pair similarity without any negative pairs, it remains uncertain whether the method can effectively handle one of the key downstream tasks, sister perturbation identification [1], where morphologically similar but biologically distinct perturbations must be distinguished.

* To support the claims of computational efficiency, the authors should report the number of FLOPs, as this is a more standard and informative metric for evaluating efficiency than parameter count alone.

[1] Three million images and morphological profiles of cells treated with matched chemical and genetic perturbations. Nature com, Chandrasekaran et al. 

[2] Supervised Contrastive Learning. NeurIPS 2020, Khosla et al.

[3] SimCLR (Self/Semi-supervised Baseline) ICML 2020, Chen et al.

### Questions
1. Are there alternative criteria or strategies for selecting anchors and target pairs in practice, and how sensitive is the model’s performance to these design choices? How does the number of positive pairs affect the model's effectiveness?
2. Since SSL encourages similarity, to what extent does this objective reduce or obscure meaningful individual variance in the learned representations?
3. Could there be scenarios where pairing samples across plates might inadvertently reinforce batch effects rather than mitigate them?
4. Why does the CropMAE-single require longer training time compared to CropMAE-cross?

### Soundness
3

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
3

### Summary
In this paper, the authors introduce the Cross-Well Aligned Masked Siamese Network (CWA-MSN), a strategy designed to align HCS image embeddings of identically perturbed cells across wells. The proposed framework outperforms existing methods in gene–gene and compound–gene retrieval, even when trained with limited data.

### Strengths
1. The paper addresses an interesting problem in HCS imaging, where biological information can be obscured by batch effects.
2. The paper is well written and easy to follow.
3. The proposed framework achieves promising results in gene–gene and compound–gene retrieval.

### Weaknesses
1. The novelty of the work is somewhat limited, as it can be viewed as an application of (Assran et al.) to HCS images.

2. I think that the evaluation tasks used (gene–gene and compound–gene retrieval) are rather limited. Did the authors apply the trained model to any other tasks?

### Questions
1. Did the authors quantify the batch effect? Does the method provide improvements in that regard?

2. Did the authors train on larger datasets? Would that be expected to further improve the results?

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
This paper introduces the Cross-Well Aligned Masked Siamese Network (CWA-MSN), designed to address challenges in extracting robust biological features from high-content screening images. Specifically, CWA-MSN targets the issue of batch effects, such as variations in experimental conditions. The framework aligns cell representations from different wells subjected to the same perturbation, ensuring consistency in the learned features despite these batch effects.

### Strengths
Aligning cell representations from different wells within the same perturbation is a reasonable approach.

The method shows improvements in performance with fewer data and model parameters compared to existing methods.

### Weaknesses
The idea of aligning representations under the same perturbation has already been proposed in [1]. Additionally, [1] also uses a self-distillation framework with a masked modeling approach. While there are differences in design, this paper employs the MSN framework, while [1] uses DINO v2, and this paper aligns images from different wells, whereas [1] constructs positive pairs from different sites within the same well. A thorough discussion is needed to clarify the unique contributions of this paper.

Another concern relates to the problem setup. While this paper uses the perturbation label to determine the target for cross-well alignment, this approach no longer adheres to a self-supervised setting. As a result, the claim of addressing batch effects in self-supervised approaches seems questionable. Additionally, this claim is not supported by experiments or visualizations.

[1] Self-supervised Representation Learning with Local Aggregation for Image-based Profiling,	arXiv:2506.14265.

### Questions
Given the distributional gap between cell images and natural images, how do you handle the different input channels and augmentations?

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
4

### Summary
This paper proposes the Cross-Well Aligned Masked Siamese Network (CWA-MSN) to address batch effects in high-content cellular imaging. The method aligns representations of cells from different wells under the same perturbation, aiming to improve robustness to experimental variations. CWA-MSN combines perturbation-guided representation alignment with masked image reconstruction to learn stable and discriminative embeddings.

### Strengths
1. The paper focuses on an important and practical problem in biological image analysis — mitigating batch effects in cell-level feature learning.
2. The proposed cross-well alignment is a reasonable idea in the context of perturbation-based experiments.
3. The method achieves consistent performance gains while maintaining moderate model complexity.  
4. The paper is clearly written and well-structured, making it easy to follow.

### Weaknesses
1. The problem and solution seem misaligned. The stated goal is to overcome batch effects (common patterns across different wells within the same batch), but the proposed method aligns different wells under the same perturbation. Therefore, the performance gain may not necessarily result from eliminating batch effects. More evidence is needed to validate that batch-effect reduction is indeed achieved.

2. The method uses perturbation labels as alignment supervision. Since perturbation type is often correlated with downstream task labels, this raises concerns about fairness when comparing with self-supervised baselines that do not use label information.

3. The necessity of masked reconstruction under this setting is unclear.  Masked modeling is effective for self-supervised representation learning; however, once perturbation labels are introduced, its contribution should be reevaluated.  A comparison between masked and purely supervised formulations would help justify the design choice.

4. The idea of aligning representations under the same perturbation has been explored in "Self-supervised Representation Learning with Local Aggregation for Image-based Profiling" (arXiv:2506.14265). The paper should discuss the relationship and distinction between CWA-MSN and this prior work, particularly regarding supervision signals and architectural differences.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
