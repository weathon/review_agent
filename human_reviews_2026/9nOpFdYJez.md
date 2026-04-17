# From Sparse to Structured: A New Paradigm for Gradient-Based Parameter-Efficient Fine-Tuning

- Decision: Reject
- Scores: 2, 6, 4, 4, 6

## Abstract
Large pre-trained models have demonstrated extensive applications across various fields. However, fine-tuning these models for specific downstream tasks demands significant computational resources and storage. One fine-tuning method, gradient-based parameter selection (GPS), focuses on fine-tuning only the parameters with high gradients in each neuron, thereby reducing the number of training parameters. Nevertheless, this approach increases computational resource requirements and storage demands. In this paper, we propose an efficient gradient-based and regularized fine-tuning method (GRFT) that updates the rows or columns of the weight matrix. We theoretically demonstrate that the rows or columns with the highest sum of squared gradients are optimal for updating. This strategy effectively reduces storage overhead and improves the efficiency of parameter selection. Additionally, we incorporate regularization to enhance knowledge transfer from the pre-trained model. GRFT achieves state-of-the-art performance, surpassing existing methods such as GPS, Adapter Tuning, and LoRA. Notably, GRFT requires updating only 1.22% and 0.30% of the total parameters on FGVC and VTAB datasets, respectively, demonstrating its high efficiency and effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces GRFT (Gradient-based and Regularized Fine-Tuning), a parameter-efficient fine-tuning approach. Unlike GPS, which selects sparse individual parameters, GRFT updates entire rows or columns of weight matrices that have the largest summed squared gradients. This structured selection reduces storage overhead and improves hardware efficiency compared to GPS. In addition, an L2 regularization term is incorporated to preserve the knowledge learned during pre-training. Experiments on FGVC, VTAB, and GLUE benchmarks demonstrate consistent performance improvements over GPS, LoRA, and Adapter methods.

### Strengths
1. The paper is clearly written and well organized. The presentation is smooth and accessible, making the core ideas and technical details easy to understand.
2. The paper shifts gradient-based PEFT from unstructured sparsity to structured row/column selection, which significantly reduces parameter storage and improves computational efficiency.

### Weaknesses
1. The novelty is somewhat incremental, mainly extending GPS from element-wise to structured row/column selection. In addition, the proposed regularization contributes only marginal performance gains according to the results presented in Table 5.
2. The experimental results raise some concerns. While GRFT’s storage efficiency is clear, the performance improvement over GPS is less intuitive. Intuitively, GPS performs finer-grained parameter updates, and the paper does not provide sufficient explanation or analysis to clarify why the structured updates in GRFT lead to higher performance.
3. The paper lacks comparisons with several recent and stronger baselines on vision tasks, such as MLAE [1] and GLoRA [2], which limits the assessment of its competitiveness. Meanwhile, the baselines in the LLM experiments are also insufficient, as GPS itself is not included and no comparisons are made with more advanced fine-tuning methods. Moreover, only three datasets from the GLUE benchmark are reported instead of the full suite, which is unusual and raises concerns about possible selective reporting of favorable results.

------

[1] MLAE: Masked LoRA Experts for Visual Parameter-Efficient Fine-Tuning. https://arxiv.org/abs/2405.18897

[2] One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning. https://arxiv.org/abs/2306.07967

### Questions
1. The paper should include a comprehensive comparison of computational efficiency between GRFT, GPS, and LoRA.

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
4

### Summary
This work proposes a parameter-efficient fine-tuning method termed GRFT, which selects and fine-tunes only the parameters with high gradients in some rows or columns. Additionally, this work incorporates regularization to enhance knowledge transfer from the pre-trained model. The proposed method can achieve good performance on three datasets for image/text classification.

### Strengths
1.  This work is very well written and easy to follow.
2.  This approach not only achieves good performance but also offers certain advantages in terms of GPU consumption and storage.
3.  The authors conduct extensive experiments on three classification datasets, and the ablation studies are also convincing.

### Weaknesses
1. The author does not adequately explain why the row/column selection scheme is more effective than the sparse selection scheme.

2. The authors only conduct experiments on classification datasets, which are prone to overfitting.

3. This work is suspected of deliberately selecting favorable data. The ablation experiments were primarily conducted on the FGVC dataset. Tables 4 and 5 show the results for each subset, but Figures 3 and 4 appear to intentionally omit the results for the Oxford Flowers subset.

4. "Eq. equation i" should be "Eq. i"

### Questions
1. Compared to GPS, what are the main contributions of the proposed method? These methods are similar in concept, both based on gradient-based selection. One performs sparse selection, while the other performs row selection. The authors should further elaborate on the contributions of the proposed method.

2. The selected masks are all pre-generated and are the same for all epochs. What if we used dynamic masks, for example, different masks for each epoch or each batch?

3. This work uses (row sum of) squares to select parameters. Are there any better metrics?

4. Can the authors provide some experimental analysis on more challenging tasks (such as object detection and segmentation)?

### Soundness
3

### Presentation
4

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
This paper introduces GRFT, a gradient based and regularized fine tuning method that selects entire rows or columns of weight matrices using the highest sums of squared gradients, greatly reducing mask storage compared to sparse, per entry selection (e.g., GPS). It provides a simple theoretical justification for the selection rule and adds an L2 regularization term to keep updates close to the pretrained weights, improving knowledge transfer. Experiments on image tasks (FGVC, VTAB with ViT) and text tasks (GLUE with LLaMA 3) report improved accuracy over GPS, Adapter Tuning, and LoRA.

### Strengths
- The paper proposes a clear and practical method for structured parameter selection, choosing entire rows or columns by the largest sums of squared gradients. This design improves upon dense sparse masks which simplifies implementation and reduces storage overhead during training and deployment.
- It provides a simple theoretical analysis for the selection rule and augments training with an L2 regularizer that keeps updates close to the pretrained weights.
- The approach shows good parameter efficiency across both vision and language settings.

### Weaknesses
- There is a lack of wall-clock runtime comparision gains against other baselines.
- The selection rule is underexplored and not adaptively tuned; the row-versus-column orientation is fixed rather than model or task adapted, leaving robustness to these choices unclear.

### Questions
- How stable is the set of selected rows or columns under random mini-batch sampling?
- Does the selection pattern imply that certain rows or columns are inherently more important for a given task or dataset. For example, are the same indices consistently chosen across runs or layers?
- Is there a way to adaptively choose between selecting rows and selecting columns instead of fixing one orientation?
- How does accuracy scale as the number of selected blocks increases, and do you observe diminishing returns?
- Can you provide the wall-clock speed and throughput of GRFT compared to other finetuning methods?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes the GRFT method to address the storage and hardware efficiency bottlenecks of existing gradient-based parameter selection methods (e.g., GPS). GRFT directly selects entire rows or columns with the largest sum of squared gradients from the weight matrix as trainable parameters. Theoretically, through first-order Taylor expansion and loss function optimization derivation, it is proven that "selecting rows/columns with high sum of squared gradients" can maximize the efficiency of loss reduction and save storage compared to GPS. Additionally, a hierarchical regularization design is proposed, which incorporates the "L2 norm difference between fine-tuned parameters and pre-trained parameters" into the loss function.

### Strengths
1. The paper proposes the GRFT method to tackle the storage and hardware efficiency issues of existing gradient-based parameter selection methods.
2. GRFT selects entire rows or columns with the largest sum of squared gradients from the weight matrix as trainable parameters, a choice theoretically justified via first-order Taylor expansion and loss function optimization to maximize loss reduction efficiency and save storage.
3. It also introduces a hierarchical regularization design, integrating the L2 norm.

### Weaknesses
This paper presents a competently executed but incrementally innovative extension of gradient-based parameter selection methods. Due to its limited originality, narrow scope of comparative experimental baselines, and modest practical significance, it falls slightly below the acceptance threshold for ICLR conference.

### Questions
1. We consider that the innovation of the work is insufficient. The novelty of the technical architecture is lacking, and the loss function is not proposed in an innovative manner. GRFT only focuses on the fixed structured unit of "rows/columns"; its selection unit is fixed, resulting in insufficient adaptability and flexibility for different feature dimensions and task difficulties. The work fails to explore more flexible parameter grouping methods, leading to relatively weak innovation.  

2. The work takes the "last L layers and classification head" as the focus of regularization. In existing studies, some methods have dynamically determined regularization strength through "layer sensitivity analysis" (e.g., applying strong constraints to high-sensitivity layers and weak constraints to low-sensitivity layers). In contrast, GRFT adopts a fixed hierarchical approach and lacks adaptive adjustment capabilities.  

3. The parameter selection (row/column selection) and regularization (L2 constraint) of GRFT are two independently implemented modules, and the work does not explore the collaborative optimization logic between the m. This results in low integration between the two core modules and insufficient integrated innovation of the overall architecture.  

4. The work mainly compares with basic Parameter-Efficient Fine-Tuning methods such as LoRA, GPS, and Adapter, but fails to include more advanced gradient-based or structured fine-tuning methods in the field. For example, it does not compare with GaLoRA [1], which has already realized "gradient-guided low-rank matrix optimization" and is directly comparable to GRFT in terms of design ideas.  

5. It is suggested to supplement the explanation of Figure 3.

[1] Zhao J, Zhang Z, Chen B, et al. GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection. International Conference on Machine Learning. PMLR, 2024: 61121-61143.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces GRFT (Gradient-based and Regularized Fine-Tuning), a parameter-efficient fine-tuning method that replaces element-wise gradient selection (as in GPS) with structured row- or column-wise updates.

The key idea is to select entire rows or columns of weight matrices whose gradients have the largest squared sums, reducing mask storage from $O(mn)$ to $O(\min(m, n))$.

The method further incorporates L2 regularization to preserve pre-trained knowledge and mitigate catastrophic forgetting.

Empirical results on FGVC, VTAB, and GLUE benchmarks show that GRFT achieves comparable or superior performance to GPS, LoRA, and Adapter Tuning while updating less than 1% of total parameters.

### Strengths
* Well-motivated approach addressing a clear bottleneck in GPS: large sparse masks and inefficient updates.
* Structured row/column selection is simple, elegant, and hardware-friendly, reducing both storage and computation.
* Theoretical justification is concise but sufficient, showing that rows or columns with the largest gradient energy maximize expected loss reduction under structured constraints.
* Comprehensive experiments across vision and NLP benchmarks with consistent improvements and detailed ablation studies.
* Easy to integrate into existing architectures and training pipelines without additional parameters or structural modifications.
* Clear presentation and logical exposition of ideas.

### Weaknesses
* Conceptual novelty is limited; the method is an incremental engineering extension of GPS rather than a fundamentally new paradigm.
* The practical implementation of the structured mask on GPU (whether physical pruning or logical freezing) is not described in detail.
* Reported accuracy improvements are modest, focusing more on efficiency than on representational gains.
* Performance appears somewhat sensitive to hyperparameters such as the number of selected rows $k$ and the regularization coefficient $\lambda$, which may affect reproducibility.
* No open-source code is provided to confirm claimed memory and runtime benefits.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
