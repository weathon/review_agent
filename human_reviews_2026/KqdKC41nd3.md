# Variational Adapter for Cross-modal Similarity Representation

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4

## Abstract
The core of vision-language models lies in measuring cross-modal similarity within a unified representation space. However, most image-text matching or multi-class image classification datasets lack fine-grained cross-modal matching annotations, forcing the continuous similarity space into binary classification boundaries. This compression induces false negative samples and significantly impairs the generalization performance of cross-modal tasks. While prior research has attempted to mitigate this by modeling intra-modal ambiguity, it often overlooks inherent annotation flaws, leading to suboptimal uncertainty allocation. To address these challenges, we propose a Variational Adapter for Cross-modal Similarity Representation (VACSR). This approach reformulates image-text matching with fine-grained semantic scarcity as a variational inference problem. It constructs a latent space for cross-modal similarity and uses regularization techniques to mitigate overfitting to binary annotations. Additionally, we introduce a distributional optimization loss to eliminate erroneous gradients caused by false negative samples. \textcolor{blue}{We validate the effectiveness of VACSR in image-text retrieval tasks using the COCO Caption dataset and two extended datasets: CxC and ECCV Caption. Furthermore, we conduct comprehensive out-of-distribution evaluations including domain generalization on ImageNet and its variants, as well as base-to-novel generalization across 11 datasets, highlighting VACSR’s robust generalization performance in a wide range of real-world situations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper deals with the fact that vision-language models (VLMs) that leaqrn a common space for both texrt and visual inputs are usually trained using sparse, matching/non-matching annotations and discard finer-grained details as well as associations that are missing from the caption (false negatives). The formulate the problem as a form of Variational inference and regularize the space so that overfitting to binary annotations is reduced. They do that via handling uncertainty of the latents: They assign lower uncertainty to positive and  hard negative samples, and lower uncertainty to false negatives. They report strong results on common datasets for cross modal retrieval.

### Strengths
The paper deals with the interesting problem of cross-modal retrieval and presents a variant of a probabilistic embeddings for the task that outperform other methods. The method models the uncertainty of false negatives in an intuitive way and shows strong results on common benchmarks, beating related methods. It is interesting that they also show that their probabilistic embeddings help for zero-shot clasisification.

### Weaknesses
1. The paper lacks a clear discussion on the relation of this paper to other recent approaches that use probabilistic embeddings. In particular, the relationship between the proposed method and the closely related PCME++ is not sufficiently discussed. The contribution would be significantly strengthened if the authors could clearly articulate the differences—both in design and empirical behavior—in the rebuttal and final version. Moreover, some qualitative results showing cases where the method improves over the top competitor would be interesting to analyse. 

2. Building on the previous point, Section 3.1 describes a similarity representation using a 2-component GMM for cross-modal retrieval, following prior work such as Chun et al. (2023) and Bai et al. (2022), as acknowledged by the authors. It remains unclear whether the architecture itself contains any novel elements. Based on the current presentation, the primary novelty appears to lie in the variance loss defined in Eqs. (8)–(10), where the L2 distance to the label is used to modulate the predicted variance. If this is the case, it would be helpful for the authors to clearly emphasize this and clarify what differentiates their approach.

3. The method uses a number of hyperparameters (e.g., $\alpha$, $\beta$, $\gamma$), but their impact on performance is not ablated or discussed. Given the complexity of the setup, some sensitivity analysis or justification for the chosen values would improve the paper's reproducibility and clarity.

### Questions
1. Could you clarify the key differences between your approach and PCME/PCME++? The “similarity representation encoder” architecture shown in Figure 2 appears quite similar.  
1b. How would PCME++ perform on ImageNet under the same experimental setup as used in Table 3?
1c. What would be the performance of PCME++ if the transformer were replaced with an MLP of the same architecture and size as used in VACSR?

2. Why do you use a single-component GMM? Is there an intuition behind this choice? Was this hyperparameter ablated? What would the performance be with one or three components, keeping everything else the same?

3. How sensitive is the model to the loss hyperparameters $\alpha$, $\beta$, and $\gamma$?

4. Table 4 seems to omit a key ablation on the impact of $L_\sigma$. What would the performance of VACSR if only the $L_\sigma$ component was removed from the loss?

5. On the more challenging and cleaner ECCV Captions dataset, the reported improvements over PCME++ appear to hold primarily for the R@1 metric. Any intuition on why that is?

6. How important is passing the variance though a sigmoid?

### Soundness
3

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
4

### Summary
This paper introduces VACSR, a method designed to address the limitations of binary annotations in vision-language models. Traditional datasets often label image-text pairs as simply “matched” or “mismatched,” which can generate false negatives and hinder semantic understanding. VACSR reframes image-text matching as a variational inference problem, constructing a latent similarity space that captures fine-grained semantics beyond binary labels. It incorporates a distributional optimization loss to assign uncertainty adaptively—reducing the effect of mislabeled samples while enhancing discrimination for informative negatives. Experiments on COCO, ECCV Caption, and CxC datasets show that VACSR improves both retrieval accuracy and diversity while using fewer parameters than prior probabilistic embedding methods.

### Strengths
1. The paper is well-structured. It effectively builds from a theoretical analysis of the limitations of standard losses to a well-explained solution using variational inference.
2. The paper gives rigorous gradient-level analysis of standard losses, which provides a solid theoretical foundation for the proposed method and its novel distributional optimization loss.

### Weaknesses
1. The novelty is limited. The core problem addressed by VACSR is learning fine-grained similarities between images and texts, which is fundamentally a local-level image-text matching problem. This area has been extensively studied in the last, representative works such SGRAF [1], IMRAM [2], and PVSE [3], which explicitly model the alignment between image regions and text phrases to capture detailed semantic relationships. Therefore, the paper fails to establish a strong motivation for why a new solution is needed in this research landscape, and the innovation appears incremental rather than foundational.
2. The paper claims to address the false negative issue inherent in binary annotations, which is a form of label noise. However, the experiments lack a controlled noisy correspondence setting, which has been a standard evaluation in prior works like PCME++. Comparisons with other robust learning methods such as NCR [4], Bicro [5], NPC [6] should be added.
3. The construction of the local similarity matrix is fundamentally reliant on the feature representations extracted by the pre-trained CLIP encoder. We all know the pre-trained data of CLIP involves some MSCOCO type data, but how does it work on the out-of-distribution scenarios?

Consequently, the overall performance and generalization of VACSR are heavily reliant on the robustness of CLIP's features. Additional experiments on out-of-domain image-text matching datasets are necessary to demonstrate the generalization capability.

**References:**

[1] Haiwen Diao, et al."Similarity reasoning and filtration for image-text matching," in AAAI, 2021, pp.1218–1226.

[2] Chen Hui, et al. "IMRAM: Iterative matching with recurrent attention memory for cross-modal image-text retrieval." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

[3] Song Yale, et al. "Polysemous visual-semantic embedding for cross-modal retrieval." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019

[4] Huang et al., “Learning with noisy correspondence for cross-modal matching”, NeurIPS, 2021.

[5] Yang et al., “Bicro: Noisy correspondence rectification for multi-modality data via bi-directional cross-modal similarity consistency”, CVPR, 2023.

[6] Zhang et al., “Negative pre-aware for noisy cross-modal matching”, “AAAI”, 2024.

### Questions
Please refer to the weaknesses.

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
3

### Summary
This paper introduces VACSR (Variational Adapter for Cross-Modal Similarity Representation), a method that reformulates image–text matching under sparse binary annotations as a variational inference problem. Instead of treating similarity as a fixed scalar, VACSR models it as a latent Gaussian distribution, thereby capturing fine-grained semantics and mitigating the adverse gradient effects caused by false negative pairs. By optimizing the ELBO over the similarity matrix, the model learns to align the regular dot-product similarity with a distributional representation, which then guides the pair-based sigmoid loss. Evaluation results on three image-text retrieval tasks and out-of-distribution tasks show the effectiveness and generalizability of the proposed method.

### Strengths
1. There is a very rich set of experiments that is well executed. It not only thoroughly evaluates the performance of the proposed method, but also investigates multiple aspects of the design.
2. By mapping the point-wise similarity score to a distribution can help capturing non-binary semantics and this paper explores the idea well.
3. The formula proposed in the paper is correct, and the logic is clear and complete.

### Weaknesses
1. **Marginal improvements from KL regularization.**
As shown in Table 4, the gain from adding the KL loss is relatively small and in some cases the variant without KL even slightly outperforms the full model. It’ll be better to provide more analysis.
2. **Clarity of Figure 2.**
The model diagram is rather dense, with multiple symbols and blocks in close proximity. It’ll be better to simply it and show the high-level idea more clearly.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
