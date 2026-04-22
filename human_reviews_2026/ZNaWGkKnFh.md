# Self-supervised Sparse Vision Concepts for Image Understanding and Reconstruction

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Self-supervised vision encoders have become critical components of modern machine learning systems. Despite remarkable advances in image understanding, generation, and multimodal alignment, the underlying representation of visual features has remained largely unchanged, constrained by historical architectures and benchmarks. This reliance on dense feature grids introduces redundancy and limits the integration of understanding and generation. We propose a novel framework that represents images with a small number of sparse tokens in the form of low-rank matrix factorization. While mathematically simple, this formulation effectively disentangles semantic and spatial information. We demonstrate that vision-only self-supervised learning under this framework yields sparse token representations that simultaneously support high-quality image understanding, detailed pixel-level reconstruction, and fine-grained semantic understanding. Together, these results highlight sparse tokens as a promising alternative to dense grids for efficient and versatile visual representation learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a self-supervised visual representation learning framework called STELLAR, which uses low-rank matrix factorization to compress images into a small number of sparse tokens. This approach supports both high-quality image understanding and pixel-by-pixel reconstruction without manual annotation. This approach learns representations by decoupling the semantics of "what" and the spatial location of "where." It then uses self-supervised clustering and optimal transport to align concepts from different perspectives. This approach achieves superior reconstruction quality and semantic expressiveness with only eight sparse tokens, surpassing existing sparse modeling methods on multiple downstream tasks.

### Strengths
1. The paper is well-motivated, that low-rank matrix factorization is used to generate small but informative sparse visual tokens, decoupling the semantic concepts and spatial localization.

2. The method provides a relatively complete theoretical basis.

3. The experiments are conducted across various tasks, including segmentation, classification, and linear probing, showing great generalization.

### Weaknesses
1. In line 728, the paper claim that the model is initialized from MAE checkpoint. However, the MAE pre-trained model already has strong generalization properties and cannot demonstrate the effectiveness of the proposed STELLAR framework. Especially for self-supervised learning, it is particularly difficult to initialize the model from scratch. It is strongly recommended that authors provide pre-training results from scratch.

2. Several self-supervised methods need to be discussed, especially for efficiency: MoCo v3[1], SiameseIM [2], and OCL[3].

3. I wonder the number of learnable latent queries is fixed (8-24) or self-adapted? It is suggested to add ablation experiments about learnable latent queries. 

[1] Chen, Xinlei, Saining Xie, and Kaiming He. "An empirical study of training self-supervised vision transformers." ICCV. 2021.

[2] Tao, Chenxin, et al. "Siamese image modeling for self-supervised vision representation learning." CVPR. 2023.

[3] Yang, Xiaoyu, et al. "One Leaf Reveals the Season: Occlusion-Based Contrastive Learning with Semantic-Aware Views for Efficient Visual Representation." ICML 2025.

### Questions
See Weakness

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
4

### Summary
The authors propose a self-supervised framework for sparse visual representations. Specifically, the authors represent images with a small number of sparse tokens via low-rank matrix factorization that disentangles semantic and spatial information. Several additional losses, including clustering loss and set alignment loss, are also introduced. Experimental results on image reconstruction and understanding tasks demonstrate the effectiveness of the proposed method.

### Strengths
-	The proposed method is well motivated. Sparse tokens are indeed a promising way for unifying efficiency, interpretability, and semantic richness in visual representations.
-	The paper is generally well-written and easy to follow.
-	The experiments are extensive and the results seem promising, while some parts need to be improved (see weaknesses).

### Weaknesses
-	Some results are missing exact descriptions. For example, Figure 3 is a bit confusing for me. What do the three colors represent? Besides, what specific number of tokens are used in Table 1 and Table 2? It seems that the authors do not ablate the effect of the number of tokens for understanding tasks.
-	According to Table 3, it seems that adding clustering decreases the performance significantly (row 2 vs. row 3 in `Ablation model versions`). Could the authors provide a justification for this?
-	The proposed method uses optimal transport matching and Sinkhorn-Knopp algorithms, which could be potentially computationally expensive. It would be better to provide a computational cost comparison (e.g., training time, memory cost) with previous sparse representation learning methods.
-	Apart from linear probing that serves as a feature extractor, it would be better to provide full fine-tuning results as well, considering fine-tuning is also one of the important benchmarks to evaluate whether the learned representations can serve as a good initialization for downstream tasks.

### Questions
I am concerned about the questions mentioned above. Given the current status of the paper, I am leaning towards borderline reject and hope the authors could address my concerns during the rebuttal.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper propose a self-supervised learning recipe in the vision domain. The authors introduce a self-supervised framework named STELLAR that replaces dense tokens with a small set of sparse tokens. STELLAR  Specifically, STELLAR learns a tiny set of sparse concept tokens and per-patch localization weights, then approximate a feature of the given input data using them, letting the model reconstruct images and transfer semantics with just a few tokens The authors employ transport (e.g., sinkhorn) to cluster the sparse visual concepts from the dataset into prototypes. Experimental results including reconstruction, linear probing, segmentation validates the effectiveness of the proposed method.

### Strengths
* Introduce an idea of learning sparse visual concept
* Provided an ablation study on each component

### Weaknesses
* Concern on technical novelty
    * Isn't it highly dependent on the MAE initialization?

* Concern on its motivation and intuition
    * From my perspective, the motivation of the proposed method seems similar to that of SemMAE [1] since SemMAE also tried to learn the find-grained information of the semantics (e.g., information of the objects' part) of the. Could the authors clarify the difference between STELLAR and SemMAE in terms of the motivation and intuition?

* The comparison in Table 2 is not reliable. Some baselines are reported far below than the results in their original papers
    * e.g., according to Table 1 in the iBOT paper, linear probing accuracies for DINO and iBOT exceed 80.0%, surpassing all linear probing results in the Table 1 in the authors' paper
    * Note that the epochs reported in the iBOT paper are effective epochs that account for multi-crop, not the actual pre-training epochs used by iBOT or DINO. They actually pre-trained only for 300 or 400 epochs.
    * Moreover, the proposed method employ MAE model parameters for initialization, which is not fair with other baselines. Also, the total epoch should be regarded as 1600 (MAE pre-training epochs) + 150/100/50 (STELLA post-training epochs) = 1750 / 1700 / 1650.

* A lot of recent self-supervised learning methods are missing.
    * e.g., the references below [1-22]
    * The proposed method should also be compared with these methods

* Some core evaluation tasks are missing
    * e.g., Detection and segmentation on COCO

* Important comparison results are missing
    * Fine-tuning performance comparison is very important in self-supervised learning area on visual data. However, 
        * I'm also suspecting that the proposed design may improve only linear-probing performance rather than fine-tuning performance. This concern is amplified by the utilization of the MAE initialization since MAE is well-known to show strong fine-tuning performance and weak linear-probing performance.



[1] Li et al., SemMAE, NeurIPS 2022

[2] Mishra et al., CAN, arXiv 2022

[3] Baevski et al., data2vec, ICML 2022

[4] Chen et al., SdAE, ECCV 2022

[5] Assran et al., MSN, ECCV 2022

[6] Dong et al., BootMAE, ECCV 2022

[7] Baevski et al., data2vec2.0, ICML 2023

[8] Wang et al., AdPE, arXiv 2023

[9] Wu et al., ExtreMa, TMLR 2023

[10] Huang et al., CMAE, TPAMI 2023

[11] Yi et al., ConMIM, ICLR 2023

[12] Yi et al., RC-MAE, ICLR 2023

[13] Chen et al., MixedAE, CVPR 2023

[14] Tao et al., SIM, CVPR 2023

[15] Wang et al., HPM, CVPR 2023

[16] Huang et al., MIRL, NeurIPS 2023

[17] Fu et al., CrossMAE, arXiv 2024

[18] Kim et al., LUT, ECCV 2024

[19] Liu et al., dBOT, ICLR 2024

### Questions
What happens if STELLAR does not use the MAE initialization?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes STELLAR, a self-supervised learning framework designed to learn sparse visual representations from images alone. The core idea of STELLAR is based on a low-rank matrix factorization, $V = LS$. The STELLAR framework is trained using a joint objective function that includes a reconstruction loss, a sparse concept clustering loss, a set alignment loss, and a KoLeo regularization term.The authors claim that this method, using as few as 8 latent tokens, can produce a single representation that simultaneously support high-quality image understanding, detailed pixel-level reconstruction, and fine-grained semantic understanding.

### Strengths
1. The paper tries to address the important problem of redundancy in dense visual representations. The motivation to learn a single, unified representation that excels at both high-level semantic understanding and low-level reconstruction is a valuable.

### Weaknesses
1. The paper's claims of novelty are further undermined by a profound misrepresentation of its core mechanism. The authors claim to "parameterize both S and L as learnable variables" 1 as part of a "low-rank matrix factorization".1These two statements are mutually exclusive. $L$ cannot simultaneously be a set of learnable parameters and a computed output. This formulation is not matrix factorization in the algebraic sense (like NMF or SVD).The paper's actual mechanism is a standard attention operation, cloaked in the language of classical optimization. $S \in \mathbb{R}^{r \times d}$ is a set of $r$ learnable "latent query vectors". $U \in \mathbb{R}^{n \times d}$ is the dense patch-level feature map from the ViT encoder. Equation 5 is a standard cross-attention operation, where $S$ acts as the query  and $U$ acts as the key. $L$ is simply the resulting $n \times r$ attention map, normalized via softmax.The "reconstruction" $V=LS$ 1 is then just an attention-pooled representation, where $S$ (the semantic concepts) are the values (V).Therefore, calling this "low-rank convex semi-nonnegative matrix factorization" is a profound misrepresentation of a standard attention mechanism. This attempts to invent novelty where none exists.
2.  The method proposed in this paper is actually very similar to TokenLearner [1], only with different presentation. At the same time, the paper keeps claiming to learn a sparse visual representation, but in practice, it still **relies on a standard visual encoder to extract dense visual feature**. A truly sparse architecture should employ a flexible backbone that can adaptively extract visual features.

[1] TokenLearner: What Can 8 Learned Tokens Do for Images and Videos? NeurIPS 2021

### Questions
1. In Appendix A.4, the manuscript states: "For efficient training, we initialized the model from public MAE checkpoint". This is a fatal confounder. The paper is presented as a novel self-supervised learning method that learns representations from scratch. But it is a fine-tuning procedure for a different existing model (MAE) actually.

### Soundness
2

### Presentation
3

### Contribution
2
