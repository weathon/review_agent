# Contrastive Order Learning for Ordinal Regression

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 4, 4, 8

## Abstract
We propose a novel contrastive learning algorithm for ordinal regression, called contrastive order learning (ConOrd), which combines the strengths of order learning and contrastive learning effectively. While contrastive learning excels at leveraging all samples in a batch, it often overlooks the inherent order among rank labels. Conversely, order learning superbly captures label ordinality but typically relies on local, margin-based comparisons, limiting its global consistency and representation power. ConOrd addresses these limitations by introducing a contrastive order loss, which adopts soft affinity and disparity weights based on rank differences, enabling fine-grained modeling of ordinal relationships across all sample pairs in a batch. Moreover, to enhance the embedding space further, we incorporate a center loss with learnable reference points, which guide compact clustering and ordinal alignment. Extensive experiments on various ordinal regression tasks, including facial age estimation, blind image quality assessment, and blind video quality assessment, demonstrate that the proposed ConOrd consistently provides state-of-the-art results and generalizes well to diverse ordinal regression scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a novel algorithm for ordinal regression called Contrastive Order Learning (ConOrd), which aims to combine the strengths of contrastive learning and order learning. Its core contribution is a contrastive order loss, which uses "soft" affinity and disparity weights, enables comparison among all sample pairs within a batch. The authors validate ConOrd through extensive experiments on diverse ordinal regression tasks, including facial age estimation, blind image quality assessment, and blind video quality assessment.

### Strengths
1 Clear Motivation and Problem Formulation. The paper is well-motivated and clearly articulates the respective limitations of standard contrastive learning (ignores order) and traditional order learning (local comparisons, limited batch exploitation). The proposed ConOrd method is a logical and well-designed solution to bridge this gap.
2 Novel and Intuitive Loss Function. The primary strength is the novel ConOrd loss. The use of soft affinity and disparity weights based on rank differences is an elegant and effective mechanism. It intuitively captures the ordinal structure by applying fine-grained attraction and repulsion forces across all pairs, which is a significant improvement over methods like RnC that use hard, binary rank-based thresholds.
3 Exceptional Empirical Performance and Generalizability. The extensive experimental validation covers a comprehensive range of tasks. The method's effectiveness is demonstrated by its outperformance across a wide array of benchmarks, from age estimation to perceptual quality scoring and physical measurements. This strongly supports the algorithm's generalizability.

### Weaknesses
1 Limited Scope of Contribution: The paper's core technical contribution is a new loss function, $L_{ConOrd}$. As clearly illustrated in Figure 1, this approach is designed to overcome the limitations of standard supervised contrastive learning (SupCon), which ignores ordinal relationships, and existing ordinal contrastive learning methods like RnC, which use a "hard-assigned" sample selection strategy that potentially overlooks valuable training pairs. ConOrd's use of soft-weighting to incorporate all pairs in a batch is a notable and intuitive refinement. However, restricting the primary contribution solely to an improved loss function raises concerns about the scope and breadth of the work. The work currently presents as a valuable but potentially incremental improvement to the contrastive learning paradigm for ordinal tasks.
2 Tenuous Connection to the Order Learning Field: The paper positions itself as "unifying the strengths of order learning (OL) and contrastive learning (CL)." However, this connection to the established order learning literature appears weak and superficial. The method's derivation, as presented in the motivation (Section 3.2), begins by analyzing the shortcomings of a standard contrastive learning method when applied to ordinal tasks. The entire approach feels more like 'injecting ordinal-awareness into contrastive learning' rather than a deep, principled fusion with the foundational theories and methods of ordinal learning.
3 Unclear Contribution of $L_{center}$: The paper's core method combines $L_{ConOrd}$ with $L_{center}$, making it difficult to isolate the true performance contribution of $L_{ConOrd}$ itself. The ablation in Table 5 compares $L_{ConOrd} + L_{center}$ against a baseline $L_{RnC}$ , but lacks an experiment evaluating $L_{RnC} + L_{center}$. If this combination also significantly outperforms the $L_{RnC}$ baseline, it would suggest that the $L_{center}$ regularization is a primary driver of the performance gains. To properly disentangle these effects and validate the superiority of the proposed loss, the authors should provide a more comprehensive ablation, ideally comparing $L_{SupCon}$, $L_{RnC}$, and $L_{ConOrd}$ both with and without the $L_{center}$ component.

### Questions
1 Could you please elaborate on the connections between the proposed method and Order Learning (OL)? Specifically, how is the proposed method rooted in OL, or what direct inspirations were drawn from foundational OL theories?
2 Please provide a more comprehensive ablation comparing $L_{SupCon}$, $L_{RnC}$, and $L_{ConOrd}$ both with and without the $L_{center}$ component.

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
5

### Summary
The paper proposes a novel contrastive order learning algorithm, ConOrd, designed for the challenging task of Ordinal Regression. The method cleverly integrates the benefits of supervised contrastive learning with fine-grained ordinal constraints. This is achieved by introducing rank difference-based affinity ($a_{ij}$) and disparity ($b_{ij}$) weights to precisely model the ordinal relationships among all samples within a batch, promoting better rank alignment and clustering in the embedding space. Furthermore, the authors introduce a Center Loss ($L_{\text{center}}$) to regularize the embedding space by pulling samples of the same rank toward their respective centers. Extensive experiments on diverse ordinal regression tasks, show that ConOrd consistently outperforms existing state-of-the-art methods, demonstrating its effectiveness and generalization ability. However, while the empirical results are strong, the paper needs more rigorous theoretical grounding and a more comprehensive set of baseline comparisons to fully justify its novelty and complexity.

### Strengths
1. The ConOrd method consistently and stably achieves state-of-the-art results across multiple benchmark datasets, including MORPH II, CLAP2015, LSQV-1080P, and KonVid-1k. Its superior performance is particularly evident in the BIQA and BVQA tasks.

2. The core innovation lies in the $L_{\text{ConOrd}}$ loss, which skillfully combines contrastive learning with ordinal structure via fine-grained, rank-aware weighting. The design of the affinity and disparity weights ($a_{ij}, b_{ij}$) that utilize continuous rank differences is an elegant solution compared to methods relying only on binary pairs or fixed margins.

3. The model's efficacy is demonstrated across three distinct ordinal regression domains (age estimation, image quality, and video quality), underscoring its versatility and strong generalization capacity.

4. The inclusion of the Center Loss ($L_{\text{center}}$) is shown in ablation studies (Figure 6, Table 6) to be crucial for the final performance, as it effectively regularizes the embedding space to achieve tighter clustering for samples belonging to the same rank.

### Weaknesses
1. The paper lacks theoretical analysis or proof of convergence for the proposed $L_{\text{ConOrd}}$ loss function. Given that contrastive losses can be sensitive to hyperparameter choices and sampling, providing theoretical insights into its stability, gradient properties, or generalization bounds is crucial, yet missing.

2. While comparisons are made against many existing ordinal regression methods, the paper lacks a comprehensive and fair comparison against the latest Transformer-based or large-scale pre-trained model baselines. For instance, a side-by-side comparison using the same ViT-B backbone (only mentioned briefly on page 7) across all tasks against SOTA methods is absent, reducing the overall persuasiveness of the performance gains.

3. The $L_{\text{ConOrd}}$ loss relies on several critical hyperparameters ($\epsilon, \gamma, \alpha_0, \beta_0$). The ablation study (Table 7, page 9) only shows 8 variations of the loss structure. A thorough sensitivity analysis (ideally a visual plot) for key parameters like $\epsilon$ and $\gamma$ is necessary to understand their impact on final performance and training stability.

4. The specific functional forms used for the affinity and disparity weights (Equations 4 and 5), such as $(|r_i - r_j| + \epsilon)^{-1}$, are clever but their mathematical derivation and justification lack depth. The authors should provide a clearer, intuitive argument for why these specific inverse distance formulations are superior to other potential weighting schemes.

### Questions
1. Regarding the $L_{\text{ConOrd}}$ loss function, how did the design of the weights $a_{ij}$ and $b_{ij}$ ensure effective suppression of gradient explosion and maintain training stability? Please provide a more in-depth discussion, either through theoretical arguments or analysis of the gradient dynamics.

2. In Table 7, you present 8 alternative configurations for $L_{\text{ConOrd}}$. Could you provide a dedicated sensitivity analysis plot for the key hyperparameters (e.g., $\epsilon$ and $\gamma$) to visually demonstrate their influence on the final performance and convergence speed?

3. Given the importance of $L_{\text{center}}$, please discuss the impact of the initialization method of the center points $\mu_m$ on the final results. What is the observed difference in performance if the centers were initialized randomly instead of being pre-calculated based on the training set?

4. Whether the proposed method can provide meaningful visual explanations, such as grad-cam?

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
This paper presents Contrastive Order Learning (ConOrd), a representation-learning framework for ordinal regression problems. ConOrd incorporates (1) a contrastive order loss that applies soft affinity and disparity weights derived from rank differences, and (2) a center loss that encourages compact feature clustering. Extensive experiments across multiple ordinal regression benchmarks demonstrate that ConOrd consistently achieves state-of-the-art performance.

### Strengths
+ The proposed method works well on multiple benchmarks.
+ The proposed loss is plug-and-play, which can be integrated with other backbones.
+ It excels at various modalities.

### Weaknesses
+ Methodological novelty is somewhat limited as it mainly combines constrative learning and ordinal learning. The center loss is adopted from the literature.
+ During the inference stage, the test-time k-NN may not be stable. 
+ Statistical analysis may be needed as there is only average performance.

### Questions
1. I am curious about the effectiveness of the test-time k-nn and the effect of k.
2. More informative comparisons, such as standard deviation and computational cost, could be reported.
3. The author should justify the reason for the L_conord configuration. Any theoretical consideration?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
A method is proposed to model ordinal regression using contrastive learning.  The key innovation is modifying the InfoNCE loss to include two factors: 1) the affinity of the two samples, and 2) the distance between the two samples.   The idea is to minimize the expected rank estimation error with respect to the contrastively learned probability distribution.  A key advantage is that all the samples in the rank order can be used, without needing to separate them into positive and negative groups, as in current contrastive learning methods.    

The second innovation is to introduce a center loss, which requires all samples within rank m to be closer to each other in the contrastively learned image feature space.  

Experiments on the standard dataset show that this method works well.

### Strengths
This paper introduced a simple but effective solution for extending contrastive learning to rank regression.  The idea and implementation of minimizing the expected rank estimation error with respect to the contrastively learned probability distribution are good ones.  The centering loss is a simple addition, and it is shown to be very effective.  

Together, the method simplifies many ad hoc designs, where one has to specify positive vs negative pairings in the contrastive learning.  The centering loss might replace the `momentum contrastive' implementation, which is used in all current contrastive learning methods. 

The paper is well presented, and the figures are precise.

### Weaknesses
In practical applications, absolute ordering training data are often difficult to obtain and may contain errors.  It would be helpful to address how robust this method is to incomplete ordering information or to errors in training data.   This is a concern since this method employs a disparity measure that assigns large weights to distant pairs.  

A more detailed comparison to Khosla et al., 2020, RnC algorithm (Zha et al., 2023), in terms of the embedding features learned would be helpful.

### Questions
See above weakness.

### Soundness
3

### Presentation
3

### Contribution
3
