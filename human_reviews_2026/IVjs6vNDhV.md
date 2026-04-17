# GenCape: Structure-Inductive Generative Modeling for Category-Agnostic Pose Estimation

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Category-agnostic pose estimation (CAPE) aims to localize keypoints on query images from arbitrary categories, using only a few annotated support examples for guidance. Recent approaches either treat keypoints as isolated entities or rely on manually defined skeleton priors, which are costly to annotate and inherently inflexible across diverse categories. Such oversimplification limits the model’s capacity to capture instance-wise structural cues critical for accurate pixel-level localization. To overcome these limitations, we propose \textbf{GenCape}, a \textbf{Gen}erative-based framework for \textbf{CAPE} that infers keypoint relationships solely from image-based support inputs, without additional textual descriptions or predefined skeletons. Our framework consists of two principal components: an iterative Structure-aware Variational Autoencoder (i-SVAE) and a Compositional Graph Transfer (CGT) module. The former infers soft, instance-specific adjacency matrices from support features through variational inference, embedded layer-wise into the Graph Transformer Decoder for progressive structural priors refinement. The latter adaptively aggregates multiple latent graphs into a query-aware structure via Bayesian fusion and attention-based reweighting, enhancing resilience to visual uncertainty and support-induced bias. This structure-aware design facilitates effective message propagation among keypoints and promotes semantic alignment across object categories with diverse keypoint topologies. Experimental results on the MP-100 dataset show that our method achieves substantial gains over graph-support baselines under both 1- and 5-shot settings, while maintaining competitive performance against text-support counterparts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces GenCape, a novel generative framework for Category-Agnostic Pose Estimation (CAPE) that learns structural relationships directly from support images. The key innovation lies in automatically inferring keypoint connectivity patterns (soft adjacency matrices) without requiring predefined skeleton graphs, keypoint identifiers, or text descriptions. The framework comprises two main components: The Iterative Structure-aware Variational Autoencoder (i-SVAE) learns instance-specific graph structures from support features using variational inference, with iterative refinement across decoder layers. The Compositional Graph Transfer (CGT) module then dynamically combines multiple graph hypotheses through Bayesian fusion and query-guided attention mechanisms.

### Strengths
This is the first CAPE method to achieve fully automatic learning of structural relationships from image support sets, removing the need for predefined skeletons, keypoint IDs, or text descriptions, which enhances both generality and practical deployment. The i-SVAE approach models structural uncertainty through variational inference, demonstrating superior robustness compared to discriminative methods like SDPNet, particularly when handling support-query mismatches or occlusion scenarios. The method consistently outperforms various baselines on MP-100, with particularly notable advantages under strict thresholds (e.g., PCK@0.05).

### Weaknesses
1. While the paper claims to evaluate cross-supercategory generalization on MP-100, the definition of supercategories appears inconsistent with the original MP-100 benchmark and prior CAPE literature (e.g., CapeFormer).
The original MP-100 dataset is widely understood to group categories into four high-level semantic domains: human body, human/animal face, vehicle, and furniture. However, this work instead uses a finer-grained 8-supercategory split (e.g., separating Felidae, Canidae, Ursidae as distinct supercategories), which blurs the line between "cross-category" and "cross-subcategory" generalization. For instance, transferring from Felidae to Ursidae involves structurally similar quadruped animals with comparable keypoint layouts—this is arguably intra-domain transfer, not the more challenging cross-domain shift (e.g., chair → human) that truly tests category-agnostic capability. Worse, the paper does not include any cross-domain transfer between the canonical four domains (e.g., furniture → human body). This omission is critical, as if the method cannot generalize from chair to person, its claim of "structure-inductive" modeling is significantly weakened.

2. The paper does not provide any comparison of computational efficiency, such as inference time, FLOPs, model size, or throughput, against baseline methods like GraphCape or CapeFormer. While it introduces additional modules (i-SVAE and CGT) that likely increase computational cost, no quantitative analysis or efficiency trade-offs are reported.

### Questions
Figures 4 and 5 reveal some localization errors. What is the primary cause of these errors, structural inference failures or visual feature ambiguity? 

Could the authors provide quantitative analysis of these failure modes? 

What is the method's robustness to scale variations, cropping, and other common transformations? 

Have additional evaluation metrics beyond PCK been considered, such as AUC or other standard pose estimation metrics?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This works suggests solving CAPE by progressivley inferring instance-specific keypoint relationships from the support, instead of using predefined annotated adjacency matrices. The authors also introduce the Compositional Graph Transfer module that aids with incoroporating the query features, thus allowing for less reliance on inferred keypoint relationships from the support. This makes the model more robust to occlusions and discepencies between the support and query. The new GenCape approach is tested on the known MP-100 benchamrk, achieving SOTA results.

### Strengths
1. The paper is written in a clear language that was easy to follow.

2. While predicting the keypoints relations from the data is not new, the novel i-SVAE and CGT components suggest some interesting insights that might interest the CAPE community.

3. The suggested approach achieves SOTA while dropping the need for predefined annotated data (keypoint connectivity) that was used by previous methods.

4. Other than the main experiment in Table 1, the design choices are justified in the ablations conudcted (Table 4, Table 5 and Table 6).

### Weaknesses
My main issues are with the presentation, not with the method. After resolving these issues, I would positively consider increasing my rating.

1. The technical text (mostly) in the Methods section:

Line 157: M_C is not defined in the right place. Move it to this sentence.

Line 179: remove 1 between.

Lines 190-195: i-SVAE also infers graphs from the support. And as you mention, there is sometimes a discrepency between the support and the query. So I'm not sure that i-SVAE alone will solve the issue mentioned in these lines. However, i-SVAE combined with CFG will.

Line 208: F_s^(l-1) is not defined properly - what is its value where l=1?

Line 212: in the second row of Equation 1, should this be F_s^(l-1) or F_s^(l)?

Equations 3 and 4: A^~(l) is defined twice?

Equation 6: F_s^(l) is in the input and output

Line 244: A^~(l) - different notation compared to Equation 3. Should be/not be in bold?

Line 248: This is not clear. Do keypoint locations are predicted in each layer? Each layer of what? the Graph Transformer Decoder? Clarify what is the output of each layer in the Graph Transformer Decoder.

Line 275: missing ')' in mu^(l 

Line 377: "More detailed comparisons.": should be a sperate paragraph?

2. Figures:

Figure 2: Consider adding CGT to Figure 2 (A^l_fused is not enough to easily follow).

Figure 3: It is challenging to interpret the adjacency matrices. Consider showing the “best” links from the adjacency matrix as colored edges in your prediction.

Figure 4: last column is AutoCape.

### Questions
You mentioned Text Graph Support as an approach for CAPE. Will fusing text with your approach might increase performance? Could you hint on how whould you incorporate text as a future work with your approach (maybe also infer it from the support?).

### Soundness
4

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
5

### Summary
The paper suggests a novel CAPE method, which utilizes predicted graph structure for enhanced keypoint localization accuracy.
The method uses a graph VAE formulation to predict the graph, and further implements it iteratively within each decoder layer.
Using CGT, several sampled graphs are combined into a query-aware graph structure that aids in localization.
The authors show competitive results on the MP100 dataset.

### Strengths
- The paper suggests a novel method that deals with a limitation of recent graph-based methods.
- The paper is well written, and the proposed solution looks solid and practical.
- SOTA results compared to other CAPE methods on the MP100 dataset.

### Weaknesses
- Using only Fs to predict the adjacency matrix suggests that the structure information is embedded in Fs in the first place.
As self-attention can be seen as an all-to-all information sharing mechanism, an explanation of why self-attention can't learn the relevant connections between keypoints should be added or even proven.
Specifically, the authors should explain how the current i-SVAE design adds to the self-attention already in the decoder.

- Iterative Graph Prediction - The suggested method works iteratively, predicting a different adjacency matrix for each decoder layer. 
An ablation study, showing why using a different adjacency matrix in each decoder layer, compared to using only one predicted adjacency matrix (using the output features of the encoder, for example), should be presented to support the iterative superiority claim.

- Qualitative skeleton visualization - Figure 3 is hard to understand. 
It would be helpful to add the skeleton visualizations on top of the images, and not only show the adjacency matrix.
Maybe the width or opacity can correspond to the weight. It is crucial to make it easier to understand what structure is actually learned.

Small Note:
- Figures 4 and 5 label your method as AutoCape instead of GenCape.

### Questions
- CGT - the adjacency matrices are sampled using the predicted mean and variance. Thus, it's not clear to me why each sample has its own mean and variance values (line 263), given that they are sampled from the same distribution.
This is further shown in equation 9, where alpha_n is not dependent on n at all.

- See weaknesses for other questions.

 I'm willing to raise my score if the authors address my concerns.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The submission focused on the task of category-agnostic pose estimation with few annotated example images. Specifically, the authors propose a novel generative-based framework named GenCape to estimate keypoints without additional textual descriptions or predefined skeletons. The authors propose an Structure-aware Variational Autoencoder to infer instance-specific adjacency matrices from support features, and also propose a Graph Transformer Decoder to progressively refine the estimated results. The experiments are conducted on a large-scale benchmark datasets, indicating the effectiveness of the proposed novel framework.

### Strengths
1. The task of category-agnostic pose estimation is interesting and fundmental for extending the category number of pose estimation.

2. The idea of using generative framework is reasonable and makes sense.

3. The proposed Structure-aware Variational Autoencoder and Compositional Graph Transfer are novel and effective to model the pose structure information.

4. The performances of proposed framework are shown on large-scale benchmark dataset, and outperform SOTA dramatically.

5. The experimental analyses are comprehensive and clear.

### Weaknesses
1. As discussed in Line 69-74, the support images may contain severe occlusions or incomplete annotations, and how does the proposed method address this issue? E.g., the query image has 2 occluded keypoints, while the support image has another 3 occluded keypoints. Can the proposed method estimate all the visible keypoints?

2. What's the complexity of proposed framework? It seems to be about O(M^2). Is the proposed method cost-effective?

3. Can the proposed method produce diverse results based on VAE sampling? How to understand the contradiction of diversity v.s. consistency in the proposed VAE-based method?

4. Closely related works are missing in the related works.&#x20;

   > 1. @inproceedings{chen2025weakshot, title={Weak-shot Keypoint Estimation via Keyness and Correspondence Transfer}, author={Chen, Junjie and Luo, Zeyu and Liu, Zezheng and Jiang, Wenhui and Li, Niu and Fang, Yuming}, booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}, year={2025} }
   >
   > 2. @inproceedings{lu2024openkd,   title={Openkd: Opening prompt diversity for zero-and few-shot keypoint detection},   author={Lu, Changsheng and Liu, Zheyuan and Koniusz, Piotr},   booktitle={European Conference on Computer Vision},  year={2024} }

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
3
