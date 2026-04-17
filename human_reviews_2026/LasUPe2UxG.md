# DecAlign: Hierarchical Cross-Modal Alignment for Decoupled Multimodal Representation Learning

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Multimodal representation learning aims to capture both shared and complementary semantic information across multiple modalities. However, the intrinsic heterogeneity of diverse modalities presents substantial challenges to achieve effective cross-modal collaboration and integration. To address this, we introduce DecAlign, a novel hierarchical cross-modal alignment framework designed to decouple multimodal representations into modality-unique (heterogeneous) and modality-common (homogeneous) features. For handling heterogeneity, we employ a prototype-guided optimal transport alignment strategy leveraging gaussian mixture modeling and multi-marginal transport plans, thus mitigating distribution discrepancies while preserving modality-unique characteristics. To reinforce homogeneity, we ensure semantic consistency across modalities by aligning latent distribution matching with Maximum Mean Discrepancy regularization. Furthermore, we incorporate a multimodal transformer to enhance high-level semantic feature fusion, thereby further reducing cross-modal inconsistencies. Our extensive experiments on four widely used multimodal benchmarks demonstrate that DecAlign consistently outperforms existing state-of-the-art methods across five metrics. These results highlight the efficacy of DecAlign in enhancing superior cross-modal alignment and semantic consistency while preserving modality-unique features, marking a significant advancement in multimodal representation learning scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the challenge of multimodal representation learning under strong modality heterogeneity. The authors argue that conventional fusion methods (e.g., concatenation) improperly mix modality-unique and modality-common semantics, leading to semantic interference and weak cross-modal alignment. To address this, the paper proposes DecAlign, which is a hierarchical decoupling–alignment framework that explicitly separates multimodal features into modality-unique and modality-common streams. The authors demonstrate DecAlign's effectiveness on four multimodal benchmarks (CMU-MOSI, CMU-MOSEI, CH-SIMS, and IEMOCAP) , which consistently outperforms 13 state-of-the-art methods across multiple metrics.

### Strengths
- The paper provides sufficient experimental results, supplementary and codes for reproducibility;
- The model's superiority is convincingly demonstrated across four standard multimodal sentiment and emotion datasets. DecAlign also outperforms a wide array of 13 baseline methods , with significant performance gains shown in Tables 1, 5, 6, and 7;
- The paper provides extensive ablation studies that validate its complex design;

### Weaknesses
- The main concern is about the complexity of the proposed framework. The final framework is a combination of many advanced techniques: specialized encoders , a cosine similarity decoupling loss , GMM fitting , multi-marginal optimal transport , a multimodal transformer , latent moment matching , and regularization. While the ablations show that most parts may be effective, this complexity causes a high computational efficiency and result reproducibility;
- The ablation study in Table 2 (right) introduces a component labeled "Contrastive Training (CT)”. What is this loss function? How is it formulated?
- The paper uses GMMs to find prototypes and then use multi-marginal OT to align them. GMMs typically require an iterative fitting process (like EM ) which can be slow and sensitive to initialization. Are there any specific reasons to choose GMM for clustering?
- One open question: the current model is mainly evaluated exclusively on four relatively small-scale, domain-specific sentiment and emotion datasets (CMU-MOSI, CMU-MOSEI, CH-SIMS, IEMOCAP). It is unclear how this framework, particularly the GMM fitting and multi-marginal OT components, would scale to larger, more general-purpose vision-language tasks (e.g., large-scale VQA or captioning).

I would like to see the author rebuttal in terms of the weakness & questions part.

### Questions
Please see weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies multimodal learning problem using hierarchical cross-modal alignment framework, which decouples multimodal features into modality-heterogeneous and modality-homogeneous components. In particular, the multimodal samples are fed into modality specific encoders, followed by unimodal encoder and common encoder for extracting heterogeneous features and homogeneous features. To decouple the heterogeneous and homogeneous feature, decoupling loss is proposed, which minimizes the cosine similarity. Then, heterogeneous features are modeled by Gaussian mixture model and homogeneous features are modeled via latent Gaussian variable. 
Total loss consists of the decoupling loss, heterogeneous and homogeneous losses, and task related loss. 
Experiments verify that DecAlign outperforms baseline models in benchmark multimodal datasets.

### Strengths
The paper considers important aspects of multimodal learning, especially homogeneity and heterogeneity. The decoupling approach for better capturing homogeneity and heterogeneity is worth looking for in multimodal learning society. Experimental results show that the proposed DecAlign method has potential in multimodal learning problems.

### Weaknesses
While I enjoyed reading the paper, I have some questions and comments. From my understanding, the DecAlign works for supervised problem, where target tasks are already known before the model is training as the task loss and heterogeneity loss requires some level of task information. But the title is multimodal representation learning, which I think has not been shown how good the learned representation is. Although the benchmark task performance is provided, transfer performance or zero-shot performance have not been shown. So I am also doubt that if the performance gain comes from the decoupling procedure or other factors such as use of more number of parameters. The paper would be more convincing if it is shown that the decoupling actually makes better capturing the shared semantical representation. 

Moreover, from my understanding the heterogeneity features capture modality-unique information which is not shared among modalities. But section 3.2 says "they frequently carry semantically aligned information when referring to the same underlying concept or object category", which is counter intuitive and hard to get the idea. More specifically, "modality-unique feature differences while preserving shared semantic structures" seems does not make sense. Every multimodal samples have their unique features but contains shared semantic structures. So, modality-unique feature should refer to unique semantic structure or meaning only in a single modality. This makes confusion whether the proposed method's superiority stems from the actual intuition the authors made or form other latent factors has not been discovered. 

Other questions are also in the next questions sections.

### Questions
1. Could the author provide representation quality from DecAlign? For example, does DecAlign still well perform for transfer learning or zero-shot learning performance?
2. Setting the K equal to the number of categories is understandable, but what if we set smaller or greater numbers for $K$? Is the exact number of categories truly important for GMM?
3. Regarding $K$ again, how to set it if we target regression task as $\mathcal{L}_{\rm task}$ can be MSE loss as shown in eq (13).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes DecAlign, a hierarchical framework for multimodal representation learning that explicitly decouples modality-unique (heterogeneous) and modality-common (homogeneous) features. To align heterogeneous features, the authors introduce a prototype-guided multi-marginal optimal transport strategy based on Gaussian Mixture Modeling (GMM), complemented by a multimodal transformer for fine-grained refinement. For homogeneous features, they enforce semantic consistency via latent distribution matching using both moment-based alignment (mean, covariance, skewness) and Maximum Mean Discrepancy (MMD) regularization in a Reproducing Kernel Hilbert Space (RKHS). The method is evaluated on four standard multimodal sentiment analysis benchmarks, where it consistently outperforms 13 state-of-the-art baselines across multiple metrics. Ablation studies and visualizations support the necessity of both decoupling and the dual-stream alignment design.

### Strengths
1. This paper has a clear motivation. The paper identifies a core challenge in multimodal learning, i.e., the entanglement of modality-specific characteristics and shared semantics. Further, this paper proposes a principled solution through explicit decoupling.
2. For experimental details, the authors present the implementation details and evaluate the performance of the system, which are interesting and convincing. 
3. DecAlign achieves consistent and significant improvements over strong baselines across four diverse datasets, suggesting robustness and generalizability. 
4. The ablation studies convincingly demonstrate the contribution of each component.

### Weaknesses
1. The paper lacks theoretical analysis (e.g., error bounds, convergence guarantees) for the proposed alignment losses or the decoupling objective.
2. All experiments are confined to multimodal sentiment analysis. It remains unclear whether DecAlign generalizes to other multimodal tasks.
3. The number of GMM components K is set equal to the number of downstream categories. This may not hold in unsupervised or open-world settings, limiting applicability. The sensitivity to this assumption is not explored.

### Questions
1. How does the DecAlign method perform on non-emotional multimodal tasks (such as image-text retrieval, action recognition)?
2. What happens if the number of GMM components K deviates from the true number of semantic categories (e.g., K too large/small)? Is the performance robust to this hyperparameter?
3. The decoupling loss uses cosine similarity to encourage orthogonality between unique and common features. Why not use more direct disentanglement objectives (e.g., mutual information minimization, adversarial invariance)? 
4. This paper adopts skewness to model non-Gaussianity. How much does this performance improvement compare to using only the mean and covariance?

### Soundness
3

### Presentation
3

### Contribution
2
