# Multimodal Cancer Survival Analysis with Learnable Queries

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Leveraging multimodal data, particularly the integration of whole-slide histology images (WSIs) and transcriptomic profiles, holds great promise for improving cancer survival prediction. However, excessive redundancy in multimodal data poses a critical challenge for model optimization and can become prohibitive. Thus, methods that effectively reduce redundancy are highly desirable. While previous approaches have achieved impressive results by clustering redundant representations, they still rely on additional prior knowledge, which limits their flexibility in capturing dynamic data changes and emerging patterns. To resolve this drawback, we propose a novel and effective approach, SurvQ, for multimodal cancer survival analysis with learnable queries, which adaptively learns representative features in a data-driven manner, reducing redundancy while preserving critical information. Our method employs two sets of learnable query vectors that serve as a bridge between high-dimensional representations and survival prediction, capturing task-relevant features. Additionally, we introduce a multimodal mixed self-attention mechanism to enable cross-modal interactions, further enhancing information fusion. Extensive experiments on five benchmark cancer datasets demonstrate that our method consistently outperforms state-of-the-art approaches, achieving the best average performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces SurvQ, a novel framework for multimodal cancer survival analysis that addresses the critical challenge of information redundancy in histology (WSI) and genomic data. Unlike previous methods that rely on predefined, knowledge-based prototypes to cluster data, SurvQ employs a data-driven approach using "learnable queries." The model utilizes two sets of learnable query vectors—one for histology and one for genomics—that interact with high-dimensional patch and pathway tokens via cross-attention. This mechanism acts as an adaptive information bottleneck, distilling vast and redundant inputs into a compact, task-relevant set of representative features. These learned queries are then fused using a unified multimodal mixed self-attention module to capture complex cross-modal interactions efficiently. Experiments conducted on five benchmark TCGA cancer datasets show that SurvQ achieves superior predictive performance.

### Strengths
1. Architectural Elegance and Efficiency: The proposed architecture is both effective and relatively simple. By using queries to drastically reduce the number of tokens before fusion, it allows for the use of a single, powerful "multimodal mixed self-attention" mechanism. 

2. Superior Empirical Performance: The method's effectiveness is strongly supported by the results.

3. Good Interpretability: The paper provides clear visualizations demonstrating that the learnable queries capture biologically meaningful information.

### Weaknesses
1. Fixed Number of Queries: A limitation, acknowledged by the authors, is that the number of learnable queries for each modality is a fixed hyperparameter. 
2. Reducing the redundancy of the attention mechanism through learnable queries is common in other fields[1]. I think this design is not novel enough.
[1] Arar M, Shamir A, Bermano A H. Learned queries for efficient local attention[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 10841-10852
3. There is no ablation experiment on the number of queries
4. The article highlights the redundancy in the input but does not provide a detailed analysis of training time or computational resource usage compared to other methods (e.g. flops), which makes it difficult to quantify whether the redundancy is addressed.
5. There are relatively few baselines for comparison, and most are not from 25 years ago.

### Questions
1. Fixed Number of Queries: A limitation, acknowledged by the authors, is that the number of learnable queries for each modality is a fixed hyperparameter. 
2. Reducing the redundancy of the attention mechanism through learnable queries is common in other fields[1]. I think this design is not novel enough.
[1] Arar M, Shamir A, Bermano A H. Learned queries for efficient local attention[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 10841-10852
3. There is no ablation experiment on the number of queries
4. The article highlights the redundancy in the input but does not provide a detailed analysis of training time or computational resource usage compared to other methods (e.g. flops), which makes it difficult to quantify whether the redundancy is addressed.
5. There are relatively few baselines for comparison, and most are not from 25 years ago.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper focuses on the problem of excessive redundancy in multimodal data in multimodal cancer survival analysis. Previous methods rely too much on prior knowledge, limiting the flexibility in capturing dynamic data changes and emerging patterns. This paper proposes SurvQ, conducting multimodal cancer survival analysis with learnable queries. By adaptively learning representative features in a data-driven manner, the method can reduce redundancy while preserving important information in multimodal data.

### Strengths
1. The proposed method is simple but effective. The approach generalizes the concept of query-based token compression (from DETR/BLIP-2) to the medical multimodal setting, replacing handcrafted prototype-based reductions with a data-driven mechanism.
2. This paper is well-written and easy to follow. The figures are clear.
3. The visualization of histology and genomic queries provides biological insight into the model’s internal representations.

### Weaknesses
1. While the empirical results are strong, the paper lacks deeper theoretical or information-theoretic analysis explaining why query-based bottlenecks improve generalization or reduce redundancy.
2. The evaluation is restricted to cancer survival prediction on TCGA datasets. It would strengthen the contribution to demonstrate the general applicability of SurvQ to other multimodal biomedical tasks.

### Questions
1. Are there any observed correlations between specific queries and known clinical or molecular subtypes?

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
4

### Summary
This paper proposes SurvQ, a multimodal cancer survival analysis framework that integrates whole-slide histology images (WSIs) and transcriptomic profiles using learnable queries. Specifically, the authors introduce two sets of learnable query vectors that interact with unimodal features via cross-attention to extract representative pathology and genomic features, followed by a mixed self-attention module for multimodal fusion. The method is evaluated on five TCGA datasets and demonstrates better performance compared to prior multimodal and prototype-based methods.

### Strengths
1. Data-driven prototype learning via learnable queries provides a simple yet effective solution.
2. The model achieves better results across multiple TCGA cohorts, clearly outperforming both unimodal and prior multimodal baselines.
3. Visualization of attention maps and the top 6 pathways provides some interpretability, linking learned queries to meaningful histological and molecular patterns.

### Weaknesses
1. Conceptually incremental — mainly adapting the learnable-query idea from prior works (e.g., BLIP-2, DETR) to this setting.
2. The baseline design of the ablation study is weak; it doesn’t isolate the effect of “learnability.” A fixed/random query baseline would make the comparison fairer.
3. Interpretability focuses on unimodal patterns, cross-modal interactions (e.g., which histology regions relate to which genomic pathways) are not explored.
4. The number of queries is fixed for all datasets, which may not be optimal.

### Questions
1. Do the authors employ any mechanism (e.g., orthogonal regularization,  contrastive objectives) to encourage diversity or competition among the learnable queries? Otherwise, queries may collapse to similar representations.
2. Could the authors show cross-modal attention maps to verify the interaction between modalities?
3. Have the authors evaluated the computational cost or memory benefit of query compression? A quantitative comparison of efficiency would make the “redundancy reduction” claim more concrete.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work proposes to utilize two sets of learnable queries to extract representative features via cross-attention, while multimodal mixed self-attention is leveraged to model cross-modal interactions.

### Strengths
1. The motivation is clear, and the manuscript is well-written.
2. The performance of the proposed method is superior to SOTA approaches.
3. The effectiveness of each component is validated by ablation studies.

### Weaknesses
1. The novelty is incremental and very simple. The core idea follows PIBD by using a set of learnable parameters to capture representative features for each modality. The difference is that PIBD enforces a risk level constraint to assist the model in learning discriminative features, while there is no prior constraint in the proposed method. Additionally, the idea of learnable queries has been explored in G-HANet [1], which has validated its effectiveness.
2. The insight about why it works is lacking. Given that there is no explicit constraint for the modelling, although the story is well-told, I'm still confused about why it achieves the intended purpose without any explicit guidance or constraint, and why it is better than PIBD with explicit conditions.

[1] Wang Z, Zhang Y, Xu Y, et al. Histo-genomic knowledge association for cancer prognosis from histopathology whole slide images[J]. IEEE Transactions on Medical Imaging, 2025.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2
