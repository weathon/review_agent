# GOAL: Balance Multimodal Learning with Gradient Orthogonalization and Adaptive Leveraging

- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Multimodal learning, integrating the information from multiple sensory modalities, is naturally expected to outperform the single-modal counterparts.
However, the heterogeneity of multimodal data often leads to two imbalance problems that impede unimodal representation learning prior to fusion.
The first problem arises from inconsistent gradient magnitudes across modalities, and the second from opposing gradient directions in a unimodal encoder due to competing losses.
While recent progress is achieved by strengthening within-modality representations,
we identify cross-modality compatibility as another critical factor for effective feature fusion.
Jointly considering these two factors for better fusion,
we propose the Gradient Orthogonalization and Adaptive Leveraging (GOAL), a parameter-free gradient modification method.
Specifically, guided by the principle that imbalanced dependency on each modality follows the inverse relationship with prediction variance, the AL dynamically re-weights gradient magnitude by utilizing prediction entropy as a variance estimator.
Furthermore, the GO ensures a synergistic update to obtain the compatible multimodal features through the projection of conflicting gradients.
Extensive experiments across various modalities and frameworks indicate that GOAL consistently and significantly outperforms existing state-of-the-art methods, providing a plug-and-play module for multimodal optimization.
Our code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses two key problems that hinder effective multimodal learning: (1) inconsistent gradient magnitudes across modalities, and (2) opposing gradient directions within a unimodal encoder. The authors argue that for better fusion, it's crucial to consider both within-modality representation strength and cross-modality compatibility. 

Their main contribution is GOAL (Gradient Orthogonalization and Adaptive Leveraging), a parameter-free, plug-and-play gradient modification method with two components: (1) Adaptive Leveraging (AL): Dynamically re-weights gradient magnitudes for each modality based on prediction entropy, which serves as a variance estimator; (2) Gradient Orthogonalization (GO): Resolves conflicting gradients by projecting them to ensure synergistic updates and produce compatible multimodal features.

The paper demonstrates through extensive experiments that GOAL consistently and significantly outperforms existing state-of-the-art methods across various modalities and frameworks.

### Strengths
1. Proposes a novel multimodal optimization method, GOAL, which simultaneously addresses within-modality representation enhancement and cross-modality compatibility.

2. The proposed Adaptive Leveraging and Gradient Orthogonalization modules are well-designed with clear theoretical justification: AL adjusts gradient magnitudes based on prediction uncertainty, and GO alleviates modality conflicts through conflicting gradient projection.

3. GOAL is parameter-free, gradient-based, and can be used as a plug-and-play module.

4. Experiments are thorough and comprehensive, validating the effectiveness of the method.

### Weaknesses
1. The paper provides limited analysis on the quantification of gradient conflicts under different tasks, especially for complex multimodal scenarios.

2. The paper cites the analysis in [1], which suggests that “modality contribution is inversely proportional to prediction variance” to achieve optimal information fusion. Intuitively, if a modality has high prediction uncertainty (high variance), it is less reliable for the final decision and should be assigned a lower weight; conversely, a low-variance modality should receive a higher weight. However, does high uncertainty necessarily indicate low shared information across modalities? In other words, does high uncertainty also imply a low contribution?

[1] Wei S, Luo C, Luo Y. Improving Multimodal Learning via Imbalanced Learning, ICCV 2025.

### Questions
1. Does high uncertainty necessarily indicate low shared information across modalities? In other words, does high uncertainty also imply a low contribution?

2. Some works such as KuDA [2] and EMOE [3] also investigate modality contributions. What are the relative advantages and limitations of GOAL compared to these methods?

[2] Knowledge-Guided Dynamic Modality Attention Fusion Framework for  Multimodal Sentiment Analysis, EMNLP 2024.
[3] EMOE: Modality-Specific Enhanced Dynamic Emotion Experts, CVPR 2025

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes Gradient Orthogonalization Adaptive Leveraging (GOAL) considering unimodal representation and cross-modal compatibility via reweights gradient magnitude using entropy prediction as variation estimator. Experiments on multiple datasets are conducted to show effectiveness of the proposed model. However, the main method considering modality laziness [1] and modality orthogonalization strategy [2] are both explored and proposed in previous methods, which quite reduce the novelty of the proposed method.
[1] Weiyao Wang, Du Tran, and Matt Feiszli. What makes training multi-modal classification networkshard? In Proceedings ofthe lEEE/CVF conference on computer vision and pattern recognition.pp.12695-12705,2020.
[2] Yake Wei and Di Hu. Mmpareto: Boosting multimodal learning with innocent unimodal assistance arXiv preprint arXiv:2405.17730.2024

### Strengths
1.	Proposed strategies are theoretically solid. 
2.	Experiments are partially effective.

### Weaknesses
1.	Modality laziness and modality adjustment are both proposed by previous methods. The method seems incremental since the difference has not been presented or described.
2.	There are two issues mentioned in Abstract. The paper should discuss the proposed solutions for both issues more clearly.
3.	More theory for both AL and GR modules should be provided with more math motivation.
4.	The orthogonality in Equ 9-10 should be described.
5.	The reported metrics are not sufficient for different datasets.
6.	Unimodal performance in Table 3 should be reported.
7.	Weights of gradients should be traced for adaptive learning.

### Questions
1.	Since primary modality mostly has greater confidence, will the weight of its gradients remain larger than other modalities, which increased the modality imbalance degree?
2.	Why does visualization of GOAL embeddings in Figure 3 not show significant effectiveness compared with baseline?

### Soundness
3

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
4

### Summary
The paper proposes GOAL (Gradient Orthogonalization and Adaptive Leveraging), a parameter-free gradient modification framework to address imbalance issue in multimodal learning. Specifically, the method consists of two components: Adaptive Leveraging (AL), which adjusts the magnitude of gradients based on uncertainty estimation, and Gradient Orthogonalization (GO), which resolves conflicts between gradients from different modalities.

The paper demonstrates the efficacy of GOAL through experiments on multiple multimodal datasets. The proposed method is a plug-and-play module that can be integrated into existing multimodal learning frameworks, making it easy to be applied.

### Strengths
- The method is evaluated on multiple multimodal datasets, demonstrating consistent improvements in both accuracy and macro F1 score compared to related methods.
- GOAL is a plug-and-play module, making it easy to integrate into existing multimodal frameworks.
- The writing is easy to follow.

### Weaknesses
- The paper suggests that the gradient direction alignment between unimodal encoders is crucial. However, it remains unclear why aligning gradients in separate unimodal encoders (whose parameters are not shared) would significantly benefit model performance. Unlike shared parameters, which face potential gradient conflicts, the gradients of separate encoders are expected to be independent. The authors should clarify the theoretical and practical motivation behind aligning these gradients.
- Suppose it is meaningful to align the direction of different unimodal encoders. The paper presents gradient conflicts in Figure 2(a) between two modalities, but it does not clearly demonstrate whether the GOAL method effectively mitigates this conflict. After applying GOAL, is there a noticeable change in the gradient conflict?
- The use of the AL module to adaptively adjust gradient magnitudes across modalities is based on the assumption that the magnitudes of gradients from different modalities may vary due to their differing contribution to the learning objective. Further theoretical and empirical evidence is needed to justify this approach. Why is it crucial to address this difference in magnitude for improving fusion performance?

### Questions
Please check the above section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces GOAL (Gradient Orthogonalization and Adaptive Leveraging), a novel gradient modification method designed to address the modality imbalance problem in multimodal learning. GOAL combines two key components: Gradient Orthogonalization (GO), which resolves conflicting gradient directions across modalities, and Adaptive Leveraging (AL), which dynamically adjusts gradient magnitudes based on modality-specific uncertainties. Extensive experiments across various multimodal benchmarks demonstrate that GOAL consistently outperforms existing state-of-the-art methods in both efficiency and generalization. These results highlight GOAL's potential for improving multimodal optimization, providing a plug-and-play solution for a wide range of multimodal learning tasks.

### Strengths
- GOAL introduces an innovative method that combines Gradient Orthogonalization (GO) and Adaptive Leveraging (AL), effectively addressing the modality imbalance issue in multimodal learning. This method not only eliminates gradient direction conflicts across modalities but also adapts the gradient magnitude dynamically to handle modality uncertainty, demonstrating strong innovation.
- Through extensive experiments, the authors validate the effectiveness of GOAL on multiple multimodal benchmark datasets. The results show that GOAL significantly outperforms existing methods in efficiency and generalization, demonstrating its strong practicality and applicability. As a plug-and-play module, GOAL can be easily integrated into various multimodal learning frameworks, enhancing its versatility.
- The writing is clear, and the structure is well-organized, with the theoretical analysis and experimental sections supporting each other, allowing readers to easily understand the workings of GOAL and the experimental results. The presentation of the figures and tables is intuitive and effective, making it easy to understand the experimental process and the comparison of results.

### Weaknesses
- The GOAL method proposed in the paper is well-explained theoretically, but lacks specific guidance on how to adjust the method in practical applications. For example, how to tune hyperparameters and optimize the weighting strategy of the AL and GO components for different tasks and datasets is not thoroughly discussed.
- Although the paper demonstrates the superior performance of GOAL on multiple datasets, issues such as gradient conflict variations and dynamic weighting adaptability in more complex applications could affect the model's effectiveness. These potential challenges are not discussed in the paper.
- Although the ablation experiments show the individual effects of the AL and GO components, there is a lack of in-depth quantitative analysis on how they collaborate, interact across different datasets, and optimize in complex and imbalanced datasets.
- Although GOAL performs excellently, the paper does not sufficiently discuss its advantages and disadvantages in terms of computational cost, training time, and memory consumption, especially whether there are potential computational bottlenecks given that GOAL is a plug-and-play gradient modification method.
- Although the GOAL method performs excellently in enhancing multimodal learning, its internal mechanism, especially the interaction between GO and AL components, remains opaque. The lack of interpretability analysis may make it difficult for readers to understand how GOAL coordinates optimization across different modalities. Future work could provide deeper interpretability analysis to help readers better grasp how GOAL improves model performance.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
