# UES: An Ultra-expanded Semantic Space for Unsupervised Domain Adaptation

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Unsupervised Domain Adaptation (UDA) offers a promising solution to address label annotation costs and dataset bias by facilitating knowledge transfer from a label-rich source domain to a related but unlabeled target domain. While the FC+Softmax+Cross Entropy loss has become the de facto standard for classification under the IID assumption, its performance degrades significantly under UDA's non-IID setting, where target domain features frequently violate decision boundaries, resulting in inter-class confusion. To overcome this limitation, we propose an innovative Distance Margin-based Ultra-Expanded Space (UES) loss, which encourages features to occupy an expanded representation space, thereby maintaining a safer distance from decision boundaries. Designed as a plug-and-play regularization term, UES can be seamlessly integrated into various classification-based UDA frameworks, offering exceptional simplicity by requiring only a few lines of code and minimal hyperparameter tuning while reducing computational overhead. Extensive experiments demonstrate that our method achieves performance improvements in nearly all tested cross-domain tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes constructing an ultra-expanded semantic space for unsupervised domain adaptation (UDA) to improve both intraclass compactness and interclass separability. Unlike conventional margin-based methods that risk decision boundary violations and negative transfer, the approach pushes features away from decision boundaries to create a more generalizable metric space. It is easy to implement, compatible with existing feature alignment techniques, and achieves consistent performance improvements and enhanced testing stability.

### Strengths
1. It seems that the loss could be combined with any feature distribution alignment technique, showing its high generality and compatibility.
2. Simple design allows straightforward integration into existing models.
3. The paper is well-motivated and seems to be reproducible.

### Weaknesses
I believe this paper has two main limitations:
1. Although the authors introduce the proposed Ultra-expanded (UE) loss with good motivation, there seems to be a lack of in-depth theoretical discussion and analysis. As a result, readers may find it difficult to gain a truly insightful understanding. Please refer to [A].
2. The experiments are based on relatively outdated baselines. Improvements over these older methods may still fall short of the performance achieved by more recent approaches, even including zero-shot methods. It remains unclear whether the proposed methods would remain effective when applied to newer methods, especially those leveraging pretrained or large-scale models.

Other issues include:
1. The paper uses overly complex and non-standard notation, and contains several typos, which would benefit from careful proofreading.
2. The comparative experiments are somewhat limited, both in terms of baselines, datasets, and experimental settings.

[A] Xu, Gezheng, et al. "Revisiting Source-Free Domain Adaptation: a New Perspective via Uncertainty Control." The Thirteenth International Conference on Learning Representations. 2024.

### Questions
I hope the authors could provide further clarification or responses regarding the theoretical analysis and experimental settings.

### Soundness
1

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a well-motivated and clearly written contribution to UDA. The idea of expanding classifier weights into an “ultra-expanded semantic space” is elegant and empirically validated. Although the method is simple, it provides notable and consistent gains over classical baselines with minimal complexity.

### Strengths
1.The proposed UES loss is conceptually simple yet effective. The theoretical motivation and geometric interpretation are clearly articulated, and the visualizations convincingly show its ability to enhance intra-class compactness and inter-class separability across domains.
2.The method introduces very few hyper-parameters, shows strong robustness to them, and can be easily integrated as a plug-and-play regularizer into existing UDA frameworks.
3.The Semantic Zoom mechanism is intuitively reasonable: using different temperatures for source and target domains is a neat way to retain semantically meaningful information while mitigating domain bias.

### Weaknesses
1.While the chosen alignment backbones (DAN, DANN, DeepCORAL, DAAN) are classic and representative, they are relatively old. The paper lacks comparison with more recent or stronger baselines, making it difficult to assess the broader applicability of UES in modern settings.
2.The margin-loss comparison focuses on Softmax, ArcFace, and Center Loss, which are also dated. The effectiveness of UES against more advanced discriminative losses remains unclear.
3.The Geometric Interpretation in Section 4.1 is insightful and explains how Wbasis enlarges the margin and thus enhances robustness. However, the analysis remains primarily at a geometric and empirical level, lacking a formal connection to existing domain generalization bounds or theoretical guarantees. Strengthening this link would substantially improve the paper’s rigor.

### Questions
1.The Semantic Zoom module employs different softmax temperatures. How sensitive is the approach to this setting, and would a learnable or adaptive temperature improve performance further?
2.Since UES constructs an expanded feature space, does it risk over-dispersion or feature instability in high-dimensional domains?
3.The experiments show that the method consistently improves accuracy and reduces A-distance, but it remains unclear how UES interacts with domain alignment losses (e.g., adversarial or moment matching). Does UES primarily benefit from better intra-domain discrimination, or does it also implicitly reduce inter-domain discrepancy?

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
The paper proposes UES (Ultra-Expanded Semantic Space), a new loss formulation for Unsupervised Domain Adaptation (UDA). The authors claim that this loss can serve as a plug-and-play regularizer for existing UDA frameworks such as DANN, DAN, DeepCORAL, and DAAN. Experiments on three benchmark datasets (Digits, Office-31, Office-Home) show improved accuracy compared to ArcFace and Center Loss.

### Strengths
1. This paper attempts to design a simple, general-purpose regularizer applicable across UDA frameworks.
2. The figures in this paper are easy to understand.

### Weaknesses
1. All baselines (DANN, DAN, DeepCORAL, DAAN) are from 2015–2019. Missing modern methods makes the claimed superiority unconvincing.
2. The related work section is outdated and clearly misses recent advances in UDA from the past few years.
3. The experiments are limited to relatively simple datasets and lack evaluations on more challenging benchmarks such as DomainNet, which is commonly used in UDA research.
4. The writing quality requires further improvement. The paper should be reorganized for better readability, and the main experiments should be integrated into the main text rather than placed in the appendix.

### Questions
Please refer to Weaknesses.

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
5

### Summary
The paper proposes an innovative approach for Unsupervised Domain Adaptation (UDA) by introducing the Ultra-expanded Semantic (UES) loss, designed to enhance both intra-class compactness and inter-class separability. The approach aims to mitigate issues caused by the traditional FC + Softmax + Cross-Entropy loss, particularly in non-IID settings where features from the target domain exhibit large intra-class variation, leading to poor generalization. The UES loss incorporates an Ultra-expanded (UE) loss term and a Semantic Zoom mechanism, both of which are used to push features further from decision boundaries while preserving critical semantic information. Extensive experiments on several datasets demonstrate the proposed method's effectiveness, showing consistent improvements over baseline methods, such as ArcFace and Center Loss.

### Strengths
- Innovative approach: proposes a new margin-based UES loss that simultaneously enhances intra-class compactness and inter-class separability for UDA.

- Clear motivation: identifies the weakness of FC+Softmax+CE under non-IID assumptions and provides a reasonable explanation for why UES loss can help.

- Strong experimental validation: consistently outperforms ArcFace and Center Loss across multiple datasets and baselines (DAN, DANN, DeepCORAL, DAAN).

- Good robustness: demonstrates stable convergence and wide tolerance to hyperparameters (e.g., expansion factor e, λ₂).

- Easy integration: the loss is simple and can be added as a regularization term to existing frameworks with minimal code changes.

### Weaknesses
- Limited theoretical analysis: the paper lacks a solid theoretical justification for why UES loss performs better than other margin-based losses.

- Insufficient ablation studies: only limited analysis on the contribution of Semantic Zoom vs. UE loss; unclear which component drives most of the gains.

- Missing computational analysis: claims low overhead but provides no quantitative evidence (training time, memory, etc.).

- No discussion on limitations: does not explore scenarios where the method might fail (e.g., extreme domain shift or noisy target data).

- Writing and structure: some sections (especially “Inspirational Discoveries”) read more like extended discussion rather than rigorous analysis, which affects clarity.

### Questions
- Could the method be extended to multi-source domain adaptation? How does the UES loss behave when there are multiple source domains?

- How does the choice of hyperparameters (e.g., expansion factor, temperature) influence the results across different tasks? A more thorough analysis of hyperparameter sensitivity would be beneficial.

- Could the method be applied to more complex domains beyond image classification, such as natural language processing or video processing?

### Soundness
3

### Presentation
3

### Contribution
3
