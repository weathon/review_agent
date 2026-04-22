# Relationship Alignment for View-aware Multi-view Clustering

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
Multi-view clustering improves clustering performance by integrating complementary information from multiple views. However, existing methods often suffer from two limitations: i) the neglect of preserving sample neighborhood structures, which weakens the consistency of inter-sample relationships across views; and ii) inability to adaptively utilize inter-view similarity, resulting in representation conflicts and semantic degradation. To address these issues, we propose a novel framework named Relationship Alignment for View-aware Multi-view Clustering (RAV). Our approach first constructs view-specific sample relationship matrices from deep features and aligns them with the global relationship matrix to enhance cross-view neighborhood consistency and facilitate accurate measurement of inter-view similarity. Simultaneously, we introduce a view-aware adaptive weighting mechanism for label contrastive learning that dynamically adjusts the contrastive intensity between view pairs based on deep-feature similarity: higher-similarity views lead to stronger label alignment, while lower-similarity views reduce the weighting to prevent enforcing agreement. This strategy promotes cluster-level semantic consistency while preserving natural inter-view relationships. Extensive experiments demonstrate that our method consistently outperforms state-of-the-art approaches on multiple benchmark datasets. Project website: https://github.com/chenzhe207/RAV.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The approach is well-motivated, combining relation alignment with view similarity measurement to offer a novel perspective for addressing semantic degradation caused by view discrepancies in multi-view clustering.

### Strengths
1. This paper effectively addresses two core issues in current deep multi-view clustering: the weak preservation of neighborhood structure and the representation degeneration caused by view discrepancies. It directly offers solutions to these problems by introducing two modules: cross-view relationship alignment and view-aware label contrastive learning. The former ensures the consistency of the neighborhood structure through alignment, while the latter dynamically adjusts the inter-view weights based on the Wasserstein distance. The relational alignment module and the view-aware weighting mechanism are ingeniously designed, and their combination exhibits strong synergistic effects.
2. The experiments cover multiple datasets, involving different types, sample sizes, and numbers of views, fully showcasing the diversity of the data.
3. The experiments include sensitivity analysis of parameters, convergence analysis, and ablation studies, validating the effectiveness of the proposed method from multiple perspectives.

### Weaknesses
1. This paper has some shortcomings in terms of literature coverage in the relevant field, as it fails to comprehensively showcase the depth of the research.
2. The description of the experimental details is insufficient, such as the lack of clarification on the choice of the Gaussian kernel bandwidth σ.
3. The description of the formulas is somewhat confusing, particularly in the case of Equation (6).

### Questions
1. In terms of references, it is recommended that the author include citations to more relevant literature, especially recent studies in the field, to enhance the completeness and reference value of the paper.
2. Although the experimental data is already comprehensive, it is recommended to analyze in more depth the performance of the method on different datasets, particularly the reasons why it performs better or worse on certain datasets.
3. Although the paper mentions the differences between this method and the SEM method, the explanation is still not clear enough. Please further elaborate on the core differences between the two.
4. This paper primarily uses the Wasserstein Distance (WD) to quantify the differences between views. It would be useful to explain why the WD was chosen as the metric, rather than other common metrics like MMD or cosine similarity.
5. In the calculation of the relationship matrix, the paper does not specify the value of the Gaussian kernel bandwidth σ, nor does it indicate whether σ was adaptively adjusted for different datasets.
6. In Equation (6), the parameters τ_F and τ_L are mentioned, but their specific meanings are not clearly defined. If the lack of clarity is due to a presentation issue or if they carry special significance, further explanation is needed to enhance the clarity and accuracy of the equation.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a multi-view clustering method that combines relational alignment and view-aware weighting, addressing two core challenges in the field: insufficient preservation of sample neighborhood structure and adaptive utilization of view similarity. The experimental design is rigorous and logically clear, providing empirical support for the effectiveness of the proposed method.

### Strengths
1. It clearly points out the shortcomings of existing methods in preserving sample neighborhood structure and adaptively utilizing view similarity.
2. The relational alignment module and the view-aware weighting mechanism are ingeniously designed, and their combination exhibits strong synergistic effects.
3. The experimental design of this paper is well-structured and thoroughly validated.
4. It breaks away from the traditional feature alignment approach, shifting towards deeper structural relationship alignment. By introducing distributional distance awareness in contrastive learning, it avoids the semantic disruption caused by forced alignment.

### Weaknesses
1. There is an issue with inconsistent formatting of mathematical symbols, which should be carefully reviewed to enhance the reader's understanding of the work.
2. The discussion of future prospects in the conclusion section is not sufficiently detailed.
3. The description of the innovations is not particularly prominent.

### Questions
1. The paper introduces view-aware adaptive weighting, particularly the Wasserstein distance-based view similarity measurement. Please provide a detailed explanation of the advantages of Wasserstein distance in handling distributional differences.
2. The paper lacks visualizations of the feature space or clustering results, such as t-SNE plots, making it difficult to intuitively appreciate the improvements achieved by the proposed method in representation learning.
3. The formatting of mathematical symbols in the paper requires careful attention, such as \mathbf{q}_{ij}^v.
4. The conclusion summarizes the key points of the paper and outlines future research directions. Please further elaborate on its research value for future work from both theoretical and practical perspectives.
5. The current abstract is clear in its writing but somewhat flat in expression. It does not fully highlight the innovation of the research. It is recommended to make the core contributions more prominent and to clearly explain the inherent connection between the two innovative points.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes two key core modules: first, aligning global and local relation matrices to maintain neighborhood consistency; second, dynamically adjusting the weights of view pairs based on WD to alleviate representation conflicts. Meanwhile, the experimental section thoroughly validates the effectiveness of the proposed method.

### Strengths
1.This paper not only considers the structural relationships between samples but also employs a more universal metric to assess the similarity between views.

2.This paper uses three major clustering evaluation metrics: ACC, NMI, and PUR. The experimental results are presented clearly, and the performance is highly convincing.

3.The conclusion not only summarizes the core aspects of the method but also offers suggestions for future research directions.

### Weaknesses
1.This paper does not provide the rationale for selecting the contrastive learning temperature parameters $τ_F$ and $τ_L$.

2.Algorithm 1 provides an overview of the training process but lacks details of the key steps.

3.The descriptions in the introduction are somewhat repetitive and redundant.

### Questions
1.This paper concatenates the features of all views to obtain $Z∈R^{N×V_d}$, and further constructs the global relation matrix. Compared to other fusion strategies (such as weighted fusion of features to construct the relation matrix), what are the advantages of choosing this strategy?

2.Does constructing the relation matrix lead to an increase in computational cost, creating certain limitations?

3.The weighting mechanism only applies to the label contrastive loss, while in the relation alignment module, all views are equally aligned with the global matrix. Could this be considered an inconsistent treatment?

4.Some descriptions in the paper are somewhat colloquial. It is recommended to revise them into a more formal and academic expression.

5.The weight between views is calculated as $1/2\(w_{v,u}+w_{u,v}\)$ in equation (12). Please explain why this representation is used.

6.Algorithm 1 is described concisely but lacks key details, such as how the relation matrix is obtained.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper proposes two modules—relationship alignment and view-aware weighting—that preserve the neighborhood structure and alleviate the representation conflict caused by low view similarity, demonstrating a certain level of innovation.

### Strengths
1. The experiment compares multiple methods on different datasets, and the comparison methods are comprehensive. 
2. The ablation study not only validates the effectiveness of each loss component but also further verifies the effectiveness of the view-aware adaptive weighting.
3. A view-aware adaptive weighting mechanism based on Wasserstein distance is introduced to dynamically adjust the weight of view pairs in label contrastive learning, avoiding the forced alignment of inconsistent views.
4. This paper focuses on sample neighborhood structure consistency rather than traditional sample-level consistency.

### Weaknesses
1. The description in the introduction of this paper has issues with insufficient coherence, and the logical flow needs improvement.
2. This paper does not clearly specify whether the contrastive method strictly follows the configuration in the original paper, lacking necessary details.
3. The conclusion section is highly repetitive of the abstract, failing to explore the core value of the innovations in depth, and lacks a thorough analysis and summary of the research findings.

### Questions
1. This paper introduces two innovative modules. Please explain and describe the inherent connection between these two modules.
2. Sections 2.1 and 2.2 still lack sufficient research and discussion on related work. Please further enrich this section.
3. This paper cites two studies related to view weighting in the related work section (Xu et al. (2023) and Wu et al. (2024)), yet the experimental comparison only includes a performance comparison with the SEM method. It is recommended to add a comparative experiment with the study by Wu et al. (2024).
4. This paper uses the Gaussian kernel function to compute the relationship matrix between samples. Please explain the reason for choosing the Gaussian kernel instead of using Euclidean distance-based methods to compute the relationship matrix.
5. When the similarity between two views is extremely low, the corresponding weight is significantly reduced, potentially leading to excessive suppression of the view pair's contribution and the loss of limited yet valuable discriminative information from that view. Does the proposed method suffer from this issue?
6. In the Implementation Details section, it is recommended to include the dimensionality settings of the encoder's hidden layers, and to specify whether a consistent network structure was used across all datasets.

### Soundness
4

### Presentation
4

### Contribution
3
