# Tackling the XAI Disagreement Problem with Adaptive Feature Grouping

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Post-hoc explanations aim at understanding which input features (or groups thereof) are the most impactful toward certain model decisions. 
Many such methods have been proposed (ArchAttribute, Occlusion, SHAP, RISE, LIME, Integrated Gradient) and it is hard for practitioners
to understand the differences between them. Even worse, faithfulness metrics, often used to quantitatively compare explanation methods,
also exhibit inconsistencies. To address these issues, recent work has unified explanation methods 
through the lens of Functional Decomposition. We extend such work to scenarios where input features are partitioned into groups 
(e.g. pixel patches) and prove that disagreements between explanation methods and faithfulness metrics are caused by between-group 
interactions. Crucially, getting rid of between-group interactions leads to a single explanation that is optimal according to all faithfulness metrics. We finally show how to reduce the disagreements by grouping features on tabular/image data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses disagreement problem in explainable ai, where different post-hoc explanation methods and metric often produce inconsistent results. The paper extend the framework of functional decomposition to analyze explanations over groups of features. It has the theoretical contribution to prove that the disagreement between various group-based explainers and high scores on common metrics share a common root cause. Therefore the paper proposes AGREED, an algorithm that adaptively finds feature partitions that minimize these interactions.

### Strengths
1. The paper is well written and easy to follow.
2. The paper has rigorous theoretical contribution.
3. The paper clearly demonstrates that both disagreements between explanation method and (un)faithfulness as measured by standard metric are functions of the same underlying between-group interaction terms.
4. The paper validated the theoretical claims in diverse modalities both synthetic and real world dataset.

### Weaknesses
1. The problem statement in the introduction is very broad. The described challenge is fundamental to almost all xai research.
2. The readability of the figures are poor, also the main text needs to provide guidance on how to interpret the results in the figure (e.g., what the takeaway is).
3. Need to cite all the explanation methods that are introduced in the paper (e.g., arch, occ).
4. The paper's argument appears to equate 'faithfulness' with 'agreement/consistency' between explainers. While the theory links common (un)faithfulness metric to interaction-driven disagreement, the author should specify the definition of 'faithfulness'.
5. The vision experiments rely on VGG16 and ResNet18, which are architectures from 2014 and 2015 which is outdated.
6. The proposed algorithm does not optimize the true interaction loss, which captures all interactions.

### Questions
Look at the weaknesses

### Soundness
3

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
The paper aims to address the problem of XAI disagreement. The aauthors extend Functional Decomposition and parition the input features/pixels into groups and prove that the primary reason of disagreement between different explanation methods is the between-group interactions.Further, they advocate eliminating between-group interactions and show that the disagreements can be reduced by adaptively grouping features/pixel.

### Strengths
1) The paper deals with an important aspects of XAI: disagreement between explanation methods and the unreliability of fidelity metrics

2) The theoretical sections are formally presented, with clear conditions, properties, and proofs.

3) The proposed method was shown to work in tabular data as well as images.

### Weaknesses
1)  The authors use L2 disagreement between attribution vectors to define their partition loss. While this choice yields a differentiable and additive objective but it emphasizes only magnitude alignment and misses out on rank alignment. Rank-based measures such as Spearman, Kendall correlation etc could capture rank consistency between explanations but are non-differentiable for the groupwise optimization. A brief justification of this design choice would enable the readers to understand the paper clearly.

2) The partition loss formulation is conceptually appealing but it requires computing multiple attribution maps per input, severely limiting scalability. This limitation should be discussed more explicitly as it constrains the applicability of the proposed framework to high-dimensional or real-time settings.

3) The authors provide the algorithm details of AGREED in supplementary and also the computation cost of O(d^2N^2). The paper would benefit from a detailed analysis of this computation cost.

4) AGREED seems to be prohibitively expensive as the cost O(d^2N^2) increases rapidly with number of features and samples. Even for 100 features and 1000 samples the number of model  inferences would be too high. Hence, there is limited scope of appying AGREED on high dimensional and real-world settings. I request the authors to clarify this limitation in the paper.

### Questions
I request the authors to address the weaknesses

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
4

### Summary
Summary
This paper addresses the disagreement problem among post-hoc explanations in explainable AI by proposing an adaptive feature grouping method (AGREED) that reduces interactions between feature groups, aiming to bring explanation methods into agreement. The work builds on prior frameworks of functional decomposition and attribution methods, extending them to disjoint feature groupings. Empirical validation on tabular and image datasets shows improvements in explanation consistency and faithfulness metrics.

Soundness
The methodology is mathematically sound and supported by formal proofs linking disagreement to between-group interactions. The algorithm is intuitive and empirically demonstrated to reduce disagreement. Results are robust on multiple datasets, though assumptions like groupwise additivity in regions and sampling baselines could affect results in practice.

Presentation
The paper is well-organized but the presentation is somewhat standard and occasionally dense. Some of the core ideas could be clarified with more intuitive or visual illustrations, especially around interpretation of grouped attributions. The empirical section is comprehensive yet could better emphasize practical significance. Comparisons to prior grouping methods on images are limited by scalability and implementation constraints, leading to incomplete benchmarking in some cases.

Literature review
Literature review is a bit outdated, check these e.g. for attributions methods such as IG and SHAP with stronger mathematical content:

A General Feature Attribution Framework under a Black-box Setting

-Y. Cai, A. Thibaud, G.Wunder, International Conference on Machine Learning (ICML'25), Vancouver, Canada, 2025

-On Gradient-like Explanation under a Black-box Setting: When Black-box Explanations Become as Good as White-box
Y. Cai, G. Wunder, International Conference on Machine Learning (ICML'24)

Contribution
-Proposes a practical adaptive grouping algorithm to reduce disagreement in feature attributions
-Unifies explanation methods under a functional decomposition framework for groups
-Extends interpretability to handle feature group interactions explicitly
-Demonstrates empirical gains on tabular and image data
-Provides theoretical insights into when explanation agreements are possible

### Strengths
-Sound theoretical grounding
-Addresses a key challenge limiting trust in explainability
-Novel focus on attributing disagreement to feature interaction

### Weaknesses
-Novelty incremental given prior functional decomposition frameworks; core insight of grouping to minimize interactions is expected
-Presentation could benefit from more accessible examples and clearer motivation for practitioner
-Limited direct comparisons with some existing grouping algorithms on images
-Interpretation of grouped attributions remains difficult without actionable visualization
-Image domain grouping requires per-sample runs, limiting real-world scalability

### Questions
-How does AGREED compare to prior grouping methods that are not feasible for images, when run on smaller image subsets?
-How practically interpretable are grouped attributions for non-expert users, especially with large feature clusters?
-Can the algorithm support global grouping for an entire dataset rather than per-sample in images?
-Would integrating semantic/region-based features improve grouping and interpretability further?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the issue that many explanation methods often disagree with each other. They formally prove that when the model is group-wise additive, using the ground truth groupings, then unfaithfulness metrics will all reduce to 0. They then prove that if we minimize the difference between Arch and Occlusion, then this will minimize the disagreement between all groups. And they achieve it by iteratively merging groups with the highest pairwise interactions. They repeat until the disagreement falls below a threshold. They conduct experiments to show that when they merge the groups progressively, the difference between different explanation methods consistently decreases, and that each method is changing to become more agreed as the loss decreases, and that the proposed method AGREED can get much lower disagreement when merging not many groups than other methods, meaning that it is merging the groups that will reduce disagreement the most.

### Strengths
1. The paper proves a novel and important theorem that if we reduce the disagreement between Arch and Occlusion, it will reduce the disagreement for all methods.
2. Their proposed method empirically finds better groups than other methods.
3. The authors conducted comprehensive experiments to demonstrate their advantage in different aspects.

### Weaknesses
1. Some of the empirical figures (e.g., Figures 4 and 5) are not very intuitive
in their current presentation, and the takeaways are not made explicit in the
captions. Clarifying the intended interpretation in the captions or main text
would significantly improve readability for non-expert readers.

2. One prior work also proves how one can achieve zero unfaithfulness when using the correct grouping structure. It would be helpful to cite this work to situate the current contribution within a more comprehensive body of related theory [1].

3. The paper considers only non-overlapping partitions, while there could be cases where groups overlap with each other a lot and only considering non-overlapping groups could make it less expressive.

[1] You et al. "Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups" ICML 2025

### Questions
1. You et al. 2025 also considers group-based attributions and categorizes insertion and deletion style errors (Definition 2.2 of Sum-of-Parts is similar to Equation (10) in this paper, and Definition A.1 is similar to Equation (9)), although they compute differences with respect to the prediction contributed by a group, rather than differences in attribution scores produced by two methods for that group. They show that when there are between-group interactions (e.g., polynomial correlations), these errors can grow exponentially (Theorem 2.3), while using the correct grouping drives the error to zero (Theorem 2.4). The theoretical results here (e.g., Theorem 3.2) appear closely related, as both hinge on the vanishing of between-group interaction terms. It would be helpful for the authors to explicitly discuss the similarities and differences with You et al. 2025, including the choice of disjoint partitions in this paper vs. allowing overlapping groups in You et al 2025.

2. In Section 5.2.1 and Figure 7, only Arch and Occlusion are shown. Since the paper's claims concern reduced disagreement across a broad set of methods, could you clarify whether the qualitative improvements also generalize to other methods like LIME/SHAP etc.? Including additional visual examples in the appendix would strengthen the qualitative evidence.

3. In functions with overlapping interaction structure (e.g., where a feature participates in multiple strongly interacting cliques), disjoint partitions may not be expressive enough. In such cases, the optimization may be forced to merge many features together, reducing interpretability. It would be helpful if the authors could comment on this tradeoff and whether and how overlapping or hierarchical groupings could be supported in future work.

[1] You et al. "Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups" ICML 2025

### Soundness
4

### Presentation
3

### Contribution
3
