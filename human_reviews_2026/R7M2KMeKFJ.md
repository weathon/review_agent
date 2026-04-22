# Tribe: Tri-Component Information Decomposition for Graph Out-of-Distribution Detection

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 4, 2, 4

## Abstract
Graph neural networks are widely used for node classification, but they remain vulnerable to out-of-distribution (OOD) shifts in node features and graph structure. Existing methods trained with standard supervised learning (SL) objectives tend to capture spurious signals from either features and/or structure, leaving the model fragile under distributional changes. To address this, we propose Tribe, a novel and effective Tri-Component Information Decomposition framework that explicitly decomposes information into feature-specific, structure-specific and joint components. Tribe aims to preserve only the label-relevant component of the joint information while filtering out spurious feature- and structure-specific information, thereby enhancing the separation between in-distribution (ID) and OOD data. Technically, we develop a novel optimisation pipeline that integrates a graph Information Bottleneck (IB) objective with carefully designed regularisations. Beyond the framework, we provide theoretical and empirical analysis showing the superiority of IB in OOD detection, with higher ID confidence and a larger entropy gap between ID and OOD data compared to the typical SL objective. Extensive experiments across seven datasets confirm the efficacy of Tribe, achieving up to 34% improvement in FPR95 over strong baselines while maintaining competitive ID accuracy. Code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper approaches the problem from the perspective of the information bottleneck in information theory, aiming to train classifiers using only the information most relevant to the labels, thereby shielding the model from interference caused by redundant information and further improving its performance on the task it investigates: out-of-distribution node detection. In simple terms, the paper employs three networks: a backbone network to extract information most critical to the labels, and two branch networks designed to disentangle redundant information arising from features and structure, respectively. Overall, the paper has a natural and reasonable motivation, excellent writing, thorough experiments, and rigorous theoretical grounding, making it a high-quality contribution.

### Strengths
1. The writing is excellent and well-structured, from which I have learned a great deal. Thank you to the authors.
2. The motivation is natural and reasonable. Although the information bottleneck is not a novel theory, I believe its successful application to out-of-distribution detection brings significant innovation and ample room for further extension.
3. I have reviewed the theoretical part and did not find any major issues.
4. The experiments are comprehensive and thorough, making the results highly convincing.

### Weaknesses
I don't have major concerns; however, although the experiments are substantial, I feel that some recent baselines are missing (even though they have been presented in the related work). If possible and feasible, including comparisons with these latest baselines, either experimentally or in the textual discussion, would further strengthen the paper.

### Questions
See the weaknesses mentioned above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
Summary
This paper introduces TRIBE, a tri-component information decomposition framework for graph out-of-distribution (OOD) detection in node classification tasks. It addresses GNN vulnerabilities to feature, structure, or joint shifts by decomposing label information into feature-specific (V), structure-specific (Q), and joint (Z) components, filtering spurious individual-input correlations via an information bottleneck (IB) objective, conditional independence regularizer, and pairwise mutual information minimization. Theoretical analysis proves IB enhances ID confidence and entropy gap for better logit-based detection. Experiments on seven datasets show up to 34% FPR95 improvement over baselines like GNNSAFE, while preserving competitive ID accuracy.

### Strengths
1. Clear Structure: The paper follows a standard academic format (abstract, introduction, related work, preliminaries, method, theoretical insights), with smooth transitions between sections. This enhances readability and guides the reader through complex ideas, from problem motivation to theoretical proofs and implementation details.
2. Theoretical Rigor: Provides solid proofs (e.g., on IB's superiority over SL in ID confidence and entropy separation), offering clear insights into why the framework improves detection under shifts.

### Weaknesses
1. Unclear motivation.

The Abstract claim that “standard supervised learning (SL) objectives tend to capture spurious signals from either features and/or structure” lacks empirical evidence. Figure 1 in the Introduction suggests that SL representations mix feature-, structure-, and label-irrelevant components, but no diagnostic experiment supports this assumption. Similar claims have already been made by prior Information Bottleneck or invariant representation learning studies, so the novelty of this motivation is limited [1,2,3].

2. Intrinsic conflicts in the mutual-information optimization.

(a) Conflict between maximizing task relevance and minimizing redundancy:
The IB objectives encourage each component (Z, V, Q) to maximize its mutual information with the label Y, but the pairwise regularization (e.g., min I(Z; V)) penalizes their overlap.
When features and structure are strongly correlated, these goals may compete, reducing predictive strength and optimization stability.

(b) Conflict between compression and conditional independence:
The compression term min I(X, A; Z) aims to remove irrelevant noise, yet excessive compression may discard the necessary X–A interactions required for min I(A; X | Z)=0.
If X and A are intrinsically dependent, these objectives can become contradictory, leading to degenerate or unstable representations.

3. Experimental limitations.

(a) The paper omits comparisons with recent strong baselines such as DeGEM and GOLD, which weakens the claim of comprehensive SOTA superiority.

(b) In 6.5 energy gap, while the article defines energy-based OOD detection in Section 3 and uses energy scores for inference, the visualization shows "greater separation between ID and OOD energy scores" without explicitly explaining how this empirical energy gap maps to or supports the theoretical "entropy gap" in Section 5 (Proposition 5.3). The scores are derived from logits (related to entropy), but the lack of a clear connection between mutual information-based theory and energy visualization may seem inconsistent.

[1]Zhang, Ge, et al. "Enhancing graph neural networks for out-of-distribution graph detection." IEEE Transactions on Neural Networks and Learning Systems (2025).
[2] Ren, Lingfei, et al. "Heterophilic graph invariant learning for out-of-distribution of fraud detection." Proceedings of the 32nd ACM International Conference on Multimedia. 2024.
[3] Li, Zenan, et al. "Graphde: A generative framework for debiased learning and out-of-distribution detection on graphs." Advances in Neural Information Processing Systems 35 (2022): 30277-30290.

### Questions
1. Could the authors provide motivational experiments or additional evidence to justify why the tri-component decomposition is necessary, especially compared to existing IB-based or invariant learning methods that also aim to reduce spurious correlations?

2. In weakness (2), how do the authors address the potential conflict during optimization?

### Soundness
2

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
This paper proposes TRIBE, a framework for graph OOD detection. The core idea is to decompose graph information into three parts — invariant, variant, and redundant components — with the goal of enhancing OOD detection by separating task-relevant and task-irrelevant information. The method leverages mutual information estimation to model the interaction among these components and integrates them into a graph representation learning framework. Experiments are conducted on several graph benchmarks to demonstrate the proposed method’s effectiveness compared with existing OOD detection baselines.

### Strengths
1. The idea of decomposing graph information into three components provides a new perspective for understanding and modeling graph representations.

2. The proposed framework attempts to connect information-theoretic principles with graph learning, which is conceptually interesting.

3. The paper includes some experimental evaluation across multiple datasets to demonstrate the general applicability of the method.

### Weaknesses
1. The core formulation of the tri-component decomposition is not clearly explained. It is unclear how the three components are defined, separated, or optimized in practice.

2. The technical novelty appears limited. The proposed framework largely combines existing concepts such as information decomposition and graph representation learning without a clear new algorithmic contribution.

3. The experiments do not provide convincing empirical support for the claimed benefits. Improvements are small or inconsistent, and there is no analysis showing that the decomposition itself enhances OOD detection.

4. The comparisons are incomplete. Recent and strong graph OOD detection methods  are missing, making it difficult to assess the true effectiveness of the proposed approach.

### Questions
Please refer to the comments listed in the Weaknesses section, which highlight points that would benefit from clarification or further empirical support.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents TRIBE, a novel framework for graph out-of-distribution (OOD) detection based on tri-component information decomposition. The authors identify that conventional supervised learning objectives in graph neural networks (GNNs) often entangle spurious correlations from node features and graph structure, resulting in poor generalization to OOD scenarios. To tackle this, TRIBE decomposes the mutual information between input (features & structure) and labels into three components: feature-specific, structure-specific, and joint-input signals. The framework incorporates an Information Bottleneck (IB) objective alongside pairwise mutual information regularization and conditional independence constraints, aiming to suppress spurious components while preserving only label-relevant joint information. Empirical evaluations across seven benchmark datasets show that TRIBE significantly improves OOD detection performance (up to 34% reduction in FPR95) while maintaining competitive in-distribution (ID) classification accuracy.

### Strengths
1. The paper introduces a principled and technically sound decomposition of predictive information into three components (feature, structure, joint). This is a non-trivial extension of IB to the graph domain, which is a valuable contribution.
2. Clear theoretical results are provided to justify why the IB objective is more suitable for OOD detection than standard supervised learning.
3. Experiments on diverse benchmarks demonstrate consistent performance gains over both non-OOD and OOD-exposed baselines.
4. The ablation study is thorough, showing the importance of each component to the overall performance.

### Weaknesses
1. While the method is conceptually interesting, the paper lacks a clear visual or algorithmic description of how all the modules (Z, V, Q networks) interact during training and inference. Figure 2 helps, but further schematic detail is needed to fully grasp the flow of gradients, loss contributions, and updates.
2. The paper relies heavily on mutual information estimators (e.g., CLUB), but does not provide sufficient detail or sensitivity analysis on the impact of approximation errors in these estimators.
3. The objective combines several components (IB terms, CI loss, PMI regularizers) with scalar weights (e.g., λ, α₁–α₃), but their selection strategy is not transparent. 
4. While the paper compares against strong OOD baselines, it does not evaluate against decomposition-based or disentangled representation methods, which could be relevant comparators in the context of learning disentangled graph representations.

### Questions
1. Why is the feature network implemented as an MLP and the structure network as a GCN?

### Soundness
3

### Presentation
3

### Contribution
2
