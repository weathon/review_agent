# Multi-Linear Subspace Distance: A New Criterion for Tensor Feature Selection

- Decision: Reject
- Scores: 2, 6, 2, 6

## Abstract
Feature selection in tensor data poses greater challenges than in vector representations, since it must capture correlations spanning multiple modes rather than treating each mode in isolation. Existing tensor-based methods partially address this but often treat the feature space as a whole, selecting features globally without respecting mode-specific dependencies. This not only overlooks cross-mode interactions but also increases computational burden, as all features must be considered at once. Moreover, they lack a principled criterion for preserving the global structure of the original tensor. In this work, we introduce Multi-Linear Subspace Learning Feature Selection (MSLFS), a framework that overcomes these limitations by distributing feature selection across modes. Specifically, MSLFS selects a small number of representative slices along each mode, whose intersections yield the most informative features. The core innovation is a multi-linear subspace distance, which provides a principled measure of how well these selected features preserve the global multi-way structure of the data, while significantly reducing redundancy and computational cost. This objective is complemented by two novel regularizations: a joint sparsity constraint that enforces coordinated sparsity across modes to identify compact, non-redundant features, and a higher-order graph constraint that preserves local manifold geometry within the induced subtensor. Taken together, these components guarantee that the overall tensor structure as well as the local neighborhood relationships are preserved. Comprehensive experiments on image recognition and biomedical benchmarks demonstrate that MSLFS consistently surpasses state-of-the-art feature selection techniques in clustering tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Multi-linear Subspace Learning Feature Selection (MSLFS), a novel tensor-based feature selection framework that operates distributively across tensor modes. It proposes a multi-linear subspace distance to preserve global tensor structure and incorporates joint sparsity and higher-order graph regularization to reduce redundancy and maintain local geometry. Extensive experiments on image and biomedical datasets demonstrate that MSLFS consistently outperforms state-of-the-art feature selection methods in clustering tasks. The method also offers computational efficiency and flexibility in selecting features through slice intersections along different modes.

### Strengths
1. The writing and presentation are relative smooth, the readers can easily follow.
2. The mathematic prove is complete, ensuring its theoretical solidness.

### Weaknesses
1. This is an incremental work. In the introduction, there is no in-depth analysis of existing methods and necessary experimental verification to prove the defects of the current methods.
2. Sparse regularization and graph regularization have been widely studied and applied in a large number of works, and no novelties can be found.
3. The author's proposed algorithm can improve the running efficiency, but no comparison of running time can be seen in the experiment.
4. In the experiment, the datasets used are old and small. In addition, I noticed that the clustering effect after feature selection is not very good. As far as I know, some clustering results based on deep networks are already very excellent. Therefore, the significance of feature selection has to be questioned.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors introduce MSLFS, a framework for unsupervised feature selection in tensor data. MSLFS distributes selection across modes and identifies representative slices with the most informative features. The joint sparsity regularization is introduced to enforce coordinated sparsity across modes. Theoretical results are given to guarantee the recovery. Experiments on image and biomedical datasets show MSLFS outperforms existing feature selection methods.

### Strengths
1. A new framework for tensor feature selection is proposed.
2. Theoretical analysis for exact recovery is given.
3. Experimental results show the performance improvement compared to existing feature selection methods.

### Weaknesses
1. The entire algorithm design and theoretical analysis are limited to 3-order tensors.
2. The parameters $\alpha$ and $\beta$ are sensitive to performance.

### Questions
1. Can the proposed method be applied to tensors of order greater than 3?
2. In line 163, “Thus, the ability of a fiber to characterize the feature space depends on how well these slices span…”. What does “these slices” refer to—perhaps the slices containing the fiber? Please clarify this statement.
3. How many features are selected in Table 2?
4. As shown in Figure 8, alpha and beta are sensitive to performance (e.g., ACC/NMI varies by about 5 in each row/column). How should these parameters be selected in practice?
5. In each experiment, do all compared algorithms select features with the same feature size and dimensions?
6. How does the running time compare with other methods?
7. In line 232, “otherwise, approximate bases yield reconstructions with errors tied to the residuals”. Is it possible to provide error bounds in this case?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a new method for feature selection in tensor (multi-dimensional) data called MSLFS. Instead of selecting features globally, the method selects representative "slices" along each mode, and the intersections of these slices form the final compact feature set. The core of this approach is a novel "multi-linear subspace distance," which measures how well the selected features preserve the global structure of the original tensor. Experiments on image and biomedical datasets show that this method outperforms existing techniques in clustering tasks.

### Strengths
1. This paper introduces a new algorithm for multilinear feature selection. A new metric for measuring the distance between two tensors are also proposed. 

2. Based on the proposed measure, this paper introduces a new tensor feature selection framework by adopting the row sparsity penalty and graph regularization.

### Weaknesses
1. The definition of the spanned space in Definition 1 is unconventional. It is defined as the space spanned by the mode-n slices, rather than the typical approach of using mode-wise directional vectors. When this definition is reduced to matrix's case, the spanned space will be conflict with the standard spanned space.

2. Definition 1 is also not a standard metric, as it does not satisfy symmetric in a metric space, since dist(S(X), S(Y)) \neq dist(S(Y), S(X)).

3. In Definitions 1 and 3, it contains the unknown variable alpha, thus the definition should contain alpha factor. 

4. In Definition 3, how to reflect the multilinear that claims in this paper? As it measures the distance between a row vector unfolded by regular tensor unfolding and  another vector, the multilinear property is not clear.

5. For the method part, I would like to say that it is a variant of HOOI with row-wise sparsity penalty and graph regularization. However, these strategies are very common and have already been discovered in many previous research works, I don't think it bring any new insights for multilinear feature selection field.

### Questions
See the weakness part for details.

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
This paper presents a novel tensor feature selection method, i.e., Multi-Linear Subspace Learning Feature Selection (MSLFS), which addresses key limitations of existing approaches in handling multi-modal tensor data. These limitations include the neglect of inter-modal dependencies, high computational costs, and the absence of a principled criterion for preserving global structure. The MSLFS framework employs a distributed selection strategy that constructs informative feature subsets through intersections of representative slices from each mode. A central contribution is the introduction of a multi-linear subspace distance, which provides a quantitative measure of how well the selected features preserve the original tensor's global multi-way geometry. Furthermore, MSLFS incorporates both a joint sparsity constraint and a higher-order graph constraint to maintain the integrity of the overall tensor structure while preserving local neighborhood relationships.

### Strengths
1.	This paper has a solid theoretical foundation and clear interpretability.
2.	The framework and methodology of this paper are complete, with a well-integrated objective function, a detailed optimization process, and a comprehensive complexity analysis.
3.	The experiments presented in this paper are solid and reliable. Extensive experiments on various benchmark datasets demonstrate that its performance consistently outperforms existing methods, and this is supported by comprehensive ablation experiments and visualization results.

### Weaknesses
1.	The related work section lacks adequate discussion of recent advances (2024–2025), featuring only one work from 2025.
2.	This model includes multiple hyperparameters (e.g., α, β, and the k-nearest neighbor algorithm). Although sensitivity analysis is provided, it will be subject to certain limitations in practical applications.

### Questions
1.	The related work section lacks adequate discussion of recent advances (2024–2025), featuring only one work from 2025. It would be beneficial to incorporate additional analyses of recent literature.
2.	When the number of tensor modes exceeds 3, does the definition of multilinear subspace distance need to be adjusted? Can MSLFS be directly extended? Are additional regularization or structural assumptions required?
3.	The tensors mentioned in Appendix 7.6 may have negative values. Therefore, the MSLFS method is extended. Has this new update rule been experimentally verified? Will it affect model performance?

### Soundness
3

### Presentation
3

### Contribution
3
