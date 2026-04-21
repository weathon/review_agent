# Interpretable Dimensionality Reduction by Feature-preserving Manifold Approximation and Projection

- Avg Score: 5.60
- Decision: Reject
- Scores: 5, 6, 5, 6, 6

## Abstract
Nonlinear dimensionality reduction often lacks interpretability due to the absence of source features in low-dimensional embedding space. We propose FeatureMAP, an interpretable method that preserves source features by tangent space embedding. The core of FeatureMAP is to use local principal component analysis (PCA) to approximate tangent spaces. By leveraging these tangent spaces, FeatureMAP computes gradients to locally reveal feature directions and importance. Additionally, FeatureMAP embeds the tangent spaces into low-dimensional space while preserving alignment between them, providing local gauges for projecting the high-dimensional data points. Unlike UMAP, FeatureMAP employs anisotropic projection to preserve both the manifold structure and the original data density. We apply FeatureMAP to interpreting digit classification, object detection and MNIST adversarial examples, where it effectively distinguishes digits and objects using feature importance and provides explanations for misclassifications in adversarial attacks. We also compare FeatureMAP with other state-of-the-art methods using both local and global metrics.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Nonlinear dimensionality reduction is a classic research area in past years. In the paper, the authors proposed a feature-preserving manifold approximation and projection method, called FeatureMAP. The proposed method employs anisotropic projection to preserve both the manifold structure and the original data density. Some methods are compared with the proposed method on FashionMNIST and COIL-20 dataset to demonstrate the effectiveness of the method.

### Strengths
1. In the paper, the authors proposed a feature-preserving manifold approximation and projection method, called FeatureMAP.
2. Some methods are compared with the proposed method on FashionMNIST and COIL-20 dataset to demonstrate the effectiveness of the method.

### Weaknesses
1. Unclear Motivation: Why perform SVD on $ W_i X_i $ instead of Xi or X? The motivation is not introduced clearly.
2. There is an error in (5), since Vi/Vj is known variable calculated in (2).
3. What are the advantages of the method compared with the existing works, especially in terms of computational complexity, running times, and robustness? 
4. What’s the significance/application value of the method in real-word applications, especially under the LLM era?
5. This method involves multiple independent steps and is not a complete optimization problem. Will this lead to difficulties in obtaining the global optimal solution and the problem of the previous step affecting the next result?

### Questions
see weakness

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
Nonlinear dimensionality reduction often lacks interpretability due to the absence of source features in low-dimensional embedding space. This paper proposes FeatureMAP, an interpretable method that preserves source features by tangent space
embedding.

### Strengths
This paper proposes FeatureMAP, an interpretable method that preserves source features by tangent space embedding. The core of FeatureMAP is to use local principal component analysis (PCA) to approximate tangent spaces.

### Weaknesses
The way to solve Eq. (11) is not described in detail. A.4 seems hard for me to implement the FeatureMAP.

### Questions
see the weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The manuscript suggests a new non-linear dimensionality reduction method, which aims to preserve important features of the original space by preserving the alignment between the local axes ("tangent space") defined at each sample. It can be seen as an extension to UMAP, where the data topology is defined through kNN, and the loss function is optimised through gradient descent. The loss function has two terms: one which minimises distortion of distances caused by the projections (like UMAP) and a second term which maximises correlations of the radii of the local axis (the singular values of the data directions around each data point). This optimisation of the alignment of the local axes aims to preserve both the global features of the source space and the data density on the low-dimensional manifold.

### Strengths
- the manuscript provides interesting intuition for why the alignment between local axes of variation needs to be preserved by nonlinear dimensionality reduction methods.
 - the manuscript shows how to extend UMAP to include the alignment of local axes in the loss function in a way which can still be optimised in practice.
 - experimental results show how the new method differs from previously proposed ones in terms of the resulting visualisation and data density preservation.

### Weaknesses
- little or no support is provided that the resulting dimensionality reduction better preserves important features of the source space, which is the main motivation of the method.
 - the comparison to other methods is mostly qualitative and makes it hard to argue the new method is somehow superior to previously proposed ones. Visualization results (Figure 12) are subjective ("clearly shows distint clusters of different digit groups"), quantitative measures (Table 1) are inconclusive, and running time is usually worse (Figure 13).

### Questions
1. Is it a correct assessment to say that the proposed method is equivalent to UMAP when considering only the topology (kNN graph) and the first term of the loss function (distance distortion)? If so, it would clarify the presentation if you would say so and not repeat the parts already part of UMAP (you can, of course, have the full details of the loss in the appendix).
2. The second optimisation term has two interpretations: preservation of local density and preservation of source features. Can you provide evidence that the method better preserves source features than UMAP or other methods?
3. A recent criticism of dimensionality reduction methods and their interpretability in the context of biology  [Chari & Pachter PlosCompBio (2023)] might be relevant to refer too. Would you say the criticism equally applies to your method or would you suggest the proposed method somehow improve the interpretability of such results?

### Soundness
2

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
3

### Summary
This paper introduces a novel interpretable nonlinear dimensionality reduction method called FeatureMAP, which aims to enhance the interpretability of reduced-dimensional results by preserving source features. FeatureMAP utilizes local Principal Component Analysis (PCA) to approximate tangent spaces and computes gradients to reveal feature directions and importance. Additionally, it embeds tangent spaces into low-dimensional space while maintaining alignment between them, preserving local anisotropic density. The authors evaluate FeatureMAP on various datasets, and compare it with existing dimensionality reduction methods.

### Strengths
1.FeatureMAP offers a new perspective by combining local PCA and tangent space embedding to enhance the interpretability of nonlinear dimensionality reduction, which is a commendable innovation.
2.The paper conducts extensive experiments on several standard datasets, validating FeatureMAP's effectiveness in preserving features and density.
3.The paper effectively demonstrates FeatureMAP's ability to distinguish different categories and explain adversarial samples through visualization results and compares the proposed meathod with some of the most advanced methods, showing its advantages in certain aspects.

### Weaknesses
1.The paper mentions the selection of parameter λ but does not discuss the impact of other parameters on the results. Guidance on parameter selection and sensitivity analysis is crucial for practical applications.
2.The paper should discuss the computational complexity of FeatureMAP, especially its performance and scalability when dealing with large datasets. This is a key factor for the feasibility of the method in practical applications.

### Questions
1.What is the computational resource consumption of FeatureMAP when dealing with large datasets? Are there optimization strategies to improve efficiency?
2. In what aspects does FeatureMAP significantly differ from the latest dimensionality reduction methods such as PHATE and SpaceMAP?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper advances manifold learning by preserving both the topological structure and source features within the tangent space, facilitating interpretability through local feature gradient analysis in the embedded tangent space. Extensive experiments in digit classification, object detection, and adversarial example analysis demonstrate the enhanced interpretability of FeatureMAP.

### Strengths
This paper introduces a newly designed method that provides interpretability for nonlinear dimensionality reduction. The theoretical proofs are thorough, the experiments are comprehensive, and the writing is clear.

### Weaknesses
Missing or incorrect punctuation following equations (e.g., Equation 11). The APPENDIX layout is somewhat unstructured.

### Questions
1.	Explicitly outline what current methods (e.g., UMAP) lack regarding feature preservation in the introduction, and provide a brief comparative summary to showcase why FeatureMAP fills a distinct gap.
2.	In the adversarial attack interpretation experiment (the right of Fig. 6), it is difficult to observe additional important features beyond the typical edges of digit 1. In other words, the saliency map shown is insufficient to explain the reasons for misclassification.
3.	Figure 13 shows that the computation time of FeatureMAP exceeds that of other methods. Is this time acceptable?

### Soundness
4

### Presentation
3

### Contribution
3
