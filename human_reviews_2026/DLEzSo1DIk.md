# When Embeddings Models Meet: Procrustes Bounds and Applications

- Decision: Reject
- Scores: 6, 6, 2, 4, 4

## Abstract
Embedding models trained separately on similar data often produce representations that encode stable information but are not directly interchangeable. This lack of interoperability raises challenges in several practical applications, such as model retraining, partial model upgrades, and multimodal search. We study when two sets of embeddings can be aligned by an orthogonal transformation. We show that if pairwise dot products are approximately preserved, then there exists an isometry that closely aligns the two sets, and we provide a tight bound on the alignment error. This insight yields a simple alignment recipe, Procrustes post-processing, that makes two embedding models interoperable while preserving the geometry of each embedding space. Empirically, we demonstrate its effectiveness in three applications: maintaining compatibility across retrainings, combining different models for text retrieval, and improving mixed-modality search, where it achieves state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the fundamental problem of interoperability between separately trained embedding models, where similar semantic information is encoded but the representations are not directly interchangeable. The authors establish a strong theoretical foundation by proving a Procrustes Bound, which rigorously demonstrates that if the pairwise dot products (cosine similarities) between two embedding sets are approximately preserved, then a close-fitting orthogonal transformation (isometry) exists to align the spaces. This tight bound formally links similarity preservation to the existence of an effective linear alignment function. Leveraging this theoretical insight, the paper conducts a series of experiments validating Procrustes post-processing across 3 different embedding alignment use cases.

### Strengths
1. The paper provides a rigorous, novel Procrustes Bound which theoretically guarantees that if cosine similarities between two embedding sets are preserved, a close-fitting orthogonal transformation (isometry) exists. This provides a formal, tight linkage between embedding similarity and direct alignability.

2. The paper provides a thorough empirical analysis of post-alignment downstream performance after aligning embedding models across a variety of use cases.

### Weaknesses
1. While the derivation of the Procrustes Bound is mathematically rigorous, the result that dot-product preservation implies the existence of a linear alignment may be an expected consequence of standard contrastive or similarity-based training objectives used for all modern embedding models. Since these models are explicitly trained to make the dot product the central measure of semantic similarity, showing that a linear, orthonormal transformation aligns them when similarities are close is arguably a finding that validates the training objective, rather than an independent, deep insight into the structure of representation space. The paper could be strengthened by expanding their analysis to non-similarity-preserving embedding spaces.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles the task of aligning embeddings produced by different models since embedding models trained separately on similar data often produce representations that encode stable information but are not directly interchangeable. It proposes a method that uses orthogonal Procrustes transformations and prove that if two embedding spaces approximately preserve pairwise dot products, there exists an orthogonal transformation that closely aligns them. Finally, the authors test the proposed approach in several scenarios: maintaining compatibility across retrainings, combining different models for text retrieval, and improving mixed-modality search.

### Strengths
The paper tackles an important problem. It provides both a theoretical foundation and also shows empirically that the proposed method achieves good results. I feel like this idea can be very useful for the research community.

### Weaknesses
My main concern is related to the comparison with state of the art which I find relatively shallow. While probably not directly comparable, it would be good to understand the difference in performance to other alignment methods, some of them being based on the Procrustes transformations such as:

[Quantized Wasserstein Procrustes Alignment of Word Embedding Spaces](https://aclanthology.org/2022.amta-research.15/) (Aboagye et al., AMTA 2022)

This leads me to my next point, about a somehow limited novelty since the idea of using the orthogonal Procrustes transformations for embedding alignment was studied before as the authors point out so it would be good to better emphasize the differences with existing methods.

### Questions
* What are some failure and limitations of the proposed method?
* Does the method handle embeddings of different dimensionalities beyond zero-padding? What are the effects of zero-padding vs performance?

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
This paper studies when two sets of embeddings can be aligned using orthogonal transformations (Procrustes alignment). The authors provide a tight theoretical bound showing that if dot products are approximately preserved between embedding sets, then there exists an orthogonal transformation that can align them well. They demonstrate this approach across three applications: maintaining compatibility during model retraining, combining different models for text retrieval, and improving mixed-modality search.

### Strengths
Well-motivated: The paper addresses a practically important challenge, efficiently aligning embedding spaces without costly re-indexing, which is highly relevant for production systems.
Comprehensive experiments: The evaluation covers three diverse applications with thorough experimental design, including baselines and ablations.
Clear presentation: The paper is well-written with good intuitive explanations and effective visualizations.

### Weaknesses
My main concern with this paper is the limited empirical impact: The key assumption that embedding models preserve dot products often doesn't hold in practice as model updates typically involve shifts in embedding geometry (otherwise there would be no need to update them). This is also mirrored by performance limitations: The results show that these simple transformations (whether orthogonal or not) rarely achieve performance between the source and target models as desired. Strong improvements only occur when the source model is very weak, limiting practical utility: in reality the updates are more likely to be incremental wrt performance - large gains can offset the cost of re-indexing. 

Overall, while the paper addresses an important problem with solid theory and thorough experiments, the practical impact is limited due to restrictive assumptions and modest empirical gains in realistic settings.

### Questions
No questions, the paper is better suited for a more focused venue.

### Soundness
3

### Presentation
3

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
The paper investigates the theoretical analysis Procrustes analysis. It establishes that if the pairwise dot products between the two embedding matrices are approximately preserved, then there is a tight upper bound on the Procrustes distance. The experiments show that procruste outperforms other baseline transformation on Model retraining, Partial upgrades and Multimodal embeddings.

### Strengths
1. The bound is mathematically tight and improves on earlier results.
2. Experiments are comprehensive and demonstrate practical relevance across diverse applications.

### Weaknesses
1. Unverified Assumption: The main theorem assumes that pairwise dot products between embeddings are approximately preserved. However, the paper does not empirically verify whether this assumption holds for real-world embedding models in the application section.
2. Lack of Theoretical–Practical Connection: While the theory focuses on bounding the alignment error (upper bound), the experiments mainly show that Procrustes performs well empirically. Are there any synthetic experiments that verify the upper bound’s dependence on $D^{1/4}$ and $\epsilon^{1/2}$.
3. No New Methodology or Improvement: The paper provides theoretical justification for the standard Procrustes method but does not propose a new algorithm or modification inspired by the theory. As a result, the contribution is primarily analytical rather than methodological.
4. Limited Applicability: The Procrustes method inherently requires embeddings to have the same dimensionality and assumes a orthogonal relationship. In real-world scenarios where embedding spaces differ in dimension or where the transformation is nonlinear, this approach is not directly applicable.

### Questions
1. Under what specific conditions does the Procrustes method work effectively, and in what situations would it fail ?
2. Why don’t the authors compare Procrustes with other commonly used alignment methods, such as Canonical Correlation Analysis (CCA) or Centered Kernel Alignment (CKA), which can also measure or enforce embedding similarity?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors study when two independently trained embedding spaces that encode stable information but are not directly interchangeable can be made interoperable via an orthogonal procrustes alignment. The main theoretical result in the paper is that if pairwise dot products are approximately preserved, i,e. The gram matrices are close in frobenius norm, then there exists an orthogonal transformation with procrustes bounded error. The algorithmic alignment uses standard orthogonal procrustes solved via SVD. Empirical results are shown on three applications, model retraining with retrieval and classification, partial upgrades for text retrieval and mixed modality search.

### Strengths
1. The theoretical bound and results are simple and improves prior dependence on N in settings which have large datasets. 

2. Orthogonal procrustes is computationally simple and it keeps the source space geometry. So the overall idea is practical and relevant for the community. 


3. The experiments cover different settings such as retraining, cross-model retrieval and multimodal search, which is exhaustive as far as the applications are concerned.

### Weaknesses
1. There are mismatches between the theory and practice. The experiments vary the number of training pairs but there is no formal generalization bound relating gram-matrix closeness on a sample to the alignment error on the full data. Also the theory requires pairwise dot product and same indexed objects for a paired X, Y in the SVD computation step, however for setting mentioned in the partial updates where raw documents are unavailable, where would the paired data come from and how representative are the measurements then? 

2. Some of the experimental details are unclear. For multimodal, it is not mentioned which modality is rotated and how the pairs are sampled. For retrieval, no MRR or Recall@k is reported. The results on classification with retraining interestingly show that training time methods beat post hoc alignment including retraining, however the authors don’t discuss this. 


3. The paper is missing some analysis and experiments on orthogonal with scale and orthogonal with whitening, and regularized linear baselines.

### Questions
See Weaknesses section above.

### Soundness
2

### Presentation
2

### Contribution
2
