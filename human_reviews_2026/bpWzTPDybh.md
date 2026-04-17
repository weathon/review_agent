# RECON: Robust symmetry discovery via Explicit Canonical Orientation Normalization

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Real world data often exhibits unknown, instance-specific symmetries that rarely exactly match a transformation group $G$ fixed a priori. Class-pose decompositions aim to create disentangled representations by factoring inputs into invariant features and a pose $g\in G$ defined relative to a training-dependent, \emph{arbitrary} canonical representation. We introduce RECON, a class-pose agnostic \emph{canonical orientation normalization} that corrects arbitrary canonicals via a simple right translation, yielding \emph{natural}, data-aligned canonicalizations. This enables (i) unsupervised discovery of instance-specific pose distributions, (ii) detection of out-of-distribution poses and (iii) a plug-and-play \emph{test-time canonicalization layer}. This layer can be attached on top of any pre-trained model to infuse group invariance, improving its performance without retraining.  We validate on 2D (images) and 3D (molecular ensembles), demonstrating fine-grained, accurate pose discovery, and matching or outperforming label-supervised
canonicalizations in downstream classification.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper seeks to build on IE-AEs by applying canonical orientation normalization to obtain a different equivalence class than with a typical IE-AE. The authors address instance-specific symmetries, such as rotations in images. The authors also present test-time canonicalization, which allows pre-trained models to be granted invariance with respect to a group action. The authors experiment with a couple of image benchmark datasets, as well as geometric graphs, with SO(2) and SO(3) symmetries.

### Strengths
1. The authors demonstrate the potential advantages of their approach over the most comparable baselines.

2. The presented method appears to be novel.

### Weaknesses
1. *The symmetries considered are limited.* The groups SO(2) and SO(3) do not constitute a diverse collection of symmetries, as lines 348-354 suggest. 

2. *This is not the first work to consider 3-d symmetry,* contrary to the claim made in the abstract. In fact, this statement can mean one of two things: (1) the symmetry group is 3-dimensional (perhaps it is a Lie group); (2) the group action acts on a three-dimensional space. In either case, the claim made by the authors would be false. In fact, I do not believe that this work is the first to consider SO(3) symmetry: for example, see "Image to Icosahedral Projection for SO(3) Object Reasoning from Single-View Images" (Klee et al., 2022).

3. *There are a few key missing references.* In particular, the idea of using the Lie derivative to discover symmetry ("A Unified Framework to Enforce, Discover, and Promote Symmetry in Machine Learning" by Otto et al., 2023) and to subsequently use the discovered symmetries to construct an invariant feature space ("Symmetry Discovery Beyond Affine Transformations by Shaw et al., 2024) seems closely related to the present method and should be discussed, if not experimentally compared with. Another key reference in symmetry discovery is "Learning Infinitesimal Generators of Continuous Symmetries from Data" by Ko et al., 2924.

4. *More experimental comparison would be helpful.* I appreciate the experimental comparison which has been made, but it would be nice to see a comparison against a representative sample of other methods seeking to enforce invariance, such as an equivariant NN or else a method or two mentioned in weakness 3.

### Questions
1. Near line 150, the authors write: "In effect, we rely on the principle that structurally similar objects generally map to nearby points in Z." Is this really true in general? The authors mention empirical evidence, but what symmetries have been considered in these experiments? What does "nearby" mean in the G-invariant latent space? I think we need to be more precise about what this d_Z norm-induced distance is, or perhaps prove that *any* norm-induced distance suffices.

2. What kind of symmetries can you learn? (It seems like the bottle-neck could be the G-invariant encoder, which may be limited in the types of symmetries it can discover.) It seems only rotations are experimented with. The claim is that the method is quite general, but a claim that *any* symmetry can be discovered is quite doubtful, particularly in light of the lack of experimental evidence.

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
The paper introduces a method for unsupervised discovery of instance-specific symmetry distributions in data. Building upon class–pose decomposition frameworks (like Invariant-Equivariant Autoencoders), it corrects arbitrary canonical poses by estimating a Fréchet mean offset and applying a canonical orientation normalization. This yields consistent, data-aligned canonical representations and enables applications such as OOD pose detection and test-time canonicalization. The method is validated on both 2D datasets (MNIST, FashionMNIST) and 3D molecular datasets (GEOM-QM9).

### Strengths
- The canonical orientation normalization via the Fréchet mean is simple yet theoretically grounded and effective for unsupervised symmetry discovery.

- The paper provides rigorous proofs and clear geometric intuition, which is scalable to both SE(2) for imaging and SO(3) for geometric graphs.

- This method can be applied to any invariant-equivariant backbone, making it broadly useful.

- The test-time canonicalization and OOD detection tasks are compelling, showing both research and applied potential.

### Weaknesses
- Lack of real-world validation in the experiments. Most experiments involve synthetic rotations or clean 3D datasets; robustness under complex, cluttered scenes for real-world objects, for example, objects in 2D images in the wild, like COCO/ImageNet, or scanned 3D objects in OmniObject3D/GSO. 

- The sensitivity to hyperparameters (e.g., k-nearest neighbors in class construction) could be explored more systematically.

- Lacks direct quantitative comparisons with very recent equivariance-learning approaches (e.g., Partial G-CNNs, VP-GCNNs) beyond conceptual discussion.

### Questions
- What is the training overhead and inference time for RECON’s symmetry detection/OOD pose detection/ test-time canonicalization/reconstruction?

- There are many learning-based methods that take take 3D object model as input and model symmetry pattern/distribution in SO(3), e.g., Implicit-PDF[1], Alignist[2]. Could you explain if RECON can be extended to a more generalized setup like the above-mentioned methods, and what would be the main advantage of RECON compared to those methods if the same datasets apply (for example, Symsol)? 

References:

[1] Murphy, K.A., Esteves, C., Jampani, V., Ramalingam, S., Makadia, A.: Implicitpdf: Non-parametric representation of probability distributions on the rotation manifold. In: Proceedings of the 38th International Conference on Machine Learning. pp. 7882–7893 (2021).

[2] Vutukur, Shishir Reddy, Rasmus Laurvig Haugaard, Junwen Huang, Benjamin Busam, and Tolga Birdal. "Alignist: CAD-Informed Orientation Distribution Estimation by Fusing Shape and Correspondences." In European Conference on Computer Vision, pp. 351-369. Cham: Springer Nature Switzerland, 2024.

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
3

### Summary
The paper introduces RECON, an unsupervised framework for discovering instance-specific symmetries from data.
RECON builds upon class-pose decomposition methods (e.g., IE-AE) and addresses the key limitation of arbitrary canonical poses by normalizing them into natural orientations through an estimated Tukey-Fréchet mean of relative transformations.
This canonical orientation normalization yields interpretable, data-aligned symmetry distributions and enables downstream applications such as (i) OOD pose detection and (ii) test-time canonicalization that grants invariance to pretrained models without retraining.
Experiments on 2D image datasets (MNIST, FashionMNIST) and 3D molecular data (GEOM-QM9) demonstrate the effectiveness of the proposed symmetry discovery method itself as well as its benefits in downstream tasks.

### Strengths
- The approach discovers symmetry distributions from unlabeled data and provides meaningful, identity-centered canonicalizations that align with intuitive notions of natural pose.

- Interesting and elegant idea: The proposed Tukey-Fréchet-mean–based canonical orientation normalization provides a simple yet effective solution to the issue of arbitrary canonicalization in class-pose methods. The modeling is conceptually clear, mathematically grounded, and intuitively appealing.

- RECON can grant group invariance to pretrained models without retraining, functioning as a plug-in canonicalization step that applies irrespective of model architecture. This design significantly enhances the method’s usability and potential impact.

- The paper is clearly structured and communicates its ideas effectively, making both the theoretical formulation and practical implications easy to understand.

### Weaknesses
- Inaccurate contribution claim. The paper claims to “-for the first time– extend symmetry discovery to 3D groups,” but prior works such as [1] and [2] have already explored 3D group symmetries.

- The current method is restricted to known transformation groups, primarily rotations (SO(2) and SO(3)), limiting its applicability to more general groups or unknown symmetry settings where the underlying group structure must be inferred.

- The method assumes that variations within each equivalence class are small and can be cleanly separated from group-induced transformations. It remains unclear how reasonable this assumption is in more complex, natural data.

- Computational cost and scalability: The reliance on k-nearest-neighbor search in the invariant space for each input (Algorithm 1) may be computationally expensive for large-scale or high-dimensional datasets.
The paper does not discuss computational cost or whether neighbor computations can be cached or reused during training and inference.

- Some key quantitative results (e.g., Figure 6) appear to be based on single runs without repeated trials or statistical analysis, making it difficult to assess robustness and reliability.
Moreover, the results in Figure 6(d) show a noticeable gap from the so-called upper bound in Table 1.

### Questions
- The equivalence class construction plays a crucial role in the proposed pipeline. Could the authors also provide hit rate results in the image domain?

- Could the authors further discuss the practical utility of discovering symmetry distributions?
It seems that the most tangible application demonstrated is test-time canonicalization, yet other works such as [3] and [4] can achieve a similar effect given a known group structure, without explicitly estimating instance-level distributions.

- In the implementation, does each equivalence class correspond to the entire semantic class, or is it constructed per-sample with a fixed number of neighbors (e.g., 25 as mentioned)?

- How would the method perform on real-world datasets where each object appears at a unique orientation and images are not repeated across poses, unlike the synthetic settings used in this paper?

Reference:

[1] Generative Adversarial Symmetry Discovery.

[2] Latent Space Symmetry Discovery.

[3] Affine Steerable Equivariant Layer for Canonicalization of Neural Networks.

[4] Equivariant Adaptation of Large Pretrained Models.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors introduce a method for identifying the natural pose of a class in an unsupervised way. In contrast to class-pose methods such as IE-AE, the proposed approach learns a data-aligned canonical. This is achieved using the machinery of IE-AE. Embeddings of IE-AE suggest transformation distributions for dataset instances in connected components. The natural pose is defined as the center of this distribution and obtained using the Frechet mean. The authors argue that a data-aligned canonical is important for various downstream tasks and demonstrate its utility for OOD detection and test time canonicalization.

### Strengths
**Originality.** The work appears original. 
 
**Quality.** I rate the quality of this work as good. The work is well motivated, the proposed approach addresses the gap in the literature, and the empirical analysis support the paper claims.
 
**Clarity.** The organization is good, and the writing is very good. 
 
**Significance.** I rate the significance of this work as good. The work has the potential for broad applicability.

### Weaknesses
No notable weaknesses

### Questions
At this phase of the review process I do not have clarifying questions.

### Soundness
4

### Presentation
4

### Contribution
2
