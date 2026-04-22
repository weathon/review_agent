# CanonNet: Spectral Canonicalization and Curvature-Driven Learning for Compact Local-Geometry Point-Cloud Operators

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 4, 2

## Abstract
To address the persistent challenges of scalability and robust local geometry representation in point-cloud processing, we propose CanonNet, a highly efficient local feature operator. 
CanonNet first employs spectral canonicalization to establish an invariant local frame for each neighborhood. It then uses a geometric learning framework, trained on synthetic surfaces, to distill fundamental curvature priors into a lightweight MLP.
This design allows CanonNet to achieve competitive performance on various benchmarks with approximately 100X fewer parameters, while also exhibiting robust domain transfer. 
Its efficiency and design make it an effective building block for deep, hierarchical models, acting as a geometric analogue to the convolution operator for capturing multi-scale features.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes CanonNet, which attempts to extract local geometric feature. It trains on synthetic surfaces to distill fundamental curvature priors to a light-weight MLP. It shows reasonable performance on curvature prediction.

### Strengths
The idea of canonical order and orientation seems useful in local point cloud feature extraction. The idea of distilling curvature priors to MLP is interesting.

### Weaknesses
* The paper lacks a justification behind the proposed method. For example, why the canonical order and orientation is defined in such way. Is it robust to any transform?
* The synthetic curvature is too simple and cannot capture real-world complex point clouds.
* The experiments evaluation is weak. It only consists of two synthetic dataset without any real-world tasks, such as classification and segmentation.
* After all above issues are addressed, the writing needs to be improved substantially.

### Questions
No.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel approach for canonicalizing local patches of a point cloud. Specifically, given a local set of points, a local Laplacian is formed and it's Fiedler vector is computed to canonically order the points in this patch. From this ordering, a local SO(3) frame is computed and the patch is expressed in its coordinate system, providing a canonical layout of the local patch. Subsequently, the authors propose to train a simple MLP on a synthetic dateset, which makes use of the proposed canonicalization to predict the local surface curvature of a given patch.  After these two stages, the pre-trained MLP can be used downstream for a variety of tasks.

### Strengths
The method for frame canonicalization is clever, makes uses of tried-and-true computational tools, and could potentially have broader applications than proposed by the authors.   This paper is also very well written and easy to understand. The proposed method is also straightforward, and appears easy to apply.

### Weaknesses
While the method itself is straightforwards, the paper has several weaknesses as follows: 

- Experiments are not compelling. Curvature estimation is a toy task and despite the authors statement, the model is not competitive with existing approaches on the descriptor retrieval task (with an almost 30% difference in accuracy in favor of prior approaches). Unfortunately, parameter efficiency is not very relevant. A more compelling version of this experiment would investigate the properties of the proposed MLP as a representation learner, where the pre-trained MLP is frozen and a second network trained on the features to predict descriptors in a supervised manner. 

- The frame canonicalization is interesting and relevant (though I have several outstanding questions, see below), and could be more broadly applied than what is considered by the authors here (e.g. as a replacement for estimated frames in 3D equivariant networks, etc.). However, tying this approach to a simple MLP and somewhat unconventional and empirically questionable pre-training regime does not seem to be the best application of canonicalization.  Specifically, the paper does not provide compelling evidence that curvature awareness is a key component of learned descriptors, nor that it makes a useful target for representation learning.  I think this paper would be more compelling if the authors instead explored a variety of different applications for their canonicalization approach. Examples could include replacing estimated frames with their canonicalized version in existing SoTA 3D equivariant networks or using their frame to replace the one estimated in SHOT. 

Overall, the paper lacks compelling applications for the proposed approach so I do not recommend acceptance at this time.

### Questions
- Despite the construction of a local Laplacian and the estimation of the Fiedler vector (which are known to be robust under permutations), it appears that the re-ordering of points can regardless be very sensitive to the sampling. For instance, removing and adding a random point to a local neighborhood would probably change the ordering of the points. How is this addressed?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a local geometry operator for point cloud processing that achieves invariance to rigid transformations and point ordering. The method learns features from a graph constructed using KNN neighborhoods, where the edge weights are computed based on Euclidean distances, ensuring invariance to rotation and translation. The proposed deep learning model is lightweight and shows good performance in curvature estimation and descriptor matching, although it is trained on synthetic data.

### Strengths
1. A local feature learning method based on the Laplacian matrix for spectral canonicalization is proposed, which is invariant to permutation and rigid transformations. 
2. The model is trained on synthetic data, which reduces the cost of data collection. 
3. Experiments are provided for curvature estimation, descriptor matching, and surface classification. The model demonstrates good generalization to unseen data in curvature estimation and descriptor matching.

### Weaknesses
1. The main concern lies in the application potential of the proposed local features. Although the features learned from synthetic data show good cross-domain performance, it is unclear whether the model can be extended to large-scale, real-world point cloud understanding tasks such as shape classification (e.g., ShapeNet or Objaverse), part segmentation, or scene-level semantic analysis.
2. An ablation study on varying neighborhood sizes (different K values in KNN) is recommended to evaluate the expressiveness of the learned features.
3. How does the model perform under variations in point cloud density?
4. Given the edge weights defined in Eq. (1), the rigid transformation invariance of the operator is straightforward. Theorem 1 could be moved to the appendix.

### Questions
1. Figure 5 is not included in the main paper.
2. For shape or surface classification, since the model is invariant to rotations, the authors could consider comparing it with existing rotation-invariant architectures, such as Vector Neurons [a] and Frame Averaging [b].

[a]. Vector Neurons: A General Framework for SO(3)-Equivariant Networks. ICCV 2021. 
[b] Frame Averaging for Invariant and Equivariant Network Design. ICLR 2022.

### Soundness
2

### Presentation
2

### Contribution
2
