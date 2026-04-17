# $\pi^3$: Permutation-Equivariant Visual Geometry Learning

- Decision: Accept (Poster)
- Scores: 8, 10, 6

## Abstract
We introduce $\pi^3$, a feed-forward neural network that offers a novel approach to visual geometry reconstruction, breaking the reliance on a conventional fixed reference view. Previous methods often anchor their reconstructions to a designated viewpoint, an inductive bias that can lead to instability and failures if the reference is suboptimal. In contrast, $\pi^3$ employs a fully permutation-equivariant architecture to predict affine-invariant camera poses and scale-invariant local point maps without any reference frames. This design not only makes our model inherently robust to input ordering, but also leads to higher accuracy and performance. These advantages enable our simple and bias-free approach to achieve state-of-the-art performance on a wide range of tasks, including camera pose estimation, monocular/video depth estimation, and dense point map reconstruction. Code and models are available at https://github.com/yyfz/Pi3.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel permutation equivariant architecture for feed-forward 3D reconstruction. To achieve permutation equivariance, positional embeddings used to differentiate between frames and learnable tokens that denote a reference view are not used. The structure is straightforward, while the performance is impressive on various tasks.

### Strengths
1. The paper is well-written and easy to follow. 

2. The paper addresses the limitation of existing feed-forward 3D reconstruction pipelines, where one view is selected as a reference view and the performance may degrade if the reference view changes. 

3. The performance in camera pose estimation, monocular/video depth estimation, and pointmap reconstruction, is state-of-the-art. 

4. Ablation study verifies the permutation equivariance of Pi3.

### Weaknesses
1. Camera pose estimation: In addition to using large angular threshold (30 degrees), more strict thresholds, e.g. 5 degrees, 10 degrees, should be used to illustrate the rotation accuracy. 

2. Point-map evaluation: 
    * On both DTU and ETH3D, the numbers of VGGT are very different from those in the original paper. Could the authors explain the reason? 
    * Since the authors already provide accuracy and completeness, it makes sense to further add Chamfer distance/F-score, which is the best metric to evaluate the overall reconstruction quality. 

3. In Table 6, CUT3R heavily relies on the sequential/temporal information. Thus, it is not a very good baseline here.

### Questions
1. To compute the scale $s$ that aligns ground truth and prediction, the other solution is to use umeyama to align the ground truth and predicted camera poses. Different form optimizing Eq. 4, this solution looks simpler. Did the authors try this?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper proposes a VGGT-like model that achieves superior performance across multiple tasks, including pose estimation, depth estimation, and point map estimation.

The performance improvement stems from a novel permutation-invariant model design. Previous methods such as Dust3R, Mast3R, and VGGT rely on a predefined reference frame during estimation, making them sensitive to frame selection and limiting their robustness. In contrast, this work attains permutation invariance by removing all frame order–dependent positional encodings and introducing a scale-invariant point map loss together with an affine-invariant pose loss.

Strengths: The proposed idea is both novel and insightful, supported by thorough and comprehensive experiments.

Weaknesses: No major issues are identified in this paper.

### Strengths
1. Novel and insightful idea: The paper makes an insightful observation about the limitation of defining a reference frame in prior models. The proposed approach to eliminate this dependency is technically sound and straightforward to implement.
2. Comprehensive experiments: The method is evaluated across multiple applications on widely used datasets. Reporting standard deviations in Table 6 is a valuable addition that strengthens the credibility of the results.
Well-presented paper: The paper is clearly written and easy to follow. Figures and tables are well designed, effectively supporting the main arguments and improving readability.

### Weaknesses
I don't see any major issues in this work.

### Questions
How is the rotation mapped to xyz in Figure 6? What's the rotation representation?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes $\pi^3$, a feedforward model for pose estimation and point cloud prediction. It investigates the limitations of previous feedforward reconstruction models, such as VGGT, whose performance depends on the selection of the reference view. To address this issue, the proposed method removes the reference-view-dependent design in the model architecture and introduces a relative camera pose loss and a local point loss. Experimental results show that the proposed method effectively resolves the aforementioned problem and achieves state-of-the-art performance.

### Strengths
- The paper addresses an interesting and important problem: the dependence of previous feedforward reconstruction models on reference view selection. The proposed method provides a simple yet effective solution.

- The proposed model achieves SOTA performance on both pose estimation and reconstruction tasks.

### Weaknesses
- Some parts of the paper lack proper citations:
    - The normal loss is identical to that in MoGe, but no citation is provided.
    - The camera pose estimation module and loss function are the same as in Reloc3r, yet there is no citation in Section 3.3.

- The proposed method requires weight initialization from VGGT, which makes the comparison with other methods, such as VGGT itself, somewhat unfair. Table 8 shows results when both models are trained from scratch; however, it remains unclear how the model performs when trained from scratch only with the local point and pose losses. Furthermore, can the model trained from scratch with the added global loss achieve performance comparable to the version initialized from VGGT?

### Questions
- For monocular depth estimation, how does the performance compare with SOTA models such as DepthAnythingv2? Besides, what is the input image resolution used for depth estimation?
- Since the MLP head with pixel shuffling introduces grid-like artifacts, why does the proposed method still adopt this design instead of using DPT heads as in previous methods?

I will raise my score if the authors can provide reasonable clarifications regarding initialization and the effect of adding the global loss.

### Soundness
3

### Presentation
3

### Contribution
4
