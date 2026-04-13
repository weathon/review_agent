## Human Reviewer 1

### Summary
The paper introduces a improved approach to applying SSMs to 3D point cloud data. The study presents NIMBA, a Mamba-based model that uses a unique reordering strategy to convert 3D point clouds into 1D sequences while preserving spatial relationships, thus eliminating the need for positional embeddings and reducing data redundancy and computational overhead. Additionally, the NIMBA model demonstrates enhanced robustness to common data transformations.

### Strengths
1. The idea is simple and the paper is easy to follow.

2. Introducing Mamba to point clouds is non-trival and further improve the efficiency and robustness is important.

3. The analysis is soundness.

### Weaknesses
1. PointMamba, the article's main comparator, has been accepted by NeurIPS, and its methodology has been updated. The authors should revise the relevant descriptions in the article accordingly.

2. In the paper, the authors assert that NIMBA does not rely on positional embedding (PE). However, the ablation study in section 4.3.1 indicates that NIMBA performs better with PE, contradicting their claim (see line 97). Additionally, the results in Table 2 also seem to include PE, leading to confusion. Furthermore, since PE is easy to compute and doesn't significantly increase computational burden, the claim that NIMBA can contribute without it raises questions, especially given that omitting PE results in decreased performance

3. The reviewer is also unsure how NIMBA validates global modeling with a sequence length N in the causal modeling Mamba. While sequence order can help preserve geometric relationships, the point patches still struggle to interact with each other. More discussion can be added.

### Questions
See Weakness. Besides, there are few other suggestions:

1. Details in the writing require verification. For instance, the caption in Table 5 seems inaccurate; a full stop is missing at the end of lines 339 and 360.

2. The reviewers understand that the authors trained from scratch to better highlight the difference between each setting. However, stronger data augmentation and pre-training fine-tuning could still be added to demonstrate the upper limit of NIMBA's performance.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
5

---

## Human Reviewer 2

### Summary
This paper provides detailed introductions to existing Mamba-based point cloud analysis strategies. Then, this paper proposes a point cloud state space method named NIMBA to remove positional embedding and avoid data replication in related methods. Experiments demonstrate that the NIMBA outperforms PointMamba in robustness while facing spatial variations.

### Strengths
1. The target to remove positional embedding and avoid data replication is valuable.
2. The overall writing is fluent and clear.

### Weaknesses
1. Since point clouds are 3D data, would ordering them along a single axis be sub-optimal?
2. Point clouds are highly spatially scattered and disordered. A manually pre-defined $r$ may not be suitable for all scenes.
3. This paper can provide an inference time analysis to present an improvement in efficiency by avoiding data replication.

### Questions
1. Does Figure 2 illustrate that feeding wrong ordering centers to MAMBA Layers?

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper introduces a novel method for processing 3D point clouds using State Space Models (SSMs), specifically the Mamba model. The key innovation is a strategy to convert 3D point clouds into 1D sequences that preserves the spatial structure without requiring data replication, thus enabling efficient sequential processing by Mamba in a permutation-invariant manner. The authors claim that their method surpasses Transformer-based models in accuracy and efficiency and does not require positional embeddings. The paper reports state-of-the-art results on ModelNet40 and ScanObjectNN datasets and demonstrates improved robustness against data transformations such as rotations and jittering.

### Strengths
1、The linear complexity of SSMs like Mamba, as opposed to the quadratic complexity of Transformer models, makes NIMBA highly scalable for high-resolution 3D data.

2、The paper shows that NIMBA is more robust to data transformations such as rotations and jittering, which is crucial for real-world applications where data can be subject to various distortions.

3、The elimination of positional embeddings and sequence replication makes the model more principled and less reliant on artificial constructs for sequence ordering.

### Weaknesses
1、One of the contributions proposed in this paper is the reordering strategy. However, this serialization strategy should be compared and discussed with the Point Cloud Mamba[1]. In the Point Cloud Mamba, many methods are discussed and compared, but they are all missing in this paper. There is even no specific discussion and comparison in the ablation studies. Furthermore, the performance of NIMBA's reordering strategy is dependent on the choice of the threshold parameter, which may require careful tuning for different datasets.

2、The paper notes that NIMBA shows limited improvement when scaled, suggesting that there may be optimization challenges that need to be addressed. There is a noted decline in performance when integrating NIMBA with Mamba2 or in hybrid architectures, indicating potential issues with model integration.

3、Comparative Analysis: The paper primarily compares NIMBA with Mamba-based models; a more comprehensive comparison with other state-of-the-art methods, especially those using different SSMs [1], [2], could provide a fuller picture of NIMBA's performance. Furthermore, this paper claims to surpass the transformer-based method, but it lacks many comparisons with such methods, such as PointBert[3], PointM2AE[4].

4、 How was the threshold parameter determined, and how sensitive is the model's performance to changes in this parameter? How does the removal of positional embeddings affect the model's interpretability, and can the learned representations be easily understood?

5、While the paper claims efficiency improvements, are there specific computational cost analyses, especially for large-scale datasets?

[1] Zhang T, Li X, Yuan H, et al. Point could mamba: Point cloud learning via state space model[J]. arXiv preprint arXiv:2403.00762, 2024.

[2] Han X, Tang Y, Wang Z, et al. Mamba3d: Enhancing local features for 3d point cloud analysis via state space model[J]. arXiv preprint arXiv:2404.14966, 2024.

[3] Yu X, Tang L, Rao Y, et al. Point-bert: Pre-training 3d point cloud transformers with masked point modeling[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 19313-19322.

[4] Zhang R, Guo Z, Gao P, et al. Point-m2ae: multi-scale masked autoencoders for hierarchical point cloud pre-training[J]. Advances in neural information processing systems, 2022, 35: 27061-27074.

### Questions
Refer to weakness part.

### Soundness
2

### Presentation
4

### Contribution
2

### Rating
3

### Confidence
5

---

## Human Reviewer 4

### Summary
The paper introduces NIMBA, a new method for point cloud analysis using state-space models (SSMs). The key contribution is a new strategy that converts 3D point clouds into 1D sequences while preserving spatial structure, eliminating the need for positional embeddings. NIMBA builds on the Mamba architecture to improve efficiency and accuracy in tasks like object classification and segmentation. The authors demonstrate that NIMBA outperforms both transformer-based and other SSM-based methods on multiple datasets such as ModelNet40 and ScanObjectNN, showing enhanced robustness to noise and spatial transformations.

### Strengths
1. The paper proposes a novel method for converting 3D point clouds into sequences without replication or positional embeddings.
2. Significance: NIMBA achieves state-of-the-art results on multiple datasets, outperforming both transformer and SSM-based baselines.
3. The methodology is well-structured, with clear comparisons to prior work, though some sections could be streamlined for better readability.

### Weaknesses
1. Scaling limitations: The model shows limited improvement when scaled, suggesting potential optimization challenges. It would be better if they could verify the effectiveness on larger point cloud datasets, like nuScenes, and Waymo.
2. Performance declines were observed when replacing the Mamba block with the Hydra block, indicating possible limitations in hybrid architectures.
3. some typos should pay attention to, e.g., 
  a) line 099: positional emebddings might be positional embeddings;
  b) line 352: flattered by ordering might be flattened by ordering;
  c) line 377: environment might be environment
  d) line 523: conclusion and Fig. 2 show
  and some formatting issues

### Questions
1. Could the authors provide more intuition behind the choice of the proximity threshold of 0.8?
2. Have the authors considered alternative strategies for scaling the model to larger datasets, like nuScenes and Waymo?
3. Hybrid models: what are the potential avenues to optimize NIMBA when integrated with hybrid architectures like Hydra?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
3

### Confidence
4