# Point-Focused Attention Meets Context-Scan State Space: Robust Biological Visual Perception for Point Cloud Representation

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Synergistically capturing intricate local structures and global contextual dependencies has become a critical challenge in point cloud representation learning. To address this, we introduce PointLearner, a point cloud representation learning network that closely aligns with biological vision which employs an active, foveation-inspired processing strategy, thus enabling local geometric modeling and long-range dependency interactions simultaneously. Specifically, we first design a point-focused attention, which simulates foveal vision at the visual focus through a competitive normalized attention mechanism between local neighbors and spatially downsampled features. The spatially downsampled features are extracted by a pooling method based on learnable inducing points, which can flexibly adapt to the non-uniform distribution of point clouds as the number of inducing points is controlled and they interact directly with point clouds. Second, we propose a context-scan state space that mimics eye's saccade inference, which infers the overall semantic structure and spatial content in the scene through  a scan path guided by the Hilbert curve for the bidirectional S6. With this focus-then-context biomimetic design, PointLearner demonstrates remarkable robustness and achieves state-of-the-art performance across multiple point cloud tasks. The code is available at https://github.com/Point-Cloud-Learning/PointLearner.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes PointLearner, a biomimetic point cloud representation learning network that aims to simulate the "focal-context" perception mechanism of biological visual systems to collaboratively capture both the local fine structure and global context dependencies of point clouds. Specifically, Point-focused attention models local fine details and global semantic dependencies through a local neighbor branch and a spatial downsampling branch. Context-scan state space uses a Hilbert curve to serialize the point cloud and integrates a bidirectional S6 model to dynamically scan and integrate the global semantic structure.

### Strengths
1. The dual-path design of local neighbor branch and spatial downsampling branch effectively balances the contribution of local structure and global semantics.
2. To address the non-uniform distribution of point clouds, the paper introduces a learnable inducing points pooling method. By enabling attention interaction between inducing points and the original point cloud, this method flexibly extracts global semantics and shows more stable performance at higher sampling rates compared to the farthest point sampling method.
3. The model demonstrates promising performance improvements over existing methods on four mainstream point cloud datasets (ModelNet40, ShapeNet, S3DIS, ScanObjectNN).

### Weaknesses
1. The paper does not provide a comparison of the computational cost of the proposed model with previous methods. Each encoder block not only incorporates a two-branch architecture but also adds a state space model for global feature modeling. This may lead to a significant increase in computational overhead, which requires further clarification.
2. The spatial downsampling branch of the Point-focused attention is capable of capturing global features, so the motivation for introducing the state space model for additional global feature modeling is not well justified. While the authors mention simulating eye jump inference, this design still seems redundant and lacks sufficient novelty, as it closely resembles the design of PointMamba.
3. The details of model training are not sufficiently clear. It is recommended to include a more detailed description of the model architecture across different datasets in the form of tables to help readers better understand the model’s structure and implementation.
4. The experimental comparison is flawed, as it does not distinguish self-supervised models, which could lead to unfair comparisons. For example, pre-trained models on ShapeNet, such as PointGST, may not directly apply to segmentation tasks in indoor scenes. To make the comparison fairer on the S3DIS dataset, it would be more appropriate to include self-supervised methods pre-trained on indoor scene data, such as Sonata.

### Questions
1. The serialization strategy in this paper converts 3D data into a 1D sequence, which may disrupt the adjacency relationships between points. Mamba3D argues that serialization is unnecessary, and StructMamba3D also forgoes serialization in favor of spatial state modeling to preserve adjacency relationships. Why does this paper still choose to adopt the Hilbert curve serialization strategy? Are there sufficient experimental results to demonstrate its advantage in terms of model performance?
2. Can the proposed method be applied in the context of self-supervised learning? Leveraging large amounts of unlabeled data to enhance feature modeling capabilities is an important direction. Increasing the dataset size can also help transformer models with large receptive fields to achieve good local or structural modeling. Investigating the impact of self-supervised pre-training on model performance and validating the model’s scalability would further enhance its applicability.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes PointLearner, a point cloud representation learning network that is inspired by biological vision mechanisms.
The method introduces two main components: (1) a point-focused attention module that mimics foveal vision by combining fine-grained local neighbor attention with coarse-grained global semantics from a novel induced point pooling strategy, and (2) a context-scan state space module that serializes point clouds using a Hilbert curve and applies bidirectional S6 to simulate saccadic eye movements for global scene inference. 
The authors demonstrate state-of-the-art performance on ModelNet40, ShapeNet, S3DIS, and ScanObjectNN datasets, with particular robustness to strong noise and varying point densities.

### Strengths
1. The analogy to foveal vision and saccadic eye movements is well-motivated, intuitively aligning with the challenges of point cloud learning, including local detail preservation and global context integration. 

2. The work effectively combines the strengths of attention (local geometry sensitivity) and state space models (long-range dependency modeling) within a biologically plausible framework, demonstrating the potential of such hybrid paradigms.

### Weaknesses
1. The biological analogy is intuitive but somewhat heuristic. 
A more rigorous theoretical or empirical analysis of how closely the modules mimic biological vision would strengthen the motivation and be helpful for understanding.

2. Visualizing heat maps could be useful for better understanding the attention responses and the advantage of the proposed method.

### Questions
What are the specific advantages of using the Hilbert curve over the Morton curve or learned serialization strategies in terms of spatial locality preservation, continuity, and computational efficiency?

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
3

### Summary
This paper proposes PointLearner, a biologically inspired point cloud learning framework that integrates local geometric modeling and global contextual reasoning. The key idea is to mimic human visual perception, which combines a point-focused attention mechanism  with a context-scan state space. Experiments on several benchmarks show state-of-the-art performance and strong robustness to noise and varying point densities.

### Strengths
1. A novel biomimetic perspective. The motivation of this paper is very interesting, which is derived from biological visual analogies, effectively balancing local attention and the understanding of global features.

2. Robust architectural design. The proposed point-focused attention module and state space module are well integrated and can effectively handle noise and sparse sampling, demonstrating good generalization ability.

3. Clear structure and writing. This paper is logically clear and easy to follow.

### Weaknesses
Lack of domain-specific motivation. Although the biological vision analogy is creative, the paper does not clearly explain why this mechanism is particularly suitable for point clouds. The proposed “foveation + saccade” process seems domain-agnostic and could also be applied to images or videos. Without a clear justification, the choice of point cloud data feels somewhat arbitrary.

### Questions
Why did the authors choose to apply this biomimetic architecture specifically to point clouds rather than to 2D images or videos, where visual perception analogies might seem even more natural?

Have the authors considered testing or conceptually discussing how this framework would perform on other vision data (e.g., images or video) ?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes PointLearner, a point cloud representation learning network that closely aligns with biological vision by employing an active, foveation-inspired processing strategy, thus enabling simultaneous local geometric modeling and long-range dependency interactions.

### Strengths
1.The motivation of explaining the network design from a biological vision perspective is interesting;
2.The argument in the Introduction regarding the limitations of local attention is correct;
3.The goal of "synergistically capturing local fine-grained structures and global contextual dependencies" is valid;
4.The experimental results demonstrate the effectiveness of the proposed network architecture;
5.The paper is well-written and easy to understand.

### Weaknesses
1.Although the motivation and starting point of the paper are sound, the core ideas of the proposed solution lack novelty;
2.The Fine-grained Perception module, which updates the center token using k neighboring tokens, is very common apart from the novel use of attention;
3.The Coarse-grained Awareness module is essentially a standard approach that uses learnable queries with cross-attention to aggregate features;
4.The Competitive Normalized Fusion is a commonly adopted strategy when a network has multiple parallel branches;
5.Most of the datasets used are pure point cloud data. To better align with the biological vision motivation, it is recommended to include more datasets with RGB information.

### Questions
see weakness

### Soundness
1

### Presentation
3

### Contribution
2
