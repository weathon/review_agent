# 3DSMT: A Hybrid Spiking Mamba-Transformer for Point Cloud Analysis

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 8, 2, 6

## Abstract
The sparse unordered structure of point clouds causes unnecessary computation and energy consumption in deep models. 
Conventionally, the Transformer architecture is leveraged to model global relationships in point clouds, however, its quadratic complexity restricts scalability. Although the Mamba architecture enables efficient global modeling with linear complexity, it lacks natural adaptability to unordered point clouds. 
Spiking Neural Network (SNN) is an energy-efficient alternative to Artificial Neural Network (ANN), offering an ultra low-power event-driven paradigm. 
The inherent sparsity and event-driven characteristics of SNN are highly compatible with the sparse distribution of point clouds. To balance efficiency and performance, we propose a hybrid spiking Mamba-Transformer (3DSMT) model for point cloud analysis. 3DSMT integrates a Spiking Local Offset Attention module to efficiently capture fine-grained local geometric features with a spiking Mamba block designed for unordered point clouds to achieve global feature integration with linear complexity. Experiments show that 3DSMT achieves state-of-the-art performance among SNN-based methods in shape classification, few-shot classification, and part segmentation tasks, significantly reducing computational energy consumption while also outperforming numerous ANN-based models.
Our source code is in supplementary material and will be made publicly available

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a hybrid architecture 3DSMT (Hybrid Spiking Mamba-Transformer), designed to achieve efficient and accurate point cloud analysis. The model combines spiking neural dynamics, state-space modeling (Mamba), and the Transformer attention mechanism, achieving a strong balance between performance and energy efficiency.

● 3DSMT is composed of three core modules:Spiking Local Offset Attention (SLOA): captures local geometric features;

●Spiking Mamba Block (SMB): performs global feature integration with linear complexity;

●Spiking Position Encoding (SPE): encodes the spatial and temporal structure of unordered point clouds.

 Experiments on ModelNet40, ScanObjectNN, and ShapeNetPart demonstrate that 3DSMT achieves state-of-the-art (SOTA) performance among Spiking Neural Network (SNN) models.

### Strengths
1.The model elegantly integrates spiking neural networks, the Mamba dynamic mechanism, and the Transformer structure, effectively balancing accuracy and energy efficiency.

2.The experimental validation is extensive and comprehensive: 3DSMT performs consistently well across both classification and segmentation tasks.

3.The feature visualization is well presented and convincingly demonstrates the model’s representational capability.

4.The ablation studies systematically evaluate the effects of timestep and data augmentation strategies, providing good empirical support.

### Weaknesses
1.The novelty is somewhat limited: compared with Spiking Point Mamba (SPM) or Spiking Transformer, 3DSMT appears more as an engineering-level integration rather than a fundamentally new learning paradigm.

2.The paper lacks discussion on real-time performance: latency and inference speed are not analyzed, which are crucial for SNN and neuromorphic applications.

3.Scalability to large-scale point clouds is not validated — experiments are limited to inputs of 1K–2K points, and it remains unclear how the method performs on significantly larger point sets.

### Questions
1.What is the processing order in the Spiking Mamba Block? Is it executed sequentially like traditional SSMs, or in parallel within each timestep? Given that Mamba inherently depends on token order, how does the model maintain global permutation invariance?

2.The paper claims that 3DSMT achieves linear complexity, yet the architecture still includes Transformer modules, whose attention mechanism typically exhibits quadratic complexity with respect to the number of tokens. Why, then, is the complexity described as linear in the **abstract**? This point is particularly important and should be explicitly discussed, as it directly affects the validity of the complexity claim in both the abstract and the main text.

Although there are some open issues and limitations, the paper demonstrates solid innovation, completeness, and clarity. I believe it represents a high-quality and meaningful contribution to the field.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors propose a hybrid spiking machine learning framework called 3DSMT (3D Spiking Machine Learning Transformer) for event-driven 3D perception tasks. This method combines the energy efficiency of spiking neural networks (SNNs) with the global modeling capabilities of the Transformer, achieving efficient 3D representation learning of event streams through cross-modal fusion and spatiotemporal feature encoding. Experiments were conducted on multiple event datasets, including event camera 3D object recognition and dynamic scene understanding tasks, demonstrating a good balance between performance and energy consumption.

### Strengths
This method combines the energy efficiency of spiking neural networks (SNNs) with the global modeling capabilities of the Transformer, achieving efficient 3D representation learning of event streams through cross-modal fusion and spatiotemporal feature encoding. Experiments were conducted on multiple event datasets, including event camera 3D object recognition and dynamic scene understanding tasks, demonstrating a good balance between performance and energy consumption.

### Weaknesses
1.	The structure diagram in Figure 1 is quite complex, and the input-output relationship logic of some modules is not intuitive enough.
2.	The lack of independent ablation analysis for key modules makes it difficult to assess the contribution of each component. It is recommended to supplement this with a clear ablation comparison table (Baseline / +SNN / +Hybrid / +Full Model).
3.	Figure 4 shows a lack of clear contrast, requiring layout adjustments to enhance readability.
4.	Lack of performance on large-scale point clouds (such as SemanticKITTI, Nuscenes)
5.	Figure 1 and Figure 5 are duplicates; it is recommended to modify either one of them.

### Questions
See the weakness.

### Soundness
4

### Presentation
4

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
In this paper, authors proposed 3DSMT, a method that combined 1) Spiking Neural Network, 2) Transformer, 3) Mamba for point cloud analysis. Results on point cloud classification and part segmentation show the effectiveness of proposed method. In summary, this is a conventional work that combined several methods from different research topics without solid motivation.

### Strengths
1. This work has a relatively good presentation.

### Weaknesses
1. Combining several methods from different topics and different domains without solid motivation is of no contribution. 

Please consider the follow questions:

a. what is the motivation of  Spiking Neural Network for point cloud analysis without focusing on explainability, low-energy cost in the Experiment section?

b .what is the motivation of Mamba block for point cloud analysis? Point cloud are naturally not causal ordered, and various solutions for model efficiency over Transformer, like linear attention, conv, MLP. Why consider Mamba block for point cloud?


2. While authors emphasized efficient global context modeling, point cloud classification and part segmentation are not enough to demonstrate the efficiency. Authors shold conduct experiments on large-scale point cloud applications, like out-door LiDAR self driving scene. 

3. The baselines are work and strong baselines are ignored. For example, simple PointMLP and PointNeXt shown promising performance using only MLP, but these methods are ignored in Table 1.

PointMLP: Ma, Xu, et al. "Rethinking network design and local geometry in point cloud: A simple residual MLP framework." arXiv preprint arXiv:2202.07123 (2022).

PointNeXt: Qian, Guocheng, et al. "Pointnext: Revisiting pointnet++ with improved training and scaling strategies." Advances in neural information processing systems 35 (2022): 23192-23204.

### Questions
I would like to empahsize that combining multiple technologies from different domains and different research directions without solid motivation or strong empirical improvements cannot be considered as research.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a hybrid Spiking Mamba-Transformer architecture named 3DSMT for point cloud analysis. It includes two main components: (1) Spiking Local Offset Attention (SLOA) used to capture local geometry features; (2) Spiking Mamba Block (SMB) is designed for global, linear-time modeling for unordered point clouds. Extensive experiments on multiple benchmarks, including ModelNet40, ScanObjectNN variants, and ShapeNetPart, demonstrated strong accuracy and energy efficiency compared to previous SNN and some ANN baselines.

### Strengths
1. The paper is well organized; the figures are readable and understandable. 

2. The experimentation is quite complete and adequate and sustains the author's claims. 

3. The proposed method looks logical and technically sound.

### Weaknesses
1. The related work analysis of ANN-based point cloud models is incomplete; very highly relevant and recent works were not discussed in this section, such as Point Transformer V3, the Hybrid Transformer-Mamba model (PoinTramba), and so on.  The author should consider expanding the ANN-based related work and give a critical comparison. 

2. Table 1 shows a huge gap between “w/o voting” and “w/ voting” on ScanObjectNN dataset, but the paper does not specify why the voting mechanism heavily influences the performance.  

3. The single-Transformer comparison primarily relies on Point Transformer v2; the experimental section does not include more recent/effective Transformer variants (like Point Transformer V3) as baselines on the three main benchmarks. This may weaken its superiority compared to recent ANN models. 

4. It seems the authors manually wrote the citation, and the format fails to comply with ICLR’s citation guidance. Please review the guidance.

### Questions
1. Could the authors specify the exact voting mechanism and whether all baselines were evaluated under the same protocol?  

2. Can you report per ScanObjectNN variants' energy cost with and without voting? 

3. In the Table 7, why does “No Order” outperform Z-order or shuffle? How do you serialize the input point cloud to satisfy the requirements of Mamba?

### Soundness
3

### Presentation
3

### Contribution
2
