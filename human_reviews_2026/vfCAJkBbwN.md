# Spatially Aware Linear Transformer (SAL-T) for Efficient Particle Jet Identification

- Avg Score: 3.60
- Decision: Reject
- Scores: 2, 4, 4, 4, 4

## Abstract
Transformers are very effective in capturing both global and local correlations within high-energy particle collisions, but they present deployment challenges in high-data-throughput environments, such as the CERN LHC. The quadratic complexity of transformer models demands substantial resources and increases latency during inference. In order to address these issues, we introduce Spatially Aware Linear Transformer (SAL-T), a physics-inspired enhancement of the Linformer architecture that maintains linear attention. Our method incorporates spatially aware partitioning of particles based on kinematic features, thereby computing attention between regions of physical significance. Additionally, we employ convolutional layers to capture local correlations, informed by insights from jet physics. In addition to outperforming the standard Linformer in jet classification tasks, SAL-T also achieves classification results comparable to full-attention transformers, while using considerably fewer resources with lower latency during inference. Experiments on a generic point cloud classification dataset (ModelNet10) further confirm this trend.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper tackles particle jet tagging. The proposed method boils down to 
1) sorting the set of points given problem-specific metric;
2) splitting the sorted sequence into chunks and computing a representation of the chunk;
3) computing attention from each token to the chunks;
4) applying depthwise convolutions on attention logits to capture local correlations.

The method is evaluated on jet tagging datasets (hls4ml, Top Tagging, Quark-Gluon) and ModelNet10, demonstrating improved accuracy over Linformer with similar computational efficiency. The experimental evaluation has significant methodological flaws that prevent drawing reliable conclusions: (1) unfair memory comparisons without Flash Attention, (2) uncontrolled parameter counts confounding performance, (3) missing comparisons to relevant SOTA point cloud methods.

### Strengths
- The approach is well motivated - high amount of data and the scale of it make using a standard transformer for on-fly predictions unrealistic.
- The paper addresses a genuine practical need for real-time jet tagging at the HL-LHC, and the focus on trigger-deployable models (microsecond latency, low memory) is valuable for the HEP community.
- The metric used for data organization is theoretically-motivated and taken from jet clustering algorithms.
- Spatial partitioning aligns with the physical intuition that nearby particles in (Δη, Δφ) space are related.
- Ablation studies are helpful, specifically table 1 shows that k_T sampling achieves better performance than p_t sorting.
- The approach outperforms the main baseline - Linformer - including when the depth is increased.

### Weaknesses
- My main concern is the limited choice of baselines for comparison:
  - The approach feels spiritually close to Transolver [1], specifically, "computing attention between regions of physical significance". Yet, the comparison is missing from the baselines and is not even mentioned in related works.
  - The approach is close to Erwin [2], which sorts particles via ball trees and computes attention over groups (balls). The comparison is yet again missing.
  - Comparison to SOTA linear point cloud transformers such as Erwin [2] and PTv3 [3] is missing, even though both have linear complexity and are designed to handle point cloud data.
  - If I understand correctly, each partition is projected to a single vector, and then attention is computed from points to projections. That is very close to what UPT does [4].
- The approach uses simple sorting to organize data. While it seems to work empirically for the dataset authors use, I am not confident that the resulting partitions always make sense. For example, assume we have a N particles, and 3/4 particles are in one half of the domain, 1/4 particles in another. Using sorting + partitioning (proposed method) would generate one proper partition and one disjoint partition where locality assumption is violated. PTv3 handles the case by adaptive voxelization and Erwin uses ball trees with dummy nodes to specifically handle such cases. I believe some statistical analysis of partitioning across any dataset would strengthen confidence in this design choice.
- The paper does not seem to use flash attention, which results in high Peak GPU memory. For this reason, I cannot accept the value and conclusion draw from it - flash attention is linear in memory and would not achieve such high values.
- The strategy of choosing bold numbers is strange to me. Transformer is consistently better than either of methods, so I do believe it should be highlighted. 
- Table 7 shows that accuracy plateaus at p=4 partitions and slightly decreases beyond that (p=8: 81.10%, p=16: 80.97% vs p=4: 81.18%). This suggests the method does not scale with increased compute budget, which is concerning. The authors should discuss why additional partitions hurt performance - is it overfitting, optimization difficulties, or a fundamental limitation of the partitioning strategy?
- Regarding Table 3 - why is the number of params different for each model? I believe a more fair comparison would be to use the same number of parameters and report FLOPs and runtime.
- Additionally, in Table 3, the Transformer achieves 92.91% accuracy with only 2k parameters, while SAL-T achieves 92.52% with 3.5k parameters. This suggests the Transformer is actually more parameter-efficient than SAL-T, contradicting the paper's efficiency narrative. 


[1] Wu, Haixu et al. “Transolver: A Fast Transformer Solver for PDEs on General Geometries.” ArXiv abs/2402.02366 (2024): n. pag.

[2] Zhdanov, Maksim et al. “Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems.” ArXiv abs/2502.17019 (2025): n. pag.

[3] Wu, Xiaoyang et al. “Point Transformer V3: Simpler, Faster, Stronger.” 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023): 4840-4851.

[4] Alkin, Benedikt et al. “Universal Physics Transformers.” ArXiv abs/2402.12365 (2024): n. pag.

### Questions
- How is the proposed approach different from Erwin [2], which also sort particles into partitions and computes attention over those partitions? Originally, I believe it uses euclidean metric, but it might be possible to change it to an arbitrary metric, including the one used by authors.
- Would it be possible to expand the set of baselines to [1,2,3]?
- Do you use flash-attn for transformer implementation? If not, that would explain high peak memory. Therefore, I would ask for reporting values for transformers with flash-attn, since it is a standard approach nowadays, not naive scaled dot product.

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
3

### Summary
The paper introduces a Spatially Aware Linear Transformer for jet tagging, with applicability to point-cloud classification. The method reduces redundant particle connectivities via a sort-and-partition mechanism, and applies a convolutional layer over the attention matrix to enhance locality by influencing nearby particles. Experiments report modest improvements over baseline models.

### Strengths
* The method outperforms baseline approaches.
* It reduces computational overhead relative to baselines while matching the accuracy of a standard Transformer.

### Weaknesses
1. Relation to TIE [1]. is also a Transformer-based method with improved attention mechanics.
2. Outdated baselines. The paper compares mainly to models before 2020. Please include stronger and more recent baselines from the same domain or with similar architectures (e.g., modern Transformer variants, locality-aware/self-attention sparsification methods, and recent point-cloud/jet-tagging Transformers) to contextualize performance.
3. Presentation and organization. The writing and layout can be improved for readability: Reduce excessive white space around tables/figures. Move Figure 2 closer to the method section where it is first referenced. Place Table 5 at the top of a page or immediately after its first mention.
4. Scalability and particle count. Current settings are relatively small (≈150 particles for jet tagging; ≈1k for point-cloud classification). Additional classification results, as suggested in [2], would further substantiate the method’s effectiveness.
5. Limited incremental gains and component effectiveness. Table 1 shows only modest improvements (e.g., +0.18% accuracy) and suggests that sorting contributes more than the convolution layer; in some settings (e.g., with the best sorting strategy $k_T$) in Table 9, removing convolution yields better accuracy. As a result, the convolution seems less effective and the novelty may reduce to only a sorting mechanism.

[1]. Shao, et al. Transformer with Implicit Edges for Particle-based Physics Simulation. ECCV 2022.
[2]. Zhang, et al. Deep Learning-based 3D Point Cloud Classification A Systematic Survey and Outlook. Displays 2023.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper presents Spatially Aware Linear Transformer, an efficient model for high-energy physics tasks such as jet tagging, which utilizes a spatially aware linear attention mechanism. The model builds on the linformer architecture and enhances it with techniques inspired by jet physics, such as spatial sorting and convolutional attention. SAL-T effectively reduces the computational cost compared to traditional transformers while maintaining competitive performance on jet tagging benchmarks.

### Strengths
The integration of spatial sorting and convolutional attention is an innovative approach that addresses both performance and computational efficiency.

 SAL-T’s use of linear attention and partitioned projections significantly reduces complexity, making it viable for real-time systems like LHC triggers.

### Weaknesses
A significant limitation of this paper is its failure to compare SAL-T with other state-of-the-art models for 3D point cloud processing, such as PointNet [1], Point Transformer [2], and Point Transformer V2 [3]: Grouped Vector Attention and Partition-based Pooling. These models are well-established in the point cloud classification space and have demonstrated superior performance on benchmarks like ModelNet10. Since the jet tagging task essentially involves 3D point cloud inputs, the omission of these models for comparison raises concerns. A more thorough comparison with these models would have provided clearer insights into SAL-T's strengths and weaknesses within the broader context of point cloud processing.

Additionally, PointNet, Point Transformer, and Point Transformer V2 have been shown to outperform SAL-T on ModelNet10, which is a key benchmark for point cloud classification. Given that jet tagging inherently involves complex point cloud-like data, it is surprising that these models, which are specifically designed to process such data, were not included for comparison. This failure to test SAL-T against widely adopted point cloud backbones leaves a gap in the paper's evaluation, limiting the ability to fully assess SAL-T's effectiveness and competitiveness within the field of 3D point cloud processing.


[1] Pointnet: Deep learning on point sets for 3d classification and segmentation. ICCV 2017.
[2] Point transformer. ICCV 2021.
[3] Point transformer v2: Grouped vector attention and partition-based pooling. NeurIPS 2022.

### Questions
See weakness

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
3

### Summary
The paper presents the Spatially Aware Linear Transformer (SAL-T), a physics-inspired enhancement of the linformer architecture tailored for particle jet tagging tasks in high-throughput environments like the CERN LHC. SAL-T introduces three main improvements over linformer: spatially aware partitioning of particles based on physically motivated sorting, localized key/value projections, and convolutional layers applied over attention logits to capture local context.

### Strengths
1. The introduction of $k_T$-based sorting and partitioned attention is motivated by the physical characteristics of jet substructure, providing a novel way to leverage domain knowledge in efficient transformer design.

2. SAL-T demonstrates near-linear computational scaling in the number of particles, maintaining inference and resource use compatible with trigger-environment constraints, as shown in Table 2 (e.g., SAL-T: 739,918 FLOPs, 303 MB peak GPU memory vs. Transformer: 2,479,918 FLOPs, 4,357 MB).

3. The significant improvement when using k_t sorting can be seen.

### Weaknesses
1. Regarding the formula for the $k_t$ metric: although it integrates the physical characteristics of jet substructure, I am concerned that if the pair of values $\Delta \eta$ and $\Delta \phi$ are swapped, the value of $k_t$ will not change. Does this mean the partitioning is now faulty?

2. Formulas 3 and 4 in the proposed method confuse me. In formula 3, $ n/p $ may not be an integer, which could cause incorrect indexing of elements. In formula 4, are $P_E$  and $K_P$ multiplied or concatenated? Additionally, Figure 1c is not clear to me.

3. Regarding the experimental results:

- In Table 2, why does SALT (no partition) consume more than twice the memory, while the GPU peak usage is smaller than SALT’s? In Figure 3, SALT shows a slight accuracy advantage with a small number of particles but falls behind when the particle count is large. The results in Tables 3 and 4 do not clearly demonstrate the performance differences between the proposed method and the baselines considered.

- The comparison to other efficient transformer variants (HEPT, Particle Dual Attention Transformer, ParMAT, LorentzNet, HEP-JEPA) is only mentioned as related work and not evaluated experimentally. In particular, a side-by-side quantitative or qualitative comparison of SAL-T against designs like P-DAT (He & Wang, 2023) or ParMAT (Usman et al., 2024), which also target local/global and efficient spatial interactions, is missing.

### Questions
See the weaknesses above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors introduce a linear attention architecture designed for particle physics data, aiming to address the quadratic complexity that leads to high resource demands and inference latency. Their model is based on the Linformer backbone and incorporates two key modifications: partitioning the key and value vectors according to spatial proximity, and applying convolutional layers to the resulting attention matrix. This model is evaluated for jet tagging using the public hls4ml dataset.

### Strengths
The authors' initiative to tackle the critical issue of inference latency in particle physics transformers is commendable. The proposed model, building on a Linformer backbone with partitioning and convolutional layers, demonstrates a measurable improvement.

### Weaknesses
While the proposed model achieves a marginal performance increase over the standard Linformer, this limited gain does not appear to justify the computational cost, resulting in an unconvincing FLOPs-performance trade-off.

### Questions
- Is the baseline transformer model (architecture and training procedure) properly optimized?
- As for the claim "partitions derived from kT-sorted sequences are more likely to group together energetic particles that are physically nearby, enhancing the effectiveness of spatially aware projection", it is not obvious from Fig. 1 that kT-sorting presents better spatial loaclity. Were other partitioning strategies explored?
- In Fig. 3, for the bin of 115-150, why are the variances of SAL-T and linformer much larger than that of the transformer?

### Soundness
2

### Presentation
3

### Contribution
2
