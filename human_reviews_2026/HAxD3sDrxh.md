# SurfelSoup: Probabilistic G-SurfelTree for Learned Point Cloud Geometry Compression

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
This paper presents SurfelSoup, the first end-to-end learned surface-based framework for dense point cloud geometry compression, with surface-structured primitives for representation. It proposes a probabilistic and differentiable surface representation, G-Surfel, which models local point occupancies using a bounded generalized Gaussian distribution. We further introduce G-SurfelTree, an octree-like hierarchy, where a decision module adaptively terminates the tree subdivision  for rate-distortion optimal G-Surfel granularity selection.
This formulation avoids redundant point-wise compression in smooth regions and produces compact yet smooth surface reconstructions. Experimental results under the MPEG common test condition show consistent gain on dense geometry compression over voxel-based baselines and MPEG standard G-PCC-GesTM-TriSoup, while providing visually superior reconstructions with smooth and coherent surface structures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a contribution by introducing SurfelSoup, an end-to-end surface-based framework for point cloud compression, which breaks away from the traditional voxel-based paradigm.  It utilizes a probabilistic surface representation called G-Surfel, modeling voxel occupancies within an octree node using a bounded 3D generalized Gaussian function, thereby facilitating the use of a differentiable distortion term in the loss function during training.  Another significant aspect is the proposed G-SurfelTree hierarchy, which enables adaptive G-Surfel granularity assignment across octree levels via a decision module and a corresponding formulation of the total expected rate and distortion, rather than relying on a predefined tree structure.

### Strengths
* The paper shows high technical rationality by introducing an end-to-end surface-based point cloud compression framework.
* The paper provides a clear and detailed explanation of the proposed method. 
* The paper conducts a thorough experimental comparison with several representative methods.

### Weaknesses
* Certain statements in the paper are ambiguous. For instance, the first surface-based framework lacks well-defined boundaries. The definition of a surface-based approach is unclear, especially since previous methods using surface triangle information and occupancy (a surface-oriented representation) cast doubt on their surface-based classification. Also, although the proposed model shares similarities with 3DGS, the paper doesn't provide an in-depth and positive discussion on their differences.
* Limited Visual and Quantitative Analysis. In the visual comparison, only the comparison between the proposed method and Unicorn is shown. The authors should present the visual effects of all methods involved in the quantitative analysis and render color-free geometries to show geometric structure compression precision. 
* The quantitative comparison reveals that the performance improvement of the proposed method is relatively limited.

### Questions
* The first surface-based framework raises some ambiguous boundaries. What exactly constitutes a surface-based approach? Previous methods have utilized surface triangle information. Do they fall under the category of surface-based? Occupancy, in essence, is also a surface-oriented representation method. Does it qualify as surface-based?
* Section 2.3 only discusses one single paper. In fact, Surfel is a fundamental geometric concept, and many research studies on 3D representation have proposed various modeling methods for surfels. The introduction in the paper is rather one-sided and incomplete.
* In the visual comparison, the paper only shows the comparison between the proposed method and Unicorn. The authors should present the visual effects of all methods involved in the quantitative analysis. * Moreover, they should render the color-free geometries to demonstrate the compression precision of geometric structures.
* There are discrete artifacts and splitting phenomena in Unicorn. The authors should conduct a more in-depth discussion on why these situations occur.

### Soundness
3

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
This paper presents  a novel end-to-end learned framework for dense point cloud geometry compression. Unlike previous voxel-based approaches such as PCGCv2, SparsePCGC, and Unicorn, SurfelSoup models 3D geometry as a composition of probabilistic surfaces (G-Surfels) rather than discrete voxels. Each G-Surfel is formulated as a bounded generalized Gaussian distribution, parameterized by a mean, covariance, quaternion rotation, and shape coefficient, which can flexibly represent local planar or curved surface patches. The method further organizes these G-Surfels hierarchically in a G-SurfelTree, analogous to an adaptive octree, where a decision module learns whether to subdivide or terminate a node based on rate-distortion optimization. During inference, the model reconstructs point clouds by binarizing occupancy probabilities, selecting voxels with the highest likelihood. Experiments on several benchmarks demonstrate that SurfelSoup achieves significantly higher compression efficiency than Unicorn and other state-of-the-art baselines while producing smoother and gap-free reconstructed surfaces.

### Strengths
The introduction of the probabilistic G-Surfel is conceptually elegant and bridges voxel-based compression with continuous surface modeling, similar in spirit to 3D Gaussian Splatting. The probabilistic surface modeling avoids voxelization artifacts, leading to visually continuous and realistic point clouds.

The entire framework is end-to-end.

### Weaknesses
This method reconstructs the point cloud based on resampling, so the reconstructed point cloud is different from the original one. Moreover, this method can only compress the geometry and cannot be used to compress the RGB colors, which reduces its value and makes the motivation less clear.

The paper still lacks a comparison with 3DGS compression methods. Essentially, both approaches aim to compress point clouds, but 3DGS compression methods also compress the color information of the scene.

### Questions
The proposed method can only compress geometry, so how were the colors in the images shown in the experiments generated?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel surface-based point cloud geometry compression method called SurfelSoup, which for the first time combines a probabilistic surface representation (G-Surfel) with a learnable tree structure (G-SurfelTree), achieving end-to-end training and optimization. The method demonstrates superior performance on multiple standard datasets, particularly excelling in smooth surface reconstruction. The paper has a clear motivation, innovative methodology, and comprehensive experimental design, contributing significantly both theoretically and practically.

### Strengths
1. It introduces the first end-to-end learned surface-based point cloud compression framework, incorporating a probabilistic G-Surfel representation based on a generalized Gaussian distribution, making surface modeling more expressive and differentiable, demonstrating strong innovation. 

2. The method is comprehensively designed, covering modules such as encoding, surface reconstruction, decision-making, and entropy coding, while addressing information leakage issues during both training and inference. 

3. Extensive comparisons with various existing methods on standard test sets are provided, along with detailed ablation studies that validate the effectiveness of the proposed approach.

### Weaknesses
1. Some formulas and process descriptions in Section 3 (Methodology), particularly the probabilistic modeling of P-SOPA and the decision module, are somewhat obscure and could benefit from more intuitive diagrams or pseudocode for clarification. 

2. The method primarily targets dense point clouds and shows limited performance on structurally complex scenes, indicating relatively high application constraints. It is recommended to analyze the model's adaptability to irregular geometric structures and propose potential improvements. 

3. Although the experimental comparisons are extensive, they could be further strengthened by including comparisons with recent implicit or explicit surface reconstruction methods.  

4. The increased model size compared to voxel-based baselines somewhat hinders practical deployment potential, warranting discussion on potential model compression or pruning strategies.

### Questions
1. Some formulas and process descriptions in Section 3 (Methodology), particularly the probabilistic modeling of P-SOPA and the decision module, are somewhat obscure and could benefit from more intuitive diagrams or pseudocode for clarification. 

2. The method primarily targets dense point clouds and shows limited performance on structurally complex scenes, indicating relatively high application constraints. It is recommended to analyze the model's adaptability to irregular geometric structures and propose potential improvements. 

3. Although the experimental comparisons are extensive, they could be further strengthened by including comparisons with recent implicit or explicit surface reconstruction methods.  

4. The increased model size compared to voxel-based baselines somewhat hinders practical deployment potential, warranting discussion on potential model compression or pruning strategies.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes SurfelSoup, the first end-to-end learned surface-based framework for point cloud geometry compression. It introduces a probabilistic and differentiable surface representation, termed G-Surfel, which models local point occupancies using a bounded generalized Gaussian distribution. A G-SurfelTree is then constructed as an adaptive octree-like hierarchy, where a decision module determines subdivision depth to achieve rate–distortion–optimal granularity.

### Strengths
The idea of this paper is motivated by the limitations of voxel-based learned point cloud compression methods. Then, a proposed G-surfel is used to reconstruct the point cloud from a coarse octree with the associated features.  The performance indicates the superiority of the proposed technique to some prior works.

### Weaknesses
1. This paper introduces a decision module to adaptively determine whether to terminate a node as a G-Surfel or further subdivide it within the G-SurfelTree. This component is claimed to be critical for achieving a rate–distortion–optimal balance and for adaptively allocating surfel granularity. But the notation in this section is not clear and needs improvement for better comprehension.
2. The bit rate of the octree compression should be analyzed. 
3. The experimental results show that the proposed method outperforms Unicorn. However, it is unclear which components contribute most to this improvement.
4. High complexity in terms of model size, encoding, and decoding time.

### Questions
1. The model configuration in section 4.2 is difficult to catch, particularly with the notation l and L.
2. The training dataset is different from most of the prior works. Besides, 8i VFB is usually used as a test dataset for the static point cloud.

### Soundness
3

### Presentation
3

### Contribution
2
