# An Efficient Global-Local Feature Extraction Architecture for 3D Point Clouds

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Accurate 3D object detection and segmentation from LiDAR point clouds require both global context and fine-grained local features. Sparse convolutions capture local geometry efficiently but have limited receptive fields, while transformers model long-range context at high memory and runtime costs and often miss fine detail. We introduce Dilated Uniform Attention with 3D Sparse Convolution (DUA-SConv), a building block that integrates attention and sparse convolution in a complementary way. Each block applies self-attention over a uniformly dilated neighborhood spanning a large, fixed region to provide coarse global context, followed by sparse convolution to recover fine-grained local features. Stacked DUA-SConv blocks form a compact backbone that achieves high accuracy in 3D detection and segmentation with low runtime and parameter count.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
DUA-SConv is a hybrid 3D backbone that combines attention and sparse convolution to balance global context and local detail in LiDAR-based 3D detection and segmentation. It introduces Dilated Uniform Attention to capture wide-range contextual information efficiently, followed by sparse convolution for precise local feature recovery. By stacking these lightweight DUA-SConv blocks, the model achieves high accuracy with reduced computational cost and parameter count.

### Strengths
**[S1] Clear motivation and supporting evidence. **

The paper provides a clear motivation by emphasizing the importance of modeling long-range receptive fields in 3D point processing. Their discussion on the limitations of conventional convolutions effectively supports the need for their proposed approach. The theoretical justification is coherent, linking the receptive field expansion to improved context aggregation. Empirical results further reinforce their argument, showing noticeable improvements in capturing global scene structures. Overall, their claim regarding the long-range receptive field is convincing and well-supported by both analysis and experiments.

**[S2] Effective performance on nuScene dataset.**

The proposed model demonstrates strong performance on the **nuScenes** dataset, effectively handling large-scale outdoor environments. Its ability to process sparse and wide-range point distributions highlights the robustness of the convolution design. Quantitative results show clear improvements over previous baselines, indicating consistent advantages in outdoor scenarios. The authors also emphasize that their method maintains efficiency while preserving accuracy across long-range interactions. Together, these results confirm that the model is particularly well-suited for outdoor perception tasks where spatial coverage is critical.

### Weaknesses
**[W1] Missing PTv3 and Sonata baselines on nuScenes**

The authors claim that their proposed convolution demonstrates strong capability in modeling long-range interactions. While the presented results are promising, this claim would be more convincing if the authors included comparisons against **PTv3** and **Sonata** on outdoor datasets such as **nuScenes**. Additionally, **Sonata** is missing from the evaluations on **S3DIS** and **ScanNet**, despite being published at **CVPR 2025**. Including it would provide a fairer and more comprehensive comparison. 

**[W2] Missing qualitative results**

Although the proposed method achieves competitive quantitative performance, the paper lacks **qualitative results** or detailed visual analysis. The authors should include qualitative comparisons with prior methods on datasets such as **nuScenes** and **ScanNet**, which would help illustrate the qualitative advantages and better support their quantitative claims.

**[W3] Slightly worse performance on SemSeg datasets**

The proposed method underperforms **PTv3** on indoor datasets, indicating potential limitations in handling short-range geometric structures. This suggests that the model may not generalize well to indoor or densely cluttered environments. To provide a more comprehensive understanding, the authors could also include experiments on **classification** or other **long-range understanding** tasks to validate the effectiveness of their proposed convolution. Moreover,

### Questions
- In **Figure 4**, the groups appear duplicated when N=64. Is there a specific reason why **azimuth** and **elevation** are partitioned with overlaps rather than being uniquely divided? Also, could the authors clarify what the **index values** in the figure represent?
- How were **azimuth** and **elevation** values computed for **ScanNet**? Depending on the normalization procedure of point clouds, these values may vary significantly. Please clarify the computation method and normalization scheme used.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents an efficient transformer design that unifies global-local LiDAR perception through uniform grouping, localized attention, and implicit relative positional encoding—achieving strong accuracy with scalable computation.

### Strengths
1. The paper is well-motivated with a clear explanation of the underlying challenge in LiDAR perception. The authors use visualizations effectively to illustrate the density imbalance across different ranges and to show why uniform grouping and local-global modeling are necessary.
2. The method achieves state-of-the-art results on both NuScenes and Waymo Open Dataset

### Weaknesses
1.Although the paper lists three main contributions, most of them boil down to the introduction of the DUA (Dilated Uniform Attention) module. The overall methodological novelty feels incremental, as the proposed framework mainly adapts existing attention mechanisms to LiDAR range representations rather than introducing a fundamentally new idea.

2.The DUA module itself is not highly innovative, it essentially performs standard attention operations on the range image domain, similar to what has been explored in prior transformer-based LiDAR perception works. The conceptual leap from previous designs is therefore relatively small.

3.The model requires transformations between the range image and sparse point cloud spaces, along with additional grouping operations. These steps likely introduce notable latency. The reported 117 ms inference time is considerably slower than recent efficient LiDAR transformers such as HEDNet and ScatterFormer, while the accuracy improvement is relatively modest, raising concerns about the overall efficiency–performance trade-off.

He, C., Li, R., Zhang, G., & Zhang, L. (2024). ScatterFormer: Efficient Voxel Transformer with Scattered Linear Attention. In Proceedings of ECCV 2024

### Questions
1. Since the model focuses on efficient global-local feature aggregation, it would be interesting to see whether the approach generalizes beyond detection to dense prediction tasks such as semantic or instance segmentation.
2. I’m curious how much latency is introduced when transforming the point cloud into the range-view representation and performing the grouping operations, before converting it back to the sparse point cloud space.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes DUA-SConv, an efficient hybrid backbone for 3D point cloud processing that addresses the challenge of capturing both global context and local detail. The core contribution is a novel "Uniform Dilated Grouping" (UDG) mechanism that applies range-dependent dilation to compensate for the non-uniform density of LiDAR data. This allows an efficient serialized transformer to learn coarse global context from large, consistent spatial regions. This global context is then refined by a 3D sparse convolution to capture fine-grained local features. Experiments demonstrate its effectiveness.

### Strengths
1.	The Uniform Dilated Grouping (UDG) mechanism is a novel and technically sound method for "equalizing" point cloud density before applying attention. This directly addresses a key limitation of prior window-based transformers, whose receptive fields are spatially inconsistent (Fig. 1b). The complementary design, using attention for coarse-global context and sparse convolution for fine-local refinement, is well-motivated and elegant.
2.	Extensive experimentation and ablation studies validate the effectiveness of the proposed method. 
3.	This paper is written and  organized well.

### Weaknesses
1.	The paper lacks intuitive visualizations of the UDG mechanism in action. While Figure 4 shows the indices of the groups, it does not provide a qualitative visualization of what a "dilated group" of points actually looks like in a real point cloud, especially when contrasted with a "naive" group. Adding such a visualization would significantly help readers understand the practical effect of UDG.
2.	The key components of the LR-DUT module, such as point serialization and the K/Q-only positional encoding, appear heavily adopted from Point Transformer V3.  The core innovation is clearly the Uniform Dilated Grouping (UDG). It would be more precise to frame the main contribution as the novel integration of UDG with an existing serialized attention framework, rather than implying the entire LR-DUT block is a novel invention.
3.	It will be more convincing to add more advanced method UniMamba[1] in Tab.1.

[1] UniMamba: Unified Spatial-Channel Representation Learning with Group-Efficient Mamba for LiDAR-based 3D Object Detection. CVPR 2025

### Questions
Refer to the Weakness.

### Soundness
3

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
3

### Summary
The paper presents an architecture that integrates the local feature extraction capability of 3D sparse convolutions with the long-range contextual modeling of dilated attention. This combination enables more effective feature learning from point clouds, which often suffer from uneven density compared to other data modalities. The method builds upon established techniques (3D sparse convolutions, point cloud serialization, transformers, and positional encoding) and introduces Uniform Dilated Grouping (UDG) strategy which forms the foundation of the proposed DUA-SConv module, the core component of the architecture. Experimental results on popular benchmarks demonstrate that the proposed network outperforms other models of similar size and achieves competitive performance compared to larger architectures.

### Strengths
The paper is overall clear, well-structured, and easy to follow. 
 
The motivation for integrating modules that capture local geometric details (via 3D sparse convolutions) with those that model long-range context (via dilated transformers) is well-justified. 
 
The proposed UDG effectively partitions point clouds into groups of approximately uniform spatial size and density which is an interesting and practical solution to the varying point density problem of LiDAR data. The authors combine this component with well-established and effective techniques in a structured and coherent manner to construct the overall architecture. 
 
Results on many popular benchmarks are reported.

### Weaknesses
The proposed method has some similarity to the Neurips 2024 paper "LION: Linear Group RNN for 3D Object Detection in Point Clouds", which applies linear RNN operators on grouped features within a window-based framework. The current paper seems to extend this learning paradigm to window transformers and the 3D Sparse Convolution is very similar to the 3D sub-manifold convolution of LION in capturing local information. This undermines the novelty, especially as performance improvements are also limited. Can the authors provide a more thorough comparison with LION?

Since the paper primarily emphasizes efficiency, it would benefit from additional comparisons of FLOPs, memory consumption, and runtime against Transformer-based models (e.g., PTv3, SphereFormer), Mamba-based models (e.g., Voxel Mamba) and LION to better justify the efficiency–performance trade-offs. The authors should include FLOPs, memory, and runtime comparisons, along with results from scaled-up versions of the proposed model. This will further strengthen the paper. 

Although the proposed architecture has the advantage of a smaller model size (i.e., fewer parameters), it does not appear to achieve state-of-the-art performance. While achieving SOTA results is not strictly necessary, an analysis of scaling strategies and an evaluation of a larger version of the model compared to current SOTA methods would provide valuable insight into the design’s potential capabilities. 

The Waymo Level 1 results are not reported. Can authors include this and keep result reporting consistent with Voxel Mamba, SAFDNet, LION as well as the UniMamba & FSHNet papers cited below. 

Comparisons do not include the most recent methods. The following papers perform better than the current method. Can you provide comparisons to these baselines: 

[1] S Liu et al. "FSHNet: Fully Sparse Hybrid Network for 3D Object Detection." CVPR 2025. 

[2] X Jin et al. "UniMamba: Unified Spatial-Channel Representation Learning with Group-Efficient Mamba for LiDAR-based 3D Object Detection." CVPR 2025. 

[3] Z Liu et al. "LION: Linear Group RNN for 3D Object Detection in Point Clouds." NeurIPS 2024. 

Why does adding positional encoding to K, Q, and V lead to lower performance compared to adding it only to K and Q (as shown in Table 4)? 

There are a few minor writing issues, though they do not affect the overall understanding of the paper. In particular, the terms “dilation” and “dilution” should be used consistently (e.g., the caption of Figure 3 should use “dilation factor” to match the terminology in the main text). Additionally, the column names in Table 5 are missing and should be included for clarity.

### Questions
Please see the Weaknesses section. I am open to changing my rating if the authors can address my comments.

### Soundness
3

### Presentation
3

### Contribution
2
