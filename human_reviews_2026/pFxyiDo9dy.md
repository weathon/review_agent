# A LoD of Gaussians: Unified Training and Rendering for Ultra-Large Scale Reconstruction with External Memory

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 0, 6, 6

## Abstract
Gaussian Splatting has emerged as a high-performance technique for novel view synthesis, enabling real-time rendering and high-quality reconstruction of small scenes. However, scaling to larger environments has so far relied on partitioning the scene into chunks - a strategy that introduces artifacts at chunk boundaries, complicates training across varying scales, and is poorly suited to unstructured scenarios such as city-scale flyovers combined with street-level views. Moreover, rendering remains fundamentally limited by GPU memory, as all visible chunks must reside in VRAM simultaneously.
We introduce A LoD of Gaussians, a framework for training and rendering ultra-large-scale Gaussian scenes on a single consumer-grade GPU - without partitioning. Our method stores the full scene out-of-core (e.g., in CPU memory) and trains a Level-of-Detail (LoD) representation directly, dynamically streaming only the relevant Gaussians. A hybrid data structure combining Gaussian hierarchies with Sequential Point Trees enables efficient, view-dependent LoD selection, while a lightweight caching and view scheduling system exploits temporal coherence to minimize the loading overhead. Together, these innovations enable seamless multi-scale reconstruction and interactive visualization of complex scenes - from broad aerial views to fine-grained ground-level details.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
His paper presents A LoD of Gaussians, a method that overcomes GPU memory limitations in city-scale 3D Gaussian Splatting by introducing a hierarchical LoD system combined with external memory management.

### Strengths
1. The core technique is novel and interesting in its details.
2. The paper includes careful analysis of memory usage.

### Weaknesses
1. I believe the experimental section is insufficient. First, the paper lacks detailed descriptions of the training settings. Second, the comparative experiments are incomplete—methods such as Grendel-GS and OccluGaussian, which are specifically designed for large-scale reconstruction, are not included for comparison. Moreover, the evaluation is performed on a very limited set of datasets, which restricts the generalizability of the conclusions.
2. The paper lacks references to important related work on Level of Detail (LOD) and large-scale rendering. The authors should consider citing the following works:
    - *OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering*
    - *Virtualized 3D Gaussians: Flexible Cluster-based Level-of-Detail System for Real-Time Rendering of Composed Scenes*
3. This paper presents a relatively detailed design of a 3DGS-based reconstruction framework for large-scale scenes, covering aspects from methodology to memory management. However, I find the novelty of the proposed method limited. The LoD framework is quite similar to HierarchicalGS, and the strategy of dynamically loading data from CPU to GPU has also been explored in prior work.

### Questions
please see the weakness above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper addresses a major limitation of Gaussian Splatting: its difficulty in scaling to large, multi-scale environments (like a city with both aerial and street views). Current methods divide the scene into "chunks," which causes artifacts at boundaries, complicates training, and hits GPU memory limits. The authors propose "A LoD of Gaussians," a framework that enables the training and rendering of massive Gaussian scenes on a single consumer-grade GPU without any partitioning.

### Strengths
Efficient large-scale scene reconstruction and rendering are of great significance. The clarity is also fine.

### Weaknesses
1. The experiments are far from sufficient. The authors provide no ablations of method design and provide little illustration of the details for scaling up compared models to the large-scale scenes. So there is no guarantee about the fairness of the comparison, showing a lack of respect for the conference. Besides, the qualitative comparison is also limited to two datasets. I would strongly recommend rejecting the paper. 
2. The innovation is limited. The proposed techniques are mostly a combination of existing methods or engineering efforts, making it insufficient to be accepted as an ICLR paper.

### Questions
See weakness.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes A LoD of Gaussians, a unified framework for out-of-core training and rendering of ultra-large-scale 3D Gaussian Splatting scenes without spatial chunking. The core idea is to store the full scene in CPU memory, build a dynamic Level-of-Detail hierarchy, and use Hierarchical Sequential Point Trees (HSPT) to efficiently generate LoD cuts for view-dependent rendering and training. A GPU-side caching and view scheduling mechanism further reduces data transfer overhead and stabilizes training.

### Strengths
1. Eliminates chunk partition issues (ghosting, bleeding, merging artifacts).

2. Scales to tens of millions of Gaussians on a single GPU, a highly impactful result.

3. HSPT is a clever and elegant hybrid between hierarchy BFS and parallel SPT cutting.

4. Strong qualitative and quantitative improvements on city-scale datasets.

### Weaknesses
1. CPU memory remains the true bottleneck; the method is not actually hardware-light.

2. Initialization assumes good camera poses and geometry, performance may collapse otherwise.

3. The method is less advantageous when scale variation is small (single-height aerial sets).

4. Some ablation discussions are descriptive rather than analytical, more measurements would clarify causality.

5. The training speed per step is slower; the improvement is in iteration count, not iteration efficiency.

### Questions
1. Could the hierarchy be partially stored on SSD with async prefetch to reduce RAM pressure?

2. How sensitive is HSPT cut correctness to the surface-area-based md metric? Any cases of cut instability?

3. Does caching introduce systematic bias in reconstructed local texture, e.g., does cache reuse correlate with oversharpening?

4. Could joint pose refinement in early stages reduce reliance on precise COLMAP input?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes a 3DGS pipeline that avoids scene partitioning by storing the full scene out-of-core (CPU RAM) and streaming view-relevant Gaussians to the GPU.

### Strengths
1. Clear systems contribution for scaling 3DGS without chunking; the combination of LoD + HSPT + out-of-core streaming is well motivated and addresses real bottlenecks.
2. Practical details on hierarchy maintenance during training and scheduling/caching, with results on large multi-scale scenes.

### Weaknesses
1. ML novelty is limited: contributions are primarily data-structure/streaming/systems optimizations around standard 3DGS training; the learning component itself is not substantially new for ICLR.
2. While large-scale results are discussed, Lack comparisons (Training time, FPS, GPU consumption) against recent large-scene 3DGS(Octree-GS CityGaussin Momentum-GS CityGS-X).
3. Can you report sensitivity curves for cache size and LoD cut parameters vs. image quality and FPS?

### Questions
This paper makes significant efforts to improve the underlying rendering and training logic of 3D-GS. It would be interesting if the authors could also discuss how their approach might be applied to models like Grendel-GS, which are more sensitive to communication latency.

### Soundness
3

### Presentation
3

### Contribution
3
