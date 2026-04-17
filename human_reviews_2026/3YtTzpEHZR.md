# LoBE-GS: Load-Balanced and Efficient 3D Gaussian Splatting for Large-Scale Scene Reconstruction

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
3D Gaussian Splatting (3DGS) has established itself as an efficient representation for real-time, high-fidelity 3D scene reconstruction. However, scaling 3DGS to large and unbounded scenes such as city blocks remains difficult. Existing divide-and-conquer methods alleviate memory pressure by partitioning the scene into blocks, but introduce new bottlenecks: (i) partitions suffer from severe load imbalance since uniform or heuristic splits do not reflect actual computational demands, and (ii) coarse-to-fine pipelines fail to exploit the coarse stage efficiently, often reloading the entire model and incurring high overhead. In this work, we introduce LoBE-GS, a novel Load-Balanced and Efficient 3D Gaussian Splatting framework, that re-engineers the large-scale 3DGS pipeline. LoBE-GS introduces a depth-aware partitioning method that reduces preprocessing from hours to minutes, an optimization-based strategy that balances visible Gaussians—a strong proxy for computational load—across blocks, and two lightweight techniques, visibility cropping and selective densification, to further reduce training cost.
Evaluations on large-scale urban and outdoor datasets show that LoBE-GS consistently achieves up to $2\times$ faster end-to-end training time than state-of-the-art baselines, while maintaining reconstruction quality and enabling scalability to scenes infeasible with vanilla 3DGS.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
LoBE-GS tackles challenges like load imbalance and inefficiencies in coarse-to-fine pipelines by introducing a depth-aware partitioning method, optimization to balance computational load, and lightweight techniques such as visibility cropping and selective densification. Evaluations show LoBE-GS achieves significantly faster training times compared to existing methods, without compromising reconstruction quality, and scales effectively to large, complex scenes.

### Strengths
1. The improvement of workload balance during parallel training is a considerable contribution towards a more efficient large-scale scene reconstruction.
2. Though the idea of visibility cropping has been implemented by V2 of CityGaussian, the exploration of the correlation between runtime and different factors is meaningful and inspiring.

### Weaknesses
1. The key contribution is the proposed partitioning strategy. To sufficiently validate the superiority, the authors should provide additional experiments on a specific baseline while only alternating the partitioning strategy, such as comparing with that of VastGaussian, CityGaussian, and Hierarchical 3DGS.  The authors should also illustrate the superiority of these baseline strategies.
2. The ablation only compares time cost, but the influence on the rendering quality is ignored, making the validation of module design incomplete.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents LoBE-GS, a load-balanced and efficient framework for large-scale 3D Gaussian Splatting (3DGS). While 3DGS has become a popular representation for real-time, high-fidelity 3D scene reconstruction, its scalability to large, unbounded scenes (e.g., city blocks) remains limited by memory and computational constraints. Existing divide-and-conquer strategies mitigate these issues by spatially partitioning scenes but introduce new bottlenecks due to load imbalance and inefficient coarse-to-fine training pipelines.

### Strengths
- The paper introduces a principled and quantitative approach to load balancing in large-scale 3DGS, leveraging the number of visible Gaussians as a proxy for computational demand—a novel insight that is both intuitive and empirically validated. The use of Bayesian Optimization for spatial partition refinement is also new in this context and provides a solid methodological contribution.
- LoBE-GS directly addresses a key bottleneck in scaling 3DGS to real-world environments. The proposed framework offers a tangible step toward city-scale Gaussian reconstruction, enabling more balanced and efficient large-scale modeling. The reduction of preprocessing time from hours to minutes is particularly impactful for practical deployments.

### Weaknesses
- Although LoBE-GS shows strong performance on multiple datasets, it would strengthen the paper to include comparisons against non-block-based large-scale approaches or hybrid systems (e.g., hierarchical or LoD 3DGS variants like CityGS-X or Momentum-GS). Moreover, an ablation isolating the contributions of each component (BO partitioning, camera selection, visibility cropping, selective densification) would provide clearer insight into their individual impacts.
- LoBE-GS assumes availability of a coarse 3DGS model for partitioning and camera selection. Discussion on how sensitive the method is to the quality of this coarse prior would be valuable, as suboptimal coarse models could degrade the partitioning accuracy.
- The method involves multiple implementation-specific optimizations (Warp kernels, GP-based BO). Publicly releasing code or detailing hyperparameter settings (e.g., BO iteration count, GP kernel choice) would improve reproducibility and adoption.

### Questions
See Weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes LoBE-GS, a load-balanced and efficient pipeline for large-scale 3D Gaussian Splatting (3DGS).

### Strengths
1. The paper identifies the “straggler block” issue in parallel 3DGS training and backs the visible-Gaussian proxy with correlation analyses and min-max balancing that directly reduces the worst-case fine-stage time.
2. Experiments span five scenes (four real, one synthetic), with both quality and runtime breakdowns; the “up to 2×” improvement is demonstrated alongside fair notes on color-alignment metrics and runtime components.

### Weaknesses
1. The core ideas (min-max balancing via a load proxy, linear-time camera assignment, pruning/controlled densification) are solid but incremental rather than conceptually radical for learning/reconstruction.
2. Can you record the specific number of Gaussians for each scene and analyze in detail the reasons behind the observed performance improvements?
3. I noticed that CityGS-X is discussed in the related work. This method eliminates the merge–partition overhead and performs parallel training across multiple GPUs, thereby naturally achieving load balance. Given that the main motivation of this paper is to improve load balancing through a better partitioning strategy, it would strengthen the paper if the authors could clarify how their approach differs from or improves upon CityGS-X.

In summary, my primary concern is about the novelty and technical advancement of the proposed approach.

### Questions
I have no more questions.

### Soundness
3

### Presentation
3

### Contribution
3
