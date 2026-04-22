# Optimized Minimal 4D Gaussian Splatting

- Avg Score: 5.33
- Decision: Reject
- Scores: 4, 6, 6

## Abstract
4D Gaussian Splatting has emerged as a new paradigm for dynamic scene representation, enabling real-time rendering of scenes with complex motions. However, it faces a major challenge of storage overhead, as millions of Gaussians are required for high-fidelity reconstruction. While several studies have attempted to alleviate this memory burden, they still face limitations in compression ratio or visual quality.
In this work, we present $\textit{OMG4}$ (Optimized Minimal 4D Gaussian Splatting), a framework that constructs a compact set of salient Gaussians capable of faithfully representing 4D Gaussian models.
Our method progressively prunes Gaussians in three stages: (1) $\textit{Gaussian Sampling}$ to identify primitives critical to reconstruction fidelity, (2) $\textit{Gaussian Pruning}$ to remove redundancies, and (3) $\textit{Gaussian Merging}$ to fuse primitives with similar characteristics.
In addition, we integrate implicit appearance compression and generalize Sub-Vector Quantization (SVQ) to 4D representations, further reducing storage while preserving quality.
Extensive experiments on standard benchmark datasets demonstrate that $\textit{OMG4}$ significantly outperforms recent state-of-the-art methods, reducing model sizes by over 60\% while maintaining reconstruction quality.
These results position $\textit{OMG4}$ as a significant step forward in compact 4D scene representation, opening new possibilities for a wide range of applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a pipeline to compress Gaussian Splatting based 4D dynamic spatial representations. The method concludes several stages, as selecting salient Gaussians based on Static-Dynamic scores, pruning low-importance primitives, merging similar Gaussians within grids, and compressing attributes with MLPs and arithmetic compressors. The resulting 4DGS can provide a highly compact storage size while maintaining the visual quality. This pipeline is applicable to multiple off-the-shelf baselines and achieves compactness.

### Strengths
1. One impressive point of this work is that this paper proposes a universally applicable pipeline as a post-processing for multiple 4DGS representations. This highlights the method's generality, which I think is meaningful for this task.
2. The SD-Score, presenting the spatial and temporal sensitivity, guides the pipeline to distinguish the important Gaussians and provides effective compression.
3. The performances are impressive compared to the baselines, achieving a low storage size of several MB to represent a 4D dynamic space.

### Weaknesses
1. Although the proposed method provides superior compactness, the proposed components in the method are not very impressive in the aspect of novelty.  The Gaussian importance score, MLP based color coding, and post-processing compressors are upgraded from 3DGS compression techniques or previously established in other 4D Gaussian compression works. These make the novelty of this work weaker, and the authors fail to distinguish their designs from the previous techniques. My own opinion is that the author can emphasize more on the generality of the proposed method: The authors can establish the whole pipeline as a more robust and general post-processing progress to all (or most of) the previous 4DGS baselines, and investigate how to specifically implement the proposed method to each distinct baselines and fit the special property of each method. 
2. Another concern is a highly engineered pipeline with many hyperparameters. The whole pipeline includes multiple thresholds, quantiles, grid sizes and merge / iteration budgets. These elements raise concerns on the robustness of the proposed method. Naturally, some scenes require high quantity of Gaussians for representation, while some other may not. Some scenes contain complex lattice or textures. The discussion or ablation on these issues should be provided for a solid presentation.

### Questions
Regarding the weaknesses listed above:
1. The authors are suggested to provide more clarifications on the generality of the proposed method, such as elaborating on how the proposed pipeline fits different baselines and analysis on the effectiveness on them. The authors are recommended to provide further attempts on other baselines and clarify the effectiveness, and in which cases the method fits better.
2. The authors are recommended to provide more justifications and ablations on the hyperparameters, and how these hyperparameters are affecting the final compression performance. Are these hyperparameters tuned for each scene or each dataset? 
3. Some previous 4D reconstruction methods use zero-shot dynamic-static distinguishment [R1, R2]. Can they benefit the proposed pipeline? These methods should also be considered or discussed.

[R1] Dai, P., Zhang, P., Dong, Z., Xu, K., Peng, Y., Ding, D., Shen, Y., Yang, Y., Liu, X., Lau, R.W. and Xu, W., 2025. 4d gaussian videos with motion layering. ACM Transactions on Graphics (TOG), 44(4), pp.1-14.

[R2] Liu, Z., Hu, Y., Zhang, X., Song, R., Shao, J., Lin, Z. and Zhang, J., 2024. Dynamics-Aware Gaussian Splatting Streaming Towards Fast On-the-Fly 4D Reconstruction. arXiv preprint arXiv:2411.14847.

I am looking forward to the authors' reply on the above issues, based on which I am willing to further adjust my review.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
OMG4 (Optimizing Minimal 4D Gaussian Distributions) constructs a compact set of salient Gaussians that faithfully represent the 4D Gaussian model.
The method progressively prunes Gaussians in three stages:
(1) **Gaussian sampling** to identify primitives critical to reconstruction fidelity,
(2) **Gaussian pruning** to remove redundancies, and
(3) **Gaussian merging** to fuse primitives with similar characteristics.

### Strengths
1. The three strategies proposed in the paper are interesting, and the writing is well-organized.
2. The proposed method achieves a significant reduction in storage while maintaining comparable performance.
3. The paper provides extensive visualization videos.

### Weaknesses
1. The experiments are too limited — most evaluations are conducted only on the N3DV dataset. Testing on large-scale dynamic scenes such as NVIDIA[1], Dynamic3DGS[2], or VRU [3] would make the results much more convincing.
2. The proposed compression and pruning strategies are all based on 4DGS; however, 4DGS has inherent limitations, such as its inability to handle fast motion or long-sequence dynamic videos. Therefore, this method is somewhat restricted in its applicability.

If the authors can demonstrate strong experimental results (both quantitative and qualitative) on large-scale datasets, this paper would deserve a score of 8 and be worth accepting.

[1] Neural Trajectory Fields for Dynamic Novel View Synthesis

[2] Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis

[3] Swift4D: Adaptive divide-and-conquer Gaussian Splatting for compact and efficient reconstruction of dynamic scene

### Questions
1. How long does the model take to train?
2. Is the model initialized using Gaussian points from all frames to represent the entire scene, followed by the sampling, merging, and pruning strategies? If so, wouldn’t this result in high GPU consumption and require long training times — possibly several hours?

### Soundness
2

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
2

### Summary
This paper introduces OMG4 (Optimized Minimal 4D Gaussian Splatting), a novel framework for compact and high-fidelity dynamic scene representation. While prior 4D Gaussian Splatting (4DGS) approaches have achieved impressive real-time rendering results, they typically require millions of Gaussians, leading to significant memory and storage overheads.

### Strengths
- The proposed Static–Dynamic Score (SD-Score) is an elegant and effective contribution that unifies spatial and temporal importance estimation for Gaussian primitives.
- The multi-stage pipeline (Sampling → Pruning → Merging) is conceptually simple yet powerful, providing interpretability and modular extensibility.
- Extending Sub-Vector Quantization (SVQ) to the 4D domain and introducing a staged quantization strategy for stability is a non-trivial and meaningful extension of prior work.
- The paper is clearly written, with strong motivation and logical flow across sections.

### Weaknesses
(1) Experimental depth and comparison limitations:
- While the reported compression ratio and reconstruction quality are impressive, the paper could benefit from a more diverse set of baselines, including recent hybrid deformation-based approaches (e.g., ADC-GS or D-NeRF variants).
- It remains unclear how OMG4 scales with scene complexity or duration—e.g., whether compression quality degrades for highly non-rigid motions or long temporal spans.

(2) Computational cost and runtime:
- The paper focuses primarily on memory efficiency, but does not clearly state the training or optimization time overhead introduced by multi-stage pruning and merging.
- It would strengthen the work to clarify whether OMG4 maintains real-time rendering throughput post-compression, and how merging affects rendering speed and differentiability.

(3) Generality of SVQ extension:
- The adaptation of SVQ to 4D is interesting, but the explanation of why staged quantization improves stability is somewhat qualitative.
- A quantitative analysis (e.g., convergence behavior, quantization error curves) would substantiate this claim.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
3
