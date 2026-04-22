# Mango-GS: Enhancing Spatio-Temporal Consistency in Dynamic Scenes Reconstruction using Multi-Frame Node-Guided 4D Gaussian Splatting

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Reconstructing dynamic 3D scenes with photorealistic detail and temporal coherence remains a significant challenge. Existing Gaussian splatting approaches modeling scenes rely on per-frame optimization, causing them to overfit to instantaneous states rather than learning true motion dynamics. To address this, we present Mango-GS, a multi-frame, node-guided framework for high-fidelity 4D reconstruction. Our approach leverages a temporal Transformer to learn complex motion dependencies across a window of frames, ensuring the generation of plausible trajectories. For efficiency, this temporal modeling is confined to a sparse set of control nodes. These nodes are uniquely designed with  decoupled position and latent codes, which provide a stable semantic anchor for motion influence and prevents correspondence errors for large movements. Our framework is trained end-to-end, enhanced by a input masking strategy and two multi-frame loss to ensure robustness. Extensive experiments demonstrate that Mango-GS achieves state-of-the-art quality and fast rendering speed, enabling high-fidelity reconstruction and real-time rendering of dynamic scenes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles real-time dynamic scene reconstruction from monocular videos based on SC-GS, aiming to model long-range, non-linear motion with temporal coherence. The authors propose Mango-GS, which couples 3D Gaussian Splatting with a node-guided multi-frame network:  (1) an MLP backbone interleaves temporal self-attention and gated fusion to propagate dependencies across a window of $T$ frames and predict per-node deformations; (2) training employs temporal input masking plus a composite loss with frame loss and motion loss. Experiments on Neural 3D Video, and HyperNeRF-vrig report consistent rendering quality improvement and stability while remaining real-time rendering speed.

### Strengths
- This paper is well-written and easy to understand.
- I appreciate the paper’s attempt to process multiple frames jointly. Handling T frames simultaneously and introducing self-attention across different timestamps make sense, and this idea could inspire future work on monocular (multi-view) dynamic reconstruction.”
- The ablation is appreciated to show the important contribution of each part.
- Across different real-world datasets, the results are consistently strong, with visualizations aligning well with the quantitative metrics.

### Weaknesses
- Missing references for some important dynamic reconstruction works:
  - [NeurIPS 2024] Grid4D: 4D Decomposed Hash Encoding for High-Fidelity Dynamic Gaussian Splatting, by Jiawei Xu et al.
  - [NeurIPS 2024] Splatter a Video: Video Gaussian Representation for Versatile Processing, by Yang-Tian Sun et al.
- Using a lightweight MLP to compute the implicit distance between control points and Gaussians is insightful. However, I don’t find the benefit of this design immediately clear from Figure 2. Have the authors explored alternative formulations, such as a codebook approach similar to VQ-VAE?
- A minor suggestion: Figure 3 does not clearly convey self-attention across time. Also, if my understanding is correct, position and timestamp should be provided jointly as input to the MLP?
- Minor typo errors:
  - L23: `which provides` -> `which provide`
  - L24: `prevents correspondence error` -> `prevents correspondence errors`
  - L695-696: `Each groups displays` -> `Each group displays`

### Questions
1. What is the approximate training time? How much overhead is there relative to SC-GS?
2. Could you clarify how the temporal input masking is constructed (random or structured across time)? What masking ratio do you use, and is it fixed or scheduled during training?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper first points out that existing Gaussian Splatting methods rely on per-frame optimization, causing them to memorize frame-specific states rather than learning true motion dynamics. To address this, it proposes Mango-GS, a multi-frame, node-guided 4D Gaussian Splatting framework that introduces a decoupled control node representation and a temporal attention network. The method aims to enable efficient, temporally consistent, and high-fidelity dynamic scene reconstruction.

### Strengths
* The paper clearly identifies a core weakness in existing dynamic 3D Gaussian Splatting methods, namely their reliance on per-frame optimization, which causes temporal inconsistency and overfitting to instantaneous states. The motivation for introducing a multi-frame modeling framework is well justified and directly addresses this limitation.

* The proposed multi-frame temporal deformation network interleaves MLP layers with temporal self-attention blocks and a gated fusion mechanism. This hybrid design effectively captures long-range motion dependencies while remaining computationally efficient. It represents a well-balanced approach between expressive modeling and real-time rendering.

* The paper is well organized, with clear reasoning and smooth transitions between sections, making the overall presentation easy to follow.

### Weaknesses
* My main concern regarding the design of Mango-GS lies in the insufficient justification for using a Transformer to address temporal inconsistency. First, the advantage of Transformer is its ability to capture long-range dependencies, but Table 2 shows that the optimal temporal window is only six frames. Is using a Transformer to model such a short sequence truly necessary, and does the computational cost justify the potential performance gain? Furthermore, several prior works have also explored multi-frame motion modeling to improve temporal coherence. The paper should include an ablation study comparing the Transformer with other mechanisms that leverage multi-frame motion information, in order to clearly demonstrate the advantages of using a Transformer in this context.

* In Figure 4, the highlighted regions should be enlarged for better visual inspection.

* In Figure 5, the visual differences among compared methods are not very clear. For example, in the *Peel-Banana* scene, the results of 4DGS and Mango-GS appear quite similar, and in the *Cut-lemon* scene, differences among all methods are subtle. The paper should select frames with more noticeable differences or to use highlighted and zoomed-in regions to better demonstrate the superiority of Mango-GS. Based on the current visualizations, I do not observe a significant advantage of the proposed method.

* In the experimental section, all compared methods are from 2023-2024. The paper should include comparisons with the latest approaches and should report detailed per-scene quantitative results.

* The paper should also evaluate the method on more challenging datasets, such as the iPhone dataset, to further validate its robustness and generalization ability.
* The discussion of related work should be expanded to include more recent studies, as the current review is not sufficiently comprehensive.
* If possible, I hope to see video results to more intuitively understand the advantage of Mango-GS in motion consistency.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a 4D scene reconstruction approach that ensures both photorealistic detail and temporal coherence. It leverages a temporal transformer to model complex motion dependencies across multiple frames, guided by a set of sparse control nodes with decoupled position and latent codes. This design enables stable motion representation and prevents correspondence errors during large movements. Trained end-to-end with multi-frame objectives, Mango-GS achieves state-of-the-art reconstruction quality and real-time rendering performance.

### Strengths
1. Decoupling the 4DGS makes sense.
2. This paper presents a good FPS performance.

### Weaknesses
1. For the quantitative comparison in Table 1. The improvement in PSNR is limited.
2. The visualized comparison is weak. It is better to provide video comparison in supplementarials.

### Questions
1. Please include the cites in Table 1. 
2. Please add the discussions with other approaches that focus on the temporal modeling of Gaussians. e.g. A, B.

[A] MotionGS: Exploring Explicit Motion Guidance for Deformable 3D Gaussian Splatting
[B] TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Mango-GS, a multi-frame node-guided framework for dynamic 3D scene reconstruction using Gaussian splatting. The key innovation lies in decoupling control nodes into canonical positions and learnable feature codes, combined with a temporal transformer that processes multiple frames simultaneously to learn motion dynamics. The sparse control nodes guide dense 3D Gaussians through learned k-NN relationships, enabling efficient temporal modeling while avoiding per-frame overfitting. Experiments on HyperNeRF and Neural 3D Video datasets demonstrate state-of-the-art reconstruction quality with real-time rendering speeds of 149.5 FPS and reasonable storage requirements of 60 MB.

### Strengths
* The decoupled node representation is a well-motivated design that elegantly addresses the neighborhood drift problem in large motion scenarios.
* The multi-frame temporal attention mechanism represents a significant departure from per-frame optimization strategies prevalent in prior work. This design enables the model to learn motion patterns rather than memorize instantaneous states, leading to improved temporal coherence as evidenced by both quantitative metrics and qualitative visualizations.
* The ablation studies systematically validate each component's contribution, and the method achieves excellent rendering speed while maintaining competitive storage efficiency.

### Weaknesses
* The theoretical justification for why the decoupled representation prevents neighborhood drift is primarily empirical. While Figure 2 provides visual evidence, a more rigorous analysis of the learned feature space and how it maintains semantic consistency under large deformations would strengthen the claims.
*  The motion-aware loss components ($L_{diff}, L_{dir}$) are mentioned but never formally defined in the main paper.
* The evaluation focuses heavily on PSNR/SSIM metrics, but temporal consistency evaluation is limited. While motion-aware loss is used during training, there are no temporal metrics in the evaluation (e.g., temporal LPIPS, or frame-to-frame stability measures). This is a significant omission for a method claiming temporal coherence as a primary contribution.
* The comparison with SC-GS shows lower performance (Table 1: 30.20 vs 31.89 PSNR), but SC-GS also uses control nodes. The paper doesn't adequately explain why their approach differs beyond adding temporal attention. A more detailed comparison highlighting the specific differences in node design and their impact would be valuable.
* The multi-frame sampling strategy uses an arbitrary 7:3 ratio between sparse stride and interpolation samplers with no justification or ablation. How sensitive is training to this ratio? Why not 5:5 or 9:1?

### Questions
* How does the method handle scenarios where objects enter or leave the scene mid-sequence? Does the k-NN relationship remain stable, or do you need special handling for appearance/disappearance events? Please provide examples or discuss this limitation.
* You claim T=6 provides the best balance (Table 2a), but the performance difference between T=6 and T=8 is marginal (28.35 vs 28.24 PSNR) while T=8 achieves higher FPS (156.2). Can you provide more analysis on this trade-off? Are there specific motion types that benefit from larger T?
* The method uses 2048 initial control nodes which are dynamically densified/pruned. What are the densification and pruning criteria?
* In Figure 5, your method shows sharper results than baselines. However, could you provide temporal consistency metrics? Static frame comparisons don't fully validate the claimed temporal coherence improvements. Have you conducted user studies comparing temporal quality?
* The top-k hard-frame loss selects the k frames with highest error per batch. What is k as a fraction of batch size? If k is too small, does this ignore most training signals? If k is too large, does this reduce to average loss? In Table 3, adding top-k loss improves LPIPS from 0.084 to 0.077, but this could simply be due to focusing on high-frequency details in difficult frames rather than improving overall consistency.

### Soundness
3

### Presentation
2

### Contribution
3
