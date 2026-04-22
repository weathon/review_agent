# Implicit 4D Gaussian Splatting for Fast Motion with Large Inter-Frame Displacements

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 2, 6

## Abstract
Recent 4D Gaussian Splatting (4DGS) methods often fail under fast motion with large inter-frame displacements, where Gaussian attributes are poorly learned during training, and fast-moving objects are often lost from the reconstruction. In this work, we introduce Spatiotemporal Position Implicit Network for 4DGS, coined SPIN-4DGS, which learns Gaussian attributes from explicitly collected spatiotemporal positions rather than modeling temporal displacements, thereby enabling more faithful splatting under fast motions with large inter-frame displacements. To avoid the heavy memory overhead of explicitly optimizing attributes across all spatiotemporal positions, we instead predict them with a lightweight feed-forward network trained under a rasterization-based reconstruction loss. Consequently, SPIN-4DGS learns shared representations across Gaussians, effectively capturing spatiotemporal consistency and enabling stable high-quality Gaussian splatting even under challenging motions. Across extensive experiments, SPIN-4DGS consistently achieves higher fidelity under large displacements, with clear improvements in PSNR and SSIM on challenging sports scenes from the CMU Panoptic dataset. For example, SPIN-4DGS notably outperforms the strongest baseline, D3DGS, by achieving +1.83 higher PSNR on the Basketball scene.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents SPIN-4DGS, a new 4D Gaussian Splatting (4DGS) method designed to handle scenes with fast-moving objects, which typically degrade reconstruction quality and cause motion blur or object disappearance. Instead of employing a time-varying, deformable formulation, SPIN-4DGS follows an explicit strategy and first constructs Gaussian sets independently per frame, then employs a lightweight neural network to refine latent features in a frame-wise manner, which are then fed into implicit decoder to predict Gaussian attributes. Experiments on the CMU Panoptic dataset, which includes sports sequences with fast motion (e.g., moving balls and players), demonstrate improved reconstruction quality over tested methods in terms of PSNR and SSIM.

### Strengths
- The quantitative and qualitative results show clear improvements over existing dynamic Gaussian splatting methods, including 4DGS and others that rely on depth supervision or deformable modeling.
- The paper offers thorough ablation studies, systematically analyzing the contributions of different components such as training iterations, refinement steps, and network design choices.
- The proposed refinement module is modular and lightweight, making it a potential plug-and-play replacement for attribute optimization stages in existing 4DGS pipelines to enhance reconstruction performance in both PSNR and SSIM metrics.
- The approach is conceptually simple yet effective, avoiding complex temporal modeling while still capturing inter-frame consistency through learned refinements.

### Weaknesses
- The paper briefly mentions other explicit 4DGS approaches, such as 4DGS (Yang et al., 2023) and 4D-Rotor-Gaussians (Duan et al., 2024), but lacks an in-depth comparison or discussion of their relative advantages and limitations. In particular, results for 4D-Rotor-Gaussians are missing from the comparison table, which limits the reader’s ability to judge empirical progress.
- The method is evaluated solely on the CMU Panoptic dataset, which features controlled indoor scenes with limited background complexity. Evaluation on more diverse benchmarks (e.g., N3DV (Li et al., CVPR 2022) and D-NeRF (Pumarola et al., 2021)), commonly used in prior dynamic scene reconstruction works, would better establish the generalizability of the proposed approach.
- The novelty claim could be articulated more clearly. The proposed frame-wise construction and refinement pipeline resembles modular post-processing strategies from prior 4DGS extensions. A more explicit positioning of the technical contribution relative to these prior methods would help.

Minor Issues
- Confirm whether cited preprints were published at major conferences (e.g., Yang et al., 2023 appeared at ICLR 2024).
- Line #177: should read f<sub>θ</sub> instead of F<sub>θ</sub>.

**References**
- Li et al. Neural 3D Video Synthesis from Multi-View Video. CVPR 2022.
- Pumarola et al. D-NeRF: Neural Radiance Fields for Dynamic Scenes. CVPR 2021.

### Questions
1. Could you elaborate on the key conceptual differences between SPIN-4DGS and other explicit 4DGS-based methods (e.g., 4DGS, 4D-Rotor-Gaussians)? Specifically, how does your per-frame refinement compare to time-slicing or deformable formulations in terms of accuracy and efficiency?
2. Please include quantitative comparisons against 4D-Rotor-Gaussians (Duan et al., 2024) to provide a more complete empirical evaluation.
3. To assess the generalizability of the proposed method, it would be valuable to add experiments on N3DV and D-NeRF datasets or discuss potential limitations when applied to more complex outdoor or cluttered environments.

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
3

### Summary
The paper proposes a novel approach for dynamic scene reconstruction, with a particular focus on handling fast object motions. The proposed model, SPIN-4DGS, predicts Gaussian attributes directly from spatiotemporal positions using a lightweight network that learns shared representations across Gaussians. Unlike conventional methods that store attributes explicitly for each Gaussian, SPIN-4DGS encodes them implicitly within the network parameters, thereby significantly improving memory efficiency. Experimental results show that the proposed approach achieves substantially higher accuracy than existing methods, especially in scenes involving rapid motion.

### Strengths
- The paper is well written and easy to follow, clearly articulating its motivation and main contributions.
- It introduces a lightweight yet effective framework for dynamic scene reconstruction that performs robustly in scenarios involving fast object motions.
- The idea of predicting Gaussian attributes directly from spatiotemporal positions is novel and well-motivated, leading to more reliable attribute learning for fast-moving objects with large displacements.
- The paper provides comprehensive ablation studies and analyses that effectively demonstrate the impact of the estimated spatiotemporal positions and the proposed attribute learning mechanism on overall performance.

### Weaknesses
- Experiments are conducted only on sequences from the Panoptic dataset, which limits the generalizability of the results. Additional evaluations on some other challenging benchmarks with complex motions such as D-NeRF, Neu3DV, MeetRoom, and DeskGames would strengthen the empirical validation.
- The paper does not include comparisons with the recently proposed MoDec-GS model, which is a closely related approach and an important baseline for dynamic scene reconstruction.
- The evaluation in Table 1 relies solely on distortion-based metrics (PSNR and SSIM) and does not report perceptual quality measures such as LPIPS, which are essential for a more comprehensive assessment.
- Several recent related works including 4DGV [1], 3D-4DGS [2], and SCas4D [3] are missing from the related work discussion and could provide valuable context and comparison for positioning the proposed method.

     [1] Dai et al., 4D Gaussian Videos with Motion Layering, ACM TOG 2025

     [2] Oh et al., Hybrid 3D-4D Gaussian Splatting for Fast Dynamic Scene Representation, arXiv 2025

     [3] Lyu et al., SCas4D: Structural Cascaded Optimization for Boosting Persistent 4D Novel View Synthesis, TMLR 2025

- The authors did not discuss any limitation of the suggested approach? For instance, could the authors provide results or discussion on the model’s performance in static or near-static scenes, where inter-frame displacements are minimal?

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

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
This paper presents SPIN-4DGS,  a practical and well-engineered solution to 4D dynamic spatial reconstruction. This paper observes the defects of previous 4DGS related methods and establishes a method with hybrid explicit and implicit modeling. The proposed method collects the spatial-temporal positions from other 4DGS methods and predicts the other Gaussian attributes based on the positional information in a feed-forward paradigm. This formulation avoids the shared properties across timesteps and achieves higher performance on sports related datasets.

### Strengths
1. This paper is well-motived. Presenting highly dynamic areas with large motions has been a long-standing problem in 4D reconstruction. This paper analyzes the origin of this phenomenon and concludes this as the temporally shared features.
2. Satisfying performance. The proposed method maintains high-fidelity reconstruction for large-motion 4D scenes, from both the quantitative results and visual effects.

### Weaknesses
1. Method formulation. The proposed pipeline tries to directly capture the spatio-temporal positions from previous 4DGS representations and build a feed-forward network to estimate the other properties. However, the position prior from other methods may introduce bias from individual methods, as different 4DGS methods show different specialties, making this design unstable and unrobust. Besides, this method trains a feed-forward network to estimate the Gaussian attributes to avoid temporally shared properties, which is a proxy of 'per-time 3DGS', i.e., the upper bound of this method is just naively maintaining a set of 3DGS for each timestep (which handles the large-motion areas best), and their method is merely trying to save these per-timestep 3DGS into a larger feed-forward network. In my opinion, this design is not of novelty or significance, and does not figures the core problem out. Besides, this method naively integrates the hash MLP and feed-forward attribute estimations, which is a similar pipeline with Instant Gaussian Stream (CVPR 25), and these techniques have been used in HAC (ECCV 24), 3DGStream (CVPR 24) and feed-forward reconstruction pipelines such as MVGaussian (ECCV 24), TranSplat (AAAI 25). 
2. Possible overclaim. This work claims in Line 104 to 106: 'our design preserves the rendering efficiency of prior 4DGS methods while improving storage and training stability, thereby enhancing the efficiency–quality balance required for practical deployment', which introduces confusions: Do we need to inference the feed-forward network multiple times before obtaining the final 4D result, or do we need to inference and store the estimated per-timestep Gaussian attributes in ahead? If the former applies, the method requires network inference before each rendering, thus the rendering speed is highly lagged. If the later applies, the method requires saving per-timestep Gaussian attributes, thus the storage overhead is huge. This leads to an overclaim of the method performance.
2. Training dataset and generality. The proposed evaluation is implemented on the Panoptic dataset, while how the feed-forward network is trained is unclear. Is the training implemented on the same dataset? It is not clear whether this paradigm guarantees a correct evaluation. If the network is trained across all scenes in Panoptic dataset, will it obtain a better performance by per-scene per-training? Besides, it is wondered how this feed-forward network obtains generality as the training and evaluation datasets are of small size. It is also interesting to investigate its performance on other 4D reconstruction benchmark.

### Questions
The main concerns are elaborated in the weakness, with confusions on the method formulation, performance claim and the training protocols. 

If the authors' rebuttal solves misunderstandings in the review, I am willing to further adjust my score.

### Soundness
1

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
This paper addresses the failure of 4D Gaussian Splatting (4DGS) to reconstruct scenes with fast motion and large inter-frame displacements, where objects often blur or disappear. The authors propose SPIN-4DGS, a novel framework that decouples spatiotemporal position estimation from Gaussian attribute learning. The core contribution is a hybrid approach where explicit 4D spatiotemporal positions (x, y, z, t) are used as inputs to a lightweight implicit network (a 4D hash encoder and MLP decoders) that predicts all other Gaussian attributes (e.g., scale, rotation, color, opacity). This design avoids the attribute collapse seen in prior work and demonstrates state-of-the-art reconstruction fidelity on challenging dynamic sports datasets.

### Strengths
* problem definition and diagnosis: The paper correctly identifies a significant and open challenge for the 4DGS paradigm: robustly modeling high-frequency dynamics. The analysis of failure modes in prior art—distinguishing between deformable methods (canonical space initialization failure ) and explicit 4D parameterizations (attribute collapse due to cross-frame interference )—is insightful and provides a methodologically sound basis for the proposed solution.
*  novelty of the implicit attribute network: The core idea of decoupling explicit positions from implicitly-learned attributes is novel. This hybrid representation allows the model to leverage the efficiency of explicit splatting while using the generalization power of an implicit network to maintain spatio-temporal consistency for attributes. This directly addresses the attribute collapse observed in prior explicit 4DGS methods .
* State-of-the-Art reconstruction fidelity: The method achieves significant quantitative and qualitative improvements on the CMU Panoptic Sports dataset, a highly appropriate and challenging benchmark for this task. The gains over strong baselines, including those using external supervision (e.g., +1.83 dB PSNR over D3DGS on 'Basketball'), are substantial and clearly demonstrate the method's effectiveness in capturing fast-moving objects.

### Weaknesses
* Efficiency-Quality Trade-off: The claims regarding rendering efficiency  are not fully substantiated and appear to be a key limitation. Table 1 indicates a significant drop in rendering speed (104 FPS) compared to the explicit 4DGS baseline (197 FPS) and especially deformable methods. This is likely due to the computational overhead of querying the implicit network for all Gaussian attributes per-frame during rasterization, which is a non-trivial trade-off for the gains in fidelity. This trade-off should be discussed more explicitly.
* Storage Footprint: The paper's claim of "reducing storage overhead"  appears overstated. The reported storage (1261MB) is only marginally better than the 4DGS baseline (1293MB). This suggests the bulk of the memory is consumed by the explicit spatiotemporal positions, and the implicit network for attributes does not offer a significant storage advantage in practice. The contribution lies in enabling reconstruction, not in compressing the representation.
* Training Scalability and Cost: The method introduces a two-stage training process. The total training cost (Stage 1 position estimation  + Stage 2 refinement  + Stage 2 implicit network training) relative to end-to-end baselines is not clearly reported. This makes it difficult to assess the practical scalability and computational budget required to deploy this method.

### Questions
* Q1:Inference Cost Analysis: Could the authors confirm that the ~2x reduction in FPS (vs. 4DGS) is due to the inference cost of the 4D Hash Encoder and MLP decoders required for every Gaussian at render time? If so, this is a critical aspect of the method's efficiency profile and warrants clarification.
* Q2: Storage Claims: Given the data in Table 1, the storage benefit appears negligible. Would the authors consider reframing this contribution? The novelty appears to be in the hybrid representation enabling high-fidelity reconstruction, not in achieving superior compression.
* Q3: Total Training Budget: Please provide a comparative analysis of the total wall-clock training time for SPIN-4DGS (including all stages) versus the primary baselines (e.g., 4DGS, D3DGS) for a single scene.
* Q4: Robustness of Stage 1: The method's success seems dependent on the Stage 1 position estimator. How does SPIN-4DGS handle cases where the explicit estimator (4DGS) completely fails to initialize any points for a fast-moving object? Does the refinement step (Sec 2.2)  only densify/prune, or can it also create new Gaussians in unobserved regions?

### Soundness
3

### Presentation
2

### Contribution
3
