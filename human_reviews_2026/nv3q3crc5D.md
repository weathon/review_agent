# Flash-Mono: Feed-Forward Accelerated Gaussian Splatting Monocular SLAM

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 2

## Abstract
Monocular 3D Gaussian Splatting SLAM suffers from critical limitations in time efficiency, geometric accuracy, and multi-view consistency. These issues stem from the time-consuming $\textit{Train-from-Scratch}$ optimization and the lack of inter-frame scale consistency from single-frame geometry priors. We contend that a feed-forward paradigm, leveraging multi-frame context to predict Gaussian attributes directly, is crucial for addressing these challenges. We present Flash-Mono, a system composed of three core modules: a feed-forward prediction frontend, a 2D Gaussian Splatting mapping backend, and an efficient hidden-state-based loop closure module. We trained a recurrent feed-forward frontend model that progressively aggregates multi-frame visual features into a hidden state via cross attention and jointly predicts camera poses and per-pixel Gaussian properties. By directly predicting Gaussian attributes, our method bypasses the burdensome per-frame optimization required in optimization-based GS-SLAM, achieving a $\textbf{10x}$ speedup while ensuring high-quality rendering. The power of our recurrent architecture extends beyond efficient prediction. The hidden states act as compact submap descriptors, facilitating efficient loop closure and global $\mathrm{Sim}(3)$ optimization to mitigate the long-standing challenge of drift. For enhanced geometric fidelity, we replace conventional 3D Gaussian ellipsoids with 2D Gaussian surfels. Extensive experiments demonstrate that Flash-Mono achieves state-of-the-art performance in both tracking and mapping quality, highlighting its potential for embodied perception and real-time reconstruction applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents Flash-Mono, a feed-forward monocular SLAM system that integrates a Transformer-based recurrent frontend, 2D Gaussian Splatting for mapping, and a hidden-state-driven Sim(3) loop closure. It achieves high-quality reconstructions and real-time performance, claiming up to a 10× speedup over optimization-based GS-SLAM methods. Overall, the paper demonstrates strong engineering execution but requires clearer validation to fully substantiate its claims of novelty and robustness.

### Strengths
- The paper is generally well written, logically structured, and easy to follow. Figures, tables, and ablations are clearly labeled and help readers understand both the system design and experimental results.

- The paper effectively combines a recurrent feed-forward frontend with a lightweight refinement backend, offering a plausible path toward faster monocular Gaussian Splatting without fully sacrificing reconstruction quality.

- The authors compare against multiple strong baselines (MonoGS, DepthGS, MASt3R-SLAM) using consistent metrics across ScanNet and BundleFusion

- The work reports concrete latency optimizations such as mixed-precision and CUDA Graph execution, showing practical awareness of real-time deployment concerns.

### Weaknesses
- Although the paper emphasizes real-time performance and a 10× speedup, the reported FPS is not clearly defined. While Appendix B.1 states that DepthGS includes the UniDepthV2 inference time, it remains unclear whether Flash-Mono’s FPS also accounts for the 20-iteration refinement stage, Sim(3) optimization, loop closure, and rendering overheads. Without these components, the comparison may not reflect true end-to-end latency. A complete runtime breakdown (frontend, refinement, loop closure, rendering) and a unified timing protocol across baselines are necessary to substantiate the claimed efficiency.

- The proposed system integrates several elements already established in previous works: 2DGS, Predict-and-Refine optimization (from existing feed-forward mapping schemes), and Transformer-based hidden state modeling (as seen in MASt3R-SLAM and CUT3R). However, the paper does not sufficiently disentangle which components are novel and which are adapted. The claimed contributions, feed-forward monocular reconstruction and hidden-state-based Sim(3) loop closure, lack rigorous ablation or replacement studies. For instance, the paper does not test substituting 2DGS with 3DGS, removing the hidden state, or comparing Predict-and-Refine with standard local BA. Consequently, the boundary of novelty remains unclear and the contribution feels primarily engineering efforts rather than new approaches.

- The evaluation focuses solely on indoor datasets, i.e. ScanNet V1 and BundleFusion, which substantially overlap with the training domains (Replica, ScanNet++, DL3DV). It is recognized that generalization to outdoor or large-scale environments may be limited due to mismatches in scene/depth scales. As a practical compromise, it is recommended to include additional experiments on unseen indoor/hybrid benchmarks such as TUM RGB-D, 7-Scenes, and ETH3D to more comprehensively assess the model’s robustness and effectiveness.

- The method relies on a CUT3R backbone pretrained on ScanNet v2. V2 mainly differs from V1 in labeling quality, where scenes between CUT3R’s training data and the evaluation sets used in this paper are still the same. Please justify this overlap/leakage.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a Gaussian-based SLAM system that adopts a feed-forward paradigm to predict the attributes of 2D Gaussians and relative poses by leveraging multi-frame contextual information. The core contribution lies in integrating a feed-forward module into the GS-based SLAM framework, effectively replacing the time-consuming optimization process used in most recent GS-based SLAM systems. In addition, a loop closure module based on the hidden states of keyframes is introduced to enhance tracking accuracy and mitigate error accumulation. Experimental results demonstrate both the effectiveness and efficiency of the proposed SLAM system.

### Strengths
1. The paper is well-written and easy to follow, and its motivation is clear and well-founded.
2. It introduces a novel feed-forward paradigm for Gaussian-based SLAM, replacing traditional optimization-based processes and significantly improving efficiency. By eliminating costly optimization steps, the proposed method contributes to the development of real-time and lightweight Gaussian-based SLAM frameworks.
3. The paper trains a recurrent feed-forward frontend model that aggregates multi-frame visual features into a hidden state via cross-attention and jointly predicts camera poses and per-pixel Gaussian properties. The hidden state is further utilized in the loop closure module to reduce cumulative drift and enhance tracking robustness.

### Weaknesses
1. Some state-of-the-art RGB-based SLAM methods, such as Photo-SLAM and DROID-Splat, are not included in the comparison. In addition, since S3PO-GS primarily focuses on outdoor scenes, it would be beneficial to include evaluations on outdoor datasets to better demonstrate the proposed SLAM system’s capability in handling long and complex outdoor sequences.
2. It would be beneficial to include a quantitative analysis of the total number of Gaussians required to represent the entire scene. This statistic could provide valuable insight into the efficiency of the proposed feed-forward prediction and map fusion modules, and demonstrate how effectively the method balances compactness with reconstruction quality.

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Flash-Mono, a monocular SLAM system that integrates a feed-forward model with 2D Gaussian splatting (2DGS) for real-time scene reconstruction and camera tracking. The key idea is to replace the traditional per-frame optimization of Gaussian attributes with a recurrent network that directly predicts poses and Gaussians from sequential inputs. The proposed system consists of three modules: a feed-forward frontend, a 2DGS-based mapping backend, and a loop closure mechanism based on hidden states. The authors claim a 10× speedup over existing GS-SLAM methods while achieving state-of-the-art performance in tracking and rendering quality on ScanNet and BundleFusion datasets.

### Strengths
The combination of a recurrent feed-forward model with 2DGS for monocular SLAM is novel. The use of hidden states as submap descriptors for loop closure is creative.  

The method achieves strong results in both tracking (ATE) and rendering (PSNR, SSIM, LPIPS), outperforming recent GS-SLAM systems.  

The paper is well-organized and easy to follow.

### Weaknesses
The evaluation is limited to indoor datasets (ScanNet, BundleFusion). It is unclear how the method generalizes to outdoor or large-scale environments.  

While the hidden state mechanism is innovative, its capacity for long-term consistency is not deeply analyzed.  

The paper does not discuss model size or memory usage, which are important for deployment on resource-constrained devices.

### Questions
1. How does the method perform in outdoor or large-scale scenes where scale variation and dynamics are more challenging?
﻿
2. Could the hidden state be further exploited for lifelong mapping or incremental learning beyond submap-based reset?
﻿
3. Have you considered comparing with non-GS SLAM systems in terms of robustness under motion blur or low-texture scenes?
﻿
4. What is the memory footprint of the model, and is it feasible for mobile or embedded platforms?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Flash-Mono, an monocular 3D Gaussion Splatting SLAM that employs the feed-forward paradigm to improve the accuracy and speed of GS-SLAM methods. The propose method trains a recurrent feed-forward fronted to predicts local camera posed and per-pixel 2D Gaussions and cobmines it with backend global mapping and loop closure to achieve final tracking and mapping results. The experimental results demonstrate the superiority of the proposed Flash-Mono.

### Strengths
The authors introduce a dedicated feed-forward Gaussion prediction based SLAM framework with sophisticated frontend and backend component designs to improve tracking and mapping performance in various scenes. The designs are reasonable and well-supported by and experimental results.

### Weaknesses
1.The feed-forward reconstruction-based SLAM is new direction for SLAM community, the authors should conduct extensive experiments to show its superiority and weakness on various SLAM scenes. That is, the authors should provide more results on indoor and outdoor benchmarks and mapping reconstruction metric like Completion and Chamfer.

2.The authors do not provide a sufficient comparison with other feed-forward SLAM method, e.g., VGGT-SLAM. Additionally, the ablation results are too simple and no failure analysis about the limitations of the proposed method.

3.The explanation of some technical details is unclear. What data is used to train the frontend network, how the training data influence the final SLAM results?

### Questions
1.When the backend mapping fusion and refinement correction are performing, what frequency does the front-end and backend interacts and how it influence the final results.

2.How the historical interpretation P_a^j is calculated, project point cloud from current frame I_j to historical submap C_a?

3.Does the pose relocalization process perform after the local map refinement or before, what is the detailed pipeline of the whole system?

4.In Loop Correction of Gaussian Map, does this simple transformation of 2DGS primitives generate bad overlap between 2DGS, resulting in bad rendering and mapping.

### Soundness
2

### Presentation
2

### Contribution
2
