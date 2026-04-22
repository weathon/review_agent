# Cross-View Yaw Estimation in Location Uncertainty with Line-Aligning Yaw Scoring

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 6, 0, 2

## Abstract
Accurate rotation estimation is crucial in autonomous navigation and AR/MR (Augmented/Mixed Reality) applications. Small angular errors can lead to significant misalignment or navigation failures. Among the three rotation angles—pitch, roll, and yaw—yaw is the most challenging to estimate, as it lacks direct geometric cues, such as gravity-aligned structures. Yaw estimation given a BEV (Bird’s Eye View) image is treated as an inseparable cross-view localization problem that accompanies location and inevitably hypothesizes the height and distance of the ground pixel. We introduce LAYS, a line-alinging yaw scoring approach that enables precise yaw estimation. We propose a 3D voting-based search that effectively isolates the 1-DoF yaw component, enabling robust estimation without relying on ground-truth position or assuming ground height. In our method, BEV pixels are matched to a ground view column based on feature similarity. Using the relative yaw of the ground column, match scores are assigned to a yaw bin for each 2D pose pixel. To address location uncertainty, our method identifies line correspondence between the ground and BEV, and formulates the problem such that one such correspondence is sufficient to determine yaw. LAYS~achieves state-of-the-art sub-degree yaw accuracy, improving from 6.55\% to 34.81\% on the Mapillary Geo-Localization dataset, 41.36\% to 67.05\% on the Ford Multi-AV dataset, and  12.39\% to 23.67\% on the VIGOR dataset, setting a new benchmark for precise localization in real-world scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the challenging problem of accurate yaw estimation, which is critical for autonomous navigation and AR/MR applications but difficult due to the lack of direct geometric cues. The authors propose LAYS, a line-aligning yaw scoring method that reformulates yaw estimation as a one-degree-of-freedom problem independent of ground height or distance assumptions. Experimental results on several benchmarks, including Mapillary Geo-Localization, Ford Multi-AV, and VIGOR, demonstrate substantial improvements in sub-degree yaw accuracy over prior approaches.

### Strengths
1. The paper introduces LAYS, a line-aligning yaw scoring method that reformulates yaw estimation as a 1-DoF problem, which is both conceptually elegant and practically effective.

2. The idea of leveraging line correspondences between BEV and ground views provides strong geometric grounding and interpretability. The approach does not rely on assumptions about ground height or distance, improving robustness in real-world scenes.

3. The authors show that isolating yaw estimation as a 1-DoF (degree of freedom) problem can yield benefits in downstream full pose (e.g., 3-DoF) localization tasks, illustrating the practical value of decoupling yaw from translation estimation.
LAYS achieves significant improvements in sub-degree yaw accuracy across multiple benchmarks (Mapillary, Ford Multi-AV, VIGOR), clearly outperforming prior methods.

### Weaknesses
The analysis of failure cases appears limited: for instance, what happens when there are few dominant linear structures (roads, lanes) visible, or when the matching line correspondence is ambiguous?

### Questions
Could the authors clarify how the method handles cases where the dominant linear structure (e.g., road lane line) is absent, ambiguous, or heavily occluded (e.g., dense vegetation, parking lots)? How robust is LAYS in such scenarios?

Could the authors can utilize tools like GeoCalib [1] to handle inputs without gravity-alignment?

The method treats yaw estimation independently of translation (x,y) estimation, have the authors experimented with feedback loops where yaw estimation is used to refine translation?

[1]GeoCalib: Single-image Calibration with Geometric Optimization, ECCV 2024

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces LAYS, a novel approach for cross-view yaw estimation that decouples orientation estimation from location uncertainty. The key innovation lies in formulating yaw estimation as a line alignment problem between ground-level and bird's-eye view (BEV) images, rather than treating it as a byproduct of joint pose estimation. The method extracts column-wise features from ground images, matches them with BEV pixels, and employs a pairwise voting mechanism to estimate yaw without relying on ground height assumptions. Extensive experiments on three benchmarks demonstrate state-of-the-art performance, particularly under challenging noise conditions.

### Strengths
The paper makes several valuable contributions. The core idea of treating yaw estimation as an independent 1-DoF problem represents a significant shift from prevailing approaches that couple orientation with location estimation. The line alignment formulation is both novel and intuitive, effectively leveraging structural correspondences between perspectives.

The technical execution is thorough, with careful attention to feature extraction, matching, and voting mechanisms. The column-wise feature aggregation with relative yaw encoding is particularly clever, as it naturally handles viewpoint differences without explicit projection models.

Experimental validation is comprehensive, spanning multiple datasets with varying characteristics. The substantial improvements over current methods are convincing, especially the performance gains under ±180° yaw noise where existing approaches struggle. The ablation studies provide solid evidence for design choices, and the demonstration that LAYS can enhance existing 3-DoF methods by reducing their search space is practically significant.

### Weaknesses
While the method excels at yaw estimation, its standalone capability for precise location estimation appears limited. The formulation naturally distributes scores along radial lines, which is optimal for orientation but suboptimal for pinpointing exact positions. The paper briefly mentions this limitation but could more explicitly discuss the implications for practical deployment.

The computational requirements of the multi-resolution scoring and exhaustive matching aren't thoroughly analyzed. In applications like autonomous navigation or mobile AR, inference efficiency matters, and some discussion of computational trade-offs would strengthen the practical contributions.

The evaluation, while comprehensive across datasets, remains within relatively structured environments. The method's performance in highly unstructured scenes (e.g., natural environments without clear linear features) remains an open question, though this is perhaps beyond the paper's current scope.

### Questions
1. Given the method's strength in yaw estimation but limitations in precise localization, have you considered hybrid approaches that combine LAYS with complementary position estimation techniques? What would be the architectural implications?

2. Could you provide more insight into computational requirements and potential optimizations? For real-time applications, are there strategies to reduce the matching or voting complexity?

3. The method assumes gravity-aligned images. How sensitive is performance to residual pitch/roll errors that might occur in practical IMU-assisted systems?

4. The line alignment concept is powerful. Might this principle extend to estimating other parameters, such as camera height, by analyzing multiple line correspondences?

5. Some failure cases or challenging scenarios would be informative. Are there particular scene types or conditions where the line alignment assumption breaks down?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper estimates the Yaw rotation in cross-view localization problem. The core idea is to match BEV image pixels to ground-view image columns based on feature similarity. The paper presents results on certain datasets with sota performance.

### Strengths
1 The paper introduces a specific formulation for yaw estimation, framing it as a line alignment problem rather than a direct pixel-level correspondence task.

### Weaknesses
1 The work focuses solely on yaw estimation, assuming ground-truth position is given. This severely limits its practical utility for real-world applications like autonomous driving or VR/AR.

2 The methodological pipeline (Column Feature Extraction, Ground-BEV Matching, Pair-wise Yaw Voting) bears a strong resemblance to parts of frameworks like OrienterNet[1], which jointly solve for location and orientation. The paper fails to clearly articulate the fundamental novelty of its components beyond this existing work.

3 The experimental setup is limited. The absence of evaluations on standard benchmarks like KITTI raises concerns about the generalizability of the reported performance.

4 As shown in Table 3, the experimental performance is not good.

[1] OrienterNet: Visual Localization in 2D Public Maps with Neural Matching. CVPR 2023

### Questions
1 The experiments appear to be conducted under the assumption of a known, noise-free ground-truth position. Could you clarify this? In real-world scenarios, positional information (e.g., from GPS) is always noisy. How would your method perform with inaccurate position inputs, and what is the expected degradation in yaw estimation accuracy?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes LAYS (Line-Aligning Yaw Scoring), a novel 1-DoF yaw estimation framework that matches BEV pixels to ground-view columns via feature similarity and uses line correspondences for pairwise yaw voting, effectively decoupling yaw estimation from location uncertainty. It addresses the challenge of accurate yaw estimation in cross-view localization, where conventional methods struggle due to the lack of direct geometric cues and dependence on ground height assumptions. The proposed work achieves sub-degree yaw precision and substantial gains across multiple datasets (Mapillary, Ford Multi-AV, VIGOR), establishing yaw as an independent, solvable subproblem and improving global pose localization accuracy.

### Strengths
1. This paper presents a method that transforms a challenging 3-DoF rotation problem into a decoupled 1-DoF line-alignment task, enabling efficient and accurate yaw estimation without assuming ground height or distance.
2. Experiment section demonstrates state-of-the-art sub-degree accuracy and consistent improvements (up to ~30% absolute gains) across multiple major cross-view localization datasets, showing robustness and generalization.

### Weaknesses
1. In [CVPR 2020 - Where am I looking at? Joint Location and Orientation Estimation by Cross-View Matching], Shi et al. proposed cross-view image retrieval based on similarity between features encoded from ground panorama and polar-transformed aerial images, which is very similar to proposed method in terms of yaw alignment; proposed work claims disentanglement of yaw with the other 2 DoF, but limited experiment is performed and presented in supporting the claim, assuming $x$ and $y$ are accurate estimation is a strong assumption; hence would summarize for limited novelty and validation for proposed work.
2. Column-wise feature matching and yaw-bin scoring may incur significant computational overhead compared to end-to-end regression models, hence limited feasibility in real-world deployment.
3. in Eq.2, the $\|\|$ notation is not clearly indicated in terms of which kind of normalization.

### Questions
1. could the authors present more thorough experiment results on validating disentanglement of yaw improves 3 DoF pose accuracy with ablations?

### Soundness
2

### Presentation
2

### Contribution
2
