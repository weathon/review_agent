# FlowAD: Ego-Scene Interactive Modeling for Autonomous Driving

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 4

## Abstract
Effective environment modeling is the foundation for autonomous driving, underpinning tasks from perception to planning. However, current paradigms often inadequately consider the feedback of ego motion to the observation, which leads to an incomplete understanding of the driving process and consequently limits the planning capability. To address this issue, we introduce a novel ego-scene interactive modeling paradigm. Inspired by human recognition, the paradigm represents ego-scene interaction as the scene flow relative to the ego-vehicle. This conceptualization allows for modeling ego-motion feedback within a feature learning pattern, advantageously utilizing existing log-replay datasets rather than relying on scenario simulations. We specifically propose FlowAD, a general flow-based framework for autonomous driving. Within it, an ego-guided scene partition first constructs basic flow units to quantify scene flow. The ego-vehicle's forward direction and steering velocity directly shape the partition, which reflects ego motion. Then, based on flow units, spatial and temporal flow predictions are performed to model dynamics of scene flow, encompassing both spatial displacement and temporal variation. The final task-aware enhancement exploits learned spatio-temporal flow dynamics to benefit diverse tasks through object and region-level strategies. We also propose a novel Frames before Correct Planning (FCP) metric to assess the scene understanding capability. Experiments in both open and closed-loop evaluations demonstrate FlowAD's generality and effectiveness across perception, end-to-end planning, and VLM analysis. Notably, FlowAD reduces 19\% collision rate over SparseDrive with FCP improvements of 1.39 frames (60\%) on nuScenes, and achieves an impressive driving score of 51.77 on Bench2Drive, proving the superiority. Code, model, and configurations will be released here.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces FlowAD, a novel framework for autonomous driving that explicitly models the ego-vehicle's motion feedback on the perceived environment. The core idea is to represent this interaction as a learnable "scene flow" in the latent feature space. The method is structured around three main components: ego-guided scene partition to create flow units, spatial and temporal flow prediction modules to model dynamics, and task-aware enhancement strategies for downstream tasks. The paper also proposes a new metric, Frames before Correct Planning, to evaluate a planner's scene understanding speed. Extensive experiments on nuScenes and Bench2Drive demonstrate state-of-the-art performance across perception, planning, and VLM-based tasks.

### Strengths
1）In this work, the authors introduce a general flow-based framework for autonomous driving. While most existing methods mainly study the planning with current observation, the authors focus on performing the control outputs that shape future sensory input, which is a relatively absence part of existing methods.
2) The proposed FlowAD framework is elegantly designed and demonstrates remarkable generality. According to the experiment, it is successfully applied to diverse baselines (such as SparseBEV, SparseDrive, and Senna) across multiple task types (detection, planning, VLM analysis).
3) The paper is well-written and easy to understand. The experimental section is thorough and convincing. 
4) The authors propose a novel and valuable metric, FCP. This new metric addresses a genuine gap in evaluation by quantifying the responsiveness and scene understanding speed of a planner, complementing traditional metrics like L2 error and collision rate.

### Weaknesses
1) While the performance are promising,  the proposed method introduces a degradation on the computational cost, as evidenced by the drop in FPS compared to baselines (e.g., SparseDrive 9.0 FPS vs. FlowAD 7.6 FPS with ResNet50). It is recommended to add more detailed discussion on the trade-off between performance and efficiency, and potential avenues for optimization.
2) The concept of "scene flow" is central to the paper. While the intuition is clear, a more precise definition or visualization of what these learned flow features represent in the latent space could strengthen the methodological explanation. For instance, how do these features qualitatively differ from standard BEV or image features?
3) The use of KL divergence loss from world models is presented as a key component for training the flow prediction modules. However, I am curious about the experimental results presented in Table 11. It shows that removing both spatial and temporal losses leads to performance worse than the baseline. This suggests that architecture without proper supervision might be harmful. It is recommended to provide a deeper analysis of why this happens and how critical the specific world model loss formulation is.

### Questions
1)The dynamic adjustment of partition size relies on fitting a circle to past ego positions. How robust is this method to noisy or high-frequency steering inputs? Did the authors experiment with simpler or more robust ways to incorporate steering velocity?
2)It seems that the local aggregation uses a fixed range of 3 flow units. Was any other adaptive mechanism considered for this aggregation to handle objects of varying sizes more effectively?
3)The paper mentions using "mechanisms from world models" and the KL loss. Could the authors elaborate on how their spatial and temporal prediction modules differ from a standard recurrent world model applied patch-wise? The novelty in the prediction architecture itself could be clarified.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces FlowAD, a unified ego-scene interactive modeling framework for autonomous driving. Unlike standard BEV-based approaches that mostly treat ego-motion as an external factor to compensate for, FlowAD attempts to integrate ego trajectory information directly into the representation space through a flow-unit partitioning scheme. The method includes an ego-guided dynamic partition, spatial-temporal flow prediction, and a task-aware enhancement module, which together aim to improve perception and planning by better capturing the interaction between the ego vehicle and the surrounding scene. A new evaluation metric, FCP (Flow Consistency Precision), is proposed to measure alignment between ego-motion and scene evolution. Experiments on standard benchmarks show consistent improvements over SparseDrive.

### Strengths
I appreciate the clarity with which the authors formulate ego-scene interaction as a modeling problem. This is a conceptual contribution with potential impact. The flow-unit partitioning is elegant and physically interpretable. The method’s modular design allows integration into various downstream tasks, which enhances its practical value. The introduction of FCP as a metric is also interesting, as it focuses on ego-scene alignment, something that existing metrics often overlook. Empirical results are consistent across multiple tasks, lending credibility to the claims.

### Weaknesses
Robustness is also under-characterized: the method hinges on ego-guided partition but offers little evidence for stability under sharp turns, high speeds, or other extreme maneuvers. The proposed FCP metric is interesting yet its external validity remains unclear—there is no systematic analysis of how it correlates with closed-loop planning metrics across routes/seeds or datasets.

### Questions
How does FlowAD compare with a token-based BEV fusion baseline and a compact world-model/diffusion-style planner under matched training and evaluation conditions?

How does FCP correlate with standard closed-loop planning metrics such as driving score, collision rate, and comfort across multiple runs or datasets?

### Soundness
4

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
3

### Summary
This paper proposes FlowAD, a general flow-based framework for autonomous driving that addresses a key limitation in existing approaches: the inadequate modeling of ego motion feedback into scene observation. Drawing inspiration from human perception and relative motion, FlowAD introduces an ego-scene interactive modeling paradigm, representing ego-scene interaction as scene flow relative to the ego vehicle. The methodology centers around an ego-guided scene partition, spatial and temporal flow prediction modules, and task-aware enhancement for downstream perception, planning, and VLM  analysis tasks. A new evaluation metric, FCP, is proposed to quantify scene understanding. The framework is evaluated across several benchmarks and tasks, demonstrating consistent gains over baselines on multiple metrics.

### Strengths
- The paper explicitly addresses the gap of neglecting ego vehicle feedback by proposing to model this effect as a “relative scene flow,” balancing physical intuition with practical data-driven learning from logged data.
- FlowAD is built with end-to-end components, making it pluggable into various autonomous driving baselines.
- Extensive experiments covering open-loop and closed-loop scenarios, multiple tasks, ablation studies, and introducing the FCP metric, effectively demonstrate the method's efficacy and its potential advantages over the baselines.

### Weaknesses
- Many world-model–style approaches like DriveDreamer [1] and Drive-WM [2] exploit action-guided future status to optimize the trajectory. The claim that previous methods “ignore feedback” seems somewhat absolute and would benefit from a more rigorous comparison and clear delineation of the differences.
- The estimation of the partition starting point and turning radius depends heavily on the accuracy of the ego vehicle’s pose/odometry. It would be better to give more analysis of sensitivity to noise, calibration errors, or timestamp drift. Besides, partitioning exclusively along the image width assumes that the dominant relative motion is horizontal. The method’s performance under conditions with significant changes in longitudinal dynamics (e.g., on slopes or during acceleration/deceleration) or notable camera pitch variations is not analyzed.
- The chosen baseline methods are primarily from 2023 to 2024, and it would be beneficial to include comparisons with more recent approaches, such as the perception-free methods SSR [3] and LAW [4].

[1] DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving. arXiv. 2309.09777.

[2] Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving. arXiv. 2311.17918.

[3] Navigation-guided sparse scene representation for end-to-end autonomous driving. arXiv:2409.18341.

[4] Enhancing end-to-end autonomous driving with latent world model. arXiv preprint arXiv:2406.08481.

### Questions
- Could the authors provide a more detailed comparison and discussion with world-model–based methods to clarify the statement made in the introduction? Furthermore, could comparisons with more recent approaches, such as the perception-free methods SSR and LAW, be included to strengthen the evaluation?
- How sensitive is the final performance to the chosen partition size and the parameters within the dynamic adjustment module? In particular, what is the model's robustness in the presence of inaccurate ego-motion estimates?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes FlowAD, a novel modeling paradigm for autonomous driving that aims to address the lack of feedback from ego-motion to future observations in current models. The authors argue this gap leads to an incomplete understanding of the driving process. FlowAD's core idea is to model this "ego-scene interaction" as a "scene flow" relative to the ego-vehicle. The paper also introduces a new FCP metric.

### Strengths
1. Modeling the ego-scene interaction as "scene flow" in the feature space, inspired by human optic flow, is an interesting approach to the problem.
2. The paper is well written.

### Weaknesses
1. The newly proposed FCP metric judges "correctness" using L2 distance to the Ground Truth (GT) trajectory. This is a strong and potentially flawed assumption, as "closeness to GT" does not equal "correct" in planning. A safer, more conservative plan (e.g., braking earlier) might deviate from the GT but would be unfairly penalized by FCP as a "slow" or "wrong" response.
2. The paper does not demonstrate how FCP, an open-loop metric based on GT, translates to or predicts better closed-loop performance. Its value in a closed-loop context is not well-argued, only discuss in open-loop is some how meaningless.
3. The paper's SOTA claim is weakened by its comparison to relatively dated baselines (e.g., SparseDrive, UniAD). This makes it difficult to assess FlowAD's true performance.
4. The "scene flow" seems to primarily model the relative motion of the static background caused by ego-motion. It is unclear how this "ego-centric" flow model effectively handles the more critical, non-ego-centric dynamics, such as another vehicle suddenly cutting in while the ego-vehicle is driving straight.

### Questions
1. Can the authors provide analysis linking a better FCP score (open-loop) to a specific, measurable improvement in closed-loop driving (e.g., faster reaction to hazards), beyond the overall driving score?
2. Could the authors provide comparisons against more recent end-to-end planners (e.g., GenAD (ECCV24), DriveTransformer (ICLR25), MomAD (CVPR25), ORION (ICCV25), etc.) on both open-loop and closed-loop?

### Soundness
2

### Presentation
3

### Contribution
2
