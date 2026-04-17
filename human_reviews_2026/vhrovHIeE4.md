# AlignDrive: Aligned Lateral-Longitudinal Planning for End-to-End Autonomous Driving

- Decision: Reject
- Scores: 4, 2, 6, 8

## Abstract
End-to-end autonomous driving has rapidly progressed, enabling joint perception and planning in complex environments. In the planning stage, state-of-the-art (SOTA) end-to-end autonomous driving models decouple planning into parallel lateral and longitudinal predictions. While effective, this parallel design can lead to i) coordination failures between the planned path and speed, and ii) underutilization of the drive path as a prior for longitudinal planning, thus redundantly encoding static information.
To address this, we propose a novel cascaded framework that explicitly conditions longitudinal planning on the drive path, enabling coordinated and collision-aware lateral and longitudinal planning. Specifically, we introduce a path-conditioned formulation that explicitly incorporates the drive path into longitudinal planning. Building on this, the model predicts longitudinal displacements along the drive path rather than full 2D trajectory waypoints. This design simplifies longitudinal reasoning and more tightly couples it with lateral planning. Additionally, we introduce a planning-oriented data augmentation strategy that simulates rare safety-critical events, such as vehicle cut-ins, by adding agents and relabeling longitudinal targets to avoid collision. Evaluated on the challenging Bench2Drive benchmark, our method sets a new SOTA, achieving a driving score of 89.07 and a success rate of 73.18\%, demonstrating significantly improved coordination and safety.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a novel end-to-end autonomous driving planning model called AlignDrive. Unlike previous approaches that jointly predict lateral and longitudinal displacements, this work first predicts the driving path, and then estimates the offset along this path. This design effectively decouples path prediction from speed prediction, leading to impressive performance improvements on the Bench2Drive leaderboard. In addition, the paper also proposes a data enhancement strategy that can supplement the number of strong interaction scenarios by adding virtual agents on the driving path. This strategy also effectively improves the planning performance.

### Strengths
1. Unlike previous works that jointly predict lateral and longitudinal displacements, this paper introduces a trajectory planning approach that decouples path prediction from speed prediction, which is a well-motivated and effective idea.
2. On the Bench2Drive leaderboard, the proposed model demonstrates significant improvements in planning performance compared to prior methods.
3. The paper and supplementary materials provide detailed implementation information, making it easier for readers to understand and evaluate the effectiveness of the proposed approach.

### Weaknesses
1. Regarding data augmentation, there is an important issue that remains unclear. Why is agent encoding (Line 251) necessary? Is it because the data augmentation is not performed during data generation (i.e., the corresponding surround-view images are unchanged), and thus the model needs this encoding step to recognize the newly added agents? If so, the feature representations of these synthetic agents would inherently differ from those detected by the perception module. A better solution might be to introduce such augmentation directly during data generation, which would eliminate the need for extra encoding operations and ensure feature consistency across agents.
2. The proposed data augmentation strategy is somewhat effective, but it also has clear limitations. It only generates agents randomly along the ego vehicle’s driving path. Therefore, the generalizability of this augmentation method to broader datasets and real-world driving scenarios remains to be validated.
3. The statements in Line 015 and Line 046: “splitting planning into two independent branches can lead to coordination failures. For example, the longitudinal trajectory may indicate a high desired speed, but executing this speed along the lateral drive path can cause collisions” are not entirely accurate. Both the joint prediction of lateral and longitudinal displacements and the proposed decoupled approach can potentially lead to prediction errors and collisions; whether collisions occur depends on the prediction accuracy itself. The current phrasing could give readers the misleading impression that the proposed method inherently avoids such issues.

### Questions
1. I find that Figures 1 and 2 are not presented clearly enough. In Figure 1, it is unclear what the examples on the far left and far right are intended to illustrate. My understanding is that they aim to demonstrate the advantages of the proposed approach over prior methods that rely on fixed temporal or spatial interval prediction, but the visualization is not very intuitive. In Figure 2, there are several interaction modules (e.g., “Contextual Interaction”), yet it is not clearly shown which queries are involved in each interaction. Although this is described later in the text, it would be much clearer if the figure itself explicitly visualized these connections. In addition, the meaning of the numbers 0.8, 0.1, etc. on the right-hand side is not explained. Do they represent probabilities or confidence scores? Authors should describe this in the image or caption. 
2. Since the proposed approach still adopts an anchor-based design, it may be inherently constrained by the dataset’s distribution. Have the authors considered an alternative strategy that directly predicts the driving path without anchors, followed by speed prediction based on that path?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces AlignDrive, a cascaded planning framework for end-to-end autonomous driving. The core idea is to tightly couple lateral (path) and longitudinal (trajectory) planning by explicitly conditioning longitudinal planning on the predicted lateral path, addressing coordination failures in prior methods. The authors also reformulate longitudinal planning as a 1D displacement prediction task along the driving path, simplifying dynamic interaction modeling. Additionally, a planning-oriented data augmentation strategy is proposed to generate diverse safety-critical scenarios, improving the robustness of the model. Experiments on the closed-loop benchmark Bench2Drive demonstrate state-of-the-art performance for AlignDrive.

### Strengths
- Clear problem definition with practical significance.

The paper addresses the critical issue of coordination between lateral and longitudinal planning in end-to-end autonomous driving, which has direct implications for safety and performance.


- Well-structured method design.

The introduction of path-conditioned planning and 1D displacement prediction simplifies the longitudinal planning task while improving dynamic interaction modeling.

- Effective data augmentation.

The proposed planning-oriented data augmentation generates diverse safety-critical scenarios, significantly improving the model’s robustness in complex interactions.

- Strong experimental results.

AlignDrive achieves superior performance on the Bench2Drive benchmark, particularly in success rate and collision avoidance metrics.

### Weaknesses
- Incremental contribution.

While the proposed method introduces path-conditioned planning and displacement prediction, the overall approach is relatively incremental. Lateral and longitudinal coupling is already a common practice in traditional autonomous driving planning, limiting the novelty of this work.

- Incomplete evaluation.

Although Bench2Drive is a challenging closed-loop benchmark, there remains a significant gap between simulation results and real-world driving scenarios. The authors should evaluate their method on semi-open-loop benchmarks, such as NavSim, to better assess generalization. Closed-loop benchmarks cannot fully eliminate the possibility of designing test-specific tricks. Open-loop evaluations would help validate the robustness of the method further.

- Writing quality needs improvement.

The paper contains many sections that exhibit patterns typical of large language models. The writing lacks conciseness and academic rigor, which detracts from the overall quality. The authors should refine the language for better readability and professionalism.

### Questions
Please further discuss the contribution and novelty of this work.

### Soundness
2

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
3

### Summary
The paper introduces a new end-to-end autonomous driving framework AlignDrive, which aims at improving coordination between longitudinal and lateral planning. The approach leverages a cascaded path-conditioned formulation that tightly couples longitudinal and lateral planning through the use of anchor-based displacement regression along the drive path. Additionally, the framework incorporates a planning-oriented data augmentation strategy that simulates safety-critical events by inserting synthetic agents and adjusting longitudinal displacements. Evaluated on the challenging Bench2Drive benchmark, AlignDrive achieves state-of-the-art closed-loop driving performance, setting a new driving score of 89.07 and a success rate of 73.18%.

### Strengths
1. AlignDrive effectively addresses the limitations of previous methods like TF++ and HiP-AD by coordinating longitudinal and lateral planning through a novel cascaded design. This tightly couples the two tasks, ensuring better path-following and collision avoidance in complex driving scenarios. This is also the major novelty of the proposed framework.

2. The framework demonstrates impressive closed-loop driving performance, achieving a significant improvement over prior methods. It sets a new benchmark with the highest driving score and success rate on the Bench2Drive dataset.

3. The authors introduce a planning-oriented data augmentation approach that simulates rare safety-critical events like vehicle cut-ins. This strategy helps the model learn to navigate challenging interactions, ultimately improving its ability to avoid collisions in dynamic environments.

4. The paper provides detailed ablation studies that clearly demonstrate the necessity of each component in the AlignDrive framework, including the path-conditioned longitudinal planning and the data augmentation strategy. This strengthens the validity of the proposed approach.

### Weaknesses
1. The visual quality of Figure 1 could be improved, as some icons are not very clear. This affects the overall presentation and could confuse readers trying to follow the diagram. Additionally, Figure 5 is hard to interpret as the figure appears overly clustered.

2. While the paper compares AlignDrive with methods like HiP-AD, it does not include a comparison with other notable previous works such as TF++, SimLingo, or Hydra-Next in Table 1. Even though some of these methods use different training data, a comparative analysis would provide a more complete context for the performance claims.

3. The paper focuses primarily on the Bench2Drive dataset. It would be beneficial to test AlignDrive on other datasets, such as nuScenes or NAVSIM, to validate its ability to generalize across different environments and real-world data.

### Questions
Please refer to the weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper explore the planning representation problem in closed-loop end-to-end driving. By identifying the problem in previous parallel lateral and longitudinal prediction methods, this paper proposes a semi-coupled lateral and longitudinal planning framework, which first predicts driving path for lateral control, and the displacement along the path for longitudinal control. A planning-oriented augmentation is further proposed based on the planning framework.

### Strengths
* A semi-coupled planning framework is proposed, which first predicts driving path for lateral control, and use this path as the prior for longitudinal planning. Longitudinal planning is reformulated as a 1D prediction problem along the drive path, focusing on dynamic interactions.
* With the planning framework, a planning-oriented data augmentation is proposed to generate diverse safety-critical scenarios.
* The method achieves SOTA close-loop performance on challenging Bench2Drive benchmark.

### Weaknesses
* Though the data augmentation is useful, the agent encode module only use bounding box state and category, would it lose some appearance information extracted from image and contrary to the end-to-end philosophy?
* The longitudinal anchor set on candidate path is not described very clear, making it a little bit hard to understand.

### Questions
* As stated in introduction, the static scene elements are captured by drive path, why cross-attention with map queries is still needed in Contextual Interaction?
* In Agent Insertion, a virtual agent is randomly initialized with a randomly sampled state. First, is it reasonable that some cases will not follow road driving rules (in Fig.6., some vehicles do not drive along the road). Second, can the sample process be more efficient, making more threatening case than random sample?

### Soundness
3

### Presentation
3

### Contribution
3
