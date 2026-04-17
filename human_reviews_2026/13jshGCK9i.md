# D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Simulation provides a cost-effective and flexible platform for data generation and policy learning to develop robotic systems. However, bridging the gap between simulation and real-world dynamics remains a significant challenge, especially in physical parameter identification. In this work, we introduce a real-to-sim-to-real engine that leverages the Gaussian Splat representations to build a differentiable engine, enabling object mass identification from real-world visual observations and robot control signals, while enabling grasping policy learning simultaneously. Through optimizing the mass of the manipulated object, our method automatically builds high-fidelity and physically plausible digital twins. Additionally, we propose a novel approach to train force-aware grasping policies from limited data by transferring feasible human demonstrations into simulated robot demonstrations. Through comprehensive experiments, we demonstrate that our engine achieves accurate and robust performance in mass identification across various object geometries and mass values. Those optimized mass values facilitate force-aware policy learning, achieving superior and high performance in object grasping, effectively reducing the sim-to-real gap.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes D-REX, a real-to-sim-to-real method to improve the performance of grasping policies. The method first leverages off-the-shelf tools to convert human videos to robot kinematic trajectories. After that, the method identifies object masses based on differentiable physics engines. The identified object masses are further utilized to develop a force-based control policy for grasping. Extensive experiments prove the effectiveness of the method. Overall, the paper makes a step forward in developing physics-aware policies utilizing real-to-sim-to-real methods and demonstrates the effectiveness of incorporating physical properties into robot policies. Main limitations lie in the inherent restrictions of the mass identification method and the restricted upper limits of only identifying and utilizing masses in the robot policy learning.

### Strengths
- Good motivations. The problem is well motivated. Real-to-sim system identification is an important way to bridge the sim-to-real gap. Beyond properties reflected from the visual appearances, such as object meshes, the paper makes a step forward and proposes to identify physical properties, i.e., masses, from a dynamic interaction sequence. After identifying masses, a force-adaptive method is developed to improve the grasping policy. 
- Reasonable methodology. Estimating masses from hand-object interaction sequences and the force-based policy are reasonable approaches. 
- Solid experiments. The authors carefully design experiments to validate the effectiveness of the object mass identification method design and the superiority of the force-based grasping policy.

### Weaknesses
- The paper utilizes the foundation pose to estimate the object pose sequences from real videos. However, the foundation pose cannot deal with axis-symmetric objects and tiny objects. Moreover, the estimated object poses always suffer from noise. Therefore, the quality of the mass identification step would be restricted by the quality of the identified masses. The applicability of the method would also be restricted to objects that the foundation pose can handle. Besides, utilizing foundation models to generate scene configurations, such as the robot mjcf, may also introduce errors. 
- When identifying masses from videos, the sim-to-real gap in other properties, such as frictions, is neglected, which would further make the estimation prone to errors.

### Questions
- How do you train the grasping position policy? 
- Could the method be extended to small objects, including both the mass identification and the grasping process?

### Soundness
3

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
1

### Summary
The paper presents a differentiable real-to-sim-to-real engine for object mass identification and grasping policy learning. Specifically, the Gaussian Splat representation is leveraged to facilitate estimation object mass through visual observations and robot control signals during interaction. Besides, a learning-based method is also proposed to train force-aware grasping policies from limited human demonstration videos. Comprehensive experiments have been conducted to validate the performances on mass identification and object grasping.

### Strengths
1.	The topic of end-to-end object mass identification is valuable and the solution using differentiable simulation is novel. 
2.	Based on the mass estimation, the proposed system achieves satisfactory performances on the object grasping, reducing the gap between simulator and real-world environment.
3.	The proposed approach outperforms strong baselines in object grasping, especially for the challenging object grasping. 
4.	The paper is well-written and the ablation studies are comprehensive.

### Weaknesses
1.	There should more details in the section of parameter identification from robot-object interactions. For instance, the rationale behind the trajectory discrepancy minimization for the object mass identification should be included. What’s the effects of the semi-implicit Euler modeling. Is there any ablation study for this learning objective?

### Questions
Please see the weaknesses.

### Soundness
3

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
5

### Summary
The paper presents D-REX, a differentiable real-to-sim-to-real framework for dexterous grasping. It combines 4D Gaussian Splatting (4DGS) for scene reconstruction and differentiable physics for identifying physical parameters (notably object mass) from real-world robot trajectories. The identified mass is then used to train a force-aware grasping policy, which achieves higher grasp success rates in both simulation and limited real-world settings.


The paper explores an interesting and relevant direction by combining differentiable physics and 4DGS for Real2Sim2Real learning. While the current experiments do not fully validate the motivation of building a high-fidelity differentiable Real2Sim pipeline, the proposed idea is novel and potentially impactful. Overall, the paper presents a promising step toward differentiable Real2Sim learning and merits a weak accept pending more thorough validation.

### Strengths
- The paper is clearly written and presents a coherent overall system.
- The proposed framework is conceptually appealing, and incorporating mass identification for robot policy learning is a novel and promising idea.  
- The experimental results demonstrate accurate mass estimation and consistent grasping improvement with mass-aware policies.

### Weaknesses
- **The evaluation of Real2Sim quality is lacking.**  
  The paper presents no analysis or evaluation of either appearance or geometry (mesh) of the generated digital scenes—offering neither quantitative metrics nor qualitative discussion. As a result, the claimed Real2Sim objective remains unsupported and unvalidated, which weakens the overall completeness and credibility of the contribution.

- **The validation of the “force-aware policy” is insufficient.**
  To validate the effectiveness of the force-based control, the authors report grasping success rates in Table 3 and Figure 5 undering different settings. A visual or quantitative comparison of force values across different settings during robot execution would better substantiate the effectiveness of the proposed force-aware policy learning.

- **The efficiency and scalability should be discussed.**  
  The authors present mass-loss curves in Figure 11 but seem to have omitted the actual optimization or training time, which is also an important factor for evaluating system efficiency. In addition, the offline 4DGS reconstruction step can be computationally expensive; however, no quantitative analysis (e.g., runtime or memory usage) is provided. Therefore, the practicality of the proposed Real2Sim pipeline for large-scale or online applications remains somewhat unclear.

- **The generalization ability of D-REX is limited.**
  The proposed pipeline requires real-world object trajectories (obtained via FoundationPose) and matched real/sim robot interactions to optimize object mass. Consequently, the generalization ability of the proposed method to novel scenes or objects, where such real trajectory data and robot interactions are unavailable, appears limited.

### Questions
1. The paper introduces two separate sets of Gaussians to represent the visual appearance and geometry of the scene. How might potential optimization misalignment between these two representations affect the accuracy of mass identification and the subsequent policy learning?
2. The Appendix mentions that the optimization of 4D Gaussian Splatting for photometric alignment is unstable and inaccurate. Could you provide detailed explanations or illustrative failure examples to clarify this issue?

## Suggestions for Improvements

1. Provide more qualitative and quantitative results/comparisons after Real2Sim stage, evaluating both the 4DGS scenes and the generated mesh.
2. Provide more visual or quantitative results/comparisons of force values during robot excution to better substantiate the force-aware policy.
3. Provide a systematic runtime analysis for different modules of D-REX to better assess the overall computational efficiency of the pipeline.
4. Minor formatting issues: Line 97: “Empirically, We” — the word “We” should be lowercase; Line 1737: “Physics-constrained identification” — a line break is recommended for proper formatting.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents D-REX, a differentiable real-to-sim-to-real pipeline that couples Gaussian Splat Representations for photorealistic 3D reconstruction with a differentiable physics engine for object-mass identification and force-aware grasp policy learning. The system jointly optimizes physical parameters (mass) from robot interaction videos and learns manipulation policies conditioned on the inferred mass, closing the sim-to-real loop for dexterous grasping tasks.

### Strengths
1. The paper demonstrates a technically competent system that merges 3DGS and differentiable physics for vision-based grasping. Authors have performed real-world experiments validating some of their claims.

### Weaknesses
**1. Pipeline composition rather than a learning contribution.**
The full system is essentially a sequential pipeline: (1) Gaussian Splatting for 3D reconstruction with VLMs, (2) System identification to calibrate physical parameters, and (3) a procedural grasping policy that uses hand-designed grasp position and orientation heuristics. There is no novel algorithmic contribution or learning formulation that connects these modules beyond standard differentiable chaining. 

**2. Hand-designed grasp prediction.**
The grasping procedure relies on manually defined rules. This is not significantly different from prior grasp pipelines that use geometry-based scoring or analytical quality metrics.

**3. No clear advantage over existing methods.**
The paper does not demonstrate how D-REX materially improves over existing differentiable grasping frameworks that already combine differentiable rendering and physics. The quantitative differences appear modest and could stem from tuning rather than a new principle. Additionally, just identifying mass, without taking materials into consideration, seems incomplete for robotics purposes.

### Questions
Please see my weaknesses section, thanks!

### Soundness
3

### Presentation
3

### Contribution
3
