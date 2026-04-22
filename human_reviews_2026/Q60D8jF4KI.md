# FastGrasp: Learning-based Whole-body Control method for Fast Dexterous Grasping with Mobile Manipulators

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Fast grasping is critical for mobile robots in logistics, manufacturing, and service applications. Existing methods face fundamental challenges in impact stabilization under high-speed motion, real-time whole-body coordination, and generalization across diverse objects and scenarios, limited by fixed bases, simple grippers, or slow tactile response capabilities. We propose **FastGrasp**, a learning-based framework that integrates grasp guidance, whole-body control, and tactile feedback for mobile fast grasping. Our two-stage reinforcement learning strategy first generates diverse grasp candidates via conditional variational autoencoder conditioned on object point clouds, then executes coordinated movements of mobile base, arm, and hand guided by optimal grasp selection. Tactile sensing enables real-time grasp adjustments to handle impact effects and object variations. Extensive experiments demonstrate superior grasping performance in both simulation and real-world scenarios, achieving robust manipulation across diverse object geometries through effective sim-to-real transfer.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a two-stage RL-based framework for object grasping with mobile manipulators. A cVAE model generates grasping poses based on object observations, and an RL policy controls the robot to reach and grasp the moving object. The sim-to-real gap for privileged observation is handled by attaching markers to the object for segmentation.

### Strengths
The writing is generally clear and logical, and the visualizations are informative. The method shows good performance in simulation. While the real-world performance reveals an unsolved sim-to-real gap, this limitation is discussed and inspires future work.

### Weaknesses
1. **Model design:** The policy encodes the object point cloud with PointNet and generates grasps with a cVAE. It remains unclear what the potential performance impact (e.g., increments or new issues) would be from upgrading to more advanced and recent models for point cloud encoding and conditional generation. Small-scale experiments (even simulation-only) and discussion would enhance the insights.
2. **Insufficient related work:** The related work sections focus mostly on recent studies, ignoring a vast number of profound works from the past few decades on dexterous grasping, manipulation, mobile robotics, and tactile-based manipulation. I would suggest the authors expand these sections for completeness.
3. **Mathematical formulation:** The mathematical formulation of the grasping score is neither a previously established metric nor is it quantitatively evaluated to demonstrate its efficacy. I would encourage a more detailed analysis (e.g., providing examples of how each score component contributes, or providing detailed quantitative scores for Fig. 4). Further, mathematical notations in Sec. 4.3, including the notation for a vector, the $\cap$ sign, etc., are very confusing. I would encourage using more standard notations and explaining in which space each symbol is defined.
4. **Sim-to-real deployment:** The real-world deployment uses color-based markers for detecting object poses, which, while sufficiently substituting for the privileged information in simulation training, limits the application of the method in the real world. Further extensions on filling this gap (e.g., distilling the policy to a vision-input one, efficient object detection, etc.) are necessary. Furthermore, other gaps exist: (i) ensuring effective grasp generation on depth-camera-captured object point clouds, and (ii) environmental obstacle perception and avoidance.
5. **Limited evaluations:** The paper only reports execution success rates in simulation. Additional evaluations, including quantitative metrics (Q1 / Ferrari-Canny metric) of the generated grasps, the error between the grasp guide and the executed grasping pose, time consumption, etc., would enhance the details.
6. **Insights on mobile manipulation:** Limited insights are provided on addressing mobile-manipulation-related challenges, including high-speed motion and the potential consequences of impact.
7. **Editorial issues:** Some citations are not in the correct format.

### Questions
1. Please explain the categorization of real-world objects according to their "geometric complexities". From Supplementary Fig. 8, I can hardly discern the basis for this "complexity".

2. Would the choice of Cartesian space end-effector observations and actions improve performance?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on mobile grasping, i.e., performing grasps without stopping. It proposes a two-stage pipeline consisting of grasp generation, grasp selection, and grasp-guided reinforcement learning. Both simulation and real-world experiments are conducted to validate the approach.

### Strengths
This paper presents a well-structured system and conducts both simulation and real-world experiments.

### Weaknesses
1. The grasp guidance does not account for arm filtering. Since potential collisions between the arm and the table can occur, filtering only based on the hand seems unreasonable.
2. The objects used in the real-world experiments have similar geometries, allowing them to be grasped using nearly identical grasp poses.
3. I acknowledge that the comparison between simulation and real-world experiments uses the same 19 real Dex objects. However, the simulation should include a larger and more diverse dataset for a more comprehensive evaluation. For example, following UniDexGrasp [1], the dataset could be divided into several subsets.
4. Under partial observation—typical in real-world scenarios—it appears difficult to obtain accurate GWC and GDC estimations simultaneously.
5. The claim of effective zero-shot sim-to-real transfer seems overstated.
6. A baseline that separates movement and grasping (e.g., static grasp) should be included to compare grasp success rates and efficiency.
7. The pipeline combining pose generation and pose-conditioned reinforcement learning is not very novel. The technical contribution of this work is limited, making it more suitable for a robotics-focused conference rather than ICLR.

[1] Xu, Y., Wan, W., Zhang, J., Liu, H., Shan, Z., Shen, H., ... & Wang, H. (2023). Unidexgrasp: Universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 4737-4746).

### Questions
1. The leap hand should have 16 dof, why only 13 in this paper?

### Soundness
3

### Presentation
3

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
The authors introduce FastGrasp, a learning-based framework for dexterous grasping with a mobile manipulator. The core challenge lies in coordinating the high-dimensional motion of a mobile base, a robotic arm, and a multi-fingered hand to grasp objects at high speed. The proposed method uses a two-stage approach: first, a CVAE generates diverse grasp candidates from an object's point cloud. Second, an RL policy learns to execute the whole-body motion guided by an optimal grasp selected via a novel "hand envelopment" metric. The system integrates tactile feedback to make real-time adjustments. Experiments show the method outperforms baselines in simulation and achieves a 20-25% success rate on a real robot.

### Strengths
- The problem of coordinating a mobile base, a 6-DOF arm, and a 13-DOF hand is non-trivial, and the paper tackles the full complexity of whole-body control, validated on a real hardware platform.
- The proposed grasp selection method based on GWC and GDC is a key contribution. Experimental results show this approach dramatically outperforms a more traditional force-direction-based selection.
- The paper includes a strong set of experiments that convincingly validate the authors' design choices.

### Weaknesses
- The final success rates of 20%-25% are still quite low, although it is much better than the two ablated baselines. A more detailed failure case analysis would be expected to understand the practical challenges. It is unclear whether failures stem from perception (segmentation), planning (poor grasp choice), or control (inability to stabilize impact dynamics). The real-world "Hard Case" objects are still relatively structured and symmetric (Fig. 8).
- The notation is inconsistent. For example, $[f_k, f_{\mathrm{thumb}}]$ is described as a "vector" but then used in an expression for the length of an intersection of intervals (Eq. 2). This makes the exact computation unclear without making assumptions. A formal definition using projection operators would be more precise.
- Misuses of `\citep` vs. `\citet` throughout the manuscript.

### Questions
- How critical are the color-based markers to the system's success? Have the authors conducted any preliminary tests using a learning-based, markerless segmentation method?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed a learning-based framework, FastGrasp, to achieve fast mobile dexterous grasping in both simulations and a real robot. FastGrasp contributes in the grasp proposal generation, grasp guidance selection, and integration of tactile feedback into reinforcement learning, demonstrating improved success rate compared to baselines.

### Strengths
- This paper achieves fast and dynamic mobile grasping, addressing a complex dexterous manipulation task.
- The paper showcased the results in both simulation and real world.

### Weaknesses
- The achieved fast dexterous mobile grasping motions look impressive in the submitted video. However, the contributions would be more convincing if more comprehensive analysis and results could be provided. Please see questions below for more details.

- The real-world success rate drops a lot compared to simulation results. Not sure if it could be considered effective zero-shot sim-to-real transfer as authors claimed. It would be great to provide some analysis on possible reasons for the performance drop or the failure cases.

### Questions
- **Object diversity:** The authors claim the proposed framework can achieve robust manipulation across diverse objects. Although total success rate is reported across various objects in simulation, success rate per object should also be reflected. Besides, the demos show limited object diversity in the submitted video, one for simulation and two for real world. 

- **Sim-to-real transfer:** From the video, trajectories in the simulation seem to have a clear different pattern from real-world trajectories in general. In simulation, trajectories look more smooth and optimal, from the starting point directly to the target object. However, in the real world, trajectories look less dynamic and optimal, from the starting point to the same level of object heights and continue to reach the objects horizontally. Does this come from sim-to-real discrepancies? Showing more trajectories would help understand if this is actually a pattern or not.

- **More metrics for performance evaluation:** Throughout the paper, success rate is the only metric to evaluate and benchmark the performance. Since fast grasping is the major contribution of the paper, it would be more convincing to compare task completion time as well. Moreover, the authors also mentioned the proposed framework addressed stability issues. Although stability is somewhat reflected in the success rate, it would be better if there is some stability-related metric to quantify and support this argument as well.

- **Baselines:** This framework is built upon Zhang et al. (2024) and Xu et al. (2023). However, it is not clear what is the improvement of this paper in the related work section or results section. None of the two works are used as baselines in this paper. The authors indeed mention Zhang et al. lacks the grasp precision for diverse objects in related work, however, no evidence is provided to support this argument. 

- **More ablation:** For ablation study in Section 5.3, ablation on some essential reward terms is missing, e.g. stable grasping reward and rapid completion reward. 

- **Limitations in safety:** Since safety is one of the crucial problems in fast mobile grasping, the paper should expand a bit more on this point in the limitations section. 

- **Limitations in object perception in the real world:** The authors attached color-based markers to objects in the real world. Instead of a solution to sim-to-real transfer, this seems more like a limitation for real-world deployment and should be mentioned in the limitations section.

- **Others:** 
  - In Section 4.3. More details should be provided for the executability filtering.

  - Policy structure in Fig. 3 should be briefly mentioned in the main text and then refer to Appendix.

  - In Section 4.4, for observation, how to get the distance between the object and the palm centre in the real-world?

### Soundness
2

### Presentation
3

### Contribution
2
