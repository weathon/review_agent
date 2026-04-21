# Reducing Symmetry Mismatch Caused by Freely Placed Cameras in Robotic Learning

- Avg Score: 3.50
- Decision: Reject
- Scores: 3, 3, 3, 5

## Abstract
Equivariant policy learning has been shown to solve robotic manipulation tasks with minimal training or demonstration data.  However, the effectiveness of equivariance depends on whether transformations of the scene align with simple transformations of the input data. This is true when the camera is in a top-down view, but in the common case where a camera views the robot workspace from the side, there is a symmetry mismatch, reducing model performance. We show that equivariant methods perform better when camera images are transformed to appear as top-down images.  Our approach is simple to implement, works for RGB and RGBD images, and reliably improves performance across different view angles and learning algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper addresses a challenge in robotic learning, where freely placed cameras cause a mismatch between input image transformations and the inherent task symmetry in robotic manipulation environments. The authors propose two preprocessing methods: reprojection of RGBD images and perspective transformation for RGB images. These techniques transform side-view images into top-down views, thus aligning the image transformations with the task symmetry. This approach is shown to consistently improve performance in robotic manipulation tasks, particularly in reinforcement learning and imitation learning setups.

### Strengths
1. The paper offers practical preprocessing methods (RGBD reprojection and RGB perspective transformation) that are simple. These methods can be applied across various robotic learning tasks without additional training or modification of the robot setup.

2. The proposed methods require only knowledge of the camera’s intrinsics and extrinsics, making them straightforward to implement without the need for privileged information. This makes the approach broadly applicable across robotic tasks.

### Weaknesses
1. Limited Technical Contribution: The technical contribution of the paper is minimal. The methods of RGBD reprojection and RGB perspective transformation are well-established and mature techniques. The paper merely applies these existing methods to Equivariant Policy Learning without introducing any significant novel ideas. As a result, the work feels more like a technical report rather than a research paper offering new scientific insights.

2. Lack of Real-World Experiments: The experiments are conducted only in six simple simulated environments, without any real-world validation. This limits the applicability and robustness of the proposed methods in practical scenarios, as real-world experiments are essential to demonstrate the effectiveness of the approach outside of controlled simulations.

3. Performance Gap with Oracle: While the proposed methods reduce the performance gap with the oracle top-down view, they do not entirely close it. The occlusion of objects and grippers, especially in cluttered environments, remains an unsolved problem.

### Questions
1. Handling Extreme Occlusions: In the RGBD setting, how might more sophisticated inpainting or occlusion handling methods (e.g., learned inpainting) improve the performance gap with the oracle? Have the authors experimented with these techniques, and what were the results?

2. Effectiveness in Real-World Scenarios: While the experiments are simulated, can the authors elaborate on the challenges and potential modifications required to apply these preprocessing steps in real-world robot learning tasks with physical cameras and hardware?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper addresses the limitations of a certain type of equivariant policy learning in robotic manipulation tasks when using side-view camera perspectives, which cause symmetry mismatches that reduce performance. The authors propose a simple method to transform side-view images into top-down representations, enhancing the performance of equivariant methods. Its effectiveness is demonstrated on RGB and RGBD images.

### Strengths
- The problem formulation is quite straightforward. The proposed method is simple, intuitive, and effective.
- The discussion on Occluded Regions for RGBD images and Out-of-plane Distortion for RGB images makes the proposed method more practical, giving it the potential to be deployed in real-world settings.

### Weaknesses
- Though very simple and effective under the tested scenario, this paper seems more like a small pre-processing module specifically designed for a certain type of SO(2) RL and IL methods. How many equivariant methods could benefit from the proposed method? I would like the authors to discuss this question, and list as many papers as possible.
- Lacking real-world experiments. I am concerned whether the proposed method would be effective as well in real-world settings. And since the proposed method is mainly designed to tackle the challenge when deploying cameras in the real world, I think real-world experiments are indispensable.

### Questions
- How many equivariant methods could the proposed method benefit?
- Would the proposed method also benefit general-purpose robot learning methods such as Diffusion Policy?
- It seems that the compared point cloud baseline is using a single-view RGBD image. What if we have access to multi-view RGBD images? Consider the scenario in [1].
- Would real-world experiments be conducted?


[1] RiEMann: Near Real-Time SE (3)-Equivariant Robot Manipulation without Point Cloud Segmentation. Gao et al. CoRL'24.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes a method to improve equivariant policy learning in robotic manipulation tasks where camera views are not ideal (e.g., side views instead of top-down). The authors present two preprocessing techniques:
- Reprojection of RGBD images to approximate top-down views by generating point clouds and interpolating missing data.
- Perspective transformation of RGB images to map the ground plane onto a top-down view.

These methods enhance performance across different learning tasks and camera angles without additional data or privileged information, making them adaptable to real-world setups. The experiments show improved policy learning outcomes in several robotic tasks by aligning image transformations with physical symmetries in the robot workspace.

### Strengths
- The paper defines a problem of "symmetry mismatch" from non-ideal camera placements in image based equivariant robotic learning. By applying reprojection and perspective transformations to side-view images, it extends the utility of equivariant learning in robotics, enabling its application in more realistic setups.
- The authors provide a thorough and well-validated empirical analysis across diverse robotic tasks and modalities (RGB and RGBD), with clear comparisons to multiple baselines. This experimental rigor strongly supports the paper's claims about the effectiveness of the preprocessing techniques.

### Weaknesses
- The problem this paper attempts to address may not be a genuine issue. When handling tabletop robotic manipulation tasks and aiming to apply O(2)-equivariant policy learning algorithms, a fundamental assumption is the availability of top-view observations. If only side-view images are accessible, a more natural approach might be to consider non-equivariant policy learning algorithms instead.

- The methods proposed in this paper lack originality. Both 3D reprojection and perspective transformation are well-established algorithms in the field of computer vision. This paper merely applies them to a specific scenario—converting side-view images of tabletop robotic manipulation scenes into top-view images—to facilitate the use of O(2)-equivariant policy networks. I view these techniques as pre-processing tricks rather than substantive innovations.

- The formulation for 3D reprojection in this paper is not entirely realistic. To perform reprojection, RGBD information is required. However, if 3D data is available, it would be more straightforward to use equivariant policy networks based on 3D groups$^{[1,2]}$ (such as SO(3), SE(3), or SIM(3)). This would eliminate the need to address issues arising from mismatched camera viewpoints.

[1] Yang, J., Cao, Z. A., Deng, C., Antonova, R., Song, S., & Bohg, J. (2024). Equibot: Sim (3)-equivariant diffusion policy for generalizable and data efficient learning. arXiv preprint arXiv:2407.01479.

[2] Chen, Y., Tie, C., Wu, R., & Dong, H. (2024). EqvAfford: SE (3) Equivariance for Point-Level Affordance Learning. arXiv preprint arXiv:2408.01953.

### Questions
- In the experimental part, the author compares many baselines (equivariant, non-equivariant, 2D, 3D), but does not clearly write out the specific structure of each baseline and the group on which their equivariance properties are defined.
- All baseline methods in this paper are based on the same framework (SACfD). To demonstrate the effectiveness of this preprocessing approach in broader scenarios, I believe it would be beneficial to include comparisons with other state-of-the-art methods.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper addresses a key issue in equivariant neural networks for agent learning to decrease the gap between sideview camera observations, which perform sub-optimally when cameras view the scene from the side rather than directly above. The authors propose two simple preprocessing techniques to reduce this gap: 

1. For RGBD cameras, they reproject the image to a virtual top-down view, and 
2. For RGB cameras, they apply a perspective transformation to align the ground plane with the image plane. 

Through experiments across multiple robotic manipulation tasks using both reinforcement learning and imitation learning, they demonstrate that these preprocessing steps significantly improve the performance of equivariant networks compared to using raw side-view images.

### Strengths
1. This work addresses a very common problem prevailing in the robotic manipulation domain i.e lack of robustness of vision based policies to viewpoints.
2. The proposed solutions are very simple (under the known camera extrinsics assumption, which is typically common in table-top robotic manipulation settings).
3. The paper is generally well written and easy to understand in a single go.

### Weaknesses
4. Related works: I believe a small discussion on point cloud models (in the context of Image reprojection) should also be discussed. Several works in the the past few years have proposed using point clouds for RL / policy learning [1, 2] and shown robustness to viewpoints [3].

5. Sample-efficiency of RGBD experiments: I don't particularly find a difference between *Point cloud equi* and *Reproj. equi* in Fig 5. and Table 1. What are the benefits of Reproj. Equi over point cloud equi?

6. Sec 5.6 (Effects of camera angle) needs to also have the PointNet++ baseline (*point cloud equi*) for the RGB-D plots. Some works have suggested that point cloud RL policies are robust to viewpoint changes [3].
---
**References:**

1. On the efficacy of 3d point cloud reinforcement learning, Zhan Ling et al., arXiv 2023
2. Point Cloud Matters: Rethinking the Impact of Different Observation Spaces on Robot Learning, Haoyi Zhu et al., NeurIPS D&B 2024.
3. Point Cloud Models Improve Visual Robustness in Robotic Learners, Skand Peri et al., ICRA 2024

---
**Rationale for current rating**: Overall I believe this is a well written paper with clear contributions. However, I have particular questions regarding the baselines (points 5, 6, 8) and generalization (point 9) and based on that, I'm voting for a weak reject. However, this is *not* my final decision and I am willing to update my score based on other reviewers' comments and authors' rebuttal.

### Questions
7. Gripper Image: Does this formulation of having a gripper image generalize to a dextrous manipulation with a non-trivial gripper? Also, it would be better if the Fig 11 (from appendix) can be moved/integrated into the main paper. This is because, the gripper representation is one of the crucial aspects of the proposed solution and having it in a visual form would make the methodology more clearer to the reader.

8. I would like to see an experiment with DrQ-v2 where image augmentaiton has shown significant sample-efficiency gains and am curious how that performs as compared to an explicit Equivariant policy. I believe the data-augmentations can be implemented in a straightforward manner within the SACfD codebase.

9. Are the models in Fig 7(a) and 7(b) test-only models are are they trained on individual camera angles? If it's trained and tested separately -- I'm curious to see how Reproj equi or Presp. equi perform on testing on OOD camera viewpoints (i.e train on one camera angle and test on rest.)

10. Are the class of Equivariant policies biased to the action space? Would the same set of architectures work for a other action spaces that are common in robotic manipulation such as end-effector pose, joint velocities, joint angle poisitions etc?

### Soundness
3

### Presentation
3

### Contribution
2
