# FreeGaussian: Guidance-free Controllable 3D Gaussian Splats with Flow Derivatives

- Decision: Reject
- Scores: 6, 5, 6, 3

## Abstract
Reconstructing controllable Gaussian splats from monocular video is a challenging task due to its inherently insufficient constraints. Widely adopted approaches supervise complex interactions with additional masks and control signal annotations, limiting their real-world applications. In this paper, we propose an annotation guidance-free method, dubbed FreeGaussian, that mathematically derives dynamic Gaussian motion from optical flow and camera motion using novel dynamic Gaussian constraints. By establishing a connection between 2D flows and 3D Gaussian dynamic control, our method enables self-supervised optimization and continuity of dynamic Gaussian motions from flow priors. Furthermore, we introduce a 3D spherical vector controlling scheme, which represents the state with a 3D Gaussian trajectory, thereby eliminating the need for complex 1D control signal calculations and simplifying controllable Gaussian modeling. Quantitative and qualitative evaluations on extensive experiments demonstrate the state-of-the-art visual performance and control capability of our method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work focuses on reconstructing scenes from monocular videos and enabling controllable editing of Gaussian motion. First, the authors decouple motion information from optical flow to constrain the Gaussian flow. Then, they use DBSCAN to aggregate Gaussian trajectories and incorporate a 3D control vector for trajectory editing.

### Strengths
1. The figures in the paper are sufficiently clear. From the qualitative results, the performance surpasses that of the compared methods.
2. The motivation of the paper is clear; reconstructing dynamic scenes from monocular videos and editing motion is meaningful.
3. The authors conducted experiments on both synthetic and real scenes and provided comparisons with recent works.

### Weaknesses
1. The expression "self-supervised optimization" in line 18 is inaccurate; in this work, the supervision of Gaussian flow comes from an additional optical flow estimator.
2. The writing of the paper needs improvement. Section 3.2 is a bit hard to follow. Additionally, there are some minor errors, such as "caculated" in L280 and "conducted" in L299. Some papers, such as Gaussianflow and 3DGS, have been repeatedly cited.
3. The relationship between the two main contributions in the paper is unclear. The constraint on Gaussian flow and the control of motion seem to have no direct connection. The authors need to clarify this to highlight the relevance of these two parts.
4. There is a lack of necessary explanation regarding "3D Spherical Vector". Why can controllable editing of trajectories be achieved without any constraints during the control phase? If only the original trajectories are trained during the training phase, why can a "manually input interactive 3D vector" be used in the control phase?
5. The proposed method seems to show only a slight improvement compared to LiveScene, and may even perform worse than LiveScene in some sets.

### Questions
1. In Section 3.2, why predict camera flow using "translation velocity v and rotational velocity ω" instead of the relative camera pose R, T? What is the difference between it and the concurrent paper MotionGS [1] ?
2. In L251, how does the predicted Gaussian flow identify dynamic 3D Gaussians through back-projection? Since flow maps are in 2D space and Gaussians exist in 3D space, how is this mapping relationship determined? Is the DBSCAN clustering distance-based?
[1] Zhu R, Liang Y, Chang H, et al. MotionGS: Exploring Explicit Motion Guidance for Deformable 3D Gaussian Splatting. NeurIPS 2024.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper addresses the reliance on manual annotation and the need for masking in the Gaussian Splatting scene control line of work. The paper presents a simple intuition that the controllable dynamic regions can be identified from optical flow maps capturing both camera motion and scene motion. The dynamic regions are then clustered using DBSCAN to alleviate computation and ensure uniformity in motions. 3D spherical vectors are used for controls in replacement of commonly employed 1D state variable.

### Strengths
The paper's motivation is clearly stated. The paper presents a simple yet effective intuition of isolating the controllable region via camera motion disentanglement from optical flow maps. The motivation for adopting a 3D spherical vector for the control signal is discussed in detail. The diagrams of the overall system are clear and easy to understand, and the experiments sufficiently support the authors' arguments.

### Weaknesses
The paper is overall nicely written and properly presented. One main concern is the robustness of the proposed control region isolation and flow guidance in monocular captures. Many monocular captures do not have reliable camera poses, which can cause an issue with the flow-based isolation strategy presented in the paper. DBSCAN could also produce undesired clustering results in scenes if the Gaussian distributions are uneven across the scene, eg. when some regions are denser or sparser. It would be reasonable to show some visual comparisons and quantitative results on HyperNeRF scenes and CoNeRF face scenes as well.

### Questions
The paper can be further improved with the following clarifications: 

1. As discussed in the Weakness section, qualitative and quantitative results on datasets with noisy camera poses would help illustrate the robustness of the proposed method.
2. The paper currently does not show any failure cases. When would the approach fail? The limitation section mentions that lighting variations would compromise the results. However, this is not sufficiently supported in the discussions. How much of a lighting variation would affect the method and to what extent?

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
4

### Summary
This work leverages dense optical flow and changes in camera motion trends to perceive dynamic regions. By decoupling the linear and angular velocities of the moving camera and the linear velocity of dynamic scene Gaussian primitives, it achieves dynamic scene reconstruction from a moving perspective. The method realizes unsupervised, semantic- and text-prior-free 4D reconstruction, offering a novel approach for unsupervised Gaussian physical simulation.

### Strengths
1. The integration of optical flow and camera motion for perceiving dynamic regions is compelling, and the mathematical decoupling of dynamic Gaussian motion and camera movement is innovative.

2. Despite some jagged artifacts in local areas, the overall FreeGaussian reconstruction shows impressive 4D reconstruction quality, enhancing 3DGS robustness for dynamic scenes. Dynamic segments can simulate real motion trajectories for dynamic simulation through Gaussian modeling.

3. The work proposes a differentiable rendering constraint for optical flow and camera motion, decouples the camera’s rotation, translation, and dynamic Gaussian velocity, and solves all motion states using gradient descent. This is integrated into Gaussian’s CUDA implementation, enabling dynamic object rendering from moving perspectives based on Gaussian primitives.

4. The paper is well-written, with comprehensive experiments, and the mathematical derivation in the rendering section is rigorous.

### Weaknesses
1. Could there be instances where the perceived region is incomplete? For instance, variations in optical flow and camera motion trends can detect dynamic areas, but due to frequent object occlusions and limited observational perspectives, it might not be possible to view the entirety of dynamic regions from multiple perspectives. Could this impact the method’s effectiveness?

2. DBSCAN requires additional parameter design to cluster Gaussians. When objects are closely spaced, the spatial distribution of Gaussian primitives might lead to inaccurate clustering results.

3. The optimization for pre-trained Gaussians in the first stage lacks sufficient geometric priors, potentially leading to inadequate high-quality scene rendering in some scenarios, which may impact subsequent reconstruction quality.

### Questions
1. The design of Eq. (6) lacks details on the specific activation functions and output processing used.

2. Could the pre-trained Gaussians in the first stage have suboptimal dynamic reconstruction, potentially impacting the final simulation results?

3. The reconstruction of Gaussians does not incorporate advanced geometric constraints, which might affect DBSCAN accuracy in later stages.

4. The loss design is under the assumption of Gaussian isotropy. However, pre-trained Gaussians seem not to include isotropy constraints, which might significantly impact this proposition and its efficacy. I find this theory quite intriguing.

5. Following the previous question, if it does indeed cause an impact, then perspective transformations A and B will be introduced in L_uGS, and the CUDA part of Gaussian cannot directly backpropagate the gradients to the linear transformations A and B of the motion state variables. Is it necessary to additionally write the Jacobian matrix for backpropagation in CUDA?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces a method to reconstruct controllable Gaussian splats from a monocular video. The contributions include a derivation of 2D scene flow (referred to as dynamic gaussian flow) rendering and a interactive control pipeline that relies on spatial clustering. The method is tested on a few synthetic datasets (CoNeRF, OmniSim, InterReal).

### Strengths
- The task of improving the controllability of gaussian splatting is interesting.
- The idea of using spatial clustering to obtain controllable objects is sound.
- The derivation of 2D scene flow from 3D scene flow and camera motion looks correct.
- The paper did extensive experiments and comparisons.

### Weaknesses
Although some components of the paper is interesting, I have major concerns about the writing and soundness.
- Many technical pieces of the method is missing, which makes it difficult to understand the design choices.
    - What is the purpose of factorizing 2D optical flow to 2D scene flow and camera flow? This make both the derivation and flow loss more complicated than Gao et al 2024 and I don't see the benefit.
    - More algorithmic details about Sec 3.3. would help. What is the distance metric used in DBSCAN algorithm. How is 2D scene flow useful there? What is the model $\Theta$ and what is ${\bf E}$?

- For the flow derivation, I don't see much technical novelty compared to Gao et al 2024. This paper only considers translation of 3D gaussians, but not rotation.

Experiments
- The paper is only evaluated on synthetic data. How does it work (even qualitatively) on real datasets, such as the ones used in Gao et al 2024?

### Questions
- Does the datasets used in the paper contain videos of dynamic objects and scenes? If so, how does baselines that only handle static scenes, such as NeRF, 3DGS work? 
- If the scene is static, would dynamic gaussian flows become always zeros.

### Soundness
2

### Presentation
2

### Contribution
2
