# QuaMo: Quaternion Motions for Vision-based 3D Human Kinematics Capture

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Vision-based 3D human motion capture from videos remains a challenge in computer vision.
Traditional 3D pose estimation approaches often ignore the temporal consistency between frames, causing implausible and jittery motion.
The emerging field of kinematics-based 3D motion capture addresses these issues by estimating the temporal transitioning between poses instead.
A major drawback in current kinematics approaches is their reliance on Euler angles.
Despite their simplicity, Euler angles suffer from discontinuity that leads to unstable motion reconstructions, especially in online settings where trajectory refinement is unavailable.
Contrarily, quaternions have no discontinuity and can produce continuous transitions between poses.
In this paper, we propose QuaMo, a novel Quaternion Motions method using quaternion differential equations (QDE) for human kinematics capture.
We utilize the state-space model, an effective system for describing real-time kinematics estimations, with quaternion state and the QDE describing quaternion velocity.
The corresponding angular acceleration are computed from a meta-PD controller with a novel acceleration enhancement that adaptively regulates the control signals as the human quickly change to new pose.
Unlike previous work, our QDE is solved under the quaternion geometric constraints that results in more accurate estimations.
Experimental results show that our novel formulation of the QDE with acceleration enhancement accurately estimates 3D human kinematics with no discontinuity and minimal implausible artifact.
QuaMo outperforms comparable state-of-the-art methods on multiple datasets, namely Human3.6M, Fit3D, SportsPose and a subset of AIST.
The code is available at https://github.com/cuongle1206/QuaMo

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes QuaMo, an online kinematics module that replaces Euler-angle dynamics with a quaternion differential equation (QDE) solved exactly on the unit sphere $S^3$, coupled with a meta-PD controller augmented by a second-order acceleration enhancement term. Given per-frame vision priors (e.g., TRACE, HMR2.0), QuaMo predicts angular velocities and integrates quaternions via the Hamilton product to reduce the motion jitter and produce temporally consistent motion from the monocular videos.

### Strengths
- A key strength of QuaMo is its fully online state-space formulation, which does not rely on future observations, enabling real-time refinement of off-the-shelf 3D pose estimators through iterative updates of angular velocity and quaternion states.
- Unlike integration approximation methods, which risk moving the estimated quaternion outside the unit sphere $S^3$ and therefore require normalization to mitigate this issue, this work uses the Hamilton product between the quaternion representation of the rotation matrix and the current $q_t$. This leads to more accurate estimations, since normalization in previous approaches turns the quaternion into a direction that does not correspond to the true rotation and distorts the trajectory.
- The proposed acceleration enhancement $\alpha$ is computed from the second-order quaternion difference of the last three reference poses. It boosts control only when the reference is changing quickly (i.e., during true fast movement) and dampens as the target is reached. This allows QuaMo to treat fast, intended motion and jitter differently.
- For evaluation, the work reports both local and global metrics (MPJPE, P-MPJPE, Accel, global MPJPE/GRE, global jitter, and foot-skating) and provides error bars across random seeds. Evaluating the proposed QuaMo using TRACE and HMR2.0 as baselines, the results show that incorporating QuaMo significantly reduces acceleration errors. While the improvements in joint-based errors for HMR2.0 in camera coordinates are marginal, the method substantially decreases errors in the world coordinate system.

### Weaknesses
- As shown in table 3, although the proposed acceleration term $\alpha$ enhances the joint-based errors, it increases the acceleration error. It would be useful to clarify when to disable/attenuate the term to prevent the rise of acceleration error.
- Although the addition of Euler integration for root translation shows an improvement in GRE, it is a first-order numerical integration method, and I suspect that small errors could accumulate over time. A figure or analysis of translation error versus time would be useful.
-  There are minor ambiguities in the writing. First, $f_\omega$ is introduced as a function of $q_t$ and $\omega_t$. Then, at line~240, the term $b_t$ is introduced as an approximation of $f_\omega$, but $b_t$ is the output of the ControlNet, which receives additional inputs.
- Minor writing issues:
  - line 240 misses close parenthesis
  - line 236 has redundant comma after imaginary
  - line 290 mentions $\kappa_I$ of Eq. 5 that does not exist.
  - In table 1 and 2, Foot skating is mentioned as FS while in table 3 FK is used.

### Questions
- Given that the input $\hat{q}_t$ is noisy and the acceleration enhancement term uses a second-order difference, it can inherently amplify high-frequency noise. How is noise propagation controlled? Is it handled by ControlNet by assigning an appropriate $\kappa_A$ depending on the level of noise present in $\hat{q}_t$?
- Can the QuaMo method be applied to per-frame approaches that estimate MANO parameters, such as HaMeR [1]? If so, what modifications would be required to adapt the quaternion-based kinematic formulation to hand-specific articulations?

[1] Pavlakos, Georgios, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, and Jitendra Malik. "Reconstructing hands in 3d with transformers." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9826-9836. 2024.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents QuaMo, an online 3D human kinematics capture framework from monocular video. The method replaces Euler angles with a quaternion differential equation (QDE) under a unit-sphere constraint to avoid discontinuity and gimbal lock. It further combines a meta-PD controller with a second-order acceleration term to better handle fast motion changes. Experiments on four benchmarks (Human3.6M, Fit3D, SportsPose, AIST) indicate improved accuracy and smoother motion over several online kinematics baselines, with ablation results supporting each component.

### Strengths
Clear articulation of the discontinuity/gimbal lock problem and why Euler angles are problematic in online capture.

Correct quaternion-based formulation with integration respecting the unit-sphere constraint.

Adaptive acceleration term adds responsiveness to the PD controller, helping in fast motion regimes.

Solid empirical evaluation across multiple datasets, and thorough ablation on rotation representations and pipeline components.

### Weaknesses
Novelty: Quaternions for rotation representation are standard in robotics, graphics, and physics simulation. The specific QDE formulation and its integration here are sound, but not a fundamentally new concept.

No cost analysis: The paper doesn’t compare training/inference speed, memory usage, or computational overhead with Euler/axis-angle setups. The practical trade-off is unclear.

Input dependency: Performance varies greatly depending on upstream reference pose quality (TRACE vs HMR2.0). There is no robustness study under noisy or degraded inputs.

Evaluation scope: Benchmarks target relatively clean, single-person motions. Multi-person, occlusion-heavy, or contact-rich scenarios are not tested.

### Questions
1. What is the training/inference time cost relative to Euler or axis-angle versions? Does the quaternion formulation require higher compute/latency?

2. How does the method perform when reference poses are distorted or noisy? Any filtering or noise-robust adaptation tested?

3. Could the approach be extended to longer-horizon online settings (multiple future poses) without sacrificing latency?

4. Would the method benefit from incorporating environment contacts (as mentioned in future work) into the controller?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper targets the challenging problem of 3D human motion capture based on visual input. To address the problem of implausible and jittery motion due to the representation of Euler angles, the paper propose to use Quaternion motions instead and an effective system based on QDE  is utilized. The experiments on several benchmarks like Human3.6M, Fit3D, SportsPose and AIST validate the effectiveness of the proposed algorithm.

### Strengths
* The paper targets the challenging problem of 3D human motion capture based on visual input, which is of great importance to the industry applications. 

* The idea of using quaternion differential equations for human kinematics capture is intersting. 

* It reports reasonable results on several benchmarks like Human3.6M, Fit3D, SportsPose and AIST.

### Weaknesses
* It should include the recent references published in the recent two years. Currently, there is no reference published in 2025.

* For the experiments, there are several releated works which are not compared. For example, [R1] is referenced in the paper but not compared in Table 1. By checking the results report in [R1], it would have obviously better results compared with the prposed algorithm. Please involve more papers published in the recent two years (2024-2025) for comparisons. 

[R1] Jihua Peng, Yanghong Zhou, and PY Mok. Ktpformer: Kinematics and trajectory prior knowledge-
enhanced transformer for 3d human pose estimation. In CVPR, pp. 1123–1132, 2024

* In Section 4.2, there are several implementation details reported in the paper. As there are many hyper-parameters defined in the paper, is there possible to provide more ablations on the setting of these hyper-parameters?

### Questions
Please address the questions raised in the weakness section. More specifically, please provide more comparisons in Table 1 to validate the effectiveness of the proposed algorithm against the state-of-the-art algorithms.

### Soundness
2

### Presentation
3

### Contribution
2
