# Zero-shot Human Pose Estimation using Diffusion-based Inverse solvers

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 8

## Abstract
Pose estimation refers to tracking a human's full body posture, including their head, torso, arms, and legs.
The problem is challenging in practical settings where the number of body sensors is limited.
Past work has shown promising results using conditional diffusion models, where the pose prediction is conditioned on both <location, rotation> measurements from the sensors. 
Unfortunately, nearly all these approaches generalize poorly across users, primarily because location measurements are highly influenced by the body size of the user.
In this paper, we formulate pose estimation as an inverse problem and design an algorithm capable of zero-shot generalization.
Our idea utilizes a pre-trained diffusion model and conditions it on rotational measurements alone; the priors from this model are then guided by a likelihood term, derived from the measured locations. 
Thus, given any user, our proposed InPose method generatively estimates the highly likely sequence of poses that best explains the sparse on-body measurements.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes InPose, a diffusion-based inverse solver for human pose estimation from sparse sensors. The model assumes access to rotations and positions of only three joints (head and two wrists). It decomposes poses into scale-free and scale-dependent components and uses a pre-trained diffusion to restore full-body poses with sparse observations. During inference, it adopts ideas from IIGDM to incorporate positional likelihood, allowing zero-shot adaptation to new body scales without fine-tuning.

### Strengths
1. The paper introduces a diffusion-based inverse inference framework for human pose estimation, which is conceptually original. It demonstrates how diffusion models for forward generation can be adapted to approximate inverse inference by adding an explicit likelihood-guidance term during sampling. 
2. The decomposition of human poses into scale-free and scale-dependent components is theoretically motivated. 
3. Within the paper’s stated assumptions, the derivation is internally consistent. Although these assumptions are physically unrealistic, the formulation itself is mathematically correct.

### Weaknesses
1. The entire formulation assumes that the three sensors directly provide joint rotations r_m. In practice, such rotations cannot be measured without knowing both adjacent body segments’ orientations. This input variable is thus non-observable, and the model seems to operate on dataset-level ground-truth artifacts rather than physically measurable signals. 
2. According to Fig.1, the conditioning joints (head and wrists) are leaves in the kinematic tree. While leaf-joint rotations may be defined as parameters in datasets, they are not constraining in the forward-kinematic chain (i.e., not required to determine downstream positions). Conditioning p(r_M|r_m) on such non-constraining variables makes the setup physically non-closed: the learned dependence is statistical rather than geometrically consistent, weakening the realism and deployability of the approach. 
3. The paper argues that zero-shot generalization is needed because users have different body scales, yet it simultaneously assumes that the new user’s bone lengths are known at inference time. If scale information is already available, a simpler and physically grounded approach would train a scale-free diffusion prior and condition on the user’s scale-dependent skeleton (e.g. scaled T-pose) together with observable head/wrists measurements (if available and physically meaningful). Such a design would achieve the same finetune-free behavior without introducing complicated conditioning mechanisms.

### Questions
1. Could the authors clarify how r_m could be physically obtained from only three sensors? Are these intended as device orientations or joint rotations, or just dataset artifacts?

### Soundness
1

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
4

### Summary
This paper proposes a new diffusion-based human pose estimation method for sensor data. The core motivation is that human skeletons have diverse limb lengths, so a conditional diffusion model fails to be trained on that diverse data. To bypass this problem, the paper proposes to train the diffusion model only on rotational data, which is independent of limb lengths. To realize this idea, the conditional diffusion formulation is separated into two parts, i.e., rotational distribution and locational distribution. The rotational distribution is absorbed in a classifier-free guidance (CFG) formulation and trained on data, while the location part is simply formulated based on geometry. To handle the location part in reverse steps, the pseudo-inverse guidance technique is utilized. This technique requires the transform to be linear, which is not satisfied by the geometric property of the rotations, so the result of the technique is approximated by a Gaussian distribution. Experiments show that the proposed method is much better generalized for diverse data.

### Strengths
- Good design. I'm impressed by the separated CFG design. This cleverly bypasses the difficulty of limb-length differences.

- Good generalization performance.

### Weaknesses
- Some crude components/approximations: The approximation for the $\Pi$GDM is somewhat crude. (i) It requires a strong assumption. The assumption in Theorem 1 requires the 6 DoF representation to be already close to a Stiefel matrix. However, this may not be true for a large part of the diffusion process. (ii) Theorem 1 is somewhat misleading. The proof only shows that the mean and the covariance of the distribution will correspond to the derived expressions. This does not imply that the resulting distribution will be close enough to a Gaussian one. It can be a widely different distribution, which still has the same mean and covariance.

- Better generalization, but worse in the best-case scenario? In Figure 3, InPose is worse than existing methods when the scale is one. The paper says that this is "expected," but why is this? I cannot find a valid reason for the proposed formation being worse with the default scale. A more thorough discussion is needed here.

### Questions
Please see the above weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an approach for estimating the 3D human pose from sparse body-mounted sensor measurements (e.g., using three sensors that track position and orientation). The method employs a diffusion-based framework, with the key contribution being the advantage of accommodating different body shapes, unlike prior methods that assume a fixed body shape. The approach is evaluated on the AMASS dataset and compared against previous work.

### Strengths
- The proposed method achieves improved robustness compared to previous work, specifically BoDiffusion and AvatarJLM.

### Weaknesses
- The evaluation is limited to synthesized data (for noise and different scales).
- The experiment using Gaussian noise is interesting, but this is not necessarily representative of the type of noise that these sensors tend to have.
- The experiments varying the body size introduce some arbitrary scaling of the upper body or the arms/legs. Is an 1.4x scaling for the torso while doing a 0.7x scaling for the arms realistic? Why not sample real motions and real bodies from AMASS for these experiments?
- I understand that the baselines are trained with a specific body shape, but how would they perform if their training data is augmented with arbitrary scaling factors for the different body parts. Is scale robustness something that can be achieved with augmentations?
- Although the setting is different, there has been recent work that operates with arbitrary body shape parameters (EgoAllo, Yi et al, CVPR 2025). How does that compare to the proposed work?
- Minor: The paper refers to the representation of Zhou et al as 6DoF. My understanding is that this representation still has three degrees of freedom, they just regress 6 values, so they refer to it as 6D.

### Questions
As I describe in the weaknesses, I would be interested in seeing:
- experiments with more realistic scale variations (e.g., with body shapes from AMASS).
- a comparison (conceptual, and potentially experimental) with the EgoAllo design.
- a version of the baselines where they see data with different scales during training (i.e., augmentations on the body shape).

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces InPose, a diffusion-based method for human pose estimation from sparse sensor data (3 sensors: head and two wrists). The key innovation is formulating pose estimation as an inverse problem, enabling zero-shot generalization across users with different body sizes—without requiring model fine-tuning for each user. InPose leverages a pre-trained diffusion model conditioned on rotational measurements, using location measurements only as a likelihood-based guidance term during inference. The method is evaluated on the AMASS dataset and compared to state-of-the-art baselines, showing strong generalization and robustness to body size and measurement noise.

### Strengths
+ Novel Inverse Problem Formulation: The decomposition of human pose into scale-free (rotational) and scale-dependent (location) components is elegant and addresses the generalization issue in prior work.
+ Zero-Shot Generalization: InPose can handle unseen body sizes and shapes without retraining, a significant practical advantage.
+ Robustness: The method is robust to measurement noise, as shown in experiments with noisy sensor data.
+ The paper provides mathematical justification for the Gaussian approximation in the likelihood term and details the propagation of uncertainty through non-linear operators.
+ Evaluations cover generalization to new body sizes, robustness to noise, and ablation studies on representation choices (6DoF vs. rotation matrices).

### Weaknesses
- Performance on Default Body Size: For users matching the training body size, baseline methods outperform InPose.
- Complexity: The method involves non-trivial mathematical machinery (e.g., modified ΠGDM, covariance propagation), which may hinder adoption. Please include complexity analysis.
- The proposed method may be hard to be used in online real-time applications, where most of the real usages cases in VR and AR require.
- The proposed method builds on top of several previous methods, reused and simplified some formulations. Please clearly define the novelty over these previous methods.

### Questions
Please address the concerns in the weakness session.

### Soundness
3

### Presentation
3

### Contribution
3
