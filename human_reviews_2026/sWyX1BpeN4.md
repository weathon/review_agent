# Manipulation as in Simulation: Enabling Accurate Geometry Perception in Robots

- Decision: Accept (Poster)
- Scores: 4, 6, 10, 6

## Abstract
Modern robotic manipulation primarily relies on visual observations in a 2D color space for skill learning but suffers from poor generalization. In contrast, humans, living in a 3D world, depend more on physical properties-such as distance, size, and shape-than on texture when interacting with objects. Since such 3D geometric information can be acquired from widely available depth cameras, it appears feasible to endow robots with similar perceptual capabilities. Our pilot study found that using depth cameras for manipulation is challenging, primarily due to their limited accuracy and susceptibility to various types of noise. In this work, we propose Camera Depth Models (CDMs) as a simple plugin on daily-use depth cameras, which take RGB images and raw depth signals as input and output denoised, accurate metric depth. To achieve this, we develop a neural data engine that generates high-quality paired data from simulation by modeling a depth camera's noise pattern. Our results show that CDMs achieve nearly simulation-level accuracy in depth prediction, effectively bridging the sim-to-real gap for manipulation tasks. Notably, our experiments demonstrate, for the first time, that a policy trained on raw simulated depth, without the need for adding noise or real-world fine-tuning, generalizes seamlessly to real-world robots on two challenging long-horizon tasks involving articulated, reflective, and slender objects, with little to no performance degradation. We hope our findings will inspire future research in utilizing simulation data and 3D information in general robot policies. We release the dataset, models for various depth cameras, along with an easy-to-use guide for sim-to-real transfer at https://manipulation-as-in-simulation.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenge of transferring robotic manipulation policies from simulation to the real world, identifying the noise and inaccuracies of real-world depth sensors as a primary obstacle.The authors propose "Camera Depth Models (CDMs)," a plug-in module designed to take raw RGB and noisy depth signals from a specific camera and output a clean, accurate, "simulation-like" metric depth map.

### Strengths
1. The paper tackles a highly relevant and practical problem in robotics: the sensor-reality gap that plagues sim-to-real transfer. The proposed approach of explicitly de-noising the real-world perception to match the clean simulation environment, rather than the other way around, is a clear and interesting contribution.
2. The introduction of the ByteCameraDepth dataset, which captures data from 7 different depth cameras across 10 modes and multiple scenes, is a valuable contribution to the community for studying and modeling depth sensor noise

### Weaknesses
1. **Missing Critical Baseline:** The paper's primary motivation is that depth noise breaks sim-to-real transfer. The proposed solution is to clean the real-world depth (Sim-Clean -> Real-Clean). However, the paper fails to compare this against the most obvious and widely-used baseline: making the policy robust to noise via domain randomization (i.e., Sim-Noisy -> Real-Noisy). It is common practice to train policies on simulated data with added noise patterns precisely to make them robust to real-world sensor imperfections. The paper only compares its depth *model* (CDM) against other depth *models* (e.g., PromptDA) but does not compare its *manipulation pipeline* against this standard sim-to-real baseline. This is a significant omission that weakens the central claim.
2. **Questionable Premise:** Related to the first point, the premise that policies *cannot* be trained on noisy data is debatable. While the paper shows that a policy trained on *clean* data fails on *noisy* data (Table 2, "None" model), this is expected. It does not prove that a policy trained on noisy data would also fail. Many existing works successfully train policies on real-world, noisy data (for imitation learning) or on noise-augmented simulation data (for sim-to-real), as the policy learns to filter or ignore the noise implicitly. The paper overstates the necessity of perfectly clean depth for manipulation.
3.  **Lack of Robustness Analysis:** For a paper focused on real-world manipulation, there is no analysis of the policy's robustness. The experiments are run in a controlled setting. There are no ablations involving physical perturbations, dynamic clutter, or other disturbances during policy execution. It is possible that the policy's success is brittle and highly dependent on the (now clean) geometric information, but lacks the robustness that might be learned from training on more varied (i.e., noisy) data.

### Questions
1. The core of the paper focuses on the CDM models. The actual manipulation policy, which is critical to the paper's claims, is barely discussed in the main text. Section 4.3 briefly mentions "imitation learning" , but the specific architecture (a ResNet encoder with a diffusion head) is relegated to Appendix D. This makes the manipulation aspect of the paper feel like an afterthought and difficult to evaluate properly.
2. The paper's title and abstract frame it as a manipulation paper, but the experimental validation for manipulation is limited. The zero-shot evaluation rests on two tasks (Kitchen and Canteen) presented in one table (Table 2). While the tasks are well-chosen, this is a small number from which to conclude that this method "enables generalizable and effective robotic manipulation"[L 880].

### Soundness
2

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
This paper introduces Camera Depth Models (CDMs), a neural network plugin designed to improve the accuracy of depth readings from consumer-grade depth cameras (like RealSense). The authors develop a large multi-camera depth dataset (ByteCameraDepth) and a novel data engine to train the CDMs by modeling camera-specific noise in simulation. By generating accurate, "simulation-like" depth data in the real world, the CDMs effectively bridge the sim-to-real gap for geometric perception. The main contribution is demonstrating that a depth-only manipulation policy, trained purely on clean simulated depth data (without real-world fine-tuning or adding noise), can transfer seamlessly to real robots and achieve high success rates on challenging long-horizon tasks.

### Strengths
1. The paper is well-structured and easy to follow. The problem is clearly defined, and the figures (especially Figure 1 and 2) effectively illustrate the proposed architecture and the core contribution of bridging the sim-to-real geometry gap.
2. This work provides a generalized, plug-and-play solution that allows researchers to fully leverage clean simulation depth without the current necessity of laborious real-world noise modeling or fine-tuning, dramatically lowering the barrier for generalizable, geometry-based robot policies.

### Weaknesses
1. In Section 3.4, subsection "Value noise model", the notation should be $N_{\text{value}}$ and $\hat{D}\_{\text{value}}$, instead of $N_{\text{hole}}$ and $\hat{D}\_{\text{hole}}$.
2. In robotic manipulation, color information is also crucial, as in practical applications, we rarely rely solely on depth as input. I'm interested in experiments demonstrating how camera depth models benefit robot manipulation tasks when RGB information is included (i.e., using RGB-D data).

### Questions
1. CDM Robustness on Novel Scenes/Objects: The sim-to-real tasks test policies on randomized positions (Fig. 3), but the objects themselves seem consistent across the real and sim environments. How does the camera depth models perform when deployed on objects or scenes drastically unseen during its noise model training?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper proposes to enhance the depth perception ability of different real-world depth sensor. This paper introduces the Camera Depth Models(CDMs) as a simple plugin on daily-use depth cameras, which is trained with high-quality paired data from simulation. Concretely, to generate paired depth image,  this paper first collects large-scale real-world depth-rgb images, and utilize these images to pretrain the hole noise model and value noise model to predict the low-quality depth image. With the hole noise model and value noise model, this method synthesizes camera depth for open-source simulation depth image to get paired simulation-real depth images to train the CDMs. The results prove the effectiveness of this method for depth-based manipulation policy sim-to-real adaptation.

### Strengths
1. This paper proposes a large-scale real-world RGB-Depth image pairs captured by seven depth cameras, which is a valuable depth-based dataset for depth prediction in the real world.

2. This paper provides a good solution to generate paired simulation-real depth images. This model enables a simulation policy to be transferred directly to a real robot for downstream tasks, achieving a zero-shot sim-to-real transfer, by simply using the CDM to clean the real camera's noise.

3. This paper provides sufficient experiments to prove that this model could achieve nearly simulation-level accuracy in depth prediction, making it possible to transfer a long-horizon simulation policy into the real-world manipulation task.

### Weaknesses
1. This article provides extensive visualization results and sim-to-real outcomes to validate the model's effectiveness, but it lacks some quantitative experimental results. For example, it lacks quantitative experiments on the predictive performance of the two pre-trained noise models.

### Questions
To what degree do these two noise models generate real-world-camera style depth images? For example, in the case of a transparent water cup, what effect does each of these models have?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose Camera Depth Models (CDMs): Lightweight neural plug-ins for existing depth cameras that convert raw RGB + noisy depth input into denoised, metric-accurate depth — effectively making real-world perception as accurate as simulation.

Dataset contribution: it proposes ByteCameraDepth, a real-world multi-camera depth dataset comprising >170,000 RGB-depth pairs.

Compared with baseline methods, this pipeline presents better performance on benchmarks.

There are also some limitation and concerns regarding the generalization scope and model capability.

### Strengths
1. CDMs act as post-camera modules, requiring no changes to robot control code or simulators. Fast (< 0.15 s latency) and generalizable across camera types.
2. By producing metric-accurate depth aligned with simulation, CDMs remove the need for domain randomization and reduce reliance on real fine-tuning, unlocking large-scale sim-data reuse.
3. Large real-camera dataset + multi-sensor evaluation. Demonstrates not only numeric improvement (RMSE ↓ 40 %) but task-level transfer (0 % → 90 % success).

### Weaknesses
1. Generalization Scope, There are multiple aspects for generalization, for example, scene generalization, robot generalization and so on. The model is trained on some collected dataset on specific robot and scenes, can it generalize to other scenarios? Please provide more analysis about zero-shot generalization capability.

2. The previous method d3roma and FoundationStereo has different settings but also valuable to compare, can you compare the generalization capability of the proposed method and these two? especially on cases like transparent and specular area.

3. Some template issues, like Fig.5 in supp.

### Questions
Figure 5 in supp materials has typesetting issues.

### Soundness
3

### Presentation
3

### Contribution
3
