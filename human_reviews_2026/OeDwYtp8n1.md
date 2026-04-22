# Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Recent advances in video diffusion models shows promise for generating robotic decision-making data, with trajectory conditions further enabling fine-grained control. However, existing methods primarily focus on individual object motion and struggle to capture multi-object interaction crucial in complex manipulation. This limitation arises from entangled features in overlapping regions, leading to degraded visual fidelity. To address this, we present RoboMaster, a novel framework that models inter-object dynamics via a collaborative trajectory formulation. Unlike prior methods that decompose objects, our core is to decompose the interaction process into three sub-stages: pre-interaction, interaction, and post-interaction, and models each phase using the dominant object, specifically the robotic arm in the pre- and post-interaction phases and the manipulated object during interaction. This design effectively alleviates the multi-object feature fusion issue in prior work. To further ensure subject semantic consistency across the video, we incorporate appearance- and shape-aware latent representations for objects. Extensive experiments on the challenging Bridge dataset, as well as RLBench and SIMPLER benchmarks, demonstrate that our method establishs new state-of-the-art performance in trajectory-controlled video generation for robotic manipulation. Project Page: https://fuxiao0719.github.io/projects/robomaster/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Summary :

RoboMaster is a trajectory‑conditioned video diffusion framework for robotic manipulation that jointly controls the robot arm and the object with a single collaborative trajectory. By decomposing each task into pre‑interaction, interaction, and post‑interaction phases and using mask‑based, appearance‑ and shape‑aware embeddings, it avoids the feature entanglement seen when controlling objects separately. It achieves state‑of‑the‑art visual fidelity and trajectory accuracy and improves downstream action planning.


Contributions:

Collaborative trajectory control for interaction: A unified trajectory that models pre-interaction → interaction → post-interaction phases, switching the dominant controller (arm → object → arm) to capture inter-object dynamics and avoid feature fusion during contact.

Appearance and shape aware subject embeddings: Mask-based tokens drawn from the initial frame’s VAE latents, expanded into shape-aware circular volumes along the trajectory to preserve object identity across frames.

Causal latent propagation: Frame-to-frame latent carryover (with overwrite at trajectory points) for smoother, temporally consistent motion during generation.

### Strengths
1. Novel collaborative trajectory design

Introduces a new way to model robot–object interactions using a single collaborative trajectory split into pre-interaction, interaction, and post-interaction phases.

Avoids the feature entanglement issues (e.g., missing or distorted objects) that plague prior methods like Tora and DragAnything.

2. High visual and physical realism

Produces smoother, more physically plausible manipulation videos with consistent object identities across frames.

Quantitatively achieves better FVD, PSNR, and SSIM, and lower trajectory errors on the Bridge benchmark.

3. Generalization and robustness

Handles diverse manipulation skills and in-the-wild scenarios.

Robust to imperfect user input—works with coarse or partial object masks and noisy trajectories.

### Weaknesses
1. Restricted to 2D pixel space

The system does not yet model depth or 3D geometry; this limits physical accuracy and makes 3D control (e.g., precise grasping) difficult.

2. Possible failure on out-of-domain inputs

Can produce incomplete or distorted objects when encountering unseen categories or backgrounds.

Still relies on training data diversity to generalize effectively.

3. Semantic dependency on user input

Relies on accurate prompts and roughly correct masks. Misleading text or poor masks may still degrade quality.

### Questions
Question 1: How does the model determine the precise transition points between these phases in practice, especially when the temporal boundaries of “interaction” are ambiguous or vary across tasks?


Question 2: How well would RoboMaster generalize to unseen robot morphologies or entirely different kinematic structures, and what adaptations (e.g., in trajectory representation or latent space) would be needed for that?

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
3

### Summary
This paper proposes RoboMaster to model interactions between a robotic arm and objects, dividing the interaction process into three stages: before and after the interaction it mainly controls the arm’s motion, while during the interaction it controls the object’s motion. The effectiveness of the proposed method is validated through visual results and simulation.

### Strengths
1. The paper argues that interaction should originate from multiple entities, including the arm and the object—this is a novel viewpoint.

2. RoboMaster exhibits impressive OOD generalization.

### Weaknesses
1. The motivation for decoupling the control signals is unclear; it is not explained how 2D trajectories help the robot learn, and the paper does not discuss the overall design in detail.

2. Section 4.5 is too brief, making it difficult to verify the method’s effectiveness for the robot; visual quality is not the core of the research—the core is whether the designed method can effectively aid robot learning.

### Questions
Could you explain in detail how the proposed model is applied to robot learning, and on that basis clarify the motivation for decomposing the action/control signals? At present, the stated motivation seems to be driven by generating better visual effects.

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
4

### Summary
The method tackles the task of trajectory-conditioned robotic manipulation video synthesis through a novel demonstration trajectory decomposition scheme that separates trajectories into a pre-interaction, interaction and post-interaction phase. This novel decomposition helps alleviate feature confusion issues observed in previous works. The method is evaluated against several baselines on the tasks of video synthesis and downstream robotic manipulation (through inverse dynamics), and manages to outperform them.

### Strengths
The paper is well-written and easy to understand. The method outperforms the baselines against which it is compared. The design choices are sensibly ablated. The work contributes a dataset of 21.000 human-annotated 2D robot manipulator trajectories. The work includes an honest discussion of its limitations.

### Weaknesses
The proposed method operates purely in image space: the generated trajectories require postprocessing by an inverse kinematics model and are not guaranteed to be realistic or executable.
Unlike its baselines, the method requires a segmentation of the provided trajectory into multiple stages by the user.
The manual masking of the interacted object could be replaced by an automatic grounding and segmentation.
A purely 2D trajectory input is very limiting, yet this is somewhat alleviated by the ability to describe the desired trajectory more specifically through a textual input.

### Questions
Can the method handle multi-step interactions, such as grasping a sponge, rotating it on a plate several times, then moving the robotic manipulator away?

### Soundness
4

### Presentation
4

### Contribution
2
