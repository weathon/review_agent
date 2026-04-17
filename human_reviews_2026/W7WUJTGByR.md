# Flow Equivariant World Modeling for Partially Observed Dynamic Environments

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Embodied systems experience the world as 'a symphony of flows': a combination of many continuous streams of sensory input coupled to self-motion, interwoven with the motion of external objects. These streams obey smooth, time-parameterized symmetries, which combine through a precisely structured algebra; yet most neural network world models ignore this structure and instead repeatedly re-learn the same transformations from data. In this work, we introduce 'Flow Equivariant World Models', a framework in which both self-motion and external object motion are unified as one-parameter Lie group 'flows'. We leverage this unification to implement group equivariance with respect to these transformations, thereby sharing model weights over locations and motions, eliminating redundant re-learning, and providing a stable latent world representation over hundreds of timesteps. On both 2D and 3D partially observed world modeling benchmarks, we demonstrate Flow Equivariant World Models significantly outperform comparable state-of-the-art diffusion-based and memory-augmented world-modeling architectures, training faster and reaching lower error -- particularly when there are predictable world dynamics outside the agent's current field of view. We show that flow equivariance is particularly beneficial for long rollouts, generalizing far beyond the training horizon. By structuring world model representations with respect to internal and external motion, flow equivariance charts a scalable route to data-efficient, symmetry-guided, embodied intelligence.
Project page: https://flowm-anonymous.github.io/Flow-Equivariant-World-Models/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Flow Equivariant World Models (FloWM), a novel framework for modeling dynamic environments that are only partially observable from an embodied agent’s egocentric viewpoint. The core idea is to unify self-motion and external object motion under the mathematical framework of Lie group flows, treating both as time-parameterized symmetries. By enforcing flow equivariance, the model maintains a structured, group-equivariant latent memory that evolves consistently with both the agent’s actions and the dynamics of unseen objects. The authors also propose a practical and scalable Transformer-based implementation of FloWM, and the experiments on 2D and 3D world modeling tasks demonstrate the effectiveness of FloWM.

### Strengths
- This paper addresses the challenges of partially-observed environments and memory limitations in world modeling, which are critical and challenging issues that previous approaches struggle to handle effectively.
- FloWM leverages the concept of flow equivariance and introduces two novel techniques: velocity channels and self-motion equivariance. These techniques elegantly unify both internal and external object motions in world modeling.
- The theoretical framework is rigorous, providing solid evidence for the effectiveness of FloWM's foundation.
- The authors also present a practical implementation of FloWM based on Transformer architecture, demonstrating its impressive performance on both 2D and 3D world modeling benchmarks. Ablation studies further validate the contribution of each component.

### Weaknesses
- FloWM assumes that both agents and objects move at a constant velocity along fixed trajectories, a strong assumption that limits its applicability to more complex tasks where motion is more variable and dynamic.
- FloWM relies on maintaining latent maps during both training and inference, which makes it less suitable for open-ended scenarios where the environment is constantly evolving and expanding.
- FloWM focuses primarily on the movement of agents and objects, but does not account for interactions between the agents and objects, which is an important aspect in embodied AI.
- Although the paper claims to present an **embodied** world model, the experimental evaluation is confined to simulated games rather than physical robotics tasks.

### Questions
- How does FloWM model non-uniform motion of agents and objects, especially when their speeds are variable or unpredictable?
- How can FloWM be applied to open-ended tasks where the task map is constantly evolving or expanding over time?
- How does FloWM account for interactions between the agent and objects, which are crucial for embodied AI?
- What is FloWM's performance in more complex physical embodied tasks, such as robotic manipulation or real-world applications?

I will raise my score if these questions are addressed well.

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
The paper introduces Flow Equivariant World Models (FloWM) for partially observed environments, unifying self-motion and external object motion as one-parameter flows and enforcing equivariance in a latent map. The model maintains a group‑structured memory with velocity channels that co-move with known actions, yielding a stable representation over long horizons. Two instantiations for 2D and 3D scenarios show improved long‑range prediction and length generalization over diffusion baselines. The approach is data‑efficient but currently relies on discretized velocities and a 3D encoder that is only approximately equivariant.

### Strengths
S1. Using flows as a unified representation for self‑motion (agent motion) and object motion is interesting and well motivated.

S2. Experiments show strong long‑horizon consistency under partial observability; the model remains stable well beyond the training horizon.

### Weaknesses
W1. The problem setup is hard to follow. Please state clearly what is observed, what is predicted, what is latent, and what the training objective is.

W2. The modeling assumptions may narrow the scope of applicable tasks. My understanding (please correct me if mistaken) is that agent actions and “world” actions must be compatible as Lie‑algebra elements; this seems to require both to live in a compatible rigid‑motion group, which may exclude domains where agent actions are non‑rigid or semantic (e.g., crouching, smiling).

W3. All experiments use textureless data; it is unclear how well the approach handles textured or cluttered scenes such as CLEVR/MOVi‑style data.

### Questions
Q1. Could this framework be applied to ProcGen‑style games, and if so what adaptations would be needed?

Q2. For driving video generation tasks akin to Wayve's GAIA‑style datasets, what difficulties do you foresee (e.g., multi‑agent stochasticity, occlusions, fine textures, non‑rigid objects)?

Q3. In Fig. 3, is the top‑down latent map h_t provided by the data or inferred from egocentric images?

Q4. In Fig. 2, observations depict overlapping digits while internal flows are defined per digit. Do you first factorize observations with an object‑centric encoder (e.g., slots), or how are digits moved independently without explicit object segmentation?

Q5. How is stochasticity handled? If the world has inherent randomness (e.g., an object moves left or right with equal probability next frame), where does this uncertainty enter the model, and can the method represent multiple futures during prediction?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Summary of Contributions:
This paper introduces a new world model architecture, Flow Equivariant World Models (FloWM), designed for partially observable environments where both the agent and other objects are moving. The core idea is to unify these two types of motion using the principle of flow equivariance. The model maintains a recurrent latent map of the world. This map is explicitly transformed (e.g., shifted or rotated) in the opposite direction of the agent's actions, which provides a stable representation even as the agent moves. In parallel, the model uses a set of "velocity channels" to predict the movement of external objects in the environment, allowing it to track them even when they are outside the agent's limited field of view. The authors show this approach works in both 2D (moving MNIST) and 3D (moving blocks) toy environments.

### Strengths
1. The central idea of the paper is very intuitive and compelling. Instead of forcing a large model to re-learn world physics from scratch at every step, the architecture builds in a strong inductive bias for motion. Using an explicit transformation to handle the agent's own movement ("self-motion equivariance") is a clever way to maintain a stable world representation.

2. The experimental results on the provided benchmarks are very strong. The proposed FloWM method significantly outperforms modern baselines like a diffusion-forcing transformer (DFoT) and a state-space model (DFoT-SSM), especially on long-term rollouts (as seen in Figure 4). The baselines clearly fail to track objects that leave the agent's view, while FloWM succeeds.

3. The ablation studies are thorough and effectively demonstrate the value of each component. The model without self-motion equivariance (no SME) fails completely, proving this is the most critical contribution for this task. The model without velocity channels (no VC) also struggles with the dynamic objects, which justifies the full architecture.

4. The abstract and introduction sections are well-written and enjoyable to read. They motivate well injecting the motion-based inductive bias into world models.

### Weaknesses
1. The main weakness is the significant and unaddressed question of scalability. All experiments are conducted on simple "toy" datasets (MNIST digits, simple colored blocks). In these settings, the objects are visually simple and come from a small, finite set.
  It is not clear how this method would perform in a real-world scenario, such as autonomous driving or robotics, where the visual space is far more complex and objects are not easily categorized.

2. The 3D Block World experiment requires a ViT encoder to learn a complex projection from a 3D first-person view to a 2D top-down map. The authors note this is not analytically equivariant and must be learned. This projection problem seems like it would become exponentially harder with complex, realistic image data, and this bottleneck could potentially negate the benefits of the equivariant recurrent map.

### Questions
1. Could the authors comment on the scalability of this approach? The leap from simple, finite-set objects (digits, blocks) to the visual complexity of real-world scenes (e.g., in driving or robotics datasets) seems very large. What are the primary challenges anticipated?

2. Related to scalability: The 3D-to-2D encoder in the 3D experiment must learn the projection, which is a key potential bottleneck. How well does this projection need to be learned for the flow equivariance to be effective? Have the authors attempted any preliminary experiments on more complex data (e.g., a common dataset like ViZDoom or a driving dataset) to see if this encoder can be trained effectively in a more realistic setting?

3. The current model uses a discrete set of velocity channels (e.g., -2, -1, 0, 1, 2). How would this framework be extended to handle a continuous range of object velocities, as would be common in the real world?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Flow Equivariant World Models (FloWM) that treat both self-motion and external object motion as Lie-group flows. It derives a generalized flow-equivariant recurrence where dedicated tokens encode self motion and internal flows track external motion. Benchmarks are a synthetic 2D MNIST World and a 3D Dynamic Block-World evaluated up to multiple rollout steps. The authors show lower error than diffusion baselines in terms of generative quality metrics (MSE and SSIM) and better long-horizon generation.

### Strengths
1. The method provides a clear formalization of flow equivariance for both self and external motion in a theoretically grounded world modeling framework. Methods that can model latent actions/causes/dynamics that are external to agent's own action are welcome and needed in the era of world models.

2. The method targets partial observability with out-of-view dynamics, which is a gap in video models that can suffer from limited context windows.

### Weaknesses
1. Beyond MNIST-World and the 3D Block benchmarks which are designed by authors, there is no evaluation on external benchmarks.

2. Diffusion baselines sometimes underperform even vs trivial ablations and all-black baseline (table 1), which highlights my point above that the benchmarks might favor FloWM’s design. Table 1 and 2 shows diffusion is worse than “no SME, no VC” (i.e. a vanilla ConvRNN) on most metrics with similar pattern in Table 2, which questions the fairness of evaluation protocol and effectiveness of VC and SME.

3. The evaluation metrics are based on generative quality only. Since the method is in the domain of equivariant representation learning, such metrics are not a good proxy for decoding actions directly. For example, one evaluation can decode actions from latents corresponding to velocity channels.

4. Baselines are limited to diffusion models. Other world modeling paradigms (e.g. JEPAs [1] or masked video models [2]) are not compared with.


[1] V-jepa 2: Self-supervised video models enable understanding, prediction and planning.

[2] VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training

### Questions
1. What is your explanation for the fact that a simple ConvRNN (“no SME, no VC”) sometimes outperforms baselines in Tables 1 and 2?

2. Can you report compute comparisons (FLOPs and wall-clock time) to baselines?

### Soundness
2

### Presentation
2

### Contribution
2
