# Rodrigues Network for Learning Robot Actions

- Avg Score: 6.00
- Decision: Accept (Oral)
- Scores: 2, 6, 8, 8

## Abstract
Understanding and predicting articulated actions is important in robot learning. However, common architectures such as MLPs and Transformers lack inductive biases that reflect the underlying kinematic structure of articulated systems. To this end, we propose the **Neural Rodrigues Operator**, a learnable generalization of the classical forward kinematics operation, designed to inject kinematics-aware inductive bias into neural computation. Building on this operator, we design the **Rodrigues Network (RodriNet)**, a novel neural architecture specialized for processing actions. We evaluate the expressivity of our network on two synthetic tasks on kinematic and motion prediction, showing significant improvements compared to standard backbones. We further demonstrate its effectiveness in two realistic applications: (i) imitation learning on robotic benchmarks with the Diffusion Policy, and (ii) single-image 3D hand reconstruction. Our results suggest that integrating structured kinematic priors into the network architecture improves action learning in various domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Rodrigues network (rodrinet), a novel graph neural network architecture that encodes the kinematics-aware inductive bias. The authors evaluate the proposed architecture on forward/inverse kinematic fitting, robot manipulation, and hand reconstruction tasks, which show that rodrinet outperforms or is on par with existing popular architectures.

### Strengths
1. The overall presentation is very clear and straightforward, give enough background to understand the proposed method. 
2. The forward/inverse kinematic fitting experiments clearly demonstrate that the proposed architecture captures the inductive bias of articulated chains. Before that, I hesitated on this because GNN message passing is very different from sequential forward kinematic computation. 
3. The Rodriguez equation is an interesting perspective, and the authors conduct experiments over a diverse range of tasks. I very appreciate the authors' efforts.

### Weaknesses
My major concerns (questions) are two below, which also link to the core weaknesses of this paper in my opinion.

1. Why robot-kinematic-inductive bias matters? 
2. Does the performance gain from RodriNet justify the cost of an unorthodox design?

[Q1]: Most of the current robotic applications use task space control (i.e., predicting the end-effector pose), aka the eef pose, which can be easily translated to joint torques with an op-space controller. In this case, the model does not need to consider the robot's own kinematic structure and can leave that to controller. 

In this paper's experiment results, the most compelling results come from the forward/inverse kinematics fitting, which, however, has faster and flexible analytical solutions. On other tasks, the benefits are very marginal. Like in the robot manipulation experiments, even when Rodrinet outperforms the diffusion policy on PickCube and StackCube, I am not entirely convinced because these two tasks should not heavily depend on the robot kinematic chain. In the hand reconstruction task, the authors explicitly mention that modifications are made to RodriNet to suit MANO's configuration space (for example, fitting the hand shape beta parameters, which I assume are not tied to joint/link). This even make the results look less convincing. 


[Q2] Even if robot-kinematic-inductive bias may not matter in every task, having them for free does not hurt. However, this proposed architecture, which looks like another GNN variant, has limitations and cannot be used for free. 

For example, all robot manipulation experiments are conducted in state space, rather than using visual inputs. This is a critical constraint. Additionally, using a GNN instead of a general transformer could make it challenging to interpolate with other transformer-based backbones. Using joint space and not task space actions can also make the prediction less visually grounded. In sum, these constraints outweigh the marginal performance gain.

A side note: joint controllers are widely used in dexterous hands; maybe this technique can find its place in hand manipulation tasks, instead of parallel grippers.

### Questions
(I list all questions in the weakness section)

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
3

### Summary
This paper introduces Rodrigues Network (RodriNet), a neural architecture that explicitly embeds articulated kinematic structure into its design. The key component is the Neural Rodrigues Operator, derived from Rodrigues’ rotation formula by separating “state-dependent” (sin θ, cos θ) terms from “structure-dependent” coefficients and making the latter learnable. This transforms parent–child relations in a kinematic tree into a learnable message-passing rule. A multi-channel variant updates per-link 4×4 features from per-joint inputs using both left and right multiplications. Stacking these operators forms Rodrigues Layers, complemented by a Joint Layer and a Self-Attention Layer for global interaction, with an optional global token handling shared variables such as base pose or gripper state.
Empirically, RodriNet (1) achieves superior fitting accuracy on forward kinematics for the LEAP Hand and improved Cartesian-trajectory prediction on the UR5 arm, (2) boosts simulated success rates in Diffusion Policy across five ManiSkill tasks, and (3) improves single-image 3D hand reconstruction on FreiHAND by replacing the transformer head in HaMeR, yielding small but consistent SOTA gains with significantly fewer parameters.

### Strengths
- Principled inductive bias: The formulation of a learnable operator directly from Rodrigues’ rotation rule is mathematically clean and physically grounded. By explicitly modeling the basis (1, sin θ, cos θ), RodriNet captures rotational structure while preserving the kinematic tree topology.

- Strong expressivity for kinematic mapping: On forward-kinematics fitting, RodriNet (Rodrigues Layers only) achieves orders-of-magnitude lower MSE and faster convergence than MLP, GCN, Transformer, or BoT baselines, demonstrating that the inductive bias aligns well with articulated geometry.

- Effective bridging of Cartesian and joint spaces: The UR5 trajectory prediction task is a well-controlled test of inverse kinematics. RodriNet achieves lower train/test MSE and maintains accuracy under reduced data regimes, showing improved data efficiency.

- Practical gains in imitation learning: Integrated into Diffusion Policy, RodriNet improves average success rate (0.61 vs. 0.58 for UNet-DP and 0.44 for Transformer-DP) across five ManiSkill tasks, with the largest gains on PickCube and StackCube. The global token cleanly handles gripper control.

- Cross-domain transfer: When replacing the transformer head in HaMeR, RodriNet attains consistent SOTA on FreiHAND with far fewer parameters (10.7 M vs. 39.5 M), indicating that the inductive bias generalizes beyond robotics.

- Thorough analysis and solid engineering: Ablations identify the Rodrigues Layer as the main contributor. Performance is stable under architectural scaling, and the custom CUDA implementation yields ~6× speedup in large configurations.

### Weaknesses
- Lack of equivariance guarantees: The learnable operator applies unconstrained 4×4 left/right multiplications, offering no guarantee of SE(3) consistency or frame equivariance. Layer composition may not preserve physically valid transformations. The method serves as an inductive bias rather than a structured constraint; adding formal regularization (e.g., orthogonality or SE(3)-aware constraints) could strengthen the theoretical foundation.

- Limited novelty relative to existing basis encodings: The operator performs a learned linear combination over {1, sin θ, cos θ} (and quadratic quaternion terms for MANO), similar in spirit to Fourier or harmonic encodings. The absence of comparisons to SE(3)-equivariant architectures (e.g., SE(3)-Transformer, EGNN) or differentiable FK layers leaves it unclear whether gains arise from kinematic-tree structure or from providing the correct trigonometric basis.

- Restricted embodiment generalization: Kernels are defined per joint within a fixed kinematic tree, and experiments are conducted per robot (LEAP Hand, UR5, Franka). There is no evaluation of cross-morphology transfer, shared parameters across repeated structures, or conditioning on structural metadata for multi-embodiment generalization.

- No real-world validation: Results in imitation learning are limited to simulation. For contact-rich tasks such as PegInsertionSide or PlugCharger, performance gains are minimal, suggesting limited robustness to real sensor or actuation noise.

- Compute tradeoffs insufficiently analyzed: Training RodriNet for motion prediction requires longer runtime (2 h 22 m vs. ~1 h 20 m for Transformer/BoT), yet compute–accuracy tradeoffs are not fully discussed.

- Scope limitations: The model handles only rotational joints and omits link geometry or prismatic motion. Its focus on imitation learning rather than closed-loop control narrows applicability to contact-rich or mobile-manipulation domains.

### Questions
- Equivariance and structure validity: Does RodriNet satisfy any formal invariance or equivariance guarantees under changes of the base frame or re-rooting of the kinematic tree? Could initializing weights from analytical FK coefficients and applying soft orthogonality constraints improve SE(3) consistency while maintaining flexibility?

- Comparison to SE(3)-aware models: How does RodriNet perform against SE(3)-equivariant architectures (e.g., SE(3)-Transformer, E3NN) or MLP/Transformer baselines with Fourier angle encodings and identical kinematic-graph wiring? This would clarify whether the Neural Rodrigues Operator contributes beyond periodic encodings and structural locality.

- Cross-morphology generalization: Can the per-joint kernels be shared across joints of the same type or conditioned on metadata (axis, link transforms) to enable zero-shot transfer to unseen kinematic trees? A small multi-embodiment study would help test generality.

- Feature interpretation: Do the learned 4×4 “link features” converge toward SE(3)-like structures in practice? Visualizing the 3×3 rotation blocks (e.g., via polar decomposition) could reveal whether the network implicitly maintains orthogonality.

- Compute tradeoffs: RodriNet requires longer training time in the UR5 task. With the CUDA kernel acceleration, what is the effective wall-clock speedup compared to a Transformer at matched accuracy? Are there memory or scaling limits for large CL/CJ?

- Simulation-to-real transfer: In tasks where simulated gains are limited (e.g., PegInsertionSide, PlugCharger), how does RodriNet handle tactile or force-feedback inputs? Any early evidence of transfer performance on hardware would strengthen the empirical case.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors investigate incorporating structural inductive biases from kinematic models in neural network design. They propose the Rodriguez layer, which combines projecting joint features through the learned Rodriguez operator and aggregating features globally with self-attention. Experimentally, the authors show that for problems where agents have a rigid body structure (e.g., forward kinematics or solving manipulation tasks with imitation learning), their model design performs substantially better than neural architectures that lack these inductive biases.

### Strengths
> Incorporating learnable kinematic models in robotics makes intuitive sense and seems to provide measurable improvements for applying machine learning on rigid body structures

> The paper is well written and presents the proposed model clearly. The diagrams help summarize the approach. 

> The experiments show compelling results to validate the efficacy of the Rodriguez layer for rigid body agents. The inclusion of both robots and animated characters helps demonstrate the model's potential across several settings of embodied agents.

### Weaknesses
> Some aspects of the work are a bit obvious. The results for forward kinematics are not surprising, particularly as the Rodriguez matrix is one of two major approaches to modelling rigid-body kinematics—the other being Denavit-Hartenberg parameters. It could strengthen the paper to show how directly predicting the kinematic model's parameters compares in this problem. 

> Some results could benefit from statistical hypothesis testing to confirm whether the observed benefits are statistically significant (e.g. the success rates in Table 2). 

> Something that might be helpful would be to include results for failure cases. Presumably, there are data modalities or scenarios in which assuming a kinematic relation negatively impacts performance. 

Writing Opinions:

Line 063 - 068:  This paragraph can be cut with almost no loss of meaningful content to the paper. 

Related Work: Consider looking at cross-embodiment works as potential related work. These models also consider agent structure as necessary to model explicitly. 

[1] Gupta, Agrim, et al. "Metamorph: Learning universal controllers with transformers." arXiv preprint arXiv:2203.11931 (2022).

[2] Xiong, Zheng, Jacob Beck, and Shimon Whiteson. "Universal morphology control via contextual modulation." International Conference on Machine Learning. PMLR, 2023.

> Figure 3: We suggest replacing "backbone" with "architecture" unless these models are pretrained, which, in our experience, is where this term is used more frequently. 
 
>line 353: replace backbone with "architecture" 

> Figure 5: Provide more information in the caption on what is shown in the diagram.

### Questions
Q1: What's the reason for not comparing it to the Denavit–Hartenberg formulation of kinematic structure? A quick search reveals that the Rodrigues formulation does have some technical advantages, but this seems the more appropriate comparison. 

Q2: What kind of features are provided to the Rodriguez layers? Do the transformations process non-kinematic related features in some way? How would these layers work with non-kinematic associated features, for instance? 

Q3: What about the author's current work that limits their representation to only spherical joints as opposed to translational joints? 

Q4: What distinguishes the Rodriguez layer from running system identification to find the model's kinematics?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces RodriNet, a neural architecture that embeds articulated-kinematics structure directly into its computation by learning a generalized version of Rodrigues’ rotation formula. Instead of treating joint values as flat vectors like MLPs or Transformers, the proposed Neural Rodrigues Operator replaces fixed trigonometric coefficients in classical forward-kinematics equations with learnable weights, enabling structured message passing along a robot's kinematic tree. RodriNet stacks these operators with joint-update and attention layers, yielding a model that naturally learns hierarchical motion patterns. Experiments show large gains in fitting forward kinematics, predicting 3D joint trajectories from Cartesian motions, improving imitation-learning performance when used as the backbone in Diffusion Policy, and achieving state-of-the-art results on 3D hand pose estimation, all with fewer parameters. Overall, the work argues for architectural priors tailored to robot embodiment as a path to more efficient and generalizable action learning.

### Strengths
- Introduces a learnable generalization of Rodrigues rotation to embed articulated-kinematics structure into neural networks, giving a principled inductive bias for articulated action learning

- Presents a clear architecture (Rodrigues Layer, Joint Layer, self-attention) that mixes local kinematic structure with global information exchange

- Demonstrates strong improvements across domains: forward kinematics, Cartesian-to-joint motion prediction, robotic imitation learning, and 3D hand reconstruction

- Shows faster convergence and better data efficiency than standard architectures like MLPs, GCNs, and Transformers in modeling articulated motion

### Weaknesses
- Requires per-joint learnable parameters and tree-structured computation, which may introduce higher architectural complexity compared to simpler universal backbones

- No ablation studying how much each component (Rodrigues layer vs joint layer vs attention) contributes, leaving uncertainty about which parts drive performance

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
3
