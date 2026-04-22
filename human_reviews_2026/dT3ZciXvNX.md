# DexMove: Learning Tactile-Guided Non-Prehensile Manipulation with Dexterous Hands

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Non-prehensile manipulation offers a robust alternative to traditional pick-and-place methods for object repositioning. However, learning such skills with dexterous, multi-fingered hands remains largely unexplored, leaving their potential for stable and efficient manipulation underutilized. Progress has been limited by the lack of large-scale, contact-aware non-prehensile datasets for dexterous hands and the absence of wrist–finger control policies. To bridge these gaps, we present DexMove, a tactile-guided non-prehensile manipulation framework for dexterous hands. DexMove combines a scalable simulation pipeline that generates physically plausible wrist–finger trajectories with a wearable device, which captures multi-finger contact data from human demonstrations using vision-based tactile sensors. Using these data, we train a flow-based policy that enables real-time, synergistic wrist–finger control for robust non-prehensile manipulation of diverse tabletop objects. In real-world experiments, DexMove successfully manipulated six objects of varying shapes and materials, achieving a 77.8\% success rate. Our method outperforms ablated baselines by 36.6\% and improves efficiency by nearly 300\%. Furthermore, the learned policy generalizes to language-conditioned, long-horizon tasks such as object sorting and desktop tidying.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors introduce DexMove, a framework for tactile-guided, non-prehensile dexterous manipulation. The core problem addressed is the lack of large-scale datasets and coordinated wrist-finger control policies for such tasks. The authors propose a hybrid data acquisition pipeline that combines a simulation engine for generating wrist-finger trajectories with a wearable device for capturing real-world tactile force data from human demonstrations. This data is used to train a flow-matching model to establish initial contact, a network called TaFo-Net to predict future contact forces, and the main DexMove-Policy, which generates goal-conditioned wrist-finger motions. DexMove surpasses a set of baselines across six objects and demonstrates generalization to language-conditioned long-horizon tasks in the real-world experiments.

### Strengths
- The hybrid data acquisition strategy is clever and effective. The hardware design is thoughtful, and the scale of the generated datasets is substantial.
- The experimental evaluation is comprehensive and rigorous. The quantitative comparison against learning-based baselines (CORN and DyWA) is strong. The supplementary video demo is very helpful and provides qualitative examples
- The demonstration of downstream applications, including structured sorting and language-driven collaboration, showcases the practical utility and downstream usages.

### Weaknesses
- The presentation is a bit hard to chew on. There are a few consistency and clarity issues throughout Sections 3 and 4. For example, $d^{\mathrm{tip}}$ in Eq. 2 is undefined. In Sec. 3.2, the author defines $V \in \mathbb{R}^{v \times 4}$, but in Sec. 4.3, the historical tactile data is denoted as $V_{-T_p : 0} \in \mathbb{R}^{T_p F \times vC}$. This reuse of the same symbol with different dimensions could be confusing. $P^{\mathrm{hand}}$ and $A^{\mathrm{hand}}$ seem to be redundant as one can always derive $P^{\mathrm{hand}}$ with FK.
- The overall framework is composed of three separate neural networks (Figure 4). Although it is validated through system-level evaluation, the authors do not provide a clear justification for this modular decomposition over a more integrated architecture. Moreover, the DexMove-Policy is conditioned on a planned force schedule $G_{1:T_f}$ predicted by TaFo-Net. This could lead to compounding errors, yet no evaluation of the system's sensitivity is reported. The choice of flow matching is also presented without justification over the cited diffusion policy. A brief discussion on why flow matching is preferred would be greatly appreciated.
- A disentanglement of the data contributions would further enhance the paper's quality. An experiment showing performance when training only on simulation data (e.g., with heuristic forces) is a critical missing baseline to validate this hypothesis. The claim that the isomorphic sensor design "minimizes the domain gap" is also not validated.

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents DexMove, a novel data-driven framework for learning tactile-guided non-prehensile manipulation using dexterous hands. It proposes a hybrid data collection pipeline that combines large-scale simulation of wrist–finger trajectories with real-world human demonstrations collected via a wearable tactile device. Based on the data collection pipeline, it further introduces a three-part policy structure that is able to establish contact, predict future tactile force, and generate tactile-informed trajectories to handle a set of non-prehensile manipulation tasks. Experiments show that DexMove achieves higher performance than existing baselines, and generalizes to long-horizon, language-conditioned tasks.

### Strengths
This paper is well motivated. The proposed hybrid data pipeline that fuses large-scale simulation with human tactile demonstrations shows potential to provide high-fidelity datasets for the community. Real-world experiment results demonstrate DexMove's superiority over single-contact and ablated baselines, showing its generation ability towards generalizing to long-horizon, language-conditioned tasks.

### Weaknesses
1. The proposed method relies on strong assumptions: it employs three calibrated depth cameras and visual markers to track object pose and obtain visual observations.

2. The paper lacks a detailed discussion of common failure cases. Understanding how and why the method fails (e.g., specific object geometries, friction conditions, or loss of contact) is critical for the audience.

3. The demonstration of language-conditioned tasks like "tidying" is presented as a qualitative strength, but it lacks quantitative evaluation. It is unclear how robust this capability is, what the success rate is for these more complex tasks, and how much the language instruction actually guides a complex rearrangement versus simply triggering a pre-learned pushing skill.

### Questions
Please refer to the Weaknesses part.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a tactile-guided framework for non-prehensile manipulation with dexterous hands. The authors propose a complete pipeline that combines large-scale simulation-generated wrist–finger trajectories with human tactile demos to learn stable contact and motion strategies. The system integrates three core components: ContactFlow Matching for establishing robust contact using flow matching; DexMove-Policy, a goal- and force-conditioned flow-based policy that generates motion trajectories; and TaFo-Net, a tactile force planner modeling inter-finger coordination over time. DexMove enables the robot to manipulate various objects through pushing, rolling, and sliding without grasping, achieving high success rates and efficiency in real-world tests.

### Strengths
1. The paper tackles the underexplored problem of tactile-guided non-prehensile manipulation using dexterous hands, a setting rarely addressed in prior work that mostly focuses on grasping or gripper-based pushing.  The combination of large-scale simulation-based wrist–finger trajectory generation with human tactile demonstrations is both conceptually novel and practically valuable, enabling scalable yet physically grounded learning.
2. The technical design is solid and coherent: three modules are integrated through a consistent flow-matching formulation, supported by clear ablations and real-world evaluations.
3. The paper presents its ideas with clear structure and illustrative figures, making complex mechanisms understandable.

### Weaknesses
1. Limited Evaluation Scope and Generalization Evidence – Although real-world experiments demonstrate promising success rates, the object set remains small and relatively homogeneous (rigid, medium-sized items). The absence of tests on soft, slippery, or deformable objects limits claims of generalization.
2. The experimental evaluation mainly covers lateral pushing, sliding, and simple rotational tasks, but lacks richer contact interaction types such as pressing, squeezing, or compliant surface exploration, which are essential to demonstrate the full potential of tactile-guided dexterous control. Expanding the benchmark to include pressing or deformable-object manipulation would provide stronger evidence of versatility.
3. Several parts of the paper, especially the method section, are densely written without clear intuitive explanation. The logical flow among the three modules (ContactFlow, DexMove-Policy, and TaFo-Net) could be clearer. A cleaner, layered schematic emphasizing data flow and module dependency would make the overall system architecture much easier to understand. Some parts of the article could be polished.

### Questions
1. How sensitive is the learned policy to tactile sensor calibration or noise? Since the method heavily relies on tactile feedback, it would be useful to know how performance changes under slight calibration errors or sensor drift over time Some sensitivity analysis is recommended.
2. Could the proposed approach be extended to handle dynamic or moving objects? I'm just curious about whether DexMove can adapt its tactile-guided policy to scenarios where the object itself is in motion (e.g., sliding or being perturbed), and what modifications would be required to handle such cases.
3. Plz see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes DexMove, a tactile-guided non-prehensile manipulation framework for dexterous hands. It first generates force-aware wrist-finger trajectories in simulation, and then collects real tactile data via a glove-like exoskeleton system, where a human wears the same tactile fingertips as the robot. Using the simulated trajectories, the authors train a force-conditioned imitation learning policy. Then, using real-world trajectories, the system learns to predict the desired forces given a target object pose and uses these predictions to guide the imitation policy. The method achieves robust planar pushing across diverse tabletop objects and generalizes to higher-level, language-conditioned tasks such as sorting and tidying. Experiments show a 77.8% success rate, outperforming gripper-based baselines and ablations by a large margin.

### Strengths
Comprehensive system design: The paper demonstrates an impressive integration of simulation, tactile sensing, human demonstration, and policy learning. The wearable tactile exoskeleton represents substantial engineering effort.

Novel hybrid data pipeline: The combination of simulated trajectories and tactile demonstrations effectively mitigates the lack of large-scale real-world tactile data and the sim-to-real gap.

Compelling demonstrations: Real-world experiments, including language-guided and long-horizon tasks, highlight its potential as a general dexterous manipulation system.

Connection to VLMs: Integration with vision-language models for goal specification expands the scope of tactile manipulation toward 
multimodal reasoning and autonomy.

### Weaknesses
Task simplicity: Despite the complex hand control, the main tasks remain planar pushing on a tabletop. The method’s advantage over simpler mechanisms (like parallel-jaw pushers) is somewhat limited by the task design.

Pose tracking with markers: Object pose estimation relies on markers, reducing realism and preventing deployment in unstructured environments.

Limited wrist motion: The dataset penalizes wrist rotations beyond ~90°, which may restrict generalization to more complex manipulations.

Evaluation diversity: The test objects and surfaces, while varied, do not yet cover dynamic or multi-object interactions; thus, generalization remains partially demonstrated.

System scalability: The wearable system and tactile glove require manual calibration and mounting. The scalability and robustness of data collection remain uncertain.

### Questions
1. How many different policies are trained in total? Specifically, are the contact-establishment and DexMove-Policy networks trained separately or jointly?

2. Are the contact and trajectory policies both trained entirely in simulation?

3. How robust is perception to occlusion, given the reliance on three depth cameras? Does failure of one camera degrade policy performance?

4. In Table 3, what exactly does “Wrist-only*” denote? Does that mean teleoperation-based wrist control with locked fingers (as opposed to a policy-based wrist-only controller)?

5. Have the authors tested beyond planar pushing, such as rolling, sliding on uneven surfaces, or multi-object rearrangement?

6. More importantly, comments on the sim-to-real gap: Is TaFo-Net essential for bridging this gap? If so, why? Is it because force information is critical, or simply because real-world tactile data are needed to fine-tune the network?

### Soundness
3

### Presentation
3

### Contribution
3
