# Endowing GPT-4 with a Humanoid Body: Building the Bridge Between Off-the-Shelf VLMs and the Physical World

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
In this paper, we explore how to empower general-purpose Vision-Language Models (VLMs) to control humanoid agents. General-purpose VLMs (e.g., GPT-4) exhibit strong open-world generalization, and remove the need for additional fine-tuning data. To build such an agent, two key components are required: (1) an embodied instruction compiler, which enables the VLM to observe the scene and translate high-level user instructions into low-level control parameters; and (2) a motion executor, which generates human motions from these parameters while adapting to real-time physical feedback.
We present BiBo, a VLM-driven humanoid agent composed of an embodied instruction compiler and a diffusion-based motion executor. The compiler interprets user instructions in context with the environment, and leverages a chain of visual question answering (VQA) to guide the VLM in specifying control parameters (e.g., motion captions, locations). The diffusion executor extends future joint trajectories from prior motion, conditioned on both control parameters and environmental feedback.
Experiments demonstrate that BiBo achieves an interaction task success rate of 90.2\% in open environments, and improves the precision of text-guided motion execution by 16.3\% over prior methods. BiBo handles not only basic interaction but also diverse motions, and even dancing while striking at a sandbag. The code will be released upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the BiBo framework, which aims to leverage off-the-shelf Vision-Language Models (VLMs) to control humanoid agents for interaction in open physical environments. The paper designs an embodied instruction compiler and a diffusion model-based motion executor to realize the translation from high-level instructions to low-level actions, and verifies their effectiveness on multiple tasks.

### Strengths
See Summary.

### Weaknesses
1.The core innovations of this study are "reducing data dependency via off-the-shelf VLMs" and "solving motion continuity issues using LDM (Latent Diffusion Model) + VAE (Variational Autoencoder)". However, the generalization ability in real physical environments has not been fully verified. BiBo’s interaction capability has not been validated in real-world scenarios (non-InfiniGen-generated simulated environments), making it impossible to prove its effectiveness under complex real conditions such as lighting changes and irregular objects.
2.The dataset used in this study has limitations, and its generalization is questionable. Experiments are only conducted on the HumanML3D dataset and InfiniGen-generated scenes, without validation on other public datasets (e.g., BABEL, AMASS). This makes it difficult to demonstrate the model’s generalization ability across a wider range of action types and scenes.
3.The evaluation metric system is incomplete. Although FID and R-Precision are used to evaluate motion quality, quantitative analysis of key dimensions in interaction tasks—such as "interaction naturalness" and "task coherence"—is omitted. It is recommended to supplement quantitative analysis of metrics like Interaction Accuracy and Task Coherence.
4.There is a lack of real-time performance analysis. Although the paper claims that BiBo supports real-time control (>20 Hz), it does not provide analysis of specific memory usage or GPU memory consumption. Performance benchmarks on typical hardware platforms should be provided to verify its deployment feasibility.
5.There is no systematic analysis of the method’s limitations. The paper does not discuss BiBo’s performance in extreme scenarios (e.g., occlusion, lighting changes) nor analyze the impact of VLM reasoning errors on the entire system. Such discussions should be supplemented in the main text or appendix.
6.In the ablation study (Table 2), control groups for "without Pose Reasoning (w/o Pose Reasoning)" and "without Joint Generation (w/o Joint Generation)" need to be added to quantify the impact of each stage on task success rates (e.g., positioning accuracy for sitting tasks, joint control accuracy for touching tasks) and clarify the necessity of the three-stage process.
7.It is recommended to supplement failure case analysis in the main text or appendix, including the category of failed tasks, root causes (e.g., pose recognition errors, insufficient collision handling), and potential improvement directions, to enhance the transparency of the research.

### Questions
See Weakness.

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
5

### Summary
This paper introduces BiBo (Building humanoId agent By Off-the-shelf VLMs), a novel framework that leverages off-the-shelf Vision-Language Models (VLMs, such as GPT-4) to control humanoid agents. The core idea is to reduce data collection and training costs by combining pre-trained VLMs with a tailored embodied system for humanoid control. BiBo consists of two main components:

- Embodied Instruction Compiler: Converts high-level natural language commands into low-level structured motor commands by reasoning over scene context.
- Diffusion-based Motion Executor: Generates smooth, human-like motion trajectories while dynamically adapting to environmental feedback using a combination of Latent Diffusion Models (LDMs) and Inverse Kinematics (IK) optimization.
The authors highlight BiBo's ability to perform diverse and complex physical interactions in dynamic environments, achieving a task success rate of 90.2% and improving motion execution precision by 16.3% compared to prior methods.

### Strengths
- VLM Agent Workflow for Complex Task Understanding

The Embodied Instruction Compiler is well-designed, using a structured three-step reasoning process (attribute analysis, pose reasoning, and joint generation) to translate high-level commands into low-level motor instructions. This design allows BiBo to accurately interpret user intent and adapt to complex tasks in dynamic physical environments, such as sitting, lifting objects, or interacting with multiple scene elements. The use of voting mechanisms and multi-view representations further improves the system's robustness in understanding intricate tasks.

- Novel Integration of CLoSD + IK for Diffusion Motion Updates

The combination of CLoSD (a physics-based motion-tracking policy) and Inverse Kinematics for refining humanoid motion is inspiring. The framework dynamically corrects motion trajectories by incorporating physical feedback from the environment, ensuring smooth and continuous motion even in challenging scenarios (e.g., collisions, external forces). This joint optimization approach enhances both the adaptability and precision of motion generation, particularly for tasks requiring fine-grained control (e.g., grasping or touching objects). The use of Latent Diffusion Models (LDMs) further enables the generation of high-fidelity motions while maintaining computational efficiency. This part could be considered the most inspiring in this paper.

### Weaknesses
- Unclear Execution of Motion with CLoSD for Dynamic Objects

The paper lacks clarity on how the generated motion trajectories are passed to CLoSD for execution. For instance, when an object moves unpredictably, does the system rely on CLoSD alone for tracking, or does it dynamically update the motion plan using feedback? What if the dynamic object encounters collision with hands? While the authors mention incorporating physical feedback into motion updates, the explanation of how BiBo handles motion retargeting or re-planning in the presence of dynamic objects is insufficient. This aspect deserves more detailed discussion and evaluation.


- Limited Discussion and Citation of Related Work

The paper does not sufficiently discuss its approach to prior work in key areas, such as LLM planning, long-term task completion, or the use of diffusion models in HSI motion generation. For instance:

[1] SIMS: Simulating Stylized Human-Scene Interactions with Retrieval-Augmented Script Generation. ICCV2025

[2] Synthesizing Physically Plausible Human Motions in 3D Scenes. 3DV2024

[3] Generating Human Interaction Motions in Scenes with Text Control. ECCV2024

It is clear that TESMO[3] is exactly the type of previous approach this paper aims to compare against: it introduces discontinuity by conditioning on past generated rather than executed results.
At the very least, providing sufficient discussion and citations would improve my impression of this paper.

- Not complicated enough environments as claimed

Most environments in demo videos feature a single piece of furniture on flat ground, not challenging enough as claimed.

### Questions
Please see the weakness

### Soundness
4

### Presentation
4

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
This paper proposes BiBo (Building humanoId agent By Off-the-shelf VLMs) — a framework that connects general-purpose Vision-Language Models (VLMs) like GPT-4 to humanoid control. The key idea is to use powerful pre-trained multimodal models to bypass costly humanoid-specific data collection and training. BiBo has two major components: 1. Embodied Instruction Compiler – translates high-level natural language instructions (e.g., “have a rest”) into structured, low-level control commands (e.g., sitting location, facing direction, joint targets) through a three-stage visual Q&A process. 2. Diffusion-based Motion Executor – a latent diffusion model (LDM) that generates continuous, physically-plausible humanoid motion conditioned on those commands and on real-time physical feedback.

### Strengths
1.	The idea of directly plugging an off-the-shelf VLM (GPT-4o) into a humanoid control pipeline is innovative. Avoids re-training large models by adding a lightweight compiler layer.
2.	The compiler–assembler analogy is clear and intuitive: the VLM acts like a “compiler” converting high-level language into structured commands, while the motion diffusion module serves as an “assembler” for physical actuation.
3.	The Latent Diffusion Model with joint decoding of executed and generated latents ensures temporal continuity and environmental awareness — addressing a major weakness in previous motion diffusion frameworks.
4.	The experiment is also comprehensive

### Weaknesses
1.	The motions shown in the video do not fully comply with physical laws. During interactions with objects, there are visible cases of hovering and penetration, which make it appear that the human keypoints are rule-based attached to the objects rather than physically constrained. The interactivity seems weaker compared with methods such as UniHSI.
2.	The motion generation in the video appears to heavily depend on the VLM’s outputs. However, the VLM tends to exhibit strong hallucination problems during grounding. It is unclear how the authors constrain the frame-to-frame consistency — both in the diffusion process and within the Embodied Instruction Compiler.
3.	During locomotion, the agent sometimes floats or teleports. Yet, in Table 3, BiBo is reported to outperform others on the skating and floating metrics. Providing a more detailed definition and evaluation standard for these metrics would make the results more convincing.

### Questions
See Weakness

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper targets the interesing problem of humanoid agents which can handle flexible and diverse interactions in open environments. To avoid collecting massive dataset to train the model, the paper presents a new solution by utilizing the capability of the strong VLMs and a diffusion-based motion generator. The former is used to generate the primitive commands based on the user instruction and the latter is used to generate the corresponding motions. Experiments show the promising results of the proposed algorithm.

### Strengths
* The idea of utlizing the strong capability of VLM to decompose the primitive commands and later handled by a motion generator is interesting. 

* The experiments show the proposed algorihtm obtains promising results in the challenging problem of interaction with the open environments. 

* The paper is well presented and the proposed algorithm should be easy to reproduce.

### Weaknesses
* The paper relies on two components to handle the target problem. On one hand, currently, even the SOTA vlms may not be able to produce the precise primitive actions. To simply the problem, the paper presents a set of predefined actions but it still cannot guarantee a robust results. On the other hand, assume vlms can produce the accurate action motions, how to obtain a good motion is not a trivial task. It should provide more justification that why the presented motion executor can produce the desired results.

* The experimental results in Table 1 is evaluated based on a setting proposed by this paper. The task as well as the setups are introduced in this paper. How about the generation of the proposed algorithm to other benchmarks? 

* For the results in Table 2, the result of "Lift" is much lower than other categories. What are the potential reasons for this? 

* Currently, BiBo are operated in the virtual setting. Is it possible to proivde some evaluations which can show that the proposed algorithm can generate to real-cases like robots?

### Questions
Please address the questions in the weakness section. More specifically, please mainly address the questions related work the experiments.

### Soundness
3

### Presentation
3

### Contribution
3
