# What Matters in RL-Based Methods for Object-Goal Navigation? An Empirical Study and A Unified Framework

- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Object-Goal Navigation (ObjectNav) is a critical component toward deploying mobile robots in everyday, uncontrolled environments such as homes, schools, and workplaces. In this context, a robot must locate target objects in previously unseen environments using only its onboard perception. Success requires the integration of semantic understanding, spatial reasoning, and long-horizon planning, which is a combination that remains extremely challenging. While reinforcement learning (RL) has become the dominant paradigm, progress has spanned a wide range of design choices, yet the field still lacks a unifying analysis to determine which components truly drive performance. In this work, we conduct a large-scale empirical study of modular RL-based ObjectNav systems, decomposing them into three key components: perception, policy, and test-time enhancement. Through extensive controlled experiments, we isolate the contribution of each and uncover clear trends: perception quality and test-time strategies are decisive drivers of performance, whereas policy improvements with current methods yield only marginal gains. Building on these insights, we propose practical design guidelines and demonstrate an enhanced modular system that surpasses State-of-the-Art (SotA) methods by 6.6% on SPL and by a 2.7% success rate. We also introduce a human baseline under identical conditions, where experts achieve an average 98% success, underscoring the gap between RL agents and human-level navigation. Our study not only sets the SotA performance but also provides principled guidance for future ObjectNav development and evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents an empirical study to analyze the contributions of three different modules in RL-based Object-Goal Navigation systems: perception, policy and test-time enhancement. Among these, the authors find that perception and test-time enhancements are most crucial. Based on the findings, they propose design recommendations and develop a modular system that outperforms prior baselines.

### Strengths
The paper presents a diagnostic study on the contributions of different modules in RL-based ObjectNav systems, and provides useful insights that will benefit the community.

### Weaknesses
Overall, I find the study reported in the paper and the insights drawn to be useful for future research. However, I have some questions (listed below) and would request the authors to provide more clarification.

### Questions
1. Corner Goal Policy - The random exploration strategy seems similar to MOPA[1], but the citation is missing.
2. Frontier-Based Policy - 
    a) There could be different ways to sample the frontier goal from the list of frontiers, such as selecting the one nearest to or farthest from the agent, etc. Did you try these variants?
    b) Did you try semantic frontier selection like VLFM[2]?
    c) Referring to MOPA, they find that the performance is sensitive to the distance at which the frontier is sampled. Did you observe similar behavior?
3. Object detector - Why didn’t you try more recent detectors (Grounding DINO[3])?
4. L260-261 “global map is only used to update the local map and is not directly used for navigation” - can you elaborate what you mean by this? I believe the global map is used to explore the environment and to memorize past observations. The local map only gives a limited view of the area and will not be sufficient in long-horizon task planning.
5. Map augmentation - can you elaborate what you mean by augmentation and what different techniques you tried?
6. Which ObjectNav dataset did you use for your experiments?

[1] Raychaudhuri et al. MOPA: Modular Object Navigation with PointGoal Agents. 2024.
[2] Yokoyama et al. VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation. 2023.
[3] Liu et al. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. 2024.

### Soundness
3

### Presentation
2

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
The paper presents a systematic empirical study of ObjectNav systems focusing on RL-based modular approaches found in the existing literature. The paper studies individual contributions of various existing subsystems (perception, policy, and test-time enhancement) in controlled experiments, and presents detailed analyses and recommendations. The paper integrates these insights to develop an ObjectNav approach that outperforms existing approaches, and also highlights the current gap with human-level performance in these settings.

### Strengths
- The paper is well-written, with clearly defined goals and scope that is relevant to the community.
- The paper unifies the existing literature in the areas of modular RL-based ObjectNav approaches with clear and valuable insights into the individual modules, along with recommendations for researchers aiming to deploy or improve these systems.
- The experiments are comprehensive and well-designed.

### Weaknesses
- Many findings in the paper reinforce existing design choices well-known to the community. There are no novel paradigms explored in the paper, neither does the paper present any theoretical contributions. Yet, I believe the paper will be useful to the community as it grounds our intuitive design choices though empirical validation.
- The experiments on the choices of observation spaces, action spaces and network architectures with regards to the policy module feels limited, especially since it considers the policy as a black-box. There are policies in ObjectNav settings that use hierarchical or model-based approaches, which are not considered in the paper but are quite relevant and important to consider. Consequently, the claim “policy improvements with current methods yield only marginal gains” feels a little too strong in this regard.

### Questions
- With regards to my above concern about limited experiments with policy module, why were other policy learning approaches not considered? I think scoping the rationale behind this would be valuable to the readers.
- Minor comment: Please put citations inside parentheses when they are not a part of the sentence (see the use of `\citet` and `\citep` and use them appropriately).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Prologue: ObjectGoal task has a large flood of papers. It started with pre-trained models and over-fitting to dataset and environment to claim prowess over SOTA and gradually the shift has happened towards prowess of LLM/MLM used when zero shot approaches ourtperform earlier pre-trained models (RL wrt env. rewards or supervised wrt VLA).

The paper:
This is an empirical study (expanded ablation) for RL agents' performance in Sim for ObjNav task. This is an engineering paper more than a research paper, but with commendable hacks/tweaks. The authors divide ObjectGoal into 3 stages - Perception (scene graph / map), Policy (RL agent), and Test-time Enhancement (rule-based heuristics / hacks). The main finding is the first module, Perception being a decider in final success, although this is expected as initial errors add to latter as cascade.

Epilogue: Hard effort has been put in for this work, however, after a few days I am pretty sure a claimed new but old-tweaked method will come proving better results than this - that is the speed of research community in this sub-domain. Lack of codebase and theory makes it hard to verify claims which are just numbers. As such for ICLR community, this does not add any value wrt learning representations for Object Goal task, which is already not known. Appreciating the work, requesting authors to focus on representation part of Object Goal -- which can be better than how a MLM represents the scene perception internally.

### Strengths
1. Extensive experiments on RL SOTA on ObjectNav task.
2. In contrast to other papers, that omit critical workarounds, this paper gives detailed logical accounts of ways to engineer better performance.
3. Section A.5 list down a modified way to evaluate, which is practical; however the SOTA probably tried their best wrt current norm of 500 step counts.
4. Fig. 3 partially presents the Failure case analysis wrt what exactly caused the failure - something commendable -- however this is too environment specific to claim generalization comments and decisions in algo/RL reward design.

### Weaknesses
1. Modular aspects of Embodied AI Goal tasks are already covered by previous works -- the philosophy is already proven. Reference: Wu, Qiaoyun, et al. "Image-goal navigation in complex environments via modular learning." IEEE Robotics and Automation Letters 7.3 (2022): 6902-6909. And CSR that shows just by changing representation, better results can be fetched in downstream tasks -- Gadre, Samir Yitzhak, et al. "Continuous scene representations for embodied ai." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
2. The policies and architectures tested are straightforward with no discussion wrt their selection choice and failure case analysis.
3. There is lack of qualitative results to support the claims.

### Questions
1. In perception, why DINO + SAMg or OWLVT2 is not used which works pretty well for open vocabulary objects now in indoor scenes? MaskRCNN is outdated in current context.
2. Line 464 is not believable - in terms of human experts. Firstly, 5 experts do not add statistical significance. Secondly, human users with RGB-D ego view only (not 3rd part FPS type view) cannot navigate complex paths as smoothly (action space) as a pre-trained agent in similar environment. Requesting repeat of experiments in ego view only.
3. In table 3, there is sudden jump in success for applying heuristics for obstacle avoidance helper.
4. Line 941: How can agent come out if it is stuck at a corner and is rotating in a loop when dynamic goals swap by oscillation?
5. Instead of quite old Sem-Exp type map, probably two extreme ends could be tested - like simple semantic 3D voxel map and 3DLLM map [https://arxiv.org/pdf/2307.12981] to detect the prowess of map representation in downstream tasks and overhead.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper examines Object-Goal Navigation (ObjectNav), in which a robot must find target objects in previously unseen environments using only onboard perception. Noting the lack of a unified analysis of what drives performance, the authors conduct a large-scale empirical study of modular RL-based ObjectNav systems, decomposing them into three components: perception, policy, and test-time enhancement. Their findings show that perception quality and test-time strategies are the primary performance drivers, while policy improvements with current methods yield only marginal gains.

### Strengths
Investigating the contributions of different components in RL is very interesting.

The paper offers practical recommendations for improving ObjectNav performance. 

The inclusion of a human baseline is also valuable, clearly illustrating the performance gap between current RL-based systems and human-level navigation.

### Weaknesses
The paper uses Habitat as the testbed, but it appears that only a single dataset is employed for experiments. It would strengthen the paper to include evaluations on additional datasets to validate the generality of the findings.

More details should be provided about the three essential modules—perception, policy, and test-time enhancement—including how each is implemented, trained, and evaluated. The descriptions are currently high-level and could benefit from clearer examples or pseudo-code.

In the Introduction, Figure 1 is referenced conceptually but not explicitly mentioned in the text. It may be better to move Figure 1 into the Introduction or clearly cite it there, as it provides an overview of the framework that helps readers understand the study structure early on.

The Test-time Enhancement Module Design Choices section needs clarification. It is not entirely clear what “test-time enhancement” means—whether it refers to inference-time heuristics, post-processing strategies, or adaptive decision rules applied without retraining. Additionally, Figure 2 does not explicitly show the four failure mode categories discussed in the text: (a) trapping in narrow spaces, (b) object misidentification, (c) repeated exploration, and (d) map overlap across floors due to undetected staircases. The figure should be updated or annotated to illustrate these cases for clarity.

The paper analyses the modules within the RL framework but uses heuristic rules for navigation instead of a learned neural policy conditioned on the top-down semantic map. The motivation for this design choice should be explained

### Questions
What are the details of the Human Experts—for example, who are they, what is their background, and how do they operate during the evaluation?

### Soundness
4

### Presentation
3

### Contribution
3
