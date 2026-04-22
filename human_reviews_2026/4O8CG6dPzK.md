# Hierarchical Instruction-aware Embodied Visual Tracking

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
User-centric embodied visual tracking (UC-EVT) requires embodied agents to follow dynamic, natural language instructions specifying not only which target to track, but also how to track—including distance, angle, and directional constraints. This dual requirement for robust language understanding and low-latency control poses significant challenges, as current approaches using end-to-end RL, VLM/VLA, and LLM-based methods fail to adequately balance comprehension with low-latency tracking. In this paper, we introduce \textbf{Hierarchical Instruction-aware Embodied Visual Tracking (HIEVT)}, which decomposes the problem into on-demand instruction understanding with spatial goal generation (high-level) and asynchronous continuous goal-conditioned control execution (low-level). HIEVT employs an \textit{LLM-based Semantic-Spatial Goal Aligner} to parse diverse human instructions into spatial goals that directly specify desired target positioning, coupled with an \textit{RL-based Adaptive Goal-Aligned Policy} that enables real-time target positioning according to generated spatial goals. We establish a comprehensive UC-EVT benchmark using over 1.7 million training trajectories, evaluating performance across one seen environment and nine challenging unseen environments. Extensive experiments and real-world deployments demonstrate HIEVT's superior robustness, generalizability, and long-horizon tracking capabilities across diverse environments, varying target dynamics, and complex instruction combinations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces HIEVT (Hierarchical Instruction-aware Embodied Visual Tracking), a framework designed to address the core challenge of User-Centric Embodied Visual Tracking (UC-EVT): enabling embodied agents to follow dynamic natural language instructions (specifying both which target to track and how to track it—e.g., distance, angle, direction) while balancing robust language understanding and low-latency control. Existing methods (end-to-end RL, VLA/VLM models, LLMs) fail to strike this balance: RL lacks flexibility for complex instructions, VLAs/VLMs suffer from inference latency, and LLMs cannot support real-time tracking.
HIEVT resolves this via a two-tier hierarchical design that decouples high-level instruction parsing from low-level control: LLM-based Semantic-Spatial Goal Aligner (SSGA) and RL-based Adaptive Goal-Aligned Policy (AGAP). Experiments validate HIEVT across 10 virtual environments (1 seen: FlexibleRoom; 9 unseen: Suburb, Supermarket, etc.) and real-world deployment on a RoboMaster EP robot.

### Strengths
- Existing RL-based trackers (e.g., Zhong et al. 2024, ECCV’24) train on fixed goals (e.g., center-frame tracking) and require retuning for new instructions. VLA models (e.g., TrackVLA, Wang et al. 2025) achieve flexible instruction following but operate at 8 FPS—too slow for dynamic targets. HIEVT’s asynchronous decoupling of SSGA (instruction parsing) and AGAP (control) enables 50 FPS tracking while handling complex instructions, a capability no prior work achieves.
- Unlike VLA/VLM models (e.g., PALME, Driess et al. 2023) that require massive annotated data, HIEVT uses offline RL trained on 1.7M trajectories (collected via noise-perturbed PID controllers) and generalizes to 9 unseen environments (SR: 0.58–0.93). This outperforms baselines like Ensembled RL (SR: 0.22–0.42 in unseen environments) and TrackVLA (SR: 0.16–0.80), which degrade sharply in novel scenes. 
- Methods like GPT-4o generate direct actions from instructions, leading to opaque, error-prone decisions. HIEVT’s spatial goal (bounding box) acts as an interpretable bridge between language and action, enabling debugging and alignment with user intent. Ablations confirm this: replacing spatial goals with CLIP embeddings or vectors reduces AR by 63%–83% (Table 5). 
- Unlike simulation-only works (e.g., Zhong et al. 2023), HIEVT is deployed on a real RoboMaster EP robot with a closed-loop control system (camera → laptop processing → robot actuation). This validates its utility for real-world applications (e.g., search-and-rescue, elderly care), a gap in most embodied tracking research.

### Weaknesses
- HIEVT depends heavily on third-party components, limiting its end-to-end utility:
  - Mask Generation: Relies on SAM-Track (Cheng et al. 2023b) for real-world segmentation and Sapiens (Khirodkar et al. 2024) for virtual environments. Failure of these tools (e.g., SAM-Track in low light) directly breaks HIEVT’s pipeline.
  - LLM Dependence: SSGA’s performance is tied to proprietary LLMs (GPT-4o outperforms Gemma3-1B by 49% in AR; Table 5). The paper does not evaluate open-source LLMs (e.g., Llama 3), which are more accessible for deployment.
  - Environment Setup: Virtual environments rely on UnrealCV (Qiu et al. 2017) and UnrealZoo (Zhong et al. 2025), which are resource-intensive and not universally available.
- The instruction set focuses on simple spatial directives (e.g., "move closer") but not ambiguous or multi-modal instructions (e.g., "track the car next to the red tree" or "follow the person speaking loudly"). This limits UC-EVT’s realism, as real users often provide context-rich commands.
-  Evaluations only include 4 target categories (humans, cows, dogs, leopards)—no small (e.g., drones), deformable (e.g., bags), or occluded (e.g., partially hidden pedestrians) targets. SOTA trackers like TrackAnything (Cheng et al. 2023a) handle broader target types, highlighting HIEVT’s narrow focus.
- The LSTM in AGAP is critical for temporal consistency, but the paper does not ablate its impact (e.g., how performance degrades with a feedforward network).
-  While CoT improves goal generation (GPT-4o with CoT achieves 80%+ accuracy vs. 36% without; Fig. 8), the paper provides no theoretical analysis of why CoT outperforms simpler prompt engineering (e.g., few-shot examples).
-  The paper extends CQL but does not compare it to other offline RL algorithms (e.g., IQL, TD3+BC) to justify its selection.
- HIEVT requires 120,000 training steps (60k per stage) on 8 GPUs (~6 days total), which is computationally expensive compared to lightweight trackers (e.g., Mask-PID requires no training). The paper does not explore optimizations like parameter-efficient fine-tuning (LoRA) or smaller policy networks to reduce resource demands.

### Questions
- How would HIEVT handle ambiguous, context-rich, or multi-modal instructions (e.g., "track the person wearing a red jacket who is talking")? Would integrating audio encoders or scene understanding models (e.g., DPT for depth) improve goal generation?
- Can HIEVT maintain performance with open-source LLMs (e.g., Llama 3 70B, Mistral) instead of proprietary GPT-4o? This is critical for accessibility and deployment in resource-constrained settings.
- How does HIEVT perform in 10+ minute tracking tasks? Would adding a drift correction module (e.g., periodic re-alignment with SSGA) mitigate temporal decay?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper focuses on the user-centric embodied visual tracking problem and introduces HIEVT, a framework designed to improve embodied visual tracking with robust language understanding and low-latency control. The system integrates an LLM-based semantic–spatial goal aligner with a reinforcement learning-based adaptive policy, achieving hierarchical control between high-level instruction reasoning and low-level continuous actions. It also leverages chain-of-thought reasoning for interpretable spatial goal adjustment and RAG for memory-based goal correction, ensuring alignment with prior trajectories.  Experiments conducted in multiple simulated environments demonstrate that HIEVT achieves strong performance, generalization, and real-time efficiency. The framework is also deployed on a real-world robot, proving its effectiveness.

### Strengths
1.  The combination of LLM-based semantic reasoning with RL-based low-level control is novel and well-motivated. The use of parallel modules (CoT for spatial reasoning and RAG for memory augmentation) enables interpretable goal generation and efficient decision-making.
2. Strong experimental results and real-time performance. The proposed method outperforms both traditional and state-of-the-art VLA-based baselines in multiple simulated and real-world environments, achieving near-perfect success rates in some settings. The real-time deployment on a physical robot demonstrates practical efficiency, robustness, and effective sim-to-real transfer.
3. The paper includes a detailed GUI for interactive testing, cross-environment evaluations on nine unseen scenarios, and an actual hardware deployment. These experiments convincingly validate the method’s robustness and adaptability.

### Weaknesses
1. The method is only tested on the authors’ self-constructed dataset, without comparison on existing EVT benchmarks such as Gym-UnrealCV. This makes it difficult to assess how well the approach generalizes beyond their setup and gives the impression that the model may be overfit to the custom environment.
2. Unclear motivation and missing ablation study for design choices. The paper introduces several components but doesn’t clearly explain their purpose or justify their inclusion through ablation studies. For instance, it’s not clear why two critic networks are needed to estimate goal-conditioned Q-values, how the trajectory priors in the memory buffer are obtained, or whether their diversity affects performance. Similarly, the choice of using CQL instead of more recent offline RL algorithms is not discussed. Without explanation or supporting experiments, many of these design choices feel arbitrary. If they are inspired by prior work, relevant references should be cited.
3. Presentation and formatting problems. There are several issues that make the paper harder to read. All references mix author names directly into the text, which is confusing. The appendix section titles are incorrect, and placing lots of figures and tables on a single page (page 7) disrupts the reading flow and makes the layout look cluttered.

### Questions
1. Equation 5 looks somewhat unclear and would benefit from additional explanation or context.
2. What does the blue background in Table 1 indicate?
3. I’m curious how the proposed framework achieves reduced latency, given that it incorporates both RAG and chain-of-thought (CoT) reasoning with LLMs. These components are typically time-consuming.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces Hierarchical Instruction-aware Embodied Visual Tracking (HIEVT), a framework for user-centric embodied visual tracking where an agent follows a target based on natural-language instructions specifying both the target and tracking conditions such as distance or viewpoint. The approach separates planning and control into two levels: a high-level module that parses language and determines spatial goals, and a low-level reinforcement learning policy that performs continuous control to maintain the specified spatial relationship. The model is trained and evaluated on a large simulated dataset including multiple unseen environments. Experiments indicate that this hierarchical design improves performance and generalization compared to prior end-to-end methods, particularly in longer or more dynamic tracking scenarios.

### Strengths
- The asynchronous planning strategy sounds reasonable and has potential to keep the building components with better ones.
- The approach achieves strong performance over various baselines.
- The paper is generally well-structured and easy to follow.

### Weaknesses
- The authors argue that the user-centric EVT is one of their contributions, but its description is rather unclear and even in Appendix. How do we define the "user-centric" EVT and how is it different from previous EVT setups? 
- Following targets has already been explored in prior work such as Puig et al. 2024. This paper aims to follow humans, but simple extension from humans to animals can be done by modifying its object meshes. Can the authors the difference from this?
  - Puig et al., "Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots," ICLR, 2024.
- For the high-level goal aligner, it seems not taking into account its history observations. Given a scenario where we have two people with the same appearances and instructed by "follow the right person," when they cross each other, the right-side people goes left and vice versa, but the high-level goal aligner does not have access to any information to distinguish them. Does the current approach address this issue?
- The high-level goal aligner is comprised of multiple modules, potentially raising error propagation issues. Can the proposed approach handle this issue?
- For the Memory-Augmented Goal Correction, its buffer maintains reference trajectories from training examples, but it is unclear how much sensitive is the success rate with the buffer to the quality and diversity of the ones in the buffer. What if some "incorrect" (e.g., following wrong targets, ...) trajectories? What if the trajectories follow suboptimal paths to the targets? These analyses are missing in the current paper.
- The threshold in Equation 3 for the goal correction is set to 0.5, but it is missing how sensitive is the choice of this values w.r.t. the success rate.

### Questions
- In Figure 2, the intermediate spatial goal seems quite different from the bounding box from VFM. Why not use the bounding box from VPM? Why do we make it 
- Does it also work with different embodiments, such as different heights, camera intrinsics, etc.? If so, how well? Including the analysis of the aspect of cross-embodiment generalization would improve the paper.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the gap between the goals in the Embodied Visual Tracking task. Basically, the users of the Embodied Visual Tracking models are provided with some user-friendly instructions, which are hard for current models to understand. To solve the problem, the authors proposed Hierarchical Instruction-aware Embodied Visual Tracking (HIEVT), which translates the user-friendly goal with the Semantic-Spatial Goal Aligner and Adaptive Goal-Aligned Policy. 

The evaluation results are very solid and prove the effectiveness of the proposed framework. However, this paper has some severe writing issues as detailed below in the "Weakness part."

### Strengths
1. This paper studies a very important problem, bridging the gap between user-friendly goals and goals for Embodied Visual Tracking models.
2. The proposed method, Embodied Visual Tracking task, is very elegant and well-established.
3. The evaluation of the paper shows the proposed method is very effective.

### Weaknesses
However, there are two severe issues with the writing in this paper:
1. The motivation and the method are not matched. In the Introduction (lines 62-74), the authors listed three weaknesses of current models: 1) Limited Comprehension, 2) Limited Generalization on unseen data, and 3) Inference Latency. However, the proposed method seems to have only addressed the weakness of Limited Comprehension, ignoring the other two.

2. The whole paper misused the commands **\citet** and **\citep.** For example, at line 57, the citations are "Existing methods for embodied visual tracking (EVT) Luo et al. (2019); Zhong et al. (2019b; 2021; 2023)." Instead, the citations should be (Luo et al. 2019, Zhong et al. 2019b; 2021; 2023). That is, both author names and years are in parentheses.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
3
