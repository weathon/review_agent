# ReinforceGen: Hybrid Skill Policies with Automated Data Generation and Reinforcement Learning

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Long-horizon manipulation has been a long-standing challenge in the robotics community. We propose ReinforceGen, a system that combines task decomposition, data generation, imitation learning, and motion planning to form an initial solution, and improves each component through reinforcement-learning-based fine-tuning. ReinforceGen first segments the task into multiple localized skills, which are connected through motion planning. The skills and motion planning targets are trained with imitation learning on a dataset generated from 10 human demonstrations, and then fine-tuned through online adaptation and reinforcement learning. When benchmarked on the Robosuite dataset, ReinforceGen reaches 80% success rate on all tasks with visuomotor controls in the highest reset range setting. Additional ablation studies show that our fine-tuning approaches contributes to an 89% average performance increase. More results and videos available in https://sites.google.com/view/reinforcegen-iclr26.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work extends an object-centric data-generation pipeline, SkillMimicGen, to long-horizon manipulation, introducing ReinforceGen, a hybrid approach that alternates between motion planning and skill execution while fine-tuning three components comprising the pose predictor, the skill policy, and the termination predictor. Improve the skill policy with residual RL on top of a BC prior; use online replanning at initiation and teacher distillation during training for better generalization; and reduce termination false positives by including a success-probability gate. Results demonstrate that with limited human demonstrations, this approach outperforms prior SkillMimicGen-style baselines on five Robosuite tasks and is comparable in performance to policies using privileged state.

### Strengths
* Hybrid pipeline using "BC + RL": explicit decomposition with clear initiation/skill/termination, targeted improvements using replanning, residual RL, termination gating.
* Competitive performance with only a few human demos; significantly outperform prior SkillMimicGen-based baselines and close to privileged-state “upper bounds”.
* Demonstrate the possibility of incorporating online RL into a generation pipeline centered on objects with no discarding of planning advantages.

### Weaknesses
* Narrow scope of evaluation: Only five tasks in Robosuite; greater task diversity, and another simulator, would strengthen external validity.
* No real-robot validation: Without any hardware result, its applicability remains uncertain under real world settings.
* Presentation clarity: Many implementation details (e.g., thresholds, reset distributions, termination gate calibration) are somewhat opaquely documented.

### Questions
* Hyperparameter sensitivity: How sensitive is performance to planning replan threshold for initiation and termination gate threshold? Learning curves or sensitivity plots across reasonable range for the hyperparameters may provide clarity.
* Algorithm selection: Why DrQ-v2? Would PPO, SAC, IQL yield similar gains under the residual setup and image inputs? Any ablations across RL algorithms?
* Embodiment Generality: Is there an extension to other embodiments, for example, multi-finger hands/dexterous manipulation? What will be the expected pain points?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The work proposes a pipeline of methods to learn a policy for robotic tasks from offline data and then fine-tune it with several expert demonstrations. In limited experimental setup, the method seems to improve over a baseline.

### Strengths
The method addresses a real problem in robot learning - learn to solve a task from a substantially small number of expert demonstrations. Learning is divided to offline and online parts. This makes sense since the compounding error can be reduced by the online part (in the same spirit as Dagger).

### Weaknesses
**Major:**

 - The work claims to contribute to robot learning, but no experiments with real robots are made - everything is simulated

 - The work is a good piece of engineering work where certain parts from existing works are put together and then experimented, but lacks  in scientific research questions or contributions

 - Results table does not include comparison to existing works despite that for the used tasks related works have been published




**Moderate:**

 - You claim that during training you have access to the object state (pose), but this is straightforward only for simulated environments - how about real environments, how this is realized without substantial extra equipment & calibration etc.?

 - In Figure 3 and text - why you use the term "distillation" as I assume you just train a pose predictor?

 - In Section 4.2 you list methods and then plainly justify that the *residual RL* is selected. I assume any recent offline RL methods should work here and if not, then some kind of comparison should be provided.

 - Clearly define what are the observable inputs to the system during inference

**Minor:**

 - what is the unit in Figure 4 since pose consists of three metric translation component and three angular orientation components? I don't think the success rate drops "sharply" - this is especially difficult to judge without knowing the units.

 - Section 4.4 assumes that a reader reads Appendix C at the same time - without the appendix the section is useless. This is not good practice as every article should be sufficiently self-contained

### Questions
This work is a good piece of engineering work, but to make it stronger and more scientific:

 - Compare to known SotA on these benchmarks and in fair settings (should be easy as everything happens in simulations)
 - IF your approach is clearly better than SotA explain what parts and why they solve the problem/limitation in the existing methods
 - Demonstrate your results with real robot tasks (simulation is not robotics as the real world brings in many challenges)

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Building on the Hybrid Skill Policy (HSP) paradigm, the paper decomposes tasks into multiple sub-tasks. It then proposes a demonstration/data-generation augmentation pipeline that couples motion planning with reinforcement learning: starting from a small number of human demonstrations, the method expands to a larger offline dataset and further refines the policy via RL. On five long-horizon robosuite tasks, it outperforms a vanilla HSP baseline, though it still lags behind approaches trained with substantially more human demonstrations and those leveraging privileged state information.

### Strengths
1. Termination Classification: The authors introduce a learned termination classifier that minimizes the gap between training and deployment by rejecting low-confidence terminations, thereby reducing train–test mismatch during execution.

2. Initiation Pose Prediction: The initiation pose predictor is continuously updated during the connection segment and can trigger replanning when necessary, which substantially reduces pose error and improves task success rates.

### Weaknesses
1. Rationale for Imitation Learning vs. Pure RL: It is unclear why imitation learning (IL) is necessary. How would a purely reinforcement-learning pipeline perform under the same training budget and environment settings? Does IL primarily improve computational efficiency (sample/compute efficiency) or final success rate—and by how much?

2. Gap to Stronger Oracles and Data Regimes: Although the method improves over a vanilla HSP baseline, it still underperforms settings that use privileged state information and/or substantially larger human-demo corpora.

3. Termination Classifier Reporting: While a termination classification scheme is proposed, the paper does not report its standalone accuracy quality.

4. Lack real-world setting to verify the capability of model.

### Questions
Reinforcement learning appears to contribute substantially to performance. How do you quantify RL’s contribution?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a learning pipeline for multi-stage robotic manipulation tasks, including a MimicGen and imitation learning stage followed by an online reinforcement learning (RL) stage. During the online RL stage, initial pose predictors and termination classifiers are learned for each subtask for subtask stitching and success detection, and residual RL is used to improve each subtask policy. Finally, all components are distilled into  an end-to-end policy for vision-based execution without privileged information. 
Experiments on some simulation tasks demonstrate that the system outperforms HSP baselines and achieves comparable performance against SPIRE trained on more human demonstrations.

### Strengths
- RL finetuning is a significant problem for MimicGen-style robotic manipulation policies.
- The system dividing the long-horizon task into motion-planning stages and RL stages effectively reduces the burdens for RL exploration.
- Experimental results demonstrate its effectiveness compared with the previous HSP method.

### Weaknesses
- The RL pipeline only works in simulation, since privileged states are required to train the initial pose predictors and termination classifiers. Given this, is it really necessary to train these modules? A much simpler approach may also works: using ground truth motion-planning poses and ground truth success detectors for RL in simulation, then distilling all modules into an end-to-end policy.
- Another simple and direct approach is not compared with: apply MimicGen, BC, and residual RL for each individual subtask in simulation, then alternately execute motion planning and these policies for distillation.
- Lacks discussion and comparison of previous works solving the similar problem [1,2,3].
- The sim-to-real performance and real-world applicability of the system is not evaluated. 

[1] Chen et al., Sequential Dexterity: Chaining Dexterous Policies for Long-Horizon Manipulation, 2023
[2] Agia et al., STAP: Sequencing Task-Agnostic Policies, 2023
[3] Lee et al., Adversarial Skill Chaining for Long-Horizon Robot Manipulation via Terminal State Regularization, 2021

### Questions
- Please see weaknesses.
- Will synthesizing more data (e.g. 10000 demos) during the MimicGen stage bring the same improvement?
- Why is DrQ-v2 used for residual RL? Do other popular RL algorithms (e.g. PPO and SAC) work?

### Soundness
2

### Presentation
2

### Contribution
2
