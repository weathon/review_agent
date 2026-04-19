# Optimizing Latent Goal by Learning from Trajectory Preference

- Decision: Reject
- Scores: 3, 5, 6, 8

## Abstract
A glowing body of work has emerged focusing on instruction-following policies for open-world agents, aiming to better align the agent's behavior with human intentions. However, the performance of these policies is highly susceptible to the initial prompt, which leads to extra efforts in selecting the best instructions. We propose a framework named \emph{\textbf{P}reference \textbf{G}oal \textbf{T}uning} (PGT). PGT allows policies to interact with the environment to collect several trajectories, which will be categorized into positive and negative samples based on preference. A preference optimization algorithm is used to fine-tune the initial goal latent representation using the collected trajectories while keeping the policy backbone frozen. The experiment result shows that with minimal data and training, PGT achieves an average relative improvement of $72.0\%$ and $81.6\%$ over 17 tasks in 2 different foundation policies respectively, and outperforms the best human-selected instructions. Moreover, PGT surpasses full fine-tuning in the out-of-distribution (OOD) task-execution environments by $13.4\%$, indicating that our approach retains strong generalization capabilities. Since our approach stores a single latent representation for each task independently, it can be viewed as an efficient method for Continual Learning, without the risk of catastrophic forgetting or task interference. In short, PGT enhances the performance of agents across nearly all tasks in the Minecraft Skillforge benchmark and demonstrates robustness to the execution environment.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces PGT, Preference Goal Tuning which leverages the current policy to generate trajectories in simulation, using human preferences or already available simulation reward predictors to generate preferences, and then subsequently employs direct preference optimization (DPO) for downstream learning on the collected preference data.

### Strengths
The paper demonstrates strong results in Minecraft domain, when comparing against baselines.

### Weaknesses
The main weakness is the novelty of this approach. The approach uses online-DPO style method to generate and leverage preference data to learn downstream policy. Direct preference optimization approaches have already been applied to many domains. Although the improvements in results in Minecraft domain are appreciable, the paper does not offer any specific analysis for the pertaining application, which makes this work incremental.

Further, the method assumes trajectory generation for current policy in simulation, which might be cost-inefficient in real domains like robotics. This limits the applicability of the proposed approach in such scenarios.

The paper generates preference dataset using privileged information (preferences), however the comparison baseline Behavior cloning does not, which raises questions on the fairness of such comparison. It would have been interesting to see comparisons against other methods that leverage such preference data for policy learning like SLIC [1]. 

Therefore, I believe the paper is currently not ready for acceptance, and needs extensive updates to improve the novelty of the approach, writing and experiments for a stronger submission. 

Minor comments: 
1. Typo in Line 89
2. Fig 3 too small to read

[1] Calibrating Sequence likelihood Improves Conditional Language Generation

### Questions
None.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors present a framework to fine-tune latent goal via preference optimization for goal-conditioned policy. Especially, the proposed method yields an evident performance improvement with minimal fine-tuning of goal latent representation, and it presents potential capability in continual learning.

### Strengths
The proposed method is simple but effective in the presented applications and tasks.

It also presents potential for continual learning without fine-tuning policy model.

The authors conducted extensive experiments to illustrate the advantageous of the proposed methods in various tasks in Minecraft envrionement.

### Weaknesses
1) Limited application: The paper conducted experiments solely on Minecraft Skill forage, leaving questions how it works in other goal-conditioned applications.

2) This method can be hard to adopt in tasks where rolling out trajectories or obtaining preferences for them is costly

3) Utilizing negative trajectory samples seems critical in this method but it is already well known method. Unless there is a specific challenge to adopt such method in openworld task such as Minecraft, the contribution of this paper becomes marginal as it is just adopting good method in another applications.

### Questions
1) Can this method be easily adapted for other goal-conditioned tasks where the goal is not limited to a language prompt but could be, for example, an image—as in robotics tasks (as the authors mentioned), possibly Franka Kitchen [1,3] or UR3 BlockPush [2,3]? If not, an explanation of the limitations would help clarify the potential of the proposed method.

2) It is unclear to me how PGT works. Could you elaborate more on PGT and how to generate g in the first place to compute $L_{PGT} (g, g_{ref})$ ? 

3) For each task, should we conduct iterative process which gathers trajectories and obtains preference them from human or reward models to get the optimal latent goal. If so, this method can be limited in generalization where the strength of the foundation model primarily lies.

4) In Table 2, is there any specific or presumed reason for STE+ not working well compared to STE in hunt task?

5) How sensitive is PGT with respect to  $\beta$ in Eq. 2? which value is used for each task? 

6) Regarding weakness 3), could you explain how the proposed method differs from or improves upon existing applications of preference learning in open-world tasks? 

Minor issue:
The same paper is referenced differently for Tajwar et al.

[1] Gupta, Abhishek, et al. "Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning." arXiv preprint arXiv:1910.11956 (2019).

[2] Kim, Jigang, et al. "Automating reinforcement learning with example-based resets." IEEE Robotics and Automation Letters 7.3 (2022): 6606-6613.

[3] Lee, Seungjae, et al. "Behavior generation with latent actions." arXiv preprint arXiv:2403.03181 (2024).

### Soundness
3

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
This paper presents a post-training method for foundation policies called Preference Goal Tuning (PGT). The PGT utilizes the preference trajectories either annotated by human experts or culmulative reward to fine-tune the latent goal representation in the goal-conditioned policies. In the experiments, PGT demonstrates the capability of increasing the performance of goal-conditioned tasks by fine-tuning relavent goal representaion structure in foundation models. PGT demonstrates good robustness in handling OOD tasks and surpass other baselines in the domain of Continual Learning (CL).

### Strengths
- The idea of utilizing preference data to fine-tune the foundation models is clear and intuitive. The cost of accquring prefence data is low, according to the claims in the paper.
- Fine-tuning only a small amount of parameters is proved effective in other related work (e.g. LoRA and other parameter efficient method). It is good to see the comparison between PGT, full fine-tuning and other parameter efficient methods. Moreover, the choice of only optimizing latent goal represenation is a good and intuitive when the foundation model is required to solve goal-conditioned tasks.
- It is good to see PGT is robust in situations where 1) with different initial prompt 2) OOD tasks.

### Weaknesses
- Minecraft environments is a good start, while more challenging tasks would be greatly appreciated.
- This is minor. I understand it is not easy to organize all sub-sections under section 4, but the title of the section 4 is not very informative. Please re-consider the title of section 4.
- "Trajectory Preference" is serious. I briefly read the derivation of the PGT loss in section A.2. From my understanding, the $\pi(\tau;g)$ denotes the action given by the policy is conditioned on the goal $g$ and the trajectory $\tau$. Under the MDP setting as indicated in section 2.1, the action is only conditioned the current state (plus goal $g$ in goal-conditioned problem). The $\tau$ in the $\pi$ should be explicitly degenerated to $s$, if I understand it correctly. I am not saying the derivation is totally wrong. In fact, the PGT loss is derived from DPO which handles bandits problem solely and I understand you treat the $\tau$ as a single step. You may consider rewording a bit to indicate PGT is still optimizing under bandits framework yet benefits from collected preference trajectories.

### Questions
- See the last argument in weakness. I would be happy and open to further discussions. 
- If I understand it correctly that you do treat each $\tau$ as a single step to optimize under DPO, it is still required to model the whole task as a MDP? I know the tasks in this work are MDP problems and I am asking wether the non-Markovian problems can be solved using PGT. Looks to me you do not even need to emphasize the MDP in section 2.1. As long as you have the $\pi(\tau, g)$and $r(\tau)$, the PGT is valid.
- When annotating the trajectories, how to judge the quality of the trajectory if their goal $g$ is different?
- This is minor, why the counter-part of the Full Fine-Tuning in table 1 is the *Soft* Prompt? What does *Soft* indicate here?
- It is possible to use foundation model itself (or any other foundation model) to judge the quality of the trajectory?

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
4

### Summary
The submission introduces preference goal tuning (PGT), which is a form of continual learning approach to finetune the a pretrained model to tackle that prompt better. PGT embeds an initial prompt to a latent representation, and over several rounds of trajectory collection + finetuning of the latent representation via a modification of DPO's objective. Overall the results of PGT are sound and indicate that PGT can  improve prompt latent representations to 1. more robust and 2. even enable the solving of tasks that humans could not prompt out of pretrained models.

### Strengths
- PGT demonstrates that it can finetune a latent representation of a prompt to be more robust, with similar results to full finetuning on in-distribution finetuning data and stronger results on out-of-distribution settings (where the agent in the Minecraft world spawns elsewhere, in a visually/structurally different biome/background etc.).
- PGT results show that it outperforms human prompt engineering. It is also great to see the authors address the limitation of this method requiring preference data to work which is generally achievable in simulation via success conditions but not for the real world and is thus not free lunch.
- The design of PGT and training setup is fairly efficient and low-cost as the decoder is frozen 
- Really interesting experiment with the "use trident" task having no success when given a normal prompt about trying to use the trident, but when the prompt's latent representation is finetuned via PGT there is some success now. It does indeed suggest the insight that pretrained foundation models have a lot of skills embedded in their weights, but need a good finetuned prompt to "unlock" that ability and leverage it. However is this the only skill where this occurs? Are there others and what are the exact numbers of the improvement (original model getting 0 success but with PGT the new latent prompt gets some nonzero success/return).

### Weaknesses
- Section A.2 about PGT loss mentions the assumption of a "Bradley-Terry-Model-liked oracle reward function" existing. I don't quite understand what it means to be a BT-model-liked reward function. Is there an example for some of the environments? Say collect wood or craft object X? If this is a reward function in the typical sense of RL literature, how dense/optimal does this reward need to be? This may be potentially a strong limiting factor of this work
- Section A.2 what does it mean the task can be subjective or objective? This needs elaboration as I don't understand how it relates to the derivations.
- This is perhaps is more of a opinion coming from a RL+embodied AI/robotics background but I find the term "Task Environment Generalization" to be a little misleading. Upon my first read I thought the OOD settings were different tasks (although hindsight that cannot make sense since you finetune for a specific task) as often times the word "environment" is interchangeably used with "task" (by e.g. gymnasium/gym libraries and various robot learning benchmarks). Maybe a better phrase is "Intra-Task Generalization" or something, I am not sure however. 
- Table 2's choice of colors for improvement numbers vs worsened numbers is a bit strange to me (maybe a cultural thing). I would normally expect green values to be = good, red = bad, but it appears flipped here.

### Questions
- What were the initial prompts like used in Figure 4? I am not particularly surprised some finetuning on newly collected preference data for the wood collecting task would do well if the initial prompts were already fairly good.
- Is it possible to map the finetuned prompt latent representation to human text? That would be interesting from an interpretability standpoint + may unveil qualitative/quantitative insights into which prompts work for a model and what tend to not work.


Happy to raise my score. I think this is a well written and thought out paper and would love to have it accepted. My primary reason for keeping a 6 at the moment is the first 2 points in the weaknesses section. My big concern is about how good of a reward function you might need. For the feature of being able to e.g. elicit hidden skills, if the reward function to help finetune the prompt to elicit hidden skills in the pretrained model is hard to write / needs to be fairly dense and optimal the approach of PGT might not be easily scalable.

### Soundness
4

### Presentation
3

### Contribution
3
