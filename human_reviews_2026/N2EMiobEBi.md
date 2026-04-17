# Split Decisions: VLM-Guided Action Sampling for Efficient RL Exploration

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Reinforcement learning (RL) offers a general framework for adapting vision-language-action models (VLAs) to new tasks, but its effectiveness is often bottlenecked by inefficient exploration. Existing strategies waste interactions on uninformative behaviors, hindering sample efficiency. We introduce Split Decisions, an exploration framework that leverages semantic priors from vision-language models (VLMs) to guide VLAs toward more promising regions of the action space. In our approach, the VLM serves as a high-level planner that proposes subgoals, while the VLA acts as a low-level controller that samples and executes actions aligned with those subgoals. This structured guidance improves both the efficiency and quality of exploration, enabling policies to discover rewarding strategies more quickly. We evaluate Split Decisions on robotic manipulation tasks in SimplerEnv under both online and offline RL settings. In online fine-tuning, it achieves up to a 31\% gain in task success with the same interaction budget, while in offline training, datasets collected with Split Decisions provide a 27.5\% improvement over prior methods. These results establish Split Decisions as a general and effective paradigm for enhancing exploration in VLA adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Split Decisions, a training-time exploration framework that uses a vision–language model (VLM) as a high-level planner to generate 2D waypoint subgoals and a vision–language–action model (VLA) as a low-level controller to sample and execute actions that are closest to those subgoals. This structure narrows the action search space and reduces wasted, low-value interactions during RL fine-tuning of robot policies.

The paper decomposes the method into two stages: 

Subgoal planning: a VLM decomposes tasks into ordered waypoint trajectories with completion conditions; waypoints guide exploration.

Guided action sampling: the VLA samples multiple candidate actions; candidates are scored by geometric proximity (using 2D/3D projection/unprojection) to the current waypoint; a top-ranked action is executed.

The following produced dataset can be used for a variety of finetuning tasks, which include online and offline finetuning.

When evaluated in both online and offline environments, split decision provides improvements. In online environments, it produces better sample efficiency, and in offline environments, it provides better success rates compared to other methods when trained on the same data.

### Strengths
1. A sound way of performing exploration with just base policies. VLMs are more commonly used for planning, as seen in [1,2], but it is nice to see this work to use VLMs for exploration directly.
2. I like the simplicity of the method with respect to the dataset generation. I think this method can be easily scaled without much concern about data suboptimality.
3. The paper is comprehensive in its ablations, as it discusses all of its design choices to some extent, validating on why each component is needed.

[1] Tang, G. et al., 2025. “KALIE: Fine-Tuning Vision-Language Models for Open-World Manipulation without Robot Data”. ICRA
[2] Black, K. et al., 2025 “$\pi_{0.5}$: a Vision-Language-Action Model with Open-World Generalization”. CoRL

### Weaknesses
1. Split Decisions requires the camera to contain knowledge of its pose in its method. When being applied on real-world tasks without such depth information (when many inference pipelines are being done in such case), you might get degenerate solutions.
2. Limited methodological innovations. There are previous works that have used VLMs to generate keypoints that can be used for reinforcement learning [1]. While I understand that the generated datasets are for different purposes, it would be better to delineate what are the differences in this stage.
3. The paper did not use any real-world benchmarks (as all of the evaluation protocols are based on simulations). I believe it will be valuable if the authors include such evaluation procedures.
4. The paper did not define some key components for consideration, namely: how are the rewards defined in online finetuning, and how are the preferences being generated in offline finetuning.
5. I found the process of generating 2D waypoints can get out of hand if you are trying to learn a long-horizon task (i.e. composing pick and place). I believe that more empirical or theoretical guarantees need to be made in this method if it were to work well in compositional tasks.

References:
[1] Lee, O. et al., 2025. “Affordance-Guided Reinforcement Learning via Visual Prompting”. IROS

### Questions
1. Is it possible to compare your method against a more RL-based methods, where you rank your actions based on a learned Q function, instead of using distances produced by a VLM?
2. On figure 4. Do you have any thoughts on why RLVLA performed better than Split Decisions when carrots are involved? My hunch is that since carrots are much more seen in OpenVLA, it is easier to perform RL on it.
3. Since OpenVLA is a binned policy [1] with uniform action binning, are there any theoretical justification in perturbing the actions with uniform distributions instead of Gaussian distributions?

References:
[1] Kim, M. et al., 2024. “OpenVLA: An Open-Source Vision-Language-Action Model”. CoRL

### Soundness
2

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
This paper proposes efficient VLA RL fine-tuning with VLM guidance. A VLM generates a sequence of subgoals, and the VLA’s predicted actions are filtered to the subset minimizing distance to those subgoals, pruning low-value choices. Additionally, for computational efficiency, the method injects Gaussian noise into a single predicted action rather than generating many candidates; for sample efficiency, it blends policy-gradient and PPO losses to mitigate near-zero probabilities on filtered actions, which would make a PPO-only objective inefficient. Across online and offline RL, it outperforms baselines on manipulation tasks such as pick-and-place and drawer opening.

### Strengths
- The method shows application in both online and offline RL, demonstrating good generality.
- By showing that filtered actions have low likelihood under the policy, the paper pinpoints why PPO-only can sometimes underperform—a useful caution for off-the-shelf use.

### Weaknesses
- **Potentially unfair baseline setup**
    + Gaussian noise is injected only into positional moves. Because position can dominate exploration in your tasks, this likely advantages your method while baselines lack this inductive bias.
    + Please clarify this and/or verify fairness with empirical results—e.g., perturb only positional action for all baselines, or adopt a faster VLA baseline for multiple samples (e.g., [1]).

- **Unvalidated benefit of combining PG + PPO**
    + Authors state low action likelihood can hinder PPO, but there’s no experiment supporting combining the PG+PPO gains. Authors can compare PPO-only, PG-only, and PG+PPO to show the benefit.
    + Authors mention using PG “early” and PPO “later.” How is the switch determined (is it a hyperparameter)? Additionally, it would be helpful if authors could specify how these can be sensitive in training.

Additional suggestions
- It could be helpful to tease apart each component—for example, cases where the VLM’s subgoal is off versus cases where the VLA can’t execute the predicted subgoal—to clarify the main bottleneck.
- It might help to add a brief limitations note. For example, waypoint-style subgoals may struggle with tasks like in-hand manipulation or screwing in assembly. You could emphasize where the method applies well and possibly sketch how it could be extended—for instance, combining with a reward that more efficiently captures those behaviors.

#### Reference
[1] Kim et al., "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success", RSS 2025

### Questions
- Q1. In Fig. 4, why is the gap between RLVLA and your method so large from the very start? Don’t both use the same pre-trained OpenVLA?
- Q2. For Fig. 4, how many seeds were used? Also showing training variance would be helpful.
- Q3. Can authors elaborate on what the “Random sampling" baseline is?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Split Decisions a exploration framework for efficiently adapting vision-language-action models (VLAs) to new robotic manipulation tasks through reinforcement learning. The key idea is to model the policy with a high-level VLM planner that generates semantic subgoals in the form of 2D waypoints to guide VLA action exploration, rather than relying on random or curiosity-driven exploration strategies.

The method operates in two stages:

1. Subgoal Planning: A VLM decomposes tasks into sequential subtasks represented as 2D waypoint trajectories with completion conditions
2. Guided Action Sampling: The VLA samples multiple candidate actions, ranks them by proximity to the current subgoal waypoint, and executes top-ranked actions

For fair comparison, authors incorporate similar action exploration mechanism to existing VLAs like OpenVLA. To incorporate similar exploration mechanisms in existing VLAs like OpenVLA, the authors propose: using action sampling through Gaussian perturbations rather than costly autoregressive generation.

The method is evaluated on SimplerEnv benchmarks derived from BridgeData and Google Robot datasets. Results show 31% improvement in online RL task success and 27.5% improvement in offline RL compared to prior methods, with strong generalization to drawer manipulation tasks (40.8% gain).

### Strengths
1. The paper proposes an interesting idea to improve exploration during RL training for VLAs by breaking down the policy into two components with a high level planner and low-level VLA controller and action ranking.
2. The results presented in the paper demonstrate that proposed method outperforms the specific instantiation of the baselines in a similar setting.
3. The ablation and generalization experiments are insightful and good addition to support the claims made in the paper.

### Weaknesses
1. The experimental evaluation setup used for comparing SplitDecisions with OpenVLA + gaussian noise sampling for action exploration seems unfair comparison. It is unclear whether explicit action exploration similar to proposed method SplitDecisions is required for autoregressive baselines like OpenVLA. In order for the experimental results and comparisons to be meaningful I would like to see the following baselines:
    1. OpenVLA RL on the same benchmark without any gaussian perturbations and explicit exploration i.e. RLVLA instantiated exactly as set up in original paper
    2. OpenVLA RL on the same benchmark with gaussian perturbations and explicit exploration as proposed in the paper
    3. SD guidance based RL training.
    
    The paper already has results for b & c but we need results for baseline a to understand whether the gaussian perturbation approach used for exploration for baseline b is actually helping or hurting the performance
    
2. Figure 4 only presents results of finetuning all models for ~300k steps which is quite less training steps or experience to make meaningful conclusions from. I would like the authors to train all 3 baselines described in point 1 atleast until saturation of a larger budget of 1M-5M training steps. Making conclusions from models that have not converged seems like a unfair comparison especially for changes in training methods. In addition, I would like to see the average success plot for all tasks combined in figure 4 rather than just per task breakdown of RL training curves.
3. The experimental evaluation setup used in the paper is also quite concerning. The paper only uses 4 tasks from SimplerEnv for training from Google Robot scene and 4 tasks from BridgeData in two separate experiments. My understanding is this is not the exact same setup as used in RLVLA paper. The RLVLA paper claims to use 16 tasks each for tasks randomized across following axes: Vision (16 tables), Semantics (16 objects), and Execution (perturbations of object and receptacle poses). I would like the authors to use the same task and dataset setup for a fair comparison and if they cannot do that I would like to understand why this is not possible. My biggest concern right now is that performance reported for RLVLA in experiments in the paper do not follow similar trend as the ones from original paper. Due to this difference I can’t be confident whether the reported improvements of SplitDecisions hold true when RLVLA is replicated accurately
4. The method section doesn’t describe how top-K sampled actions during exploration stage/SD guidance stage where K > 1 are used for policy update in standard PPO algorithm. The vanilla PPO implementation requires 1 rollout per task where each step has only 1 action and for each task execution we have single task reward. When sampling more than 1 action at each timestep how are the complete task rollouts collected and incorporated in the PPO update needs to be explained in detail in methods section. With the details reported in the paper I don’t understand how the ablation presented in table 3 is conducted. If at each step there are 20 candidate actions that a agent takes then the number of rollouts for each task is going to increase exponentially with steps.
5. The paper should also add additional details on how did the authors control for samples/FLOPs per update to make the comparison fair. From my understanding, SplitDecision can include greater than 1 action per timestep for updating the policy which increases the number of samples per rollout/update that can be used for RL. For the three baseslines I mentioned in point 1 authors need to control for training updates or samples for a fair comparison.

### Questions
Mentioned in weaknesses section.

My main concern with the paper in its current state is the experimental setup and baseline instantiation used for comparison. Due to lack of proper motivation on augmenting a relevant baseline RLVLA with random sampling for exploration and missing baselines that do not incorporate vanilla RLVLA I am not confident in the results presented in the paper. I recommend the authors to run the relevant baselines for fair comparison. In addition, the method section doesn't include some details about how multiple action samples per step are used during training which is critical information for the paper. Due to the concerns I have outlined in weaknesses section I recommend a rejection rating for the paper in its current state. I am happy to increase my rating if authors can add missing baselines, improve the experimental setup and fix the methods section.

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
The paper proposes Split Decisions, a framework designed to improve exploration efficiency in reinforcement learning (RL) for vision-language-action (VLA) models. It leverages vision-language models (VLMs) as high-level planners that generate semantically meaningful 2D waypoint subgoals. These subgoals guide the VLA’s low-level action sampling, biasing exploration toward more promising regions of the action space. The framework integrates seamlessly with standard online and offline RL methods (PPO and DPO), and the authors demonstrate consistent gains across multiple robotic manipulation benchmarks, achieving meaningful improvement in online RL and in offline RL relative to baselines such as RLVLA and GRAPE.

### Strengths
- **Empirical rigor:** Comprehensive experiments across SimplerEn, BridgeData, and Google Robot benchmarks confirm consistent and significant gains.  
- **Clear methodology:** The paper includes detailed algorithmic descriptions, ablation studies, and prompt templates for reproducibility.  
- **Complementary to existing approaches:** The method can be paired with standard RL algorithms without architectural modifications.  
- **Strong clarity:** Figures and appendices are well-organized and enhance understanding of the exploration pipeline.

### Weaknesses
- **Limited novelty:** The paper does not clearly articulate how Split Decisions advances beyond existing VLM-guided exploration methods such as ExploRLLM (Ma et al., 2024) or similar cited works. Nor does it compare against them.
- **2D-only subgoal representation:** Current implementation restricts subgoals to 2D projections, limiting applicability in complex 3D manipulation tasks.  Assumes inverse projection from 2D to 3D is available. This limits the generalizability of the methods to other domains.
- **High computation cost:** Sampling multiple candidate actions from large autoregressive VLAs remains resource-intensive.  They also assume having a simulator that simulates multiple actions from a current state and chooses the one that ends closest to the desired waypoint.
- **Sensitivity to VLM quality:** The method’s success heavily depends on the semantic quality of the chosen VLM (e.g., Gemini 2.5 Pro outperforms ChatGPT 4.1 by a large margin).  
- **No real-world validation:** Results are entirely from simulation; the absence of physical robot trials weakens claims of practical generalization.  
- **Limited theoretical framing:** Improvements are well-documented empirically but not analytically grounded in exploration theory.

### Questions
1. How does Split Decisions fundamentally differ from other VLM-guided exploration methods, such as ExploRLLM?  
2. How robust is the approach when the VLM produces noisy or incorrect subgoals?  
3. How can in the real world, the action that brings the robot closest to the keypoint be chosen?  
4. Could 3D-aware waypoint prediction or affordance-based subgoals address the current 2D limitations?  
5. Have the authors tested whether the framework generalizes to non-manipulation domains (e.g., navigation or embodied instruction following)?

### Soundness
3

### Presentation
3

### Contribution
2
