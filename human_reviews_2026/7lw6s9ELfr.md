# MVR: Multi-view Video Reward Shaping for Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Reward design is of great importance for solving complex tasks with reinforcement learning. Recent studies have explored using image-text similarity produced by vision-language models (VLMs) to augment rewards of a task with visual feedback. A common practice linearly adds VLM scores to task or success rewards without explicit shaping, potentially altering the optimal policy. Moreover, such approaches, often relying on single static images, struggle with tasks whose desired behavior involves complex, dynamic motions spanning multiple visually different states. Furthermore, single viewpoints can occlude critical aspects of an agent's behavior. To address these issues, this paper presents Multi-View Video Reward Shaping (MVR), a framework that models the relevance of states regarding the target task using videos captured from multiple viewpoints. MVR leverages video-text similarity from a frozen pre-trained VLM to learn a state relevance function that mitigates the bias towards specific static poses inherent in image-based methods. Additionally, we introduce a state-dependent reward shaping formulation that integrates task-specific rewards and VLM-based guidance, automatically reducing the influence of VLM guidance once the desired motion pattern is achieved. We confirm the efficacy of the proposed framework with extensive experiments on challenging humanoid locomotion tasks from HumanoidBench and manipulation tasks from MetaWorld, verifying the design choices through ablation studies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes to use multiple views to shape VLM-guided rewards for learning policies. They ground VLM alignment scores from multiple views with the environment state and then use a learned state-based reward function to guide the policy in addition to using original task rewards. They evaluate this method, called MVR, on HumanoidBench and MetaWorld across many tasks.

### Strengths
**Experimental Thoroughness:** The authors properly perform a wide array of ablations and experiments in two separate environments, each with many tasks. This paper is overall thorough, with a lot of analysis also in the appendix. 

**Well-motivated:** The approach is well-motivated, demonstrating the importance of using multiple views for reward shaping, and the reward shaping term for $r^{\text{VLM}}$ is mathematically sound.

**Results:** Overall, results are good and demonstrate the benefits of the approach over a decent set of baselines.

### Weaknesses
**Quite a bit of recent (and not-so-recent) prior work missing:** the related works section is quite short, missing some important recent work that is very relevant to this paper, and imo not comprehensive enough for an ICLR publication on robot learning. In fact, there are no reward model learning papers cited past 2024. All of the below reward learning work have been arxiv’ed and/or published beyond the 30 day ICLR limit, and I’m sure that I am still missing some important work:

- [GVL](https://generative-value-learning.github.io/) can generate *video-based progression* rewards, zero-shot, using the Gemini VLM for a variety of downstream applications
- [VLC](https://arxiv.org/abs/2405.19988) fine-tunes a VLM (clip4clip) to produce video-based rewards for trianing RL policies
- [ReWiND](https://rewind-reward.github.io/) trains a video-language input transformer to produce *dense* rewards and demonstrates real-world RL fine-tuning
- [PROGRESSOR](https://arxiv.org/abs/2411.17764) also performs real-world reward-weighted-regression with learned reward estimator from videos
- [Rank2Reward](https://rank2reward.github.io/) learns a reward function with an extra GAIL term from video
- The citation for RL-VLM-F is wrong; there are two citations at L619 and L623, the L619 one is for a completely different paper but with the RL-VLM-F title.

**Notation:** This is a minor issue, but re-using $h$ $h$$h(o, o’)$ and $h(s, s’)$ is a bit confusing given the function $h$ itself in either case is applying two slightly different operations, and $h(o, o’)$ actually depends on $l$ too. 

**Writing:** At some points, the writing jumps around a bit. Things could be made clearer:

- 4.1 **matching paired comparisons**: introduces a problem in the first paragraph (preserving orderings), but writing a little bit more at L153/154 to introduce the intuition behind Eq. 3 and how it solves the problem in the first paragraph would help with clarity
- L187 starts with “A straightforward solution to the challenge of similarity fluctuation is…” when was this term introduced? It directly starts with a solution to a challenge not mentioned in the prior paragraph, this interrupted the flow while reading. Either introduce the problem again here or hint at it in the previous paragraph.
- L198: $n(s)$ isn’t defined. I can understand what it is but this can be clearer by either defining it or stating that this is the “*average* representation of a state sequence”
- L192: please explain why this exact parameterization for $f^{\text{MVR}}$ was done. It took me as a reader extra time to think about *why* (i interpreted as a fixed, learnable orthogonal projection—actually, whether $g^{rel}$ is learnable isn’t even specified here—that helps align state representations), but a well-written paper should clearly *explain.*

**Baselines:** Related to the above point on prior work, I’d like to see a modern VLM-only based baseline to demonstrate the need for the state alignment. For example, using a Qwen-3-VL (or to match params, SmolVLM-500M) model to zero-shot process video sequence input, at the same frequency as MVR (so not sparse reward like with RoboCLIP), for RL-VLM-F/GVL would make sense here.

### Questions
How effective do the authors think MVR will be with only sparse environment reward? This is important in manipulation settings where good task reward functions can be hard to obtain in the real world.

What’s the difference between the main paper Metaworld results and Appendix B.7 “pixel-based metaworld experiments”? Is it just the policy inputs?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces VLM-based reward shaping for dynamic, suboptimal-view scenarios by using multi-view video–text similarity. First, to lower the cost of video-based rewards, this method distills each video into a low-dimensional state that preserves the task-relevance score. Second, to mitigate VLM bias toward particular viewpoints, it regularizes the representation so that similar videos map to nearby embeddings. The learned state then guides policy learning by rewarding states that best match the task’s textual description.

### Strengths
Strong motivation and sound strategy
- The paper clearly motivates its aim of guiding policies toward optimal motion patterns. The VLM guidance automatically decays during training, curbing early suboptimal actions and ensuring convergence to the true task reward (not the shaping signal), achieving the goal in a principled way.

Thorough empirical analysis
- The paper provides comprehensive analyses, baselines, and ablations of their design choices.
- The paper acknowledges cases where it trails a task-reward-only baseline (TQC), likely due to domain mismatch from human-trained VLMs; this limitation is reasonable, and proposed mitigations (e.g., domain adaptation) are clearly outlined.

### Weaknesses
Loss of temporal information?
- Although the method trains state sequences to match task relevance, the final reward in Eq. (9) is computed from a single timestep. Does this discard temporal information, thereby weakening the benefits of a video-based approach? This concern may be due to my misunderstanding; please clarify how temporal cues are preserved.

Notation & Clarity
- L187: I believe the term “similarity fluctuation” isn’t defined and was confusing—especially in the opening problem paragraph. Authors can replace it with a clearer term or add a brief definition.
- (minor) L250–L255: When describing the full framework, adding a pointer to Algorithm A1 would help. I was looking for it while reading.
- (minor) It was hard to tell when state/observation denotes a sequence vs. a single timestep. A short notation note (e.g., bold = sequence, regular = single step) would help. If this is obvious to your audience, feel free to skip.
- (minor) L85: I believe the paper does not include real-world tasks.

### Questions
- The authors use ViCLIP for RoboCLIP to ensure fairness. How would results differ with the original S3D VLM? Given that VLM choice can significantly impact performance, if applicable, could you report a backbone comparison and indicate whether your method remains strong across different VLMs?

- Additional suggestion: the benefits of multi-view are well shown in Humanoid and Meta-World. The method would shine even more in occlusion-heavy settings (e.g., humanoid locomotion on obstacle terrain, conveyor-belt picking). Adding experiments in such scenarios would make the paper more solid.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel dense reward shaping method for reinforcement learning via VLM and real-time multi-view videos. The proposed method utilizes VLM to select the top-K reference multi-view videos and trajectories for shaping state-based dense rewards via the learned reference model fMVR. With this learned reward function, this paper conduct comprehensive experiments accross both the humanoidbench and metaworld to showcase its effectiveness.

### Strengths
1. The proposed method utilize multi-view reference videos for dense reward shaping, which can be generalizable to various different tasks without a hugh amount of human efford for reward design.

2. The multi-view reference videos can improve the spatial understanding, which enhance the training efficiency.

3. This paper conducts extensive experiments accross 19 tasks accross two simulation benchmarks.

### Weaknesses
1. During the earily stage of the training process, the video quality might be pretty low, even choose the top-k trajectories. Those low-quality data might bring limited guidance to finish the task. Especially for some challenging tasks such as stick pull and hammer in MetaWorld.

2. Some previous works rely on generative model with reference trajectories and videos for reward shaping. Such as using reference trajectory [1], generated robot videos [2, 3, 4], and generated object motions [5, 6] from cross-embodiment dataset. Although those methods required reference data from robot or human, it might be helpful to discuss those related work. Also, some reward shaping ideas from those papers like conditional entropy and ranking might be good baselines for MVR.

3. I think the tables and writings have some space to be improved, such as the table 4...

I am willing to support this paper if the author can address my concerns

[1]. Peng et al., DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills, SIGGRAPH 2018

[2]. Escontrela et al., Video Prediction Models as Rewards for Reinforcement Learning, NeurIPS 2023

[3]. Huang et al., Diffusion Reward: Learning Rewards via Conditional Video Diffusion, ECCV 2024

[4]. Yang et al., Rank2Reward: Learning Shaped Reward Functions from Passive Video, ICRA 2024

[5]. Yu et al., GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning, ICCV 2025

[6]. Han et al., Learning Prehensile Dexterity by Imitating and Emulating State-only Observations, RA-L

### Questions
No specific questions

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Multi-View Video Reward Shaping (MVR), a reinforcement learning framework that uses multi-view video-text similarity from pre-trained VLMs for reward shaping. Unlike image-based methods that misguide agents toward static poses, MVR captures dynamic motions and mitigates viewpoint bias by introducing a state-dependent reward shaping that provides visual guidance early in training and automatically decays as the agent masters the task. Experiments on HumanoidBench and MetaWorld show MVR outperforms prior VLM-based methods, confirming its effectiveness in learning both dynamic and static skills

### Strengths
- This paper improves prior methods that measure rewards solely through static image-text similarity by introducing video-text-based rewards.

### Weaknesses
- Since both HumanoidBench and MetaWorld rely on reward engineering for task performance, the proposed method does not fundamentally solve the problem of reward shaping.
- In Section 3, it is confusing and inappropriate to use the same function notation fff for both the off-the-shelf VLM’s text-similarity function and the function defined within the MVR framework, as they are conceptually different.
- The writing could be improved for clarity. For instance, in Section 4.1, two challenges—(1) the semantic gap between states and videos, and (2) differences in viewpoints—are mentioned. “Matching Paired Comparisons” appears to address the semantic gap, while “Regularizing State Representations” handles viewpoint differences. These two should be presented more in parallel for better readability.
- The performance gains are marginal compared to the baseline, especially on MetaWorld. The improvement attributed to introducing multi-view inputs, which is claimed as the paper’s main contribution, is not significant.
- The task selection criteria for each benchmark are unclear. It is necessary to specify how tasks were chosen. Moreover, in DreamerV3 + MetaWorld, the number of RL training environment steps required for convergence varies across tasks, yet the paper uniformly uses 1M training steps, making it difficult to fairly compare performance.
- In Figure 3b, adding multiple viewpoints does not show statistically significant improvement.
- Some related works are missing in the citations: [1], [2].
- Using only three random seeds is insufficient to demonstrate statistical significance. More seeds are needed for reliability.

**References**\
[1] Subtask-Aware Visual Reward Learning from Segmented Demonstrations, ICLR 2025.\
[2] ReWiND: Language-Guided Rewards Teach Robot Policies without New Demonstrations, CoRL 2025.

### Questions
- For 64-frame videos, this may be too short for locomotion tasks but potentially too long for simple manipulation tasks such as those in MetaWorld, where meaningful subtasks are limited. Does the video length affect downstream policy performance?
- In Figure 4, why is t-SNE used? Would it not be more intuitive to visualize visual observations corresponding to the top/bottom-N frames ranked by task reward or MVR-shaped reward instead?
- The experiments use only exocentric views. Are there ways—or any experiments conducted—to incorporate egocentric views (for humanoids) or wrist-camera views (for robotic arms)?
- The default reward function in MetaWorld is a manually engineered dense reward. Was training conducted instead using a sparse reward?

### Soundness
2

### Presentation
2

### Contribution
2
