# Open-World Reinforcement Learning over Long Short-Term Imagination

- Decision: Accept (Oral)
- Scores: 8, 8, 8, 8

## Abstract
Training visual reinforcement learning agents in a high-dimensional open world presents significant challenges. While various model-based methods have improved sample efficiency by learning interactive world models, these agents tend to be “short-sighted”, as they are typically trained on short snippets of imagined experiences. We argue that the primary challenge in open-world decision-making is improving the exploration efficiency across a vast state space, especially for tasks that demand consideration of long-horizon payoffs. In this paper, we present LS-Imagine, which extends the imagination horizon within a limited number of state transition steps, enabling the agent to explore behaviors that potentially lead to promising long-term feedback. The foundation of our approach is to build a $\textit{long short-term world model}$. To achieve this, we simulate goal-conditioned jumpy state transitions and compute corresponding affordance maps by zooming in on specific areas within single images. This facilitates the integration of direct long-term values into behavior learning. Our method demonstrates significant improvements over state-of-the-art techniques in MineDojo.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a long short-term world model to address the limited imagination horizon of traditional world models.
The authors leverage MineCLIP to detect task goals and generate intrinsic rewards, enabling the model to jump directly to these goal states when feasible.
For policy optimization over these jump-based transitions, they train modules to predict future value, interval timesteps, and cumulative rewards.

The paper is well-organized, and its key contributions are:
(1) It introduces a method for generating target states in MineDojo (or possibly in other 3D-RPG games).
(2) It demonstrates the feasibility of training a world model with jumping transitions and optimizing policies over such transitions.

### Strengths
1. It introduces a method for generating target states in MineDojo (or possibly in other 3D-RPG games).
2. It demonstrates the feasibility of training a world model with jumping transitions and optimizing policies over such transitions.
3. The illustration is clear and the writing organization is good.

### Weaknesses
1. Since MineCLIP is an important tool for this work, I suggest the author include a brief introduction of what MineCLIP can do in section 3.2/3.2.1 or an appropriate position. This would help readers who are not familiar with research on MineDojo to understand this paper.

2. In the abstract, "We argue that the primary obstacle in open-world decision-making is improving the efficiency of off-policy exploration across an extensive state space." This seems not closely connected to the main contribution of this paper. I suggest paraphrasing it to highlight "across a long horizon" or something that is more related to the topic.

3. Though the method sounds promising for solving MineDojo tasks, it may not be a general method for all kinds of open-world games. Such as in 2D games or fixed camera control tasks.

    (a)  Before seeing the target for the first time in the training process, there won't be a reasonable goal-directed reward or jumping option, the exploration still requires extensive enumerates.

    (b) The crop and resize operation (or assumption) is only useful for 3D visual navigation tasks.

    (c) When the target is occluded, the world model still needs to perform step-by-step imagination.

If these are true, I suggest the authors include a sentence or so in the limitation section to clarify it.

### Questions
1. Jumping transitions

    (a) I don't get the reason and rule for this dynamics threshold on lines 248 to 250. Could you clarify this?

    (b) The jumpy transition in the world models is a novel contribution, and I think it's worth more explanation for intuitive understanding. Like how many times would the jumping transition be triggered during the training, and what is the predicted average $\Delta'_t$ in the imagination (how many short-term steps are saved)?

2. I suggest adding a numerical result (possibly in the appendix) for the results in Figures 4 and 5.

3. For the parallel imagination in section 4.3. Is the post-jumping state connected with the bootstrapping $\lambda$-return of the pre-jumping states? How is the conclusion on lines 464 to 466 be made?

4. For the model-based RL review on lines 487-489, I suggest adding another two world model papers for completeness: (1) Zhang, Weipu, et al. "STORM: Efficient stochastic transformer based world models for reinforcement learning." Advances in Neural Information Processing Systems 36 (2024). and (2) Alonso, Eloi, et al. "Diffusion for World Modeling: Visual Details Matter in Atari." arXiv preprint arXiv:2405.12399 (2024). (NeurIPS24 Spotlight)

### Soundness
3

### Presentation
3

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
This paper introduces an exploration approach, LS-Imagine, to improve model-based reinforcement learning from pixels on open-world environments (with large state spaces). 

The paper aims to address the short-hoirzon limitation of model-based approaches relying on repeated autoregressive single timestep prediction. This is achieved by first generating affordance maps of observations by zooming into different patches and determining if there are objects associated with the current task in those patches using a vision-language reward model (MineCLIP in this work) distilled into an affordance map predictor. This affordance map is used to generate an intrinsic reward associated with the observation. The kurtosis of the affordance map is then used to determine if the object associated with reward is near or far away (binary, based on a threshold). If the object is deemed to be near, a standard 'short-term' world model using autoregressive single timestep prediction is used to generate trajectories for an actor and critic to learn from. Alternatively, if the object deemed to be far away, a 'long-term' world model using autoregressive multi-timestep interval prediction is used (although ignoring actions). This enables the world model to generate a trajectory over a larger number of environment timesteps in the same number of autoregressive prediction steps, therefore potentially enabling the generated trajectory to get closer to the object of interest. This long-term trajectory is used to train the critic but not the actor. The combination of these models is termed a Long Short-Term World Model, giving the approach its name: LS-Imagine.

LS-Imagine is applied to 5 MineDojo tasks and outperforms the baselines in terms of success rate and per-episode steps after training. Ablations demonstrate the benefit of both affordance-based intrinsic reward and the use of long-term imagination.

In summary, the paper attempts to tackle the important problem of long horizon world modelling and provides an interesting and novel approach to address this problem in 3D environments. While it is a relatively complex method and somewhat limited to Minecraft, I believe the ideas and insights warrant acceptance.

### Strengths
- **Significance**
  - Long-horizon world modeling and reinforcement learning in open-world environments are important problems.
  - The proposed approach is insightful and successfully addresses these problems.
- **Originality**
  - The proposed approach involves the combination of multiple novel and inisightful components.
- **Quality**
  - Overall the quality of the paper is relatively high, with the method reasonably clearly explained and analyzed.

### Weaknesses
- **Clarity**
  - Some aspects of the paper are not particularly clear. The main one is the use of the word 'jumpy' throughout the paper. The meaning of this word is assumed, but is not defined in the paper or standard usage as far as I'm aware, and is relatively unscientific, so I feel it is not the right word to use. 'Multi-step' state transitions seems more appropriate. If the authors were attempting to highlight that the number of steps can vary, then 'variable-step' transitions would be better. At the very least, 'jumpy' should be properly defined at the beginning of the paper.
  - Similarly affordance maps may not be familiar to all readers and the exact meaning of this term can vary. For example, a short clarification early in the paper such as ''...affordance maps, that elucidate which parts of an observation may be associated with reward, ..." would be helpful.
  - Some other unclear aspects/minor mistakes include:
    - L326: "employs *an* actor-critic algorithm to learn behavior *from* the latent state sequences..."
    - L351: Grammar is slightly wrong and confusing, should be: "Notably, since long-term imagination does *not* involve actions, we do not optimize the the actor when long-term imagnation is adopted." 
    - L354: Also worth highlighting the difference with the DreamerV3 loss. "The loss of the actor is therefore equivalent to DreamerV3, with an additional factor of $(1-\hat j_t)$ to ignore updates from long-term imagination:"
    - L361: "on the chellenging..."
    - L500: Doesn't make sense. Maybe "Our work is also related to affordance maps for robotics." is sufficient?
    - "*Learning to Move with Affordance Maps*" [1] should likely also be compared and cited here.

- **Limitations**

  - This approach has important limitations that are not mentioned. In particular, the approach is limited to embodied agents navigating a 3D environment in which there are objects associated for which reward is obtained by approaching them. Therefore the approach assumes, for example:
    - Observations are of a 3D environment
    - Actions are navigation actions of an embodied agent
    - Rewards are assoiated with identifying or moving towards objects
    - A reward model is available to identify high reward regions of observations
  - The experiments are limited to Minecraft for which these assumptions hold. This approach would likely not work as well even in Crafter [2] for example, which provides a 2D 'open-world' analogue of Minecraft, since objects do not become larger as you move towards them.
  - The approach also relies on both the long-term and short-term models being used, given only the short-term model is able to update the actor. While the thresholding of $P_{jump}$ can partially be used to address this, this is not particularly robust, and still requires some close-up objects in initial exploration for the standard one-step world model to be used, so the approach may not work as well in very sparse environments.
  - There is still significant value of the approach despite these limitations, and the paper is reasonably scoped, but they should be included in the limitations at the end of the paper, which are currently overly brief and narrow.

  **References:**

  [1] "*Learning to Move with Affordance Maps*", Qi et al., 2020

  [2] "*Benchmarking the Spectrum of Agent Capabilities*", Hafner, 2021

### Questions
- Why was the word 'jumpy' chosen? Could a more precise word be used instead?
- The authors say on L42 that MBRL approaches optimize short-term experiences of "typically 50 time steps". Where did 50 come from? My understanding is that MBRL methods such as DreamerV3 commonly use an even shorter horizon of $\sim 16$ timesteps.
- Could the limitations be expanded upon to cover the general application of LS-Imagine beyond Minecraft?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies Model-based RL in Open-World environment, specifically Minecraft. The authors propose a long short-term imagination method to improve the imagination process of the dreamer algorithm. The main idea is to use an affordance map to identify the distance object of interest and train a long-term transition model to skip the approaching steps so that the imagination can be more task-relevant. Results from 5 tasks in Minecraft showcase the improvement of the method over Dreamerv3 and other two baselines.

### Strengths
- The paper is mostly well written. 
- The method proposed is novel and the results are promising comparing to the baselines.

### Weaknesses
- Although the high-level idea is straight-forward, the implementation is overcomplicated. 
- The method feels very ad-hoc to the Minecraft tasks studied in this paper. It doesn't come into my mind about any other relevant tasks other than Minecraft where the proposed method can be applied.

### Questions
- The term "Gaussian matrix" in line 215 is confusing. Based on the context later, I think you are referring to Gaussian kernel. If that is correct, what is the $\sigma$ you use in the paper? Is the method sensitive to different value of $\sigma$?
- In Section 3.2.2, you mentioned the affordance map generator is pretrained on the random exploration data. How do you make sure that the task relevant observations are present in the random data to train a meaningful generator? For example, if the task is to collect a diamond, and there certainly won't be any diamond seen by the random agent. Will the method still help to solve the task?
- I am confused by the model analysis part with the parallel pathway. Things explained in Section 4.3 sound the same with Section 3.4. Could you elaborate what is the difference between the sequential pathway and the parallel pathway?
- Some relevant citations are missing: [1] studies the hierarchical world model which uses a similar strategy of doing the long-term imagination on the higher-level states; [2] studies a similar problem that short imagination horizon may not enough to cover behaviour of interest. I would suggest the author to discuss these works.
- There are no error bars for the middle and right plots in Figure 7. Are these results only based on one seed?
- What is the computation cost of this method?
- Maybe I missed some parts, but how does the policy learn the behaviour that the long-term imagination skip? For example, in the "Harvest log in plains" task, the go toward the tree is always skipped by the long-term imagination if the model learns well. Since the skip transition doesn't have gradient attached to the action, I wonder how does the policy learn to walk toward the tree.

[1] Gumbsch *et al.*, Learning Hierarchical World Models with Adaptive Temporal Abstractions from Discrete Latent Dynamics, ICLR 2024 

[2] Hamed *et al.*, Dr. Strategy: Model-Based Generalist Agents with Strategic Dreaming, ICML 2024

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
The paper introduces a novel hierarchical model-based reinforcement learning (MBRL) framework named Long Short-Term Imagination (LS-Imagine), designed to address the challenge of short-sightedness in open-world decision-making with high-dimensional visual inputs, such as MineDojo. In LS-Imagine, imagination horizon can be extended with a specialized world model. Specifically, a affordance maps are predicted to guide the jumpy switch. Two world models are trained to capture transitions at different temporal resolutions, short-term state transition and long-term state transition. Agent learning is conducted during imagination, a common strategy utilized in most background planning method.  The authors provided experimental results on Harvest tasks from the MineDojo benchmark, showed superior results compared with baseline methods.

### Strengths
1. The jumpy prediction technique within the long-term imagination framework is innovative as it departs from the fixed interval approach prevalent in previous work, offering increased flexibility in jumpy prediction
2. The paper is well-organized and clearly written.

### Weaknesses
1. The proposed method employs a hierarchical structure, yet the baseline comparisons are made with flat learning methods. Including comparisons with hierarchical MBRL methods like Director[1] could greatly strengthen the paper.
2. Equation 9 appears to have an inconsistency in the time indexing; should the bootstrapping term $R^\lambda_{t+1}$ be $R^\lambda_{t+\hat{\Delta}_{t+1}+1}$ ? 
3. The use of  $\lambda$ -return in evaluating the policy might introduce bias since it should be evaluated with on-policy data,, but the predicted jumpy state, $\hat{z}_{t+1}$, might not aligned with the learning policy.
4. The paper focuses on Harvest tasks. Including results from other complex, long-horizon tasks, such as the Tech Tree task group from MineDojo, would better demonstrate the framework’s effectiveness.

[1]: Hafner, Danijar, et al. "Deep hierarchical planning from pixels." Advances in Neural Information Processing Systems 35 (2022): 26091-26104.

### Questions
1. Could the authors elaborate more on how the long-term transition data is collected? The phrase from the appendix still confusing to me, specifically, how is the reward of the state helps measure the long-term transition.

### Soundness
2

### Presentation
3

### Contribution
3
