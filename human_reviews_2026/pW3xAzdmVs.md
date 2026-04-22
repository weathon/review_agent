# SLA-v3: Spatial Linkability-Aware and Novelty-Encouraging State Heuristic for Exploration

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 6

## Abstract
Efficient exploration continues to be a pivotal challenge in reinforcement learning (RL), particularly in environments characterized by sparse rewards. While intrinsic motivation (IM) has proven effective for tackling hard exploration tasks, current IM approaches often struggle with the detachment-derailment (D-D) problem. This issue significantly curtails their effectiveness, especially in settings with extremely sparse rewards. Although methods like Go-Explore address D-D by explicitly archiving states to ensure revisitation, their dependency on state restoration limits their practical application in procedurally generated environments. In this paper, we argue that the root cause of the D-D problem lies in the underlying topological transition structure of the environment. Specifically, we observe that certain states become persistently difficult to traverse and revisit reliably when subjected to exploratory noise. To overcome this, we introduce a novel IM framework centered on state traversal difficulty. Within this framework, we propose the $\textbf{S}$patial $\textbf{L}$inkability-$\textbf{A}$ware $\textbf{a}$nd $\textbf{N}$ovelty-$\textbf{E}$ncouraging $\textbf{S}$tate $\textbf{H}$euristic ($\textbf{SLAANESH}$), abbreviated as $\textbf{SLA-v3}$. SLA-v3 tackles the D-D problem by utilizing the shortest-path quasi-metric from the initial state ($S_0$) as a heuristic for traversal difficulty. This mechanism generates sustainable exploratory incentives, particularly encouraging visit to hard-to-traverse states. Furthermore, SLA-v3 integrates a novelty detector, which serves to warm up the heuristic and effectively prevent stagnation in unproductive dead-end paths. Our extensive experimental evaluations on MiniGrid and challenging Atari environments (PitFall! and Montezuma's Revenge) robustly demonstrate the superior efficacy of SLA-v3.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents an intrinsic reward method that encourages the RL agent to visit state which are hard to reach under random exploration. This property of a state is modelled by training a model to predict the timestep corresponding to the first occurrence of a state in an episode. The proposed intrinsic reward then guides the agent towards states that are hard to reach as predicted by this model and once this is achieved, it decreases, by design, to allow for exploration at the frontier of the agents state visitation distribution. Overall, the idea is sound and the motivation to draw inspiration from Go-Explore while attempting to get rid of the assumptions of the latter is relevant. The experiments presented and the evaluation protocols are rigorous and extensive although sometimes not adhering to standards (especially in terms of baselines compared and environment modifications).

### Strengths
The combination of the proposed quasi-metric for state reachibility with the design of an intrinsic reward to appropriately maximize it is relevant and well executed. While similar ideas have been proposed in prior work (e.g. Go-Explore), this paper makes a step forward in proposing a method which relies less in specific properties of the environment to achieve good performance (e.g. like the ability to reset the environment to a desired configuration).

Although I have some issues with the formatting of the paper (e.g. citations, wording, clarity), the ideas presented flow nicely and are organized appropriately throghout the paper, allowing the reader to somewhat easily follow. The Figures are well designed and also help. The mathematical principles used to design the reward function are rigorously presented and clear.

The method achieves good performance in MiniGrid and 2 hard-exploration Atari environments (although modified). The experiment protocols are appropriate and the results provide statistically significant conclusions. The authors thoroughly analyzed the behavior of the agents in the environments under different conditions (noise, sticky actions), and learning dynamics under different hyperparameters, which provides valuable insights into the performance of SLA-v3.

### Weaknesses
In terms of formatting, the current citation style makes it hard to read the paper. I reocmmend using \citep{} instead of \cite{} unless directly referencing the authors of a paper.

I find this statement unclear: "Intrinsic Motivation [...] enhances the reward signal by leveraging historical experience replay, thereby facilitating more effective policy optimization". IM methods don't have to forcefully use historical experience replay, and hence it is not a good definition of these methods. 

Generally, I recognize AI-generated text in several sentences of the paper, which although is not necessary a bad thing, and knowing that authors ackonwledge this in the Appendix, I personally dislike it and find it less clear at times. I would suggest making the effort of writing original text to improve the clarity of the presentation.

The paper omits relevant citations to prior work in using temporal distances in reinforcement learning [1,2] (similarly used in this work to define the quasi-metric for state reachibility), in the state-dependent reachibility properties in MDPs [3,4] (see the line of work in *empowerment* in RL), and intrinsic motivation [5] - only to list a few papers. 

The paper does not justify some design decisions of the presented method: e.g., why is RND chosen as the additional source of novelty? why not a different intrinsic reward method?

How and why were the baselines selected? Other open-source intrinsic reward algorithms which have been evaluated in both MiniGrid and Atari environments were omitted (e.g. Disagreement, ICM, E3B, NGU...) [5]. Showing that SLA-v3 outperforms these in standard evaluation settings would help strengthen the claims of the paper a lot. 

The Pitfall and MR games from Atari were modified to include additional information (e.g. room index) which can dramatically facilitate the exploration problem. Why weren't the aforementioned baselines not evaluated in these settings as well?

Finally, I believe it would be interesting and very relevant to broaden the evaluation of the algorithm to other popular environments like ProcGen, other hard-exploration Atari games, or Crafter, while comparing the performance of SLA-v3 with currently omitted but highly relevant baselines.

With this, I think the paper presents an interesting method which is rigorously motivated and introduced, but lacks clarity and provides minimal evidence of the value the algorithm could yield (e.g. superior performance compared to well-established baselines). I think improving these 2 points will render the paper publishable.

[1] Park, Seohong, Oleh Rybkin, and Sergey Levine. "Metra: Scalable unsupervised rl with metric-aware abstraction." arXiv preprint arXiv:2310.08887 (2023).

[2] Stooke, Adam, et al. "Decoupling representation learning from reinforcement learning." International conference on machine learning. PMLR, 2021.

[3] Latyshev, A. K., and A. I. Panov. "Skill Learning with Empowerment in Reinforcement Learning." Pattern Recognition and Image Analysis 34.3 (2024): 535-542.

[4] Eysenbach, Benjamin, et al. "Diversity is all you need: Learning skills without a reward function." arXiv preprint arXiv:1802.06070 (2018). 

[5] Yuan, Mingqi, et al. "Rlexplore: Accelerating research in intrinsically-motivated reinforcement learning." arXiv preprint arXiv:2405.19548 (2024).

### Questions
Why is RND chosen as the additional source of novelty? Why not a different intrinsic reward method?

How and why were the baselines selected? Many open-source, relevant intrinsic motivation baselines, which have been previously evaluated on the environments used in this paper were omitted [1].

[1] Yuan, Mingqi, et al. "Rlexplore: Accelerating research in intrinsically-motivated reinforcement learning." arXiv preprint arXiv:2405.19548 (2024).

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper shows that an intrinsic reward based on a model of the time distance from the start can be very effective on some really sparse-reward games such as Montezuma's revenge and Pitfall.

### Strengths
- very nice introduction and related work section

- excellent experimental results on difficult RL problems: in particular on Montezumas Revenge and Pitfall games

- the proposed intrinsic reward defined in Eq. (4) is interesting: you get rewarded for states that are further away from the start than all other states along the trajectory, so practically avoiding going backwards.

- Eq. (5): combines the intrinsic reward of Eq. (4) with novelty detector (here Random Network Distillation, RND).  Basically, the intrinsic reward from Eq. (4) modulates the intrinsic reward from a novelty detector.

### Weaknesses
- the "shortest-path quasi-metric" suggests that there is some graph-theoretic shortest path involved, however, the heuristic just learns the time index.  don't get me wrong, I like simple ideas and solutions, but then why call it "shortest-path quasi-metric"?  I see that you use it to motivate your idea, the SLA heuristic, but then you do not use it at all (and instead just the time index).

- you only explain the intrinsic reward that you want to use (Eq. (5)), but then how do you create the whole RL algorithm with it.  That part is completely missing!  

- also what model/network are you using for $H_θ$?

- figure 1 is confusing:  is it generated?  the overlay grayed-out neural network makes no sense and looks like an indicator for AI-generated content.

- you didn't mention whether and how you used LLMs for writing the paper!

### Questions
- the title: Spatial Linkability-Aware and Novelty Encouraging State Heuristic.  Why so complicated!  Aren't you just encouraging to explore states far away from the start?  What is linkability-awareness?

- Figure 1, left: the easy-to-traverse states are close to the start, the hard-to-traverse states are far away.  Aren't there many many more hard-to-traverse states in high dimensions?  Or are you using some special properties of the state topology, that there are not so many hard-to-traverse states?

- Figure 1, middle: what are the boxes in the pink box?  why a box with "Architecture Choice"?  What do the arrows mean?  The middle panel looks automatically generated with an LLM (e.g., the grayed-out overlayed neural net).  It doesn't make much sense to me!

- Figure 1, right: similar, is this just a generated figure?  I find it confusing!  Why the "Agent" blue discs on the arrows?  Why converging arrows to "Novelty?  Looks all fancy, but does it really mean anything?

- Figure 1, caption: also the caption isn't a real sentence.  There is an extra verb "has been reached".???

- line 201: you write that your agent switches from **go** to **explore**?  when?  how?  you don't show us the whole algorithm!

- is the shortest-path quasi-metric a "quasi-metric"?  what are the requirements to be a quasi-metric?

- You define in Eq. (1) the SLA Heuristic.  Where do you use it?

- For the SLA Heuristic, you have to define equality of states.  Are you using the equality on pixalated images (like GoExplore)?  Or how do you do that?  E.g. in Montezuma's revenge there is a skull going left and right, so two states are rarely exactly the same...

- Line 180: "the ground truth shortest-path quasi-metric serves as an up-
per bound t".  Isn't that the other way around?

- Also, on the RHS in Eq. (1) you write $s_t$.  But how is that variable bounded?  What is it?  Does it range over all possible states of all possible trajectories you have seen so far?  Or over all theoretically possible trajectories?  The $\min$ should specify what it is ranging over.

- what is $H_θ$ in Eq. (2)?  I know it is your heuristic model, but you do not mention  that connection in the text.

- line 193: what is "functionally equivalent"?  you mean the value model in PPO has the same (type) signature as the heuristic model?  Or do they compute similar things?

- line 196: why is no neural architecture search necessary?  Maybe learning $H_θ$ requires a different architecture than the value model of PPO.

- Figure 7: do have the curve of Go-Explore?  you just show that you reach and exceed the "Go-Explore Final" score.  But are you reaching it faster?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the Detachment-Derailment (D-D) problem in sparse-reward reinforcement learning. The authors argue that the root cause of D-D lies in the environment's topological transition structure, where some states are inherently difficult to traverse and revisit.

They propose SLA-v3, a novel intrinsic motivation (IM) framework centered on a "state traversal difficulty" heuristic. This heuristic, H_{sla}, is defined as the shortest-path quasi-metric (minimum number of steps) from the initial state S_0. The method learns to approximate this metric by training a heuristic model H_{\theta} to predict the minimum episodic timestamp (step count) for each state, using a downward-biased loss. The intrinsic reward encourages visiting states with higher heuristic values than those seen so far in the episode. This is combined with a novelty detector (RND) to prevent stagnation.

The method is evaluated on MiniGrid and the challenging Atari environments (PitFall!, Montezuma's Revenge), demonstrating strong performance against IM baselines.

### Strengths
1. **Clear Problem Motivation:** The paper provides a clear definition of the Detachment-Derailment (D-D) problem, grounding it in the environment's topological structure.
2. **Novel Heuristic:** The core idea of using the shortest-path quasi-metric from the initial state as a heuristic for "traversal difficulty" is novel, intuitive, and a sensible approach to tackling the D-D problem.
3. **Practical Approximation:** The method for learning this heuristic by approximating the minimum episodic timestamp from a replay buffer is a practical and clever approach. The downward-biased loss (Eq. 2) is well-suited for this minimum-seeking objective.
4. **Strong MiniGrid Analysis:** The method shows compelling performance in procedurally generated MiniGrid environments. The visualization in Figure 3 provides strong evidence that the learned heuristic captures a meaningful representation of task progress, showing monotonically increasing values at key sequential stages (e.g., key acquisition, door unlocking).

### Weaknesses
1. **Mismatch between Heuristic and Stochastic Environments:** The core SLA heuristic (H_{sla}(s) = d(s_0, s)) is defined as the shortest path, a concept well-defined in deterministic settings. However, the paper evaluates this in stochastic environments (e.g., sticky-action PitFall) and claims it "remains effective". The theoretical justification for this is weak. In a stochastic environment, a state might be reachable in 10 steps (with 1% probability) and 50 steps (with 99% probability). The heuristic's loss function (Eq. 2) is designed to optimize towards the minimum (10 steps). It is unclear why this optimistic, minimum-seeking objective is the most appropriate or stable choice for a stochastic setting, as opposed to an expected path length or a more risk-aware metric.*
2. **Reliance on Domain Knowledge in Atari:** The paper rightly argues against Go-Explore's dependency on state restoration, which limits its applicability. However, the impressive results on PitFall! and Montezuma's Revenge also "integrate carefully designed domain knowledge," specifically "room indices and agent positional information". This reliance on privileged information significantly weakens the claim of a general, domain-agnostic solution and makes the comparison to other methods less direct. It raises the question of how much of the exploration challenge is being solved by this domain knowledge versus the heuristic itself.
3. **Incomplete Literature Review and Missing Citations:** The related works section (2.1), while covering the main paradigms, omits several significant and foundational papers in the exploration space. This incomplete review of the literature weakens the paper's positioning and fails to properly contrast the proposed method with other established approaches. Specifically, the following key citations are missing:
   * **Foundational Novelty/Diversity Methods:** 
     * Tang et al. (2017). "# exploration: A study of count-based exploration for deep reinforcement learning." Advances in neural information processing systems 30.
     * Hong et al. (2018). "Diversity-driven exploration strategy for deep reinforcement learning." Advances in neural information processing systems 31.
   * **Alternative Exploration Frameworks:**
     * Jin et al. (2020). "Reward-free exploration for reinforcement learning." International Conference on Machine Learning. PMLR.
   * **Alternative State Metrics:**
     * Wang et al. (2023). "Efficient potential-based exploration in reinforcement learning using inverse dynamic bisimylation metric." Advances in Neural Information Processing Systems 36.
   * **Recent Related Work:**
     * Hao et al. (2025). "Llm-explorer: A plug-in reinforcement learning policy exploration enhancement driven by large language models." Advances in Neural Information Processing Systems 2025.

### Questions
* (Re: Weakness 2) How critical is the domain knowledge (room indices, agent position) to the performance in the PitFall! and Montezuma's Revenge experiments? How does SLA-v3 perform compared to baselines (e.g., RND) if this domain knowledge is removed and all methods must learn from pixels alone?
 * (Re: Weakness 1) The heuristic H_{\theta} is trained to approximate the minimum observed timestamp. In a highly stochastic environment, this objective seems optimistic. Could the authors comment on how this minimum-seeking objective affects policy learning and stability compared to a potential average-seeking objective (e.g., approximating the mean timestamp)?
 * The method combines a PPO-based RL algorithm (with separate value functions), the SLA heuristic model H_{\theta} (trained with Eq. 2), and a novelty detector (RND, which has its own loss). Could the authors please provide the exact formula for the overall optimization objective used in the implementation, showing how all these different loss components are combined and weighted?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the Detachment-Derailment problem in sparse-reward exploration. The proposed solution, SLA-v3, utilizes a traversal difficulty heuristic (shortest-path distance from $S_0$) to incentivize visiting these difficult states. This method achieves strong results on hard benchmarks (MiniGrid, Atari) without relying on state restoration.

### Strengths
- Strong performance on challenging hard-exploration benchmarks
- intuitive design (heuristics)
- clearly written, well-motivated

### Weaknesses
- The paper's primary motivation is to overcome Go-Explore's reliance on state restoration (teleportation). However, the Go-Explore paper itself introduced a "policy-based" (goal-conditioned) variant specifically to address this limitation, which also does not require state restoration. This paper appears to overlook this highly relevant baseline.
- The central argument against state restoration is its impracticality in real-world applications, with robotics being the most prominent example. The paper's claims of practical applicability would be  strengthened by demonstrating SLA-v3's effectiveness in such a domain (e.g., solving Franka Kitchen-gym tasks or other manipulation/locomotion challenges; hard-exploration robotics benchmark).

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3
