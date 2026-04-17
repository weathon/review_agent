# From Solo to Symphony: Orchestrating Multi-Agent Collaboration with Single-Agent Demos

- Decision: Reject
- Scores: 4, 2, 2, 4, 4

## Abstract
Training a team of agents from scratch in multi-agent reinforcement learning (MARL) is highly inefficient, much like asking beginners to play a symphony together without first practicing solo. Existing methods, such as offline or transferable MARL, can ease this burden, but they still rely on costly multi-agent data, which often becomes the bottleneck. In contrast, solo experiences are far easier to obtain in many important scenarios, e.g., collaborative coding, household cooperation, and search-and-rescue. To unlock their potential, we propose Solo-to-Collaborative RL (SoCo), a framework that transfers solo knowledge into cooperative learning. SoCo first pretrains a shared solo policy from solo demonstrations, then adapts it for cooperation during multi-agent training through a policy fusion mechanism that combines an MoE-like gating selector and an action editor. Experiments across diverse cooperative tasks show that SoCo significantly boosts the training efficiency and performance of backbone algorithms. These results demonstrate that solo demonstrations provide a scalable and effective complement to multi-agent data, making cooperative learning more practical and broadly applicable.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces "Solo-to-Collaborative RL (SoCo)", a framework designed to leverage single-agent (solo) demonstrations to accelerate multi-agent reinforcement learning (MARL) in cooperative settings. The authors argue that while MARL typically requires expensive multi-agent data, solo demonstrations are often more abundant and easier to obtain (e.g., in robotics or coding tasks). SoCo addresses challenges like observation mismatch and domain shift by: (1) pretraining a shared solo policy via behavior cloning on solo demos; (2) decomposing multi-agent observations into solo-compatible views during training; and (3) using a policy fusion module with a learnable gating selector (to choose among candidate solo actions) and an action editor (to refine actions via residual corrections). The framework is plug-and-play with CTDE-based MARL algorithms like MATD3 and HATD3. Experiments on nine tasks across four benchmarks (Spread, LongSwimmer, MultiHalfCheetah, MultiWalker) demonstrate improved sample efficiency and final performance.

### Strengths
The framework is a fresh angle compared to prior work focused on multi-agent data (e.g., offline MARL or multi-task transfer).

Results show consistent efficiency gains  and performance boosts, with ablations isolating component contributions (e.g., gating selector vs. random alternatives).

### Weaknesses
**Assumptions and Generalizability:** The framework assumes decomposable, structured observations, which may not hold in more complex or unstructured environments. It also relies on rule-based decomposition, limiting applicability to tasks without clear solo-multi alignments. Extension to stochastic policies or non-CTDE paradigms is mentioned but not demonstrated.

**Limited Scope of Baselines and Tasks:** While MATD3/HATD3 are solid, comparisons to other CTDE algorithms (e.g., QMIX, MAPPO) or recent transferable MARL   would strengthen claims. Using real-world offline data would help as well.

 **Theoretical Analysis:** No formal analysis of convergence or transfer guarantees, which    would add depth.

### Questions
The correction strength L is task-specific and requires tuning, potentially undermining plug-and-play claims. Ablations show sensitivity (e.g., small L over-relies on solos, large L underuses them), is it able to do the auto-tuning for it.

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
3

### Summary
This paper introduces Solo-to-Cooperative Reinforcement Learning (SoCo), a framework designed to leverage solo demonstrations to improve the efficiency of multi-agent reinforcement learning (MARL). SoCo first pretrains a shared solo policy from individual-agent data and then adapts it for cooperation through a policy fusion mechanism consisting of a gating selector and an action editor. Experimental results across multiple cooperative benchmarks demonstrate that SoCo accelerates convergence and achieves competitive or superior performance, showing that solo experiences can serve as a scalable and effective complement to multi-agent data.

### Strengths
1. The paper tackles a practical and underexplored challenge in MARL - how to utilize single-agent (solo) demonstrations to enhance multi-agent cooperation. This "solo-to-cooperative" perspective is conceptually novel and practically valuable for domains where cooperative data are costly or difficult to obtain.
2. The proposed SoCo framework is modular and algorithm-agnostic, allowing seamless integration with existing MARL methods (e.g., MATD3). This design highlights strong flexibility and reusability, making SoCo a potentially general framework for future MARL research.

### Weaknesses
1. The experimental environments used in the paper are not fully representative of mainstream MARL benchmarks. Evaluating SoCo on widely recognized cooperative environments such as SMAC or MaMuJoCo would significantly strengthen the empirical evidence and better demonstrate its capability in handling complex coordination scenarios.
2. The paper lacks comparisons with recent advances in transferable reinforcement learning and multi-agent imitation learning methods. Without such baselines, it is difficult to judge the generality and competitiveness of SoCo relative to the state of the art. Incorporating methods from the past year would make the empirical study more comprehensive and convincing.
3. The notations in Figure 1 are inconsistent with those in the main text, in particular, the figure uses subscripts starting from 1, while the text uses k = 0. Unifying the notation and indexing conventions would improve clarity and technical precision.

### Questions
1. Could the authors provide a more concrete or simplified explanation of the observation decomposition process? A small illustrative example would greatly help clarify how observations are partitioned into solo views and used in the SoCo framework.
2. The authors claim that in the 2-agent MultiHalfCheetah environment, the absence of multi-objective tasks allows for an "effective isolation of the gating selector's influence."Under this environment, does the decomposition process still generate multiple solo views? Please clarify this point.If the gating selector is not truly isolated, please explain the following:When L = 0, the MultiHalfCheetah experiment yields near-zero returns, indicating that direct transfer of solo policies fails due to domain shift. However, in the 3-agent Spread ablation study with the same setting (L = 0), the performance remains high. These two results seem inconsistent - could the authors clarify the underlying reason for this discrepancy?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents Solo-to-Collaborative RL (SoCo), a framework designed to leverage single-agent (“solo”) demonstrations to accelerate cooperative multi-agent reinforcement learning. The premise is well-motivated.. it is often easier to obtain solo demos than multi-agent demos. However, solo demos do not show any cooperation even if they do have signals on how each agent should act. The key technical challenge is how to determine when to mimic the solo demos and when to learn to cooperate. The method first trains a shared solo policy via imitation learning, then during multi-agent training decomposes each agent’s local observation into multiple solo-compatible views and reuses the pretrained policy to generate candidate actions. These candidates are fused through a learnable module consisting of a gating selector (for resolving ambiguity and selecting a solo action) and an action editor (which applies residual corrections to adapt to cooperative dynamics).

The method is evaluated in simulation comparing two MARL algorithms with and without SoCo. The results show improved speed of convergence and better convergence with the SoCo. Overall, this is a well written paper and explores and well-motivated idea. However, I have reservations about the limited empirical evaluation and no comparisons with other methods (including PegMARL) that address a similar problem.

### Strengths
- The empirical results clearly show improvement with the SoCo framework on top of two existing MARL algorithms.
- The paper addresses a well motivated and under-studied problem and the approach seems sound. 
- The idea of action editor is interesting and perhaps can find applications beyond the solo-to-multi setting addressed in this paper. The key challenged addressed in this paper is domain shift and the action editor idea can be used in other settings where domain shift can be observed.

### Weaknesses
My main reservation is the lack of comparisons with other baseline methods and limited evaluation. For example, the paper claims that PegMARL (Yu et al., 2025) "assume sufficient multi-agent data" but that is not true. Most of the experiments in the PegMARL paper are with single agent demonstrations (see results in Sec 5.1). In fact, it seems that PegMARL would be a more general version since it can handle both single agent as well as multi-agent demonstrations (Sec 5.2). Given that, it seems that SoCo should be benchmarked against PegMARL.

The empirical evaluations also only consider two MARL algorithms (with a similar flavor) and compare those with and without SoCo. It would seem appropriate to benchmark with other MARL algorithms (e.g., MAPPO) and other simpler baseline algorithms (naively using single agent demos to bootstrap learning multi-agent policies). 

The paper also seems to focus on cases where the agents are homogenous. It would be interesting to discuss whether the framework extends to heteregenous cases where agents may have specialized capabilities or roles to play.

### Questions
- How does SoCo compare with other methods solving similar problems including PegMARL?

- How does SoCo compare with other MARL algorithms such as MAPPO? 

- How easy/difficult is it to disentangle joint observations into solo ones? It seems like this is done by hand, if so, how can this generalize to other MARL settings?

- How does the quality of the demonstrations affect the performance?

- Can you comment on whether SoCo extends to heterogeneous agents with different observation spaces, action spaces, or specialized roles?

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
3

### Summary
This paper proposes Solo-to-Collaborative RL (SoCo), a framework for leveraging solo (single-agent) demonstrations to accelerate multi-agent reinforcement learning. SoCo pretrains a shared solo policy via behavioral cloning, then adapts it during cooperative training through a policy fusion mechanism combining a gating selector (for choosing among candidate actions) and an action editor (for refining actions via residual corrections). The approach is evaluated on nine cooperative tasks across four scenarios (Spread, LongSwimmer, MultiHalfCheetah, MultiWalker), showing improvements in training efficiency and final performance compared to MATD3 and HATD3 baselines.

### Strengths
1. The paper addresses a practically relevant scenario where solo demonstrations are easier to collect than coordinated multi-agent trajectories in domains like collaborative coding, household robotics, and search-and-rescue.

2. The paper is generally well-written and easy to follow.

3. The policy fusion mechanism is intuitive, addressing goal ambiguity through the gating selector and domain shift through the action editor, with a modular design that allows flexibility for different scenarios.

### Weaknesses
1. The paper cites PegMARL (Yu et al., 2025), which is incorrectly characterized. According to its abstract, PegMARL utilizes "personalized expert demonstrations" that are "tailored for each individual agent" and "solely pertain to single-agent behaviors without encompassing any cooperative elements," which is functionally identical to what this paper calls "solo demonstrations." This is an important baseline that should be compared experimentally.

2. The approach requires "well-defined, structured, and decomposable" observations (page 4, lines 189-193), which can be restricted: it cannot handle unstructured observations (e.g., raw images, point clouds) and requires manual design of decomposition rules for each environment. Additionally, the method requires tasks to be decomposable into solo subtasks. For example, in a multi-robot box-lifting task, even if observations are decomposable, the task itself is not—a single robot cannot meaningfully lift the box alone. How can solo demonstrations be collected for such fundamentally non-decomposable tasks?

3. The gating selector is defined as g^i_φ : O_i → R^{G_i} (Section 3.3.1, line 220), where G_i is the number of solo views for agent i. However, Gumbel-Softmax requires a fixed output dimension. If different local observations o^i_t correspond to different numbers of solo views, how does the gating selector handle variable output dimensions?

4. Does SoCo only support continuous action spaces? The action editor performs residual adjustments (Equation 4-5), which only makes sense for continuous actions; and all experimental environments seem to use continuous action spaces. If so, this is a significant limitation that should be explicitly stated. If not, how does it work for discrete action spaces?

5. Inference appears computationally expensive, requiring: (1) local observation decomposition, (2) solo policy forward pass, and (3) policy fusion (gating + action editor), rather than a traditional single forward pass through a policy network.

6. The training curves appear to compare only the joint training phase. However, SoCo uses 1M transition samples to pretrain the solo policy, while MATD3 and HATD3 train from scratch. If this pretraining cost is considered, the performance gaps in Figures 2d-2i may not be significant enough.

7. The evaluation environments seem to be relatively simple, mainly particle environments and multi-agent MuJoCo. How would the approach perform on more challenging benchmarks like Google Research Football or SMACv2?

### Questions
see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose to collect trajectories from the single-agent counterparts of multi-agent tasks, use behavior cloning (offline reinforcement learning) to first obtain a shared prior policy $\pi_{\text{solo}}$, and then let each multi-agent actor fine-tune based on $\pi_{\text{solo}}$ to train on the multi-agent tasks. The authors demonstrate the superiority of their method on many continuous tasks.

### Strengths
The paper takes a novel perspective, proposing to use a more general single-agent task corresponding to a multi-agent task as a kind of initialization for the multi-agent problem.

It also shows good empirical performance, requiring fewer training steps within the multi-agent environments.

### Weaknesses
**Shortcomings in the comparative experiments:**

1. Since the method is positioned as a “plug-in,” it should be combined with more backbones to demonstrate extensibility. For the authors’ continuous-action settings, baselines such as MADDPG and SAC should be included.

2. The paper claims the method can extend to discrete-action environments, yet all reported experiments are on continuous control. The authors should add concrete results on discrete-action tasks.

3. I would like to see ablations on the action candidates—for example, a setting that uses only the agent’s own decomposed observation ooo as the **sole** action candidate.

4. In a sense, the method reuses a single-agent initialization for multi-agent models. It is understandable that this leads to significantly faster convergence than the baselines. However, the **final** performance improvements appear less pronounced (although accelerating convergence is itself a strong advantage). I would like to see, for Figure 2 (g, h, i), how much improvement remains when training to full convergence.

### Questions
see weakness.

### Soundness
2

### Presentation
3

### Contribution
3
