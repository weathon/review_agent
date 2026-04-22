# Offline Multi-Agent Reinforcement Learning via Sequential Score Decomposition

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Offline cooperative multi-agent reinforcement learning (MARL) faces unique challenges due to the distribution shift between online and offline data collection. While online MARL typically converges to a single coordinated joint policy, offline datasets are often mixtures of diverse cooperative behaviors, resulting in highly multimodal joint behavior distributions. In such settings, independent policy regularization often misaligns joint policy contraints and leads to severe distribution shift. To address this, we propose OMSD, which sequentially decomposes the joint behavior policy into individual conditional distributions and leverages diffusion-based generative models to provide modality-coordinated regularization for each agent. Combined with centralized critic guidance, OMSD achieves coordinated exploration within high-value, in-distribution regions, and avoids out-of-distribution joint actions. Experiments across multiple datasets on various continuous control tasks demonstrate that OMSD consistently achieves state-of-the-art performance, especially in challenging multimodal scenarios. Our results highlight the necessity of modality-aware coordination for robust offline MARL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors focus on and attempt to solve a core limitation in most approaches to offline cooperative multi-agent reinforcement learning. Current methods that regularize each agent independently to match dataset behavior fail completely on multimodal joint policies. The authors coin the term Combinatorial Mode Shift exposing the exponential explosion of inconsistent joint actions when multimodal behaviors are factorized across agents. There are a number of clear contributions that the authors have made. The first is the formalisation of CMS and the associated theoretical analysis. The second is the redefining of the score decomposition into the SSD. Then the OSMD algorithm based on this with each agent learning a conditional diffusion model estimating the gradient of log mu_i. The final contribution is the empirical results showing superior performance to BRPO and diffusion-planner baselines by a large margin. Given the theory, it is not surprising that the improvements are largest on multimodal and low-quality datasets.

### Strengths
CMS is a well-reasoned theoretical articulation of a practical and important pathology in offline MARL.
The sequential conditioning is a small minimal structural change that is shown to prevent mode explosion. Using the diffusion model for score estimation rather than generation is also well-principled. The performance improvements show up where expected
The visualizations and toy examples clearly show why existing methods fail as well as how OMSD fixes it.
Many papers in MARL do not include any error bars on results, which is a huge failing in the community, and so the fact that this has been done consistently is a big plus.

### Weaknesses
There are a number of important weaknesses

OMSD’s reliance on a fixed agent order introduces a potential symmetry breaking. In symmetric teams, a fixed order could bias solutions toward asymmetric equilibria but in heterogeneous teams, order could encode hierarchies. So different orders may converge to distinct local optima of comparable quality. There doesn't seem to be any analysis of learned or time-varying orderings, which could preserve symmetry while still keeping the coordination. A formal discussion of when ordering choice matters would be an important improvement.

All experiments involve ≤ 4 agents, which is much less than most real MARL applications. The paper doesn't give any data on wall-clock cost, memory usage, or performance beyond very toy domains.

It seems that during training, agent i optimizes against deterministic prefixed actions from current policies but at test time, all agents act simultaneously and stochastically. Therefore, the training distribution differs from execution. There isn't any justification (in the form of theory) that explains why this sequential BR training gives coherent parallel execution. Although the experiments indicate that it does work, it would strengthen the argument to figure out why.

The centralized IQL critic is an important weakness. Poor critic estimates would be able to propagate through all the agents. The method would maybe benefit from conservative or ensemble critics and diagnostics for critic reliability.

I don't think that it's critical, but it would be useful if it could be proved that SSD prevents the exponential mode growth in general continuous settings rather than just showing it empirically in some limited domains.

In addition, there are no experiments beyond 4 agents or on discrete tasks (SMAC). There are no ablations for randomized ordering, diffusion noise schedules, or runtime cost. There is no test on unimodal datasets to confirm graceful degradation.

Finally, the authors seem to use ideas, if not code, from Coordination Failure in Cooperative Offline MARL by Tilbury et al, but it is not cited.

### Questions
In addition to the points in the weakness, diffusion models clearly provide good score estimates but generally need a lot of compute.  Why is there no comparison to simpler density estimators?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper considers an offline multi-agent RL problem. In particular, the authors study a challenge setting where the joint behavior policy is not unique, e.g., multi-modal. The authors show that traditional policy decomposition with independent assumptions might fail to provide a meaningful referencing policy when learning a new policy via the offline datasets. To better capture the structure of the joint policy, a new decomposition rule/structure is proposed. Specifically, the agents's policy can depend on other agents' actions. A diffusion based neutral network framework is considered to learn the policy. Experiments show that the proposed method outperforms other decomposition methods without the extra conditional "actions".

### Strengths
I think the problem setting is interesting. The decomposition rule in the literature sometimes might over-simplify the relations among agents. As a result, the learned policy for each agent could be quite sub-optimal. Besides, a lot of methods do regularize the new policy using the behavior policy, which could introduce bias and errors. Diffusion models are used to better learn the relations, i.e., the policy condition on certain agents' actions. I believe that different policy architecture could make a difference in the final results.

### Weaknesses
The paper is not very easy to follow, especially the motivations are not well-explained in the introduction and in earlier sections. I lost a bit of the flow when reading it. Some math definitions are not well-documented, and there are some technical flaws. The multi-modal aspect of this MARL problem is not well tested in the experiments. The overall idea of conditioning all agents' actions is not very new.

### Questions
- Q function:the Q function defined in line 163 is simply wrong. the action $a_1$ is not sampled from a given policy, it should be a given fix action. The following actions after $a_1$ are sampled from the policy $\pi$.

- Line 122-123: what does the $\partial R ^x$ mean? Is the reward function differentiable? Is this assumption necessary? For instance, you work is more on the application side with two small theorems. Did you use this assumption in the theorems? In the experiments, do the problem settings satisfy this assumption?

- Figure 1: what is the problem setting for these two agents? It is not clear from the contexts.

- Agent identity: in the paper, you mentioned that one source of multi-modal could come from the loss of agent identities. In you simulation, are agents indexed? If not, how would you define "prefix".

- Performance gain: Can you comment on how much more resources, .e.g, gpu training time and inference time, are needed comparing to traditional decomposition method? Does your method scale well if you increase the number of agents? Which part is the most difficult/intense in the training? Is it the diffusion model?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes OMSD (Offline MARL with Sequential Score Decomposition), a framework designed to address key challenges in Offline MARL. 
The paper first points out a fundamental limitation of existing Offline MARL algorithms: 
they often rely on a policy factorization assumption. 
This assumption is frequently violated, as offline datasets inherently possess a 'multi-modal' characteristic, mixing various expert strategies. The paper argues that this incorrect assumption ignores complex cooperative relationships and leads to a critical distribution shift, which it names Combinatorial Mode Shift (CMS).
To solve this problem, OMSD introduces 'sequential decomposition'. 
This models the agents' actions as a sequence, where each agent decides its action after observing the actions of its predecessors, 
rather than acting simultaneously. 
This sequential behavior model ($\hat{\mu}$) is learned using a score-based diffusion model. 
The CTDE policies are then updated by (1) maximizing the total team reward (Q-value) and (2) regularizing the policy to remain close to the learned sequential behavior model.

### Strengths
A key strength is the paper’s novel use of a score-based diffusion model to address the Combinatorial Mode Shift (CMS) problem, enabling the policy to capture multi-modal cooperative behavior from offline data.

### Weaknesses
- Lack of Novelty

The paper’s novelty is somewhat limited. The core challenge, termed CMS, is a well-known issue of multi-modality in offline MARL. [1] Furthermore, the learning objective itself is a standard offline RL formulation.
The main contribution seems to be the application of this objective to a sequential diffusion policy for behavior policies $\mu$.
The different point from existing methods is the specific policy update rule (eq. (4)).

- Diffusion Behaivor policy

The proposed algorithm relies on the learned behavior policy (the data-collecting policy) using a diffusion models. 
This approach neccessitates a multiple learned network, including a joint Q, behavior policy and CTDE policy, which significantly increases training complexity. 
While the diffusion policy is to capture the agent-dependencies, it may not be the most efficient method for this task. 
Diffusion models are highly expressive generative models; employing such a high-capacity model soley to represents the behavior policy seems uncessarily complex.

[1] AlberDICE: Addressing Out-Of-Distribution Joint Actions in Offline Multi-Agent RL via Alternating Stationary Distribution Correction Estimation

### Questions
See Weakness

### Soundness
3

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
4

### Summary
The paper introduce a new MARL algorithm using reverse KL-divergence as a regularization term to handle multi-modality in diverse datasets. To handle the distribution shift issue due to policy factorization in regularization term commonly used in offline MARL literature, the paper proposed sequential decomposition of the behavior policy and used that to train decentralized policies for individual agents. Experimental results show that the proposed MARL algorithm outperforms substantially other offline MARL methods on the MPE benchmark.

### Strengths
1. The paper provides a detailed theoretical analysis of distribution shift arising when joint policies are factorized into local marginal policies. This analysis effectively highlights the limitations of existing offline MARL methods that rely on such factorization to define their learning objectives.

2. The proposed sequential decomposition of behavior policies within the regularization term is a promising direction. The empirical results on the MPE environment demonstrate notable performance gains over state-of-the-art baselines.

### Weaknesses
1. The proposed algorithm demonstrates strong performance primarily on simpler environments (e.g., MPE). However, on more complex benchmarks such as MaMuJoCo, it outperforms the baselines in only 5 out of 9 tasks. This raises concerns about the scalability and general effectiveness of the method in more challenging multi-agent settings, including MaMuJoCo and potentially other complex domains such as SMAC.

### Questions
1. As you show in the paper that, policy factorization can lead to a significant distribution shift. However, in the policy learning part, the learnable joint policy is still factorized into local marginal policies of individual agents. Does that mean our learning outcome of agents' policies will still suffer from this distribution shift?

2. Can you provide some insights on the performance of the proposed method in complex domains?

### Soundness
3

### Presentation
3

### Contribution
3
