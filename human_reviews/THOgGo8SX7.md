# Efficient Reinforcement Learning for Global Decision Making in the Presence of Local Agents at Scale

- Decision: Reject
- Scores: 5, 5, 5

## Abstract
We study reinforcement learning for global decision-making in the presence of local agents, where the global decision-maker makes decisions affecting all local agents, and the objective is to learn a policy that maximizes the joint rewards of all the agents. Such problems find many applications, e.g. demand response, EV charging, and queueing. In this setting, scalability has been a long-standing challenge due to the size of the joint state space which can be exponential in the number of agents. This work proposes the \texttt{SUBSAMPLE-Q} algorithm, where the global agent subsamples $k\leq n$ local agents to compute a policy in time that is polynomial in $k$. We show that this learned policy converges to the optimal policy on the order of $\tilde{O}(1/\sqrt{k}+\epsilon_{k,m})$ as the number of subsampled agents $k$ increases, where ${\epsilon}_{k,m}$ is the Bellman noise. Finally, we validate our theoretical results through numerical simulations in demand-response and queueing settings.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper considers a Multi-agent Reinforcement Learning problem where the global agent will make decisions for all local agents. This work proposes SUBSAMPLE-Q algorithm and prove its convergence. The authors also provide simulations to support their result.

### Strengths
This paper proposes SUBSAMPLE-Q algorithm to sample $k$ out of $n$ agents to update policy. The authors show that the time is polynomial in $k$. They also provide numerical result to support the finding.

### Weaknesses
My primary concern lies with the novelty of this work. The runtime of the approach in the paper is polynomial in $k$, but this comes at the expense of exponential dependence on $|S_l|$. In most of the case, $|S_l|$ is expected to be significantly larger than $k$ and $n$. Therefore, $k^{|S_l|}$ can perform worse than $|S_l|^k$, which suggests the proposed algorithm may not offer any advantages over standard Q-learning.

Additionally, the presentation of this work is poor and leaves several aspects unclear. For instance, the goal of Algorithm 2 is not clear to me. I would suggest the authors clarify the goal in the paragraph before those two algorithms.

Besides, there are lots of typos in this paper, including:

1. In line 110, I think it should be $s^0 \sim d_0$. 

2. In line 204, I think the $\hat{Q}_k$ should be $\hat{Q}^t$. 

3. The third line in algorithm 2 is unfinished. 

4. The notation $\pi_{k,m}$ is introduced in line 220 but the definition of $m$ is not explained until line 249. 

I strongly recommend that the authors carefully review and revise the paper to address these issues.

### Questions
Please see my comments in Weakness section.

### Soundness
3

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
2

### Summary
The authors consider the setting of reinforcement learning in the context of distributed control. That is, a bunch of local agents are governed by a single global agent. The global agent is the only one in power to enact a policy. The state transitions of the local agents are determined by their previous state and the previous state of the global agent. The state transition of the global agent depends on the current state of the global agent and its enacted action. The reward obtained at each iteration is a sum of a global agent reward which depends on the action enacted and the average of the local agent rewards which do not rely on the action. The objective to optimize is the infinite horizon discounted reward. The main bottleneck of these problems is the exponential growth of the state space with the number of local agents. The paper proposes a technique of initially sampling k local agents, to learn their corresponding empirical deterministic optimal policy, and later using random sampling of k agents at each iteration to learn a random optimal policy.

### Strengths
1. The problem considered has wide applications across multiple domains such as power grid control, EV logistics planning, queuing system control, etc.

2. The suboptimality scales as $O(\frac{1}{\sqrt{k}})+O(\frac{1}{\sqrt{m}})$, where $k$ is the number of local agents sampled at each iteration and $m$ is the number of samples obtained at each iteration to solve for the deterministic optimal policy as a function of $k$.

3. Although they model the agents to be homogeneous for the most part, some heterogeniety is introduced by attaching a type to each agent which is transition invariant.

### Weaknesses
1. The total number of samples required for $T$ iterations of the algorithm is $Tm$, where $m$ is required to be large for lower suboptimalities. Moreover, they assume access to a generative model which is quite often an unrealistic assumption.

2. The heterogeniety modelled in the local agents is incredibly mild and the functional aspect of the problem largely treats them as homogeneous.

3. For applications such as queuing control, etc, the average reward is a more meaningful metric, since discounted reward objective doesn't capture stability issues.

4. It is unclear as to what the role of $T$ is in the final bounds (ie Theorem 3.4).

### Questions
Please refer above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper considers a setting where there is a global decision making agent and there are many local agents. The paper frame this problem as a MDP problem with a global decision making agent and $n$ local agents. The paper proposes a subsampling based Q learning algorithm named SUBSAMPLE-Q and provide theoretical guarantees for the performance gap between the learned policy and the optimal policy. The paper also provides some numerical simulation experiments.

### Strengths
The paper considers a rather interesting RL setting with a global decision maker and local agents. The setting itself is fairly novel to me. 

The theoretical results also seem to be interesting especially making the bound dependent on $k$.

### Weaknesses
In assumption 2.1, it assumes state space for local and global agent and the action space — all are finite. Essentially it’s a tabular setting. Given the RL theory these days mostly moved away from tabular setting and at the very minimum considers linear MDP setting (Jin et al 2020), the paper seems to lack generality beyond this finite setting.

The paper is very dense in notation and it’s very difficult to follow. I must add as a disclaimer that, I am not familiar with many of the related literature around this paper. However, even after coming from RL theory background, I found it tiring and difficult to follow the notations and setups. 

The significance of the theoretical result is not clear. I think the paper would benefit more if the significance and importance are highlighted more.

### Questions
What would be the main difficulty in setting this problem up in non-tabular setting?

### Soundness
3

### Presentation
2

### Contribution
2
