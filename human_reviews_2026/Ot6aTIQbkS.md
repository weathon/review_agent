# Reinforcement Learning for Admission Control in Two-Sided Queueing Systems

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 4, 2

## Abstract
Two-sided queues are a useful formalism for modeling two-sided markets, as well as more general systems in which work is conserved. Furthermore, in practical applications the arrival rate of different entities is often unknown, and may vary based on the state. General-purpose reinforcement learning algorithms may struggle at scale due to the dependency on the diameter of the Markov Decision Process (MDP), which often scales exponentially over the state space in queueing systems. To solve these issues, we present an algorithm with a diameter-independent regret bound, for the problem of admission control in a two-sided queue. Where $S$ is the size of the state space, $N$ is the number of types, $T$ is the number of steps and $\kappa$ is the ratio between the upper and lower rate bounds, our algorithm can be shown to have a regret bound of $\tilde{O}(\kappa^{3} S^{1.5} \sqrt{T}+\kappa^{2.5} S^{1.5} \sqrt{NT})$. We then show that this can significantly outperform general-purpose algorithms in an empirical study.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work considers the problem of admission control in two-sided queues. At one side of the queue, tasks arrive from customers. At the other side, tasks are consumed by available servers.  The decision-making problem involves whether to admit or reject incoming customers and servers. The proposed algorithm relies on a monotonicity assumption about the arrival rates (i.e., that they keep increasing). The algorithm is shown to have more favorable regret than existing generic algorithms. It is also evaluated empirically on a few toy problems.

### Strengths
S1. The paper has a very thorough theoretical treatment and is well-written.

### Weaknesses
W1. The clarity of the work could be improved in my opinion, as I detail in my comments below.

W2. This is subjective, but I find the setting of the paper somewhat convoluted and of limited practicality given the key monotonicity assumption. The assumption is not supported with arguments or real-world examples. Is it realistic to assume that customer rates keep decreasing while server rates keep increasing? However, the paper is clearly on the theoretical side.

### Questions
C1. Regarding "the state can be represented as an integer" (L92): is this true? While the number of customers / servers indeed needs to be tracked, this seems to be a lossy representation. Why are the types of customers and servers not represented in the state also? Is it the case that rewards for abandonments of existing customers / servers are type-independent? How can this be justified given acceptance rewards are type-dependent?  

C2. Other works on RL for queue control worth considering citing are [1] and [2]. [1] in particular has a similar model where customers (tasks) and servers belong to different classes that influence the rewards received.

C3. Some notions related to queuing theory are not explained.  Could you clarify transient states and sojourn times?

C4. Small comments:
- $\Lambda(s)$ is used on L106 before being defined in Assumption 1
- L180: Alg 3 seems to be a much more compact version of Alg 4, do you want to refer to Alg 3 instead here?
- L528: "Ionnaidis" -> "Ioannidis"

### References

[1] Staffolani, A., Darvariu, V. A., Bellavista, P., & Musolesi, M. (2023). RLQ: Workload allocation with reinforcement learning in distributed queues. IEEE Transactions on Parallel and Distributed Systems, 34(3), 856-868.

[2] Jali, N., Qu, G., Wang, W., & Joshi, G. (2024, April). Efficient reinforcement learning for routing jobs in heterogeneous queueing systems. In International Conference on Artificial Intelligence and Statistics (pp. 4177-4185). PMLR.

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
1

### Summary
This paper studies admission control in two-sided queueing systems. To address the exponential dependence on the MDP diameter that arises in general-purpose reinforcement learning algorithms, the authors propose a method with a diameter-independent regret bound. They derive a regret of $\tilde{O}(S^{1.5}\sqrt{T})$ and demonstrate the empirical performance of their algorithm using synthetic datasets.

### Strengths
This paper is the first to propose a reinforcement learning algorithm for two-sided queues with state-dependent arrival and service rates.

### Weaknesses
1. The novelty of the proposed algorithm is not clearly articulated. The method appears to rely heavily on UCRL-based techniques. Additionally, it remains unclear why incorporating state-dependent arrival and service rates presents a fundamental challenge beyond existing approaches.

2. Assumption 1 appears to be quite strong, yet no justification is provided to support its validity in practical systems. A more detailed discussion on when such monotonicity assumptions hold in real-world queueing scenarios would be valuable.

### Questions
1. The paper would benefit from a clearer explanation of why handling state-dependent arrival and service rates poses additional challenges compared to existing approaches.

2. It would be helpful if the paper could elaborate more clearly on the differences from UCRL-AC (Weber et al., 2024).

3. Assumption 1 is quite strong and currently lacks justification. Could the authors provide theoretical or empirical support for why such monotonicity conditions are expected to hold in real-world two-sided queueing systems? Moreover, it remains unclear why the algorithm relies on Assumption 2 instead of directly using Assumption 1; further clarification of the relationship and necessity of these assumptions would improve the paper.

\textbf{Minor comment:} $\Lambda(s)$ and $M(s)$ on page 2 should be defined before they are referenced.

### Soundness
2

### Presentation
2

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
This paper proposes UCRL-TSAC, a reinforcement learning algorithm for admission control in two-sided queueing systems, where both customers and servers arrive stochastically and must be matched. General-purpose RL algorithms such as UCRL2 suffer from regret bounds that depend on the MDP diameter, which grows exponentially with the number of states in queueing models. The authors develop a model-based diameter-independent algorithm.

### Strengths
- Novel diameter-independent regret bound for MBRL for two-sided queueing. 

- Provides rigorous bias analysis showing linear dependence on state space under mild monotonicity assumptions.

### Weaknesses
- I have general question regarding the problem formulation. Firstly, while the customer has types, the state is only the queue length which does not distinguish the customer types. So is the abandonment rate and also there is no compatibility issue between different types of customers and servers. The type only comes into play in terms of the reward.  I guess all of these are intentional to make the problem simple enough, but I wonder whether any real-world problems can fit into such a formulation, and whether the model is an important one in the queueing literature. 

- I am very confused by the action and the reduced action in Line 121.  The original action is defined as all possible subsets of types, but I think this is an artifact of how the problem is formulated as the state does not contain any information of the type. Therefore, the action had to prescribe the set of types it would admit. If I were to formulate the problem, I’d make the state a tuple (q, TYPE) where q is the queue length, and TYPE is set to be the type of the arrived customer/server on the event of customer/server arrival. On the event of abandonment, TYPE is set as some special value. The action space would then just be a binary action of whether to admit or not. This way, the policy just needs to look at TYPE to know what is the event (arrival of a specific type, or abandonment) and decide whether to admit or now. The current action space in the paper is very confusing to me and I do not fully understand the restriction in Line 121. For line 126 it reads “… there exists a particular type i …” but in the equation in Line 121, it is “\forall i”

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors consider the usage of reinforcement learning for admission control in two sided queuing networks. The approach is model based, that is an UCRL type of analysis - where you first build an estimate of the MDP and then solve for the optimal policy. The state space consists of net number of customers or servers. The arrival rates for both types are allowed to be state dependent. The abandonment rates for both types are also allowed to be state dependent. The servers are _assumed_ to be arriving at a higher rate as the number of customers increase and vice versa. The action space considers a restricted set of actions. The main contribution is to arrive as regret bounds which do not scale with the diameter. The authors also perform some experimental validation.

### Strengths
1. Considers an average reward approach for this problem, which is technically challenging but better captures the objective. 
2. Provides diameter independent bounds with state dependent rates of arrival and departure. 
3. Considers a reduced action space with might help with computational complexity issues.

### Weaknesses
1. The paper needs to be better written. The model (MDP) is not specified clearly. Notations are used much before they are defined. Not all assumptions are in the main body. 
2. Literature survey is insufficient. There has been quite some development in this domain which is very relevant to the topic studied in this paper that has not been discussed. I am adding a few of these references below. 
3. The authors mention that other works assume the stability of policies. Since the problem considered in this paper assumes a finite state space, stability is an inherent assumption of the work. 
4. These problems typically fall in the regime of countable state space systems. That is, most applications have incredibly large state spaces, that a model based approach where the transition matrix is explicitly constructed being the right way to solve for this problem is unclear. Some of the references below handle countable state space systems, making their guarantees meaningful in large scale systems. 
5. The modeling assumptions are very strong. For example the abandonment rates are very specific and dont always the reality. The arrival rates are also subject to strong assumptions. The overall setting considered to be quite stylized and not quite general.
6. Even though its the RL setting, the bounds on these arrival rates and abandonment rates are assumed to be known. 
7. The constant $\kappa$ can scale exponentially with the state space size making the lack of diameter dependence sort of moot. 

It will help if the authors can better contrast their work with the rest of the work in literature and identify the exact contributions, as that is very unclear as of now. 




Refs:
1. Score-Aware Policy-Gradient and Performance Guarantees using Local Lyapunov Stability Comte et al.
2. Queueing network controls via deep reinforcement learning Dai and Gluzman
3. Performance of npg in countable state-space average-cost rl Murthy et al
4. Near-Optimal Regret-Queue Length Tradeoff in Online Learning for Two-Sided Markets Yang et al.

### Questions
See above.

### Soundness
2

### Presentation
1

### Contribution
2
