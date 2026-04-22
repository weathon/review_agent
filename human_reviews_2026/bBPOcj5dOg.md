# Queue Length Regret Bounds for Contextual Queueing Bandits

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
We introduce contextual queueing bandits, a new context-aware framework for scheduling while simultaneously learning unknown service rates. Individual jobs carry heterogeneous contextual features, based on which the agent chooses a job and matches it with a server to maximize the departure rate. The service/departure rate is governed by a logistic model of the contextual feature with an unknown server-specific parameter. To evaluate the performance of a policy, we consider queue length regret, defined as the difference in queue length between the policy and the optimal policy. The main challenge in the analysis is that the lists of remaining job features in the queue may differ under our policy versus the optimal policy for a given time step, since they may process jobs in different orders. To address this, we propose the idea of policy-switching queues equipped with a sophisticated coupling argument. This leads to a novel queue length regret decomposition framework, allowing us to understand the short-term effect of choosing a suboptimal job-server pair and its long-term effect on queue state differences. We show that our algorithm, CQB-$\varepsilon$, achieves a regret upper bound of $\widetilde{\mathcal{O}}(T^{-1/4})$. We also consider the setting of adversarially chosen contexts, for which our second algorithm, CQB-Opt, achieves a regret upper bound of $\mathcal{O}(\log^2 T)$. Lastly, we provide experimental results that validate our theoretical findings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new framework called contextual queueing bandits (CQB). The problem involves a single queue and multiple servers. Jobs arrive with diverse "contextual features" (e.g., job size, user profile). The system's agent must decide which job to assign to which server in real-time to maximize the service (departure) rate. The key challenge is that these service rates, which are modeled by a logistic function of the job's context and an unknown server-specific parameter, are not known in advance and must be learned simultaneously. The paper presents an algorithm that achieves an $\tilde{\mathcal{O}}(T^{-1/4})$ regret in the stochastic setting, and a $\mathcal{O}(\log^{2}T) regret in the adversarial setting.

### Strengths
1. New problem setting: The paper considers a formulation of queueing bandits where an incoming job is associated with a context vector $x$, which is matched with a corresponding feature vector $\theta$ of a server. The service rate of the server is the logistic function of the inner product of $x$ and $\theta$. The key challenge is that $\theta$ is unknown and needs to be estimated by balancing exploration of more servers and exploitation of the currently estimated best-matching server of a job in the queue.

2. Analysis Technique: The paper clearly identifies and names the "queue state misalignment" phenomenon, arising due to the mismatch between the context vector set in the queue of a candidate policy and that of the optimal policy. The paper presents a new analytical technique that uses queue length regret decomposition based on policy-switching queues and a novel coupling argument. This technique cleverly isolates the short-term impact of a suboptimal choice from its long-term effect on the queue state.

3. Regret Guarantees: The paper provides the first provable decaying queue length regret bound $\tilde{\mathcal{O}}(T^{-1/4})$ for this setting. This is a strong result, as it shows the queue length under the learning policy converges to that of the optimal policy. The $\mathcal{O}(\log^{2}T)$ regret for the adversarial setting is also a solid finding.

### Weaknesses
1. Simplistic Service Model: I believe that the service model considered by the paper is too simplistic and detached from practice. See my questions below for more details.

2. Regret Lower Bound: The main regret bound for the stochastic setting is $\tilde{\mathcal{O}}(T^{-1/4})$. The paper does not provide a lower bound to show if this rate is optimal.

3. Traffic assumptions: Assumption 3.4 is somewhat strong. It requires that for every job context $x$, there exists a server $k^*$ that can process it with a service rate greater than the arrival rate $\lambda$ by at least $\epsilon$. This guarantees system stability if the optimal policy is followed. Also, the paper assumes a uniform rate of arrivals, and it is unclear how the policy and regret analysis can be extended to bursty arrivals, which are common in practice.

4. The two-phase policy with a pure exploration phase followed by $\epsilon$-greedy is not very practical, but it seems to be the tractable algorithm for which the authors can give guarantees.

### Questions
1. Are you considering only work-conserving policies where all the servers are kept busy whenever there is a job in the central queue? For the context-based service rate setting, I believe that a non-work-conserving policy can be better in terms of queue length and latency -- a job might benefit from waiting in the central queue while its matching server is unavailable rather than being routing to a sub-par server. Routing policies of this nature have been considered in https://arxiv.org/pdf/2402.01147 and https://ieeexplore.ieee.org/document/1103637. Could you please cite these works and clarify how the setting of this work differs from those in the previous papers? If I understand correctly, you are assuming that only one server is active at any given time.

2. In Figure 1, what do you mean by 'success' or 'failure'? Once a job is assigned to a server, the time taken to serve it will be determined by the logistic service rate model. So won't that server remain occupied for the next few time slots until the job finishes? The authors seem to be using a one-shot service success/failure model where a job is served with probability $\sigma(\mu)$, that is, success, and otherwise it is dropped.

I believe that the practicality and impact of this work can be improved significantly by enhancing the service model to consider multiple servers being used simultaneously for different jobs, and moving away from this sucess/failure type service model.

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
3

### Summary
This paper introduces a new framework called contextual queueing bandits, which extends classical queueing bandit models by incorporating heterogeneous job contexts and learning unknown service rates governed by a logistic model. The authors use a non-trivial performance metric queue length regret to measure the gap between the learning policy and the optimal policy. To analyze this, the paper develops a policy-switching queue coupling technique that decomposes queue length regret into short-term (instantaneous) and long-term (state-misalignment) effects. Then building on this, two algorithms are proposed: one for the stochastic context setting (i.i.d. job features) and achieving a vanishing regret bound of $O(T^{-1/4})$; another for the adversarial context setting (arbitrary features) an achieving a polylogarithmic regret bound of $O(\log^2 T)$. The authors also conduct simulation experiments to demonstrate queue length convergence.

### Strengths
1. The paper extends classical queueing bandits to a contextual setting, where job features directly affect service rates.  The metric of queue length regret is a non-trivial and operationally meaningful objective compared to standard cumulative reward regret.

2. The writing is good. The queue dynamics, logistic service model, and filtration definitions are clearly presented. The results for both stochastic and adversarial contexts are also clearly separated.

3. The policy-switching queue construction and coupling argument are technically elegant and could inspire future queue length regret analysis in other dynamic systems.

4. The paper establishes the first provable decaying queue-length regret bound in the stochastic contextual setting by exploiting Chebyshev’s sum inequality for oppositely ordered sequences. This result is both elegant and novel, and I appreciate this contribution.

### Weaknesses
1. The logistic service rate model needs to be further examined and discussed.

2. Although the coupling-based decomposition is theoretically interesting, the operational meaning of the $O(T^{-1/4})$ scaling could be clarified. For example, what does vanishing queue length difference imply in real scheduling terms?

3.  The length setup for the pure exploration phase requires prior knowledge of the traffic slackness parameter. It would be valuable to discuss whether this dependence can be relaxed.

### Questions
1. How sensitive are the results to the logistic model assumption? Could the framework handle other contextual service model?

2. Is the coupling argument extendable to multi-queue settings (each queue associated with a server)?

3. Is it possible to remove the dependence on slackness parameter in the pure exploration phase to develop a more adaptive policy?

4. As mentioned in the weakness part, the operational meaning of the vanishing queue length could be clarified.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies contextual queueing bandits, a new context-aware framework for scheduling while simultaneously learning unknown service rates. Individual jobs carry heterogeneous contextual features, based on which the agent chooses a job and matches it with a server to maximize the departure rate. The service/departure rate is governed by a logistic model of the contextual feature with an unknown server-specific parameter. To evaluate the performance of a policy, the authors consider queue length regret, defined as the difference in queue length between the taken policy and the optimal policy. The main challenge in the analysis is that the lists of remaining job features in the queue may differ under the taken policy compared to the optimal policy for a given time step, since these two policies may process jobs in different orders. To handle this challenge, the authors propose the idea of policy-switching queues equipped with a sophisticated coupling argument. This leads to a novel queue length regret decomposition framework, which exhibits the short-term effect of choosing a suboptimal job-server pair and its long-term effect on queue state differences. The authors show that the proposed algorithm, CQB-$\varepsilon$, achieves a regret upper bound of $O(T^{ −1/4})$. The authors also study the setting of adversarially chosen contexts, for which the proposed second algorithm, CQB-Opt, achieves a regret upper bound of $O(\log^2 T)$. Lastly, the authors provide experimental results to validate their theoretical findings.

### Strengths
1. The studied problem, contextual queueing bandits, is interesting and novel, which can be applied to various scenarios such as job scheduling and wireless networks. This problem may attract attention in both online learning and queueing theory communities.
2. The authors design two algorithms, i.e., CQB-$\varepsilon$, which achieves a regret upper bound of $O(T^{ −1/4})$ for the regular contextual setting, and CQB-Opt for the adversarially chosen context setting, which achieves a regret upper bound of $O(\log^2 T)$.
3. The challenges and developed techniques are clearly explained.

### Weaknesses
1. This paper should discuss and compare with the following work, which studies RL with queueing states, in formulation and results. For example, can the formulation of the following work encompass the formulation of this paper?

Yashaswini Murthy, Isaac Grosof, Siva Theja Maguluri, and R. Srikant. Performance of NPG in countable state-space average-cost RL. arXiv preprint arXiv:2405.20467 (2024).

2. The proposed algorithm, CQB-$\varepsilon$, uses the explore-then-exploit (ETE) strategy, which is known to be suboptimal regarding the instance-independent regret bound compared to the UCB and Thompson sampling based strategies. The authors should discuss more on why they use the ETE strategy?
3. Discussions and comments are lacking following the main theoretical results, e.g., Theorems 5.5 and 6.3. For example, the authors should discuss more on the intuition behind the $O(T^{ −1/4})$ and $O(\log^2 T)$ regret bounds. Why are the regret bounds neither $O(T^{ −1/3})$ as the traditional ETE regret bound, nor $O(T^{ −1/2})$ as the traditional contextual bandit regret bound?
4. The current baseline in empirical evaluations is the random sampling policy, which is too simple. It would enhance this paper if the authors can compare with more baselines which are from (or the adaptations of) existing queueing bandit works.

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies a contextual queueing bandit problem, where each job is associated with distinct contextual features. In each round, the decision maker selects a job based on its context and assigns it to a server, aiming to maximize the overall job departure rate. A key challenge arises from queue state misalignment, as different policies lead to different queueing states over time. To address this, the authors propose the CQB-$\epsilon$ algorithm, which alternates between a pure exploration phase and an $\epsilon$-greedy phase. Using a telescoping-based analysis technique, they establish a regret bound of $\tilde{O}(T^{-1/4})$. Furthermore, the paper extends the framework to an adversarial context setting, where the CQB-Opt algorithm achieves sublinear regret under the assumption of the existence of at least one good server.

### Strengths
The strengths of the paper are summarized below.

- The paper introduces a novel contextual queueing bandit framework in which arriving jobs have distinct contextual features. The problem is well-motivated by practical applications such as personalized recommendation systems and LLM inference workloads.

-The paper addresses the analytical challenges arising from queue state mismatch, which complicates the regret analysis in queueing-based decision processes.

- The work extends beyond the stochastic setting by developing and theoretically analyzing an algorithm for the adversarial context setting, providing provable sublinear regret guarantees.

### Weaknesses
The weaknesses of the paper are given below. 

- The main contribution lies in extending existing queueing bandit frameworks to the heterogeneous contextual setting, but many of the key algorithmic and analytical techniques are adapted from prior work rather than fundamentally novel.

- The algorithmic design of the proposed methods follows standard extensions of existing bandit algorithms. While the regret analysis is mathematically interesting, it does not lead to new insights in algorithmic design for queueing bandits.

- The experimental evaluation is limited to synthetic datasets, which restricts the demonstration of practical effectiveness. Incorporating real-world datasets or additional application scenarios would strengthen the empirical validation.

### Questions
What are the insights provided by the regret analysis which show the new challenges comparing to the existing queueing bandits?

### Soundness
3

### Presentation
2

### Contribution
2
