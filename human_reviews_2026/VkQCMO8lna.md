# Scalable Cooperative Multi-Agent Reinforcement Learning with Adaptive Communication Range

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 2, 6, 6

## Abstract
It has long been recognized that cooperative multi-agent reinforcement learning (MARL) faces scalability challenges, as the state-action space grows exponentially with the number of agents. Existing approaches typically improve scalability by filtering out communication with low-relevance agents. However, such filtering often relies on global state and fixed graph distribution, limiting adaptability in large-scale, communication-constrained environments. In this paper, we propose a scalable MARL method, called Adaptive Communication Range PPO (ACR-PPO), that models communication-based decision-making under communication budget constraints as a sequential process: a communication policy first selects each agent’s communication range within a given budget, followed by a behavior policy that chooses actions based on the included neighbors. More importantly, we provide a theoretical guarantee of monotonic performance improvement under communication budget constraints. Experimental results across diverse scenarios show that our approach preserves policy performance while significantly reducing communication cost through adaptive range control.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces **ACR-PPO**, a networked multi-agent reinforcement learning (MARL) method that incorporates communication among agents. The authors provide a theoretical analysis of how the communication range affects performance improvement and evaluate the proposed method on several cooperative tasks. The results suggest that ACR-PPO can significantly reduce communication cost while achieving comparable performance in some tasks.

### Strengths
1. The paper is well-written and clearly structured.
2. It provides a theoretical analysis of the communication range under communication cost, which could be of interest to researchers in the MARL with communication community.

### Weaknesses
Overall, the novelty of the paper is not clear to me, particularly regarding the discussion of the “global state.” While the paper attempts to connect its approach to Networked MARL with communication, the discussion of related work is limited. The differences between this paper and the latest achievements in the Networked MARL literature are not clear. 

Moreover, the GACML and AC2C methods mentioned by the authors have already employed local information for determining communication (although they employ centralized critics only during training), which contradicts the authors' claim "_these methods typically require the global state_".

The baseline comparison does not include communication-based methods such as GACML and AC2C. Without these comparisons, it is difficult to assess the effectiveness of the proposed method.

In Figure 1, the definition of communication cost is unclear. PPO itself does not involve communication among agents, so its communication cost should be zero. Why would communication be necessary if a regularized PPO variant (PPO + regularization) can outperform ACR-PPO on the presented tasks?


**Other Comments**

- **Line 43:** Mean-field MARL does not typically involve or claim explicit communication among agents. Why is it included in this introduction and related work?
- **Line 101:** Please clarify which appendix is being referenced.
- **Definition 1:** This definition is said to follow Qu et al., but the assumptions and meaning underlying it are not clearly stated.
- **Equations (10) and (11):** These are somewhat confusing.
    - In Eq. (10), since ki,t is given, it is unclear why it appears in the sampling process.
    - In Eq. (11), if Q is replaced using Eq. (10), both s and ki,t​ appear to be sampled twice.

### Questions
See weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Adaptive Communication Range PPO (ACR-PPO), a scalable cooperative multi-agent reinforcement learning algorithm that adaptively adjusts each agent’s communication range under a cost budget. It is based on the spatial correlation decay concept proposed in the literature, and different from Qu 2020, it uses a PPO formulation as the objective. It formulates communication-aware decision-making as a constrained optimization problem and jointly optimizes a communication policy and a behavior policy. The paper provides theoretical guarantees of monotonic performance improvement even with limited communication. Experiments on traffic, power-grid, and vehicle-platooning benchmarks show that ACR-PPO maintains near-centralized performance while significantly reducing communication cost.

### Strengths
-	The definition of the time-varying range Q/V function is novel
-	The decomposition of the policy into a range-selection and then a control part is also novel. 
-	The proposed approach has diverse and interesting applications which are of value to this community.

### Weaknesses
My major concern is Lemma 1. Firstly, Lemma 1 does not say what rho is (which I believe should be gamma, the discounting factor, by looking at the proof). Then, looking at the proof in eq. (31), the second equality uses $\rho_{i,t} = \rho_{i,t}’$ for $t\leq \kappa$. This is not something that can be “required”. I think this is only true, if in the transition (eq. 1) and policy factorization, each agent only depends on 1-hop neighbors. If, for example, the transition and policy factorization depends on 2-hop neighbors, then $\rho_{i,t} = \rho_{i,t}’$ will be true up to $t\leq \kappa/2$ as each step the difference will propagate for 2 hops. Now that the authors are using a $\kappa$-hop dependence in the transition and policy, then $\rho_{i,t} = \rho_{i,t}’$ is only true up to t=1. 

I believe this is a major flaw of the paper. While I found the concepts to be interesting, I would suggest fixing this flaw before the next submission.

### Questions
See weakness.

### Soundness
1

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
2

### Summary
This paper proposes ACR-PPO, a scalable cooperative MARL framework that enables agents to dynamically adapt their communication range under a budget. The method models decision-making as a sequential process, where a communication policy first selects a range and a behavior policy then acts based on the gathered information. Supported by a theoretical guarantee of monotonic performance improvement, experiments show that ACR-PPO offers competitive performance in networked environments.

### Strengths
1. The paper tackles a relevant and practical problem. Improving the scalability of communication in MARL is crucial for real-world applications, and the proposed approach of dynamically adjusting the communication range under a budget is a promising and well-motivated direction.
2. The authors provide a formal guarantee for monotonic policy improvement under communication constraints, which adds significant rigor to the proposed method.

### Weaknesses
1. The results are a bit difficult to interpret. For instance, while Figure 2(c) provides a valuable snapshot of the communication range changing within one episode, it would be beneficial to include more comprehensive statistics. A richer analysis might include:
    - Visualizations or statistics on how the communication range varies across different agents in a single episode. Do some agents consistently require a larger range than others?
    - An analysis of how the average communication range evolves over the entire training process (i.e., across episodes), not just within a single late-stage episode.
    - Case studies in different scenarios showing how the learned communication strategy adapts to different environmental demands.
2. The current baselines compare against full communication (CPPO) and no communication (DPPO). To make a more compelling case for the dynamic nature of ACR-PPO, it would be valuable to include a baseline with a fixed, moderate communication range. This would directly test whether dynamically adjusting the range offers a significant advantage over a well-chosen, static communication heuristic.

### Questions
1. How does the effectiveness of ACR-PPO vary with the underlying graph topology? For example, does the method's performance change significantly in graphs with larger diameters, larger size or different topology?
2. The paper frames the decision from the receiver's perspective, where an agent's communication policy determines the range from which it receives information. In a real-world implementation, this would seem to require a "request: phase where the receiver informs potential senders. The overall policy then becomes multi-stage: communication neighbors selection - request - communication - action policy. Does this introduce significant latency or overhead that might affect the applicability of the method? Is there any potential ways to reduce the latency?
3. Could the authors clarify the assumptions regarding agent homogeneity? Are the policies, communication budgets, and local network structures assumed to be uniform across all agents?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes ACR-PPO, an adaptive communication range policy optimization algorithm for cooperative multi-agent reinforcement learning (MARL) under communication cost constraints. The approach formulates the optimization process as a sequential two-policy update mechanism: communication policy and behavior policy. The idea is intuitive and make sense. The authors have provided detailed theoretical proofs and derivations, along with experimental results across multiple benchmarks, which validate the effectiveness of the proposed method.

### Strengths
1.The work directly tackles an important bottleneck in scalable MARL—the exponential growth of inter-agent communication cost—and proposes a mechanism that can flexibly adapt to dynamic and resource-constrained environments.
  
2.Theoretical Guarantees: The paper provides well-developed analysis, which enhances the credibility of the proposed method. 

3.Strong Empirical Validation: Quantitative results demonstrate that ACR-PPO can adaptively trade off performance and cost, often achieving comparable performance with dramatically reduced communication.

### Weaknesses
1. While this work achieves comparable performance with dramatically reduced communication via a sequential process, it is common sense that such sequential strategies typically involve a trade-off between time and space. If feasible, the authors should provide additional comparative experiments related to time (frequency) to clarify the overhead incurred. In my opinion, controlling frequency (time delay) should also be a critical factor for the real-world performance of multi-agent systems.

2. The goal of this paper is scalability (see Abstract, Introduction, Section 4). While the authors’ approach shows promising potential, the compared methods do not seem to be tested in larger-scale environments. If there exist experimental setups for larger scenarios in this field, the authors should supplement such experiments. Large-scale experiments are not as straightforward as theoretical verification; they typically involve more factors that need to be considered, such as communication latency.  Without experiments in such scenarios, the authors’ claims about scalability are weakened-—a topic frequently discussed in the other domains.

3. Given my limited familiarity with this field, I am not certain about the reasonableness of the selected baselines. However, I notice that only one 2025 paper is cited for comparison. If there exist more recent relevant works, the authors should typically include performance comparisons with them.

4. The paper is somewhat overly focused on theoretical rigor, and the authors are encouraged to provide more implementation details. Regarding the theoretical section, the authors are also advised to offer more intuitive explanations—this would facilitate quick comprehension for researchers from other fields who may be interested in the work.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
