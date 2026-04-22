# Incentives in Federated Learning with Heterogeneous Agents

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Federated learning promises significant sample-efficiency gains by pooling data across multiple agents, yet incentive misalignment is an obstacle: each update is costly to the contributor but boosts every participant. We introduce a game-theoretic framework that captures heterogeneous data: an agent’s utility depends on who supplies each sample, not just how many. Agents aim to meet a PAC-style accuracy threshold at minimal personal cost. We show that uncoordinated play yields pathologies: pure equilibria may not exist, and the best equilibrium can be arbitrarily more costly than cooperation. To steer collaboration, we analyze the cost-minimizing contribution vector, prove that computing it is NP-hard, and derive a polynomial-time linear program that achieves a logarithmic approximation. Finally, pairing the LP with a simple pay-what-you-contribute rule—each agent receives a payment equal to its sample cost—yields a mechanism that is strategy-proof and, within the class of contribution-based transfers, is unique.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies incentive alignment in federated learning (FL) with heterogeneous data distributions. Unlike prior work assuming data exchangeability, agents' utilities depend on *who* contributes samples. The authors show that uncoordinated play yields unbounded Price of Stability, prove that computing optimal allocations is NP-hard, design a polynomial-time LP approximation with logarithmic guarantees, and propose a strategyproof "pay-what-you-contribute" mechanism that is essentially unique within contribution-based transfers.

### Strengths
1. **Novel heterogeneity model:** Properly captures that agents benefit more from their own data than others' (validated empirically in Figure 1), addressing a key gap in FL incentive literature that assumed exchangeable data.
2. **Comprehensive theoretical analysis:** Rigorous results spanning game theory (unbounded PoS), computational complexity (NP-hardness), approximation algorithms (log-factor LP), and mechanism design (uniqueness of PWYC). The progression from negative to positive results is well-structured.
3. **Clean technical contributions:** The Set Cover reduction (Theorem 2) is elegant, the LP analysis properly handles infinite hypothesis classes via VC theory, and the mechanism uniqueness result (Theorem 4) provides interesting theoretical justification for simple mechanisms.
4. **Clear presentation:** Well-motivated with good intuitive examples (warm-up in Section 2.2, two-player illustration in Section 3) before formal results.

### Weaknesses
1. **Restrictive modeling assumptions:** Binary utilities (agents only care about meeting thresholds, not exceeding them) and the need for external subsidies in PWYC limit practical applicability. These limitations deserve more prominent discussion in the main text, not just Appendix B.
2. **Weak empirical validation:** Experiments relegated to appendix with small hypothesis classes (|H| ≤ 50) and limited datasets. For ICLR, stronger validation on realistic FL scenarios is expected. Scalability to neural networks unclear.
3. **Loose approximation bounds:** The O(d²(log(...))²/log(1/δ)) guarantee for infinite H may be impractical. No discussion of when this is acceptable or tighter analysis for special cases. No lower bounds showing necessity of logarithmic factors.
4. **Limited practical guidance:** Unclear how to implement the LP allocation for neural network hypothesis classes. The transition from known distributions (Section 4) to unknown (Section 5) could be smoother.

### Questions
1. Can you provide lower bounds showing the logarithmic approximation is necessary, or identify problem classes admitting better approximations?
2. Can you formally characterize when budget-balanced mechanisms exist in your setting, or prove impossibility?
3. How would the LP approach work for neural network hypothesis classes in practice?
4. How do results extend to heterogeneous accuracy requirements (different ε_i) or sequential/multi-round FL?
5. Can you provide concrete payment comparisons between PWYC and approximate VCG mechanisms?

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
This paper studied a game-theoretic framework in federated learning with heterogeneous data, where an agent’s utility depends on the specific samples that each agent supplies. The agents aim to meet a PAC-style accuracy threshold while minimizing cost. Theoretically, first, the paper showed that without coordination the game can fail badly and pure equilibria is not guaranteed to exist. Second, with coorporation, it is NP-hard to obtain the cost-minimizing contribution vector. The paper then developed a polynomial-time linear program that achieves a logarithmic approximation. Lastly, the paper proposed a mechanism with payment rule that is strategy-proof and unique within the class of contribution-based transfers.

### Strengths
Overall I found this work to be interesting and well-presented, and it provided a useful set of threotical analysis and results which extended upon the current literature on incentives in FL with heterogeneous data. 
- Theoretically, this paper introduced a concrete utility model based on PAC-style threshold guarantees. And such utility model allowed further analysis on the necessity of coordination and the payment-based mechanism. While most of the existing work did not use specific utility models, this framework could be a good extension on those.
- the strategyproof mechanism used a simple pay-what-you-contribute rule that is natural and intuitive
- while the analysis assumed linear cost functions for agents, the results are flexible to hold with more general convext cost functions

### Weaknesses
- I found the utility function assumed for the agents to be strict, in particular the payoff is an indicator function that's either 1 or 0. In reality the payoff to agent may be a more general increasing function of the number of samples, such as (1-loss) illustrated in Figure 1. Can the results be extended to more general utility funcitons?

### Questions
Results in section 5 assumed that the planner knows c_i. In practice it's more likely that the agents also report their cost function c_i. Is it possible to have a strategyproof mechanism where the agents report both distributions and their costs?

### Soundness
3

### Presentation
3

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
The paper models federated learning with heterogeneous agents as a contribution game in which each agent wants a PAC-style guarantee on its own distribution Utilities depend on who supplies each sample, not just the total mass. The authors show that uncoordinated play can be pathological: pure NE may not exist and even the best equilibrium can be arbitrarily worse than cooperation. A central planner seeking the cost-minimizing contribution vector faces an NP-hard decision problem, but a linear program delivers logarithmic-type approximation. Coupling this LP with a "pay-what-you-contribute" transfer is strategy-proof and, within contribution-based mechanisms, essentially unique.

### Strengths
- Clear formalization of heterogenous valuations in FL where benefits depend on contributor identity; the PoS lower bound and the local-obliviousness–based uniqueness of PWYC are novel angles versus exchangeable-data models. 
- Connects learning theory, game theory, and mechanism design for realistic FL, offering a tractable planner and a strategy-proof payment rule that could inform practice.
- Experients small but targeted, including FEMNIST/Shakespeare experiments that validate the heterogeneity premise.

### Weaknesses
-  The analysis assumes realizable PAC learning and ERM and requires pairwise disagreement masses for (potentially many) hypothesis pairs. For infinite $\mathcal{H}$ LP, a $\lambda$-net is constructed using the VC dimension. Estimating these quantities reliably for modern models is nontrivial and makes the linear programming approximation intractable.
- PWYC is strategy-proof but typically not budget-balanced. There is no bound on the planner’s deficit or a characterization of near–budget balance vs. individual rationality. Adding such bounds would strengthen the practical message.
- The uniqueness of contribution-based payments is proved for two agents under local obliviousness; it is unclear how far this extends to k>2 or to other approximation algorithms.

### Questions
- In theorem 3, why is the approximation error independent on $\epsilon$ for finite $\mathcal{H}$?
- If an agent can split into multiple identities or collude, does PWYC remain strategy-proof and does the LP allocation change in harmful ways?
- What are examples of practical FL protocols that satisfies the information condition in section 5?

### Soundness
2

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
This paper uses the framework of game theory to study the incentives of agents in federated learning to contribute local data in a heterogeneous learning setup.
It models the utility functions of agents by assuming that each agent has a PAC accuracy objective—achieving at most $\varepsilon$ empirical risk with probability at least $1-\delta$—which yields a payoff of 1 if and only if the goal is achieved, while collecting and contributing data incurs a linear cost.
Based on this formulation, the paper shows that a non-coordinated equilibrium may not exist or can be arbitrarily inefficient compared to the coordinated social optimum.
It proves that computing the social optimum is NP-hard but provides a scheme to approximate it via linear programming (LP).
Finally, it proposes the Pay-What-You-Contribute (PWYC) scheme, where each agent is paid according to its data-collection cost, which is strategyproof in the sense that agents have no incentive to misreport their local data.

### Strengths
The study of incentivizing agents to participate in collaborative federated learning and preventing free-riding is of clear interest to the community. The paper is generally well-written, and the conceptual takeaways of each section are fairly clear and interesting. It provides a good theoretical prototype for studying agents’ social behavior in fully heterogeneous federated learning settings.

### Weaknesses
My key concern is that although the high-level ideas are compelling, the analysis relies on multiple layers of simplifications and idealized assumptions, which makes the overall significance of this work questionable.

**1. Assumption on sample collection.**
Section 2.1 explicitly requires that the local datasets should be pooled into a collective dataset $S$, and that the global model is directly trained via ERM on $S$. To my understanding, this does not conform to the philosophy of federated learning, which aims to keep local data private. In practice, algorithms such as FedAvg are used to indirectly train a central model (hypothesis) over $S$, which introduces a gap between what the federated learning mechanism can achieve and the true minimizer of ERM. The paper seems to ignore this gap and does not explain or justify this simplification.

**2. Binary payoff structure.**
It is assumed that the payoff from model training is given by the binary variable $a_i^{\epsilon,\delta}$, indicating whether the PAC learning objective is satisfied or not. This is an unrealistic assumption. One might argue that such a simplification is necessary for analytical tractability, but it is not merely a matter of convenience. For instance, I do not see a straightforward way to generalize notions such as Assumption 1 (self-sufficiency) or the socially optimal data-contribution profile to the case of non-binary payoffs.

**3. Requirement to fix $m_i$ in advance.**
This is another stringent and unrealistic assumption, as in practice, local model optimization can be performed independently, and the utilization of local data should be dynamically adjustable. It is also unclear how the central sample-allocation scheme would operate when the hypothesis space’s covering number or VC dimension, as well as the agents’ explicit data-collection costs, are unknown. The framework quickly becomes impractical due to $m_i$ being pre-fixed, especially when it becomes difficult to approximate these quantities accurately.

**4. Is PWYC making federated learning meaningless?** I am uncertain about the implications of the results in Section 5. The PWYC mechanism simply reimburses each agent for its data-collection cost, meaning that the central server essentially subsidizes all data acquisition. However, if the server pays for everything, this seems to undermine the motivation for federated learning in the first place, as it assumes the existence of an entity that can simply “buy all data” and conduct centralized training. The result in Section 5.2 is a bit disappointing in this regard, as its message appears to be that this is essentially the only case where agents can be sufficiently incentivized to participate in federated learning without misreporting data.

### Questions
Please see the Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
2
