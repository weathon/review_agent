# Online Multi-Agent Control with Adversarial Disturbances

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
Online multi-agent control problems, where many agents pursue competing and time-varying objectives, are widespread in domains such as autonomous robotics, economics, and energy systems. In these settings, robustness to adversarial disturbances is critical. In this paper, we study online control in multi-agent linear dynamical systems subject to such disturbances. In contrast to most prior work in multi-agent control, which typically assumes noiseless or stochastically perturbed dynamics, we consider an online setting where disturbances can be adversarial, and where each agent seeks to minimize its own sequence of convex losses. Under two feedback models, we analyze online gradient-based controllers with local policy updates. We prove per-agent regret bounds that are sublinear and near-optimal in the time horizon and that highlight different scalings with the number of agents. When agents' objectives are aligned, we further show that the multi-agent control problem induces a time-varying potential game for which we derive equilibrium tracking guarantees. Together, our results take a first step in bridging online control with online learning in games, establishing robust individual and collective performance guarantees in dynamic continuous-state environments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focus on online control in multi-agent linear dynamical systems with adversarial disturbances.

### Strengths
1. The presentation is very clear, with careful organization and precise mathematical exposition.

2. The proofs are detailed and technically solid. Although I did not check every appendix proof line by line, the arguments in the main text appear correct and internally consistent.

3. The multi-agent setting is both natural and valuable — it extends the single-agent adversarial control framework to a broader and practically relevant domain.

### Weaknesses
1. The algorithmic framework and proof techniques largely follow Agarwal et al. (2019), with relatively limited methodological novelty beyond extending to multiple agents.

2. The paper lacks any empirical validation, even simple synthetic experiments. Given that multi-agent setups are especially suitable for simulation, this absence weakens the practical impact.

3. Some assumptions (e.g., global strong stability) are relatively strong, and it is unclear how restrictive they are in practice.

### Questions
1. Could the authors highlight more explicitly what are the key technical differences and challenges compared to Agarwal et al. (2019)? For example, what specific parts of the regret analysis or equilibrium proof required new ideas due to the multi-agent coupling?

2. In the Aggregated Control Learning setting, the improved regret bound relies on a global strong stability assumption. Could the authors give a simple counterexample showing that without this assumption, the result may fail?

3. It would significantly strengthen the paper to include simple multi-agent simulations (e.g., 2–3 agents in a linear system) to illustrate how the proposed algorithm performs and whether the regret empirically scales as predicted.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work is a follow up to Agarwal 2019 in the multi-agent setup.  In particular, the authors study two setups:
1. Each agent observes their own actions;
2. Each agent additionally observes the accumulated actions.

The authors show that the second setup lead to better N dependence in the regret.

Further, the authors consider a special setup when all agent has the same cost function. Here the approximation error of nash equilibrium can be bounded as in a online convex optimization problem, and hence the average action converges to the "nonstationary" Nash equilibrium.

### Strengths
This work extends [Agarwal 2019] in an interesting direction. Analyzing regret / convergence / stability for multi-agent systems can always be challenging. To address the challenge, the authors identify nontrivial multi-agent setups in which learning can happen.

Further, the paper is techinically sound and easy to follow. The authors may benifit from a better comparison of their analysis to the original analysis in Agarwal, and highlight the technical difficulty.

### Weaknesses
My concern is about how nontrivial the analyses are. For the three results:

1. When each individual only observe their own actions: it seems one doesn't need to care about other actions because the perturbation is already adversarial. Further, the disturbance caused by other agents' action is bounded due to the system being stable. Therefore, it seems one can just apply [Agarwal] result directly with N different copies.

2. When the individual agent can observe actions from other agents, they can infer the perturbation and get better "problem coefficients" (e.g. disturbance size) when apply Agarwal. 

3. When all agents share the same cost, the problem is just online learning, except that its not exactly convex. However, when use DAC class, the problem is not very far away from a convex surrogate with larger decision space.

I am not 100% sure my understanding is accurate, but after reading the results, all the bounds seem expected, and relatively straight-forward.

### Questions
See weakness

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses online control in multi-agent linear dynamical systems (LDS) with time-varying convex costs and adversarial disturbances, contrasting with prior work on noiseless or stochastic systems. The authors propose a decentralized, gradient-based algorithm (Algorithm 1) using Disturbance Action Controller (DAC) policies, aiming to achieve both sublinear individual regret and collective equilibrium tracking. The analysis is conducted under two information models. First, in an independent learning setting (observing only the state), the paper establishes a per-agent regret bound of $\tilde{\mathcal{O}}(N^2 \sqrt{T})$. Second, in an aggregated control learning setting (observing state and others' aggregated inputs), the bound is improved to $\tilde{\mathcal{O}}(N \sqrt{T})$, and further to a near-optimal $\tilde{\mathcal{O}}(\sqrt{T})$ with an additional Lipschitz assumption, removing the dependence on $N$. Finally, in a common interest setting (a time-varying potential game), the paper demonstrates that the no-regret algorithm successfully tracks the evolving Nash equilibria.

### Strengths
1. The paper tackles a challenging and relevant problem at the intersection of online non-stochastic control and online learning in games.
2. The paper provides a strong theoretical analysis. The analysis under two different information structures (Settings 1 and 2) provides a clear understanding of the value of information and the price of decentralization. The $\Omega(\sqrt{T})$ lower bound confirms the optimality of the bounds with respect to $T$.
3. The equilibrium tracking result (Theorem 4.1) is a valuable contribution.

### Weaknesses
1. The algorithm assumes that agents have perfect knowledge of the system dynamics ($A$ and their own $B_i$). This is a common but significant limitation. While unknown dynamics is mentioned as future work, a brief discussion in the main text about the specific challenges this would introduce (e.g., the need for system identification competing with the adversarial disturbances) would be beneficial.

2. The algorithm requires agents to possess a stabilizing linear controller $K_i$ a priori. More critically, the improved results in Setting 2 (Theorem 3.4) rely on Assumption 4, which is a global strong stability condition on the joint controller $(K_1, \dots, K_N)$. Moreover, the paper (below Assumption 4) suggests this can be centrally precomputed, which weakens the decentralized claim for Setting 2 and should be stated more explicitly in the main text. How practical is it to obtain this global controller in a multi-agent setting?

3. The regret bound in Theorem 3.2 depends on $U^2$, where $U$ is an assumed upper bound on the control inputs of other agents. This seems somewhat circular. While the algorithm's projection step bounds the agent's own DAC parameters, it's not immediately obvious that this guarantees $||u_t^j|| \le U$ for all $j \neq i$, especially when other agents are also learning. Do you need to prove that ``if all agents run Algorithm 1, then the control output $u_t$ for all will be bounded. Thus, the assumption $||u_t^j||\le U$ for all $j\in[N]$ needs more justification.

4.  The paper does not consider a fully decentralized control setting in that (i) each agent has access to the global state and (ii) each agent observes the aggregated control input from all the other agents. In a fully decentralized setting, each agent may only have access to its local state, and the communication among the agents is through a general graph structure. The authors should more clearly describe this point in the paper and also point to references in the literature that study fully decentralized control setting with local states and communication graph.

5. Theorem 3.2 provides regret against the class of strongly stable linear controllers ($\Pi_i^{\text{lin}}$), whereas the main results for Setting 2 (Theorems 3.4 and 3.5) are against the DAC policy class ($\Pi_i^{\text{DAC}}$). While DAC is a standard comparator class used to ensure convexity in online control, this is a weaker benchmark than $\Pi_i^{\text{lin}}$. This distinction should be explained and discussed more clearly in the paper.

### Questions
Greatly appreciated if the authors could address the weaknesses mentioned above.

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
4

### Summary
This paper considers a linear dynamical system that evolves via inputs of N agents each with a different cost function, subject to adversarial disturbances. In this setting, the paper provides (A) a O(N^2T^0.5) regret bound when the agents can observe the state against linear policies, (B) a O(NT^0.5) regret against recently introduced disturbance-action policies, (c) and a bound on the sub optimality with respect to the best response produced due to individual regret minimization.

### Strengths
The paper puts forward and interesting setup, and it is an interesting question as to what equilibria arise due to self-centered learning behaviors in this setting.

The closest work (Ghai et al) I am aware of only considers the cooperative setting: where the feedback is limited, but the cost function is shared. This paper on the other hand models disparate costs per agent.

### Weaknesses
In my reading, a major weakness is that many results (Theorem 3.2. 3.3, 3.4) in the paper follow blackbox (entirely or almost) from known results, and hence the contribution of the present work in these contexts is restricted to framing.

Note that the regret is defined as treating other agents' actions fixed. Thus, Theorem 3.4 is in fact a corollary of earlier work on online non-stochastic control, purely by including the other players' actions in the "disturbances". The size of disturbances inflated by a factor of N, and this can substituted in the earlier results blackbox.

This also extends to Theorem 3.4, which primarily is a Lipschitz constant computation. Additionally, notice that here a DAC policy may not be super meaningful since ideally the agent should also respond to the other agents actions (which the linear policy does allow the agent to do).

The authors note that Theorem 3.3 does not follow from existing online convex regret lower bounds, due to policy regret differences. But consider any LDS with (A,B) = (0,0), this reduces to an online convex game over controls, and now standard T^0.5 lower bounds apply. So, I don't see the barrier here.

In contrast to the above, I really like the set of questions in Section 4. However, Theorem 4.1 provides a non-varnishing bound, thus there is no implication of convergence to an equilibrium-like notion.

### Questions
Can the authors comment on points I might have misunderstood above in the weakness section? Setting aside subjective judgements, I think we should be able to agree on facts.

### Soundness
3

### Presentation
3

### Contribution
2
