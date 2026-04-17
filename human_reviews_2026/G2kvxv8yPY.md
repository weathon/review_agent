# A Bit of Freedom Goes a Long Way: Classical and Quantum Algorithms for Reinforcement Learning under a Generative Model

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
We propose novel classical and quantum online algorithms for learning finite-horizon and infinite-horizon average-reward Markov Decision Processes (MDPs). Our algorithms are based on a hybrid exploration-generative reinforcement learning (RL) model wherein the agent can, from time to time, freely interact with the environment in a generative sampling fashion, i.e., by having access to a "simulator." By employing known classical and new quantum algorithms for approximating optimal policies under a generative model within our learning algorithms, we show that it is possible to avoid several paradigms from RL like "optimism in the face of uncertainty" and "posterior sampling" and instead compute and use optimal policies directly, which yields better regret bounds compared to previous works. For finite-horizon MDPs, our quantum algorithms obtain regret bounds which only depend logarithmically on the number of time steps $T$, thus breaking the $O(\sqrt{T})$ classical barrier. This matches the time dependence of the prior quantum works of Ganguly et al. (arXiv'23) and Zhong et al. (ICML'24), but with improved dependence on other parameters like state space size $S$ and action space size $A$. For infinite-horizon MDPs, our classical and quantum bounds still maintain the $O(\sqrt{T})$ dependence but with better $S$ and $A$ factors. Nonetheless, we propose a novel measure of regret for infinite-horizon MDPs with respect to which our quantum algorithms have $\operatorname{poly}\log{T}$ regret, exponentially better compared to their classical counterpart. Finally, we generalise all of our results to compact state spaces.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The manuscript deals with regret bounds of online RL in MDPs with finite horizons and MDPs with infinite horizons. Classical and quantum algorithms in the special case with oracles are considered. This is a purely theoretical work, covering a total of 49 pages.

### Strengths
* Very careful preparation with comprehensive discussion of the literature.
* Innovative approach to this established research topic.

Further comments:\
Although it is unusual to mention selected publications in the abstract, I think this is good here because it makes it very concrete and carefully presented.

The use of many footnotes is rather unusual. However, I think it fits in with the very careful style of the work.

One could criticize the enormous length of the appendix, but I think it is well done, because the main text can be understood well without the appendix, and the appendix provides useful additional information.

### Weaknesses
I can't identify any clear weaknesses.

Perhaps your own contribution could be mentioned in one central place and described in more concrete terms there.

Further comments:

In “One of the most famous measures is that of regret,” I don't like the word “famous.”

Similarly, in “The secret of the improved performance,” I don't like the word “secret.”

It's probably a matter of taste, but I think “Conclusion” is more accurate than “Conclusions.”

„a RL“ -> „an RL“

### Questions
* Which relevance do the results have for further research?
* Which relevance could the results potentially have for future applications?

### Soundness
4

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
4

### Summary
This paper proposes quantum and classical algorithms for online reinforcement learning in both finite-horizon and infinite-horizon MDPs. The authors design quantum versions of variance-reduced backward induction and robust value iteration, achieving query complexities that improve the best classical bounds by factors of $\sqrt{A}$ and smaller H-dependence. They introduce an online “exploration–generative” protocol and prove regret bounds that are minimax-optimal for classical algorithms and enjoy an exponential-in-T advantage for a newly defined expected regret when quantum oracles are used. Results are extended to continuous state spaces via Hölder-smooth discretization.

### Strengths
1.	The paper provides solid theoretical proofs for its results. 
2.	For the online learning problem of MDPs, when we introduce quantum algorithm, it is difficult to define proper regret. The authors introduce a novel model with classical exploration phases and classical/quantum generative phases to solve this difficulty.

### Weaknesses
1.	In section 2 (to compute optimal policies), the authors consider undiscounted version of MDPs (both finite-horizon and infinite-horizon). In my understanding, the discounted version is more important and the quantum algorithm in this version has already been proposed. The motivation and technique challenges of the undiscounted version are not clearly explained. 
2.	In section 3 (online learning version), the authors introduce a novel model which splits the interaction into two types of phases: classical exploration phases and classical/quantum generative phases. Although this idea solves the difficulty to define proper regret of the quantum algorithm, it undermines the most critical challenge in online learning: the algorithm must balance exploration and exploitation. This significantly reduces the problem's difficulty. I don’t think it is reasonable. 
3.	In section 3, the authors require that the length of generative phase is at most $O(\tau)$ where $\tau$ is the length of the previous exploration phase. We must have some limitations of generative phase, but such a restriction appears arbitrary and lacks justification.

### Questions
1.	Related to weakness 1: Is there any special motivation to consider quantum algorithm for finite-horizon and infinite-horizon undiscounted MDPs? Comparing to the quantum algorithms in previous work for infinite-horizon discounted MDPs, what is the main technique challenges for undiscounted version?
2.	Related to weakness 2: In practical scenarios, is there any motivation for proposing such model?
3.	Related to weakness 3: why should we require that the length of generative phase is at most $O(\tau)$？

### Soundness
4

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
The authors propose both classical and quantum online algorithms for reinforcement learning (RL) tasks. These algorithms are rooted in an exploration-generative RL framework, wherein the agent can occasionally interact freely with the environment by utilizing access to a simulator of the environment. The authors claim that their proposed algorithms address and mitigate common RL challenges, such as “optimism in the face of uncertainty” and “posterior sampling.”

### Strengths
The paper is well-written and engaging. The discussed related work is extensive and provides a good overview on similar research.

### Weaknesses
However, the applicability of the proposed algorithms is strictly confined to RL scenarios where a generative model with comprehensive knowledge of the state and action spaces—as well as the reward function—is available. Moreover, it assumes the transition probabilities can be queried through an oracle. Such oracles are not novel in RL research and are frequently associated with methods categorized as model-based RL. Past research has already explored quantum oracles extensively, particularly focusing on achieving quantum advantage via quantum sampling.

A key point of concern is the lack of clarity regarding the benefits of separating the exploration phase (classical) from the policy learning phase (classical/quantum). If the agent has access to a flawless oracle, it is unclear why a classical exploration on the MDP (Markov Decision Process) would be necessary. Additionally, the rationale behind limiting access to the oracle (referred to as “the true MDP”) needs to be elaborated. For the paper to be more impactful, it should include a clear justification for the algorithm’s design choices as well as an explanation of how, in practical settings, the oracles representing the true MDP might be obtained.

Minor Issues:
The citation format in the abstract, “Ganguly et al. (arXiv’23) and Zhong et al. (ICML’24),” is unconventional and should follow standard citation styles.
Page 9: There is a typo in “a RL.” It should be corrected to “an RL.”

### Questions
Why is the access to the oracle limited?
Why is the oracle not used to learn an optimal policy offline?
How can in practical settings oracles representing the true MDP be obtained?

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
In the context of quantum RL, this paper studies both computing near-optimal policies under a generative model for (generalized) tabular MDPs, and
online reinforcement learning (regret minimization) in finite-horizon and infinite-horizon average-reward MDPs. The basic setting for this generalized tabular MDP is that the state space $\mathcal{X}$, although continuous, can be approximated with an $\epsilon$-net by the Holder continuous condition, reducing the problem to a standard tabular RL.

For the generative model setting, this paper proposes a novel algorithm with a quantum maximum finding subroutine based on the classical value iteration algorithm, which reduces the sample complexity from $O(A)$ to $O(\sqrt{A})$ in the finite horizon MDP. The usage of quantum mean estimation also provides a quadratic improvement with regards to $\epsilon$, which has been investigated in the literature [1]. For infinite-horizon average-reward MDP, this paper revises the classical (extended) value iteration in favor of quantum mean estimation, and achieves a sample complexity of $\tilde{O}(\Lambda |\mathcal{S}_n| \sqrt{A} / (1 - \nu) \epsilon)$ in the case that the Bellman operator is a $\nu$-contraction.

For the general online exploration setting, this paper studies the regret minimization with a new formulation. The whole exploration process is divided into two phase: the exploration phase and the generative phase. The agent is allowed to interact *freely* with the quantum MDP in the generative phase, as if there is a generative model and without incurring any regret. In the exploration phase, the agent executes current policy and incurs corresponding regret. The total length of a generative phase must be the same order of previous exploration phase. This paper proposes a unified algorithm for both finite horizon MDP and infinite horizon MDP, where the agent uses the algorithms of generative model setting to approximate the optimal policy in the generative phase, and the execute this policy in the exploration phase. The regret for the finite horizon setting is $\tilde{O}(\min(HSA, H^2S\sqrt{A} \log T))$, improving on the order of $S, A$ of previous works. The regret for the infinite horizon setting is $\tilde{O}(\Lambda \sqrt{T} + \Lambda S \sqrt{A} \log^2 T)$, where the term $\Lambda \sqrt{T}$ can be omitted if an expected regret is used so as to achieve the $\mathrm{poly} \log T$ regret.

[1]. Zhong et al., Provably efficient exploration in quantum reinforcement learning with logarithmic worst-case regret.

### Strengths
1. The application of quantum computing (e.g., the quantum mean estimation subroutine) to online exploration of RL is an interesting and important problem. This paper pushes the boundary of this problem.

2. As far as I know, this is the first paper investigating quantum RL in the setting of infinite-horizon average-reward MDP. It shows that the sample complexity can also be improved quadratically with the help of quantum mean estimation.

3. The idea of applying quantum maximum finding subroutine in the hoeffding-style value iteration algorithm is novel, saving a $\sqrt{A}$ in the regret bound and sample complexity.

### Weaknesses
1. There is a major problem in the formulation of this "exploration-generation" two-phase procedure, which assumes that the agent can use the oracles as *a generative model* in the generation phase *without incurring any regret*. There are indeed many works in the literature of classical RL using this idea of "lazy update" to design sample-efficient algorithms such as [1, 2], but none of these works assume the access to a generative model nor assume the data collection phase incurs no regret. They can use this lazy update because *they use the data from the "exploration" phase to estimate the value function directly*, instead of collecting new data to estimate the value function in a new "generation" phase. In this work, however, the agent is able to use a generative model to collect grand new data in the generation phase, which means the agent is able to explore the MDP in an **arbitrary** state-action distribution. On the other hand, the core assumption of online RL for unknown environments is that the agent should create favorable state-action distribution themselves by taking good policies to find the high-rewarded state and action. The basic assumption here is, you can **NOT** collect any data from the states that you never know how to get there in an online exploration problem (e.g., see all of the references from line 365 to 367 in this paper), which is also the case for any real-world tasks. Therefore, there is a large gap between the real online exploration RL and the two-phase model proposed in this paper. All the results of Section 3 shall be RL with a generative model instead of online exploration.

2. For the results of the quantum algorithms in the infinite horizon MDP, there is an extra contraction measure $\nu$ of the Bellman operator in the sample complexity, which does not appear in classical RL literature like current SOTA [5]. The sample complexity bound has a $(1 - \nu)^{-2}$ dependency for the classical setting and a dependency of $(1 - \nu)^{-1}$ for the quantum setting. This term mainly comes from the pipeline of value iteration of Algorithm 5. What is the reason to introduce $\nu$ here? Is it because if one uses a conventional decomposition of regret like [6] then an extra martingale term of $O(\sqrt{T})$ will appear? This contraction term somehow makes it harder to evaluate the sample complexity bound of Result 2.

[1]. Zhong et al., Provably efficient exploration in quantum reinforcement learning with logarithmic worst-case regret.

[2]. Jacksch et al., Near-optimal Regret Bounds for Reinforcement Learning.

[3]. Fruit et al., Efficient Bias-Span-Constrained Exploration-Exploitation in Reinforcement Learning.

[4]. Ayoub et al., Model-Based Reinforcement Learning with Value-Targeted Regression. 

[5]. Zhang et al., Sharper Model-free Reinforcement Learning for Average-reward Markov Decision Processes.

[6]. Bartlett et al., REGAL: A Regularization based Algorithm for Reinforcement Learning in Weakly Communicating MDPs.

### Questions
1. What is the data collected in the exploration phase used for in Algorithm 1?

2. As the essential setting of this paper is tabular MDP, why is the paper using a continuous state space $\mathcal{X}$ with a $1/n$-covering $|\mathcal{S}_n|$ but all of the bounds (except for $\mathcal{X} = [0, 1]^D$) depends on $|\mathcal{S}_n|$? It is quite a common sense that one can use the $\epsilon$-net trick to apply the results of tabular RL to a continuous space, but this does not lead to any improvement since the size of the $\epsilon$-net has the same order as the original space $\mathcal{X}$. The analysis for the discretizaiton error of $\mathcal{S}_n$ in the Holder continuity seems to only overly complicate the paper. A right example here is the Theorem 1 of [4], which uses a log-covering number as the complexity measure. 

3. Is is possible to use the reduction from an average-reward MDP to a discounted reward MDP mentioned in [5] to get rid of the contraction measure $\nu$?

4. Why the paper is consistently using "backward induction algorithm" to name the well-know value itration algorithm in RL?

5. What is $\mathcal{L}0$ in line 1993?

6. It would be better to provide extra sections in the appendix to introduce the algorithms and summarize the results since the algorithms are not given in the main context, instead of mixing the algorithms and the full proof into a single section.

[4]. Ayoub et al., Model-Based Reinforcement Learning with Value-Targeted Regression. 

[5]. Zhang et al., Sharper Model-free Reinforcement Learning for Average-reward Markov Decision Processes.

### Soundness
3

### Presentation
2

### Contribution
2
