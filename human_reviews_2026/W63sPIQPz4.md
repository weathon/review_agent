# Learn to change the world: Multi-level reinforcement learning with model-changing actions

- Decision: Reject
- Scores: 4, 8, 2

## Abstract
Reinforcement learning usually assumes a given or sometimes even fixed environment in which an agent seeks an optimal policy to maximize its long-term discounted reward. In contrast, we consider agents that are not limited to passive adaptations: they instead have model-changing actions that actively modify the RL model of world dynamics itself. Reconfiguring the underlying transition processes can potentially increase the agents' rewards. Motivated by this setting, we introduce the multi-layer configurable time-varying  Markov decision process (MCTVMDP). In an MCTVMDP, the lower-level MDP has a non-stationary transition function that is configurable through upper-level model-changing actions. The agent's objective consists of two parts: Optimize the configuration policies in the upper-level MDP and optimize the primitive action policies in the lower-level MDP to jointly improve its expected long-term reward.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper formalizes reinforcement learning with model-changing actions—actions that modify the environment’s dynamics—and introduces multi-level configurable MDPs where an upper-level MDP selects (or configures) the lower-level transition kernel, and a special time-variant configurable MDP (TVCMDP) with continuous configuration under a cost budget. Theoretical pieces include a linearization of value w.r.t. transition kernels, a convex surrogate for TVCMDP with exponential configuration costs, and error bounds for bi-level estimation and configuration uncertainty. Algorithms include a model-based bi-level value iteration procedure that alternates solving lower-level MDPs and an upper-level configuration MDP. Experiments on a synthetic TVCMDP, a synthetic bi-level MDP, and two standard tasks—Cartpole and Gridworld/Block-world—demonstrate that learned configuration policies can outperform non-configuring or random-configuring baselines and move performance closer to an “oracle” fixed-best kernel.

### Strengths
The paper makes the idea of environment-reconfiguration actionable by cleanly separating configuration (upper level) from control (lower level) and by treating the lower-level transition kernel itself as the state of the upper level—an uncommon but crisp abstraction; it provides a tractable convex approximation for TVCMDP via a linearization of the value function and a budgeted exponential cost, along with bi-level error bounds tying configuration/estimation deviations (δc, δg, Δ) to value degradation; the algorithmic template (solve lower-level values/policies, then plan over kernels) is simple and broadly applicable; and the case studies (Cartpole with discrete parameterizations; Block-world with slip parameter; synthetic TVCMDP/bi-level) collectively show that configuration can materially improve returns over non-configuring or random policies and approach an oracle that fixes the best kernel.

### Weaknesses
1. Empirical scope and baselines are limited for claims about generality and efficiency: the upper-level uses value iteration (Cartpole) or DQN (Block-world) over small, discretized configuration spaces, and lower-level policies are DQN or value iteration; there is no comparison to known configurable-MDP solvers or modern bilevel/meta-RL approaches, making it hard to attribute gains to the proposed formulation rather than to problem simplicity. 

2. Assumptions in theory are strong and somewhat idealized: known reward functions and the ability to estimate discrete kernel sets for the upper level; while the error bounds are clean, experiments do not probe sensitivity to misspecification or to continuous/large kernel spaces. 


3. Evaluation design can blur configuration power with training protocol advantages: for Cartpole the upper level deterministically switches among four handcrafted environments and evaluates over only 20 episodes per setting; Block-world precomputes rewards via offline VI and discretizes $\alpha$ into 1000 points, raising questions about scalability and online sample efficiency when such precomputation is infeasible. 


4. Problem framing relative to prior configurable-MDP work could be sharper: while the paper argues that “the lower-level kernel as upper-level state” is novel, the empirical section does not directly compare to CMDP baselines (e.g., gradient or Stackelberg formulations) on the same tasks to evidence distinct benefits of an explicit upper-level MDP abstraction.

### Questions
1. For TVCMDP, how sensitive are solutions to $\alpha, \beta$, and budget $B$ in the exponential cost, and where does the linearization break (e.g., $‖x‖$ bounds), empirically? 


2. How does performance scale when the configuration state space grows (e.g., $>4$ Cartpole parameter sets or multi-parameter continuous kernels)? 


3. Could you report wall-clock, env-step budgets, and seeds for upper- and lower-level training, and provide return-vs-time curves to separate algorithmic from engineering speedups?

4. In the synthetic bi-level study, can you vary $\delta_c$, $\delta_g$, and $\Delta$ to empirically validate the proved error bounds (Lemmas 2–3) and show slopes consistent with theory?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces the multi-layer configurable time-varying Markov decision process (MCTVMDP) with model-changing actions that actively modify the RL model of world dynamics itself. The new RL framework proposed in this paper has the potential and mechanism of breaking out of the current model, through actions that change or improve the underlying MDP. The authors also study special cases of the proposed RL mechanism with model-changing actions; they consider configurable RL for time-varying environments and multi-level (including bi-level) environment-changing RLs. They propose a multi-level value iteration algorithm for solving multi-level configurable RL problems. Lastly, there are some experiments done in Cartpole and Block-world environments to conclude that the proposed framework is adaptable to more complex and continuous RL environments and to testify to the feasibility of the bilevel configurable MDP model.

### Strengths
Although the notion of configurable MDP (CMDP) has previously been proposed, the main strengths can be summarized as follows:
1. The primary contribution of this research lies in the study of multi-layer configurable MDPs. 
2. The authors employ upper-level MDP abstractions/learning to model/improve configuration actions. 
3. Additionally, they deal with time-varying non-stationary lower-level MDPs (time-varying transition kernels), which were not investigated by these previous works.
4. They introduce a model-based value iteration algorithm.

### Weaknesses
Some questions and concerns must be addressed for both theory and experiments. 
1. How do the upper-level model and lower-level model connect?
2. On what basis does the agent select actions? 
3. In the Bi-level model-based value iteration algorithm, how much time does it take to converge to the optimal policy?
4. How would your bi-level model-based value iteration algorithm’s performance change with different environments and different algorithms other than the DQN?
5. Why did you specifically choose to employ the DQN algorithm?
On page 18,  figure 3, why is the performance of Oracle (blue bar) closest to the performance of the bi-level MDP configuration?

Minor comments:
1. On page 9, “We present the configurable Carpole”, should be Cartpole.
2. On page 5, section 3, line 1: Should be TVCMDP instead of TVCDMP.

### Questions
I would highly recommend including a conclusion that summarizes the main notion and contributions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a MDP "inception" framework: a two-level process where the upper-level MDP action picks a configuration that affects the transition kernel of the lower-level MDP. Rewards of the upper level MDP are defined as the expected return of the next episode of the lower-level optimal policy under the chosen configuration after substracting a cost for "changing the world": the configuration is picked by perturbing previous transition kernel and the cost function reflects the price of the perturbation (chosen as exponential function).
The authors also provide a first-order value linearization to turn continuous kernel tweaks into a convex program, as well as estimation and propagation error bounds (based on classical results from RL). The experiments use the famous cartpole environment as toy example to validate the theoretical claims.
The related work is addressed, especially Configurable Markov Decision Processes, which is probably the most relevant. "Contextual Bilevel Reinforcement Learning for Incentive Alignment" also presents similar ideas, but in the case where the "world changes" are not controlled by the upper-level MDP.

### Strengths
The authors claim novelty relative to configurable RL by targeting time-varying, non-stationary RL and continuous configuration. While prior work (as stated in the summary) also addresses similar settings, the lower–upper decoupling perhaps allows a clearer formulation of estimation and error bounds by leveraging well-established results from tabular RL. The writing is clear and easy to follow. The main idea can be understood even by readers with limited prior exposure to reinforcement learning.

### Weaknesses
While the proposed two-level formulation offers an appealing perspective and an intuitive conceptual separation, the same setting can be expressed within the standard Markov Decision Process (MDP) framework: this can be achieved by augmenting the state and action spaces such that the action space becomes the Cartesian product of the configuration and the lower-level action spaces. In such formulation, the configuration component is selected once at the beginning of each episode, while the lower-level actions are kept as is. The state space remains unchanged, but the configuration affects the transition dynamics; consequently, the overall process can be represented by a single transition kernel conditioned on the joint action pair (configuration, action). This perspective implies that the proposed hierarchical structure may be viewed as a reformulation of an augmented flat MDP, rather than a fundamentally new class of decision process.

We also raise concerns regarding the level of academic rigor in the paper’s presentation. The introduction, while comprehensive, often reads as overly verbose and could benefit from greater synthesis. For example, the second and third paragraphs substantially overlap in content and could be merged, and the sporadic use of quotation marks around common reinforcement learning terminology (e.g., “MDP,” “infinity,” “infinite-horizon”) yields a conversational tone, which should be avoided in a conference submission. Adopting a more precise and concise tone throughout would significantly enhance the professionalism and readability of the paper.

While the CartPole simulations are not sufficient indicators, we believe that the main flaw of this paper lies in the optimization objective:
- The authors used a linearization $V_\pi(P^\pi + x) \approx N + MxN$, with $M = \gamma (I - \gamma P_\pi)^{-1}$ and $N = (I - \gamma P_\pi)^{-1} r_\pi$ for a fixed $\pi$, so a more accurate notation would be $N^\pi + M^\pi x N^\pi$.
- Accordingly, the objective becomes $\max_{x_k} \max_{\pi} N^\pi + M^\pi x N^\pi$, and $N^\pi$ can no longer be omitted, contrary to the claim in the paper.
- Moreover, the objective is no longer convex, at least in the general case, as it is not shown to be jointly convex in both $x$ and $\pi$. A simple example is the function $(x, y) \mapsto xy$, which is convex in each variable individually, but not jointly convex.
- Even if we disregard this issue (which by itself undermines the proposed approach), the linearization is only valid in a neighborhood of the current $P^\pi$, where $|x_{ij}| \leq \max P^\pi_{i,j}$ (not merely $|x_{ij}| \leq 1$). Consequently, if the proposed scheme is to be followed, the modification $x$ of the kernel must be bounded at each step rather than freely vary in the $[-1, 1]$ interval, thereby requiring a "trust-region" optimization formulation.
- The cost bound does not seem to be used in Algorithm1 nor in its low-level formulation.

### Questions
The main question at this point concerns the linearization. We expect a solid and rigorous proof of any proposed modification to salvage the paper. Although we recognize that such an effort may be substantial within the given time frame, we are willing to raise our rating if it is adequately addressed.

### Soundness
1

### Presentation
2

### Contribution
2
