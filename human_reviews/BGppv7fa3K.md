# Principal-Agent Reinforcement Learning: Orchestrating AI Agents with Contracts

- Decision: Reject
- Scores: 6, 5, 5, 6

## Abstract
The increasing deployment of AI is shaping the future landscape of the internet, which is set to become an integrated ecosystem of AI agents. Orchestrating the interaction among AI agents necessitates decentralized, self-sustaining mechanisms that harmonize the tension between individual interests and social welfare. In this paper we tackle this challenge by synergizing reinforcement learning with principal-agent theory from economics. Taken separately, the former allows unrealistic freedom of intervention, while the latter struggles to scale in sequential settings. Combining them achieves the best of both worlds. We propose a framework where a principal guides an agent in a Markov Decision Process (MDP) using a series of contracts, which specify payments by the principal based on observable outcomes of the agent's actions. We present and analyze a meta-algorithm that iteratively optimizes the policies of the principal and agent, showing its equivalence to a contraction operator on the principal’s Q-function, and its convergence to subgame-perfect equilibrium. We then scale our algorithm with deep Q-learning and analyze its convergence in the presence of approximation error, both theoretically and through experiments with randomly generated binary game-trees. Extending our framework to multiple agents, we apply our methodology to the combinatorial Coin Game. Addressing this multi-agent sequential social dilemma is a promising first step toward scaling our approach to more complex, real-world instances.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper formulates a general framework of principal-agent Reinforcement Learning (RL), which tries to train multi-agent RL agents without centralized control, but instead by using a protocol where a principal agent tries to guide (intentionally misaligned) agent behavior by assigning payments. The paper first constructs a hidden-action principal-agent stochastic game, then propose an meta-algorithm for finding Subgame Perfect Equilibrium (SPE). After analyzing the theoretical property of such meta-algorithm, the paper discusses the possibility of learning it with RL, and then finally how to extend to multi-agent RL. Several experiments show that the proposed method works well in the coin game and tree MDPs

### Strengths
1. This paper addresses a very interesting problem as it considers a novel scenario where the principal needs to regulate agents' behavior but does not have centralized control on each agent. This is even more interesting considering the potential future use of regulating a herd of LLM agents with a learned principal, many of which can self-reflect and improve according to a protocol but we do not have training access to.

2. This paper is well-written and well-structured. It builds up a framework with many complicated notions, but the paper managed to consider one component at a time: first it is the game itself, then an idealized solution, then function approximator (RL), and finally multi-agent. Generally, the paper is easy to follow.

3. The paper gives detailed analysis and proof for its theoretical performance (e.g., convergence property), which is important for a new, economic-inspired RL framework. In the mean time, the main text is kept clean without too many unnecessary notations using intuitive explanations.

### Weaknesses
1. The hyperparameter, $\alpha$ in line 406, could be tricky. As stated by the authors, $\alpha$ is set as a heuristic approach to downweight the importance of payment minimization compared to social welfare (payment minimization cannot hurt social welfare), and thus $\alpha$ should be set very small. However, if $\alpha$ is indeed very close to $0$, the principal agent can simply use its reward to overwhelm all selfish intentions (e.g. sent 1% of its own reward to each individual agent, which is a very large amount for them) and the algorithm would degenerate to independent RL with cooperative objective. Thus, there exists some balance of $\alpha$ that may require tuning. It would be better if the authors can conduct an ablation study on $\alpha$ to show how it impacts the balance between social welfare and payment minimization.

2. There is essentially only one gridworld experiment for the proposed algorithm in multi-agent RL (tree MDP in the appendix is for single-agent). It would be better if more and harder environments could also be tested, such as particle world ones (environments like SMAC-like ones would be even better for camera ready (if accepted) / next submission (if rejected)).

**Minor issues**

1. The notations in the beginning of Sec. 2.2 are unclear. For example, $o$ is an outcome, but is then used as a number ("o-th coordinate of a contract"); expressions like "contract corresponds to the outcome $o$" would be better.

2. The punctuation "." at the end of the caption of Tab. 1 is missing.

3. The current contribution part is not stating the paper's contribution, but is rather an outline of the paper. It would be better if the outlines can be more briefly summarized into a paragraph, with each highlight on key novel aspect as one sentence.

### Questions
I have a question: can the solution proposed by this framework improve social welfare on a more complicated, closer-to-mainstream MARL environment (e.g. a continuous version of coin game in a multi-agent particle world style)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work introduces a specific contract mechanism into MARL, defined a principal-agent stochastic game based on this mechanism, and solved the game using a straightforward value-based RL algorithm, conducting some experiments as well.

### Strengths
- Introducing contracts into MARL is an issue of interest to the community.
- This work is comprehensive, thoroughly addressing the bilevel MARL problem related to the contract mechanism.

### Weaknesses
- Algorithm 1 is vanilla, and lines 3 and 4 imply solving an MDP, which can be very time-consuming in a larger environment. Although a value-based method is used and is model-free, it is a primitive algorithm. Some analyses are straightforward to the community. The main contribution lies in introducing the contract mechanism and reformulating the game. 

- Is the contract in this work a mechanism, or is it a problem to be solved?
    - As a problem: The contract reformulates a new game, and the logic of the experimental design is to verify that their proposed algorithm can solve this problem. So, could some other existing algorithms potentially solve this problem?
    - As a mechanism: For instance, with this contract mechanism, the prisoner’s dilemma can be effectively resolved, and the coin game can achieve high scores. In this sense, this work should compare with other algorithms to demonstrate the advantages of the contract. The experiments in Figure 2 are clearly insufficient.

- The contract has a very specific meaning here, where after an agent takes an action and a result is achieved, the principal pays the agent based on the result. This setting is a variant of incentivizing agents.
    - In this way, this setting is closely related to LIO ("learning to incentivize other learning agents"). The authors cited LIO but did not discuss the differences or conduct comparative experiments.
    - In LIO, players are symmetric and do not need to wait for an optimization problem to converge at one level (in the spirit of online cross-validation). And the incentivization occurs during the training phase without changing the game’s structure. So, could similar methods be directly applied?

- Coin Game might not highlight the advantages of the contract. A more suitable scenario would be a task with division of labor, such as the cleanup task. There is already work on MARL contracts in this scenario, "Formal Contracts Mitigate Social Dilemmas in Multi-Agent Reinforcement Learning." And LIO also uses this scenario.

Minors
- There is a large margin below Figure 1 and Figure 2.
- Some single quotation marks in the text should be changed to double quotation marks.

### Questions
1. How do the distinct and possibly conflicting objectives affect the meta-algorithm bilevel optimization procedure compared to same downstream tasks?
2. Nudging is an interesting solution for learning approximation error. However, who provides such nudging, i.e. in equations (20)-(22)? How does nudging change the discontinuous utility of the principle? 
3. Besides, in the contracts of our general life, breach of contract is also an effective method to steer an agent’s behavior. What is the relationship between breach and nudging?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studies a problem where a principal must guide the actions of an agent (or several agents) to maximize the principal's utility using payments. This problem is formalized using a Markov decision process, where the principal chooses a policy that determines the payment to the agent depending on the state and outcome. The authors propose and analyze a simple meta-algorithm that ensures convergence to subgame perfect equilibrium (SPE). Moreover, experiments are conducted on the Coin Game environment, which is a medium-sized two-player social dilemma. The experiments show the effectiveness of the proposed approach for maximizing the principal's utility (in this case, social welfare).

### Strengths
- The intersection of reinforcement learning and contract design is an interesting area of study. 
- The problem formulation is neat and well motivated. 
- The paper is well-written and overall a pleasure to read.

### Weaknesses
- I struggle with identifying the technical novelty and the significance of the contributions:  
	- From a technical standpoint, it appears that by considering SPE the problem becomes fairly simple and boils down to performing backwards induction (using standard RL techniques). In particular, as far as I can tell, the results of Theorem 3.3 and 3.4 follow straightforwardly from SPE+ well-known RL results. Could you highlight any specific novel technical challenges under the proposed problem setup? 
	- Conceptually, the idea of a principal orchestrating agents in social dilemma to resolve miscoordination and maximize social welfare has been considered before (this is also essentially what much of mechanism design is about). In Appendix A.3, you argue that your work differs in perspective from prior work on "solving" social dilemmas by modifying the agents' rewards. In particular, to distinguish your work from the others you write "While the principal effectively modifies agents’ reward functions, the payments are costly." Are not the payments also costly in the models studied in prior work? For instance, in Yang et al. (2020), the agents transfer the rewards to other agents which is of course costly to the paying agent. Maybe the work of [Jackson and Wilkie (2002)](https://www.jstor.org/stable/3700662) is also relevant / of interest here.  
- From my undestanding, principal-agent theory is broader than just contract design, and it would be helpful to the reader to discuss this (i.e., principal-agent theory) related work as well. For instance, [Myerson (1982)](https://www.sciencedirect.com/science/article/abs/pii/0304406882900064), [Zhang and Conitzer (2021)](https://arxiv.org/abs/2105.06008), [Gan et al. (2022)](https://arxiv.org/abs/2209.01146). There is also work on Bayesian persuasion which is part of the principal-agent problem family. Appendix A.2 does make it sound like principal-agent problems are just contract design problems.

### Questions
- $E_t$ in Proposition 4.1 is not defined in the main text. Overall, Proposition 4.1 is very hand-wavy. For example, you write that "This utility can be achieved through small(?) additional payments that counteract the agent's deviations, ...". I'm aware that more details are provided in the appendix, but the proposition should be stated rigorously or, alternatively, be made an informal remark.  
- In your opinion, why would the centralized orchestrating of agents (i.e., the principal-agent perspective) be more appealing to resolve incentive issues than the large-scale decentralized individual bargaining perspective many other works take? (You mentioned some of these works in Appendix A.3).

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
3

### Summary
The paper proposes to apply principal--agent theory to multi-agent RL with the goal of applying their methods to resolve social dilemma-like situations between AI agents, allowing different AI agents to collaborate more effectively. They define principal--agent MDP as a stochastic game with turns, where at each time-step principal observes the state and chooses its action (a contract), the agent observes state and principal's action and then chooses its action. As a solution concept, they choose the subgame-perfect equilibrium. They present a meta-algorithm that is essentially a bilevel optimization problem, which when solved, proven to find the SPE in their setting. Then they propose a Q-learning based approach to instead learn the SPE rather than computing it exactly. They present an early theoretical result connecting the approximation error in Q functions to a bound on principal's utility. They test their approach in the Coin Game environment in terms of both social welfare and convergence.

### Strengths
- The problem of orchestrating an ecosystem of AI agents to collaborate with each other is an interesting and relevant problem.
- The application of principal--agent theory and its combination with RL appears to be novel and also interesting.
- The paper is easy to read, and ideas are expressed clearly.

### Weaknesses
- The presented empirical and theoretical results appear too weak. Convergence of the meta-algorithm is not surprising, what is left out is the computational complexity of solving the bilevel optimization problem. Empirical results are also limited, and there is no comparison to other MARL algorithms that tackle social dilemmas (see the question).

- The connection to the motivating example seems lost once the Introduction is over. More discussion at the end about how this allows for orchestrating of AI agents is needed.

### Questions
1. Is there a reason why you have not benchmarked your method against other MARL methods that allow agents to gift each other rewards? For example [1,2,3]? I would expect these to be at least discussed as an alternative approach.

2. For the case of orchestrating AI agents (your motivating example), what would be the advantage of your principal--agent framework over things like adaptive mechanism design?

[1] Yang, Jiachen, et al. "Learning to incentivize other learning agents." Advances in Neural Information Processing Systems 33 (2020): 15208-15219.

[2] Lupu, Andrei, and Doina Precup. "Gifting in multi-agent reinforcement learning." Proceedings of the 19th International Conference on autonomous agents and multiagent systems. 2020.

[3] Kolumbus, Yoav, Joe Halpern, and Éva Tardos. "Paying to Do Better: Games with Payments between Learning Agents." arXiv preprint arXiv:2405.20880 (2024).

### Soundness
3

### Presentation
3

### Contribution
3
