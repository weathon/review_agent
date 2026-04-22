# Learning a Game by Paying the Agents

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
We study the problem of learning the utility functions of no-regret learning agents in a repeated normal-form game.
Differing from most prior literature, we introduce a principal with the power to observe the agents playing the game, send agents signals, and give agents *payments* as a function of their actions.
We show that the principal can, using a number of rounds polynomial in the size of the game, learn the utility functions of all agents to any desired precision $\varepsilon > 0$, for *any* no-regret learning algorithms of the agents.
Our main technique is to formulate a zero-sum game between the principal and the agents, where the principal chooses strategies among the set of all payment functions to minimize the agent's payoff. 
Finally, we discuss implications for the problem of *steering* agents. We introduce, using our utility-learning algorithm as a subroutine, the first algorithm for steering arbitrary no-regret learning agents to a desired equilibrium without prior knowledge of their utility functions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies an active, non-equilibrium inverse game problem interacts with any no-regret players by giving payments and signals. The obejctive is to learn the utility functions of players under a equivalent game. The main idea is to cast the principal–agent interaction as a zero‑sum game where the principal’s mixed strategy is a payment vector. Then, authors consdier the problem of steering no-regret learning algorithm to desired result without the access to utility functions of players.

### Strengths
- The zero‑sum formulation simple yet interesting, which requires almost no modeling of the agent’s learning rule beyond no‑regret property.

- The results of this paper hold for arbitrary no‑regret learner, and authors also provide a lower bound implying the tightness of $\epsilon$ and necessarity of $M$.

- The proposed approaches are extended to the problem of steering no-regret learner without the access of utility function. Authors show that the principal-optimal CEP characterizes the value that the principal can achieve.

- The paper is well-written and easy to follow. I appreciate that authors offer clear intuitions for the most of places.

### Weaknesses
- The assumption in (1) needs to hold for all signals, which seems strong to me. In fact, many contextual algorithms ensure provable regret bounds on average over contexts, not uniformly over all contexts.

- The sample complexity shown in Theorem 4.3 has polynomial dependence on M and C, but it does not match the given lower bound. 

- The rate of steering agents to achieve the optimal CEP is $T^{-1/4}$, which seems suboptimal.

### Questions
- I would like to hear whether it is possible to relax the requirement on the signal in Theorem 4.3. Could the authors clarify or discuss potential directions for such relaxation?

- In Section 3, it is mentioned that the knowledge of the mixed strategy $x_t$ could be relaxed. Could the authors elaborate on how the proposed algorithm should be modified in that case?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the problem of learning a game by observing the agents play the game using no-regret algorithms and paying them to influence their behavior. The actor in this game seeks to learn the utilities of the game, up to strategic equivalence, through picking payments and providing signals to the agents in each round of play, under the assumption that the agents play arbitrary, possibly correlated, no-regret algorithms. It is important to note that the agents are in fact assumed to have contextual no-regret, i.e. they have vanishing regret for each signal/ no-regret over strategies mapping signals to actions. The main result of the paper shows that it is possible to learn the game in number of rounds polynomial in the size of the payoff matrix and that there are matching lower bounds, up to polynomial factors. 

The paper first shows a slick method to estimate the utilities of a single agent using payments, by defining a two-player zero-sum game with a unique equilibrium and then shows how to construct no-regret dynamics to generate a sequence of payments that approaches this equilibrium, which in turn reveals the utility function of the agent.  This method is then lifted to learn the utilities of all agents in sequence in a multi-agent game, by rotating through the actions of all other players and using signals to fix their behavior for each learning phase. Other results in the paper include using this result to steer agents to desirable correlated equilibria while starting with no initial information about the game.

### Strengths
The main strength of the paper is in posing a natural question about learning a game by observing no-regret behavior and making meaningful progress in answering it by extending existing technical machinery. In particular, the algorithm to find the utility of a single agent is an elegant construction to re-use no-regret dynamics to find the equilibria of a game despite not seeing full information feedback.

### Weaknesses
The main weakness of the paper is the strong assumption made about the agents having contextual no-regret over the signals employed by the actor learning the game. This gives the actor tremendous leverage over the players practically for "free" since it allows them to "forget" the history of play by switching signals. This is exploited in obtaining an almost direct reduction from multiple players to one player. While this might well be a necessary assumption, there is insufficient technical justification for it.

### Questions
Connected to the main weakness highlighted above -- how necessary is the assumption about no-regret for each signal? For instance, is it at all possible to learn the utilities of a multiplayer game with just a vanilla external regret assumtion?

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Dear authors,
I am finalizing my evaluation and will upload my full review within the next two days. Thank you for your patience.

Best regards,
—Reviewer

### Strengths
TBC

### Weaknesses
TBC

### Questions
TBC

### Soundness
2

### Presentation
2

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
This game addresses a natural question of learning a player's utility functions over time and studies this from an online learning perspective. Although this is a standard problem, the paper differs from the literature in two important ways: first they assume that the players are not in equilibrium. Secondly, they assume that the players are playing a no regret algorithm. The main contribution is a polynomial time algorithm that learns the utilities upto an error of $\epsilon$. They further show that they can guide the agent towards a particular correlated equilibrium with payments (also known as steering). 

The model is as follows. In each round the principal payment functions $P^t_i: A_i \to [0,B]$  gets added to the agent rewards. The principal can also send a private signal. The agent which maximizes the sum of the payment and their own utility will play a mixed strategy $x^t$ which is observed by the principal. The paper's central insight is to reduce this to a zero-sum game between the principal and agent. Effectively the unique optimal in this game is for the principal to offer a payment of $p = 1-u$ the utility, thus making the agent indifferent between the actions. Then the principal can read the utility function by simply taking $p=1-u$. This is because the utility is zero-mean and if the principal updates their weights according to a no-regret algorithm like projected gradient descent (note that the mixed strategy $x^t$ is the derivative of the principal's loss function $(u+p) \cdot x^t$), it will eventually converge to a point where the optimal utility is $u+p$ is a constant. This is a simple but elegant idea that allows the principal to discover the agent's true utility. 

To handle the general setting when there are $n$ agents, the principal uses signals to effectively freeze the other agent's actions.

### Strengths
I think the paper outlines a simple and beautiful idea to learn the utility of any agent in a normal formed game. The dependence and the  learning rate and the associated regret is tight.

### Weaknesses
The complexity to learn scales exponentially on the number of agents and their action profiles. Their lower bound suggests that this task is too hard for many practical games. I also wonder a conference such as EC  might be a better fit than ICLR.

### Questions
N/A

### Soundness
4

### Presentation
3

### Contribution
3
