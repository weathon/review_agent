# Exploration in the Face of Strategic Responses: Provable Learning of Online Stackelberg Games

- Avg Score: 5.00
- Decision: Reject
- Scores: 5, 6, 6, 5, 3

## Abstract
We study online leader-follower games where the leader interacts with a myopic follower using a quantal response policy. The leader's objective is to design an algorithm without prior knowledge of her reward function or the state transition dynamics. 
Crucially, the leader also lacks insight into the follower's reward function and realized rewards, posing a significant challenge. 
To address this, the leader must learn the follower's quantal response mapping solely through strategic interactions --- announcing policies and observing responses. 
We introduce a unified algorithm, Planning after Estimation, which updates the leader's policies in a two-step approach.  
In particular, we first jointly estimate the leader's value function and the follower's response mapping by maximizing a sum of the Bellman error of the value function, the likelihood of the quantal response model, and a regularization term that encourages exploration. The leader's policy is then updated through a greedy planning step based on these estimates. Our algorithm achieves a $\sqrt{T}$-regret in the context of  general function approximation. 
Moroever, this algorithm avoids the intractable optimistic planning and thus enhances implementation simplicity.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studies online leader-follower Markovian games where a leader interacts with a myopic follower using a quantal response policy. The leader lacks knowledge of the follower's reward function, posing a challenge in learning through strategic interactions. The authors introduce the PES algorithm, which estimates the leader’s value function and the follower’s response mapping by maximizing the sum of the Bellman error of the value function, the likelihood of the quantal response model, and an additional regularization term. This approach allows for policy updates and achieves sublinear regret, avoiding complex optimistic planning. Finally, the authors show how their algorithm applies to RLHF problems.

### Strengths
- The problem formulation is clear, and I believe it could be of interest to the RL community.
- The authors propose a possible application of their algorithm to RLHF settings, which is currently a hot topic. Thus, this represents a strength.

### Weaknesses
- At line 149, the authors refer to a specific communication layer in Markovian Stackelberg games, but I believe it should be called "commitment layer".
- It is a bit confusing in the initial section when the authors say, "we use the notion of sample complexity", and then they provide a regret upper bound.
- Given the current version of the paper, I cannot evaluate the theoretical guarantees of the algorithm proposed in the paper, as the authors do not provide any discussion about the tightness of their regret upper bound. 
- Furthermore, several lemmas are taken from previous papers. The authors should better outline the technical steps to derive their results.

### Questions
- The use of entropy to model the followers' bounded rationality is something that exists in the literature? Can you provide additional related works that employ the same approach? 
- The paper shares several features with (Chen et al., 2023). Could the authors clarify the main differences between this work and theirs?
- Is the regret upper bound tight?

### Soundness
3

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
3

### Summary
The paper studies the online stackelberg game, where a leader and a myopic follower interact with the environment and receive feedback. The paper proposed a Planning After Estimation algorithm and showed its regret bound. The authors also applied the algorithm to RLHF framework, exhibiting its potential application.

### Strengths
Compared to Chen et al. (2023), the PES algorithm is easy to implement with rigorous theoretical guarantees. The PES algorithm is fairly easy to understand with two steps. The algorithm is also applicable to the RLHF framework, which shows a potential nice application to real-world problems.

### Weaknesses
However, I feel that the assumption of bounded rationalness is quite restricted. From my understanding, the follower takes the best strategy with an extra regularized term, assuming that the follower has the knowledge of the environment such as the reward function. Does this assumption make sense in real-world application? What if the follower is generally unaware of the environment and has to learn from scratch?

### Questions
Can the online stackelberg game be formulated as a pure single-agent RL game? It seems that the quantal response model of the follower can be treated as one part of the environment. If so, why cannot traditional single-agent RL algorithms be applied to the game?

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
4

### Summary
This paper investigates an online learning problem focused on the leader’s optimal policy in an episodic Markov Stackelberg game. In each round, the leader announces a policy, prompting the follower to respond based on a myopic quantal response strategy. The leader then observes the resulting trajectory and receives bandit feedback, even though both the reward functions and transition model are unknown beforehand. To address this, the authors propose an algorithm, Planning after Estimation, which learns the follower’s reward function and the leader’s value function through function approximation. The algorithm operates by first estimating the leader’s value function and the follower’s response model, followed by a planning step to refine the leader's policy. This approach is efficient, simpler to implement, and achieves sublinear regret under certain assumptions. Furthermore, the paper analyzes the algorithm’s application in a special case relevant to Reinforcement Learning with Human Feedback (RLHF).

### Strengths
1. Originality:  The PES algorithm is a fresh approach to solve Stackelberg equilibria in online settings with general function approximation, emphasizing ease of implementation.
2. Quality: Extensive proofs are presented for sublinear regret and computational feasibility, underlining the method's theoretical robustness.
3. Clarity: This paper presents a clean and implementable algorithm.
4. Significance: It  solves the proposed online learning problem.
5. Application to RLHF: Extending PES to an RLHF context demonstrates its practical relevance and adaptability in settings beyond purely theoretical constructs.

### Weaknesses
1. Contribution: Currently I do not assess the contribution of this work as very strong. As there are existing works on the same problem, to better clarify the significance of this paper, it is necessary to present a quantitative comparison between previous results and the proposed algorithm, but it seems missing in the paper. 
Although the regret has a $\tilde{O}(\sqrt{T})$ dependence on $T$, this result is somewhat restrictive: (1) The regret depends on the decoupling coefficients and covering numbers, which may further depend on $H$ and $T$, while the typical value of covering numbers is not discussed; (2) The algorithm implicitly assumes that $\ita$ is known to the leader; (3) It is based on a function approximation approach, requiring corresponding assumptions for tractability. 
Moreover, the result seems less meaningful for the turn-based special case, which includes the presented case study on RLHF model. This is because when the follow er's response only depends on the leader's action instead of mixed strategy, the leader is essentially facing a standard Markov decision process, which is already well-studied.
2. Writing: The writing is acceptable but could be improved in some parts. 
(1) The abstract and most early parts of the introduction focus on explaining Stackelberg games, but do not clarify that the model involves a Markov state transition, which is part of the main challenge. (2) The definition of sample complexity in the preliminary section is redundant, as it is not used. 
(3) Some formulas need more explanation, such as (3.1) $\nu_h$, (4.8) the meaning of the substracted term, (6.1) $\beta$.
(4) Line34 + " interactions can be sequential..."  Unclear, especial with the use of "can" Does it may the authors were not sure of their use here
(5)cLine 42+ The use of "her" is not clear. Whose gender is female here?

### Questions
1. As mentioned in the related work section, some existing works have studied the considered problem and the similar problem where the follower plays the best response instead of a quantal response. Please discuss how the effectiveness of your algorithm compares with these existing results, especially regarding the regret bounds.
2. What is the technical motivation for the first term $W_1^{U,\theta}(s_1)$ in the loss function?
3. The results focus on a general function approximation approach, where restrictive assumptions are required. If the Markov Stackelberg Game has finite state and action spaces, could you derive a regret bound for your algorithm that only depends on $H,T,|S|,|A|,|B|$ and $||u||,||r||$?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper studies an online Stackelberg game from the leader's perspective. The leader interacts with a follower who behaves according to a quantal response model. The leader does not know her own utilities, nor does she know the follower's. However, the leader observes the interaction history and attemps to learn the follower's behavior model based on this observation. The aim is to learn an optimal policy to use against the follower. It is assumed that the follower is myopic. And the paper proposes to use a class of parameterized functions to approximate the follower's reward function, whereby the task reduces to estimating the parameter that minimizes the loss with respect to samples collected. Based on the estimation, a greedy planning algorithm is proposed and the authors analyzed the regret of this algorithm.

### Strengths
The paper uses some nice ideas, which are novel in the literature on learning in Stackelberg games, such as using function approximation for learning the follower's response model. The authors also made an effort to make the model more realistic by assuming, e.g., the leader doesn't know the follower's realized rewards. The paper also made an interesting connection with RLHF.

### Weaknesses
The key contribution is somewhat unclear and the overall results look incremental. Although the authors have put significant good effort into deriving the results, the outcomes look overall like a string of engineering work without strong key insights. It is therefore unclear how much implications the work offers to similar problems and to the literature in general. While the paper made some realistic assumptions, this is weakened by some others, e.g., the follower is myopic. The paper also seems to miss important related works in the literature on Stackelberg games, e.g.:

- Inverse Game Theory for Stackelberg Games: the Blessing of Bounded Rationality, Wu et al., 2022.
- Learning and Approximating the Optimal Strategy to Commit To, Letchford et al., 2009.
- Learning optimal strategies to commit to, Peng et al., 2019.
- Computing optimal strategy against quantal response in security games, Yang et al., 2012.

Typos and minor issues: 
- Abstract: First sentence looks confusing at first glance, as if the leader is using a quantal response policy. 
- Equation (3.3): I suppose $\pi^*$ should be $\pi^\star$.
- Line 219: "Let ... to be" --> "Let ... be".
- Assumption 1: double parentheses seem unintended.

### Questions
How far away are the proposed techniques from addressing the same problem but with a far-sighted follower? What are the main obstacles?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper considers designing a policy for the leader in a Stackelberg game where the followers' reward function is unknown, requiring online learning.

The authors consider a setting with boundedly rational followers, where boundedly rational is defined as containing a regularization term, which is constrained to be strongly convex. 

This paper introduces an algorithm called "Planning after estimation" (PES). PES follows two steps. For the first step, estimating the leaders' value function and the followers' quantal response model, the authors introduce a combined loss function that balances likelihood loss of the follower, Bellman loss of the leader, and an exploration term. For the second step, updating the leaders' policy, is done greedily. 

The authors show that their approach achieves a sublinear $\tilde{\mathcal{O}} (d_c \sqrt{T})$ regret (where $d_c$ is a decoupling coefficient), and theoretically instantiate this with reward functions from RL with human feedback, used to train LLMs.

### Strengths
1. Stackelberg games are widely used across a variety of settings, and as the paper discusses, are applicable to RL with human feedback used to train LLMs. It is easy to justify the motivation of this paper, as in many real-world setting we would not know the followers' reward function a priori. Thus, realistically the leader would have to plan with imperfect information of the followers' response --- and learning this policy needs to be done in a computationally tractable (sample efficient) way.

2. The authors formulate an approach for solving online Stackelberg games with bilevel optimization, first estimating the followers' response and the leaders' own value function, then greedily planning after these estimates. Their approach contrasts to Chen et al. [2023] which instead builds confidence sets over the followers' response function, which is more computationally complex as the constrained optimization problem requires solving coupled confidencee sets. 

3. The authors include analysis to show their approach, under certain assumptions, leads to a  small decoupling coefficient and achieves $\mathcal{O} (\sqrt{T})$.

### Weaknesses
It appears that this paper modifies the text size of the title (it looks vertically compressed), which is against submission policy.

There are no empirical results validating this approach; the "case study" just instantiates the reward functions. Given that one of the biggest benefits that the authors mention is that their method is more computationally tractable than Chen et al., this is a major weakness. 

It is therefore not possible to determine whether this approach works in practice, or is just a hypothetical framework. 



Specific comments

Writing could be improved in some points for easier reading.
- line 15: "posing" -> "which poses
- line 95: "in specific" -> "specifically"
- line 98: "omnisicent" -- as in rational and deterministic?
- paragraph 116-127: several things are capitalized ("Our" and "Multi") which should not be. 
- line 132: "the set of probability measure on X" awkward wording
- line 114: "denotes" -> denote
- line 151: "step any step" extra word

### Questions
Writing could be improved in some points for easier reading:
- line 15: "posing" -> "which poses
- line 95: "in specific" -> "specifically"
- line 98: "omnisicent" -- as in rational and deterministic?
- paragraph 116-127: several things are capitalized ("Our" and "Multi") which should not be. 
- line 132: "the set of probability measure on X" awkward wording
- line 114: "denotes" -> denote
- line 151: "step any step" extra wrod



Questions

1. Line 55: Why would the problem be ill-posed when the follower is fully rational? Shouldn't the problem be easier under perfect rationality? 

2. Line 63: You say that Chen et al.'s approach for optimistic planning over confidence intervals considers a problem that is highly intractable --- but then how did they solve it? If the problem is very limited, e.g., only scales to a small problem size, it would be helpful to specify, and specify in which way your work overcomes these barriers. 

3. Line 114: "Our algorithm is not only easy-to-implement but also easier to show theoretical guarantee" --> as in the math is easier? what does this mean?

4. Model: the transitions $P_h$ and reward function $u_h$ and $r_h$ suggest that the transitions and reward functions change at every timestep. Why is that the case? Shouldn't the transitions and reward functions be stationary? How does this evolution happen?

### Soundness
2

### Presentation
2

### Contribution
2
