# Combinatorial Reinforcement Learning with Preference Feedback

- Decision: Reject
- Scores: 5, 6, 6, 5

## Abstract
In this paper, we consider combinatorial reinforcement learning with preference feedback, where a learning agent sequentially offers an action—an assortment of multiple items—to a user, whose preference feedback follows a multinomial logit (MNL) model. This framework allows us to model real-world scenarios, particularly those involving long-term user engagement, such as in recommender systems and online advertising. However, this framework faces two main challenges: (1) the unknown value of each item, unlike traditional MNL bandits (which only account for single-step preference feedback), and (2) the difficulty of ensuring optimism with tractable assortment selection in the combinatorial action space. In this paper, we assume a contextual MNL preference model, where mean utilities are linear, and the value of each item is approximated using general function approximation. We propose an algorithm, MNL-V$Q$L, that addresses these challenges, making it both computationally and statistically efficient. As a special case, for linear MDPs (with the MNL preference model), we establish a regret lower bound and show that MNL-V$Q$L achieves near-optimal regret. To the best of our knowledge, this is the first work to provide statistical guarantees in combinatorial RL with preference feedback.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper considers combinatorial reinforcement learning with preference feedback, where a learning agent sequentially offers an action—an assortment of multiple items—to a user, whose preference feedback follows a multinomial logit (MNL) model. This framework can model real-world scenarios, particularly those involving long-term user engagement, such as in recommender systems and online advertising. However, this framework faces two main challenges: (1) the unknown value of each item, unlike traditional MNL bandits (which only account for single-step preference feedback), and (2) computational complexity due to the combinatorial action space. In this paper, the authors assume a contextual MNL preference model, where mean utilities are linear, and the value of each item is approximated using general function approximation. The authors propose an algorithm, MNL-VQL, that addresses these challenges, making it both computationally and statistically efficient. As a special case, for linear MDPs (with the MNL preference model), the authors establish a regret lower bound and show that MNL-VQL achieves near-optimal regret.

### Strengths
1.	The authors design a computationally efficient algorithm, called MNL-VQL, and provide a regret upper bound for the parameterized contextual MNL preference model and general function approximation for item values. This is the first theoretical regret guarantee in combinatorial RL with preference feedback. 
2.	For the special case of linear MDPs, the authors establish a regret lower bound, and algorithm MNL-VQL achieves a regret upper bound which nearly matches this lower bound.

### Weaknesses
1.	The writing of this paper needs improvement. This paper is too dense.
2.	There is no experiment provided. It is hard to validate the effectiveness and implementability of the proposed algorithm, and the provided theoretical results.
3.	The authors should highlight more on the computational challenge brought by the combinatorial action space, and novel techniques used to overcome this computational challenge, using concise words to convey the intuition behind, perhaps in the introduction.
4.	This work looks like a combination between combinatorial RL with preference feedback and existing RL works with function approximation. The authors should elaborate more on technical novelty, e.g., provide a paragraph to summarize it.

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
2

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
This paper studies the combinatorial RL with preference feedback, where the agent aims to select an assortment of items for users and get preference feedback to improve long-term user engagement in online recommender systems. To tackle the uncertainty in both the user choice model and the state transitions, the authors propose to use the contextual MNL preference model and RL with function approximation to model the problem. Then they propose a computationally and statistically efficient algorithm MNL-VQL, with rigorous theoretical guarantee.

### Strengths
1. This paper frames a new combinatorial RL problem with preference feedback. Based on the decomposition model in SlateQ (Ie et al, 2019), the authors explicitly use contextual MNL preference assumption to model the user’s single item choice after the agent recommends a combinatorial action (i.e., an assortment), which is the key structural property to maintain the traceability of the problem. For the state transition and reward modeling, the authors use RL with general function approximation to guarantee the generalizability of the model.
2. The authors naturally combine the MNL bandit leaning for the user’s MNL choice model and the model-free VOQL to learn the item-level Q-functions. To obtain optimistic choice probabilities, they propose a new method to judiciously assign UCB or LCB for each selected item in the assortment considering the relative value of outside option $a_0$.  Then they also propose a provably efficient method to compute the policy based on the optimistic assortment Q-values.
3. The authors also provide theoretical guarantees for the regret upper bounds as well as discussions for the tightness of the regret bounds considering several degenerate cases, such as contextual MNL bandit with $H=1$ and RL with linear function approximation

### Weaknesses
1. Although quite natural and well-motivated, the problem setting heavily relies on three previous works: the SlateQ (Ie et al, 2019) as the basic slate-based combinatorial RL model, the contextual MNL choice model for user behavior Oh & Iyengar (2019), and model-free RL with non-linear function approximation in Agarwal et al. 2023. The algorithm is also a combination of Lee & Oh (2024) for MNL model learning and VOQL in Lee & Oh (2024). The main contribution beyond the above combination is how to adjust the UCB/LCB for MNL model learning as well as an efficient combinatorial optimization subroutine. In general, the paper does not provide extremely novel algorithmic components/insights, but only essential parts to make the combination work. 
2. The theoretical results are not validated by any empirical results, which may not give readers enough evidence that structural assumptions will hold for real-world application scenarios and confidence that the proposed algorithm would work well in practice.

### Questions
1. Please comment on whether I am correct about the contribution and novelty of the paper. One interesting direction to improve the novelty of the paper is to discuss a new setting where more than one item can be adopted by the user, such as the DCM bandit choice model (Katariya et al., 2016), which is more interesting and practical when users may purchase several items recommended by the system. 
2. For the empirical evaluation, one valuable comparison, that I would like to see, is to use the similar simulation setting of Section 5 of SlateQ (Ie et al, 2019) to demonstrate the algorithm's performance, which is a simulated environment that mimics real-world recommender systems. For baseline algorithms, the authors could use SARSA and Q-learning with different combinatorial optimization methods (e.g., greedy, top-K as in Ie et al, 2019) as well as the holistic optimization that uses the non-decomposed Q-learning method that treats each assortment atomically as a single action. The authors are also encouraged to compare with other possible baselines as well as the real world data as in Section 6 in SlateQ (Ie et al, 2019).

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
2

### Summary
The paper tackles Combinatorial Reinforcement Learning (RL), which refers to as a class of RL problems where the action space is combinatorial, meaning that the agent selects a combination or subset of base actions from a set of possible base actions. The paper proposes a computationally efficient algorithm and provides theoretical regret guarantee.

### Strengths
* The paper tries to integrate the concept of multinomial logit user feedback into reinforcement learning, which could model the practical cases in applications like online retail and advertising, etc.

### Weaknesses
* In Line 48, "However, these studies take a myopic approach, optimizing for immediate rewards without considering the long-term impact on user behavior." -> I cannot agree with this. I believe that any approaches that balance exploration-exploitation take into account the long-term impact, including bandit algorithms.
* The regret definition seems strange to me. The definition in Line 192 seems like the regret is only measured on state $s_1$.
* The organization of the paper is not very easy to follow.

### Questions
* I am confused with the role of function class mentioned in Assumption 2. Personally I don't regard such formulation as standard in RL literature, rather, it seems more like a way to simplify the function approximation, in the sense that the problem is formulated more in a bandit setting side.
* Another confusion of mine is regarding the state space. It is not explicitly defined in the paper whether the state space is finite or not. If it's finite, then by estimating value function of each state action pair, I am more interested in the sample complexity of the proposed algorithm.

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
This paper studies combinatorial reinforcement learning with preference feedback problems in which a learner offers a set of actions (referred to as assortment) to a user, and he prefers one of the actions, where his preference follows a multinomial logit (MNL) model. The goal is to learn a policy for selecting an assortment from a combinatorial action space in each step of every episode that minimizes the total regret, i.e., the sum of the difference between the state value of the optimal policy for a given initial state and state value of the policy for given initial state.

The authors proposed an algorithm named MNL-VQL (MNL Preference Model with Variance-weighted Item-level Q-Learning) for this problem with a sub-linear regret guarantee, even achieving near-optimal regret bounds for linear Markov Decision Processes (MDPs).

### Strengths
**The following are the strengths of the paper:**
1. This paper extends the existing MNL bandit setting to the combinatorial RL setting. This problem is motivated by many real-world applications like online recommendation and advertising, where long-term user engagement is modeled using state transitions and preference feedback over shown assortments.

2. The authors propose the MNL-VQL algorithm for combinatorial reinforcement learning with preference feedback problems. They have shown that their algorithm enjoys sub-linear regret bounds guarantees, showing near-optimal performance for linear MDPs.

### Weaknesses
**The following are the weaknesses of the paper:**
1. It is unclear how the proposed algorithm is computationally efficient as it has to search all possible assortments (that can be exponentially large) to find the assortment to users (as defined in Eq. (9)). Is there a way to solve the optimization problem in Eq. (9) efficiently, avoiding the exponential cost of enumerating all assortments?

2. The proposed algorithm has many hyperparameters, and it is unclear how one can choose them in practice. Furthermore, without any experiment results (or ablations study), it is unclear how the proposed algorithm will behave in practice. Are there any guidelines for selecting the key hyperparameters, or is it possible to have a small experiment demonstrating the algorithm's sensitivity to different choices of hyperparameters?

3. This paper considers time-inhomogeneous MDPs, but it is unclear how the non-stationarity (in transition kernel and reward function) across episodes is dealt with. Can you point where in algorithm we can see explicit treatment of these time-varying dynamics?

4. The rewards can not be sparse (Line 249) because rewards in each step are needed to get estimates of the item-level Q-functions (Line 281 and Line 287), also needed for estimating the second moment (Line 294).

### Questions
Please address the weaknesses of the paper. I have a few more questions/comments:
1. How does one deal with the problem of changing horizon length (e.g., horizon length cannot be fixed in real-life applications like online recommendations and can be different for different users)?

2. What is the value of generalized Eluder dimension for different function classes, e.g., linear, neural network, and kernel regression with different kernels? Is it always finite, irrespective of function class?

3. Why is the outside option $a_0$ considered when selecting the assortment, as it is already part of every assortment? Without this, pessimistic utility estimates (Eq. (7)) are not needed. Right?

**Minor comment:**
1. It is clear enough how the pessimistic estimates ($j = -2$) of item-level Q values are used in Eq. (10). My understanding is that it is part of the **otherwise** case.

2. I find it hard to follow the paper from page 5 to page 9 (except page 6). The authors can make this part of the paper more readable by clearly connecting different parts of the algorithm.

I am open to changing my score based on the authors' responses.

### Soundness
3

### Presentation
2

### Contribution
3
