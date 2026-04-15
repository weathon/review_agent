# No Algorithmic Collusion in Two-Player Blindfolded Games with Thompson Sampling

- Decision: Reject
- Scores: 5, 8, 6, 5

## Abstract
When two players are engaged in a repeated game with unknown payoff matrices, they may be completely unaware of the existence of each other and use multi-armed bandit algorithms to choose the actions, which is referred to as the ``blindfolded game'' in this paper. We show that when the players use Thompson sampling, the game dynamics converges to the Nash equilibrium under a mild assumption on the payoff matrices. Therefore, algorithmic collusion doesn't arise in this case despite the fact that the players do not intentionally deploy competitive strategies. To prove the convergence result, we find that the framework developed in stochastic approximation doesn't apply, because of the sporadic and infrequent updates of the inferior actions and the lack of Lipschitz continuity. We develop a novel sample-path-wise approach to show the convergence.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents a negative result in the area of repeated games and learning.  They study a scenario to two players, each having two actions.  The players are "blindfolded" in that they only observe past actions and payoffs of themselves.  The main result shows that if the players both use Thompson sampling, then then two players convers to the unique Nash equilibrium under mild conditions, i.e., no collusion arises.  Proving this result requires new technical ideas beyond the typical approaches in the literature due to the lack of simultaneous updates and the lack of global Lipshitzness.

### Strengths
Learning in games is a thriving area and understanding when agents converge or not to different forms of equilibria is timely and interesting.  In particular, quantifying what factors lead to collusion (or the lack of it) is a central question in the literature. 

The paper is clearly written and the authors do a solid job of placing their results in the literature. 

The inclusion of numerical experiments in what is primarily and theoretical paper is appreciated.

### Weaknesses
The paper considers a very narrow setting - two-players, two-actions with both using a very specific algorithm.  The contribution of the paper would be larger if if could state a broader class of models/algorithms also maintained "no collusion".  As written, the result feels more like a curiosity worth investigating further than a major contribution, i.e., it shows a property of an example without exploring the limits of where the property holds.   Example are, of course important, but the contribution would be larger if the setting could be expanded to probe the limits by considering, e.g., more players, more actions, other learning rules, etc.

A particular limitation of interest to generalize is the specific assumption of Thompson sampling as the algorithm.  Studying a broader class of algorithms would increase the level of contribution significantly.

Table 1 highlights some challenges as compared to "classic" approaches in the literature.  However, there are a wide variety of modern approaches proposed in the past few years as this literature has exploded.  The paper does not highlight the challenges with applying these modern analyses (many of which consider more general settings) to the current results in the paper.  As starting point would be to highlight the challenges associated with applying the techniques in the papers mentioned in the last paragraph of Section 1.  See the question below.

### Questions
How much more generality in the model could your proposed techniques handle?  What generalizations to the model will require new techniques? What more general class of algorithms could your techniques apply to?

What are the technical contributions in your analysis relative to more recent approaches than those summarized in Table 1.  For example, those mentioned in the last paragraph of Section 1.

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
This work studies the dynamics of a repeated game where the players play using Thompson sampling based on the realized payoffs of their actions. This is referred to as a "blindfolded" game since the players do not get to see the impact of the other players' actions, or even model the other players, and play purely based on the payoffs of the actions (which are influenced by the actions chosen by the other players). Thompson sampling is an algorithm with a no-regret guarantee in multi-armed bandits problems with a stationary distribution for each arm, which is not necessarily the case in the scenario of blindfolded games, due to the adaptive and non-stationary behavior of other players. The work establishes that in two-player, two-action games satisfying some conditions including the existence of a unique pure strategy Nash Equilibrium (PSNE), the dynamic of play converge in the last iterate to this PSNE. The work is situated in the context of previous papers studying algorithmic collusion in repeated pricing games where price setters use reinforcement learning to set prices and end up converging to collusive, non-equilibrium strategy profiles. In particular, this result is framed as a counterpoint to theoretical analysis of Hansen et. al. showing UCB algorithms (another class of no-regret algorithms for multi-armed bandits) can converge to collusive strategy profiles.

### Strengths
The main strength of this paper is in studying the intricate dynamics of agents employing Thompson sampling against each other.  Unlike no-regret algorithms such as hedge/ multiplicative weights, Thompson sampling is a mis-specified algorithms in the context of games, and there is a notable lack of tools to analyze the dynamics they induce. This result is a non-trivial first contribution towards understanding these dynamics. The paper contains some interesting technical tools, including setting up a natural state space to track the dynamics and uses a sample-path-wise approach, building upon tools from Tsitsiklis 1994 to prove the result. Even though the result only holds for under some fairly restrictive conditions, it sets up some interesting open problems for future research to pursue.

### Weaknesses
The main weakness of the result is the set of restrictive conditions on the class of games, such as only two actions, a unique pure strategy Nash Equilibrium and assumptions about the payoff of the Nash Equilibrium not being much worse than the other action profiles. In particular, it is not clear if these even includes the simple pricing games that related work, such as Hansen et. al., Calvano et. al. study. These pricing games offer a compelling reason to study the emergence of collusion in the dynamics. Additionally, it is clear that results such as the folk theorem show the existence of algorithmic strategies (however artificial) that induce collusion in these pricing games, making it interesting to study if particular algorithmic strategies result in dynamics leading to collusion. It would help if the work could explain why these games may be interesting and discuss if collusion is possible under different algorithmic strategies employed by the players. The other weakness is the lack of discussion about how different parts of the proof slot together and exactly why Thompson sampling, as opposed to UCB/ other exploration-exploitation algorithms, provably converges to the Nash.

### Questions
Why is this class of games interesting? How does it compare the pricing games studied by related work? Is collusion possible at all using different algorithms for the players? Are any of the conditions absolutely necessary? Are there counterexamples showing the lack of one of these conditions breaks the result - one candidate that stands out in particular is the assumption about a unique PSNE.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors study repeated play in two-player, two-action games that admit a unique pure Nash equilibrium. They assume a minimal information setting. In particular, across all rounds, players only observe their own payoffs. In addition, the player's payoffs are subjected to zero-mean normally distributed noise.

Their main result, Theorem 1, states that under Assumptions 1 and 2, if both players assume a Bayesian point of view and update some particular prior distributions based on Thompson sampling, then almost surely they will converge to the unique Nash equilibrium of the game. The authors' ultimate claim is that under these assumptions algorithmic collusion is almost surely not possible.

### Strengths
1. The information setting studied by the authors is quite realistic.
2. The analysis, to the best of my knowledge, is correct.

### Weaknesses
1. Assumption 1 is indeed mild. However, Assumption 2 seems to be quite strong for the study of algorithmic collusion. In particular, under Assumption 2, the game's unique Nash equilibrium cannot be, as the authors point out, much worse than any other outcome. Therefore, independently of the algorithm used, gains from an algorithmic collusion can be minimal at best. Would the authors provide some further analysis of the implications of this assumption towards this particular claim?
2. A possible weakness is also the implicit assumption that the game's unique Nash equilibrium is pure. The uniqueness of the game's Nash equilibrium is a reasonable assumption for the study of the algorithm's convergence. However, if the game doesn't have a special structure, e.g., being a potential game, this equilibrium might not be pure. Could the authors further motivate this assumption?

### Questions
Kindly refer to the weaknesses.

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
4

### Summary
This paper mainly focuses on dynamics in two-player blindfolded games, where both players follow Thompson sampling to choose their actions.
Under some assumptions, the author demonstrates that the game dynamics converge to the pure Nash equilibrium.
The proof utilizes the fact that the game dynamics can be viewed as a special form of stochastic approximation.
Furthermore, the author also experimentally shows that each player's strategy converges to a pure Nash equilibrium in some games.

### Strengths
* The problem is well-motivated. Analyzing the celebrated MAB algorithm in games is of great importance to the research community.
* It seems novel to provide last-iterate convergence guarantees for Thompson sampling in online learning in games.

### Weaknesses
My primary concerns are centered on the assumptions on two-player games.
Firstly, the last-iterate convergence results are only provided for **two-action** games.
The analysis heavily depends on the fact that, in the two-action games, the action probability can be formulated in a closed-form expression (as in Eq. (7)).
However, I am curious if similar expressions can be derived for more general games.
Secondly, the assumption regarding the existence and uniqueness of the pure Nash equilibrium appears to be strong.
I am uncertain about the practical applications of games with these assumptions.
If these assumptions limit the practical application of this study, it would be advantageous to provide any convergence or divergent results in games that do not adhere to these assumptions, such as two-player zero-sum games.

Moreover, a recent study [1] primarily provided last-iterate convergence rates in two-player zero-sum normal-form games and Markov games under bandit feedback.
I believe the bandit feedback setting closely resembles the blindfolded setting.
Hence, the relationship with [1] should be clarified.

[1] Cai, Y., Luo, H., Wei, C.-Y., and Zheng, W. Uncoupled and convergent learning in two-player zero-sum Markov games. NeurIPS, 2023.

### Questions
My main concerns and questions are outlined in Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
