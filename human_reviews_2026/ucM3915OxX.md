# Nearly-Optimal Bandit Learning in Stackelberg Games with Side Information

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
We study the problem of online learning in Stackelberg games with side information between a leader and a sequence of followers. In every round the leader observes contextual information and commits to a mixed strategy, after which the follower best-responds. We provide learning algorithms for the leader which achieve O(T^{1/2}) regret under bandit feedback, an improvement from the previously best-known rates of O(T^{2/3}). Our algorithms rely on a reduction to linear contextual bandits in the utility space: In each round, a linear contextual bandit algorithm recommends a utility vector, which our algorithm inverts to determine the leader's mixed strategy. We extend our algorithms to the setting in which the leader's utility function is unknown, and also apply it to the problems of bidding in second-price auctions with side information and online Bayesian persuasion with public and private states. Finally, we observe that our algorithms empirically outperform previous results on numerical simulations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies online learning for contextual Stackelberg games under bandit feedback. This paper uses a reduction that lets the leader act in a utility space and then plug in linear contextual bandits algorithms. Instantiating the reduction with OFUL gives $\tilde{O}(\sqrt{T})$ regret when contexts are adversarial and follower types are stochastic; a logdet-FTRL instantiation gives $\tilde{O}\left(K^{2.5}\sqrt{T}\right)$ regret when contexts are stochastic and follower types are adversarial. The paper also extends to unknown leader utilities  with $\tilde{O}\bigl(\mathrm{poly}(d,K,A_\ell,A_f)\sqrt{T}\bigr)$ regret.

### Strengths
A clean reduction that operates in the leader’s utility space, enabling off-the-shelf contextual bandit algorithms to close the bandit-feedback gap to $\tilde{O}(\sqrt{T})$, with extensions to unknown utilities and applications such as auctions and Bayesian persuasion.

### Weaknesses
1. The paper restricts the leader’s strategy space to the set of approximate extreme points of contextual best-response regions. It would be more interesting to allow approximate follower responses, i.e., followers who play $\delta$-suboptimal best responses, and analyze how such deviations affect the reduction and regret guarantees.
2.  The main results rely heavily on prior techniques—off-the-shelf linear contextual bandit algorithms, the Contextual Follower Best-Response Region framework, and $\delta$-approximate extreme points—so the paper’s technical novelty is unclear; overall, the contribution appears incremental, primarily composing existing components.

### Questions
No specific questions

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
This paper studies contextual Stackelberg games with multiple follower types under bandit feedback. The authors propose a reduction to linear contextual bandits in the leader’s utility space, enabling near-optimal learning algorithms for two settings: (i) adversarial contexts with stochastic followers, and (ii) stochastic contexts with adversarial followers. In both cases, the algorithm achieves $\tilde{\mathcal{O}}(T^{1/2})$ regret, improving upon the previous best-known $\tilde{\mathcal{O}}(T^{2/3})$ rate. The framework also extends to second-price auctions and Bayesian persuasion with side information through discretized action spaces.

### Strengths
- This work closes the gap between lower and upper regret bounds for bandit-feedback contextual Stackelberg games, achieving the optimal rate up to logarithmic factors.

- The reduction to linear contextual bandits in utility space is conceptually clean and broadly applicable as shown by examples.

- Extension to unknown utilities: Maintains near-optimal regret under a linearity assumption, showing robustness of the reduction.

- The paper is well structured and the main theorems and algorithmic structure are well organized and technically sound.

### Weaknesses
- The per-round time of the proposed algorithm is exponential which limits the practicability of the algorithm, which is also evident from the small toy example of Sec. 3.5. 

- The authors assume a finite and known set of follower types and full knowledge of their utility functions (except in the linear “unknown utility” variant). 

- The discretization of the leader’s strategy space $\mathcal{E}_t$ is important to keep the action space finite but introduces an approximation error on the regret that is only bounded asymptotically.

### Questions
1) In limitations, it is mentioned that a possible path forward towards reducing the runtime by means of smoothed analysis. Could the authors comment on the conceptual idea and how promising might be?

2) How restrictive is the leader assumptions on the followers, i.e., cardinality, types, and utility, in practice? Do the authors see any path forward at relaxing these assumptions (cardinality and types)?

3) Is it possible to say antyhing about the $\mathcal{O}(1)$ approximation error in the regret due to the discretization?

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
3

### Summary
This paper treats the problem of online learning in Stackelberg games between a learner and a follower whose type changes over time. Both the leader and the follower have access to a $d$-dimensional side / contextual information variable $z_t$, and their payoffs are determined as follows:
- For the leader, as a function $u(z_t,a_t,b_t)$ of the context $z_t$ at time $t$ and the actions $a_t$ and $b_t$ of the leader and follower respectively.
- For the follower, as a time-varying function $u_t(z_t,a_t,b_t)$ of the context $z_t$ at time $t$ and the actions $a_t$ and $b_t$ of the leader and follower respectively. [The variability of $u_t$ over time is what the authors dub as the leader facing "a sequence of followers"]

The authors assume that the context $z_t$ or the follower's payoff function $u_t$ can be determined adversarially—but not both, as in that case, it is not possible to attain sublinear (Stackelberg) regret. In terms of information available to the players, they assume that the ensemble of the follower's payoff functions is known to the leader (though not the identity of each payoff function at each instance), the leader's payoff function could be either known or unknown to the follower, and the follower observes the leader's mixed strategy at each instance (so they can best-respond to it).

The authors provide a reduction to linear contextual bandits and, depending on whether the sequence of followers or contexts is adversarial, they provide the following guarantees:
1. *For adversarial contexts and stochastic followers:* an $\mathcal{O}(K\sqrt{T} \log T)$ regret bound when the recommendation subroutine in Algorithm 1 is the OFUL algorithm of Abbasi-Yadkori et al (2011).
2. *For adversarial follower sequences and stochastic contexts:* an $\mathcal{O}(K^2.5 \sqrt{T} \log T)$ when the regret-minimizing subroutine in Algorithm 1 is instantiated to the logdet-FTRL algorithm of Liu et al (2024).
The authors also provide a version of these theorems for the case where the leader's utility is unknown. In that case, the above bounds continue to hold in expectation with $K$ replaced by $d A_l A_f$, where $A_l$ is the number of actions available to the leader, and $A_f$ to the follower.

Finally, the authors describe how their results could be applied to the context of bidding in auctions with side information (Section 4.1) and Bayesian persuasion (Section 4.2).

Overall, I found the paper's topic to be on the fringe of ICLR topics—too specialized on a setting that is not central to generalist ML/AI audiences. My "weak accept" recommendation reflects my cautiously optimistic belief that the paper *could* be accepted if it garners sufficient support from experts in the domain. [I am knowledgeable in the general field but not an expert; still, I am skeptical regarding the difficulty of the reduction to contextual bandits or the novelty behind the meta-use of the .recommend subroutine in Algorithm 1.]

Otherwise, if the paper's contribution is not championed by an expert in the field, I believe this paper would be much better suited to a more specialized venue like EC. I will wait for the authors' rebuttal and the discussion with the rest of the review panel to converge.

### Strengths
The paper is well-written and, as far as I could tell, the analysis is sound and the results correct. I am not a dedicated expert in this literature but, as far as I can tell, the authors' contributions are positioned properly within the relevant literature.

### Weaknesses
My main objection to the paper is the paper's learning setting: I understand that the paper's framework may be sometimes referred to as "bandit learning" in the context of Stackelberg games but, otherwise, the assumptions made in the paper are fairly stringent and exacting. In particular, as was mentioned above, the authors assume that the ensemble of the follower's payoff functions is known to the leader, and the follower observes the leader's mixed strategy at each instance (so they can best-respond to it). Both assumptions go against the spirit of the partial information setting in bandits where it is assumed that players only observe their payoffs / losses—and even if this term is used in a part of the Stackelberg literature, it is a misnomer, which should not be propagated in a generalist venue like ICLR (or any of the other generalist learning venues, like NeurIPS, COLT, or ICML).

Other than that, as I said above, I found the paper's topic to be on the fringe of ICLR topics—too specialized on a setting that is not central to generalist ML/AI audiences.

If the paper's contributions are deemed sufficiently impactful by an expert in the field (I am not an expert on Stackelberg games and/or contextual bandits), I wouldn't find either of the above weakenesses as an obstacle to acceptance. Come what may however, I would insist on the authors' changing the "bandit learning" terminology to avoid clashing with established terminology in online / game-theoretic learning.

### Questions
None; see above.

### Soundness
3

### Presentation
3

### Contribution
2
