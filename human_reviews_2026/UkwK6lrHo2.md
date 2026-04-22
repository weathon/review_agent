# Simultaneously Perturbed Optimistic Gradient Methods for Payoff-Based Learning in Games

- Avg Score: 4.80
- Decision: Reject
- Scores: 2, 6, 6, 4, 6

## Abstract
We examine the long-run behavior of learning in a repeated game where the agents operate in a low-information environment, only observing their realized payoffs at each stage. We study this problem in the context of monotone games with unconstrained action spaces, where standard gradient schemes may lead to cycles, even with perfect gradient information.  To account for the fact that only a \emph{single} payoff observation can be made at each iteration—and no gradient information is directly observable—we design and deploy a simultaneous perturbation gradient estimation method adapted to the challenges to the problem at hand, namely unbounded action spaces, gradients and rewards. In contrast to single-timescale approaches, we find that a two-timescale approach is much more effective at controlling the (unbounded) noise introduced by payoff-based gradient estimators in this setting. Owing to the introduction of a second timescale, we show that the proposed simultaneously perturbed optimistic (SPOG) algorithm converges to equilibrium with probability 1. In addition, by developing a new method to assess the rate of convergence of two-timescales stochastic approximation procedures, we show the sequence of play induced by SPOG converges at an asymptotic $\tilde{\mathcal{O}}(n^{-2/3})$ rate in strongly monotone games. To the the best of our knowledge, this is the first convergence rate result for games with unbounded action spaces, and it is faster than the sharpest known convergence rates for single-observation, payoff-based learning in strongly monotone games with bounded action spaces.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a new algorithm, SPOG, designed to solve strongly monotone games under bandit (zeroth-order) feedback. The algorithm achieves a convergence rate of $\tilde{O}(n^{-2/3})$, improving upon the rate attained in previous work.

### Strengths
- SPOG obtains a faster convergence rate compared to the previous method (Table 1).
- The empirical evidence shows that SPOG is faster than the baselines.

### Weaknesses
The big-O notation in the main theorem (Theorem 4.2) conceals factors that are **exponentially large**. Lemma 3.1 holds only for sufficiently large $n$, and an examination of its proof shows that this threshold $n$ is itself exponentially large. Specifically, in line 709, it requires $\Theta(\frac{n^{a-d}}{\log n})=\frac{\alpha_{n-1} R_n}{\delta_n} \leq \min( \frac{1}{4M\sqrt{N}}, \frac{1}{2\gamma M} )$, while the optimal convergence rate is obtained when $a=d$ (Theorem 4.2). 
Consequently, this exponential dependence propagates into the convergence rate stated in Theorem 4.2, which relies on Lemma 3.1.

Furthermore, I recommend that the authors **explicitly specify** the required $n$ in Lemma 3.1 rather than hiding it within the asymptotic notation. Since the main claim of the paper concerns achieving a faster rate, even the constants involved are significant and should be made transparent.

### Questions
Given the exponentially large factor implicated in Theorem 4.2, I encourage the authors to evaluate SPOG on a broader suite of (larger) games and to include additional baselines. Without such expanded empirical evidence, the current experiments are insufficient to substantiate the claimed efficiency of SPOG.

### Soundness
2

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
This paper studies payoff-based learning in monotone games. In each round, each player observes only zeroth-order (bandit) feedback of their own payoff and cannot observe others. The action space is assumed to be unconstrained (unbounded). The authors propose a novel algorithm, SPOG. It has two main components: (i) an adjusted SPSA gradient estimator that controls variance by using the previous payoff as a baseline, and (ii) a simultaneously perturbed optimistic-gradient update that adapts the OG+ style separation of learning rates to the payoff-based setting where gradients must be estimated. For monotone games, the iterates converge asymptotically to a Nash equilibrium; for strongly monotone games, the authors prove a last-iterate convergence rate of $n^{-2/3}$.

### Strengths
- A new last-iterate convergence-rate result for payoff-based learning in monotone games.
- The combination of learning-rate separation with an adjusted gradient estimator is technically interesting and the associated analysis is nontrivial (at least for me).

### Weaknesses
- Regarding rates in payoff-based learning, additional related work and comparisons to known lower bounds are needed. For example:  
  Fiegel, Côme, et al., "The Harder Path: Last Iterate Convergence for Uncoupled Learning in Zero-Sum Games with Bandit Feedback," ICML.  
  Cai, Yang, et al., "Uncoupled and Convergent Learning in Two-Player Zero-Sum Markov Games with Bandit Feedback," NeurIPS 2023.  
  Fiegel et al. appear to show an $n^{-1/4}$ lower bound.

- Can the results be extended to the constrained setting? If not directly, what are the main obstacles?

- The input $\gamma$ appears to require knowledge of game-dependent constants $L, G$ (and possibly $N$). Discussion of robustness or adaptive tuning would help.

- Can you obtain a rate for the merely monotone case ($\mu=0$)? One possibility is to add a player-wise regularizer to make the game strongly monotone and then anneal the regularization strength with $n$; can this yield a rate?

- The experimental section focuses on very simple cases. For the two-player zero-sum bilinear game, comparisons with other payoff-based methods (e.g., Fiegel et al.) would strengthen the empirical evidence.

### Questions
1. Do you believe the $n^{-2/3}$ rate is tight? Is it possible to provide a lower bound?
2. How does your setting differ from the bandit convex optimization (BCO) model of Shamir (2013), where $\Omega(n^{-1/2})$ lower bounds (for any algorithm) are known? Clarifying the modeling/algorithmic differences and the lower-bound constructions would help reconcile how one might "break" the $n^{-1/2}$ barrier. Does your approach imply fast rates in BCO as well, or do crucial assumptions differ?
3. Can your method be extended to two-point zeroth-order or stochastic first-order settings?
4. How does the rate depend on the strong monotonicity parameter $\mu>0$? From the proof of Proposition 4.4, it superficially looks like the argument might go through with $\mu=0$; where exactly do you use $\mu>0$?
5. In the experiments, can you compare against other payoff-based methods for two-player zero-sum bilinear games?

Minor: Line 105: What is $\mathcal{V}$?

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
This paper proposes Simultaneously Perturbed Optimistic Gradient (SPOG), a single-observation  learning algorithm for repeated games with unbounded  action spaces and  monotonicity. It extend Simultaneous Perturbation Stochastic Approximation (SPSA) to policy optimization by introducing structured random perturbations that reduce variance while maintaining unbiasedness. Theoretical analysis shows unbiasedness and convergence under standard Lipschitz and bounded-noise assumptions. Empirically, SPPO is evaluated on continuous-control benchmarks  and a large-scale humanoid locomotion task.  It achieves comparable or superior performance to baselines with fewer samples and reduced variance.

### Strengths
- The paper provides the first last-iterate convergence and explicit rate for unconstrained action spaces. Handling unbounded domains is technically nontrivial and important for continuous games.
- The two-timescale analysis and the way SPSA+ interacts with learning-rate separation to control unbounded variance are novel.
- Many prior results assume compactness. The paper removes this assumption, which is a significant step.

### Weaknesses
- The claimed improvement beyond $\Omega(n^{-1/2})$ lower bounds for one-point ZO methods should be discussed more carefully. Since SPSA+ reuses a prior payoff observation, the effective oracle information per iteration differs from the standard one-point model used in classical lower bounds. 
-  The paper lacks finite-sample complexity bounds  compared to ES or policy-gradient methods.
- Convergence and the $n^{-2/3}$ rate require delicate  step-size relations in Assumptions 2/3.

### Questions
- Put the core parameter constraints  in a compact table or boxed recommendation to help practitioners.
- Can the theoretical convergence be strengthened to finite-sample rates or expected suboptimality bounds?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies payoff-based learning in monotone games with unconstrained continuous action spaces, where agents only observe their own realized payoffs and must estimate gradients from a single function evaluation per round. The authors design Simultaneously Perturbed Optimistic Gradient (SPOG), which combines (i) optimistic gradient dynamics with learning-rate separation and (ii) a single-observation SPSA+ zeroth-order gradient estimator that reuses a previous payoff as a baseline to reduce variance, together with projections onto slowly expanding balls to keep iterates controlled in the unbounded domain. Under standard smoothness and monotonicity assumptions, they prove that SPOG’s slow iterate converges almost surely to a Nash equilibrium in all monotone games. They further show a last-iterate rate $\mathcal{O}(n^{-2/3})$ in strongly monotone games. This rate is claimed to be the first convergence rate result for payoff-based learning in unbounded monotone games, and to improve on the best known rates for one-point zeroth-order methods in bounded games by effectively turning a one-point estimator into a “two-point for the price of one” via the baseline trick.

### Strengths
* The proposed algorithm has the rate result for unconstrained monotone games and strictly faster than the $\mathcal{O}(n^{−1/2})$ rates known for one-point ZO methods.
* Beyond qualitative convergence, the paper develops a rate analysis for two-timescale stochastic approximation.

### Weaknesses
* The convergence proofs crucially rely on projecting both fast and slow iterates onto balls of radii $R_n$ and $3R_n$ growing like $\log n$. This is mathematically convenient but changes the dynamics relative to the original unconstrained game; the paper does not really discuss whether or how this affects behavior in realistic problems.
* The algorithm requires carefully tuned sequences $\alpha, \beta, \delta, R$ with nontrivial exponent constraints (e.g., 0<d\leq a<b<1, a+b>1) and coupled summability conditions, plus an upper bound on $$\gamma. This makes the method hard to instantiate in practice, and there is little guidance on how robust convergence is to mis-tuning.
* The empirical section consists of a single 1D bilinear zero-sum game, comparing SPOG to OG/OG+ on the distance to equilibrium. It is too minimal to validate the algorithm’s behavior in higher dimensions, with more players, or under different noise levels.

### Questions
* Is the projection into slowly expanding balls essentially a proof artifact, or do you believe it is algorithmically necessary in practice?
* How sensitive is SPOG’s behavior to mis-specified exponents and constants in Assumptions 2–3? It will be more convincing to have empirical evidence or theoretical intuition on choosing these parameters adaptively.
* For merely monotone games you prove a.s. convergence but no rate. Do you expect a quantitative rate to be achievable under additional structural assumptions?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This study develops a learning algorithm that achieves last-iterate convergence in unconstrained monotone games under zeroth-order feedback. The proposed method consists of three key components: an adjusted SPSA gradient estimator, a learning-rate separation, and a two-timescale approach. The authors prove that the proposed method converges to an equilibrium with probability 1 in general monotone games. Moreover, a faster last-iterate convergence rate, $\tilde{\mathcal{O}}(n^{-2/3})$, is derived in strongly monotone games.

### Strengths
To the best of my knowledge, the derived convergence rate of $\tilde{\mathcal{O}}(n^{-2/3})$ is the fastest known for strongly monotone games under payoff-based feedback. This significant improvement is expected to lead to faster last-iterate convergence rates in merely monotone games, although this study only establishes an asymptotic result in such games.

Moreover, in the experiments, I am surprised that the proposed method’s trajectories—though noisy—converge faster than baseline methods even in a non-strongly monotone game.

### Weaknesses
While I understand that this paper primarily focuses on theoretical analysis, I am curious about the empirical performance of the proposed method in larger games, particularly in high-dimensional settings. Furthermore, providing empirical results on strongly monotone games would be preferable, as the improvement in the last-iterate convergence rate in these games is a significant contribution of the paper.

### Questions
My main concerns and questions are stated in the Weaknesses section. Additionally, I have the following questions and comments:

- (Introduction) The publication year is missing in “(Daskalakis et al.)”.
- (Introduction) Does “even optimistic gradient (OG) methods, which incorporate a recency bias, have been shown to exhibit trajectories of play that orbit an equilibrium, failing to converge” refer to results under noisy feedback?
- (Discussion) The authors mention that “this exceeds the optimal lower complexity bound of $\Omega(n^{-1/2})$ for one-point zeroth-order algorithms.” I am curious why this inconsistency exists. Is the proposed method outside the class of one-point zeroth-order algorithms considered by Shamir (2013) and Ba et al. (2025)?

### Soundness
3

### Presentation
3

### Contribution
3
