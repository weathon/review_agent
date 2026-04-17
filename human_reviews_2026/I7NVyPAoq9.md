# Thompson Sampling Algorithm for Stochastic Games

- Decision: Reject
- Scores: 8, 6, 8, 4, 4

## Abstract
We study a stochastic differential game with $N$ competitive players in a linear-quadratic framework with ergodic cost, where $d$-dimensional diffusion processes govern the state dynamics with an unknown common drift (matrix). Assuming a Gaussian prior on the drift, we use filtering techniques to update its posterior estimates. Based on these estimates, we propose a Thompson-sampling-based algorithm with dynamic episode lengths to approximate strategies. We show that the Bayesian regret for each player has an error bound of order $O(\sqrt{T\log(T)})$, where $T$ is the time-horizon, independent of the number of players. This implies that average regret per unit time goes to zero. Finally, we prove that the algorithm results in a Nash equilibrium.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper extends Thompson Sampling (TS) to stochastic differential games in a linear-quadratic framework with ergodic cost. Bayesian regret of each player is proved to be of order $O(T\log T)$ which matches the best-known rates for single-player LQ control (up to logarithmic factors) and Nash equilibria can be found. Technically, the most challenging parts in this paper are two-folds: (1) each player has only access to their own states without overly idealized knowledge of the whole system; (2) the bound is independent of the number of players. The contribution of this paper is solid, the proofs are technically sound and the presentation is clear.

### Strengths
(1) This paper considers the partially obseravle setting, i.e., each agent only has access to its own states without knowledge of the whole system.

(2) The bound is independent of the number of players.

(3) The proofs are mathematically sound and well organized. The authors clearly justify their assumptions and discuss their appropriateness.

### Weaknesses
(1) It would be better to have more ablations on hyper-parameters in experiement part.

(2) It would be better to add experimental results on empirical convergence rate to Nash equilibria.

### Questions
(1) Is your regret bound tight in terms of dimension $d$ or other parameters?

(2) Is it possible to show last-iterate convergence rate to Nash equilibria?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies an $N$-player nonzero-sum stochastic differential game with linear dynamics driven by a $d$-dimensional Brownian motion. Unlike most of the SDG literature, each player observes only their own state with a common unknown drift parameter. Thus uncertainty arises from both the state dynamics and the model parameters. Building on Bardi & Priuli (2014), they embed affine state-feedback strategies in a continuous-time TS policy for the LQ SDG. With an episode stopping rule, they prove a Bayesian regret bound of order $O\big(d||\sigma^i\||^3\sqrt{T\log T}\big)$, matching known TS rates in LQ control in terms of $T$. They also show the policy constitutes a Nash equilibrium. The experiments align with the theory.

### Strengths
* Proposes a continuous-time TS policy with adaptive episode lengths for an ergodic LQ SDG under partial information, where all players share an unknown drift parameter $A$ and learn it solely from their own state processes. The paper proves a Bayesian regret bound of order $O(\sqrt{T\log T})$ and establishes a corresponding equilibrium result.  
* Experimental results support the theory, further demonstrating performance in higher-dimensional settings and robustness to prior misspecification.

### Weaknesses
* The assumption of a common unknown prior is somewhat restrictive. However, given the first focus on a partial-information SDG with both equilibrium and sublinear regret guarantees, it may be acceptable within the current problem setup.
* Section 2 is difficult to follow with the current explanation (especially for readers familiar with TS but not with SDGs like me). Providing more intuition about the parameters and their roles would improve accessibility.  
* Some notation appears before being defined or is left undefined (see Question 1). Clarifying notation order and completeness would enhance readability.

### Questions
1. In Section 2, the authors use a norm notation $|| \cdot ||$. In Bardi & Priuli (2014) this notation denotes the spectral norm, i.e., the largest eigenvalue in the positive-semidefinite case. However, in Section 3 the same symbol is defined as the Frobenius norm. Please clarify whether the norm in Section 2 (especially in Assumption A4) refers to the Frobenius or the spectral norm.  
   1.1. The notation $[\alpha^{-i}, \alpha]$ is defined in line 210 (p. 4), but it first appears on page 3. The definition should appear before its first use.  
   1.2. In Proposition 3.2, the symbol $:=^{i}$ is used without definition. It likely means $:=E^{i}$; please make this explicit.  

2. From Proposition 3.3, the obtained regret seems to scale as $O\big(\min(\|\sigma^{i}\|^{3}\sqrt{d},d)\sqrt{T\log T}\big)$, which is smaller than the form stated in Theorem 3.1. Can you clarify why Theorem 3.1 presents the looser bound? was it for simplicity, or are there technical reasons for preferring that form?  

3. What is the main difficulty or bottleneck when extending the analysis to heterogeneous parameters $A^i$? I would (naively) expect the resulting regret to include $N$-dependent constants in such cases. The authors’ perspective on this point would be valuable.

### Comments

1. L75: the authors appear to cite Abeille & Lazaric (2018) as a paper studying the *Bayesian* regret of TS. However, this reference actually analyzes the *frequentist* regret of TS in linear–quadratic (LQ) control settings (as far as I understand). 

2. There are duplicated references (e.g., Abeille & Lazaric 2018a,b; Bardi & Priuli 2014a,b). Please review the reference list carefully and remove any unintentional duplicates.

### Typos

1. L692: missing “*” in $Q$.
2. L1182-L1185: missing “(” before $(\sigma)$.
3. L1193: missing $1/2$ factor.
4. L1211: missing “(” before $\Lambda^i$.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper propose a Thompson sampling method for solving stochastic different games (with linear–quadratic dynamics) with homogeneous player setting in a Bayesian inference framework. The authors show that the proposed method enjoys state of the art regret bound and leads to Nash equilibrium.

### Strengths
1. This paper propose a novel approach for solving multi-player SDGs using Thompson Sampling, extending prior work's approach. 
2. The framework also relax assumptions in previous work on independence of the players. 
3. The results on Nash equilibrium is very interesting and potentially impactful to the field.

### Weaknesses
I am not in this field, so my feedback might be limited and please correct me if I am wrong. That said, there are a few of my concerns
1. The scope of this work is quite limited. The authors assume there is no coupling between the players from the dynamics side, but only through costs. Although the authors motivates the scenarios in the intro, but the applicability of such framework remains elusive, and quite restrictive. 
2. The tools the authors used seem to be borrowed from prior works. I.e., the adaptive episode length idea, the continuous time framework, and TS itself is extensively studies in the literature. So what's the true contribution of this work, besides attaching all the techniques together?

### Questions
See above.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper treats the question of equilibrium learning in general, non-zero-sum stochastic differential games, that is, games where the players' payoff depends on the actions chosen at each instance of time and a state that evolves following a stochastic differential equation (an open-loop control).

An equilibrium strategy in this context is an admissible selection of actions in $\mathbb{R}^d$ over time which is maximizes unilaterally the long-run (Cesàro) average payoff of each player over time. This is in turn characterized by an associated ensemble of Hamilton-Jacobi-Bellman equations, a set of nonlinear partial differential equations that provide necessary and sufficient conditions for unilateral optimality.

The specific state evolution model of the authors is of the form
$$
dX(t) = (AX(t) - \alpha(t)) \, dt + \sigma\, dW(t)
$$
where $X(t)$ is the state of the game at time $t$, $\alpha(t)$ is the players' action profile at time $t$, $A$ is a coupling matrix, and $W(t)$ is an ordinary Brownian motion. Under certain assumptions for $A$ and the (quadratic) cost functions of the game, it is known by prior work that the associated set of HJB equations admits a unique solution (also of quadratic form).

The authors consider the case where the coupling matrix $A$ is unknown, so the players cannot solve the associated HJB equations. Their main contribution is an episodic Thompson sampling (TS) algorithm which starts by sampling from an initial Gaussian prior distribution and proceeds episode-by-episode by observing the state and updating the posterior following a specific stopping criterion. Based on this algorithm, the authors are able to show that (i) the players' regret is bounded as $\mathcal{O}(\sqrt{T \log T})$; and (ii) the proposed algorithmic policy is a Nash equilibrium with probability $1$. The authors also validate these results through numerical simulations in games with 10, 50 and 100 players.

### Strengths
I am not sufficiently knowledgeable in stochastic differential games to provide an expert opinion but, as far as I could tell, the authors' analysis is sound, and the positioning of their contributions in the surrounding literature is fair. The contribution itself seems in line with what could be expected from a good paper in the field.

### Weaknesses
My main concern with this paper is its thematic alignment with ICLR. Even though I could easily see this paper published in a top-tier control venue (IEEE CDC, TAC or SIOPT SICON), the fit with ICLR is very slim. This would be less of an issue if the field of (stochastic) differential games were more accessible from a technical standpoint but, as it currently stands, the paper's technical content and contributions would only be accessible to an infinitesimally thin slice of ICLR's generalist audience—even its more theoretical side.

My "reject" recommendation reflects precisely this: it should not be taken as a criticism of the paper's technical content (which I cannot assess at the level of a dedicated expert), but as an assessment of the suitability of this paper for ICLR as a whole.

### Questions
None (see above), except for a remark: the paper is not about *stochastic* games in the sense of Shapley (also referred to as Markov games in ML), but stochastic *differential* games. This should be made clear in the title, as the two fields are quite disjoint (Ι was definitely expecting something different coming in from the title).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies an $N$-player linear–quadratic stochastic differential game with ergodic cost where the drift matrix $A$ is unknown but common to all players. Each player observes only their own state, maintains a Bayesian posterior over  $A$ with Gaussian prior, and plays using a Thompson Sampling policy with adaptive episode lengths. The authors prove a Bayesian regret bound $\mathcal{O}(\sqrt{T}\log T)$ for each player, which is independent of $N$. They show that the TS policy profile constitutes a Nash equilibrium. They provide a continuous-time posterior update (Proposition 3.1), decompose the regret into sampling/strategy/mismatch terms (Proposition 3.2) and bound each (Proposition 3.3), plus a coupling argument establishing equilibrium (Theorem 3.2). Experiments simulate regret scaling and show normalization by $\sqrt{T}\log T$ remains bounded.

### Strengths
* The paper proves the Bayesian regret bound for ergodic $N$-player games, matching best-known orders for LQ control while not depending on number of players $N$.
* Propose Thomspon sampling algorithm to handle the unknown $A$ setting, proving the Nash equilibrium of the TS profile under additional stability conditions.

### Weaknesses
* The model is based on the previous work Bardi & Priuli (2014a). This paper handles the setting where the matrix $A$ is unknown but does not explain why this setting is meaningful in practice.
* This paper focus on the theoretical side, the author use a full section (Section 2) to introduce existing results but does not claim their technique contribution over previous work.

### Questions
* The numerical experiment section is confusing. It lack explanation to help the reader understand how the experiment is conducted.
* Assumption 3.1 (2) seems non-trival, can you provide concrete examples or experiments to verify this assumption?
* The proof technique seems standard, please explain the technique contribution.

### Soundness
3

### Presentation
2

### Contribution
2
