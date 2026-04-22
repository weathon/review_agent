# On the $O(1/T)$ Convergence of Alternating Gradient Descent–Ascent in Bilinear Games

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 2, 6

## Abstract
We study the alternating gradient descent-ascent (AltGDA) algorithm in two-player zero-sum games. 
    Alternating methods, where players take turns to update their strategies, have long been recognized as simple and practical approaches for learning in games, exhibiting much better numerical performance than their simultaneous counterparts.
    However, our theoretical understanding of alternating algorithms remains limited, and results are mostly restricted to the unconstrained setting. 
    We show that for two-player zero-sum games that admit an interior Nash equilibrium, AltGDA converges at an $O(1/T)$ ergodic convergence rate when employing a small constant stepsize. This is the first result showing that alternation improves over the simultaneous counterpart of GDA in the constrained setting.
    For games without an interior equilibrium, we show an $O(1/T)$ local convergence rate with a constant stepsize that is independent of any game-specific constants. 
    In a more general setting, we develop a performance estimation programming (PEP) framework to jointly optimize the AltGDA stepsize along with its worst-case convergence rate. 
    The PEP results indicate that AltGDA may achieve an $O(1/T)$ convergence rate for a finite horizon $T$, whereas its simultaneous counterpart appears limited to an $O(1/\sqrt{T})$ rate.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the problem of the average convergence rate of online gradient descent-ascent in bilinear zero-sum games. The main results include:  
* A computer-assisted framework to find the optimized step sizes for a given time horizon.  
* A proof of the $O(1/T)$ average convergence rate of online gradient descent-ascent in constrained bilinear zero-sum games with interior equilibrium. For games without interior equilibrium, a local result is also provided.

Numerical results are also provided to support the theoretical findings.

### Strengths
* The purposed PEP framework provide new tools to proof the convergence rate of learning algorithms in games.
* The result of $O(1/T)$ convergence rate of online gradient descent-ascend for constrained zero-sum games greatly improves the results of (Bailey et al., 2020) in the unconstrained setting. 
* The proof techniques, especially the construction of the decayed energy function and its relation to the duality gap, are interesting.
* The paper is well-written and easy to follow, and the motivation and the strategy of proofs are clearly stated.
* Numerical results coincide well with the theoretical findings.

### Weaknesses
One weakness I see is that the PEP part (Section 4) is a bit rough. For example, a crucial point of how to reduce the infinite-dimensional nonconvex optimization problem INNER to a solvable finite-dimensional problem is not clearly stated in the main text.

Moreover, I think two recent works of (Feng et al., 2025), which studied the seperation between alternating and simultaneous momentum dynamics in zero-sum games, and (Hait et al., 2025), which studied the regret properties of alternating learning dynamics in convex games, are also relevant to this work. It would be good if the authors could include them in the related work section.

A minor problem: In Figure 1, the label of the x-axis should be T instead of N.

Reference:
1. Feng et al., Continuous-Time Analysis of Heavy Ball Momentum in Min-Max Games . ICML 2025
2. Hait et al, Alternating regret for online convex optimization. COLT 2025

### Questions
What are the obstacles to extending the current framework to the analysis of alternating Mirror Descent algorithms in zero-sum games (Wibisono et al., 2022)?

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
4

### Summary
The paper studies alternating gradient descent ascent for bilinear games and provides three results, a rate of $O(1/T)$ for average convergence in games with interior Nash Equilibria, a local convergence of the same rate without the aforementioned restriction and a reformulation of the method under a performance estimation framework that allows the computation of an optimized stepsize. Regarding the first two contributions, the authors identify a non-decreasing energy function associated with the duality gap, that facilitates the analysis. The last contribution requires a series of transformations to reduce the original problem to a tractable form.

### Strengths
The paper addresses an important open problem and makes substantial progress towards resolving it. The approach for deriving the convergence rate may be based on a simple energy function but is quite technical and involved. The same holds true regarding the PEP framework. I also found the presentation clear.

### Weaknesses
The only weakness I can point to is that the existence of an interior NE is not a very weak assumption, like say the uniqueness of a NE.

### Questions
Could the authors elaborate on any approaches that they perhaps try but fail to generalize the result completely?

I will also add some minor comments:
In Lemma 8, you can conclude the monotonicity of the energy function without using the fact that the NE lies in the interior, since the third and fourth terms of equation 20 are jointly nonnegative. 

In line 1418, “Lemma 12” instead of “12”.

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
The paper studies Alternating Gradient Descent–Ascent (AltGDA) in two-player zero-sum constrained bilinear games. While previous works have shown an empirical faster rate of AltGDA and some improved $O(T^{-2/3})$ convergence rate, this paper proves an $O(1/T)$ convergence rate when an **interior** Nash equilibrium (NE) exists, using a constant stepsize dependent on the value of the interior NE. Specifically, the sufficient condition is to have the stepsize no larger than the smallest non-zero equilibrium mass. It contrasts this with simultaneous GDA’s typical $O(1/\sqrt{T})$ behavior. The paper also shows a local $O(1/T)$ rate for non-interior games under a universal constant stepsize, with the bound expressed via a gap parameter linked to a maximal-support NE. Finally, a performance-estimation-programming (PEP) SDP is used to numerically optimize stepsizes and visualize a near $O(1/T)$ curve for finite $T$.

### Strengths
- The paper is well motivated by the empirical success of AltGDA and establishes the first $O(1/T)$ rate for AltGDA with constraints under an interior NE. This is a clear improvement over simultaneous GDA while using a constant stepsize. The proof leverages a monotone energy decay, which is neat conceptually and interesting in its own right.  
- The use of the PEP framework is also interesting to show the benefit of using alternation. The SDP study provides tangible (though with approximations) finite-time evidence for the benefits of alternation and suggests structured.

### Weaknesses
- One main weakness is that even under the assumption that there exists an interior equilibrium, the stepsize $\eta$ depends on the unknown equilibrium masses. Specifically, the global $O(1/T)$ theorem (interior NE $(x,y)$) requires a stepsize scaled by $\min( \min_i x_i, \min_j y_j )$, which are unknown in practice and may be extremely small. No adaptive rule is provided that attains the same rate without such knowledge.  
- In addition, if the interior NE is near the boundary, the admissible constant stepsize can be impractically tiny. There is no quantification of worst-case constants or a detection-and-retuning mechanism, and this will also leads to a bad convergence guarantee.
- The other concern is that the global analysis relies on the existence of an interior NE. When no interior NE exists, the same energy monotonicity fails; the paper does not provide an in-process diagnostic to decide which regime one is in.
- For the second result with local convergence rate, while the learning rate now is not dependent on $\min( \min_i x_i, \min_j y_j )$, this requires the initial point to be close to an NE, with the distance also dependent on these problem-dependent constants.

### Questions
- Is there an adaptive stepsize rule (e.g., backtracking based on duality-gap trends or energy-surrogate monotonicity) that achieves the same $O(1/T)$ rate when an interior NE exists, without using $\min\{\min_i x_i^*, \min_j y_j^*\}$ a priori?  
- I wonder whether there is some way to detect the existence of an interior NE in an online fashion. For example, can persistent monotone decrease of the energy and support persistence be shown to imply interior-NE structure (or neighborhood), while non-monotonic energy indicates the alternative regime?  
- Can the aurthors explain more on the choice of the learning rate $\eta$. Is that possible to bound the largest admissible constant stepsize using spectral/simplex geometry of $A$ alone, or give conditions under which $\min( \min_i x_i, \min_j y_j )$ are bounded away from zero?  
- Can this analysis be generalized to two-player general-sum game?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the AltGDA algorithm in the constrained min-max setting. It gives a convergence rate $O(1/T)$ under interior NE assumptions.

### Strengths
The paper gives $O(1/T)$ convergence for the AltGDA algorithm under interior NE assumptions in the constrained case.

### Weaknesses
The algorithm is known, so the main claimed novelty is the analysis. Lemma 2 is the key new piece, which follows once interior NE existence is assumed. The result and proof are straightforward, leading the contribution to appear incremental since Theorem 1 follows almost immediately.

The authors fail to sufficiently discuss long-known theoretical results of $O(1/T)$ for the constrained min-max setting, with only a brief mention of optimistic methods, though there is a rich line of literature here (e.g. [1-3]), nearly all of which has been ignored by the authors.

The authors' results hold only for games with an interior Nash equilibrium, or else for local convergence. Previous first-order algorithms achieve $O(1/T)$ rates without these conditions. The authors should explain this point better and clarify how their first-order results are weaker than previous first-order results.

[1] Korpelevich, Galina M. "The extragradient method for finding saddle points and other problems." Matecon 12 (1976): 747-756.

[2] Nemirovski, Arkadi. "Prox-method with rate of convergence O (1/t) for variational inequalities with Lipschitz continuous monotone operators and smooth convex-concave saddle point problems." SIAM Journal on Optimization 15, no. 1 (2004): 229-251.

[3] Nesterov, Yurii. "Dual extrapolation and its applications to solving variational inequalities and related problems." Mathematical Programming 109, no. 2 (2007): 319-344.

### Questions
Can the authors say more about how their analysis differs from the analysis in Bailey et al. (2020)?

Can the authors explain what they believe is the purpose of analyzing alternating first-order methods, since optimal first-order methods have already been known for several decades?

Can the authors explain what they believe to be the non-trivial parts of their analysis?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies alternating gradient descent–ascent with Euclidean projections in two-player zero-sum bilinear games over convex compact sets. They focus on constrained matrix games where alternation is widely used in practice but poorly understood theoretically. The authors first show that when the game admits an interior Nash equilibrium, AltGDA with a suitably small constant stepsize enjoys an ergodic  $\mathcal{O}(1/T)$ convergence rate in duality gap. Giving the first result where plain alternation provably improves over simultaneous GDA, they prove a local $\mathcal{O}(1/T)$ convergence result for general bilinear games. Beyond these analytic results, the paper develops a performance estimation problem (PEP) formulation that encodes AltGDA’s worst-case behavior as an SDP and numerically optimizes stepsizes. The PEP evidence suggests AltGDA may in fact achieve an $\mathcal{O}(1/T)$ rate more broadly for finite horizons, while simultaneous GDA appears limited to $\mathcal{O}(1/\sqrt{T})$, and experiments on random matrix games corroborate the faster empirical convergence of AltGDA.

### Strengths
* The combination of a global result under an interior NE and a local result around a maximal-support NE gives a fairly complete picture.
* The energy termed $\mathcal{E}$ used to capture the two-phase dynamics (boundary hits vs. interior cycling) and to bound residual terms is conceptually interesting.
* Experiments verify the theoretical results.

### Weaknesses
* The global convergence rate requires an interior NE, which is quite restrictive in game applications.
* The PEP-based evidence for global convergence in general convex compact sets is compelling but not a proof.
* The analysis assumes deterministic gradients. Many game-solving methods operate with sampling or bandit feedback, it is unclear whether the energy-based analysis or PEP insights extend to noisy gradients.

### Questions
* In Theorem 1, $\eta$ depends on $\min x_i, \min y_i$, which are unknown. Can you provide data-driven or adaptive rules that maintain the $\mathcal{O}(1/T)$ rate?
* Are there lower bounds showing that (a) AltGDA cannot do better than $\mathcal{O}(1/T)$ in this setting, or (b) the constants in Theorems 1–2 are not too pessimistic?
* Which parts of your proof crucially use bilinearity? Could the energy-based argument or the Lemma 1–2 telescoping scheme extend to smooth convex–concave problems with Lipschitz gradients but non-linear coupling?

### Soundness
3

### Presentation
3

### Contribution
2
