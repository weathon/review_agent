# Towards a Sharp Analysis of Offline Policy Learning for $f$-Divergence-Regularized Contextual Bandits

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Many offline reinforcement learning algorithms are underpinned by $f$-divergence regularization, but their sample complexity *defined with respect to regularized objectives* still lacks tight analyses, especially in terms of concrete data coverage conditions. 
In this paper, we study the exact concentrability requirements to achieve the $\tilde{\Theta}(\epsilon^{-1})$ sample complexity for offline $f$-divergence-regularized contextual bandits. For reverse Kullback–Leibler (KL) divergence, arguably the most commonly used one, we achieve an $\tilde{O}(\epsilon^{-1})$ sample complexity under single-policy concentrability for the first time via a novel pessimism-based analysis, surpassing existing $\tilde{O}(\epsilon^{-1})$ bound under all-policy concentrability and $\tilde{O}(\epsilon^{-2})$ bound under single-policy concentrability. We also propose a near-matching lower bound, demonstrating that a multiplicative dependency on single-policy concentrability is necessary to maximally exploit the curvature property of reverse KL. 
Moreover, for $f$-divergences with strongly convex $f$, to which reverse KL *does not* belong, we show that the sharp sample complexity $\tilde{\Theta}(\epsilon^{-1})$ is achievable even without pessimistic estimation or single-policy concentrability.
We further corroborate our theoretical insights with numerical experiments and extend our analysis to contextual dueling bandits. We believe these results take a significant step towards a comprehensive understanding of objectives with $f$-divergence regularization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper provides a sharp theoretical analysis of offline contextual bandits under f-divergence regularization, a class of objectives underlying many modern offline RL and RLHF algorithms. The key contribution is a refined characterization of the sample complexity required to find an ε-optimal policy with respect to the regularized objective, rather than the unregularized reward.

For the reverse KL divergence, the authors prove an Ō(ε⁻¹) sample complexity under single-policy concentrability—strictly improving those requiring stronger all-policy coverage assumptions. They also provide a matching lower bound showing that the dependency on the single-policy concentrability coefficient is necessary.

For general f-divergences with strongly convex f, they establish that the same fast rate Θ̃(ε⁻¹) is achievable without any coverage dependence, supported by both upper and lower bounds. The analysis leverages a novel moment-based argument that couples pessimism with the curvature of the regularized objective, providing a new proof technique beyond the standard performance-difference or simulation-lemma analyses.

Empirical results on two-armed synthetic bandits corroborate the theoretical rates and demonstrate distinct scaling behaviors between KL and strongly convex f-divergences.

### Strengths
- Sharp and rigorous theoretical results: The paper achieves the first tight sample complexity bounds for offline f-divergence-regularized bandits, closing important gaps in the literature. The improvements—from stronger coverage assumptions to Ō(ε⁻¹) under single-policy coverage—are both technically and conceptually relevant.

- Broader generality beyond KL: Extending the framework to strongly convex f-divergences (e.g., χ²) and proving that fast rates can be achieved without any concentrability assumptions is conceptually strong. It clarifies how curvature of the regularizer directly impacts coverage requirements.

- Balanced theoretical and empirical validation: Although simple, the experiments convincingly verify the predicted n⁻¹ scaling and the absence of coverage dependence for strongly convex f. The theoretical lower bounds are also near-matching, strengthening the claims.

- Relevance and timeliness: The results have clear implications for RLHF, offline RL with KL or χ² regularization, and policy optimization under limited coverage, making the work both theoretically rigorous and practically relevant.

### Weaknesses
- The experiments only involve two-armed synthetic bandits. Although sufficient to verify scaling, a richer empirical study (e.g., linear contextual or synthetic high-dimensional bandits) would strengthen the practical credibility.
- Som part of the writting is unclear (see questions)

### Questions
- $\pi_{\gamma}$ in the LHF of the equation in Lemma 2.14? 
- I don't understand Lemma 2.14, does the lemma hold for every $\lambda$, for for some $\lambda$ ($\exists \lambda$)? 
- How critical is the assumption that the behavior policy equals π_ref? Would the same rates hold when π_ref differs from the logging policy?
- For the χ²-regularized case, could you provide intuition or experiments showing the trade-off between strong convexity (α) and the bias introduced by over-regularization?
- Could your analysis provide insights into why KL regularization remains dominant in practice despite its stronger coverage requirements?

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
2

### Summary
This paper provides a tight statistical analysis of offline contextual bandits with f-divergence regularization, focusing on the widely used reverse KL and on general strongly convex f-divergences. The authors establish the $O(1/\epsilon)$ sample-complexity results under single-policy concentrability for reverse-KL-regularized objectives, A novel pessimism-based proof technique is used by leveraging the curvature of the reverse-KL objective and a moment-based lemma. For strongly convex f-divergences, the authors further show that the same rate can be achieved without any coverage assumptions, supported by a matching lower bound and simple empirical verification.

### Strengths
1. This paper established a near-optimal sample complexity for offline regularized bandits.
2. The moment-based argument and integration of pessimism with curvature properties seem novel in offline bandits.
3. The extension to strongly convex f-divergences provides a unified theoretical view.

### Weaknesses
1. Experiments are limited. Only toy two-armed-bandit cases are shown, perhaps including some real dataset would strengthen the arguments.
2. The paper seems to put more effort on introducing the analysis for KL-divergence regularized Bandits, while the title suggest general f-divergence. Perhaps including discussions on how to handle general f-divergence regularized bandits would make the main body match with the title.

### Questions
The paper mentioned that for strongly convex f-divergence, we do not need the data coverage assumption. I am wondering if we use the local convex f-divergence, can KL analysis be applied for this case? Or there is a fundamental challenge here.

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
The paper investigates the theoretical offline reinforcement learning with general function approximation. They provide sharper lower and upper bounds with a proposed KL-divergence-regularized simple algorithm, where they also prove single-policy concentrability is efficient and necessary. Also, they investigate the f-divergence-regularized algorithm, which can converge normally without coverage assumptions. The results are very novel in this area. A few experiments are conducted to corroborate their theoretical results.

### Strengths
1. The paper is well-written. The story is clear and consistent.
2. The contributions should be solid. a) The f-divergence could provide another way to understand the requirement for offline learning. b) The refined mean-value-type risk upper bound has some technical improvements.

### Weaknesses
1. The results in the f-divergence algorithm are not compared with the global optimal policy, which is not the same as the KL-divergence. In other words, it seems to replace the coverage assumption with another assumption that the global optimal policy is the optimal policy they defined in 3.1. It is still an important contribution, but it may be overstated.
2. The numerical experiments are very simple, more like a sanity check rather than a validation.

### Questions
1. Could you provide more intuition for Lemma 2.14? It appears central to your KL-regularized algorithm. Also, this technique seems to originate in prior work—can you clarify the differences in proofs between this one and the previous one? 
2. For the f-divergence regularized algorithm, is my understanding in the weakness correct? If no, please justify it. If yes, please elaborate a little more about what is the potential impact or benefit of it.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the sample complexity of regularized policy optimization with general $f$-divergence regularizations in contextual bandits. For the special case of KL-divergence regularization, the authors derive an $\mathcal{O}(\epsilon^{-1})$ upper bound when the reference policy provides sufficient coverage of the optimal regularized policy. They further present a matching lower bound, up to a coverage parameter, that depends on a weaker density-ratio coverage of the optimal regularized policy, indicating that some coverage assumption is necessary for efficient learning. Finally, for $\alpha$-strongly convex function divergences, they establish a minimax-optimal sample complexity that does not depend on any coverage parameters.

### Strengths
1. Relaxation from all policy coverage to optimal policy coverage assumption with clever use of pessimistic bonus term
2. The lower bound of Theorem 2.11 has a multiplicative coverage term, showing some coverage assumption is needed for any efficient algorithm
3. The work shows that for $f$-divergence-regularized objectives, if $f$ is strongly convex, then no coverage assumption on the reference policy is necessary.

### Weaknesses
1. The worst-case gap between $C^{\pi^\*}$ and $D^2_{\pi^\*}$ scales as $|S||A|$. This linear dependence on the size of the state space can render Algorithm 1 inefficient, even when $C^{\pi^\*}$ is a constant. A more detailed discussion or empirical illustration of how these two coverage measures relate in practice would strengthen the paper.
2. Theorem 3.4 constructs only a specific instance of an $\alpha$-strongly convex $f$ (a scaled $\chi^2$ divergence) that matches the upper bound, rather than establishing a general $f$-dependent lower bound.

### Questions
1. The coverage term of lowerbound is bounded by $ \min( C^{\pi^\*}, \mathcal{O}(\exp(\eta) )) $ in Theorem 2.11. Similarly, the coverage term of sample complexity of Algorithm 1 can be bounded by $ \min( D^2_{\pi^\*}, \mathcal{O}(|S||A|\exp(\eta) )) $. Can this be improved to $ \min( D^2_{\pi^\*}, \mathcal{O}(\exp(\eta) )) $?
2. Have the authors tried to show similar upper bound for $f$-divergence-regularized CB when $f$ is only locally strongly-convex by adding an appropriate bonus term similar to Algorithm 1?

### Soundness
3

### Presentation
3

### Contribution
2
