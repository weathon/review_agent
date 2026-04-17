# Online Prediction of Stochastic Sequences with High Probability Regret Bounds

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4, 8

## Abstract
We revisit the classical problem of universal prediction of stochastic sequences with a finite time horizon $T$ known to the learner. The question we investigate is whether it is possible to derive vanishing regret bounds that hold with high probability, complementing existing bounds from the literature that hold in expectation. We propose such high-probability bounds which have a very similar form as the prior expectation bounds. For the case of universal prediction of a stochastic process over a countable alphabet, our bound states a convergence rate of $\mathcal{O}(T^{-1/2} \delta^{-1/2})$ with probability as least $1-\delta$ compared to prior known in-expectation bounds of the order $\mathcal{O}(T^{-1/2})$. We also propose an impossibility result which proves that it is not possible to improve the exponent of $\delta$ in a bound of the same form without making additional assumptions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the classical universal prediction problem as in Merhav & Feder (1998). 
Here, we assume some underlying distribution (random process) $P$ that generates a sequence $Z_1, \cdots, Z_T$. 
We are given an approximating distribution $Q$ of $P$, so that at each time step, the learner predicts the Bayesian-optimal decision under $Q$. 
It is known (and fairly straightforward) that the expected regret between the Bayesian-optimal rule under $Q$ and the (unobservable) Bayesian strategy under $P$ is bounded above by $O(\sqrt{T \cdot D(P\|Q)})$. 
This paper strengthens the result by establishing a high-probability bound of the form $O(\sqrt{T D(P\|Q)/\delta})$, where $\delta$ is the confidence parameter. 
The paper further demonstrates that the $1/\sqrt{\delta}$ dependency (and for total variation, the $1/\delta$ dependency) cannot be improved if no additional structure is known between $Q$ and $P$.

### Strengths
1. In my opinion, the main technical contribution of the paper is the lower bound (Theorem 5), which shows that with only a global bound on the divergence between the approximate $Q$ and the underlying true $P$, the dependency on $\delta$ cannot be better than polynomial. This bound contrasts with much of the prior results in learning with expert advice and bandit settings, where the dependency is typically polylogarithmic.

2. The paper instantiates their upper bound in several settings where an approximate $Q$ is obtainable, such as countable $Z$, i.i.d., and Markov settings.

### Weaknesses
1. The upper bound proof is fairly simple and follows from a straightforward application of martingale concentration inequalities.

2. I'm not sure how significant this result is, as in many natural settings we already have high-probability bounds that depend logarithmically on $\delta$. The lower bound in this paper feels more like a pathological example. For instance, the examples in Section 6 only show the upper bounds; if the authors could demonstrate any **natural** settings where a polynomial dependency on $\delta$ is inevitable, the significance of the paper would be substantially boosted.

3. All the results in this paper are based on the hypothetical approximating distribution $Q$. It would be better to have some results that depend on the specific problem structure or on particular algorithms.

### Questions
I think the negative result in this paper is similar in spirit to the work of I. Aden-Ali, Y. Cherapanamjeri, A. Shetty, and N. Zhivotovskiy, *“The One-Inclusion Graph Algorithm is Not Always Optimal,”* in COLT 2023. It would be helpful for the authors to discuss it, as both works highlight cases where polynomial rather than logarithmic dependencies are unavoidable.

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
3

### Summary
This paper studies online prediction of stochastic sequences over a known finite horizon T and derives high-probability regret bounds that complement prior in-expectation results. The authors establish a bound of order $O(T^{-1/2} \delta^{-1/2})$ holding with probability at least $1−\delta$, which mirrors the form of earlier expectation bounds (e.g., $O(T^{-1/2})$). They further prove an impossibility result showing that the dependence on $\delta$ cannot be improved without additional assumptions.

### Strengths
1. To the best of my knowledge, it’s the first work to provide high-probability regret bounds for the universal prediction of stochastic sequences without i.i.d. assumption or expert advice.

2. The paper is well-organized with a clear statement of problems and a structured presentation of results.

3. It is nice to see that the dependence on $\delta$ is already tight in their high probability bound.

### Weaknesses
1. Practical Applicability: Although the mismatched prediction results are general, their application to universal prediction relies on the construction of a suitable parametric family and may be intractable for a concrete problem.

2. This paper assumpts that the learner should know some distribution $Q$ which is assumed to be “sufficiently similar” to the unknown distribution $P$. I think it is quite a strong assumption. It would be better if the authors can add some discussions about how to learn $Q$.

3. Limited Discussion about Computation: The paper does not address how to computationally implement the proposed strategies. It may be computationally complex to apply the methods in practice.

### Questions
1. Can the authors provide a concrete example to construct a tractable $Q$ for a non-i.i.d. process and analyze how the regret bounds perform in practice?

2. Can the results be extended to more general settings, i.e. the loss functions are unbounded but may have heavy-tail or light-tail distribution?

3. This paper does not discuss the dependence on the constant $L$. Is it necessary to consider the improvement of $L$ and is it possible to achieve a tighter regret bound?

### Soundness
2

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
This paper studies universal online prediction of stochastic sequences with a known horizon and upgrades classic in-expectation guarantees to high-probability bounds under general bounded losses. It analyzes a mismatched predictor under a reference law and then transfers the results to universal prediction via Bayesian mixtures. The main results give pathwise and distributional high-probability bounds with the same $1/\sqrt{T}$ scaling as expectation bounds, plus an impossibility result that explains the sharp dependence on the confidence parameter. The paper also summarizes consequences for countable alphabets and standard i.i.d./finite-memory Markov families.

### Strengths
- Clear modular proof strategy (martingale concentration around an empirical TV term, plus information-distance control) that feels broadly reusable
- First high-probability regret guarantees in this stochastic-sequence setup with general bounded loss and non-i.i.d. data
- Useful universal-prediction corollaries via mixtures; the rate summary is easy to parse.

### Weaknesses
- I find the dependence on the confidence parameter heavy: once the random pathwise term is de-randomized, the bounds pick up $1/\delta$ or $1/\sqrt{\delta}$ factors. I get that the impossibility result shows this is unavoidable in full generality, but for practical confidence targets the guarantees feel conservative. Readers would be curious to see whether under mild added assumptions (mixing, exp-concavity/log-loss) one can recover something closer to $\log(1/\delta)$.
- From a practicality angle, calibration seems missing. The universal consequences rely on large mixtures, which are intractable and may hide large constants. I’d feel more confident with a small synthetic study (finite-alphabet, or simple parametric families) that reports actual constants and clarifies the assumptions behind the rate table.

### Questions
Please address the concerns raised above.

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
This paper studies the universal prediction problem of stochastic sequences. Given a sequence $Z_1, \ldots, Z_{T}$, the goal is to give a prediction at every step $t$ based on the previous observations $Z_1, \ldots, Z_{t-1}$ that achieves a small error compared to the optimal strategy that has prior knowledge about the underlying distribution of the sequence. The authors show that $(T\delta)^{-1/2}$ regret can be achieved with probability $1-\delta$, and the polynomial dependence on $\delta^{-1}$ cannot be improved in the algorithm analysis.

### Strengths
1. Prediction of stochastic sequences is a fundamental problem, and a high-probability guarantee is often more relevant for practitioners than expected error. This work fills a gap in previous work in that only an expected regret was known.
2. The authors show that in the high-probability setting, the same convergence rate of $T^{-1/2}$ w.r.t the time horizon can be achieved. The algorithmic framework has some generality, in that similar bounds are also obtained for other settings with different assumptions on the input domain.

### Weaknesses
1. The main weakness is that the dependence on $\delta$ is not ideal, and it seems that the lower bound result does not hold generally for all algorithms. Specifically, Theorem 5 appears to be specific to some class of policies satisfying equation (3) in the paper. Thus, it only shows that the *error analysis* for the proposed algorithm is in a way optimal, but does not seem to be a fundamental limit that generally applies to all algorithms.

### Questions
1. It would be great if the authors could confirm whether Theorem 5 generally holds for all algorithms, or only algorithms that satisfy certain structures or assumptions. If it is the latter case, why would it be reasonable to focus on these algorithms?
3. For what distribution families do the authors think could achieve $log(1/\delta)$ dependence?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper revisits the classical problem of universal prediction of stochastic sequences and aims to complement existing in-expectation regret bounds in (Merhav and Feder, 1998) by deriving high-probability bounds. The setting involves a learner predicting stochastic outcomes over a known finite horizon with losses measured by a bounded function. They show $O(T^{-1/2}\delta^{-1/2})$ convergence rate with probability at least $1-\delta$, compared to the classical $O(T^{-1/2})$ rate in expectation. The authors also establish an impossibility result, proving that the bounds cannot be improved significantly without stronger assumptions.

### Strengths
- The paper studies a fundamental problem.
- The paper is well-written and has a clear related-work section.
- The paper presents an impossibility theorem clarifying the optimality of the proposed dependence on $\delta$

### Weaknesses
- The paper heavily focusses on theoretical analysis while leaving numerical experiments as future research directions.
- The related work section could be tightened. Citations to multi-armed bandit and MDP literature feel tangential since those problems involve decision-making and exploration-exploitation tradeoffs whereas the present paper studies passive sequence prediction. Such citations distract from the main focus on universal prediction although there might be possible methodological overlap.

### Questions
- The paper extends classical expected regret guarantees to high-probability bounds. Could the authors better illustrate practical scenarios or domains (e.g., time-series prediction, online compression, or universal coding) where such high-probability guarantees provide tangible benefits over expected bounds?

### Soundness
3

### Presentation
3

### Contribution
3
