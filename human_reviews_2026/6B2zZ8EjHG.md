# Bayesian Optimization by Minimum Filling Distance Search

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Bayesian Optimization sequentially queries objective function evaluations, often focusing on the expected utility of evaluating corresponding candidates under uncertainty with a learned probabilistic model of underlying true objective functions. We propose a new filling distance based acquisition function, termed Minimum Filling Distance Search (MFDS), to explicitly takes into account the location of the previous queried observations so that acquisition iterations can avoid oversampling and therefore explore the whole design space more efficiently. For multi-objective optimization, in addition to efficiently approaching the Pareto front, the queried candidates by MFDS are well spread over the entire Pareto set. We provide an asymptotical convergence proof and empirically evaluate MFDS performances, demonstrating the improvement over existing methods using other acquisition functions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new acquisition function for Single-Objective (SO) and Multi-Objective (MO) Bayesian Optimization (BO), designed to minimize the expected closest distance between an optimal input and a corresponding reference set. In both the SO and MO settings, the acquisition function is formulated as a one-step look-ahead approach. The proposed method has been evaluated on multiple synthetic benchmark functions, demonstrating its effectiveness in improving solution quality and Pareto front coverage.

### Strengths
- The proposal of leveraging the average minimum distance as a metric for formulating the acquisition function is, to the best of my knowledge, novel and interesting.
- The paper considers both the SO and MO settings and demonstrates that a similar formulation can be applied, with experiments cover both settings.

### Weaknesses
- Limited synthetic benchmarks: The paper benchmarks on both single-objective functions (Gaussian mixture, Shekel, Forrester) and multi-objective problems (2-dimensional GMM, DTLZ2, and one real-world problem RE3-5-4) which is plausible. However, the benchmark complexity is more of proof of concept level, and the claim could be much more strength with more complicated empirical comparison.
- Unclear attribution of performance gain: One unclear aspect is that the authors directly start with a one-step look-ahead formulation (Eq. 14, Eq. 17). Since one-step look-ahead methods already provide a performance benefit, it is not clear whether the performance gain of this algorithm stems from the new acquisition function formulation or from the look-ahead strategy itself. To ensure fair comparison, the standard acquisition functions should also be evaluated with one-step look-ahead. If the authors believe the look-ahead strategy is a core contribution, an ablation study should be included.
- The complexity and scalability: the algorithm has its fundamental complexity and scalability issue, which would restrict its apllicability in moderate-high input dimensionalities.

### Questions
- How is the  $X_n$ is set if it is different from $X$ ?

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
The authors propose to use as an acquisition function for Bayesian optimization the expected minimum distance between the design points and the true minimizer.
They develop a convergence proof and demonstrate this method on both single and multi-objective problems.

### Strengths
It's nice to have some theoretical results.
Namely, the authors show that the acquisition function converges to zero.
It would be more ideal to have a bound/rate on regret or something like this but since there are also fairly extensive numerical results I think this is enough.

The numerical results seem adequate, with a decent number of benchmark comparators, uncertainty quantification on the iteration number, and five benchmark applications, which is perhaps sufficient.

### Weaknesses
I found some of the discussion about exisiing BO methods surprising, and I think readers would appreciate more context for these claims:

i) "As a result, when an inappropriate prior is set for the probabilistic model, BO may fall in local optima (Wang & de Freitas, 2014)." What part of that article are you specifically referring to when you say that inappropriate priors lead to local optima?

ii) "[Entropy Search] design works well for objective functions with a single global optimum. For objective functions in real-world applications, however, there is often not just one single optimum." I think I unerstand why, conceptually, ES assumes a single global optimum. But have we actually observed any failure cases empirically in the multi-optimum case? Since it is well defined even if one of the local optima is only epsilon worse than the other, it's not obvious to me that it would actually fail in practice. Furthermore, the subsquent Max-Value entropy search of Wang and Jegelka would seemingly solve this anyways unless I'm issing something.

I like some of the intuition built in the intro about why this method is useful under model misspecification, but I ultimately did not fully understand the derivation of the acquisition function. It would be helpful to in particular say a bit more about how the second term in the definition was obtained.

Though a reasonable idea conceptually, there does not appear to be much in the way of analyitcal tractability for this acquisition function, and the authors' experiments are on small iteration counts. This will limit the applicability of the method.

### Questions
1) I'm a bit confused with the main definition of the acquisition function u at the bottom of page 4.  The first term does not seem to depend on x. Is it supposed to be D_n\cup\{x,y\} instead of just D_n?

2) At the end of the day, X_N is going to be the set of previously observed design points at time N, or not?

3) Doesn't this method suffer from a similar assumption of a unique global optimum as you mention in your criticism of ES?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a new acquisition function for Bayesian Optimization, Minimum Filling Distance Search (MFDS). MFDS chooses the next point by minimizing the expected distance between the sequence of sampled locations and (i) the posterior distribution of the global minimizer (single‑objective) or (ii) the posterior distribution of the Pareto set (multi‑objective). This explicitly leverages where past samples are, aiming to avoid oversampling and to cover regions with high probability of optimality. The authors prove an asymptotic convergence result for the single‑objective case (under specific assumptions) and empirically compare MFDS to other popular acquisition functions.

### Strengths
- Nice and intuitive geometric representation of the objective. This acquisition function makes sense.
- Theoretical proof of the convergence. 
- Results showing that this method outperforms others (though more tests should be performed).

### Weaknesses
- Theory is limited just to 1D. 
- Claiming theoretical proof of convergence and then mentioning the "almost surely" converges should be clarified better.
- Very limited tests are performed. The authors should use more test functions and average results across different seeds.

### Questions
- Fig.1 why don't you show comparison between different acquisition functions of the next selection point where the set of all the current points is the same? Current comparison seems to be unfair. 
- why are n-1 iterations done in one way and the last iteration is a greedy approach? This seems like a heuristic.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper proposes an acquisition function for Bayesian optimization. The proposed acquisition function considers the minimum distance between the sampling path and the set of optimal solutions. This acquisition function can also be used for multi-objective optimization. The authors prove the convergence of the proposed method and demonstrate its superior performance through experiments.

### Strengths
- The problem is well-motivated, and considering the past observations is intuitively important for Bayesian optimization.
- Experimental results demonstrate better performance of the proposed method than existing baseline methods.

### Weaknesses
I am not an expert in Bayesian optimization, so I could not fully understand the advantage of the theoretical result over existing theoretical guarantees in this field. In particular, I did not fully understand what benefit we obtain from Theorem 1. Does $\lim_{n\to \infty} \min_{x'} G(x' \cup X_n, \mathcal{D}_n) = 0$ indicate that $x_n$ converges to the optimal solution(s)? Is there a similar convergence guarantee for other acquisition functions?

### Questions
My main concerns and questions are outlined in the Weaknesses section. Additionally, I have the following questions:

- In Eq. (8), what is the definition of $P$?
- In Eq. (12), it seems difficult to compute the integral involving $\mu$ exactly. How can we compute this integral? What is the computational complexity?
- In Section 5.1, what does the equation $f_{GM}(x) = -0.5,\mathcal{N}(-\mu, 2\sigma^2) - 0.55,\mathcal{N}(\mu, \sigma^2) + 1$ mean? Does it mean that the probability density functions of the Gaussian distributions $\mathcal{N}(-\mu, 2\sigma^2)$ and $\mathcal{N}(\mu, \sigma^2)$ are used?

### Soundness
3

### Presentation
3

### Contribution
3
