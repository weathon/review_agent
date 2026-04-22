# Stochastic Adaptive Gradient Descent Without Descent

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
We introduce a new adaptive step-size strategy for convex optimization with stochastic gradient that exploits the local geometry of the objective function only by means of a first-order stochastic oracle and without any hyper-parameter tuning. The method comes from a theoretically-grounded adaptation of the Adaptive Gradient Descent Without Descent method to the stochastic setting. We prove the convergence of stochastic gradient descent with our step-size under various assumptions, and we show that it empirically competes against tuned baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigate Adaptive Gradient Descent without Descent method to the stochastic convex optimization setting. It proposes multiple adaptive step-size strategies that preserve the desirable properties of adaptive methods in the deterministic case. The authors also conduct experiments to validate the effectiveness of their proposed approach.

### Strengths
- This paper is well written and easy to follow. The motivation is clearly presented and well explained.
- The authors provide experimental results to support their claims.

### Weaknesses
- My primary concern lies in the contribution and the novelty of this paper. The core idea of this paper is the decomposition (11).  Could the authors summarize the key technical challenges in their proof?
- In Assumption (2-iii), the authors assume $x_k$ is bounded. Why not simply introduce a projection operator to ensure boundedness instead?
- In Appendix C, it is unclear what the order of $S_K$. The authors should explicitly state or derive convergence rate respect to $k$.
- The authors should also compare different batch sizes in their experiments to better illustrate the scalability and robustness of their method.

### Questions
Please refer to __Weakness__.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new adaptive step-size strategy for stochastic gradient methods for strongly convex optimization problems. The approach adapts the step size based solely on first-order information, without requiring any problem-specific parameters about the level of convexity and smoothness, that are typically not known, or manual tuning. It is derived from a theoretical extension of the Adaptive Gradient Descent Without Descent method to the stochastic setting. The paper provide convergence guarantees under several assumptions and demonstrate through experiments that the method performs well, especially, that it is less sensitive to initial step-size selection than some other step-size strategies for SGD.

### Strengths
The paper studies on adaptive step-size selection, which addresses a critical aspects of stochastic optimization. Step-size choice strongly influences both convergence speed and stability, and many existing work that establish convergence do so by tuning the step-size to problem parameters such as the level of convexity or smoothness, which are rarely known in advance. The proposed method offers a valuable contribution by automatically adapting the step size to the local geometry of the objective function in an online manner, without requiring prior information or manual tuning. This data-driven adaptivity enhances the method’s practicality and robustness, making it potentially useful for a wide range of real-world optimization problems.

The experiment results make a good case for how the adaptive approach is less sensitive to hyperparameters. The baseline algorithms seem reasonably well justified, though, I think for strongly convex problems O(1/t) step-size should be better than O(1/sqrt(t)) stepsize.

The theoretical development is clear and rigorous. The authors provide a well-motivated adaptive rule and prove convergence under standard assumptions, which strengthens the overall credibility of the approach.

The paper makes a useful contribution by extending adaptive step-size ideas, previously explored in deterministic settings, to the stochastic regime. This connection to existing theory is well-articulated and strengthens the paper’s motivation.

### Weaknesses
From a theory standpoint, I do not see a clear advantage of the proposed adaptive step-size scheme. While I acknowledge that adaptivity can be beneficial in practice by removing the need to know parameters such as the Lipschitz constant L or the strong convexity parameter \mu, this does not appear to yield any improvement in theoretical convergence guarantees. In particular, you don't improve theoretical performance for, e.g., O(1/t) step-size that does not have knowledge of L or \mu.

In general, knowledge of the Lipschitz constant is not required to ensure convergence or even a convergence rate in either stochastic or deterministic optimization; both can achieve convergence with appropriately diminishing step sizes. In deterministic optimization, however, diminishing step sizes are typically suboptimal, as they result in slower (sublinear) convergence. In such cases, using a constant step size proportional to 1/L provides improved linear convergence, which motivates adaptive schemes such as Malitsky & Mishchenko (2020) that effectively approximate 1/L without requiring prior knowledge of L.

In contrast, in stochastic optimization, standard diminishing step sizes such as α_t = O(1/t) already achieve O(1/t) convergence rate in the strongly convex and L-smooth setting, without requiring knowledge of L or \mu. Therefore, from a theoretical standpoint, adapting the step size to local curvature does not improve the convergence rate. The main benefit of such adaptive step-size rules would thus be in practice, potentially improving empirical stability or constant factors, rather than providing stronger theoretical guarantees.

Another limitation of the experimental results is the lack of statistical analysis or evidence of robustness across multiple runs. The reported results appear to be based on a single execution, which makes it difficult to assess the variability and reliability of the observed performance. Including averages and standard deviations over several independent runs would strengthen the empirical evaluation and support the claimed advantages of the proposed method.

### Questions
I feel Figure 1 is not so convincing because there is much information lacking. For example, how long do you run the algorithms? Is it always the same number of steps? How are the step-sizes selected for other algorithms? Is this from a single run? How do you know you were not just lucky?

Why do you only consider strongly convexed functions? Is the idea limited to convexity? Or is it easy to extend to non-convex problems?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes AdaSGD, a parameter-free adaptive step-size strategy for stochastic convex optimization that extends “Adaptive Gradient Descent Without Descent” (AdaGD) (a previous paper) to the stochastic setting. The key idea is to estimate a local curvature proxy using the previous mini-batch, yielding three practical variants of an adaptive rule (Eq. (4)) that do not require tuning any problem-dependent scale (apart from a theory-only $\epsilon$-like exponent $\delta$). The method uses a Lyapunov construction adapted from the deterministic analysis and proves (i) almost-sure convergence under standard assumptions via Robbins–Siegmund and (ii) a rate $\mathbb{E}\|x_{k+1}-x^*\|^2\leq C/k^{0.5+\delta}$ for strongly convex objectives (Variant III). Empirically, on linear, ridge, logistic, and Poisson regression, AdaSGD is markedly robust to the initial step size and performs comparably to tuned baselines, while requiring two gradient evaluations per iteration.

### Strengths
The paper offers a clear and principled stochastic adaptation of the deterministic AdaGD scheme by computing the step size from the previous mini-batch, yielding three concrete variants in Eq. (4) and a concise implementation in Algorithm 1. The analysis is careful: a Lyapunov function (Eq. (6)) is constructed to obtain a one-step descent-in-expectation inequality, leading to almost-sure convergence under Assumption 2 when the steps are square-summable (Theorem 3.2), and to an explicit rate $\mathbb{E}\|x_{k+1}-x^*\|^2\leq C/k^{0.5+\delta}$ in the strongly convex case for Variant III (Theorem 3.4). 

The experimental section (Sec. 4) is focused and reproducible: objectives (linear, ridge, logistic, Poisson) are written explicitly, datasets are described with sizes and batch settings and the evaluation protocol uses epochs and mini-batch sampling with replacement. Figures 1 and 2 convincingly show AdaSGD attains competitive sub-optimality across problems. 

The paper is generally well-written and organized; the step-size rule is easy to implement, and the connection to the deterministic method is explained clearly in Sec. 2.

To sum up:
1. The paper has solid mathematical analysis with clear assumptions, statements and proofs.
2. The writing is generally clear with nice flow.
3. The work includes experimental work that compares the proposed method with other works.

### Weaknesses
The empirical scope is narrow and lacks statistical rigor: results are reported on a small suite of convex problems (with one larger dataset, w8a), but there are no error bars, multiple seeds, or variance measures in Figs. 1-3, making it hard to assess significance. Furthermore, baseline coverage is limited: while SGD (with and without decay) and AdaSGD-MM are included, widely used parameter-free/adaptive competitors (such as Adagrad/Adam/Polyak Stepsizes) are not evaluated in Sec. 4, so the practical advantage over common alternatives remains unclear. 

Furthermore, on the practicality side, the per-iteration cost is higher due to the second gradient evaluation, although the paper argues this is offset by avoiding grid search; there is no wall-clock comparison to prove that claim.

However, my main concern with this work is the significance/originality. As the paper clearly mentions, this work extends the deterministic results of a previous paper to the stochastic setting. I appreciate the techniques used in order to achieve that, but it is not clear to me what is the *significance* of this extension.

### Questions
See weaknesses.

To sum up:
1. Could you add empirical comparisons with a few parameter-free/adaptive baselines mentioned in the related work on the same convex tasks?
2. Can you include wall-clock/time-to-target comparisons that account for the second gradient evaluation per iteration and contrast them with typical hyperparameter grid searches for SGD?
3. Can you explain why the extension to the stochastic regime is important? Is it a theoretical benefit? Is it a practical benefit?

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
4

### Summary
The paper, proposed an efficient stochastic variant of the deterministic AdaGD of Malitsky and Mischchenko (2020). The proposed method does require any hyper-parameter tuning, and the papers provides two main theorems one on almost sure convergence (theorem 3.2) for three settings/cases, and a theorem in convergence in expectation for strongly convex component functions. Experiments also show the performance of the proposed method.

### Strengths
The paper is well written and the idea is easy to follow. The authors did a very nice job in the narrative of the paper. The explanation of their main idea, difference with previous approach, and the theoretical analysis is clearly presented. 

The results as far as i know, are the first of their kind as an analysis for the stochastic version of AdaGD of Malitsky and Mischchenko (2020) was never successfully proposed.

### Weaknesses
1. On Assumption 1: 
What exactly does Assumption 1 mean? This looks like an artificial condition to impose, just to make the proof work. Why is it not in the statements of the main theorems? How can one guarantee that this is satisfied? Can the authors provide more explanation on this, and some even constructed examples that this assumption is satisfied?

2. The authors claim the following in Figure 1: "Our methods (AdaSGD, the blue curves) always perform well with small values of $\lambda_0$ and thus do not require step-size tuning to compete with optimally-tuned SGD and ..." This statement, by simply observing Figure 1 is clearly not accurate. For example ,for $\lambda =1$ or $\lambda =10$ then the method does not converge. This makes the method precisely parameter-dependent. For claiming a tuning-free algorithm, one needs to have full independence in any parameter choice.

3. I would appreciate it if the labels were provided for any figure, not just Figure 2, which forces the reader to go back and forth between two figures to understand what is happening. 

4. On presentation: the paper devotes 1.5 pages to related work (not closely related to the paper) and then two full pages to the lemmas related to bounding specific quantities. I find this unnecessary, as I would like to have this space for extra numerical evaluation of the method and convergence analysis. The authors decided to move to Appendix A, the important information on what classes of problems they really solve, which, in my opinion, should have been in the main paper (instead of the lemmas on bounding quantities that can be moved to the appendix)

### Questions
Please see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
