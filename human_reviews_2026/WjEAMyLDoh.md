# Sharp asymptotic theory for Q-learning with \texttt{LD2Z} learning rate and its generalization

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
Despite the sustained popularity of Q-learning as a practical tool for policy determination, a majority of relevant theoretical literature deals with either constant ($\eta_t\equiv \eta$) or polynomially decaying ($\eta_t = \eta t^{-\alpha}$) learning schedules. However, it is well known the these choices suffer from either persistent bias or prohibitively slow convergence. In contrast, the recently proposed linear decay to zero (\texttt{LD2Z}: $\eta_t=\eta(1-t/n)$) schedule has shown appreciable empirical performance, but its theoretical and statistical properties remain largely unexplored, especially in the Q-learning setting. We address this gap in the literature by first considering a general class of power-law decay to zero (\texttt{PD2Z}-$\nu$: $\eta_t=\eta(1-t/n)^{\nu}$). Proceeding step-by-step, we present a sharp non-asymptotic error bound for Q-learning with \texttt{PD2Z}-$\nu$ schedule, which then is used to derive a central limit theory for a new \textit{tail} Polyak-Ruppert averaging estimator. Finally, we also provide a novel time-uniform Gaussian approximation (also known as \textit{strong invariance principle}) for the partial sum process of Q-learning iterates, which facilitates bootstrap-based inference. All our theoretical results are complemented by extensive numerical experiments. Beyond being new theoretical and statistical contributions to the Q-learning literature, our results definitively establish that \texttt{LD2Z} and in general \texttt{PD2Z}-$\nu$ achieve a best-of-both-worlds property: they inherit the rapid decay from initialization (characteristic of constant step-sizes) while retaining the asymptotic convergence guarantees (characteristic of polynomially decaying schedules). This dual advantage explains the empirical success of \texttt{LD2Z} while providing practical guidelines for inference through our results.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper develops a sharp asymptotic theory for Q-learning with the LZ2D step size schedule.

The authors establish several asymptotic results, including almost sure convergence, functional central limit theorems, and last-iterate convergence rates. 

The paper highlights that the proposed LZ2D step size achieves improved convergence behavior compared to classical polynomial or constant step-size schemes. 

Empirical results demonstrate the performance of LZ2D on synthetic Q-learning experiments, supporting its claimed theoretical benefits.

### Strengths
1. **Comprehensive theoretical contributions.**

    The paper provides a complete asymptotic characterization — including almost sure convergence, convergence rates for the last iterate, and uniform approximation (Section 4). This set of results is new for Q-learning with the particular step size and strengthens the theoretical foundation of this step size choice.

2. **Clear theoretical presentation.** 

    The main theorems are well organized and connected logically. The assumptions are explicitly stated, and the paper maintains consistency with the established stochastic approximation framework, which aids readability despite the technical depth.

3. **Empirical validation.** 

    Figures 1 and 3 compare different step-size rules and suggest that the proposed LZ2D schedule achieves faster or more stable convergence. The figures are clearly presented and help visualize the claimed benefits.

### Weaknesses
1. **LZ2D makes an online algorithm effectively offline.**

    The LZ2D step-size schedule $\eta_{t,n}$ depends explicitly on the total sample size $n$, making it unsuitable for online learning. This fundamentally alters the nature of Q-learning, which is originally designed for streaming data.
    * If new samples continue arriving after the prescribed $n$, it is unclear how to update $\eta_{t, n}$.
    * The paper should discuss whether asymptotic results remain valid if $n$ grows or is mis-specified.
    * This issue is critical for any claim of practical superiority. 

    From this perspective, LZ2D transforms Q-learning from an online algorithm to an offline one, which undermines its applicability in reinforcement learning.

2. **Trade-off between last-iterate convergence and asymptotic normality is missing.** 

    The paper claims both fast last-iterate convergence and asymptotic efficiency, but these are known to be mutually exclusive under standard stochastic approximation theory.
    * As shown in ROOT-SGD (Li, Mou, Wainwright, Jordan, 2020), the last iterate typically converges at rate $O(1/T)$, while asymptotic normality applies to the averaged iterate with $\sqrt{T}$ scaling.
    * In this paper, Theorem 3.5 shows that for PD2Z-$\nu$ step sizes, the scaling becomes $n^{\nu/(2(\nu+1))}$, slower than $\sqrt{n}$, except in the limit $\nu \to \infty$. This trade-off is not acknowledged. The current presentation gives the impression that LZ2D dominates all prior step-size rules, which is mathematically misleading.

       **Reference:** Li C J, Mou W, Wainwright M, et al. Root-sgd: Sharp nonasymptotics and asymptotic efficiency in a single algorithm[C]//Conference on Learning Theory. PMLR, 2022: 909-981.

3. **Experimental comparisons are incomplete and possibly biased.**

    * Figure 1 compares LZ2D with polynomial and constant step sizes but does not include linearly decaying step sizes ($\eta_t \propto 1/t$), which have theoretical guarantees for Q-learning (Li et al., Operations Research, 2024).
    * The paper also omits analysis of averaged iterates, which are standard when polynomial decay is used.
    * The hyperparameters (e.g., exponent $\alpha$ in polynomial decay) are not reported. Without tuning details, the apparent dominance of LZ2D may reflect unfair parameter choices rather than intrinsic superiority. Overall, the empirical claims in Figure 1 and 3 appear “too good to be true” without further clarification or ablation.

        **Reference:** Li G, Cai C, Chen Y, et al. Is Q-learning minimax optimal? a tight sample complexity analysis[J]. Operations Research, 2024, 72(1): 222-236.

4. **Unexplained discrepancy in Brownian motion approximation (Figure 3).**

     The right panel of Figure 3 shows a large performance gap between the LZ2D method and the Brownian motion benchmark, despite both satisfying the same functional CLT.
    * The paper does not explain why the LZ2D trajectory aligns so much better.
    * This may be due to slow convergence of remainder terms, but no quantitative discussion is given.
    * Additionally, there is an inconsistency: Line 455 defines the measurement as $\max_{k_n \le t \le n} |\sum_{l=t}^n (Q_l^c - Q^*)|{\infty}$, while Figure 3 uses $\max_{1 \le t \le n}$. Which definition is correct, and does LZ2D introduce the $k_n$ term while the Brownian benchmark does not? Clarifying this is important, since it affects the interpretation of the approximation quality.

### Questions
Please see the Weaknesses part. I will increase my point if the weaknesses are addressed well.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studied the theoretical properties of tabular synchronous Q-learning with linear decay to zero (LD2Z) and more general power-law decay to zero (PD2Z) step sizes.
They first provided non-asymptotic L^p error bound, which showcases that, comparing to polynomial decaying step sizes, the error bound of PD2Z could rapidly decay from initialization (an advantage of constant step sizes). Numerical experiments were provided to support the theoretical results. The paper also showed the asymptotic normality and the stronger time-uniform Gaussian approximation of the tail Polyak-Ruppert averaging of their algorithms.

### Strengths
The paper is well-written and clearly presents the significance of the research and its contributions. 
Specifically, the step size of LD2Z (or more general PD2Z) is more practical in real-world applications, such as in generative models like LLMs. However, theoretical analysis of Q-learning algorithms with this type of step size has remained underdeveloped, and this paper uses rigorous theoretical results to demonstrate that the LD2Z step size strategy can simultaneously retain the advantages of constant step sizes (fast initial convergence) and polynomial decay step sizes (asymptotic normality of the tail average).
Additionally, the paper provides a strong approximation result that is stronger than the functional CLT-like results reported in previous works.
Extensive numerical experiments further validate the theories proposed in this study.
Overall, this represents a very solid theoretical work.

### Weaknesses
1. Goldreich et al. (2025) have provided the L2 non-asymptotic error bounds and asymptotic properties of the LD2Z step size for Stochastic Gradient Descent (SGD) under the strong convexity condition. It is recommended that the authors clarify the differences and innovations in the proof techniques of this paper compared to previous works.

2. To my understanding, although the tail-averaged Q-learning with LD2Z step size exhibits asymptotic normality, its asymptotic variance is presumably not asymptotically optimal—in contrast, the asymptotic variance under polynomial decay step sizes is optimal. It would be beneficial if the authors could illustrate, either through theoretical analysis or experimental results, the discrepancy in asymptotic variance between the two step size strategies.

### Questions
see weakness

### Soundness
4

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
The paper develops sharp theory for Q‑learning under a power‑law decay‑to‑zero stepsize schedule. This work then establishes explicit non‑asymptotic error bounds, a central‑limit theorem for a Polyak–Ruppert averaging estimator, and a time‑uniform strong invariance principle. Experiments on a 4×4 FrozenLake‑style gridworld corroborate the theory. This work provides the first non‑asymptotic analysis of LD2Z for Q‑learning.

### Strengths
The main strength of this work lies in providing the first non asymptomatic analysis of Q-learning with PD2Z step-size schedules. The analysis is clear, sound, and explains why LD2Z achieves both fast early convergence and asymptotic stability.

### Weaknesses
The main weakness of this work lies in its empirical evaluation, while this work is mainly concerned with learning theory, tabular Q-learning is simply enough to run on any modern laptop (or colab). Ideally, using the publicly available code to generate figure 6.9 from the book by Sutton and Barto 2018 with your learning rate scheduler would be beneficial.

### Questions
comments 

1. line 054: li et al is missing a year.
2. line 241: aforementioned, decay followed by --> decay is followed by

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
The paper studies sample complexity of Q-learning under the power-law decay to zero (PD2Z) step-size including LD2Z. The authors prove a convergence rate result using the PD2Z step-size, which has a regime of fast decay and stabilizing phase. The authors also provide a result for tail Polyak Ruppert averages that converges to normal distribution of which the superioity over standard PR is suppported by experimental result. Lastly, an approximation result on the partial sum of the iterates is provided.

### Strengths
1. The paper studies the first result of Q-learning using PD2Z step-size. It inherits the fast convergence benefits of a constant step-size with the diminishing error properties of using a polynomial step-size.

2. The paper provides interesting properties not only convergence rate but also tail PR result and strong invariance property.

### Weaknesses
1. Importance of studying convergence rate in Q-learning : The literature of Q-learning focused on the sample complexity on $S,A$ or $\frac{1}{1-\gamma}$ dependency, which is lack of here.

2.. The discount factor $\gamma=0.1$ seems to be too small. It differs from usual choice of $\gamma=0.99$.

3.. Figure 4 seems to be an important result for tail PR case but it is deferred to Appendix.

4.. The authors study a result for Q-learning with PD2Z but the unique challenge in applying the particular step-size to Q-learning compared to optimization literature is not clear.

### Questions
1. The quantile transformaiton using the uniform random vairable $U$ is not clear. $R$ is defined as a function on $S\times A$ but the authors use $R(s,a,U)$.

2. Can we derive Assumption 3.2 from the definitions?

3. From Figure 1, the performance of constant step-size and LD2Z step-size does not seem to differ much.

### Soundness
2

### Presentation
2

### Contribution
2
