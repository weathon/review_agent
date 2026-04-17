# Derivative-Free Optimization via Monotonic Stochastic Search

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 2

## Abstract
We consider the problem of minimizing a differentiable function $f:\mathbb{R}^d \to \mathbb{R}$ using only function evaluations, in the zeroth-order (derivative-free) setting. We propose three related monotone stochastic algorithms: the \emph{Monotonic Stochastic Search} (MSS), persistent Monotonic Stochastic Search (pMSS), and MSS variant with gradient-approximation (MSSGA). MSS is a minimal stochastic direct-search method that samples a single Gaussian direction per iteration and performs an improve-or-stay update based on a single perturbation. For smooth non-convex objectives, we prove an averaged gradient-norm rate $\mathcal{O}(\sqrt{d}/\sqrt{T})$ in expectation, so that $\mathcal{O}(d/\varepsilon^2)$ function evaluations suffice to reach $\mathbb{E}||\nabla f(\theta^t)||_2 \le \varepsilon$, improving the quadratic dependence on $d$ of deterministic direct search while matching the best known stochastic bounds. In addition, we propose a practical variant, pMSS, that reuses successful search directions with sufficient decrease, and establish that it guarantees $\liminf{t\to\infty}||\nabla f(\theta^t)||_2 = 0$ almost surely. Since MSS relies solely on pairwise comparisons between $f(\theta^t)$ and $f(\theta^t+\alpha_t s_t)$, it falls within the class of optimization algorithms that assume access to an exact ranking oracle. We then generalize this framework to a stochastic ranking-oracle setting satisfying a local power-type margin condition, and demonstrate that a majority vote over $N$ noisy comparisons preserves the $\mathcal{O}(d/\varepsilon^2)$ gradient complexity in terms of iteration count, given suitably designed oracle queries. MSSGA uses finite-difference directional derivatives while enforcing monotonic descent. In the smooth non-convex regime, we show that the best gradient iterate converges almost surely at a rate of $o(1/\sqrt{T})$ almost surely. To the best of our knowledge, this result provides the first $o(1/\sqrt{T})$ almost-sure convergence guarantee for gradient-approximation methods employing random directions. Furthermore, our analysis extends to the classical Random Gradient-Free (RGF) algorithm, establishing the same almost-sure convergence rate, which has not been previously shown for RGF. Finally, we show that MSS remains robust beyond the smooth setting: when $f$ is continuously differentiable, the iterates satisfy $\liminf{t\to\infty}||\nabla f(\theta^t)||_2=0$ almost surely.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposed two algorithms for zeroth-order optimization: the Monotonic Stochastic Search (MSS) algorithm and its gradient-approximation variant (MSSGA), and established their convergence properties for non-convex, convex, and strongly convex settings.

### Strengths
The article is relatively well-written, with appropriate discussions and citations of relevant work.

### Weaknesses
1. The upper complexity bounds achieved in the article are all known, and although a slightly different algorithm is used, this does not constitute sufficient novelty for the article to be accepted by ICLR. For example, in lines 107-109, the authors state, “The key difference, however, is that our algorithm enforces monotonic improvement by rejecting any update that does not lead to a smaller value of the objective function.” However, I do not believe this is an innovative point; it is simply a straightforward approach. Furthermore, for the stochastic setting (when the returned gradient oracle has noise), I am uncertain whether such a strategy remains viable.

2. The writing of the article is poor. For instance, in Section 1, the "Our Contribution & Related Work" section is overly lengthy and lacks emphasis, spanning two pages yet making it difficult to identify the core contributions of the paper and how they differ from previous work. As a standard for a qualified paper, I believe this paragraph needs to be completely rewritten.

3. There are no experiments presented, and I doubt the practical value of the algorithms proposed in the article.

### Questions
Please see the weakness part.

### Soundness
1

### Presentation
1

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
This paper proposes a new class of **monotonic stochastic search (MSS)** algorithms for **derivative-free optimization (DFO)**.
Unlike classical random search or evolutionary strategies, MSS imposes a *monotonic descent constraint* on noisy function evaluations, thereby improving stability under stochastic perturbations.

The authors analyze three major settings:

1. **Smooth nonconvex functions** — MSS achieves sublinear convergence in expectation and almost surely, with
   [
   \mathbb{E}|\nabla f(x_T)| = O(\sqrt{d}/\sqrt{T}),
   ]
   without assuming convexity or PL-type conditions.

2. **Convex functions** — The algorithm guarantees function value convergence
   [
   \mathbb{E}[f(x_T)] - f^* = O(d/T).
   ]

3. **Strongly convex functions** — A faster geometric rate is achieved,
   [
   \mathbb{E}[f(x_T)] - f^* = O!\big((1 - \mu/(dL))^T\big).
   ]
   Here, the PL inequality is used only as a consequence of strong convexity, not as an independent assumption.

Overall, the paper provides a unifying stochastic framework that recovers known DFO rates while improving robustness to noise.

### Strengths
1. **Comprehensive Theoretical Coverage**
   The paper systematically treats nonconvex, convex, and strongly convex regimes in a unified manner, providing clear asymptotic rates for each case.
   The inclusion of the **nonconvex L-smooth case without PL assumptions** is particularly commendable.

2. **Novel Monotonicity Principle**
   The “monotonic stochastic search” idea—using noisy evaluations to enforce descent direction without explicit gradients—is both conceptually simple and practically valuable.
   It bridges classical stochastic approximation and derivative-free optimization.

3. **Mathematical Rigor**
   Proofs are clean and self-contained.
   The paper references classical results (Nesterov, 2013; Ghadimi & Lan, 2016) appropriately while extending them to stochastic zeroth-order settings.

4. **Clarity of Structure**
   Each assumption and theorem is clearly labeled and motivated. The algorithmic structure is easy to follow.
   The division of results (nonconvex / convex / strongly convex) is pedagogically clear.

5. **Relevance and Generality**
   DFO remains a vibrant area for large-scale simulation-based learning and black-box optimization.
   This work offers a theoretically grounded yet computationally feasible method.

### Weaknesses
1. **Limited Empirical Validation**
   The experiments are minimal, mainly synthetic quadratic functions and low-dimensional benchmarks.
   Demonstrations on higher-dimensional or noisy black-box tasks (e.g., reinforcement learning, hyperparameter tuning) would strengthen the impact.

2. **Mild Novelty in Algorithmic Design**
   While the monotonicity mechanism is interesting, it resembles prior stochastic line search or acceptance–rejection DFO strategies.
   The novelty is thus more in the **analysis** than in the **algorithm itself**.

3. **Dependence on Smoothness Constants**
   The theoretical guarantees assume global L-smoothness and bounded variance of the function evaluations—standard but relatively strong assumptions for DFO.

4. **No Adaptive Mechanism for Query Efficiency**
   The paper could discuss how to reduce the dependence on the dimension (d), since rates scale as (O(\sqrt{d})) or (O(d)), which is suboptimal for high-dimensional problems.

5. **Strongly Convex Analysis Relies on PL-type Result**
   Although acceptable as a corollary of strong convexity, the use of the PL inequality should be more clearly separated as a *derived property*, not an assumption.

### Questions
1. Can the monotonic stochastic search idea be combined with adaptive sampling (e.g., covariance adaptation or coordinate selection)?
2. How robust is MSS to biased noise or nonstationary stochasticity in function evaluations?
3. Would it be possible to extend the analysis to nonsmooth (but Lipschitz) objectives?
4. Can the dependence on (d) be improved via random subspace or low-rank approximation techniques?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposed two stochastic zeroth-order optimization algorithms for smooth/nonsmooth optimization, MSS and MSSGA, which are based on DDS and gradient approximation. Convergence rates under nonconvex, convex and strongly convex scenarios are provided. Also asymptotic convergence result in the non-Lipschitz smooth case is provided.

### Strengths
1. The proposed algorithms are very simple, which should be easy to implement in practice.
2. The propsoed algorithms achieved good convergence guarantees and matched existing best results.

### Weaknesses
1. While the proposed MSS/MSSGA algorithms are elegant and minimalistic, the proposed algorithms' complexities do not outperform existing ones, it lacks a discussion on the motivation of the study.
2. There lacks a thorough theoretical/empirical comparison on the proposed algorithms with closely related works, for example STP and GLD as authors mentioned. It is not clear what is the advantage of the proposed algorithms.
3. The writing is a bit sloppy, for example, the "Our Contribution & Related Work" part is very lengthy and full of notations, which is hard to follow and identify the detailed contributions, I suggest a revision.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies derivative-free optimization where only function evaluations are available. The authors propose two algorithms: Monotonic Stochastic Search (MSS) and MSS with Gradient Approximation (MSSGA). At each iteration, MSS samples a single random direction s_t from a distribution D and moves to the point that minimizes f among $\theta_t, \theta_t +\alpha_t s_t$ where $\alpha_t$ is a step size. MSSGA additionally uses finite differences to approximate the directional derivative. The main results show that MSS requires $d/\epsilon^2$ samples for non-convex and smooth problems (Thm. 2), MSSGA uses $\frac{d}{\epsilon}$ for smooth and convex optimization (Theorem 5), and $d \log \frac{1}{\epsilon}$ for strongly convex objectives (Theorem 6). The paper shows a convergence result for potentially nonsmooth (but still differentiable) objectives in Thm. 7.

### Strengths
1. MSS uses only one new function evaluation per iteration and enforces monotonicity, this is an advantage over competitor algorithms (e.g. Stochastic Three-Point method).
2. The proofs are clear and mirror GD-style analysis, e.g. Lemma 1 gives GD-like expected decrease that straightforwardly leads to the $\sqrt{d}/\sqrt{T}$ bound.
3. The paper provides an almost surely o(1/\sqrt{T}) rate for the best iterate in the smooth non-convex case (Thm. 4) for MSSGA provided the smoothing sequence $\gamma$ decays appropriately, and a similar result is shown for MSS in Remark 2.

### Weaknesses
1. While using only one function evaluation instead of two is attractive, this is (a) a constant improvement, and (b) seems to actually show up in the convergence analysis. Comparing your Lemma 1 against Lemma 3.5 from [1], both have the same form of linear progress in (\alpha|\nabla f|) minus a quadratic penalty. STP works with normalized directions (i.e. $\mathbb{E}|s|^2=1$), this corresponds to putting $\mu_D=\sqrt{2/(\pi d)}$ in their lemma. If we rescale your Gaussian ($s_t\sim\mathcal{N}(0,I)$) to that normalization (i.e., divide by $\sqrt{d}$) and matches the stepsizes, your linear‑term constant becomes (1/\sqrt{2\pi d}), i.e., a factor of 1/2 smaller than STP due to using only one side instead of ($\pm s$). Since STP uses two evaluations per iteration and MSS uses one, the per‑function‑evaluation constants essentially tie. In other words, if we accept the convergence analysis in both papers, then the one function evaluation of MSS is cancelled out by having to do more iterations overall. If you include some experimental comparison, or improve the analysis, then you could still show an advantage of MSS over STP.
2. I am not 100% sure what novelty is really claimed here, especially in the almost sure convergence results. Or in the proofs. Can you please make that more clear? The proof of MSS is very similar to the proof of STP in [1]. Also, the contributions section is currently rather difficult to read and very long, if you could shorten it to bullet points to better quantify what separates your work from prior work that'd be great.
3. While most proofs are clear, some of the notation is a bit difficult to parse (like $A_{\theta^t}^{--}, A_{\theta^t}^{++}, A_{\theta^t}^0$) maybe name these sets differently instead of using this many sub/superscripts?

As it stands, I lean towards rejecting this manuscript, but am open to changing my mind if my concerns are addressed.

[1] Bergou, E. H., Gorbunov, E., & Richtarik, P. (2020). Stochastic three points method for unconstrained smooth minimization. SIAM Journal on Optimization, 30(4), 2726-2749.

### Questions
1. Can you please address my concerns in the weaknesses section? In particular, a comparison with STP that takes into account *total complexity* rather than just per-step complexity while matching the distribution of noise used.
2. Can you clarify if there are new technical tools used for MSS compared to prior work?

### Soundness
3

### Presentation
2

### Contribution
2
