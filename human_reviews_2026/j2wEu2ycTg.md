# Poisson Midpoint Method for Log Concave Sampling: Beyond the Strong Error Lower Bounds

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
We study the problem of sampling from strongly log-concave distributions over $\mathbb{R}^d$ using the Poisson midpoint discretization (a variant of the randomized midpoint method) for overdamped/underdamped Langevin dynamics. We prove its convergence in the 2-Wasserstein distance ($\mathcal W_2$), achieving a cubic speedup in dependence on the target accuracy ($\epsilon$) over the Euler-Maruyama discretization, surpassing existing bounds for randomized midpoint methods. Notably, in the case of underdamped Langevin dynamics, we demonstrate the complexity of $\mathcal W_2$ convergence is much smaller than the complexity lower bounds for convergence in $L^2$ strong error established in the literature.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the Wasserstein-2 convergence of both the overdamped and underdamped Poisson Randomized Midpoint Methods (PLMC) for sampling from *strongly log-concave* distributions, with a focus on the dependencies of dimension ($d$), accuracy ($\varepsilon$) and condition number ($\kappa$) in the oracle complexity. 

The analysis builds on the intuition that one step PLMC of size $\eta$ *approximately implement* $k$ step of LMC with size $\eta/k$. Accordingly, the convergence proof of PLMC is devided into two parts: convergence analysis of LMC and **trajectory-gap quantification between LMC and PLMC**.  The main theoretical novelty lies in the latter, where the authors construct a refined coupling between the Brownian motions driving the two dynamics, enabling a tight control of their deviation. Combining this with existing convergence results for overdamped LMC ([1]) and underdamped LMC ([2]), sharper bounds in terms of the accuracy dependency are proved: $\mathcal{O}(\varepsilon^{-2/3})$ for overdamped PLMC and $\mathcal{O}(\varepsilon^{-1/3})$ for underdamped PLMC.  

[1] Alain Durmus, Szymon Majewski, and Bła˙ zej Miasojedow. Analysis of Langevin Monte Carlo via
Convex Optimization. The Journal of Machine Learning Research, 20(1):2666–2711, 2019.

[2] Arnak Dalalyan and Lionel Riou-Durand. On sampling from a log-concave density using kinetic
Langevin diffusions. Bernoulli, 26(3), 2020.

### Strengths
1. The theoretical derivation in the paper is solid. 

2. The complexity bound beats all existing bounds in terms of accuracy dependency.

3. The improved bounds also help to understand an misunderstanding on the information lower bound derived in [1], suggesting that better bounds than $\mathcal{O}(\varepsilon^{-2/3})$ can be achieved if the desired accuracy metric is $W_2$/ KL-divergence.

### Weaknesses
The presentation of the paper require improvement. As far as I am concerned, more discussion on 

(1) good performance of PLMC compared to LMC; 

(2) intuition behind of the constructed coupling, along with why it implies better complexity; 

(3) whether the current derived bound is sharp;

should be added.

### Questions
1. Typos:

(1) line 85-86: at the end of the line, *for underdamped LMC* should be *for underdamped PLMC*;

(2) line 194-197: the summation should be from $j=0$ to $j=i-1$.

2. The proofs can be organized in a better way. For example, both the high-level idea for convergences of overdamped and underdamped PLMC should be talked about. Currently, the part related to underdamped PLMC is not very informative.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper studies sampling from strongly log-concave distributions. It establishes that the Poisson midpoint discretization of both overdamped and underdamped Langevin dynamics achieves an accelerated convergence rate under the  W2 distance, significantly improving upon existing bounds.

### Strengths
he paper is very well-written. It provides a clear description of the problem, a thorough comparison of its theoretical results with existing works, and insightful proof techniques that illustrate the reasons behind the achieved convergence rates.

### Weaknesses
It is recommended to exchange the positions of Sections 2.2 and 2.3 for a more logical flow. Additionally, the origin of the Poisson midpoint method should be clarified: is it a novel proposal of this work, or is it adopted from prior literature?

### Questions
1. To better highlight the novelty of the convergence rate, could the authors provide a more detailed explanation of the limitations in prior work that prevented them from achieving the same rate?

2. The role of the parameter p in the analysis should be clarified. Furthermore, does the method maintain its performance advantage for ill-conditioned problems where the condition number kapa is very large?

3. Could the authors include numerical experiments to empirically validate the theoretical findings?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper analyzes the Poisson midpoint method (proposed in an earlier work) in the strongly convex setting. It obtains rates which scale as $\varepsilon^{-1/3}$ with respect to the desired inverse accuracy, which bypasses conjectured lower bounds. It does so using a Wasserstein central limit theorem.

### Strengths
The paper obtains surprising low accuracy sampling guarantees with oracle complexity scaling as $\varepsilon^{-1/3}$. This defeats a conceptual lower bound.

The paper obtains these new rates in a conceptually orthogonal way to any prior analysis, which potentially opens the doors to successive works adapting these techniques to other settings.

I do not want to overelaborate, but I think these are clear and very salient contributions, and I believe the techniques in this paper deserve more exposure.

### Weaknesses
In principle, improved rates for low accuracy algorithms in $\varepsilon$ are not useful if they come at the expense of dimension dependence, due to the existence of high accuracy algorithms. In particular, if we take $\varepsilon = \tilde{\varepsilon}/d^{1/2}$ here, then we observe worse dimension dependence $d^{3/8}$ in the second term (as $p \to \infty$). However, I do not see this as a severe issue.

This paper explores only the simplest principled setting; many settings still fall outside the purview of this paper, for instance the LSI setting or the low friction setting.

### Questions
The comparison with Altschuler et al. (2025) is made twice, which seems redundant.

Can this be adapted to SDEs without convergent drifts? This may be helpful to numerical analysts.

38: Ito -> It\^o

41-42 is not a complete sentence.

The guarantee $W_2^2 \lesssim \frac{\varepsilon^2 d}{\alpha}$ is non-standard in terms of the quoted rates. It would be preferable if one absorbed this $d$ into the factor $\varepsilon^2$, which is always possible. Indeed, this is done in Table 1 and it is a bit puzzling why this is not done elsewhere.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper studies the Poisson Midpoint Method (PLMC) for overdamped and underdamped Langevin dynamics on strongly log‑concave targets. It proves Wasserstein‑2 convergence with improved oracle complexities: for overdamped (Cor. 1), and for underdamped (Cor. 2),. The analysis combines a tight W2 bound for Gaussian + 1-D perturbation (Lemma 1, adapted from Zhai) with contractive couplings. Tables 1–2 (p. 6) compare against LMC/RLMC and emphasize a cubic speedup in (ε) over Euler-Maruyama. The paper also clarifies the distinction between strong (L2) error lower bounds and weak W2 guarantees.

### Strengths
Important question, clear positioning: The introduction and §1.1 make a precise case that strong (L2) lower bounds for ULD do not preclude faster W2 rates, and the results indeed obtain $\tilde O(ε^{-1/3})$ for the underdamped case (Theorem 2, Cor. 2). 

Clear presentation on technical novelty and algorithmic efficiency.

### Weaknesses
Typo: In Eq (2), the coefficient on Brownian term should be $\sqrt{2\gamma}d$, in order to achieve right invariant distribution.

No empirical study: The paper is purely theoretical. There are no experiments illustrating constants, stability, or the practical effect of hyper-parameters.

### Questions
No.

### Soundness
4

### Presentation
3

### Contribution
3
