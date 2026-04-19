# Ito Diffusion Approximation of Universal Ito Chains for Sampling, Optimization and Boosting

- Decision: Accept (poster)
- Scores: 3, 6, 8

## Abstract
In this work, we consider rather general and broad class of Markov chains, Ito chains, that look like Euler-Maryama discretization of some Stochastic Differential Equation. The chain we study is a unified framework for theoretical analysis. It comes with almost arbitrary isotropic and state-dependent  noise instead of normal and state-independent one as in most related papers. Moreover, in our chain the drift and diffusion coefficient can be inexact in order to cover wide range of applications as Stochastic Gradient Langevin Dynamics, sampling, Stochastic Gradient Descent or Stochastic Gradient Boosting. We prove the bound in $\mathcal{W}_{2}$-distance between the laws of our Ito chain and corresponding differential equation. These results improve or cover most of the known estimates. And for some particular cases, our analysis is the first.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work considers a rather general and broad class of Markov chains,
Ito chains that look like Euler-Maryama discretization of some Stochastic
Differential Equation. However, I am not fully convinced of the importance of this work.

### Strengths
This paper studies a rather general and broad class of Markov chains and studies the convergence rates.

### Weaknesses
1. The theoretical results do not provide any new insight to the community.
2. The analysis is based on standard techniques.
3. The paper lacks a empirical study to verify the theoretical findings.

### Questions
None

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies a broad class of Markov chains called Ito chains, which allows for almost arbitrary isotropic and state-dependent noise instead of normal and state-independent one. The paper analyzes the coupling between Ito chains and its anagolous SDEs, and provides an upper bound for $\mathcal{W}_2$ distance between laws of the two. Their results cover many of the known estimates. In particular, some techniques such as Window Coupling and an augmented version of Girsanov's Theorem are of independent interest.

### Strengths
Originality: The originality is high. The authors study a quite general class of Markov chains and analyze the upper bound between SDE, which is also a popular research area.
Clarity: the clarity is good. The paper sequentially introduces several auxiliary SDEs/Markov chains to eventually bridge the gap between the processes of interest: $X_k$ and $Z_t$. The motivation of each proposed auxiliary processes are clearly stated and is easy to follow for readers.
Significance: The theoretical contribution is quite good for this area. The authors consider non-Gaussian and state-dependent chains which is not so deeply investigated in literature. The window coupling and "modified" Girsanov theorem may be of independent interest.

### Weaknesses
There are some mistakes and typos in proofs:
 - In Appendix, the equations in the middle of page.18 (the one after (10)), a noise matrix $\sigma(Y_{\bar{\eta}k}^S)$ before $\zeta_k^S(X_{Sk})$ is missing. The same issue appears in most of the later equations in this proof.
 - In the same equation, the last two terms have a wrong scaling w.r.t. $S$. Specifically, they should be $4\mathbb{E}\left[\left\|\frac{1}{\sqrt{S}}\sum_{i=1}^{S-1} \sigma(X_{Sk}) (\epsilon_{Sk+i} - \frac{1}{\sqrt{S}}\zeta_k^S(X_{Sk})) \right\|^2\right]$ and $4\mathbb{E}\left[\left\| 
\sum_{i=0}^{S-1}(\sigma(X_{Sk}) - \sigma(Y_{\bar{\eta}k}^X)) \zeta_k^S(X_{Sk}) \right\|^2\right]$. Consequently, some following terms should be multiplied by $S^2$ or so. (It seems not to affect the final upper bound, though. It is better to still check the whole roadmap of proofs.)
 - I wonder how the last term in the top equation on page.19 is derived ($4M_0\|X_{Sk}-Y_{\bar{\eta}k}^X\|^2$). It seems merely using Assumption 1 (4) is not enough. Or maybe there should be some additional assumptions?

### Questions
The window coupling seems to be among the crucial parts in this paper. Could authors explain if this is first proposed by this paper, or already appears in literature? It is best to provide some references.

The paper gives better bounds for SGD with Gaussian/non-Gaussian cases [1]. Could authors compare the theoretical techniques used in two papers and explain where do such improvements probably come from?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper considers a general class of Markov chains called Ito chains that cover a wide range of applications including sampling, optimization, and boosting. The paper provides bounds for the approximation error in W_2 distance between such Ito chains and the corresponding stochastic differential equation (which can then be used to study the Markov chain). In several applications, the bounds improve upon previous results or are completely new.

### Strengths
The results in this paper provide novel and general bounds on the approximation error for Ito chains using the corresponding stochastic differential equation. The results cover a broad range of applications in sampling, optimization, and boosting (for example, SGLD, SGD, and Stochastic Gradient Boosting) and are novel in several such applications (for example, the results cover almost arbitrary isotropic and state-dependent noise). The proof involves several new ideas and the presentation is quite clear.

### Weaknesses
It would be nice if the author(s) could comment on whether Assumption 1 in Section 2 is typically satisfied/easy to verify in applications.

### Questions
It would be nice if the author(s) could comment on whether Assumption 1 in Section 2 is typically satisfied/easy to verify in applications.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
