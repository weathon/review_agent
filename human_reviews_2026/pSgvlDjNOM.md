# Provable Anytime Ensemble Sampling Algorithms in Nonlinear Contextual Bandits

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
We provide a unified algorithmic framework for ensemble sampling in nonlinear contextual bandits and develop corresponding regret bounds for two most common nonlinear contextual bandit settings: Generalized Linear Ensemble Sampling (GLM-ES) for generalized linear bandits and Neural Ensemble Sampling (Neural-ES) for neural contextual bandits. Both methods maintain multiple estimators for the reward model parameters via maximum likelihood estimation on randomly perturbed data. We prove high-probability frequentist regret bounds of $\mathcal{O}(d^{3/2} \sqrt{T} + d^{9/2})$ for GLM-ES and $\mathcal{O}(\widetilde{d} \sqrt{T})$ for Neural-ES, where $d$ is the dimension of feature vectors, $\widetilde{d}$ is the effective dimension of a neural tangent kernel matrix and $T$ is the number of rounds. These regret bounds match the state-of-the-art results of randomized exploration algorithms in nonlinear contextual bandit settings. In the theoretical analysis, we introduce techniques that address challenges specific to nonlinear models. Practically, we remove fixed-time horizon assumptions by developing anytime versions of our algorithms, suitable when $T$ is unknown. Finally, we empirically evaluate GLM-ES, Neural-ES and their anytime variants, demonstrating strong performance. Overall, our results establish ensemble sampling as a provable and practical randomized exploration approach for nonlinear contextual bandits.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a unified ensemble sampling framework for nonlinear contextual bandits under two settings: GLM-ES for generalized linear bandits and Neural-ES for neural bandits. The authors provide regret analysis and empirical results on the proposals.

### Strengths
1. Clear problem setting, unified algorithm template (Alg. 1), and concrete GLM/Neural instantiations.

2. The anytime design: Doubling-trick conversion with constant-factor loss, addressing unknown horizon.

### Weaknesses
Some over-claims or lack of novelty. The algorithms adapt known ensemble-/perturbed-history ideas to GLM/neural settings; much of the value is synthesis + anytime conversion rather than a fundamentally new exploration principle. (Claims of “first” high-probability bounds are credible but incremental.)

### Questions
1. Anytime hyperparameters: For the doubling-trick variants, how sensitive are results to $T_0$ and base $b$? Any rule-of-thumb for choosing them beyond the theory constant?

2. Scalability with large $K$ or even continuous actions: Can the framework extend to large or continuous action sets (e.g., via approximate maximization or feature maps), and would the regret proofs still go through?

3. Warm-up guidance (GLM-ES): Beyond the theoretical construction, what’s a simple recipe (iterations, step sizes) that preserves optimism with high probability in practice?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an application of ensemble sampling (Lu & Van Roy, 2017) to GLM bandits and neural bandits.
For the bandit problems with a fixed arm set of size $K$, the proposed method uses $O(K \log T)$ estimators in the ensemble and achieves a $\tilde{O}(d^{3/2}\sqrt{T} + d^{4.5})$ regret bound for GLM bandits and a $\tilde{O}(\tilde{d}\sqrt{T})$ regret bound for neural bandits.

### Strengths
This is the first theoretical analysis of ensemble sampling beyond the linear contextual bandit setting.

### Weaknesses
### 1. Proof requires further clarification

For GLM-ES:

- In Section B.3, the analysis begins with $H_ t$ but ends with $\bar{H}_ t$. The switch happens in line 1280 without any justification. Until line 1280, the analysis requires properties of $H_ t$, and after line 1280, the analysis requires $\bar{H}_ t$, so it doesn't seem to be a simple typo.

- The analysis, especially in Appendix B.4, requires $\sigma_ R \ge b_ 1 + b_ 2 R_ t \sigma_ R$. Tracking the definitions, it seems like $b_ 1 = \tilde{\mathcal{O}}(d)$ and hence $\sigma_ R \gtrsim d$ must hold. However, line 1433 assumes $\sigma_ R \approx (d \log T)^{\frac{1}{2\varepsilon}}$ with $\varepsilon = \frac{2}{3}$, which implies $\sigma_ R \approx (d \log T)^{\frac{3}{4}}$.
The choice of $\varepsilon = \frac{2}{3}$ is also curious, since from Eq. in line 1436, $\varepsilon \rightarrow 1$ would yield a better bound on the warm-up time, although it is still invalid due to the reason above.

- For analyses that use stochastic optimism, one has to show that the conditional probability of optimism is bounded below, conditioned on the history up to the previous time step. However, this paper only shows the lower bound of the total probability. Especially in Appendix B.4, I think the relationship between the shown facts and the conditional probability of optimism should be explained more clearly.

- Lemma A.7 is not proved.

- In line 1055, while the previous equation proves $\ddot{b}(c) \le e \ddot{b}(u)$, it suddenly claims $\ddot{b}(c) \le 2 \ddot{b}(u)$.

- In line 1330, I don't see why $\mathbb{P}(U_ t^\top \mathbf{Z}_ t^j \ge (b_ 1 + b_ 2 R_ t \sigma) \lVert U_ t \rVert_ 2) - \frac{\delta}{T} \ge \frac{p_ N}{2}$ should be true when no relationship between $\delta, T$ and $p_ N$ is given.

For Neural-ES:

- I don't think the proof of Lemma C.4 would be mostly the same as Jia et al. (2022), since the perturbation is not sampled freshly as in Jia et al. (2022).

- The last steps of Appendix C require corrections. $\sum_ {t=K+1}^T \lVert g(X_ t; \theta_ 0) / \sqrt{N} \rVert_ {A_ t^{-1}}$ is not bounded by $\tilde{d}(\log (1 + TK / \lambda) + 1)$.
In the next step, the inequality suddnely jumps to $\tilde{O}(\tilde{d}\sqrt{T})$.
The relationship between $\alpha_ T, \beta_ T$ and $\tilde{d}$ is not clearly shown.

- Line 1617  would make sense only when there is a conditional expectation on the left-hand side with conditional probability on the right-hand side. Even under the expectation, only the lower bound on the total probability is provided and not on the conditional probability.

- Eq. in line 1680 is given without any justification.

- In the proof of Lemma C.5, Eq in line 1666 should present the upper bound of $h(X_ t)$ not the lower bound for the following logic that ensures optimism.

Due to these issues, I am not convinced that the theorems are true.

---

### 2. Slightly misleading presentation

While I acknowledge that some terms could be used loosely, I find the current title, abstract, and the introduction slightly misleading.

- "Contextual bandits" commonly refers to the setting where the reward of the arm, or the arm set itself, changes at every time step. For instance, "linear bandits" would refer to the fixed-arm-set case and "linear contextual bandits" would refer to the changing arm-set case (Lattimore and Szepesvari, 2020). In this work, the contextual setting is introduced as the changing arm-set setting with the arm set denoted as $\mathcal{X}_ t$ in Section 3 Problem Setting. However, it is suddenly replaced by $\mathcal{X}$ starting from line 255, where the G-optimal design is defined, then Theorems 5.5 and 5.7 also assume a fixed finite arm set $|\mathcal{X}| = K$. I don't think the paper should state that they consider the contextual setting. 

- One of the contributions of the paper is proposing an anytime version of ensemble sampling using the doubling trick. However, using the doubling trick for ensemble sampling is already discussed in Remark 3 of Janz et al. (2024a).

- While a lot of works use the term "nonlinear" for the setting with a general function class, this work is limited to GLMs and functions with a small NTK norm. I acknowledge that this point is clear in the abstract and the introduction, and the term could be used flexibly, so this is a minor concern compared to the previous ones.

---

### 3. Missing/misstated definitions

- $\mathcal{X}$ (without the subscript $t$) is not defined.

- $\mathcal{F}_ t$ is not defined within the paper.

- $\eta$ and $J$ are not defined in Lemma C.4. This is because the lemma is adapted from Jia et al. (2022) which trains the neural network with $J$ gradient descents of step size $\eta$, whereas this paper simply finds the minimizer of the loss function.

- The problem setting for GLM states that $Y_ t = \mu(X_ t^\top \theta^* ) + \eta$ with sub-Gaussian $\eta$, however the analysis seems to require that $Y_ t$ follows the natural exponential family distribution and not any sub-Gaussian noise, especially in Appendix A.1.

- At the end of Lemma C.4 (line 1539-1540), the definitions of $A_ K$ and $\lambda_ K$ are given, but are not used within the lemma. I think they should be at the end of Lemma C.5, or the defintion of $\sigma_ R$ in Lemma C.5 should be moved to Lemma C.4.

---

In addition, I see a lot of resemblance to Janz et al. (2024a), Lee & Oh (2024), and Jia et al. (2022) in the appendix, but in some parts, the authors do not mention these papers and proceed. I recommend the authors clearly state where the analysis originally came from, and cite proved results in these papers whenever possible, not just occasionally.

### Questions
Minor typos:

- Line 284 width L -> width N

- Lines 1266, 1275 are identical, when I think a different equation is intended for line 1275.

- In the proof for Neural-ES, I think $\theta_ 0$ should be replaced by $\theta_ 0^j$.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an ensemble sampling algorithm for K-armed contextual bandits with non-linear reward models.
Specifically, the authors develop a unified framework that covers both cases where the reward follows a generalized linear model (GLM) with respect to the arm’s context and where the reward function lies in a reproducing kernel Hilbert space (RKHS) with a bounded norm under the neural tangent kernel.
The proposed algorithm achieves regret bounds of $O(d^{3/2} \sqrt{T})$ for GLM rewards, and $O(\tilde{d} \sqrt{T})$ for RKHS-bounded rewards, given an ensemble of size $\Omega(K \log T)$.
Moreover, the paper addresses the case where the total number of rounds $T$ is unknown, employing a doubling trick to establish asymptotic regret guarantees.
The superiority of the proposed algorithm is validated through synthetic experiments.

### Strengths
1. Ensemble sampling has shown strong empirical performance in practice but has primarily been analyzed under linear reward models. This work extends the analysis to more general settings, including GLM and RKHS-bounded reward models, and demonstrates that the proposed algorithm achieves regret bounds matching those of existing linear or randomized algorithms.

2. The paper is clearly written and easy to follow. The literature review and discussion of related work are well-organized and help position the contribution effectively.

### Weaknesses
1. The notation throughout the paper should be carefully reviewed. For example, 
- context feature is denoted by both $x$ and $X$ in different places
- $H_t$ is defined as the Hessian of the loss function at time $t$, yet the term $ \|| X_t \|_{H_t^{-1}}$ is used later as a bonus term, which may lead to confusion
- Does this work allow the context to change over time? The main theorem (Thm. 5.5 & 5.7) assumes a fixed arm set ($|\mathcal{X}| \le K$, but the problem setting (line 157) uses a varying arm set.
- Definition of $\theta_t$ in appendix seems missing. 
- By definition, $\theta^\{j\_t}\_t$ is an estimator estimated using data up to time $t$. Then, shouldn't the behavior of $X_t^\top \theta^{j\_t}\_{t-1}$ be analyzed in optimism?

2. The description of the warm-up phase could be expanded. In Remark 4.2, the authors note that while previous GLM-bandit studies use warm-up rounds to ensure the invertibility of the empirical Gram matrix, the proposed method employs them to guarantee constant-probability optimism. Since the purpose differs, it would be helpful to elaborate on how the rounding procedure in Algorithm 3 operates—for example, how it differs from exploring for $\tau$ rounds using the G-optimal design solution $\zeta$ as the policy.

3. Although this issue is not unique to this paper, the neural network size required for Neural-ES appears to be very large (see the condition in Lemma C.4), which may limit the practical applicability of the algorithm.

### Questions
1. In Remark 5.4, the authors state that the expected Gram matrix is not necessarily assumed to have a positive minimum eigenvalue. However, under this case, how is the existence of the optimal design distribution (\zeta) guaranteed?

2. Neural-ES employs a different warm-up procedure from GLM-ES. What allows Neural-ES—despite dealing with a more general reward model—to avoid the complex warm-up required in the GLM-ES setting?

3. How does the MLE concentration result in Lemma A.3 differ from Lemma A.2 in Sawarni et al. (2024)?

4. The doubling trick (Besson & Kaufmann, 2018) is a well-established approach in online learning to handle unknown $T$, with known regret bounds. How does the result in this paper differ from the standard doubling-trick analyses already available in the literature?

5. In line 1244t, the sufficient condition for optimism is written by $H_t$, but it was changed to $\bar{H}_t$ in Eq. (B.8). Could you explain the process in a little detail?

### Soundness
3

### Presentation
2

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
This paper introduces Provable Anytime Ensemble Sampling, a unified framework for ensemble sampling in nonlinear contextual bandits with two concrete instantiations: Generalized Linear Model Ensemble Sampling (GLM-ES) and Neural Ensemble Sampling (Neural-ES), and extends both to the anytime setting via a doubling-trick schedule. The core mechanism is randomized exploration through reward perturbation while reusing past perturbations so that per-round computation remains constant. Under a fixed and finite action set with bounded features, the authors derive high-probability regret guarantees for GLM-ES (under M-self-concordance and a positive lower bound on the link derivative over the working domain) and for Neural-ES (under neural tangent kernel conditions for sufficiently wide networks). The anytime variants achieve the same asymptotic regret up to a constant factor, and experiments on linear, logistic, distance, and quadratic synthetic environments show competitive cumulative regret with noticeably reduced computational cost and no need to know the time horizon in advance.

### Strengths
The framework unifies randomized exploration across linear and nonlinear bandits with a single, conceptually clean recipe that reuses historical perturbations to keep updates inexpensive. The theory is clearly presented and provides high-probability guarantees, together with an anytime extension that is simple and practical. The GLM instantiation reduces the warm-up complexity relative to prior perturb-and-estimate methods, and the paper is generally well written and organized.

### Weaknesses
The empirical setup lists hyper-parameter choices and network sizes but gives little justification for why these values were selected, and it lacks systematic sensitivity/ablation results for ensemble size, regularization, perturbation variance, or the anytime schedule. Neural experiments are limited to a small multilayer perceptron, so it remains unclear whether the computational and statistical behavior persists for wider/deeper architectures used in practice. 

Several modeling assumptions that are central to the guarantees (fixed finite action set, bounded features, GLM regularity, and wide-network/NTK-style training for Neural-ES) are not emphasized in the discussion of applicability. The paper would also benefit from clarifying the data-generation details in the synthetic environments (e.g., how the unknown parameter and action features are sampled) and from commenting on the ensemble size scaling with the number of actions, which may be large in real applications.

### Questions
1.Could the authors provide a more systematic sensitivity and ablation analysis covering ensemble size, regularization, perturbation variance, and the anytime schedule (e.g., base segment length and growth factor), including both regret and runtime/memory effects?
2.For Neural-ES, do the computational and theoretical advantages extend beyond small MLPs to wider/deeper networks commonly used in practice (e.g., larger MLPs or transformer backbones), and what concrete width/optimization regimes are required for the stated guarantees to remain valid?

The evaluation uses synthetic environments, which is standard for this area but leaves open how the approach behaves in more complex or real-world contexts (e.g., with changing action sets, larger feature spaces, or modern neural architectures).

### Soundness
3

### Presentation
3

### Contribution
2
