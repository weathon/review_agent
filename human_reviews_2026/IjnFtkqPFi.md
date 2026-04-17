# Efficient Simple Regret Algorithms for Stochastic Contextual Bandits

- Decision: Reject
- Scores: 6, 8, 6, 4

## Abstract
We study stochastic contextual logistic bandits under the simple regret objective. While simple regret guarantees are known for the linear case, no such results existed for the logistic setting. Building on ideas from contextual linear bandits and self-concordant analysis, we propose the first algorithm that achieves simple regret $\tilde{\mathcal{O}}(d/\sqrt{T})$. Notably, the leading term of our regret bound is free of $\kappa=\mathcal O(\exp(S))$, where $S$ is a bound on the magnitude of the unknown parameter vector, while the algorithm remains computationally tractable for finite action sets. We also introduce a new variant of Thompson Sampling adapted to the simple-regret setting, which yields the first simple regret guarantee for randomized algorithms in stochastic contextual linear bandits. Extending these tools to the logistic case, we obtain a Thompson Sampling variant with regret $\tilde{\mathcal O}(d^{3/2}/\sqrt{T})$, again free of $\kappa$ in the leading term. The randomized algorithms, as expected, are cheaper to run than their deterministic
counterparts. Finally, we conducted a series of experiments to empirically validate these theoretical guarantees.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper considers stochastic contextual bandits under the simple regret objective, studying both the linear and logistic models. The simple regret framework focuses on learning a policy that performs well on future data after a certain exploration horizon, which is different from minimizing cumulative regret during learning.
For the linear model, the paper first presents a deterministic algorithm called MULIN. The algorithm selects actions which maximize predictive uncertainty and achieves a simple regret bound of
$\tilde{O}(d/\sqrt{T})$ where $d$ is the dimension of the feature space and $T$ the number of rounds. They also propose a randomized Thompson Sampling variant which obtains a bound of
$\tilde{O}(d^{3/2} /\sqrt{T})$, matching the dependence on dimension known from cumulative regret settings.

For the logistic model, which introduces nonlinearity through a sigmoid link function, the paper proposes two analogous algorithms. The deterministic version, MULOG, extends the uncertainty-based exploration approach by leveraging self-concordant analysis of the logistic loss. It achieves a simple regret bound of $\tilde{O}(d/\sqrt{T})$ with the leading term free of the exponential constant $\kappa$. The randomized counterpart, THATS, attains a similar bound of
$\tilde{O}(d^{3/2} /\sqrt{T})$ with lower computational cost.

These results are claimed to be the first theoretical guarantees for simple regret minimization in stochastic logistic contextual bandits. The paper develops new analytical techniques to handle randomization and nonlinearity, including a decreasing-uncertainty lemma for the logistic case. Empirical studies on synthetic data corroborate the theoretical findings, showing that the proposed algorithms outperform selected baselines.

### Strengths
**Originality**:
This paper introduces the 1st simple-regret analysis for logistic contextual bandits, a nontrivial extension from the linear bandit setting. It develops new analysis tools for randomized algorithms under simple regret, distinct from cumulative regret frameworks. It corrects a known technical mistake in prior work (Zanette et al., 2021) and extends the self-concordant logistic analysis of Faury et al. (2020).

**Quality**:
Theoretical results are mostly solid, with good proofs, detailed derivations, and clear dependence on problem parameters like $T$ and $d$.
The regret bounds seem tight and match known minimax orders in the linear case.
Empirical evaluations, though synthetic, corroborate the theoretical claims and illustrate performance differences between algorithms.

**Clarity**:
The paper has a good clarity. Pseudocode is provided for all four main algorithms.


**Significance**:

This work establishes a foundation for simple regret optimization in nonlinear contextual bandits. This seems an important step toward more challenging generalized linear models.

### Weaknesses
There is no major technical flaw detected. But I do have a few concerns. 

(1)
The alignment argument used for Thompson sampling assumes isotropy. It is not immediately clear to me why this holds.


(2) The analysis for logistic bandits relies on that the constructed matrices $L_t$ uniformly lower-bound the true curvature. It is not immediately clear to me why this derived property of $L_t$ holds. 

Please also see my questions below.

### Questions
(1) The paper claims the derived bounds “essentially tight,” but no lower bound for logistic simple regret is provided. Or am I missing something here?

(2) Does the analysis of theorems implicitly rely on independence between the data used to construct $L_t$​ and the estimator $\hat{\theta}_t$?

(3) The regret analysis relies critically on the claim that the constructed matrices $L_t$​ form valid uniform lower bounds on the true Hessians. This is motivated by the minimization of $\mu(\varphi^\top \theta )$ over confidence sets $W_i$​.
I wonder why this inequality holds with high probability for all tt?
In other words, why cannot $\mu(x)$ tend to 0 at a fast rate as $x$ increases? Or is this actually an assumption instead of a derived property?

(4) The regret bound for the Thompson Sampling–based algorithms (SIMPLELINTS in Theorem 3 and THATS in Theorem 4) relies on a dimension-only alignment guarantee inherited from the linear case. However, in the logistic setting, it is unclear why $L_t$ is still isotropic? If it is not, then wouldn’t this break the analysis?

### Soundness
2

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
3

### Summary
This paper studies the stochastic contextual bandits with simple regret in both linear and logistic models. At the beginning of each round $t$, the learner observes a context $S_t$ sampled from unknown $\nu$. Then, the learner
chooses an action $A_t \in \mathcal{A}(S_t)$ given the past information. After that, they receive the reward $X_t \sim \mathcal{P}(\cdot | S_t, A_t)$.
The goal of the learner is to learn the policy $\pi$ that maps the context space to context-dependent action space. The value of policy $v(\pi)$ 
 is defined as the expected reward, where the randomness is due to unknown distribution $\nu$. The simple regret is now defined as $R(\pi):=\mathrm{sup}_{\pi}  v(\pi)-v(\pi)$.
 
Here, the optimal policy is  $\pi (s) = \mathrm{argmax}_{a \in A(s)}   \mu(\phi(s,a))^{\top} \theta^*)$.

The deterministic algorithm is very simple direct uncertainty maximization (MULIN)). It chooses the point that maximizes the uncertainty score $A_t = \mathrm{argmax}_a \| \phi(S_t,a) \|{V^{-1}_t}$ in the rounds $t=1, \ldots, T$. And finally compute $\hat{\pi}: s \rightarrow \mathrm{argmax}_a \phi(s,a)^{\top} \hat{\theta}_T$ where $\hat{\theta}_T $ is the minimizer of the loss function in the regression. The property of decreasing uncertainty is a key lemma together with the reliability of the confidence sets. 
The second algorithm is a random one, Thompson Sampling (SIMPLELINTS) algorithm for linear contextual bandits and Try Hard Thompson Sampling (THATS) for logistic models.  All algorithms are efficient and the bounds are essentially tight.

### Strengths
1. The methods and analysis are unified in the sense that we can understand the intuitive and important theoretical property in the linear bandits and then the natural extension to the logistic model is described.  The paper is effortless to follow, and the text is flawless.
2. SIMPLELINTS (Theorem 3) achieves $\tilde{O}(d^{3/2}/\sqrt{T})$. Based on it, they also analyze a randomized logistic algorithm based on TS. These randomized methods have computational advantages over the deterministic method.
3. The dominant term in the simple is $\kappa$-free. This highlights the difference in complexity between cumulative vs simple regret.
4. Empirical comparisons of the proposed algorithms are also provided to support theoretical findings.

### Weaknesses
The motivation to study the simple regret in practice is not discussed.

### Questions
- How do the techniques share with pure exploration/best arm identification?
- What are the motivations/advantages to focus on simple regrets over cumulative regret or pure exploration/best arm identification setting?

### Soundness
4

### Presentation
4

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
This paper studies contextual bandits with stochastic contexts under the simple-regret objective. Featuring a new analysis framework, the authors propose fully online algorithms for both the linear and logistic cases. A central point is that, through the use of a curvature-aware algorithm, the logistic simple-regret bounds have a leading term independent of $\kappa$, which can be as large as $\kappa \approx e^S$. To the best of my knowledge, this $\kappa$-free result for the logistic setting is new. The paper also provides empirical results illustrating why slope-weighted uncertainty is essential for efficient exploration in logistic models.

### Strengths
The logistic simple-regret setting is well motivated, and the work fills a clear gap in the literature. To my knowledge, this is the first paper to remove the dependence on the curvature constant $\kappa$ from the leading term of the regret bound. The construction of a monotone surrogate Hessian and the associated decreasing-uncertainty lemma are non-trivial and address the main technical challenge in logistic models, where the uncertainty depends on the unknown slope $\mu'(z)$. These ideas are conceptually elegant and potentially reusable in other generalized linear or reward-free settings.

### Weaknesses
Although there are no fatal theoretical flaws, the paper contains numerous typographical and consistency issues that make verification difficult. The most important ones are:

- In both MULIN and SIMPLELINTS, the design matrix $V_{t+1}$ is never updated. The pseudocode should include  
  $V_{t+1} \leftarrow V_t + \phi(S_t, A_t)\phi(S_t, A_t)^\top.$

- From Equation (17), we have $\mathcal{V}_{t+1} \subseteq \mathcal{V}_t$, but it is reversed in Rows 1420-1421.

- The term $\varphi(s,a)^\top \theta^{\mathrm{Log}}_{T+1}$ at Rows 1014-1015 is duplicated and should appear only once in the expression for the prediction-error decomposition.

### Questions
I have no further questions.

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
This study investigates simple-regret minimization in stochastic contextual bandits with linear and logistic models. To address this problem, the authors introduce deterministic max-uncertainty algorithms (MULIN/MULOG) and randomized variants designed for simple regret (SIMPLELINTS/THATS).

### Strengths
- The authors propose an effective algorithm for simple-regret minimization in stochastic contextual bandits. The regret guarantees are reasonable, and the authors provide sufficient explanations for their derivations.
- In particular, the finite-sample analysis is strong.

### Weaknesses
1. In my understanding, several other studies address simple-regret minimization in stochastic contextual bandits. For example, Kato et al. (2024) develop policy-learning algorithms in this setting. Their goal is to train a policy that minimizes simple regret in a best-arm-identification setting, and they characterize regret bounds using the VC dimension, which covers certain linear and logistic models. Theoretically, that analysis may be somewhat coarse, but could those results be applied to the authors’ setting?

- Kato, Okumura, Ishihara, and Kitagawa. "Adaptive Experimental Design for Policy Learning."

2. The result of this study is important, though not particularly surprising. I believe the problem itself is not especially difficult, and with sufficient effort, researchers could reasonably arrive at similar results. That said, this does not diminish the contribution of the study, as it offers meaningful implications for both theoretical and practical research. However, at least for me, the result is not especially exciting.

3. Is there a lower bound?

### Questions
1. I would like to understand the relationship between the proposed method and the Top-Two Thompson Sampling(TTTS)  algorithm introduced by Russo (2016). Although Russo’s method was originally developed for the fixed-confidence setting, it is now widely regarded as the standard variant of Thompson Sampling in pure exploration problems.

- Russo (2016). "Simple Bayesian Algorithms for Best Arm Identification"

2. Why do the authors use a variant of Thompson Sampling? In my understanding, for example, TTTS algorithms is used for avoiding complicated computation of optimal allocation ratios. Are there any advantages in the use of a variant of Thompson sampling in this setting?

3. Line 2: "Firstly, We" → "Firstly, we"

### Soundness
3

### Presentation
3

### Contribution
3
