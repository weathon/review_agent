# Horizon-Free Regret for Linear Markov Decision Processes

- Decision: Accept (poster)
- Scores: 6, 5, 6, 8

## Abstract
A recent line of works showed regret bounds in reinforcement learning (RL) can be (nearly) independent of planning horizon, a.k.a. the horizon-free bounds. However, these regret bounds only apply to settings where a polynomial dependency on the size of transition model is allowed, such as tabular Markov Decision Process (MDP) and linear mixture MDP. We give the first horizon-free bound for the popular linear MDP setting where the size of the transition model can be exponentially large or even uncountable.  In contrast to prior works which explicitly estimate the transition model and compute the inhomogeneous value functions at different time steps, we directly estimate the value functions and confidence sets.  We obtain the horizon-free bound by: (1) maintaining multiple weighted least square estimators for the value functions; and (2) a structural lemma which shows the maximal total variation of the inhomogeneous value functions is bounded by a polynomial factor of the feature dimension.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces an algorithm for linear MDPs that achieves a regret bound independent of the horizon length ($H$). It specifically analyzes regret by grouping indices with the same maximal total variance for the optimal value function, allowing value functions within the same group, even if values are inhomogeneous, to share samples. Despite its computational inefficiency, this work addresses the research question of whether it's possible to obtain a horizon-independent regret bound in linear MDPs, extending the prior research on horizon-free regret analysis in linear mixture MDPs.

### Strengths
In a linear MDP (simply assuming a known reward function), by setting $\theta^\*\_h = \mu^\top V^*\_{h+1}$ for the transition kernel parameter $\mu$, since the action value for each state-action pair is given by $Q^*_h(s,a) = r(s,a) + \phi(s,a)^\top \theta^*_h$, hence finding the optimal action at h-step is the same as solving a linear bandit problem. Based on this perspective it seems like an interesting approach to group $H$ bandit problems and share samples to get a $H$-independent regret bound.

### Weaknesses
- While the analysis of horizon-free regret in linear MDPs, as an extension of the algorithm for linear mixture MDPs, is intriguing, the technical novelty in this paper seems somewhat limited. As mentioned in related work, Zhang et al. (2021b) initially proposed a horizon-free algorithm for linear mixture MDPs, and Zhou & Gu (2022) introduced a computationally efficient horizon-free algorithm. Although it is acknowledged that Linear MDPs become fundamentally more challenging as the dimension of unknown parameters grows with the increasing state space, given that the order of regret bound for minimax algorithms with $H$ dependence in both linear MDPs and linear mixture MDPs is the same (Hu et al., 2023), it is regrettable that this paper only presents a computationally inefficient algorithm despite the existence of computationally efficient $H$-free regret bounds in linear mixture MDPs.

(Hu et al., “Nearly minimax optimal reinforcement learning with linear function approximation”, ICML 2022)

- Providing a slightly clearer explanation of the method in the main text would be beneficial. For instance, as mentioned in Technique 3, calculating an upper bound for $\max\_{h' \in [H]} \sum\_{k=1}^K \sum\_{h=1}^H \mathbb{V} (P\_{s^k\_h, a^k\_h}, V^*\_{h'+1})$ is required, but it was stated that a bound for $ \sum\_{k=1}^K \sum\_{i=1}^{\log\_2 H +1} \max\_{h' \in [H\_i]} \sum\_{h \in H\_i} \mathbb{V} (P\_{s^k\_h, a^k\_h}, V^*\_{h+1})$ was given. A more explicit explanation of how these two bounds are connected would enhance clarity.

### Questions
- While the set of all possible features $\Phi$ is not precisely defined, if we define it as $\Phi:=  \\{ \phi(s,a) : (s,a) \in \mathcal{S} \times \mathcal{A} \\}$, then the convexity of $\Phi$ implies that for all $\phi_1, \phi_2 \in \Phi$ and \lambda \in [0,1], $\lambda \phi_1 + (1 - \lambda) \phi_2 \in \Phi$. This appears to be a highly restrictive assumption. It would be beneficial to explicitly mention this assumption as a Key assumption like Assumption 1-3 if it is essential, especially considering it is not utilized in existing linear MDP literature (e.g., Jin et al. 2020b).

- On page 7, could you please provide an explanation for the statement, "$\mathbb{V}(P_{s,a}, v) = \phi(s,a)^\top (\theta(v^2)) - (\phi(s,a)^\top \theta(v))^2$" being described as a linear function of the matrix ?

- Could you clarify how the last inequality in equation (11) holds?

- Typo?: It seems that while $l_h^*$ is defined below equation (10), $l_h$ is not defined. Please provide a definition for $l_h$.

- What is the meaning of "size of transition model"?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper considers a regret bound of linear MDP that has a mild dependency on the horizon $H$. The main results are emphasized in the Thm 1 (2nd page). The transition kernel is uniform over timesteps, the Q function is discountless, and the time horizon H is fixed. The challenge here is the optimal action may depend on the remaining time (i.e., should the model explore more or not), which is reflected in the index $h$ in $Q_h$, $V_h$. The process is sequential; before the beginning of each episode k, we declare policy $\pi^k$ to play it and observe rewards, and evaluation is based on online measure (regret). Assumptions are that 1: bounded total rewards, 2: linear transition kernel (dim: $S \times d$) and rewards (dim: $d$) with constant scale per feature. Assumption 3 states that the transition and reward share parameter \theta (dim: $d$).

The contribution of this paper is to show horizon-free regret (the regret of subpolynomial dependence on H) for MDP under assumption 2 (compared with similar results with assumption 3 by Zhou and Gu 2022). The paper proposed an algorithm (Algs 1--3 combined) to show this regret. The algorithm seems to be computationally inefficient given it defines some quantities that depend on v (a real variable with bounded norm). While the contribution of this paper is written in the paper and it seems fine, I do not consider the paper to be well-written and ready for publication. Since I am not convinced of the correctness of the claim, I lean to be negative on the paper, but I do not have a strong position . I have several questions in the corresponding section for the sake of clarity.

### Strengths
* The paper considers interesting question whether linear MDP is learnable with horizon-free sample complexity. The paper considers the setting where $\theta_r, \mu$ are unknown, which is surely interesting to the community. 
* Sections 1--2 are well-written, except for related work section where many papers are cited without specific connection, but overall, nice.

### Weaknesses
* The algorithm is computationally inefficient with respect to $v$, so the algorithm here is more conceptual than practical.
* The structure of the paper can be improved. It seems the structure follows Zhou & Gu (2022), but the demonstration of "technical challenge for 3.5 pages without demonstrating algorithm (the algorithm is introduced after that) does not make much sense. At least, this 3.5 pages does not help the reader to be convinced of the results more than 1-page summary.
* There are many typos, which makes it hard for the reader to be convinced of the results.

### Questions
My largest question is about Algorithm 1. If I understand correctly, these algorithms require optimization over continuous space. First, $\mathcal{W}(\epsilon)$ is not defined in this paper; similar quantities such as $\mathcal{W}$ and $\mathcal{W}_\epsilon$ are defined twice, so I assume these are equivalent. Lines 6--9 in Alg1 loops over $v \in \mathcal{W}(\epsilon)$, which, defines quantities $(\hat{\theta}, \tilde{\theta}, \Lambda^k(v), b^k(v, \phi))$ as a function of $v$. $\mathcal{U}^k$ is also optimized on the space of $v$.
* $\tilde{\theta}(v)$ is not used in the line 10 of Alg 1 and after?

On assumption 2: Is assumption (2d) standard? It seems that the number of unknown parameter $\mu \in \mathbb{R}^{\mathcal{S}\times d}$ can be arbitrarily large and it is highly nontrivial to have regret bound that does not depend on the size of $\mathcal{S}$.

On the similarity with the (contextual) linear bandit problem. The linear bandit problem is where $x$ is disclosed (contextual is $x$ is $x(t)$) and reward is $r(t) = x^\top \theta$ where $\theta$ is an unknown quantity. In the MDP of this paper, transition has uncertainty and the uncertainty on the reward $r(t)$ w.r.t. transition kernel is much more involved.

The following are rather minor opinions:
* VOFUL: Is it by any means related to OFUL? If I understand correctly, OFUL (e.g., Abbasi-Yadkori et al. 2011) is an algorithm for online bandit algorithm whereas Alg 3 is for the uniform confidence bound (also referred as self-normalized bound or martingale bound).
* Typos. Many periods are missing, for examples:
> p2: See Section 3 for more details. Due to space limitation, we postpone the full proof of Theorem 1 to Appendix B
> p6: That is, we need to use all the samples along the trajectory...
> p7: Fix some eps>0 

* Lines 5, 11, 13 in Alg 1 are comments and can be used some special characters for indicating that, such as $//$

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper provides the first horizon-free bound for linear MDP. The corresponding algorithm is also novel since this algorithm estimates the value function and corresponding uncertainty directly instead of estimating the transition probability.

### Strengths
1. This is the first bound for the linear MDP problem.

2. The approach is novel and interesting.

### Weaknesses
1. It would be better to have some numerical illustrations to see whether the given algorithm has better numerical performances.

### Questions
1. I wonder whether this algorithm is numerically efficient.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose a novel algorithm for reinforcement learning (RL) in linear Markov Decision Processes (MDPs). The key feature of the presented algorithm is horizon-free regret bounds, that means that the regret of the algorithm scales only polylogarithmically with a problem horizon.

### Strengths
- The first algorithm for linear MDPs that attains nearly horizon-free regret bounds;
- Very detailed discussion on novelties of the proposed approach;

### Weaknesses
- Algorithm is computationally unfeasible: it requires to optimize over the confidence region of all models of linear MDPs, that is unfeasible in MDPs with infinite number of states;
- Proof of Lemma 11 is unclear for me, see section Questions;

### Questions
Here I would like to present questions to proofs and additionally indicate misprints that I found. I would like to increase my score if all the clarifications regarding proofs will be given.

Questions regarding Lemma 4:

- Why it holds $f(l-y)/y^{d-1} ≤ f(l-x)/x^{d-1}$?
- Is there any references that will show why (16) hold?

Lemma 11:

- Possible misprint: After an inequality (a) there is missed square over $v$ in variance and also there should be $v^2(s^k_{h+1})$ in the second term (not $v^2(s^k_h)$);
- Definition of $\bar v(s)$ is not clearly written: is it  $\bar v(s) = \max_{a} P_{s,a} v?
- It is not clear why (c) holds because it is not easily understandable why the presented $\bar v$ maximizes $v(s^k_{h+1})$, or I did not understand the definition of $\bar v$.
- Why (d) holds? What is a constant $c$?

Misprints and undefined or confusing notation.

- eq. (8) — V^* without h
- Equations (9) and (13) — missed square in \bar \sigma^k_h in the second multiplier after Cauchy-Schwartz inequality;
- After equation (9) it claims that \sigma^k_h is variance, whereas before $\sigma^k_h$ was a square root of variance, it is very confusing;
- After eq. (10) — $l^*$ in the beginning and after l without *;
- After definition of $W_{\epsilon}$: w.r.t. should be with points not commas;
- Proof of Lemma 5: $O$ is not defined;
- After equation (28): use of Var as variance whereas usually it was denoted by $\mathbb{V}$, and after it confusing claim $\sigma^k_h(v) \geq Var$.
- After equation (29) — no brackets around $i$ in the inequality $\bar \Lambda^k_i \leq \Lambda^k(\bar V^k_{h+1})$.
- After equation (30) — forgotten subscript $(i)$ in the definition of $\bar \Lambda^k$ and it is not clear why $\bar \Lambda^k + \sum_{h=1}^H … = \bar \Lambda^{k+1}$.
- Equation (43) — tilde in not over $O$ but over all the expression;
- Last parargraph of page 21: underfined norm type $\Vert \bar v - v \Vert_i$;
- Lemma 12 — missed norm type in the statement (is it $\ell_2$ norm or $\ell_\infty$ as it used later?).

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good
