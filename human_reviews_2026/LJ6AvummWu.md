# SOReL and TOReL: Two Methods for Fully Offline Reinforcement Learning

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Sample efficiency remains a major obstacle for real world adoption of reinforcement learning (RL): success has been limited to settings where simulators provide access to essentially unlimited environment interactions, which in reality are typically costly or dangerous to obtain. Offline RL in principle offers a solution by exploiting offline data to learn a near-optimal policy before deployment. In practice, however, current offline RL methods rely on extensive online interactions for hyperparameter tuning, and have no reliable bound on their initial online performance. To address these two issues, we introduce two algorithms. Firstly, SOReL: an algorithm for safe offline reinforcement learning. Using only offline data our Bayesian approach infers a posterior over environment dynamics to obtain a reliable estimate of the online performance via the posterior predictive uncertainty. Crucially, all hyperparameters are also tuned fully offline. Secondly, we introduce TOReL: a tuning for offline reinforcement learning algorithm that extends our information rate based offline hyperparameter tuning methods to general offline RL approaches. Our empirical evaluation confirms SOReL's ability to accurately estimate regret in the Bayesian setting whilst TOReL's offline hyperparameter tuning achieves competitive performance with the best online hyperparameter tuning methods using only offline data. Thus, SOReL and TOReL make a significant step towards safe and reliable offline RL, unlocking the potential for RL in the real world.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Current offline RL methods rely on extensive online interactions for hyperparameter tuning, and have no reliable bound on their initial online performance. To address these two issues, the authors introduce two algorithms. Firstly, SOReL: an algorithm for safe offline reinforcement learning. Using only offline data their Bayesian approach infers a posterior over environment dynamics to obtain a reliable estimate of the online performance via the posterior predictive uncertainty. Crucially, all hyperparameters are also tuned fully offline. Secondly, they introduce TOReL: a tuning for offline reinforcement learning algorithm that extends their information rate based offline hyperparameter tuning methods to general offline RL approaches. Lastly, they conduct empirical evaluations.

### Strengths
1. The problem of fully offline RL is interesting and important.
2. The authors propose to solve the problem through a Bayesian view, which is an interesting idea.
3. There are empirical evidence supporting the theoretical findings.

### Weaknesses
1. My main concern is about Theorem 2, since the last equation in line 274 is not correct. As N becomes sufficiently large, LHS is a constant while RHS converges to 0. If this is a typo, please provide the correct form.

2. Discussions about Theorem 2 and its assumption is missing in the main part. Therefore it is hard to understand the implication of the Theorem. Some high-level discussions about the assumption in the main part would be helpful.

3. The algorithms tune the hyperparameters fully offline, while it is unclear whether the approach is efficient. The PIL function includes multiple expectations and the posterior distribution, which seems hard to optimize directly. Are the optimization problems in both algorithms efficient in general?

### Questions
Please refer to the weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces SOReL and TOReL, two new algorithms that enable safe and reliable offline reinforcement learning by providing offline hyperparameter tuning and regret estimation, a problem so far underexplored in prior literature, thereby improving sample efficiency and performance guarantees without needing online interactions and moving offline RL closer to practical applicability.

### Strengths
- the theoretical considerations & derivation of the algorithms, leveraging PIL to select hyperparameters for the model appear novel and significant
- the method appears to be able to well approximate the true regret, enabling fully offline hyperparameter tuning and thus bringing offline RL closer to real world applicability
- the method is very flexible and can be used in principle with any existing offline RL algorithm of a users choice
- the authors demonstrate accurate regret estimation on a variety of tasks / datasets

### Weaknesses
While I find the proposed method highly appealing in many ways, I find the paper has a couple of major weaknesses:

1) A key contribution appears to be the PIL-based tuning of the model hyperparameters. I would argue however, that this is the "easy" part of offline RL hyperparameter tuning, since we can resort to simple supervised learning techniques like holding out 10% of the training data & measuring prediction performance on this set so select hyperparameters. I would argue the proposed method is much more sophisticated (directly tuning for regret minimization instead of dynamics prediction performance), and works likely better - but since this is a large part of the paper's contribution, I'd argue we still need a comparison against such a simple baseline.

2) Apart from the PIL-based model parameter tuning, this kind of approach (bayesian model + sampling from posterior to make policy e.g. safe) has been studied in many prior works, such as [1,2] (consider adding in related work). No need for a direct comparison in experiments, but I'd argue the novelty in that sense it at least somewhat limited.

3) Probably the biggest one: I would agree that so far no method has really solved the offline hyperparameter tuning / selection problem & your work towards achieving this appears highly promising, but I would not agree to your assessment in the related work section, that your method is the first to "carry out all hyperparameter tuning [...] using only offline data". Other methods have been developed for fully offline policy evaluation / hyperparameter selection (see e.g. the baselines in [3] and all the derivatives that were developed). While you compare your algorithm's performance against an online tuning algorithm and an oracle, the truly interesting thing to understand (at least to me it seems that way), would be whether the proposed method performs better than other offline tuning / selection methods - without these experiments, I believe the key merit of your method is not empirically validated.

[1] Depeweg, Stefan, et al. "Learning and policy search in stochastic dynamical systems with bayesian neural networks." ICLR 2017
[2] Kaiser, Markus, et al. "Bayesian decomposition of multi-modal dynamical systems for reinforcement learning." Neurocomputing 416 (2020): 352-359.
[3] Fu, Justin, et al. "Benchmarks for deep off-policy evaluation." ICLR (2021).

### Questions
- please clarify if I am mistaken: PIL is only used for model hyperparameter tuning - afterwards, to tune policy, we simply use the final selected model to quantify regret (via median return), correct?
- if I am not mistaken, you refer to PIL with many different terms (posterior inforamtion loss, posterior information distance, information rate, ...) - this is a bit confusing, could you elaborate / align?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses two critical issues in offline reinforcement learning that have largely been overlooked: the lack of offline hyperparameter tuning metrics, and the absence of reliable offline regret approximation methods . The authors propose a Bayesian framework for offline RL, leveraging the Posterior Information Loss (PIL), the expected KL divergence between a learned model and true environment dynamics as an offline tuning signal. From this framework, they introduce two methods: SOReL (Safe Offline RL), which provides reliable offline regret estimates using predictive uncertainty to enable safe deployment, and TOReL (Tuning for Offline RL), which extends SOReL's offline hyperparameter tuning methodology to general model-based and model-free offline RL algorithms. The paper provides rigorous theoretical analysis showing that regret is bounded by the PIL (Theorem 1) and that Bayesian offline RL achieves the optimal convergence rate (Theorem 2). Empirical evaluations on D4RL, Adroit, and proprietary Brax datasets demonstrate that TOReL identifies near-oracle hyperparameters using only offline data, saving 20K to >200K online samples compared to existing online hyperparameter tuning methods, while SOReL accurately approximates true regret across diverse offline datasets.

### Strengths
**Well-motivated problem formulation.** The paper identifies and articulates two concrete, practically important issues in offline RL that have been underexplored: the reliance on online interactions for hyperparameter tuning and the lack of performance guarantees before deployment. The motivating examples (healthcare, robotics) and the cycle diagram (Figure 1) effectively communicate why these issues matter for real-world applications. This clear problem framing strengthens the contribution's significance.

**Rigorous theoretical analysis with frequentist justification.** Theorems 1 and 2 provide formal regret bounds in terms of PIL, with Theorem 2 establishing the optimal convergence rate for Bayesian offline RL under local asymptotic normality assumptions. The frequentist justification for a Bayesian approach is valuable and non-obvious. Proposition 1  decomposes PIL for Gaussian world models into interpretable terms (MSE and predictive variance), facilitating practical implementation and understanding.

**Comprehensive empirical validation.** The paper includes extensive experiments across multiple environments (gymnax, Brax, D4RL) with careful ablations and statistical testing (e.g., Pearson correlation in Table 3). The comparison against online UCB-based hyperparameter tuning (Figure 3) provides compelling evidence of sample efficiency gains. Both SOReL and TOReL experiments demonstrate practical effectiveness, with particular strength in the near-oracle performance of ReBRAC+TOReL across diverse datasets (Table 1).

### Weaknesses
**Gap between theory and practice.** The theoretical regret bound (Theorem 1) is noted to be "too conservative" and is not used in practice; instead, the posterior predictive median (Eq. 5) is employed as a heuristic. This disconnect undermines the theoretical contribution's direct utility. The paper acknowledges that the bound's tightness depends critically on model accuracy relative to discount factor 
\gamma, a constraint difficult to satisfy in practice (e.g., only pendulum-v1 yields non-trivial bounds in Figure 10b).

**Missing comparisons with related work.** The paper does not compare against other offline hyperparameter tuning methods beyond citing them (e.g., Wang et al. 2022, Paine et al. 2020, or any related work of the same nature). Direct empirical comparisons would strengthen the work. There are some numerical experiments, but visual trends of training/eval curves would be helpful.

### Questions
**Q1:** Can the theoretical bound (Theorem 1) be tightened? Given that the bound is often too conservative in practice, are there domain-specific refinements or tighter bounds for structured environments that could improve practical utility?

**Q2:** Why use the posterior predictive median rather than other quantiles (e.g., mean, upper confidence bound)? The choice of median is justified as a "compromise" but lacks formal justification. Have you tested other regret approximation metrics?

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
4

### Summary
This paper considers two salient problems with offline RL: (a) what offline metrics are useful for tuning hyper-parameters of an offline RL policy search procedure, (b) how to estimate the online regret associated with deploying a policy outputted by an offline RL procedure? Towards this, the paper utilizes Bayesian perspective to consider the use of posterior information loss (PIL) for purposes of solving the aforementioned problems, developing TOReL (for tuning) and SOReL (for estimating regret).

### Strengths
- The paper addresses important problems that plague the design of offline RL algorithms and these issues limit the applicability of offline RL in general. 
- The empirical results appear pretty compelling as well.

### Weaknesses
- The paper can do a better job with presenting connections with lowerbounds in offline RL connecting the regret estimation against hardness of policy evaluation; for instance, see [1], but there are probably other works that build on this. Why does the proposed procedure actually work in light of some of these stark negative results?

[1] Wang et al: What are the Statistical Limits of Offline RL with Linear Function Approximation? 2020.

### Questions
- Regarding estimators proposed by this paper: can you mention again how one can rely on point estimates of these quantities and whether they can be strengthened using estimation procedures building on [2]?

[2] Agarwal et al: Deep Reinforcement Learning at the Edge of the Statistical Precipice, 2021.

### Soundness
2

### Presentation
3

### Contribution
3
