# Towards Safe and Optimal Online Bidding: A Modular Look-ahead Lyapunov Framework

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 8

## Abstract
This paper studies online bidding subject to simultaneous budget and return-on-investment (ROI) constraints, which encodes the goal of balancing high volume and profitability. We formulate the problem as a general constrained online learning problem that can be applied to diverse bidding settings (e.g., first-price or second-price auctions) and feedback regimes (e.g., full or partial information), among others. We introduce L2FOB, a Look-ahead Lyapunov Framework for Online Bidding with strong empirical and theoretical performance. By combining optimistic reward and pessimistic cost estimation with the look-ahead virtual queue mechanism, L2FOB delivers safe and optimal bidding decisions. We provide adaptive guarantees: L2FOB achieves $O (\mathcal{E}\_r(T,p)+(\nu^* / \rho)  \mathcal{E}\_c(T,p))$ regret and $O (\mathcal{E}\_r(T,p)+\mathcal{E}\_c(T,p))$ anytime ROI constraint violation, where $\mathcal{E}_r(T,p)$ and $\mathcal{E}_c(T,p)$ are cumulative estimation errors over $T$ rounds, $\rho$ is the average per-round budget, and $\nu^*$ is the offline optimal average reward. We instantiate L2FOB in several online bidding settings, demonstrating guarantees that match or improve upon the best-known results. These results are derived from the novel look-ahead design and Lyapunov stability analysis. Numerical experiments further validate our theoretical guarantees.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper provides a framework and an algorithm for ROI and budget constrained bidding in repeated auctions. 
The framework relies on the notions of look-ahead virtual queue tracing, instead of using a more standard primal-dual approach. 
The results are expressed in term of error made by prediction oracles. 
The framework is used to revisite previous works.

### Strengths
The topic of budget and ROI constrained bidding  has already attracted a lot of attention, 
and I find the framework proposed by the authors practical and intuitive (compared too, for instance, dual approaches).

### Weaknesses
* The writing is not always clear, and I urge the authors to check each paragraph clarity.
* What would, IMHO, what could really improve the paper would be: 
1. to clarify the positioning with respect to the other papers on the topic (stochastic vs adversarial env, slack assumtions...). A table would be very welcome. 
2. to better explain what the Lyapunov Framework is in general, it feels like some references and explanation could be added on the topic. For instance, could the authors tell the readers where the name comes from?
 
* the practicality of the assumption should be more discussed. For instance, how do you compute the "E(...)" in the algorithm in practice?

### Questions
* see weakness

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper formulates the general online bidding problem as an online learning problem with both the budget and target return-on-investment (ROI) constraints. Under this general learning problem, the authors proposed a primal–dual algorithm using a Lyapunov perspective, and they also show an regret upper bound for this algorithm with no budget overuse but some ROI violations. Additionally, for several specific settings, this upper bound matches the optimal regret order.

### Strengths
1. To my best knowledge, the idea of using Lyapunov function is novel.

2. The theoretical result holds in a very general framework, and also achieves strong performances in several specific but important settings.

3. Numerical results are also included in the appendix to illustrate the performance.

### Weaknesses
1. My main concern is about the compatibility of the learning oracle and the proposed algorithm with partial information feedback (bandit setting). Specifically, when showing sqrt{T} results in Section 5, the authors state that bounds on \Epsilon(T,p) can be obtained following some other papers. However, their arm selection process might be different from this paper. How to guarantee that the bound still holds? Could the authors elaborate more on these bounds and provide a more detailed version of the proofs? I will raise my score is this question is addressed.

2. It seems the algorithm works when only one budget constraint exists. This assumption may limit the applicability. I wonder whether a similar statement holds when there are multiple budget constraints, especially Lemma 3?

### Questions
See the weaknesses section.

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
4

### Summary
The paper study the online bidding problem under budget (hard) and ROI constraints, in stochastic contextual settings. The authors propose an algorithm based on a Lyapunov framework which attains regret and violations that scale linearly in the error of some online learning oracles which are assumed to exist and are used to estimate the reward and the constraints functions. Finally, the authors show how the setting they study generalise many well known problems (e.g., first price auctions) and the specific guarantees of the algorithm in the aforementioned settings.

### Strengths
The main strength of the paper is the setting studied. Indeed it generalises many "online bidding" problems, thus being of interest for the community. Second, I find particularly interesting the fact that the algorithm presented does not require Slater's condition to obtain the desired bound.

### Weaknesses
I have the following concerns on the paper:

1. First, the authors miss a fundamental related work ([1]). In [1], the authors study an online learning problem with budget and roi constraints as in this paper. The only difference is that the authors study a non-contextual setting. Moreover, [1] requires some form of Slater's condition on the ROI constraint to obtain the optimal bounds. Nonetheless, [1] attains guarantees for adversarial settings, too.
2. I find peculiar that the context distribution is assumed to be known. I believe it is not standard in the literature of contextual bandits, while it seems fundamental for the employment of the algorithm, since the expectation on the context distribution is computed at each round.
3. Paragraph "Assumptions and Baselines" is not formal enough. Specifically, Equation (2) is meaningless as it is. First, it is not clear over which distribution the expectation is taken. Moreover, if the expectation is taken over the context distribution, as it seems given that $v_t$ is not used in (2), the problem collapse to a non contextual online learning problem.  
4. Throughout the paper, the authors claim that they bound the positive ROI violation (using $()^+$). Nonetheless, the standard metric is used in the theorems. 
5. Finally, I don't understand the proof step at line 299. I think the inequality holds paying a $\sqrt T$ factor thanks to the Azuma inequality. Instead, the authors claims that it works as it is. 

[1] "Online Learning under Budget and ROI Constraints via Weak Adaptivity", ICML 2024

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the problem of online bidding with a budget constraint and a return-on-investment (ROI) constraint. The authors develop a general, modular framework for this problem and propose an algorithm called L2FOB. This algorithm uses a combination of optimistic reward estimates, pessimistic cost estimates, and a one-step look ahead to optimize for the bids. The authors prove adaptive bounds on regret and ROI constraint violation in terms of errors of the reward and cost estimators.

### Strengths
* The authors develop a modular framework and algorithm that can be applied to many specific settings (as they do in Section 5).

* The proposed algorithm uses a nice look-ahead function to bound a potential function, which they then use to bound the Lyapunov drift and resulting regret bound. This is a neat idea. This idea seems similar to one-step model predictive control in control theory.

* The bounds on regret and ROI constraint violation are adaptive in terms of the error bounds of the reward and cost estimators. The authors achieve this regret bound without Slater's condition, although the trade-off (as they note) is that instead of strict ROI constraint satisfaction they incur some violation.

### Weaknesses
(Combined weaknesses and questions into one section.)

* The look-ahead idea is nice. As I mentioned in the strengths section, this seems similar to one-step model predictive control in control theory. It would be nice to discuss this area of related work in the paper. On a related note, how helpful would it be to consider a $k$-step look-ahead?

* The problem setup assumes that the context distribution is known and uses it to compute the expectations over $v$ in lines 4, 5 and 7 in Algorithm 1. Is it necessary to assume a known context distribution? How does your algorithm behave if the updates are noisy - either through point estimates, empirical estimates using the history, or an approximately known context distribution?

* Can your ideas be generalized to non-bidding problems in online optimization with constraints? How well does your algorithm scale with the number of constraints?

* The ROI constraint is not satisfied strictly, although this comes with the advantage of not assuming Slater's condition.

### Questions
Please see the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
