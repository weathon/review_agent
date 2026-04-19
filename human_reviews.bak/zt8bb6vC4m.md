# Pricing with Contextual Elasticity and Heteroscedastic Valuation

- Decision: Reject
- Scores: 6, 6, 6, 3

## Abstract
We study an online contextual dynamic pricing problem, where customers decide whether to purchase a product based on its features and price. We introduce a novel approach to modeling a customer's expected demand by incorporating feature-based price elasticity, which can be equivalently represented as a valuation with heteroscedastic noise. To solve the problem, we propose a computationally efficient algorithm called "Pricing with Perturbation (PwP)", which enjoys an $O(\sqrt{dT\log T})$ regret while allowing arbitrary adversarial input context sequences. We also prove a matching lower bound at $\Omega(\sqrt{dT})$ to show the optimality (up to $\log T$ factors). Our results shed light on the relationship between contextual elasticity and heteroscedastic valuation, providing insights for effective and practical pricing strategies.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper studies an online dynamic pricing problem by considering a novel model with feature-based price elasticity.  The authors provide a novel algorithm, ``Pricing with Perturbation (PwP)," that efficiently solves this pricing problem and obtains near-optimal regret, which matches the lower bound of regret up to log terms.

### Strengths
1. The presentation is clear. Beginning with the introduction part, the paper clearly lists its comparisons and generalizations from previous work. Later in the main text, the intuition of the algorithm is also well described. The assumptions made in the paper are also clearly listed and justified.

2. The novelty of the algorithm and its technical contributions are sound. The proposed Pricing with Perturbation (PwP) algorithm is smart and can efficiently solve the problem of a lack of fisher information.

3. Discussions on potential extensions of the work are discussed in detail in the appendix.

### Weaknesses
1. The motivation for this contextual price elasticity seems unclear.

2. Certain assumptions, such as $x^\top \eta$ having a positive lower bound, lack a real-world explanation.

3. Lack of applying this framework to real-data studies

### Questions
1. Can the authors present certain real-world motivations for this contextual price elasticity? e.g., why is it reasonable to rely on the context $x_t$, and is it reasonable to assume that for all $x_t$, $x_t^\top \eta$ is positive all the time? 

2. About the linear assumption on $x_t^\top \eta$, can this be generalized to some non-linear function of $x_t$? Also, when $x_t$ is stochastic, can the assumption of $x_t^\top \eta>0$ be relaxed to $E[x_t^\top \eta]>0$, where $E[\cdot]$ is the expectation over $x$?

3. Can the authors provide a real-world (or semi-real) data study? on evaluating the performance of algorithms in real-life situations.

4. In terms of the presentation of simulation results, could the authors present log-log plots and compare them with the $1/2 log T$ curve? Since it would be hard to see the regret order if they are not presented in this way,

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper investigates a context-based dynamic pricing problem, where customers decide whether to purchase a product based on its features and price. The authors adopt a novel approach to formulating customers’ expected demand by incorporating feature-based price elasticity. The paper provides a matched regret bound for the problem.

### Strengths
Generally speaking, from my point of view, the paper is well written. I really enjoy reading the discussions the authors make, including the relationship between two different formulations and Section 4.1.1. The technical part is solid. The idea of perturbation, though not completely novel, is quite interesting.

### Weaknesses
1.	In my opinion, Ban and Keskin (2021) should be given more credits. As far as I know, Ban and Keskin (2021) is the first to consider the heterogenous price elasticities which are formulated to be linear with context. At least when introducing the formulation, I think the paper should be cited and discussed more.
2.	I understand that a known link function is a good starting point and a common practice. One direction that I think might further improve the paper is to consider (or at least discuss about) an unknown link function. The reason why I mention this point is that Fan et al. (2021) studies a problem with unknown noise distribution. According to equivalence of the two formulation, it seems that it is not undoable to consider a version without knowing the link function. 
3.	About the Perturbation, similar ideas can be found in the dynamic pricing literature (see, e.g., Nambiar et al. 2019). From my perspective, the only reason why the time horizon $T$ should be known in advance is because we need it to calculate $\Delta$. Nambiar et al. (2019) dynamically change the magnitude of the perturbation, which may potentially help the current algorithm to get rid of the known time horizon $T$. Please correct me if I am wrong.

Reference:
Gah-Yi Ban and N Bora Keskin. Personalized dynamic pricing with machine learning: High-dimensional features and heterogeneous elasticity. Management Science, 67(9):5549–5568, 2021.

Jianqing Fan, Yongyi Guo, and Mengxin Yu. Policy optimization using semiparametric models for dynamic pricing. arXiv preprint arXiv:2109.06368, 2021.

Mila Nambiar, David Simchi-Levi, and He Wang. Dynamic learning and pricing with model misspecification. Management Science, 65(11):4980-5000, 2019.

### Questions
See above.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper unifies the ``linear demand'' and the ``linear valuation'' by proposing a new demand model where each item has a feature-dependent price elasticity. The authors devise an effective online optimization algorithm that can achieve a nearly optimal regret bound. Some numerical simulations are conducted to empirically show the effectiveness of the proposed approach.

### Strengths
S1. A new demand model for the contextual pricing problem.

S2. The proposed algorithm has a regret bound close to the theoretical lower bound.

S3. Numerical simulations are conducted.

### Weaknesses
W1. Although the proposed demand model extends existing models by considering the feature-dependent price elasticity, the proposed model and online algorithm still rely on linear forms of elasticity and valuation. Remember ICLR is a deep learning conference. A potentially more suitable treatment may be substituting the linear functions with a neural tangent kernel and then devising online algorithms correspondingly.

W2. What is the major technical challenge if we replace the uniform \alpha with a feature-dependent price elasticity? The authors may want to discuss more the impact of introducing feature-dependent price elasticity terms on algorithm design as well as regret analysis. 

W3. As the authors mention in Ethic issues, personalized pricing may have fairness issues. Therefore, it is essential to discuss how to deal with the cases when we add some fairness regularization terms or fairness constraints to the optimization problem. 

W4. Still about personalized pricing. As the objective is purely the interest of the platform, I would like to see discussions or experimental results on how the personalized pricing algorithm affects customer well-being metrics such as consumer surplus.

### Questions
W2

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this work, the authors face the problem of contextual dynamic pricing in a heteroscedastic environment. The authors face this applicative problem by proposing a new theoretical framework. They provide a lower bound on the expected regret for the setting. Then, the authors provide an algorithm, for which they discuss the upper bound, which matches the lower bound up to log factors. The authors also provide a numerical validation of the solution.

### Strengths
The work faces a problem of interest from the applicative point of view. 

The relevant literature is properly discussed.

### Weaknesses
The presentation can be improved, in particular from the introductory part.

The main concern is about the theoretical analysis of this paper. Indeed, an important focus of this work is related to heteroscedasticity, which is its differential part w.r.t. existing literature. However, this phenomenon is not highlighted in the analysis. For example, in Thr 4.5, the authors retrieve a bound in which such a phenomenon is not highlighted, and the result presented is already present in the literature. Furthermore, the result presented is known for a setting that is simpler than the one presented in this paper, so it holds in this scenario.

### Questions
See weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
