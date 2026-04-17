# Distributionally Robust Conditional Conformal Prediction

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Conformal prediction (CP) constructs prediction sets with a marginal coverage guarantee of $1 - \alpha$, assuming the calibration distribution $P_{XY}$ and test distribution $Q_{XY}$ are identical. Under distribution shift, existing approaches align calibration and test conformal scores only at the marginal level, which helps preserve marginal coverage. However, ignoring their mismatched conditional score distributions can lead to poor conditional coverage at individual test inputs. In response, we introduce the conditional coverage gap (CCG) and its expectation over $Q_X$ to quantify the robustness of the conditional guarantee. To study how a distribution shift is propagated from data to conformal scores, we use the Wasserstein distance between $P_{XY}$ and $Q_{XY}$ to bound the expected CCG. This bound implies that an invertible transformation between $P_{XY}$ and $Q_{XY}$ via Wasserstein minimization can promote robust conditional coverage. Lastly, we implement the idea by Branched Normalizing Flow (BNF), a two-branch structure where the $X$-branch transports test inputs from $Q_X$ to $P_X$ to obtain prediction sets with conditional guarantee on $P_{Y|X}$, and the $Y$-branch inversely maps these sets with preserved conditional guarantee on $Q_{Y|X}$. Extensive experiments on nine datasets demonstrate that BNF consistently reduces CCG with improved coverage robustness across various confidence levels under distribution shift.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies constructing prediction intervals in the presence of distribution shifts from 
the source to the target environment. The proposed method is based on the branched normalizing flow and mainly applies to the 
case where the target environment is a mixture of source environments. The method is 
tested in a rich set of numerical experiments.

### Strengths
1. The question studies in this paper is of interest, and the proposed method provides
some new perspectives on this problem.
2. The proposed method has been tested on a relatively rich sets of data.

### Weaknesses
1. The exposition of the manuscript could be improved. At times, the paper's motivation and contribution are misaligned. For example, it is claimed in the abstract that *"However, these methods often neglect the dependence between input features and conformal scores, leading to degraded conditional coverage for individual
test inputs"*, whereas the paper does not propose scores or methods to improve conditional coverage; instead, it focuses on mitigating conditional distribution shift. Elsewhere, some statements appear inaccurate—for instance, ‘these existing methods ignore the dependency of conformal scores on input features,’ which is not correct.

2. The proposed method requires learning a transformation form $P_{XY}$ to $Q_{XY}$, 
which is in general impossible because $Y$ is not observed in the target environment and 
therefore $Q_{Y\mid X}$ is not identifiable. The paper considers a special case 
where $Q_{X,Y}$ is in the convex hull of the source distributions (where $Q_{Y \mid X}$ can be 
identified from the shift of covariate distribution). This assumption feels restrictive; 
it would help to discuss additional scenarios where the method is applicable or provide concrete examples illustrating its usefulness.


3. The guarantees provided by the model is a bit unclear. The results, in e.g. Appendix O, 
leave terms like $\alpha_{\text{i.i.d.}}$ and $\alpha_{\text{trans}}$ without specifying how 
these terms depend on the instances (e.g., sample sizes, dimensionality, number of sources, and shift magnitude).

### Questions
1. Continuing with my comment #2 in the "Weaknesses" section, I wonder if the authors could 
provide other examples where the proposed method is applicable beyond the convex hull
model. 

2. I am a bit confused by the bound in equation (14): is the right-hand side bound 
really informative for the task at hand. To be specific, if the target is to evaluate 
the conditional coverage, it is confusing to me why the covariate shift matters. 
Consider the case of pure covariate shift, and we have a very accurate prediction model;
then one would expect the left-hand side to be very small while the right-hand side 
to be large. In other words, minimizing the right-hand side is far from the true objective.

3. [1] also considers multi-source conformal prediction; how does the proposed method compare to theirs?

**Reference:**
1. Liu, Yi, et al. "Multi-source conformal inference under distribution shift." Proceedings of machine learning research 235 (2024): 31344.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses conditional coverage under calibration-test distribution shift. It (i) defines a CCG and its expectation over the test feature law ICG as measures of conditional coverage robustness under distributional shift; (ii) derives upper bounds on ICG using Wasserstein distance between calibration and test joint distributions. Building on this insight, the authors propose Branched Normalizing Flow (BNF) and an Augmented BNF variant which attempt to map the test joint distribution to the calibration joint via an invertible flow. After transforming inputs, they apply an adaptive conformal algorithm on the normalized data and inverse-transform prediction sets to the original label space. Empirically, they evaluate on synthetic and real datasets and introduce ASCG as an empirical metric for conditional coverage. The results show improved ASCG and stable marginal coverage compared to the baseline methods.

### Strengths
1.	Figure 1 is very detailed and helps in understanding the presented framework.
2.	The empirical results show that the proposed method performs better than the baseline methods.

### Weaknesses
1.	The abstract and introduction are somewhat misleading: They describe a general distribution shift between the calibration and test, while the proposed method requires additional information about the shift.
2.	Limited applicability: the proposed approach requires knowledge about the test distribution, unlike many existing methods that consider a general distribution shift.
3.	The theoretical assumptions are hard to verify in practice.
4.	The metric ASCG only considers marginal coordinates and ignores multivariate conditional failures. Also, the effect of M is not analyzed. Furthermore, I believe the authors should compare the methods using the WSC metric as well.

### Questions
### Questions
1. In line 212, the paper suggests limiting the sensitivity of the score to changes in y conditioned on x. However, shouldn’t we want the score function to remain responsive to variations in y given x? Could the authors clarify this point?
### Minor comments
1. The worst slab coverage was first introduced in “Knowing what You Know: valid and validated confidence sets in multiclass and multilabel prediction” by Cauchois et al.

### Soundness
3

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
3

### Summary
Suppose we have procedure that generates prediction sets $C$ such that $P_{(X,Y) \sim P_{XY}} (Y \in C(X) \mid X=x) = 1-\alpha$ for all $x$ (CQR, for example, is one way to approximately achieve this). In general, $C$ will no longer have this $1-\alpha$ conditional coverage if instead $(X,Y) \sim Q_{X,Y}$. Now suppose we have invertible mappings $f_X$ and $f_Y$ such that for $(X,Y) \sim Q_{XY}$, we have $(f_X(X), f_Y(Y)) \sim P_{XY}$. Then, for $(X,Y) \sim Q_{XY}$, we can construct the conditional-coverage set in the mapped space and then map back to the original space, inheriting the conditional coverage property from the mapped space.

Theoretically, the paper provides an upper bound on the (integrated) conditional-coverage gap between $P_{X,Y}$ and $Q_{X,Y}$ in terms of the Wasserstein distance between $P_{XY}$ and $Q_{XY}$. This is useful in the multi-source domain generalization setting, as it suggests an optimization objective for learning the mapping (normalizing flow) that is in terms of the Wasserstein distance. The method is validated on several synthetic and real world datasets and the proposed method is shown to decrease the average-slice coverage gap.

### Strengths
The problem the paper solves of maintaining conditional-coverage under distribution shift is an interesting one, and the proposed solution is also conceptually interesting.

### Weaknesses
The writing is severely lacking and does not do justice to the underlying ideas, which are nice but currently presented in an inaccessible way. This is true at a micro level (typos, grammar mistakes, and weird/ambiguous wording) and also at a macro level (the information is not presented in an optimal way). I will elaborate on a few places for improvement, but not comprehensively. I encourage the authors to spend time checking grammar and thinking about ways to further simplify the presentation.

First, as context: I am quite familiar with conformal prediction but have very limited knowledge of distributional robustness. Ideally, your paper should be accessible to people like me. 

* Line 40: depend on *the* specific test input
* Line 49-50: “Various upper bounds .. proposed to maintain marginal coverage guarantee” — what do you mean by this? I think perhaps it is not so much “maintain” but rather to derive $\epsilon$ in $P(Y \in C(X)) \geq 1-\alpha-\epsilon$
* Line 85-94: As someone with no distributional robustness background, I got nothing from these paragraphs. Is there a way to briefly provide either more of the DR background that is needed or to describe this in a more intuitive way?
* Line 95: I don’t know what “To enhance expressiveness” means here
* Line 97: “sale prediction” -> “sales prediction”?
* Line 100: The results *show*
* Line 116: independent *from*
* Line 119: “Theoretically” -> “Formally”
* Line 125: “almost inaccessible” -> not practically achievable
* Line 164: The double subscript in $F_{P_{V|x}}$ is a lot. Can you just write $F_P$?
* Definition 1: I would consider moving this, or something like it, to the introduction. If you want to talk about Wasserstein distance in the introduction, it should be defined (as least informally)
* Line 199: 2-product *metric*
* Eq 15: I spent a while trying to figure out what $\bar{X}_{n+1}$ and $\bar{Y}_{n+1}$ had been defined, only to realize that you were defining the output of the mapping to be that. I would reorder this and write $(\bar{X}_{n+1}, \bar{Y}_{n+1}) := f_(\theta)(…)$
* Line 255-257: These requirements on $f_{\theta}$ are not very clear. Can you also write it out mathematically?
* Line 286: *has* the conditional guarantee 
* Line 304: There is an extra space before the first word
* Sec 4.2: What is the intuition for why the Gaussian noise helps? 
* Line 342: You should mention that this is the expectation assuming $\lambda$ is uniform on the simplex
* Line 345+: I believe “calibration set” is meant throughout and not “training set”

My overall comment about the Introduction is that there is too much technical content/jargon that should be explained more carefully if there is space; otherwise, it would be better to provide more high-level/intuitive summaries. I also found Figure 1 quite hard to follow.

### Questions
* How do you ensure invertibility of the normalizing flow? More background on distributional robustness would be useful throughout the paper.
* It is fine that you use average-slice coverage gap as your main metric, but I would still like to see the results for worst-slice coverage, as this is the metric used in previous papers. Do you have these results in the Appendix somewhere? 
* The theory is described in the case where we have a calibration distribution $P_{XY}$ and a single test distribution $Q_{XY}$. Does your method apply to this setting? How does it perform? 

Overall, I think the paper has a lot of promise and I would be happy to increase my score if these weaknesses can be addressed, especially the writing quality.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a distributionally robust framework for conditional conformal prediction to maintain valid conditional coverage guarantees under distribution shifts. The authors first introduce the Conditional Coverage Gap (CCG) to quantify coverage robustness at individual test points and its expectation, the Integrated Coverage Gap (ICG), as an overall robustness measure. Theoretically, they derive an upper bound for ICG using the Wasserstein distance between the calibration and test joint distributions, revealing how data-level shifts propagate to the conformal score space. Motivated by this bound, they design a Branched Normalizing Flow (BNF) that learns an invertible mapping between the test and calibration distributions via Wasserstein minimization. Its branched structure enables separate transformation of inputs and labels, allowing test-time input normalization without the true label. An Augmented BNF variant enhances expressiveness using Gaussian noise.

### Strengths
It introduces a theoretically grounded framework for achieving distributionally robust conditional coverage in conformal prediction, formalized through new metrics—the Conditional Coverage Gap (CCG) and Integrated Coverage Gap (ICG)—and linked to the Wasserstein distance between distributions.

### Weaknesses
The paper appears to mix many concepts including adaptivity of prediction sets, conditional coverage, and the shift of conditional distributions. While these elements may share some qualities, they are fundamentally different and often can be solved in distinct ways. For example, the motivation proposing IGG is $F_{P_{V|x}}(\tau(x))$ is different from $F_{Q_{V|x}}(\tau(x))$ when $P_{V|x} \neq Q_{V|x}$, but even if there is no gap between $F_{P_{V|x}}(\tau(x))$and $F_{Q_{V|x}}(\tau(x))$, it doesn’t mean that the conditional coverage is guaranteed. And even if $P_{V|x} = Q_{V|x}$, there is also not conditional coverage guarantee. Besides , there is a problem in mathematics that $\tau(x)$is not only depended on $X_{n+1} = x$, but depended on the calibration data, so the probability of (3) takes over the product measure of test data and calibration data. In this case, $P(s(X_{n+1}, Y_{n+1} \leq \tau (x))|X_{n+1} = x)$ $\neq F_{Q_{V|x}}(\tau(x))$ and $P(s(X_{n+1}, Y_{n+1} \leq \tau (x))|X_{n+1} = x)$ $\neq F_{P_{V|x}}(\tau(x))$.

### Questions
1. About  Augmented BNF, how does $\varepsilon$makes an effect on $f_{\theta_{Y}}^{aug}(y;\varepsilon)$? 
2. How does Theorem3 suggest a surrogate objective for Augmented BNF by $\sum_{k=1}^{K} \lambda_{k}W(P_{XY}, f_{\theta}^{aug},$ #$D^{k}_{XY}).$  
3. Why Algorithm 1 loop N epochs?
4. please supplement the detailed experimental settings such as the form of $f_{\theta_Y}(x, \varepsilon)$ and your datas' distribution.
5. What is the relationship between CCG and conditional conformal prediction? Why do you bound it?

### Soundness
2

### Presentation
2

### Contribution
2
