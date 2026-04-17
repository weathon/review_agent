# SIM-Shapley: A Stable and Computationally Efficient Approach to Shapley Value Approximation

- Decision: Reject
- Scores: 4, 2, 2, 8

## Abstract
Explainable artificial intelligence (XAI) is essential for trustworthy  machine learning (ML), particularly in high-stakes domains such as healthcare and finance. Shapley value (SV) methods provide a principled framework for feature attribution in complex  models but incur high computational costs, limiting their scalability in high-dimensional settings. We propose Stochastic IterativeMomentum for Shapley Value Approximation (SIM-Shapley), a stable and efficient SV approximation method inspired by stochastic optimization. We analyze variance theoretically, prove linear $Q$-convergence, and demonstrate improved empirical stability and low bias in practice on real-world datasets.
In our numerical experiments, SIM-Shapley reduces computation time by up to 85\% relative to state-of-the-art baselines while maintaining comparable feature attribution quality. Beyond feature attribution, our stochastic mini-batch iterative framework extends naturally to a broader class of sample average approximation problems, offering a new avenue for improving computational efficiency with stability guarantees. Code is publicly available at https://anonymous.4open.science/r/SIM-Shapley.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents SIM-Shapley (SIM-SHAP) as a novel approximation method to estimate Shapely values. SIM-SHAP is quite related to many amortization and approximate methods relying on the weighted least squares representation of the Shapley value. SIM-SHAP extends the well known KernelSHAP approxmation method and Unbiased KernelSHAP (Covert and Lee, 2021) with an exponential moving average throughout the sampling and Shapley value computation procedure. The paper concludes by evaluating the approximation quality of SIM-SHAP compared to state-of-the-art methods. All in all the empirical evaluation shows that SIM-SHAP does not set the new state of the art but is comparable to it.

The following contains the **references** used throughout the review:
- [1] https://inria.hal.science/hal-03414720v1/document
- [2] https://arxiv.org/pdf/2506.11849
- [3] https://ojs.aaai.org/index.php/AAAI/article/view/29225
- [4] https://www.sciencedirect.com/science/article/abs/pii/S0305054808000804

### Strengths
- **Important Research:** Computation of Shapley Values is relevant for more and more application areas most prominently Explainable AI. Black box (game-agnostic) estimation methods are important with better methods proposed every year. Therein, this work contributes to an important research field. The most important aspect of the contribution is the dynamic nature of the estimation, and that it natively supports estimation of current estimation progress (which is often a positive argument brought in favor of [4]).
---
- **Well Written and Good Presentation:** The paper is well written and clear to follow. The contribution is clearly presented and well organized. The presentation good. 
---
- **Noation:** The mathematical notation and presentation is good.
---
- **Good Baselines:** The paper compares itself already to the most important baseline methods. While the current SOTA [2] is missing, the current selection is already quite well chosen. Some improvements can be done.
---
- **Code:** I greatly appreciate the submission of the source code and the quality of the code. Thank you for including the demo notebooks they are interesting to go through.

### Weaknesses
- **Empirical Evaluation:** While the empirical evaluation is generally well done, it would be nice to also include some of the methods described in the below point in the evaluation such as [1] and [4]. The comparison with [4] is interesting since it is usually also very efficient but a bit more limited in terms of performance and thus serves as a good baseline how hard the problems are and how impressive the gains of improvements are. I also do not like that the empirical evaluation does not compare against *real ground truth* values at least in a couple of cases, but estimates them with KernelSHAP running longer. Real ground truths can be easily achieved by computing via brute force for datasets with a moderate number of features (16 features needs approximately $2^{16} \approx 64k$ model calls). This should be doable for some settings.
---
- **Missing Related Work:** While the paper compares itself to modern methods to estimate the Shapley value empirically, the related work is not very well prepared in the current version of the manuscript and the majority of the related work is moved into the appendix. The whole SIM-SHAP method seems to me very related to [1] and potential follow-up works on this. I feel like this should be described in the manuscript. Some other impactful Shapley value estimation methods ([3] and [4]) are also missing. Most prominently, the RegressionMSR method proposed in [2] would need to be discussed in the work, since it presents the current state of the art in terms of Shapley value estimation. Since the paper was only released rather recently, I don't think an empirical evaluation is necessary (of course interesting) but delineating against is would definitely improve the work.
---
- **Doubts about LeverageSHAP usage:** I have my doubts about LeverageSHAP not being able to be used on classification tasks and the notion of it being imputer-dependent since LeverageSHAP is a black-box estimation method. While the *implementation* of LeverageSHAP may be limited to these tasks, it should be easily transferrable. Also see question 1.
---
For me this work is **quite borderline** (in its current state sitting a bit on the reject side), but depending on the resolution (or partial resolution) of my concerns, I think **this paper could be accepted.**

### Questions
### Question 1
You write
> SIM-Shapley uniquely provides both game and imputer agnosticism—capabilities absent in existing methods

and 
> Leverage- SHAP (Musco & Witter, 2025) employs statistical leverage scores to improve sampling efficiency but is limited to mean-value imputation and specific task types

How is LeverageSHAP **not** imputer invariant? LeverageSHAP is basically KernelSHAP but a smarter coalition selection / sampling. The imputer used behind the value function does not influence the applicability of LeverageSHAP, no?

---
### Question 2
How does your method compare to [1]?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper considers the problem of estimating Shapley values of a general value function $v$. They propose a linear regression-based method similar to KernelSHAP. Instead of solving the weighted and constrained least squares problem once, they iteratively solve a sequence of regression problems. When solving the next regression problem, they draw $m$ new samples and add a regularization term on the "moving average" solution of the regression problems so far.

### Strengths
N/A

### Weaknesses
Big:

1. I think the biggest structural issue in this paper is how they discuss and compare to prior work. Across the discussions and experiments, they consider different subsets of the estimators FastSHAP, SimSHAP, LeverageSHAP, and KernelSHAP. The first issue is that they selectively compare to these algorithms e.g., they'll sometimes ignore LeverageSHAP or FastSHAP depending on the experiment. The bigger issue is that they don't consider other SOTA estimators which include PermutationSHAP, and (two recent methods) RegressionMSR and ProxySPEX.

2. The paper ignores fundamental properties of estimators, or says things that are just incorrect about them:
    * They claim prior estimators can't be used for general value functions $v$, but, except for methods like TreeSHAP (which isn't mentioned in the paper), this is just false. Basically all near-SOTA Shapley value estimators are agnostic to the value function. (I think they know this e.g., they present a "variant" of their estimator for global explanations in Appendix C but the only change is one line about how the value function $v$ is computed.) It is true that *implementations* of these algorithms are generally tied to specific value functions but, I would say a) the onus for modifying the standard implementation is on researchers if they want experiments on a new kind of value function, and b) the shap-iq library is extracting the value function logic to an arbitrary class that makes it easy to switch which value function is used.
    * They treat KernelSHAP and LeverageSHAP as different algorithms. In reality, they're the same algorithm except a) the sampling distribution is different, and b) LeverageSHAP solves a constrained regression problem whereas KernelSHAP uses very large weights on the empty set and full sert. The fact that the experiments in the current paper have such a big gap in performance between these two methods indicates something is strange about the implementation.
    * FastSHAP is fundamentally a different *kind* of estimator because it learns a neural network to predict the Shapley values with respect to *multiple* value functions. So if you're only estimating the Shapley value of a single value function, it doesn't really make sense to use FastSHAP. (To be fair, this is mentioned in passing in a runtime table in the experiments.)

3. As for their algorithm, I'm confused bordering on concerned. Instead of solving one regression problem, they iteratively solve a sequence of algorithms. Intuitively, I don't see the value in distributing the total budget into a sequence of worse solutions than just solving the regression problem once (but hey, I could be wrong about this). The bigger concern I have is that, as written, Algorithm 1 is given a budget of $m$ (like the other estimators), but then makes $m$ independent samples for each of the $T$ iterations. Effectively, this algorithm is getting to see $T * m$ evaluations of the value functions, whereas the algorithms they compare to gets only $m$ evaluations. If this is how the experiments are implemented, they're deeply unfair because SimShapley is getting way more access to $v$ than the other algorithms.

4. As mentioned earlier, their experiments are inconsistent in terms of which algorithms they compare to. And they present performance on different datasets in different ways. Pessimistically, this would indicate cherrypicking of experimental results. My **strong** suggestion is to standardize their experiments: Plot bias as a function of sample size in one big plot with one subplot per dataset. If you want, make a similar plot for time, and a similar one for "global" value functions (again, the only difference here is how $v$ is defined so, even if it means modifying some implementation code, you should compare to *every* Shapley value estimator that you mention). This would be a) more visually appealing, and b) much easier to see overall performance of each estimator in a fair way.

Here are the recent (very good) Shapley value estimators you should compare to:

ProxySPEX: https://arxiv.org/abs/2505.17495

RegressionMSR: https://arxiv.org/abs/2506.11849

Minor:

* There's already an algorithm called "SimSHAP" so calling this one "SimShapley" seems confusingly similar. Especially because this algorithm seems to be quite different from "SimSHAP".

### Questions
* Is there a typo in Equation 7b? I.e., should it be $\delta^{(n+1)} = t \beta^{(n)} + (1-t) \delta^{(n)}$.

* In Table 2, you run KernelSHAP with $m=64$ and SimShapley with $m=64$ and some $T$ (I assume greater than 1). How many evaluations does each algorithm get? Is it that KernelSHAP gets $64$ and SimShapley gets $64T$? What do you set $T$ to?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces SIM-Shapley (Stochastic Iterative Momentum for Shapley values), a new method for approximating Shapley values, which are a key tool for feature attribution in explainable AI (XAI). The primary challenge with Shapley values is their exponential computational complexity, making them impractical for many real-world applications. The authors propose SIM-Shapley to address this by reformulating the Shapley value calculation as a stochastic optimization problem.

### Strengths
1.	The proposed method covers both local and global explanations.
2.	Clear reformulation of KernelSHAP as a constrained stochastic optimization with a simple EMA update and a closed-form per-iteration solution. 
3.	The method stays model-agnostic.

### Weaknesses
1.	The authors claim that their method fundamentally re-conceptualizes SV computation. However, the core of their process is a series of mature stochastic optimization techniques.
2.	The early stopping method (Eq.11) is heuristic and has no sensitivity analysis to the parameter epsilon.
3.	The choice of the parameter xi in Eq. 12 is crude and lacks analysis.
4.	The authors claim that the method can be used for Shapley Interactions, but provide no experiments to demonstrate this.

### Questions
1.	Line 159: The constraint in equation 5 should preserve the efficiency property of SV.
2.	The main convergence theorem targets the SIM-Shapley fixed point $\beta^*$ that depends on $\lambda$, not the unregularized KernelSHAP target $\beta$. Hence, the theory guarantees a fast approach to $\beta^*$rather than a fast approach to $\beta$. Figure 1 suggests a small bias for small λ, but there is no explicit bound for $‖beta^*−\beta‖$ or a rate as $\lambda \to 0$.
3.	The proof of Theorem 1(Appendix B.1) seems to rely on a key assumption $mathrm{Var}(\delta^{(j)})\le \mathrm{Var}(\beta)$, which is not always right.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes SIM-Shapley, a Shapley value approximation method based on stochastic iterative momentum. Its core idea is to reformulate Shapley value computation as a stochastic optimization problem. The key innovations include: (1) achieving linear Q-convergence through Exponential Moving Average (EMA)-based updates and adaptive mini-batch sampling, with a variance contraction rate of $(1-t)^2$; (2) introducing three stability mechanisms: $\ell_2$ regularization, negative-sampling detection, and initialization bias correction; (3) supporting both local and global explanation modes while being game-agnostic and imputer-agnostic. Experiments show that SIM-Shapley is 60%-85% faster than baselines such as KernelSHAP and SAGE across classification, regression, image, and clinical tasks, with a 10%-15% reduction in bias.

### Strengths
1. The paper is well-written, with a clear structure that allows readers to easily follow the authors' reasoning.
2. The comparison between SIM-Shapley and other Shapley value computation methods is thorough and clear, enabling readers to readily grasp the paper's contributions.
3. Relevant experiments effectively demonstrate the superiority of the proposed method, particularly in accelerating Shapley value computation.
4. The paper addresses interpretability, a critical issue in deep learning models. By focusing on accelerating Shapley value computation, this work holds significant importance—it facilitates the adoption of deep learning models in high-reliability industries.

### Weaknesses
The paper is relatively comprehensive, and there are no major fundamental issues. However, two minor points require attention:
1. **Paper Formatting:** The authors are requested to review the paper's formatting. For instance, on Page 21 of the supplementary materials, some figures/icons clearly exceed the paper's margins.
2. **Evaluation Metrics:** The paper primarily uses the error between estimated Shapley values and the ground truth to evaluate performance. Readers are curious about how the proposed method performs under a broader set of evaluation metrics, such as Deletion/Insertion and mu-fidelity.

### Questions
Refer to the "Weaknesses" section above.

### Soundness
3

### Presentation
3

### Contribution
3
