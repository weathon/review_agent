# Provably Accurate Shapley Value Estimation via Leverage Score Sampling

- Avg Score: 7.33
- Decision: Accept (Spotlight)
- Scores: 8, 6, 8

## Abstract
Originally introduced in game theory, Shapley values have emerged as a central tool in explainable machine learning, where they are used to attribute model predictions to specific input features. However, computing Shapley values exactly is expensive: for a model with $n$ features, $O(2^n)$ model evaluations are necessary. To address this issue, approximation algorithms are widely used. One of the most popular is the Kernel SHAP algorithm, which is model agnostic and remarkably effective in practice. However, to the best of our knowledge, Kernel SHAP has no strong non-asymptotic complexity guarantees. We address this issue by introducing *Leverage SHAP*, a light-weight modification of Kernel SHAP that provides provably accurate Shapley value estimates with just $O(n\log n)$ model evaluations. Our approach takes advantage of a connection between Shapley value estimation and agnostic active learning by employing *leverage score sampling*, a powerful regression tool. Beyond theoretical guarantees, we find that Leverage SHAP achieves an approximately 50% reduction in error compared to the highly optimized implementation of Kernel SHAP in the widely used SHAP library [Lundberg & Lee, 2017].

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper modifies the Kernel SHAP algorithm, which approximates the Shapley value and model-agnostically quantifies the role of each feature in a model prediction. The Kernel SHAP did not enjoy theoretical convergence guarantees, and the authors propose a modification of this algorithm that offers a theoretical convergence guarantee and similar numerical performance.

### Strengths
This paper is very well written, introduces the context of their work beautifully, and provides both a theoretical and practical contribution to the field.

### Weaknesses
A weakness is that it might feel niche, but as a non-specialist in interpretable AI, I cannot judge the importance of the Shapley values. If this information is important, then the author's contribution is quite important because it removes some level of heuristic thanks to their theoretical contribution.

### Questions
N.A.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper develops a computationally efficient method for approximating Shapely values, which are useful in interpretable machine learning. The proposed method provides a modification of the well-known Kernel SHAP method, with additional non-asymptotic theoretical convergence guarantees and better empirical performance. The authors use a modified sampling without replacement approach to optimize their method and report experiments with ablation studies for the optimizations.

### Strengths
- Estimating Shapley scores accurately and efficiently is an important problem in explainable machine learning. The paper provides a theoretically principled approach for this problem.
- The approach seems to outperform Kernel SHAP and optimized Kernel SHAP baselines in the experiments.

### Weaknesses
- The main theoretical result (Theorem 1.1) is somewhat unsatisfactory as it does not directly compare the true and estimated Shapely values. The authors address this via Corollary 4.1, but it has a non-intuitive $\gamma$ term which is can be large and makes the approximation guarantees weaker. Are there conditions under which $\gamma$ is guaranteed to be small? This would better help understand the limitations of current theoretical results.
- The experiments could include more baselines like (Jethani et al., 2021), (Mitchell et al., 2022b), and (Yang & Salman, 2019) for a more comprehensive comparison with the state-of-the-art.
- The technical novelty in proving the new theoretical results also appears to be limited. The main result seems similar to the active learning guarantee Theorem 1.1 of Shimizu et al. ICLR 2024, and it is not clear what additional technical insights are needed to develop the current result. A discussion of novel technical insights needed to develop the current result would be helpful.

Typos:
- Line 155, Line192 finte, Line 255 contrained
- References are incorrectly bracketed

### Questions
- Which algorithm in the literature does the Optimized Kernel SHAP in the experiments correspond to?
- Do the theoretical guarantees for leverage SHAP continue to hold when optimizations like paired sampling, and sampling without replacement are not applied? Is there a difference in the guarantees with and without the optimizations?
- In Table 2, Leverage SHAP w/o Bernoulli sampling seems to outperform in some cases. Is there a way to understand this? An analysis or discussion around this would be very useful to better understand the method's behavior.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper provides a new algorithm for approximating Shapley values with provable guarantees. Shapley values have widespread application in ML as a way of formalizing individual feature contributions to a model's final prediction, although this paper considers the fully general setting in terms of a generic value function $v : 2^{[n]} \to \mathbb{R}$ (where $n$ is the number of features or "players"). 

The algorithm builds on a known way of formulating the Shapley value as the solution of a certain $2^n$-dimensional linear regression problem. The widely used "Kernel SHAP" algorithm approximates the solution by essentially subsampling the rows of the design matrix in a certain way. But a more principled approach is to subsample according to leverage scores, a well-studied concept in statistics. The catch is that naively, leverage score take time polynomial in the size of the design matrix (so $2^{O(n)}$) to compute. The key idea the authors exploit to get around this is that for the specific Shapley design matrix, the leverage scores can actually be written down in a simple closed form.

This allows them to efficiently solve the underlying regression problem and carry over known guarantees from the leverage score toolbox. Specifically, they show that the estimated Shapley values are close to the true Shapley values in a certain sense (both in terms of the optimum achieved and the values themselves). They show further refinements using paired sampling without replacement.

Finally, they show various experiments demonstrating that the resulting algorithm indeed outperforms the previous best Kernel SHAP algorithm in practice.

### Strengths
Shapley values are a basic and important topic in interpretable AI and beyond, finding wide application in practice. The problem of efficiently estimating them well is a very well-motivated one. This paper makes a very nice and useful contribution to this problem. The key theoretical insight of analyzing the form of the leverage scores is simple but very clever and elegant, and allows them to make use of a very well-studied toolbox in statistics (although there is still technical work to be done). It immediately feels like the "right" way to solve the problem. The resulting algorithm is theoretically sound, clean, simple, as well as effective in practice. The paper is also very clearly written, with a clear description of all the relevant context as well as clear exposition in general. I did not verify all the proofs in complete detail but they seemed correct to me.

### Weaknesses
I do not see any major weaknesses. I do think would be helpful for the authors to discuss the limitations of the Leverage SHAP algorithm a bit more (e.g. does it strictly dominate all prior algorithms?), and provide some context on what still remains open in this space (see below for related questions).

### Questions
1. It would be nice to have a fuller picture of the Shapley value problem from a broader approximation algorithms standpoint. The linear regression approach is ultimately just one approach to the problem. Do we have a sense for the best possible approximation one can hope for? Are there any computational inapproximability results?
1. Also, it's natural to ask whether additional structure makes the problem easier (e.g. for the specific setting of feature attribution). I realize this is indeed true for decision trees and such. But I am curious if the authors have an idea of a "dream result" under less restrictive but still reasonable structural assumptions, especially for feature attribution. For feature attribution for a general black box model, is the Leverage SHAP approach likely to be the best?
1. One thing that was not totally clear to me from the experiments is whether Leverage SHAP strictly dominates Kernel SHAP at every point in the running time vs accuracy graph. That is, for any fixed running time budget, is it always better to run Leverage SHAP? Theoretically, one concern could be that Leverage SHAP necessarily requires sampling $m = \Omega(n \log n)$ rows (IIUC based on Thm 1.1), whereas I believe Kernel SHAP allows you to pick $m$ arbitrarily (albeit without a guarantee). Thus perhaps for small running time budgets, maybe Kernel SHAP can sometimes be more effective than Leverage SHAP. Or perhaps Kernel SHAP is just better optimized as a practical implementation.
I think Figure 3 (sample size $m$ vs error) nearly answers this question, but my question is whether there is any subtlety in how $m$ translates to actual running time. And in general if there is any catch to the "strictly dominates Kernel SHAP" question.

Minor nit: in two places (lines 132 and 266) there is a typo: "principal" -> "principle".

### Soundness
4

### Presentation
4

### Contribution
3
