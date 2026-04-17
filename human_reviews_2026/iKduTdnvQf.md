# Managing Solution Stability in Decision-Focused Learning with Cost Regularization

- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Decision-focused learning is an emerging paradigm that integrates predictive modeling and combinatorial optimization by training models to directly improve decision quality rather than prediction accuracy alone. Differentiating through combinatorial optimization problems represents a central challenge, and recent approaches tackle this difficulty by introducing perturbation-based approximations that enable end-to-end training. In this work, we focus on estimating the objective function coefficients of a combinatorial optimization problem. We analyze how the effectiveness of perturbation-based techniques depends on the intensity of the perturbations, by establishing a theoretical link to the notion of solution stability in combinatorial optimization. Our study demonstrates that fluctuations in perturbation intensity and solution stability can lead to ineffective training. We propose to address this issue by introducing a regularization of the estimated cost vectors which improves the robustness and reliability of the learning process. Extensive experiments on established benchmarks show that this regularization consistently improves performances, confirming its practical benefit and general applicability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper analyzes gradient estimators for decision-focused learning/differentiation through discrete optimization and provides theoretical arguments that the "scale" of extra ingredients in these estimators (e.g., random perturbations) should be set appropriately relative to the scale of coefficients in the objective function.

### Strengths
Stability is a real issue in any attempt to include differentiable optimization in machine learning pipelines. Existing pipelines are prone to uneven performance depending on a number of hyperparameter choices. It makes sense that normalizing the scale of the coefficients in the optimization problem could be useful to help deal with some of these issues, or make setting other hyperparameters easier.

### Weaknesses
The bulk of the paper is spent on theoretical analyses that do not add a great deal of information compared to what is already known. It is not surprising that when the hyperparameters of existing methods are set on an inappropriate scale, their gradient estimators would no longer be tracking a useful quantity. The question is more (a) how to remedy such issues, and (b) if doing so resolves some of the overall stability issues in training with optimization in the loop. The paper makes a couple of suggestions about (a) , but it's not clear if this is addressing a large issue in the overall pipeline, or whether training instability is largely a result of other characteristics besides setting the right scale for these hyperparameters. The empirical results don't show a very consistent or large improvement from any particular strategy compared to unmodified existing approaches. Even when there is an improvement, it is often very slight (not really practically significant, and unclear whether statistically significant).

### Questions
NA

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
This paper studies the impact of solution stability on the convergence of the DFL training process. The authors consider a classical DFL setting, which involves a ML estimator used to output the cost function coefficients for an optimization problem with a fixed feasible space.

They focus on methods that work by perturbing the estimated parameters and the impact of such perturbation on the decision problem solution. Scale has a key role here: too small perturbations do not cause the solution to change and hence provide no useful information for gradient descent; too large perturbations lead to radical changes of the optimal solution and risk causing instability.
The paper also explains how a poorly chosen scale can make a number of "experience based" DFL method to become equivalent to (usually less performant) "imitation based" methods.

The authors then propose to address the issue by applying a mapping to the predicted parameters (perturbed on unaltered) that normalizes the scale of the vector. In a compact empirical evaluation, the method proves capable of improving the performance of some DFL techniques in some cases

### Strengths
I think this work makes a very good case for how controlling the scale of perturbations (or of sampling processes) can dramatically affect the behavior of DFL training methods that makes use of such idea (which are at this point many and among the best performers).

The discussion on how different classes of method become either ineffective, or collapse to solution imitation, is well done and convincing, even if somewhat informal.

I also believe that the proposed normalization technique can be useful in the considered setting, by far the most common in the DFL literature and involving decision problems with linear costs.

### Weaknesses
The key issue I see in this work is that the proposed approach does not appear to address the analyzed problem. Based on the formulation from eq. (19), the normalization mapping is applied to the parameter vector just before it is fed to the optimization process (the f mapping). In a perturbation based approach, this means that normalization would be applied to the perturbed parameters, after the scale mismatch as already done all the damage extensively documented in the first half of the paper.

The proposed normalization seems instead well suited to address a second, more widely acknowledged, issue in the considered setting, i.e. the fact that the optimization mapping is scale-invariant (as stated by the authors in Property 3).

Overall, the analysis the take most of the paper is devoted to one problem, but the proposed mitigation and the empirical evaluations are about another, different, problem.

As a secondary, but still significant, weakness, the reported improvements are not particularly large in all but one or two cases.

The remaining issues I could find are minor. Here are some of them:

* The lack of differentiability is not the real issue in the considered setting (the solution function is differentiable almost everywhere). The true problem is that the gradient is 0, and therefore non-informative from an optimization standpoint, whenever it is defined. There's confusion on this point in several places in the paper (but this is easy to fix)
* The rationale for the g mapping is extremely unclear, especially since it appears to be virtually ignored in the entire paper, including the empirical evaluation (where it is explicitly an identity function)
* The g mapping also appears to have discrete output, meaning it cannot be differentiable as stated at line 115
* The first proposed normalization maps the parameter vectors onto the surface of the unit sphere, rather than into the unit sphere (this is a good thing, actually)
* Several of the considerations at lines 169-175 appear to be true by construction, and hence do not see to add much to the discussion

### Questions
* Is the normalization mapping actually applied to the cost vector just before the optimization mapping?
* Why is the g mapping included in the formulation?

### Soundness
1

### Presentation
3

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
The paper focuses on decision-focused learning (DFL) and predicting coefficients in the objective function. The authors introduce the notion of solution stability and use this perspective to investigate different types of DFL techniques. They show that, in all examined perturbation-based DFL methods, solution stability around cost estimates is hard to predict, leading to a loss of learning signal or a shift from experience-based learning to imitation. To address this issue, they propose a regularization of the estimated cost vectors.

### Strengths
1. The writing is generally clear.
2. The viewpoint of interpreting different decision-focused learning methods through the concept of solution stability is novel and interesting.

### Weaknesses
1. The introduction and explanation of the four properties in Section 3 could be clearer; adding examples may aid understanding.
2. There are some typos—for example, inconsistent capitalization of the initial letter in “property.”

### Questions
1. DFL techniques are generally expected to outperform prediction-focused learning. However, according to Table 1, their performance is worse than that of the MSE training method, especially in the SP1 case. Could you explain this phenomenon?
2. Recently, there have been DFL studies that predict coefficients in the constraints. Can the notion of solution stability be used to analyze these works as well?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper analyzes why perturbation-based Decision-Focused Learning (DFL) methods often become unstable or ineffective when training models to predict cost coefficients for MILPs.
The key insight is that the effectiveness of perturbation-based gradient signals depends on the relative scale between the learned cost vector and the perturbations.
If this scale is poorly matched, gradients either vanish or push the model toward pure imitation, resulting in performance collapse.

To address this, the paper introduces a cost regularization strategy that controls the stability radius of the optimization mapping, thereby improving the quality of the gradient signal. Two forms of regularization are studied:
- L2-normalization of cost vectors (rn)
- Projection into a bounded L2-ball with radius κ (rp)

Experiments show that regularizing cost vectors improves training stability and decision performance across multiple discrete optimization benchmarks.

### Strengths
This paper provides a meaningful conceptual clarification and a practical normalization mechanism that helps stabilize a widely used—but often fragile—class of DFL methods. The insight linking stability radius with learning dynamics is both useful and broadly relevant.

### Weaknesses
1. I found some notations and definitions are not rigorous in the paper, see Questions.

2. The paper lacks discussion on other perturbed optimizers beyond the MILP case.

### Questions
1. The dimension of Eq. (6) does not seem correct. What is the dimension of the regret loss $L^r$?

2. $f$ is piecewise constant and hardly differentiable. Why you can still write $\nabla_\theta f(\theta)$? It does not seem well defined.

### Soundness
2

### Presentation
2

### Contribution
2
