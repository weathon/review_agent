# Adaptive Proximal Gradient Optimizer: Addressing Gradient Inexactness in Predict+Optimize Framework

- Decision: Reject
- Scores: 1, 3, 1

## Abstract
To achieve end-to-end optimization in the Predict+Optimize (P+O) framework, efforts have been focused on constructing surrogate loss functions to replace the non-differentiable decision regret. 
While these surrogate functions are effective in forwarding training, the backpropagation of the gradient introduces a significant but unexplored problem: the inexactness of the surrogate gradient, which often destabilizes the training process. To address this challenge, we propose the Adaptive Proximal Gradient Optimizer (AProx), the first gradient descent optimizer designed to handle the inexactness of surrogate gradient backpropagation within the P+O framework. 
Instead of explicitly solving proximal operations, AProx uses subgradients to approximate the proximal operator, simplifying the computational complexity and making proximal gradient descent feasible within the P+O framework. We prove that the surrogate gradients of three major types of surrogate functions are subgradients, allowing efficient application of AProx to end-to-end optimization.
Additionally, AProx introduces momentum and novel strategies for adaptive weight decay and parameter smoothing, which together enhance both training stability and convergence speed.
Through experiments on several classical combinatorial optimization benchmarks using different surrogate functions, AProx demonstrates superior performance in stabilizing the training process and reducing the optimality gap under predicted parameters.

## Human Reviews

## Human Reviewer 1

### Rating
1

### Rating Number
1

### Confidence
4

### Summary
The "predict then optimize” (P+O) framework is a two-step approach for decision-making in scenarios where optimization is dependent on uncertain data. First, a predictive model (e.g., neural network) estimates unknown parameters or outcomes (e.g., demand, prices, costs) based on historical or contextual data. Next, using these predictions as inputs, an optimization model (LP solver) determines the best decision according to an objective function (e.g., maximizing profit, minimizing cost) under given constraints.

The framework assumes that the true prediction parameters are available when training the predictive model. However, training with direct supervision on these parameters (e.g., with squared loss) ignores the downstream performance metric -- the regret. Unfortunately, regret is not differentiable, or the gradients are zero, which does not allow end-to-end training. Therefore, several methods proposed surrogate losses (like SPO+) or surrogate gradients (IMLE, CMAP, DBB, NID - as denoted in the paper).

The paper under review proposes an optimizer designed explicitly for the P+O framework that should utilize the surrogate gradients better than existing general optimizers. The method is inspired by proximal gradient descent and utilizes existing ingredients like momentum, adaptive lr, and smoothing.
The work claims some convergence guarantees in the convex case with a bound on convergence rates and empirically compares them to existing popular optimizers.

### Strengths
The paper tries to tackle an important problem in the popular P+O framework. Proving convergence is not straightforward, even in a simple setting with exact gradients.

### Weaknesses
The paper actually does not contain what promises. It seems that it actually only adds a squared loss on costs (prediction parameters) to an existing surrogate (or, equivalently, adds a gradient of squared loss to an existing surrogate gradient) in an obfuscated way and then basically uses a custom version of Adam optimizer and weight decay.

Specifically, It recalls the proximal gradient descent (Eq 8) but does not use it ("In practice, computing the proximal operator $prox_{\eta R}$ can be impractical in P+O problems. Instead, we utilize the existing surrogate gradient..."). Equation 9 then reveals that this surrogate with the gradient of $\ell^2$ norm is used.

Next, Theorem 1 is not proven correctly (and probably does not hold in this form):
- Equation 13 in the statement contains constant $\delta$, which is not quantified and is mentioned only in Eq. 5 in section 2.2. Here it requires that the bound is uniform in $\hat c$. It is mentioned that it is non-negligible. Indeed, the true gradient is always zero or undefined (since it is the gradient of the solution of an LP, i.e., of a piecewise constant function). Therefore $\delta=\sup_{\hat c}\|g(\hat c)\|$, where $g(\hat c)$ is the surrogate gradient.
- It is not clear, what $d_k$ is, since $\nabla R(\hat c_k)$ is not uniquely defined.
- l.834: The inequality does not hold (for instance, take $\eta$ close to $1/L$, then LHS is close to zero, but RHS is negative (close to $-\delta/L\|d_k\|$))
- L836: 'Higher order terms' $L\eta^2\delta^2$ cannot just 'be neglected' as there is no limit taken. Also, here it is incorrectly assumed that $\delta$ 'is small.'

The proof of Corollary 2 is wrong.
- Equation 44 does not imply that the sum converges. Consider for instance the sums $\sum_{k=1}^N 1/k -\sum_{k=1}^N 1/\sqrt k\le 0$, they both diverge.
(I skipped reading the proof of the next corollary)

The paper is full of incorrect or inexact statements:
- l.67 "The problem of gradient inexact caused by the agent function for P+O under the end-to-end framework has not been emphasized, and research is lacking." The above-mentioned papers are devoted to exactly this.
- l.36 "end-to-end approaches are also an emerging topic in the decision-making process."
2. Inexact Gradient Challenge in P+O Framework
- “The existence of errorbound can mislead the direction of descent, which will eventually lead to the problem of unstable or non-convergence of the training process.” The true gradient is always zero (or nonexistent); hence any informative descent direction will actually increase this delta. Therefore, there is no connection between this delta and some unstability or non-convergence (or maybe it is completely opposite, that for convergence, it is required delta to be large.)
3.  Adaptive Proximal Gradient Optimizer (AProx)
- l.165 "$R(\hat c)$ is not trivial". What does it mean?
- l.172 "The number 1/2 as coefficients of $f(\hat c)=\tfrac12|\hat c-c|^2$ is to avoid its excessive influence on the  gradient of the composite function." No, it is to avoid unnecessary constant 2 in the proximal gradient step.
- l.188 "This approach effectively integrates the proximal operator implicitly and allows us to proceed without its explicit computation." No. it just ignores the proximal map completely.
4. Theoretical Convergence Analysis
- l.240 "It is worth noting that Lemma 2 rests on the fact that R(·) is a convex function. In the solution approaches of P+O, most of the constructed surrogate functions can satisfy convexity." It is not true that "CMAP...involve convex functions" does not imply it is convex. "DBB uses linear interpolations..." (this is correct) but does not ensure that the result is convex (which is not, in general), similarly for IMLE.

It uses nonstandard or misleading terminology:
- l.59 "agent function" and "agent gradient" for surrogate loss/gradient.
- l.64 "discovergence"
- It often uses the term (and notation) 'gradient' for objects that are not, in fact, real gradients but surrogate ascent direction.
- "training rounds" (l.352 ), "step size" (l.367) and "calendar hours" (l.506) are used instead of epochs
- l. 80 "We ... give an inference on the rate of descent."
- l.74 "We propose the inexact surrogate gradient problem"
- l.77 "the optimizer, which is improved on the proximal gradient."
- l.156 "to address the inexact gradient challenge in Predict+Optimize (P+O) challenge"
- l.170 "we used the l2 paradigm term for the prediction error"

Experiments
- I am not able to understand the setting of the benchmarks in Tables 1, 3 and 5. It is claimed that "Table 1 shows the step size and training time per epoch required for convergence when training with several different regrets." I guess that "regrets" means "surrogates" and refers to one of the methods of IMLE, NID, CMAP, SPO, or DBB. However, it is not described how the statistics were calculated. Also, the std in the tables is so large that no conclusions can be drawn from it.
- I do not understand the experiment setup. The metric used (relative optimal gap) measures the performance of the trained model and not the optimizers. However, the training is not mentioned. Next, it is not clear how significant the results are. No statistical testing or even a std was reported.

Overall
- The paper proposes an enhancement to optimization within the P+O framework but lacks clarity and rigor in both theoretical claims and experimental evaluation.
- The main contribution—adding a squared loss to an existing surrogate with a custom optimizer—is presented obfuscated, does not bring any novel insights in the field, or does not help to understand the existing methods better.
- Theoretical issues arise, especially in Theorem 1 and Corollary 2, where the proofs contain major flaws.
- Misleading terminology and insufficient setup description and statistical analysis in experiments limit the work’s impact.

### Questions
I have no questions

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work proposes to use an adaptive proximal gradient optimizer in order to address issues arising in inexact gradient computations in predict+optimize works. The idea is to first add a smooth function $f$ to the regret $R$. Next, this work integrates adaptive learning rate, momentum, and parameter averaging in the minimization of $\Phi = f + R$.

### Strengths
1) The proposed work is interesting and aims to tackle a well-known issue arising in the non-differentiability of the loss function in Predict & Optimize.
2) The numerical experiments show promising results for the proposed approach. 
3) The introduction and related works are well-written.

### Weaknesses
Major Comments:

1) The proof of the main theorem is incorrect. The inequality in Line 772 does not necessarily hold as a result of Line 765. The authors should revisit the proof and correct these details. 
2) The paper claims that this paper uses a proximal update. However, the update is given by $\hat{c}_{k+1} = \hat{c}_k - \eta (\nabla f(\hat{c}) - g(\hat{c}))$, where  $g$ is an inexact gradient estimate of the non-smooth loss/regret term. The authors claim this is "implicitly a proximal update" in line 188, but this resembles a subgradient descent instead. It would benefit the paper if the authors could further elaborate on how this update relates to or a proximal update, or revise their claims if they cannot justify this connection.

3) The main theorem relies on $R(\hat{c}) = c^\top(z^\star(\hat{c}) - z^\star(c))$ being convex in $\hat{c}$. However, it is not obvious that the regret is convex. This paper would benefit if it either provides a proof of convexity for the regret function, or discuss the implications if this assumption does not hold and how it might affect the validity of the results.

Minor Comments:

1) The authors should update their references. For example, "Differentiation of Blackbox Combinatorial Solvers" is not cited properly, as it is already a published article. 
2) The paper uses $\nabla R$ and $g$ interchangeably. However, $\nabla R$ is the true gradient of the regret (and assumes $R$ is differentiable), whereas $g$ is an approximation. The authors should update this in, e.g., Line 5 of the pseudocode and in line 259.  
3) Line 052: "non-differentiate" -> "non-differentiable"
4) Line 068: "gradient inexact" -> "inexact gradients"
5) $R(\hat{c})$ is never explicitly written. It would make the paper more readable if the authors explicitly defined it in Section 2
6) Line 161: "introduces" -> "introduced"
7) Line 233 "approachfocuses" -> "approach focuses"

### Questions
1) How do you tune hyperparameters $\eta$, $\lambda$, $\beta_s, \beta_m, \beta_p$? 
2) What does Table 7 show? Entries are denoted by "yes" and "no". There is no description in the caption. It is also not referenced in the main draft. The authors should remove unreferenced tables or reference them in the main draft. 
3) Line 368 states that Table 1 shows training with "several different regrets". However, Table 1 only shows optimization algorithms and not regrets. Is this line referencing the wrong table?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
1

### Rating Number
1

### Confidence
3

### Summary
This paper addresses the predict and optimize framework, which utilises learning algorithms to predict parameters for optimization problems in an end to end fashion. Unforutnately, incorporating the optimization stage into the problem results introduces nonsmoothness into the objective. The paper addresses this issue by utilizing a proximal framework. The authors analyse the theoretical convergence and practical performance of their algorithm.

### Strengths
- The topic of the paper is interesting.
- The paper provides a review of related works.

### Weaknesses
There is a major error in the proof of theorem 1: on line 772 a lower bound is incorrectly combined with an upper bound. Since this is the basis for the main convergence result, this error compromises the theoretical results presented.  

I believe there is another error in the proof of corollary 2 (883-886). The sequence $||d_k|| $ need not converge to zero. For example, consider, $||d_k|| = (\delta\eta)/c$ which satisfies (44) but clearly does not converge to zero.

I also believe that there are major flaws in the specification of Algorithm 1. For example on line 3 the $ \hat{c_k} = \hat{c}(\theta_k) $, while in line 10 $ {\hat{c}}_{k+1} $ is computed as an update sequence based on the gradient. On line 12 a set of smoothed $ \tilde{\hat{c}}_k $ are computed but not utilised (so far as I can tell). 

Another concern I have with is the smoothing function selected by the authors. To compute the gradient of $f(\hat{c}) = \frac{1}{2} || \hat{c} - c||^2$ with respect to $\hat{c}$ requires knowledge of the "true cost parameters", which precludes practical implementation. 

Additionally, paper is significantly hindered by the quality of the writing with many awkward and confusing sentences, confusing notation and typos. The paper is difficult to follow due to theses issues. For example, in section 2.1 equation (3) is stated with no relation to the previous paragraph and (4) is stated with no discussion. There are issues like this in almost every section of the paper.

### Questions
See the weaknesses section. If the authors can clarify the theoretical concerns and substantially improve the quality of the text, I will be happy to take a second look at the paper.

### Soundness
1

### Presentation
1

### Contribution
1
