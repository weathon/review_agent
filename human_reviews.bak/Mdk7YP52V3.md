# Understanding Pathologies of Deep Heteroskedastic Regression

- Decision: Reject
- Scores: 5, 8, 3, 1

## Abstract
Several recent studies have reported negative results when using heteroskedastic neural regression models to model real-world data. In particular, for overparameterized models, the mean and variance networks are powerful enough to either fit every single data point (while shrinking the predicted variances to zero), or to learn a constant prediction with an output variance exactly matching every predicted residual (i.e., explaining the targets as pure noise). This paper studies these difficulties from the perspective of statistical physics. We show that the observed instabilities are not specific to any neural network architecture but are already present in a field theory of an overparameterized conditional Gaussian likelihood model. Under light assumptions, we derive a nonparametric free energy that can be solved numerically. The resulting solutions show excellent qualitative agreement with empirical model fits on real-world data and, in particular, prove the existence of phase transitions, i.e., abrupt, qualitative differences in the behaviors of the regressors upon varying the regularization strengths on the two networks. Our work thus provides a theoretical explanation for the necessity to carefully regularize heteroskedastic regression models. Moreover, the insights from our theory suggest a scheme for optimizing this regularization which is quadratically more efficient than the naive approach.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies learning both the mean and the variance functions using deep neural networks. Estimation of the variance term posts additional difficulty since training may fall into two undesirable scenarios: (1) the inverse variance goes to zero, which means the heteroscedasticity is not learned, or (2) the inverse variance goes to infinity, which means the training data are overfitted.

This paper presents a categorization of the possible scenarios depending on how much regularization is applied to the mean function and variance function. For both mean and variance, there is a potential memorization vs. generalization distinction. In the 2D phase diagram, the interaction of the mean and variance functions results in 5 categories. 

Then the authors use heuristic arguments and propose numerical approximation to nonparametric
free energy, which aligns with experiments in relatively simple settings.

### Strengths
1. (main) Conceptually interesting: a richer understanding of regularization for both mean and deviation.
2. Uncertain quantification: an important topic, this paper provides some ideas about how to tame overparametrization

### Weaknesses
1. (main) Is there a sharp phase transition? This paper lacks quantitative measurement and results. It would be interesting to calculate, at least under certain simple generative models, the free energy and check if there is a first-order/second-order phase transition. 
2. (main) Technically speaking, not sure which part of the paper is innovative---for example, Eqns 7--10, are they new or semi-new (i.e., similar derivations are obtained in a different context)?  I would be skeptical that Eqns 7--10 are entirely new. Also, the proposed regularization is well-studied in the literature.
3. Data experiments are simple, but I am mostly fine with that, since this paper is mainly proof-of-concept.
4. It is a bit handwaving when transitioning from parameter norm regularization to gradient regularization
$$
\int \alpha || \nabla \hat \mu(x) ||_2^2 dx, \qquad \int \beta || \nabla \hat \Lambda(x) ||_2^2 dx.
$$
I feel that there are missing gaps between parametric models vs nonparametric models, though the idea can be understood intuitively.

### Questions
See the above section

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies heteroskedastic regression problem in the framework of field theory. By modeling parametrized neural network using continuous functions, reparametrizing regularization strength, proposing continuous regularization terms, and approximating the integral over $y$ by a single point, a computationally feasible nonparametric free energy approximating the log likelihood of deep heteroskedastic regression is derived. The reparametrized regularization strength is perceived as order parameters. The field model is solved numerically on a lattice. Abrupt change in the expressiveness of the model and the loss is observed and is interpreted as phase transition. Similar patterns also emerge when using real data and neural networks. The field-theory model implies that one-dimensional hyperparameter searching suffices.

### Strengths
A field-theory model is proposed and can explain the pathological behavior of heteroskedastic regression. The model can produce phenomena which appears in regressing tasks with various realistic data sets, indicating the insight obtained from this model is universal to some extent. This makes the results in the paper convincing. The process of deriving the field-theory model is supported by solid reasoning in general, and the experiment is performed using realistic data sets. This paper is well written in general; the figures are informative.

### Weaknesses
Although a field-theory model is proposed, little analytical result regarding the phase transition is obtained. There are small issues regarding the writing. I leave the details in Questions.

### Questions
* Below equation (7), the authors ‘consider the scenario in which the inner integral is approximated using a single MC sample’. I wonder if there is any justification for this approximation (experiment, argument, reference, etc.).
* I don’t understand the sentence below equation (9): ‘Interestingly, both resulting relationships include a regularization coefficient divided by the density of $x$.’ Does the word ‘regularization coefficient’ refer to the term with Laplace operator, which originates from the regularization term?
* Typo: on top of equation (1), (i.e., $y_i \sim \mathcal{N}(\mu_i, \Lambda_i)$), $\Lambda_i$ or $\Lambda_i^{\frac{1}{2}}$?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper examines the behaviour of heteroskedastic regression models. By regularizing the model, first with differing levels of weight decay on the mean and covariance functions, and then extending this to the corresponding Dirichlet energies, the authors appeal to tools from statistical mechanics and the calculus of variations in order to derive a system of elliptic partial differential equations that give necessary conditions for energy minimization. This admits a phase diagram in terms of the regularization parameters, describing a two-parameter family of solutions that exhibit phase transitions between different regions of qualitative behaviour. Experimental validation of this behaviour is verified, and the two-dimensional family is reduced to a single dimension for the purposes of hyperparameter optimization.

### Strengths
The paper is well written and presented. Drawing insight on tools in machine learning via adjacent fields is always valuable.

### Weaknesses
There is a large conceptual leap from the weight decay formulation to using the Dirichlet energy as a regularizer. While the two coincide for linear models, that alone is a tenuous link. Other work has drawn (similarly loose) links to implicit regularization via backwards error analysis of predictive networks trained with SGD, and probably warrants mentioning https://arxiv.org/pdf/2209.13083.pdf.

A single Monte Carlo sample is used in the construction, without further discussion or investigation on the limitations of doing so.

Taking the Dirichlet energy with respect to $p(x)$ may be interesting and warrants discussion (or future work). Assuming $p(x)$ to be uniform for the purposes of numerics is concerning, and doing so may help alleviate this issue.

### Questions
Can the authors please address the highlighted weaknesses

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work studies the challenge of conditional variance estimation "from the perspective of statistical physics".  The authors studied the behavior of the regularized learning objective 
$$\rho\cdot  E(\log p_N(y\mid \mu(x), \Lambda^{-2}(x))) + (1-\rho)\bigl(\gamma \\|\nabla\mu\\|\_{L_2(P_x)}^2 +  (1-\\gamma) \\|\nabla\Lambda\\|\_{L_2(P_x)}\^2\bigr)$$ 
where $\mu,sigma$ are the conditional mean and variance functions to be estimated, in the extreme cases of no regularization ($\rho=1$), "no data" ($\rho=0$) and no mean regularization ($\gamma=1$), and presented simulation studies.

### Strengths
N/A

### Weaknesses
My main concern is that the theoretical results are irrelevant and trivial.
- For example, the "no regularization" regime in Prop. 1 does not describe any reasonable learning algorithms, all of which introduce regularization explicitly or implicitly (e.g. through restrictions to certain function classes, algorithmic regularization through e.g. gradient descent, etc.). If the authors wish to study a nonparametric estimator such as the one defined in their Eq (7), they should impose constraints on the functions (e.g. Sobolev) and carefully choose a rate of vanishing regularization strength *in accordance with the function class*. If the authors wish to study estimators without explicit regularization -- as is common in the analysis of overparameterized models -- they should specify the form of implicit regularization (e.g. gradient descent / gradient flow; model parameterization).
- Furthermore, the challenge of conditional variance estimation arises from overfitting, yet the main result is stated for a *population objective* without any account for sample size.

Additionally, the references to statistical physics appear completely unnecessary.  Calling Eq. (7) a "nonparametric free energy" does not provide any new insight. The proof of the main result also makes no use of techniques or ideas from statistical physics.

### Questions
The authors are encouraged to study the proposed learning objective in a relevant and non-trivial regime, and possibly to familiarize themselves with notions in learning theory and nonparametric statistics.

## Post-rebuttal update

I appreciate the authors' response, but it does not address my original concerns, therefore my recommendation remains unchanged.  

Regarding the author's question, I have asked for a more precisely formulated asymptotic analysis where the rate of convergence (to 0 or 1) of the regularization hyperparameters are explicitly quantified; this is only possible with at least some concrete conditions imposed to the model family (e.g. DNN models with a certain parameterization).

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor
