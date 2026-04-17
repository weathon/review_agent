# Dimension-Free Decision Calibration for Nonlinear Loss Functions

- Decision: Accept (Poster)
- Scores: 8, 8, 8, 4

## Abstract
When model predictions inform downstream decisions, a natural question is under what conditions can the decision-makers simply respond to the predictions as if they were the true outcomes. The recently proposed notion of decision calibration addresses this by requiring predictions to be unbiased conditional on the best-response actions induced by the predictions. This relaxation of classical calibration avoids the exponential sample complexity in high-dimensional outcome spaces.
However, existing guarantees are limited to linear losses. A natural strategy for nonlinear losses is to embed outcomes $y$ into an $m$-dimensional feature space $\phi(y)$ and approximate losses linearly in $\phi(y)$. Yet even simple nonlinear functions can demand exponentially large or infinite feature dimensions, raising the open question of whether decision calibration can be achieved with complexity independent of the feature dimension $m$. We begin with a negative result: even verifying decision calibration under standard deterministic best response inherently requires sample complexity polynomial in $m$. 
To overcome this barrier, we study a smooth variant where agents follow quantal responses. This smooth relaxation admits dimension-free algorithms: given $\mathrm{poly}(|\mathcal{A}|,1/\epsilon)$ samples and any initial predictor $p$, our introducded algorithm efficiently test and achieve decision calibration for broad function classes which can be well-approximated by bounded-norm functions in (possibly infinite-dimensional) separable RKHS, including piecewise linear and Cobb–Douglas loss functions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies prediction for downstream decision-making via the lens of decision calibration. Decision calibration requires that predictions are calibrated with respect to a decision-making rule, which incurs loss depending on the prescribed action and label. The paper focuses on non-linear losses that admit (possibly infinite-dimensional) linear representations, e.g. in an RKHS. Prior work on decision calibration handle linear losses and losses with finite-dimensional linear representations. 

The main results are:
- Under best response, verifying decision calibration requires sample complexity polynomial in the dimension of the prediction space. 
- Under smooth best response, there is an auditing algorithm for losses in RKHS with sample complexity independent of the dimension. 
- The paper then gives an algorithm that post-processes an initial predictor to a decision-calibrated predictor--under smooth best response and for losses in RKHS. The sample complexity is independent of the dimension and scales polynomially in the number of actions and 1/epsilon.

### Strengths
- I think this paper makes solid contributions to the study of calibration for decision-making. The proposed algorithm is a fairly significant extension of previous results. The authors also give a lower bound that has implications for existing calibration algorithms. 
- The technical contribution is a uniform convergence argument for decision calibration under smooth best response.
- Overall the paper is well-written and the relation to previous work is clear.

### Weaknesses
- The idea of uniformly approximating non-linear losses comes from previous work of Gopalan et al (2024b) and Lu et al (2025) in similar contexts. The algorithmic template follows that of Gopalan et al (2022) and Zhao et al (2021) (but it was not previously clear how to handle the infinite-dimensional case).
- The uniform approximation framework is introduced but not explicitly used; even the examples in Appendix F seem to have exact representations. It would be nice to spell out the end-to-end decision calibration guarantees for losses that can be uniformly approximated and/or the examples given in Appendix F. 

Minor things:
- The term “feature space” is used for both $X$ and $H$; I would suggest changing one. 
- The decision rule $k$ has domain $X$, but the paper is only concerned with decision rules that are functions of the predictions (or loss estimators), right?

### Questions
- Why is smooth best response able to give stronger results intuitively-speaking? I think it would benefit the paper to add some intuitive explanation behind the separation, e.g. why smooth best response “fixes” the hard example of Theorem 3.1. 
- Can the results under smooth best response be extended to any decision rule that is near-optimal and smooth?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies decision calibration in high-dimensional space and nonlinear loss functions. The main results in the paper include:
* Lowerbound for deterministic best response: distinguishing decision calibration requires a sample complexity of $\Omega(\sqrt{m})$. 
* Dimension-free decision calibration under smooth best response: under the quantal response model, the paper designs an algorithm that audits decision calibration, where the sample complexity is invariant of dimension. 
* dimension-free audition then leads to a dimension-free algorithm for decision calibration under quantal response.

### Strengths
The paper addresses a well-motivated open problem. The main message in the paper is strong: under smooth best responding, decision calibration can be achieved without dependence on dimensionality. The main technical contribution in the paper is the dimension-free audition result. The algorithm design follows from existing work and the existence of the audition algorithm. 

The paper is well organized. The class of functions considered here seem quite general (including nonlinear losses linearizable or uniformly approximable in an RKHS). It would be great if the authors could include more discussions on the class of loss functions admitted by the problem setup and the dependence on norm. It would also help me if there are examples not admitted in the class, and how decision calibration cannot be achieved.

### Weaknesses
The work builds on established techniques, yet delivers useful results.

### Questions
See my comment above.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes improved upper and lower bounds on the problem of decision calibration in the batch setting, when faced with potentially nonlinear losses and/or high/infinite dimensional feature spaces. The contribution is roughly threefold. First, a lower bound is provided, which establishes that under the pure best response decision rule (deterministic selection of the best alternative), a poly(m) complexity in the dimension m is required just to audit for the presence/absence of decision calibration. 

Second, in contrast to the first result, an efficient auditing scheme is developed for various loss classes in RKHS that has complexity independent of the ambient dimension. Third, this dimension-free auditing scheme is extended to provide an efficient dimension-free decision calibration algorithm in this setting.

### Strengths
This is a good paper; the proposed array of results are both insightful, creating previously unknown separations, and employ a variety of techniques that may be useful for follow-up work.

On the insights side, it is an important message that decision calibration can be done in a dimension free way; equally important, however, is the complementary message that the dimension-dependence can go away for smooth decision rules but not for the canonical nonsmooth one. This draws distant parallels to other, quite distinct, instances of smooth vs. nonsmooth issues in (regular) (multi)calibration, but is very much a decision-focused phenomenon here.

On the techniques side, here are a couple that I want to point out. First, extending from the Gopalan et al constructions, that applied to full calibration rather than the decision variant, is not fully trivial and required some careful constructions. Second, on the constructive side, constructing the dimension-free calibration algorithm using the auditing algorithm is also not as straightforward as it may seem; among all else, it required “implicit patching”, a simple but useful trick.

### Weaknesses
I believe this is a good-quality and innovative paper, and only have relatively smaller qualms with it. 

First, the negative result doesn’t quite fully align with algorithmic impossibility of calibration itself (and I’m not convinced how difficult to solve this misalignment problem is in this context). Briefly (and as quickly mentioned in the corresponding section of the paper), the proof by covering shows that auditing for calibration is not possible to do without incurring a dimension-driven dependence. Given that all existing (decision) calibration algorithms depend on auditing of miscalibration, it is thus possible to conclude that the currently studied ubiquitous algorithmic template doesn’t allow for dimension-free calibration. However, there might ostensibly be an off-chance whereby decision calibration is possible to perform dimension-free by eschewing direct auditing. This off-chance possibility thus represents a gap that must be discussed more prominently; this includes a broader perspective — auditing/testing vs calibration tasks have both come up in the literature before e.g. for vanilla calibration, so are there any insights into how fundamental this gap is to bridge?

Secondly, the paper touches upon the following important consideration, which is to my knowledge fairly novel in the literature: That decision calibration can exhibit tractability separations depending on the nature of the decision rule. In this case, the very least that must be done is to align the presentation and the terminology surrounding the decision rules that are studied. In particular, given that (1) the hard best response rule comes up in the 0-temperature limit of quantal response, and (2) the bounds obtained in the upper bound side for quantal response have an explicit dependence on the inverse temperature beta, it would be clean to define and refer to decision rules throughout the paper in terms of that temperature. (Even on the current terminology level, the quantal response mapping is also called other names such as “smooth” etc. so that can be streamlined.) 

In particular, I would consider the story about the influence of the temperature on the hardness of the problem somewhat fully addressed if the upper bounds involving beta were also accompanied by lower bounds in beta (ignoring other parameters besides beta); matching bounds may be a lot to ask for from the technical perspective, so any, even slow, lower-bounding dependence on beta would suffice to drive the above point home.

### Questions
My substantive questions in this case are limited to the issues in the Weaknesses section above. In addition to those, I have observed a fair amount of typos and grammar issues, which would be tedious to list here (but suffice it to say, they are prominent even in the abstract) --- so to improve the writing, these need to be fixed.

### Soundness
3

### Presentation
2

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
Prior work on decision calibration requires that predictions are unbiased conditional on events relevant to action selection under a linear loss function. This paper extends the decision calibration framework to a setting where decision makers to have nonlinear loss functions.

### Strengths
- Most prior work in decision calibration, e.g. Zhao et al 2021, Sahoo et al 2021, focuses on linear losses, so this work makes a contribution to the area by considering nonlinear loss functions.

- This work proposes to approximate nonlinear loss functions using a feature mapping $\phi: \mathcal{Y} \rightarrow \mathcal{H}$, so that a nonlinear loss function $\ell(y, a)$ can be approximated by a linear loss function $\ell^{*}(\phi(y), a)$. It is quite common and natural to express a nonlinear function through a basis expansion.

- The main theoretical contributions include a lower bound for dimension-free decision calibration, a dimension-free sample complexity guarantee of decision calibration under nonlinear losses and quantal response, and improved sample complexity results for decision calibration.

### Weaknesses
- The introduction could be significantly strengthened by making clear why existing approaches for decision calibration fall short for nonlinear loss functions. In addition, it would also be helpful to emphasize the challenge that arises when moving from linear loss functions to nonlinear loss functions. Is decision calibration more difficult to achieve for nonlinear loss functions because we may require unbiasedness of higher-order moments of the outcome? Is this the motivation for embedding outcomes into a feature space?

- In a related vein, the link between nonlinear loss functions and achieving decision calibration for higher-dimensional outcomes could be made much more clear in the introduction. The abstract of the paper emphasizes moving from linear losses, which are commonly studied in decision calibration, to nonlinear loss functions. So, my initial interpretation was that this paper would focus on a loss functions that depend on higher-order moments of the outcome. However, the introduction very quickly jumps from a discussion of linear losses to a discussion of achieving decision calibration for high-dimensional outcome spaces.  

- Since the authors propose to express a nonlinear function as a linear basis expansion, is it necessary to achieve some calibration notion for predictions of the potentially high-dimensional outcome $\phi(Y)$. I would expect the sample complexity of decision calibration to depend on the dimension of $\phi(Y)$ as in Zhao et al, 2021, the sample complexity depends on $C$ the number of classes.  It would helpful to clarify what's different/new about this analysis that allows us to sidestep the dependence on the dimension of $\phi(Y)$.

- It would be useful to clarify in the introduction why the authors consider under the quantal response, and what limitations are in considering the quantal response rather than just the best response rule.

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
3
