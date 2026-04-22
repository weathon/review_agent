# Understanding SGD with Exponential Moving Average: A Case Study in Linear Regression

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 4, 6, 6

## Abstract
Exponential moving average (EMA) has recently gained significant popularity in training modern deep learning models. However, there have been few theoretical results explaining the effectiveness of EMA. In this paper, to better understand EMA, we establish the risk bound of online SGD with EMA by performing a case study on overparameterized linear regression, which is one of the simplest overparameterized learning tasks that shares similarities with neural networks. Our results indicate that (i) the variance error of SGD with EMA is always smaller than that of SGD without averaging, and (ii) unlike SGD with iterate averaging from the beginning, the bias error of SGD with EMA decays exponentially in every eigen-subspace of the data covariance matrix. Additionally, we develop proof techniques applicable to the analysis of a broad class of averaging schemes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper focuses on methods for linear regression that run stochastic gradient descent (SGD) while maintaining an exponential moving average (EMA), i.e., a smoothed version of the SGD iterates. The main results provide upper and lower bounds on the bias and variance of the smoothed weight vector. 

The derived bounds depend on:
- the learning rate for SGD,
- the averaging parameter,
- the spectrum of the feature covariance matrix,
- bounds on the fourth moments of the feature matrix, and
- a decoupling assumption between the label noise and the features.

Overall, this is a reasonably clear paper that provides some novel theoretical results for a specific method. There seems to be growing interest in these types of methods within the community, so the topic may be of sufficient relevance. One could argue that the specific results are somewhat incremental given recent work on averaging schemes, but the key innovations in the proof technique appear to extend more generally to any time-dependent averaging scheme.

There is room for improvement in the presentation. In particular, the notation is quite heavy, and I found the discussion of the results difficult to follow at times.


Suggestions for improvement: 
- The discussion in Section 4.2 about interpreting the different terms in the variance was difficult to follow. Breaking the argument into separate parts or labeling the individual terms more explicitly would help the reader understand the contribution of each component.-
- Proposition 4.3 was also challenging to parse. It would help to provide more intuition for the different parameter regimes (sample size, learning rate, averaging rate) and to clarify their practical meaning. The red highlighting was not helpful for me.
- It would be valuable to include some guidance for practice: How should one choose the parameters in realistic scenarios? Is there a way to adjust the level of averaging adaptively during training? 
- It would also improve readability if the colors were matched between subfigures (a) and (b) in Figure 1.

### Strengths
- The explicit bounds on the bias and variance is an interesting and novel result.  It extends previous work on iterate and tail averaging to a more general, time-dependent averaging framework.

- The assumptions are clearly stated and are quite general. . 

- While the basic setup of the proof follows closely from prior work, the main technical novelty seems to be the ability to handle the general time-dependent averaging scheme described in Section 7, which includes many variants of SGD as special cases.

- The outline of the proof in Section 7 is helpful.

### Weaknesses
-   The presentation is dense and notation-heavy, making several sections difficult to follow.

- The paper could include more practical guidance — e.g., how to select the averaging parameter alpha based on the observed data. 

- It is not clear to me how much this method improves upon other averaging schemes, so the overall impact may be limited.  For example, analysis of non-linear models would likely have broader impact.

### Questions
- Would it be possible to state a prove  a general theoretical result for any generic averaging scheme? What really matters in the weights to ensure good performance? Can the averaging be data-dependent? 

- My interpretation of Table 1 and the figures is that all reasonable averaging schemes are somewhat equivalent. It this a reasonable takeaway or is there really some special about exponential moving average that I am missing?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper  investigates the generalization behavior of constant step-size Stochastic Gradient Descent (SGD) when combined with Exponential Moving Average (EMA), using high-dimensional linear regression as a prototypical setting. The authors derive instance-dependent and dimension-independent upper and lower bounds for the excess risk, which are tied to the spectral properties of the data covariance matrix. The analysis provides a clear decomposition of the error into an "effective bias'' and an "effective variance.'' The effective bias is shown to decay exponentially in each feature subspace. The effective variance is demonstrated to be jointly controlled by the EMA parameter $\alpha$, the learning rate $\delta$, the number of iterations $N$, and the covariance spectrum. A key theoretical contribution is the direct comparison of EMA with tail averaging, revealing that under appropriate conditions, their variance terms are comparable and can even be equivalent in magnitude. The work also extends the analysis to the mini-batch SGD setting and discusses the resulting critical batch size.

### Strengths
- This paper provides the first fine-grained, instance-dependent risk bounds for SGD with EMA in the high-dimensional linear regression regime. The spectral decomposition analysis, which demonstrates that the bias decays exponentially along each eigendirection, is a solid theoretical contribution. The resulting dimension-free upper and lower bounds, characterized by two effective dimension thresholds, are technically sound and insightful.
- A significant strength is the unified framework used to compare EMA against classical strategies like no averaging, full-iterate averaging, and tail averaging. The analysis lucidly establishes a formal connection between EMA and tail averaging, particularly their equivalence in variance magnitude under a specific condition. This helps clarify theoretically why EMA is often observed to be a superior or highly competitive averaging strategy in practice.

### Weaknesses
- The primary weakness lies in the strength of Assumption 3.3, the fourth-moment lower bound. This assumption ($\mathcal{M}\circ A - HAH \succeq \beta \mathrm{tr}(HA) H$) effectively requires a uniform "energy'' lower bound from the fourth-order moments in all directions. This is a very strong condition that is unlikely to hold for many practical high-dimensional data distributions, especially those that are non-isotropic, heavy-tailed, or mixtures, where tail subspaces might have weak energy or significant variations in directional kurtosis. This assumption appears to be a technical convenience for achieving a tight lower bound rather than a verifiable data property, which limits the robustness and generalizability of the conclusions.
- There is a notable disconnect between the paper's stated motivation and its actual contributions. The abstract and introduction heavily motivate the work by citing the widespread use of EMA in training diffusion models. However, the theoretical analysis and experiments are confined entirely to linear regression (with a brief extension to a single ReLU neuron in the appendix). The paper offers no specific analysis or empirical evidence related to the unique loss landscapes, model architectures, or training dynamics of diffusion models, failing to fully bridge the gap between the simple theoretical setting and the complex, practical application it claims to address.
- The experimental validation feels incomplete and offers limited practical guidance. The authors select the learning rate $\delta$ based on the derived theoretical threshold  but do not perform any systematic sweep or ablation study over this hyperparameter. Without comparing performance across a range of learning rates, it is impossible to assess the necessity, conservativeness, or near-optimality of this theoretical threshold in practice, which diminishes the practical value of the theoretical findings.

### Questions
Could the authors please comment on the realism of Assumption 3.3 and the implications of relaxing it, even if it impacts the tightness of the lower bound? Additionally, to bridge the gap between motivation and scope, I suggest either revising the abstract's focus on diffusion models to better match the linear regression results, or adding a discussion on how these insights might qualitatively extend to such complex models. Finally, to strengthen the empirical validation, would it be possible to add a learning rate ablation study for $\delta$ to test the necessity and conservativeness of your theoretical threshold? I am willing to raise my score if these concerns are satisfactorily addressed.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper theoretically studies why EMA works with SGD. Specifically, it provides both upper and lower bound for the excess risk of EMA through variance and bias decomposition. Experiments are also conducted to show the variance and bias error of EMA to verify the theoretical claims.

### Strengths
1. The paper is the first to theoretically bound the excess risk of SGD with EMA despite the prevalance of this method.
2. There're some techinical novelty as pointed out by the authors from line 385-393, that previous analysis doesn't work due to the inhomogeneous $\beta_t-\beta_{t+1}$.
3. The experimental results seem to align with proposition 4.3 well.

### Weaknesses
1. The setting of linear regression is perhaps a bit simple given EMA is used in much harder optimization problems.
2. The proofs of Theorem 4.1 and 4.2 seem to largely follow the previous analysis, especially [1], and I'm not sure how to evaluate the contribution of this proof given I'm unfamiliar with this area and the related work.

[1] Zou, D., Wu, J., Braverman, V., Gu, Q., & Kakade, S. M. (2023). Benign overfitting of constant-stepsize SGD for linear regression. Journal of Machine Learning Research, 24(326), 1-58.

### Questions
1. Would it be possible to extend linear regression to more complicated (convex) functions, or more specifically the loss function in diffusion models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work studies the SGD algorithm (also extended to minibatch SGD) with EMA under the high-dimensional linear regression, which can be viewed as  a simplified model of overparameterized neural networks. This work provides a unified analytical framework for different averaging methods and shows that EMA is better on the bias error and variance error at the same time. The simulation experiments also support their results.

### Strengths
Strengths: 

* From the theoretical perspective, this work clearly shows the advantage of the EMA averaging method compared with other methods.

### Weaknesses
1. As we know that high-dimensional linear regression is an important task, would it be possible to conduct experiments in a real-world setting and show that EMA has a better performance?

### Questions
Question:

Q1: Could this work discuss the theoretical or technical challenge of Adam with EMA intuitively?

Q2: I understand that these high-order assumptions have been used in previous works. However, it is also helpful to discuss these assumptions.

Minor Comments:

It would be better to add the theoretical line in Figure 1 to do a better comparison.

### Soundness
3

### Presentation
3

### Contribution
3
