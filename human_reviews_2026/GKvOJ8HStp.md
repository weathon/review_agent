# Optimal Regularization for Performative Learning

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
In performative learning, the data distribution reacts to the deployed model—for example, because strategic users adapt their features to game it—which creates a more complex dynamic than in classical supervised learning. One should thus not only optimize the model for the current data but also take into account that the model might steer the distribution in a new direction, without knowing the exact nature of the potential shift. We explore how regularization can help cope with performative effects by studying its impact in high-dimensional ridge regression. We show that, while performative effects worsen the test risk in the population setting, they can be beneficial in the over-parameterized regime where the number of features exceeds the number of samples. We show that the optimal regularization scales with the overall strength of the performative effect, making it possible to set the regularization in anticipation of this effect. We illustrate this finding through empirical evaluations of the optimal regularization parameter on both synthetic and real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the interaction between regularization and performative effects in performative learning, a framework where the data distribution adapts to the deployed model (e.g., through user behavior or feedback loops). The authors focus on ridge regression and study how the optimal regularization parameter should be tuned to mitigate performative shifts. The analysis is conducted under strong modeling assumptions: the data-generating process is fully Gaussian, the conditional model for $y \mid x$ is linear with both predictive and spurious features, and the performative shift acts linearly on the covariates. Within this stylized setting, the paper characterizes how the optimal regularization depends on the strength and direction of the performative effect.

### Strengths
The paper provides an analysis of how regularization interacts with performative effects in ridge regression. It offers closed-form expressions for the optimal regularization in population regimes (small number of covariates and over-parameterized), supported by empirical validation.

### Weaknesses
1) **Modeling assumptions are overly restrictive** and cast doubt on the generality of its conclusions. The entire analysis is built on a fully Gaussian linear framework, where:  
- the performative effect acts only on the covariates and the dependence is linear;  
- the conditional model for $y \mid x$ is linear with additive Gaussian noise; and  
- The covariates are mean-zero Gaussian.

This highly stylized formulation makes the mathematics tractable but reduces conceptual novelty. More importantly, my main concern about this specific model's assumptions is that the proportionality between the optimal ridge parameter and the performative strength follows directly from the linear–Gaussian algebra rather than from a general property of performative learning. Consequently, it is unclear whether the claimed insights extend to nonlinear predictors, heteroskedastic noise, or performative effects that act on the label rather than on the covariates, or even to more realistic yet still structured settings such as those involving strategic agents.

2) **The paper’s relationship to prior work (Perdomo et al., 2020)** is somewhat misrepresented.  

The authors claim their model generalizes the performative regression model of Perdomo et al.; however, the opposite is true.  
The setup in this paper is a special case of the general framework in Perdomo et al.:  
- their model corresponds closely to *Example 2.2* in Perdomo et al., where the performative shift is encoded through a linear transformation of the covariates. The only substantive difference from *Example 2.2* in Perdomo is the inclusion of spurious covariates and that in Perdomo, $y$ is binary, and in this work, it is continuous.  
- in contrast, Perdomo et al. allow a general distributional shift in the joint $(X,Y)$, while this paper restricts the shift to $X$ only;  
- the outcome $Y$ in Perdomo et al. is not required to be linear or Gaussian, whereas here it is; and  
- Perdomo et al. consider general loss functions, while this paper is limited to the squared loss.  

Hence, the contribution should be regarded as an analytical refinement of a specific subcase within Perdomo’s framework, rather than as a generalization.

3. **Recent work in high-dimensional setting**. The paper also overlooks recent works that already study high-dimensional performative settings without relying on such restrictive Gaussian-linear models and squared loss assumptions. For instance, [1] analyzes regret minimization under general performative feedback, and their convergence only depends on the zooming dimension (which can be much smaller than the parameter dimension); and [2] establishes dimension-independent convergence results in the strategic setting with general loss. These works demonstrate that meaningful high-dimensional analysis is possible under more general conditions.

## References
[1] Jagadeesan, Meena, Tijana Zrnic, and Celestine Mendler-Dünner. "Regret minimization with performative feedback." International Conference on Machine Learning. PMLR, 2022.

[2] Bracale, Daniele, et al. "Learning the distribution map in reverse causal performative prediction." arXiv preprint arXiv:2405.15172 (2024).

### Questions
1. Do you also assume $\mathbb{E}( \theta^* )=0$ for Equation (7) to hold? The derivation seems to require the population parameter to be centered; otherwise, the cross-term $\mathbb{E}( (\theta^* )^\top A^T \Sigma A \theta^* )$ would not simplify as stated.

2. Why do you introduce the empirical version of $\theta_k$ in Eq. (3), (13) and (14) when the rest of the analysis is conducted entirely at the population level? This switch in notation could confuse readers, as it is unclear whether finite-sample randomness ever plays a role in the theoretical results.

3. In the abstract, you state:  
   > “We show that, while performative effects worsen the test risk in the population setting, they can be beneficial in the over-parameterized regime.”  
   
However, it seems that it is not the *performative effect itself* that is beneficial, but rather that in the presence of performativity, optimal regularization helps reduce the variance or uncertainty of the estimated parameters, thereby improving performance. Could you clarify or restate this claim to better reflect the mechanism driving the improvement?

4. After Equation (17), you refer to $\widehat{ R  } _ {eq}$, but I don't find it defined in the text.  Could you indicate where this expression is formally introduced?

5. The performative effect is modeled as a diagonal linear transformation. Would the analysis still hold if $D$ had off-diagonal entries, i.e., if the performative shift mixed predictive and spurious features?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates what happens for linear regression applied in performative settings (where data reacts to the deployed model) when the model is optimally regularized via L2 regularization. Both the under parametrized and overparametrized are studied theoretically.

### Strengths
The theory agrees with the numerical experiments

It makes sense to study a technique like regularization for this setting

### Weaknesses
The paper has very low self-containment and lacks intepretation. As someone who is vaguely familiar with performative prediction, I could not understand the setting from reading this paper. For example; Assumption 1 introduces various variables, such as a, b, c, d. Yet their interpretation is not mentioned. Also $a$ seems to have a covariance; the interpretation for me is not clear. Are we then in a Bayesian setting and is there a prior on $a$? Or are we in a frequentist setting and we do typically consider worst-cases with respect to $a$? Because the setting is not clearly introduced, this really hampered my reading of the whole paper. 

The writing is very technical; I cannot get the main points easily. Even the Figure captions are so technical, with very little interpretation, I cannot understand their points.

### Questions
"In this section, we tackle the population regime where there are enough samples from D(θk) at each deployment to compute exactly the next regressor, as would typically happen in a low-dimensional setting."

How is this possible in the presence of noise?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates how regularization can mitigate performative effects, e.g., situations where the data distribution depends on the deployed model. The authors focus on high-dimensional ridge regression and study both the population regime and the over-parameterized regime. In particular, it shows that 1) in the population regime, the optimal regularization is proportional to the magnitude of the performative effect and can mitigate performance degradation; 2) in the over-parameterized regime, performative effects can actually improve test risk when properly regularized. The authors derive closed-form characterizations of the optimal ridge coefficient and deterministic equivalents for performative risk, supported by both synthetic and real-world experiments (Housing, LSAC datasets).

### Strengths
Overall, the paper is well-written and well-structured. It provides one of the first analytical treatments of regularization under performativity, linking performative dynamics to the scaling of optimal regularization. The mathematical contributions are rigorous, and the main theorems (Theorem 1, 3, 4) are clearly stated. The finding that performative effects can improve performance in the over-parameterized regime (contrary to intuition) is conceptually interesting and could have a broader impact on regularization with the presence of performativity.

### Weaknesses
My biggest concern with the paper is that it relies on restrictive modeling assumptions: in particular, the analysis assumes Gaussian features and linear label shifts (Assumption 1). This limits generalizability to nonlinear or heavy-tailed data distributions, which are common in real-world performative settings. In addition, the paper only focuses exclusively on $\ell_2$ regularization; other forms (ℓ₁, dropout, early stopping) are only mentioned in future work but not studied.

While the theoretical results are rigorous and technically sound, many expressions (e.g., Theorem 3 and its expansions) are algebraically heavy and include higher-order terms that obscure intuition. Even though the authors provide closed-form expressions, the results are not immediately interpretable without significant algebraic unpacking. A more intuitive discussion or simplified special cases (e.g., isotropic $\Sigma$ etc) would help readers understand the qualitative behavior of the optimal regularization.

### Questions
In Eq. (4), the excess risk is defined as the test risk under the unshifted distribution $D(\theta = 0)$. While this isolates the performative effect and prevents evaluation bias, it seems somewhat counterintuitive to me; in reality, a deployed model is evaluated on the induced distribution $D(\theta^∗)$. Could the authors clarify the motivation for this evaluation choice and discuss how the conclusions might differ if the test risk were computed on $D(\theta^∗)$ instead?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This submission considers "performative prediction" setting introduced by Perdomo et al. (2020). Performative prediction is a learning setting where a decision-maker repeatedly deploys a model and nature responds with a slightly altered distribution. A key example is when individuals may strategically alter their features in order to receive better classification under the current predictive model. 

This submission works in the setting where the decision-maker uses ridge regression; it focuses on exploring the interaction of regularization and performativity. The authors prove several theorems about the convergence and excess risk of this process. They also run some experiments.

Mathematically, after the decision-maker deploys a model $\theta$, the data arise from distribution $D(\theta)$ as follow: $x_i \sim \mathcal{N}(0,\Sigma)$ and $y_i \sim x^T \theta^* + x^T D \theta + \mathcal{N}(0,\sigma^2)$. Here $D$ is a matrix that controls the performative effects. The sequential retraining process may converge to some $\theta^{\infty}$. The theorems evaluate performative risk with respect to $D(\theta=0)$. In contrast, the prior work I am familiar with evaluates excess risk on $D(\theta^{\infty})$, i.e., at the equilibrium we actually reach.

### Strengths
I think this is an interesting area in which to work. I found the paper polished and easy to read. 

The issue I address below may not be fundamental, in which case I would view the paper favorably.

### Weaknesses
As mentioned above, this paper evaluates excess risk on the $D(\theta=0)$ distribution, which in general will not be the equilibrium distribution. In contrast, the work on performative prediction I am familiar with evaluates risk on the equilibrium distribution (see [1]). It is not clear to me why the results here are of interest.

The authors provide some reasoning: "This ensures that the final model is not evaluated on shifted distributions, and it is particularly relevant for long-term fairness, as it prevents bias amplification over time... and discourages reliance on spurious feature." I do not understand how these address the concern above.

However, I am not an expert in the area so I look forward to discussion with the authors and other reviewers.

[1] Hardt, Moritz, and Celestine Mendler-Dünner. "Performative prediction: Past and future." Statistical Science 40.3 (2025): 417-436. [https://arxiv.org/abs/2310.16608](https://arxiv.org/abs/2310.16608)

### Questions
Why did you choose to evaluate excess risk with respect to $D(\theta=0)$?

How do the results change if you evaluate with respect to $D(\theta^*)$?

### Soundness
4

### Presentation
3

### Contribution
1
