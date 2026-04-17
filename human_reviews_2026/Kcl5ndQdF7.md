# On the Rate of Convergence of Kolmogorov-Arnold Network Regression Estimators

- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Kolmogorov-Arnold Networks (KANs) offer a structured and interpretable framework for multivariate function approximation by composing univariate transformations through additive or multiplicative aggregation. This paper establishes theoretical convergence guarantees for KANs when the univariate components are represented by B-splines. We prove that both additive and hybrid additive-multiplicative KANs attain the minimax-optimal convergence rate $O(n^{-2r/(2r+1)})$ for functions in Sobolev spaces of smoothness $r$. We further derive guidelines for selecting the optimal number of knots in the B-splines. The theory is supported by simulation studies that confirm the predicted convergence rates. These results provide a theoretical foundation for using KANs in nonparametric regression and highlight their potential as a structured alternative to existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a theoretical framework for analyzing the convergence rate of Kolmogorov-Arnold Networks on Sobolev spaces of smoothness r. The scaling exponent $2r/(2r+1)$ highlights that KANs do not suffer from curse of dimensionality because the scaling exponent for MLPs is $2r/(2r+d)$ where $d$ is the input dimensionality. The authors further derive how to optimally choose grid size k and support their conclusions with numerical experiments. These results provide a theoretical foundation for using KANs in non-parametric regression.

### Strengths
* The paper is well motivated and nicely written
* The paper presents a theoretical contribution on the convergence rate of KANs
* Theoretical claims are supported by numerical experiments

### Weaknesses
* Generalization of the result to deeper KANs. Theorem 1 and 2 seemed to assume two-layer KANs. Can this result be extended to deeper KANs?
* The optimal number of knots $k$ is not supported by numerical experiments.

### Questions
* Theorems 1 and 2 seemed to assume two-layer KANs. Can this result be extended to deeper KANs?
* In Corollary 2, the optimal number of internal knots is derived. For smooth functions (say $f(x) = {\rm sin}(x)$) when $r\to\infty$, how should one choose the optimal $k$?
* In abstrac, $n$ is used without defnition.
* a nitpick, since $k$ is reserved for the optimal knot number, it's better to use a different subscript letter in $a_k$ (line 402).

### Soundness
3

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
This paper considers estimating functions from a certain model class using KANs. The model class is closely related to the KAN architecture and essentially consists of functions which can be expressed using KANs with a certain amount of smoothness. The resulting rates are shown to be minimax optimal, and numerical experiments are performed. I think that the paper has quite an interesting potential idea, however, it seems to me that the authors have not described the estimator that they are analyzing. This issue would need to be addressed for the results to make sense.

### Strengths
Function estimation using KANs is becoming increasingly popular, so the problem addressed is of significant importance. The paper is also well-presented and written.

### Weaknesses
There are appear to be some significant technical details missing. Specifically, in Theorem 1 what exactly is the estimator $\hat{f}$? You say that $\hat{g}\_q$ and $\hat{\psi}\_{qj}$ are constructed via least squares with back-fitting. What does this mean, exactly? I can't find an explanation elsewhere in the paper. Note that only the composite function $f$ is sampled, you do not have access to the true $g_q$ and $\psi_{qj}$ to do regression on. This point confused me a bit, could the authors please be explicit about what estimator they are studying.

### Questions
Could the authors also please clarify what algorithms are used for training their KANs in the experimental results section? If these algorithms differ from the estimator which is theoretically analyzed, please indicate this.

If these issues about the precise definition of the estimator are addressed, I would be willing to raise my score.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the statistical behavior of Kolmogorov–Arnold Networks (KANs) in the nonparametric regression. KANs are based on the Kolmogorov–Arnold representation theorem, which guarantees that any continuous multivariate function can be expressed as a sum of univariate functions applied to linear combinations of input variables.

The paper focuses on a setting of KANs where each univariate component is modeled using B-spline basis functions. The paper then places this architecture within the framework of classical nonparametric estimation and provides a complete analysis of its convergence properties under standard smoothness assumptions.

The main result shows that, under Sobolev smoothness of order r, both additive and hybrid (additive–multiplicative) KAN models achieve the same minimax-optimal mean-squared convergence rate as traditional spline or kernel estimators. The paper also offers a practical rule for choosing the number of spline knots and validates the theoretical predictions with some numerical experiments.

### Strengths
The major contribution of this paper is that for both additive and hybrid KAN estimators, they obtained the minimax-optimal mean-squared error rate under Sobolev smoothness assumptions. This places the models on a solid theoretical foundation within nonparametric statistics.

Moreover, using these results, the paper provides the guidance on the choice of the optimal number of spline knots $k \asymp n^{1/(2r+1)}$.

### Weaknesses
The main theorems establish the classical minimax rate for Sobolev regression functions. While they considered Kolmogorov–Arnold Networks (KANs) in the nonparametric regression, they only considered additive and hybrid (additive–multiplicative) KAN models, see Equation (11) in the paper. Using B-spline basis functions to obtain minimax optimal regression estimators for Sobolev regression functions is standard and has been studied in many existing papers. Due to this simple form of the model in Equation (11), it is not clear if this is a significant contribution to the statistical and machine learning community at ICLR.

Moreover, the paper [1] studied minimax optimality of a more complex model, the deep composition of anisotropic Besov functions, which seems to cover the setting of this paper, at least, for the theoretical contributions and using B-splines. I think the theoretical contributions of this paper then may not be adequate. Also, in recent regression literature such as [1,2,3], adaptivity to unknown smoothness parameters are usually studied. While this paper provided a suggestion of picking the optimal number of spline knots, it relies on the unknown parameter r.

Overall, I think this paper appears to be a subset of [1], or has a lot of overlaps with [1], making its theoretical contributions unclear. Its contributions seem standard and combinatorial.

[1] Suzuki, Taiji, and Atsushi Nitanda. "Deep learning is adaptive to intrinsic dimensionality of model smoothness in anisotropic Besov space." Advances in Neural Information Processing Systems 34 (2021): 3609-3621.

[2] Shi, Zhaoyang, Krishna Balasubramanian, and Wolfgang Polonik. "Adaptive and non-adaptive minimax rates for weighted Laplacian-Eigenmap based nonparametric regression." International Conference on Artificial Intelligence and Statistics. PMLR, 2024.

[3] Chazal, Frédéric, Ilaria Giulini, and Bertrand Michel. "Data driven estimation of Laplace-Beltrami operator." Advances in Neural Information Processing Systems 29 (2016).

### Questions
I may have some misunderstand of the paper but please clarify the questions from Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper establishes the theoretical convergence rate of Kolmogorov–Arnold Networks (KANs) in nonparametric regression. By representing the univariate components of KANs with B-splines, the authors connect KANs to classical sieve estimation theory. They prove that both additive and hybrid KANs achieve the minimax-optimal convergence rate under Sobolev smoothness assumptions. Experiments on synthetic data confirm that the empirical convergence behavior matches the theoretical rate.

### Strengths
1. This paper provides a complete theoretical treatment of KAN convergence rates, bridging the empirical success of KANs with rigorous nonparametric learning theory. This significantly advances the theoretical foundation of KAN research.

2. By parameterizing univariate KAN components using B-splines, the authors successfully unify KANs with the sieve estimation framework, showing that structured neural architectures can inherit the statistical optimality of spline estimators while maintaining interpretability.

3. The simulation experiments are well aligned with the theoretical analysis, showing empirical slopes near the theoretical exponent

### Weaknesses
1. The theoretical results rely on restrictive assumptions, such as all univariate functions belonging to Sobolev spaces with known smoothness and bounded constants, which may limit their applicability to practical, high-dimensional learning settings.

2. The experimental validation focuses solely on smooth synthetic functions under low-noise settings. No results are provided for higher-dimensional, noisy, or real-world datasets, limiting the empirical generality of the conclusions.

3. The paper compares KANs only to standard MLPs. Including baselines such as DeepONet, SIREN, or kernel regression networks would more clearly demonstrate the practical advantages of KANs in terms of interpretability, stability, and efficiency.

### Questions
See the weakness.

### Soundness
3

### Presentation
3

### Contribution
3
