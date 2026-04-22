# VARSHAP: A Variance-Based Solution to the Global Dependency Problem in Shapley Feature Attribution

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Feature attribution methods based on Shapley values, such as the popular
SHAP framework, are built on strong axiomatic foundations but suffer
from a critical, previously underappreciated flaw: global dependence. As
recent impossibility theorems demonstrate, this vulnerability is not
merely an estimation issue but a fundamental one. The feature
attributions for a local instance can be arbitrarily manipulated by
modifying the model's behavior in regions of the feature space far from
that instance, rendering the resulting Shapley values semantically
unstable and potentially misleading.

This paper introduces VARSHAP, a novel feature attribution method that
directly solves this problem. We argue that the source of the flaw is
the characteristic function used in the Shapley game — the model's
output itself. VARSHAP redefines this game by using the reduction of
local prediction variance as the characteristic function. By doing so,
our method is, by construction, independent of the model's global
behavior and provides a truly local explanation. VARSHAP retains the
desirable axiomatic properties of the Shapley framework while ensuring
that the resulting attributions are robust and faithful to the model's
local decision landscape. Experiments on synthetic and real-world
datasets confirm our theoretical claims, showing that VARSHAP provides
stable explanations under global data shifts where standard methods fail
and demonstrates superior performance, particularly in robustness and
complexity metrics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents VarSHAP as a novel value function for Shapley-based feature attribution. VarSHAP reformulates the classical value function looking at the expected model output (i.e. class-wise logits for classification, model output for regression) as a variance of the model output. The paper shows that this new value function addresses some known issues with feature attribution scores. 

The following references are used in the remainder of the review:

## References for Review
- Fumagalli et al. 2025: https://proceedings.mlr.press/v258/fumagalli25a.html
- Sobol, 2001: https://www.sciencedirect.com/science/article/abs/pii/S0378475400002706
- Bordt, 2024: https://proceedings.mlr.press/v206/bordt23a/bordt23a.pdf

### Strengths
- **Significance:** The paper studies an important question, in that attribution methods if applied poorly may lead to inconsistent or wrong interpretations. Therein, the paper addresses a good research topic and proposes an interesting solution. Simply reformulating the value function and thus introducing a quick fix to a problem allows the traditional black box estimation methods (KernelSHAP or Shapley interaction methods) to still be applicable.
- **Synthetic Evaluations:** I like the use of synthetic evaluations for analyzing the core concepts of the work.
- **Well Written:** The paper is generally well written.

### Weaknesses
- **Framing and Motivation:** In my opinion, it is generally well known that the feature distribution plays a crucial role in interpreting feature-based explanations such as feature attribution. The recent AISTATS paper (Fumagalli et al, 2025) unifies this issue by linking cooperative game theory with functional ANOVA decomposition. The resulting framework makes it quite clear that the more information is modeled in the removal mechanism (conditional vs. baseline) the more influence the distribution has on the resulting explanations (e.g. attributions). This is generally not new and in my opinion not a drawback but an important positive side-effect of these methods. In your experiments (Section 3.1) you are **changing** the model function to be explained. While, yes at $x=(0,0)$ both models predict 0 output since you designed them to be identical at this point. But around this single point the models do not behave the same because of the introduction of an interaction between $X_1$ and $X_2$. This interaction is also influencing the attribution of the $X_2$ feature since the Shapley value is influenced by all interactions in the model (this is also the second dimension in the framework in Fumagalli et al, 2025). If you were to compute all Shapley interactions (for example after Bordt (2023)) then the attribution of $X_2$ will become zero again since it does not influence the model individually and there are no correlations between the $X_1$ and $X_2$. Hence, my point is that this _problem_ is not really a problem but a correct behavior of the attribution methods. I do not think that this is per-se a drawback for looking into different value functions (like it is done here) but I do think that this discussion should be **stronger substantiated** and compared to with more important **related methods** addressing this issue already (Sobol indices, Shapley interactions). 
- **No Source code Available**: The submission contains **no** code. This is quite a big problem for proper reviewing. I wanted to check the computational efficiency of the variance estimation (which the authors write is easy, but do not show), when I noticed this.
- **Limited Evaluation:** While I wholeheartedly agree that for a contribution like this a proper synthetic evaluation is absolutely necessary (which the paper includes), but the synthetic examples (with three datasets) basically makes up the whole experimental evaluation. Section 3.3 paints a relatively unclear picture of the empirical implications of the proposed value function change. VarSHAP seems to be on par with the traditional value function. It is unclear how easy it is how to translate the paper to different data modalities and/or implement it in traditional data science pipelines. Experiments on real world data would greatly improve this.
- **Missing Ablations:** The paper does not analyze the influence of traditional parameters such as dataset size, feature size, data modalities to VarSHAP. This makes it quite challenging to gauge the practical implications of the method.

### Questions
- **Q1:** How does your value function compare against Definition 5 (page 6) of Fumgalli et al. (2025)? How does the new value function relate to the well known Sobol indices?

- **Q2:** What are computational limits of your method? In lines 286-293 you describe the issue but you do not show any analysis in your empirical evaluation (particularly in regards to Section 3.3). Is this a bottleneck?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a new feature attribution method, named VARSHAP,  that aims to eliminate the global dependence problem that affects the approximations of KernelSHAP. Instead of using the model’s direct output as the characteristic function, the proposed method defines it as the reduction in prediction variance.
The authors claim that VARSHAP maintains the axiomatic properties of Shapley values. The empirical evaluation shows that VARSHAP can provide robust attributions.

### Strengths
-The paper proposes a novel idea that approximates the Shapley values, but for a different objective compared to the traditional Shapley value explainers.

-The paper is generally clear and easy to follow. 

-The evaluation includes case studies designed to measure certain properties of the proposed explanation method, as well as a benchmark evaluation.

### Weaknesses
-The authors claim, in line 66, that the problem is in the Shapley value concept itself, yet the proposed solution uses the same Shapley value concept. I find this statement quite strong and potentially incorrect. I agree that the solution proposed by KernelSHAP can be inaccurate and can be improved, and that is not an issue of an implementation, but that does not extend to the Shapley value concept.

-The proposed approach uses marginal expectations to marginalize features out of coalitions, but does not discuss the baseline removal approach [1].

-One of the desired properties of Shapley value methods is the local accuracy, i.e., the solution matches the prediction of the underlying model, which makes their interpretation intuitive. On the other hand, VARSHAP proposes values that sum to "the negative of the initial total variance under full local perturbation", which I think is unintuitive and makes it difficult to explain the predictions, especially when the user is not an expert in machine learning or a statistician. 

-The proposed method violates the consistency property of Shapley values with respect to the original prediction, i.e., the Shapley value increases or stays the same if a player’s contribution grows or stays the same. In other words, if a feature $\beta$ negatively affects the prediction and a feature $\gamma$ positively affects it, but both ($\beta$ and $\gamma$) have similar effects on the prediction variance, both will be assigned the same importance, which I think makes the interpretation of the outcome more challenging. This property is promoted under the "sign independence", which I cannot understand why it can be considered a desired property.

-KernelSHAP, which can be no better than random guessing (according to the paper), is outperforming the proposed VARSHAP with respect to fidelity. Additionally, VARSHAP never outperformed the competitors with respect to faithfulness in the LATEC benchmarking. 

-I doubt that the proposed approach is showing superior performance to KernelSHAP, as claimed in the conclusions. Additionally, VARSHAP has worse computational complexity than KernelSHAP.

-The method is not compared to the unbiased KernelSHAP [2], which, given a sufficiently large number of samples, converges to the true Shapley values. I also think it addresses the global dependence that has been put as the central problem of this paper.


[1]-Sundararajan, M. and Najmi, A. The many Shapley values for model explanation. In III, H. D. and Singh, A. (eds.), Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 9269–9278. PMLR, 13–18 Jul 2020.

[2]-Covert, I. and Lee, S.-I. Improving kernelshap: Practical Shapley value estimation using linear regression. In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, volume 130, pp. 3457–3465, April 2021.

### Questions
1- Why can the sign independence be considered a desired property?

2- How to explain the outcome of VARSHAP to a user who is not an expert in machine learning?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This manuscript identifies a feature of inherent in Shapley values, namely the global dependence identified by Bilodeau et al. (2024): as Shapley's value is a global estimator, it takes into account 'value added' across the data distribution.  Thus, 'local' explanations (e.g. feature importance for the third observation) can be influenced by altering 'irrelevant' relationships in the data.

For example (q.v. $\S 3$), consider Dataset 1, comprising of patient types A and B.  Now consider Dataset 2, identical to Dataset 1, but adding a new patient type, C.  Shapley's value for a patient in group A changes from data sets 1 to 2, although the data-generating process has not changed.

The manuscript introduces VARSHAP which, instead of performing a weighted sum of 'value added', performs a weighted sum of variances.  Thus, whereas Shapley's value measures the change in a model's prediction resulting from knowing a feature's actual value (rather than a baseline value), VARSHAP measures the change in the variance of the model's predictions.

As the variances are derived from perturbations of the Gaussian, they are thin-tailed, so effects fade quickly with distance - attenuating the global dependence identified in Bilodeau et al.

In addition to the example above, the paper presents:
1. ($\S 3.2$) an example in which LIME (a linear, local approximation method) is fooled by a non-linear feature, while SHAP and VARSHAP correctly identify its irrelevance.
1. ($\S 3.3$) LATEC benchmark results comparing faithfulness, robustness and complexity scores of VARSHAP, SHAP and LIME on a range of models and datasets.  The authors conclude, ``For faithfulness, KernelShap ... often ranks top [while] VARSHAP variants excel in robustness ... [and] complexity metrics''.

### Strengths
Generally, I think that there are strong arguments for interpreting large ML models - and, thus, that there is an ongoing need for good research in this area.

Further, the authors extend Bilodeau et al.'s argument from marginal SHAP attributions to conditional SHAP attributions.

Finally, the example in $\S 3.1$ does raise concerns about how SHAP and LIME have been applied in that environment.

### Weaknesses
The feature attribution literature has been strongly rooted in Shapley's value and variants.  Most contributions to the literature identify perceived mathematical weaknesses of e.g. the Shapley value, and propose a mathematical variant to resolve it.

A small literature, though, returns to the motivating question: does the proposed measure help us explain or interpret the ML model?  This paper is detached from that question.  As such, in my view, the motivation is immediately weakened: yes, this is a tweak on the Shapley value that one can perform and, yes, it scores well in some metrics - but leaves open the question of whether it does anything to help the metrics implicitly motivating the literature: do developers or users better understand?

On the mathematics alone, the perturbations assume independence of features (p.4).  I would regard this as a secondary concern: any novel paper leaves open research questions.

Independently of this, I found the exposition less crisp than I would have liked:
1. There is a lengthy verbal introduction that relies on assertions, rather than carefully defining terms or providing intuitions.  This left me unclear about what was being asserted, why it was a problem, and so on.  

1. When seeing a modification of Shapley, I would like to know early on which Shapley axioms are being replaced, and with what.

1. Some of the assertions seem misleading.  For example, the description of SHAP opening $\S 2.1$ is vague enough to seem to better describe a partial derivative than Shapley's value.

1. Proposition 1 uses terms like "aims to satisfy" and "fundamentally".  Functions don't have purpose: they satisfy or don't; I don't know what it means to "fundamentally" measure.  As the proposition's conclusion is that the function "must take the following general form", one would expect - instead - a phrasing like: "Suppose that f satisfies axioms A, B, C and D.  Then f has the form..."  Further, A, B, C and D should be defined before they are used - rather than a page later.  (Similarly, on p.6, the word "precisely" tends to be added to sentences.)

1. Section $\S 3.3$ *could* be the strongest argument for VARSHAP, showing more general performance, rather than just in special cases.  For this to help the authors' argument, though, it needs to be properly set up rather than raced through: *explain* faithfulness, robustness and complexity metrics in a sentence or two, to convince the reader that this mean something from an interpretability point of view.

### Questions
For the example given in $\S 3.1$, how was set membership encoded?  If $\\{A, B, C \\}$ is a feature set, then the example seems to be looking for an interaction between set membership and the $X$ variables - something that the base Shapley value cannot disentangle.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper revisits the Shapley feature attribution framework and proposes VARSHAP, which replaces the traditional expected-value characteristic function with a variance-based one.
The authors show that under three axioms (zero property, sign independence, additivity), the unique valid transformation is $d(x) = x^2$, leading to a variance-based expected marginal contribution. The method thus measures each feature’s contribution to the variance of the model output rather than its raw expectation.
They claim this resolves the “global dependency” flaw of SHAP, where correlated features distort importance scores. Empirical results on synthetic datasets and benchmark interpretability tasks demonstrate that VARSHAP produces stable and faithful attributions, particularly when strong dependencies exist between features.

### Strengths
Clear and formally correct derivations.

Theoretical framing aligns Shapley, variance decomposition, and interpretability in a unified narrative.

Empirical results are consistent and easy to reproduce.

Addresses an important practical flaw (feature correlation) with simple mathematical machinery.

### Weaknesses
Limited novelty: Strong overlap with Sobol sensitivity analysis; the variance-based approach is not a new concept.

Under-citation: Lacks acknowledgment of prior equivalence results (e.g., Da Veiga, S. (2021)) showing Sobol and Shapley variance connections.

No empirical stress test: Only toy regressions; no nonlinear or high-dimensional domains.

Overclaiming impact: The method does not “solve” dependency. It merely changes the objective from mean to variance.

No practical pipeline: Absent complexity or estimator analysis for real SHAP implementations.

Unclear interpretability gain: It is debatable whether variance-based attributions are easier or harder to interpret in applied settings.

References:
Da Veiga, S. (2021). Kernel-based ANOVA decomposition and Shapley effects--Application to global sensitivity analysis. arXiv preprint arXiv:2101.05487.

### Questions
How exactly does VARSHAP differ mathematically from Sobol total-effect indices $S_{T_i}$?

Can you prove that VARSHAP satisfies the same orthogonality decomposition as functional ANOVA?

How would VARSHAP behave if the model output has negligible variance but large mean shifts?

Would replacing variance with another second-moment measure (e.g., covariance of residuals) produce similar properties?

Can you quantify computational complexity compared to KernelSHAP and Sobol estimation?

### Soundness
3

### Presentation
3

### Contribution
2
