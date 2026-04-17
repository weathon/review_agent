# Learning Dynamic Causal Graphs Under Parametric Uncertainty via Polynomial Chaos Expansions

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 2

## Abstract
Existing causal discovery methods are fundamentally limited by the assumption of a static causal graph, a constraint that fails in real-world systems where causal relationships dynamically vary with underlying system parameters. This discrepancy prevents the application of causal discovery in critical domains such as industrial process control, where understanding how causal effects change is essential. We address this gap by proposing a new paradigm that moves beyond static graphs to learn functional causal representations. We introduce a framework that models each causal link not as a static weight but as a function of measurable system parameters. By representing these functions using Polynomial Chaos Expansions (PCE), we develop a tractable method to learn the complete parametric causal structure from observational data. We provide theoretical proofs for the identifiability of these functional models and introduce a novel, provably convergent learning algorithm. On a large-scale chemical reactor dataset, our method learns the dynamic causal structure with a 90.9% F1-score, nearly doubling the performance of state-of-the-art baselines and providing an interpretable model of how causal mechanisms evolve.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a novel and elegant method for learning dynamic causal graphs, where the causal relationships change as a function of system parameters. The problem of discovering dynamic causal structures is both timely and challenging, and the proposed algorithm, which cleverly combines concepts from different families of causal discovery methods, represents a relevant and interesting contribution.

While the core idea is strong and I am inclined to recommend acceptance, the paper in its current form could be further improved by addressing several smaller issues concerning ambiguity, adding more detailed discussions, and expanding the empirical evaluation. I believe that incorporating these suggestions will further increase the paper's impact and clarity.

### Strengths
*   **Novelty and Significance:** The paper tackles the important and under-explored problem of learning causal graphs that are not static but change dynamically with system parameters.
*   **Elegant Methodological Contribution:** The proposed algorithm provides a novel synthesis of constraint-based and score-based causal discovery principles, creating a powerful and interesting hybrid approach.
*   **Timeliness:** The research area is of high interest to the machine learning community, and this work is a valuable early contribution to a nascent field.
*   **Theoretical rigor:** The authors put visible effort into providing a solid theoretical analysis of their proposed algorithm.

### Weaknesses
### **Areas for Improvement**

The paper is promising, but its clarity and empirical validation could be strengthened in several key areas.

**1. Clarity and Methodological Precision:**

*   **Missing Details and Citations:** Several sections are very dense and would benefit from additional detail and citations to support the claims made. This is particularly true for:
    *   Line 161 onwards: A citation is needed for the statement made here.
    *   Line 205: Please explicitly name the specific optimization algorithm used in your work.
    *   Line 223: The origin or justification for this property is unclear. Please elaborate.
    *   Section 220 onwards: This section is difficult to follow. Please expand on the concepts presented and provide citations for established results to improve comprehensibility.
    *   The index `α'` appears to be used without a formal definition in the main text.

*   **Ambiguity of Temporal Dimension:** There seems to be a discrepancy in the problem formulation. Equation (1) presents a static formulation without a time dimension. However, Equation (5) and later references to time-series causal discovery algorithms imply that the data `X` has a temporal component. Please clarify whether the proposed method is designed for time-series data, i.i.d. samples, or both. This is a crucial detail for understanding the method's scope and applicability.

*   **Acyclicity Constraint:** Score-based causal discovery algorithms typically deploy an explicit acyclicity constraint in addition to a sparsity regularizer. The objective function only appears to include a sparsity penalty ($\lambda \Vert (E, \Theta) \Vert_0$). Could you clarify if an acyclicity constraint is used, and if so, how it is enforced? If not, please explain how the acyclicity of the learned graph is guaranteed.

**2. Experimental Evaluation and Analysis:**

*   **Baseline Performance Analysis:** In the experiments, the static baseline algorithms perform surprisingly poorly, even though the causal parameters in the dataset have a strictly positive range. In such a case, one might expect a static method to converge to average coefficients, which, however, should still represent the proper causal connections.  Could the authors comment on why the baselines fail so decisively? Is this potentially due to hyperparameter choices? A more in-depth discussion would improve the reliability of the empirical results.

*   **Value of Synthetic Experiments:** While the chosen real-world benchmark is interesting, the paper would be strengthened by including synthetic experiments. This would allow for a more controlled, fine-grained analysis to:
    *   Demonstrate precisely under which conditions (e.g., speed of parameter change, noise levels, sample size) the proposed method outperforms static baselines.
    *   Identify the limitations and potential failure modes of the algorithm.

*   **Additional Datasets:** The empirical evaluation currently relies on a single ground truth graph. To demonstrate broader applicability and robustness, the authors should consider evaluating on additional benchmarks. There are several well-established datasets that could be suitable, such as:
    *   Time-series benchmarks: [4], [5]
    *   I.I.D. sample benchmarks with varying contexts: [6], [7]

*   **Missing Related Work:** The discussion of prior work could be expanded. While the field is new, there are some relevant works that should be cited and discussed, such as [2] and [3].

### **Minor Weaknesses**

*   **Terminology:** As suggested by [1], the authors might consider using "Structural Causal Model" (SCM) instead of "Structural Equation Model" (SEM) to avoid confusion with the different usage of the term in the social sciences.
*   **Cross-Referencing:** Please add backlinks to algorithm listings (e.g., on lines 196, 234) and to proofs in the appendix to improve readability and navigation.

### Questions
1.  The proposed method uses a hybrid approach: skeleton discovery via a constraint-based method, followed by a score-based method for orientation and parameterization. Is this two-stage process strictly necessary? Have you explored whether a purely constraint-based approach (e.g., a PC-style algorithm using your proposed conditional independence test) is theoretically sound or empirically viable?
2.  A discussion on the limitations and potential failure modes of the proposed algorithm would be highly valuable for readers. Under what conditions (e.g., type of dynamics, data scarcity, violation of assumptions) would you expect the method to perform suboptimally?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors leverage polynomial chaos expansion to identify time-varying causal graphs with parametric uncertainty. Polynomial chaos expansion appears to be an adequate tool for a relevant but relatively understudied topic in causal inference. Therefore, the paper certainly has merit. The proposed approach outperforms several standard baselines.

### Strengths
1) The paper is overall well-written and easy to follow.

2) The considered problem setting is clear and relevant, and nicely embedded into the bigger streams of causal inference literature.

3) The usage of polynomial chaos expansion is innovative.

4) The experiments show a clear advantage over existing methods.

### Weaknesses
1) While I agree that time-varying causal graphs are not studied very often, I think it is not fair to say that all existing methods are limited by the assumption of a static causal graph. There is prior work that addresses this problem setting and deserves discussion. For instance, work on dynamic Bayesian networks (Song et al., "Time-varying dynamic Bayesian networks," NeurIPS 2009), or Huang et al., "Causal discovery and forecasting in nonstationary environments with state-space models," ICML 2019, address similar problem settings. Thus, I think the claims should be toned down a bit as well.

2) A similar comment on the claim that traditional causal discovery assumes that all uncertainty is epistemic and rises solely from finite data samples. There is a lot of work on additive noise models, where the uncertainty is clearly not purely epistemic, even if it is not about parameter uncertainty.

3) Some claims when developing the independence test require a bit more justification, see questions. It would also be good to state that proofs can be found in the appendix.

4) Comparing to some methods that themselves consider time-varying graphs would strengthen the evaluation.

5) References to the algorithm are shown as ??

### Questions
1) What are the "standard regularity conditions" that you assume for the independence test?

2) From where does it follow that under these standard regularity conditions, the estimates are asymptotically independent and follow a standard normal distribution?

### Soundness
3

### Presentation
3

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
The paper presents a method for learning causal relationships between a
collection of measured process variables, where the strength of causal links
depend on uncertain parameters.
This type of model is relevant in process control applications, for instance,
where operating conditions can change the causal graph of process variables.
The authors propose to model the relationships between each process variable
and its `causal parents' as linear equations, where the coefficients are
functions of the parameters. They use Polynomial Chaos Expansion (PCE)
to represent the coefficient functions.
Based on this model, the authors develop a conditional independence test used
to initialise the learned causal graph. This is improved using a natural
gradient method to optimise a likelihood score.
Gradient updates are weighted with the Fisher matrix to improve convergence.
On the theoretical front, the authors provide an identifiability theorem, a
sample complexity bound and an analysis of the convergence of the gradient
method, using standard assumptions.
Finally, the method is validated on an industrial dataset, where it achieves
state-of-the-art performance across a variety of standard metrics.

### Strengths
* Original formulation of the problem in a new setting, together with a
  computationally tractable method. The integration of PCE is original and
  appears to be a very good fit for the addressed problem.

* Comprehensive theoretical analysis of the proposed method.

* Experimental validation demonstrates strong performance compared to
  alternative methods in the literature.

### Weaknesses
* The assumption of the parameters $\xi$ having a known distribution seems quite
  restrictive; the paper would benefit from a discussion of this and possible
  workarounds.

* As per the paper the polynomial basis functions used in PCE are dependent on
  the distribution of $\xi$. Section 4 does not discuss what assumptions were in
  place regarding the distribution of $\xi$, or in fact which basis was used for
  the experimental results.

* It's peculiar that the main algorithm, presented in appendix A.5, is not part
  of the main body of the paper. Additionally, there are a lot of steps there
  which are not discussed elsewhere in the paper: "non-Gaussianity/residual
  methods", "Test for nonlinear relationship using MI and residual analysis"
  (the abbreviation MI seems to be undefined), "Consider adding edge
  (i, j)" (what is meant by "consider" here?). Unless these steps are explained, at
  least in the appendix, the paper seems to lack reproducibility.

### Questions
* The wording of Assumption 2 is unclear. In "or the collection of coefficient
  functions is non-degenerate", what is meant by "or"? Is this an equivalent
  reformulation of the previous statement, or an alternative but orthogonal
  assumption that will lead to the same results?

* Related to the above, in the proof of Thm. 1, it seems to me that
  non-degeneracy is explicitly used to derive eq. (25). However, the text refers
  to Assumption 1, the relevance of which I am not able to see here. Is this a
  typo?

* Regarding eq. (15) in the proof of Thm. 1: is it clear that the elements of
  $A$ are in $L^2(\Xi)$, so that it can be expanded in this way?

* The text of the proof of Theorem 2 could be improved, e.g., by writing an
  expression for $\hat{\theta}_{ij,\alpha}$; avoid writing that the expectation
  is "approximately $P\sigma_\epsilon / m$, being more explicit; do not redefine
  $\kappa$ but refer back to Section 3.3.

* In Section 2.4, it is written that the Fisher matrix is block-diagonal, but in
  the proof of Thm. 3, this is refined to diagonal. This should be corrected for
  consistency. More generally, some more details on the derivation of the
  expression for the Fisher matrix would be appreciated.

* Please provide references in Section 2.2 for the principal results
  mentioned there, such as the choice of basis depending on the distribution,
  the decay of the spectral error, and the hyperbolic truncation schemes.

* Note the several broken references to the Algorithm environment.

### Soundness
4

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
4

### Summary
Summary. The paper proposes PCT-CD, which models edge weights as functions of operating parameters via Polynomial Chaos Expansion, claims identifiability, and reports large gains on a single industrial dataset.

### Strengths
Strengths.
- Clear statement of goal: parameter-dependent causal effects. 

- Simple decomposition of pipeline: CI testing in the PCE basis, then score-based refinement. 

- Readability: paper is generally well written and easy to follow.

### Weaknesses
1. The introduction asserts that existing score-based methods “provide only point estimates without uncertainty quantification,” and that constraint-based methods relying on independence tests are insufficient in practice. These are too narrow and partially incorrect. There is a long line of Bayesian and bootstrap approaches for DAG posteriors and uncertainty over graphs and parameters; many constraint-based methods are paired with robust CI procedures and stability devices. The paper needs careful scoping and citations when criticizing “traditional” methods.

2. The text (line 185) suggests that “traditional CI tests… are insufficient,” then replaces them with a bespoke PCT-CI definition without an empirical comparison against strong CI tests matched to data characteristics. The use of  PCT-CI is not justified methodologically. The argument reads as assertion rather than evidence. Provide comparisons to well-tuned kernel-based CI, partial correlation with robust estimation, or recent conditional mutual information estimators, on synthetic and real settings.

3. Key background on parameter-varying or context-specific causal discovery, time-varying DAGs, covariate-dependent SEMs, and distribution shift causal structure is largely missing. The PCE/QoI citations cluster in early classics (Wiener 1938; Xiu & Karniadakis 2002) and very recent 2024–2025 engineering pieces, leaving a gap spanning two decades. This pattern suggests coverage of a few base methods plus several very recent items rather than a thorough survey. Expand the related work to include context-specific independence, conditional DAGs, regime-switching causal models, and Bayesian DAG posteriors with uncertainty. 

4. All results come from one proprietary refinery dataset with 9 variables and 11 asserted ground-truth edges. This is neither diverse nor standard. The head-to-head table mixes methods with very different assumptions and tuning needs, without transparent hyperparameter search, preprocessing, or split protocol. Strong directed baselines might perform well on such process-control data if tuned with domain knowledge; a single domain does not support claims of general superiority. Use public benchmarks (synthetic with controlled parametric variation, cause-me style datasets, time-varying DAG suites), release code, and report robust model selection.

5. The comparison list includes methods that assume static graphs and others that model nonlinearity, but the paper then generalizes conclusions about “traditional methods.” If the claim is about advantages under parametric variation, show controlled synthetic experiments where the ground-truth edge functions vary with ξ, sweep PCE order and noise types, and compare to alternatives explicitly designed for nonstationary or context-specific structure. Current evidence is insufficient to support broad claims.

Minor issues:
- Typographical placeholders (“Algorithm ??”) and inconsistent section cross-references. 

- Ambiguity about whether ξ’s joint law is known a priori or estimated, and how misspecification impacts tests and scores. 

- The statement that score-based methods “provide only point estimates” ignores common uncertainty add-ons; rephrase and cite carefully.

### Questions
1. You state that “traditional methods provide only point estimates and ignore uncertainty.” Which classes of methods are you referring to specifically, and which uncertainty-aware baselines did you exclude and why?

2. Is the distribution of the operating parameters ξ assumed known or estimated from data? If estimated, how sensitive is your method to misspecification?

3. Does the causal graph change with ξ (structure-varying), or are only edge weights functional in ξ while the graph is fixed? Clarify the formal model class.

### Soundness
2

### Presentation
2

### Contribution
2
