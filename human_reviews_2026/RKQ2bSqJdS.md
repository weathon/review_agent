# Beyond Formula Complexity: Effective Information Criterion Improves Performance and Interpretability for Symbolic Regression

- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Symbolic regression discovers accurate and interpretable formulas to describe given data, thereby providing scientific insights for domain experts and promoting scientific discovery. However, existing symbolic regression methods often use complexity metrics as a proxy for interoperability, which only considers the size of the formula but ignores its internal mathematical structure. Therefore, while they can discover formulas with compact forms, the discovered formulas often have structures that are difficult to analyze or interpret mathematically. In this work, inspired by the observation that physical formulas are typically numerically stable under limited calculation precision, we propose the Effective Information Criterion (EIC). It treats formulas as information processing systems with specific internal structures and identifies the unreasonable structure in them by the loss of significant digits or the amplification of rounding noise as data flows through the system. We find that this criterion reveals the gap between the structural rationality of models discovered by existing symbolic regression algorithms and real-world physical formulas. Combining EIC with various search-based symbolic regression algorithms improves their performance on the Pareto frontier and reduces the irrational structure in the results. Combining EIC with generative-based algorithms reduces the number of samples required for pre-training, improving sample efficiency by 2~4 times. Finally, for different formulas with similar accuracy and complexity, EIC shows a 70.2% agreement with 108 human experts' preferences for formula interpretability, demonstrating that EIC, by measuring the unreasonable structures in formulas, actually reflects the formula's interpretability. We provide code and data in https://anonymous.4open.science/r/EIC-91B2.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new evaluation metric in the field of symbolic regression—the Effective Information Criterion (EIC). This metric relates "the loss of significant digits or the amplification of rounding noise as data flows through the system" to the structural reasonableness of formulas, assessing expression quality from the perspective of numerical stability. By incorporating EIC into the evaluation and search process to filter out numerically ill-conditioned expressions, the authors claim that this criterion helps generate more interpretable formulas.

### Strengths
1) EIC introduces a new evaluation metric from the perspectives of numerical analysis and information theory, to some extent addressing the limitations of traditional complexity and accuracy measures.
2) EIC is able to filter some unreasonable nested expressions (e.g., exp(exp(exp(x)))) or numerically ill-conditioned formulas.
3) The authors validate EIC within GP and MCTS-based symbolic regression frameworks, across multiple scenarios including noisy data, black-box problems, and human interpretability assessments.
4) On the SRBench dataset, GP methods incorporating EIC achieve complexity–accuracy Pareto frontier performance.

### Weaknesses
(1) Lack of in-depth analysis and validation regarding the effect and generalization ability of EIC across different datasets. 
(2) Absence of intuitive evidence demonstrating the practical improvement and interpretability brought about by the EIC.
(3) Insufficient discussion regarding the potential limitations of EIC
(4) Some experimental details are not clear.

### Questions
1)	On the white-box subset of SRBench, is the improvement consistent across most datasets, or concentrated in specific ones? What proportion of datasets show improvement, and what is the average improvement magnitude? Are there notable cases of accuracy degradation? Additionally, have the authors analyzed characteristics of datasets that benefit most (e.g., variable count, input scaling) to assess the generality of the method?
2)	In scientific domains where variables exhibit drastically different numerical scales, does EIC’s evaluation criterion affect formula discovery under real sampling ranges? For example, in the SRSD dataset, where variables and constants are physically meaningful, some formulas involve variables with ranges $U_{log}(10^{-29},10^{-27}) and U_{log}(10^6,10^8)$ . How does EIC behave under such multi-scale conditions?
3)	Since the core contribution lies in proposing EIC, it would be informative to show representative formula examples with and without EIC (e.g., EIC-GP vs. vanilla GP). Listing several illustrative formulas would make it easier to see how EIC improves interpretability in practice.
4)	The paper claims that EIC facilitates discovery of true underlying formulas. Could the authors report the solution rate improvement? Moreover, as $R^2 > 0.999$ typically indicates precise fits, it would be helpful to report how many formulas reach this stricter threshold under the EIC-based approach.
5)	Please clarify whether the reported $R^2$ values are based on the median or the mean across runs.
6)	In the "expert preference experiment" (Section 4.4), it is recommended to report inter-rater agreement (e.g., Fleiss’ κ) to strengthen the statistical reliability of expert evaluations.
7)	Since EIC measures the amplification of propagated numerical errors, would directly applying it during the search process risk false penalization of inherently sensitive but valid functions (e.g., $exp(x^2 + y^2)$ or Gaussian functions)?
8)	Please check the paper for formatting and typographical issues. For example, line 456 should refer to Figure 8 (not Figure 14), and “Implementaion” should be corrected to “Implementation”.

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
5

### Summary
The paper proposes the “Effective Information Criterion” (EIC), which is intended to have physical meaning (i.e., significant digits loss, noise gain) for equations. EIC is then used as a proxy for interpretability in symbolic regression (SR), which is argued to be better than length in SR. The paper then uses EIC as a form of regularization in SR algorithms, and they also perform experiments to check the alignment of EIC with what human experts think is interpretable.

### Strengths
The presentation, details and motivation from lines 1 to 188 are of high quality. The paper introduces the work in an intuitive manner that makes it seem natural that this is a ‘right’ approach.

I am convinced by the paper that the motivation of EIC is well justified and would be impactful if executed well.

The alignment with human experiments is valuable experiment to the SR community and is a strong addition to their other results.

### Weaknesses
Multiple errors at the core of the paper, specifically from line 189 onwards, with some errors which I am not even sure are fixable. I list the key reasons which affected my score below. **If addressed adequately in the rebuttal, I will increase the score by multiple stages to be in the acceptance range**.

First, in line 189, the equation given, f(x) = (x+10^100) – 10^100, is given as an example with suffers from “catastrophic cancellation”, is this not just a data structure/programming issue? For example, is this not just an issue of using the wrong data structure for numerical precision? Also, simplification (which has been done in SR [1]) mechanisms can remove this error. It is not clear why this example is relevant to EIC, is the EIC of this equation not just 0, according to equation (4)? What is the point of this equation when on one hand the paper claims it is “bad” because it suffers from catastrophic cancellation, yet under the new contribution, EIC, it is perfect at a value of 0? Is this not a contradiction?

[1] Cao, Lulu, et al. "Genetic programming symbolic regression with simplification-pruning operator for solving differential equations."

Second, it seems that the computation in the experiments highly depend on the underlying data structure (e.g., float64, binary128, etc., or even a custom one) used to perform computations. This is not stated clearly in the paper, and I am unable to find what computation precision is used for the experiments.

Third, the LARGEST error, which is the key reason for the "reject” score, is in lines 207-208, when giving the equation “$M = 1 - \frac{1}{2} \log_{10}(12\delta_r^2)$”. This relation is a major part of EIC, and affects the correctness of the WHOLE paper. There is no justification for how the equation is obtained, and it is flawed if the paper is assuming that the output noise is uniform. Is the paper saying that for equations in SR, say f(x)=1/x, injecting a uniform random noise to x will result in a uniform noise to f(x) as well? I suspect the paper made the serious mistake of assuming that uniform input noise will result in uniform output noise, which is not true for many equations in SR (e.g., non-linear operators), even with small noise.

Fourth, lines 194 to 196 is problematic too. The paper claims “truncating a variable x to N significant digits can be expressed as…”. This can be disproved with a simple counter example, where x=3, and we choose N=1, then using the definitions (e.g., the range of noise in line 195 and M=log_10|x| in line 196), x_hat ranges from about 1.5 to 4.5. Here, 1.5 is not a truncation of x=3 to N=1 significant digits and 4.5 is not a truncation of x=3 to N=1 significant digits either. This affects the “physical meaning” of EIC.

Fifth, the white-box and black-box results are inconsistent in the paper itself. For instance, there are algorithms which are present in Figure 6 which are NOT present in the corresponding Table 7. Likewise, there are algorithms which are present in Table 7 which are NOT present in the corresponding Figure 6. Also, using aggregated “rank” in Figures 5 and 6 (note: the computation of ranking is not clear in the paper so I am assuming the same process as SRBench is used) has been shown to lead to paradox where the algorithms on the Pareto frontier is different when the absolute value of the metrics are used instead of “ranks” [2]. It would be better to show a plot with the axis using the absolute value of the metrics, instead of aggregated ranks, this way, the absolute difference in metric values can be seen as well. Finally, confidence intervals/error bars should be added to the plot as well, as done in the EIC plot.

[2] Fong, Kei Sen, and Mehul Motani. "Pareto-Optimal Fronts for Benchmarking Symbolic Regression Algorithms."

### Questions
Please address each of the 5 weaknesses above, especially the third weakness. Critically, for the third weakness, please provide a clear, step by step, reasoning for how “$M = 1 - \frac{1}{2} \log_{10}(12\delta_r^2)$” is obtained.

### Soundness
1

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
5

### Summary
This paper proposes a new criterion, EIC (Effective Information Criterion), which extends the traditional concept of formula complexity to encourage more rational structure discovery in symbolic regression and scientific modeling.
EIC balances data fit and structural information and is presented as a generalization of AIC/BIC, aiming to achieve model simplicity, interpretability, and robustness simultaneously.

The authors incorporate EIC into multiple symbolic regression methods (GP, MCTS, E2ESR, SNIP, SR4MDL) and evaluate it on the SRBench benchmark.

### Strengths
- Novel theoretical perspective:

  The idea of extending AIC/BIC to account for structural information is conceptually original. EIC offers a quantitative measure of “structural rationality,” which is a meaningful addition to symbolic regression research.

- Consistent empirical benefits:

  Incorporating EIC reduces overfitting and yields simpler, more plausible symbolic expressions without severely compromising predictive performance.

- Breadth of experiments:

  The method is tested across several benchmarks and models, showing stable improvements.

- Improved interpretability:

  Both human and LLM-based evaluations indicate that equations produced under EIC are perceived as more interpretable, suggesting that EIC aligns with human notions of simplicity and meaningfulness.

### Weaknesses
- Lack of rigorous theoretical grounding:

  While EIC is information-theoretically sound, its formal relationship with AIC/BIC or MDL (e.g., asymptotic equivalence, bias correction) is not derived.
  The framework lacks a clear statistical foundation comparable to classical information criteria.

- Missing baseline comparisons:

  The paper claims that EIC improves training stability, yet it does not include direct comparisons with other regularization techniques.
  As a result, the unique contribution of EIC relative to established regularizers is unclear.

- Empirical parameter tuning:

  The weighting coefficient $\alpha$ is chosen empirically (grid search or scheduling) without a theoretically justified selection rule.

- Limited linkage between theory and implementation:

  The theoretical notion of “effective information” and its numerical estimation (via Jacobian perturbation) are not tightly connected in the exposition.

- Related work could be deepened:

  The paper would benefit from a more systematic comparison to the Information Bottleneck theory, MDL, and other information-based regularization frameworks.

### Questions
1. How does EIC differ from the other baselines in terms of its effect on generalization and optimization dynamics?
1. Does EIC possess any asymptotic or consistency properties similar to AIC/BIC when viewed as an information criterion?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Effective Information Criterion (EIC) to score formulas in a symbolic regression (SR) context by how stably they process information under finite precision. The claim is that EIC correlates with interpretability, improves search when added as a regularizer, and improves sample efficiency for "generative SR" by filtering pretraining data. The experiments cover "white-box" and "black-box" tasks, search methods, three generative models, and a small human study.

So, I appreciate the motivation: length alone is not a good proxy (strong agree), and numerical fragility shows up in SR outputs. But in some sense this is unavoidable, and the solution proposed in this paper introduces even larger problems. In its current form, I have major concerns about EIC that leads me to recommend rejection. Fundamentally, EIC looks to not be invariant to how a formula is written or to the choice of units, which is an even bigger problem than the formulation of SR already has. I don't believe this concern can be addressed by a rewrite of the paper. Furthermore, the evaluation mixes roles (search loss vs selector) in a way that makes it hard to separate real gains.

### Strengths
- Clear motivation and an easy-to-implement scoring procedure.
- Breadth in evaluations using search integration, generative pretraining filter, and a human preference study.
- The general idea of downweighting cancellation-prone expressions is useful as a diagnostic.
- Good writing and visuals; the narrative is straightforward to follow.

### Weaknesses
1. EIC is not invariant to algebraic rewrites or to units/scales. 3*x and x/0.3333... are algebraically identical yet score differently because EIC depends on the concrete parse/encodings. This means that even changing units (kg <-> lb) changes constants and can flip selections. For scientific use this is a non-starter, and in SR representation choices materially affect behavior.
2. Taking the maximum over all subformulas is an ad-hoc patch with no apparent theoretical basis. It hard-codes representation dependence (parenthesization/temporaries change the max) and penalizes algebraically harmless subterms. I view this as a fundamental issue with the approach, and the patch as a symptom of a deeper issue.
3. The noise model is internally inconsistent and lacks sensitivity analysis. The derivation assumes uniform rounding <-> significant digits, the implementation uses Gaussian relative noise, and sigma is fixed while calling the method "parameter-free." There is no sigma sweep and no stated near-zero policy for $(\tilde{y} - y)/y$. Rankings can flip under these choices.
4. The evaluation is circular and confounded. The paper introduces a metric, optimizes it during search, then declares wins under that metric; baselines are not optimized for EIC. Reported R^2/length shifts are also confounded by in-loop coefficient refitting, early stopping that peeks at test, an EIC threshold chosen from the test distribution, and dropping ~39% low-R^2 runs. These choices inflate the gains.
5. I think the human study is a bit of an overreach on interpretability. The human study uses low-D problems, preselects pairs to have large EIC gaps, and does not report inter-rater reliability or per-task effect sizes. Using LLM votes as a proxy for expert judgment does not strengthen the claim. At best this shows a correlation in a narrow setting.
6. I believe the evaluation is entangling search with selection, and this makes it seriously difficult to evaluate if there is any actual benefit to this method, ignoring the theoretical inconsistencies. So, in practice, SR will yield a frontier of accuracy vs complexity, and users might pick a mid-complexity point for generalization. But, the procedure effectively bakes in "pick mid-complexity" and then credits EIC for the generalization, while other methods might simply (by default) select the most accurate expression, which is high complexity. That does not demonstrate that EIC measures something fundamental. To fix this, I think you need to separate the roles of search and selection. I recommend using EIC for selection, and then evaluating using some existing search code.
7. The meaning of the "tiers" on Pareto plots is not defined precisely.

### Questions
1) If I canonicalize each expression (symbolic simplify and constant rationalization) and then recompute EIC, how often does the ranking change on your benchmarks? Please provide some examples.
2) How sensitive are the main results to sigma, to the choice of uniform vs Gaussian noise, and to the near-zero policy for $(\tilde{y} - y)/y$?
3) Can you separate EIC-as-selector and evaluate only that, compared to some other selection method?
4) What is the exact definition of the Pareto "tiers" in the method comparison plot, and how are ties and noise handled?
5) Why should a scientist use a formula selection criteria that is highly sensitive to unit and representation choices?

### Soundness
1

### Presentation
3

### Contribution
2
