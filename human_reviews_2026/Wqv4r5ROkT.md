# Exogenous Distribution Learning for Causal Bayesian Optimization

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
Maximizing a target variable as an operational objective within a structural causal model is a fundamental problem. Causal Bayesian Optimization (CBO) approaches typically achieve this either by performing interventions that modify the causal structure to increase the reward or by introducing action nodes to endogenous variables, thereby adjusting the data-generating mechanisms to meet the objective. In this paper, we propose a novel method that learns the distribution of exogenous variables-an aspect often ignored or marginalized through expectation in existing CBO frameworks. By modeling the exogenous distribution, we enhance the approximation fidelity of the data-generating structural causal models (SCMs) used in surrogate models, which are commonly trained on limited observational data. Furthermore, the ability to recover exogenous variables enables the application of our approach to more general causal structures beyond the confines of Additive Noise Models (ANMs) and single-mode Gaussian, allowing the use of more expressive priors for context noise. We incorporate the learned exogenous distribution into a new CBO method, demonstrating its advantages across diverse datasets and application scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes EXCBO, a Causal Bayesian Optimization method that explicitly learns the distribution of exogenous noise variables in structural causal models rather than marginalizing them out. The authors use an encoder-decoder framework to recover exogenous variables from observational data and model their distribution with Gaussian Mixture Models. They introduce the Decomposable Generation Mechanism (DGM) as a generalization of Additive Noise Models (ANM), prove exogenous recovery under DGM, and provide regret analysis. Experiments on synthetic and real-world datasets show EXCBO can outperform baselines when noise is multimodal with moderate variance.

### Strengths
1. This paper addresses a gap in existing CBO methods by explicitly modeling exogenous distributions rather than marginalizing them out or assuming simple Gaussian noise.

2. The encoder-decoder framework is intuitive and practical, using standardization to recover exogenous variables from observational data. The DGM formulation is more general than the ANM.

### Weaknesses
1. The authors use 2-component Gaussian Mixture Models to model $p(\hat{U})$ without any justification. Could the authors elaborate on a) why GMMs were chosen over other flexible density estimators? b) Why exactly 2 components? c) How sensitive are the results to this choice?

2. Regarding the theoretical analysis part:

a) The authors mention that there exists a constant $a$ in Theorem 4.1, but according to Equation (16), $a = \mathrm{sign}[f_b(z)/c]$ actually depends on $z$. Please clarify whether a is truly constant or the independence claim needs modification.

b) For BGM (Theorem F.2), $\hat{U} \perp\perp Z$ is explicitly assumed as a premise. For DGM (Theorem 4.1), it is claimed to be proven as a conclusion. However, the DGM proof relies on $a$ being constant (which is a concern in 2(a)).

c) In line 1037, the authors stated that $\sigma_{\phi}(z) = c|f_b(z)|$. Can the authors elaborate on whether this equation is correct?

3. Regarding the experiments:

a) MCBO is missed in the Dropwave experiments in Figure 4. I wonder if the authors could provide justification for it? From Figure 9, it seems that MCBO performs comparably with EXCBO on the Dropwave experiments.

b) Figure 4 shows EXCBO's advantage decreases as $\lambda$ and $\sigma$ increase. Can the authors elaborate more on it and provide more explanation?

4. The paper is notation-heavy with many symbols, which makes it a little bit hard to understand and follow. It would be better if the authors could add more illustrative examples to improve clarity

5. (Minor) The indentation and margin of the beginning of many paragraphs (e.g., the first paragraph in Section 3, Section 3.4) should be adjusted.

### Questions
Please see the questions in the Weaknesses part.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces EXCBO, a causal Bayesian optimization framework that relaxes the restrictive additive Gaussian noise assumption by estimating the exogenous variable distribution from data.
Using an encoder decoder surrogate (EDS), EXCBO recovers latent residuals, models their distribution with a Gaussian Mixture Model, and integrates the estimated p(U) into the optimization process.
Empirical results on synthetic and real-world structural causal models show improvements over standard CBO baselines under multimodal or non-Gaussian noise.

### Strengths
Strengths
Clear motivation: Identifies a genuine limitation in prior CBO frameworks that assume additive Gaussian noise, a well-motivated problem in causal optimization.

Intuitive methodology: The encoder-decoder formulation for exogenous recovery is conceptually sound and connects structural causal modeling with modern regression techniques.

Reasonable empirical evidence: Experiments demonstrate that EXCBO improves performance in settings with multimodal or non-Gaussian noise distributions, aligning with its theoretical motivation.

Practical relevance: Learning a more realistic noise model can be useful for real-world decision-making tasks

### Weaknesses
Weaknesses
Conceptual and Theoretical
Incremental novelty: The paper combines known ideas from heteroscedastic Gaussian Processes, nonlinear ICA, and latent-variable Bayesian optimization rather than introducing fundamentally new theory or algorithms.

Limited theoretical depth: The recoverability theorem is a restatement of standard residual properties under independence. The regret bound simply inherits results from GP-UCB without considering estimation uncertainty from the exogenous step.

No analysis of identifiability or robustness: The paper does not explore what happens when the DGM assumption fails, when noise is correlated with parents, or when the graph is misspecified.
Algorithmic
Minor procedural change: The algorithm is essentially FNBO plus a residual normalization and GMM fitting step. The “learning” component is non-iterative and computed once before optimization.

Omission of MCBO: The most relevant baseline (MCBO) is missing from Figures 4 and 5. The justification (“computationally expensive”) is weak and unsupported by runtime data.

Unexplained high initial rewards: In the reward progression plots, EXCBO starts significantly higher than other methods. The paper does not explain whether EXCBO uses pretraining or a different initialization, raising concerns about comparability.

Experimental
Inconsistent reporting: The number of experimental runs or seeds is clearly stated (four) only for the Dropwave dataset. Other benchmarks have uncertainty bars but no run count.

Limited noise diversity: Only Gaussian and two-component Gaussian mixture noises are tested. There are no experiments with heavy-tailed, skewed, or heteroscedastic noise beyond the DGM structure.

No robustness or ablation studies: The effects of the encoder-decoder, the GMM modeling, or independence violations are not separately tested.

Partial tabular reporting: Tables 1 and 2 summarize results for small-scale tasks, but not for larger experiments or real-world cases.

Writing / Presentation
Clear overall, but contains typographical errors ( “STATMENT”, “LLMS”, extra parenthesis in “do(XI := f(ZI, A, UI) )”.

Related works is missing critical prior work related to Heteroskedastic Gaussian Processes, Latent Variable Bayesian Optimisation and Non Linear ICA, all closely related to this work

Minor formatting inconsistencies in math expressions and figures.

 Minor Comments
Ensure consistent reporting of the number of runs/seeds across all experiments.

Add a clear explanation for EXCBO’s higher starting reward to rule out unfair initialization.

Include MCBO results (even partial) or provide runtime justification with quantitative data.

Standardize reference formatting and include missing citations to heteroscedastic GP and nonlinear ICA literature.

Proofread for typographical errors listed earlier.

### Questions
see above

### Soundness
2

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
4

### Summary
The authors built on top of the Sussex MCBO/Aglietti CBO work, incorporating ideas from the exogenous variable learning literature. 

They have successfully incorporated previous reviewer feedback, it seems, and improved their presentation and results.

I am happy to increase my score once my three questions are addressed.

### Strengths
- The authors present an interesting, novel contribution to the CBO literature, placing it well in context and current literature.
- Exploring the incorporation of EX in CBO is valid and this contribution is therefore relevant to readership.

### Weaknesses
- See questions

### Questions
1. Why do you not benchmark against CBO by Aglietti et al as a baseline? Is this conceptually incompatible? The code is available and runs without code changes, though requires careful specification of initialisation points, AFAIK.
2. Why do the convergence plots in Figure 6 seem to have different starting points? Presumably, they were initialisation with the identical random sample. Please clarify!
3. “Each figure presents the mean performance over four random seeds” Why can’t you run more seeds? Four seeds seems to be enough for distinguishable error bars, but I am just curious.

Thanks in advance!

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes EXCBO, a causal Bayesian optimization method that learns the distributions of exogenous variables to better model multimodal noise, in the decomposable generation mechasm. The authors incorporated these learned exogenous distributions into the Bayesian optimization process to improve sample efficiency and regret performance.

### Strengths
The paper is clearly written and easy to follow, and the proposed method is conceptually straightforward.

### Weaknesses
### Weak real-world motivation / failure case.
Without demonstrating or discussing the failure mode of existing works, it's hard to be motivated why we need the proposed method. 

### Restrictive design (one action per node).

The method assumes a known mapping where each intervened variable $X_i$ has its own continuous action $A_i$ that directly enters its mechanism. I think it's too restrictive, since in the system, either we want to know the best hard-intervention, or best soft-intervention (find the optimal action values AND their parents). It's hard to come up with a real-world scenario that will be matched with this assumption.

### \tau-SCM appears to rebrand a standard Markovian assumption.

The \tau-SCM is simply an SCM with X = f(Z,U) where Z and U are independent. This is a usual Markovian SCM assumption. I don't see any reasons why authors create a new terminology for already existing notions. Also, it's unclear what \tau stands for. 

### Definition 3 (EDS) is not mathematically rigorous.

Defining the encoder/decoder via “a regression model such that E[X] exists and ϕ() can model the conditional mean µϕ() and variance σϕ().", as in Def. 3, is not considered as a mathmatical definition with no rigoursness. 

### Extra smoothness assumptions vs prior CBO.

Their identification results require differentiability, whereas prior GP-based CBOs don't assume differentiable structural f in the model statement. Therefore, the claimed “generalization” depends on added smoothness conditions.

### Contradiction: $\hat U=h(Z,X)$ but $\hat U\!\perp\! Z$

Theorem 4.1. mentioned that $\hat U \perp  Z$, while $\hat U$ is a function of $Z$; i.e., $\hat U=h(Z,X)$. I think this is contradictory.

### Questions
1. If we infer $U$, we can actually recover the SCM. Then a better optimization algorithm can be found. Please discuss. 

2. This model's performance depends on the performance of the encoder models. Please take this account in your errro analysis.

### Soundness
2

### Presentation
2

### Contribution
1
