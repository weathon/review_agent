# Robust Equation Structure Learning with Adaptive Refinement

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6, 4

## Abstract
Symbolic regression (SR) aims to automate scientific discovery, but often truncates the hypothetico–deductive cycle, focusing on hypothesis and experiment while lacking systematic analysis. We introduce RESTART, a framework that closes this loop by adding a principled analysis stage to diagnose and correct structural errors. RESTART features two core mechanisms: a short-term refinement process that uses boosting to identify unexplained signals and guide an LLM toward targeted corrections, and a long-term structure library that distills successful refinements into reusable code snippets for cumulative knowledge. On LLM-SRBench across Physics, Biology, and Materials Science, RESTART achieves lower error and higher accuracy than state-of-the-art baselines. It also generalizes robustly, recovering near-exact functional forms on out-of-distribution data, representing a significant advance toward fully automated scientific discovery.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes RESTART (Robust Equation STructure learning with Adaptive RefinemenT), a symbolic regression (SR) framework that explicitly closes the hypothesize → experiment → analyze loop. 

In experiments, RESTART reports lower NMSE and higher symbolic accuracy than existing methods.

### Strengths
- The proposed method achieves better performance than existing LLM-based symbolic regression systems.

- The modular design that separates initialization, refinement, and structure retention could enable future extensibility.

### Weaknesses
Overall, this paper suffers from jargon-heavy and somewhat hand-wavy writing. Many parts need clearer, more concrete explanations and consistent notation. Below are detailed comments for revision:

Line 82: The term “structure proposal” is unclear. Please define explicitly what constitutes a structure—e.g., is it an equation template, code snippet, or symbolic expression tree?

Line 83: Define “equation exemplars.” Are these example equations used in few-shot prompts, or previously discovered hypotheses stored in a buffer?
Line 84: The notation for ( q_p ) and ( \tau_{1:k} ) is confusing. Replace ( \tau_{1:k} \sim q_p^k ) with the clearer and standard form ( \tau_i \sim q_p,\ i=1,\ldots,k ). Avoid the unconventional ( q_p^k ) notation.

Line 85: Explain “best-of-k” clearly—does it refer to selecting the lowest-loss sample among ( k ) generated equations?

Line 146: Remove the period in the section title. Also, “EFFICIENT INITIALIZATION” is vague—rename it to something more descriptive, such as “Initialization via Mapping-Based Estimator.”

Line 150: Specify what nonlinear and cross-variable dependencies are being modeled. Clarify what “using another estimator” means here—does the initialization merely provide a stronger approximation, or is it computationally more efficient?

Line 158: Equation (3) is syntactically incorrect. Replace ( g_t \in \arg\min_g L(g(f_t(x),x)) ) with the correct assignment form ( g_t \gets \arg\min_g L(g(f_t(x),x)) ). Additionally, clearly define ( t ) (iteration index) and ( g_t ) (the symbolic residual or exploration function). The following explanation seems incoherent—please rewrite it with precise mathematical meaning rather than generative phrasing.

Line 202: Clarify what ( L ) denotes (loss function) and what ( r^{f_t} ) refers to (is it the residual, ( y - f_t(x) )?).

Line 226: Variable names are inconsistent—change ( S_r ) → ( s_r ) and ( S_a ) → ( s_a ) to match prior definitions.

Line 231: Define what you mean by a “high-value structure.” Also, specify what the symbol h represents in the tuple ( (name, desc, h) ).

Line 295: Explain all symbols in the NMSE equation: ( y ) (true target), ( \bar{y} ) (mean of targets), and ( \hat{y} ) (predicted output). Introduce them before using the formula.

Line 359: The font size of Table 2 is excessively large. Remove unnecessary LaTeX sizing commands and ensure uniform table formatting.

Section 5.2.2: The example expression ( \sqrt{c_1 |x_1| / x_2} ) does not appear particularly challenging; classic search-based SR methods can likely recover it easily. Provide a stronger justification for why this case is nontrivial.

Line 450: In the expression “( i\Delta = E - V_0 + \hbar J_0 ),” remove the leading i—it appears to be a typographical error.

Figure 5a: Define both ( R^2(\text{data}) ) and ( R^2(\text{equation}) ) explicitly. Clarify how each is computed (e.g., is ( R^2(\text{data}) ) the fit to measured data and ( R^2(\text{equation}) ) the symbolic agreement with the analytical form?).

### Questions
Please try to address the concerns raised in the previous section and provide a revision of the paper during the rebuttal.

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
This paper introduces RESTART, an LLM-based symbolic regression framework that combines LLM-based hypothesis generation with several guidance mechanisms: (1) efficient initialization, (2) short-term refinement via boosting-style residual analysis, and (3) long-term knowledge retention through a structure library. Experiments on LLM-SRBench show better performance compared to baselines including LLMSR, with good efficiency gains.

### Strengths
* The framework makes intuitive sense: fitting symbolic functions to residuals provides data-driven feedback rather than generic error signals, and the structure library of motifs offers a principled way to accumulate good patterns.
* Good evaluation setup and empirical results on LLM-SRBench problems. The efficiency analysis showing comparable performance with 25% of iterations is valuable. Figures are well-designed and good ablation studies are provided in the main paper and the appendix.
* The writing is clear, literature coverage is reasonable, and the appendix provides useful implementation details. The problem motivation around completing the scientific discovery cycle is well-articulated.

### Weaknesses
* Limited novelty: The contribution consists of fairly straightforward additions to the LLM-SR framework: pre-trained initialization, residual fitting for guidance, and a pattern library. While sensible, these feel incremental: residual analysis resembles gradient boosting adapted to SR, and structure libraries conceptually overlap with recent concept library approaches, though implemented differently.
* Residual fitting assumption: The boosting formulation fits g(x) to the residual (y - f_t(x)), which appears to assume the missing component is additively separable from the current hypothesis. This may not hold when the true structure involves multiplicative or other nonlinear compositions with respect to f_t(x). See questions for details.
* Evaluation gaps: Only numeric accuracy metrics are reported (NMSE, Acc_threshold), not symbolic correctness. Variation across problems isn't visualized beyond 4 selected examples. The case study lacks baseline comparisons and doesn't provide the prompt used. Several design choices (fitness scoring, ablation scope) need better justification. See questions for specifics.

### Questions
1. Most SR backends like E2E don't condition on f_t(x) form. Is g actually receiving f_t(x) or just fitting x to the residual? What exactly does each subproblem backend receive?

2. How does the method handle cases where the missing structure isn't additively separable (e.g., f*(x) = h(x) · k(x))? Can you provide benchmark examples and show if performance degrades on such problems? Intuitively, the residual model would not work well in such cases and would suggest functions that are not appropriate. This might guide the framework towards predicting more complex linear forms (potentially with nonlinear basis functions). To be clear, even with this limitation, the additional component could be valuable in many use cases, but it would be helpful to discuss this in the paper and provide analysis on the effect on final discovered forms.

3. Can you report symbolic accuracy metrics (e.g., symbolic recovery like in LLM-SRBench) alongside numeric error? Can you show performance distributions across all problems rather than just aggregated means?

4. For the case study: what was the exact prompt used? How do baselines like LLM-SR perform on the same problem? The problem is published in 2025, after the cutoff date of many LLMs, but how do you rule out that the LLM is recalling or deriving the simple equation form rather than discovering it?

5. What does "Additive" mean in Figure 4? Why are the ablations conducted only on 2 problems, and how are they selected? Is there an ablation on the fitness score design (absolute, relative, and weighted s_fit)?

Suggestions:
* I would suggest authors to bring examples of prompts and how the two mechanisms work to the main paper for clarity. 
* I was wondering if the authors could provide the code for reproducibility and evaluation.
* minor: some typos exist in the paper (for example: "Libarary -> "Library" in Figure 1)

### Soundness
3

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
5

### Summary
This paper proposes RESTART, a robust large-language-model-based framework for symbolic equation discovery that integrates principled short-term and long-term analytical guidance into the discovery process. The authors evaluate RESTART on 239 problems from the recent LLM-SRBench benchmark and report substantial improvements over the state-of-the-art baseline LLM-SR. The paper is generally well-motivated, clearly written, and supported by extensive experimental evaluation and insightful analyses.

### Strengths
- Although individual ideas (boosting-style refinement, structure library) are not novel alone, their integration into the discovery framework is novel and seem to provide considerable performance improvement. 
- Experiments on a wide range of datasets from LLM-SRBench are thorough and demonstrate consistent gains over state-of-the-art baselines.
- The paper includes comprehensive ablation studies and a meaningful real-world physics case study which strengthen their evaluation.
- The paper is generally very well written and easy to follow.

### Weaknesses
- I noticed that authors are only reporting results on 93 out of 128 LSR-Synth from the llm-srbench benchmark. why is that and what criteria is behind this selection of 93 out of 128 problems?

- In pg 6, Acc₀.₁ is described as the symbolic accuracy from [1], but it is actually a stricter numeric metric rather than the symbolic accuracy (SA), which in [1] is computed via an LLM-as-judge for symbolic equivalence. I would suggest authors to also report SA in their results (see fig. 11 in [1]).

- The analysis in case study of sec 5.2.2 on real experimental data is insightful, however, with the current report of results it's not clear what are the symbolic forms of equation discovered by RESTART and how's the advantage of these new hypothesis compared to the ones from LLM-SR baseline. Providing symbolic forms similar to Figure 12 for some examples or at least this case study could be helpful to better understand the problem.

- Some abrupt jumps in Fig. 4a suggest high variance in the observed results. Are these from a single run or averaged over multiple runs? Can you also report the confidence of these curves across different runs? 

- Authors have not shared to the anonymous version of code in their submission. It would be good if authors can also share their code if paper is accepted so that research community can also build on it.

**Minor Comment:**
- It would be helpful to include the LLM backbone model in the captions of Tables 1 and 2 for clarity.

[1] LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models, ICML 2025

### Questions
check weaknesses section

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
5

### Summary
The paper “Robust Equation Structure Learning with Adaptive Refinement (RESTART)” presents a novel framework for Symbolic Regression (SR) that explicitly models the full loop of hypothesis → experiment → analysis → refinement using large language models (LLMs).

While prior SR methods have typically lacked a distinct analysis stage, RESTART introduces a refinement mechanism driven by error analysis, making it the core of its contribution.

The system consists of three main modules:
1. Informative Initialization – Uses existing mapping-based estimators (e.g., E2E) to generate initial hypotheses.
1. Targeted Refinement (short-term guidance) – Analyzes residuals and guides the LLM to make localized improvements.
1. Long-term Structure Library (long-term guidance) – Stores reusable symbolic structures as executable code snippets.

Experiments on LLM-SRBench (covering physics, biology, and materials science) show that RESTART achieves 20–30% reduction in NMSE, +5–10% improvement in Symbolic Accuracy, and better OOD generalization than baseline SR systems.

A case study also demonstrates the rediscovery (with correction) of a physical law reported in a Nature paper (July 2025 issue), illustrating RESTART’s potential for real-world scientific discovery.

### Strengths
1. Formalizing the “Analyze” stage in SR

   * RESTART defines “analysis” as an explicit symbolic sub-task ( g_t(f(x), x) ) that learns residual errors and guides refinement in a *boosting-like* manner.
   * The LLM is not just a generator but an *autonomous analyst*, producing targeted corrections based on error structure—a clear conceptual advance.

2. Two-level guidance design (short-term and long-term)

   * The model separates immediate error-based refinement from cumulative structure learning.
   * Unlike multi-agent or evolutionary systems, RESTART combines *directionality* (via residual analysis) with *knowledge reuse* (via a library).

3. Comprehensive baseline comparison

   * Evaluated against search-based (e.g., GP), mapping-based, and LLM-based SR methods (LLM-SR, LaSR, PySR, AI-Feynman).
   * RESTART shows consistent improvements in both ID and OOD conditions, supporting its analytic-guided hypothesis refinement.

4. Empirical validation through case study

   * Successfully rediscovered a quantum particle energy–velocity relation from *Nature* (2025/7/3).
   * This demonstration links RESTART’s symbolic reasoning to genuine scientific modeling tasks.

### Weaknesses
1. Ablation study limited to NMSE only

   * Figures 4b and 4c report NMSE-based comparisons only.
   * The analysis does not include *Symbolic Accuracy* (structural correctness of equations), which is crucial for validating RESTART’s core claim—enhanced symbolic understanding.

2. Contribution of “Analyze” phase not isolated for structure accuracy

   * While NMSE degradation is shown when omitting the Targeted Refinement module, the improvement in symbolic structure accuracy is not quantified.

3. Potential data leakage concerns

   * The *Nature* case study uses data from July 2025, but the paper does not ensure that the LLMs used (GPT-4o, Qwen3, DeepSeek) were trained before that publication.
   * Without such guarantees, the rediscovery results risk reflecting memorization rather than inference.

4. Lack of theoretical grounding

   * The residual decomposition function ( g_t ) lacks convergence analysis or formal justification.
   * No quantitative discussion on search space reduction induced by the Analyze stage.

5. Unaddressed SR-specific pitfalls

   * Generated equations can still be overly complex (spurious terms).
   * The paper does not explain how the Structure Library maintains the validity or consistency of stored code snippets.

### Questions
1. How does the inclusion of the Analyze phase quantitatively improve Symbolic Accuracy (beyond NMSE)?
1. Can you guarantee that the LLMs used were not trained on the *Nature (2025)* article or its equation?
1. How is the quality of the Structure Library maintained—how do you prevent incorrect or redundant structures from accumulating?
1. Do you plan to include structural metrics (e.g., symbolic edit distance, tree isomorphism rate) in future evaluations?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents RESTART (Robust Equation Structural Learning for Symbolic and Causal Discovery), a unified framework aimed at closing the loop of scientific discovery by coupling symbolic regression, robust structural learning, and long-term knowledge accumulation. 
The authors evaluate RESTART on LLM-SRBench across Physics, Biology, and Materials Science domains, as well as synthetic causal-discovery settings with noisy or adversarial perturbations. Results show that RESTART outperforms state-of-the-art GP-, RL-, and LLM-based symbolic regression systems in both in-domain accuracy and out-of-distribution (OOD) generalization, while remaining more robust to noise and spurious correlations.

### Strengths
1. The integration of symbolic regression with robust structural learning is good. The paper operationalizes the full scientific-discovery cycle, something most prior works treat only partially. The two-level guidance mechanism (boosting + structure library) provides a compelling framework for iterative theory refinement.
2. Experiments span multiple scientific domains, several baseline categories (GP, RL, mapping, LLM), and realistic noise regimes. Ablations and the physics case study demonstrate how each component contributes to performance and stability. The OOD gains are particularly meaningful for genuine scientific discovery.

### Weaknesses
1. While the integration is original, the constituent elements (boosting-based residual learning, NOTEARS-style differentiable structure learning, structure libraries) each draw heavily from prior art. Clarifying what RESTART fundamentally adds beyond combining them would strengthen the contribution.
2. The two-layer optimization (boosting subproblems + adversarial robustness) roughly doubles compute. Table 4 lists per-solver times but lacks an end-to-end comparison normalized by total wall-clock or FLOPs. Claims about “25 % fewer iterations’’ are anecdotal without systematic cost analysis.
3.. RESTART introduces many tunables (fitness-gating, adaptive weights, divergence parameters, temperature schedules). The paper does not analyze sensitivity or provide heuristics for new domains.
4. The paper highlights success stories but gives little insight into when the approach fails.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
