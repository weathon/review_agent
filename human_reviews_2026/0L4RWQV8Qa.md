# LLM-based Symbolic Regression with Tool-Augmented Multi-Objective Optimization

- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
Symbolic Regression (SR) aims to discover analytical equations from observational data and plays a central role in scientific modeling. While recent Large Language Model (LLM) based approaches show promise, they face two key limitations. First, they lack dedicated data analysis mechanisms to uncover variable dependencies, which reduces the efficiency of equation discovery. Second, most methods rely on single-objective evaluation focused solely on fitting error. This neglect of structural complexity and generalization often causes models to converge prematurely to local optima, limiting their ability to explore the broader equation space. To address these issues, we propose Tool-Augmented Multi-Objective Symbolic Regression (TAMOSR), a unified framework that integrates external analytical tools (e.g., correlation analysis, mutual information, periodicity detection) to extract structural priors and guide equation generation, while simultaneously optimizing for accuracy, complexity, and generalization via a multi-objective evaluation module with a dynamic Pareto front. TAMOSR employs two collaborative LLM modules: a Meta Strategy Generator, which selects tools and synthesizes structural optimization strategies based on Pareto-optimal equations, and an Equation Generator, which produces new candidate equations accordingly. The system operates in a closed loop, continuously refining both strategies and equation structures. Experiments on diverse scientific benchmarks demonstrate that TAMOSR outperforms existing SR methods in accuracy, generalization, and search efficiency, offering a scalable and adaptable paradigm for scientific discovery.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a LLM-driven SR framework. Two cooperating LLMs are used: a Meta Strategy Generator (chooses tools, summarizes patterns from the current Pareto set) and an Equation Generator (produces candidates from those strategies).

### Strengths
* The curated toolbox is well-motivated.
* The listed case studies look nice.

### Weaknesses
* The paper repeatedly references LLM-SR but never provides a clear definition or background. Readers unfamiliar with LLM-SR may struggle to understand the method and results fully.
* The set of comparison methods is too small and the number of datasets is limited, which weakens claims of generality and robustness.
* Key steps are not described in sufficient detail. For example, how the Meta Strategy Generator and Equation Generator interact, and how the multi-objective optimization actually guides the search and selection of candidates.
* There is little experimental rationale for selecting LLaMA-3.1-8B-Instruct and GPT-4o-mini. The paper should explain these choices and include sensitivity analyses across alternative models.
* Similarly, the paper does not provide empirical justification for the chosen analysis tools. It should clarify why these tools were selected and how much each contributes to performance (e.g., via ablation studies).
* The paper lacks mean±std over independent seeds and significance tests across all tables/plots.
* The experiments do not include comparative analyses of key hyperparameters. Sensitivity studies are needed to assess robustness and to guide practical deployment.

### Questions
Here are some suggestions for the authors:
* Specify the core workflow.
* Expand baselines and datasets; ensure fair budgets.
* Provide empirical justification for base LLM choices (LLaMA-3.1-8B-Instruct & GPT-4o-mini) and add a small model-selection study over alternatives.
* Add tool selection ablations and contribution analysis.
* Provide Sensitivity experiments to key hyperparameters.

### Soundness
3

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
5

### Summary
This paper proposes TAMOSR, an LLM-based framework for symbolic regression that integrates external analytical tools and multi-objective optimization into LLM-based equation discovery. Two cooperative LLMs work iteratively: a Meta Strategy Generator selects tools and synthesizes strategies from Pareto-optimal equations, while an Equation Generator produces candidates. The system optimizes for accuracy, complexity, and generalization simultaneously. Experiments on physics, chemistry, and biology benchmarks show improvements over existing methods including LLM-SR and PySR.

### Strengths
* The integration of analytical tools for variable relationship extraction is well-motivated and appears useful for this domain. The tools cover multiple aspects of data analysis that could inform equation structure, and this opens further exploration of this research direction.
* The paper is clearly written with comprehensive ablations in the appendix. The choice of baselines is reasonable, figures are informative, and the tool categorization is detailed.
* Multi-objective optimization for symbolic regression balancing accuracy, complexity, and generalization is sensible. The use of Pareto fronts and diversity-guided sampling could encourage exploration beyond single-objective approaches.

### Weaknesses
* Potential test data leakage in optimization loop: The multi-objective evaluation explicitly includes NMSE_OOD during training iterations. This means the OOD test split is used for model selection and equation refinement throughout the optimization, not just for final evaluation. However, standard practice would use only training data, during optimization, then evaluate on held-out test sets once. This undermines claims as the model has been optimized (trained) on the test. Tables 1-2 report aggregated metrics without ID/OOD breakdown, while Figure 3 shows them separately, which is inconsistent.
* Baseline performance discrepancies and lack of reproducibility: Some of the reported baseline numbers such as LLM-SR results appear substantially worse than those in the original LLM-SR paper (Shojaee et al.). Even accounting for different model versions, this gap is concerning. Additionally, attempting to reproduce the reported equations (e.g., the Oscillator 1 equation from the main paper and appendix) on the dataset does not yield the claimed 1e-15 or 1e-13 NMSE values. And the code is not released, making verification impossible.
* It is not clear if the small variations of results in some problems (e.g., oscillators) are statistically significant or run-to-run variations. And no mention is found of the number of runs or any uncertainty quantification.

### Questions
1. Can you clarify the data splits used during optimization? Specifically, is NMSE_OOD computed on the actual test OOD split provided in LLM-SR and LLM-SRBench repos during the iterative loop?
2. Why are the some of the baseline results (in particular LLM-SR) worse than reported in their original paper? Can you provide details on baseline reproduction?
3. Can you please provide the final discovered equations by TAMOSR for other benchmark problems such as oscillator 2, bacterial growth, and stress-strain? The final equations for Oscillator 1 are reported to have NMSE in the order of 1e-15 to 1e-13, but independent verification of these equations on the dataset yields very different results. Are there additional components or settings that would explain this? Would it be possible for the authors to provide the code for review and reproducibility?
4. The tool ablation shows removing any category hurts performance, but how sensitive is the method to the specific tool selection by the Meta Strategy Generator? Is there analysis on which tools are selected most frequently, and whether the LLM's tool choices are actually meaningful or just arbitrary?
5. Tables 1-2 report aggregated metrics while Figure 3 shows ID/OOD separately. What test set does table 1-2 indicate? Could you provide complete ID/OOD breakdowns in tables for all methods?
6. How many random seeds/runs were conducted for each experiment?

### Soundness
2

### Presentation
4

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
The paper proposes TAMOSR, a framework for symbolic regression that integrates LLM-guided equation discovery frameworks with external analytical tools and a multi-objective optimization process. The system employs two cooperating LLMs, a Meta Strategy Generator that performs variable and structure analysis and an Equation Generator that synthesizes new equations.

### Strengths
- The paper is ambitious in scope, attempting to combine analytical tool invocation, structural analysis, and multi-objective optimization in a unified loop for LLM-based equation discovery frameworks.

### Weaknesses
- Some parts of the writing are very unclear. For instance, in sec 3.2, the authors include OOD performance as one of the feedback objectives within the discovery loop. This is conceptually incorrect as OOD data should remain unseen during the discovery process and reserved for final evaluation. Using OOD feedback in the optimization effectively leaks test information into training. 
- The paper reports results on only 36 out of 128 LSR-Synth tasks from LLM-SRBench. The selection criteria for these subsets are unclear and should be justified. What's the selection criteria for these 36 out of 128 problems? The larger-scale of datasets in LLM-SRBench allows to evaluate methods in a more generalizable setting and across different metrics like symbolic accuracy (SA)
- What's the symbolic accuracy (SA) for results reported in the Table 2? I would suggest authors to also add this metric to their reported comparison results.
- It seems that the LLM-SR results in Table 1 and Figure 4 appear notably worse than those reported in the original paper [1], even though this work uses gpt-4o-mini which is a stronger model than gpt-3.5-turbo used in the original paper (also verified in [2]) The authors should clarify this discrepancy.
- Figure 4 shows some abrupt jumps that suggest high variance in models' performances. Are these curves based on a single run or averaged over multiple runs? I would suggest to also report confidence intervals or variance estimates across runs for clarity.
- Could the authors clarify which dataset and LLM backbone are used for results in Figure 5? Also, why rely on HV and IGD metrics which seem to be older and less intuitive metrics for recent program space of hypotheses in current LLM-based frameworks? If the goal is to measure symbolic proximity to ground truth, it would be better to use symbolic accuracy/equivalence (SA)  metric with the LLM-as-judge metric from [2] which allows fairer comparison with the existing baseline results already in the benchmark [2] 


[1] LLM-SR: Scientific Equation Discovery via Programming with Large Language Models, ICLR 2025

[2] LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models, ICML 2025

### Questions
included in the weaknesses

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a symbolic regression method that augments an LLM with external analytical tools and uses a multi-objective loop to select equations by in-domain accuracy, out-of-domain accuracy, and an AST-length complexity proxy. The system aims to improve generalization and interpretability relative to prior LLM-driven SR methods.

### Strengths
+ Clear articulation of problems with prior LLM-SR pipelines that ignore data structure and optimize a single objective.
+ Sensible overall architecture that separates tool-driven analysis from candidate generation and uses multi-objective selection.
+ Empirical results are promising on the reported tasks, and some ablations are also provided.

### Weaknesses
- Two of the three objectives are accuracy measures that mirror the main evaluation metrics. Optimizing directly for the reported metrics risks can potentially inflate perceived gains (especially for the OOD metric.
- AST length is a coarse and gameable proxy for simplicity. Algebraically equivalent forms can vary widely in AST size, and there is no canonicalization or comparison against alternative complexity notions or human interpretability.
- The three main contributions (tool use, multi-objective selection, meta-strategies) are integration and prompt-engineering choices rather than a new algorithmic idea or learning objective.
- Compute parity is not demonstrated rigorously. There is no matched token budget or GPU-hour accounting to support efficiency claims.
- No SRBench++ results are provided.
- The ID/OOD split remains under-specified. Exact percentiles, whether ID is a Cartesian product of per-dimension intervals, and whether parameters are fit on ID-only or full training data are not clearly stated.
- The pre-sorting filtering rule is described with lower-bound thresholds on NMSE, which is logically inverted. Low NMSE is good. This needs correction and concrete threshold values.
- The algorithm does not clearly specify how the first generation is seeded when the Pareto front is empty.
- The diversity score and parent sampling are under-defined. The asymmetry and potential division-by-zero issues are not addressed, and there is no sensitivity study for temperature or parent set size.
- Seeds, split definitions, fitter hyperparameters, and front size caps or tie-breaking rules are not fully specified.

### Questions
- Why is AST length the right complexity objective here, and how do you prevent gaming via algebraic re-expression? Have you tried canonicalization or MDL-style costs?
- Why do you expect a separate LLM “strategy” module to be necessary and beneficial in SR beyond better prompts for the generator? 
- What are the exact ID/OOD percentiles per dimension, and is the ID region a Cartesian product of per-dimension intervals? 
- How is the first generation seeded when the Pareto front is empty? Please give the concrete initialization policy.
- Can you report a discovery-rate metric for exact or near-exact recovery over a library of known equations, not just per-task anecdotes?

### Soundness
3

### Presentation
3

### Contribution
1
