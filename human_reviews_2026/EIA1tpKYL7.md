# Doubly-Robust LLM-as-a-Judge: Externally Valid Estimation with Imperfect Personas

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
As Generative AI (GenAI) systems see growing adoption, a key concern involves the external validity of evaluations, or the extent to which they generalize from lab-based to real-world deployment conditions. Threats to the external validity of GenAI evaluations arise when the source sample of human raters and system outputs used to obtain a system quality estimate differs from the target distribution at deployment time. In this work, we propose a doubly-robust estimation framework designed to address this evaluation sampling bias. Key to our approach is the use of synthetic "persona" ratings -- produced by prompting an LLM evaluator (i.e., an LLM-as-a-judge) to behave as a human rater with specific sociodemographic characteristics. Our doubly-robust framework combines these informative yet imperfect persona ratings with human ratings obtained under evaluation sampling bias to produce statistically valid system quality estimates. In particular, we show that our approach yields valid system quality estimates when either: (i) a model trained to predict human ratings using persona ratings and source data observed under sampling bias, or (ii) a reweighting model that corrects for sampling bias is of sufficient quality. We validate our framework theoretically and via a novel Persona Simulation Framework (PSF) designed to systematically manipulate persona quality and the degree of evaluation sampling bias present in source data. Our work provides a principled foundation for combining imperfect persona ratings with human ratings observed under sampling bias to obtain valid system quality estimates.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper solves the problem of external validity in LLM evaluations with limited, biased, or missing human labels. The authors propose a doubly-robust estimator that deals with covariate shift and selective bias. The estimator follows the standard augmented inverse-probability weighted (AIPW) form. The authors further adapt the Riesz loss to estimate the weighting function directly. The experiments validate the approach through three types of datasets: synthetic datasets, semi-synthetic (PRISM) datasets, and real-world (DICES) datasets.

### Strengths
- The paper focuses on the timely topic: external validation using LLM. The effectiveness shown by the experiments implies usefulness of this framework in real application.
- The paper establishes the methodologies with high clarity and is grounded in theory.
- The experiment design is comprehensive. The PSF allows systematic manipulation of covariate shift, selection bias, and persona quality and provides transparent benchmarks for future work.

### Weaknesses
- Though repeated for times, it is still not clear what this doubly-robustness means for the LLM-as-a-judge framework. I would really appreciate it if the authors can clarify this (see details in questions).
- Conceptual novelty is over-claimed. The estimator is essentially the AIPW/doubly-robust estimator widely studied in missing-data and causal-inference literature.

### Questions
- (Related to the first weakness) What does these conditions mean (quoting from abstract): "either (i) a model trained to predict human ratings using persona ratings and source data observed under sampling bias, or (ii) a reweighting model that corrects for sampling bias is of sufficient quality"? Can the authors explain more on this? What could be an example where the conditions are violated?
- Why coverage is a fair metric for comparing the methods in the experiment? How are the confidence intervals constructed by the baselines?

### Soundness
3

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
3

### Summary
This paper studies the external validity of LLM-as-a-judge when there are covariate shift and selection bias in human ratings. It proposes a doubly robust estimator that combines the LLM-data-augmented regression estimator and the reweighting estimator, and proves the asymptotic normality of the estimator. In the numerical experiments, it proposes a persona simulation framework to vary the degrees of covariate shift, selection bias, and persona quality. Experiment results show reduced bias, good confidence interval coverage and confidence interval width.

### Strengths
1. The research problem of external validity in LLM-as-a-judge is practically meaningful.
2. The paper is well written.
3. The proposed doubly robust estimator jointly handles covariate shift and selection bias. The theoretical results are solid.
4. The experiment results are compelling, and demonstrate the advantages of the proposed estimator over baselines.

### Weaknesses
1. While the theory is solid, the core mathematical ideas involved (doubly robust estimation, Riesz loss) are standard in the literature.
2. The validity of the method hinges on the assumption that the covariates $W=(X,V)$ captures all factors that influence the rating $Y$ and the probability of completing a rating $C$. This can be a strong assumption in practice. 
3. In the numerical experiments, the data points are projected to 15 dimensions (line 323). It is questionable if this 15-dimensional vector is sufficiently rich to satisfy the assumption mentioned in the previous point, which leads to a potential mismatch between theory and experiment.

### Questions
1. When the data points are projected to 15 dimensions in the experiment, how are these 15 dimensions selected to retain some predictive signal?

### Soundness
4

### Presentation
4

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
This paper studies how to estimate the external validity of human evaluations when there is a distribution shift between lab raters and real-world raters. The authors propose a doubly robust estimation framework that accounts for both covariate shift and selection bias, aiming to make lab-based evaluation results more representative of deployment settings. Experiments on three datasets show improved estimation quality over baselines.

### Strengths
- The paper rigorously formulates the problem.
- The motivation is strong: external validity is a real issue for GenAI evaluation, especially in high-stakes domains like healthcare.
- The proposed method is well motivated, and the experiments demonstrate performance gains.

### Weaknesses
- The writing can be improved for clarity and flow.
    - Figure 1 is difficult to follow in the introduction and may be better placed in the experiments section.
    - The paper is notation-heavy, which may hinder readability; simplifying or deferring some notation might help.
- All three datasets are somewhat synthetic. While this is understandable, the lack of real-world data weakens the external claims.It would be helpful to include qualitative insights into how varying levels of covariate shift manifest in the datasets.

### Questions
- Q1: Covariate shift is controlled in the embedding space. Could you share qualitative examples or interpretations of how source and target distributions differ? Which factor—content (V) or rater features (X)—contributes more to the observed shift?

- Q2: In your motivating example where students label source data and experts label target data, the raters have very different characteristics. Have you considered manually constructing such a scenario to validate model robustness under extreme rater mismatch?

### Soundness
3

### Presentation
2

### Contribution
3
