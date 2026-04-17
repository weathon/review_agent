# Synergizing Large Language Models and Task-specific Models for Time Series Anomaly Detection

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
In anomaly detection, methods based on large language models (LLMs) can incorporate expert knowledge by reading professional document, while task-specific small models excel at extracting normal data patterns and detecting value fluctuations from training data of target applications. Inspired by the human nervous system, where the brain stores expert knowledge and the peripheral nervous system and spinal cord handle specific tasks like withdrawal and knee-jerk reflexes, we propose CoLLaTe, a framework designed to facilitate collaboration between LLMs and task-specific models, leveraging the strengths of both models for anomaly detection. In particular, we first formulate the collaboration process and identify two key challenges in the collaboration: (1) the misalignment between the expression domains of the LLMs and task-specific small models, and (2) error accumulation arising from the predictions of both models. To address these challenges, we then introduce two key components in CoLLaTe: a model alignment module and a collaborative loss function. Through theoretical analysis and experimental validation, we demonstrate that these components effectively mitigate the identified challenges and achieve better performance than both LLM-based and task-specific models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes CoLLaTe, a framework that integrates a task-specific time-series anomaly detector (TSADM) with a large language model (LLM) for enhanced anomaly detection. The core idea is to combine the numerical pattern recognition strength of TSADMs with the domain knowledge reasoning capability of LLMs. CoLLaTe introduces two main components:
(i) an alignment module that harmonizes the anomaly-score “expression domains” of the two models by fitting a half-Gaussian density to the LLM’s score distribution and learning a mapping that makes the TSADM scores follow this distribution; and
(ii) a fusion module that combines both scores through a conditional network trained with a collaborative loss, designed to mitigate the “error accumulation” observed with MSE-based objectives.

The paper provides theoretical analysis of the proposed loss and presents empirical results on four datasets, showing strong F1-score improvements and robustness to distribution shifts. Ablation studies further validate the contributions of each module.

### Strengths
S1: The paper’s identification of two key integration challenges, score misalignment and error accumulation are is both insightful and practically relevant for fusing heterogeneous detectors such as LLMs and TSADMs. The qualitative evidence in Fig. 1(b–d) (p. 3) effectively illustrates the semantic gap between their scoring behaviors and motivates the proposed alignment mechanism.

S2: The proposed architecture (Fig. 2, p. 3) is lightweight and modular, using a small conditional MLP that can be flexibly applied to various TSADM–LLM pairs. The reported memory footprint and the manageable training/inference latency demonstrate strong engineering efficiency and potential for real-world integration.

S3: The evaluation covers four datasets and includes module ablations, hyperparameter sensitivity analyses, and distribution-shift tests, consistently showing improved performance. This breadth of experimentation strengthens confidence in the method’s robustness and generality.

S4: The discussion on how MSE objectives can propagate bias when both sources are imperfect—and how the proposed pairwise collaborative loss mitigates such accumulation—is well reasoned. The theoretical properties, such as preserving relative ordering differences (Theorem 2), add useful interpretability to the optimization behavior.

### Weaknesses
W1: Selecting the best GPT-4 run for collaboration (p. 20) introduces potential selection bias, giving the fused model an unfair advantage and likely inflating reported performance. A more rigorous evaluation would use fixed or randomly sampled LLM outputs, or report mean and variance across multiple runs to ensure fairness.

W2: The assumption that the LLM’s score distribution follows a half-Gaussian form, and aligning the TSADM outputs to this distribution appears brittle and under-justified. The paper does not specify the parameterization of the mapping function M(⋅) or enforce monotonicity constraints. As a result, the alignment objective in Eq. (3) risks overfitting to an arbitrary density rather than learning a semantically meaningful correspondence.

W3: The paper omits precise mathematical definitions for 𝐷_intra and 𝐷_inter, referencing them only through qualitative visualizations in Fig. 3. Without explicit formulas or explanations of their metrics, scaling across variates, or windowing behavior, it is difficult to reproduce or interpret the reported sensitivity analyses.

### Questions
Q1: What is the functional class of M(⋅)? Is it constrained to be monotone? How do you avoid degenerate mappings (e.g., many inputs mapped to a single bin)? Please provide the exact architecture (layers/activations) and regularizers.

Q2: How many LLM runs were attempted per subset before picking the “best”? What were the decoding parameters? 

Q3: In Table 3 (unseen distributions), did you tune hyperparameters on held‑out subsets only? Please clarify the selection protocol to ensure no test information was used.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces CoLLaTe, a collaborative framework that integrates Large Language Models (LLMs) with task-specific anomaly detection models (TSADMs) to improve time-series anomaly detection.

### Strengths
1. The paper explicitly identifies two bottlenecks—domain misalignment and error accumulation—and addresses them through clearly designed modules backed by theoretical proofs.
2. CoLLaTe achieves state-of-the-art F1 scores on multiple datasets and shows robustness to distribution shifts, which strengthens the practical relevance of the framework.
3. The theoretical results (Theorem 1, Lemma 1, Theorem 2) are well-connected to experimental observations, lending credibility to the proposed loss design.

### Weaknesses
1. While the idea of combining LLMs and specialized models is timely, the actual implementation—alignment + weighted fusion—is relatively straightforward.
2. LLM inference is computationally heavy; integrating it with TSADM could raise latency issues.
3. The proofs rely on smoothness and distributional assumptions that may not hold in practice (e.g., half-Gaussian score distribution).

### Questions
1. Are the LLM outputs frozen or fine-tuned jointly with the TSADM and conditional network, and why?
2. How sensitive is the alignment module to the choice of the fitted distribution (half-Gaussian)? Would a non-parametric mapping (e.g., histogram matching) perform similarly?
3. Can the aligned scores or the conditional network outputs be interpreted to explain why a specific time point is deemed anomalous?

### Soundness
3

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
The paper proposes **CoLLaTe**, a framework that collaborates a task-specific time-series anomaly detector (TSADM) with an LLM. Two main components are introduced:
1. a **score-alignment module** to map TSADM/LLM anomaly scores into a unified semantic/distributional space;
2. a **collaborative loss** intended to mitigate “error accumulation” when combining the two sources.  
    Experiments on four datasets (two public, two private) show that naïve combination can hurt performance, while alignment + collaborative loss recover and sometimes improve results (small gains on Mustang, larger gains on Mackey/Flight). The motivation is practical and the ablations are informative, but related work coverage, key baselines, theoretical rigor, and cost–benefit analysis are insufficient.

### Strengths
- **Well-motivated engineering problem:** addresses the real friction of score semantic mismatch and error accumulation when mixing TSADM and LLM signals.
- **Clear components & useful ablations:** alignment + collaborative loss are shown necessary (naïve fusion degrades; adding components repairs performance).
- **Some empirical gains:** meaningful improvements on several datasets; the workflow matches practical “alarm triage / false-positive reduction” use cases.
- **Potential for deployment:** conceptually compatible with industrial pipelines.

### Weaknesses
1. **Theory: unclear definitions and shaky derivations**
    

- **Alignment objective.** Main text moves from a binned, non-differentiable objective to a smooth surrogate but skips formalities. Let (f) be the target density (paper uses half-Gaussian). The non-diff objective is essentially  
$$
    \min_{M}; -\sum_{i=1}^N \tfrac{c_i}{\sum_j c_j}\log!\Big(\int_{(i-1)/N}^{i/N} f(S),dS\Big)  
    ;+; \lambda_1!\Big(\tfrac{1}{n}\sum\nolimits_{k} M(s_k)-\hat\mu\Big)^2  
    ;+; \lambda_2(\cdots)^2,  
$$
    where (c_i) counts mapped scores falling in bin (i). The paper then states (for (N!\to!\infty)) the surrogate  
$$
    \min_{M}; -\tfrac{1}{n}\sum\nolimits_{k}\log f(M(s_k)) + \lambda_1(\cdots)^2+\lambda_2(\cdots)^2.  
$$
    However, the derivation omits the constant (\log(1/N)) term and provides no error bound (e.g., Riemann-sum or dominated-convergence justification) for replacing the binned cross-entropy with the continuous NLL. Please provide a formal proposition with assumptions and a uniform convergence (or at least consistency) statement.
    
- **Choice of (f).** Using a half-Gaussian for scores (S $\ge$ 0) is plausible but not justified. Given scores lie on a bounded interval in practice, Beta / logit-normal or isotonic calibration might be more appropriate. Please provide goodness-of-fit tests (e.g., KS, AD) and a robustness study across families; otherwise alignment risk is model-misspecified.
    
- **Collaborative loss / “error accumulation” theorem.** The paper states the MSE-based combination yields an optimum $\hat S^\star = y + \lambda_1\varepsilon_s + \lambda_2\varepsilon_S$ and then lower-bounds $\mathbb{E}[(\hat S^\star - y)^2]$ by $(\lambda_1\mu_s+\lambda_2\mu_S)^2$ (Jensen). This ignores variance and covariance:  
    $$
    \mathbb{E}[(\hat S^\star-y)^2]  
    = \lambda_1^2(\sigma_s^2+\mu_s^2)+\lambda_2^2(\sigma_S^2+\mu_S^2)+2\lambda_1\lambda_2\operatorname{Cov}(\varepsilon_s,\varepsilon_S),  
    $$
    which reduces to $(\lambda_1\mu_s+\lambda_2\mu_S)^2$ only under **zero variance** (or degenerate) conditions. Moreover the stationarity condition mixes $\hat S$ as a deterministic function of parameters with random errors $\varepsilon$. Please **restate the probability space**, clarify whether $\hat S$ depends on sample noise at optimum, and **include variance/covariance** in the bound (or give conditions—unbiasedness, independence, bounded variance—under which your inequality holds).
    
- **Undefined key quantities.** (D_{\text{intra}}) / (D_{\text{inter}}) are described verbally (“within-patch / across-patch distances”) but **lack formulas**. For example, if (x_t) are embeddings and (p(t)) is the patch index,  
    $$
    D_{\text{intra}}(t)=\tfrac{1}{|P(t)|-1}!\sum_{j\in P(t)\setminus{t}}! d(x_t,x_j),  
    \quad  
    D_{\text{inter}}(t)=\tfrac{1}{K-1}!\sum_{k\ne p(t)}! d(\mu_{p(t)},\mu_k),  
    $$
    with (d) (Euclidean/DTW/cosine) and (\mu_\cdot) specified. Please provide exact definitions, normalization, and metric choice in the main text.
    
- **Convergence claim.** The $(O(T^{-1/4}))$ rate borrowed from prior work requires Lipschitz smoothness, bounded stochastic gradients/noise, step-size schedule, etc. The paper should verify each assumption for the specific loss (including the alignment regularizers) and report constants (at least qualitatively).

2. **Positioning & baselines**    
- Missing **closest paradigms**: (i) _LLM→TSAD distillation_ (training-time guidance; different collaboration locus), (ii) _TSAD→LLM two-stage refinement_ (post-hoc vetting). Without these, contribution boundaries blur.
- Missing **simple fusions**: score avg / max / vote and a learned meta-fuser (e.g., logistic regression/XGBoost on ($S_{\text{TSADM}},S_{\text{LLM}}$) with calibration) are standard sanity checks to justify your more complex design.

2. **Evaluation & practicality**
- **Cost–benefit** is unclear: when strong TSADM already yields high F1, LLM adds only +2–3 points in places, yet latency/API cost appears substantial. Provide throughput/latency/cost vs. gain curves and scenarios where LLM is decisively worth it.
- Some narrative claims (e.g., “LLM excels at point anomalies”) are not consistently supported; include a breakdown by anomaly type/length.

### Questions
1. **Add closest & simple baselines:** Include _LLM→TSAD distillation_, _TSAD→LLM two-stage refinement_, and simple score fusion (avg / max / vote). Report gaps relative to CoLLaTe.
2. **Theory fixes:**
    - Give formal definitions and computation for $(D_{\text{intra}})$ and $(D_{\text{inter}})$.
    - Re-state/prove the error-accumulation result with a consistent probability space and explicit variance terms.
    - Justify the alignment distribution with goodness-of-fit tests and robustness analysis.
3. **Where does LLM truly help?** Provide case studies and breakdowns (point vs. contextual, short vs. long anomalies) showing indispensable improvements.
4. **Explain weak TSFM baselines:** Document implementations, tuning budgets, and why they underperform here.

### Soundness
2

### Presentation
2

### Contribution
2
