# Interpretable Probability Estimation with LLMs via Shapley Reconstruction

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Large Language Models (LLMs) demonstrate potential to estimate the probability of uncertain events, by leveraging their extensive knowledge and reasoning capabilities. This ability can be applied to support intelligent decision-making across diverse fields, such as financial forecasting and preventive healthcare. However, directly prompting LLMs for probability estimation faces significant challenges: their outputs are often noisy, and the underlying predicting process is opaque. In this paper, we propose **PRISM: Probability Reconstruction via Shapley Measures**, a framework that brings transparency and precision to LLM-based probability estimation. PRISM decomposes an LLM’s prediction by quantifying the marginal contribution of each input factor using Shapley values. These factor-level contributions are then aggregated to reconstruct a calibrated final estimate. In our experiments, we demonstrate PRISM improves predictive accuracy over direct prompting and other baselines, across multiple domains including finance, healthcare, and agriculture. Beyond performance, PRISM provides a transparent prediction pipeline: our case studies visualize how individual factors shape the final estimate, helping build trust in LLM-based decision support systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents PRISM, a method that uses Shapley-style contrasts computed via LLM prompts to:

- Estimate per-factor contributions to a probability prediction;
- Reconstruct a calibrated probability by summing those contributions (and a base logit). 

The authors also propose a tabular variant (Tabular-PRISM) that batches many contrast pairs into a single prompt, and a more general comparative-prompting procedure for unstructured factors. 

The authors then evaluate PRISM in zero-shot settings on several tabular benchmarks (Adult, Stroke, Heart Disease, Lending, and apple price and football match predictions) using GPT-4.1-mini and Gemini-2.5-Pro. They report improved AUROC / AUPRC versus direct prompting, several LLM prompting baselines, ICL variants and noticeably SOTA approaches such as BIRD. Finally, the authors show example Shapley attributions and analyses of feature interactions and cost/complexity.

### Strengths
- The idea is well-grounded in classical explainable ML methods, and the authors demonstrate that you can port Shapley-esque attribution techniques to the LLM prompting setting.
- The authors have studied more efficient variants of PRISM (i.e. Tabular-PRISM) to reduce the cost of LLM queries. They have also reported detailed runtime numbers.
- The authors have conducted a large set of experiments across prompting approaches and datasets, and compared against SOTA approaches (e.g. BIRD)
- The writing and presentation of the paper are mostly clear.

### Weaknesses
- PRISM requires computing differences $f(x_S \cup \{i\}) - f(x_S)$ where $f$ a scalar (logit), which is mapped from a textual response of the probability via the inverse sigmoid function $\sigma^{-1}$. However, the authors have not shown that these verbalized probabilities are stable subject to e.g. paraphrase and temperatures (or if the final outcome of the model is robust to these changes). The paper acknowledges LLMs are noisy but argues relative differences are reliable, but this claim needs more empirical support.
- The number of queries needed scale linearly with feature set size, and thus it may not scale nicely to high-dimensional predictive problems.
- The paper compares primarily to LLM prompting variants and BIRD. However, a conventional supervised model (e.g., gradient boosted tree) trained on available labeled data is absent. Even in zero-shot paper the relevant comparison is: if labeled data were available, would a standard ML model (with Shapley explanations) be better? This is important to contextualize PRISM’s role (a useful zero-shot tool vs an alternative to traditional models).

### Questions
- In Table 1, the improvements of PRISM are relatively small, especially given that the results are shown on 300-sample subsets. Would it be possible for the authors to show / highlight statical significance for PRISM against the second-best methods for each?
- While it's nice that the authors have reported runtime of PRISM and Tabular-PRISM, would it be possible to report monetary cost of these methods (as well as those of BIRD and ICL) to ensure a more concrete understanding of the cost?
- The authors have used $K=10$ for most of their experiments. Is there a practical reason for the choice? Would difference choices lead to qualitatively different performance?

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
3

### Summary
PRISM reframes LLM probability estimation as compare-and-aggregate: estimate factor-wise Shapley contributions reliably (LLMs are better at relative comparisons than absolute calibration), then reconstruct a calibrated probability. This yields transparent predictions that are competitive or better than direct prompting across several domains.

### Strengths
- PRISM’s main win is that the final probability is explicitly explained as base logit + per-factor Shapley contributions. 
- The method is motivated by the observation that LLMs are more reliable at pairwise comparisons than at absolute probability statements, so PRISM asks the model to do the former and only then reconstructs the latter. That is a sensible, model-aware design choice.
- PRISM is run on both GPT-4.1-mini and Gemini-2.5-Pro with essentially the same recipe, and on tabular tasks from different domains (medicine, finance/credit, income). So the framework is not tied to a single vendor or to a single dataset.

### Weaknesses
- All experiments are binary and zero-shot. Multi-class is only discussed as a possible one-vs-all extension and few-shot is explicitly deferred because demonstrations confound attribution. So we don’t yet know whether PRISM still produces clean factor attributions once the prompt contains examples or multiple labels.
- The main quantitative evidence is four tabular-ish binary tasks plus two small real-world case studies (apple prices, football matches). That’s not enough to claim broad generality, especially for unstructured, long, or noisy inputs.
- On Stroke, Gemini-2.5-Pro with simple 1-shot score prompting slightly outperforms PRISM, which suggests PRISM is not a universal upgrade.

### Questions
- Have you tried intentionally inconsistent factor sets (e.g., “age=29, severe chronic heart failure, elite athlete”) or prompts that reverse the evidence mid-query to see whether the pairwise contrasts stay stable? This would tell us how brittle PRISM is when the LLM’s own explanation policy is stressed.
- You fix K=10 for tabular experiments. How sensitive are AUROC/AUPRC and, more importantly, the stability of factor attributions to K?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper focus on two pain-points of LLM probability estimation: High variance of predicted value and lack of interpretability. The author proposed PRISM that decompose a single overall probability estimate into feature-level marginal contributions, sum them in logit space for additive reconstruction, then map back with the sigmoid; estimate the contributions via Shapley values. This improves robustness and provides per-feature (factor-level) explanations.

### Strengths
- This paper propose a novel idea to transform "probability estimation" to "reconstruction of Shapley marginal contributions in logit space". This creatively grafts the classical additivity of Shapley values onto LLM “verbal” probabilities. It also introduces Tabular-PRISM with batched paired comparisons and reference-sample imputation, together with a reference-specific Shapley definition and a formal proposition guaranteeing additive reconstruction—advances at both the definitional and algorithmic levels
- Presents clear algorithmic implementations, quantifies query and evaluation complexity, contrasts 1-shot vs. n-shot settings, and reports wall-clock costs
- Conducting comprehensive experiments span multiple standard tabular benchmarks and two families of mainstream models, supplemented by calibration curves, conditional-Shapley visualizations, and real-world case studies
- Clear explanation on why addition occurs in logit space, how paired comparisons are constructed, and how background sets are sampled to approximate Shapley values, with illustrative figures

### Weaknesses
- The central premise of the paper is that a probability reconstructed from Shapley values, $P_{PRISM} = \sigma(\phi_0 + \sum \phi_i)$ 1, is more accurate and less "noisy" than an LLM's direct, holistic probability estimate22. This is a significant claim that lacks a strong theoretical foundation

- The Tabular-PRISM variant, which is used for all the main benchmark experiments in Table 1, is critically dependent on the choice of a single "reference instance" $r$6. The paper defines the calculated Shapley values $\phi_i^{(r)}$ as explicitly "reference-specific". But the choice of $r$ appears to be set once "around the population average" and never be analyzed

- The method’s key assumption is that pairwise comparison  $(x_{S\cup{i}} \text{ vs. } x_S)$ is more stable than absolute scoring, with additive reconstruction carried out in logit space; however, the experimental baselines are mainly direct probability prompting (including self-consistency) and BIRD. It lacks comparative baselines that aggregate pairwise judgments using traditional probabilistic models such as the Bradley–Terry–Luce model

- Lack of discussion on the choice of unstructured data from apple price and football matches. Why seven decomposed aspects for apple and five for football? And it would be better if there is more precise description about the factor-extraction process

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
### Summary
This paper proposes PRISM, a procedure for obtaining calibrated, interpretable probabilities from large language models without task‑specific training. The method decomposes a prediction into factor‑level contributions computed by comparative prompting and aggregated as Shapley values on the logit scale. For input factors $x=(x_1,\dots,x_m)$, PRISM estimates each contribution $\phi_i$ by contrasting outputs with and without that factor across sampled contexts, then reconstructs the probability as


$$
\hat p(x) = \sigma\Big(\phi_0 + \sum_{i=1}^m \phi_i\Big),
$$


where $\phi_0$ is a base logit and $\sigma$ is the logistic function. Tabular‑PRISM batches many contexts in a single query and uses a reference instance to impute missing fields, reducing query cost and avoiding ``unknown = risky" biases. Experiments on four tabular datasets (Adult Income, Heart Disease, Stroke, Lending) with two LLMs show PRISM is typically best or near‑best on AUROC/AUPRC, and the factor attributions are faithful to the final probability. Two small text case studies illustrate feasibility beyond tables.

### Strengths
PRISM offers a transparent, end‑to‑end recipe: factor impacts are estimated explicitly, sum to the final logit, and explain the probability. On standard tabular tasks, PRISM is often competitive or superior to strong prompt baselines across two LLMs. Tabular‑PRISM is a practical engineering improvement that reduces query count while preserving interpretability. Visualizations of factor interactions provide diagnostics that typical prompting pipelines lack.

### Weaknesses
Scope is limited to zero‑shot binary classification; multi‑class outcomes are not evaluated, and the unstructured text studies are small. The method’s query cost scales with the number of factors and sampled contexts; even with batching, total evaluations may be non‑trivial for large‑scale deployment. The final probability depends on a chosen base logit $\phi_0$; while ranking metrics are unaffected, calibration and thresholded decisions may be sensitive, and a deeper analysis is warranted. The overall novelty is moderate: PRISM combines known ingredients rather than introducing a fundamentally new estimator. Finally, performance on text depends on factor extraction quality, which can vary.

### Questions
1. Please report calibration metrics (ECE, Brier) and sensitivity to different choices of $\phi_0$. Do simple post‑hoc calibration methods help?
2. Provide budget‑versus‑quality plots varying the number of permutations $K$ and the factor count $m$; this would guide practitioners in selecting $K$ under cost constraints.
3. Can PRISM be extended to multi‑class outcomes (e.g., vector logits with softmax reconstruction or one‑vs‑rest)? Any pitfalls you anticipate?
4. For the unstructured cases, can you quantify robustness to noisy factor extraction or provide an automated factorization pipeline?
5. Will you release prompts/code/reference instances for Tabular‑PRISM to ensure reproducibility?

### Soundness
3

### Presentation
2

### Contribution
2
