# PULSE: Benchmarking Large Language Models for ICU Time Series Classification

- Decision: Reject
- Scores: 2, 2, 6

## Abstract
Large language models (LLMs) are increasingly used on multimodal clinical data, yet their performance on high-stakes intensive care unit (ICU) time series data remains under-characterized. We introduce PULSE, a comprehensive benchmark evaluating 17 models, including conventional learners, deep learning, and instruction-following LLMs, across three datasets (HiRID, MIMIC-IV, eICU) and three clinical endpoints (mortality, sepsis, acute kidney injury). In standard within-domain settings, we find that Gradient Boosted Decision Trees (LightGBM) remain the state-of-the-art, achieving mean AUROCs up to 0.916. Frontier LLMs come close (best mean AUROC of 0.893, OpenAI o3), but show sensitivity to the prompting technique. Crucially, while conventional machine learning and deep learning models suffer performance degradation when tested in unseen domains (e.g., XGBoost AUROC dropping to $\approx$0.511, when trained in MIMIC IV and tested in eICU) due to distribution shift, zero shot and few shot prompting and hybrid reasoning LLM workflow demonstrate robust performance. This establishes LLMs not merely as reasoning engines, but as the pragmatic ``day-zero" solution for institutions lacking the labeled data required to train conventional models. PULSE provides all code, configuration files, and a public results dashboard to enable transparent, reproducible comparison and rapid community extension. We expect PULSE to serve as a common yardstick in the years to come, for developing reliable LLMs for multimodal time series data in critical care.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper provides a framework to evaluate the ability of LLMs to predict clinical outcomes in the ICU based on preprocessed clinical time series data. It leverages an existing multicentre harmonisation pipeline and adds an LLM prompt module. Referencing known issues with LLM calibration, the study proposes novel metrics to evaluate LLM predictions.

### Strengths
- The study builds on top of an established benchmark across multiple hospitals and clinician-designed clinical outcomes.  
- The authors evaluate a range of state-of-the-art proprietary LLMs as well as open source alternatives.

### Weaknesses
- The motivation for this study is unclear. If data has already been extensively preprocessed, is it surprising that a specifically trained XGBoost outperforms the LLM?
- If—as the discussion alludes to—the motivation is instead grounded in the potential off-the-shelf performance of LLMs to be used until enough local data is available to train a dedicated XGBoost model, then the authors should include experiments that thoroughly investigate this aspect. The multicentre nature of the underlying YAIB framework would allow for such experiments. 
- If—as the last section of the results alludes to—the motivation lies in obtaining humand-readable explanations, then the authors should include a thorough assessment of the model explanations that goes beyond a not further described qualitative analysis of the Hybrid Reasoning Agent, most of which appeared to defer to the underlying XGBoost score as an explanation.

### Questions
- The presented Summary Agent and Hybrid Reasoning Agent aren't agents or agentic approaches under standard definitions. They operate on predefined rules and have a hard-coded control flow. They do not interacte with the environment or themselves decide on tool use. For an agent, the LLM would itself decide which data to fetch, if and which conventional ML model to call, etc. 
- I do not understand why the PULSE score and its arbitrary combination of AUC, AUPRC, and MCC was introduced. I also couldn't find a statement on which weights were used in the experiments or why.
- The performance metrics for some models warrant scrutiny. For example in Figure 2, GRU performs particularly bad for sepsis (AUC ~0.33) while its close sibling LSTM is best-in-task. This is unexpected and differs  from the original YAIB results, where LSTM and GRU are relatively closely matched across most tasks. To give another example from Figures 32  and 33, Gemma, Mistral, and to a lesser extent Llama all have poor AUC of ~0.5 for predicting sepsis in eICU but a corresponding AUPRC ~0.5, which is an extremely high AUPRC given the low prevalence of sepsis and AUPRC < 0.1 seen in other models. It is not clear how this can be the case.

### Soundness
2

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
4

### Summary
This paper measures the accuracy of conventional ML methods, deep learning, and LLMs for the task of predicting various labels for patients based on ICU time series.

### Strengths
The logic behind the research is sound, and the design of methods and experiments appears valid. There are no obvious mistakes. The paper is written well and easy to read, with only a very few English errors.

### Weaknesses
The findings are not surprising, nor are the methods. The authors did not take the opportunity to go beyond a correct but standard and relatively superficial understanding of the ICU prediction domain.

Line 418 has a valid and interesting observation: "because one-shot and few-shot LLM approaches require no site-specific training, they are a pragmatic “day-zero” option for hospitals with limited labeled data or MLOps capacity." For ICU prediction to be useful outside research, it must be applicable in hospitals not used for research. LLMs can make predictions for such hospitals. Conventional ML models can do so also, but when they are trained on data from specific hospitals, other hospitals are out-of-domain. The work in this paper would be more interesting with cross-hospital results for the conventional ML methods. If a model is trained on two of the three HiRID, MIMIC-IV, and eICU, how accurate is it on the out-of-domain third? Can the 200 eICU hospitals be separated, to evaluate out-of-domain accuracy on each after training on the 199 others?

The submission praises the previous benchmark YAIB but says it was not designed to accommodate LLMs. This is true, but why not extend YAIB? That would increase comparability with previous results, compared to creating a new benchmark, as done in this submission.

The paper should be more incisive in highlighting the limitations of LLMs for prediction. When given GBDT predictions as input, the LLMs provide plausible-sounding explanations, but their accuracy is worse than the accuracy of the GBDT predictions provided as input! The explanations appear sensible, but the null hypothesis has to be that they have no actual factual value.

Like most AI papers nowadays, this one has very few references that are more than just a few years old. The lessons of older research are neglected, or assumed to be recent. In particular, the finding that conventional ML methods are more accurate than linguistic methods mirrors findings that date back many decades that humans are less accurate than data-driven methods. See among many other papers:

Dawes & Corrigan (1974), “Linear models in decision making,” Psychological Bulletin

WG Baxt. “Prospective validation of artificial neural network trained to identify the presence of acute myocardial infarction.” The Lancet. 1996.

### Questions
Please explain where you disagree with the weaknesses discussed above.

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
3

### Summary
The paper presents PULSE, a benchmark to compare conventional ML, deep learning, and large language models on ICU time‑series for three clinically important tasks: hospital mortality, acute kidney injury, and sepsis. It standardizes datasets, preprocessing, prompting/agent workflows, metrics (including a reliability‑aware PULSE Score) and reports cost/latency trade‑offs alongside accuracy.

### Strengths
- Clear motivation, the work fills the gap between text‑centric LLM benchmarks and tabular ICU benchmarks by a head‑to‑head evaluation on structured time‑series. 
-  A strong side is the pipeline that prevents leakage: harmonization, hourly resampling, explicit missingness indicators, forward‑fill plus train‑set imputation, and chronological splits that can reflect realistic deployment.
- The benchmark covers a fair number of datasets, 3 public ICU cohorts, and a wide range of models (17).
- Hybrid Reasoning Agent grounds the LLM in XGBoost risk and top features which is an interesting approach. 
- Operational realism, the paper reports token usage, latency, throughput, and a cost-performance Pareto analysis that helps choose optimal deployments. 
- Artifacts, configurations, predicted outputs, and an interactive leaderboard are planned for public release.

### Weaknesses
- All most important results use a limited test subset (100 stays x 10 windows per dataset), which reduces statistical power and may change rankings and cost Pareto frontiers on full test sets.
- Figures lack uncertainty quantification, no CI or error bars.
- PULSE score inconsistency: the metric penalizes models that are LLMs differently from no-LLM, the authors have to elaborate on that. 
- Baselines can be under‑tuned, ConvML and ConvDL mostly use defaults without dataset‑specific HPO which may understate baselines or distort comparisons. 
- The codebase was not included in the supplementary materials of the review, therefore I was unable to assess how easy it is to run the benchmark which is an important aspect for the community.

### Questions
- How does the PULSE Score’s CCF asymmetry affect cross-family comparisons and leaderboard fairness? Please consider comparing the 3 policies and reporting their impact or discuss them: (a) discard cases where an LLM’s label contradicts its probability, (b) ignore the label text and evaluate all models solely on probabilities (derive labels server-side at a fixed threshold), (c) add a unified consistency/calibration penalty that applies to all models. 
- What α/β/γ weights were actually used in the PULSE metric and how sensitive is the leaderboard to them? 
- How do LLM results change if you serialize timestamps (possibly truncated or selected) instead of min/mean/max summaries under the same token budget? Now the Hybrid Agent takes XGBoost probabilities computed from raw numeric inputs, while LLMs see only aggregated values. For fairness, could you test serialized numeric tables or raw-value snippets for LLMs so comparisons are apples-to-apples?

Minor:
- Line 144, typo: basline -> baseline
- Line 365 makrer -> marker

### Soundness
3

### Presentation
4

### Contribution
3
