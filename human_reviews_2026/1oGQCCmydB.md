# Conversational Time Series Foundation Models: Towards Explainable and Effective Forecasting

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
The proliferation of time series foundation models has created a landscape where no single method achieves consistent superiority, framing the central challenge not as finding the best model, but as orchestrating an optimal ensemble with interpliability.  While Large Language Models (LLMs) offer powerful reasoning capabilities, their direct application to time series forecasting has proven ineffective. We address this gap by repositioning the LLM as an intelligent judge that evaluates, explains, and strategically coordinates an ensemble of foundation models. To overcome the LLM's inherent lack of domain-specific knowledge on time series, we introduce an R1-style finetuning process, guided by SHAP-based faithfulness scores, which teaches the model to interpret ensemble weights as meaningful causal statements about temporal dynamics. The trained agent then engages in iterative, multi-turn conversations to perform forward-looking assessments, provide causally-grounded explanations for its weighting decisions, and adaptively refine the optimization strategy. 
Validated on the GIFT-Eval benchmark on 23 datasets across 97 settings, our approach significantly outperforms leading time series foundation models on both CRPS and MASE metrics, establishing new state-of-the-art results.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores a promising direction by incorporating LLM-based reasoning into time-series foundation model ensembles, offering a potentially impactful perspective for the field. The idea is compelling, though several aspects of the proposed “reasoning” mechanism require further clarification and refinement.

### Strengths
- This paper has a clear motivation. No single TS foundation model dominates; an ensemble is necessary.
- Interesting idea of LLM-guided ensemble refinement and SHAP-grounded explanation.
- Solid benchmark evaluation and competitive results.
- Attempts to bridge quantitative optimization with reasoning-style explainability.

### Weaknesses
- The ''reasoning'' mainly resembles metric-based weight selection, not fully realized temporal reasoning.
- Limited practical scope: focuses on model choice rather than broader forecasting problem understanding.
- The link between explanations and numerical optimization appears post-hoc.

### Questions
- Q1:  What reasoning capability is actually learned beyond metric-based weight tuning? Ablations that remove or randomize the agent would help clarify its actual contribution. For example, ablations that isolate the effect of the LLM agent from SLSQP.

- Q2: Disconnection between ''semantic reasoning'' and numerical training dynamics. The agent claims to adjust weights using causal insights. However, weights remain trained purely via losses, not via reasoning signals. Is there any feedback loop from reasoning to model update? Currently, the reasoning seems post-hoc rather than interventional.

- Q3: I am still concerned about its limited real-world applicability. Actually, forecasting is not limited to model selection alone. In practice, forecasting involves: problem contextualization, data understanding, domain-driven scenario reasoning, and structural model engineering, but the agent framework focuses narrowly on selecting among pretrained models. The paper claims to be an explainable and effective forecasting. I wonder whether the agent can be involved in reasoning about data quality or pattern regimes before optimization. Can it propose feature engineering strategies? Can it detect when ensemble models are all mismatched?

- Q4: How is the confidence score calibrated, and how should it be interpreted operationally? Additionally, have the authors evaluated the robustness of the agent under distribution shift or non-stationary settings?

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
This paper proposes TSOrchestra, a framework that employs an LLM as a reasoning agent to orchestrate ensembles of time series foundation models. Instead of directly predicting numerical values, the LLM plays a role as an intelligent judge that (1) evaluates candidate models' outputs, (2) optimizes ensemble weights through iterative reasoning, and (3) explains the causal rationale behind its decisions. Extensive experiments on various datasets and settings demonstrate that TSOrchestra achieves state-of-the-art performance and strong interpretability.

### Strengths
- The paper introduces a novel framework that leverages LLMs as orchestrators that coordinate multiple time series foundation models through reasoning-based ensemble optimization. 
- The framework is theoretically supported, showing that ensembles outperform a single model. 
- Experiments are comprehensive, covering diverse datasets and domains. Ablation studies and sensitivity analyses are thoroughly conducted. Results are strong across various metrics.

### Weaknesses
- The scalability and latency of multi-turn reasoning and optimization are not analyzed. The computational cost of repeated optimization, SHAP evaluation, etc. may be high. It's unclear whether this framework is practically useful. 
- The interpretability claims could be further validated throgh user studies or quantitative metrics. 
- The framework's robustness to noise is not evaluated. It would be helpful to test on noisy datasets such as stock prices or socia media traffic, where temporal signals are weak. 
- While the reasoning framework is novel, it would be interesting to see some failure or error cases during reasoning. It would be valuable to include examples where the reasoning agents makes incorrect or inconsistent adjustments and analyze why such cases occur. I don't expect them to be included in the main context though.

### Questions
Please see the weaknesses above.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new framework, TSOrchestra, which leverages a large language model (LLM) as a reasoning agent to coordinate ensembles of time-series foundation models (e.g., Moirai, Sundial, Toto). The central argument is that while LLMs perform poorly when directly used for numerical forecasting, they can analyze the current problem context and dynamically combine specialized forecasting models to achieve improved performance. The authors further claim that the ensemble weighting process can be both interpretable and causally grounded. The paper also includes a theoretical proof of the superiority of ensemble methods via a proposed Temporal Incompatibility Index.

### Strengths
- The idea of positioning an LLM as a meta-optimizer or reasoning controller for existing time-series foundation models is interesting and timely.

- The work clearly identifies the limitations of direct LLM forecasting and attempts to use reasoning for ensemble coordination instead of numerical prediction.

- The introduction of the Temporal Incompatibility Index is conceptually appealing, and could inspire further study on heterogeneity in time-series regimes.

- The paper reports improved empirical results on the GIFT-Eval benchmark, showing the potential of the proposed approach.

### Weaknesses
- The training process for the LLM agent is underexplained. It remains unclear how the SFT training data are constructed: who wrote or generated the reasoning traces, how ground-truth ensemble weights were determined, and what constitutes a “correct” reasoning trajectory. Without this, reproducibility and credibility of the training pipeline are limited.

- The paper does not convincingly justify why LLM-based reasoning is necessary when all metrics (MAE, MSE, etc.) are already computable and could be directly optimized via standard numerical ensemble methods. The claimed “forward-looking” reasoning remains theoretical; the LLM has no access to future data and ultimately still optimizes empirical risk on past data.

- The interpretability claim is weak. The generated textual explanations may describe or rationalize weight differences post hoc, but the paper provides no evidence that these explanations are faithful or trustworthy beyond synthetic SHAP correlations. The argument that “language-model explanations make the process auditable” is not substantiated.

- The writing is overly complicated and poorly organized. Many equations (e.g., SFT loss, GRPO, standard RL formulations) are well-known and unnecessary in full detail, taking space away from the truly novel aspects (how faithfulness scores are computed, how the reasoning is encoded in prompts, etc.).

- The Temporal Incompatibility Index, though potentially interesting, is poorly integrated into the main text—it appears mainly in the appendix and is not clearly used in the actual training or inference pipeline.

- Empirically, the improvement over simple static SLSQP ensembles is moderate, and the paper lacks ablations showing how much each component (LLM reasoning, SHAP faithfulness, multi-round reflection) contributes.

### Questions
1. Training data construction: How exactly is the SFT training dataset generated? Are the reasoning traces human-written, rule-based, or extracted from ensemble optimization logs? How is the “ground-truth” ensemble weight defined for supervised training?

2. Role of LLM reasoning: If MAE/MSE/SMAPE are already computed during optimization, why can’t we simply use these metrics to derive an optimal ensemble directly? What additional information or reasoning capability does the LLM bring beyond what numerical optimization (e.g., SLSQP) already provides?

3. Distinction from standard ensemble optimization: How does the proposed reasoning process differ from conventional multi-objective or adaptive ensemble methods that also update weights based on past performance?

4. Interpretability evaluation: How is interpretability or “faithfulness” quantitatively measured? The SHAP-based alignment is mentioned, but are there any human evaluations or independent checks to confirm that LLM-generated explanations are trustworthy?

5. Forward-looking reasoning: The paper argues that the LLM provides forward-looking adaptation to “dynamic incompatibility,” yet both SLSQP and the LLM operate only on existing data. How is this “forward reasoning” realized in practice without access to future observations?

6. Stability across runs: Have the authors tested the stability of the proposed reasoning process? Since LLM reasoning involves sampling and multi-round interactions, do repeated runs under the same setup yield consistent ensemble weights and similar performance? If not, how large is the observed variance?

7. Effect of iterative optimization: The examples show multi-round reasoning, but the paper does not present a quantitative analysis of performance vs. iteration number. Is the improvement monotonic? How is the stopping criterion decided, and does it generalize across datasets?

8. Choice of base models: Why were Moirai-2, Sundial, and Toto selected as the ensemble members? Are they complementary in their forecasting behavior (e.g., trend vs. seasonality vs. local fluctuations)? Would other strong foundation models such as TimesFM, Chronos, or Timer yield similar results?

9. Interpretability of base models: The argument for LLM-based interpretability assumes that each base model has distinct, human-understandable biases, yet the paper does not describe these differences. How do the outputs of these three models differ, and how does the LLM reasoning capture such distinctions?

10. Temporal Incompatibility Index (TI): The paper introduces the TI Index theoretically but never reports its empirical values. How is TI computed for the datasets or models used, and how does it influence the ensemble weighting decisions? Without actual TI results, it is hard to assess its practical meaning.

### Soundness
2

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
3

### Summary
For the problem of time series forecasting, the authors propose to use LLM Judge to evaluate, explain and coordinate an ensemble of foundational time series models. They use R1-style finetuning process, guided by SHAP-based faithfulness scores, which teaches the model to interpret ensemble weights as meaningful causal statements about temporal dynamics.

### Strengths
Agentic solutions for time series forecasting is a novel and undiscovered area and is currently lacking in current benchmarks which are heavily populated by foundation or deep learning models.

Building the agentic workflow on top of the ensemble backbone is intuitive as it both lies on a strong foundation yet gives enough space to the agent to make decisions through adjusting weights.

The paper thoroughly explains the building structure of the agent strenghtened with equations and visuals where necessary. The methodology is clearly understood by only reading the main content, except for the data generation part that was used for SFT and RL. It would good to have a few sampes showing the conversations the model is trained for.

The main experiments show the ensemble can achieve good results compared to other agentic and foundation models.

### Weaknesses
The experiments miss a critical ablation isolating the key contribution—the LLM’s control over ensemble weights. The current ablations focus on design choices for the agent itself, but not on how much value the LLM brings compared to a simple ensemble where weights are optimized without LLM intervention. Without this, it’s hard to assess the true benefit of the proposed architecture.

Moreover since the approach fundamentally builds on ensembling, it should be benchmarked against a broader suite of ensemble methods. This would provide a fairer and more comprehensive evaluation. Ensembling is already a well-studied method in time series forecasting.

The paper claims that the agent dynamically adjusts to time-varying model performance, but the described mechanism does not convincingly support this. The agent’s control seems limited to selecting which metric to optimize next through SLSQP, without direct weight manipulation. The demos in the appendix support this limitation, suggesting the agent lacks true flexibility or depth in its control strategy.

While I find the aim of this paper interesting and highly valuable I believe it is not mature enough for publication yet. Mainly I believe the agentic system lacks depth and does not have much flexibility on weight control and moreover the conducted experiments fail to show how the proposed agent framework is useful compared to a few baseline ensembles which is already a highly well studied area for time series forecasting. Without these experiments I am concerned most of the benefit comes from the ensembling and not the agent’s adjustments.

### Questions
See the weakness section

### Soundness
2

### Presentation
2

### Contribution
1
