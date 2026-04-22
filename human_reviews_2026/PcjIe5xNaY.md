# Reasoning on Time-Series for Financial Technical Analysis

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
While Large Language Models have been used to produce interpretable stock forecasts, they mainly focus on analyzing textual reports but not historical price data, also known as Technical Analysis. This task is challenging as it switches between domains: the stock price inputs and outputs lie in the time-series domain, while the reasoning step should be in natural language. In this work, we introduce Verbal Technical Analysis (VTA), a novel framework that combine verbal and latent reasoning to produce stock time-series forecasts that are both accurate and interpretable. To reason over time-series, we convert stock price data into textual annotations and optimize the reasoning trace using an inverse Mean Squared Error (MSE) reward objective. To produce time-series outputs from textual reasoning, we condition the outputs of a time-series backbone model on the reasoning-based attributes. Experiments on stock datasets across U.S., Chinese, and European markets show that VTA achieves state-of-the-art forecasting accuracy, while the reasoning traces also perform well on evaluation metrics judged by industry experts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents verbal technical analysis (VTA) that integrates verbal reasoning through LLMs with latent time series forecasting to achieve interpretable financial prediction. The authors argue that existing LLM-based approaches primarily analyze textual financial information (e.g., reports or news), and VTA is distinct from them since it focuses on time series reasoning, transforming numerical data into natural language descriptions. Through extensive experiments across multiple stock datasets, VTA demonstrates state-of-the-art forecasting accuracy. In addition, VTA achieves superior expert evaluation scores compared to GPT and DeepSeek.

### Strengths
- Each component of the framework (time series reasoning, forecasting, etc) is well-motivated and properly designed. 
- Experiments are comprehensive, covering a wide range of datasets, baselines, and metrics.
- The expert-based reasoning evaluation is practically valuable. 
- The authors provided their code anonymously for reproduciblity.

### Weaknesses
- Some parts of the paper are unclear:
  - The discussion of Figure 1 should be expanded. Currently, the y-axis and evaluation criteria are ambiguous. It would be helpful to explicitly describe what each score measures and how they were computed.
  - The notation *s* in Eq. (10) and Figure 3 should be aligned for consistency.
  - Some terms in Table 5 (e.g., volatility, drawdown) should be clearly defined in the main text for non-finance readers.
- The scope of the paper needs justification:
  - The paper focuses only on financial datasets. Please elaborate on why the financial domain is particularly challenging (if so) for time series reasoning. 
  - Please discuss whether VTA could generalize to other domains (e.g., healthcare, energy).
- There already exists a recent work that explores LLM-based time series explanation. The paper should clearly position VTA with respect to this approach by discussing or empirically comparing it.
  - TimeCAP: Learning to Contextualize, Augment, and Predict Time Series Events with Large Language Model Agents (AAAI 2025) 


In summary, while the idea is interesting and timely, the paper needs a clearer positioning with the existing work. ALso, several aspects of the presentation could be improved.

### Questions
- How does VTA perform when applied to non-financial time series data?
- Does VTA show consistent reasoning patterns across markets (U.S. vs. China vs. Europe)?

In addition, please refer to the weaknesses above.

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
This paper proposes Verbal Technical Analysis (VTA) — a hybrid reasoning framework that aims to make time-series stock forecasting interpretable by combining:
- A verbal reasoning LLM fine-tuned via a proposed Time-GRPO objective (using inverse-MSE reward);
- A latent time-series backbone model conditioned on the reasoning outputs through joint conditional training.

Experiments across U.S., Chinese, and European stock datasets suggest VTA achieves better MSE/MAE than prior time-series LLMs (e.g., Time-LLM, CALF) and generates reasoning traces preferred by human financial experts. The paper also evaluates downstream portfolio performance using Sharpe ratio and CAPM/Fama-French models.

### Strengths
pros:
1. Interesting conceptual direction — bridging LLM reasoning and numerical forecasting is a timely and under-explored research frontier, particularly for financial data.

2. Clear problem motivation — the authors highlight the lack of interpretable time-series reasoning methods and position VTA as a step toward explainable financial LLMs.

3. Engineering completeness — the framework, ablations, and appendices (notably the extensive tables on pages 6–9 and expert evaluation in Appendix C) are well-executed and professional.

4. Human evaluation effort — 25 finance professionals were surveyed; results show statistically significant preference for VTA reasoning over GPT-4.1 and DeepSeek-R1 (Table 7).

5. Empirical results — consistent improvement across four datasets; the method performs competitively even under portfolio-level evaluation (Table 5).

### Weaknesses
1. The “Time-GRPO” formulation largely reuses GRPO (Shao et al., 2024) with an inverse-MSE reward; conceptually it is a direct extension rather than a fundamentally new RL objective. The “joint conditional training” (Figure 3) borrows heavily from classifier-free guidance (Ho & Salimans 2022) and conditional diffusion principles. The novelty lies more in application assembly than in algorithmic innovation.
2. The paper never establishes whether the verbal reasoning causally improves forecasting accuracy, or merely adds correlated signals. The conditioning between textual reasoning and time-series output feels heuristic; no ablation demonstrates whether removing reasoning entirely (vs. random text conditioning) degrades performance beyond noise.
3. The “expert evaluation” metrics (Clarity, Depth, etc.) are subjective and lack reproducibility; no public dataset of reasoning traces is released. The forecasting datasets are relatively small (StockNet, A50, EUROSTOXX 50). Improvements of ≈2–3% MSE over CALF (Table 2) may not be statistically meaningful beyond the reported seeds.
4. Does not compare to simpler text-free multimodal adapters (e.g., cross-attention-based transformers with interpretable heads). The authors claim SOTA forecasting “and interpretability,” but do not quantify interpretability beyond expert survey averages.

### Questions
1. How sensitive is VTA to the choice of base LLM (e.g., Qwen vs Llama)?

2. Can the verbal reasoning traces be used independently for explainable trading, or are they only auxiliary features for the time-series decoder?

3. Would random or adversarial reasoning degrade forecasting performance — i.e., is the reasoning genuinely informative?

4. Is the “inverse-MSE reward” stable during optimization? Any reward hacking observed?

### Soundness
2

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
3

### Summary
The authors propose a method for training a model that can reason over financial time series. The model works by jointly learning over textual and numerical representations of time series data. The authors conduct an evaluation by comparing the method to several other LM-based time series forecasting methods and conduct a human evaluation to measure the plausibility of the model's generated traces.

### Strengths
- Well-written, easy to understand
- This paper contrasts with others in the area by combining neural representations of LMs with text embeddings. Most other papers focus only on time series expressed as text or on neural embeddings, not both.
- Strong choice of baselines and third-party datasets
- The use of experts for annotation is useful and lends confidence towards the model's performance

### Weaknesses
On the whole I think this is a very strong paper. However, without details on how baselines were tuned it's difficult for me to have confidence in the method's performance. If the authors can provide this information then I'll happily increase my score to an accept.   

To be clear, my primary concern is that the proposed method could be outperformed by (1) simple statistical methods or (2) strongly hyperparameter-tuned baselines.

### Questions
- Why do we need dedicated financial time series reasoning? Why aren't the other times series LLMs useful for this job? 
- How does performance compare to simpler models? "Are Language Models Actually Useful for Time Series Forecasting?" Tan et al. 2024 shows that many LLM-based methods for time series forecasting are outperformed by simple baselines (like mean prediction). 
- Could the authors provide more information on how they tuned their baselines? Im concerned that the baseline models could outperform the proposed method given sufficient tuning.

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
4

### Summary
This paper proposes a novel framework called Verbal Technical Analysis (VTA), which combines verbal (LLM-based) and latent (time-series model-based) reasoning for interpretable financial time-series forecasting. The framework introduces a reinforcement learning training pipeline (Time-GRPO) to optimize reasoning quality, and conditioning mechanisms to guide forecasts based on reasoning outputs. VTA is evaluated on multiple datasets across U.S., Chinese, and European markets, achieving state-of-the-art results in both forecasting accuracy and interpretability.

### Strengths
The paper proposes a novel framework, Verbal Technical Analysis (VTA), which effectively integrates large language models with time-series forecasting to produce both accurate and interpretable stock predictions.

It introduces an innovative reinforcement learning objective (Time-GRPO) using inverse MSE as a reward to improve the quality of verbal reasoning aligned with forecast accuracy.

The model generates natural language reasoning traces that are rated highly by financial domain experts, adding a valuable layer of explainability for real-world financial decision-making.

The evaluation is rigorous, including ablation studies, expert human assessment, and practical validation through portfolio optimization, demonstrating both technical soundness and practical utility.

### Weaknesses
The proposed training pipeline is multi-staged and complex, involving cold-start RL, supervised fine-tuning, and joint conditional training, which may limit reproducibility and increase implementation overhead.

The paper does not compare VTA’s reasoning or performance against human-crafted technical analysis rules or human analysts, which would be a natural baseline for such a task.

The paper does not compare VTA with financial time seires LLMs, e.g., Kronous, which are related to financial analysis.

### Questions
How well does VTA scale to longer prediction horizons or more volatile, high-frequency financial time-series?

Does the verbal reasoning always align with the actual forecast, or are there cases where they contradict, and how is that handled?

How does VTA perform under market regime shifts or structural breaks, such as during crises or unexpected economic events?

### Soundness
3

### Presentation
3

### Contribution
2
