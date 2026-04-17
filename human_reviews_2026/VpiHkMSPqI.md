# LLM-as-a-Prophet: Understanding Predictive Intelligence with Prophet Arena

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
With the rapid progress of large language models (LLMs) trained on every available piece of data, it becomes increasingly challenging to reliably evaluate their intelligence due to potential data contamination and benchmark overfitting. To overcome these challenges, we investigate a new angle of benchmarking LLMs' intelligence by evaluating their capability in forecasting real-world  future events, a paradigm we call "LLM-as-a-Prophet". Such forecasting tasks require combination of sophisticated capabilities while remaining free from data contamination or overfitting. To systematically evaluate such predictive intelligence of LLMs, we introduce $\texttt{Prophet Arena}$, a general evaluation benchmark that continuously collects live forecasting tasks and decomposes each task into distinct pipeline stages, supporting our controlled and large-scale experimentation. Our comprehensive evaluation reveals that many LLMs already exhibit impressive forecasting capabilities, reflected in, e.g., their small calibration errors, consistent prediction confidence and promising market returns. However, we also uncover key bottlenecks even in frontier models, such as inaccurate event recalls, misunderstanding of data sources and slower information aggregation compared to markets when resolution nears.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper aims to explore whether large language models (LLMs) possess genuine predictive intelligence. The authors argue that forecasting can serve as a rigorous and unified test of intelligence, as it integrates reasoning, calibration, and evidence aggregation.

### Strengths
1) The paper is overall well written and easy to follow. 

2) The study is methodologically sound and supported by a comprehensive empirical evaluation. However, the benchmark’s reproducibility and transparency are limited, as key implementation details (event selection, context construction) are not fully released.

3) The work offers a valuable new perspective on evaluating LLMs’ predictive and reasoning abilities. While the conceptual framing is interesting, the paper’s technical contribution remains limited. More importantly, a benchmark’s ultimate value lies in its ability to guide future model development by revealing actionable weaknesses or providing diagnostic insights. Prophet Arena, however, stops at evaluation and does not clearly demonstrate how its results can inform the improvement of model architectures, reasoning mechanisms, or calibration strategies. The work could be strengthened by explicitly linking benchmark findings to concrete modeling directions or by releasing more reproducible implementation details.

### Weaknesses
1) Although the paper claims to release part of the dataset, the full event selection, context construction, and preprocessing pipeline are not disclosed. No implementation details, scripts, or reproducible framework are provided for replicating Prophet Arena. As a result, the benchmark cannot be independently reproduced or audited, which weakens its scientific transparency.

2) Prophet Arena primarily repackages existing components—prediction market data, web retrieval, and standard evaluation metrics (Brier score, ECE, return). The framework does not introduce new modeling techniques or algorithmic insights, serving mainly as an evaluation setting. Comparisons are limited to a single market-based baseline, lacking systematic benchmarks against structured forecasting models or classical probabilistic methods.

3) The paper proposes using “real-world forecasting” as a proxy for intelligence, but does not justify why this task is representative, stable, or objectively measurable. It remains ambiguous whether Prophet Arena is intended as a benchmark, a task suite, or a conceptual lens for studying intelligence.

4) (Important) The paper does not explain which cognitive abilities are actually being measured (e.g., reasoning, retrieval, knowledge recall, uncertainty estimation). Without disentangling these capabilities, the source of success or failure in forecasting is unclear. This makes the benchmark weak in diagnostic interpretability and limits its use for targeted model improvement.

5)  Experimental results show that LLMs do not outperform the market baseline in prediction accuracy or expected return. The models’ predictions are often overly conservative or poorly calibrated, as acknowledged by the authors. Hence, the study fails to demonstrate real-world utility or unique advantages of LLMs in open-domain forecasting.

6) (Minor Points) Figure 10 is blurry, and Figure 11 (right) has a formatting issue — the “Category” label is not fully visible.

### Questions
1) Do the authors view Prophet Arena primarily as a diagnostic tool for understanding LLM reasoning, or as a performance benchmark for future model comparison?

2) How can the results from Prophet Arena inform or guide the next stage of model development? For example, can specific reasoning or calibration failures observed in forecasting tasks translate into actionable model improvements?

3) Since the event selection, context construction, and preprocessing pipelines are not fully released, how do the authors ensure reproducibility and fairness across different models and time periods?

4) Other questions are discussed in the Weakness section.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a live benchmark for evaluating LLMs’ ability to forecast future real-world events over time, addressing a core question with direct relevance to industrial decision-making. The results demonstrate both promising predictive intelligence and persistent bottlenecks in reasoning, evidence integration, and temporal information assimilation.

### Strengths
- This paper proposes a live and realistic benchmark that directly measures LLMs’ forecasting ability on future real-world events, highly relevant to industrial applications.

- Strong empirical design including probabilistic scoring, calibration, and market-based evaluation, providing a multi-angle understanding of model forecasting capability.

- Clear analysis connecting reasoning quality and belief update dynamics.

- Well-articulated motivation and clean methodology presentation.

### Weaknesses
- No measurement of alignment between human judges and model judges, raising concerns about rating validity.
- Reliance on a single web-search agent without explicit evidence quality assessment may confound reasoning evaluation.

### Questions
1. Can the authors report performance differences across prediction horizons (short vs mid-term)? Since most events resolve within days or hours, current results may reflect late-stage market imitation rather than true foresight. Distinguishing performance across different lead times would reveal whether models are actually anticipating future outcomes, or simply reacting better as more information becomes available near resolution. 

2. Retrieval module lacks noise-robustness. News access relies solely on a search agent (GPT-4o) with minimal noise filtering or source-quality controls. How to improve evidence quality?

3. Since LLMs function as judges without calibration against human scoring, there is a risk of alignment bias or shared model failure modes. Do the authors have plans for human adjudication or inter-rater agreement studies to validate reasoning evaluation?

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
3

### Summary
1. The manuscript perfectly proposes = LLM-as-a-Prophet, testing LLMs’ ability. This actually helps forecast future real world events
2. Introduces Prophet Arena, this is a live benchmark which is using prediction-market data and real-time web context.
3. The manuscript also evaluates array of 22 LLMs. This represents diverse number of large language of models.
4. The manuscript Measures forecasting loss (Brier), calibration error (ECE), and market return.
5. GPT-5R shows best calibration and lowest loss; all models still lag human markets.

### Strengths
1. The manuscript presents novel paradigm: which is real-world, contamination-free forecasting benchmark.
2. The manuscript also presents comprehensive metrics: which is a good blend of accuracy and calibration.
3. Good part is about the open-sourced subset for reproducibility.

### Weaknesses
1. The Event recall errors and approximate temporal memory which could be a potential weakness.
2. The weakness Conservative probability estimates vs markets.
3. Dependence on search/source quality; not all domains benefit equally.
4. Limited profitability (returns < 1).
5. Incomplete foresight near event resolution; calibration still imperfect.

### Questions
1. Can you tell me how robust are results to prompt or search-engine variations along with some metrics to compare and evaluate?
2. Could multi-agent or ensemble LLMs outperform single models and how this multi agentic workflow will be working together?
3. How to improve knowledge precision and can you tell me what can be done to improve it?
4. Can MCPs be used to do the same evaluation.

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
The paper proposes evaluating LLM capabilities by forecasting live Kalshi questions. They setup an automated pipeline to extract, test, and resolve LLM forecasts. The evaluation results are summarized using both standard forecasting and calibration metrics like brier score and ECE, as well as (hypothetical) market returns and relative advantage between different LLMs. The paper has interesting analysis of LLM behaviours, ranging from: analysis across topics, effect of market data vs retrieved GPT 4o search results, different tendencies to assign extreme probabilities to forecasts, and analysis of reasoning traces and failure modes.

### Strengths
1. The paper introduces a useful live benchmark for measuring LLM capabilities via forecasting open Kalshi events.

2. I like the relative advantage based metrics used for comparing language models. This mitigates issues in existing benchmarks where questions can vary in difficulty (sometimes not even being "forecasting" questions as future information is leaked).

3. The paper has detailed analysis of LLM forecasting behaviour and interesting insights across diverse ablations in Section 4 and the appendix.

### Weaknesses
1. The paper does not show awareness of existing literature in LLM forecasting. For example, probabilistic forecasting is mentioned as the first "distinguishing feature" of the benchmark in the introduction. However, this has been the standard used in existing papers in LLM forecasting [1]. I am also concerned about the supposed "introduction" of "LLM-as-a-prophet paradigm". LLM forecasting has been an active area of study for the last 3 years [2]. I do not see the value of adding a new term, especially given the connotations of the word "prophet". 

2. The use of GPT-4o based search as context across models could be a confounder. It is perhaps possible that the OpenAI models perform better on this benchmark because the retrieved context is more "in distribution" for them (as it comes from GPT 4o search + outputs) than other model families. Further, GPT 4o search is more likely to surface external context that bridges knowledge gaps in OpenAI models.

[1] Approaching Human-Level Forecasting with Language Models. Danny Halawi, Fred Zhang, Chen Yueh-Han, Jacob Steinhardt

[2] Forecasting Future World Events with Neural Networks.
Andy Zou, Tristan Xiao, Ryan Jia, Joe Kwon, Mantas Mazeika, Richard Li, Dawn Song, Jacob Steinhardt, Owain Evans, Dan Hendrycks

### Questions
1. I am slightly confused about the "hypothesis" / framing in the Introduction about predicting the next word leading to predicting the next event. Its definitely an interesting hypothesis, but one that can only be tested with base models without post-training which moves models away from "next-token-prediction" behavior. Could you either shift this hypothesis to be less prominent, or test it?

2. It would be useful to shift the benchmark construction methodology to the main paper, as currently the main paper barely talks about it. To make space, section 3.1, 3.2, and 4.1.1 can be shifted to the appendix as they are not unique to this work.

### Soundness
3

### Presentation
3

### Contribution
3
