# Rethinking Multimodal Time-Series Forecasting Evaluation

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
We introduce a new context-enriched, multimodal time series forecasting benchmark TimesX. TimesX contains a wide selection of high-quality real-world time series with diverse domains and textual contexts obtained from an automated data generation pipeline, which helps address three main issues of existing multimodal forecasting benchmarks: (1) poor generalization due to the small scale and synthetic nature of benchmark data, (2) very limited types of textual contexts in the benchmarks, and (3) an inability to mitigate data leakage in evaluation. We conduct a thorough empirical study of zero-shot multimodal forecasting approaches on TimesX. Our results suggest that many approaches that perform well on existing benchmarks may fail on TimesX. In contrast, simple ensemble methods that leverage rich textual context accompanying time-series can outperform strong baselines on the TimesX benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces TimesX, a novel, context-enriched, multimodal time-series forecasting benchmark designed to address key shortcomings in existing evaluation datasets, specifically poor generalization due to synthetic and small-scale data, limited types of textual contexts, and the inability to mitigate data leakage. TimesX is the first real-world, large-scale benchmark, featuring 19 diverse domains and 190 variables, each linked to comprehensive and verifiable textual contexts including Metadata, Calendar features, Covariates, and Time-Stamped Events. A key contribution is its automated, leakage-free data generation pipeline, which employs a hypothesizer-verifier-enricher framework and strict time isolation to ensure context quality and long-term validity against continuously updated foundation models. The authors demonstrate that existing zero-shot approaches struggle with TimesX, and simple ensemble methods leveraging the rich contextual data can outperform strong baselines on this more realistic platform.

### Strengths
1. Unprecedented Scale and Realism: TimesX is a first-of-its-kind, large-scale, cross-domain benchmark built entirely from real-world time series and textual data across 19 domains and 190 variables, directly addressing the poor generalization and synthetic bias prevalent in prior work.

2. Robust Data Leakage Mitigation: The benchmark incorporates a novel, automated data generation pipeline with strict time isolation and an updating mechanism, ensuring tasks occur after a model’s knowledge cutoff date and guaranteeing the benchmark's leakage-free and long-lived validity against future foundation models.

3. High-Quality and Comprehensive Textual Context: TimesX provides a rich, fine-grained collection of four distinct context types (Metadata, Calendar, Covariates, and Time-Stamped Events) and uses a sophisticated Hypothesizer-Verifier-Enricher framework to ensure their quality and verifiability. This high-quality context is empirically shown to be critical for model performance, reducing error compared to lower-quality contexts.

### Weaknesses
1. Limited Focus on Novel Model Architectures: The paper's empirical evaluation focuses predominantly on existing zero-shot baselines (LLMs, TFMs) and compositional methods (TEXTREV, FUNCREV), but does not propose or evaluate a dedicated multimodal architecture specifically designed to leverage the unique, high-quality structure of the TimesX context, leaving the full potential of the benchmark unquantified.

2. Complexity and Accessibility of Data Generation: The proposed Hypothesizer-Verifier-Enricher multi-agent pipeline for generating time-stamped events is highly sophisticated and relies on advanced LLMs and automated tools, which may make the reproducibility and cost-effective updating of the benchmark challenging for the broader academic community without access to comparable resources.

3. Absence of Fine-tuning Evaluation: While the authors rightly focus on zero-shot evaluation, the paper explicitly notes that TimesX does not offer a pretraining or fine-tuning dataset, thereby limiting its utility for researchers aiming to develop and benchmark new fine-tuned multimodal TSF models that could potentially exploit the richness of the context more effectively than zero-shot methods.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a new benchmark called TimesX for multimodal time series forecasting, addressing three key challenges: (a) insufficient generalization gap due to small-scale or synthetic datasets, (b) data leakage issues, and (c) limited types of textual context. To tackle (a), the authors propose 190 diverse datasets. For (b), they implement a time isolation approach, ensuring the target forecast date occurs after the model's public knowledge cutoff to prevent data contamination, along with an automated data refresh mechanism for evaluating future pretrained models. To address (c), the authors expand the textual context by including metadata, calendar events, covariates, and other event types. They benchmark several existing models on TimesX, demonstrating the limitations of previous benchmarks and highlighting the advantages of their proposed benchmark.

### Strengths
1. Benchmarking correctly multimodal time series forecasting is important and timely

2. The benchmarks contain a lot of datasets

### Weaknesses
1. The justification for the dataset’s relevance is weak. The claim that small-scale and synthetic datasets hurt generalization is debatable. How do the authors demonstrate that their benchmark is more realistic than others? They mention that synthetic datasets focus on certain multimodal aspects, but how can they prove their benchmark doesn't have the same limitations, given its focus on specific elements (e.g., metadata, calendar events, covariates)?

2. The authors should provide more details on the origin of the time series data. Are these datasets from previous benchmarks? What are the new datasets, and where do they come from? It would also be useful to visualize the datasets (e.g., PCA, t-SNE) to show their diversity, check for distribution shifts, and explain the train/test split process.

3. The originality of the paper seems limited. The main innovation the three agents managing dataset generation resembles prompt engineering for time series description. More in-depth analysis is needed on the dataset’s relevance, including PCA representations, distribution shifts, outlier analysis, LLM hallucinations, and the quality of time series descriptions.

### Questions
1. Could the authors clarify the origins of the time series datasets? Specifically, are they used in any well-known benchmarks (even non-multimodal ones)? A table listing the datasets and indicating where they have been previously used would be helpful.

2. It would be useful to perform a basic analysis of the time series. What are the distributions over time, across samples, and over channels for each dataset? Do these distributions appear realistic, or do they seem trivial? Additionally, what is the correlation profile between channels and the time structure?

3. The process involving the three agents for text generation seems promising, but how can we be sure it doesn't lead to hallucinations? Is the generated text redundant with information already present in the time series? Could the authors consider introducing a gradual difficulty in the generated text where the same dataset would have progressively more informative descriptions (even within the same event category)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes TimesX, a large, real-world multimodal TS forecasting benchmark that pairs numeric series with four kinds of text context (metadata, calendar, covariates, and time-stamped events). It contains time series forecasting benchmark over 19 diverse domains and 190 variables in total. The key design choices are (i) time isolation (evaluate only after model knowledge cutoffs) with an auto-refreshable pipeline to mitigate leakage, and (ii) a multi-agent hypothesizer-verifier-enricher workflow to to create an event corpus with verifiable facts, URL, and accurate timestamps.

### Strengths
- The benchmark is large-scale and diverse covering different geographical locations, spanning over 2.5 years (Jan 2023 - Jun 2025), and over 19 domains with 10 variables per domain, and two granularities (weekly and daily)

- The hypothesizer-verifier-encricher agent framework can help truthfulness and mitigate data leakage issues.

### Weaknesses
- Evaluation horizons are selected to be after the model cutoff date, but there is no guarantee that the hypothesizer-verifier-encricher framework will not retrieve data from the future horizon/beyond the cutoff data (since it works based on web-search to my understanding).

- The authors claim the benchmark can be refreshed for a new pretrained model with a later cutoff date, but the paper does not provide details on how the refresh works or what the refresh exactly entails? Does it amount to just prompting the model to not look beyond a certain date?  If so, that is not a contribution of this work since any LLMs can be prompted to search for information within a time window.

### Questions
- How many tasks for each domain should be made clear in the main text

- In figure 1, which tasks from CiK do the authors run the 3 models on? Is it over a single dataset, or over multiple CiK datasets?

- Throughout the manuscript, authors report geometric mean. Is there a reason why this was chosen over the more commonly used arithmetic mean?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces TimesX, a novel multimodal time series forecasting benchmark designed to address limitations in existing benchmarks. The key contributions are: (1) A large-scale, real-world dataset spanning 19 domains with 190 variables, featuring diverse textual contexts (metadata, calendar events, covariates, and timestamped events); (2) A data generation pipeline with strict time isolation to prevent data leakage and enable automatic updates; (3) Empirical analysis showing that simple ensemble methods outperform complex LLM-based revisions on real-world data, challenging conclusions from synthetic benchmarks. The benchmark emphasizes real-world applicability, leakage prevention, and high-quality context alignment through a multi-agent workflow.

### Strengths
Large-scale: a new benchmark to combine real-world data, leakage prevention, and automated context generation. The multi-agent workflow for event extraction is innovative.
Quality: Extensive evaluation (13 methods across 190 variables) with robust metrics (MASE). The pipeline design ensures reproducibility.
Clarity: Well-organized, with clear examples (e.g., gas price case study) and appendix support.
Significance: Challenges prevailing assumptions about LLM superiority in synthetic benchmarks and provides a practical foundation for future research.

### Weaknesses
Coverage gaps: Most series are daily or weekly. Future versions could add lower-frequency series to broaden applicability.
Language & region bias: Current textual contexts are English-centric and skew toward North-American events. Authors could discuss plans for multilingual expansion.

Hyper-parameter sensitivity: Peak-detection thresholds (θ, Kmax, Lmax) that drive event recall are fixed; an ablation or sensitivity plot would strengthen robustness claims.

Evaluation metrics: Relies solely on MASE. Adding CRPS or interval scores could illuminate probabilistic calibration differences between TFMs and LLMs.

The context generation pipeline relies on LLMs and web searches, which may inherit biases or scalability limitations. A cost/error analysis of the pipeline would strengthen practicality.

### Questions
How scalable is the multi-agent context generation pipeline to larger datasets or domains with sparse events (e.g., rare diseases)?

Could the benchmark incorporate uncertainty quantification metrics (e.g., prediction intervals) to assess model reliability?

Have you explored fine-tuning LLMs/TFMs on TimesX, and how might it alter the conclusions about ensemble methods?

What steps are taken to mitigate potential biases in event data sourced from web searches (e.g., geographic or media bias)?

### Soundness
4

### Presentation
2

### Contribution
4
