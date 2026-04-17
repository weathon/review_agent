000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Time series captioning, the task of describing numeric time series in natural language, requires numeric reasoning, trend interpretation, and contextual understanding. Existing benchmarks, however, often rely on synthetic data or overly simplistic captions, and typically neglect metadata and visual representations. To close this gap, we introduce **CaTS-Bench**, the first large-scale, real-world benchmark for Context-aware Time Series captioning. CaTS-Bench is derived from 11 diverse datasets reframed as captioning and Q&A tasks, comprising roughly 465k training and 105k test timestamps. Each sample includes a numeric series segment, contextual metadata, a line-chart image, and a caption. A key contribution of this work is the scalable pipeline used to generate reference captions: while most references are produced by an oracle LLM and verified through factual checks, human indistinguishability studies, and diversity analyses, we also provide a human-revisited subset of 579 test captions, refined from LLM outputs to ensure accuracy and human-like style. Beyond captioning, CaTS-Bench offers 460 multiple-choice questions targeting deeper aspects of time series reasoning. We further propose new tailored evaluation metrics and benchmark leading VLMs, highlighting both their strengths and persistent limitations. Together, these contributions establish CaTS-Bench and its captioning pipeline as a reliable and extensible foundation for future research at the intersection of time series analysis and foundation models.

![0_image_0.png](0_image_0.png)

## 1 Introduction

Effective interpretation of time series data is a cornerstone of decision-making in domains ranging from financial markets and healthcare monitoring to climate analysis and industrial automation. Yet, distilling raw numeric sequences into concise, human-readable summaries remains a labor-intensive task, requiring domain expertise, statistical know-how, and careful visualization. Automating this process through *time series captioning* (TSC) not only accelerates insight discovery but also democratizes access to complex temporal analytics, enabling non-experts to ask natural-language questions and receive meaningful explanations without writing code or inspecting raw charts.

# Cats-Bench: Can Language Models Describe Numeric Time Series?

Anonymous authors Paper under double-blind review

## Abstract

1 Large language models (LLMs) and vision-language models (VLMs) have demonstrated remarkable prowess in text generation and visual reasoning, respectively. However, when applied to time series, they reveal critical deficiencies: LLMs exhibit well-documented limitations in precise numeric extrapolation, temporal continuity, and uncertainty quantification (Tang et al., 2025; Merrill et al., 2024; Tan et al., 2024; Cao & Wang, 2024). While VLMs have shown promise in visual pattern recognition tasks such as trend and anomaly detection from plots (Zhou & Yu, 2025), their capacity for fine-grained numeric time series reasoning remains largely underexplored. These limitations underscore a broader challenge: existing evaluation resources fail to reflect the complexity of real-world temporal signals, leaving model improvements unguided by the demands of true data-driven applications. In response, the community has proposed Time Series Captioning (TSC) as a more natural task for foundation models, leveraging their generative and reasoning capabilities to narrate trends, anomalies, and context in prose (Trabelsi et al., 2025; Jhamtani & Berg-Kirkpatrick, 2021). However, current benchmarks remain narrow, often synthetic or restricted to simple trend labels, and exclude rich metadata or visual modalities. Consequently, progress in model architecture, pretraining, or finetuning cannot be measured against challenges that mirror real deployment scenarios, slowing adoption in high-stakes sectors where accurate temporal interpretation is essential.

To fill this gap, we introduce **CaTS-Bench**, the first large-scale, multimodal benchmark explicitly designed for *context-aware* time series captioning and reasoning. We define "context-aware" to mean that captions are informed by both the metadata (units, domain labels, dates, region, etc.) and visual cues that provide semantic and numeric grounding. By mining 11 real-world datasets across various domains, CaTS-Bench provides 20k triplet samples drawn from 570k time steps of curated data, each paired with (1) rich metadata containing contextual information, units, and domain-specific cues (Dong et al., 2024; Wang et al., 2024); (2) a corresponding line plot image, enabling the use of VLMs (Chen et al., 2024a; Zhou & Yu, 2025); and (3) a reference caption produced by a scalable oracle-based pipeline and validated through factual checks, human indistinguishability studies, and diversity analyses. To further strengthen reliability, we additionally release a *human-revisited subset* of test captions: sampled from multiple LLM candidates and carefully edited by the authors to remove inaccuracies, speculative claims, and linguistic repetitions. This subset complements the larger benchmark with high-fidelity, human-styled references. Beyond captioning, CaTS-Bench also includes 460 challenging multiple-choice questions spanning time series matching, caption matching, plot matching, and comparative reasoning, designed to expose models' blind spots in numeric precision and multimodal alignment. All data samples are made available here. We further propose new evaluation metrics tailored to time series captioning that move past generic N-gram overlap to reward numeric fidelity and coverage. Our comprehensive experiments on leading VLMs reveal that, in both zero-shot and finetuned settings, models can produce fluent text but fail to reliably capture quantitative details without specialized adaptation. A key finding is that VLMs fail to effectively leverage the visual cues provided for time series captioning, pointing to a significant limitation in current multimodal architectures. Our analysis identifies clear room for improvement, such as better leveraging visual cues, enhancing multimodal alignment, and incorporating dedicated numeric reasoning modules. These findings pave the way for a new generation of foundation models capable of translating complex temporal data into actionable narratives. In summary, the contributions of this paper are: 1. **Scalable Captioning Pipeline**: A reproducible pipeline for generating high-quality time series captions. It anchors LLM outputs in factual metadata, validates them through factual checks, human indistinguishability studies, and diversity analyses, and is extensible to new datasets.

2. **CaTS-Bench**: A multimodal, context-aware benchmark for time series captioning and reasoning, featuring time series segments, rich metadata, visual plots, and factually grounded captions. Most references are LLM-generated via the pipeline, while a curated subset of human-revisited test captions ensures high-fidelity, human-styled references alongside the larger benchmark.

3. **Diagnostic Q&A Suite**: Four multiple-choice tasks designed to isolate capabilities in series matching, caption grounding, visual reasoning, and comparative analysis.

4. **Comprehensive Evaluation**: Zero-shot and finetuned assessments of state-of-the-art VLMs, revealing strengths, failure modes, and clear directions to advance time series understanding.

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107

## 2 Related Work

LLMs are increasingly being repurposed for time series analysis (Zhang et al., 2024; Liu et al., 2024a), with early efforts primarily focused on forecasting. These approaches span prompt engineering (Liu et al., 2024a; Chatzigeorgakidis et al., 2024), modality alignment (Liu et al., 2024b; Sun et al., 2023; Liu et al., 2024c; Pan et al., 2024), discretization (Ansari et al., 2024; Jin et al., 2024), and specialized finetuning (Zhou et al., 2023; Chang et al., 2023). Such studies highlight that LLMs pretrained on text can reason over temporal data, but subsequent work also shows consistent weaknesses in handling long-range dependencies, numeric precision, and structured reasoning, particularly in forecasting and anomaly detection (Tang et al., 2025; Merrill et al., 2024; Tan et al., 2024; Cao & Wang, 2024; Zeng et al., 2023).

Table 1: Comparison of TSC benchmarks.

| Dataset                                   | # Timesteps   | Modality                | Sources Metadata   | Captions   | TSC Q&A       |    |    |
|-------------------------------------------|---------------|-------------------------|--------------------|------------|---------------|----|----|
| TADACap (Fons et al., 2024)               | N/A           | Visual                  | 4                  | Minimal    | Patterns Only | ✓  | ✗  |
| TRUCE (Jhamtani & Berg-Kirkpatrick, 2021) | 34k           | Numeric                 | 2                  | ✗          | Patterns Only | ✓  | ✗  |
| TACO (Dohi et al., 2025)                  | 2.46b         | Numeric                 | 8                  | ✗          | Expressive    | ✓  | ✗  |
| CaTS-Bench                                | 570k          | Numeric + Text + Visual | 11                 | Rich       | Expressive    | ✓  | ✓  |

Building on these foundations, researchers have explored Time Series Captioning (TSC), a task more aligned with the generative strengths of language models. TSLM (Trabelsi et al., 2025) introduces an encoder–decoder trained on synthetic cross-modal data; TADACap (Fons et al., 2024) retrieves domain-aware captions for visualized time series; TRUCE (Jhamtani & Berg-Kirkpatrick, 2021) employs a truth-conditional framework to validate simple trend patterns; and TACO (Dohi et al., 2025) scales up caption corpora using LLM-based synthetic generation. While each provides valuable insights, they remain limited in scope: TADACap and TRUCE are domain-specific and pattern-oriented, while TACO's reliance on templates restricts contextual richness (See Table 1). Beyond these, standard time-series archives such as UCR (Chen et al., 2015), UEA (Bagnall et al., 2018), and Monash (Godahewa et al., 2021) support classification and forecasting but not generative captioning. Similarly, benchmarks like PISA (Xue & Salim, 2023) target prompt-based forecasting, omitting metadata entirely. Recent evidence shows that incorporating auxiliary modalities (metadata, domain context, or visual renderings) can significantly improve both interpretability and predictive performance (Zhou & Yu, 2025; Dong et al., 2024; Chen et al., 2024a; Wang et al., 2024; Kim et al., 2024; Williams et al., 2024; Liu et al., 2025; Tang et al., 2023). Yet no benchmark to date integrates large-scale numeric series, expressive captions, rich metadata, and multimodal grounding. CaTS-Bench fills this gap by offering the first benchmark that unifies numeric time series, metadata, and visuals with both expressive captions and Q&A tasks for systematic evaluation for TSC.

## 108

109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

## 3 Cats-Bench

In this section, we illustrate the entire data curation pipeline and the design of the benchmark tasks. While examples generated from this pipeline can be directly used for TSC evaluation, we further enrich the scope of CaTS-Bench by providing an additional suite of Q&A tasks constructed from the same data, enabling a more fine-grained examination of time series and caption reasoning abilities.

## 3.1 Data Curation

We build **CaTS-Bench**, a comprehensive benchmark curated from 11 diverse real-world source datasets spanning domains: climate (Jha, 2023; Ritchie, 2021), safety (of Los Angeles, n.d.; of Public Health, n.d.), USA border crossing (U.S. Department of Transportation, n.d.), demography (Aziz, 1985), health (European Centre for Disease Prevention and Control, 2024; Food and Agriculture Organization of the United Nations, 2024), sales (Hassan, 2020; Chen, 2015), and agriculture (USDA
Economic Research Service, 2024). See Appendix B for more details on the source datasets. The overall data pipeline is shown in Figure 2. Each source dataset provides a full-length time series per entity (e.g., country, city, product), and to generate samples, we apply a random window cropping strategy. For each dataset, we define a valid range of window lengths and randomly select a size for each crop; see Appendix C for our range calculation. The number of windows sampled from a dataset 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 depends on its total time steps, ensuring fair representation. The domain-specific number and lengths of the time series windows are illustrated in Table 2. Each time series window is augmented with a **metadata JSON** file with contextual information (domain, location, start time, etc.), a **line plot** image with randomized visual style (color, width, figure size), a **ground truth caption** produced by querying an oracle LLM (Gemini 2.0 Flash) with a structured prompt that includes: (i) the serialized numeric values of the cropped segment and (ii) metadata enriched with numericly grounded information, including both the historical and sample-specific mean, standard deviation, minimum, and maximum. An example of the prompt is available in Appendix N.1. We emphasize that time series captioning lacks inherent ground truth at the level of a single canonical description: multiple valid ways exist to describe the same series depending on focus and phrasing. To provide consistent references at scale, our primary captions are generated by an oracle LLM, but anchored strictly in the underlying data. The oracle receives full contextual metadata (not available at evaluation time) and is instructed not to include any external knowledge, ensuring captions remain factual and context-grounded. This design makes captions a practical proxy for evaluation and challenges models to reason from multimodal inputs rather than mimic the oracle. Furthermore, we randomize time series window sizes and plot styles to prevent overfitting and better reflect real-world variability in length and visualization styles.

Agriculture Air Quality Crop Original Time **Series**

Generated Caption Oracle LLM
Covid
[18, 25, 36, 35, ...., 22]

Time Series Sample Location: Delhi Measure: Temperature (C) Sample Frequency: Hourly Start Date: 2023-01-17 Min: 7.9 Max: 19.3
.....

"<image> Here is a time series about hourly Temperature (C) in the Indian city of Delhi: \n
[18, 25, 36, 35, ...., 22] ........... Write a caption for this time series data".

Final **Prompt**
"Write a caption for this time series data."

11 Data **Sources**
Walmart Metadata Prompt **Template**
Figure 2: Overview of the CaTS-Bench semi-synthetic data generation pipeline. A time series window is cropped, metadata is attached, and an oracle LLM generates a reference caption. See Appendix L for examples and Appendix H for the quality verification protocol.

Table 2: Dataset outline by domain. AQ: Air Quality, Border: Border Crossing, Demo: Demography, Injury: Road Injuries, Calories: Calories Consumption, Agri: Agriculture

| Metric                    | Overall   | AQ Border Crime Demo Injury COVID CO2 Calories Walmart Retail Agri   |      |      |      |     |      |     |      |      |      |     |
|---------------------------|-----------|----------------------------------------------------------------------|------|------|------|-----|------|-----|------|------|------|-----|
| # Source Time Steps       | 287M      | 286M                                                                 | 397k | 38k  | 14k  | 37k | 720k | 34k | 234k | 6k   | 7k   | 49k |
| # Samples Generated       | 20k       | 4.4k                                                                 | 3.2k | 764  | 598  | 756 | 5.5k | 732 | 2.1k | 544  | 551  | 835 |
| # Train Samples           | 16k       | 3.5k                                                                 | 2.6k | 611  | 478  | 604 | 4.4k | 585 | 1.7k | 435  | 440  | 668 |
| Avg. Sample Length        | 29.1      | 65.3                                                                 | 21.2 | 76.8 | 11.6 | 5.9 | 75.8 | 9.5 | 12.2 | 12.2 | 22.4 | 7.3 |
| # Test Samples            | 4k        | 886                                                                  | 646  | 153  | 120  | 152 | 1.1k | 147 | 422  | 109  | 111  | 167 |
| Avg. Sample Length        | 26.1      | 66.0                                                                 | 21.2 | 76.9 | 5.0  | 3.6 | 73.0 | 8.7 | 5.5  | 11.8 | 8.1  | 7.5 |
| # Human-revisited Samples | 579       | 0                                                                    | 0    | 153  | 120  | 0   | 0    | 0   | 0    | 109  | 0    | 167 |
| Avg. Sample Length        | 25.7      | -                                                                    | -    | 76.9 | 5.0  | -   | -    | -   | -    | 11.8 | -    | 7.5 |

To prevent information leakage, we partition each source dataset temporally before generating the samples. Specifically, the first 80% is used for generating training samples, whereas the last 20% is reserved exclusively for generating test samples. Random window cropping is applied separately to the training and test partitions. This strategy ensures that the model is evaluated on future, unseen data relative to the training set. The actual benchmark samples consist of the test split resulting from this process. We leave the training split of the data for optional training. Our final semi-synthetic dataset version contains 20k examples, split into roughly 16k training samples and 4k test samples.

Detailed statistics and source of our data are reported in Table 2. Human-Revisited Subset. We also release a curated subset of test captions that have been revisited by humans. These captions were first sampled from multiple LLM candidates (Gemini 2.0 Flash, GPT-4o, Gemma 27B, and Llama 90B) using the above data pipeline, and then carefully refined by the authors to eliminate factual errors, speculative statements, and redundant phrasing. Drawn from the domains of agriculture, crime, demography, and Walmart sales, this subset provides high-fidelity, human-styled references that complement the larger benchmark.

## 3.2 Quality Validation Of Semi-Synthetic Captions

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 To ensure the quality of CaTS-Bench, we conducted a series of comprehensive verification studies addressing the core concern of semi-synthetic data: whether captions generated by the oracle model (Gemini 2.0 Flash) are factual, unbiased, and linguistically diverse. These analyses demonstrate that semi-synthetic captions in CaTS-Bench provide high-quality references and stable benchmarks for TSC, and thus are a sufficient proxy for human-written descriptions in practical scenarios. We verified caption quality through three complementary studies below (with full details in Appendix H). Manual Validation. We manually checked ∼2.9k captions (72.5% of the semi-synthetic test benchmark) across statistical claims (min, max, mean, STD) and trend descriptors (up/downward, stable, fluctuating). Accuracy exceeded 98.6% on average across all categories (Table 9) which confirms that captions faithfully reflect underlying series properties. Human Detectability Study. In a blind test with 35 participants, subjects attempted to distinguish our captions from those written by humans. Accuracy was near random at 41.1%, suggesting that our captions are indistinguishable from human-authored ones and no evidence of oracle-specific bias. Diversity and Bias Analysis. Captions consistently drew from a wide variety of statistical and temporal descriptors (Table 12), and embedding-based similarity analysis across nine embedding models revealed minimal template reliance. Pairs of captions that were almost semantically identical, measured as embedding cosine similarity > 0.95, were rare, averaging 2.3% of occurrences (Table 13). Comparisons with human captions ( H.4.4) indicate that Gemini's outputs are stylistically intermixed with human text, while N-gram analysis (H.4.2) confirms high lexical diversity.

## 3.3 Time Series Captioning

TSC requires generating a detailed, coherent narrative that highlights the key characteristics of a given time series. During evaluation, each model is presented with a standardized multi-part prompt that combines four elements: the **numeric series** itself, embedded as raw time-indexed values (e.g.,
[25.3, 26.1, 26.8, ...]); **contextual metadata** such as measurement units, data source, sampling interval, and domain tags (e.g., "Hourly temperature readings from Rome, May 2000"), which excludes explicit statistics like mean or maximum since the model must infer them; a **visual** input in the form of a line-plot image that allows vision-language models to ground their descriptions in visual trend cues; and a fixed-format **instruction template** containing the directive for caption generation (see Appendix N.2). By standardizing this multi-part prompt, we evaluate models on their ability to recognize numeric trends (e.g., rising or falling segments, peaks, and troughs), integrate metadata cues, and utilize visual features to produce context-aware captions.

## 3.4 Q&A Multiple-Choice Tasks

We introduce a suite of multiple-choice Q&A tasks designed to probe different reasoning skills in time series understanding. All tasks are automatically derived from the same source data used for captioning, with questions generated from task-specific, fixed templates (see Appendix J.1 for examples). To increase difficulty, an initial pool of 4k questions per type was filtered by removing those correctly answered by Qwen 2.5 Omni. Appendix J.2 shows that this filtering produces genuinely harder questions, rather than reflecting Qwen-specific weaknesses only. Ambiguous Time Series Matching questions were manually checked to ensure a single correct answer. From the remaining 7k challenging questions, a random subset of 460 was sampled as the final test set, including 100 each for time series matching, caption matching, and plot matching, and 40 each for amplitude, peak, mean, and variance comparison tasks. Question types are described below. Time Series Matching. Given a caption, the model must retrieve the correct time series from distractor candidates created via shuffling, temporal reversal, and Gaussian noise. These perturbations prevent simple numeric lookup and require alignment with both values and trends (see J.3 for details). Caption Matching. Given a time series, the model must select the correct caption from distractors composed of random captions and perturbed variants of the ground truth (see Appendix N.5, N.6). This isolates caption understanding from free-form generation. Plot Matching. Given a caption and its numeric series, the model must select the correct line plot from the candidates, testing visual grounding and the ability to link language with visual patterns. Time Series Comparison. Given two time series, select the correct comparative statement from a pair of options (e.g., "Series A peaks earlier than Series B" or "Series B has a higher volatility than Series A"). This task challenges models to perform temporal and statistical comparison, a setting where many language models currently struggle (Merrill et al., 2024).

## 3.5 Evaluation Metrics

To comprehensively evaluate model-generated captions against the ground truth in TSC, we employ a diverse set of metrics that target linguistic quality, statistical inference, and numeric fidelity. For Q&A, we adopt accuracy as the evaluation metric, as each question is designed to have a single correct answer. Below, we describe each metric used for TSC in our evaluation framework. Standard Linguistic Metrics. We assess caption similarity using standard NLP metrics, including DEBERTA SCORE (Zhang* et al., 2020), BLEU (Papineni et al., 2002), ROUGE-L (Chin-Yew, 2004), METEOR (Banerjee & Lavie, 2005), and SIMCSE (Gao et al., 2021; Liu et al., 2019). Together, these metrics capture both surface-level linguistic overlap and deeper semantic similarity. This ensures that evaluation does not merely reflect stylistic resemblance but instead rewards accurate semantics of the underlying time series phenomena. Refer to Appendix F for more details. numeric Fidelity Metrics. Since TSC involves reporting exact or approximate numeric values, we introduce two tailored metrics to quantify numeric accuracy, both bounded within [0, 1]. The choice of the 5% tolerance is discussed in Appendix F.2. 1. **Statistical Inference Accuracy.** While models are explicitly prompted to discuss descriptive statistics, they demonstrate varying abilities to accurately infer and verbalize statistics such as the mean, standard deviation, minimum, and maximum based on the raw time series and metadata. To evaluate this behavior, we report the percentage of captions in which these statistics are mentioned and fall within a 5% relative error, using offline-computed true values. Importantly, captions are not penalized for omitting statistics; only wrongly reported values are considered errors. This metric primarily measures hallucination, favoring omission over incorrect numeric claims.

2. **Numeric Score.** For each ground truth caption, we extract all numeric values (excluding timerelated ones like year or month) and search for the closest numeric value in the generated caption.

A match is recorded if the closest value is within a 5% relative tolerance. We compute *Accuracy* (mean of 1−min{relative_error, tolerance}) over all matched numbers), *Recall* (fraction of ground truth numbers matched), and a *Final Score* as a weighted combination: λA ·Accuracy+λR ·Recall, with λA = 0.3 and λR = 0.7 to emphasize recall over precision, as omitting critical numbers is more severe than minor numeric rounding imprecisions. While the previous metric targets numeric hallucinations, this one focuses on penalizing captions that omit numeric details.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 4 Experiments

We evaluate a broad range of VLMs on CaTS-Bench, covering both proprietary and open-source models, with the latter also tested after finetuning on our captioning training set (details in Appendix D). For TSC, we additionally consider a *program-aided* (PAL) model (Gao et al., 2023). All models are prompted with the same template-based format to ensure fair comparison, avoiding task- or architecture-specific prompt engineering. Appendix E provides the full model list and a description of PAL, while Appendix O outlines the human baseline that participated in our Q&A evaluation.

## 4.1 Time Series Captioning

To ensure fair comparison across the domains, we report macro-averaged scores for each metric, mitigating sample size imbalances, as some domains contain more data, and preventing any domain from disproportionately influencing the results. We benchmark leading VLMs on TSC using the

| Category         | Model            | DeBERTa F1   | SimCSE   | BLEU   | ROUGE-L   | METEOR   | Numeric   |       |       |       |       |       |       |
|------------------|------------------|--------------|----------|--------|-----------|----------|-----------|-------|-------|-------|-------|-------|-------|
| HR               | SS               | HR           | SS       | HR     | SS        | HR       | SS        | HR    | SS    | HR    | SS    |       |       |
| Proprietary      | Gemini 2.0 Flash | 0.665        | 0.688    | 0.856  | 0.858     | 0.079    | 0.137     | 0.248 | 0.318 | 0.221 | 0.279 | 0.634 | 0.677 |
| Gemini 2.5 Pro   | 0.657            | 0.668        | 0.857    | 0.845  | 0.069     | 0.088    | 0.236     | 0.267 | 0.247 | 0.284 | 0.681 | 0.714 |       |
| Claude 3 Haiku   | 0.658            | 0.682        | 0.853    | 0.856  | 0.064     | 0.112    | 0.241     | 0.291 | 0.236 | 0.300 | 0.601 | 0.623 |       |
| GPT-4o           | 0.661            | 0.681        | 0.863    | 0.865  | 0.071     | 0.112    | 0.233     | 0.284 | 0.236 | 0.296 | 0.627 | 0.644 |       |
| InternVL 2.5 38b | 0.664            | 0.688        | 0.871    | 0.868  | 0.072     | 0.129    | 0.244     | 0.305 | 0.255 | 0.331 | 0.659 | 0.685 |       |
| LLaVA v1.6       | 0.627            | 0.650        | 0.824    | 0.820  | 0.052     | 0.086    | 0.215     | 0.259 | 0.233 | 0.287 | 0.455 | 0.517 |       |
| LLaVA v1.6 34b   | 0.639            | 0.655        | 0.821    | 0.825  | 0.060     | 0.094    | 0.221     | 0.265 | 0.232 | 0.285 | 0.547 | 0.560 |       |
| Idefics 2        | 0.602            | 0.604        | 0.784    | 0.698  | 0.024     | 0.040    | 0.192     | 0.226 | 0.140 | 0.162 | 0.424 | 0.455 |       |
| SmolVLM          | 0.592            | 0.594        | 0.755    | 0.693  | 0.027     | 0.044    | 0.194     | 0.224 | 0.154 | 0.178 | 0.431 | 0.474 |       |
| QwenVL           | 0.619            | 0.643        | 0.821    | 0.890  | 0.049     | 0.082    | 0.209     | 0.249 | 0.214 | 0.261 | 0.445 | 0.504 |       |
| QwenVL PAL       | 0.664            | 0.685        | 0.864    | 0.843  | 0.066     | 0.108    | 0.237     | 0.292 | 0.226 | 0.282 | 0.564 | 0.613 |       |
| Llama 3.2 V      | 0.653            | 0.671        | 0.852    | 0.850  | 0.072     | 0.118    | 0.239     | 0.290 | 0.252 | 0.315 | 0.650 | 0.685 |       |
| Gemma 3 27b      | 0.648            | 0.667        | 0.863    | 0.863  | 0.065     | 0.085    | 0.222     | 0.263 | 0.257 | 0.309 | 0.641 | 0.668 |       |
| d ainePretr      | LLaVA v1.6       | 0.712        | 0.758    | 0.896  | 0.907     | 0.134    | 0.285     | 0.312 | 0.445 | 0.300 | 0.441 | 0.693 | 0.732 |
| d                | Idefics 2        | 0.711        | 0.759    | 0.894  | 0.908     | 0.132    | 0.290     | 0.309 | 0.452 | 0.298 | 0.437 | 0.691 | 0.733 |
| Finetune         | InternVL-2.5 8b  | 0.638        | 0.655    | 0.817  | 0.809     | 0.053    | 0.088     | 0.215 | 0.259 | 0.229 | 0.282 | 0.582 | 0.594 |
| QwenVL           | 0.703            | 0.643        | 0.892    | 0.790  | 0.126     | 0.082    | 0.302     | 0.249 | 0.297 | 0.260 | 0.683 | 0.504 |       |
| SmolVLM          | 0.604            | 0.613        | 0.817    | 0.781  | 0.051     | 0.091    | 0.228     | 0.269 | 0.220 | 0.265 | 0.556 | 0.643 |       |

semi-synthetic and human-revisited captions separately as ground truth. Selected results are shown in Tables 3 and 4, with complete results in the Appendix G. Table 3: Selected evaluation results of generated captions against human-revisited (HR) and semisynthetic (SS) ground truths. **Bolded** and underlined scores denote first and second places. Semi-synthetic (SS) Captions as Ground Truth. Our experiments show that finetuning substantially improves performance across most metrics. Proprietary models such as GPT-4o and *Gemini* generally outperform Claude. Among open-source models, finetuned Idefics 2 and LLaVA
v1.6 Mistral achieve strong gains, in some cases surpassing proprietary baselines, underscoring the effectiveness of finetuning for both linguistic quality and numeric precision. QwenVL PAL shows marked improvements over standard QwenVL and even takes the lead on statistical inference metrics (as shown in Table 4), highlighting code execution as a practical enhancement for tasks where numbers matter. Given the semi-synthetic nature of ground truths in this experiment, we assessed the robustness of evaluation along two axes. First, to account for the stochasticity of LLM outputs, we repeated inference three times on ∼ 600 test samples across five representative models; variance was vanishingly small (often 10−6; Appendix H.5), confirming that our single-run results are stable and reliable. Second, to test sensitivity to linguistic style, we paraphrased a subset of ground truth captions using multiple architecturally distinct LLMs while strictly preserving all factual content and numeric details, generating variants of ground truths differing only by linguistic style. The paraphrasing prompt is provided in Appendix N.3. Re-evaluating baseline outputs against these paraphrased captions as ground truth yielded model performance rankings largely consistent with those based on the original Gemini captions, with a mean Spearman Correlation of 0.9266 Table 4: Representative statistical inference scores under ground truths. E.g., *Mean* indicates statistical inference of the series mean. **Bolded** and underlined scores denote first and second places.

Category Model Mean Max Min HR SS HR SS HR SS
Pr oprie taryGemini 2.0 Flash 0.536 0.651 0.982 0.985 0.936 0.917 Gemini 2.5 Pro Prev. 0.323 0.494 0.987 0.994 **0.977 0.971** Claude 3 Haiku 0.833 0.693 0.980 0.977 0.934 0.898 GPT-4o 0.817 0.700 **0.992** 0.990 0.938 0.921 InternVL 2.5 38b 0.858 0.784 0.982 0.966 0.930 0.887 LLaVA v1.6 Mistral 0.667 0.644 0.871 0.864 0.751 0.743 LLaVA v1.6 34b 0.410 0.445 0.817 0.843 0.727 0.698 Idefics 2 0.806 0.616 0.891 0.903 0.840 0.806 QwenVL 0.656 0.565 0.795 0.822 0.678 0.657 QwenVL PAL **0.973 0.903** 0.985 0.980 0.978 0.942 Llama 3.2 Vision 0.467 0.594 0.956 0.952 0.895 0.877 Gemma 3 27b 0.734 0.694 0.978 0.968 0.904 0.864 Pr etrai ned Finetu ned LLaVA v1.6 Mistral 0.928 0.828 0.987 0.976 **0.981** 0.926 Idefics 2 0.958 0.885 0.988 0.985 0.967 **0.927** InternVL 2.5 (8b) 0.750 0.597 0.830 0.904 0.734 0.779 QwenVL 0.952 0.565 0.973 0.822 0.963 0.657 SmolVLM 0.640 0.590 0.914 0.898 0.772 0.777 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 7 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Human-revisited (HR) Captions as Ground Truth. We repeat the evaluation using humanrevisited captions as ground truth, further confirming the benefits of finetuning. Open-source models like Idefics 2 and LLaVA v1.6 Mistral gain substantially in text quality and numeric accuracy, often surpassing proprietary baselines on linguistic metrics and nearing them on numeric ones. Proprietary models such as GPT-4o and *Gemini* still lead on some language-focused metrics, but their advantage shrinks when finetuned open-source models are included. Meanwhile, the PAL model excels in statistical inference thanks to code execution. Overall, these results confirm that finetuning not only enhances average performance but also improves numeric reliability, positioning open-source models as strong contenders when paired with targeted adaptation.

## 4.2 Q&A Tasks

Figure 3 summarizes model performance on our Q&A tasks, while Table 17 provides detailed results. Performance is highly variable, and even proprietary models occasionally fail to exceed random chance on some tasks. No model consistently dominates across all categories. Models handle binary-choice time series comparisons better, likely due to the narrower range of options. Matching a time series to a caption is harder than the reverse, and plot matching is the most challenging, highlighting a key VLM weakness:
linking numeric patterns with visual features. Proprietary models (GPT-4o, Gemini 2.0 Flash) lead, while among open-source models, Phi-4 M.I. excels in time series and statistical reasoning.

Finetuning on TSC yields mixed results: some models (e.g., Phi-4 M.I., Idefics 2) gain in specific sub-tasks, while others drop in performance. Notably, finetuning often fails to improve Q&A accuracy, likely due to task misalignment and catastrophic forgetting. As Table 17 shows, humans achieve the highest overall scores, though top models sometimes outperform them on distractionprone tasks. Notably, all models perform near-random on plot matching, whereas humans score nearly perfectly. Despite the tasks' apparent simplicity, they reveal fundamental limitations in VLMs' temporal reasoning capabilities which suggests the need to address basic time series understanding before tackling more complex applications.

![7_image_0.png](7_image_0.png)

## 4.3 Role Of The Visual Modality

Visual Modality Ablation. We perform a modality removal experiment by stripping away the time series plot and providing only the associated textual metadata and the numeric values of the time series. This quantifies the contribution of the visual channel and enables a better understanding of the model's captioning performance. We evaluate a selected subset of pretrained baselines to assess their intrinsic reliance on vision. Full results can be found in Appendix I.1. Our experiments suggest that the additional contribution of the visual modality to caption quality is insignificant for most models. As shown in Figure 4, most models show only marginal performance drops, or even slight gains, when the time series plot is removed, suggesting a strong dependence on and metric-specific correlations shown in Table 11 (full discussion in H.3). These results corroborate that our evaluation framework is stable and reliably gauges caption quality rather than biased surface-level stylistic alignment.

Models such as QwenVL, LLaVA 1.6 and Claude 3 Haiku maintain strong performance with visual input, but the performance gap (∆) remains modest, underscoring the underuse of plot-based information. Interestingly, the numeric score tends to decline when visual input is removed, hinting at weak but present reliance on the plot for numeric reasoning. These results point to a subtle yet important misalignment: models are exposed to visual data but often fail to meaningfully reason with it. This phenomenon is not limited to line plots, as discussed in I.3, even more expressive visual forms (e.g., Gramian Angular Fields and recurrence plots) fail to trigger visual reasoning of current VLMs in TSC.

```
Gemini 2.0 Flash
                   -0.018 -0.060 -0.032 -0.036 -0.057 -0.047
                   -0.005 -0.012 -0.009 -0.009 -0.014 -0.017
                   -0.050 -0.048 -0.044 -0.016 -0.067 -0.051
                   -0.009 -0.066 -0.025 -0.020 -0.079 -0.092
                    0.001 0.001 0.007 0.015 0.000 0.086
                   -0.028 -0.118 -0.040 -0.033 -0.109 -0.131
                    0.002 -0.004 -0.012 -0.002 0.001 -0.015
                    0.006 0.003 -0.005 0.004 0.002 0.019
                   -0.010 -0.013 -0.038 -0.025 -0.049 -0.007
                                                                                                   0.125
                                                                                                   0.100
                                                                                                   0.075
                                                                                                   0.050
                                                                                                   0.025
                                                                                                  0.000
                                                                                                  0.025
                                                                                                  0.050
                                                                                                  0.075

```

=

VL
L

Idefics2-8B
LLaVA v1.6 Mistral Claude 3 Haiku

| Gemini 2.0 Flash InternVL QwenVL Phi-4 M.I. SmolVLM Llama 3.2 Vision Idefics2-8B LLaVA v1.6 Mistral Claude 3 Haiku   |
|----------------------------------------------------------------------------------------------------------------------|

DeBERTa F1 SimCSE BLEU ROUGE-L METEOR Numeric Figure 4: Performance deltas between VL (vision-language input)
and L (text-only input). Each cell shows ∆ = VL − L. Blue indicates better performance due to the visual input; red the opposite. Visual Attention Analysis. To better understand how VLMs process plots during caption generation, we examined their attention maps. The analysis revealed minimal visual grounding: models concentrated predominantly on textual elements in the plots (e.g., axis labels and titles), with limited evidence of attending to the actual line trends. Attention to visual patterns was sporadic, weak, and inconsistent, suggesting that learned parameters largely disregard visual cues in favor of textual priors. This qualitative evidence highlights the gap between nominal multimodal input and actual integration. Full results are reported in Appendix I.2 and Figure 7.

It is important to note that the under-utilization of visual inputs observed in our experiments is not a limitation of CaTS-Bench itself, but rather a reflection of current VLM capabilities. The benchmark explicitly provides both time series plots and rich metadata, creating ample opportunity for multimodal reasoning. That most models default to textual priors instead of leveraging visual signals highlights a critical gap in the field. We view this as an opportunity for future research: developing models that better integrate plot-based information with textual and numeric cues to advance the broader goal of genuine multimodal understanding in time series analysis.

## 5 Conclusion

textual priors over visual understanding. In particular, models such as Idefics2, Phi-4 M.I., and InternVL perform better in text-only settings on most metrics, hinting that generation is largely driven by language pretraining or instruction tuning rather than true visual interpretation.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 We introduced CaTS-Bench, the first large-scale, multimodal benchmark for context-aware time series captioning and reasoning. Built from 11 diverse real-world datasets, it combines numeric series, metadata, visual plots, and validated captions to provide a challenging testbed beyond synthetic or narrow benchmarks. A key contribution is not only the benchmark itself, but also the scalable data curation pipeline we developed to generate high-quality captions. This pipeline leverages an oracle LLM anchored in metadata, rigorous verification through factual checks, diversity analyses, and a complementary human-revisited subset, making it both scalable and extensible to new domains. Our evaluation of leading VLMs revealed both progress and limitations. Finetuning greatly improves open-source models, enhancing fluency and numeric fidelity, while proprietary models show stronger performance overall. A consistent weakness lies in multimodal grounding: models largely ignore visual inputs, with plot matching emerging as the most difficult task. These findings reveal a critical gap in multimodal alignment and point toward the urgent need for models that can genuinely integrate numeric, textual, and visual cues. By releasing CaTS-Bench together with its evaluation suite, we provide the community with not only a rigorous foundation for advancing time series reasoning, but also a practical methodology for generating reliable, context-rich captions at scale, paving the way for more robust multimodal understanding in the future.

9

## Ethical Statement

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 The development of CaTS-Bench was guided by a commitment to ethical research practices. All datasets used in this work are publicly available and do not contain personally identifiable information (PII). The domains, such as climate, public health, and agriculture, were chosen for their public relevance and data accessibility. Our use of an oracle LLM to generate semi-synthetic reference captions was a deliberate design choice to ensure scalability, particularly for a subjective task like captioning, where a single ground-truth is ill-defined. We have taken extensive measures to validate the quality, factual accuracy, and diversity of these semi-synthetic captions, as detailed in Section 3.2 and Appendix H, to mitigate the risk of propagating systemic biases from the oracle model. Our human-revisited test set is also an attempt to further ensure evaluation reliability. For our human evaluation studies, all participation was voluntary. We obtained informed consent from all participants, who were university students. The study's purpose was clearly communicated, and all responses were collected anonymously to protect participant privacy, as shown in an example of a consent form in Appendix O.

## Llm Usage Statement

Large Language Models played a central role in multiple stages of this work. 1. LLMs were employed as **data generators**, producing semi-synthetic captions that serve as ground truth references in CaTS-Bench.

2. LLMs were employed as **data extractors**, for example to parse statistical claims from captions during our evaluation analyses.

3. LLMs, more precisely VLMs, served as **baselines** in our experiments as captioning models for evaluation.

4. LLMs were employed as a **writing assist tool** to polish the presentation of the paper, while the authors retain full responsibility for all content.

## Reproducibility Statement

To ensure the reproducibility of our research, all components of our work will be made publicly available upon publication. 1. **Data:** The complete CaTS-Bench dataset, including the numeric time series, metadata, generated plots, oracle-generated and human-revisited captions, and the diagnostic Q&A suite, are released at https://huggingface.co/datasets/a9f3c7e2/CaTSBench.

2. **Code:** We will release the source code for the entire data curation pipeline, model finetuning scripts, and the evaluation suite. The code will be hosted in a public repository to allow for complete replication of our results and to facilitate future research.

3. **Models and Environment:** All open-source models used in our experiments are explicitly named with version details provided in Appendix E. For proprietary models, we specify the exact model endpoints used at the time of the experiments. Detailed finetuning hyperparameters and hardware specifications are documented in Appendix D.

4. **Evaluation:** Our evaluation protocol relies on standard, well-established linguistic metrics and novel metrics that are precisely defined in F. All prompts used for caption generation, quality verification, and LLM-based scoring are provided in Appendix N to ensure that our evaluation can be replicated consistently. Furthermore, we conducted a robustness check (Appendix H.5), which demonstrated minimal variance across multiple runs, confirming the stability of our results.

Importantly, LLMs did not contribute to research ideation or decision-making. All factual claims, analyses, and conclusions are the responsibility of the authors.

## References

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. Phi-4 technical report. *arXiv preprint arXiv:2412.08905*, 2024.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Abdul Fatir Ansari, Lorenzo Stella, Ali Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Bernie Wang. Chronos: Learning the language of time series. *Transactions on Machine Learning Research*, 2024. ISSN 2835-8856. URL https:
//openreview.net/forum?id=gerNCVqqtR. Expert Certification.

Anthropic. The claude 3 model family: Opus, sonnet, haiku. Model card, Anthropic, March 2024. URL https://www-cdn.anthropic.com/ de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf.

Saad Aziz. Population collapse. https://www.kaggle.com/datasets/saadaziz1985/
population-collapse, 1985. Accessed: 2025-05-01.

Anthony Bagnall, Hoang Anh Dau, Jason Lines, Michael Flynn, James Large, Aaron Bostrom, Paul Southam, and Eamonn Keogh. The uea multivariate time series classification archive, 2018. arXiv preprint arXiv:1811.00075, 2018.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. *arXiv preprint arXiv:2308.12966*, 2023.

Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, pp. 65–72, 2005.

Rui Cao and Qiao Wang. An evaluation of standard statistical models and llms on time series forecasting. *arXiv preprint arXiv:2408.04867*, 2024.

Ching Chang, Wen-Chih Peng, and Tien-Fu Chen. Llm4ts: Two-stage fine-tuning for time-series forecasting with pre-trained llms. *arXiv preprint arXiv:2308.08469*, 2023.

Georgios Chatzigeorgakidis, Konstantinos Lentzos, and Dimitrios Skoutas. Multicast: Zero-shot multivariate time series forecasting using llms. In 2024 IEEE 40th International Conference on Data Engineering Workshops (ICDEW), pp. 119–127. IEEE, 2024.

Daqing Chen. Online Retail. UCI Machine Learning Repository, 2015. https://doi.org/10.

24432/C5BW33.

Mouxiang Chen, Lefei Shen, Zhuo Li, Xiaoyun Joy Wang, Jianling Sun, and Chenghao Liu. Visionts:
Visual masked autoencoders are free-lunch zero-shot time series forecasters. arXiv preprint arXiv:2408.17253, 2024a.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. *arXiv preprint arXiv:2412.05271*, 2024b.

Lin Chin-Yew. Rouge: A package for automatic evaluation of summaries. In Proceedings of the Workshop on Text Summarization Branches Out, 2004, 2004.

Yanping Chen, Eamonn Keogh, Bing Hu, Nurjahan Begum, Anthony Bagnall, Abdullah Mueen, and Gustavo Batista. The ucr time series classification archive, July 2015. www.cs.ucr.edu/
~eamonn/time_series_data/.

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Kota Dohi, Aoi Ito, Harsh Purohit, Tomoya Nishida, Takashi Endo, and Yohei Kawaguchi. Domainindependent automatic generation of descriptive texts for time-series data. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2025.

Jiaxiang Dong, Haixu Wu, Yuxuan Wang, Li Zhang, Jianmin Wang, and Mingsheng Long.

Metadata matters for time series: Informative forecasting with transformers. arXiv preprint arXiv:2410.03806, 2024.

European Centre for Disease Prevention and Control. Download today's data on the geographic distribution of covid-19 cases worldwide. https://www.ecdc.europa.eu/en/publications-data/ download-todays-data-geographic-distribution-covid-19-cases-worldwide, 2024. Accessed: 2025-04-03.

Elizabeth Fons, Rachneet Kaur, Zhen Zeng, Soham Palande, Tucker Balch, Svitlana Vyetrenko, and Manuela Veloso. Tadacap: Time-series adaptive domain-aware captioning. In Proceedings of the 5th ACM International Conference on AI in Finance, pp. 54–62, 2024.

Food and Agriculture Organization of the United Nations. Faostat - food balance sheets. http:
//www.fao.org/faostat/en/\#data/FBS, 2024. Accessed: 2025-04-03.

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models, 2023. URL https://arxiv.org/ abs/2211.10435.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.

Yasser Hassan. Walmart dataset. https://www.kaggle.com/datasets/yasserh/
walmart-dataset, 2020. Accessed: 2025-04-03.

Abhishek S. Jha. Time series air quality data of india (2010–
2023). https://www.kaggle.com/datasets/abhisheksjha/ time-series-air-quality-data-of-india-2010-2023, 2023. Accessed:
2025-05-01.

Harsh Jhamtani and Taylor Berg-Kirkpatrick. Truth-conditional captioning of time series data. In EMNLP, 2021.

Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, and Qingsong Wen. Time-LLM: Time series forecasting by reprogramming large language models. In International Conference on Learning Representations
(ICLR), 2024.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Kai Kim, Howard Tsai, Rajat Sen, Abhimanyu Das, Zihao Zhou, Abhishek Tanpure, Mathew Luo, and Rose Yu. Multi-modal forecaster: Jointly predicting time series and textual data. arXiv preprint arXiv:2411.06735, 2024.

Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? *Advances in Neural Information Processing Systems*, 37:87874–87907, 2024.

Chen Liu, Shibo He, Qihang Zhou, Shizhong Li, and Wenchao Meng. Large language model guided knowledge distillation for time series anomaly detection. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, pp. 2162–2170, 2024a.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. SimCSE: Simple contrastive learning of sentence embeddings. In *Empirical Methods in Natural Language Processing (EMNLP)*, 2021.

Rakshitha Godahewa, Christoph Bergmeir, Geoffrey I. Webb, Rob J. Hyndman, and Pablo Montero-
Manso. Monash time series forecasting archive. In Neural Information Processing Systems Track on Datasets and Benchmarks, 2021.