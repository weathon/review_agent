# StarEmbed: Benchmarking Time Series Foundation Models on Astronomical Observations of Variable Stars

- Decision: Reject
- Scores: 4, 4, 6, 0

## Abstract
Time series foundation models (TSFMs) are increasingly being adopted as highly-capable general-purpose time series representation learners. Although their training corpora are vast, they exclude astronomical time series data. Observations of stars produce peta-scale time series with unique challenges including irregular sampling and heteroskedasticity. We introduce $\texttt{StarEmbed}$, the first public benchmark for rigorous and standardized evaluation of state-of-the-art TSFMs on stellar time series observations (``light curves''). We benchmark on three scientifically-motivated downstream tasks: unsupervised clustering, supervised classification, and out-of-distribution source detection. $\texttt{StarEmbed}$ integrates a catalog of expert-vetted labels with multi-variate light curves from the Zwicky Transient Facility, yielding $\sim$40k hand-labeled light curves spread across seven astrophysical classes. We evaluate the zero-shot representation capabilities of three TSFMs ($\texttt{Moirai}$, $\texttt{Chronos}$, $\texttt{Chronos-Bolt}$) and a domain-specific transformer ($\texttt{Astromer}$) against handcrafted feature extraction, the long-standing baseline in the astrophysics literature. Our results demonstrate that these TSFMs, especially the $\texttt{Chronos}$ models, which are trained on data completely unlike the astronomical observations, can outperform established astrophysics-specific baselines in some tasks and effectively generalize to entirely new data. In particular, TSFMs deliver state-of-the-art performance on our out-of-distribution source detection benchmark. With the first benchmark of TSFMs on astronomical time series data, we test the limits of their generalization and motivate a paradigm shift in time-domain astronomy from using task-specific, fully supervised pipelines toward adopting generic foundation model representations for the analysis of peta-scale datasets from forthcoming observatories.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper evaluates the performance of time series foundation models in astronomy science. The authors introduce a new benchmark dataset (StarEmbed) comprising over 40,000 hand-labeled light curves spanning seven astrophysical classes. On this new dataset, the authors conducted unsupervised clustering, supervised classification, and out-of-distribution source detection tests. The results demonstrate that although these TSFMs were not trained on astronomical observation data, they can outperform existing astrophysics-specific models in certain tasks. Particularly noteworthy is the exceptional performance of TSFMs in out-of-distribution detection, where they significantly surpass domain-specific models. The authors emphasize that these experimental findings are driving a paradigm shift in astronomy data analysis.

### Strengths
- The paper is well written and easy to follow.
- It presents a qualified benchmark study: introducing a new dataset which is very important for time series research, performing diverse experiments on it, and revealing several noteworthy experimental results.

### Weaknesses
- The paper's technical contribution is limited.

### Questions
- StarEmbed is a time series classification benchmark. Why did the authors choose to test Chronos and Moirai, which are designed for forecasting, rather than specialized time series classification foundation models in the paper?

### Soundness
2

### Presentation
3

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
This paper introduces StarEmbed, a benchmark designed to evaluate state-of-the-art time series models on stellar time series data. The authors assess the models on three tasks: unsupervised clustering, supervised classification, and out-of-distribution source detection. The results show that Chronos models perform particularly well, especially in the out-of-distribution source detection task.

### Strengths
1. A benchmark is always a good contribution to the community to encourage and push the frontier of models and comparing on a far basis the algorithm

2. The authors make an effort on the experimental side to perform host of experiments.

### Weaknesses
1. The paper's contribution feels limited. It mainly introduces a benchmark and tests existing models, without offering a novel architecture, fine-tuning, or a new foundational model based on Chronos. A more substantial contribution, such as a new model or an innovative approach (e.g., boosting/bagging), would make the work more impactful and valuable to the community. Since Chronos is already well-known, testing it on the benchmark alone doesn't constitute a major contribution. While introducing the benchmark is still valuable, it’s not enough on its own to make the paper stand out.

2. If the focus is on the benchmark, more effort should have gone into its statistical analysis. This includes examining the distribution across samples, channels, and time, studying distribution shifts, and analyzing the train/test split to understand the benchmark’s difficulty and challenges. Frequency statistics and two-dimensional visualizations could help illustrate the benchmark's significance. Additionally, including more baseline models for comparison such as Toto, Moment, or PatchTST would strengthen the paper’s contribution and provide a more comprehensive evaluation.

### Questions
1. Did the authors perform any statistical analysis of the benchmark, such as examining the distribution across samples, time, and channels? It would also be useful to explore frequency diversity, wavelet decomposition, and PCA/t-SNE representations to better understand the data.

2. How is the train/test split performed, and what unique challenges does it present? Is there meaningful signal in the benchmark, or is it just noise? The authors should discuss whether the benchmark contains realistic, non-trivial information and offer insights into its significance.

3. It would be beneficial to test the benchmark on other foundational models from the literature, as well as models commonly used in time series forecasting, to provide a more comprehensive evaluation.

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
4

### Summary
This paper introduces a benchmark for evaluating time series foundation models (TSFMs) on astronomical light curves from periodic variable stars. The authors curate ~40k labeled light curves from ZTF across seven astrophysical classes and evaluate several TSFMs, a domain-specific model (Astromer), and hand-crafted features on three tasks: unsupervised clustering, supervised classification, and out-of-distribution (OOD) detection.

### Strengths
1. Novel scientific benchmark -- scientific datasets have nonstandard and interesting properties (as highlighted in an intro figure -- heteroskedacticity, irregularity, etc), and are a good stress test of ML models. The paper has a nice introduction of astro time series data for non-experts.
2. Overall a careful, comprehensive benchmark -- The three-task evaluation framework (clustering, classification, OOD detection) provides a thorough assessment of model capabilities of different types. The use of expert labels, careful data cleaning and stratified splitting procedures are well-designed.
3. Good reproducibility / data availability (expected for datasets and benchmarks track paper)
4. Findings are interesting, in particular the generalizability of general-purpose TSFMs to astro data.

### Weaknesses
1. Literature review could be a bit more thorough, and the paper put in broader astro and time series FM context. E.g., https://arxiv.org/abs/2504.20290, https://arxiv.org/abs/2408.16829 and https://arxiv.org/abs/2405.17156 are some relevant papers, not necessarily directly on transients, but nevertheless applicable to supernova time series science. https://arxiv.org/abs/2405.13867 is relevant as well.
2. As far as I can tell there is only zero shot performance. One/few shot results would be extremely relevant given foundation model context.
3. Stats rigour -- only 3 seeds for non-deterministic methods seems insufficient, confidence intervals not universal
4. Comparison mostly at the performance level. For astro, speed is often paramount (in particular for transient science

### Questions
1. Can you provide any analysis of which hand-crafted features drive the strong baseline performance? Could help understand what TSFMs might be missing.
2. Would you expect results to change substantively in the 1/few shot setting?
3. Are general purposed TSFMs more costly/slower than astro specific ones?
4. Overall, it's hard to tell how comprehensive the comparison to astro methods is. There are lots of other methods (e.g. template fitting, other NN based methods that are referenced but not compared). I would be curious to what extent some of these are relevant or not.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper introduces an astronomy-focused benchmark for time series classification and regression. It does not propose new modeling methodology, but rather evaluates existing domain-specific deep time series models (e.g., Astromer, Chronos). The empirical finding that hand-crafted features outperform deep models across most tasks is interesting, and highlights a real and current limitation of existing “foundation” time series models in astronomy.

However, the paper stops at this observation. It does not provide a clear hypothesis, insight, or direction on why this is the case or how future time series models might be improved. As a result, the contribution feels limited to benchmarking — the results themselves are somewhat inconclusive, and the broader impact beyond the domain remains unclear.

### Strengths
* The paper is clearly compiled with in-depth knowledge of astronomy and the current state-of-the-art in deep learning models for astronomical time series data
* The experiments seem to span a variety of astronomic time series tasks
* The paper is very extensive, also including the appendix analyses. It clearly captures a large amount of work, which I think it would be better appreciated (and reviewed with more domain knowledge) in a domain-specific journal rather than ICLR.

### Weaknesses
* inconclusive results: hand-crafted features are best across most tasks, but no hypothesis is given how to improve the current models. Also no model or training algorithm is proposed
* the paper seems to miss a body of work around the time series classification and benchmarking community, mainly the UCR archive and the time series models benchmarked on these very diverse tasks. Including these non-deep learning models as comparisons may fill a gap between underperforming deep models and the very well performing hand-crafted features

### Questions
* Beyond deep time series classification, how applicable are the models tested in the time series classification community that are benchmarked e.g., on the UCR Time Series Classification Archive (https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/) for these tasks? Given that hand-crafted features are performing well, these non-deep learning models from UCR may also be fairly competitive on the benchmarks 
* Would state-space-models like Mamba also be applicable in this area of applications?

### Soundness
2

### Presentation
3

### Contribution
1
