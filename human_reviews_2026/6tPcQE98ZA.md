# Benchmark Datasets for Lead-Lag Forecasting on Social Platforms

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Social and collaborative platforms emit multivariate time-series traces in which early interactions—such as views, likes, or downloads—are followed, sometimes months or years later, by higher impact like citations, sales, or reviews. We formalize this setting as Lead-Lag Forecasting (LLF): given an early usage channel (the lead), predict a correlated but temporally shifted outcome channel (the lag). Despite the ubiquity of such patterns, LLF has not been treated as a unified forecasting problem within the time-series community, largely due to the absence of standardized datasets. To anchor research in LLF, here we present two high-volume benchmark datasets—arXiv (downloads → citations of 500K papers) and GitHub (pushes → stars/forks of 1.4M repositories)—and outline additional domains with analogous lead–lag dynamics, including Wikipedia (page-views → edits), Spotify (streams → concert attendance), e-commerce (click-throughs → purchases), and LinkedIn profile (views → messages). We documented all technical details of data curation and cleaning, verified the presence of lead-lag dynamics through statistical and classification tests, and benchmarked deep learning architectures alongside parametric and non-parametric baselines for regression. Our study establishes LLF as a novel forecasting paradigm and lays an empirical foundation for its systematic exploration in social and usage data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel task termed lead-lag forecasting (LLF) in multivariate time-series analysis. The key objective of LLF is to predict a correlated but temporally shifted outcome channel (the lag) based on observations from an early usage channel (the lead). To facilitate research in this direction, the authors construct two large-scale datasets specifically designed to support model training and evaluation on the LLF task. Furthermore, they conduct comprehensive empirical analyses, benchmarking a diverse set of deep learning, parametric, and non-parametric prediction models.

### Strengths
**S1**. The paper formally defines a new task, contributing a novel perspective to multivariate time-series forecasting.

**S2**. The datasets collected for this task are large-scale and well-constructed.

**S3**. The authors provide a comprehensive analysis of the datasets, offering valuable insights into their characteristics.

### Weaknesses
**W1**. The overall task setup is highly similar to **popularity prediction**, as both aim to forecast future trends based on early observations. To improve the paper’s contribution, the authors should provide a clearer and more rigorous definition of the lead-lag concept. Specifically, lead-lag could be characterized as the temporal ordering of behavioral changes between correlated signals, such as an increase in access volume typically preceding a subsequent increase in citation volume.  Moreover, the paper should better articulate the significance and broader applicability of this new task—what practical or theoretical insights does LLF bring to other domains?

**W2**. It is unclear why the study uses a fixed five-year prediction window instead of exploring shorter or longer forecasting horizons, which could offer greater temporal insight and practical value.

**W3**. Since the work is presented as a benchmark study, it would be beneficial to include comparisons with a wider range of existing SOTA methods to provide a more comprehensive evaluation.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Lead-Lag Forecasting (LLF) as a formalized prediction task, focused on predicting long-term outcomes from early, correlated signals. The authors present two large-scale benchmark datasets derived from arXiv and GitHub to support this task. These datasets facilitate the prediction of long-term metrics (citations and forks) from short-term engagement signals (accesses, pushes, and stars) over a five-year horizon, using an initial observation window of 30 to 365 days. The paper provides a thorough statistical analysis of the lead-lag dynamics within these datasets and establishes performance benchmarks using a range of standard machine learning and deep learning models.

### Strengths
1. *Valuable and Novel Datasets*: The paper releases two large-scale, novel datasets for a well-defined and relevant forecasting problem. The scale (millions of papers and repositories) and the long-term nature seem very useful.
2. *Clear Problem Formulation*: The concept of Lead-Lag Forecasting is articulated clearly, providing a solid framework for future work. The paper is well-written and easy to follow.
3. *Comprehensive Benchmarking*: The authors have made a significant effort to benchmark the datasets with a broad set of standard baselines, ranging from simple regressions to Transformer-based models. This provides a useful starting point for researchers looking to develop new methods.
4. *High Transparency*: The data curation process is described in detail, and the hyperparameters for the baseline models are transparently reported, which is crucial for reproducibility and building upon this work.

### Weaknesses
1. *Missing citation*: The paper fails to acknowledge that previous work has already investigated the influence of earlier accesses of arXiv papers on the citation count (Tim Brody, Stevan Harnad, Leslie Carr (2006): "*Earlier Web usage statistics as predictors of later citation impact*").
2. *Limited new insights*: As a dataset paper, the primary contribution is not methodological, so it is not stricly required that the paper introduces any new modelling methodology. However, it would be useful if the paper adds more ablations and explores potential failure modes or performance variations of the baseline models. 

    For example:
- Do the models perform uniformly across different arXiv subject areas (e.g., computer science vs. physics)?
- For GitHub, does prediction accuracy for forks vary with factors like the popularity of the repository? A more fine-grained analysis would add significant value to the benchmark.

### Questions
1. To maximize the utility of this work for the community and ensure reproducibility, do you plan to release the source code for running the baseline models and generating the benchmark results?
2. The datasets span several years during which platform dynamics may have changed (e.g., arXiv's HTML preview was introduced and GitHub's user interface and features like "trending"). Did you investigate whether the lead-lag relationships are stable over time? For example, is the predictive power of early accesses on citations the same for a paper from 2014 as for one from 2018?
3. You mention removing "bots" from the arXiv access logs. Could you elaborate on how these thresholds were determined and whether a sensitivity analysis was performed? It's possible that legitimate, highly active researchers or institutional proxies could be misclassified, potentially skewing the data for high-interest fields.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a Lead-Lag Forecasting problem, a prediction task that uses early "lead" signals (e.g., views, downloads) to forecast delayed "lag" outcomes (e.g., citations, forks), and presents two large-scale benchmark datasets: Arxiv and GitHub. The study considers two task types: classification and regression. Statistical analysis confirms strong correlations between early lead signals and long-term outcomes, and baseline experiments with methods demonstrate the performance improving as lookback windows increase.

### Strengths
S1. An interesting prediction question is proposed, and two large-scale datasets are provided.

S2. The statistical analysis is provided to confirm the relations between the lead and the lag.

### Weaknesses
W1. I am not quite convinced by the problem setting. Why is the lag signal not included in the lead signals? The previous signals of the lag are very likely available in both the ArXiv and GitHub scenarios. With such inclusion, several related works have been previously proposed, such as [R1].

W2. The proposed datasets might be so simple that Linear Regression can achieve great performance. It casts doubt on whether the relation between the lead and the lag is complex enough for further discovery.

W3. Two datasets are not enough to fully evaluate a method's capabilities on the question. More datasets in real-world applications might help, such as recommendation scenarios.

[R1] Entire Space Multi-Task Modeling via Post-Click Behavior Decomposition for Conversion Rate Prediction.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces Lead-Lag Forecasting (LLF) as a novel time series forecasting problem, where the objective is to predict long-term "lag" outcomes based on early "lead" indicators from different channels on digital platforms. 

To support research in LLF, the authors construct and release two large-scale benchmark datasets from arXiv (accesses → citations) and GitHub (pushes/stars → forks), with up to 5-year forecasting horizons.

### Strengths
Clearly introduces and formalizes Lead-Lag Forecasting (LLF), which fills a significant gap in time-series research.

Curates two valuable datasets (arXiv, GitHub) with careful preprocessing, long-range horizons, and minimal survivorship bias.

### Weaknesses
The work mostly benchmarks existing models and does not propose new LLF-specific architectures or techniques.

Although included, the use of the Time-MoE foundation model does not yield significant gains, and its contribution appears limited or inconclusive.

### Questions
Could the authors provide case studies or explainability tools to better understand prediction errors, particularly for high-impact cases?

### Soundness
3

### Presentation
3

### Contribution
2
