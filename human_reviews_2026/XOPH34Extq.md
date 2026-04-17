# TabStruct: Measuring Structural Fidelity of Tabular Data

- Decision: Accept (Oral)
- Scores: 10, 4, 8, 6

## Abstract
Evaluating tabular generators remains a challenging problem, as the unique causal structural prior of heterogeneous tabular data does not lend itself to intuitive human inspection. Recent work has introduced structural fidelity as a tabular-specific evaluation dimension to assess whether synthetic data complies with the causal structures of real data. However, existing benchmarks often neglect the interplay between structural fidelity and conventional evaluation dimensions, thus failing to provide a holistic understanding of model performance. Moreover, they are typically limited to toy datasets, as quantifying existing structural fidelity metrics requires access to ground-truth causal structures, which are rarely available for real-world datasets. In this paper, we propose a novel evaluation framework that jointly considers structural fidelity and conventional evaluation dimensions. We introduce a new evaluation metric, global utility, which enables the assessment of structural fidelity even in the absence of ground-truth causal structures. In addition, we present TabStruct, a comprehensive evaluation benchmark offering large-scale quantitative analysis on 13 tabular generators from nine distinct categories, across 29 datasets. Our results demonstrate that global utility provides a task-independent, domain-agnostic lens for tabular generator performance. We release the TabStruct benchmark suite, including all datasets, evaluation pipelines, and raw results. Code is available at https://github.com/SilenceX12138/TabStruct.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper proposes a benchmarking framework including a novel metric for tabular data generators. The benchmarking framework is applied to 29 tabular SCM-based or real-world datasets and 13 generators.

The work has several contributions:
* A good motivation for introducing yet another benchmark for tabular data generators.
* A novel way of assessing the performance of data generators, based on structural fidelity, inspired by a causal perspective.
* A benchmarking framework with datasets, generators, and the evaluation pipeline. 
* Many insights into existing failure modes and strengths of tabular data generators. 
* An extreme commitment to reproducibility and scientific rigor in code and all parts of the paper. 
* A detailed and extended overview of related work for all parts of the work, from motivation to benchmark to insights.

### Strengths
The authors have executed all the above-mentioned contributions with an extremely high quality. The motivation and reasons for the benchmark are novel and original, and well related to prior work. The work seems highly significant, as it improves upon the benchmarking landscape for tabular data generators and may finally deliver actionable insights that allow us to identify good, usable generators for practical applications. The paper is written very clearly, and all information is easily accessible to the reader.

### Weaknesses
There are no notable weaknesses as far as I can tell. Some of my questions might allude to potential weaknesses, but nothing concrete enough to mention here.

### Questions
* Using SCM-based datasets to benchmark tabular data generators is well motivated. Yet, it is hard to tell if datasets, for example, about physical laws, are representative of real-world tabular data. Real tabular data is often noisier than datasets we have / can get from data with a ground-truth SCM. Thus, it is hard to predict how useful and representative the subset of SCM-based data will be for TabStruct in the future to guide generator development. Have the authors considered ways to get more such data or more realistic data? For example, priors of tabular foundation models such as TabPFN often use SCM-based data for pretraining that might be more realistic. Likewise, they usually add noise to the SCMs' data. 
* A core use of SCM-based data could be, as mentioned, to check for extrapolation instead of interpolation. This would also point to potentially benchmarking an entirely new kind of generators, that is, generators that can generate non-IID data. This would be akin to model benchmarks for non-IID data, e.g., https://arxiv.org/abs/2406.19380, rather than purely IID data, e.g., https://arxiv.org/abs/2506.16791. Given this comparison, TabStruct and the SCM-based data could enable benchmarks for non-IID data generators, and the Global CI would be appropriate for this. At the same time, Global utility might need to use non-IID models/validation splits. Was this intended by the distinction of interpolation vs extrapolation in Figure 1? And how does this framing position the current benchmark? 
* The current formulation of the Global utility metric utilizes a binary output of the CI test. While theoretically well motivated, I was wondering if a more continuous output of the CI test might not be better suited. For example, one could return the alpha value at which the test fails, rather than 0/1. Then, the "decision boundary" of the Global utility would be smoother. Moreover, it would be more similar to the smoothness of global utility (which averages over continuous values) and maybe provide a higher correlation. Or, one could use a similar significance test in global utility, such as a Wilcoxon test at alpha 0.01 to test if Perf(D_ref) is equal to Perf(D) over samples or bootstraps. 
* Why was balanced accuracy chosen as a metric for classification tasks? It is well established in model benchmarks that metrics such as ROC AUC for binary and log loss for multiclass classification are more appropriate for comparing models (e.g., see https://arxiv.org/abs/2506.16791 and its cited/related benchmarks and https://pages.cs.wisc.edu/~dpage/cs760/roc.pdf). I think here, a non-threshold-based metric would also be more appropriate. The impact of this choice might be less severe because AutoGluon was used, which, if optimized for balanced accuracy, should have employed threshold tuning by default (as required for evaluation for balanced accuracy). Do you know if threshold tuning was used in your experiments? On that note, the current formulation could also create some numerical problems for outliers where RMSE approaches 0. It might be worth improving the implementation with safeguards against numerical issues when computing relative metrics. 
* The description of nested validation in Line 314 is very hard to parse, in my opinion. The appendix and text make it clear enough later. But the work could benefit from shortly introducing the validation method in the first sentence of the paragraph. I think in this case, a nested repeated shuffle split would be one way to describe it (?).  Furthermore, is stratification used for splits for classification datasets? If not, this might be important for imbalanced classification datasets in the future. 
* How does the framework control randomness and seeds of the methods, data splits, and tuning?
* The correlation analysis in Section 4.1 is nice and insightful, but it remains unclear if the correlation is high enough to be representative enough for a good benchmark. There might be ways to test or verify that the correlation is sufficiently high by checking if model rankings are also correlated well enough -- especially for generators that have a very high, similar performance. Have you looked into such experiments next to the results in the appendix (Figure 6 etc)? The answer might depend on how valuable the small model difference might be, or if preserving the ranking of models is required. The danger is that developers might start to optimize for a potentially noisy metric when comparing only the best models or marginal improvements over the best models. 
* The preprocessing as described in Appendix D.2 might be very suboptimal for some of the ML models from AutoGluon (as they expect to do their own model-specific preprocessing). But this make-everything-continuous preprocessing is needed for many of the generators, correct?  
* As a final question / side note, a public, easy-to-share leaderboard of the methods might be cool, such as a Hugging Face leaderboard.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the problem of evaluating tabular data generators. The paper argues that the existing benchmarks are insufficient because they either neglect the structural properties of the tabular data, or are biased towards downstream task performances, and are often limited to toy examples where the causal graph is known.

The paper proposes to incorporate structural fidelity as a core evaluation dimension, and introduces a metric called, "global utility", which is an SCM-free metric for assessing structural fidelity. A comprehensive benchamark " TabStruct" is proposed to evaluate 13 generators across 29 datasets including real-world and SCM-based data.

### Strengths
+ the proposed problem is highly relevant. The paper identified a significant gap in the evaluation of tabular generative models and moves beyond simple ML efficacy (performance on downstream tasks) and density estimation
+ the proposed metric is novel (global utility) and interesting. 
+ the empirical validation is rigorous and comprehensive with 13 generators from 9 categories and 29 datasets. It is also commendable that the authors chose some challenging datasets where the models don't easily achieve perfect scores.
+ Tabstruct library is a valuable open-source contribution

### Weaknesses
- The global utility metric is a heuristic proxy, not a formal measure. Although the paper provides strong empirical evidence it is important to acknowledge that it is a proxy. Although it is acknowledged in the paper, it should probably be emphasized more. 
- The metric measures "predictability" rather than the "causal structure". A generative model could learn powerful but spurious correlations that allow for excellent cross-prediction of variables. This would make it score high on global utility, but be structurally unfaithful.
- The "full-tuned" configuration of the global utility metric requires training an ensemble of predcitors for each of the columns in the dataset. That is computationally expensive, especially for high-dimension data. Although the "tiny-default" is much faster and seems to be stable ranking wise, the fundamental approach is very expensive and this could lead to scalablity issues in practice.
- The finding that the BAyesian networks and GOGGLE perform poorly on structural fidelity metrics is counter-intuitive. Although there is some explanation in the paper, this should probably be probed more deeply. Where is the failure happening?

### Questions
1. Could you elaborate on the theoretical gap between high global utility and the preservation of the Markov equivalence class? Are you aware of any hypothetical data-generating processes where a model could achieve high global utility while  violating key conditional independencies? How likely is this to happen in practice?
2. The high correlation between the global utility and the global_CI is taken as the validation of the global utility; but, the global_CI score is an imperfect ground truth because  the CI test have limitations themselves. So how does this inherent noise and potential inaccuracies of the CI-based ground truth affect your confidence in the correlation values and the conclusions you draw from it?
3. global utility is normalized by the performance on the reference data. How does this metric behave in low-data regimes? What is the relationship between the reliability of the global utility and the size and quality of the reference dataset?
. How does the computational cost of global utility scale with the number of features?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a comprehensive benchmark for evaluating tabular generative models. It identifies a key gap in existing evaluation measures and proposes new metrics to address this issue. While most evaluation metrics focus on dimensions such as density estimation, privacy preservation, or machine learning efficiency, these metrics are not tailored to assess whether synthetic data preserves the underlying structural (causal) relationships among variables. To address this, the paper proposes a new evaluation dimension called structural fidelity, and a new metric, global CI, to measure this new dimension.  Because, this metric requires knowledge about the underlying structural causal model (SCM), which is almost always unknown for real datasets, the paper also introduces a SCM free heuristic metric, global utility, which correlates strongly with global CI and can be used as a proxy to measure structural fidelity.  

The paper performs extensive evaluations using 13 tabular generators across 29 datasets (including both expert-validated causal and real-world datasets) and provides many interesting insights about the performance of popular tabular generators and of the evaluation metrics.

### Strengths
The paper addresses the very important (and difficult) problem of evaluating the performance of tabular data generators.

It identifies a blind spot (structural fidelity) in the current evaluation metrics and proposes new metrics to measure it.

The experimental setup is as rigorous as it gets.

The very careful and detailed analyses of both the different generators and the different metrics provide corroboration to many observations in previous studies, as well as several new and valuable insights. 

The paper provides a very thorough discussion of the limitations of the proposed metrics. 

The framework’s modular design allows researchers to easily plug in new datasets, generators, and metrics, making it a great resource to the synthetic tabular data generation community.

### Weaknesses
The one point I think is missing from the paper is a comparison against the detection test metric (C2ST), which measures data fidelity by evaluating the discriminating ability of classifiers trained to discriminate between synthetic and real data. C2ST is a very popular metric, widely used in the field, which also aims to measure fidelity at a full dataset scale.  

Given the large scale of the experiments presented on the paper, I understand that it would likely be difficult to include extensive comparisons during the short time frame of the discussion window. However, I think the paper would benefit from providing at least some preliminary comparisons of C2ST against the proposed global utility metric.

### Questions
Some of the benchmark datasets are very large (e.g., Higgs, and SCM datasets). How did the paper handle the TabPFN model in situations where the evaluation dataset had more than 10,000 rows? Was TabPFN removed from the predictor’s ensemble in these cases? (My understanding is that the current version of TabPFN can only handle datasets with up to 10,000 rows.)

This is only a suggestion, that the paper might find useful for future work. In Table 2, in addition to the 13 generators, the paper includes the reference (training) data, $D_{ref}$, for direct evaluation. This helps better ground the interpretation of the scores achieved by the generators, showing that the metrics can distinguish between high- and low-quality data.  But, perhaps, another interesting comparison would be to include direct comparisons not only to the reference set but also to a separate independent set (of the same size as the reference set) from each dataset (which is not touched during training). Since the independent set, $D_{indep}$, is by construction independent and identically distributed to the reference set, it could be used to estimate the performance of an ideal generator, truly capable of generating independent samples from the same distribution as the reference data. 

Having these comparisons would potentially provide complementary information for the $D_{ref}$ direct evaluations. For instance, in the case of privacy metric such as the DCR, while $D_{ref}$ will produce a score equal to zero, $D_{indep}$ would estimate the median DCR score we would expect to see in an ideal generator. This would represent a baseline value and sort of “ground truth” value the generators should be aiming to. (While low DCR values indicate low privacy, high DCR scores are not necessarily a good thing either, since they usually indicate low data fidelity.) Additionally, the direct evaluation of $D_{indep}$ could also be potentially used to check when models are generating data that is too close to the training set (e.g., when a model is generating data $D_{syn}$ with better fidelity scores than $D_{indep}$). 

To generate this independent set the paper would need to change its current data split and re-split the full dataset of each benchmark into a test set, a validation set, a training (reference) set, and a independent set of the same size as the training set. While this would require re-running all the analyses in that paper (what is certainly not feasible during the short discussion phase) perhaps the paper might want to consider implementing these comparisons in the future, as they might provide additional insights (and the paper aims for TabStruct to be an ongoing effort aiming to continue to evolve). 

Minor suggestions:

Line 1515: “TarStruct” -> “TabStruct”

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a benchmark framework to assess methods for generating synthetic data based on existing datasets. The main novelty is that it introduces a metric to measure structural fidelity based on a global utility score without the need for ground-truth causal graphs. This enables the study of structural fidelity for real-world data where such graphs are typically unavailable. In empirical experiments, they first demonstrate a high correlation between the new metric and existing metrics on synthetic tasks (where the ground truth is available) and then assess the performance of different generators on real-world tasks.

### Strengths
**Novelty** The paper introduces a new metric that complements the assessment of data generators on real-world tasks

**Clarity** I enjoyed reading the paper as it is straightforward, with a good structure

**Impact** Studying tabular generators is a very relevant task for a broader audience at ICLR and a much-needed research direction directly related to developing better foundation models for tabular data

### Weaknesses
**Aggregation of Metrics** I understand the importance of aggregating metrics to provide concise and interpretable results. However, a considerable amount of information may be lost in the aggregation process. It would be valuable to include some indication of the variability or distribution of utility scores across features. For instance, are there notable outliers where the generator fails to capture feature usefulness, or are all features consistently represented? Such an analysis could strengthen the empirical insights.

**Readability** (This did not influence my rating, as it is straightforward to address.)
The paper employs multiple text styles (boldface, italics, color) without a clear or consistent rationale. For example, in lines 117-131, it is unclear why certain words are boldfaced. Similarly, the colored text and structure of (the huge) Table 2 do sufficiently aid visual comprehension or highlight the key messages. For example, consistent colors for the Top-3 and Bottom-3 results would help visualize similarities across columns. Moreover, showing real-world and SCM results for each metric side-by-side would support following the content of Section 4.2.

**Ethics Statement** This is not an ethics statement but a summary/conclusion.

### Questions
[clarification] Would reaching a utility score >1 be possible?

[clarification] What is the size of the generated dataset D_synth with respect to D_ref? I assume they are the same size, but it might also be interesting to analyze how the proposed metrics behave for larger or smaller synthetic datasets relative to the reference data.

### Soundness
3

### Presentation
2

### Contribution
3
