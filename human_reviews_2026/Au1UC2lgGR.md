# What is the right direction for time series anomaly detection benchmarking: evidence from evaluation of linear models

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 4, 2, 4

## Abstract
Time series anomaly detection (TSAD) progress has been accompanied by a persistent increase in architectural sophistication. In this work, we revisit this trend and demonstrate that a simple score based on a closed-form solution for an ordinary least squares (OLS) regression model outperforms state-of-the-art deep learning baselines. Through extensive evaluation on both univariate and multivariate TSAD benchmarks, we show that linear regression achieves superior accuracy and robustness while requiring orders of magnitude fewer resources. 
Our further analysis identifies the types of anomalies that can and cannot be reliably captured by linear models, providing insights into their strengths and limitations. Overall findings indicate that current benchmarkings would benefit from inclusion of simple methods as well as more intricate problems that would do require deep learning-based solutions. Thus, future research should consistently include strong linear baselines and, more importantly, develop new benchmarks with richer temporal structures pinpointing the advantages of deep learning models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper argues for including strong linear baselines in time-series anomaly detection evaluations and for developing newer benchmarks with richer, more complex interactions to expose where linear methods may fail.

### Strengths
I enjoyed reading this paper. It starts from a clear hypothesis: in some cases we don’t need complex deep-learning methods for time-series anomaly detection. The authors then show that plain linear regression can outperform recent deep models on several benchmarks. The mathematical treatment—covering linear regression and the reduced-rank variant used here—is helpful, as is the discussion in Section 3.3.

The paper uses 16 baseline methods as comparators.  Furthermore, it uses 5 univariate benchmarks and 4 multi-variate benchmarks.   

I appreciate that the paper also identifies that both rank (in reduced rank regression setup) and the width of the temporal window determines the overall performance.

### Weaknesses
The paper side-steps the issue of threshold in anomaly detection.  How would one select this threshold?

### Questions
I found Section 3.4 hard to follow. It appears aimed at characterizing which anomalies the linear-regression approach can detect, but the answer isn’t fully clear. It’s nice to see that the Gaussian-process (GP) anomaly score is captured by linear regression under a finite-history assumption; however, it remains unclear which anomaly types are detectable under the GP-based score. Could the authors clarify this mapping and provide concrete examples?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper revisits the effectiveness of simple linear models for time series anomaly detection and demonstrates that OLS regression can outperform several state-of-the-art deep learning baselines. The authors provide both empirical evidence and theoretical justification for this observation, emphasizing the importance of including OLS linear estimators as strong baselines in future TSAD benchmarking efforts.

### Strengths
* The paper addresses a timely and important question in the TSAD community, where increasingly complex neural architectures are often proposed without systematic comparison to simple, rigorous baselines.
* The findings are intriguing and supported by theoretical insights, highlighting the competitiveness of closed-form OLS estimators relative to gradient-based optimization methods.

### Weaknesses
* Incomplete evaluation suite and limited linear baselines.
* Generality of the theoretical argument.
* Scalability and hyperparameter concerns.

Please find the detailed comments in the following section.

### Questions
* Can the derivation in Section 3.4 be generalized to multivariate or non-separable kernels? Would cross-correlated channels require additional assumptions on the covariance structure? How robust is the linear equivalence under nonstationary processes commonly found in real-world scenarios?
* Better elaboration of evaluation measures is needed. Are F1 scores in the tables computed point-adjusted techniques? The threshold-independent evaluation measures such as AUC-PR and VUS-PR [1] should also be considered.
* The findings that close-form solution outperforms graident-based methods are interesting. Stronger modern linear baselines such as DLinear [2] and MLPMixer [3] should be included for a more rigorous comparison.
* The paper would benefit from a more clear illustration of runtime performance of OLS, especially in the case of high-dimensional input and its scalability to its graident-based solution counterpart. 
* The model’s dependence on window length (as illustrated in Figure 2) suggests that performance is highly sensitive to this hyperparameter. However, it is unclear how window size were selected for Tables 1–2 for OLS and baselines.
* How did the authors divide the datasets according to different types of anomalies? As many existing TSAD datasets are known to be homogeneous or label-flawed, the paper would also benefit from evaluating on a curated and heterogeneous benchmark such as TSB-AD [4].
* Typo in line 118-119: 'patchifies thef requency'.


[1] Paparrizos, John, et al. "Volume under the surface: a new accuracy evaluation measure for time-series anomaly detection." Proceedings of the VLDB Endowment 15.11 (2022): 2774-2787.

[2] Zeng, Ailing, et al. "Are transformers effective for time series forecasting?." Proceedings of the AAAI conference on artificial intelligence. Vol. 37. No. 9. 2023.

[3] Tolstikhin, Ilya O., et al. "Mlp-mixer: An all-mlp architecture for vision." Advances in neural information processing systems 34 (2021): 24261-24272.

[4] Liu, Qinghua, and John Paparrizos. "The elephant in the room: Towards a reliable time-series anomaly detection benchmark." Advances in Neural Information Processing Systems 37 (2024): 108231-108261.

### Soundness
2

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
5

### Summary
The paper revisits time-series anomaly detection (TSAD) benchmarks and demonstrates that a simple Ordinary Least Squares (OLS) linear regression model with lagged inputs can outperform many state-of-the-art deep learning models across diverse univariate and multivariate datasets. The authors provide a theoretical justification linking OLS to Gaussian Process conditional density estimation, evaluate the approach on major TSAD benchmarks, and emphasize the need for stronger linear baselines and richer benchmark datasets. They further show that Reduced-Rank Regression (RRR) enhances performance in multivariate scenarios.

### Strengths
1. The OLS-based baseline outperforms numerous deep TSAD methods across multiple benchmarks.  
2. Evaluations cover both univariate and multivariate settings, employing event-level metrics for comprehensive analysis.  
3. The paper connects OLS with the Gaussian Process conditional density perspective for anomaly scoring.

### Weaknesses
1. The paper’s most significant weakness is its contribution: its core claim—that simple linear models perform strongly on current TSAD tasks, indicating the need for better datasets and evaluation protocols—was already articulated in 2020 (published in TKDE 2021) by Renjie Wu and Eamonn J. Keogh, “Current Time Series Anomaly Detection Benchmarks Are Flawed and Are Creating the Illusion of Progress.” Despite citing this work, the paper still evaluates on the very datasets flagged as flawed. Moreover, NeurIPS 2024 introduced an improved benchmark—Qinghua Liu and John Paparrizos, “The Elephant in the Room: Towards a Reliable Time-Series Anomaly Detection Benchmark”—which the paper neither adopts nor compares against, further weakening its claim to novelty.
2. Many of the datasets used are known to favor simpler models. Despite this stated motivation, no new challenging benchmark is introduced.
3. Design choices (e.g., lag size, ridge-term sensitivity) are mentioned only briefly, with no systematic ablation or robustness evaluation.
4. The theoretical analysis primarily supports linearity within a finite-history Gaussian Process framework, leaving unclear how well it extends to the nonlinear anomaly regimes commonly observed in real-world applications.

### Questions
1. Can the authors provide stronger empirical evidence using synthetic benchmarks that favor nonlinear anomalies? (Currently, most anomalies appear linear-friendly.)  
2. How sensitive is the OLS model to lag window and other hyperparameters across datasets? The paper briefly mentions tuning, but lacks a systematic analysis.  
3. Would kernelized linear models (e.g., kernel ridge regression) help bridge cases where deep models still outperform—such as those with seasonal or nonlinear trend patterns?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates the performance of ordinary least squares (OLS) regression and reduced rank regression (RRR) models applied as a forecaster on a lag window to solve the time series anomaly detection (TSAD) task. The point of the paper is that such a simple model is most of the time more reliable and performs better than other more complicated methods.

The paper first sets some background and provides a justification of the choice of linear models with the argument that Gaussian processes adjusted for the case of TSAD (discretized and with a kernel restricted to past events), have an expected next timestep value which linearly depends on past values and gets optimal values on the solution of the corresponding OLS problem.

In the experiments section, multiple recent methods are compared with the OLS model on univariate and multivariate time series. In the majority of the cases OLS and RRR perform the best. Additionally, some ablation is performed on the window size and rank on the two methods, a speed comparison with the other methods and a scoring split on different categories of anomalies.

### Strengths
- The theoretical justification using the Gaussian process is a nice addition to the paper. Though this type of models is still not realistic enough to capture different real-life time series behavior, it is still quite a broad class of models and it is interesting to know their forecasting behaves and can be learned like a linear model.

- The exposition and paper structure is quite clean and reads well.

- The choice of experiments is also quite good in the sense that both uni- and multi-variate series are covered and there are ablations on hyper parameters like the window width. The split and evaluation on different anomaly types is also interesting.

### Weaknesses
- There are some hyper parameters on which both methods depend on and would have high impact on scoring, but it is never mentioned what exact values they are set to and whether they vary per dataset. For example the window size (there is already an ablation on it where one can see it has an impact), the rank on RRR and there is also the difference order which briefly appears on figure 2 but is never mentioned or defined, though it can have significant impact e.g. on PSM and SWAT. Actually the values on table 2 seem to be the optimal scores appearing on figure 2.

- There are some discrepancies between table 2 and figure two. Specifically the RRR score on SMAP is 0.7719 on table 2 , while the maximum value it achieves on figure 2 is around 0.716. Why is this the case? What is the connection between the two?

- The paper argues about the usage of simple linear models in TAD as an alternative/criticism to existing complicated models. This is not a new topic and there are already publications which study and benchmark such models. Just to name a couple: "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark" introduces large benchmarking datasets and concludes that simple methods like PCA (practically linear), POLY and others perform best. "Position: Quo Vadis, Unsupervised Time Series Anomaly Detection?" also studies such simple models also including PCA, simple thresholding and argues for the empirical linear separability of anomalies on different datasets. It is a bit weird that the topic is presented as novel and that none of those simple strong baselines are not included in the scoring.

- Though there is a brief mention on the different metrics used, event F1 score and F1 k-delay, there are three metrics present on the evaluation tables which are not explicitly mapped to those two mentioned metrics and never defined:  "F1" "B-F-5" "E-F-5". Given the long history of differences between metrics and flawed evaluation methods, it is expected from the authors to provide a very clear definition of the metrics used.

### Questions
- On lines 198 and 207: The notation $\{(x_i = i, y_i)\}^T_{i=1}$, is a bit non-conventional and confusing, especially given that the $x_i$ name was used above for a different purpose (lagged feature vectors). I think it would be clearer to just use $\{(i, y_i)\}^T_{i=1}$.

- Could you please explicitly define the metrics: "F1" "B-F-5" "E-F-5"?

- What values of hyper parameters are used in your experiments, on each dataset?

- How are the scores of figure 2 and table 2 connected?

### Soundness
3

### Presentation
3

### Contribution
2
