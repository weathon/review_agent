# High-Dimensional Online Change Point Detection with Adaptive Thresholding and Interpretability

- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
Change point detection (CPD) identifies abrupt and significant changes in sequential data, with applications in human activity recognition, financial markets, cybersecurity, manufacturing, and autonomous systems. While traditional methods often struggle with the computational demands of high-dimensional data, they also fail to provide explanations for detected change points, limiting their practical usability. This paper introduces a CPD framework that enhances both interpretability and scalability by leveraging the Sliced Wasserstein (SW) distance. Our contributions are fourfold: (1) we present a method to transform multivariate data into one-dimensional time series using the SW distance, enabling compatibility with existing CPD methods; (2) we derive theoretical insights, demonstrating that random slices of the SW distance follow a Gamma distribution, which facilitates statistical hypothesis testing for CPD; (3) we propose a novel self-adapting online CPD algorithm based on an adaptive threshold for a given significance level $q$; and (4) we propose a model-specific framework for generating contrastive explanations for annotated change points. We find that our method outperforms popular (online/offline) change point detection methods by reducing the number of false positives by at least 48% on average while also providing interpretable change points and maintaining competitive or superior detection performance, making it practical for deployment in high-stakes applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors consider the problem of nonparametric online change point detection. They suggest an algorithm based on sliced Wasserstein distance. Because of the reduction to one-dimensional projections, the authors significantly reduce computational expenses. Moreover, they establish (see Theorem 3.1) that the Wasserstein distance between random projections tends to the gamma distribution as the dimension increases. This suggest an approach for automatic threshold choice based on asymptotic distribution of the sliced Wasserstein distance. The authors illustrate the performance of their algorithm on synthetic and real-world data.

### Strengths
Theorem 3.1 about asymptotic distribution of the sliced Wasserstein distance (as $d \rightarrow \infty$). allows for adaptive choice of the threshold.

### Weaknesses
1. While Proposition 3.2 gives an intuition about running length of the detection procedure, its detection delay remains unexplored. This leaves open the question of optimality of the suggested approach.

2. Some important references are missing. A variant of CUSUM was also studied in [Yu et al., 2023]. Methods based on different f-divergences and density-ratio estimation were considered in [Liu et al., 2013] and [Hushchyn, A. Ustyuzhanin, 2021]. Their change-point detection methods are based on KLIEP [Sugiyama, 2008], uLSIF [Kanamori et al., 2009], and RuLSIF [Yamada et al., 2013]. Approaches based on deep neural networks (with a statistics similar to the one used in GANs) were suggested in [Hushchyn et al., 2020], [Puchkin, Shcherbakova, 2023]. Finally, in [Cao et al., 2018] and [Markovich, Puchkin, 2024] the authors proposed algorithms based on online convex optimization and prediction with expert advice. I would suggest the authors to incorporate these references into the literature review during revision.

[Sugiyama, 2008] M. Sugiyama, T. Suzuki, S. Nakajima, H. Kashima, P. von Bunau, and M. Kawanabe. Direct importance estimation for covariate shift adaptation. Annals of the Institute of Statistical Mathematics, 60(4):699–
746, 2008.

[Kanamori et al., 2009] T. Kanamori, S. Hido, and M. Sugiyama. A least-squares approach to direct importance estimation. Journal of Machine Learning Research, 10:1391–1445, 2009.

[Yamada et al., 2013] M. Yamada, T. Suzuki, T. Kanamori, H. Hachiya, and M. Sugiyama. Relative density-ratio estimation for robust distribution comparison. Neural Computation, 25(5):1324–1370, 2013.

[Liu et al., 2013] S. Liu, M. Yamada, N. Collier, and M. Sugiyama. Change-point detection in time-series data by relative density-ratio estimation. Neural Networks, 43:72–83, 2013.

[Cao et al., 2018] Y. Cao, L. Xie, Y. Xie, and H. Xu. Sequential change-point detection via online convex optimization. Entropy, 20(2):108, 2018.

[Hushchyn et al., 2020] M. Hushchyn, K. Arzymatov, and D. Derkach. Online neural networks for change-point detection. Preprint, arXiv:2010.01388, 2020.

[Hushchyn, A. Ustyuzhanin, 2021] M. Hushchyn and A. Ustyuzhanin. Generalization of change-point detection in time series data based on direct density ratio estimation. J. Comput. Sci., 53:Paper No. 101385, 8, 2021.

[Puchkin, Shcherbakova, 2023] N. Puchkin and V. Shcherbakova. A Contrastive Approach to Online Change Point Detection. In Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 5686–5713, 2023.

[Yu et al., 2023] Y. Yu, O. H. M. Padilla, D. Wang, and A. Rinaldo. A note on online change point detection. Sequential Analysis, 42(4):438–471, 2023.

[Markovich, Puchkin, 2024] Score-based change point detection via tracking the best of infinitely many experts. Preprint, arXiv:2408.14073, 2024.

### Questions
1. According to my experience, the threshold choice based on asymptotic distribution of the test statistic may be not the best one (this agrees with one of limitations you mentioned in Section 5). For instance, the automatic threshold choice for the kernel-based test statistic [Li et al., Seq. Anal., 2019] lead to frequent false alarms. Has a similar problem occurred, for instance, on the Room Occupancy data (where the time series has only 4 components)? Could you elaborate on how you chose a threshold in that case?

2. Could you add a comparison of your algorithm with any of [Liu et al., 2013], [Hushchyn et al., 2020], [Markovich, Puchkin, 2024]? All these papers considered the human activity recognition data set. Besides, [Hushchyn et al., 2020] checked performance of their procedure on the MNIST data set, while [Markovich, Puchkin, 2024] applied their approach to the Room Occupancy data.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Sliced Wasserstein Change Point Detection (SWCPD), a novel framework for change point detection (CPD) in high-dimensional data.
 
The cornerstone of this framework is a new insight: the theoretical result showing that random Sliced Wasserstein slices follow a Gamma distribution for adaptive thresholding. This finding is leveraged by the authors practically for CPD. Multivariate data is transformed into univariate signals using Sliced Wasserstein (SW) distance and then a hierarchical procedure for generating contrastive explanations of detected changes is implemented.
 
The method is evaluated on synthetic and real-world datasets (MNIST, Human Activity Recognition, Occupancy) and compared against established baselines (BOCPD, e-divisive, KCP, OT-CPD, ClaSP).

### Strengths
- Novel theoretical characterization connecting SW projections to Gamma distributions, enabling principled statistical hypothesis testing for CPD.

- First method to combine Sliced Wasserstein distance with adaptive thresholding and contrastive explanations in a unified framework.
 
- Theorem 3.1, Propositions 3.2, and supporting lemmas are seemingly sound mathematical foundations

- Comprehensive experimental evaluation including synthetic and real datasets, with ablation studies and parameter sensitivity analysis

- Reproducible methodology with clearly described experimental settings and baselines.
- Strong empirical results, particularly in reducing false positives
 
- Figures and tables generally well-designed; boxplots in Figure 10 effectively communicate parameter sensitivity.
 
- Relevant to high-stakes domains (finance, healthcare, autonomous systems) where both accuracy and interpretability are critical.

- SWCPD has computational efficiency, which is meaningful in practice for streaming applications.

### Weaknesses
Theoretical Limitations
- Assumption Gap (in Section 3.1): Theorem 3.1 requires d → ∞ for the Gamma distribution approximation, but practical datasets have moderate dimensions. While Table 5 shows empirical validation works for d ≥ 20, the paper would benefit from:
- Providing finite-sample bounds or approximation error rates for moderate d
- Specifying minimum dimension requirements for reliable performance
- Discussing behavior when theoretical assumptions are violated
- Temporal Correlation (in Section 3.2): The moving average smoothing (step 2 of the algorithm) applies to temporally correlated sliding windows, yet the paper assumes i.i.d. random projections for statistical validity. The paper acknowledges this, but dismisses it without rigorous justification.
- Method of Moments Limitations: MoM estimation is sensitive to outliers and heavy-tailed distributions. 

The paper should:
- Compare MoM to maximum likelihood estimation empirically
- Analyze robustness to distribution misspecification
- Provide diagnostic criteria for when the Gamma approximation fails

Hyperparameter Selection (Section 4.2):
- Tolerance criterion (Algorithm 2, line 3): The stopping criterion β̂ ≤ tol lacks justification. What is an appropriate tolerance value? How sensitive are explanations to this choice?
- Window length w: Figure 3 shows high sensitivity to w, but no principled selection strategy is provided beyond grid search. Next steps should develop data-driven selection methods (e.g., cross-validation, information criteria).
- Wasserstein order p: The choice between p=2,4,6 appears dataset-dependent (Table 2, Figure 3) with no clear guidance. Provide decision rules based on drift characteristics.

Experimental Concerns
- Limited Dataset Diversity: Four datasets (two relatively simple: MNIST, Occupancy; one tied: HAR with 0.59 e-divisive).
- Missing challenging scenarios: high-noise environments, gradual drifts, multimodal distributions, very high dimensions (d > 100).
- Financial time series and sensor data mentioned in introduction but not evaluated.

Statistical Rigor:
- Confidence intervals on metrics would strengthen claims (only AUC has std in Table 3).
- No runtime comparisons with non-OT baselines (Table 8-9 only compare OT methods).
With L=5000 projections, practical speed claims need verification for streaming applications with strict latency requirements.

### Questions
To strengthen the background around Wasserstein-based change detection, it would be helpful to frame the discussion in consideration of prior OT/WD approaches, including WATCH and its lifelong extension (LIFEWATCH), which perform unsupervised CPD by monitoring Wasserstein shifts in high-dimensional streams. Beyond these, Cheng et al. (ICASSP’20) proposed an optimal transport CPD and segment-clustering pipeline grounded in the Wasserstein two-sample test. From a statistical perspective, Horváth et al. (Annals of Statistics, 2021) analyze sequential monitoring via a weighted Wasserstein distance with asymptotic and finite-sample guarantees, which I believe is a useful context for your hypothesis-testing claims. 

Can you provide finite-sample error bounds or convergence rates for the Gamma approximation in Theorem 3.1? Specifically, what is the approximation error for moderate dimensions (d = 10-50) commonly seen in practice?

What is the practical minimum dimension d for reliable performance? Table 5 shows empirical validation, but can you provide theoretical guidance or diagnostic criteria for when the method should or shouldn't be applied?

You acknowledge temporal correlation (in line 218) but dismiss concerns without rigorous justification. Can you provide theoretical analysis or simulation studies quantifying the impact on confidence interval validity? 

Can you show empirical evidence of robustness across different correlation structures? Can you provide guidelines for adjusting parameters when correlation is strong?

How should practitioners set the tolerance parameter “tol” in Algorithm 2? Can you provide a sensitivity analysis or specify the selection criteria?

Several improvements appear modest given the reported uncertainty. For example, HAR shows 0.85 ± 0.12 (yours) vs. 0.84 ± 0.15 (ClaSP), and Occupancy shows 0.59 vs. 0.52-0.58 (baselines). Can you provide statistical significance tests (such as paired t-tests or Wilcoxon tests) to confirm these improvements are not due to random variation?

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
3

### Summary
In this paper, the authors present a framework for online change point detection in high-dimensional data with Sliced Wasserstein distance. The idea is to transform multivariate data into one-dimensional projections, derives that random SW slices follow a Gamma distribution for hypothesis testing. An adaptive thresholding algorithm was proposed based on a significance level, and aimed to offer contrastive explanations via geometric properties of projections.

### Strengths
> The paper addresses scalability and explainability in high-dimensional CPD, which is a growing concern in sequential data analysis.

> It includes a derivation showing SW slices approximate a Gamma distribution, enabling some statistical testing.

> Experiments on synthetic and a few real datasets show minor reductions in false positives in certain setups.

> The adaptive threshold and projection-based explanations offer a straightforward way to add interpretability.

### Weaknesses
- While the paper evaluates on time series (e.g., HAR) and image-based streams (e.g., MNIST), it fails to compare against state-of-the-art CPD methods tailored to each modality. For time series, it misses deep learning-based SOTA like LSTM/Transformer CPD (e.g., recent kernel methods for multivariate series), relying instead on general baselines like CUSUM or OT-CPD. For images (MNIST as sequential data), it ignores computer vision-specific CPD techniques, such as video anomaly detection models (e.g., autoencoders or flow-based methods for change detection in image sequences), limiting claims of superiority.
  - A general methodology for fast online changepoint detection
  - DeepLocalization: Using change point detection for Temporal Action Localization
  - Adaptive Block-Based Change-Point Detection for Sparse Spatially Clustered Data with Applications in Remote Sensing Imaging

- Only a handful of datasets are used (synthetic, HAR, MNIST, Occupancy), with small sequences (e.g., 200 samples per class in MNIST). No broader testing on diverse high-stakes domains like finance (e.g., stock data) or cybersecurity (e.g., network traffic benchmarks), despite mentioned applications. Results may not generalize.

- Relies on i.i.d. assumptions in windows, which may not hold for correlated time series or image streams. Adaptive thresholding requires tuning (e.g., α, window size), and computational costs for many projections (L=5000) aren't quantified for real-time online use.
Underdeveloped Interpretability: Contrastive explanations via projections are basic; no user studies or quantitative metrics (e.g., fidelity) to validate if they truly aid understanding in practice.

- Theoretical insights (Gamma approximation) are asymptotic (d→∞), potentially weak for moderate dimensions. Overall, incremental over existing SW/OT-based CPD (e.g., Cheng et al., 2020b), without strong novelty.

### Questions
>> The paper lacks comparisons to modality-specific CPD SOTA. Could the authors explain the reason? I understand it has used HAR (time series) and MNIST (image streams), but baselines are general (e.g., CUSUM, OT-CPD). For time series, why not include deep CPD methods like Transformer-based detectors (e.g., from recent works on multivariate series)? For images, why omit CV-specific SOTA like optical flow or autoencoder-based video CPD? This omission weakens claims of "competitive or superior performance."

>> How generalizable are results beyond the limited datasets? Experiments focus on synthetic data, HAR, MNIST (as streams), and Occupancy, with small scales (e.g., 200 samples/class). Have you tested on larger, domain-specific benchmarks like financial time series (e.g., S&P 500 drifts) or cybersecurity traces (e.g., KDD Cup anomalies)? Without this, applicability to "high-stakes" fields seems overstated.

>> I wonder about the robust of the Gamma approximation in practice. Theorem 3.1 assumes d→$\infty$ for SW slices ~ Gamma, but datasets like HAR (d~561) or MNIST (d=784) are finite. Appendix shows p-values, but could you provide ablation on approximation error for lower d, or when distributions violate assumptions (e.g., non-Gaussian)? Moreover, the experiment is not comprehensive. It doesn't consider how the adaptive thresholding is sensitive to parameter choices. Algorithm depends on α, window size, lookback Kmax. Appendix grid search on MNIST/Occupancy shows variations, but no sensitivity analysis across all datasets. For low-drift synthetics, how does it avoid false positives/negatives compared to fixed thresholds in OT-CPD?

>> The discussion is not thorough. How about the computational trade-offs for online deployment? With L=5000 projections and window sizes w=50, online CPD might be slow for streaming data. Grid searches vary α, w, Kmax, but no runtime benchmarks vs. baselines. How does it scale for real-time apps (e.g., autonomous systems), and why no efficiency metrics?

>> I am confused about the validation of the interpretability of explanations? Projection-based contrastive explanations identify "discriminative features," but lack quantitative evaluation (e.g., explanation accuracy via perturbation tests) or qualitative user studies. On HAR/MNIST, do they align with domain knowledge (e.g., sensor features in activity shifts)? Could you add metrics like faithfulness? What about correlated data or non-i.i.d. windows? Windows assume independence, but time series (HAR) and image streams (MNIST) often have temporal correlations. This could bias SW distances or MoM estimates. Could you discuss mitigations, like decorrelation steps, or test on autocorrelated synthetics?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes a novel method for explainable online change-point detection in high-dimensional data using the sliced Wasserstein distance. By transforming a multivariate time series into a one-dimensional signal, the method both detects change points and identifies which univariate series contribute most to the change. The detector uses a self-adaptive threshold, updating it for each window.

### Strengths
1. The paper tackles an interesting practical task: detecting change points and identifying which components of the time series contribute to those changes.

2. It provides theoretical guarantees for the distribution of the Sliced Wasserstein distance.

3. Visualizations and experiments on synthetic and real-world datasets demonstrate the method’s interpretability and change-detection ability, compared with multiple baselines.

4. The paper includes an ablation study with various parameter settings and reports runtime across different time-series lengths.

### Weaknesses
1. Although the paper targets high-dimensional data, the datasets in the experiments are relatively low-dimensional. The largest dimension is 50 in the change point task with synthetic datasets. It would be beneficial to include higher-dimensional settings in the explainability section as well. In addition, the runtime as a function of the dimension $d$ is not reported.
2. The main paper claims that the Sliced Wasserstein distance follows a Gamma distribution, but the theory is proved only for $p=2$.
3. In the synthetic-data explainability section, interpretability is shown only for Gaussian mean-shift cases. It would be more complete to include variance shifts and other distributions. In the synthetic change-point detection section, there is no baseline comparison.

Some minor comments:
1. Some notation appears before it is introduced. For example, $\delta$ in line 233, $K_{max}$ on line 249.
2. The notation of $p$ and $q$ is sometimes confusing and used interchangeably. For example, on lines 167 and 259. $p$ should denote the Wasserstein order, and $q$ the confidence level or quantile?

### Questions
1. In the experiments, what tolerance is used to terminate Algorithm 2 at line 3?
2. The number of change points is relatively small compared with the data dimension and the series length. What happens if you simulate many change points and shift a larger subset of series, for example, more than 2 change points across more than 3 series when $d=50$? 
3. Is it challenging to provide a theoretical guarantee for change-point detection? For example, can the current theory be extended to quantify the uncertainty of correctly localizing the change point?
4. Is the dimensionality of the HAR dataset used in the experiments 250? What does “selected 25 instances” mean?

### Soundness
3

### Presentation
2

### Contribution
3
