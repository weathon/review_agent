# TCD-Arena: Assessing Robustness of Time Series Causal Discovery Methods Against Assumption Violations

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4

## Abstract
Causal Discovery (CD) is a powerful framework for scientific inquiry. 
Yet, its practical adoption is hindered by a reliance on strong, often unverifiable assumptions and a lack of robust performance assessment.
To address these limitations and advance empirical CD evaluation, we present **TCD-Arena**, a modularized, highly customizable, and extendable testing kit to assess the robustness of time series CD algorithms against stepwise more severe assumption violations. 
For demonstration, we conduct an extensive empirical study comprising around 30 million individual CD attempts and reveal nuanced robustness profiles for 33 distinct assumption violations. 
Further, we investigate CD ensembles and find that they have the potential to improve general robustness, which has implications for real-world applications. 
With this, we strive to ultimately facilitate the development of CD methods that are reliable for a diverse range of synthetic and potentially real-world data conditions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces TCD-Arena, a modular, open-source testbed for stress-testing time-series causal discovery (CD) methods under stepwise assumption violations. It defines three target graphs—WCG (lag-window graph), SG (lagged summary graph), and INST (instantaneous graph)—and implements 27 violation families (e.g., observational/innovation noise structures, latent confounding, faithfulness violations, nonlinear mechanisms, non-stationarity, sample length, data quality, scaling). For each violation, five intensity levels are simulated across 8 data regimes (varying T, D, L, sparsity, and presence of contemporaneous effects). The study evaluates eight CD methods spanning major paradigms (cross-correlation, VAR/GVAR, VARLiNGAM, PCMCI/PCMCI+, DYNOTEARS, NTS-NOTEARS, and CausalPretraining), with extensive hyper-parameter searches, producing over 50 million runs. 

Key findings: (i) Granger-based approaches are comparatively robust for lagged edges (WCG/SG); (ii) Dynotears and PCMCI+ are strongest on instantaneous edges (INST) while VARLiNGAM lags; (iii) deep models (CausalPretraining, NTS-NOTEARS) underperform simpler ones; (iv) under-specifying the max lag hurts all methods, whereas over-specifying is mostly benign; (v) ensembles (simple average, linear, MLP, ConvMixer) improve average robustness.

### Strengths
1. First comprehensive time-series-focused robustness arena with graded violation intensities rather than binary toggles; covers 27 violations with clear, reproducible generators.
2. Separating WCG/SG/INST clarifies evaluation of lagged vs. instantaneous structure—often conflated in prior work.
3. Massive run budget with systematic hyper-parameter sweeps; sensible aggregation (AUROC across regimes × levels) and ablations on lag mis-specification and hyper-parameter sensitivity.
4. Ensembling insight shows that meta-learners fed only with method outputs (no raw input) increase robustness and reduce variance across violations.

### Weaknesses
1. AUROC on adjacency recovery is threshold-free but may obscure class-imbalance and orientation errors. SHD, F1/AUPRC, precision, and arrow-orientation accuracy would provide complementary views; some are included in the appendix but not highlighted.
2. The DGP (Eq. 2) is additive with univariate link functions and a fixed max lag; some violations (e.g., nonlinearity via RBF/mixtures) are still univariate. Methods exploiting multivariate interactions or cycles may not be assessed.
3. Non-stationarity is simulated by redrawing coefficients A while keeping the skeleton fixed; real systems also change skeletons, contexts, regimes, and sampling rates.
4. Only one faithfulness-violation pattern is implemented; non-Gaussian noise is introduced for innovation but not matched to LiNGAM’s identifiability conditions (e.g., independent non-Gaussian; instantaneous vs lagged). The result that VARLiNGAM is not advantageous under this violation could reflect misspecification.
5. Now it seems only MCAR is used for missingness. More missingness mechanisms, such as MAR and MNAR, should be examined.
6. Meta-learners are trained on simulated violations; transfer to real data under distribution shift is acknowledged but not tested. The Pareto oracle is instructive but unattainable. Guidance on how to pick ensembles in practice is limited.
7. Given 50M runs, a brief computational cost profile (GPU hours, carbon) and per-method runtime/memory would help users scope experiments.

### Questions
1. Could TCD-Arena add skeleton changes and context-specific mechanisms (e.g., regime switches) to stress methods designed for heterogeneous data?
2. What meta-features (if any) about datasets improve selection? Did you try cross-violation training and measure out-of-violation generalization?
3. Interesting result that ↑𝐿 is safe while ↓𝐿 is harmful. Can you show sample-complexity curves for varying 𝑇 under ↑𝐿 (variance vs bias trade-off)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces TCD-Arena, a modular benchmarking toolkit for time-series causal discovery (CD) with a focus on robustness under stepwise assumption violations. It evaluates 8 CD methods spanning major paradigms across 27 violation types (each with graded severity) and multiple data regimes, totaling 50M runs. The study contrasts recovery of three graph notions (GWCG, GSG, GINST), analyzes hyperparameter sensitivity, and explores ensembling of CD methods as a route to improved robustness. Results suggest (i) notable variability in robustness across methods and violations, (ii) simple Granger-style approaches are comparatively robust for lagged effects, (iii) deep methods can trail simpler baselines, and (iv) ensembling provides tangible gains.

### Strengths
I found the paper interesting and I do think that there is a need more papers like in the causal discovery community.

This paper study the robustness to violated assumptions is central for CD adoption

Evaluating GWCG, GSG, and GINST clarifies what each method is able to recover.

It studies hyperparameter sensitivity analysis which is important and often ignored. This it has a   practical value for deployers.

If the toolkit is released as open-source with configs and seeds, TCD-Arena could become a useful community resource.

### Weaknesses
* It is very difficult to understand the contribution of the paper without going to the Appendix.

* Despite the motivation that real-world ground truth is scarce, the study remains fully synthetic. This limits the external validity; even a small real or semi-synthetic case  would strengthen claims.

* Many violations are reasonable, but several design choices (e.g., faithfulness violations via path cancellation; innovation noise blends; stationarity via coefficient resampling) feel one specific instantiation among many. It’s unclear how sensitive findings are to alternative parameterizations. More ablations/justifications would help.

* Some algorithms do not target GINST, others require max-lag specification, and assumptions differ (e.g., non-Gaussianity). While you note this, the aggregated “robustness” comparisons may conflate target mismatch with method weakness. Clearer tracks per target/assumption would avoid this.

* Meta-learners trained on synthetic violations may not transfer across domains/violations. The practical ensembles’ training protocol need clearer specification.

* Recent CD algorithms for time series causal discovery are not included. 


* While the topic of causal discovery robustness is important, this paper is primarily a benchmarking and dataset-style contribution based entirely on synthetic data. It does not introduce a new learning algorithm, theoretical insight, modeling principle, or a new real-data benchmark, types of contributions that, to my understanding, are typically expected for ICLR’s main track. Even though ICLR does occasionally accept benchmark-oriented papers, such works usually have a broad and transformative impact (for example, by introducing a new real-world benchmark or establishing a widely adopted evaluation framework). In its current form, the contribution of this paper seems more incremental in scope (even though I think it is very interesting and needed). That said, I may not be fully aware of the current editorial stance on such submissions, and I would defer to the area and meta chairs regarding venue fit. Given its focus, the work might be better suited for a Datasets and Benchmarks track at a major AI or ML conference, where such empirical frameworks are the main focus.

### Questions
* Can you consider testing methods on larger graphs?

* Can you include at least one semi-synthetic or interventional real dataset to validate ranking consistency (even if partial ground truth)? 

* Do your qualitative conclusions hold under AUPRC, (adj)SHD, and orientation error measures? 

* Any cases where AUROC misleads? 

* Why that specific cancellation pattern? Have you tried near-unfaithful settings (small but nonzero effects) and do trends persist? 

* How were search budgets matched across methods? Can you report per-method best-vs-average gaps in the main text and show robustness profiles over the top-k configs? 

* How do you prevent leakage from the meta-learner training to evaluation regimes? Any evidence of transfer to unseen violation types or severities? Stationarity violation. 

* You resample coefficients while keeping the skeleton fixed. Have you tested changing skeletons over segments (structural breaks)? 

* Can you include or at least discuss clearly recent algorithms like TiMINo[1], varFCI [2], LPCMCI [3], PCGCE [4] J-PCMCI[5], CBNB [6], SpaceTime [7] 


* Can you discuss clearly the contribution of this paper with respect to other ICLR benchmark papers on causal discovery. Like for example: [8] (which was cited but not thoroughly discussed)

Minor:

* You might consider citing [9] in addition to Pearl (2009) and Peters et al. (2017) when referring to a comprehensive review of causal discovery in i.i.d. settings, as this book provides a more in-depth treatment of constraint-based causal discovery methods (PC and FCI) than the other two references.


* In much of the existing literature, the WCG and summary graph are defined differently: both typically include instantaneous relations. It may therefore be helpful to explicitly state that your definitions diverge from the conventional ones. You might also consider using distinct notation (e.g., LWCG for lagged window causal graph and LSG for lagged summary graph) to avoid confusion. Additionally, note that the literature also defines a third type of graph, the extended summary causal graph, which, as I understand it, combines features of the lagged summary graph and the instantaneous graph.


References:

* [1] Peters et al. Causal Inference on Time Series using Restricted Structural Equation Models. Neurips, 2013

* [2] Malinsky and Spirtes. Causal Structure Learning from Multivariate Time Series in Settings with Unmeasured Confounding. KDD workshop on CD. 2018

* [3] Gerhardus and Runge. High-recall causal discovery for autocorrelated time series with latent confounders. Neurips, 2020

* [4] Assaad et al. Discovery of extended summary causal graphs from time series. UAI. 2022

* [5] Gunther et al. Causal Discovery for time series from multiple datasets with latent contexts. UAI, 2023

* [6] Bystrova et al. Causal Discovery from Time Series with Hybrids of Constraint-Based and Noise-Based Algorithms. TMLR, 2024

* [7] Ameche et al. SpaceTime: Causal Discovery from Non-Stationary Time Series. AAAI, 2025

* [8] Cheng et al. CAUSALTIME: REALISTICALLY GENERATED TIMESERIES FOR BENCHMARKING OF CAUSAL DISCOVERY. ICLR, 2024.

* [9] Spirtes et al. Causation, prediction, and search. MIT press. 2000.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces TCD-Arena, a large-scale and modular benchmark framework for evaluating the robustness of time series causal discovery (CD) algorithms under systematically controlled assumption violations. This study evaluates eight representative CD algorithms through over 50 million runs across 27 assumption violations. Additionally, this work explore hyperparameter sensitivity, model misspecification and ensembling strategies that improve robustness. TCD-Arena is released as an open-source toolkit to support reproducible, large-scale robustness evaluation for time series causal discovery research.

### Strengths
The paper presents a comprehensive and rigorously executed empirical benchmark that systematically evaluates the robustness of time series causal discovery methods across diverse assumption violations. By encompassing 27 distinct violation scenarios and analyzing eight representative algorithms, the study establishes a new empirical standard for assessing reliability in time series causal discovery.

### Weaknesses
The authors conducted a commendable and well-structured benchmark study. However, there are a few aspects that could improve the paper’s readability and completeness:
- In line 408 and line 411, the references to Fig. 3b are mistakenly written as Fig. 2b.
- In line 2420, the right panel of Figure 18 shows missing results for NTS-NOTEARS under certain conditions.
- Including a runtime comparison across different algorithms would be valuable for practitioners, as it would help guide the selection of time-series causal discovery methods in real-world applications.

### Questions
- In line 654 and line 771, the references to Ormaniec et al. and Yi et al. appear to be incomplete. Both works were published at ICLR 2025. Moreover, the title of Yi et al. is fully capitalized, which is inconsistent with the other citation styles. Providing complete and correctly formatted references would help readers in the community quickly identify related advances.
- Could the authors provide more details about the GVAR method and the code source used in the experiments? It seems that this corresponds to an extended version of the standard VAR model, as Table 11 indicates that the method has two hyperparameters, coeff and p-val. Including a brief methodological description would make the work more complete. Also, unless there is a specific reason to retain this name, it might be preferable to change it, since another work [1] is also named GVAR, which could lead to confusion.
- Could the authors elaborate on how AUROC was computed in Figure 1? My understanding is as follows: for each assumption violation scenario, there are five different violation levels, and for each level, 100 datasets are sampled. Each method, under each hyperparameter configuration, computes AUROC across all these datasets. The results are then averaged to obtain an average AUROC per hyperparameter configuration, and the best-performing configuration is reported. Please confirm whether this interpretation is correct.
- The paper defines robustness (line 363) in a way that differs from prior works such as Montagna et al. and Yi et al., which report results based on the best-performing hyperparameter setting per violation. Could the authors discuss the differences between these two evaluation strategies and their respective advantages or limitations? Although there may not yet be a community-wide consensus, a discussion on this distinction would be valuable.

[1] Marcinkevičs R, Vogt J E. Interpretable models for granger causality using self-explaining neural networks[J]. arXiv preprint arXiv:2101.07600, 2021.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents TCD-Arena, a modularized and extendable testing kit designed to assess the robustness of time series causal discovery (CD) algorithms against assumption violations. The authors use this toolkit to conduct a large-scale empirical study, evaluating eight distinct CD methods across 27 different assumption violations, each with stepwise increasing severity. The study involves over 50 million individual CD attempts. The paper also provides an initial investigation into ensembling CD methods, which can improve general robustness. The TCD-Arena toolkit is presented as an open-source package to facilitate future research and comparability.

### Strengths
1. The paper's most significant strength is the sheer scale and thoroughness of the empirical study, evaluating 8 CD methods against 27 distinct violations with increasing severity, totaling over 50 million CD attempts.
2. The paper addresses the critical and practical problem of algorithm robustness to assumption violations, which is a known barrier to the adoption of CD methods.
3. The paper provides a novel investigation into ensembling CD methods as a strategy to improve general robustness, which is an under-explored area in the literature.

### Weaknesses
1. This level of resource-intensive evaluation is an "overkill" and is not a realistic or practical standard for most researchers to adopt. Instead of clarifying robustness, such a convoluted framework may deter researchers, raise the barrier for reproduction, and make the results more difficult to interpret. The paper does not convincingly argue that this massive complexity offers proportionate benefits over simpler, more targeted robustness tests.
2. The main paper's evaluation relies exclusively on AUROC. This omits other standard and widely-used metrics in the CD literature, such as True Positive Rate (TPR), False Discovery Rate (FDR), and Structural Hamming Distance (SHD). This makes the results difficult to align and compare with many other benchmark studies.
3. The authors rightly concede that the ensemble results are a "theoretical proof-of-concept". The paper does not provide a clear path for how these ensembles could be deployed in a real-world scenario where ground truth is unavailable and domain adaptation is a challenge.
4. The paper restricts some methods, like PCMCI and PCMCI+, to linear conditional independence tests, even though they can handle nonlinear cases. While this is noted in the appendix, it means the study is not evaluating the full capabilities of these specific algorithms.

### Questions
1. Could the authors justify the decision to rely on AUROC as the primary metric in the main paper and to exclude common metrics like TPR, FDR, and SHD from the study entirely? These are standard in the field, and their omission is a significant weakness.
2. What was the rationale for the specific selection of the eight algorithms? Were more recent, state-of-the-art methods for time series causal discovery considered?
3. Regarding the ensembles: Beyond the "proof-of-concept," what do the authors see as a realistic path to practical application? How could a meta-learner, trained on this synthetic data, be reliably applied to real-world data where the types and severity of violations are unknown?

### Soundness
2

### Presentation
3

### Contribution
2
