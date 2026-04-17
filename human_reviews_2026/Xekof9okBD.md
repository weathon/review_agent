# Efficient Multivariate Temporal Point Process via Monotone Alternating Splines

- Decision: Reject
- Scores: 8, 2, 4, 4

## Abstract
Temporal point processes (TPPs) have widespread applications across various domains. Compared to modeling the conditional intensity of a TPP, modeling its conditional cumulative intensity function (CCIF) improves efficiency and eliminates approximation error. However, currently, the parametrization of CCIF remains limited and scarce. This paper proposes a novel method called the Monotone Alternating Spline (MAS). By leveraging two distinct interpolation and extrapolation components, MAS provides a broad framework for modeling CCIF, offering greater flexibility and efficiency. Theoretically, MAS's interpolation grants a strong fitting ability, while its extrapolation guarantees robust generalization. Extensive experiments illuminate that MAS achieves state-of-the-art performance on both synthetic and real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes **Monotone Alternating Splines**, a framework for parametrizing conditional cumulative intensity function of temporal point processes. The idea is to avoid numerical integration of the conditional intensity function during likelihood estimation, as this is computationally expensive and error prone. The conditional cumulative intensity is modelled in two parts: using monotone splines to fit the function within a defined interval (interpolation) and using a simple monotonic function (i.e. not a spline) to extrapolate outside this interval.

Unlike neural network-based methods, the use of splines means we get analytical derivatives, avoiding the need for automatic differentiation. Theoretical guarantees are offered for fitting and generalization, and empirical results on five synthetic and four real-world datasets indicate good performance.

### Strengths
The paper addresses a well-defined and important problem, namely the inefficiency and inflexibility of modelling TPPs. The proposed MAS framework appears to be an intuitive solution that tackles the limitations of CIF-based models and (cumulative) neural network models. The validation on nine datasets, including four real-world datasets, shows consistently competitive performance with all baselines. Standard deviations are reported for experimental results, adding some quantification of uncertainty to the claims.

The generalization analysis in Section 4 provides a clear, interpretable theoretical bound that decomposes the error into interpolation, extrapolation and complexity terms. This is nice to have for a practical method. The argument and evidence for MAS efficiency, thanks to its analytical derivatives, is convincing. The ablation study directly validates theoretical tradeoffs. The conceptual figures are quite clear and efficiently use the available space (assuming ICLR guidelines allow `wrapfig`).

### Weaknesses
The main criticism is in the presentation of some of the results. Figure 4(e) shows training time for what appears to be a single run, which is surely highly variable. The mean and standard deviation/error over multiple runs would be more convincing. Similarly, Figures 4(a-d) could benefit from some uncertainty quantification.

Furthermore, the scaling of timestamps by constant factors appears a bit _ad-hoc_. Is MAS sensitive to this scaling hyperparameter?

The paper's primary empirical evidence is presented in Tables 1 and 2, which are dense, difficult-to-read "leaderboard" tables. The use of bolding and underlining to highlight the top two models is a clear sign that the table format itself is failing to make the results accessible. The raw scores are not something that any reader is likely to need to look up directly, at least not in the main section of the paper. More rigorous and modern methods for comparing multiple models across multiple datasets are available, including the use of statistical tests (rather than reporting raw scores) and visual presentation, using small multiples or a critical difference diagram. See Demšar (2006; ["Statistical Comparisons of Classifiers over Multiple Data Sets." _JMLR_](http://jmlr.org/papers/v7/demsar06a.html)) or works by Edward Tufte.

### Minor points

1. In the introduction it is written _"asynchronous timestamps in continuous time, which sets them apart from conventional synchronized time series data..."_ This seems to be an odd use of the term "asynchronous" and "synchronous" when it would appear the authors mean really "irregular" and "regular", here.

2. The abbreviation "(TPPs)" is introduced in the abstract but never re-used.

3. The Figures use quite a modest amount of space and this is laudable but some of the labels and arrows are quite small, and only readable when zooming in, maybe some parts could be made slightly bigger.

### Questions
1. Could you please clarify how many runs were used to compute the means and standard deviations in Tables 1 and 2?

2. For Figure 4(e), could you provide the mean and standard deviation for the training times over multiple runs?

3. The data scaling in the appendix (Section D.2) seems necessary to get MNN-based models to converge. Did you test if MAS is also sensitive to this scaling, or does it converge robustly even without it?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Monotone Alternating Splines (MAS), a new method for efficiently modeling multivariate temporal point processes through the conditional cumulative intensity function (CCIF). Unlike conventional methods via monotone neural networks, MAS combines spline-based interpolation and simple monotone extrapolation, ensuring both flexibility and computational efficiency. The interpolation enhances fitting ability, while the extrapolation guarantees generalization. Experiments on both synthetic and real-world datasets demonstrate that MAS outperforms existing methods in terms of predictive performance.

### Strengths
- The manuscript is clearly written and easy to follow.
- The result demonstrating higher accuracy than other major models is impressive from a practical standpoint.

### Weaknesses
- The primary concern lies in the technical contribution’s weakness. The idea of modeling the cumulative intensity function with a monotone spline itself is not particularly novel [1]. While separating the modeling of interpolation and extrapolation is practically useful, it remains a minor methodological refinement. Also, representing the cumulative intensity function with a monotone spline is mathematically equivalent to representing the intensity function with a nonnegativity-constrained spline. Since the latter approach has already been explored (e.g., [2]), it is unclear what additional significance the proposed idea of modeling the cumulative intensity function using a monotone spline truly provides.
- If the paper’s main contribution lies in its empirical accuracy (Tables 1 and 2), the experimental setup requires further improvement. For example, in fitting a Hawkes process with a simple triggering kernel and limited data, simpler models naturally achieve higher accuracy. Neural network–based models tend to underperform when the dataset is small unless their architecture (layer depth, unit size, and especially the smoothness of activation functions) is carefully tuned to the dataset. To fairly demonstrate that the proposed model outperforms existing methods in terms of both flexibility and generality, the baseline monotone NN architectures should be determined via cross-validation, and the comparison should be conducted across multiple data sizes.

[1] Bantis et al., "Survival estimation through the cumulative hazard function with monotone natural cubic splines", Lifetime data analysis, 2012.

[2] Loaiza-Ganem et al. "Deep random splines for point process intensity estimation of neural population data", NeurIPS, 2019.

### Questions
- The explanation of the synthetic data is insufficient. For the Hawkes process datasets, over what time span was the data generated? What was the scale of the dataset used for training?
- The experimental results only report accuracy. Since the paper claims efficiency as a key advantage of the proposed model, the training time should also be provided.
- Why is the second condition in Theorem 3.1 necessary? The explanation refers to the Hawkes process, but by adopting a jump-type triggering kernel, one can construct a Hawkes process that does not satisfy this condition. Therefore, this second condition may not be a general requirement for cumulative intensity functions.
- Theorem 3.1 is presented as self-evident, but references should be provided for its proof.
- The paper states that MNNs lack expressive power (lines 180-181), but the reason or evidence for this claim should be explained.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The work proposed to use monotone alternating splines (MAS) to parameterize the cumulative conditional intensity function for modelling temporal point process. The MAS consisting of interpolation parts and extrapolation parts. This approach does not rely on numerical integration and does not have the problems of low computational efficiency and numerical approximation error. Theoretical study shows the approaches ability to fit complex TPPs and generalize effectively. Empirical study also shows competitive performance of the approach against a series of baselines on fitting TPP sequences and next event forecasting.

### Strengths
1. The work is well motivated, aiming at tackling some important and significant problems in TPP modelling, including the computational efficiency and approximation error.
2. Experiment results on next event predictions and log likelihood evaluation demonstrates the strong performance of the work against the baselines.

### Weaknesses
1. Despite the solid motivations, the innovation of the work can be summarized as a new approach toward parameterize the cumulative intensity function. This innovation itself is incremental.
2. Even improved efficiency is a central contribution of the work, good quantitative performance on TPP benchmarks are also critical for a TPP model. Many recent works on temporal point process evaluate TPP models on multiple-events long-horizon predictions [1]. Lacking such experiment results makes the empirical study of the work relatively weaker.
3. There are some recent SOTA TPP models not compared against in the model, including C-Diff [2], Hypro [3], Decomposable TPP [4]. The work should include these models in the comparison.
4. The ablation study results shows that the model performance is easily impacted by the selection of hyper-parameters including, interpolation length, number of knots, and interpolation function. However, the selection of these hyper-parameters are purely empirical without principled guidance. The existence of these hyper-parameters also makes the contributions of the work less significant.
5. The work claims that one of the limitations of existing intensity-based TPP approaches is the approximation error due to Monte-Carlo integration. This claim is not fully substantiated. In other words, can the performance of CIF-based models be improved and close the gap with the proposed approach with more accurate numerical integration?

References

[1] Xue, S. EasyTPP: Towards Open Benchmarking Temporal Point Processes, ICLR 2024 

[2] Zeng, Mai. Interacting diffusion processes for event sequence forecasting. ICML 2024.

 [3] Xue, Siqiao, et al. "Hypro: A hybridly normalized probabilistic model for long-horizon prediction of event sequences." NeurIPS 2022 

[4] Panos, Aristeidis. "Decomposable Transformer point processes." NeurIPS 2024.

### Questions
1. Is the same set of hyper-parameters like the number of interpolation nots or the interpolation length applied to all datasets or different hyper-parameters are selected for different datasets? Is it possible to dynamically change the hyper-parameters based on the local context of the event sequence? For example, should we have shorter interpolation lengths or more interpolation nots where event occurrences are denser?
2. If the next event happens at a very distant future, is the conditional NLL likelihood evaluation or the forecasting of that distant future events going to be overly reliant on the extrapolation function?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes **MAS** (Monotone Alternating Splines) for multivariate temporal point processes. It parameterizes the **conditional cumulative intensity function (CCIF)** after each event via a piecewise monotone spline for interpolation on $((t_n, t_n+L])$ and a simple extrapolator (linear or exponential) on $((t_n+L,\infty))$. Formally,  This decouples local flexibility (splines) from global monotonicity (tail), aiming to avoid architectural constraints of monotone neural nets (MNN).

Theory: (i) an interpolation error bound for the spline part; (ii) a generalization bound decomposing probability, extrapolation, and complexity terms; (iii) representability of Poisson/Hawkes. Experiments compare against CIF-based (RMTPP/THP/SAHP), CCIF/MNN (FullyNN/EMTPP), TriTPP (spline/flow), and intensity-free/score-matching baselines, reporting competitive or better NLL/RMSE on several datasets.

### Strengths
- **Clean parameterization**: Alternating “spline (local) + tail (global)” design for CCIF is intuitive, implementable, and potentially more flexible than weight-constrained MNNs.    
- **Theory present**: Interpolation error and a three-term generalization bound provide a lens to discuss trade-offs (($\Delta, L, Z$)).
- **Empirical competitiveness**: Broad set of baselines and datasets; MAS is often competitive and occasionally best.
- **Engineering practicality**: The decoupling offers a reasonable path to stability and monotonicity in multivariate TPPs.

### Weaknesses
1. **Related-work inaccuracies / mispositioning**
- States/leans that **TriTPP only suits NHPP**; this conflicts with the **time-rescaling theorem** (general TPPs with integrable intensity admit such mappings). The issue harms credibility of the positioning.
- Overgeneralizes that “most monotone splines don’t preserve monotonicity outside the interpolation interval.” In practice, **RQS with linear tails** (as used in neural spline flows and TriTPP variants) ensures global monotonicity; the paper should contrast **its tail choice** (e.g., exponential) as a design tuned to TPPs, not as if splines lack tails.

2. **Formula/technical errors**
- **Hawkes CCIF corollary** uses $(t_n)$ in every exponential term; it should sum over all past events (t_i) with $(\exp(-\beta (t - t_i)))$. This is a visible typo in a central sanity-check.

3. **Completeness**
- Several wins are within one standard deviation; 
- Missing key baselines: contemporary CDF/CCIF (e.g., CuFun/UMNN-style monotone flows) and parametric MLE Hawkes on synthetic data. Without them, it’s hard to assess where MAS truly dominates.

4. **Hyperparameters/fairness**
- MAS introduces extra knots (p) and window (L). Although defaults are given (e.g., (p=10, L=6)) and some ablations exist, the selection guidance from the bound is not operational (unknown constants), and per-dataset sensitivity isn’t systematically reported.
- Some results show higher variance for MAS than baselines, hinting at stability/tuning issues.

### Questions
1. Please correct the Hawkes corollary and re-run any sanity checks that depend on it.
2. Can you reposition TriTPP under the time-rescaling lens and explain, with evidence, when MAS’s exponential tail outperforms RQS linear tails (e.g., long-tail arrival gaps, jump structure)?
3. Include CuFun/UMNN-style monotone flow baselines and MLE Hawkes on synthetic Hawkes. If excluded, justify concretely (code unavailability, compute).
4. Provide operational guidance for ((p, L)): e.g., a heuristic tied to expected event density / hazard smoothness; analyze sensitivity systematically.

### Soundness
2

### Presentation
2

### Contribution
2
