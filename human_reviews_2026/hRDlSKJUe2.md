# Taking the GP Out of the Loop

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Bayesian optimization (BO) has traditionally solved black-box problems where evaluation is expensive and, therefore, observations are few. Recently, however, there has been growing interest in applying BO to problems where evaluation is cheaper and observations are more plentiful. Scaling BO to many observations, $N$, is impeded by the $\mathcal{O}(N^3)$ cost of a na\"{\i}ve query (or $\mathcal{O}(N^2)$ in optimized implementations) of the Gaussian process (GP) surrogate. Many methods improve scaling at acquisition time, but hyperparameter fitting still scales poorly. Because a GP is refit at every iteration of BO, fitting remains the bottleneck. We propose Epistemic Nearest Neighbors (ENN), a lightweight alternative to GPs that estimates function values and epistemic uncertainty from $K$-nearest-neighbor observations. ENN has $\mathcal{O}(N)$ acquisition cost and, crucially, omits hyperparameter fitting, making ENN-based BO also $\mathcal{O}(N)$. Because ENN omits hyperparameter fitting, its uncertainty scale is arbitrary, making it incompatible with standard acquisition methods. We resolve this by applying a non-dominated sort (NDS) to candidate points, treating predicted values ($\mu$) and uncertainties ($\sigma$) as two independent metrics. Our method, TuRBO-ENN, replaces the GP surrogate in TuRBO with ENN and its Thompson-sampling acquisition with this NDS-based alternative. We show empirically that TuRBO-ENN reduces proposal generation time by one to two orders of magnitude compared to TuRBO and scales to thousands of observations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes Epistemic Nearest Neighbors (ENN) as a lightweight alternative to Gaussian Processes (GPs) in Bayesian Optimization (BO), aiming to address scalability issues in BO with many observations. ENN estimates function values and an uncalibrated uncertainty using K-nearest neighbors, entirely avoiding GP hyperparameter fitting. To handle the lack of uncertainty calibration, the authors introduce a non-dominated sorting (NDS) acquisition strategy combining mean and uncertainty. The method, TuRBO-ENN, replaces the GP and Thompson sampling in TuRBO with ENN and NDS. Experiments on analytic benchmarks and RL simulators suggest comparable performance to TuRBO but with one to two orders of magnitude lower wall-clock time.

### Strengths
This paper identifies a genuine bottleneck in BO for large-N regimes and focuses on practical scalability rather than marginal gains in theoretical accuracy. ENN is simple: thus it maintains interpretability and offers immediate computational savings.

### Weaknesses
1. The core surrogate (weighted K-NN) is not new, while the acquisition strategy based on Pareto dominance between mean and uncertainty has been explored before (e.g., De Ath et al., 2021). Thus, the paper’s main contribution is engineering-oriented rather than conceptual.
2. There is no convergence or regret bound specific to ENN-based acquisition. 
3. Because ENN’s “epistemic” variance is based purely on distance, it may not reflect model uncertainty in any principled sense. This undermines the interpretation of the “exploration” component in BO and raises questions about generalizability beyond trust-region settings.
4. The conclusion suggests “near state-of-the-art quality” while only showing parity with a reduced TuRBO baseline. The claim that the method “removes the GP bottleneck” is somewhat overstated given that GP approximations (e.g., inducing points, Vecchia) already achieve near-linear scaling.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes BOMO for Bayesian Optimization with Many-Observations. This is introduced in their TuRBO-ENN algorithm which replaces GP surrogate with a K-nearest-neighbors surrogate called Epistemic Nearest Neighbors (ENN) and avoids calibrated uncertainty by doing non-dominated sorting inside the TuRBO trust region.

### Strengths
The paper attempts to bring together several concepts to push Bayesian optimization (BO) to many observations settings. These, however, contradicts the very essence of BO which is typically targeted at optimizing black-box or expensive-to-evaluate functions with limited observations. The motivation presented by the authors read more like a succinct literature review with no clear case for such formulations. 

Nonetheless, the authors attempted to think beyond conventional thinking by bringing together several formulations that are uncommon in BO settings to make a case for their proposition. Their formulations, however, lacks depth with their assumptions rife with several canonical settings including noise-free formulation, Euclidean-distance variance notion, and uncorrelated estimates of $f(x)$.

### Weaknesses
1. Unclear conceptual workflow; the proposition seems heuristically motivated with no clear theoretical formulations.
2. The empirical settings and results are unclear and lack comprehensive approaches. The authors claimed that another method (Vecchia GP) exists but seemingly provided a computational cost concerns as a basis for not comparing. 
3. The performance of TuRBO-ENN as presented in the results are consistently behind conventional methods used for BO. The authors downplayed this as a separate focus despite making an earlier claim of making SOTA algorithm faster while producing comparable-quality solutions.
4. The paper is weakly presented with several mash up of literatures.

### Questions
1. The authors presented three properties which formed the basis of their work but mostly lacks justification. Can a sufficient narrative on the validity of those properties be presented?

2. The paper is rife with a lot of assumptions for simplicity including the properties in 1, noise-free derivation, Turbo-driven mechanics, and so on. A proper attempt to tackle the aforementioned problem in complex settings is needed. Can this be done?
 
3. At some points, the authors used the notion of D rather than N to make a case in their experiments. Can this be clarified as it contradicts the fundamental proposition of many-observations?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new surrogate model for Bayesian optimization.
The key motivation is that the commonly used surrogate model, GPs, has a cubic time complexity, which scales poorly when the number of queries increases.

To this end, this paper proposes epistemic nearest neighbors (ENN) as the BO surrogate.
This surrogate does not require model fitting.
The predictive mean is a weighted combination of the \\(K\\) nearest neighbors' labels, and the predictive variance is harmonic mean of the distances to the \\(K\\) nearest neighbors.

Empirically, the proposed surrogate (with a custom acquisition strategy and trust regions) yields competitive BO performance on many tasks.
Meanwhile, the surrogate model runs very fast as it does not require modeling fitting.

### Strengths
1. The proposed surrogate model is very simple.
It is based on \\(K\\) nearest neighbors, which could scale up to very large datasets thanks to highly optimized libraries nowadays.
The downside, however, is that the prediction of the ENN is often mis-calibrated.
And thus this surrogate model cannot be used with standard acquisition functions.
Instead, they have to resort to sampling non-dominated points in the trust regions.

1. It is also quite promising that even this simple model yields quite competitive BO performance.

### Weaknesses
1. The proposed method does cause BO performance regression, though the regression is often not significant.

1. The biggest concern that I have is that the experiments are not the best to validate the proposed method.
In particular, the maximum number of queries is 300.
At this scale, GP inference is still very fast despite the cubic time complexity.
Thus, it would be better to validate the proposed method in high throughout settings with tens of thousands of evaluations.
At the current stage, it is still not entirely clear if the proposed method is of practical use.

1. There is another potential concern when applying this surrogate model to high dimensional problems.
The proposed surrogate model is not compatible with existing acquisition functions because the predictive variance is often mis-calibrated (and also probably because the predictive mean/variance are non-smooth).
As a result, this surrogate model has to rely on discretizing the trust region and non-dominated sort on the discretization.
However, as the dimension increases, the size of the discretization should also increase, possibly exponentially.

### Questions
1. One thing would be interesting to try is running non-dominated sort acquisition for GPs.
By using the same acquisition, this checks how much the performance deterioration is caused by the surrogate model.

1. Since the proposed method also employs a trust region.
Does the trust region shrink/expand in the same way as TuRBO?
I am curious how often do the \\(K\\) nearest neighbors fall into the trust region.

### Soundness
3

### Presentation
3

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
This paper proposes TuRBO-ENN, a scalable alternative to Gaussian-process (GP) based Bayesian optimization (BO) for problems with many observations or BOMO. Traditional BO suffers from $O(N^{3})$ scaling due to GP fitting and hyperparameter optimization, which becomes prohibitive when data are plentiful. The authors introduce Epistemic Nearest Neighbors (ENN), a lightweight surrogate that estimates mean function values and epistemic uncertainty from K nearest neighbors. So no training, no kernel inversion and no hyper. optimisation. The query and acquisition cost with no hyperparameter fitting is $O(N)$. Because ENN’s uncertainty is uncalibrated, standard acquisition rules like UCB or EI are invalid; instead, the paper proposes a non-dominated sorting based acquisition, jointly optimizing for large predicted value and uncertainty.

### Strengths
- Linear scaling (O(N) -  avoids the GP fitting bottleneck, enabling thousands of observations.

- No hyperparameter fitting - eliminates costly hyper. optimization and matrix inversion.

- Deterministic surrogate - simple, stable, and fast to evaluate using nearest neighbors.

- Calibration-free acquisition - Pareto non-dominated selection, again without tuning, so fast.

- Comparable performance - achieves nearGP solution quality with 10–100× faster proposal times across noise free benchmarks.

### Weaknesses
- An obvious weakness is that all the numerical experiments are conducted in the noise free setting, the authors justify this choice by mentioning that many simulation based optimisation porblems are noise free. 
- Since ENN’s variance isn’t a probabilistic posterior, it cannot distinguish epistemic from aleatoric uncertainty, so handling noise properly would be non-trivial. The authors push this to future work but I think this is a big caveat and i am uncertain if this framework would stand in those settings. 
- The ENN surrogate takes no account of correlations between inputs as no posterior covariance is learnt. I believe this seriously limits its extrapolation abilities. 
- This method is not applicable in settings where either posterior modelling is critical (with full covariance) or where there is inherent noise. So the method is only applicable in a very limited setting. 

It’s basically a lazy learner, like K-nearest-neighbors regression repurposed for Bayesian optimization.

Typos and other minor issues:

 - Line 67, spelling
- Citations all over the paper are not in the correct format, they should be in parentheses for readability.

### Questions
- How does the performance of TuRBO-ENN depend on the dimensionality 
D, given that distance measures become less informative in high-dimensional spaces?
- How does TuRBO’s trust-region size interact with 𝐾 or the non-dominated sorting procedure in any systematic way?
- What are the theoretical or empirical limits of scalability.. at what 𝑁 or 𝐷 does the nearest-neighbor search itself become the new bottleneck?
- I am wondering if the performance improvements partly stem from TuRBO’s trust-region mechanism rather than the ENN surrogate itself?

### Soundness
2

### Presentation
2

### Contribution
2
