# High-Order Dynamics Modeling of Time Series with Attractor-Guided Adaptive Filtering

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Explicit, equation-discovery models promise transparent mechanisms and strong extrapolation for time-series dynamics. Yet most existing methods impose first-order structure, even when the true system depends on multiple lags. This mismatch is typically absorbed by inflating the latent state via ad-hoc augmentation, which erodes identifiability, complicates learning, and weakens interpretability. Compounding the issue, defaulting to Kalman-style updates in nonlinear or weakly stable regimes is brittle: inference degrades away from fixed points, biasing parameter estimates and reducing predictive reliability.

We introduce a framework for \emph{adaptive high-order dynamics modeling}. Given an $m$-dimensional series, we \emph{initialize the latent dimension to $m$} and estimate the Markov order $p$—the minimal number of past states needed to predict the next—via a conditional mutual information test. Rolling statistics assess proximity to attractors and drive \emph{stability-aware} filter selection. Starting from $(p,m)$, an inference–learning loop evaluates candidate structures and guides a unidirectional search that converges to $(\hat p,\hat m)$ together with the associated system parameters. Across benchmark datasets, the resulting models yield more flexible latent dynamics and consistently improve predictive accuracy over state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a method for explicit high-order dynamics modeling of time series data. The authors argue that many dynamical systems exhibit dependencies extending beyond first-order Markov assumptions, and that collapsing these dependencies into a large latent dimension hinders interpretability and stability. Their proposed framework begins by estimating an initial order using conditional mutual information (CMI), then identifies attractor proximity through rolling mean and covariance metrics to switch between Extended Kalman Filtering (EKF) and Particle Filtering (PF). The learning proceeds through an EM-style loop that alternates between state inference and parameter updates, while performing a directed search over model order (p) and latent dimension (m). Experiments on synthetic systems and the dysts benchmark suite suggest improved coefficient recovery and prediction accuracy compared to baselines such as LaNoLeM and MIOSR.

The paper addresses the problem of learning explicit high-order latent dynamics with stable and interpretable structure. The theoretical framing is coherent, and the experimental results on purpose-built systems support the main claims. However, the evidence on general benchmarks is weaker and somewhat orthogonal to the core contribution. The significance of the work lies in unifying order detection, adaptive filtering, and stable parameter learning within a single pipeline. This is a valuable conceptual contribution, though further empirical validation is needed to establish its robustness and scalability.

### Strengths
The paper is well motivated: most latent dynamics models assume first-order structure, yet many physical systems exhibit higher-order dependencies. The use of conditional mutual information for order detection is principled and nonparametric. The attractor-proximity metrics are intuitive, computationally simple, and seem ideal for changing dynamic regimes. The augmented first-order formulation for learning p-th order systems is effectively implemented, and the inclusion of a structured regularizer for the top-level matrix enhances identifiability. The experimental section demonstrates that the proposed method recovers system coefficients faithfully and switches filters in a manner consistent with stability predictions.

### Weaknesses
Despite its motivation, several aspects of the method remain heuristic or underexplored. The directed search over (p, m) lacks guarantees and is only briefly evaluated; its robustness to initialization and to model mis-specification is unclear. The attractor diagnostics based on rolling statistics could confound nonstationarity or slow drift with instability, and there is little quantitative analysis of false-switch rates or sensitivity to the window parameter. While the experiments on synthetic high-order systems validate the main premise, the tests on dysts benchmarks primarily probe filtering robustness rather than the proposed benefit of higher-order modeling. Reporting is another weakness: variance across runs, statistical significance, and runtime scaling with dimension and sequence length are not provided. Finally, the identifiability discussion is incomplete; although the authors acknowledge non-uniqueness in (p, m) decompositions, they offer no diagnostic to determine when differing solutions represent equivalent dynamics.

### Questions
1.	How sensitive is the directed search over (p, m) to the initialization (p0, m0)? Have you benchmarked its convergence against a small exhaustive grid search?
	2.	How frequently does the attractor-guided filter switch incorrectly under stationary but noisy conditions, and how sensitive is this to the window length W?
	3.	Can you quantify the computational cost (particle counts, EM iterations, wall-clock time) for both the EKF and PF regimes as p and m grow?
	4.	When the learned (p, m) differs from the ground truth, can you show that the resulting dynamics are equivalent under augmentation or reparameterization?

To strengthen the paper, it would help to (1) include ablations that isolate the contribution of each component (CMI initialization, attractor diagnostics, and elastic-net regularizer), (2) compare against explicit high-order baselines such as latent AR(p) models, (3) quantify the reliability of the EKF/PF switching mechanism, and (4) report variance and runtime to assess scalability. A case study on a real system known to exhibit delay or memory effects would also enhance external relevance.

### Soundness
2

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
The paper introduces a framework for modeling time series using higher-order state-space representations. It proposes an initialization heuristic for the Markov order and state dimension. Then, the algorithm goes through an inference-learning loop to both learn the model parameters and the optimal Markov order and state dimension. In the inference stage, the paper uses an adaptive filtering method based on stability proximity. The algorithm is tested on both synthetic and real datasets.

### Strengths
* The problem is well-motivated, making it easy to understand the challenge and what the paper is trying to solve.

* The Preliminaries section was written clearly and easy to follow. Explicitly showing the matrices (e.g., Equation 25) helps the reader understand the operations.

### Weaknesses
* There are several places in the paper where the boundary between the text and citation is unclear, consequently making sentences grammatically wrong and hard to follow. In addition, Figures and Tables do not have captions that explain the key notations and takeaways, which are crucial to help the reader to understand them easily without referring to the main text. Finally, based on the results shown in the paper, it is hard to understand the key takeaways and conclude which method is doing how much better. I think the authors could improve how the tables/results are summarized and also use plots to qualitatively show what the differences in the metrics mean intuitively.

* The paper lacks an explanation of how some of the heuristics are chosen, making the proposed method less sound. For example, it is unclear why L_0 in equation (20) is defined the way it is.

### Questions
* Could you please explain the line after equation (21), where it says “Near a stable equilibrium…hence m_t -> 0”?

* How does the proposed method depend on how the baseline in equation (20) is chosen? In other words, how sensitive is the method to the estimated baseline?

* I’m not sure if the synthetic data experiments in Table 1 are high-order and high-dimensional. What are the dimensions in each system? Could you stress test the proposed method on even higher-order systems?

### Soundness
2

### Presentation
1

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
This paper presents an adaptive framework for learning interpretable state-space models that explicitly capture higher-order temporal dependencies. The method estimates both the minimal Markov order and latent state dimension through a structured search guided by conditional mutual information and validation loss. It integrates a stability-aware hybrid inference scheme, switching between Extended Kalman and particle filters depending on attractor proximity, using local stability metrics to balance efficiency and robustness. The framework is evaluated on synthetic and benchmark dynamical systems, where it consistently outperforms prior baselines such as LaNoLeM and MIOSR in reconstructing system coefficients, recovering attractor structures, and improving predictive accuracy, particularly in nonlinear or noisy regimes.

### Strengths
The paper introduces a conceptually elegant and interpretable formulation that unifies high-order dynamical modeling, attractor-aware filtering, and structural adaptivity within a single framework. Its principled use of conditional mutual information for order estimation and attractor-guided switching yields strong empirical gains while maintaining transparency in learned representations. The results demonstrate notable improvements in both parameter recovery and forecasting fidelity across multiple synthetic systems, validating the method’s robustness to noise and nonlinearities. The work is well-motivated, technically detailed, and addresses an important gap between black-box sequence models and interpretable dynamical systems modeling.

### Weaknesses
Despite strong conceptual contributions, the approach is computationally demanding due to repeated inference–learning loops and structured searches over orders and dimensions, with no formal scalability analysis. The filter-switching heuristic, while effective, may introduce sensitivity to noise and threshold tuning, and its stability under rapidly varying regimes is not empirically tested. The framework’s reliance on polynomial basis functions limits its expressiveness for systems with non-polynomial or discontinuous dynamics, and interpretability may diminish as polynomial order grows. 

Furthermore, while there is a rich existing literature on recovering dynamics through Hankel-based methods, empirical dynamical modeling (EDM) and and it deep learning based variant DeepEDM,  the paper provides limited discussion or comparison to these approaches, which could contextualize the proposed method’s novelty and contribution more clearly.

Finally, the experiments are confined to synthetic and benchmark datasets, leaving the real-world applicability and generalization performance unexplored.

### Questions
1. How does the method scale computationally with sequence length and latent dimension compared to baselines?
2. How robust is the attractor-guided filter switching under noise or nonstationary dynamics?
3. Can the approach handle non-polynomial or discontinuous systems, or be extended beyond polynomial bases?
4. Strengthening the related work section would further improve the quality of the work.
5. The color scheme and design of Fig 1 could be improved.

### Soundness
3

### Presentation
3

### Contribution
3
