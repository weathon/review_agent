# Online Differential Privacy Bayesian Optimization with Sliced Wasserstein Compression

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 2, 6

## Abstract
The increasing prevalence of streaming data and rising privacy concerns pose significant challenges for traditional Bayesian optimization (BO), which is often ill-suited for real-time, privacy-aware learning. In this paper, we propose a novel online locally differentially private BO framework that enables zero-order optimization with rigorous privacy guarantees in dynamic environments. Specifically, we develop a one-pass Gaussian process compression algorithm based on the sliced Wasserstein distance, which effectively addresses the challenges of kernel matrix scalability, memory efficiency, and numerical stability under streaming updates. We further establish a systematic non-asymptotic convergence analysis to characterize the privacy–utility trade-off of the proposed estimators. Extensive experiments on both simulated and real-world datasets demonstrate that our method consistently delivers accurate, stable, and privacy-preserving results without sacrificing efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces an online locally differentially private (LDP) framework for Bayesian optimization (BO) tailored to streaming data environments. It proposes a zero-order optimizer using Gaussian processes (GPs) with a novel Sliced Wasserstein Compression (SWC) algorithm to bound kernel dictionary growth while maintaining numerical stability and privacy guarantees. The method embeds Gaussian noise into gradient approximations derived from the GP posterior, ensuring per-iteration LDP. Non-asymptotic convergence rates are provided for strongly convex and smooth (non-convex) losses, matching SGD-like performance without gradient access. Experiments on synthetic (linear, logistic, ReLU, Sine, Friedman) and real (Uber fares) datasets demonstrate superior accuracy and stability over LDP-SGD and non-private baselines, especially in nonlinear settings.

### Strengths
- Novelty: Fills a clear gap in online, gradient-free, LDP BO for streaming data. The SWC algorithm is innovative, using sliced Wasserstein distance to bound dictionary size $(O(1/\kappa)^p)$ while preserving posterior fidelity—extending batch private BO to dynamic, untrusted settings without historical data storage.

- Theoretical Rigor: Provides non-asymptotic convergence rates (Theorems 4.4–4.5) under decaying stepsizes, achieving SGD-optimal $O(t^{-\alpha})$ error in strongly convex cases and $O(t^{-(1-\alpha)})$ gradient norms in smooth non-convex settings. Bounds explicitly capture privacy $(\epsilon, \delta)$, compression $\kappa$, and BO approximation errors, with corollaries for RDP/GDP—stronger than heuristic privacy in prior online BO. Assumptions (e.g., smoothness, RKHS) are standard and justified.

- Practical Efficiency: $O(1)$ per-iteration time/space via SWC avoids $O(t^3)$ kernel inversions in standard BO. Algorithm 1 is clean and implementable; privacy via Gaussian mechanism is flexible (extendable to Laplace).

- Empirical Validation: Thorough experiments across parametric (linear/logistic/ReLU) and nonparametric (Sine/Friedman) synthetic tasks, plus real Uber fares data. LDP-BO outperforms LDP-SGD in MSE/prediction error, especially nonlinearly, with stable convergence under varying $\epsilon / \kappa$. Function fitting plots visually confirm better generalization vs. non-private DNN. Reproducibility statement and code promise are excellent.

- Clarity and Structure: Well-written with intuitive flowchart (Fig. 1), detailed appendices (e.g., DP preliminaries, proofs), and ethical/reproducibility statements. Related work comprehensively covers BO, online learning, and private estimation.

### Weaknesses
1. Assumption Strength: Bounded sensitivity (Ass. 4.1) and RKHS membership (Ass. 4.2) are common but may limit applicability to heavy-tailed or non-stationary streams (e.g., real-world sensor data). Strong convexity (Ass. 4.3) yields optimal rates, but non-convex analysis (Thm. 4.5) only guarantees stationarity—discuss implications for multimodal BO objectives more. Assumption 3.2 (non-expansive compression) is mild but relies on prior consistency (Lemma C.1); empirical verification would help.
2. Experimental Scope: Baselines are appropriate (LDP-SGD, non-private BO/SGD/DNN), but missing comparisons to other scalable GPs (e.g., inducing points in Balandat et al., 2020) or private online methods (e.g., DP contextual bandits from Ding et al., 2021). Real-data ablation on $\kappa$ sensitivity or varying $\delta$ is absent; Uber fares is compelling but single-domain—add multi-domain (e.g., finance, healthcare) for broader impact. No runtime/memory plots to quantify $O(1)$ claims vs baselines.
3. Privacy-Utility Trade-off: While theory bounds noise (e.g., $O(B^2 \log(1/\delta)/(\lambda \epsilon^2))$), experiments use fixed/varying $\epsilon$ but not composition over $t$ (e.g., total $\epsilon_t$ sum). Discuss practical budget allocation (e.g., decaying $\epsilon_t$) or advanced mechanisms (e.g., RDP for tighter composition). SWC privacy amplification isn't analyzed—does compression interact with LDP?
4. Minor Technical/ Presentation Issues: Acquisition function (GI in Eq. 3) optimization isn't detailed (e.g., how to solve argmin GI efficiently online?). Theorem 3.1 claims per-iteration LDP but composes to $\max{\epsilon_1,...,\epsilon_t}$-LDP—clarify if adaptive $\epsilon_t$ allows better total budget. Appendix E details are good but could include hyper parameter tuning (e.g., $\eta_t$ choice).

### Questions
1. SWC Mechanism (Sec. 3.2, Alg. 2): The use of sliced Wasserstein for compression is elegant and avoids density estimation pitfalls. However, computing $SW_2(\rho_{D_{-j}}, \rho_{\tilde{D}_t})$ for each $j$ in $I$ seems $O(M_t^2)$ per iteration (with $m=100$ projections)—how does this achieve true $O(1)$ amortized? Suggest approximating via subsampling directions or low-rank updates. Besides, if line 8 is not satisfied in the first iteration of the loop, does that mean it cannot obtain a compressed dictionary $D_t$, and is it necessary to dynamically adjust the size of $\kappa$? Also, Thm. 3.3 bounds $M_t ≤ O((1/\kappa)^p)$, but $p=5$ in expts: discuss curse-of-dimensionality for high-$p$ (e.g., $>10$).
2. Privacy Integration (Sec. 3.1, Thm. 3.1): Clipping to $B$ before Gaussian noise is standard, but sensitivity $| g_t - \tilde{g}_t | ≤ 2B$ assumes $| \mu |$ bounded—link to Ass. 4.1 more explicitly. Why Gaussian over Laplace (better for $l_1$-sens)? Corollaries B.1–B.2 are nice extensions; consider empirical comparison of RDP/GDP noise levels. Question: In streaming, how to handle adaptive adversaries querying outputs?
3. Theoretical Analysis (Sec. 4): Proofs are solid (e.g., leveraging projection non-expansiveness in Lem. C.3), but the BO error term $(L + p\kappa)$ assumes GI minimization over $p+1$ points (Wu et al., 2023)—is this feasible online? For non-convex (Thm. 4.5), the $O(1/T^{1-\alpha})$ rate is good, but compare quantitatively to non-private BO rates (e.g., Müller et al., 2021). Minor: $t_0$ definition in Thm. 4.4 could be explicit.
4. Experiments (Sec. 5): Strong results, but add ablation: (i) MSE vs $\kappa$ (e.g., $0.05–0.5$) on Uber data; (ii) total privacy budget $(\Sigma, \epsilon_t)$ impact; (iii) dictionary size evolution over $t$ to verify bounded $M_t$. Runtime: LDP-BO vs. LDP-SGD on $t=20k$? Fig. 2/5: Why focus on first coeff? Average over all or report full $| \theta |_2$ error.

5. Broader Impact/Ethics: Privacy in streaming BO enables sensitive apps (e.g., personalized medicine), but discuss failure modes (e.g., utility collapse at $\epsilon<<1$) or biases in SWC (e.g., projection directions). Reproducibility: Provide exact $\eta_t$ (e.g., $\alpha=0.75$?), kernel choice (RBF?), and seed for 100 reps.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel online, gradient-free Bayesian optimization (BO) framework with local differential privacy (LDP) guarantees, addressing the limitations of traditional BO in dynamic environments. Experimental results show significant performances of the proposed framework.

### Strengths
1.	This is the first work to tackle online Bayesian optimization under the local differential privacy model. 
2.	The proposed Sliced Wasserstein Compression (SWC) is a novel algorithm that can manage the kernel dictionary in streaming data environments.
3.	The paper is well-organized, with a clear introduction, detailed methodology, thorough theoretical analysis, and comprehensive experimental results.

### Weaknesses
1.	The SWC algorithm requires multiple calculations of the slice Wasserstein distance, which consumes more computing resources.
2.	The method introduces several new hyperparameters: the compression budget, and the clipping bound. The sensitivity of the algorithm to these choices is not explored.
3.	The paper could enhance its originality by more clearly distinguishing its contributions from existing methods.

### Questions
1.	Why is it necessary to use the SWC method specifically for compression? What is the innovation of SWC?
2.	How does this method perform in high-dimensional environments?
3.	How to set compression budget, and the clipping bound in practice. Is performance robust to their selection, or does it require extensive tuning?

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
This paper introduces an online, locally differentially private (LDP) framework for Bayesian Optimization (BO) in streaming data environments. The proposed algorithm, termed LDP-BO, enables gradient-free, privacy-preserving, and memory-efficient optimization by integrating two key components: (1) a Sliced Wasserstein Compression (SWC) scheme for maintaining a bounded-size Gaussian process (GP) kernel dictionary, and (2) a per-iteration LDP mechanism that injects Gaussian noise directly into the gradient-free update rule.

The authors provide non-asymptotic convergence guarantees under both strongly convex and general smooth loss functions, showing that the method achieves SGD-like rates while ensuring local privacy. Theoretical analysis is supported by experiments on synthetic regression models, nonlinear function fitting, and a real-world Uber fares dataset, demonstrating lower mean-squared error and higher stability than LDP-SGD baselines.

### Strengths
1. The paper proposes a novel synthesis of online learning, Bayesian optimization, and local differential privacy. The integration of Sliced Wasserstein Compression for kernel dictionary control within an LDP framework is creative and technically non-trivial.

2. The theoretical contributions are strong and carefully derived. The non-asymptotic convergence bounds explicitly separate the effects of privacy noise, compression error, and initialization, offering clear interpretability.

3. The paper is well-written, with precise notation and a clear exposition of algorithms and assumptions. The inclusion of supporting lemmas and appendices strengthens reproducibility and theoretical transparency.

4. Addressing privacy-preserving online Bayesian optimization fills a clear gap in the literature, as prior DP-BO work focused mainly on batch or centralized settings. The method provides a promising foundation for future research on online private learning and real-time optimization.

### Weaknesses
1. While the technical setup is solid, the paper could better articulate why online Bayesian optimization under LDP is practically necessary. The current motivation remains abstract, lacking concrete scenarios (e.g., edge devices, adaptive sensor control) that require both streaming and privacy simultaneously.

2. Although SWC ensures bounded dictionary size, it remains unclear how compression affects optimization accuracy or convergence in practice. The theoretical bounds include a κ-dependent error term, but there is no empirical or interpretive analysis showing how varying κ impacts the privacy–utility trade-off.

3. The SWC algorithm, while theoretically elegant, may incur non-trivial computational overhead due to repeated projection and sorting operations in high dimensions. The paper lacks runtime benchmarks or scalability analysis to substantiate the claimed O(1) per-iteration cost.

4. The experiments, though diverse, remain relatively small-scale. There are no comparisons against recent scalable or privacy-aware BO methods (e.g., DP-BO via posterior sampling or distributed DP frameworks). Moreover, results focus mainly on MSE rather than BO metrics such as regret or sample efficiency.

5. The algorithm assumes per-iteration privacy budgets and full control over the noise mechanism, which may not be feasible in realistic streaming environments or federated deployments.

### Questions
1. Could the authors provide a clearer interpretation of the privacy–utility–compression trade-off, perhaps through ablations varying ε, κ, and step size η?

2. The Sliced Wasserstein Compression is the most distinctive element. Could the authors demonstrate its empirical effect (e.g., compare with random feature truncation or Nyström methods)?

3. How does LDP-BO handle non-stationary streaming data, where the underlying distribution shifts over time?

4. The theoretical convergence rates resemble those of noisy SGD. Can the authors clarify what benefits BO (as a surrogate model) provides over standard LDP-SGD in this setting?

5. Could the method extend to Reinforcement Learning or contextual bandit frameworks where privacy and exploration–exploitation interplay is central?

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
3

### Summary
An  online locally differentially private Bayesian optimization (LDP-BO) framework that enables zero-order optimization with rigorous privacy guarantees in dynamic environments is propoised in this work, which conducts Bayesian optimization (BO) in an online, streaming setting and provides Local Differential Privacy (LDP) guarantees. A systematic non-asymptotic convergence analysis to characterize the privacy–utility trade-off of the proposed estimators is carried out as well as experiments on both simulated and real-world datasets.

 demonstrate that our
method consistently delivers accurate, stable, and privacy-preserving results with-
out sacrificing efficiency.

The core technical contributions are a zeroth-order LDP optimizer based on Gaussian Process (GP) surrogate models and a Sliced Wasserstein Compression (SWC) algorithm to keep the GP's kernel dictionary from growing unboundedly.

### Strengths
- LDP-BO framework, enables zero-order optimization with rigorous privacy guarantees in dynamic environments.
- The framework for complex objective fuctions, thru a zeroth-order optimizer that eliminates the need for gradient information.
- Non-asymptotic analysis for the proposed framework.

### Weaknesses
- Ablations and sensitivity are thin.
- There are no comparisons to private kernel/GP baselines, private BO/bandit methods, or online CDP/LDP surrogates; the Uber-fares experiment even compares private GP to a linear LDP-SGD, which is mismatched in model capacity.
- Assumptions are strong.

### Questions
- Sensitivity and mechanism calibration need tightening. 
- Convergence results require bounded gradients (Assump. 4.1), RKHS membership of 𝐿(⋅,𝑧_𝑡) (Assump. 4.2), and strong convexity/smoothness (Assump. 4.3) for Theorem 4.4, which do not hold for many realistic BO targets.

### Soundness
3

### Presentation
2

### Contribution
3
