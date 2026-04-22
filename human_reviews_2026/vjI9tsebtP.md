# One Intervention per Component is Enough: Towards Identifiability in Linear Stochastic Dynamics from Steady State

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
We study the problem of recovering the parameters of a multivariate Ornstein–Uhlenbeck (OU) process from steady-state observational and interventional data. In many applications, such as large-scale gene perturbation experiments, only stationary “snapshot” measurements are available, making standard stochastic differential equation estimation methods that rely on time-series trajectories inapplicable.
We first establish an identifiability result: one intervention per strongly connected component (SCC) of the drift graph suffices to recover all OU process parameters generically up to a global scaling factor. This holds provided that the SCC condensation graph is connected with a single root and certain spectral nondegeneracy assumptions hold. We propose a recursive learning algorithm that orders SCCs topologically and, for each component, isolates its marginal dynamics and solves a linear system derived from the steady-state moment equations, leveraging parameters recovered for upstream components.
Building on this theoretical foundation, we propose a regularized least-squares estimator that jointly minimizes residuals of the steady-state mean and covariance equations across observational and interventional data. Experiments on synthetic and real datasets demonstrate the effectiveness of our method in recovering parameters and predicting unseen interventions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper analyzes identifiability of linear stochastic dynamics (Ornstein–Uhlenbeck processes) from steady-state observational and interventional data.
It proves that one intervention per strongly connected component (SCC) suffices to identify the system up to a global scale under a “row-zeroing” intervention model and diagonal diffusion.
A regularized least-squares estimator is proposed, with synthetic and Perturb-seq experiments illustrating the approach.

### Strengths
•	Clear theoretical framing and well-presented identifiability proofs.
	•	The “one intervention per SCC” condition is simple and interpretable.
	•	Empirical evaluation includes both synthetic and real (Perturb-seq) datasets.

### Weaknesses
•	The intervention model (zeroing off-diagonals of one row while keeping the diagonal) is restrictive (independent dynamics) and not biologically realistic.
	•	The estimator is a straightforward least-squares fit; no methodological innovation beyond the theoretical analysis.
	•	No comparison to existing system-identification or Optimal Experimental Design (OED) approaches
	•	Focused entirely on linear OU dynamics, limiting relevance to broader ML audiences.
	•	Overall contribution is more aligned with causal inference and system identification than with core machine-learning or representation-learning research, not as relevant to the ICLR community.

### Questions
1.	How sensitive is the identifiability result to imperfect or partial interventions?
	2.	How can extensions be handled for non-diagonal diffusion or nonlinear dynamics?
	3.	Have you compared against classical gradient-based or Bayesian OED estimators using the same steady-state moments?

### Soundness
3

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
The paper studies identifiability of linear multivariate OU dynamics from steady-state observational + interventional snapshots. Its core result: one intervention per strongly connected component (SCC) identifies parameters up to a global scale when the SCC condensation graph is connected with a single root and mild spectral non-degeneracy holds. The proofs tie directly to a practical estimator that jointly fits means and covariances. Experiments on synthetic and single-cell data show sensible recovery and prediction for unseen interventions.

### Strengths
1. **Well-motivated problem.** 
The paper targets the steady-state snapshot regime, where trajectories are unavailable, which is common and practical in applications such as single-cell perturbation screens.
2. **Concise identifiability guarantee.**
It shows that a single intervention per SCC (under a single-root condensation DAG) is sufficient to identify parameters up to a global scale, relaxing prior requirements that often needed interventions on many or most nodes.

### Weaknesses
1. **Missing basic definitions in the main text.**
Key notions: e.g., strongly connected component (SCC) and steady-state mean/covariance, are not clearly defined where they first appear. A brief one-sentence explanation for each would improve accessibility for readers who are not specialists in stochastic dynamics.

2. **Assumptions are hard to map to results.**
Assumptions 1–3 are introduced in the appendix, making it difficult to see how each supports a specific theorem. Providing a short, intuitive explanation next to each theorem (what the assumption rules out and why it is needed) would significantly aid comprehension.

3. **Main-text proof density.**
The detailed proof of Theorem 3 occupies a full page and renders the exposition equation-heavy. Consider moving the derivation to the appendix and keeping a high-level proof sketch in the main text.

4. **Restrictive diffusion model.**
The assumption of diagonal diffusion is strong. Many real systems exhibit correlated noise. Some discussion of robustness or extensions would be helpful.

### Questions
1. **Generic vs. strong assumptions.**
Which assumptions are intended to be generic (holding except on a measure-zero set) and which are strong modeling or structural? Please add brief intuitive explanations for each assumption and, if feasible, a theoretical discussion of genericity (beyond the numerical evidence in Appendix C).

2. **Resolving the global scale.**
The identifiability result is up to a global scaling factor. For applications requiring absolute parameters, what concrete strategies do you recommend to fix the scale (e.g., anchoring a known rate/variance or using an external calibration)?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the identifiability of linear stochastic dynamical systems (multivariate OU processes) when only snapshot data is available under both observational and interventional conditions. The paper proves that one intervention per strongly connected component (SCC) of the underlying graph is sufficient to identify the entire system up to a single global scaling factor. They also provide a recursive algorithm that learns the SCC structure from interventional mean shifts and recovers parameters within each SCC using steady-state mean and covariance equations jointly. A practical regularized least-squares objective is proposed to fit all observed (interventional and observational) moments and tested in synthetic and real-world experiments.

### Strengths
- the task of recovering causal dynamics from steady-state data is very relevant in scientific and machine learning settings
- the rule behind the one intervention per SCC is clean and simple, offering practical guidance for experimental design
- the theoretical results are rigorous and supported by synthetic and real-data experiments
- the focus on "snapshot" data fits many real-world applications, such as biology, where temporal measurements are often available

### Weaknesses
- since in this linear setting identifiability directly corresponds to causal discovery, the related work section could better emphasize this by referring to similar works in the field. See questions below.
- the paper refers to strongly connected components (SCCs) but it seems that it does not provide a definition. I would recommend adding it
- perfect interventions is not really realistic in a real-world setting, especially in biological perturbations. I think that the theoretical results are interesting, however I am not sure how insightful the real-world experiments are. This learning framework is based on strong modeling assumptions (linear SDE, diagonal diffusion, perfect interventions) which are likely not met in these applications. Perhaps this could be mentioned by the authors.

### Questions
- I am missing the link to these references on causal discovery within dynamical systems. Can the authors elaborate on these? For causal discovery of SDEs [1,2] and with interventions [3,4]. 
- how can these results be used for practical guidance in experimental design if one does not know a priori which ones are the SCC?
- Remark 1 states that spectral nondegeneracy assumptions hold generically within SCCs, however how does this extend to sparse settings? 

[1] Boeken, Philip, and Joris M. Mooij. "Dynamic structural causal models."

[2] Manten, Georg, et al. "Signature kernel conditional independence tests in causal discovery for stochastic processes." (2024).

[3] Hansen, Niels, and Alexander Sokol. "Causal interpretation of stochastic differential equations." (2014).

[4] Zweig, Aaron, et al. "Towards Identifiability of Interventional Stochastic Differential Equations." (2025).

### Soundness
3

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
The paper studies the problem of recovering the parameters of a causal linear system given by a multivariate Ornstein-Uhlenbeck (OU) process from steady-state observational and interventional data. The drift matrix \Lambda is the key parameter of the OU process that differentiates observational and interventional data. Observational data for the $i$th random variable (RV) is sampled from an underlying OU process. Interventional data of $i$th RV is obtained from a modified OU process driven by $\tilde \Lambda^{(i)}$, which isolates $i$ from its regulators ($j\neq i$). The OU parameters are recovered from observational and interventional first- and second-order moments $\mu$ and $\Sigma$, and $\tilde \mu$ and $\tilde \Sigma$. The paper is theory-focused. A (drift) graph $\mathcal{G}(\Lambda)$ is formed by non-zero entries of the drift matrix $\Lambda$. The authors prove that under some conditions, if $\mathcal{G}$ has $N$ strongly connected components (SCCs), then the number of interventions necessary to estimate the parameters of OU up to a scaling factor equals $N$ (one intervention per SCC). Based on the analysis, the authors propose a least-squares estimator of OU parameters. Empirical simulations on real and synthetic datasets demonstrate good performance and corroborate theoretical results.

### Strengths
- The paper conducts a thorough theoretical analysis of how interventions aid the estimation of OU parameters
- The estimator is developed from the theoretical analysis, quantifying the information needed to estimate the parameters
- It is well-written, supported by a number of recent publications
- Much of prior work focused on either identifiability of OU parameters or interventions to learn stationary stochastic differential equations (SDE) models. Only very recent work (Rohbeck’24) combined both. The authors build on that work and provide identifiability results under weaker assumptions.
-  Beyond the theoretical identifiability results, the paper contributes a concrete estimation algorithm that operationalizes the theory. The proposed regularized least-squares estimator jointly fits the steady-state mean and covariance equations under observational and interventional data, incorporating sparsity priors on the drift matrix. This bridges the gap between abstract identifiability proofs and practical parameter recovery. The algorithm is also evaluated on both synthetic and real biological datasets (Perturb-seq), demonstrating that the theoretical insights translate into tangible performance gains and confirming the model’s robustness in realistic, noisy settings.

### Weaknesses
- Insufficient number of baselines. The authors reference a number of relevant works. However, both experiments compare only against a single baseline, which is “similar” to prior work. Hence, it is difficult to evaluate the relative performance. In my opinion, it would be helpful to compare against traditional and non-interventional estimators.
- Lack of empirical support. The main claim is that we only need a single intervention per SCC. However, in the experiment on the synthetic dataset, the authors only analyze whether the drift graph meets conditions from Thm. 3. What about generating a graph with $N$ SCCs and showing that, on average, $N$ interventions (one per SCC) are needed for a good estimate? For the number of SCCs smaller than $N$, the error is expected to increase, while for larger numbers, it would marginally decrease. One could also add directed two-cycles to confirm that the gain w.r.t. the work of Dettling’23 is even more significant in this case.
- Minor clarity comments. One or two sentences in the main text describing what the topological order of SCCs intuitively means (Remark 2) would help. Also Intervention definition (l. 107) should be $\Lambda_{kj}$ in the first row. Lastly, $x_\inf$ (l. 092) and $\oplus$ (l. 215) are undefined.

### Questions
- (10) assumes that we have access to observational and interventional moments. How sensitive is this model to the measurement errors? What if some of the parameters (e.g., diffusion matrix) are misspecified? 
- An experiment on a real dataset states that the scaling factor cancels out when estimating the first-order moment $\mu$. But would it also cancel when estimating the covariance $\Sigma$? If it does not cancel out, what are the practical implications of this scaling ambiguity?
- Which assumptions from Thm. 3 can be relaxed without losing identifiability?

### Soundness
4

### Presentation
3

### Contribution
3
