# Linear Causal Representation Learning by Topological Ordering, Pruning, and Disentanglement

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Causal representation learning (CRL) has garnered increasing interests from the causal inference and artificial intelligence community, due to its capability of disentangling potentially complex data-generating mechanism into causally interpretable latent features, by leveraging the heterogeneity of modern datasets. In this paper, we further contribute to the CRL literature, by focusing on the stylized linear structural causal model over the latent features and assuming a linear mixing function that maps latent features to the observed data or measurements. Existing linear CRL methods often rely on stringent assumptions, such as accessibility to single-node interventional data or restrictive distributional constraints on latent features and exogenous measurement noise. However, these prerequisites can be challenging to satisfy in certain scenarios. In this work, we propose a novel linear CRL algorithm that, unlike most existing linear CRL methods, operates under weaker assumptions about environment heterogeneity and data-generating distributions while still recovering latent causal features up to an equivalence class. We further validate our new algorithm via synthetic experiments and an interpretability analysis of large language models (LLMs), demonstrating both its superiority over competing methods in finite samples and its potential in integrating causality into AI. Source code is available at [the anonymous link](https://anonymous.4open.science/r/creator-883D/).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents CREATOR, a novel algorithm for linear causal representation learning (CRL) from heterogeneous environments. The work focuses on a setting with a linear structural causal model over the latent variables and a linear mixing function mapping these latents to observations. The proposed CREATOR algorithm operates in three stages: inferring a topological ordering of the latent variables, pruning the resulting dense graph to identify the sparse causal DAG, and finally disentangling the latent features to the underlying causal DAG.

### Strengths
The main strength of this work lies in its successful relaxation of several restrictive assumptions made in the recent, closely related work of Jin & Syrgkanis (2024). Specifically, the proposed method does not require the distribution of exogenous noise to be identical across different environments, nor does it assume that different noise components within the same environment must have different distributions.

### Weaknesses
1. The manuscript's strong assumption of a linear model (linear SCM and mixing function) limits its applicability to complex, real-world scenarios which are often nonlinear.

2. The problem setup assumes all environments share the same observed variables, whereas in reality, environments may have only partially overlapping sets of observed variables. Considering this case would better reflect real-world conditions.

3. The manuscript is technically dense and lacks intuitive examples or explanations for its core mechanisms, making it difficult to follow.

4.  Similar to Jin & Syrgkanis (2024), a discussion comparing this work with the following papers is necessary:
    * A versatile causal discovery framework to allow causally-related hidden variables, ICLR, 2023.
    * Generalized independent noise condition for estimating causal structure with latent variables, JMLR, 2024.

5.  The manuscript claims that (line 345) it “yields a more efficient pruning procedure” than the method in Jin & Syrgkanis (2024). To substantiate this claim, an explicit efficiency comparison is needed, which would further enhance the contribution of the work.

### Questions
1. Lines 89–90 state that the algorithm can ``provably identify latent features and their causal mechanisms up to an equivalence class,'' while line 193 claims that ``the latent features and causal DAG can be uniquely recovered.'' This appears somewhat contradictory, as an equivalence class is not a unique DAG. Clarification would be appreciated.

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
The paper introduces CREATOR, a new algorithm for linear causal representation learning (CRL) designed to relax some of the assumptions made in LiNGCReL (Jin & Syrgkanis, 2024).
The model assumes a linear structural causal model over latent variables and a linear mixing to observations. The method consists of three stages: (i) ordering and feature recovery using independence criteria, (ii) DAG pruning based on rank analysis across multiple environments, and (iii) feature disentanglement to recover latent variables up to a structural equivalence class.
Theoretical claims are supported by identifiability proofs relying on non-Gaussianity and independence arguments, and empirical validation is provided on small synthetic datasets and a qualitative case study on large language models (LLMs).

### Strengths
* The paper is well structured and formally presented. The proposed model formulation is clear and consistent with recent linear CRL frameworks such as those by Squires et al. (2023) and Jin & Syrgkanis (2024).
* The attempt to connect linear CRL with representation analysis in large language models is conceptually appealing and relevant to ongoing discussions on the linear representation hypothesis (Arora et al., 2016; Park et al., 2024).
* The relaxation of environment-specific assumptions, allowing heterogeneous noise distributions, addresses a practical limitation of earlier models and enhances the potential applicability of the method.

### Weaknesses
* **Assumption relaxation and identifiability.**
  The claim of weaker assumptions is debatable. Although CREATOR removes the requirement of identical noise across environments, it still depends on strong non-Gaussianity and independence conditions, similar to those used in LiNGAM-based approaches. The improvement over LiNGCReL thus appears more incremental rather than fundamental.

* **Dependence on ICA and independence testing.**
  The algorithm’s first stage heavily relies on ICA and associated independence measures, meaning the identifiability largely stems from ICA theory rather than a novel mechanism. In addition, the theoretical condition for independence is replaced in practice by the HSIC statistic (Gretton et al., 2005), which lacks formal consistency guarantees and is known to be unstable in empirical causal discovery (Rolland et al., 2022).

* **Fragility of pruning and disentanglement stages.**
  To the best of my understanding, the rank-based pruning assumes exact linear independence across environments and may be highly sensitive to small perturbations or finite-sample noise. This can propagate errors into the disentanglement stage, resulting in cascading inaccuracies, a problem noted in previous multi-stage causal methods (Varıcı et al., 2024b). Competing approaches such as Buchholz et al. (2023) adopt more robust regularization strategies.

* **Empirical limitations.**
  The synthetic experiments are limited to low-dimensional data (d ≤ 7, n = 1000) and only compared to LiNGCReL, without evaluation against broader baselines such as Squires et al. (2023) or nonlinear CRL models. The LLM case study is based on a pre-defined DAG and qualitative interpretation of latent “concepts,” making causal conclusions difficult to validate. Moreover, no runtime, ablation, or stability analyses are reported.

* **Presentation and clarity.**
  Some notation, such as the equivalence relations ($\sim \pi$, $\sim \Delta$, $ \sim sur$) is introduced formally but lacks intuitive explanation. Including a small illustrative example (e.g., a 3-node toy system as in Ahuja et al., 2023) would improve readability.

### Questions
* Could the authors clarify in what specific sense Assumptions 1–3 are weaker than those in Jin & Syrgkanis (2024)? Does the notion of identifiability here correspond to ancestral, Markov, or exact equivalence?
* How is the HSIC test implemented in practice (kernel choice, thresholds), and how sensitive is the ordering step to deviations from true independence?
* Could disentanglement quality be assessed with additional metrics, such as mutual information or subspace similarity, beyond LocR²?
* In the LLM study, how do the authors verify that the assumed DAG reflects causal rather than syntactic relations? Would it be possible to compare the recovered features to known linguistic embeddings for validation?

### Soundness
3

### Presentation
2

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
The paper introduces CREATOR (Causal REpresentation leArning via Topological Ordering, Pruning, and Disentanglement), a linear causal representation learning (CRL) method that assumes (i) a linear SCM over latent variables and (ii) a linear mixing from latents to observations. Relative to prior linear-CRL work, the method weakens distributional assumptions by requiring only non-Gaussian, independent noise (≤1 Gaussian component) and allows noise distributions to vary across environments. Synthetic experiments show improved LocR² (latent recovery) and SHD (DAG accuracy) versus the previous LiNGCReL algorithm; a small LLM case study illustrates feasibility under the “linear representation hypothesis.”

### Strengths
- Assumption 1 permits environment-specific noise distributions and only requires non-Gaussian independent components ($\leq 1$ Gaussian), and Theorem 1 establishes identifiability up to ∼sur with ≥ d environments. This is a meaningful step beyond prior linear-CRL assumptions.

- On synthetic tasks across $d\in\{2,3,5,7\}$ and $K\in\{d,2d\}$, CREATOR improves LocR² and SHD over LiNGCReL (Fig. 2; Fig. 3 in Appendix).

- LLM case study is neat as a proof-of-concept.

### Weaknesses
- While improving the performance of LiNGCReL, it appears that the underlying idea of the identification algorithm is largely the same, limiting the novelty of this paper.

### Questions
- How does CREATOR behave when d is under- or over-estimated?

- In noisy finite samples, do you use singular-value thresholds or bootstrap ranks for deciding the 1-rank drop?

### Soundness
3

### Presentation
3

### Contribution
2
