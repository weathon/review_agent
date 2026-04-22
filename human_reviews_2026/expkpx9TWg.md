# Causal Effect Identification in the Presence of Latent Confounding with a Single Imperfect Proxy Variable

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
We consider the problem of identifying the causal effect of a treatment on an outcome in the presence of latent confounding. Many existing works utilize proxies of the latent confounder to adjust for it indirectly, typically requiring multiple proxies. Within the framework of latent variable linear non-Gaussian acyclic model (lvLiNGAM), we propose a causal effect identification procedure requiring only a single proxy. Moreover, this proxy can be agnostic, which means that: first, it can have an arbitrary causal relationship with the treatment/outcome; second, this causal relationship is not required to be known a priori. The complexity of the agnostic proxy precludes identifying the causal effect via a simple analytical formula. Consequently, our procedure is designed to first derive several candidate solutions from cross-cumulants and then isolate the valid solution by examining certain independence relationships. We present and prove a series of new theoretical results, which collectively establish the soundness of our procedure: given the observational population distribution, it correctly identifies the true causal effect when identifiable, and correctly reports unidentifiability otherwise. Also, we conduct experiments to validate the correctness of our theoretical results.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper tackles the long-standing challenge of causal effect identification under latent confounding when only one proxy variable is available. Within the lvLiNGAM framework, the authors propose a new algorithm that works with a single agnostic proxy, meaning the proxy’s causal relation with treatment and outcome can be arbitrary and unknown. Their method first derives candidate causal effects using cross-cumulants and then selects the correct one via independence tests. Theoretically, they claim asymptotic consistency and, if identification is impossible, explicit detection of unidentifiability. Experiments on synthetic and real datasets demonstrate effectiveness and consistency advantages over GRICA and other baselines.

### Strengths
1. The paper investigates an important and fundamental problem in causal inference, causal effect identification and estimation in the presence of latent confounders.

1. It proposes an algorithm that extends existing methods to handle agnostic proxies, broadening the applicability of proxy-based causal identification.

### Weaknesses
1. The linear non-Gaussian additive noise assumption may be too restricted and limit its applications. The method collapses under even slight model misspecification (e.g., near-Gaussian noise or weak nonlinearity). Could the authors discuss possible extensions to nonlinear settings?
1. How can interval confidence or uncertainty quantification be obtained for the estimated causal effect?
1. The novelty over prior works (e.g., [1, 2, 3]) is insufficiently discussed. The key methodological distinctions and theoretical advances should be made clearer.
1. The writing is poor. Many theoretical results are listed without sufficient intuition or illustrative examples, making the paper difficult to follow.
1. The proposed method appears to require a large sample size for accurate effect estimation. The authors should discuss small-sample behavior or variance reduction strategies.
1. The experimental evaluation lacks robustness checks, such as tests under noisy proxies, weak confounding, or varying non-Gaussianity strength.
1. More baselines should be included to strengthen the empirical comparison beyond GRICA and cumulant-based methods.
1. The method requires multiple cumulant estimations and independence tests for several possible graph configurations. No complexity or runtime analysis is provided.



[1] Cai, Ruichu, et al. "Causal discovery with latent confounders based on higher-order cumulants." *International conference on machine learning*. PMLR, 2023.

[2] Kivva, Yaroslav, Saber Salehkaleybar, and Negar Kiyavash. "A cross-moment approach for causal effect estimation." *Advances in Neural Information Processing Systems* 36 (2023): 9944-9955.

[3] Chen, Wei, et al. "Identification of causal structure with latent variables based on higher order cumulants." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 38. No. 18. 2024.

### Questions
Please see above

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the challenging problem of identifying causal effects in the presence of latent confounders. Within the lvLiNGAM framework, the authors propose a novel method that requires only a single proxy variable for the latent confounder. The approach derives candidate solutions using cross-cumulants and then selects the valid one through independence tests.

### Strengths
1. The method is designed to handle settings with multiple latent confounders, substantially relaxing key limitations of existing approaches and broadening its applicability to more complex causal structures.
2. The theoretical foundation, built upon the non-Gaussianity assumption of lvLiNGAM, is sound. The property of explicitly reporting unidentifiability is a crucial and honest feature, preventing users from drawing false conclusions when the data is insufficient for identification.

### Weaknesses
1. The proposed method appears to rely on large sample sizes to achieve satisfactory performance, which may limit its utility in data-scarce scenarios.
2. Although higher-order cumulants provide valuable statistical information, their estimation can be sensitive to sampling variability. Moreover, in settings involving multiple latent variables, the iterative application of the procedure may lead to error accumulation, potentially compromising estimation accuracy.

### Questions
See above.

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
3

### Summary
This paper tackles causal effect identification under latent confounding using only a single proxy within the lvLiNGAM framework. Unlike prior methods requiring multiple or structurally known proxies, the proposed approach allows an *agnostic proxy*—one whose causal links to treatment or outcome are arbitrary and unknown. Exploiting non-Gaussianity, the method first derives candidate causal effects from cross-cumulants and then selects the valid one via independence tests. It guarantees asymptotic consistency: with sufficient data, the true causal effect is recovered if identifiable, while unidentifiability is explicitly reported otherwise. This work broadens causal inference applicability in realistic settings with limited or poorly understood proxies.

### Strengths
1. The "agnostic proxy" concept fills a critical gap in existing literature. 
2. The proposed method is supported by rigorous theoretical analysis.
3. Experimental results demonstrate that the proposed approach.

### Weaknesses
1. The method relies on cross-cumulant estimators with high variance in small samples, leading to poor performance.
2. The approach is confined to the lvLiNGAM framework and cannot handle nonlinear.
3. The method jointly relies on both independence tests and cross-cumulant constraints for causal structure discrimination. If either component yields an incorrect judgment—due to estimation noise or finite-sample errors—the resulting causal effect estimate may be suboptimal or even incorrect.

### Questions
1. Have you considered any robustness strategies to solve the depends critically on both independence tests and cross-cumulant constraints?
2. Xu et al. demonstrate identifiability under nonlinear structural causal models in Figure 1(a). How does your method relate to or differ from their framework? Do you see potential for integrating your cross-cumulant + independence test approach with nonlinear identifiability results to extend applicability beyond the linear lvLiNGAM setting?


[1] Kernel single proxy control for deterministic confounding.

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
4

### Summary
The paper studies identification of the causal effect of a treatment $T$ on an outcome $O$ under latent confounding, assuming lvLiNGAM (linear, non-Gaussian, acyclic) structure. The key contribution is an identification and estimation procedure that requires only a single proxy $Z$ of the latent confounder $L$, and that proxy can be agnostic, i.e., it may have arbitrary and a priori unknown causal relationships with $T$ and/or $O$. The method first generates candidate values for the causal effect from cross-cumulants (solving a quadratic with coefficients given by cumulant polynomials) and then selects the valid candidate via independence tests. The authors prove asymptotic consistency and provide algorithms for both the ``basic agnostic proxy'' setting and a generalized setting where $Z$ can be placed anywhere relative to $(L,T,O)$; in the latter, they first decide (via independence tests) whether the graph is in a family where identification is possible and otherwise report unidentifiability.

### Strengths
- Identification with just one proxy whose causal links to $T$ and $T$ need not be known or restricted (beyond lvLiNGAM).
- The paper delineates when the effect is identifiable vs. provably unidentifiable  and provides Algorithm 1 that either outputs the effect or correctly returns “unidentifiable.”

### Weaknesses
- While common in applications, the setup focuses on one treatment–outcome pair and a single proxy. Extensions to multiple treatments/outcomes, multiple proxies, or networked settings are not developed here.

- Performance can degrade at modest sample sizes because cumulant estimation is high-variance.

### Questions
- Which concrete independence tests (and thresholds) are recommended in practice for the selection step? How sensitive is performance to these choices in moderate $n$ regimes?

- In Fig. 7, subplots for Fig. 3(a), the curves of the proposed method are not observable. It is better to update the plots. Moreover, the subplots for Fig. 3(b), Beta, the results are not consistent with the plots in (Tramontano et al., 2025). Moreover, there are two versions of the cumulant algorithm there and the cumulant with minimization has a better performance.

- If several agnostic proxies are available, can your framework be combined across proxies?

- The paper frequently compares to methods that assume the causal structure is known. However, in lvLiNGAM with latent confounding, one can first recover the causal order from observational data using recent cumulant–rank–based discovery results (e.g., Schkoda et al. 2024), and then apply the effect-identification step (such as (Tramontano et al., 2025)). In what sense is the present contribution novel beyond this discover-and-apply pipeline?

### Soundness
3

### Presentation
2

### Contribution
2
