# Multi-View Causal Discovery without Non-Gaussianity: Identifiability and Algorithms

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 6, 2

## Abstract
Causal discovery is a difficult problem that typically relies on strong assumptions on the data-generating model, such as non-Gaussianity. In practice, many modern applications provide multiple related views of the same system, which has rarely been considered for causal discovery. Here, we leverage this multi-view structure to achieve causal discovery with weak assumptions. We propose a multi-view linear Structural Equation Model (SEM) that extends the well-known framework of non-Gaussian disturbances by alternatively leveraging correlation over views. We prove the identifiability of the model for acyclic SEMs. Subsequently, we propose several multi-view causal discovery algorithms, inspired by single-view algorithms (DirectLiNGAM, PairwiseLiNGAM, and ICA-LiNGAM). The new methods are validated through simulations and applications on neuroimaging data, where they enable the estimation of causal graphs between brain regions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a new framework for causal discovery that eliminates the need for the conventional assumption of non-Gaussianity and instead exploits a multi-view data structure to ensure identifiability of causal relations. Traditional approaches generally assume a single view, where it is theoretically impossible to determine causal direction from data without strong additional assumptions. One of the most prominent models, LiNGAM, achieves identifiability by assuming non-Gaussian, independent disturbances. The proposed model, LiMVAM (Linear Multi-View Acyclic Model), is a linear structural equation model (SEM) that handles multiple related data views jointly. The authors show that identifiability can be guaranteed by leveraging second-order statistics (SOS), the covariances across views, without relying on higher-order statistics. Specifically, they prove that if (i) each variable pair is correlated in at least some views, (ii) the correlation patterns are not identical across views, and (iii) the overall graph formed by all views is connected, then both the causal ordering and the coefficient matrices are identifiable using SOS alone. Based on this theory, the paper proposes three new algorithms, PairwiseLiMVAM, DirectLiMVAM, and ICA-LiMVAM, which can be interpreted as multi-view extensions of existing LiNGAM-based methods. In particular, ICA-LiMVAM generalizes ICA-LiNGAM as a special case. Through simulations under both Gaussian and non-Gaussian settings, PairwiseLiMVAM and DirectLiMVAM achieve higher accuracy and faster computation than prior approaches such as ICA-LiNGAM and Multi-Group DirectLiNGAM. Applications to brain-imaging (MEG/fMRI) data further demonstrate that the proposed methods recover physiologically plausible causal relationships.

### Strengths
This paper makes an important theoretical contribution by formalizing the general intuition that performing multiple instances of statistical causal discovery can enhance estimation robustness. Under a multi-view formulation, the authors provide a clear set of assumptions and a rigorous logical development showing that causal ordering can be uniquely identified using only second-order statistics (SOS)—without relying on the conventional non-Gaussianity assumption. To the best of my knowledge, this is the first work to demonstrate such identifiability purely from SOS. The paper is clearly written and well structured: the connection to the well-known LiNGAM model is carefully explained, and the mathematical exposition is detailed and persuasive. The demonstration that ICA-LiMVAM generalizes ICA-LiNGAM as a special case further clarifies the positioning of the proposed approach.

The simulation experiments are well designed and convincingly show that the proposed methods perform as predicted under the stated assumptions. Moreover, experiments on real MEG and fMRI datasets demonstrate that the approach is not only theoretically sound but also practically useful.

By leveraging multiple views rather than imposing unrealistic distributional assumptions, the proposed framework provides a promising and potentially high-impact direction for improving the robustness of statistical causal discovery in realistic settings.

### Weaknesses
In practical data analysis, it may be difficult to ensure that the assumptions of cross-view correlation and diversity are satisfied. In particular, Assumption 1 requires that for each pair of variables, (i) the correlation is non-zero in at least some views, and (ii) the correlation structures across views are not perfectly proportional. However, in high-dimensional real-world datasets, many variable pairs often exhibit near-zero correlations (i.e., near independence), providing little informative variation, while other groups of variables may be dominated by common latent factors, resulting in nearly identical correlation structures across views. A deeper discussion of how frequently such situations may arise in practice, and how they might be detected or mitigated, would substantially strengthen the paper and improve the practical applicability of the proposed method.

Although this point could also be framed as an opportunity rather than a limitation, one intriguing extension would be to actively generate multiple views, for example, via bootstrap resampling, and apply the proposed framework to enhance the robustness of statistical causal discovery. Such an idea could even lead to new approaches for evaluating the reliability or confidence of inferred causal relations, a problem that has long been challenging. Discussing these possible directions would make the paper’s impact and future potential even greater.

### Questions
- In practical datasets, how likely is it that the assumptions of cross-view correlation and diversity are violated? Could the authors provide a deeper discussion on how frequently such violations might occur in real-world settings, and what strategies could be used to detect or avoid such situations?

- Would it be possible to actively generate multiple views, for example, through bootstrap resampling, and apply the proposed method to substantially enhance the robustness of statistical causal discovery? Could this idea potentially evolve into a new approach for assessing the reliability or confidence of estimated causal relationships?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a multi-view linear SEM (LiMVAM) for causal discovery that exploits correlations across views to obtain identifiability and scalable algorithms without requiring non-Gaussian noise. Two SOS-only, residual-recursive procedures, PairwiseLiMVAM (likelihood-ratio with closed-form LR) and DirectLiMVAM (cross-covariance Frobenius criterion), recover a shared causal ordering and then estimate per-view coefficients with joint FGLS. A complementary “shared-disturbance” formulation (with per-view scalings) connects LiMVAM to Shared ICA, yielding identifiability even when some disturbances are Gaussian. Conceptually, LiMVAM generalizes multi-environment/non-stationary ideas by treating environments as “views” and using diversity of second-order structure to replace classical non-Gaussianity assumptions.

### Strengths
a. The multi-view setting is both common and important. This paper extends two single-view causal-discovery algorithms to the multi-view case and removes the need for non-Gaussian noise assumptions.

b. The estimators are fast and straightforward: closed-form likelihood-ratio and cross-covariance tests within a recursive-residual scheme, followed by joint FGLS for edge weights, essentially no tuning required.

c. The real-world experiments are good, including MEG/fMRI case studies with cross-subject stability checks that support the practical value of the approach.

### Weaknesses
a. The contribution is close to existing multi-environment/multi-domain lines. Several works also target identification of linear systems in multi-view or multi-domain settings but are not discussed. For example, [1] addresses causal structure learning for linear relations without relying on non-Gaussian noise or inter-view correlation; [2] studies identification under heterogeneous/noise-variance shifts across domains; and invariance-based approaches such as [3] provide a related lens on leveraging distributional stability across environments. A clearer positioning relative to these would sharpen the paper’s novelty claims.

b. In sections 4.2–4.4, many key conditions and results deferred to the appendix. It would improve readability and verifiability to promote the central statements to the main text as formal theorems/definitions (with precise assumptions), leaving proofs and extended discussion to the appendix.

c. In synthetic experiments, multi-environment baselines beyond LiNGAM-style variants are limited.


[1] Ghassami A E, et al. Multi-domain causal structure learning in linear systems[J]. Advances in neural information processing systems, 2018, 31.

[2] Adams J, et al. Identification of partially observed linear causal models: Graphical conditions for the non-gaussian and heterogeneous cases[J]. Advances in Neural Information Processing Systems, 2021, 34: 22822-22833.

[3] Peters J, et al. Causal inference by using invariant prediction: identification and confidence intervals[J]. Journal of the Royal Statistical Society Series B: Statistical Methodology, 2016, 78(5): 947-1012.

### Questions
a. Lines 96–100: the partial-ordering example is unclear. If the adjacency/structural matrix is the zero matrix, the DAG has no edges, so all variables are mutually independent; in that case there is no informative partial order beyond the trivial one?

b. On the assumption that views must be correlated, does this imply a shared latent factor across views?

c. Theorems 1–2: if each view’s coefficient matrix $B^i$ is identifiable, why is an additional assumption needed to recover the causal ordering $P$? Is $B^i$ identified only up to a common permutation of variables (leaving the topological order ambiguous), or is $B^i$ in Theorems 1–2 different from the structural matrix in Eq. (3)?

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
This paper tackles the formidable challenge of uncovering causal structure from observational data by capitalizing on multi-view architectures, wherein multiple correlated datasets (or "views") share an underlying causal dependency system. Transcending the conventional prerequisite of non-Gaussianity for reliable identifiability (as is characteristic of LiNGAM), the authors propose an innovative multi-view linear SEM framework.This framework achieves identifiability under substantially less stringent assumptions by exclusively utilizing heterogeneous Second-Order Statistics (SOS) across the distinct views. The study furnishes rigorous theoretical guarantees of identifiability, devises streamlined multi-view extensions of established algorithms (DirectLiNGAM, PairwiseLiNGAM, and an ICA-based approach), and empirically validates and benchmarks these methodologies on both synthetic and large-scale neuroimaging datasets.

### Strengths
- The paper delivers a significant contribution by establishing the identifiability of linear SEMs using only second-order statistics, eliminating the reliance on non-Gaussianity
- The generalization of DirectLiNGAM and PairwiseLiNGAM to the multi-view setting is addressed in a thoughtful manner, yielding novel fast SOS-based algorithms. Additionally, the adaptation of ICA-LiNGAM methodology to accommodate multi-view shared disturbances is conceptually elegant.
- The paper is well-organized and clearly written.

### Weaknesses
- What is the difference between Ghassami[1] and Perry[2] in this paper?

- In Section 3, It would be better that the assumption, "All adjacency matrices ${B}_i$ share the same causal ordering" be explicitly highlighted (or formal definition).

- The definitions of some superscripts and subscripts in the text are easily confusing. For example, in Equation (8) of Section 4.5, which represents different views, the superscripts $i$ and $i'$ hinder readability. 

- In Section C.2，Why is matrix $B$ in the form of $diag(cov(x^1,y^1),...,cov(x^m,y^m))$?

- The paper only conducted limited baseline comparisons, without benchmarking against recently emerged causal discovery methods.




## References
> 1、AmirEmad Ghassami, Negar Kiyavash, Biwei Huang, and Kun Zhang. 2018. Multi-domain causal structure learning in linear systems. In Proceedings of the 32nd International Conference on Neural Information Processing Systems (NIPS'18). Curran Associates Inc., Red Hook, NY, USA, 6269–6279.

> 2、Ronan Perry, Julius von Kügelgen, and Bernhard Schölkopf. 2022. Causal discovery in heterogeneous environments under the sparse mechanism shift hypothesis. In Proceedings of the 36th International Conference on Neural Information Processing Systems (NIPS '22). Curran Associates Inc., Red Hook, NY, USA, Article 792, 10904–10917.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This article addresses the challenging problem of causal discovery, which traditionally relies on strong assumptions like non-Gaussianity. The authors introduce a novel approach that leverages multi-view data.

### Strengths
This article addresses the challenging problem of causal discovery, which traditionally relies on strong assumptions like non-Gaussianity. The authors introduce a novel approach that leverages multi-view data.

### Weaknesses
I have a concern regarding the step in the identifiability proof where independence is concluded from the vanishing covariances. As is well-known, vanishing covariance does not generally imply independence. Since the proof seems to leverage this implication, its validity depends critically on the underlying distributional assumptions. Please clarify how the non-Gaussianity of the disturbances, potentially via the framework of the Darmois–Skitovich theorem, guarantees that this implication holds in the proposed multi-view model. A more detailed explanation in the manuscript would be essential. The second order of statistics may not be sufficient for the estimation.

### Questions
See above.

### Soundness
1

### Presentation
1

### Contribution
1
