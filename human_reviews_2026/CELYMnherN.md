# A Federated Generalized Expectation-Maximization Algorithm for Mixture Models with an Unknown Number of Components

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
We study the problem of federated clustering when the total number of clusters $K$ across clients is unknown, and the clients have heterogeneous but potentially overlapping cluster sets in their local data. To that end, we develop FedGEM: a federated generalized expectation-maximization algorithm for the training of mixture models with an unknown number of components. Our proposed algorithm relies on each of the clients performing EM steps locally, and constructing an uncertainty set around the maximizer associated with each local component. The central server utilizes the uncertainty sets to learn potential cluster overlaps between clients, and infer the global number of clusters via closed-form computations. We perform a thorough theoretical study of our algorithm, presenting probabilistic convergence guarantees under common assumptions. Subsequently, we study the specific setting of isotropic GMMs, providing tractable, low-complexity computations to be performed by each client during each iteration of the algorithm, as well as rigorously verifying assumptions required for algorithm convergence. We perform various numerical experiments, where we empirically demonstrate that our proposed method achieves comparable performance to centralized EM, and that it outperforms various existing federated clustering methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors discuss clustering in the federated setting. Data is distributed across clients, who share cluster centroids but follow different cluster weights, specifically, no client has data from all clusters. . The effective novelty of the paper is the ability of the proposed FedGEM to automatically infer the number of clusters across the federation. 
The authors discuss convergence properties of FedGEM in the general case and for the special case of GMMs. The paper concludes with a variety of empirical studies, specifically showing that FedGEM works even if "real" data is not well-separated, thereby invalidating a core prerequisite for the theoretical analyses (concavity).

### Strengths
The paper appears to be thorough and well-executed. The authors spend time analyzing a lot of aspects of their method in an extensive Appendix. The methods limitations are well-covered along with its strengths.
The problem of federated clustering is highly relevant, the method is original and automatic cluster count detection unlocks a new capability. 
Note that I did not check derivations.

### Weaknesses
I do not follow the paper's motivation about OEM fault detection. How exactly does this relate to (federated) clustering? The authors claim the dimensionality of the data poses a problem - however all of their experiments are well within the range of internet-transferable sizes. (The largest dataset across clients comes in at ~17mb). 
Although the paper describes some limitations of the proposed method, I see a few more that I think need to be addressed:
- The entire analysis and experimental result assumes full client participation. A realistic federation might only have a subset of clients participate at every round. The server-side aggregation would require rethinking
- The final aggregation radius needs to be determined through cross-validation. This is highly impractical in the federated setting as it would involve running the entire algorithm end-to-end multiple times. Furthermore, the optimal radius depends on the data geometry, scaling the feature space would require a new radius.
- It is unclear how anisotropic covariances would change the nature of aggregation radii
- Cross-validation incurs a huge price for DP guarantees

missing communication cost quantification. The authors note that AFCL requires clients sharing arrays of the same size as the local data. That price might be worth wile to pay (in terms of communication cost) if the total number of communication rounds is smaller due to a faster convergence rate. Especially as the number of clusters per round is variable, per-iteration costs of transmitting centroids could be high.

### Questions
If the authors could discuss the limitations I believe to be missing in the paper and also discuss communication costs, I'll consider raising my score!

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This submission presents a federated generalized expectation-maximization (FedGEM) algorithm for clustering in a setting where clients have heterogeneous data and the total number of clusters across the clients is unknown. FedGEM involves clients performing local EM steps to identify local cluster centers and then constructing uncertainty sets around these centers. These sets (centers and radiuses) are communicated to a server, which infers overlaps between local clusters to collaboratively train shared cluster parameters and estimate the possible total number of clusters. The authors provide a theoretical analysis of the convergence of the FedGEM and its ability to correctly infer the cluster count. The empirical evaluation demonstrates that FedGEM achieves state-of-the-art performance, outperforming the mentioned baselines.

### Strengths
- Sound Validation: The extensive experimental results demonstrate the effectiveness of the FedGEM. The results in Table 1 show that FedGEM outperforms AFCL, the only other federated baseline that operates without knowing K. Also, FedGEM is competitive with federated methods that are given the true K in advance. The sensitivity study further indicates the robustness of FedGEM, showing strong performance even when theoretical assumptions like well-separated clusters are violated.
- Theoretical Analysis: This submission is supported by a theoretical analysis. The authors provide probabilistic convergence guarantees for the general FedGEM algorithm under standard assumptions (Theorems 1, 2, and 5). Also, this submission provides a detailed analysis for the isotropic GMM setting, where they prove the First-Order Stability (FOS) condition for multi-component GMMs (Theorem 6).

### Weaknesses
- Generality: The implementation, theoretical verification, and experiments are based on an isotropic GMM. However, its performance on real-world data with arbitrarily shaped clusters might be limited.
- Computational Complexity: The server is required to identify cluster overlaps involves pairwise comparisons between all local components from all clients. This is a computational complexity that scales quadratically with the total number of local components across the network. Though the scalability study in the appendix shows good performance, this could become a bottleneck in FL settings with thousands of clients.
- Presentation and Grammar Issues: 1) grammar issue: line 96-97; 2) the names of the mentioned methods should be in the same font if the authors would like to emphasize

### Questions
- How to set `final_aggregation_radius`? Could you provide more practical heuristics for setting this value? How sensitive are the ARI and the estimated K to this hyperparameter?
- Others, please refer to weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces FedGEM, a federated generalized EM algorithm for mixture models with an unknown global number of components (\\(K^\*\\)). Each client runs a local GEM on an isotropic GMM with fixed mixture weights and covariances, updating only component means and computing uncertainty radii. The server merges overlapping components based on these uncertainty sets and performs constrained parameter updates. Theoretical results establish finite-sample and population convergence, as well as a sufficient condition for correct (\\(K^\*\\)) recovery. Experiments on synthetic and benchmark datasets show competitive clustering accuracy and moderate scalability.

### Strengths
# originality
Federated EM with unknown global K via uncertainty-set–based merging at the server is an interesting angle relative to standard federated clustering/EM, and distinct from k-means-style approaches

# quality

The paper presents local convergence results for the GEM variant, as well as finite-sample versus population map deviation bounds under the stated assumptions.

the radius subproblem has a stated unique solution and a low-complexity solver with a convergence and complexity discussion

# clarity

Problem setup and the server pseudo-code for super-clustering/aggregation are explicit

Assumptions are listed and some are justified in an appendix

# significance

Unknown K across many clients arises in FL. The approach could be a useful building block if the server-side merging is robust and the hyperparameters are chosen well. Empirical results indicate competitive performance and a reasonable scaling trend

### Weaknesses
# originality

The method is positioned generally ("mixture models with unknown K"), but the analysis and implementation hinge on isotropic Gaussians with fixed weights; this narrows the contribution relative to the stated ambition. Extending to anisotropic covariances or learning pi would be more compelling.

# quality

weights fixed, covariance fixed to identity, Kg known locally. the paper also does not study the effects of mispecified pi.

The overlap-detection step entails pairwise checks over all client components, i.e., quadratic in the total number of components. the paper relies on empirical timing rather than giving a tight complexity analysis for this stage. This should also be discussed as a limitation.

it is not analyzed whether a client (or all clients) stuck at a sharp local maximizer will "snap back" after the server’s within-set update. 



# clarity

Assumption 6 (supremum variable vs. the conditioning argument of M / $\hat{M}$
The role and definition of the strong concavity parameter ​$\lambda_g$ are not introduced where first used (Assumption 3)

# significance

Final aggregation radius $\epsilon^{final}$ is a user hyperparameter. While a sufficient condition wrt Rmin is given, Rmin is not known. The paper should indicate a protocol for choosing this hyperparameter

No discussion of communication/computation trade-offs vs. alternative K estimation strategies (e.g., centralized model selection surrogates, Bayesian nonparametrics) in FL.

### Questions
See weaknesses.

Other questions:

Anisotropic models: Do you foresee obstacles to extending the convergence and finite-sample analysis to full-covariance Gaussians (or even tied/diagonal covariances)? What breaks in the proofs?

Is fixing pi essential for your bounds, or could pi be updated (with constraints) without derailing the strong concavity/FOS arguments?

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
3

### Summary
This paper proposes FedGEM: a federated generalized expectation-maximization algorithm for the training of mixture models with an unknown number of components. The algorithm relies on each of the clients performing EM steps locally, and constructing an uncertainty set around the maximizer associated with each local component. The central server utilizes the uncertainty sets to learn potential cluster overlaps between clients, and infer the global number of clusters via closed-form computations. 

This paper performs a thorough theoretical study of our algorithm, presenting probabilistic convergence guarantees under common assumptions. Subsequently, this paper studys the specific setting of isotropic GMMs, providing tractable, low-complexity computations to be performed by each client during each iteration of the algorithm, as well as rigorously verifying assumptions required for algorithm convergence. This paper also performs various numerical experiments.

### Strengths
1. The structure of the paper is relatively clear. 
2. The theoretical derivation and proof in the paper are quite thorough.

### Weaknesses
There are only two compared algorithms that do not assume prior knowledge of K. Among them, one is the algorithm proposed in 1974, and the other is AFCL, which uses entirely different datasets from those in this paper. These two aspects result in flaws in the experimental design, making it difficult to truly reflect the algorithm's performance.

### Questions
As above.

### Soundness
3

### Presentation
3

### Contribution
3
