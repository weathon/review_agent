# OrthoSolver: A Neural Proper Orthogonal Decomposition Solver For PDEs

- Decision: Accept (Poster)
- Scores: 6, 6, 2

## Abstract
Proper Orthogonal Decomposition (POD) is a cornerstone reduced-order modeling technique for accelerating the solution of partial differential equations (PDEs) by extracting energy-optimal orthogonal bases. However, POD's inherent linear assumption limits its expressive power for complex nonlinear dynamics, and its snapshot-based fixed bases generalize poorly to unseen scenarios. Meanwhile, emerging deep learning solvers have explored integrating decomposition architectures, yet their purely data-driven nature lacks essential physical priors and leads to modal collapse, where decomposed modes lose discriminative power.
To address these challenges, we revisit POD from an information-theoretic perspective. We theoretically establish that POD's classical energy-maximization criterion is, in essence, a principle of maximizing mutual information. Guided by this insight, we propose OrthoSolver, a neural POD framework that generalizes this core information-theoretic principle to the nonlinear domain. OrthoSolver iteratively and adaptively extracts a set of compact and expressive nonlinear basis modes by directly maximizing their mutual information with the data field. Furthermore, an orthogonality regularization is imposed to preserve the diversity of the learned modes and effectively mitigate mode collapse. Extensive experiments on seven PDE benchmarks demonstrate that OrthoSolver consistently outperforms state-of-the-art deep learning baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces OrthoSolver, a Neural Operator architecture that combines efficient decomposition with robust nonlinear modeling by exploiting an information-theoretic perspective on dimensionality reduction. The method introduces two key innovations to the neural operator community: an information-theoretic nonlinear approach for identifying reduced modes (enhance expressivity), and an orthogonality regularization term that prevents redundant feature extraction (reduce mode collapse). The approach is both elegant and theoretically sound. The authors validate their method on five different 1D and two different 2D PDEs, and further investigate the key design choices of their framework through an in-depth ablation study.

### Strengths
The paper addresses the problem of combining efficient mode decomposition (to extract relevant features) with robust nonlinear modeling, while simultaneously preventing mode collapse. This problem is relevant as it clearly captures the dichotomy between traditional Reduced Order Model (ROM) techniques, which have solid mathematical foundations but suffer from linearity limitations, and Neural Operator–based data-driven modeling, which offers high nonlinearity but often lacks theoretical soundness. Among the main strengths of the paper, I highlight the following:

1. An original application of mutual information optimization to guide PDE mode decomposition, which could potentially impact other fields such as ROM and manifold learning.  
2. A clear and well-structured review of the Neural Operator literature.  
3. The introduction of a Basis Orthogonality Constraint to promote diversity among the learned basis functions.  
4. A comprehensive experimental section that includes both 1D and 2D PDEs.  

The derivation presented in Appendix C.2 appears sound, and the experimental details are sufficiently complete to allow for reproduction of the results.

### Weaknesses
My primary concern is robustness and comparison with state-of-the-art models. Given these clarifications in an author response, I would be willing to increase the score. 

1. A more in-depth analysis and explanation of the model’s performance (Table 1) are needed. In particular:  
   - Important models [1, 2, 3] are missing and should be included as benchmark baselines. The field is evolving rapidly and it is important to compare against the best models.
   - The evaluation metric for Table 1 is not clearly explained — is the L2 error averaged over time?  

2. How were the hyperparameters for both your model and the baselines chosen? In the **Hyperparameter Settings** section, this information is not specified. Are the hyperparameters the same as those used in the PDEBench paper’s baseline results (see Tables 2–9 in [4])? Are the results better than the ones reported by PDEBench (in order to claim SOTA).

3. Can you provide variance results across multiple network initialization seeds for the different problems? This is important to ensure that the results are not cherry-picked for a specific initialization and that the methodology is robust.


***References***

[1] Zhdanov, Maksim, Max Welling, and Jan-Willem van de Meent. "Erwin: A tree-based hierarchical transformer for large-scale physical systems." ICML, 2025.

[2] Alkin, B., F  ̈urst, A., Schmid, S., Gruber, L., Holzleitner, M., and Brandstetter, J. Universal physics transformers: A framework for efficiently scaling neural operators. NeurIPS, 2024.

[3] Luo, H., Wu, H., Zhou, H., Xing, L., Di, Y., Wang, J., and Long, M. Transolver++: An accurate neural solver for pdes on million-scale geometries. CoRR, 2025.

[4] Takamoto, Makoto, et al. "Pdebench: An extensive benchmark for scientific machine learning." Advances in Neural Information Processing Systems 35 (2022): 1596-1611.

### Questions
1. Are the modes interpretable? For example, in classical POD, the most energetic modes are those that explain the greatest variance. Do we observe a similar behavior here? It would be helpful to plot the modes in order of energy level to assess their interpretability.
2. In Table 2, why does the model without the reconstruction constraint seem to perform almost as well as the full model? I would have expected this constraint to be very important, as it measures the discrepancy between the original function and its reconstruction from the full set of extracted modes.

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
4

### Summary
The paper builds a connection between POD and mutual information maximization under linear-Gaussian assumptions. Based on this, the authors proposed a data-driven neural proper orthogonal decomposition solver for PDEs. The mutual information is encoded into the total loss function, which will increase the training difficulty. The authors conduct many numerical experiments and many comparisons with other methods to demonstrate the performance of the proposed method.

### Strengths
Providing a large amount of the empirical results.

### Weaknesses
1. The equivalence between variance and mutual information only holds under the strong linear Gaussian assumption, which is rarely satisfied in nonlinear partial differential equations. This article does not explain why this connection is still meaningful or useful in the highly nonlinear cases it aims to address.
2. Too many penalty terms have to be introduced into the loss function. Is it very hard to optimize?

### Questions
Refer to Weaknesses

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
The authors introduce a paradigm shift for POD away from optimal energy-based modes to maximizing mutual information among modes. The work also introduces a novel learning architecture for mode decomposition/training in neural PDEs. Strong empirical performance is reported across several PDE benchmarks.

### Strengths
- The paper tackles relevant challenges in neural PDEs including principled mode selection and training.
- The work bridges statistical objectives (MI) and reduced-order modeling, which is of broad interest.
- The paper presents a novel and well motivated architecture for performing mode decomposition and training in neural PDEs.
- This architecture achieves SOTA results across several benchmark PDEs.

### Weaknesses
- The theoretical novelty appears limited under the standard Gaussian assumptions. In particular, for Gaussian data MI-optimal linear projections coincide with KLT/PCA, i.e., POD. This is well known (e.g., [1]) in information-theoretic sources.
- The authors make claims about overcoming mode collapse are made without supporting empirical evidence or sufficient study.

[1] Burges, Christopher JC. "Dimension reduction: A guided tour." Foundations and Trends® in Machine Learning 2.4 (2010): 275-365.

### Questions
**Q1. Theoretical novelty.** Under linear-Gaussian models, MI maximization over orthonormal projections selects the KLT principal subspace, i.e., POD (see [1]). What is new in the theoretical results beyond restating the equivalence.

**Q2. Ablation study.** I find the ablation results appear to show almost monotonically decreasing MSE for Advection and 1D-NS as the order increases. Here $K=4$ appears to be an outlier. Please consider reporting a full ablation study across the modes, including at least $K=3$ and $K=5$.

**Q3. Mode collapse.** The paper makes claims about resolving mode collapse, but there is no supporting empirical evidence of when mode collapse occurs or that the method presented effectively overcomes mode collapse. I recommend the authors either reduce the claims about mode collapse or provide a supporting study.

[1] Burges, Christopher JC. "Dimension reduction: A guided tour." _Foundations and Trends® in Machine Learning_ 2.4 (2010): 275-365.

### Soundness
3

### Presentation
4

### Contribution
2
