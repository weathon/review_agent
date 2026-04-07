## Summary
This paper argues that the principles of Support Vector Machines (max-margin, KKT conditions) are inherently Euclidean and thus not directly applicable in the original "non-Euclidean" statistical space of the data, where distance is governed by covariance. It proposes a Covariance-Adjusted SVM (CSVM) that uses class-specific Cholesky decomposition to whiten data, solves the SVM in the transformed Euclidean space, and derives an iterative algorithm (SM) to estimate population covariance from samples. Empirical results on five binary datasets show CSVM often outperforms standard SVM kernels and global whitening methods.

## Strengths
- **Clear, intuitive motivation rooted in geometry:** The paper effectively connects the Mahalanobis distance, data whitening, and vector space concepts to argue for class-specific preprocessing, providing a clear and accessible rationale for its approach.
- **Practical iterative algorithm for a real problem:** The proposed SM Algorithm is a concrete and novel heuristic to address the practical challenge of applying class-specific whitening without test labels, framing it as an iterative label estimation and covariance update problem.
- **Thorough empirical comparison on diverse data:** The method is validated on five datasets from different domains, showing consistent improvements over several standard SVM kernels and two common whitening techniques (PCA, ZCA) across multiple metrics (accuracy, F1, AUC).

## Weaknesses
- **Overstated and unsubstantiated theoretical claims:** The core lemmas (2.1, 2.3) make strong, sweeping claims (e.g., KKT conditions are "invalid" in non-Euclidean spaces) that are not rigorously proven. The derivation shows the margin formula changes under a transformation, but this does not invalidate the optimization framework; it merely redefines the geometry. This overclaim undermines the paper's theoretical contribution.
- **Missing comparison with the most relevant prior work:** The paper dismisses prior covariance-incorporating SVMs (e.g., Minimum Class Variance SVM, Mahalanobis-distance-based SVMs) for alleged "gaps" and "dimensional inconsistencies" without a detailed explanation or a direct empirical or mathematical comparison. This omission makes it impossible to assess whether CSVM represents a substantive advance over existing techniques.
- **Confusing and inconsistent theoretical narrative:** Lemma 2.2 claims an N-class problem yields N distinct classifiers in the input space, which is non-standard and poorly explained. This claim is not reconciled with the final proposed algorithm, which outputs a single classifier after iterative adjustment, creating internal inconsistency and confusion for the reader.
- **Incomplete algorithmic analysis and ablation:** The SM Algorithm is presented heuristically without analysis of its convergence, sensitivity to initialization, or computational complexity. Furthermore, no ablation study isolates the contribution of the *iterative* algorithm from the simpler (and likely major) benefit of *class-wise whitening*, leaving the source of performance gains ambiguous.

## Nice-to-Haves
- An ablation study comparing: (a) global whitening + SVM, (b) class-wise whitening + SVM (non-iterative), and (c) the full iterative SM Algorithm.
- A synthetic 2D experiment to visually demonstrate and validate the core geometric claim that the margin splits according to class covariance.
- Reporting statistical significance tests (e.g., over multiple data splits) to bolster the empirical claims, given that some performance differences in the tables are small.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Strength:** "The paper is well-written" (Too generic).
- **Weakness:** "The derivation of the margin in Eq. 9 is inconsistent because it uses the Euclidean formula in a non-Euclidean space." (The paper's derivation is consistent: it computes the margin in the Euclidean space and then uses the transformation to express it in the input space. The critic misunderstands the pullback operation.)
- **Weakness:** "The algorithm's step (e) is an arbitrary post-processing step." (While the adjustment heuristic is not derived from a new objective function, it is a direct consequence of the derived margin ratio (Eq. 14), making it a reasoned design choice, not purely arbitrary.)
- **Weakness:** "Missing details like dataset sizes and hyperparameter tuning harm reproducibility." (While more details are always helpful, the absence of these specifics is a common minor shortcoming, not a core flaw that invalidates the results. It is moved to a suggestion.)
- **Weakness:** "The paper does not discuss broader impact." (This is not a standard expectation for a methodological paper of this type.)

## Novel Insights
The paper provides a coherent vector-space interpretation for why data whitening (e.g., via PCA/ZCA) improves model performance: it frames whitening as a transformation from a non-Euclidean statistical space (where distance is Mahalanobis) to a Euclidean space, where the geometric foundations of algorithms like SVM are naturally valid. This perspective cleanly unifies preprocessing and model geometry. Furthermore, the iterative SM Algorithm presents a novel, practical strategy for performing class-conditional whitening in the absence of test labels, a common real-world constraint.

## Suggestions
- Reframe the theoretical claims to be more precise and modest. Focus on deriving how the optimal classifier under a Mahalanobis geometry leads to the proposed optimization adjustments, rather than claiming the entire SVM framework is "invalid."
- Add a direct experimental comparison with at least one key prior method (e.g., Minimum Class Variance SVM) to substantiate the claim of addressing gaps in prior work.
- Clarify the narrative around Lemma 2.2. Either provide a clear explanation of how multiple classifiers are reconciled into a final decision rule or reformulate the lemma to avoid confusion with the single-classifier algorithm.
- Formalize the SM Algorithm with pseudo-code, specify the convergence criterion, and include a basic analysis (even empirical) of its convergence behavior and sensitivity.