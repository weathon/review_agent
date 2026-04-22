# On the Geometry of Uncertainty: Equivariant Uncertainty Estimation for Molecular Vector Properties

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Accurate and reliable uncertainty estimation (UE) for vector-valued physical properties is crucial for scientific discovery in fields like drug  and materials discovery. For example, atomic forces are central for finding optimal structures and identifying equilibrium systems, and a fundamental requirement of these vectors is that they must be equivariant to 3D rotations. However, existing UQ methods often fail to respect these geometric constraints, leading to poorly calibrated uncertainty and degraded predictive performance. To address the problem, we introduce a novel framework for equivariant multivariate evidential regression. Our key contribution is a new parameterization for the covariance matrix, which is provably equivariant, identifiable, and correctly parameterized. We further introduce a calibration metric based on the Mahalanobis distance to rigorously assess equivariant multivariate uncertainty. Extensive experiments on molecular property prediction benchmarks, including MD17, OC20, and QM7-X, show that our framework consistently outperforms established baselines, achieving state-of-the-art predictive accuracy with well-calibrated uncertainty estimates. This work provides a principled and practical approach to uncertainty estimation for equivariant vector-valued properties, paving the way for more trustworthy machine learning applications in scientific discovery.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a new method for uncertainty estimation of vector properties (of molecules). 
They construct a new prediction head to enable equivariant multivariate evidential regression. They provide equivariance of the covariance matrix by using exp(\alpha) I to the FF^T term. They evaluate the predicitve quality and calibration on several datasets including OOD splits.

### Strengths
- The idea seems simple yet effective
- The approach does not require sampling during inference
- Good empirical performance

### Weaknesses
- I would like to see a small hyperparameter study on the force loss contribution, as I believe it is quite critical to see how good the performance can be, also on energy, and on the tradeoff between energy and force prediction.
- The text can be improved: there are some typos, and for example, a missing reference to tables 3 and 4
- I would like to see more methods. The number of baselines seems a bit limited to me

### Questions
- In the results section, you introduce VBP of Liu, then in the table write VBD and in the text SBP. Are the first and the last typos and you mean VBD each time?
- Can you provide more insights and ablations on the effect of alpha?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Equivariant Multivariate Evidential Regression (EMER), a novel framework for uncertainty estimation (UE) of vector-valued molecular properties, such as atomic forces, which need to respect physical symmetries. This is achieved through a novel parametrization of the covariance matrix that is equivariant to rotations. A set of experiments on three molecular datasets using a newly introduced calibration metric is presented to compare the proposed method to some selected baselines.

### Strengths
* The problem of designing vector-valued uncertainty estimation methods that are equivariant to 3D rotations is interesting and impactful.
* The idea of extending Multivariate Evidential Regression (MER) by re-parametrizing the covariance matrix is sound.
* The paper is mostly clear and well-written.
* The authors do a good job in motivating their approach and discussing related works.

### Weaknesses
* The main issue with this paper is the experimental setup. Specifically, the authors propose a new calibration metric and rely solely on it to compare the different models and draw their conclusions. At the same time, this new metric is not sufficiently explained or analyzed, making it difficult to use as a reliable evaluation metric. The authors should (i) better explain this metric by comparing it to existing metrics (please add references in section 3.3) and analyzing it on known benchmarks, and (ii) use other metrics that are commonly used in the literature, such as the calibration score from [1].
* Another serious limitation is the following: the main motivation behind this work is that uncertainty estimation methods for vector-valued molecular properties should be equivariant to rotations; however, this need is neither experimentally validated nor reflected in the results. Specifically, the raw proposed method (EMER) consistently performs worse than Dropout. Given the absence of other baselines, it is challenging to assess the primary contribution of this paper (see the next point).
* The authors only compare to selected sampling-based methods. However, in order to assess the effect of adding equivariance to the UE method, the authors should compare it to related baselines that do not employ equivariance, such as the original MER [2] and LNK [1]. Specifically, the authors need to compare EMER with MER and EMER-Dropout with MER-Dropout, applying the same dropout approach to MER for a fair comparison.
* The presentation of the main contribution of this paper, which is the novel parametrization of the covariance matrix, is very ambiguous. Specifically, $F$ is sometimes referred to as a vector and at other times as a matrix. Also, it is not discussed how the proposed parametrization behaves in terms of over- or under-parametrization of the covariance matrix.

**References**

[1] Wollschläger, Tom, et al. "Uncertainty estimation for molecules: Desiderata and methods." International conference on machine learning. PMLR, 2023.

[2] Meinert, Nis, and Alexander Lavin. "Multivariate deep evidential regression." arXiv preprint arXiv:2104.06135 (2021).

### Questions
1. In Equation 2, what is $m$? Additionally, please formally define all variables, including their domains and dimensions, such as the NIW parameters and the Gamma function.
2. The current setup predicts a covariance matrix for each atom in the molecule. Can the method be extended to capture correlations between different forces and between forces and energy?
3. Is there any reference for the proposed calibration metric?
4. What are VBP, VBD, and SPD mentioned in the experiments section? What is "Benchmark" in Table 1?

### Soundness
1

### Presentation
3

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
The paper focuses and addresses a critical gap in machine learning for physical systems: the lack of reliable geometrically consistent uncertainty estimation (UE) for vector-valued properties, such as atomic forces and dipole moments. The authors highlight that while scalar UE naturally satisfies equivariance, applying standard multivariate UE techniques to vectors often violates fundamental SO(3) rotational symmetries. To this end, they introduce EMER, a sampling-free framework that extends evidential regression to satisfy geometric constraints.

### Strengths
- **Well-Motivated**: The paper is well-written and structured. The motivation of the paper is very clear: while uncertainty estimation (UE) for scalar properties in scientific machine learning is mature, it fundamentally fails for vector-valued properties (like atomic forces) because standard multivariate formulations violate essential SO(3) equivariance constraints. The proposed solution EMER introduces a novel contribution through its parameterization of the Normal-Inverse-Wishart scale matrix as $$\Psi(\alpha,F)=\exp(\alpha)I+FF^T$$, which combines an invariant scalar isotropic term with an equivariant rank-one term to guarantee both positive definiteness and equivariance, effectly removing the limitations of prior MER approaches that relied on non-equivariant Cholesky factorizations. 
- **Reasonable Metrics**: Additionally, the authors found the drawbacks of the existing calibration metrics and proposed a rotation-invariant Mahalanobis distance-based metric for calibration to address the lack of suitable evaluation tools for the specific domain.
- **Empirical Breadth**: The experimental validation is comprehensive, utilizing three established benchmarks (MD17, OC20, QM7-X) and testing across multiple SORA backbone architectures (PaiNN, NequIP, MACE).

### Weaknesses
- **Lack Visualization**: The authors can think of a way to add more visualizations for readers to better understand their motivation and solution. 
- **Metric Design**: Although the metric design sounds good to me for vector-valued problems, it still needs to convince people why not use traditional calibration metrics. Like why do not we use traditional calibration metrics, e.g., ECE, PICP, NLL, etc., on different dimensions of the vectors and take an average? This seems to be able to estimating the calibration of vectors?
- **Lack Discussion**: From the results, it seems that dropout itself is not that useful, but EMER+dropout is super useful. Can the authors elaborate more about this phenomenon?

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies uncertainty estimation for multivariate SO(3)-equivariant regressors (e.g., vector forces, dipoles). Prior UQ methods either break equivariance (single-pass, scalar-oriented approaches) or require expensive sampling like ensembles/MC dropout; when they do sample, they only indirectly yield equivariant covariances and are costly at inference. The authors propose EMER (Equivariant Multivariate Evidential Regression), a single-pass, sampling-free framework that parameterizes uncertainty so it is provably equivariant, positive-definite, and identifiable, and pairs it with a rotation-invariant calibration test based on Mahalanobis distance behavior for vectors. They also introduce a lightweight dropout-regularized variant to stabilize the encoder and keep predicted covariances well-conditioned. Across MD17, QM7-X, and OC20-style setups, EMER delivers state-of-the-art accuracy with markedly better calibration than ensembles, Bernoulli dropout, and variational Bayesian dropout, with ablations showing why the isotropic term and a low-rank design are crucial for stability.

### Strengths
1. Uncertainty estimation is crucial for scientific discovery, especially in molecular and physical systems where data naturally follow equivariance constraints. The paper is among the first to explicitly address uncertainty estimation in equivariant settings.

2. The proposed EMER framework is conceptually clear, mathematically grounded, and provides a provably equivariant and positive-definite uncertainty parameterization.

3. Experiments across multiple benchmarks (MD17, QM7-X, OC20-style) show strong performance both in predictive accuracy and uncertainty calibration.

### Weaknesses
1. While the paper emphasizes the inefficiency of sampling-based methods, it does not compare against more recent generative or diffusion-based uncertainty estimation approaches (e.g., Two for One: Diffusion Models and Force Fields for Coarse-Grained Molecular Dynamics).

2. The method’s efficiency comes from its parameterized uncertainty form, which may limit expressiveness compared to non-parametric sampling-based alternatives. A discussion or quantitative comparison of this trade-off would strengthen the claims.

3. The experimental section primarily compares to older dropout and variational Bayesian dropout methods. Including more contemporary or stronger ensemble baselines would make the empirical validation more convincing.

4. While calibration metrics are useful, demonstrating how the proposed uncertainty improves downstream tasks such as active learning, model-based control, or molecular discovery would highlight practical impact.

5. Table 1 suggests that most calibration improvements may stem from dropout regularization rather than the new covariance parameterization; a clearer disentanglement of these effects would be helpful.

### Questions
See the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
