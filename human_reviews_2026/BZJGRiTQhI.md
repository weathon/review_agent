# Generative Conformal Prediction with Optimized Coverage Allocation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Conformal prediction provides model-agnostic uncertainty quantification with guaranteed coverage, but conventional methods often yield overly conservative uncertainty sets, particularly in multimodal or heterogeneous settings. This inefficiency arises from two sources: (i) limited expressiveness of the predictive model and (ii) simplistic nonconformity scores design. Most existing approaches advance only one of these axes, leaving the other underexplored. We propose *generative conformal prediction with* ***O**ptimized* ***R**anking and **C**overage **A**llocation (ORCA)*, a three-stage framework that advances both aspects jointly. ORCA leverages generative models to capture the full conditional distribution and introduces a rank-dependent optimization procedure that adaptively allocates coverage for efficiency while maintaining validity. We cast this coverage allocation as an optimization problem, derive an exact mixed-integer linear programming formulation, and show that the solution converges asymptotically to the oracle density-level set. Across synthetic, semi-synthetic, and real datasets, ORCA produces substantially more efficient uncertainty sets than state-of-the-art baselines, demonstrating robust gains in scenarios where conventional conformal prediction methods fail.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Problem context: Conventional CP methods lead to conservative prediction sets, mainly because (i) poor choice of predictive model (ii) simple form of nonconformity scores. Recent approaches expand in only one direction.

This paper presents ORCA, a three-stage conformal prediction (CP) framework designed to construct volume-efficient conformal prediction sets. The procedure combines: (i) generative modeling to approximate the full conditional distribution $p(y|x)$, (ii) vectorized nonconformity scores, and (iii) a rank-dependent mixed-integer linear programming (MILP) optimization to allocate coverage for improved volume-efficiency while maintaining finite-sample coverage guarantees. Experiments are conducted on synthetic, semi-synthetic, and real datasets. 

The paper is clearly written and easy to follow, contributing to the growing literature on improving CP efficiency through optimization-based allocation mechanisms.

### Strengths
1. Studies an important limitation of conformal prediction, its inherent conservativeness, through a structured three-stage approach.
2. The integration of generative sampling with optimization-based coverage allocation is practically appealing.
3. The exposition and algorithmic description are clear and reproducible.
4. Finite-sample validity results are consistent.
5. Experimental section is elaborate in terms of methods compared as well as diversity of real datasets.

### Weaknesses
1.  The method assumes that sampling from a trained generative model effectively explores the true conditional distribution $ p(y | x) $. 
However, ORCA learns the model-implied distribution $\hat{p}(y|x),$ which may diverge substantially from the true one. 
Theoretical results (e.g., Proposition 3.4, Theorem 3.5) are stated with respect to $p(y|x),$ but the algorithm operates entirely under $\hat{p}(y|x)$. It appears that $p(y|x)$ and $\hat{p}(y|x)$ are used interchangeably in the theoretical derivations. 
If these two distributions differ significantly, the validity of the stated results becomes questionable. 
In my view, this point requires further clarification.

---

2. In Stage 3, the true set size is approximated by the sum of individual ball volumes (Eq. in Lines 232-235). This is a potentially loose proxy, particularly in high dimensions where overlaps between the balls may be non-negligible. The paper does not analyze or quantify this approximation gap, leaving uncertainty about the accuracy of the optimization objective.

---

3.  The method’s dependence on the generative model warrants a robustness study. Experiments where the generative model is intentionally misspecified would be valuable to evaluate ORCA’s reliability when $ \hat{p}(y|x) \neq p(y|x) $. Without this, the practical limits of the approach remain unclear.

---

4. The MILP-based allocation stage could be computationally expensive for large calibration sets or high-dimensional outputs. The paper does not discuss computational cost, or runtime comparisons to other optimization-based CP methods. Clarification of scalability would be helpful.

---

5. While the pipeline is well-structured, its components, generative sampling (e.g., Wang et al. 2023, PCP), optimization-based allocation (Bai et al. 2022), and vectorized nonconformity scores, are individually not new. The contribution is thus more integrative than conceptual.

### Questions
1. How is the connection between the true conditional distribution $p(y|x)$ and its model-based approximation $\hat{p}(y|x)$ established in theory?

2. Have you tested ORCA under deliberate model misspecification to assess robustness to misspecification in the generative model?

3. What is the computational complexity of the MILP optimization stage, and how does it scale with sample size and dimensionality?

4. Since the approach appears to learn the conditional distribution itself, is it possible to derive conditional coverage guarantees (asymptotically)? If not, could you comment on the main challenges or theoretical barriers in this direction?

---

**Minor Comments and Suggestions:**

1. There are a few minor typographical errors (for example, “inclduing healthcare” in Line 33-34.).

2. What is Eff? Is it prediction set efficiency (as in caption of Table 1) or prediction set size (as stated in Line 368)?

3. The definition of the $m$-nearest neighbor distance, which is used in the score construction, currently appears only in the Appendix. It would improve readability to explicitly reference or restate this definition in the main text (e.g., in Section 3.1) or redirect reader to the def, since it plays a key role in understanding the nonconformity score formulation.

### Soundness
2

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
2

### Summary
This paper proposes ORCA (Optimized Ranking and Coverage Allocation) for generative conformal prediction (CP) that jointly improves model expressiveness and coverage efficiency. 
Traditional CP methods rely on single-point predictors or simple nonconformity scores, while ORCA employs a generative model to sample from the conditional distribution, then formulates an optimization-based coverage allocation using rank-dependent radii over these generated samples. 
The key idea is to allocate coverage more effectively by solving a mixed-integer linear program (MILP) that ensures exact finite-sample validity while minimizing prediction set size.

### Strengths
1. The paper makes a clear contribution by jointly addressing two challenges in CP: model expressiveness and score design. The integration of generative modeling with an optimization-based conformal layer is interesting.

2. The exact MILP reformulation of the coverage allocation problem is good. 

3. The experiments cover synthetic distributions, MNIST, and real-world datasets.

### Weaknesses
W1. Although the MILP formulation guarantees optimality, its computational cost could become prohibitive for large K (number of generated samples) and n_1 (exploration data), which has been acknowledged by the authors, the current approach might be impractical for large-scale or online settings.

Could the authors discuss possible approximate or relaxed solvers for the MILP formulation (e.g., LP relaxation, greedy coverage allocation) and their *empirical* performance trade-offs at least?

W2. The performance of ORCA seemingly depends on the quality of the generative model. If the conditional samples poorly approximate p(Y|X), the optimization may misallocate coverage, degrading efficiency and validity. 

How sensitive is ORCA to the choice of generative model architecture (e.g., VAE or diffusion)? Does the optimization procedure remain effective when the generative samples change. 


W3. Concerning experiemnts:

3a. Experimental comparison with differentiable or learned conformal prediction methods (e.g., neural calibration approaches) should be better.

3b. Figure 2 and Figure 3 are confusing:
For Figure 2, please clarify more experimental details (e.g., settings, goals, and results). For example, the reason you show 3 figures for each digit, and where is "target image"?
For Figure 3, the area of the regions are not reported. Besides, the explanation of ORCA's disconnected results are no so convinced.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

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
This paper introduces ORCA (Optimized Ranking and Coverage Allocation), a novel framework for generative conformal prediction that addresses the inefficiency of conventional uncertainty quantification methods in complex distributional settings. ORCA implements a three-stage approach: (1) exploring distributional geometry via generative sampling with density-based ranking using nearest neighbor distances, (2) formulating coverage allocation as an optimization problem solved exactly through mixed-integer linear programming to minimize set size while maintaining validity, and (3) calibrating the optimized thresholds to ensure finite-sample coverage guarantees. The method produces adaptive, often discontinuous prediction sets that expand in high-density regions and contract in sparse areas. Theoretical analysis proves ORCA maintains exact validity and asymptotically converges to the oracle highest-density region. Empirical evaluation on synthetic and real datasets demonstrates improved efficiency over baselines.

### Strengths
1. The paper addresses a fundamental tension: maintaining statistical validity while achieving computational and practical efficiency. This is particularly vital as modern machine learning increasingly encounters multimodal, heterogeneous data distributions where traditional approaches fail.

2. The paper offers deep insight into how coverage validity and efficiency can be jointly optimized. It reframes CP as a coverage allocation optimization problem, bridging statistical guarantees with optimization and generative modeling. 

3. ORCA achieves highly efficient, adaptive, and discontinuous uncertainty regions: a significant improvement over prior CP approaches such as CQR, PCP, and CRD.

### Weaknesses
1. The efficiency gains heavily depend on the quality of the generative model. When the generative distribution poorly approximates the true conditional distribution, ORCA might produce efficient but misleading uncertainty sets. The paper lacks analysis of performance degradation under systematic model misspecification.

2. Despite the MILP reformulation, the optimization remains computationally intensive for large K and $n_1$. The authors mention this briefly but don't provide concrete complexity analysis or practical guidelines for parameter selection in resource-constrained settings.

3. The m-nearest neighbor ranking assumes sufficient sample density to meaningfully estimate local structure. In the configurations specified in line 366, the ranking could be unstable, particularly in regions of low probability mass. It is better to provide sensitivity analysis or confidence bounds on the ranking quality.

### Questions
Practical MILP solvers often terminate at near-optimal solutions due to time constraints, what is the solution quality when the solver fails to achieve global optimality within reasonable time limits?

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
This work proposes a three-stage framework generative conformal prediction with Optimized Ranking and Coverage Allocation (ORCA) to improve the efficiency of conformal prediction. The paper aims to reduce the inefficiency originating from two sources: limited model capacity and expressiveness, and simplistic nonconformity scores. The framework uses generative models to capture the full conditional distribution and a local density proxy ranking mechanism to prioritize high-density regions. This is followed by optimal coverage allocation to identify the threshold vector given the ranked distance vectors. The authors provide finite-sample validity guarantees and asymptotic results for convergence to high-density regions. The paper also includes experiments on multiple synthetic and real datasets.

### Strengths
The motivation of the paper is clear and important for improving efficiency of conformal prediction. The writing is clear for the most part and the stages of the framework are reiterated and explained; section 3.2 can benefit from more clarity in notation. The empirical evaluation is extensive and multiple relevant baselines have been considered.

### Weaknesses
1. The paper discusses the goal of ORCA to capture the full conditional distribution and the ability to adaptively adjust radii based on density. Given this, I believe the paper warrants an analysis of the conditional coverage of the method. While the paper reports worst-slab coverage in the experiments, the theory lacks this discussion. Studying these implications and tradeoffs is important, especially as the experiments do not show improvement in conditional coverage (e.g., Table 2). [1] could be a relevant paper to refer to that uses rank and density notions to improve conditional coverage (another paper that uses density notion for smaller sets is [2]).
2. Conformal prediction’s strength lies in finite-sample guarantees. While the paper shows finite-sample validity, the results for optimal coverage allocation are weak and have asymptotic properties. Results that provide finite-sample guarantees e.g., high probability bounds/rates with explicit dependence on n, K, m will strengthen the contribution. 
3. While the efficiency of ORCA is slightly better compared to PCP, I couldn’t find discussion on the computational overhead and runtime of ORCA. This discussion is important given the comparisons studied.

[1] Jivat Neet Kaur, Michael I. Jordan, and Ahmed Alaa. Conformal Prediction Sets with Improved Conditional Coverage using Trust Scores, 2025.
[2] Rui Luo and Zhixin Zhou. Density-sorted prediction set: Efficient conformal prediction for multi-target regression. Pattern Recognition, 2026.

### Questions
Comments:
1. Typos: p5 l269 chosen, p6 l302 quantifies
2. It seems the space between the headings and text has been artificially reduced beyond the limit to accommodate content e.g., Section 3.4, Section 5. I would like to flag this here.

### Soundness
2

### Presentation
3

### Contribution
3
