# Random Projections for Spectral Algorithms in Mis-specified Setting: Sobolev Norm Learning Rates and Minimax Optimality

- Avg Score: 5.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 8

## Abstract
Random projections (RP) offer an effective approach to reducing computational and storage costs while preserving the geometric structure of the data. However, existing studies primarily focus on the optimal generalization performance of specific kernel-regularized algorithms with RP in the well-specified setting under restrictive conditions. In this paper, we provide a comprehensive and improved analysis of the generalization performance of RP-based spectral algorithms under general conditions, without increasing computational complexity. By leveraging the embedding property of the RKHS and a refined analysis of the operator similarity, we establish optimal learning rates in Sobolev norms that match the minimax lower bounds up to logarithmic factors. For both randomized sketches and Nystr\"{o}m sub-sampling (uniform or leverage-based), we show that the projection dimension needed for optimality is proportional to the average or maximal effective dimension, yielding a significant reduction in computational cost while maintaining the statistical efficiency.  Our results do not rely on the uniform boundedness assumption on the target function and hold for a broad range of source conditions, i.e., $s\geq \alpha-1/\beta$, where $s,\beta$, and $\alpha$ denote the smoothness index, capacity index, and the embedding index, respectively. In the benign case when $\alpha=1/\beta$, the optimality holds for all $s\in (0,2\tau]$ with $\tau$ denoting the quantification index. Experimental results confirm our theoretical findings and demonstrate the practical effectiveness of RP.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the generalization error of scalar-valued regression estimators derived from spectral algorithms within RKHS, specifically when combined with Random Projection techniques (including Nystrom methods). The authors aim to provide a unified analysis that incorporates several refinements previously studied in isolation: handling the misspecified case (by taking into account the embedding index $\alpha$), relaxing assumptions on the target function's boundedness, using general source conditions, applying general spectral algorithms, incorporating random projections, and analyzing convergence in Sobolev norms. The main results presented are convergence rates (upper and lower bounds) under these combined conditions, with a claim that these rates avoid the saturation effect typically seen in kernel methods.

### Strengths
* The paper tackles an important and relevant problem for the ICLR community: understanding the theoretical properties of scalable kernel methods (like those using Nyström or other random projections) under realistic assumptions (mis-specification, general smoothness).
* The paper attempts to cover a wide range of modern theoretical aspects relevant to kernel regression, potentially offering a more complete picture.

### Weaknesses
Despite the interesting goal, the paper suffers from significant weaknesses, including potentially erroneous central claims and numerous presentation errors. These issues undermine the reliability of the results and require major revisions.

**Major Issues**

* **Erroneous Claim on Saturation Effect**: The paper claims that the derived rates ``do not exhibit saturation effect.'' This contradicts established theoretical results for kernel methods, which shows that saturation is unavoidable for KRR (Zhang, 2023). The error appears to stem from Lemma 8, specifically the application of Eq. (13), which seems to ignore the condition $\nu \leq \tau$. A correct application likely leads back to the standard saturation condition $s \leq \tau + \gamma$ (see [1]).
* **Issues with General Source Condition (GSC)**: The paper emphasizes the GSC as a significant generalization, but its presentation is flawed ($\tau$ used before definition, parameter $s$ missing from $f_{\rho} \in \Omega_{\phi,R}$). More critically, the analysis later appears to rely on the property $P\phi(A)P=\phi(PAP)$ (P projection, A operator). It's questionable if any commonly used non-linear $\phi$ (like the Hölder case) satisfies this under the paper's assumptions. This potentially invalidates the applicability of the results.

**Other Issues**

* **Uniformly Bounded Eigenfunctions (UBE)**: Example 1 (line 1041) suggests that the UBE assumption is a common or mild condition, citing prior work that makes this claim. This is mathematically incorrect and spreading this misconception is problematic. See [2] for a counter-example of the claim.
* **Lack of Precision Regarding Measures**: Examples 2 \& 3 claim certain RKHSs are "benign" without specifying the crucial dependence on the underlying probability measure. These properties often hold only for specific measures (like the uniform measure). Similarly, Section C.3.2 discussing Sobolev embeddings needs to specify the assumed measure. Without this, the claims are ill-defined or potentially false.
* **Lower bound**: The theorem statement claims only a bound on the effective dimension is needed, but the proof appears to use lower and upper bounds on eigenvalues. The theorem statement should precisely list all necessary assumptions. Assumption 3 and the mis-specification parameter $\alpha$ are mentioned in the theorem and discussion but seem irrelevant to the lower bound analysis. As for previous lower bound (e.g. Zhang 2024) the bound is valid for any smoothness level irrespective of $\alpha$.
* **Typos** (064-083)  stiuded, projectins (twice), (097) ,our, (143) the reproducing property holds that, (263) that leverage operator spectral, (326) plain Nyström, (350) an projection, (772-1487-1546-1563-1698-1713-) various typos with norms, (808) wwe, (1111) determined by $r$, (1131) by allowing the index function by allowing the index function, (1442) Assumption 2 (should it be 4?)...
* **Redundant Proofs & Lack of Attribution**: Section C.6 and the proof of Lemma 4 appear to reproduce technical results already established in the literature (Zhang 2024) without clear attribution or justification for their inclusion. This makes the paper appear more technically dense than necessary and obscures the novel contributions. Similarly, how different are the proofs of Lemma 16 and 17 compared to Theorem 13 and 15 in Zhang 2024? Lemma 23 can be found in Fischer & Steinwart. 

[1] Blanchard and Mucke. Optimal rates for regularization of statistical inverse learning problems, 2018.
[2] Minh, Niyogi, and Yao. Mercer’s theorem, feature maps, and smoothing, 2006.

### Questions
* **Mercer Assumptions:** The analysis assumes the domain $\mathcal{X}$ is compact and the kernel $K$ is Mercer. However, much of the modern analysis of kernel methods relies on weaker conditions, often only requiring the kernel to be square-integrable w.r.t. the measure to ensure the integral operator is Hilbert-Schmidt \citep{fischer2020sobolev}. Could you clarify why the more restrictive classical Mercer conditions are invoked or needed?

* Could the authors detail how they go from $||O_{K,n,\lambda}^{1/2} g_{\lambda}(O_{K,n})P C_{K,n,\lambda}^{1/2}||$ to $||O_{K,n,\lambda} g_{\lambda}(O_{K,n})||$ (1682-1684), I could not convinced myself of this step. 

* Have the authors considered extending their analysis to the vector-valued setting (relevant for conditional mean embeddings [1]) or incorporating recently proposed relaxations of noise assumptions [2]?

[1] Li, Meunier, Mollenhauer, Gretton. Optimal rates for regularized conditional mean embedding learning. 2022
[2] Mollenhauer, Mucke, Meunier, retton. Regularized least squares learning with heavy-tailed noise is minimax optimal. 2025.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper provides a comprehensive and improved analysis of the generalization performance of RP-based spectral algorithms under general
conditions, without increasing computational complexity. The focus is on generalization performance in mis-specified settings (where the target function may not lie in the RKHS) under general conditions, without assuming uniform boundedness on the regression function. Matching upper and minimax lower bounds in Sobolev norms are established to show the optimality of the algorithm. Numerical experiments confirm theoretical rates and practical benefits.

### Strengths
* This paper is well-written and the structure is clear, despite the technical complexities. A table summarizing the related results is provided, allowing for a easy comparision.

* This paper presents a comprehensive analysis of the spectral algorithms with random projection, unifying Radomized Sketches, plain Nystrom and ALS Nystrom. The settings are general regarding the source condition, embedding properties and algorithms. Moreover, the assumptions are discussed in detail.

* The theory in this paper is strong, proving the minimax optimal rates of SARP. Moreover, the required projection dimension is less than or equal to that required in the literature, providing theoretical guarantees for reducing computational costs.

* Empirical experiments are provided to validate the theory.

### Weaknesses
* Assumption 4 cannot hold for $\phi(u) = u^{s/2}$ when $s >2\tau$. Consequently, the saturation effect still holds for KRR, in contrast to the claim in the paper. See the proof of Lemma 8 and Proposition 2.

* The technical contribution in this paper seems to be marginal. The proof idea and steps seem to be standard. A overview of the technical novelty can be provided and emphasized.

* It would improve the contribution of this paper to propose practical criteria for choosing the projection dimension.

### Questions
1. While the proof in this paper is erroneous, is it possible that indeed SARP does not suffer from the saturation effect? Do we have supporting empirical evidence?


1. Is the current requirement of the projection dimension $m$ minimal for the optimal rates? Is it possible to establish a lower bound for it, or what are the difficulties here?

2. Can you weaken requirement that $\mathcal{X}$ is compact? What will be the impact of non-compactness to the effectiveness of random projection methods?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper develops a comprehensive study about the generalization performance of RP-based spectral algorithms under more general conditions. 

The main modification of assumptions in this paper includes two parts. Firstly, it proposes to impose classical Bernstein condition on the noise $\epsilon$ for the purpose of replacing uniform boundedness of $f_p,$ which may be difficult to satisfy in practice. Then, the paper expands traditional H\"older source condition to assumption 4 that covers more general cases. 

Under the new and general assumptions, this paper first derives a minimax lower bound for the learning rate in Sobolev norms, which is the first to generalize current conclusions to both well-specified and mis-specified cases. Afterwards, it presents the sharp learning rates for general spectral algorithms with random projections (SARP), which recovers previous result for spectral algorithms without RP. Furthermore, it finds the optimal learning rates that up to a logarithmic factor of the minimax lower bounds. Finally, paper applies above theorem to three algorithms, including randomized sketches, plain and ALS Nystr\"om sub-sampling.

### Strengths
1. This paper is a solid work with in-depth theoretical proofs.

2. This paper is the first to establish bounds under general conditions, which have not been fully solved by previous works. 

3. This work includes a large number of appendices to explain their findings, including Table 2, which provides a detailed comparison of various conditions.

### Weaknesses
1. Although this paper is built on more general conditions, all bounds in this paper recover previous results without getting a tighter bound. This may weaken the contribution of this work.

2. The lack of thorough discussion about specific cases of generalization of conditions (e.g. mis-specified setting and uniformly boundedness). Since one of the most important contribution of this paper is the generalization of conditions, a more thorough discussion about it can greatly demonstrate the contribution of this work. It's not very clear to me what new situations are covered in practice because of mis-specified setting and no uniformly boundedness on $f_p$.

3. Lack of comparison about conditions with previous works. There are some other works also built on more general conditions according to Table 2. Moreover, this work introduces embedding as a condition, which is not required in most other works. Thus, the general condition of this work is in doubt and needs more discussions to prove.

### Questions
1. Why projection dimension needed for optimality is proportional to the empirical effective dimension? The conclusion is not so straightforward to me based on current presentation.

2. How to select appropriate projection dimension and regularization parameter in practice?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a comprehensive theoretical analysis of Spectral Algorithms with Random Projections (SARP), establishing their minimax optimality under very general conditions, particularly in the common scenario where the true function may not belong to the model's hypothesis space (the "mis-specified setting").

### Strengths
By applying recent developments in spectral algorithms, the authors analyzed random projection methods and derived convergent results.

This is an interesting research direction that merits further investigation in future work.

### Weaknesses
I did not check the entire proof. The paper would benefit from a more careful revision of the writing.

1. Can you provide more examples such that assumption 2 holds?

2. Could you be more careful on the assumption 3? It is ambiguous for non-expert.  

This is an interesting topic, however, it needs to be more careful on presentation.

### Questions
Same to the weakness.

### Soundness
3

### Presentation
2

### Contribution
3
