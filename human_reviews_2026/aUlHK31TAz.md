# On Coreset for LASSO Regression Problem with Sensitivity Sampling

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6

## Abstract
In this paper, we study coreset construction for LASSO regression, where a coreset is a small, weighted subset of the data that approximates the original problem with provable guarantees. For unregularized regression problems, sensitivity sampling is a successful and widely applied technique for constructing coresets. However, extending these methods to LASSO typically requires coreset size to scale with O(\mathcal{G}d), where d is the VC dimension and \mathcal{G} is the total sensitivity, following existing generalization bounds. A key challenge in improving upon this general bound lies in the difficulty of capturing the sparse and localized structure of the function space induced by the \ell_1 penalty in LASSO objective. To address this, we first provide an empirical process-based method of sensitivity sampling for LASSO, localizing the procedure by decomposing the functional space into separate components, which leads to tighter estimation error. By carefully leveraging the geometric properties of these localized spaces, we establish tight empirical process bounds on the required coreset size. These techniques enable us to achieve a coreset of size \tilde{O}(\epsilon^{-2}d\cdot(\log^3 d\cdot\min\{1,\log d/\lambda^2\}+\log(1/\delta))), which ensures a  (1\pm\epsilon)-approximation for any \epsilon,\delta\in(0,1) and \lambda > 0. Furthermore, we give a lower bound showing that any algorithm achieving a (1+\epsilon)-approximation must select at least $Omega(\frac{d\log{d}}{\epsilon^2}) rows in the regime where \lambda=O(d^{-1/2}). Empirical experiments show that our proposed algorithm is at least 4 times faster than the existing LASSO solver and more than 9 times faster on half of the datasets, while ensuring high solution quality and sparsity.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces ``LASSO-Sens``, the first coreset construction method for the standard LASSO regression problem based on sensitivity sampling.
The key challenge in applying coreset techniques to LASSO stems from the complex, non-smooth geometry of the function space induced by the l1 penalty, which complicates standard analysis . The authors overcome this barrier by developing a localized empirical process method that effectively decomposes the function space into independent residual and l1 penalty components.
This new analysis yields a provably tight coreset size up to logarithmic terms. The authors confirm this bound is nearly optimal by providing a matching lower bound .

Experimental results supplement the theory, demonstrating that ``LASSO-Sens`` is significantly more efficient (four times faster) than the standard LASSO solver and substantially outperforms a uniform sampling coreset baseline, all while maintaining high solution quality and sparsity.

### Strengths
**High-Quality Presentation:** The paper is exceptionally well-written and easy to follow. The authors do an excellent job of situating their work within the existing literature, clearly motivating their approach and making the theoretical breakthroughs highly compelling.

**Significant Theoretical Contribution:** The primary contribution, a comprehensive and provably near-tight coreset size bound, is a significant theoretical achievement. Overcoming the noted analytical hurdles of bounding the sampling error for the LASSO objective is a noteworthy accomplishment. *Though, I am unfamiliar with this field and defer to other reviewers to confirm the novelty of this result within the broader literature.*

**Comprehensive Analysis:** The paper's theoretical claims are well-supported, complete with a matching lower bound in the relevant regime, which confirms the near-optimality of the proposed method. *Again, I defer to other reviewers to confirm that the relationship between $\lambda$ and $d$ is common for comparing the here proven upper and lower bounds.*

### Weaknesses
The primary weakness of the paper lies in its experimental validation. While the theory is the main focus, the empirical results section feels underdeveloped and, in some cases, seems to contradict the narrative that LASSO-Sens is the superior practical approach.

**Overstated Claims of Empirical Superiority:** The text emphasizes multiplicative speedups of ``LASSO-Sens`` over the full LASSO procedure. However, this comparison obscures a more critical one: the performance against the LASSO-Uniform baseline.

**LASSO-Sens vs. LASSO-Uniform:** Across many of the provided plots (Figures 1, 2, and appendix figures), ``LASSO-Sens`` does not demonstrate a clear, significant advantage over ``LASSO-Uniform`` in terms of final loss . In several instances (e.g., Synthetic $\lambda = 0.5, 1$), ``LASSO-Uniform`` even appears to achieve a better loss.

**Lack of Statistical Rigor:** The plots do not include statistical error bars, and the text does not clarify if the results are from a single replication or averaged over many trials. Without this, it is impossible to determine if the small observed differences between ``LASSO-Sens`` and ``LASSO-Uniform`` are statistically significant or simply noise. Given that ``LASSO-Uniform`` is a lighter-weight approach (avoiding the sensitivity score computation), its comparable effectiveness in these experiments weakens the practical argument for the proposed method.

### Questions
**Statistical Significance:** Are the results in Figures 1 and 2 (and appendix figures) from a single run or averaged over the 10 trials mentioned? Could you please add error bars (e.g., standard error or 95% CIs) to the plots to allow for a proper statistical comparison between ``LASSO-Sens`` and ``LASSO-Uniform``?

**Practical Justification:** Given that ``LASSO-Uniform`` performs comparably, and sometimes better, than ``LASSO-Sens`` in the provided experiments, could you further emphasize the empirical justification for using the more complex sensitivity sampling method? The theoretical advantage is clear, but the practical advantage over this strong, simple baseline is not.

**Cost of Sensitivities:** Do the runtimes reported for ``LASSO-Sens`` (e.g., in Table 1) include the pre-processing time required to compute the sensitivity scores? A clear breakdown of the (sampling vs. solving) costs for both coreset methods would be essential for a fair comparison.

### Soundness
3

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
This paper studies coresets for standard LASSO (least-squares + $\ell_1$ penalty) using sensitivity sampling. The authors propose a sensitivity-based sampling algorithm for augmenting matrix A′ = [A −b] and localizing the function class into a residual $\ell_2$ part and an $\ell_1$‑penalty part, then, they apply empirical process and chaining tools to bound Gaussian diameter and metric entropy on the localized sets, and finally uses those bounds to show a coreset of size is smaller than linear regression.

### Strengths
- Localizing the analysis into residual $\ell_2$ and $\ell_1$ penalty components and applying Gaussian/chaining bounds on each piece is a fresh and appropriate way to tackle the complexity introduced by the $\ell_1$ term.

### Weaknesses
- A thorough discussion and comparison with the existing literature, such as Avron et al and Chhaya et al, is missing.

### Questions
1. In Theorem 7, why $\log(\frac{1}{\delta})$ is an additive term in the coreset size?

2. The lasso negative result in Chhaya et al seems to be general, that is, a smaller strong coreset or sketch or summarization is impossible for any $\lambda$, i.e., the claim is independent of how the coreset is constructed and then analyzed. Can you clarify how your result negates this claim and if it does not negate, then how do you give a strong coreset guarantee on lasso. 

3. How is the coreset size using the localization of the function class related to other standard regularized regressions, such as ridge (Avron et al) and modified lasso?

### Soundness
2

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
1

### Summary
The paper considers the problem of coreset construction with sensitivity sampling for LASSO. Note that the coreset construction via sensitivity sampling is a known method. The main contributions of this paper are providing theoretical guarantees for sensitivity sampling in LASSO. The authors then provide some experiments to validate their methods.

### Strengths
The paper is clear and easy to follow.

### Weaknesses
First things first, I am not an expert in this field, so my evaluation might be unreliable. I did not check the proof, and I will revisit it in the rebuttal phase. Here are my two cents on the paper's weaknesses.

1. Comparison with sketching methods: I see that the experiments of this paper were only conducted with Vanilla LASSO, sensitivity sampling, and uniform sampling. However, they did not compare this method with other approaches for handling big data, like projection methods/sketching. As a general audience, I expect a comparison here, to see if I should really care about sensitivity sampling for LASSO, or if advanced sketching methods for LASSO should work for me/or even work better than the proposed method, when the number of samples ($n$) is large.

2. Comparison with modified LASSO coreset: again, I see that the problems of sensitivity sampling with modified LASSO (using $\|x\|_1^2$ instead of $\|x\|_1$) are also considered. The authors should also compare their method with this one. Although the authors claimed that the modified LASSO introduces some unexpected correlation between features, it is good to know how their performance compares to their proposed approach. 

3. Cost of sensitivity score calculation: I see that the computation of the coreset score scales cubically ($O(d^3)$) with the number of features $d$. It should not be the problem for the case $d \ll n$ as the authors considered. However, in a general high-dimensional case, $d$ will scale with $n$, and if $d/n = \alpha$ for a constant $\alpha$, there would be a serious problem. I expect that in such a case, sketching methods would work better. Can the authors comment on this point?

4. Dependence on regularization parameter $\lambda$: I see that the bound on the coreset size includes a term proposional to $1 / \lambda^2$. I expect that things would get ugly when $\lambda$ is close to 0 (e.g., weak regularization). Can the authors comment on this point?

Overall, I am uncertain whether the experiments presented in this paper are sufficient. However, I think that the main punchline of this paper is not totally about the experiments, but the theoretical guarantees the authors established for the coreset construction with sensitivity sampling for LASSO. However, I am not an expert and can only have lukewarm support for this paper, with a low confidence score.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3
