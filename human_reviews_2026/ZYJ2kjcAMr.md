# A Kernel Distribution Closeness Testing

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
The distribution closeness testing (DCT) assesses whether the distance between a distribution pair is at least $\epsilon$-far. Existing DCT methods mainly measure discrepancies between a distribution pair defined on discrete one-dimensional spaces (e.g., using total variation), which limits their applications to complex data (e.g., images). To extend DCT to more types of data, a natural idea is to introduce maximum mean discrepancy (MMD), a powerful measurement of the distributional discrepancy between two complex distributions, into DCT scenarios. However, we find that MMD's value can be the same for many pairs of distributions that have different norms in the same reproducing kernel Hilbert space (RKHS), which potentially have different closeness levels, making MMD less informative when assessing the closeness of multiple distribution pairs. To mitigate the issue, we design a new measurement of distributional discrepancy, norm-adaptive MMD (NAMMD), which scales MMD's value using the RKHS norms of distributions. Based on the asymptotic distribution of NAMMD, we finally propose the NAMMD-based DCT to assess the closeness level of a distribution pair. Theoretically, we prove that NAMMD-based DCT has higher test power compared to MMD-based DCT, with bounded type-I error, which is also validated by extensive experiments on many types of data (e.g., synthetic noise, real images).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes NAMMD (Norm-Adaptive Maximum Mean Discrepancy), a normalization of the standard MMD that divides the discrepancy measure by the sum of the RKHS norms of the kernel mean embeddings. This normalization is intended to adjust for scale differences between distributions and to improve robustness when comparing distributions with different variances or magnitudes. The authors demonstrate that under certain norm conditions (Theorem 9), NAMMD can outperform MMD, and they apply the resulting statistic to distribution closeness testing (DCT), comparing it with Canonne’s total-variation-based DCT. Experimental results show promising improvements in detection power across several benchmarks.

### Strengths
- The paper introduces a conceptually simple yet elegant modification of MMD that aims to adapt to distributional scales, a long-standing issue in kernel-based two-sample testing.

- The theoretical analysis (especially Theorem 9) provides partial intuition for when NAMMD may outperform MMD.

- The empirical results indicate improved detection power in some settings, particularly for scale-shifted distributions.

- The work situates itself in the broader line of kernel-based hypothesis testing and could inspire further studies on adaptive or normalized discrepancies.

### Weaknesses
- The motivation for the specific normalization by the sum of RKHS norms remains unclear. It is not obvious why this normalization is preferable to other kernel-based measures such as Kernel Canonical Correlation Analysis (KCCA; Akaho, 2001) or Hilbert–Schmidt Independence Criterion (HSIC; Gretton et al., 2005), both of which normalize by feature-space variance or covariance to account for scale differences. Clarifying the conceptual distinction between NAMMD and these established approaches would help position the contribution more precisely.

- The paper does not clearly situate NAMMD relative to prior normalized or relative MMD variants, including Normalized MMD (Muandet et al., 2012) and Relative MMD (Bounliphone et al., 2015). It remains somewhat ambiguous whether NAMMD introduces a fundamentally new normalization principle or whether it can be viewed as a reformulation of these earlier approaches.

- The paper should explain how p-values are computed in the NAMMD-based test (e.g., via asymptotic approximation, permutation, or bootstrap). Since the normalization modifies the scaling of the MMD statistic, its asymptotic variance and null distribution could differ from those of standard MMD. A comparison would clarify how the normalization affects calibration and test power.

- Theorem 9 considers only a specific norm condition favoring NAMMD. A more systematic analysis of how kernel bandwidths or scale ratios between P and Q influence this condition would clarify whether it is merely sufficient or also necessary. 

- In Section 5.1, only average test power is reported. The empirical type-I error rates of both Canonne’s total-variation-based DCT (Canonne et al., 2023) and the NAMMD-based DCT are not shown, making it unclear whether both tests operate at the same nominal significance level. Reporting empirical type-I error under the null hypothesis would make the comparison more rigorous and interpretable.

[References]
- Akaho, S. (2001). A kernel method for canonical correlation analysis.
- Gretton, A., Bousquet, O., Smola, A., & Schölkopf, B. (2005). Measuring statistical dependence with Hilbert–Schmidt norms.
- Muandet, K., Fukumizu, K., Sriperumbudur, B. K., & Schölkopf, B. (2012). Learning from distributions via support measure machines.
- Bounliphone, W., Bellet, A., & Tommasi, M. (2015). A test of relative similarity for model selection in generative models.
- Canonne, C. L., Kamath, G., & Steinke, T. (2023). Distribution closeness testing via total variation.

### Questions
- Conceptual distinction — Could the authors clarify how NAMMD differs conceptually from correlation-based kernel measures such as KCCA (Akaho, 2001) or HSIC (Gretton et al., 2005)? In what sense does the normalization by the sum of RKHS norms capture a different form of scale or variance adjustment?

- Relation to prior normalized MMDs — How does NAMMD relate to previously proposed normalized or relative MMD variants, such as those by Muandet et al. (2012) or Bounliphone et al. (2015)? Is the proposed normalization theoretically novel, or can it be interpreted as a reparameterization of these approaches?

- Statistical inference — How are p-values estimated in the NAMMD-based test? Since the normalization changes the scaling of MMD, does its null distribution differ in variance or asymptotic form? A short explanation or comparison would improve reproducibility and theoretical clarity.

- Norm condition in Theorem 9 — Theorem 9 focuses on a specific norm condition that favors NAMMD. Could the authors discuss whether this condition might fail, and what happens in such regimes? For example, would NAMMD reduce to standard MMD, or behave pathologically?

- Empirical validity — In Section 5.1, the comparison with Canonne’s total-variation-based DCT lacks type-I error reporting. Could the authors include empirical type-I error rates under the null to ensure that both methods are calibrated at the same nominal significance level (e.g., α = 0.05)?

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
The paper studies the problem of distribution closeness testing (DCT), extending the usual two-sample test. The authors propose to use a normalized version of maximum mean discrepancy for this task. Asymptotic theories are developed for their tests.

### Strengths
1, The experiments are broad and cover both synthetic and real-world datasets.
2, The paper identifies an important but underexplored problem: distribution closeness testing (DCT)

### Weaknesses
1, the paper proposes a normalizing approach, but the paper does not rigorously prove that this scaling yields an optimal variance normalization or minimizes any formal criterion (e.g., unbiasedness or asymptotic efficiency). Thus, NAMMD’s normalization remains heuristic rather than theoretically grounded.

2, I find it somewhat concerning that the theory does not unify DCT and TST despite superficial similarity. The paper reverts to permutation calibration, admitting that the asymptotic distribution does not apply when $\varepsilon = 0$. 

3, While the paper provides many figures and comparisons, key experimental parameters (e.g., sample sizes for Fig. 1, variance of estimators) are not reported, limiting reproducibility and interpretability of the less informative claim.

4, Although DCT is motivated as a closeness test, the practical interpretation of $\varepsilon$ (what level of MMD/NAMMD difference implies model transferability or acceptable domain shift) is not concretely defined. In the ImageNet experiments, the choice of $\varepsilon$ and its connection to performance metrics appear ad-hoc.

5, Although DCT is motivated by high-dimensional tasks, the paper does not analyze NAMMD’s behavior under the curse of dimensionality.

6, Most importantly, this papers does not discuss or compare NAMMD to other variance normalized versions of MMD.

### Questions
I find figure 1 very misleading and have the following questions:

1, how is the $p$-value computed? Do you use permutations to get the $p$-value.

2, Assuming the p-values are computed using permutations, it depends on the sample size, which is not reported. The number of repetitions is not reported either.

3, Theoretically, when P is not equal to Q, MMD converges to a positive number. While if you permute the samples, it converges to 0. So, p-value is expected to be small.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, authors propose a testing statistic called nom-adaptive MMD for distribution closeness testing. Comparing to the traditional MMD, NAMMD adds a normalisation term, 
and fixes the issue that MMD is the same for kernel mean embeddings with different RKHS norms.  The authors several a testing procedure to examine the closeness of the distributions and shows that the proposed NAMMD has a higher testing power on toy and real world problems.

### Strengths
The closeness test is indeed different from traditional two-sample tests and represents an underexplored area in hypothesis testing, with potential downstream applications such as classifier transfer. 

The issue that kernel mean embeddings in MMD can have different RKHS norms but yield the same MMD value is a legitimate concern, and the solution proposed in Equation (1) is both elegant and easy to implement in practice. 

The proposed NAMMD method appears to consistently achieve higher testing power than MMD and total variation–based test statistics, as demonstrated in the experimental results.

The paper is generally well-written.

### Weaknesses
The main concern with this paper is that its technical innovation and motivation appear to be unrelated. From Figures 1(a) and 1(b), I can indeed see that MMD is not ideal when a and b have the same MMD value but their kernel mean embeddings have different RKHS norms. However, it is unclear why this becomes an issue specifically in the context of DCT. Without establishing this connection, the motivation remains weak. If DCT is susceptible to this particular issue of MMD, the authors should clarify in the introduction.

The authors also do not clearly explain what distinguishes the DCT test statistic from traditional TST test statistics. From the definition in line 117, it seems that as long as a TST statistic outputs some form of dissimilarity measure, it could be used for DCT as well. This further adds to the confusion about why we should focus on the RKHS norm issue discussed in the previous section, as it is not evident how normalizing the MMD resolves any problem when using a TST statistic (MMD) for DCT.

In the introduction: 

> "Besides, extending these methods using continuous total variation involves the estimation of the underlying density functions of the distributions [25, 26]" 

However, this is not accurate. There have been many efforts to adapt total variation to continuous data. For example, [1] propose a general nonparametric estimator for integral probability metrics (of which TV, MMD, and Wasserstein distances are special cases). Similarly, [2] introduce a general framework for estimating f-divergences (of which TV is one example), which has been widely adopted in generative model training.

[1] https://arxiv.org/abs/0901.2698
[2] https://arxiv.org/abs/1606.00709

The empirical comparison also does not include other widely used discrepancy measures beyond MMD and TV in the discrete example. For instance, f-divergence–based measures have been used in TST and could serve as a natrual candidate test statistics for DCT.

### Questions
What is the main reason that authors consider MMD over other test statistics? For example, Wasserstein distance and classic divergence, both can be extneded to continous variable settings without estimating densities (see [1] and [2]). 

Line 085: Specifically, the MMD value can be the same for many pairs of distributions that have different norms in the RKHS Hκ, which potentially have different closeness levels. 

Figure 1 a and b are not helpful demonstrating this as P and Q are equally close in both figures, depending on how close you zoom. Could authors find a better illustration of this problem? 

In Figure 1, c and d, why is it not desirable to have MMD stays constant while the p-value of TST changes? 

Line 312, how realistic is the condition ||mu_p1||^2 + ||mu_q1||^2 <= ||mu_p2||^2 + ||mu_q2||^2? It looks like that the main results in the proof depends on this particular assumption and there are some explanations in Section 6.2. However, it seems to only suggest the sum of variance of kernel embedding of p1 and q1 should be greater than that of p2 and q2. How strengient is this condition in reality? Could this condition be translated into some more interpretable?

### Soundness
3

### Presentation
3

### Contribution
2
