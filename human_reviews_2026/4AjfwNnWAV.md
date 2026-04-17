# Measuring Uncertainty Calibration

- Decision: Accept (Poster)
- Scores: 2, 6, 8

## Abstract
We make two contributions to the problem of estimating the $L_1$ calibration error of a binary classifier from a finite dataset. First, we provide an upper bound for any classifier where the calibration function has bounded variation. Second, we provide a method of modifying any classifier so that its calibration error can be upper bounded efficiently without significantly impacting classifier performance and without any restrictive assumptions. All our results are non-asymptotic and distribution-free. We conclude by providing advice on how to measure calibration error in practice. Our methods yield practical procedures that can be run on real-world datasets with modest overhead.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the problem of estimating the L1 calibration error of a binary classifier from a finite dataset. The paper has two main contributions:
(1) upper bounds on the calibration error under bounded variation assumptions using a variant of total variation denoising, and 
(2) a perturbation method that enforces bounded derivatives, enabling kernel-based estimation. Both approaches are non-asymptotic and distribution-free.

### Strengths
- The proposed methods are non-asymptotic and distribution-free
- The empirical validation includes both synthetic experiments with known ground truth and real-data experiments. Real-data experiments (Amazon Polarity, CIFAR, IMDB, Spam) demonstrate practical applicability.

### Weaknesses
**Missing related work**

The paper would benefit from a more complete discussion of recent literature, e.g. 
https://proceedings.neurips.cc/paper_files/paper/2020/file/26d88423fc6da243ffddf161ca712757-Paper.pdf also addresses distribution-free calibration in binary classification, establishing fundamental limits and impossibility results in the absence of distributional assumptions.

**Insufficient experimental analysis**

The experimental section is almost entirely descriptive rather than analytical. It reads more like preliminary notes or a work-in-progress draft than a finished scientific contribution. 
For example, the paragraph on Real Data Experiment provides only vague descriptive claims without referencing any table/figure, or any scientific insight into why the method performs better.
The lack of analysis, missing experimental details, and superficial treatment of results suggest rushed submission.

**Overall presentation of the paper can be improved**

The paper would benefit from clearer descriptions of tables and figures, explicitly stating what is shown and what conclusions should be drawn. For instance, the table below Figure 3 lacks a caption and is not referenced in the text. 
Footnote 6 appears incomplete, as it does not contain any text. 
Such inconsistencies and lack of polish suggest insufficient attention to detail in the preparation of the paper and raise concerns about whether similar oversights exist in the rest of the paper.

**Overstating contributions**

The conclusion that “we empirically demonstrated it is possible to measure calibration error on a real task of practical importance” might be overstated given that the result is an upper bound, not a point estimate. It would be fairer to emphasize certification rather than estimation per se.

**Limitations not discussed**

For example, the limitation to binary classification is mentioned only in passing, yet this significantly restricts applicability given that most real-world calibration problems are multiclass. A thorough limitations section would strengthen the paper by setting appropriate expectations.

### Questions
How should h be chosen in practice?

ECE uses finite binning and is computationally cheap. How far apart are the derived bounds from ECE estimates?

### Soundness
3

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
3

### Summary
Paper introduces two new approaches for estimating binary classifier output calibration and especially upper bounds for calibration error (CE). These are  derived under bounded variation (with total variation denoising), and perturbation of classifier outputs (with kernel density estimator). Empirical evaluation shows the performance under perturbations (real data), and comparison of proposed bounds against Lipschitz bucketing on gap between the upper bound and true CE (synthetic data) and upper bound tightness (real data) with promising results.

### Strengths
The proposed methodologies are generally sound. To my knowledge, the derived estimators are new contributions to binary classification calibration domain. Paper is well-written and new calibration error upper bounds are motivated by the limitation of previous works and literature. Proposed theoretical results are supported by experimental results on both synthetic and real data. Although targeted on very specific classification setup, the contribution could provide useful information for the community, especially for those working on trustworthy ML. Overall, well-defined contributions to specific problem.

Summary of strengths
- Theoretically sound
- Motivated by the limitation of previous work
- Novel contributions (theoretical with techniques that could apply in practice)

### Weaknesses
There are also some weaknesses affecting the clarity of the presentation and significance of the results. First, although the motivation and derivation of the proposed theoretical results seem ok, the presentation could be improved and supported by illustration of the problem at the beginning, including the problem setup and possible limitations of previous work. Second, the experimental evaluation could be more versatile to fully support the proposed techniques, including the comparison with several real datasets (in addition to Figure 3), and other previous approaches presented in the related work section (in addition to Lip+Bkt). Also, different metrics of calibration accuracy and computational efficiency, and statistical significance of the results in comparison, could improve the practicality and the significance of the proposed contributions  and claims.

Summary of weaknesses
- Lack of illustrative example at the beginning  (motivation of the calibration problem setup, and limitation of previous approaches of vanilla binning etc.)
- Limitations in experiments (comparison to other previous approaches, performance metrics, computational complexity)

### Questions
- Why upper bounds on real data (Figure 3) is showed only for Amazon polarity set and not for other datasets?
- Would it be possible to show the performance also against other baselines mentioned in the related works? and with other performance metrics related to calibration accuracy?
- What is the computational complexity / efficiency of the proposed approaches?
- Are the results statistically significant? 

Other minor comments:
- expectaiton -> expectation (page 3)
- Table X (after Figure 3) does not have caption is not referred in the text at all.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes two variants of upper-bounded calibration error evaluation methods: bounded variation and bounded derivatives. 
Their method of bounded derivatives achieves the tightest bound compared to the baseline.

### Strengths
* The paper introduced novel ways to calculate bounds for calibration error. The authors explained these bounds in detail and provided proofs. The explanations in the main part of the paper were very well written, providing most of the necessary intuitions and insights to be able to follow the paper.
* The related work is well-connected to this work.
* The experiments complement the paper and showcase that their method is giving the best results.

### Weaknesses
* The paper would have benefitted from illustrations about the difference of $\eta$ and $\hat{\eta}$ for TV denoising and for kernel smoothing. Such illustrations could be done for some moderately non-monotonic function, e.g. with total variation above 1 but below 2. While the descriptions were all understandable for me eventually, I think most readers would benefit from such illustrative figures. There was some space remaining for this (if I understand correctly that LLM usage, reproducibility statement, and ethics statement can be outside the 9-page-limit).

* Some of the notation was not described in equations (4) and (5). In particular, TVB(delta') in Eq.(4) and B(delta) in Eq.(5) have not been explicitly defined. I did not find the explicit definitions in the Appendix either. However, conceptually it is understandable what these terms mean.

* It would also be nice to see experiments about how classical calibration evaluation methods such as ECE perform in relation to the proposed methods. It is understandable that ECE does not provide such useful upper bounds, but a comparison would help to understand better the shortcomings of ECE.

* It was not clearly justified why Lipschitz bucketing was used in the experiments as the only earlier method to compare with. 

* Minor issue: at the end of section 3, in the paragraph about Concentration, the notation starts with $\hat{\sigma}^2_n$ but then switches to $\hat{\sigma}^2_{X_i}$. The latter is confusing, because it does not depend only on a particular $X_i$, but all $X_1,\dots,X_n$. Hence, I suggest staying with $\hat{\sigma}^2_n$.

* The section on related work seems to be a bit out-of-date, with one reference in 2024, but the rest are older. Several works on evaluating classifier calibration published in 2025 have been omitted:

Dieye et al (2025). When standard calibration metrics fail in evaluating classifier calibration: A simulation study. In ClaDAG 2025.

Kängsepp et al (2025). On the usefulness of the fit-on-test view on evaluating calibration of classifiers. Machine Learning, 114(4), p.105.

Maalej et al (2025). Evaluating Calibration Techniques for Reliable Predictions. In International Conference on Machine Learning and Soft Computing (pp. 159-175)

### Questions
1) What are the exact definitions of TVB(delta') in equation (4) and B(delta) in equation (5)?

2) Why was Lipschitz bucketing used in the experiments as the only earlier method to compare with?
	* Are there any other methods for comparison, or is Lip+Bkt the only or the best bounded method? 
	* In row 420, the phrase "our proposed" goes with NW but not with TV denoising. This is slightly confusing. I thought TV denoising was also first used in the context of calibration in this paper. Can the TV denoising method be also stated as "our proposed"?

3) How do classical calibration evaluation methods such as ECE perform in relation to the proposed methods? It is understandable that ECE does not provide such useful upper bounds, but a comparison would help to understand better the shortcomings of ECE.

4) One usecase of measuring calibration is when comparing different calibration methods and deciding which method works best, by giving the smallest calibration errors. Could the proposed methods be used to rank calibration methods? Would calculating the lower bound in addition to the upper bound be helpful with ranking which calibration method is better?

5) Why is Lip+Bkt not given as an option in practical advice, as it has tighter bounds than TV?

### Soundness
4

### Presentation
4

### Contribution
4
