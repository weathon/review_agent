# When three experiments are better than two: Avoiding intractable correlated aleatoric uncertainty by leveraging a novel bias--variance tradeoff

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 6, 2

## Abstract
Real-world experimental scenarios are characterized by the presence of heteroskedastic aleatoric uncertainty, and this uncertainty can be correlated in batched settings. The bias-variance tradeoff can be used to write the expected mean squared error between a model distribution and a ground-truth random variable as the sum of an epistemic uncertainty term, the bias squared, and an aleatoric uncertainty term. We leverage this relationship to propose novel active learning strategies that directly reduce the bias between experimental rounds, considering model systems both with and without noise. Finally, we investigate methods to leverage historical data in a quadratic manner through the use of a novel cobias-covariance relationship, which naturally proposes a mechanism for batching through an eigendecomposition strategy. When our difference-based method leveraging the cobias-covariance relationship is utilized in a batched setting (with a quadratic estimator), we outperform a number of canonical methods including BALD and Least Confidence.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a novel framework for active learning that leverages the bias–variance–noise decomposition to better distinguish between reducible and irreducible sources of uncertainty. It introduces acquisition strategies that selectively target epistemic uncertainty, model bias, or a combination of both, enabling more principled experiment selection. They evaluate their methods on a set of synthetic toy experiments designed with controlled noise assumptions, showing that their approach outperforms standard uncertainty-based baselines such as BALD and LC.

### Strengths
1. The paper is well written.

2. The separation of uncertainties through the bias–variance–noise decomposition is intuitive and well-motivated, making the proposed framework conceptually clear and broadly applicable across different active learning settings.

### Weaknesses
1. The paper does not consider or compare against recent state-of-the-art active learning methods such as BatchBALD [1], BADGE [2], and BAIT [3]. These approaches are well-established benchmarks for active learning.

2. The paper relies entirely on toy experiments to demonstrate its effectiveness. While these experiments are useful for illustrating theoretical properties, the method is not purely theoretical and would therefore benefit significantly from evaluation on real-world datasets.

3. The structure of the paper is somewhat confusing. Section 2 includes a numerical experiment comparing the proposed methods to BALD and showing improved performance, yet BALD is subsequently disregarded in the main numerical experiment section, Section 4.

4. It is not explicitly stated what models are being trained in the experiments. Are they are neural networks, Gaussian processes, random forests, or linear models?

5. There is no limitations or weakness sections.

6. Minor Issues:

- The caption for Figure 1 blends into the body text and should be visually separated for improved readability.  
- A visualization of each dataset listed in Table 2 would make the paper more accessible and easier to interpret.  
- On line 157, should the expression be $\delta_{\inf}(x) = 0$ instead of $\delta_{\inf} = 0$?
- The legends in Figures 2 and 4 are confusing. For example, in Figure 4 the solid blue line is unlabeled, and the legend references a “random” (solid black) line that does not appear in the figure. Consider moving legends outside the plot and ensuring every line has a corresponding label.

[1] Kirsch, A., van Amersfoort, J., & Gal, Y. (2019). BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning. NeurIPS 2019.

[2] Ash, J. T., Zhang, C., Krishnamurthy, A., Langford, J., & Agarwal, A. (2020). Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (BADGE). ICLR 2020.

[3] Kim, J., Lee, Y., & Shin, J. (2021). BAIT: Bayesian Active Learning by Information Theoretic Measures. NeurIPS 2021.

### Questions
1. Why is Random better than BALD? Figure 2

2. Why does Figure 4 & 5 include no BALD or other baseline methods?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a novel quantity derived from the bias-variance trade-off of statistical models that can serve as an optimization target for Active Learning (AL) acquisition functions. 
They explain the derivation of this quantity in great detail and construct a series of toy experiments with different structures in the noise term to showcase the properties of their proposed method.
As this novel quantity is unknown in practice, the authors propose several way of approximating it and benchmark the variants of their methods against existing AL methods, as well as "cheated" version of their method that has direct access to the target quantity.

### Strengths
- Very well structured construction of the novel quantity that the authors are planning to optimize
- Principled experiment to show the potential of they method (if a perfect approximation would be possible)
- Multiple approaches to approximate the novel quantity and initial comparison against competitors
- Clear path for improving upon the proposed method

### Weaknesses
- (Minor) It is well known in AL that uncertainty-based methods (like the proposed one) are struggling for small labeled sets, as the model F cannot be appropiately trained. A comparison of Fig. 2 and Fig. 6 (App. A) clearly hints in this direction. For that reason, we really would have liked to see a comparison against a diversity based method like TypiClust or ProbCover to acertain, if the "cheat" variant of the proposed algorithm can surpass diversity based acquisition functions.
- (Minor) Your comparison between "BiasReduction" and "PEMSE" variants (Fig. 4) is lacking depth. Both methods appear to be equal in performance, which should only be true if we only use MSE as metric. The advantage the PEMSE method can generate can only be visible in the calibration of the network, which should be easy to check by comparing $\sigma_F$ to $\sigma_Y$ at all 2500 points of your data grid. This way you could highlight the advantage that PEMSE should have over BiasReduction for model training.

### Questions
- In Fig. 2, do you mean "Cheat" and "Approximated" instead of "False" and "True"?
- Is Fig. 4 missing a curve for Random or is "NA" supposed to be random?

### Soundness
4

### Presentation
4

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
The paper presents a novel approach to active learning (AL) that seeks to optimize data acquisition by explicitly 
addressing the bias-variance tradeoff in noisy, heteroskedastic and correlated real-world settings. The core idea is to 
leverage a decomposition of the pointwise expected mean squared error into epistemic uncertainty, bias, and irreducible 
aleatoric uncertainty. The authors introduce a "cobias-covariance tradeoff" and propose a quadratic bias estimation method, 
to guide data point acquisition that effectively reduces model bias and epistemic uncertainty.

### Strengths
The paper offers a compelling motivation for using the bias-variance tradeoff as a guiding principle for AL acquisition 
functions. The explicit consideration of heteroskedastic and correlated noise is generally an interesting and practically
relevant setting.

### Weaknesses
- **Major Presentation and Structural Issues:** The paper's most significant weakness lies in its poor presentation and 
lack of clear structure. The narrative is disjointed, frequently mixing theoretical derivations with experimental 
results and ablation studies across various sections (e.g., experimental results in Section 2.4, followed by another 
dedicated Section 4). This significantly hinders readability and comprehension. A dedicated section for related work is 
also notably absent, making it difficult to contextualize the contributions.
- **Lackluster Problem Statement and Notation:** The problem statement and introduction of notation (Section 2.1) are vague
and inconsistent. Here are a few examples:
  - The variable $Y$ is initially introduced as a "random variable in space $\mathcal{Y}$" but later treated as a 
  stochastic process $Y(x)$
  - What does $\mathbb{E}_\mathcal{Y}[]$ mean? Is $\mathbb{Y}$ a probability space?
  - $Q_k \cap L_k = \emptyset$ (section 2.3.1) should be $Q_k \cap L_{k-1} = \emptyset$
  - Equation 7, introducing a "gradient" with respect to an integer step 'k', is mathematically ill-defined.
- **Insufficient Experimental Detail in Main Text:** Crucial information about the experimental setup, such as the specific 
model architecture used is omitted from the main body. I acknowledge that Appendix B contains details on the experiments.
However, this makes it impossible for readers to fully grasp the methodology without referring to supplementary material, 
which should ideally contain additional details, not core methodology.
- **Limited Experimental Scope:** The evaluation only considers three specifically designed toy examples. While these toy 
examples aim to highlight specific aleatoric noise conditions (i.e. no noise, uncorrelated noise and correlated noise), 
the absence of evaluation on more (realistic) benchmarks raises questions about the generalizability and practical 
applicability of the proposed methods.
- **Unclear Algorithmic Variants and Baselines:** The descriptions of algorithmic variants and baselines are not consistently 
clear, making it challenging to understand how comparisons were made.

Based on the significant issues in presentation, clarity, and structural organization, alongside some fundamental 
inconsistencies in mathematical notation and insufficient experimental detail in the main text, I strongly recommend 
rejection of this paper in its current form.

### Questions
No specific questions, please comment on the points below "Weaknesses"

### Soundness
1

### Presentation
1

### Contribution
2
