# Smoothing-Based Conformal Prediction for Balancing Efficiency and Interpretability

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Conformal Prediction (CP) is a distribution-free framework for constructing statistically rigorous prediction sets. While popular variants such as CD-split improve CP’s efficiency, they often yield prediction sets composed of multiple disconnected subintervals, which are difficult to interpret. In this paper, we propose SCD-split, which incorporates smoothing operations into the CP framework. Such smoothing operations potentially help merge the subintervals, thus leading to interpretable prediction sets. Experimental results on both synthetic and real-world datasets demonstrate that SCD-split balances the interval length and the number of disconnected subintervals. Theoretically, under specific conditions, SCD-split provably reduces the number of disconnected subintervals while maintaining comparable coverage guarantees and interval length compared with CD-split.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper is well-written, easy to follow, and clearly presents its main ideas.

Problem context: Popular conformal prediction (CP) methods, particularly conditional-density-based approaches such as CD-split, achieve strong efficiency but often produce discontinuous prediction sets. The study is motivated by the observation that, when the conditional distribution is complex or multimodal, CD-split often yields prediction sets composed of many small, disjoint intervals. Such fragmentation reduces interpretability and practical usability in domains like healthcare or finance, where users prefer simple and contiguous prediction intervals. The proposed smoothing-based approach aims to directly address this interpretability concern without compromising statistical validity.


Paper's Proposal: The authors propose a smoothed conformal prediction framework, SCD-split, which incorporates a smoothing step into the CD-split procedure to improve the connectivity of prediction sets. The method retains the formal coverage guarantees of split conformal prediction while aiming to produce more interpretable, connected intervals. The key idea is to apply smoothing directly to the fitted conditional density function $\hat{f}(y \mid x)$ before constructing prediction sets. Smoothing scheme such as Fourier smoothing is explored. The approach introduces a validation-based tuning procedure to select the smoothing parameter $\sigma$, targeting a user-specified number of connected components ($K_{\text{target}}$) in the prediction set. 

The method is evaluated on synthetic data and two real-world datasets (bio and bike).

### Strengths
1.  The paper is well-organized, easy to follow, and maintains a good balance between theoretical exposition and algorithmic intuition. Figures effectively convey the motivation and qualitative effect of smoothing on prediction set connectivity. These visual cues were helpful to grasp the ideas. 

2. The paper deals with an important problem setup at the intersection of CP and interpretability issues. 

3. The work connects classical results from functional analysis (smoothing by convolution) to show that smoothing can reduce the number of disconnected components in the level sets of the predictive density. This linkage is conceptually appealing and mathematically elegant.

4. The proposed SCD-split method introduces a simple smoothing step that can be applied to any conditional-density-based conformal prediction procedure. The design is straightforward to implement.

5.  Finite-sample marginal coverage guarantee of split conformal prediction is ensured.

6. The method allows users to choose $K_{target}$.

### Weaknesses
1. The paper equates interpretability solely with the number of disconnected components in prediction sets. This is motivated through healthcare and finance examples. However, in the real data experiments, there is no discussion on how the constructed prediction sets, with reduced number of disconnected intervals, help towards 'interpretability' as motivated earlier in the paper. What interpretable insights does SCD-split provide at the end is not discussed at all.



2. The paper repeatedly states that SCD-split achieves a trade-off between length of prediction sets and number of connected components. However, the algorithm explicitly targets only one quantity, connectivity, via the smoothing parameter $\sigma$. Empirically, Table 1 shows that SCD-split often improves both metrics simultaneously (shorter lengths and fewer disconnected intervals), so, what is the claimed trade-off? What trade-off is actually being optimized or observed? If smoothing consistently reduces both fragmentation and length, this requires a theoretical explanation.

3. Smoothing the estimated conditional density $\hat{f}(y|x)$ might introduces bias (does it in your case?), which can affect the efficiency (expected length) of conformal prediction sets. The paper claims that efficiency is ``preserved'' but provides no theoretical quantification of the bias-efficiency trade-off. Can the authors quantify how smoothing impacts the expected length of the sets, either asymptotically or in finite samples?

4. Many existing conditional density estimators (e.g., kernel-based, mixture, or flow-based models) already include built-in smoothing or regularization mechanisms. SCD-split may thus resemble a CD-split procedure with a smoother $\hat{f}(y|x)$. The paper does not  differentiate its smoothing step from simply using a more regularized or smoother model. In what sense is SCD-split fundamentally different from CD-split with a smoother conditional density estimator? Is the key contribution only the explicit control of $K_{\text{target}}$?

5. The core methodological idea, applying a smoothing operator to the conditional density estimate before constructing conformal prediction sets, is conceptually simple and only a small modification of existing frameworks. Since, smoothed version of  conditional density estimate is itself a conditional density estimate. The novelty of SCD-split lies mainly in re-purposing smoothing to target a user-specified number of connected components ($K_{target}$) which is attained through validation set, not in developing a new conformal framework or theoretical principle. 

6. In Theorem 4.4 (interval length bound), in my opinion, the result should be stated as a high-probability bound rather than a deterministic one. The proof relies on concentration/deviation inequalities that hold with high probability only. 

7. Theorem 4.4 relies on conditions such as Lipschitz and lower bounded conditions on densities, which usually vanishes at the tails. Can you provide some examples of densities which are multimodal, satisfy the bounds in the Theorem 4.4?


Overall, the idea of smoothing-based conformal prediction is intuitively appealing, all of the above concerns regarding the conceptual and methodological contributions of the paper, the clarity and limited novelty of the theoretical results, and the experimental validation without meaningful interpretability analysis collectively lead to a low score.

### Questions
1. The paper does not discuss how this notion of interpretability relates to other notions of interpretability in uncertainty quantification (e.g., feature attribution, model transparency, or local explanation of uncertainty).  Why should connectivity of prediction intervals be considered a superior or more relevant notion of interpretability compared to other notions? 

2. What is the target number of connected components $K_{target}$ chosen for real data experiments? I couldn't find these details in the paper.

3. Throughout the paper, authors use `interval length' for the length of prediction set with disconnected intervals. It should be replaced with 'prediction set size/volume'. 

4. The bound in Theorem 4.4 can be very large in practice. Can you comment on why is it relevant?

5. The same reference `Yaniv Romano, Evan Patterson, and Emmanuel Candes. Conformalized quantile regression.' is cited in three different versions.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the interpretability challenge of conformal prediction (CP) methods based on conditional density estimation, which often produce prediction sets with multiple disconnected intervals that are difficult to interpret. The authors propose SCD-split, a novel framework that integrates Fourier-based smoothing operations into the CP procedure to control prediction set connectivity while maintaining statistical validity. SCD-split allows users to specify a desired number of intervals based on domain knowledge, with validation-based tuning to achieve this target. Theoretical guarantees demonstrating that smoothing preserves marginal coverage, does not increase interval count under general conditions, and strictly reduces intervals under specific structural assumptions while maintaining bounded length increases. Empirical validation on synthetic and real-world datasets showing SCD-split achieves favorable trade-offs between validity, efficiency, and interpretability compared to existing methods, particularly for complex multimodal distributions.

### Strengths
1. The paper identifies that existing CP literature has predominantly focused on coverage and efficiency (interval length) while largely neglecting connectivity/interpretability.

2. The paper establishes that interpretability in regression CP should be multi-dimensional, measured jointly by interval length and connectivity. This concept reframes how the community should evaluate prediction set quality, moving beyond single-metric optimization.

3. While smoothing is well-established in signal processing, this is the first work to systematically incorporate it into conformal prediction frameworks to directly control prediction set structure.

### Weaknesses
1. The paper only guarantees marginal coverage not conditional coverage. While this is a known limitation of split CP, the paper doesn't adequately discuss how smoothing might differentially impact conditional coverage across the covariate space.

2. The example illustration in Figure 1 is too conceptual and not convincing enough, would it possible to provide some concrete real-life examples (such as in lines 94 - 102 but with visual explanations) on multimodal distributions and explain why it is important to enhance the connectivity? 

3. The paper briefly mentions that if $K_{target}$ exceeds the true number of modes, the method provides diagnostic feedback. It'd be better to offer more detailed guidance on how users should revise $K_{target}$ and analyze what happens when it is misspecified.

4. The paper equates interpretability with "small number of intervals", but in reality, users may care about which values are included, not just how many intervals exist.

5. The proposed method is only tested on two real-world datasets: Bio and Bike, which is insufficient to demonstrate broad applicability.

### Questions
1. In regions where the original density estimate is already smooth, would additional smoothing be unnecessary and could degrade local coverage?

2. Would aggressive smoothing merge truly distinct modes, and therefore hide important uncertainty information?

### Soundness
3

### Presentation
3

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
The paper proposes SCD-split, a method to control the fragmentation of conformal prediction sets by applying Fourier-based smoothing to the estimated conditional density before constructing prediction intervals. The user specifies a desired number of disjoint intervals (K_target), and the method adaptively tunes the smoothing parameter (σ) via validation to approximate this target while maintaining valid coverage. 

The authors provide theoretical guarantees on coverage, interval connectivity, and length, and empirical results on synthetic and two real-world datasets demonstrate that SCD-split produces more interpretable prediction sets with coverage comparable to CD-split and shorter average lengths than other baselines. Moderate smoothing effectively merges spurious modes and enhances interpretability, whereas excessive smoothing can lead to overly broad, single-interval sets.

### Strengths
The proposed SCD-split approach is well motivated and integrates Fourier-based smoothing into conformal prediction through a clear and principled formulation. 

The theoretical analysis, including proofs for coverage preservation, bounded length increase, and non-increasing interval count, is well structured and contributes significantly to the paper’s technical strength.

Results on synthetic data (e.g., Table 1) convincingly demonstrate that smoothing reduces the number of intervals while maintaining valid coverage, validating the theoretical claims in practice.

### Weaknesses
Most weaknesses relate to the real-world experimental analysis, which appears limited and somewhat unclear. The evaluation is limited to only two real datasets, constraining the assessment in terms of robustness and generality. Some aspects of the experimental design aren’t clearly articulated and there are some results that seem inconsistent, as stated next:

1. The method section (L247) mentions that the base estimator can be a random forest or neural network, but it is not specified which was actually used, nor are implementation details or hyperparameters provided.

2. The choice of the target number of intervals (K_target) is not stated, and it is unclear what value was used for Table 3 or how it affected results.

3. The useful ablation on the smoothing parameter σ (Table 1) is not repeated for real-world datasets.

4. In Table 3 (Bio dataset), SCD-split produces on average more intervals than CD-split (σ=0), contradicting the theoretical claim that larger σ should not increase interval count, this doesn't seem to be discussed.

5. In Table 3, the presentation is inconsistent, why is the largest num being highlighted for bio but not for bike?. Overall, the gains in length and number of intervals appear modest. The targeted number of intervals (K_target) isn’t highlighted  as mentioned in point 2.

6. It is not evident whether the chosen real-world datasets are well suited to demonstrate the advantages of the proposed method. A discussion of which data characteristics make the approach most beneficial would strengthen the empirical narrative.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces SCD-split, a smoothing-based extension of density-based split conformal prediction (CD-split) designed to produce more interpretable prediction sets—specifically, sets with fewer disconnected intervals—while retaining valid coverage and competitive interval length. The motivation stems from a limitation of the conditional-density conformity score, which often yields highly fragmented prediction sets when densities are multimodal. The method applies a smoothing operation (e.g., Fourier low-pass filtering) to the estimated conditional density before running the split conformal procedure, with a user-specified target for the desired number of intervals. The authors establish several theoretical guarantees: finite-sample marginal coverage, a result showing that smoothing cannot increase the number of intervals, a structural condition under which it strictly decreases, and a bound on the potential increase in interval length induced by smoothing. Experiments on both synthetic and real datasets demonstrate that SCD-split successfully matches the target number of intervals while preserving coverage and maintaining competitive length, yielding more interpretable prediction sets than CD-split and other baselines.

### Strengths
- **Novel interpretability objective**: Controls the number of disconnected intervals in conformal prediction sets, addressing a dimension of interpretability that is rarely discussed in prior work.
- **Clear methodological contribution**: Introduces smoothing (e.g., Fourier low-pass filtering) of conditional density estimates within a split-conformal framework, implemented in a concrete algorithm.
- **Strong theoretical support**:  
  - Marginal coverage is preserved.  
  - Number of intervals is guaranteed not to increase, and may strictly decrease under structural conditions.  
  - Provides a bound on how much smoothing can increase interval length.
- **Solid empirical results**: Demonstrates improved interpretability (fewer intervals) while maintaining competitive coverage and interval length across synthetic and real datasets.
- **Practically motivated**: Highlights application scenarios (e.g., healthcare, finance) where having a small number of intervals improves decision-making.

### Weaknesses
- **Choice of conformity score and practical relevance**  
  The method is specialized to the conditional-density conformity score, which is less commonly used in practice than alternatives such as conformal quantile regression (CQR) or residual-based scores. Since conditional density estimation is typically more difficult than regression estimation, the added computational and statistical burden warrants justification. Although the conditional-density score is known to be optimal in a certain sense [1] and this is briefly mentioned in Section 3.1, the paper would benefit from a fuller discussion of when this score is practically advantageous and why its theoretical optimality may outweigh its higher estimation cost.

- **Scope of coverage guarantees**  
  The theoretical guarantees focus on marginal coverage. Though this is fairly standard in conformal prediction.

- **Tuning and practical usability**  
The procedure requires user-specified tuning choices, including the target number of disjoint intervals and a smoothing parameter grid, but the paper offers limited guidance on how these should be chosen in practice. Although the authors note that the number of intervals can be selected based on interpretability considerations, there is an inherent trade-off: increasing the amount of smoothing improves interpretability but can degrade the quality of the conditional density estimate, potentially leading to less efficient prediction sets. A more detailed discussion of this trade-off—and practical strategies for navigating it—would help clarify how the method should be used in real applications (maybe after Theorem 4.4 on the interval length bound).

### Questions
1. **Tuning the number of intervals:**  
   Would it be reasonable in practice to select the target number of intervals \(K_{\text{target}}\) (or the smoothing level) via cross-validation? Although this would technically violate the finite-sample coverage guarantee, similar tuning strategies have been used in conformal prediction (e.g., Gibbs, Cherian, and Candès, 2023, who cross-validate a ridge penalty). Do the authors view this as a viable practical alternative, and is there any empirical evidence on how much coverage might be affected? 

2. **Applicability beyond continuous outcomes:**  
   Is the method intended only for continuous response variables, or can it be adapted to discrete or categorical outcomes? Since the approach relies on estimating and smoothing a conditional density, clarification on its scope—and on what modifications would be required for non-continuous \(Y\)—would be helpful.

### Soundness
3

### Presentation
3

### Contribution
3
