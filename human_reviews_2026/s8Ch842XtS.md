# Novelty Detection with Augmented Localized Conformal $p$-values

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Novelty detection is an important area of research in both statistics and machine learning.
In this paper, we focus on the conditional novelty detection problem, where novelties arise from the relationship between different variables. We first adopt the conformal inference framework and propose the Augmented Localized Conformal $p$-values (ALCP), constructed by recalibrating an augmented conditional distribution estimator. This estimator efficiently captures conditional information by incorporating both calibration and test data into its kernel estimation. We show that the resulting $p$-values are valid in finite samples and can improve detection efficiency. Based on ALCP, we then develop a novel conditional novelty detection algorithm, along with a data-driven bandwidth selection method that ensures finite-sample false discovery rate (FDR) control while enhancing detection power. Both simulated and real data experiments demonstrate the advantages of the proposed ALCP approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes ALCP for conditional novelty detection. It performs localized conformal calibration with kernel weights and augments the calibration using test data through density-ratio reweighting to handle shift from novel points, then applies a recalibration step that yields finite sample valid p-values and pairs with Benjamini–Hochberg to control the FDR. It also introduces a data driven bandwidth rule with a small correction to keep validity. On synthetic data and a real housing dataset, ALCP shows higher detection power at controlled FDR than localized and marginal conformal methods.

### Strengths
The paper introduces ALCP for conditional novelty detection, combining localized conformal calibration with test-set augmentation via density-ratio reweighting to handle shift from novel points; it provides finite-sample valid p-values and supports FDR control with Benjamini–Hochberg, includes a data-driven bandwidth rule with a correction that preserves validity, and shows higher detection power at controlled FDR on synthetic data and a real housing dataset.

### Weaknesses
- The contribution feels incremental. It extends localized conformal prediction with test-set augmentation and a data-driven bandwidth, and the route to FDR control seems to rely on existing results.
- The paper does not clearly explain how it differs from prior conformal methods or which guarantees are new, nor does it clearly lay out the challenges it aims to solve or rigorously reason through their theoretical and practical implications, so the proposed fixes to LCP read as a modest mix of existing ideas without a convincing impact.
- Bandwidth selection requires recomputing auxiliary p-values and running BH across many bandwidths, which may be impractical at scale or with many hyperparameters.
- There is little analysis of how kernel choice, density-ratio estimation, and randomization affect power; the bandwidth study does not report selected values or their variability. key baselines (standard anomaly detectors and closely related conformal methods) are missing, making the incremental benefit of the proposed score hard to judge.

### Questions
See Weaknesses.

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
4

### Summary
This paper studies the conformal outlier detection for the conditional distribution shift. The authors design a framework to construct p-values based on kernel estimation that are sensitive to conditional shift, and leverage a randomization technique to preserve finite-sample validity of the p-values. The p-values are then coupled with a conditional calibration strategy to ensure valid FDR control. They also present a strategy to adaptively choose the bandwidth in kernel estimation while preserving the validity of p-values and FDR control. The power improvement of the proposed method is demonstrated in simulations and a real-world case study of housing price.

### Strengths
1. The paper is clearly written and easy to follow in general. 
2. The use of various techniques are valid and sound. Although none of them seems completely new, a combination of them leads to a solution to a somewhat useful problem.

### Weaknesses
1. The conditional testing problem seems to be at odds with the FDR control, and the statement of conditions is a bit unclear (see my questions). 
2. The scale of numerical evaluations is a bit limited.
3. The guarantees are still marginal FDR, which is not very new.
4. The theoretical understanding of power boost can be deepened.

### Questions
1. While the authors propose to test the null hypothesis $P_{Y|X}=Q_{Y|X}$, the FDR control is still with regard to a marginal exchangeability condition $P_{X,Y}=Q_{X,Y}$. Is my understanding correct? If yes, then the motivation and claim of addressing conditional shift may be insufficiently supported. In particular, it is possible that the null hypothesis $P_{Y|X}=Q_{Y|X}$ holds, but due to covariate shift the marginal null hypothesis $P_{X,Y}=Q_{X,Y}$ is not true and can be rejected, leading to inflated type-I error. 
2. Following the above question, the simulations should also consider settings with covariate shift to show the robustness of the methods, even if the theory can only be limited to marginal test.
3. Are there theoretical results, such as asymptotic analysis, that can illustrate why your method can improve power?

### Soundness
3

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
This paper addresses the problem of novelty detection, where the goal is to test whether a test point was drawn from an inlier distribution $P$ or from a different distribution $Q \neq P$. Specifically, the authors consider conditional novelty detection, where the inlier distribution takes the form $P=P_{W|X} \cdot P_X$, and the hypothesis of interest is whether a test point is drawn from $P_{W|X}$ or from a distinct conditional distribution $Q_{W|X} \neq P_{W|X}$.

The authors propose a new score function based on an estimate of the conditional CDF, which incorporates information from test points to improve estimation accuracy. They also introduce a data-driven method for selecting the kernel bandwidth for each test point to enhance detection power. The authors prove that the proposed method controls the marginal FDR and provide empirical results demonstrating this control and its enhanced power.

### Strengths
*   **S1:** The paper introduces a new score function, an estimate for the conditional CDF, which offers a novel approach to the localization problem in conditional novelty detection.
*   **S2:** The paper provides a theoretical result on the convergence rate of the proposed estimate for the conditional CDF, formally justifying the benefit of using test data.
*   **S3:** The p-values are proven to be valid, and the overall procedure provides finite-sample FDR control by utilizing the conditional calibration method (Fithian and Lei (2022), "Conditional calibration for false discovery rate control under dependence").
*   **S4:** The idea of using the test data to improve the estimation of the conditional CDF can be potentially useful beyond the specific application mentioned in this paper.
*   **S5:** The experiments demonstrate FDR control and show improved power compared to baseline methods.

### Weaknesses
* **W1:**  The experiments are too simplistic. The paper is framed as a novelty detection work, but the experiments have the flavor of regression. The authors use classic regression datasets and synthetically generated outliers rather than established novelty detection benchmarks. To convince readers of the method's practical utility, a real-world use case is needed. See related question Q5.

* **W2:**  The experimental discussions are limited and lack depth:
     - The evaluation of the data-driven bandwidth selection is limited. It would be very helpful to report the actual bandwidth values selected and illustrate how they vary across different settings. This would emphasize that the choice of bandwidth is non-trivial and highlight cases where it changes in ways that are difficult to predict or set a priori.
     - The experimental settings for the appendix studies (E.1, E.2, E.3) are not specified, it is unclear which data were used.
     - The experiments in Appendices E.1 and E.2 lack sufficient explanation for the observed trends. The authors propose selecting the parameter $w$ based on the accuracy of $\hat{r}$, yet Appendix E.2 neither examines this relationship nor clarifies the results, providing limited practical insight. This leaves no practical guidance for practitioners on how to set $w$ when applying the proposed method. Moreover, the description also does not specify which data were used, leaving it unclear whether $r$ can be computed. I would expect to see an experiment with simulated data (in addition to real-data exp.), where $r$ is known, demonstrating the performance as a function of the accuracy of $\hat{r}$ and different values of $w$.

* **W3:**  The novelty is somewhat limited when viewed in the context of prior work.
     - The main innovation is incorporating test data. The correction using $\hat{r}$ is related to weighted conformal prediction, and the data-driven selection based on the rejection set size is similar in spirit to the method in Marandon et al. (2024), "Adaptive Novelty Detection with False Discovery Rate Guarantee," to whom clearer credit should be given in the main text.
     - The FDR control result is an instantiation of the conditional calibration framework from Fithian and Lei (2022), "Conditional calibration for false discovery rate control under dependence". While the full proof is valuable, this connection should be emphasized before the theorem to properly credit the proof technique.

To conclude, the core idea of using test data to improve the score function is valuable. However, the main reason for my relatively low score (but still positive) is that the authors do not fully convince the reader that this is a practically worthwhile approach. Since the paper's goal is to improve power, this must be demonstrated through more realistic and comprehensive experiments.

### Questions
1.  In the simulation details for the real data experiment, the authors mention in lines 661-662 "the null test set is combined with the excluded outliers to construct the complete test set." Does this mean the entire set of outliers was used for each of the 200 repetitions? If so, this choice needs to be justified.
2.  In the novelty detection section (around line 277), $\hat{r}$ is referred to as fixed, but it necessarily depends on the data $\mathcal{D}_c \cup \mathcal{D}_u$. Could you clarify what it is "fixed" with respect to, and whether this affects the validity of Theorem 3?
3.  The computational complexity of the method appears to scale quadratically with the number of test points, $m$. The paper notes a runtime of 30 seconds for sample sizes in the hundreds. What are the practical limitations of this method when $m$ is in the thousands? A brief discussion on scalability should be added.
4. The authors mention that one should choose $w$ based on the accuracy of $\hat{r}$. However, if we have a relatively accurate $\hat{r}$, shouldn't it be reasonable to use it as a score function, as it quantifies the likelihood ratio? The proposed approach is probably more robust, but a short discussion about this would be helpful.
5. In lines 450-451, the authors mentioned "by training a prediction model on the full dataset and take outliers as the 10% of samples with the largest nonconformity scores from both the high-price and low-price groups." Which predictive model is used here? Is it the same one used in the experiments? This should be clarified.
6. Minor comment: There are no details on how the authors estimate $\hat{r}$ (neither in section 4 nor in appendix C). It only appears in Appendix E.1, where it is mentioned that a GLM algorithm was used for estimating r in the main manuscript. This information should be included in section 4 or appendix C.
7.  Minor comment: The sentence in line 916, "both have power reduced with this algorithm," appears to contradict the results in Table 2 for the ALCP method with the quantile score, where power actually increases. Please check and revise this.

### Soundness
2

### Presentation
3

### Contribution
2
