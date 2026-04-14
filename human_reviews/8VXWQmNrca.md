# Conformal Bounds on Full-Reference Image Quality for Imaging Inverse Problems

- Decision: Reject
- Scores: 3, 6, 6, 5

## Abstract
In imaging inverse problems, we would like to know how close the recovered image is to the true image in terms of full-reference image quality (FRIQ) metrics like PSNR, SSIM, LPIPS, etc. This is especially important in safety-critical applications like medical imaging, where knowing that, say, the SSIM was poor could potentially avoid a costly misdiagnosis. But since we don't know the true image, computing FRIQ is non-trivial. In this work, we combine conformal prediction with approximate posterior sampling to construct bounds on FRIQ that are guaranteed to hold up to a user-specified error probability. We demonstrate our approach on image denoising and accelerated magnetic resonance imaging (MRI) problems.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents a conformal prediction framework for constructing bounds on the recovery error with respect to a full-reference quality metric (FRIQ) in imaging inverse problems. The paper leverages split conformal prediction to build bounds that are guaranteed to hold marginally over exchangeable samples of the joint distribution of (measurement, reconstruction). 
A posterior sampler is used to generate adaptive bounds (ie which change across measurements in the dataset),  
The paper proposes to use a predictor (based on simple regression models, such as splines) to improve the quantile estimates, in order to reduce the number of posterior samples needed to obtain an accurate quantile.

### Strengths
- applies split conformal prediction to the linear inverse problems, with a generalization to FRIQ metrics.

### Weaknesses
Main weaknesses:
- the contribution of the paper is not novel enough in my opinion: the theory of split conformal prediction is defined with respect to any confirmity score, so I believe the use of general FRIQS as confirmity scores is not a very novel extension. 
- the proposed quantile regression method seems to be the main methodological contribution of the paper, which is based on a relatively classical 1-dimensional interpolation method, and doesn't offer much improvement over a standard empirical quantile estimate. 
- the image denoising results are not convincing: using a posterior sampler increases the computational fold 32 x diffusion_steps -fold with respect to an end-to-end reconstruction method, only to obtain a bound on the PSNR which is less than 1dB better than the non-adaptive bound (see figure 4).

A smaller weakness:
- the mathematical notation is often too heavy, rendering simple concepts hard to understand. Equation 13 is unclear: the dependence of $\bar{z}_i$ on $\theta$ is missing, $\lambda$ is the same symbol as used in equations 9 and 10 for calibrating the intervals, and here has another meaning.

### Questions
- why is the posterior mean in equation 17 computed using different samples than those used for approximating quantiles?
- why does the paper restrict the method to approximate posterior samplers? one could use any uncertainty quantification method to compute adaptive quantiles.

### Soundness
3

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
This work considers the problem of estimating an $\alpha$-quantile interval on the full reference image quality (FRIQ) metrics in the context of imaging inverse problem. Here FRIQ metrics refers to metrics like PSNR, SSIM, and LIPIPS that compares the reconstructed image with the ground-truth image. The main novelty of the work is the proposal of a method that can estimate FRIQ's interval **without** accessing the ground-truth image, by combining posterior sampling and conformal prediction. The procedure of the proposed method can be roughly schemed as follows:

1. Use an off-the-shelf (approximate) posterior sampling algorithm to reconstruct an single estimate and set of posterior samples.
2. Compute the estimated FRIQ metric between the reconstruction and posterior samples (not using the testing groundtruth image).
2. Given a set of calibrate pairs of the groundtruths and measurements other than the testing groundtruth, perform conformal quantile regression to estimate the bound.

Experiments on image denoising (dataset=FFHQ) and accelerated multicoil MRI (dataset=fastMRI) tasks were conducted to validate the proposed method.

### Strengths
1. The considered problem is novel and of sufficient interest to the computational imaging community.
2. The application of conformal prediction to imaging is also relatively new.
3. The current manuscript properly discusses the limitations of the proposed method.

### Weaknesses
1. The clarity of the method section needs to be improved.
2. The mathematical notations are abused, which hinders first-time readers to quickly understand the method.
3. The above two points jointly affect proper interpretation of the experimental results.

### Questions
1. The definition of adaptiveness seems slightly confusing. From line 216-219, the adaptive methods appear not to depend on the test realization ($\hat{x}_0$) as well. Furthermore, the explanation in line 231-232 is quite vague.
2. The text between line 176-214 appears to focus on the construction of FRIQ samples for the test image, while the reminder of the subsection describes how to apply this construction to all the posterior samples and then conduct CP. The authors are advised to give titles to these texts for clarity.
3. [Section 3.2] It is clear that the non-adaptive method produces just one interval C after calibration. However, the adaptive method seems to produce multiple intervals C's (manifest in $\beta_i$) for each $z_i$. How can eq. 11 be obtained given multiple bounds?
4. [Section 3.3] It appears that the quantile prediction network requires $c$ number of $z_i$ as input. Another limitation of the proposed method is that the network needs to be retrained if $c$ is changed.
5. [Section 3.3] Eq 14 and 15 are not explained. Again, how eq. 16 is obtained remains unclear.
6. [Fig. 1] An addition to figure 1 that illustrates the calibration step is highly recommended.
7. [Fig. 2] Due to the lack of clarity in the method section, it is hard to interpret the left four plots in figure 2, as well as figure 5. Especially, the relationship between $\beta$ and $z$.
8. Can the authors explain what a single Monte-Carlo trial contain? Also, T is not properly defined.
8. Many symbols are used before proper definition or (potentially) doubly defined
    - $U_0$ was used before proper definition.
    - Eq. (2) was not explained.
    - the regularization weight $\lambda$ has been used to denote the empirical miscoverage earlier.

Overall, the current manuscript needs some significant improvement on its clarity. However, I still think the considered problem is interesting, and would look forward to the authors' response.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes conformal bounds on full-reference image quality metrics without access to the true image, which can be utilized in safety-critical applications, such as calculating the extent of possible in accelerated MR imaging given a predefined error tolerance.

### Strengths
**Novelty and potential real-life application**

* The paper utilizes conformal prediction (CP) , which is well-suited for generating uncertainty bounds, to calculate bounds on full-reference image quality metrics. This can be particularly useful in medical imaging, especially in scenarios like the multi-round MRI acquisition described in lines $463-464$. 
* Multiple bounds, including an adaptive bound and its improved version, are provided, and numerical comparisons between each bound are made. Theoretical claims are supported by the detailed experiments provided in appendices.

### Weaknesses
**Most of the weaknesses are identified and discussed in the Limitations paragraph (Line $509$)**

* As stated, the method may not work if the calibration and test data distributions are significantly different. I acknowledge it requires more work to make the method more robust to distribution shift, but a simple experiment demonstrating the sensitivity to the such shifts can be added.
* The performance difference between the adaptive bound and the improved adaptive bound appears incremental in Figure 4 and 8.

### Questions
* How do the authors explain the insensitivity of Mean Conformal Bound to the Number of Image Samples $c$, which may not be intuitive?
* In a resource-constrained setting, could calculating adaptive bounds be challenging, especially for a real-time implementation?

**Suggestions**

* Adding a numerical experiment to demonstrate the effect of a simple distribution shift between the test and calibration data would be interesting and increase the paper's impact. 
* As a theoretical limit, the acceleration factor which can be resolved using parallel imaging (PI) is constrained by the number of receiver coils; thus, it is better to state line $413-414$ as "For $R>1$, the inverse problem _might_ become ill-posed." It is guaranteed to become ill-posed for single coil imaging, but for PI, coil sensitivity maps affect the linear system as well.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents a method for constructing prediction intervals for Full-Reference Image Quality (FRIQ) metrics in imaging inverse problems based on conformal prediction. By leveraging a calibration set and empirical FRIQ samples generated from posterior reconstructions, the approach provides intervals that offer coverage guarantees on the true quality metric for new test images. Specifically, the method selects a parameter $\lambda$ to balance the interval width and the desired error rate, ensuring that the intervals meet a specified confidence level, typically $1−\alpha$. Numerical evaluations on natural image denoising and accelerated MRI reconstruction demonstrate the potential practicality of this approach.

### Strengths
1, The research question is significant from various perspectives, especially for trustworthy machine learning, as it addresses the need for reliable uncertainty quantification in image reconstruction tasks.

2, Another strength of this paper’s novelty is its integration of conformal prediction with approximate posterior sampling to construct statistically rigorous bounds on FRIQ metrics for imaging inverse problems, offering guaranteed coverage with a user-specified error probability. This approach provides robust uncertainty quantification in complex imaging tasks where data distributions are unknown.

### Weaknesses
1, The paper in its current form lacks a more comprehensive review of other uncertainty quantification methods, such as Bayesian approaches (e.g., Monte Carlo dropout), which are widely used in imaging reconstruction. Including these comparisons would better highlight the proposed method’s advantages in terms of coverage guarantees and reliability.

2, Similarly, the paper also lacks numerical comparisons with other uncertainty quantification methods beyond conformal prediction. Including these would provide a clearer assessment of the proposed method’s effectiveness relative to established approaches.

3, The proposed approach seems to rely on an exchangeability assumption between calibration and test data, which may not hold in real-world imaging due to distribution shifts. This could limit the method's robustness, especially with diverse or evolving datasets, like those in medical imaging across different populations or devices.

4, The numerical results in Sections 4.1 and 4.2 show no clear performance gain between adaptive bound estimation and its improved learning variants, which may weaken the methodological contribution of the proposed approach.

### Questions
1, Is the choice of calibration $d_{cal}$ critical to the success of this method? How would the intervals behave if $d_{cal}$ were generated using a different model than the one used for predictions? Given that $d_{cal}$  is assumed to represent the true distribution, how sensitive is the method to shift between the calibration distribution and the test distribution?

2, How does the number of FRIQ samples influence the quality of the prediction interval? Is there a minimum number of samples required for reliable interval construction? Since the empirical miscoverage rate uses an indicator function, does it introduce any quantization errors at the same time?

3, The reconstruction results in Figures 2, 3, and 4 rely on MMSE approximations, which result in poorer perceptual quality. Could the authors instead use a single DDRM sample for reconstruction and test the coverage, potentially improving perceptual quality while evaluating the coverage of this approach?

4, Consistency in notation across sections, particularly between Sec. 2 and Sec. 3, would improve readability and understanding. Making sure that the terms introduced in the CP background (Sec. 2) align directly with their usage in the adaptation in Sec. 3 could bridge any gaps for readers. This way, each section builds on the previous one without introducing confusion.

### Soundness
3

### Presentation
2

### Contribution
3
