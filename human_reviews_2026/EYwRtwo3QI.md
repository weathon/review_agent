# QUTCC: Quantile Uncertainty Training and Conformal Calibration for Imaging Inverse Problems

- Decision: Reject
- Scores: 2, 2, 2, 8

## Abstract
Deep learning models often hallucinate, producing realistic artifacts that are not truly present in the sample. This can have dire consequences for scientific and medical inverse problems, where accuracy is more important than perceptual quality. Uncertainty quantification techniques, such as conformal prediction, can pinpoint outliers and provide guarantees for image regression tasks, improving reliability. However, existing methods predict fixed quantiles and utilize a linear constant scaling factor to calibrate uncertainty bounds, resulting in larger, less informative bounds. We propose QUTCC, a quantile uncertainty training and calibration technique that enables nonlinear, non-uniform scaling of quantile predictions to enable tighter uncertainty estimates. Using a U-Net architecture with a quantile embedding, QUTCC can predict the full conditional distribution of quantiles for each image. After conformal calibration, QUTCC can predict pixel-wise uncertainty intervals that satisfy coverage guarantees and also estimate a pixel-wise conditional probability density function. We evaluate our method on image denoising, quantitative phase imaging, and compressive MRI reconstruction. Our method successfully pinpoints hallucinations in image estimates and consistently achieves tighter uncertainty intervals than prior methods while maintaining the same statistical coverage.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The submission focuses on conformal risk control for imaging inverse problems. The submission proposes to train a quantile-conditional neural network to jointly estimate all quantiles $q \in [0,1]$ of the pixel-wise conditional distribution of the ground-truth image given a noisy observation. With this network, the submission introduces an algorithm to estimate the conformalized quantile $\hat{q}$ that guarantees coverage (in the conformal risk control sense) of the ground-truth image.

Experiments compare the proposed method with one existing baseline method.

### Strengths
* Uncertainty volume minimization in high-dimensional inverse problems is an important problem.
* Training a quantile-conditional network is an interesting idea.
* Estimating the full posterior density using quantile regression is novel in the context of imaging inverse problems.

### Weaknesses
* Presentation of theory is hand-wavy.
* Experimental results provide limited evidence in support of the proposed method.

I will expand on a few questions below and I am looking forward to discussing with the authors!

### Questions
**Conformal Risk Control vs. Risk-Controlling Prediction Sets**

In its current formulation, there appears to be a disconnect between the guarantee mentioned in the text, the one written in Eq. (2), and the one provided by the algorithm. If I understand correctly:

* $x,y$ are high-dimensional objects, let us say $\mathbb{R}^d$ (this should be specified clearly in the text)
* Lines 183-185 state that the prediction set should "contain at least $1 - \alpha$ of the ground-truth pixels with probability $1- \delta$". This is the notion of risk-controlling prediction set of [1] (Definition 1)

$$\mathbb{P}[ \mathbb{E}[\frac{1}{d} \sum_{j \in [d]} \mathbb{I}[x_j \notin C_{\lambda}(y)_j]] \leq \alpha]\geq 1 - \delta.$$

* However, Eq. (2) requires the **entire** input to be contained at once in the prediction set, which is a stronger guarantee. In practice, Eqs. (5) and (6) refer to the risk-controlling notion, not Eq. (2).

* Finally, the algorithm implements a conformal risk control procedure that does not provide a high-probability claim, but in expectation only (Algorithm 1 does not depend on $\delta$). That is, the guarantee provided by conformal risk control is: 

$$\mathbb{E}[\frac{1}{d} \sum_{j \in [d]} \mathbb{I}[x_j \notin C_{\lambda}(y)_j]] \leq \alpha.$$

In fact, the submission uses the same procedure described in [2], Eq. (4). Note that to achieve the stronger notion of high-probability risk control, binary search would not be a valid algorithm, but other strategies, such as backtracking or fixed hypothesis testing might be used [3].

Could the authors clarify this apparent inconsistency?

**Multiple quantile estimation and quantile crossing**

The submission acknowledges that for the proposed method to be statistically valid, the risk must be monotonically non-increasing with the quantile level. This implies that no quantile crossing can happen, but the training procedure proposed in the submission does not prevent quantile crossing by design. In fact, Table 3 in the appendix shows that quantile crossing does happen (even if minimally). 

Experimentally, this does not seem to affect the validity of the calibration procedure, but it renders the guarantees of the proposed method vacuous in the general sense.

Could the authors propose technical ways to address this robustly? For example, see [4], who also propose to learn multiple quantiles simultaneously in order to estimate conformalized histograms of the posterior distribution (related to the goal of Sec. 3.4.1, although different in methodology and application).

Could the authors include a formal statement of the guarantees provided by the proposed method? This will improve clarity of presentation, even if the result might follow from standard properties of conformal prediction.

**Definition of the loss function**

In its current formulation, the loss in Eq. (4) does not faithfully represent the sampling of the quantile levels from a uniform distribution.

Could the authors state the loss function as an expectation over a population, clearly stating the distribution quantile levels are sampled from at training time?

**Estimating the conditional distribution**

It might be helpful to distinguish between the true quantile function, the empirical quantile function, and the estimated quantile from the trained network. Note that in the general sense, it is not true that the quantile function is the inverse of the CDF, as the latter might not be strictly increasing. 

I am not sure I understand what coverage guarantee the estimated PDF provides. Could the authors expand on this point and clearly state the guarantee(s)? Is this similar to the concept of conformal predictive distributions [5]?

As an aside, it might be worth mentioning existing Bayesian approaches that provide estimates of these distributions by design, with, for example, Gaussian processes.

**Experimental results**

Table 1 and Figure 2 provide weak evidence that the proposed method significantly provides shorter intervals compared to im2im, as all differences are well within one standard deviation of the results. The result in Figure 3 is more convincing. Could the authors provide the equivalent of Figure 2 stratified by pixel intensity? That might be a more compelling example of the advantages provided by the proposed method.

If the main advantage of the proposed method is to provide shorter interval lengths, it is important to compare with existing works that address this in imaging inverse problems [6,7].

I am not sure I follow the choice of training two different models, one for im2im and one for QUTCC. If I understand correctly, one could use im2im to calibrate the predictions of a network trained with multiple quantile regression as well. Could the authors expand on the choice of using different networks instead of keeping the network fixed and changing the calibration procedure only? That might provide a more straightforward comparison between the two methods.

---

**Minor comments**

* "Where accuracy is more important than perceptual quality" may be too broad in its current formulation. Image quality evaluation is a nuanced and complementary field of research, especially in medical applications. Uncertainty quantification is a well-motivated endeavor even without this comment.
* The emoji in the title seems unrelated to the contribution of the paper? I would consider removing unless it contributes to the submission.
* Typo in Eq. (8), $\geq 1-\alpha$?
* In its current formulation, does Algorithm 1 always terminate? Since the step size does not change, isn't there a chance the thresholds will start oscillating?
* The claim that no existing methods have addressed minimal interval length or conditional coverage in imaging inverse problems is too strong as there exist prior works that have proposed ways to tackle parts of these issues [6,8].
* The claim that prior works use linear scaling of the interval bounds might also be too broad, as several works have used additive parametrizations that do not scale intervals bounds multiplicatively, or different strategies such as masking.

---

**References**

[1] Angelopoulos et al. "Image-to-Image Regression with Distribution-Free Uncertainty
Quantification and Applications in Imaging" (2022).

[2] Angelopoulos et al. "Conformal Risk Control", (ICLR 2024).

[3] Angelopoulos et al. "Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control", (2022).

[4] Sesia and Romano "Conformal Prediction using Conditional Histograms", (2021).

[5] Vovk et al. "Conformal Predictive Distribution with Kernels" (2018).

[6] Teneggi et al. "How to Trust Your Diffusion Model: A Convex Optimization Approach to Conformal Risk Control" (2023).

[7] Belhasin et al. "Principal Uncertainty Quantification with Spatial Correlation for Image Restoration Problems" (2023).

[8] Fischer et al. "Subgroup-Specific Risk-Controlled Dose Estimation in Radiotherapy" (2024).

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose a quantile embedding to enable nonlinear and non-uniform scaling of quantile predictions, thereby allowing for tighter pixel-based conformal interval lengths. The authors demonstrate their method on inverse problems using a U-Net, are able to estimate the full conditional distribution of quantiles for each image, and estimate the conditional CDF.

### Strengths
1. The manuscript is approachable and well written. The figures are clear.
2. The quantile embedding for image-to-image regression tasks is novel.
3. Estimating the conditional distribution and being able to mitigate quantile crossing offer practical utility.

### Weaknesses
1. The performance improvements compared to Im2Im-Deep (Table 1 and Table 2) are not entirely convincing and are marginal at best.
2. The crux of the method is the quantile embedding. However, the paper does not compare Im2Im-Deep on an equal playing field due to the different formulations (Im2Im-Deep uses symmetric, while QUTCC uses asymmetric). Specifically, it is unclear whether the performance gains are due to the asymmetric formulation, the quantile embedding, or both. It would make sense that the asymmetric formulation would perform better for higher pixel intensities, because NN models tend to accurately capture the more moderate/prevalent intensities well at the expense of less prevalent intensities. An apples-to-apples comparison between Im2Im-Deep with an asymmetric formulation would be beneficial to clarify this issue. I would be more than happy to increase my score with greater clarity and an additional baseline.
3. The manuscript would benefit from a more rigorous statement of the conformal guarantee.
4. The generation of the conformalized PDF is interesting, but it can be viewed as repeatedly applying the main calibration procedure at different $\alpha$ levels (brute force) rather than a fundamentally new technique.

### Questions
1. Why place a quantile embedding after each layer instead of picking a particular layer? An ablation study would be interesting to see. 
2. In the supplementary, QUTCC requires more training epochs. Does the QUTCC training regime (sampling q randomly) require more careful hyperparameter tuning compared to baselines?
3. The conditional PDF is estimated using finite differences. How sensitive is the shape of the resulting PDF to the number and spacing of the quantile queries used to build the CDF?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose quantile uncertainty training and conformal calibration (QUTCC), a method for training quantile regression neural networks to estimate the full quantile function (as opposed to a fixed set of quantiles, e.g., 0.05 and 0.95), and how to obtain conformalized prediction bounds using this approach. While the proposed method is general, the authors focus on its application for image-to-image regression. Extensive experiments are conducted across five imaging inverse problems and compared against an appropriate baseline (Im2Img-UQ). The authors claim that the proposed method consistently estimates narrower uncertainty intervals compared to the baseline, while achieving the desired level of risk.

### Strengths
Novelty and relevance: While the idea of adding conditioning mechanisms to enable a quantile regression network to learn the full quantile function is not novel (see [1, 2]), its application for uncertainty prediction in inverse imaging problems is novel, relevant, and timely. 

Extensive experiments: The authors conduct a thorough evaluation of their proposed method on five different datasets, selecting an appropriate baseline for comparison.  

References: 
[1] Ostrovski, Georg, Will Dabney, and Rémi Munos. "Autoregressive quantile networks for generative modeling." International Conference on Machine Learning. PMLR, 2018.
[2] Dabney, Will, et al. "Implicit quantile networks for distributional reinforcement learning." International conference on machine learning. PMLR, 2018.

### Weaknesses
Major weaknesses: 
Insufficient improvements over baseline: My main criticism of this work is its minor (possibly zero?) improvements over the baseline Im2Im-UQ. The results shown in Table 1 and Figure 2 suggest nearly identical performance. Even in Figure 4, the selected outputs of the baseline and proposed methods are extremely similar. Consequently, it is difficult to justify that predicting the full quantile function versus 2 fixed quantile values (0.05 and 0.95) offers any real benefit. 

Minor weaknesses: 
Alternate architectures: The authors chose to fix the UNet architecture (ResNet-18 backbone) used for the experiments to ensure a fair comparison between the proposed method (QUTCC) and the baseline method (Im2Im-UQ). While I do think this is completely appropriate, it would be nice to see this comparison for a variety of architectures. Nevertheless, this is a very minor point, and I don’t believe it would substantially alter the main results of this work.

### Questions
Are there any other benefits of estimating the full quantile function over a few fixed quantile values apart from obtaining tighter prediction intervals? 

Are there any differences in the model’s internal representations between using QUTCC vs. fixed quantiles?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces QUTCC, a framework for uncertainty quantification in imaging inverse problems. It proposes a quantile-conditioned U-Net that predicts conditional quantiles as a function of a sampled quantile level, trained using a pinball loss akin to quantile regression. Similar to how diffusion models condition on timesteps, QUTCC conditions on quantiles, enabling the network to predict the full conditional distribution of image intensities. A nonlinear, non-uniform conformal calibration step then adjusts quantile bounds to guarantee statistical coverage while producing tighter uncertainty intervals. The model can be queried at test time to yield median predictions, pixel-wise confidence intervals, and even per-pixel probability density functions (PDFs). Across multiple imaging tasks—including denoising, MRI, and quantitative phase imaging—QUTCC achieves smaller, sharper uncertainty bounds and better hallucination detection compared to prior methods like Im2Im-UQ.

### Strengths
The paper presents a principled and rigorous approach to uncertainty estimation in imaging inverse problems. Its key contribution—the quantile-conditioned U-Net trained with a pinball loss and paired with nonlinear conformal calibration—represents a natural yet nontrivial extension of prior conformalized quantile regression methods such as Im2Im. The methodology is conceptually elegant and statistically grounded, providing a unified framework for pixel-wise uncertainty quantification, coverage guarantees, and conditional density estimation. The paper is clearly written, with informative figures that effectively illustrate both the conceptual flow and empirical results. Experimental validation across multiple imaging modalities and noise models (MRI, QPI, Gaussian, Poisson, and real noise) convincingly demonstrates robustness and generality, highlighting the method’s potential significance for reliable deep imaging.

### Weaknesses
While the paper is technically sound and well presented, several aspects could be improved to strengthen its contribution. First, the related work section omits an important body of literature on multi-modal prediction and multi-hypothesis uncertainty estimation using multi-head or mixture-based networks (e.g., Learning in an Uncertain World: Representing Ambiguity Through Multiple Hypotheses by Rupprecht et al., Hierarchical Uncertainty Exploration via Feedforward Posterior Trees by Nehme et al.). These works share conceptual overlap with simultaneous quantile prediction and discussing them in the related work section can better contextualize the paper’s novelty. Second, the interpretation of pixel-wise marginal distributions raises conceptual concerns: modeling uncertainty independently for each pixel neglects spatial correlations that are fundamental to imaging tasks. It remains unclear how these marginal PDFs can be practically leveraged beyond visualization, and Figure 5 does not convincingly demonstrate their utility. Finally, the quantitative results in Table 1, while consistent, show only marginal improvements over the baseline, and stronger ablations or uncertainty-coverage tradeoff analyses would help substantiate the claimed performance gains.

### Questions
Questions and suggestions for the authors:
1. Could the authors comment on the potential complementarity between multi-modal prediction frameworks (e.g., multi-head or mixture density networks) and quantile-based uncertainty estimation? It would be interesting to understand whether quantile conditioning could be integrated into architectures designed for multi-hypothesis prediction.
2. In Equation (8), shouldn’t the inequality be $\mathbb{P}(\\hat{\\mathbf{x}}\_{lower}(k)\\le\\mathbf{x}\\le\\hat{\\mathbf{x}}\_{upper}(k)) \\ge 1 - \\alpha$? Can you clarify this?
3. The calibration procedure in Algorithm 1 appears similar in spirit to the approach used in Im2Im. Could the proposed conformal calibration, in principle, have been implemented within that framework as well?
4. What practical insight or downstream utility is gained from the pixel-wise marginal distributions shown in Figure 5? As presented, these marginals appear difficult to interpret or use for actionable uncertainty quantification.
5. In Table 1, is the comparison between QUTCC around the same predictor? QUTCC reports median-based estimates, is Im2Im also doing that? or is it using mean prediction? Aligning the central statistic across methods is crucial and could affect the reported results.
6. How sensitive is the proposed model to the quantile sampling strategy used during training? For example, do uniform versus importance-weighted quantile samples impact calibration quality, coverage, or stability during optimization?

### Soundness
4

### Presentation
3

### Contribution
3
