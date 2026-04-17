# COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
In clinical applications, the utility of segmentation models is often based on the accuracy of derived downstream metrics such as organ size, rather than by the pixel-level accuracy of the segmentation masks themselves. Thus, uncertainty quantification for such metrics is crucial for decision-making. Conformal prediction (CP) is a popular framework to derive such principled uncertainty guarantees, but applying CP naively to the final scalar metric is inefficient because it treats the complex, non-linear segmentation-to-metric pipeline as a black box. We introduce COMPASS, a practical framework that generates efficient, metric-based CP intervals for image segmentation models by leveraging the inductive biases of their underlying deep neural networks. COMPASS performs calibration directly in the model's representation space by perturbing intermediate features along low-dimensional subspaces maximally sensitive to the target metric. We prove that COMPASS achieves valid marginal coverage under the assumption of exchangeability. Empirically, we demonstrate that COMPASS produces significantly tighter intervals than traditional CP baselines on four medical image segmentation tasks for area estimation of skin lesions and anatomical structures. Furthermore, we show that leveraging learned internal features to estimate importance weights allows COMPASS to also recover target coverage under covariate shifts. COMPASS paves the way for practical, metric-based uncertainty quantification for medical image segmentation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper targets uncertainty intervals on downstream clinical metrics derived from segmentations, rather than pixel-wise errors. It proposes COMPASS: apply linear perturbations along the most metric-sensitive low-dimensional directions in a network’s intermediate representation, and use the perturbation magnitude as the nonconformity score for split CP to obtain tighter intervals.
Under exchangeability and a nestedness assumption, the method provides marginal coverage guarantees; density-ratio–weighted CP is used to recover target coverage under covariate shift. Empirically, across four medical segmentation tasks, COMPASS (especially the Jacobian variant) achieves shorter intervals while maintaining coverage, and is closer to target coverage under distribution shift.

### Strengths
1.	Replaces per-sample adversarial search with a global low-dimensional “sensitivity subspace,” substantially reducing the computational burden of FCP; practically deployable.
2.	Provides standard split-CP coverage under a nestedness condition, and naturally extends WCP with feature/gradient-based weighting to handle coverage under shift.
3.	Demonstrates weighted CP effectiveness under adversarial label/class shift.
4.	Offers empirical evidence for “why it is more efficient”: a power-law relation between score and error with tail compression, plus visualizations showing monotone metric changes under latent-space directional perturbations.

### Weaknesses
1. Nestedness assumption is strong and insufficiently vetted. For intervals S_β (x_i )=⋯(Section2.3), monotonicity/nestedness in beta is nontrivial with deep networks and nonlinear metrics. The main text lacks a systematic diagnostic/guarantee mechanism (e.g., monotonicity tests, conservative envelopes, or projection fixes), which is critical for theoretical validity.
2. Feature subspace construction and robustness. Collapsing Jacobians by channel-summing before PCA may sacrifice spatial sensitivity; the choice of layer L, cross-layer comparisons, and PCA stability under small samples/noisy gradients are under-analyzed. A single global subspace may underfit heterogeneous distributions (different anatomies/modalities). Do we need grouped/stratified subspaces or online adaptation?
3. Baselines and scope. The main text omits feasible approximations/variants of FCP; include a practically runnable approximate FCP or feature-perturbation baseline for a fair comparison. Only area (a differentiable metric) is evaluated; adaptation and experiments for nonsmooth/compositional metrics (perimeter, Hausdorff distance, morphology indices, or multi-metric joints) are missing.
4. Coverage guarantee depends on fixed Delta learned only from training data. If Delta adapts using calibration or test distributions, do we need data splitting or multiplicity adjustments to preserve validity?

### Questions
1. How do you detect nestedness of S_β (x_i ) in practice? If responses to ±β are asymmetric or non-monotone, what is the remedy—monotone regression, direction re-estimation, or constructing a conservative upper envelope to retain coverage?
2. What is the variance of the PCA subspace across random splits/initializations? Would subspace bagging reduce variance and improve coverage stability?
3. Beyond a LiRPA oracle, can you include approximate adversarial search or low-rank projections + Lp balls as simplified FCP to compare under matched compute budgets?
4. For tiny targets or filamentary anatomy (very small areas), are the intervals still stable? Please report coverage and interval length stratified by target size.
5. If commercial black-box models expose only logits, what are the gains of COMPASS-L versus COMPASS-J? Please specify the minimal API needed (which gradients/intermediate layers) and provide a deployment guide.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The submission focuses on metric-guided conformal prediction for image segmentation models with applications in medical imaging. The submission proposes to construct prediction sets by perturbing the feature representations of pretrained segmentation models along the PCA directions of the gradients of the downstream metric of interest. The method is extended to conformal prediction under distribution shifts.

Experiments compare the proposed method with several existing baselines across a variety of medical imaging datasets.

### Strengths
* The paper is well-written.
* Metric-guided conformal prediction is a relevant topic.
* The proposed method is novel, and experimental evaluation is comprehensive.

### Weaknesses
* Lack of theoretical guarantees on the monotonicity of the sets makes theorems vacuous for the proposed method in a general sense.
* Method is limited to functions of the segmentation mask only, instead of mask and input.

I have a few clarifying questions and comments and I am looking forward to discussing with the authors!

### Questions
**Missing citation to Belhasin et al. (2023)**

The idea of using principal directions to construct prediction sets was introduced by [1] in the context of imaging inverse problems. Although the current submission studies a different problem, it will be important to include this reference.

**Monotonicity of the downstream metric**

The submission acknowledges that nestedness and monotonicity are assumptions of the theoretical results. However, the proposed algorithm does not satisfy these assumptions in practice. Even if this does not seem to affect the experimental results, it does render the theorems vacuous for the proposed method in a general sense. Figure 6 provides evidence that the principal directions do approximate monotonic changes in the downstream metric.

Could the authors expand on technical solutions to guarantee monotonicity? If not, the claim that validity of the algorithm follows by Theorem 1 should be reworded.

**Extending approach to downstream functions of both the image and the mask**

The submission focuses on one downstream task, segmentation area, that only depends on the segmentation mask. Could the authors expand on potential limitations and hurdles to extend the proposed method to:

1. Multi-class segmentation problems.

2. Downstream metrics that also depend on the input image and not the segmentation mask only? This might align better with the motivating example of radiomics in the introduction.

**Dimension of $J_i$**

Could the authors clarify what $C,H$ and $W$ represent? I assume these are not the input image number of channels and spatial dimensions, but the size of the feature embeddings at the chosen layer? Are these vectors flattened in practice to compute the principal directions?

**Experiments**

How many eigenvectors where chosen? Does performance of the proposed method change with $L$? 

Which layer was used to compute the directions? Does this choice matter? It could be interesting to include visualizations of some directions.

In Figure 3, why do unweighted CP methods outperform weighted ones?

**Intuition behind Sec. 3.3**

I am not sure I understand the intuitive explanation given in Sec. 3.3. I agree that a skewed distribution has a lower quantile, but how does a lower quantile imply smaller interval lengths here? These quantiles are computed along different directions, which are then pushed forward through another, potentially nonlinear function? 

---

**Minor comments**

- It might be worth citing a recent work by Mossina and Friedrich, "Conformal Prediction for Image Segmentation
Using Morphological Prediction Sets" (2025).
- Theorem 1: technically, differentiability does not matter for the theoretical result? If I understand correctly, it is needed to compute the principal directions only.
- Line 175: "optimal" in what sense?

---

**References**

[1] Belhasin et al. "Principal uncertainty quantification with spatial correlation for image restoration problems" (2023).

### Soundness
3

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
4

### Summary
This paper introduces COMPASS, applying SCP to medical image segmentation tasks, specifically targeting uncertainty quantification for derived metrics like organ size. COMPASS leverages the inductive biases inherent in deep neural networks by performing calibration in the model’s representation space. The method perturbs intermediate features along low-dimensional subspaces that are most sensitive to the target metric. COMPASS yields tighter, more efficient prediction intervals compared to traditional CP methods.

### Strengths
1. COMPASS calibrates uncertainty in the model’s intermediate feature space, rather than directly on the output metric. This leverages the inductive biases of deep neural networks, resulting in more efficient and tighter prediction intervals
2. COMPASS incorporates a weighted variant to address distribution shifts.

### Weaknesses
1. The method assumes that perturbations along certain directions in the feature space will produce monotonic changes in the metric, this assumption may not hold in all cases, particularly in highly non-linear models, which could lead to suboptimal intervals or coverage guarantees in some real-world scenarios.
2. the weighted COMPASS variant helps correct for moderate distribution shifts, but the validity of the coverage guarantee relies on accurately estimating the density ratio for the calibration and test sets, and errors in this estimation could compromise the method’s effectiveness in practice. 
3. COMPASS's performance is heavily reliant on the quality of the pre-trained model’s feature representations. If the model’s features are poorly aligned with the downstream metric, the perturbation directions may become inefficient or non-monotonic, leading to invalid or wide prediction intervals.

### Questions
- The paper mentions using PCA to select sensitive perturbation directions, which might be computationally intractable in high-dimensional feature spaces. Please provide further discussion or proof that this method remains efficient in very deep or high-dimensional neural networks
- Theoretically, COMPASS provides valid coverage guarantees under exchangeability and nestedness assumptions. However, in real-world applications, these assumptions might not always hold. how the method ensures valid coverage in scenarios where these assumptions do not fully apply?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces COMPASS, a novel framework for conformal prediction (CP) tailored to medical image segmentation tasks. Unlike traditional CP methods that operate on scalar outputs or pixel-level segmentation masks, COMPASS generates uncertainty intervals directly for clinically relevant downstream metrics (e.g., organ size) by perturbing intermediate neural network features along directions sensitive to the target metric. The method leverages PCA on Jacobians to identify low-dimensional subspaces that are sensitive to the metrics, enabling efficient and valid interval estimation. COMPASS is evaluated across four medical segmentation datasets and demonstrates superior statistical efficiency and robustness under covariate shift compared to existing CP baselines.

### Strengths
1. The paper moves beyond pixel-level CP to metric-level CP, addressing a critical gap in medical image analysis---uncertainty quantification for derived metrics rather than raw segmentation outputs. The method is conceptually good and has the potential to improve trustworthiness in clinical settings.

2. The paper introduces a clever and principled way to perturb latent features along sensitive directions derived from PCA on Jacobians.

3. The authors provide rigorous coverage guarantees under exchangeability and nestedness assumptions.

4. The proposed method produces tighter intervals than existing methods across multiple datasets, maintains valid coverage and adapts well to covariate shifts, while outperforming baselines in both interval efficiency and robustness.

### Weaknesses
1. The underlying assumptions---exchangeability and nestedness---may not hold in practice.

2. The method relies heavily on feature quality, i.e., the alignment between extracted features and the evaluation metrics.

3. The experiments focus on segmentation tasks involving area estimation. It remains unclear how well COMPASS generalizes or scales to other metrics (e.g., texture, shape) or modalities (e.g., 3D volumes, multi-class segmentation). In particular, defining a differentiable metric function for such cases may be challenging or even infeasible.

4. The computation of Jacobians can be prohibitively expensive for high-dimensional feature maps. Moreover, the calibration step still requires multiple forward passes, which can be computationally costly for large models or datasets.

### Questions
1. How sensitive is COMPASS to violations of the exchangeability and nestedness assumptions? Have you observed practical cases where these assumptions fail, and how does that affect performance or calibration stability?

2. Is there any way to relax or empirically verify these assumptions before applying COMPASS?

3. Since COMPASS relies on the alignment between features and evaluation metrics, how can we assess or quantify this alignment in practice?

4. Would using features from different network layers or models influence the calibration quality of COMPASS? If so, how can one optimally select which layer’s features to use for Jacobian computation?

5. The experiments focus on area-based segmentation metrics—how would COMPASS handle non-differentiable metrics such as texture or shape? Is it feasible to apply COMPASS to 3D volumetric segmentation or multi-class settings, and what challenges would arise?

6. Has any attempt been made to test COMPASS on non-segmentation tasks, such as regression or classification, to evaluate generality?

7. How do you handle the computational overhead of large Jacobians for high-dimensional feature maps?

8. Given that the calibration step requires multiple forward passes, are there strategies to make it more efficient for large-scale datasets or models?

### Soundness
3

### Presentation
3

### Contribution
3
