# Enhancing Image-Conditional Coverage in Segmentation: Adaptive Thresholding via Differentiable Miscoverage Loss

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Current deep learning models for image segmentation often lack reliable uncertainty quantification, particularly at the image-specific level. While Conformal Risk Control (CRC) offers marginal statistical guarantees, achieving image-conditional coverage, which ensures prediction sets reliably capture ground truth for individual images, remains a significant challenge. This paper introduces a novel approach to address this gap by learning image-adaptive thresholds for conformal image segmentation. We first propose AT (Adaptive Thresholding), which frames threshold prediction as a supervised regression task. Building upon the insights from AT, we then introduce COAT (Conditional Optimization for Adaptive Thresholding), an innovative end-to-end differentiable framework. COAT directly optimizes image-conditional coverage by using a soft approximation of the True Positive Rate (TPR) as its loss function, enabling direct gradient-based learning of optimal image-specific thresholds. This novel differentiable miscoverage loss is key to enhancing conditional coverage. Our methods provide a robust pathway towards more trustworthy and interpretable uncertainty estimates in image segmentation, offering improved conditional guarantees crucial for safety-critical applications. The code is available at https://github.com/bjbbbb/Conditional-Optimization-for-Adaptive-Thresholding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper targets uncertainty quantification for high-risk image segmentation and argues that standard Conformal Risk Control (CRC) only offers marginal (overall) risk control, leaving large per-image variability. It proposes Conformal Risk Adaptation (CRA): recast threshold selection as a weighted quantile problem (eliminating grid search), and use a two-stage calibration—(i) monotone non-parametric pixel-probability recalibration (e.g., isotonic regression), then (ii) stratified calibration at the image level based on the total predicted probability mass to improve conditional risk behavior.

### Strengths
1.	Recasting risk control as weighted quantile estimation yields a closed-form threshold solution, avoiding discretization error and the computational overhead of grid search.
2.	Ranking by normalized cumulative probability mass makes the method self-adaptive to images with different “brightness/confidence” profiles—directly addressing the source of conditional-risk heterogeneity.
3.	The calibration design is sensible: pixel-level monotone recalibration improves the fidelity of total mass estimates; stratified, image-level calibration provides group-conditional control that improves fairness and stability.
4.	Code repository provided; good reproducibility.

### Weaknesses
1. Theory scope/assumptions insufficiently specified: In Theorem 1, the “arbitrary scoring function” premise should clarify what monotonicity/thresholdability conditions are required so that CRC-style isotonicity and risk upper bounds hold. For stratified calibration (Eq. (7)–(8)), assumptions on the stratification function g(fixed a priori vs. data-dependent) are unclear; if Kand stratification thresholds are tuned on the calibration set, the finite-sample guarantees may be compromised without data splitting or multiplicity control. Connections to group-conditional CP and finite-sample smoothing/quotas need tightening.

2. Using $\sum_j \hat{p}_j$ as the stratifier could correlate strongly with target area/prevalence, potentially systematically treating small objects or low-contrast images differently. Analyze fairness/bias impacts.

3. Missing comparisons to spatially aware/topology-aware conformal methods or pixel-grouping approaches; Only reporting t-tests is weak—please add effect sizes and confidence intervals; The Coverage Gap (mean absolute deviation) may be high-variance for small ∣Y∣; consider tail-risk metrics (CVaR, quantile gaps).
	
4. Multi-label extension is under-specified (class coupling, probability-sum constraints, shared vs. per-class CRA) and lacks guarantees.

### Questions
1. Stratification guarantees (Eq. (8)): Must g be fixed before training? If Kand strat thresholds are chosen from calibration data, do you require data splitting or multiple-testing correction to maintain finite-sample guarantees?
2. Ties & discretization: CRA scores rely on cumulative probability mass. How do you handle ties (e.g., quantized/duplicate probabilities)? Do ties induce threshold jitter or systematic conservativeness in coverage?
3. Stratifier robustness: Why choose $\sum_j \hat{p}_j$ specifically? Have you evaluated alternatives (e.g., entropy, peakiness, predicted foreground fraction), and their effects on conditional risk and normalized size?
4. Multi-class extension: For multi-class segmentation, do you compute per-class quantiles or a shared joint threshold? How do you preserve both overall and class-conditional risk guarantees?
5. Fairness & small targets: On small-volume/rare-object subgroups, does CRA reduce Coverage Gap and FN hotspots (false-negative heatmaps)? Please report size-stratified results to substantiate the “per-case stability” claim.
6. Comparisons & ablations: Could you add results combining CRA with spatial priors/shape regularization (e.g., CRFs/morphology) and an ablation of CRA components (isotonic recalibration / stratification / weighted quantile) to show the accuracy–size–risk trade-offs?
7. Constants & upper bound B(Eq. (3)): What is the source and magnitude of B? If different loss bounds or surrogate risks (e.g., smoothed FNR) are used, how do the threshold solution and guarantees change?

### Soundness
2

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
5

### Summary
This paper addresses the challenge of achieving image-conditional coverage in image segmentation tasks, a crucial problem for safety-critical applications. The authors introduce two novel methods for adaptive thresholding: Adaptive Thresholding and Conditional Optimization for Adaptive Thresholding. These methods aim to improve uncertainty quantification in segmentation tasks by learning image-specific thresholds that control the FNR at an image level, rather than relying on marginal guarantees that apply to the entire dataset. The COAT method, in particular, introduces a differentiable loss function that allows end-to-end optimization for the target conditional coverage, enabling more reliable and interpretable uncertainty estimates. Through extensive experiments, the authors show that COAT consistently outperforms other methods, such as CRC and AA-CRC, in controlling the FNR and achieving more consistent and accurate conditional coverage across different images and datasets.

### Strengths
1. The paper tackles the significant challenge of achieving image-specific uncertainty quantification in segmentation tasks. I think conditional guarantees are critical.  
2. The COAT method introduces a novel end-to-end differentiable miscoverage loss, allowing for direct optimization of image-conditional coverage without the need for pre-calculated thresholds, this enhances the flexibility and accuracy of uncertainty estimates

### Weaknesses
In some applications, such as medical diagnostics, controlling both false positives and false negatives is essential, and the method could benefit from a more balanced approach to risk control. For example, is it possible to simultaneously derive threshold intervals that control both the FNR and FPR? An analysis of the balance between the two would be valuable. 

It would be valuable to extend the evaluation to multi-class segmentation problems, where each pixel could belong to more than one class. 

the paper focuses on image level, in some cases, different regions within an image may have varying degrees of importance. For example, in medical imaging, certain organs or lesions may require stricter coverage guarantees than others. Introducing class-specific or region-specific thresholds could enhance the reliability

### Questions
If the goal is to control the FDR (not FPR), the monotonicity between FDR and the threshold can be discussed. Additionally, it would be beneficial to explore how to optimize other performance metrics while controlling the FNR.


I will consider adjusting the score based on the response.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses a critical limitation in deep learning-based image segmentation: the lack of reliable image-conditional uncertainty quantification. While methods like Conformal Risk Control (CRC) provide marginal guarantees on performance metrics (e.g., False Negative Rate) across a dataset, they fail to ensure reliability for individual images, a crucial requirement for safety-critical applications. To bridge this gap, the authors propose learning image-adaptive thresholds to achieve image-conditional coverage.
The paper introduces two methods: first, a supervised baseline named AT (Adaptive Thresholding), which treats threshold prediction as a regression task on pre-computed optimal thresholds. The primary contribution is COAT (Conditional Optimization for Adaptive Thresholding), a novel end-to-end differentiable framework. COAT introduces a differentiable miscoverage loss, which uses a soft, sigmoid-based approximation of the True Positive Rate (TPR). This allows the model to directly optimize for the desired conditional coverage via gradient-based learning, circumventing the need for pre-computed targets.
Experimental results across multiple datasets and base models demonstrate that COAT significantly reduces the "Coverage Gap"—the average deviation from the target coverage per image—compared to existing methods like CRC, while still upholding the marginal coverage guarantee. The work presents a robust pathway towards more trustworthy and reliable instance-wise uncertainty estimates in image segmentation.

### Strengths
1. Originality
The paper introduces a novel formulation of conditional coverage for semantic segmentation, addressing a key limitation of traditional conformal prediction methods that only ensure marginal coverage. The proposed COAT framework is conceptually original in two ways:
(1) It reframes adaptive threshold selection as a differentiable optimization problem using image-specific TPR-based objectives.
(2) It presents a new end-to-end surrogate loss that approximates conditional miscoverage, enabling training of a threshold predictor that generalizes across diverse image characteristics.
This combination of conditional risk control and differentiable threshold learning is innovative and extends conformal prediction into a domain where such guarantees are rare and previously unattainable.
2. Quality
The methodological quality is strong. 
The algorithmic design is principled, with clear motivation, precise mathematical formulation, and a calibration step that preserves marginal coverage guarantees. Experiments are extensive, evaluating multiple backbone architectures and several conformal baselines. Results are reported across 20 random data splits, with means and standard deviations, showcasing statistical rigor.
The paper also includes a theoretical justification proving asymptotic conditional coverage under reasonable assumptions.
3. Clarity
The paper is generally well written and easy to follow. Algorithm 1 is detailed, explicit, and matches the narrative description. Figures provide intuitive interpretations of coverage behavior and prediction stability. The appendix further improves clarity by discussing theoretical assumptions, asymptotic properties, and the role of LLM-assisted content generation.
4. Significance
The paper makes a substantial contribution to reliable machine learning for safety-critical applications. Ensuring per-image conditional coverage is significantly more relevant than marginal guarantees in medical settings, where rare under-covered samples can lead to severe consequences. The proposed COAT method improves reliability not only statistically but also visually, producing prediction sets with fewer false negatives and more consistent behavior across images.

### Weaknesses
While the paper presents a strong and compelling contribution, there are several areas where it could be improved to enhance its clarity, reproducibility, and impact.
1. The definition of the soft mask Msoft(X), using temperature parameter T to control the sharpness of the approximation. The paper provides no rationale for this specific design choice. To solidify this core component of the methodology, the authors should clarify if this formulation offers specific theoretical or empirical advantages, or provide an ablation study comparing it to the more conventional approach.
2. The paper positions COAT as a more advanced, end-to-end extension of the supervised AT baseline. However, the experimental results group the comparison of AT and COAT together with external baselines. A more focused, head-to-head comparison is needed to clearly demonstrate the value added by COAT's end-to-end differentiable framework. A direct analysis isolating the performance difference (especially in Coverage Gap) between AT and COAT would be crucial to empirically justify the increased complexity and novelty of the COAT approach.
3. The paper does not provide a quantitative analysis of the computational overhead. While the appendix mentions the hardware used for training, it omits a comparison of training times and, more critically, inference latency against the other methods. For safety-critical domains like autonomous driving, the practicality of a method depends heavily on its speed. A comparison of the inference time per image for CRC, AT, and COAT would be a vital addition for assessing the method's real-world viability.
4. The qualitative analysis focuses primarily on successful examples where COAT performs well. While useful, this presents an incomplete picture of the method's capabilities. To provide a more nuanced understanding of its limitations, the authors are encouraged to include and discuss some failure cases or instances where COAT still struggles to close the coverage gap. This would offer valuable insights into the types of images for which adaptive thresholding remains a challenge.

### Questions
1. The differentiable loss is a core contribution of this work. Its soft mask is defined as Msoft(X)=σ( (pb(X) - τb(X)) / T ). Could the authors please provide the rationale behind this formulation?
2. The paper presents COAT as an advancement over the supervised AT baseline. While the results in Table 1 are strong, AT and COAT are compared within a group of other methods. To better isolate the specific contribution of the end-to-end differentiable framework, could the authors provide a more direct, head-to-head comparison between AT and COAT?
3. For safety-critical applications, the computational cost of a method is a crucial factor. While the appendix notes the hardware used, a comparative analysis of performance is missing. Could the authors please provide the average inference latency (e.g., in milliseconds per image) for the CRC, AT, and COAT methods? This would be essential for understanding the practical trade-offs involved in adopting the proposed adaptive frameworks.
4. The qualitative results effectively showcase instances where COAT succeeds. To provide a more complete picture of the method's capabilities, could the authors include or discuss any failure cases or types of images where COAT still struggles to achieve the target coverage? An analysis of such challenging cases would offer deeper insights into the method's limitations and highlight promising directions for future research.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of achieving image-conditional coverage in conformal image segmentation, where existing Conformal Risk Control (CRC) methods only provide marginal guarantees. The authors propose two novel methods: Adaptive Thresholding (AT), which formulates threshold prediction as a supervised regression task, and COAT (Conditional Optimization for Adaptive Thresholding), an end-to-end differentiable framework that directly optimizes for image-conditional coverage using a soft approximation of the True Positive Rate (TPR) as its loss. The approach enables image-specific threshold learning, offering more reliable and interpretable uncertainty estimates for segmentation models, particularly in safety-critical domains.

### Strengths
The paper tackles an important and underexplored problem in conformal prediction—extending coverage guarantees from marginal to image-conditional levels. The proposed COAT framework is technically sound and conceptually novel, offering a differentiable formulation for conditional risk optimization. The introduction of a soft TPR-based loss is elegant and allows gradient-based training, which is rarely explored in this context. The work is clearly written, well-motivated, and highly relevant to safety-critical applications such as medical imaging. Experimental results (if included) would likely demonstrate meaningful improvements in both conditional reliability and interpretability over standard CRC baselines.

### Weaknesses
While the proposed idea is novel and well-motivated, several limitations remain.

Lack of empirical depth: The paper would benefit from more extensive experiments across diverse segmentation datasets and backbone architectures to demonstrate the generality of AT and COAT beyond the presented setting.

Limited comparison scope: The baseline comparisons appear restricted to CRC-based approaches; including Bayesian or ensemble-based uncertainty methods could strengthen the evaluation.

Theoretical guarantees: Although COAT introduces a differentiable loss to approximate conditional coverage, the paper lacks a clear theoretical justification of when or why this approximation yields valid conditional guarantees.

Ablation clarity: It is not entirely clear how much of the observed improvement stems from the adaptive thresholding itself versus the differentiable optimization scheme; more detailed ablation or visualization would help clarify this.

Practical considerations: The additional computational cost and training stability of COAT are not well quantified, which could limit adoption in large-scale or real-time applications.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2
