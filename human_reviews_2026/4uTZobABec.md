# Learning a distance measure from the information-estimation geometry of data

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 8

## Abstract
We introduce the Information-Estimation Metric (IEM), a novel form of distance function derived from an underlying continuous probability density over a domain of signals.
The IEM is rooted in a fundamental relationship between information theory and estimation theory, which links the log-probability of a signal with the errors of an optimal denoiser, applied to noisy observations of the signal.
In particular, the IEM between a pair of signals is obtained by comparing their denoising error vectors over a range of noise amplitudes.
Geometrically, this amounts to comparing the score vector fields of the *blurred* density around the signals over a range of blur levels.
We prove that the IEM is a valid global distance metric and derive a closed-form expression for its local second-order approximation, which yields a Riemannian metric.
For Gaussian-distributed signals, the IEM coincides with the Mahalanobis distance.
But for more complex distributions, it adapts, both locally and globally, to the geometry of the distribution.
In practice, the IEM can be computed using a learned denoiser (analogous to generative diffusion models) and solving a one-dimensional integral.
To demonstrate the value of our framework, we learn an IEM on the ImageNet database.
Experiments show that this IEM is competitive with or outperforms state-of-the-art supervised image quality metrics in predicting human perceptual judgments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper innovatively proposes the Information-Estimation Metric (IEM), a distance measure derived from the local geometric properties of probability distributions (denoising errors and score vector fields). It breaks through the limitation that traditional supervised perceptual metrics rely on manual annotations, validates its unsupervised learning capability on the ImageNet dataset, and outperforms most existing methods in image quality assessment tasks. The overall theoretical derivation is rigorous and the experimental design is comprehensive.

### Strengths
1.Solid Mathematical Foundation. The paper reliably demonstrates the mathematical significance of its work.
2.Comprehensive Experimental Validation and Credible Conclusions. The paper verifies the advantages of IEM over other metrics through experiments and tests.

### Weaknesses
1. The drawback of low computational efficiency of the IEM. 
2.The experimental scope is relatively limited, with insufficient validation of the IEM’s performance across a wide range of noise types.

### Questions
More experiments on different datasets should be more convincing.
Could you provide more evidence of the demand for IEM?

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
2

### Summary
This paper proposes the IEM and primarily validates it on image quality assessment tasks. The core idea involves measuring distribution shifts, and the method is not inherently limited to visual data. The experimental results demonstrate its performance on specific datasets like TQD.

### Strengths
*   **General Formulation:** The proposed IEM is not fundamentally tied to visual data, suggesting potential applicability to other media types like audio or natural language, as it makes no specific assumptions about the modality.
*   **Interesting Motivation:** The motivation based on "humans exhibit complex patterns of sensitivity that vary with the direction of the signal’s perturbation" is compelling and aligns with nuanced aspects of perception.

### Weaknesses
*   **Insufficient Experimental Validation:**
    *   The paper's core motivation and problem formulation do not appear exclusively focused on Image Quality Assessment (IQA), yet validation is conducted *only* within IQA tasks. This is insufficient. If the goal is a general measure of distribution shift, experiments on other modalities (e.g., sound, text) are necessary to substantiate the claim of generality.
    *   Within the IQA validation, there is a lack of experiments on real-world images with complex, mixed distortions. This is crucial for verifying the method's ability to assess information changes caused by complicated, realistic impairments.
*   **Lack of Comparative Analysis:** A critical weakness is the absence of a comparative analysis between IEM and other established information-theoretic distance measures like Kullback-Leibler Divergence (KLD), Jensen-Shannon Divergence (JSD), and Earth Mover's Distance (EMD). Such a comparison is essential for demonstrating IEM's potential superiority or unique advantages.
*   **Inadequate Explanations:**
    *   The significant performance discrepancy of `IEM_sq` (poor on photo quality assessment but good on TQD) lacks a thorough explanation from either a theoretical or experimental analysis perspective.
    *   The notation `IEM_fw` is introduced without a clear explanation.
*   **Parameter Sensitivity:** The IEM demonstrates high sensitivity to the parameter `Γ` (Gamma), with the "optimal" value varying drastically (e.g., from `1/4` to `10^6`) between different IQA tasks (photographic vs. texture images). This instability is a practical concern that needs addressing.

### Questions
1.  **Generalizability and Scope:** The paper's motivation seems broader than IQA. Have you considered, or could you perform, validation on non-visual modalities (e.g., audio, text) to truly demonstrate IEM's general applicability for measuring distribution shifts?
2.  **Theoretical Interpretation and Measurement:** A key motivation relates to sensitivity varying with the *direction* of perturbation. In high-dimensional spaces (e.g., latent spaces), how can the "direction" of distributional change be effectively measured? Furthermore, how might the similarity or difference between these "directions" of change impact IEM's performance? Is IEM inherently sensitive to this directional information?
3.  **Application to Generative Models:** Could IEM be applied to quantify the difference between generated and real images, thus serving as a metric for evaluating the performance of generative models?
4.  **Comparative Advantage:** How does IEM's performance compare directly to other information-theoretic measures like KLD, JSD, and EMD on your tasks? This comparison is vital for establishing IEM's value.
5.  **Explanation of Results and Definitions:** What explains the contradictory performance of `IEM_sq` (good on TQD, poor on photo quality assessment)? Can you provide a theoretical hypothesis or detailed experimental analysis?
6.  **Robustness to Parameters:** The performance is highly sensitive to `Γ`. What is the underlying reason for this sensitivity? Are there strategies to make the method more robust to the choice of `Γ`, or to determine an optimal `Γ` in a more principled, task-agnostic way?
7.  **Sensitivity to Denoiser Choice:** Is the performance of the proposed method sensitive to the selection of the specific denoiser used?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors introduce a novel distance metric called the Information-Estimation Metric (IEM). The IEM is defined based on fundamental principles connecting probability density with local geometry (i.e., connecting the estimation error to the gradient of the log of the marginal). Authors provide a rigorous theoretical framework, as well as experiments to further explain the usefulness of the proposed metric. Hence, I recommend this paper to be accepted.

### Strengths
The paper is well-written, the motivation is clear and sound. Theoretical framework is thorough, experimental section is provided. While the computational complexity can be an issue, authors still provided experimental results, which is appreciated.

The paper introduces a new distance metric that relies on the geometry of the distribution. The idea is interesting and useful. One strength is that the method is not informed by any downstream task, and is rather derived directly from the probability distribution of the data. Second, the IEM does not rely on the manifold hypothesis, and is well-defined for any valid probability density.

The IEM depends directly on the prior $p_{\mathbf{x}}$, rather than through an observation model, i.e., the representation $p_{\mathbf{y} | \mathbf{x}}$, that may or may not be dependent on the prior. And, the IEM is a proper metric, from which the authors derive a local metric $\mathbf{G}$ such that it is a one-way relationship. I.e., the IEM is not a geodesic distance.

Lastly, the experiments show interesting results -- the distance metric can predict human perceptual judgement.

Overall, the paper presents a significant theoretical advancement in connecting information theory, estimation theory, and geometry to derive perceptually meaningful distance metric.

### Weaknesses
An obvious weakness, as the authors acknowledge in the Discussion section, is the computational complexity of the approach. Numerical estimation of an integral is expensive, and there exist other works that provide a metric but have a smaller cost. It would be great if authors could come up with a way to make this practical and usable for the community. Also, the method needs training a diffusion model, which is also expensive.

The paper provides a metric for continuous distributions, which is often not available or well-defined for all types of data.

The experiments are focused on natural images. One of the strengths that the authors mention is that the IEM does not rely on the manifold hypothesis. While I understand that the theory of IEM does not rely on the manifold hypothesis, it strikes me that natural images \textit{do} lie on a manifold. It would make the paper stronger to see experiments on data that do not necessarily lie on a manifold (e.g., text).

Despite these limitations, the paper presents a significant theoretical advancement in connecting information theory, estimation theory, and geometry to derive perceptually meaningful distance metrics.

### Questions
How do authors expect to solve the computational complexity issue?

Have authors conducted experiments on other types of data?

The authors show that for Gaussian distributions, the IEM coincides with the Mahalanobis distance. Are there other known probability distributions where the IEM has a closed-form expression?

How does the IEM relate to other geometrically-informed distance measures like optimal transport distances (e.g., Wasserstein distance)?

The paper focuses on supervised versus unsupervised approaches. Could a semi-supervised approach yield better results with less labeled data?

How stable is the IEM to small perturbations in the underlying data distribution? Is there a way to quantify its robustness?

How might the IEM perform in domains where human perceptual ground truth is unavailable or difficult to obtain?

Could the IEM be applied to other perceptual tasks like anomaly detection or out-of-distribution detection?

How does the local metric G(x) derived from the IEM compare to other local metrics derived from different principles in manifold learning?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a well-defined (proper) distance metric called the Information-Estimation Metric (IEM) induced by a probability distribution. Specifically, the metric is computed by comparing the denoising errors between the input signals when subject to varying levels of noise. ​It is built on the pointwise ​I-MMSE and the Tweedie-Miyasawa formulas. Inspiration is also drawn from the generative process in a diffusion model. The metric is shown to be the Mahalanobis metric when the underlying distribution is Gaussian. As an application, it is shown to correlate well with human judgment when applied to image pairs. Other interesting applications are identified for future investigation.

The paper makes a solid contribution with several potential follow-up avenues. The experiments are both illustrative and convincing, and the writing is top-notch. ​

### Strengths
​The formulation is novel and has many potential applications.

​The exposition and the illustrations are excellent.

### Weaknesses
A better connection between IEM and the image quality experiment could be made.

### Questions
Could you please provide some intuition on how iso-IEM contours look like at larger scales (or for larger epsilon)?  

As a follow-up question, can you please comment on the IEM landscape in the context of image quality assessment? Specifically, how do images that lie on an iso-IEM contour or a ball with the reference at the center look like? How do they compare with, say, iso-VIF or iso-LPIPS images?

Can you please point to the prior distribution assumed for the IQA experiments? I may have missed this detail. 

Why does the IEM correlate well with human judgment?

Can you please comment on the motivation for exploring the connection with the KL divergence? From what I understand, the IEM is defined for points drawn from the same underlying prior data distribution. The KL divergence, on the other hand, compares distributions. The translation of p_x does answer this to some extent, but some more insights could help the readers.

### Soundness
4

### Presentation
4

### Contribution
4
