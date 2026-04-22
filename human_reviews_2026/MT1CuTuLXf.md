# ReconstructionNet: A Neural Network Architecture for Uncertainty-Aware Predictions with Explainability

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Uncertainty estimation quantifies a model’s confidence in its predictions, fostering calibrated trust among users. Existing approaches face two key limitations: (1) most capture only a single type of uncertainty, and (2) they incur additional training or inference overhead. We propose ReconstructionNet, a neural network that addresses these limitations by modeling the joint input–output distribution with class-specific autoencoders. This enables simultaneous prediction and estimation of both aleatoric and distributional uncertainty in a single pass. Across five real-world datasets, ReconstructionNet matches or surpasses baseline classifiers while producing uncertainty estimates with greater reliability, selectivity, robustness to false negatives, and strong out-of-distribution detection. Furthermore, ReconstructionNet’s architecture naturally supports uncertainty explanations, revealing how individual features contribute to prediction uncertainty without extra computation. Experiments demonstrate that these explanations highlight misclassified regions consistent with human intuition. Together, these contributions establish ReconstructionNet as a unified framework for trustworthy and interpretable artificial intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes ReconstructionNet, which models the input-output joint distribution using class-specific autoencoders. In a single forward pass, the model simultaneously provides predictions, distributional (reconstruction error) uncertainty, and aleatoric uncertainty (entropy of the prediction). The reported classification performance on multiple real-world datasets (tabular and medical images) is comparable to or better than baselines.

### Strengths
The proposed uncertainty estimation method does not require additional multiple sampling or other modules, and it distinguishes between aleatoric and distributional uncertainty. The paper also provides relatively thorough theoretical proofs.

### Weaknesses
1. The related work on Bayesian Methods in Section 1.2.1 appears somewhat outdated. The statement that they are "usually computationally expensive" is somewhat one-sided. In recent years, many studies on Bayesian uncertainty estimation have also been computationally efficient, such as Bayesian Last Layer-type methods ([1] and [2]), which reduce the computational overhead of forward propagation by restricting stochasticity to specific layers.

2. In Section 3.2, the authors use θ₁ and θ₂ to distinguish between the two types of uncertainty in the joint probability formulation. However, this seems to be used only for conceptual characterization and does not appear to be involved in the actual network optimization or uncertainty evaluation process.

3. Using reconstruction error as uncertainty seems to heavily rely on the idea of reconstruction-based Out-of-Distribution (OOD) detection. The assumption inherent in the network architecture is that samples within a class are considered in-distribution, while samples from different classes are treated as out-of-distribution.


[1] Harrison, James, John Willes, and Jasper Snoek. "Variational Bayesian Last Layers." Fifth Symposium on Advances in Approximate Bayesian Inference.


[2] Hu, Xinyue, et al. "Enhancing Uncertainty Estimation and Interpretability with Bayesian Non-negative Decision Layer." The Thirteenth International Conference on Learning Representations.

### Questions
1.   What is the rationale behind treating the poorly reconstructed part (i.e., high reconstruction error) as having high distributional uncertainty? How is it ensured that this model is well-calibrated? It is recommended to supplement the experiments with metrics such as Expected Calibration Error (ECE) and Negative Log-Likelihood (NLL).

2. A major concern arises from the training cost: there is one autoencoder per class. How is the training overhead managed? How scalable is the method for large-scale datasets with a large number of classes (e.g., ImageNet-1K)?

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
3

### Summary
This paper focuses on the problems of uncertainty and the explanation in neural networks. Existing uncertainty estimation methods either capture only one type of uncertainty or require high computational cost, and they typically lack explanations regarding the source of uncertainty. To address these limitations, this paper proposes ReconstructionNet, a new architecture designed to predict and quantify different types of uncertainty while providing built-in interpretability. Experiments conducted on both tabular and image datasets demonstrate the effectiveness of the ReconstructionNet in terms of the prediction accuracy, reliable uncertainty estimates, and intuitive uncertainty explanations that highlight regions contributing to model uncertainty.

### Strengths
1. The proposed neural network architecture offers multiple desirable properties, including the ability to estimate both aleatoric and distributional uncertainty within a single model, while also providing inherent uncertainty explanations without the need for additional modules.
2. The paper includes a theoretical analysis of the proposed uncertainty explanations, discussing properties like the sensitivity and consistency.

### Weaknesses
1. The paper claims that the proposed architecture can capture different types of uncertainty and minimize epistemic uncertainty. However, epistemic uncertainty is only briefly mentioned in Table 1 as part of the prediction performance. There is no explicit evaluation, baseline comparisons, or in-depth analysis of its estimation performance.
2. The paper focuses solely on uncertainty estimation methods in the experiments, but does not include or discuss uncertainty calibration methods such as Temperature Scaling [1] or Parametrized Temperature Scaling [2].
3. The uncertainty evaluation mainly considers the correlation and AURC metrics. The other commonly used uncertainty error metrics, such as Expected Calibration Error (ECE) or Adaptive Calibration Error (ACE), are not included.
4. In terms of uncertainty explanation evaluation, the paper discusses gradient-based and perturbation-based methods in the Related Work section, but it does not provide any quantitative or qualitative comparisons with these methods to validate the overall performance of its explanations.

[1] On Calibration of Modern Neural Networks. ICML 2017.

[2] Parameterized Temperature Scaling for Boosting the Expressive Power in Post-Hoc Uncertainty Calibration. ECCV 2022.

### Questions
1. The paper only presents uncertainty explanations for image data. How does the proposed method perform on tabular data, and how much interpretable or supportive explanation information can be derived in that context?
2. The construction details of the proposed architecture are not clear. Is ReconstructionNet built upon existing neural network architectures, and can it be adapted or integrated with different model types?

### Soundness
2

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
3

### Summary
This paper proposes a novel uncertainty estimation method capable of modeling different types of uncertainty. This method relies on a class-specific autoencoder and computes uncertainty by calculating the discrepancy between samples and their reconstructions.

It is important to note that this method is similar to reconstruction-based anomaly detection methods; thus, a sufficient comparison with anomaly detection methods and an enumeration of their differences are necessary.

Additionally, the experiments of this model have been mainly conducted on simple datasets, and how to extend it to large-scale experiments also requires further clarification.

### Strengths
1. can model different types of uncertainty, including aleatoric, epistemic, and distributional uncertainty.

2. Can give an explanation for uncertainty. 

3. Achieving good performance compared to baseline models.

### Weaknesses
1. The connection and difference between the proposed method and autoencoder-based anomaly detection methods.

2. How to scale to large-scale datasets.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a reconstruction-guided image classification with uncertainty quantification. They compute aleatoric uncertainty using entropy-based calculations of the classification probabilities, and epistemic uncertainty using the reconstruction error. Their proposed method is evaluated with six different methods of UQ in six different datasets. Experimental results demonstrate the superior performance of their proposed method.

### Strengths
**1. Good literature review.** The paper is enriched with discussion of related works in uncertainty quantification. They have discussed Bayesian, Evidential, and Deterministic methods of uncertainty quantification. 

**2. Strong experimental results.** This paper conducts extensive evaluation with different datasets and compares with different baselines. They report prediction performance, quality of estimated uncertainty, and OOD detection.

### Weaknesses
**1. Limited novelty:** The reconstruction loss as a regularizer for a image classification task is not novel and have been used in many other previous works [1,2]. A proper literature review of those works is required to better understand the novelty of this work. 

**2. Poor writing:** The paper is filled with many definitions which are standard in ML and UQ literature. Those definitions are not the contribution of this paper and including them wastes writing resources. 

**3. Confusing uncertainty decomposition.** Uncertainty in deep models are divided into aleatoric and epistemic uncertainty. The distribution uncertainty is a subset of epistemic uncertainty specific to domain-generalization. However, the paper interchangeably utilize these terms and it beccomes confusing to the reader. 

**4. Scalability.** Another serious limitation of this paper is its architecture choice. It essentially creates $C$ encoder to perform the reconstruction and classification. However, as class number $C$ grows, this paper's method will struggle to generalise. This is an important limitation of this model's architecture choice. For example, this method is not applicable to any ImageNet classification system.

[1] Le, L., Patterson, A. and White, M., 2018. Supervised autoencoders: Improving generalization performance with unsupervised regularizers. Advances in neural information processing systems, 31.

[2] Ghifary, M., Kleijn, W.B., Zhang, M., Balduzzi, D. and Li, W., 2016, September. Deep reconstruction-classification networks for unsupervised domain adaptation. In European conference on computer vision (pp. 597-613). Cham: Springer International Publishing.

### Questions
N/A

### Soundness
1

### Presentation
2

### Contribution
1
