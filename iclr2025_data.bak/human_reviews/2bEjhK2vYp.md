## Human Reviewer 1

### Summary
The paper proposed a novel attribution algorithm for feature-level attribution on self-supervised learning. Compared to other feature-level attribution. The method is designed to meet prerequisites that the interpretation should not rely on 1) downstream task 2) other samples (other than the argumentation) and 3) model architectures. Authors present some experiments to justify the new method (SSLA) is effective.

### Strengths
- The prerequisites are designed to resolve the problem caused by other factors. And the method are design to reflect this spirit.
- The diagram are clear and help the readers to understand the method.

### Weaknesses
- I am not sure if the perquisites can be widely accepted by the community. For example, what is the downside (empirically / theoretically) if a downstream task is considered during the attribution process?
- Lack of comparison between different attribution methods. One interesting problem could be what's the difference between SSLA result v.s. other methods that relies on downstream tasks.
- Minor presentation suggestion
  - Equation 1 and 2 seems to be a little redundant.
- I am willing to raise the rate if the effectiveness of SSLA on some traditional evaluation methods (on downstream tasks) are proved (at least) to be correlated

### Questions
- In Figure 2, why we have a full R^n shape for a0, a1, ..., a_T?
- There seems to be no reason seperate the snowflake and light blue arrow in Figure 2?

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
2

---

## Human Reviewer 2

### Summary
This paper addresses the interpretability of SSL models, focusing on the challenge that existing interpretability methods often rely on downstream tasks or specific model architectures. To overcome these issues, the authors propose three fundamental prerequisites for SSL interpretability:
1. The interpretation should not introduce information from downstream tasks.
2. The interpretation process should not introduce samples other than the current sample.
3. The interpretation process should not be restricted to specific model architectures.

Based on these prerequisites, they introduce the Self-Supervised Learning Attribution (SSLA) algorithm. SSLA redefines the interpretability objective by introducing a feature similarity measure. 
They also propose a new evaluation framework tailored to SSL tasks, arguing that traditional interpretability evaluation methods are impractical due to the absence of explicit labels and suitable baselines in SSL settings. Experiments are conducted using five representative SSL methods (BYOL, SimCLR, SimSiam, MoCo-v3, MAE) on the ImageNet dataset with ResNet-50 as the backbone. They compare SSLA against a random masking baseline, demonstrating that SSLA can more effectively identify important features that influence the SSL model's representations.

### Strengths
* **Novel Focus on SSL Interpretability:** The paper addresses an important and under-explored area—the interpretability of SSL models without reliance on downstream tasks or specific architectures.
* **Clear Prerequisites:** The authors clearly outline three prerequisites for SSL interpretability methods, providing a solid foundation for their approach.
* **Architecture-Agnostic Approach:** SSLA is designed to be independent of specific neural network architectures, potentially making it broadly applicable across different SSL models.
Recognition of Evaluation Challenges: The authors recognize the limitations of traditional interpretability evaluation methods in the context of SSL and attempt to propose a new framework tailored to SSL tasks.

### Weaknesses
* **Insufficient Empirical Evaluation:** The experimental evaluation is limited. The authors only compare SSLA to a random masking baseline. There are no comparisons with other existing attribution methods adapted to SSL, making it hard to gauge the effectiveness of SSLA.
* **Limited Dataset and Model Diversity:** Although experiments are conducted on the ImageNet dataset using ResNet-50, the evaluation lacks diversity in both datasets and model architectures. The claim that SSLA is architecture-agnostic is not fully supported without experiments on different architectures.
* **Evaluation Methodology Concerns:** The proposed evaluation framework is novel but not thoroughly validated. The authors argue that traditional evaluation methods are unsuitable for SSL interpretability but do not provide sufficient empirical evidence or theoretical justification. It is unclear whether the metrics used effectively measure interpretability in SSL contexts.

### Questions
*  **Comparison with Existing Methods:** Have you considered adapting existing attribution methods like Integrated Gradients or Grad-CAM to SSL settings for comparison? How does SSLA perform relative to these methods?
*  **Validation of Evaluation Framework:** How have you validated the effectiveness of your proposed evaluation framework? Have you conducted any studies or experiments to show that it correlates with human intuition or ground truth attributions?
*  **Testing on Diverse Architectures:** Given that experiments are only conducted with ResNet-50, have you tested SSLA on other architectures?
*  **Handling SSL Randomness:** How does SSLA account for the randomness inherent in SSL methods, such as stochastic data augmentations? Does this randomness affect the stability of the attribution results?
*  **Computational Overhead:** What is the computational cost of SSLA compared to standard inference? Is it feasible to apply SSLA to large-scale models and datasets?
*  **Generalization to Other Domains:** Can SSLA be applied to SSL models in domains other than computer vision?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper proposes SSLA, a feature attribution method for self-supervised learning (SSL) tasks. In particular, the method is designed without dependency of downstream tasks. The method starts by defining the usefulness of SSL model as its ability to preserve representation of data after transformation. Then it addresses the significance of features by attributing this usefulness to features iteratively. The paper then conducts feature masking experiment to demonstrate the effectiveness of the method.

### Strengths
1. The paper proposed a novel feature attribution method for SSL model that does not rely on downstream tasks. 
2. The claim made in the paper is supported by both theoretical derivations and experiments. 
3. The paper is well written and easy to follow. The discussion of prerequisites of an attribution method for SSL may spark interesting discussions.

### Weaknesses
1. The paper relies on the independence of downstream tasks, which make the comparison of this method and existing methods difficult. Hence, it is difficult to address the effectiveness of this method. 
2. The derivation of theorems rely heavily on first order approximation. Though it is common, the paper does not provide analysis on error bounds, which downgrades the trustworthiness of the method.
3. The two main components of the method lack motivation. The first one is using cosine-similarity of features before/after transformation as a measure of usefulness of SSL model. The correlation (or even causality) of this and "SSL as learning representation" is not clear. The second one is the iterative method. The author may consider justify why we need an iterative method to attribute the importance.
4. Although the paper proposes the method to be independent of downstream tasks, its evaluations still rely on downstream tasks, which seems counter-intuitive(Line 179 -180). Moreover, since the evaluation is dependent of downstream tasks, the author may consider compare their method to other SSL attribution methods that rely on downstream tasks.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper indicates that the introduced additional samples from downstream task would impede the interpretability of Self-Supervised Learning (SSL). To tackle this issue, the authors try to propose a new interpretability objective by introducing a feature similarity measure, decoupling the interpretability process from the reliance of downstream tasks.

### Strengths
- The additional information introduced by prediction head and downstream tasks may influence the interpretability of SSL is a crucial and challenging  breakthrough point.
-  Three fundamental prerequisites for the interpretability of SSL proposed by this paper sounds reasonable.

### Weaknesses
- Despite the author indicating the potential weakness for the interpretability of SSL, the alignment ability for data augmentation is just one side of current self-supervised learning. The other component, which is often regarded as an extra design to prevent training collapse intuitively, is essentially to ensure that the representation divergence is sharp enough to cluster the data distribution by latent categories. More details can be found in [1] and [2]. Therefore, only adopting the extent of augmentation that is invariant to evaluate the related influence of variables is quite biased.

   - [1] Awasthi, Pranjal et al. “Do More Negative Samples Necessarily Hurt in Contrastive Learning?” International Conference on Machine Learning (2022).
   - [2] Weiran Huang, Mingyang Yi, Xuyang Zhao, and Zihao Jiang. Huang, Weiran, et al. "Towards the generalization of contrastive self-supervised learning." arXiv preprint arXiv:2111.00743 (2021).

### Questions
**Reviewing summary**
- As listed in the weaknesses, I think the authors did not incorporate the divergence into their consideration, which can be regarded as the most critical component of contrastive self-supervised learning, making their criterion sound unreasonable. Despite their indicating the unreasonable aspects regarding the interpretability of SSL in relation to downstream tasks, this results in my score of 3.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4