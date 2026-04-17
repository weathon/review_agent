# Towards Efficient Fairness Image Retrieval with Disentangled Information Suppression

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Deep hashing has emerged as an effective method for large-scale image retrieval, improving computational efficiency by converting high-dimensional data into compact binary codes. Despite its success, recent studies reveal that deep hashing methods may exhibit fairness issues, leading to biased or discriminatory retrieval results across demographic groups. To jointly improve retrieval accuracy and group fairness, we introduce Disentangled Information Suppressed Hashing (DISH), a framework that learns fair and discriminative representations. DISH employs a disentangled encoder to decompose each image into factor-specific representations. To encourage semantic concentration and interpretability, a disentangled consistency objective is introduced to enforce factor-level stability under augmentation and align semantic evidence with latent factors.
Furthermore, an information suppression module is designed to mitigate sensitive information leakage through probability-driven channel masking, channel-wise adversarial learning, and conditional covariance regularization. These components work collaboratively to eliminate sensitive signals both within and between feature channels while preserving semantic discriminability. Extensive experiments on multiple benchmarks show that DISH substantially outperforms state-of-the-art deep hashing baselines in retrieval accuracy while achieving better fairness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Disentangled Information Suppressed Hashing (DISH), a new framework for deep hashing in image retrieval that aims to improve both retrieval accuracy and group fairness. The authors identify that standard deep hashing methods can perpetuate biases present in training data, leading to discriminatory results. DISH addresses this by first using a disentangled encoder to separate an image's features into distinct, factor-specific channels. It then employs a multi-faceted strategy to suppress sensitive information within these channels before the features are converted into binary hash codes. This suppression mechanism includes probability-driven channel masking, channel-wise adversarial learning, and a conditional covariance regularizer to minimize information leakage about sensitive attributes (like age or gender). Finally, a semantic alignment loss ensures the resulting hash codes remain effective for retrieval. The authors conduct extensive experiments on the UTKFace and CelebA datasets, demonstrating that DISH consistently outperforms state-of-the-art deep hashing methods by achieving higher retrieval accuracy while simultaneously reducing bias across multiple fairness metrics.

### Strengths
The core idea of intervening in the continuous feature space to suppress sensitive signals before hashing is a sound and promising direction for fair retrieval.

The framework attempts a comprehensive approach to bias mitigation by targeting sensitive information both within and across different feature channels.

The paper includes an extensive set of experiments and ablation studies that validate the effectiveness of each component of the proposed framework on the tested datasets .

### Weaknesses
1. Missing Key References: The related work section overlooks some relevant prior art. For instance, "Deep Hash Distillation for Image Retrieval" (Jang et al., ECCV 2022) explores knowledge distillation for hashing, which is a relevant technique for learning compact and effective representations that is not discussed.

2. Complex and Heuristic Objective Function: The overall training objective is a complex combination of four different loss terms. This requires balancing multiple hyper-parameters that appear to be chosen heuristically. This complexity raises concerns about the method's generalizability; the carefully tuned balance may not work well for different datasets or tasks, limiting the work's practical contribution.

3. Outdated Backbone and Limited Evaluation: The experiments exclusively use a ResNet-50 backbone. While this may be necessary for fair comparison with some older methods, ResNet-50 is an outdated architecture. The evaluation should have included a much stronger, modern backbone (e.g., a Vision Transformer) to demonstrate the true potential and robustness of the DISH framework. Furthermore, the evaluation is limited to facial attribute datasets (UTKFace, CelebA). The method's effectiveness on more general natural image datasets remains unproven.

### Questions
Could you provide an explanation for the performance degradation observed when increasing the number of hash bits? For example, in Table 1, when moving from 64 to 128 bits, the MAP score for DISH decreases from 73.67 to 73.33, and the DP fairness metric worsens from 5.63 to 6.10. This is counter-intuitive, as longer hash codes are typically expected to have more capacity and lead to better, not worse, performance.

Given the complexity of the loss function, how sensitive is the model to the choice of the hyper-parameters $\lambda_{1}$ and $\lambda_{2}$? The sensitivity analysis shows a "stable ridge"5, but this seems highly dependent on the specific datasets used. Have you explored any principled methods for setting these weights beyond a simple grid search, which could improve the framework's generalizability?

### Soundness
3

### Presentation
2

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
This paper proposes an image hashing model to improve the group fairness of image retrieval. Several loss terms are proposed to regularize the model by improving the disentanglement of representation, minimizing the mutual information between the representation and sensitive attribute, and semantics alignment of hash-code. Experiments on several benchmark dataset illustrations the effectiveness of  proposed method.

### Strengths
1. Presentation. The paper is well structured with a logical flow and well-defined sections.

2. Empirical Evaluation. The experimental evaluation on benchmark datasets is comprehensive and convincing. The inclusion of detailed plots and visualizations effectively illustrates the results and provides clear insights into model behavior. The figures are well-designed and aesthetically appealing.

### Weaknesses
1. Originality and Significance. All the proposed methods seem to have been discussed in the previous work [1].
The novelty of this paper seems to be limited to changing the operator from hash-code space to continuous representation space. 

2. Reproducibility. The author doesnot include the implementation as supplementary. Will the author open source their code?

3. Inconsistent Reported Results from Previous Works. The reported empirical results of some baseline models seem to be significantly different from previous works. For example: Table 1: FATE: MAP = 70.01 ±1.27, while in [1]: Table I: MAP = 59.12
Could the author explain?

4. Unclear Math Notation. Please see Question 2, 4. 

5. Some proposed technics lack motivation. Please see Question 1, 3, 5. 

6. Some related works are not cited. For example: 

Paper related to variational information bottlenecks/variational mutual information bound/mutual information neural estimation: [2][3]

Existing effort applying variational information bottlenecks on deep hashing: [4]

[1] Zhang, Fan, et al. "FATE: Learning Effective Binary Descriptors With Group Fairness." IEEE Transactions on Image Processing 33 (2024): 3648-3661.

[2] Alemi, Alexander A., et al. "Deep variational information bottleneck." arXiv preprint arXiv:1612.00410 (2016).

[3] Belghazi, Mohamed Ishmael, et al. "Mutual information neural estimation." International conference on machine learning. PMLR, 2018.

[4] Wang, Y., Zhou, M., & Qian, X. Hashing with Uncertainty Quantification via Sampling-based Hypothesis Testing. Transactions on Machine Learning Research.

### Questions
1. Expression 188-189: "However, direct optimization is intractable due to the latent factors. Therefore, we instead optimize
the evidence lower bound (ELBO) of the log-likelihood." Why direct optimization is intractable? Why using ELBO as surrogate? 
 Could the author explain what does the variational posterior in expression (6) means ?

2. What is $\theta$? Is $\theta$ the encoder parameters? Why use $p_{\theta}(k|x_i)$ instead of $p(k|x_i)$ and $p_{\theta}(y_i|x_i, k))$ instead of $p(y_i|x_i, k))$?

3. Is $K$ a very large number? Why variational inference is needed? It seems that if $K$ is not very large, expression (6) can be computed analytically. 

4. Does the $s$ in line 197: $s_{I, k}(a)$ mean the same thing as $s$ in line 234: $s_i \in \\{1,\dots, C_s\\}$? If not, please change. 

5. Section 4.3: Channel Masking. It seems that this is just reweighing the representation based on their distance to the prototype. Why "This multiplication attenuates channels with lower pθ(k|xi), concentrating semantics into more informative factors."?

6. Which base model the author build their model upon? I ask this because in Table 4: it seems that the first row doesn't correspond to any model in table 1. Could the author explain?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the Disentangled Information Suppressed Hashing (DISH) framework to address fairness issues in deep hashing for large-scale image retrieval (e.g., retrieval bias caused by sensitive attributes such as age, gender, and race). It decomposes images into factor-specific features via a disentangled encoder, strengthens semantic focus and stability by combining disentangled consistency learning, reduces sensitive information leakage via information suppression modules (probability-driven channel masking, channel-level adversarial learning, and conditional covariance regularization), and preserves feature discriminativity via Hamming space semantic alignment. Experiments on the UTKFace and CelebA datasets show that 16-bit hash codes outperform baseline methods such as OrthoHash and FATE.

### Strengths
1. The DISH framework proposed in this paper addresses the core shortcomings of existing deep hashing methods by innovatively intervening in the continuous feature space before binarization, thereby improving the flexibility and effectiveness of sensitive information suppression.
2. The information suppression module adopts a "three-layer collaborative strategy" to comprehensively cover sensitive information leakage scenarios "within the channel - between channels - globally".
3. The paper fully demonstrates the superiority of DISH through large-scale experiments.

### Weaknesses
1. This paper primarily focuses on the "face image retrieval" scenario and does not cover non-face image domains. This "single scenario + single data type" verification model makes it challenging to demonstrate the DISH framework's adaptability to broader, large-scale image retrieval tasks.
2. The paper only designs experiments for single-dimensional sensitive attributes and does not involve complex scenarios of "multiple sensitive attributes superimposed".
3. The DISH framework introduces several complex components, which inevitably increase the number of parameters, computational cost, and training time significantly compared to traditional deep hashing methods.

### Questions
1. The paper proposes that "the unentangled encoder decomposes an image into factor-specific representations", but does not specify the actual semantics of these "factors".
2. The paper demonstrates "optimal across all components" through ablation experiments but does not analyze interactions and redundancy among components.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses fairness issues in deep hashing–based image retrieval systems, which can produce biased results across demographic groups. The authors propose DISH, a fairness-aware framework that learns disentangled and debiased hash representations in the continuous feature space before binarization. The model introduces three key modules: Disentangled Consistency Learning, which stabilizes factor-level semantics under augmentations and enforces interpretability; Information Suppressed Learning, which mitigates sensitive information leakage using probability-driven channel masking, channel-wise adversarial learning, and conditional covariance regularization; and Semantic Alignment, which maintains discriminative power in the Hamming space. Theoretically, the framework minimizes mutual information between sensitive attributes and channel representations, providing formal fairness guarantees. Empirical results on UTKFace and CelebA datasets demonstrate that DISH achieves state-of-the-art retrieval accuracy and fairness, outperforming prior methods such as FATE. Ablation studies further validate that each component contributes to improved fairness–utility trade-offs.

### Strengths
* The paper clearly identifies the under-explored issue of fairness in deep hashing–based image retrieval and motivates the need for addressing fairness before the binarization stage. The authors effectively argue that operating in the continuous feature space allows flexible and effective bias suppression.
* Extensive experiments on multiple benchmarks (UTKFace and CelebA) demonstrate clear Pareto improvements — DISH achieves higher retrieval accuracy and better fairness metrics (DP, EOP, EOD) than several baselines. The consistency across different settings adds credibility to the method’s robustness.
* Various ablation and component analysis. The paper includes detailed ablations and comparisons (e.g., different masking strategies, loss components, adversarial granularity) that clarify the contribution of each module. This level of experimental transparency is commendable and strengthens the empirical claims.

### Weaknesses
* Limited methodological novelty. The overall framework of DISH largely follows the established fair representation learning paradigm—combining disentangled representation learning with adversarial debiasing to suppress sensitive information. Similar concepts have been explored in prior works such as Adversarial Fair Representation (Zhang et al., 2018), β-VAE (Higgins et al., 2017), FactorVAE (Kim & Mnih, 2018), FFVAE (Creager, 2019) and fairness-oriented disentanglement models like Zhu et al. (2024) and Zhang et al. (2025) that explicitly separate semantic and sensitive attributes using adversarial objectives. Compared with these, the main difference of DISH is the addition of a binary hashing loss for retrieval; thus, the core contribution feels incremental rather than conceptually novel.
* Insufficient comparison to existing fair-representation methods. The paper does not thoroughly analyze how DISH compares to or improves upon well-known fair-representation frameworks such as Adversarial Fair Representation (Zhang et al., 2018) or Fair-VAE-style models. It remains unclear why adding the hashing loss directly to these existing methods would not achieve similar fairness-accuracy trade-offs. More empirical or theoretical justification for integrating disentanglement with binary hashing would strengthen the claim of methodological superiority.
* Limited exploration of training pipelines. The work assumes that fairness should be enforced jointly during hashing training, but does not evaluate a two-stage approach—e.g., learning a fair representation first and then fine-tuning with a hashing objective. Such baselines would clarify whether joint optimization is indeed necessary or simply convenient.
* High sensitivity to hyperparameters and training cost. The framework introduces at least two major hyperparameters (λ₁ and λ₂) to balance adversarial and covariance regularization terms, in addition to several architectural choices such as the number of channels K. The paper does not provide a principled method for setting these values, raising concerns about scalability, reproducibility, and tuning cost for deployment on larger or more complex datasets.

### Questions
* The proposed DISH framework integrates disentanglement, adversarial debiasing, and hashing into a single pipeline. However, several existing fair-representation methods (e.g., Adversarial Fair Representation, FairVAE, or disentanglement-based adversarial models like Zhu et al., 2024; Zhang et al., 2025) could potentially be combined with a binary hashing loss in a similar way. Could the authors clarify why  disentanglement, adversarial debiasing are better than these baselines in terms of combination of the hashing loss?

### Soundness
3

### Presentation
4

### Contribution
2
