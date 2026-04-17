# Inconsistency-Aware Minimization: Improving Generalization with Unlabeled Data

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Estimating the generalization gap and developing optimization methods that improve generalization are crucial for deep learning models, for both theoretical understanding and practical applications.
Leveraging unlabeled data for these purposes offers significant advantages in real-world scenarios.
This paper introduces a novel generalization measure, $\textit{local inconsistency}$, derived from an information-geometric perspective on the parameter space of the neural networks. A key feature of local inconsistency is that it can be computed without explicit labels.
We establish theoretical underpinnings by connecting local inconsistency to Fisher information matrix and loss Hessian.
Empirically, we demonstrate that local inconsistency correlates with the generalization gap.
Based on these findings, we propose Inconsistency-Aware Minimization (IAM), which incorporates local inconsistency into the training objective.
We demonstrate that in standard supervised learning settings, IAM enhances generalization, achieving performance comparable to that of existing methods such as Sharpness-Aware Minimization.
Furthermore, IAM exhibits efficacy in semi- and self-supervised learning scenarios, where the local inconsistency is computed from unlabeled data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper defines local inconsistency as the change in a model’s output distribution under small parameter perturbations, shows it can be computed with a single trained model and unlabeled data, and links it to the Fisher Information Matrix and loss curvature. It introduces Inconsistency-Aware Minimization (IAM) in two variants (IAM-D, IAM-S) and provides a generalization bound that replaces a Hessian term with the top FIM eigenvalue under near-interpolation (Theorem 4.1). Experiments show that the inconsistency measure correlates with generalization (Figure 1), IAM improves over SGD/SAM/ASAM on vision benchmarks (Table 1), gives ImageNet gains with ResNet-50 (Table 2), and benefits semi/self-supervised setups including FixMatch and SimCLR (Table 3, Figure 3).

### Strengths
1. The local inconsistency can be computed from a single trained model using only unlabeled data. It is directly differentiable, enabling optimization within standard training pipelines.
2. The paper establish a generalization bound (Theorem 4.1), validating the single-model approximation theoretically
3. Algorithm 1 estimates the local inconsistency using power iteration updates without explicit eigen decomposition, achieving linear complexity versus quadratic. This makes the method computationally practical for large networks.
4. The method shows consistent improvements across supervised learning (Tables 1,2), semi-supervised with limited labels (Table 3), and self-supervised SimCLR (Figure 3). Success across multiple paradigms suggests genuine generality.

### Weaknesses
1. Theorem 4.1 assumes near-interpolation and many other assumptions. The bound validity remains unclear if these assumptions are violated. The paper didn't illustrate why the assumptions are reasonable, and also didn't show whether the assumptions are valid in practical training.
2. Algorithm 1 use K=1 for efficiency, but the paper provides no empirical verification on why K=1 is enough.
3. In Table 2, ImageNet results compare only against SGD baseline with no SAM or ASAM comparison, despite SAM being the primary supervised-learning baseline on CIFAR datasets (Table 1)

### Questions
1. Table 2 shows IAM-S results for ResNet-50 at 100 and 200 epochs but lists "-" for 400 epochs with no explanation—did training fail, diverge, or was it simply not run?
2. What about the computational efficiency compared to baselines?
3. All results are on CNN type model. Could you test the transformers like ViT?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a model intrinsic measure dubbed local inconsistency, which can be used to predict generalization gap and regularize model training. This local inconsistency measure is grounded in information-geometry and is closely related to Fisher information matrix, Hessian matrix, and Gauss-Newton matrix. Moreover, the paper presents an algorithm to estimate local inconsistency empirically, based on which inconsistency-aware minimization (IAM) is proposed to train DNNs with improved generalization. Experimental results on image classification are provided under supervised, semi-supervised and self-supervised settings to validate the effectiveness of IAM.

### Strengths
1. The proposed measure has solid theoretic grounding.
2. The proposed measure can be estimated without labels.
3. The writing is clear and the proposed method is well positioned among related works.

### Weaknesses
1. In Table 1, results for both IAM-D and IAM-S are presented, where IAM-S gives better performance in general. In Table 2, only IAM-S is presented. However, for semi and self-supervised settings, only results for IAM-D are provided. What's the reason of this inconsistency? Is it because IAM-S works better in supervised learning while IAM-D works better in semi and self-supervised settings? If this is the case, then more analysis is needed to show why this happens and provide practical guidance on which variant of IAM should be preferred in which setting.
2. The number/type of model architectures used in experiments is quite limited. Experiments on more models, especially transformer-based models like ViT, are needed to show the generalizability of the proposed method.
3. How good is the convergence of Algorithm 1? In experiments, K is typically selected to be 1. Will more steps lead to better results? An ablation study on K could help.
4. The statement in Line 233-234 assumes there're only C nonzero eigenvalues. This assumption needs to be justified, for example, with some empirical results.
5. In Algorithm 2, is using the current minibatch good enough for computing $\delta_K$? Why not sample another class-balanced subset for computing $\delta_K$?
6. In Table 2, the performance of SGD with 400 epochs is worse than that of 200 epochs, which indicates overfitting. It's better to report results based on the best performing epoch (early stopping).
7. In Table 3, it's better to include FixMatch+SAM for comparison.
8. For self-supervised learning, it's better to provide linear probing results on CIFAR-100 as well.

### Questions
1. In Eq. 3, why is the constriant over $\delta$ based on $L_2$ norm but not $L_{\infty}$ norm which is more popular in adversarial training?

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
This paper introduces a new generalization measure called local inconsistency, which quantifies a model's output divergence with respect to parameter perturbations. Theoretically, the measure is shown to be governed by the Fisher Information Matrix, providing a robust, complementary signal to traditional sharpness. Based on this, the authors propose Inconsistency-Aware Minimization (IAM), an optimization framework that achieves generalization improvements comparable to SAM while showing performance improvement in unlabeled-data regimes like FixMatch and SimCLR.

### Strengths
1. The paper is clearly written and easy to follow.
2. Although the core idea is similar to SAM, it is still interesting to explore replacing the minimax loss with KL-divergence.
3. The paper provides some interesting theoretical insights.

### Weaknesses
1. Although the proposed method is a reasonable attempt at introducing a new measure, yet it gives little to no advantage over existing SAM and its variants.
2. I recommend that the authors consider incorporating FisherSAM [1], where $\rho$ is constrained within the KL region. This might lead to additional interesting results.

    [1] Fisher SAM: Information Geometry and Sharpness Aware Minimisation

3. Using gradient-based optimization to solve the inner maximization problem is valid and is similar to [2]. However, it could incur too much extra computation. 

    [2] Regularizing neural networks via adversarial model perturbation

4. Based on my understanding and experience, directly optimizing KL-divergence may introduce difficulties, particularly due to its unbounded nature and potential numerical instability.

5. Does the method use the batch data in Algorithm 1? For supervised learning, I see no clear reason not using batch data.

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
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces "local inconsistency", a novel generalization measure for deep learning models that can be computed using only unlabeled data. This measure quantifies the model's output sensitivity to parameter perturbations. Based on this, the authors propose Inconsistency-Aware Minimization (IAM), an optimization method that incorporates this measure into the training objective. The paper demonstrates that IAM improves generalization in supervised learning (comparable to Sharpness-Aware Minimization, SAM) and, more importantly, enhances performance in semi-supervised and self-supervised learning scenarios where labeled data is scarce.

### Strengths
- The paper introduces $S_{\rho}$,  a novel measure of local output sensitivity. Its strength is that it can be computed from a single model using only unlabeled data.
- Due to its label-agnostic nature, it can be effectively applied to semi-supervised learning and self-supervised learning. This is a clear advantage over methods like SAM that depend on a labeled loss.
- Unlike other inconsistency measures that require training multiple models, "local inconsistency" can be computed from a single model. 
- The proposed measure is theoretically grounded, with clear connections established to the Fisher Information Matrix (FIM) and the loss Hessian (Section 4.2), providing a solid information-geometric motivation.

### Weaknesses
- Similar to SAM, IAM is a min-max optimization algorithm. It requires an inner maximization step (Algorithm 1) to find the perturbation, which, even with K=1 step, effectively doubles the computational cost (requiring two gradient computations) per training step compared to standard SGD.
- For computational efficiency, the inner maximization step is approximated using only one step (K=1) of gradient ascent. While Appendix C argues this is meaningful, it is still a strong approximation and may not find the true worst-case perturbation.
- The method introduces new, sensitive hyperparameters, most notably $\rho$ and $\beta$. As shown in the appendix heatmap, these values are dataset-dependent and have a significant impact on performance.

### Questions
- The K=1 step for the inner maximization is justified for efficiency. Have you performed an ablation study on the number of steps K (e.g., comparing K=1 vs. K=3, as mentioned in Appendix E)? This would clarify if a more accurate estimation of the worst-case perturbation (K > 1) actually leads to better generalization, or if the K=1 approximation is sufficient.
- The hyperparameters $\rho$ and $\beta$ seem critical and dataset-dependent. Could the authors provide more intuition or a more principled guideline for setting these values beyond grid search?
- The paper proposes two variants: IAM-D and IAM-S. The results in Table 1 seem mixed (IAM-S is better on CIFAR-100, while IAM-D is slightly better on SVHN). Are there recommendations for when to use one versus the other? For instance, are there specific scenarios where IAM-D might be preferred over IAM-S?

minor
- (Typos) Page 17 and Page 19: pertubation -> perturbation

### Soundness
4

### Presentation
3

### Contribution
3
