# SORA: Free Second Order Attacks in Fast Adversarial Training

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Adversarial Training (AT) is a leading defense against adversarial examples but often suffers from *Catastrophic Overfitting* (CO) in efficient single-step variants, where robustness to multi-step attacks collapses despite high single-step performance. 
    We address this failure mode with two contributions. 
    First, we identify *Epsilon Overfitting* (EO), a previously overlooked phenomenon in which fixed perturbation magnitudes exacerbate CO, and show that introducing perturbation variability significantly improves robust generalization across different architectures and datasets. 
    Second, we propose **PertAlign** (Perturbation Alignment), a theoretically grounded, computationally negligible metric that predicts CO onset by measuring gradient alignment across attack stages. 
    Leveraging these insights, we introduce **SORA**, an adaptive step-size adversarial training method that dynamically adjusts perturbations based on loss-surface geometry. 
    SORA consistently prevents CO, achieves state-of-the-art robustness and clean accuracy, and generalizes across datasets and architectures using a single fixed set of hyperparameters.
    Extensive experiments on diverse datasets and architectures, show that SORA matches or surpasses the robustness of prior methods while delivering higher clean accuracy and superior efficiency.
    Code is available at [https://anonymous.4open.science/r/2026_ICLR_SORA](https://anonymous.4open.science/r/2026_ICLR_SORA).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose an efficient metric to measure the non-linearity of loss landscape during adversarial training. By utilizing this metric, the optimal step size for adversarial attack can be calculated. The experiments on diverse datasets and architectures demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper is written well
2. The method is computationally efficient 
3. The theoretical analysis on PerAlign, which is to measure the non-linearity of the loss landscape, is rigorous

### Weaknesses
1. **Marginal improvement:** On CIFAR10 and CIFAR100, SORA only has slight improvement compared to NFGSM and AAER. What are the advantages of your method compared to them?
2. **Abnormal baseline performance:** The performance of GradAlign, ZeroGrad, ATAS, AAER is surprisingly low on PathMNIST and TissueMNIST. It is well known that AT is sensitive to hyperparameters. Did you tune their hyperparameters to ensure the optimal performance on these datasets?
3. **Some baselines are missing:** The results of critical baselines, e.g., ATTA [1], Fast-BAT [2], NuAT [3], are missing
4. **Lack of results on high-resolution datasets:** You only have the results on low-resolution datasets. It is still unknown whether your method can be scaled up to high-resolution datasets, e.g., ImageNet-100.

[1] Haizhong Zheng et al. Efficient adversarial training with transferable adversarial examples

[2] Yihua Zhang et al. Revisiting and advancing fast adversarial training through the lens of bi-level optimization.

[3] Gaurang Sriramanan et al. Towards efficient and effective adversarial training.

### Questions
See weaknesses

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
This paper addresses the problem of Catastrophic Overfitting (CO) in single-step adversarial training (AT). The authors identify a phenomenon termed "Epsilon Overfitting" (EO), where models overfit to the specific perturbation magnitude used during training. They propose Perturbation Alignment (PertAlign) to predict the onset of CO, and leverage these insights to develop SORA, an adaptive step-size AT method. SORA dynamically adjusts perturbation sizes based on an approximation of the local loss curvature. Extensive experiments across multiple datasets and architectures show that SORA can prevent CO and achieve competitive robust accuracy with minimal computational overhead.

### Strengths
1.	Comprehensive Evaluation: The paper provides a thorough empirical evaluation across a diverse set of datasets (including challenging medical imaging benchmarks) and model architectures. 
2.	Identification of Epsilon Overfitting (EO): The observation that fixed, large perturbation magnitudes can lead to overfitting on specific ε values is an interesting and well-documented analysis. 
3.	Low-Cost Metric: The proposed PertAlign metric is computationally cheap, as it reuses gradients already computed during the standard training process. 
4.	Practical Algorithm: The SORA algorithm itself is relatively simple to implement and integrates seamlessly into existing fast AT pipelines.

### Weaknesses
1.	Significance of the Core Problem: A fundamental question remains: How critical is Catastrophic Overfitting in the broader landscape of adversarial robustness? While CO is a known failure mode in single-step AT, the paper does not sufficiently articulate why CO remains a pressing, unsolved issue that warrants a new solution, especially given that multi-step methods like PGD-10 are still considered the gold standard for high robustness, albeit at a higher cost.
2.	Novelty and Depth of Contributions: 1) Epsilon Overfitting (EO): While the term is new, the underlying idea—that fixed-step attacks can lead to non-robust, overfitted decision boundaries—is a core intuition behind many existing adaptive and multi-step methods. The claim that EO is a "previously overlooked phenomenon" may be an overstatement; it is more accurately a new and specific characterization of a known class of problems. 2) Perturbation Alignment (PertAlign): The theoretical derivation connects PertAlign to the Hessian, which is a solid contribution. However, from a practical standpoint, PertAlign can be perceived as a relatively trivial trick: it is essentially the cosine similarity between the gradient at a random start and the gradient after one FGSM step. What is it essentially different from GradAlign?
3.	Marginal Practical Gains: The experimental results show that SORA's advantages are often marginal. In many tables, SORA's robust accuracy is within less than ~1% of the best-performing single-step baselines (e.g., NFGSM, AAER). While it achieves better clean accuracy on some datasets like PathMNIST, the overall improvement in the trade-off between robustness and accuracy is not dramatic. Hence, the net practical benefit of adopting SORA is not overwhelmingly compelling.
4.	Theoretical Grounding Justification: The theoretical analysis in Appendix A is technically sound, deriving the optimal step size under a quadratic loss assumption. However, the justification for this approximation in the context of deep neural networks is weak. The highly non-convex and complex loss landscapes of modern DNNs are far from quadratic, and the paper does not provide evidence that this local approximation holds well enough in practice to be truly meaningful, beyond serving as a heuristic inspiration for the algorithm.
5.	Comparison to Competitors: The paper positions SORA as a state-of-the-art method. However, when compared to strong baselines like PGD-10 or TRADES, SORA consistently lags in robust accuracy (as expected, given their higher computational cost). Among its single-step peers, it is a strong contender but not a clear dominator. The claim of "superior efficiency" is true relative to multi-step methods, but its advantage in time/memory over other single-step methods (see Figure 6) is minimal, and its performance gain is similarly slight.

### Questions
See the weaknesses.

1.	Beyond preventing CO, what specific, significant advantage does SORA offer over other CO-robust single-step methods, given that the accuracy improvements are often marginal?
2.	Can you provide empirical evidence validating the quadratic loss assumption in deep networks? How sensitive is SORA's performance to scenarios where this assumption breaks down?
3.	The concept of adapting to local curvature is not new. How is SORA's approach fundamentally different or better than prior adaptive step-size methods like ATAS, beyond the specific heuristic used?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces and analyze Epsilon Overfitting, demonstrating its importance for robust generalization and CO. The authors propose PertAlign, a efficient metric for early and reliable CO prediction. The authors verify the effectiveness of their method on different methods and datasets.

### Strengths
1.	The motivation and method are clear, and the visualization is helpful.
2.	The authors provide sufficient evidence, including extensive experiments and ablation studies, to support the effectiveness of their proposed method. 
3.	This paper is well-written, making it easy to follow.

### Weaknesses
1.	Misleading or Overstated Claims  
(1) The fact that fixed perturbation magnitudes exacerbate CO has been thoroughly studied in prior work [1,2]. It is already well-known that larger perturbations easily lead to CO [3,4]. This is not a “previously overlooked phenomenon” as the authors claim.  
(2) The paper argues that existing measures incur a high computational cost. But [4, 5] already provides a time-efficient CO indicator without extra backwards.
2.	Weak Novelty Foundation  
(1)	The “epsilon overfitting” part merely repeats existing research on decision boundary distortion. Many works already explain and measure distorted boundaries. The more important insight is to explain: how does boundary distortion dynamically during training, and what causes the spontaneous and initial onset of distortion.  
(2)	PertAlign is fundamentally the same as GradAlign, with only a different radius for landscape measurement. Moreover, various efficient landscape-measurement techniques already exist [5]. 
3.	Effectiveness of the Method  
(1)	Most machine learning method requires hyperparameter searching per dataset and architecture. Using “universal hyperparameters” is beneficial but not an excuse to deny baseline performance.  The authors should ensure fair comparisons by tuning baselines on different dataset and architecture, such as SENet and PathMNIST.  
(2)	The method reduces CO by dynamically decreasing step-size, which raises the concern that it simply converges to a trivial solution where the perturbation strength is too small to trigger CO. To rule out this trivial solution, the authors must verify performance under more challenging perturbation budgets (e.g., 32/255), since N-FGSM maintains robustness at 16/255.

[1] Understanding Catastrophic Overfitting in Single-step Adversarial Training  
[2] Fast Adversarial Training with Adaptive Step Size  
[3] Fast is better than free: Revisiting adversarial training  
[4] Eliminating Catastrophic Overfitting Via Abnormal Adversarial Examples Regularization  
[5] Efficient local linearity regularization to overcome catastrophic overfitting

### Questions
Refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates the catastrophic overfitting issue in accelerated adversarial training and propose a metric called PertAlign to predict  CO in advance. PertAlign is based on the second order analyses on the input loss landscape and can be used to adaptive choose the step size when applying FGSM in adversarial training. The authors provide both theoretical motivations and the experimental results to validate the method (SORA) proposed.

### Strengths
++ PertAlign is theoretically motivated and is a low-cost indicator predicting CO in advance.

++ The method is simple and in a plug-and-play manner.

++ The experiments are relatively comprehensive on various models and datasets. The time / memory scatter shows SORA in competitive among fast adversarial training methods.

### Weaknesses
I have the following concerns about this paper:

1. Limited novelty: the relationship between the craggy loss landscape and catastrophic overfitting is actually not new. 

2. The motivation of PertAlign is based on the second order approximation (Lemma 4.1). However, the step size $\alpha$ is generally quite large in one-step adversarial training, sometimes it can even be larger than $\epsilon$, I am not sure if the higher-order terms in the proof of Lemma 4.1 can be ignored. If not ignored then the approximation will not be accurate.

3. While the experiments are comprehensive, the performance improvement on AA, the most reliable evaluation, is very marginal (From Table 7 to 24). In most tables, the performance difference between SORA and the strongest baseline among fast adversarial training methods, is very small and smaller than the performance variance.

4. The experiments focus on the $l_\infty$ attacks and image classification problems, it would be better to include $l_2$ or $l_1$ perturbations and other tasks. In addition, the SORA pseudo-code seems to consider $l_\infty$ perturbations only, is it applicable to $l_2$ or $l_1$ cases? If not, what modifications do we need to make?

5. In addition to Table 3, more ablation studies are expected, such as sweeping the values of new hyper-parameters ($\alpha_max$, $\alpha_0$, $\beta$ in SORA's pseudo-code)

Minor issues:

1. Some missing literature:

    * About second-order curvature in adversarial training: "Robustness via Curvature Regularization" (CVPR 2019)
    * About fast adversarial training or geometry: "YOPO: You only propagate once" (NeurIPS 19), 

2. In Table 5's caption: "The values in each cell correspond to clean, FGSM and PGD-10 accuracies." while I see only two results per cell.

### Questions
Please address the questions in the weakness section:

1. What is the key difference between PertAlign and existing work? (incl. the missing literature mentioned) If using second-order approximation is the key, then it would be better to demonstrate how tight the second order approximation is in the input loss landscape.

2. Why the improvement is marginal? What is the practical advantages of the proposed method?

3. The method's performance on tasks other than image classification.

4. The adaptation of the proposed methods to $l_2$ and $l_1$ cases and the corresponding experimental results.

5. More ablation studies on the newly introduced hyper-parameters.

### Soundness
3

### Presentation
3

### Contribution
2
