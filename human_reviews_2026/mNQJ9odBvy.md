# SECA: Self-Guided Model Calibration

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Deep learning models frequently exhibit poor calibration, where predicted confidence scores fail to align with actual accuracy rates, undermining model reliability in safety-critical applications. We propose a novel train-time calibration method named **SECA** (**Se**lf-guided Model **Ca**libration), a **hyper-parameter-free** approach designed to improve predictive calibration through dynamic confidence regularisation. SECA constructs adaptive soft targets by fusing batch-averaged model predictions with one-hot ground-truth labels during training, thereby creating a self-adaptive calibration mechanism that adapts target distributions based on the model's predictive behaviour. This leads to well-calibrated predictions without additional hyper-parameter tuning or significant computational overhead. Our theoretical analysis elucidates SECA's underlying mechanisms from entropy regularisation, gradient dynamics, and knowledge distillation perspectives. Extensive empirical evaluation demonstrates that SECA consistently achieves superior calibration performance compared to the Cross-Entropy loss and other state-of-the-art calibration methods across diverse architectures (CNN, ViT, BERT) and benchmark datasets in visual recognition and natural language understanding.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes SECA (Self-Guided Model Calibration), a training loss function to improve neural network calibration. SECA constructs adaptive soft targets by combining one-hot ground-truth labels with batch-averaged predictions from samples of the same class. For each sample with label $y_i$, the hybrid target is defined as $\tilde{q}{i,c} = q{i,c} + \mu_{y_i,c}$, where $\mu_{y_i}$ is the average prediction across all batch samples with label $y_i$. The training loss becomes $L_{SECA} = -\sum_{c=1}^{C} \tilde{q}{i,c} \log p_{i,c}$.The method is evaluated across CNNs, Vision Transformers, and BERT on vision (CIFAR-10/100, ImageNet) and NLP tasks (DBpedia, 20 Newsgroups), showing consistent calibration improvements.

### Strengths
1. he paper evaluates across diverse architectures (ResNet, ViT, BERT) and domains (vision, NLP), which is more thorough than most calibration papers that focus primarily on CNNs. Results show consistent improvements, with particularly strong performance on CIFAR-100 (81% ECE reduction) and ImageNet (37.3% ECE reduction).
2. The paper provides three complementary views (entropy regularization, gradient dynamics, knowledge distillation) to understand SECA's mechanism. The decomposition showing SECA = Cross-Entropy + KL(μ_yi||p_i) + H(μ_yi) is particularly insightful.
3. Training times remain competitive with baseline Cross-Entropy (32.08 vs 30.83 hours on ImageNet), significantly faster than complex methods like MDCA (43.60 hours).
4. The paper is well-written with clear motivation, methodology, and comprehensive ablations in the appendix including batch size sensitivity and compatibility with post-hoc calibration.

### Weaknesses
1. The method's core assumption that each batch contains sufficient samples per class is problematic. For ImageNet (1000 classes) with typical batch sizes (256-512), some classes won't be represented in each batch. The paper doesn't explain how $\mu_j$ is computed when class $j$ has no samples in the current batch and authors provide no analysis of minimum viable batch sizes or behavior with single-sample-per-class scenarios
2. While the three perspectives are intuitive, the theoretical analysis lacksf ormal convergence guarantees or bounds on calibration improvement and rigorous characterization of when SECA improves/fails
3. Despite claiming results are averaged over 5 runs, no error bars, confidence intervals, or statistical tests are provided. Given the sometimes small differences between methods, this is concerning.
4. Missing comparison with recent 2023-2024 calibration methods.
5. No analysis of failure modes or dataset characteristics where SECA underperforms.

### Questions
1. How exactly is $\mu_j$ computed when class $j$ has no samples in the current batch? Is it set to uniform distribution, previous batch's value, or something else?
2. What is the minimum batch size required for SECA to work effectively? How does performance degrade with very small batches (e.g., 1-8 samples)?
3. How does SECA perform on severely imbalanced datasets where some classes appear very rarely during training?
4. Can you provide confidence intervals and statistical significance tests comparing SECA to baselines?
5. How are hyperparameters for baseline methods chosen? Using original paper settings may not be fair

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
3

### Summary
This paper presents a new training-based calibration method that attempts to prevent overconfidence by using soft targets that combine the one hot ground-truth class with a component based on the batch-averaged prediction for a corresponding class. By doing so, this forgoes the need for additional hyperparameters or significant computational overhead while maintaining competitive performance with alternate calibration training methods across various architectures and datasets. A theoretical analysis is provided as additional motivation for the method along three aspects, showing how soft targets act as a KL-divergence-like regularizer and that it in theory should produce more favourable gradients for balancing out excessive confidence.

### Strengths
1. Paper is overall clear and easy to follow. Authors comprehensively test three different kinds of architectures and compare with a wide variety of calibration methods, showing the effectiveness of their method. 

2. A great deal of analytical work is presented covering important real-world considerations, such as dataset shift, combination with post-hoc calibration methods, and computational efficiency which helps further motivate the method.

3. The provided theoretical analysis brings further insight and aids to ground the method.

### Weaknesses
1. Although the method does appear to be strong and theoretically justified, given the vast numbers of calibration methods already and how temperamental their performance often is depending on various factors, the main advantages of this method are that it is (mostly) parameter free and has a small training footprint, but even in the experimental results for the other methods using the ideal parameters (which is often only 1 parameter) from their original papers without tuning leads to very competitive performance (in fact the performance of many methods is very close to each other). Furthermore, it is stated they were not specifically tuned for each model, so if this was done their performance could be even better, so it is difficult to definitively say SECA is superior with these assumptions.

2. In addition, although the method is hyperparameter free in that it does not introduce new hyperparameters, it is sensitive to batch size which appears to need some optimization based on figures 6-9 in the appendix where in a few cases using higher batch size leads to worse performance (most notably figure 7 on CIFAR-10). This presents a limitation in the method and should be discussed given that it practice it might take optimization for ideal performance.

3. Class-wise calibration consistency is discussed as justification for the method in the Knowledge Distillation Perspective section but no experimental results are provided to validate this assertion. Looking at per-class ECE similar to the MDCA paper would help to validate this point.

4. Table 2 shows computational time for the different methods, but other loss based methods like focal loss that should not add much, if any, noticeable computational overhead are shown to be significantly slower than SECA and cross entropy training which makes it appear that the comparison is not fair despite claiming the same number of training steps. Can the authors comment on this and say how training times were tracked for each calibration method?

### Questions
See Weaknesses

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
This paper proposes an adaptive loss modification approach applied within each training batch. By refining the loss function across all output classes, the method aims to enhance model calibration.

### Strengths
1. The experimental section is comprehensive, and the results clearly demonstrate the method’s effectiveness in improving calibration.
2. The paper provides a multifaceted analysis of the method from entropy, gradient, and knowledge-distillation perspectives.

### Weaknesses
1. The motivation for this approach remains unclear. Neither the motivation section nor the discussion of loss-function issues sufficiently explains the rationale behind the proposed method.
2. The logical explanation of how the method improves calibration lacks clarity. It is not evident how the proposed approach ultimately enhances model calibration. Moreover, while the authors acknowledge similarities with existing methods, they do not adequately distinguish their approach from prior work.
3. The novelty of the method is not sufficiently good.

### Questions
1. What is the ultimate outcome toward which the self-guided mechanism drives the model? Specifically, is this self-guided dynamic loss function stable during optimization, and does it converge to a particular state? What are the convergence properties of this method during optimization?
2. What is the underlying motivation for introducing the self-guided mechanism? If its purpose is merely regularization or soft labeling, why is self-guidance necessary? The approach appears to introduce only an additional term during optimization, as shown in Equation (8), which resembles a dynamic soft-labeling strategy.
3. What motivates Equations (5) and (6)? What is the conceptual significance of incorporating the batch-averaged prediction into the hybrid target formulation?
4 . Why does this method not compromise accuracy?

### Soundness
2

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
3

### Summary
This paper proposes a new train-time calibration method called SECA (Self-guided Model Calibration). The method dynamically adjusts the training target distribution by fusing the batch-averaged model predictions with the ground-truth labels, thereby achieving self-adaptive confidence calibration. The authors argue that this approach can improve the alignment between predictive confidence and true accuracy without introducing additional hyperparameters. The paper further provides theoretical analysis from the perspectives of entropy regularization, gradient dynamics, and knowledge distillation. Extensive experiments across different model architectures (e.g. CNN, ViT, BERT) and tasks in vision and language domains show that SECA achieves better calibration performance compared with traditional cross-entropy and other existing calibration methods.

### Strengths
1. The paper is well-structured with a clear logical flow and appropriate level of detail.  
2. The theoretical analysis is logically coherent and easy to follow.  
3. The experiments are diverse and conducted across multiple architectures/datasets.

### Weaknesses
1. The form of $\hat{q}$ should be more precise, as there may be a normalization process that has been omitted.  
2. The derivation of Equation (12) is not accurate. The derivative of the first term should be $p_{i,c} - q_{i,c}$, but the derivative of the latter term may not simply be $-\mu_{y,c}$; thus, the equation may not hold as written.  
3. The experiments are not comprehensive and consequently not trustworthy. In comparison papers, results are often reported both before and after grid search temperature tuning, but this paper does not include such comparisons.

### Questions
Why were ResNet-32 and ResNet-56 chosen, while previous works like DualFocal[1] and AdaFocal[2] typically used ResNet-50 and ResNet-110?

Other questions are listed in the Weaknesses section.

[1] Linwei Tao, Minjing Dong, and Chang Xu. Dual focal loss for calibration. In International Conference on Machine Learning, pp. 33833–33849. PMLR, 2023.
[2] Arindam Ghosh, Thomas Schaaf, and Matthew Gormley. Adafocal: Calibration-aware adaptive focal loss. Advances in Neural Information Processing Systems, 35:1583–1595, 2022.

### Soundness
2

### Presentation
3

### Contribution
2
