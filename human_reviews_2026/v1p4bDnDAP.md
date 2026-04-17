# Two Birds with One Stone: Neural Tangent Kernel for Efficient and Robust Gradual Domain Adaptation

- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Gradual Domain Adaptation (GDA) bridges large distribution shifts through intermediate domains, yet faces challenges in computational overhead and error accumulation. In view of these problems, we propose GradNTK, a novel framework to employ the Neural Tangent Kernel (NTK) as one stone to "hit" two birds of the efficiency and robust issues in GDA. 
On one hand, by exploiting the short-time dynamics of wide neural networks, GradNTK instantiates an NTK-induced Maximum Mean Discrepancy (MMD) as a differentiable domain-alignment metric that enforces smooth transitions between adjacent domains while maintaining near-linear computational cost. 
On the other hand, the same NTK dynamics generate a prospective utility function to weight source/target samples by their shift sensitivity, enabling curriculum-guided gradual adaptation while avoiding error accumulation.
Experiments on Portraits, Rotated MNIST and CIFAR-100-C demonstrate superior performance (e.g., 95.1\% on Rotated MNIST, 99.5\% on Color-Shift MNIST), while reducing training time by 1.8× compared to prior GDA methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses a general and important problem in machine learning, namely gradual domain adaptation (GDA). The authors propose GradNTK, a new framework that employs the Neural Tangent Kernel (NTK) to mitigate computational overhead and error accumulation in GDA. Specifically, the framework introduces an NTK-MMD loss and a sample reweighting function to facilitate domain transition. While the paper is clearly written and easy to follow, the overall novelty remains limited because both NTK-based matching and sample reweighting strategies are well-studied in the broader domain adaptation literature.

### Strengths
-	The paper is well-organized and the presentation is clear..
-	The proposed GradNTK framework integrates NTK-MMD loss and sample reweighting, and the experiments show some degree of effectiveness in GDA scenarios.

### Weaknesses
-	Both neural kernel methods [1–4] and sample reweighting techniques [5–8] have been extensively explored in domain adaptation. Applying them to the gradual domain adaptation setting is a straightforward extension and does not provide sufficient novelty for ICLR.
-	The overall experiment design is not sufficient to verify the effectiveness. More diverse datasets and larger backbones are preferred.
-	The comparison in Table 4 appears to be unfair. Test-time adaptation (TTA) methods only require the source model, whereas the proposed framework requires access to source data at each adaptation stage. Moreover, many recent TTA methods [9-12] achieve better performance with less information and lower computational cost. As a result, the empirical advantage of the proposed framework remains unclear.

**References:**

[1] Jacot, Arthur, Franck Gabriel, and Clément Hongler. "Neural tangent kernel: Convergence and generalization in neural networks." Advances in neural information processing systems 31 (2018).

[2] Jia, Sheng, et al. "Efficient statistical tests: A neural tangent kernel approach." International Conference on Machine Learning. PMLR, 2021.

[3] Cheng, Xiuyuan, and Yao Xie. "Neural tangent kernel maximum mean discrepancy." Advances in Neural Information Processing Systems 34 (2021): 6658-6670.

[4] Shimizu, Eiki, Kenji Fukumizu, and Dino Sejdinovic. "Neural-kernel conditional mean embeddings." arXiv preprint arXiv:2403.10859 (2024).

[5] Tachet des Combes, Remi, et al. "Domain adaptation with conditional distribution matching and generalized label shift." Advances in Neural Information Processing Systems 33 (2020): 19276-19289.

[6] Guo, Zong, et al. "Gradual domain adaptation with sample transferability exploitation for person re-identification." 2022 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2022.

[7] Ru, Jinghan, et al. "Imbalanced open set domain adaptation via moving-threshold estimation and gradual alignment." IEEE Transactions on Multimedia 26 (2023): 2504-2514.

[8] Chen, Hong-You, and Wei-Lun Chao. "Gradual domain adaptation without indexed intermediate domains." Advances in neural information processing systems 34 (2021): 8201-8214.

[9] Wang, Qin, et al. "Continual test-time domain adaptation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[10] Döbler, Mario, Robert A. Marsden, and Bin Yang. "Robust mean teacher for continual and gradual test-time adaptation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[11] Press, Ori, et al. "Rdumb: A simple approach that questions our progress in continual test-time adaptation." Advances in Neural Information Processing Systems 36 (2023): 39915-39935.

[12] Song, Junha, et al. "Ecotta: Memory-efficient continual test-time adaptation via self-distilled regularization." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Questions
Please refer to the Weakness section.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces GradNTK, which integrates the Neural Tangent Kernel (NTK) into Gradual Domain Adaptation (GDA). It uses NTK in two roles:
1. An NTK-induced Maximum Mean Discrepancy (MMD) is employed as a differentiable alignment loss between adjacent domains, aiming to improve efficiency and smoothness.
2. An NTK-based reweighting function is used to assign sample weights based on “shift sensitivity,” intending to mitigate error accumulation during gradual adaptation. 
 
Experiments on Rotated MNIST, Color-Shift MNIST, Portraits, CIFAR-10-C, and CIFAR-100-C show the method’s effectiveness for GDA.

### Strengths
1. The derivation connecting NTK linearization, witness functions, and MMD appears to be pedagogically clear.
2. Using NTK for both alignment and reweighting appears conceptually elegant.
3. Replacing the traditional MMD with NTK-based short-time dynamics reduces memory overhead.

### Weaknesses
1. Most technical ingredients—NTK linearization, MMD witness formulation, NTK-MMD, and pseudo-labeling—are well-established concepts. The contribution seems to be primarily a straightforward combination.
2. Large portions of Section 3 appear to reiterate textbook material, which may obscure the core novelty. It would be advisable to condense these derivations and clearly separate known results from new contributions to make the key insight stand out.
3. As shown in Table 5, removing the NTK-reweighting module causes almost no drop in accuracy, suggesting that this component contributes little to the claimed “robustness.”
4. Beyond TENT, several test-time adaptation (TTA) methods such as CoTTA [a] and RMT [b] also address gradual and continual adaptation scenarios, but without access to source data. This makes TTA a more challenging setting compared to GDA. It would be valuable to clarify the conceptual and practical position of GDA relative to these recent TTA frameworks—specifically, what assumptions GDA relaxes or strengthens, and in what scenarios it is preferable. Including such discussion and comparisons would provide a more informative contextualization of the proposed method.
5. All datasets used involve relatively smooth or synthetic domain shifts. It is unclear whether GradNTK would maintain its effectiveness under more realistic domain gaps.

[a] Continual Test-Time Domain Adaptation. CVPR2022. \
[b] Robust Mean Teacher for Continual and Gradual Test-Time Adaptation. CVPR2023.

**Minor comments**
- In Figure 1, “(i) GDA” should be “(i) UDA.”
- Line 51: citation formatting error.

### Questions
Additional questions beyond those in Weaknesses:

1. Can the authors clarify whether GradNTK requires explicit intermediate domain labels, or could it operate in a fully online streaming manner (closer to TTA)?
2. Does the NTK reweighting function actually change sample selection over time, or does it remain nearly uniform (or static) in practice?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work studies the gradual domain adaptation problem, where the gap between the source dataset and target dataset could be tracked by the continuous intermediate domains between them. To address the computation cost and error accumulation in existing methods, this work proposes the Neural NTK method as an efficient distance estimation and develops an NTK-based weight to reweigh the loss in risk estimation. Experiment results show that the proposed method achieves promising performance compared with advanced methods.

### Strengths
+ The motivation of improving distribution discrepancy estimation and gradual error propagation is reasonable and meaningful.

+ The developed method with the Neural NTK estimator and weight is technically sound.

+ The empirical performance is significant compared to the advanced methods, and the empirical analysis is consistent with the theoretical results.

### Weaknesses
+ The validity of the proposed method in empirical scenarios needs further clarification, i.e., convergence of approximation, metric property and guarantees of reweighing objective.

+ The math rigor could be improved, e.g., some notations are unclear.

### Questions
Q1. Despite the requirement of infinite width of NTK for good approximation, it is also uncertain that the constructed parameterized MMD via $\Theta$ still ensures metric property. Specifically, since the MMD in Eq. (12) is restricted to the function space parameterized by $\Theta$, could it still measure the distance between distributions? (recall that the metric property of kernel MMD is only satisfied by specific kernels like Gaussian). 

Q2. The weights constructed in Eq. (15) need further justification. Since the weight is used in the risk estimation in Eq. (22), which seems to be similar to the common importance reweighting strategy, it would be important to show some theoretical results that such a weight could benefit the model learning, e.g., reduce the bias of risk estimation.

Q3. How to understand the Neural NTK for gradual discrepancy estimation in practical scenarios with finite width. Are there quantitative results that could control the error? 

Q4. Recall that there are actually typical covariate shift methods that also consider the importance reweighting technique to reduce the gap between source risk and target risk. It would be interesting to demonstrate the significance of the proposed method compared with these methods, i.e., the difference of weight construction. 

Q5. What is the definition of $\pi_\mathbb{X}$ in Line 173? In my understanding, $Z$ is the push-forward distribution under $r_\psi$, then $\pi_\mathbb{X}$ seems to be redundant.

Q6. What is $\Delta_\mu$ in Eq. (13)? Does it imply the difference of mean values of source and target distributions? i.e., similarly defined as $g$ that is the difference of the Neural NTKs $f_T$ and $f_0$.

### Soundness
3

### Presentation
3

### Contribution
3
