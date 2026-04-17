# Robust Backdoor Removal by Reconstructing Trigger-Activated Changes in Latent Representation

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Backdoor attacks pose a critical threat to machine learning models, causing them to behave normally on clean data but misclassify poisoned data into a poisoned class. 
Existing defenses often attempt to identify and remove backdoor neurons based on Trigger-Activated Changes (TAC) which is the activation differences between clean and poisoned data. 
These methods suffer from low precision in identifying true backdoor neurons due to inaccurate estimation of TAC values.
In this work, we propose a novel backdoor removal method by accurately reconstructing TAC values in the latent representation. Specifically, we formulate the minimal perturbation that forces clean data to be classified into a specific class as a convex quadratic optimization problem, whose optimal solution serves as a surrogate for TAC. We then identify the poisoned class by statistical test based on extreme selection bias of the class with the smallest norm of perturbations, and leverage the perturbation of the poisoned class in fine-tuning to remove backdoors. 
Experiments on CIFAR-10, GTSRB, and TinyImageNet demonstrated that our approach consistently achieves superior backdoor suppression with high clean accuracy across different attack types, datasets, and architectures, outperforming existing defense methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a backdoor defense method that reconstructs Trigger-Activated Changes (TAC) in the latent representation of a poisoned model. The method formulates the reconstruction of TAC as a convex quadratic optimization problem that finds the minimal perturbation forcing all clean samples to be classified into a specific class. The poisoned class is identified by comparing perturbation norms, and then fine-tuned the model by using the corresponding perturbation to neutralize backdoor effects. Thereby achieving effective defense against backdoor attacks.

### Strengths
1. The paper formulates the reconstruction of Trigger-Activated Changes (TAC) as a quadratic convex optimization problem, offering a systematic approach to analyze backdoor effects.
2. This paper provide detailed theoretical modeling and derivation, demonstrating that the proposed defense method admits stable solutions.
3. The method is empirically compared with several recent defense techniques, showing its effectiveness and robustness in practice.

### Weaknesses
1.	The method requires solving one convex QP per class, which may become impractical for large-scale models.
2.	The approach is limited to single-target scenarios and does not address multi-target or multi-trigger backdoors.
3.	The method’s performance depends heavily on thresholds α and β, but no adaptive or learning-based tuning mechanism.
4.	Experiments are conducted only on ResNet models and image datasets, which may limit the generalizability of the results.
5.	Low poisoning rates may reduce the accuracy of detecting poisoned classes, thereby affecting the overall defense performance.

### Questions
1.	A low poisoning rate may increase the minimum perturbation required to misclassify clean samples into poisoned classes, which could affect the selection of poisoned categories and ultimately influence the overall defense results. Experiments with varying poisoning rates could be added to demonstrate that the proposed method remains effective even in low poisoning rates.
2.	Manually tuning α and β, where α controls poisoned class identification and β balances backdoor defense with task accuracy, is time-consuming and often leads to unstable or suboptimal performance. I wonder if it is possible to use an adaptive strategy to make the process more efficient and reliable

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
5

### Summary
This paper proposes a novel backdoor removal framework that reconstructs Trigger-Activated Changes (TAC) in the latent representation of neural networks to achieve robust backdoor defense. The method computes minimal perturbations that force a model to misclassify clean data into each class, identifies the poisoned class through statistical outlier detection in the L2-norm of these perturbations, and then fine-tunes the model using the perturbation corresponding to the identified poisoned class. Experiments on CIFAR-10, GTSRB, and TinyImageNet demonstrate improved defense performance compared to state-of-the-art methods while maintaining high clean accuracy.

### Strengths
- The idea of reconstructing TAC in the latent representation through convex quadratic optimization offers a neat and interpretable surrogate approach that does not rely on poisoned data. This reformulation is novel and mathematically well-grounded.
- The mathematical explanation is solid and convincing, although it is also not easy to understand.
- The empirical evidence of using the smallest-perturbed class is clear and convincing.
- The experiments are solid with a comprehensive comparison with the baselines. And leave nearly no improvement for future research.

### Weaknesses
- There is a lack of clear outlines for the appendix content, making it hard to find the remaining experiments and the desired explanations.
- Solving multiple convex programs per class may be nontrivial for large-scale models (e.g., high-dimensional latent spaces or hundreds of classes). No analysis of time or resource overhead is given.
- The extensive experiments related to the scalability are needed to further verify the effectiveness of the proposed method. For example, the experiments on a larger model (e.g., ViT) and a more complex dataset (e.g., ImageNet). The current results (e.g., Table 2) show that the SOTA baseline (e.g., SAU) already performs good enough, weakening your contribution in this field.

### Questions
Can you provide more evidence from a bigger scale (e.g., weakness 3 above) to show the superiority of your method? Or can you provide some intuitive explanations to show that we need your contribution for the community? It can be either insights (e.g., how reconstructed TAC contributes to future research) or empirical results (e.g., how your method solves the corner cases that are previously unsolved).

### Soundness
4

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
5

### Summary
This paper proposes a novel backdoor defense framework that removes backdoors from neural networks by reconstructing Trigger-Activated Changes (TAC), the differences in neuron activations between clean and poisoned data, without needing poisoned samples. The TAC reconstruction is performed by computing a minimal perturbation for each class.

### Strengths
- Extensive experiments on multiple datasets and attacks demonstrate better or comparable performance over prior methods.

- The presentation of the paper is easy to follow.

- The motivation is clear, and the proposed method addresses an important problem.

### Weaknesses
- The experiments are primarily on ResNet-18.

- The method assumes one poisoned class, which may limit performance in multi-target or all-to-all attacks.

- Experiments do not include large datasets, such as ImageNet-1K.

- The performance is not significantly better than all baselines, such as FT-SAM.

### Questions
Thanks for the interesting work. I have a few questions and suggestions.

- Computational overhead. As the proposed method requires computing "minimal perturbation" for every class. What is the computational cost of this method?

- Why not directly remove the high-TAC neurons? If the proposed TAC reconstructing method is effective, removing the high-TAC neurons should also work. In addition, the authors could also provide some figures to demonstrate the reconstructed TAC values, like Figure 2 in [A].

- Transformer-based architectures, such as ViT. I suggest the authors include experiments on more architectures, such as ViT.

[A] Towards Backdoor Stealthiness in Model Parameter Space

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an approximate TAC-based method for backdoor removal and defense. In particualr, in the feature space, perturbations are optimized to force clean data to be classified into a specific class. Generated perturbations are used to distinguish between benign and backdoor samples and then utilized in fine-tuning to remove backdoors. This defense method is inspired by TAC, while it is also closely related to feature space backdoor defenses.

### Strengths
This method uses clean data to generate the perturbations, making it suitable for realistic defender settings where poisoned data are unavailable. Later, perturbations can be used for both detection and removal.

### Weaknesses
Comparison with feature-space defenses. While the paper is inspired by Trigger-Activated Changes (TAC), its practical implementation closely resembles feature-space backdoor defenses[a]. However, the paper provides limited comparative analysis with these prior methods. A deeper comparison would strengthen the contribution and clarify the novelty.

Adaptive evaluation. The work does not evaluate the defense under adaptive or defense-aware backdoor attacks. Since the proposed method depends on the assumption that poisoned-class perturbations exhibit smaller L2 norms, an attacker aware of this could manipulate with this regard. Testing against attacks that minimize perturbation norms would provide more substantial evidence of robustness.

Hyperparameters. The defense relies on several dataset-specific hyperparameters, such as the outlier threshold. The paper gives limited guidance on how these parameters generalize across datasets or model architectures. In addition, reproducibility could be improved by reporting computational cost and sensitivity analyses.

[a]Towards Stable Backdoor Purification through Feature Shift Tuning. NeurIPS 2023.

### Questions
Compare with feature space defenses, discuss adaptive attacks, discuss the generalization w.r.t hyperparameters

### Soundness
2

### Presentation
2

### Contribution
2
