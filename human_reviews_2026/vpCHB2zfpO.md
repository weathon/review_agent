# S2AP: Score-space Sharpness Minimization for Adversarial Pruning

- Decision: Reject
- Scores: 4, 4, 4, 0

## Abstract
Adversarial pruning methods have emerged as a powerful tool for compressing neural networks while preserving robustness against adversarial attacks. These methods typically follow a three-step pipeline: (i) pretrain a robust model, (ii) select a binary mask for weight pruning, and (iii) finetune the pruned model. To select the binary mask, these methods minimize a robust loss by assigning an importance score to each weight, and then keep the weights with the highest scores. However, this score-space optimization can lead to sharp local minima in the robust loss landscape and, in turn, to an unstable mask selection, reducing the robustness of adversarial pruning methods. To overcome this issue, we propose a novel plug-in method for adversarial pruning, termed Score-space Sharpness-aware Adversarial Pruning (S2AP). Through our method, we introduce the concept of score-space sharpness minimization, which operates during the mask search by perturbing importance scores and minimizing the corresponding robust loss. Extensive experiments across various datasets, models, and sparsity levels demonstrate that S2AP effectively minimizes sharpness in score space, stabilizing the mask selection, and ultimately improving the robustness of adversarial pruning methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces S2AP, a plug-in method for adversarial pruning that minimizes sharpness in the score-space—the space of importance scores used to select which weights to prune. By perturbing these scores during mask optimization, S2AP stabilizes the pruning process, reduces sensitivity to small score variations, and consistently improves adversarial robustness across various models, datasets, and sparsity levels, without altering the core logic of existing pruning methods.

### Strengths
+ The visualizations of the experimental results are rich and nicely done.

+ The workflow of the proposed S2AP method is clearly presented and easy to understand.

### Weaknesses
+ The motivation needs to be further clarified. In particular, why the score-space optimization can lead to sharp local minima in the loss landscape? Additional empirical results or theoretical analysis are required to support this claim.

+ The flatness of the robust loss landscape has been already explored to improve adversarial robustness (e.g., Ref. [1]). What is the difference between S2AP and existing studies in terms of measuring the flatness of the robust loss landscape?

+ Another major concern is about the computational cost. It seems that S2AP introduces several time-consuming update processes during the pruning. A comprehensive empirical study regarding the computational cost would help the reader better understand the cost of the S2AP method.

[1] AdvRush: Searching for adversarially robust neural architectures, ICCV 2021.

### Questions
Please refer to the Weaknesses part.

### Soundness
3

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
4

### Summary
This paper introduces S2AP, a plug-in method for adversarial pruning that minimizes sharpness in the score space. By perturbing and optimizing these scores to reduce loss landscape sharpness, S2AP stabilizes mask selection and enhances adversarial robustness.

### Strengths
1. The work extends the concept of sharpness minimization from the parameter space to the score-space setting.  
2. It proposes a score-space sharpness minimization approach tailored for adversarial pruning.  
3. The proposed method improves adversarial robustness and outperforms existing pruning baselines such as HARP and HYDRA.

### Weaknesses
1. The motivation is somewhat weak, and key concepts such as the sensitivity of top-k selection to score variations and the link between score-space sharpness and robustness are insufficiently explained.

2. The fairness of comparison may be questionable since it is unclear whether HARP and HYDRA are fine-tuned with AWP as in S2AP. Results under identical fine-tuning setups should be provided.

Minor：  
3. The explanation of “trade-off in generalization” and its connection with ∆ is unclear.

4. The proposed method appears sensitive to hyperparameters (γ), which vary significantly across settings; this should be discussed as a limitation.

5. Incorrect citation formatting on page 1, line 43.

### Questions
The statement “minor score variations can lead to large changes in the selected top-k parameters” is not clearly explained. Could the authors provide a more intuitive explanation or a simple illustrative example to clarify this phenomenon?
Moreover, what is the relationship between the score-space loss landscape and adversarial robustness? Understanding this connection is essential to justify the motivation of the proposed score-space sharpness minimization.

Are HARP and HYDRA also fine-tuned using the AWP strategy? If not, the comparison in Table 1 and Table 2 might be unfair, since fine-tuning with AWP can significantly influence robustness. Could the authors report results where S2AP adopts the same fine-tuning setup as HARP and HYDRA to ensure a fair comparison?

On page 6, line 319, the authors state that “∆ remains mainly positive, showing that S2AP improves over Orig. without introducing a significant trade-off in generalization.” However, it is unclear what exactly the “trade-off in generalization” refers to. What is the relationship between ∆ and generalization?

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
This paper proposes a new Adversarial Pruning (AP) method, i.e., S2AP (Score-space Sharpness-aware Adversarial Pruning). The method explicitly minimizes “sharpness” in the score space, improving the stability of mask selection during pruning and thereby enhancing adversarial robustness. The authors extend the traditional concept of sharpness minimization in the parameter space to the optimization of space score, introducing the concept of “score-space sharpness minimization.”

### Strengths
1.	The idea of transferring “sharpness minimization” from the parameter space to the score space for mask optimization is novel.
2.	S2AP is a plug-in module that can be integrated into any score-based pruning framework without modifying the original objective.
3.	Extensive experiments demonstrate S2AP’s generality and robustness across different scenarios.
4.	They propose a “mask stability” metric to quantitatively verify that S2AP makes the mask search process smoother and more stable.

### Weaknesses
1.	The paper contains several formatting problems, including incorrect citation format (e.g, line 43) and missing punctuation (e.g. line 245).
2.	Limited comparison with existing sharpness-aware methods. While the authors acknowledge that S2AP is inspired by sharpness-aware approaches such as SAM [1] and AWP [2], no experiments compare S2AP directly with SAM or AWP under the same conditions. As a result, it remains unclear how much the proposed method truly improves adversarial robustness compared to these prior approaches.
3.	For S2AP’s finetuning, the authors use AWP approach to minimize sharpness in the classical weight space. While an ablation suggests that S2AP’s robustness improvement is not entirely due to AWP, part of the gain still comes from AWP approach, and the paper does not propose any novel contribution for the finetuning procedure itself.

[1] Foret et al., Sharpness-aware minimization for efficiently improving generalization. In International Conference on Learning Representations.

[2] Wu et al., Adversarial weight perturbation helps robust generalization. Advances in neural information processing systems.

### Questions
1.	S2AP is applied after 5 warm-up epochs. What is done during the 5 warm-up epochs and why exactly 5 epochs were chosen?
2.	Could the authors provide more experimental results comparing S2AP with SAM and AWP under the same conditions to clarify how much the proposed method truly improves adversarial robustness compared to these prior approaches？
3.	In Table 5, the authors report mask robust accuracy on CIFAR-10 and SVHN across sparsity levels using ResNet18, VGG-16, and WideResNet-28-4. However, clean accuracy under the same settings is not shown. Could the authors provide the corresponding clean accuracy results for these models and sparsity levels to give a more complete evaluation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
S2AP is a new plug-in method presented in this paper, for adversarial pruning that aims to improve robustness by minimizing score-space sharpness during mask selection. The authors argue that existing adversarial pruning methods suffer from unstable mask selection due to sharp local minima in the score-space loss landscape. S2AP introduces a min–max optimization over perturbed importance scores to flatten the loss surface and stabilize pruning decisions. The method is evaluated across multiple datasets (CIFAR-10, SVHN, ImageNet), architectures (ResNet, VGG, ViT), and pruning methods (HARP, HYDRA).

### Strengths
1. The paper identifies a clear and important issue, mask instability due to sharp score-space transitions. It provides empirical evidence of its impact on robustness.
2. Experiments span multiple datasets, models, and pruning methods, with consistent improvements in robust accuracy and mask stability.
3. S2AP is designed to integrate with existing pruning pipelines, making it practically useful.

### Weaknesses
My concerns are listed in following aspects.

1. The overall idea appears to be incremental, which is a direct extension of sharpness-aware minimization in weight-space. Applying this to score-space is intuitive but not conceptually novel. Similar ideas have already been explored in AdaSAP and S2-SAM, which also target sharpness minimization for sparse training.

2. The min–max formulation is plausible but lacks rigorous analysis. There is no convergence proof, no bounds on sharpness reduction, and no formal comparison to weight-space perturbation methods like AWP. The sharpness metric used (Hessian eigenvalues) is empirical and not tied to generalization guarantees.

3. The use of Hamming distance between masks across runs is a coarse proxy for stability. It does not capture functional similarity or robustness of the resulting subnetworks. More nuanced metrics (e.g., gradient similarity, spectral gap analysis would provide deeper insight.

4. Although ViT is included, S2AP is only applied to HYDRA due to HARP’s layer-wise constraints. This limits generalizability. Recent works on structured pruning for transformers show that fairness and robustness trade-offs are complex and require more tailored approaches.

5. The paper reports a ~15% increase in pruning time but does not analyze scalability or runtime implications for large models. This is especially relevant for transformer-based architectures and real-world deployment.

### Questions
Please refer to my weakness discussion for questions.

### Soundness
2

### Presentation
2

### Contribution
2
