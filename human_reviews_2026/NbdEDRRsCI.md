# Many Eyes, One Mind: Temporal Multi-Perspective and Progressive Distillation for Spiking Neural Networks

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 6, 2, 6

## Abstract
Spiking Neural Networks (SNNs), inspired by biological neurons, are attractive for their event-driven energy efficiency but still fall short of Artificial Neural Networks (ANNs) in accuracy. Knowledge distillation (KD) has emerged as a promising approach to narrow this gap by transferring ANN knowledge into SNNs. Temporal-wise distillation (TWD) leverages the temporal dynamics of SNNs by providing supervision across timesteps, but it applies a constant teacher output to all timesteps, mismatching the inherently evolving temporal process of SNNs. Moreover, while TWD improves per-timestep accuracy, truncated inference still suffers from full-length temporal information loss due to the progressive accumulation process. We propose **MEOM** (**M**any **E**yes, **O**ne **M**ind), a unified KD framework that enriches supervision with diverse temporal perspectives through mask-weighted teacher features and progressively aligns truncated predictions with the full-length prediction, thereby enabling more reliable inference across all timesteps. Extensive experiments and theoretical analyses demonstrate that MEOM achieves state-of-the-art performance on multiple benchmarks. Code is available at https://github.com/KaiSUN1/MEOM.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents MEOM (Many Eyes, One Mind), a Knowledge distillation (KD) framework for spiking neural networks (SNNs). It employs Temporal Multi-Perspective Distillation (TMPD) to introduce temporal variances to the ANN teacher output. It also utilizes Temporal Progressive Distillation (TPD) to align with the full-length prediction for truncated inference progressively.

### Strengths
1. The proposed MEOM considers the temporal variances of SNNs.
2. This paper provides theoretical analyses for the proposed submodules.

### Weaknesses
1. The motivation of TMPD requires further clarification. Lines 51-69 claim that it is improbable for outputs across timesteps to remain identical. A more effective strategy would incorporate diverse temporal supervisory signals. However, as illustrated in Fig. 1, both final logits and membrane potential distribution do not exhibit significant differences across timesteps. Furthermore, the impact of such variations on SNN performance has not been verified.
2. The proposed TMPD is more like a data augmentation method that introduces perturbations in the temporal dimension, rather than providing richer temporal supervision. Theorems 1 and 2 merely demonstrate that introducing perturbations results in higher temporal covariance and lower gradient variance, thereby proving the effectiveness of this data augmentation approach. They do not prove that TMPD provides richer temporal supervision. I believe this data augmentation method is not only applicable to KD but also equally effective for general SNN training.
3. The proposed MEOM method does not show significant performance improvement over state-of-the-art methods. Compared to TWSNN, its accuracy gains across various datasets are less than 1%.
4. The overall training objective introduces three hyperparameters $\alpha$, $\beta$, and $\gamma$. The impact of hyperparameter settings on the effectiveness of the proposed method remains unclear. The robustness of hyperparameter settings across different tasks is also unclear.

### Questions
1. Please analyze the temporal variance of SNNs and its effect on SNN performance.
2. Please analyze the impact and robustness of the hyperparameter introduced in MEOM.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MEOM, a novel SNN distillation framework that tackles two key issues: static teacher supervision and poor truncated inference. It employs Temporal Multi-Perspective Distillation (TMPD) to generate diverse teacher signals and Temporal Progressive Distillation (TPD) to align predictions across timesteps. MEOM achieves state-of-the-art results, demonstrating significant improvements in both final accuracy and performance under truncated inference.

### Strengths
1. The paper is well written, clearly structured, and logically sound. 
2. The critique of existing TWD methods is insightful. Identifying the "static teacher vs. dynamic student" mismatch and the "information loss in truncated inference" as two practical and important problems is a key contribution. 
3. The experimental results are convincing. The authors demonstrate state-of-the-art results across multiple datasets (CIFAR, ImageNet) and architectures (ResNet, Spiking Transformer), while a dedicated "time flexibility" experiment shows the method's advantage in truncated inference. Furthermore, thorough ablation studies clearly isolate the contributions of each proposed component (TMPD and TPD) confirming their complementary benefits.

### Weaknesses
1. Masking strategy in TMPD. The use of random masks in TMPD is a simple and effective way to generate diverse teacher signals. However, this might not be the optimal strategy. The mask generation is based on a fixed random sampling and does not adapt to the student's learning state or the characteristics of different timesteps. Exploring the connection between masking strategies and temporal-wise features could potentially lead to further performance improvements.
2. Interplay between TMPD and TPD. The paper presents TMPD and TPD as two complementary components. However, TMPD is designed to introduce "diversity" while TPD is designed to enforce "consistency," two goals that could, at some level, be seen as being in tension. While the experiments show their combination is effective, the paper could benefit from a deeper discussion on how these components work synergistically rather than counteracting each other.

### Questions
1. Regarding the mask design in TMPD, ehe random mask strategy is simple and effective. Have you explored other, more sophisticated masking approaches, such as learnable masks or masks that are dynamically adapted based on the timestep? Do you think such strategies could offer additional performance gains?
2. Regarding the progressive alignment in TPD, the TPD achieves progressive consistency by aligning the "cumulative average prediction" of consecutive timesteps. How does this compare to a simpler strategy of directly aligning the cumulative prediction at each step with the final prediction (at time T)? Do you think the "step-wise" smoothing is crucial for training stability?
3. Regarding the synergy between TMPD and TPD, as TMPD is designed to introduce "diversity" and TPD is designed to enforce "consistency," two goals that could somehow be in tension. Could you elaborate on a deeper level how these two seemingly opposing goals work synergistically within the MEOM framework?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a knowledge distillation method for SNN training, called MEOM (Many Eyes, One Mind). The method consists of two parts: Temporal Multi-Perspective Distillation (TMPD), which creates the teacher's feature for each time-step, and Temporal Progressive Distillation (TPD), which averages activation across different time-steps and calculates CE losses between the average results and targets.
Experiment shows that the method has better accuracy than BKDSNN with Spikformer-8-384 on ImageNet.

### Strengths
1. The paper organization is clear.
2. The proposed distillation methods are understandable.
3. The results in CIFAR10/100 is good.

### Weaknesses
1. Limited novelty:
The proposed work introduces two techniques to enhance the performance of knowledge distillation. However, the TMPD method only provides a marginal improvement (around 0.3%) on CIFAR-10/100, which is within the range of training variance and thus not convincing. Moreover, the TPD approach is rather straightforward and not inherently tied to the distillation framework. Therefore, it could also be applied to conventional BPTT-based training, which weakens its novelty.

2. Insufficient experimental validation on ImageNet:
The experiments on ImageNet lack sufficient ablation studies, making it difficult to attribute the performance gain solely to the proposed techniques. In addition, the authors should consider evaluating on a larger model, such as the S-8-768 structure, where BKDSNN achieves 79.9% accuracy, to strengthen the evidence.

3. Missing analysis of training overhead:
The proposed distillation-based methods are expected to introduce additional training overhead, especially due to the TMPD methods. The authors should include a comparison of the GPU memory footprint and total GPU hours between the proposed methods and standard distillation baselines.

### Questions
Please see the weaknesses.

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
4

### Summary
This paper introduces MEOM (Many Eyes, One Mind), a unified knowledge distillation framework for spiking neural networks (SNNs). It integrates two complementary modules: Temporal Multi-Perspective Distillation (TMPD), which enhances temporal diversity via masked teacher features, and Temporal Progressive Distillation (TPD), which gradually aligns truncated and full-length predictions to improve temporal consistency.

### Strengths
1.The paper clearly identifies the weaknesses of prior temporal distillation approaches and motivates the need for temporal diversity and consistency.
2.The unification of TMPD (“Many Eyes”) and TPD (“One Mind”) provides an intuitive and theoretically supported structure.
3.Experiments cover multiple benchmarks, showing both performance improvement and robustness under truncated inference.
4.The paper’s exposition and proofs (information gain, convergence robustness) provide reasonable theoretical support for the design choices.

### Weaknesses
1. TMPD uses random, static masks. While effective, the design choice appears heuristic. Could adaptive or learnable masks yield further benefit? Consider comparing random vs. learned mask distributions or adding mask diversity analysis. 
2. TPD enforces progressive step-wise alignment but may neglect long-range temporal dependencies. A long-range consistency variant or additional analysis on global alignment would make the argument more complete.
3. Assuming the final timestep is globally optimal may not hold universally. A calibration or stability analysis across timesteps could validate this assumption, or adaptive target-timestep selection could be explored.
4. Evaluation focuses on ResNet-style SNN backbones, generalization to more complex or recent SNN backbones (e.g., Spiking Transformer, hybrid architectures) remains unexplored.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
