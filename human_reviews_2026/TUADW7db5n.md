# Batch Pruning by Activation Stability

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 2

## Abstract
Training deep neural networks remains costly in terms of data, time, and energy, limiting their deployment in large-scale and resource-constrained settings. To address this, we propose Batch Pruning by Activation Stability (*B-PAS*), a dynamic plug-in strategy that accelerates training by removing batches that contribute less to learning. *B-PAS* monitors the stability of activation representations across epochs and prunes batches whose activation variance exhibits minimal change, indicating diminishing learning utility. Applied to ResNet-18, ResNet-50, and the Convolutional vision Transformer (CvT) on CIFAR-10, CIFAR-100, SVHN, and ImageNet-1K, *B-PAS* reduces training batch usage by up to 57\% with no loss in accuracy, and by 47\% while slightly improving accuracy. Moreover, it achieves up to 61\% savings in GPU node-hours, outperforming prior state-of-the-art pruning methods with up to 29\% higher data savings and 21\% greater GPU node-hour savings. We further demonstrate the generalization of *B-PAS* by extending it to GPT-2 fine-tuning, showing that activation stability can serve as an effective pruning signal beyond vision models. These results highlight activation stability as a powerful internal signal for efficient training, offering a practical and sustainable path toward data and energy-efficient deep learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces B-PAS (Batch Pruning by Activation Stability), a dynamic data pruning method designed to accelerate DNN training. The core idea is to permanently remove training batches whose internal activation statistics have stabilized across epochs, under the assumption that such batches contribute little to further learning. The method is presented as a lightweight, plug-and-play module. Experiments on various datasets and architectures show that B-PAS can reduce training data and GPU time while maintaining baseline accuracy.

### Strengths
1.  **A Simple and Effective Pruning Heuristic:** The paper proposes using activation stability as a signal for data pruning. This is a straightforward idea that leverages the model's internal state rather than relying on external metrics like loss. The "plug-and-play" nature of the module makes it easy to integrate into existing training pipelines.

2.  **Solid Efficiency Gains on Large-Scale Datasets:** The experiments demonstrate clear improvements in training efficiency, particularly on ImageNet. The method achieves a notable reduction in data usage and GPU hours compared to a full-dataset baseline, outperforming the InfoBatch method in terms of data savings while matching its accuracy.

### Weaknesses
1.  **High Sensitivity to Manually-Tuned Hyperparameters:** The method's performance is critically dependent on the `δ` threshold, which requires an extensive and costly grid search to tune (e.g., 45 configurations for CIFAR-10). The paper offers no automated way to set this parameter, which undermines its practical utility and the goal of saving computational resources.

2.  **Lack of Theoretical Insight:** The work is entirely empirical and lacks a theoretical foundation. It does not explain *why* activation stability is a reliable proxy for a batch's training utility. Without this analysis, the method remains a heuristic without a clear understanding of its potential failure modes.

3.  **Limited Baselines:** The comparison is mostly limited to InfoBatch. Other relevant data selection techniques like coreset selection are not discussed or compared against.

### Questions
1.  The notation in Section 2.2 for the second-to-last batch is `B_{n_i}-1`. This could be misinterpreted. Is the intended notation `B_{n_i-1}`? Please clarify.

2.  Given the high sensitivity to the δ schedule, have you explored any adaptive mechanisms for setting the threshold? For example, could δ be dynamically adjusted based on the validation set performance?

3.  The method's core metric is the mean standard deviation of activations. Have you investigated other metrics to quantify activation stability? For instance, using different statistical moments (e.g., kurtosis), information-theoretic measures (e.g., entropy), or different aggregation methods across layers (e.g., maximum vs. mean)? An ablation on this choice would strengthen the justification for the proposed metric.

4.  Regarding the non-monotonic accuracy curves in Figure 3: do you have a hypothesis for why a moderate increase in the pruning threshold can sometimes lead to slightly better generalization performance before it starts to degrade?

5.  The method relies on permanent pruning. Did you analyze the characteristics of the pruned batches? Is there a risk that this process could introduce bias by disproportionately removing samples from certain classes or domains?

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
3

### Summary
To reduce the training costs of large-scale models, people have considered to reduce the number of training iterations via data pruning. This work proposes a data pruning method which dynamically prunes batches with small changes in activation variances across successive training epochs. Empirical results show that the proposed method can achieve better training efficiency without accuracy loss.

### Strengths
1. This work proposes to assess the contribution of batches using changes in their activation variances, which directly utilizes activation outputs from the forward pass, requiring no substantial additional computation, making it simple and efficient.

2. The paper is well written and easy to follow, clearly describing the motivation of the work and the proposed algorithm.

### Weaknesses
1. In line 53, the paper criticizes the existing methods, saying that they often rely on complex heuristics. However, the proposed methods also introduce additional hyperparameters that need to be tuned.

2. In the experiments, only one baseline (i.e., InfoBatch) is compared with the proposed method, which reduces the significance of the empirical results. Other recent data pruning algorithms, such as (He et al, 2024), are not compared.

3. Intuitively, the performance of the proposed method and the selection of the hyperparameters are likely to be greatly influenced by the batch size. However, this important factor was not considered in the ablation study.

### Questions
The proposed method is only tested on image classification tasks. Can it be generalized to other tasks (e.g., text generation) or models with a larger scale?

### Soundness
2

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
4

### Summary
This paper introduces "Batch Pruning by Activation Stability" (B-PAS), a novel, dynamic data pruning strategy to accelerate deep neural network training. The method operates as a lightweight, plug-and-play module that permanently removes batches from the training process if their contribution to learning has diminished. The core mechanism leverages a simple, model-internal signal: the stability of post-ReLU activation feature maps. By tracking the change in the mean standard deviation of activations for each batch across consecutive epochs, B-PAS prunes batches where this change falls below a dynamic threshold. The authors provide comprehensive empirical validation on various models (ResNet, CvT) and datasets (CIFAR, ImageNet), demonstrating up to 57% data savings and a 61% reduction in GPU node-hours on ImageNet without sacrificing accuracy, significantly outperforming prior art. The paper also introduces the Data Savings Index (DSI), a hardware-agnostic metric for data efficiency.

### Strengths
Novelty and Simplicity: The core idea of using the stability of post-ReLU activation variance as a pruning signal is original, computationally cheap, and mechanistically sound. It provides an elegant, model-internal alternative to more complex methods that rely on loss or gradient statistics.   

Great Empirical Rigor: The method is thoroughly validated across diverse architectures (ResNet, CvT) and datasets (CIFAR, SVHN, ImageNet-1K). The comprehensive ablation studies and the significant outperformance of a strong baseline (InfoBatch) on ImageNet provide convincing evidence of the method's quality and effectiveness.   

Significant Practical Impact: The demonstrated efficiency gains are substantial, with up to a 61% reduction in GPU node-hours on ImageNet without accuracy loss. The introduction of the hardware-agnostic Data Savings Index (DSI) is also a valuable contribution to the community for standardizing efficiency reporting.

### Weaknesses
Critical Dependency on Batch Normalization: The method's effectiveness collapses without Batch Normalization, as shown in the ablation study (DSI drops from 25% to 2%). This severely limits its applicability to the growing number of modern architectures, such as many transformers, that do not use BN.   

Reduced Effectiveness on Transformers: B-PAS achieves significantly lower data savings on the Convolutional Vision Transformer (CvT) compared to CNNs (14% vs. 47% DSI on ImageNet). This suggests the pruning signal is architecture-dependent and questions the method's generalizability beyond CNNs.   

Practicality Concerns of Hyperparameter Tuning: The method's performance is highly sensitive to the pruning threshold schedule ($\delta_s, \delta_e$), requiring extensive grid searches (e.g., 45 settings for CIFAR-10) to find the optimal configuration. This tuning overhead could offset the efficiency gains in practice.   

Deviation from Standard Training Practices: The requirement to fix batch compositions at initialization to enable tracking is a departure from the standard practice of reshuffling the dataset each epoch. This could have unexamined negative effects on model generalization.

### Questions
On Fixed Batch Composition: Since using fixed batches deviates from standard shuffling, what is the impact on generalization? Does intra-batch shuffling fully mitigate potential overfitting to specific sample groupings?

On Batch Normalization Dependency: Given the method's strong reliance on Batch Normalization, does the stability signal depend on BN's per-batch standardization? Have you tested its effectiveness with other schemes like LayerNorm or in Normalizer-Free architectures?

On Transformer Performance: Could you elaborate on why B-PAS is less effective on the CvT architecture? Is the activation stability signal inherently weaker in transformers, and did you investigate tracking activations at different points within the transformer blocks?

On Hyperparameter Sensitivity: The $(\delta_s, \delta_e)$ schedule required extensive tuning. Are there any practical heuristics for setting these hyperparameters for new tasks without performing a costly grid search?

On the Nature of Pruned Batches: Could you provide a qualitative analysis of the pruned batches? For instance, are they primarily "easy" examples, and does the pruning process introduce any significant shift in the class distribution of the remaining data?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Batch Pruning by Activation Stability (B-PAS), a dynamic plug-in framework designed to accelerate deep neural network training by pruning low-utility data batches based on the stability of their internal activations.

### Strengths
1) The paper is well-written and visually clear.
2) The paper introduces a pruning method by exploiting activation stability to identify redundant data batches that contribute less to learning.

### Weaknesses
1) The conceptual originality is modest. Leveraging activation stability for pruning has been well studied in existing studies. The contribution appears more incremental than fundamental.
2) The experimental comparison with existing baselines is insufficient. The evaluation primarily compares with InfoBatch, lacking a lot of related SOTA pruning methods.
3) The paper proposes that activation stability reflects diminishing learning utility. But no direct evidence is provided to support this claim.

### Questions
1) Can authors compared the proposed method with more SOTA pruning methods?
2) Can the proposed method be applied to other domains like LLMs?

### Soundness
2

### Presentation
3

### Contribution
2
