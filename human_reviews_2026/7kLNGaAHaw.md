# Architecture-Agnostic Test-Time Adaptation via Backprop-Free Embedding Alignment

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 10

## Abstract
Test-Time Adaptation (TTA) adapts a deployed model during online inference to mitigate the impact of domain shift. While achieving strong accuracy, most existing methods rely on backpropagation, which is memory and computation intensive, making them unsuitable for resource-constrained devices. Recent attempts to reduce this overhead often suffer from high latency or are tied to specific architectures such as ViT-only or CNN-only.
In this work, we revisit domain shift from an embedding perspective. Our analysis reveals that domain shift induces three distinct structural changes in the embedding space: translation (mean shift), scaling (variance shift), and rotation (covariance shift). Based on this insight, we propose Progressive Embedding Alignment (PEA), a backpropagation-free and architecture-agnostic TTA approach. By applying a novel covariance alignment procedure at each intermediate layer, PEA efficiently corrects the embedding distortions with only two forward passes. Extensive experiments demonstrate that PEA achieves state-of-the-art performance in both accuracy and efficiency, while also proving versatile across different architectures including ViTs and CNNs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a new backpropagation-free test-time adaptation method to calibrate the pre-trained model to the target domains using two forward passes per sample. The authors observe that the distribution shift can cause three distinct structural changes in the features of intermediate layers. Based on this observation, they design a Progressive Embedding Alignment (PEA) to correct the feature distortions. One forward is used to estimate the level of distribution shift, and the other is used to recompute the intermediate features to tackle the distribution shift. Moreover, they also use the exponential moving average and data argumentation to enable stable and robust adaptation. Experimental results demonstrate the effectiveness of the proposed method.

### Strengths
1. The authors empirically reveal that the distribution shift can cause the three distinct structural changes in the feature space, including translation (mean shift), scaling (variance shift), and rotation (covariance shift).
2. The authors propose a Progressive Embedding Alignment (PEA) to calibrate the feature without updating the parameters of the model, which can originally mitigate catastrophic forgetting. 
3. The results of the ablation study demonstrate the necessity of each component in the proposed method.

### Weaknesses
1. The authors propose a feature re-computation method to calibrate the pre-trained model. However, a similar idea that updates the intermediate features has been proposed by the previous method[A]. What is the essential difference and advantage of the proposed method? 
2. The authors show the distinct structural changes of the ViT model in Figure 1. However, it is unclear whether this observation is related to this specific architecture. More discussion or experimental results on CNN models or models with different normalization layers should be provided.
3. The proposed method requires computing the statistics of a batch of test samples, so the distribution of different domains in the same batch will greatly affect the accuracy of the statistical distribution. When the test samples within the same batch come from different domains (mixed data domains), the proposed method may suffer from unreliable distribution estimation and cannot guarantee a stable performance.
4. The Spike detection is proposed to detect the domain changes and then reset the estimated statistical information for the new domain. However, it is unclear how to decide the threshold \theta_{ent} for different domains, especially under challenging test scenarios such as imbalanced label shifts per SAR.
5. The data enrichment is designed to enhance the estimation of the target batch distribution by generating K augmented views for each test sample. However, it is unclear how the hyperparameter K affects the adaptation performance and the computational cost.
6. More BP-free test-time adaptation methods should be discussed and compared in the experiment, including but not limited to [B-D].

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work proposes PEA, a backpropagation-free and architecture-agnostic test-time adaptation (TTA) approach that aims to enhance adaptation while reducing computational bottlenecks. Unlike most TTA methods that rely on backpropagation—which is computationally expensive at test time—or are architecture-specific, PEA provides a general alternative. The authors observe that the representations of shifted domain samples across layers are geometric transformations of the source representations, primarily translation, scaling, and rotation. Building on this insight, PEA performs two forward passes: the first collects batch statistics per layer, and the second applies a correction based on source statistics. Experimental evaluations on ResNet-50 and ViT-B across corrupted datasets (CIFAR-10-C, CIFAR-100-C, and ImageNet-C) demonstrate accuracy gains when combined with test-time augmentation. Additional experiments on low-resource devices further highlight the method’s memory efficiency.

### Strengths
* The paper is well written and easy to follow.

* The paper proposes a method with two interesting properties: backpropagation-free and architecture-agnostic, which increases the applicability of TTA across different models and devices.

* The evaluation on low-resource devices is a valuable contribution, highlighting the method’s efficiency in practical settings.

### Weaknesses
* The paper emphasizes the existence of discrepancies in representations between source and target distributions and their variation across layers. However, this adds limited new understanding. By definition, distribution shifts alter centroids and other statistical descriptors, and much of the domain adaptation literature already focuses on aligning source and target distributions.

* The hypothesis that shifts reduce to simple geometric transformations in embedding space is overly strong, and the visualizations in Figure 1 and Appendix A do not provide convincing evidence. For example, as long as representations differ, a translation between centroids is always possible and therefore not surprising. Moreover, these are linear operations, and evidence from only two transformations is insufficient to claim generality. Non-linear or more complex transformations may well occur in intermediate representations, and the current evidence does not rule this out.

* A critical assumption is that source statistics for all layers are accessible at test time. This may not hold in many realistic scenarios and sets the method apart from approaches that do not make such an assumption. This should be acknowledged explicitly as a limitation.

* The conclusions in lines 160–163 and 139-141 overlook prior fine-grained analyses, such as Sahoo et al. and Maharana et al., which emphasize that not all layers are equally affected and advocate for layer-wise adaptation. 


* The detection mechanism based on the entropy of predictions is limited. Entropy is a weak OOD detection score compared to alternatives available in the literature (e.g., OpenOOD benchmark).

References

Sahoo, Sabyasachi, et al. "A layer selection approach to test time adaptation." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 19. 2025.

Maharana, Sarthak Kumar, Baoming Zhang, and Yunhui Guo. "Palm: Pushing adaptive learning rate mechanisms for continual test-time adaptation." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 18. 2025.

OpenOOD benchmark: https://github.com/Jingkang50/OpenOOD

### Questions
* Class representations appear already well separable after only 3 blocks (out of 9). How do you explain this observation?

* Line 167: “This systematic shift does not lead to a random feature distortion; rather, it manifests as a geometric rotation and shearing of the entire feature cloud.” Why would we expect a random feature distortion in the first place?

* In pass1 paragraph of line 290: How are K views of multiple augmentations obtained within a single forward pass? Please clarify this point.

* For clarity, in the main text, please specify what a “block” corresponds to in the ViT architecture, and explicitly indicate that the referenced block 3 is the third out of nine blocks.

* Please cite prior work using test-time augmentation (e.g., Simonyan and Zisserman, 2014).

Reference

Simonyan, Karen, and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014).

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
This paper proposes Progressive Embedding Alignment (PEA), a backpropagation-free and architecture-agnostic Test-Time Adaptation (TTA) method. The key idea is to interpret domain shifts as three geometric transformations in the embedding space—translation (mean shift), scaling (variance shift), and rotation (covariance shift)—and correct them progressively via Whitening-Coloring Transform (WCT)–based covariance alignment. PEA performs only two forward passes per test batch, without modifying model weights, and can be applied to both CNNs and ViTs. Experiments on CIFAR10-C/100-C and ImageNet-C show that PEA achieves competitive accuracy with minimal memory and latency, demonstrating its suitability for real-time or edge deployment.

### Strengths
The embedding-space interpretation of domain shift (translation, scaling, rotation) is elegant and intuitive. It provides a clear geometric motivation for the proposed covariance alignment strategy.

PEA is architecture-agnostic for both CNNs and ViTs.

The fully forward-only, memory-light, and backprop-free natures make PEA ideal for edge devices. It achieves low latency (~0.3s) and small memory footprint (<1GB) while maintaining strong accuracy.

Device-level tests on Jetson Orin Nano enhance practical relevance.

### Weaknesses
As the proposed method relies on statistical estimation, its performance degrades under small batch sizes. And, I am wondering how does the method performs when the batch size is as small as 2? Moreover, with augmentation enabled, could PEA still operate effectively with a batch size of 1?

### Questions
How to decide the threshold for detecting domain changes (Eq. 7)  across different models and domains during online testing? An ablation study (with and without Eq.7) and sensitivity analysis on this threshold would be helpful.

How does the number of augmented views in PEA affect performance? More detailed sensitivity analyses on this factor are preferred.

How does PEA perform under mixed domain shifts? It would be helpful to see results for both CNN and ViT backbones in such scenarios.

The reported results of SPA appear to be lower than those in the original paper. Could the authors clarify this discrepancy? Since SPA is a strong backpropagation-based method, it is understandable that PEA (focus on forward-only) has lower performance than SPA, but ensuring consistent and accurate SPA results would make the comparison more convincing.

Suggestion:

FOA also includes an alignment component, namely “activation shifting,” which appears conceptually related to the proposed PEA. However, PEA performs alignment in a more fine-grained, layer-wise, and comprehensive manner. It would be much better to discuss this relationship in the paper.

I would like to consider increasing my score if these are well addressed.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper proposes **Progressive Embedding Alignment (PEA)**, a novel Test-Time Adaptation (TTA) method that is **backpropagation-free**, **architecture-agnostic**, and highly efficient. The authors aim to overcome two major limitations of existing TTA work: 1) the high computational and memory overhead of backpropagation-based methods (e.g., Tent, EATA), which makes them unsuitable for edge devices , and 2) the architecture-specificity of recent efficient methods, which are often tailored for either CNNs or ViTs, but not both.

The core idea is motivated by a principled analysis of domain shift. The authors empirically demonstrate that domain shifts induce a combination of three consistent geometric distortions in the intermediate embedding space:
1.  **Translation** (mean shift) 
2.  **Scaling** (variance shift) 
3.  **Rotation** (channel-wise covariance shift) 

PEA works by explicitly correcting these distortions. In an offline step, it pre-computes and stores the source-domain statistics (mean $\mu_{s,l}$ and covariance $\Sigma_{s,l}$) for each block $l$. Then, at test time, it uses a two-pass, forward-only procedure:
* **Pass 1:** Estimates the statistical distance between the current batch and the source to compute a layer-wise alignment weight $w_l$.
* **Pass 2:** Applies a Whitening-Coloring Transform (WCT) to realign the batch features to the source statistics. The final, corrected feature $F_l'$ is a weighted blend of the original and aligned features, $F_{l}^{\prime}=(1-w_{l})F_{l}+w_{l}Y_{l}$, which prevents over-correction .

### Strengths
- Clear Motivation and Novelty: The analysis of domain shift as a geometric transformation (translation, scaling, rotation) in the embedding space is intuitive and provides a strong foundation for the proposed method. The idea of reversing these distortions via a weighted WCT, without any gradient updates, is a novel and elegant approach to TTA.

- Exceptional Efficiency (Memory and Latency): The paper's primary contribution is its practicality. By being backprop-free and requiring only two forward passes , PEA is significantly more efficient than SOTA methods like CMF, SPA, or EATA. This is powerfully demonstrated in the edge device evaluation (Table 4), where PEA runs successfully on a Jetson Orin Nano, while numerous backprop-based baselines (Tent, EATA, CMF, etc.) fail due to OOM errors.

- Strong Empirical Results: PEA (especially with augmentation) achieves state-of-the-art or highly competitive accuracy on all three benchmarks (ImageNet-C, CIFAR10-C, CIFAR100-C). 

- Generality (Architecture-Agnostic): This is a key advantage. The method works "out-of-the-box" for both CNNs (ResNet-50) and Transformers (ViT-Base) using the identical procedure

### Weaknesses
- Source Data Requirement: The method's primary limitation, which is acknowledged by the authors26, is that it is not "source-free." It requires access to the original source training dataset in the offline stage to pre-compute the source statistics ($\mu_{s,l}$, $\Sigma_{s,l}$)27. While this is a valid setup for TTA, it is a stronger assumption than methods that only require the pre-trained model.

-  Storage Overhead: While the inference compute/memory is low, PEA introduces a storage cost by requiring the mean vector and covariance matrix for each aligned block to be stored. The paper mentions this is minimal (~30MB for ViT-Base), but this cost should be explicitly reported for all models in the main tables to provide a complete picture of the resource trade-offs.

- Statistics Estimation: The method's effectiveness hinges on the quality of the estimated test-time statistics ($\mu_{t,l}$, $\Sigma_{t,l}$)28. While the EMA helps, the authors note that performance could degrade with extremely small batches (e.g., BS=1) or high class imbalance29. A formal evaluation at BS=1, a true "online" setting, would be valuable to fully test this boundary condition.

### Questions
- Are recent methods used to reduce the overhead of backprop only tied to vit or cnn or have high latency? if so can you share related work to that?

-  In equation 1 why you didn't use the mahalanobis distance between each input in the batch and the source distribution instead of relying on the batch statistics and in that case even for bs 1 the method shall work preoperly out of the box without having to think about the scale difference between mean and variance as the current distance can be overwhelmed by one measure over the other (if we have more mean shift or more variance). 

- Can you ablate the choice of distance and what happens if we simply use the shift in mean ?

- how do you select the fixed threshold used to flag spikes in feature alignment

- Can you ablate the choice of k views augmentaion to see its effect on the main results ?

### Soundness
4

### Presentation
4

### Contribution
4
