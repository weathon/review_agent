# Expressive yet Efficient Feature Expansion with Adaptive Cross-Hadamard Products

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Recent theoretical advances reveal that the Hadamard product induces nonlinear representations and implicit high-dimensional mappings for the field of deep learning, yet their practical deployment in resource-constrained vision models remains largely unexplored. To address this gap, we introduce the Adaptive Cross-Hadamard (ACH) module, a novel operator that embeds learnability through differentiable discrete sampling and dynamic softsign normalization. This facilitates highly efficient feature reuse without incurring additional convolutional parameters, while ensuring stable gradient flow. Integrated into Hadaptive-Net (Hadamard Adaptive Network) via neural architecture search, our approach achieves unprecedented efficiency. Comprehensive experiments demonstrate state-of-the-art accuracy/speed trade-offs on image classification tasks, establishing Hadamard operations as specific building blocks for efficient vision models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces the Adaptive Cross-Hadamard (ACH) module, an operator that embeds learnability through differentiable discrete sampling and dynamic softsign normalization. The ACH module is designed to enable parameter-free feature reuse while stabilizing gradient propagation. It is integrated into the Hadaptive-Net (Hadamard Adaptive Network) using neural architecture search to achieve high efficiency.

### Strengths
(1) The paper's approach of leveraging the Cross-Hadamard Product from the perspective of feature expansion to enhance model efficiency is novel and interesting.

(2) The paper is well-written, providing a comprehensive description from its methodology to the implementation details.

### Weaknesses
(1) Insufficient Motivation for Technical Choices: The paper lacks a clear and in-depth discussion of the motivation behind its core technical components. For instance, in the sections on "HADAMARD FOR CHANNEL EXPANSION" and "DIFFERENTIABLE DISCRETE SAMPLING," the paper focuses more on the implementation details (how) rather than the fundamental reasons (why) these specific techniques are suitable or advantageous. This makes it difficult for readers to fully appreciate the design principles of the proposed method.

(2) Potentially Limited Novelty in Sub-components: The novelty of some technical components appears to be incremental. The section on "DIFFERENTIABLE DISCRETE SAMPLING," for example, seems to rely heavily on the combination of existing techniques like the Gumbel-Topk trick and the ECA module. While combining existing ideas can be a valid contribution, the paper does not sufficiently articulate what makes this specific combination novel or non-trivial.

(3) Limited Experimental Scope: The experimental evaluation is confined to CNN-based architectures. In the current deep learning landscape, demonstrating effectiveness on Transformer-based models is crucial for showing broad applicability, especially for a method focused on efficiency. The absence of such experiments makes the generalizability of the proposed Hadaptive-Net unclear.

### Questions
(1) In the sections on "HADAMARD FOR CHANNEL EXPANSION" and "DIFFERENTIABLE DISCRETE SAMPLING," the paper primarily describes how these techniques are applied but lacks a detailed explanation of why they were chosen. Could the authors elaborate on the underlying motivation and principles for using these specific technologies? The current description in this area feels somewhat underdeveloped.

(2) Regarding "DIFFERENTIABLE DISCRETE SAMPLING," it seems that several of the components, such as the "Gumbel-Topk trick" and the "ECA module," are existing techniques. Could the authors clarify the novelty of their approach in this context and how these existing tricks are combined or adapted in a non-trivial way?

(3) The experiments primarily use CNN-based backbones. Have the authors evaluated or considered the applicability of their method to efficient Transformer architectures? The lack of such experiments may limit the perceived scope of the contribution.

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
This submission aims to achieve efficient and expressive feature recombination for lightweight vision models. It proposes Adaptive Cross-Hadamard (ACH) module, which leverages (i) learnable Hadamard (element-wise) products, (ii) discrete channel sampling via Gumbel-TopK, and (iii) dynamic softsign normalization for efficient feature expansion and recombination within CNNs. The optimal placement of ACH blocks alongside standard GhostNet-style linear expansion blocks is is discovered via differentiable neural architecture search to build the Hadaptive-Net.

Experiments are conducted on image classification (CIFAR-100, ImageNet-1K) and object detection (COCO) tasks. The results show Hadaptive-Net’s competitive accuracy-latency trade-offs compared to some baselines. Empirical analysis and theoretical complexity comparison such as ablation studies, architecture search, Grad-CAM visualizations are also conducted. Practically, it also presents engineering optimizations for GPU execution.

### Strengths
**(S1)** Clear drive and originality. This work identifies redundancy in existing lightweight vision models like GhostNet and inverted bottlenecks, which is a topic of practical signifance. It presents a mathematically grounded alternative in Sec. 3.1, leveraging the non-linear benefits of Hadamard products. This moves beyond previous linear cheap operations (e.g., GhostNet, gating-based models) and explores a new, parameter-free path to feature learning.

**(S2)** Technical soundness of method. The ACH module is well-engineered to address practical learning and stability constraints, including discrete channel selection via Gumbel-TopK sampling in Sec. 3.2, dynamic softsign normalization in Sec. 3.3, and NAS integration strategy in Sec. 3.4. In particular, I appreciate the ablation study in Fig. 3 which provides evidence for the design choices. The dramatic accuracy drop (73.57% to 64.39%) when replacing DySoft with vanilla BatchNorm is striking and supports the claims of stability. Appendix presents hyper-parameters, hardware, training schedules, and code release, which aids reproducibility.

**(S3)** Focus on deployment and GPU optimization. Section 4.2 shows efforts to bridge theoretical and real-world inference latency, such as custom CUDA kernels (direct-indexing and parity-balanced) and algorithmic choices for ACH operation. Fig. 5 visualizes runtime differences under varying tensor shapes. This focus on implementation details is useful for helping with real-world deployment in resource-constrained scenarios. 

**(S4)** Broad experiments and competitive results. Ablation studies in Fig. 3 and Tab. 1, 2, 5 break down the contribution of each module component and validate plug-and-play integration on multiple efficient models. NAS-derived architecture choice is also examined in Tab. 3 and Appendix. Tab. 7 provides direct comparisons on CIFAR-100 and ImageNet-1K, considering both accuracy and efficiency metrics (latency, FLOPs). Notably, it achieves higher accuracy than recent MobileNetV4-L (74.73% vs. 74.38%) with less than 1/3 of FLOPs (669M vs. 2170M).

**(S5)** Visualization and interpretability. Grad-CAM in Fig, 6 and 7 show that ACH enables more semantically meaningful feature extraction and achieves improved early focusing compared to linear baselines. Diagrams like Fig. 2 (ACH pipeline), Tab. 1 (activation comparisons) are also well-constructed and informative.

### Weaknesses
**(W1)** In Appendix A.4, complexity ratio for Ghost/ACH modules is handled via “given $m \ll n$” simplifications, but experiment settings do not clarify the $m:n$ ratio in real deployments. This may hinder the reproduction or subsequent improvements in the community. I encourage the authors to provide more technical details on this point in revision.

**(W2)** For object detection in Tab. 6, the gains of Hadaptive-Net-L over GhostNetV3 or MobileNetV4 are a bit marginal. This raises the question whether it actually overcomes the efficiency wall or simply offers a modest incremental boost within a specific operator. I recommend the authors add direct clarification or discussion on this point in related section. This would significantly alleviate readers' concerns.

**(W3)** Seems no evidence is offered as to robustness under distribution/drift, input permutation, adversarial queries, or transfer to other domains (e.g., audio, language). This limits the “deep learning primitive” claim in Sec. 6. I recommend the authors consider slightly tempering this claim in the revised manuscript.

**(W4)** Incomplete literature review. Several important prior work [1] [2] [3] [4] [5] that applies Hadamard products in efficient vision models, attention/bilinear pooling, or edge detection are not included and discussed. I recommend the authors incorporate related discussions in the revised manuscript, which would not only improve the completeness but help emphasize the contributions and novelty of ACH within existing studies. It would be better to show the direct comparisons with [4] and [5].

**(W5)** The fine control of hyper-parameter $\tau$ is managed solely through heuristics and lacks sensitivity analysis experiments to support the choices. 

---

## Reference

[1] Hadamard Product in Deep Learning: Introduction, Advances and Challenges, TPAMI 2025. This survey covers the use of Hadamard products in deep learning, especially for efficiency and nonlinear fusion. 

[2] Hadamard Product for Low-rank Bilinear Pooling, ICLR 2017. It proposes efficient attention musing Hadamard products relevant to feature recombination methods. 

[3] Unmixing Convolutional Features for Crisp Edge Detection, TPAMI 2021. It explores element-wise feature fusion similar to Hadamard approaches, directly related to efficient feature expansion.

[4] MogaNet: Multi-order Gated Aggregation Network, ICLR 2024. It is a representative efficient CNN model that uses gated Hadamard products for multi-order feature aggregation.

[5] Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model, ICML 2024. It first utilizes SiLU-gated Hadamard products within bidirectional state-space blocks to enable selective feature propagation.

### Questions
Most of my major concerns and related recommendations have been stated in the Weaknesses section. I encourage the authors to focus their efforts on addressing those points, as they are critical for strengthening the manuscript in the rebuttal stage.

The following are more specific, minor questions to help the authors think more deeply about certain design choices and experiment setups, which I hope might be helpful for this and future work:

**(Q1)** Could the authors provide theoretical analysis or insights into why dySoft normalization outperforms others in the ACH setting? Perhaps via variance/convergence analysis or more direct connections to gradient stability?

**(Q2)** Is there empirical evidence that ACH modules offer gains outside vision, such as language, audio? How do they perform in settings with heavy distribution shift? IMHO, such explorations may get closer to the heart of the question.

**(Q3)** It seems that the NAS leaves most early layers as Ghost modules and prefers ACH only at high dimensions. Do the authors see this as a limitation or a natural consequence of Hadamard product behavior? Is hybridization always preferable to pure ACH insertion? I believe this worths deeper investigation.


---

## Justification:

This submission presents strengths in motivation, technical soundness, thorough experiments, and practical engineering contributions. However, several concerns exist such as the lack of clarification for some technical details, limited literature review, and more. Overall, I am leaning towards acceptance and first give a rating of 6. I would be glad to raise my rating if thoughtful responses and improvements are provided. Conversely, if most of the concerns remain unaddressed, I may also lower my score. I am also open to follow-up discussions with the authors to help further strengthen this work.

I hope these comments help my fellow reviewers and ACs understand the basis of my recommendation.

### Soundness
2

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
4

### Summary
The authors introduce the Adaptive Cross-Hadamard (ACH) module, utilizing channel attention-guided feature gating and differentiable discrete sampling along with Hadamard (element wise). This module, when stacked can efficiently and effectively model nonlinear representations and high-dimensional relationships. The authors develop a novel and effective training methodology utilizing the Gumbel TopK trick to limit the compute of the model across channels as well as a novel hardware-efficient normalization scheme due to the nature and requirements of this new model. The authors use architecture search to demonstrate that this module is preferred over existing methods and demonstrate its computational efficiency. Finally, especially at small scale, the authors Hadaptive-Net models clearly outperform competing, less efficient models on top1 accuracy for the CIFAR-100 and ImageNet1K datasets. As a result, the authors have obtained a more efficient, more effective model.

### Strengths
Through the introduction of the Adaptive Cross-Hadamard (ACH) module these results develop a novel method for training pairwise Hadamard products with very few parameters and utilize Hadamard operations as building blocks for efficient vision models, which is relatively unexplored. Moreover the authors' module offers parameter-free feature reuse, reducing computational cost. The ACH module has strong accuracy/speed trade-offs, appealing for real-world and edge deployment and a hardware friendly implementation. Finally, in evaluation the ACH based classification models offer state of the art accuracy/speed tradeoffs.

### Weaknesses
I think it would be very helpful to have a precise description of the end to end method listed as an algorithm at some point. In particular, equation 10 is somewhat ambiguous to me. It is not very clear how the tensor X is added to the cross Hadamard product (e.g. what is the $\oplus$ operator doing?) It is also  somewhat confusing at first blush what components of the model are trainable and which aren't. Finally the dysoft operator is somewhat ambiguously applied to the output of the pairwise Hadamard computation.

### Questions
Please provide a clear and concise mathematical description of the algorithm, preferably with descriptions of the input and output dimensions of the individual operators. More precise definitions and clear descriptions of the algorithm will make this paper much stronger.

 Additionally, do the authors believe this method would be useful for tasks beyond classification? If so, please provide some justification or evidence.

### Soundness
4

### Presentation
3

### Contribution
3
