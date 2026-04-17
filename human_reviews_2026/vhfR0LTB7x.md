# Distilling SNN Students from ANN Teachers via Spiking Neural Architecture Search

- Decision: Reject
- Scores: 4, 4, 4, 4, 6

## Abstract
Bridging the performance gap between Spiking Neural Networks (SNNs) and Artificial Neural Networks (ANNs) under low timesteps remains a critical challenge in the SNN community. Recent work uses either ANN-supervised training or automated architecture design to narrow the gap. However, the combination of ANN-supervised training and SNN architecture search remains unexplored, leaving room for further improvement of SNN performance. To address this, we propose Distilling SNN Students from ANN Teachers via Spiking Neural Architecture Search (DSAS) method. It is a training-free spiking neural architecture search method that leverages pre-trained ANN teachers to discover efficient and high-performance SNNs with few timesteps. Specifically, DSAS employs an evolutionary neural architecture search guided by two novel metrics, i.e., Multi-layer Activation Similarity (MAS) and Threshold-guided Gradient Similarity (TGS). MAS aligns ANN and SNN feature maps, yet TGS ensures gradient alignment while tuning spiking activation thresholds. Experiments demonstrate that DSAS achieves state-of-the-art accuracy with four timesteps on both convolution-based and transformer-based search space, effectively narrowing the performance gap of ANN and SNN. For example, DSAS discovers architectures that achieve 65.50% top-1 accuracy on Tiny-ImageNet and 81.97% on CIFAR-100. Available code: https://anonymous.4open.science/r/DSAS-5764

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces DSAS, a spiking neural network (SNN) architecture search method that leverages pre-trained artificial neural networks (ANNs) to guide the design of high-performance SNNs. The paper aims to narrow the performance gap between SNNs and ANNs, especially when limited to low timesteps. The method proposes two metrics, Multi-layer Activation Similarity (MAS) and Threshold-guided Gradient Similarity (TGS), to guide the search process. The experimental results show improvements over some existing methods.

### Strengths
- The idea of combining ANN-guided learning with neural architecture search is interesting.
- The approach of using representation-based metrics to guide the architecture search rather than relying on iterative training reduces the computational overhead. This is a notable strength, as it significantly cuts down the search cost and improves the scalability of the method.
- The results demonstrate that DSAS performs well across multiple benchmarks, including Tiny-ImageNet and CIFAR-100, achieving competitive accuracy with fewer timesteps. The paper shows that DSAS can improve performance relative to several state-of-the-art methods, which supports the authors' claims about the method's efficacy.

### Weaknesses
- According to Equations 4 and 9, the alignment of activations and gradients is conceptually aimed at minimizing the angular difference, ensuring that the student SNN follows a similar trajectory to the teacher ANN. However, the paper does not adequately explore the optimization implications of this alignment. Could the authors provide further theoretical insights into how DSAS contributes to improving SNN optimization?
- Some details of the paper are difficult to follow. For instance, the meaning and use of the parameter K in TGS are unclear. Additionally, the term "KD" is not defined in the paper. Moreover, there are a few typographical errors: for example, "mateices" should be "matrices" (Line 187) and "Sigmod" should be "Sigmoid" (Line 238). A more thorough examination and clearer presentation of these details are needed. 
- The paper predominantly focuses on a narrow search space that includes standard CNN components. While this choice is understandable, it restricts the exploration of architectures beyond traditional CNN-based designs. This raises the question of whether the search method demonstrates clear advantages over heuristic-based hyperparameter tuning, especially given the relatively small scale of the experiments presented.

### Questions
- Why was PyramidNet specifically chosen as the ANN teacher model for DSAS?

- Is it possible for DSAS to be extended for selecting the optimal type of spiking neuron for SNNs?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The DSAS proposed a unique combination of ANN-supervised training and architecture search. It enhances classification accuracy by employing the Mean Activation Similarity (MAS) metric to effectively align the internal feature representations between the teacher (ANN) and the student (SNN). Furthermore, DSAS introduces the Threshold Gradient Similarity (TGS) metric, which adaptively adjusts the SNN's neuron firing thresholds; this crucial step facilitates the SNN's learning process by aligning its backpropagation steps with those of the ANN.

### Strengths
- The proposed method, DSAS, is noteworthy as its novelty lies in the unique combination of ANN-supervised training and architecture search.
- MAS metric is effectively employed to align the internal feature representations between the teacher (ANN) and student (SNN), thereby enhancing overall classification accuracy.
- TGS metric is introduced to adaptively adjust SNN neuron firing thresholds, facilitating the SNN's learning process by aligning its backpropagation steps with those of the ANN.

### Weaknesses
- The MAS metric overlooks the temporal dynamics inherent to SNNs. By relying solely on the average spike rate, the temporal information is disregarded. Consequently, distinct temporal spike patterns (e.g., '0011' vs. '1100') are the same when using MAS.
- In TGS, I think comparing SNN and ANN gradients is hard because SNNs have the time dimension ($T$). As the time steps increase, the SNN's time-based gradient becomes very different from the standard ANN's gradient, making a direct comparison difficult. How can the author solve the problem?
- The experimental comparison section is deficient. Specifically, a crucial ablation study may miss: How does the performance of the proposed architecture search method compare when the search is performed within the same defined search space but without utilizing ANN pre-trained weights, instead employing an alternative search strategy?
- In Supplemental Material F, the "The evolutionary trajectory" should be the change in accuracy after retraining, rather than merely reporting the fitness function metric. Reporting the fitness metric alone does not sufficiently demonstrate the search's efficacy; only accuracy improvements can validate the search process. Furthermore, a comparison against a random choice baseline should be included.
- The quality of the writing requires substantial revision due to numerous grammatical and expression errors. Specific issues include:
    - The "DSAS" should be introduced by its full name in the Abstract, as it is the first appearance.
    - Acronyms are discouraged in Section 074 captions.
    - In Line 148, the reference to "ANN student" appears erroneous and should likely be "SNN student"?
    - In Line 353, the acronyms "C-10" and "C-100" should be explicitly defined in the figure caption.
    - In Figure 1, the legend "W: Weight" should be corrected to "W: Width," and the legend's overall descriptive detail is insufficient.

### Questions
See weakness

### Soundness
2

### Presentation
1

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
This paper proposes DSAS, a novel method for designing efficient Spiking Neural Networks by integrating knowledge distillation with neural architecture search. Its core innovation lies in introducing a Multi-layer Activation Similarity metric, which guides the search process by measuring the activation similarity between the SNN student and the ANN teacher across multiple intermediate layers. The authors conduct experiments on multiple datasets including Tiny-ImageNet, ImageNet, and CIFAR, demonstrating that DSAS significantly outperforms existing methods in terms of accuracy and parameter efficiency, with low search cost.

### Strengths
1.The paper accurately identifies the shallow problem in existing SNAS methods, where search algorithms tend to favor shallow networks to quickly match the final output of the teacher model.

2.MAS is the central contribution of this paper. By extending alignment from the final output to multi-layer intermediate features, this metric cleverly guides the search direction.

### Weaknesses
1.The experiments are conducted solely on traditional CNNs. Currently, Transformer architectures have become mainstream in fields such as computer vision. To demonstrate the generalizability and state-of-the-art relevance of the proposed method, the authors should include experimental results on converting trained Transformer models from ANN to SNN.

2.In the ImageNet comparison, DSAS uses PyramidNet101 as its teacher model, while many baseline methods use ResNet-34 or ResNet-50. Using a more powerful teacher model may give DSAS an unfair advantage. It is unclear how much of the performance gain is attributable to the superior search method itself versus the stronger teacher, requiring a more rigorous analysis. And how would the performance of DSAS change if its teacher model were replaced with ResNet-34, the same as used by baseline methods.

### Questions
As mentioned in weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel training-free Spiking Neural Architecture Search (SNAS) method, named DSAS. Its core objective is to leverage the knowledge of pre-trained Artificial Neural Network (ANN) teachers to automatically discover high-performance and energy-efficient Spiking Neural Network (SNN) student models under low timestep conditions. The paper includes two main metrics: Multi-layer Activation Similarity (MAS) and Threshold-guided Gradient Similarity (TGS). The experimental section comprehensively covers four static image datasets (ImageNet, Tiny-ImageNet, CIFAR-10, CIFAR-100) and two neuromorphic datasets (CIFAR10-DVS, DVS128-Gesture), with systematic comparisons against various existing distillation and architecture search methods.

### Strengths
The authors conducted extensive experiments across six datasets (with both static and neuromorphic tasks). The evaluation includes diverse aspects of the architecture and model. 

The paper's motivation is clear. The core problem is sharply focused on "how to design efficient SNN architectures that are both high-performing and require few timesteps." The approach of fusing ANN teacher knowledge with automated architecture search provides a solid and reasonable foundation for the research direction.

### Weaknesses
The main concern is that, the use of knowledge distillation (KD) for student network search is not new. The idea of MAS appears quite similar to Relation Similarity Metric in Dong et al. (2023), essentially extending an ANN-based method to the SNN domain. The authors should clarify the key differences between the two activation-similarity mechanisms and highlight the unique challenges, if any, in applying this approach to SNNs.

I recommend reporting the total network design time. Since the pipeline involves both the search stage and the teacher model pretraining, the overall cost should be included to provide a fair comparison.

The methodology for energy evaluation is unclear. Please describe how the energy consumption is calculated to allow the reader to better understand and reproduce the results.

### Questions
See the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a training-free neural architecture search (NAS) method, named DSAS, for distilling a high-performance SNN from a pre-trained ANN. DSAS is an evolutionary NAS approach that introduces a multi-layer activation similarity (MAS) metric to align features between the ANN teacher and the SNN student, and a threshold-guided gradient similarity (TGS) method to enhance the SNN’s approximation of ANN gradients. Experimental results show that SNNs obtained by DSAS achieve competitive performance compared with state-of-the-art SNNs.

### Strengths
- The proposed search method relies on evaluation-based algorithms, resulting in substantially lower search costs compared to existing spiking NAS approaches.

- The paper provides a comprehensive empirical evaluation, including ablation studies, parameter analyses, and comparisons with multiple baselines.

### Weaknesses
- The search framework is primarily designed for spiking CNNs, without consideration of transformer-based architectures. Moreover, the SNN candidates in the search space are identical to those in EMO-SNAS [Song, TEVC 2025], which limits the novelty of the architectural design.

- DSAS performs architecture searches separately for each dataset and achieves competitive results; however, the paper lacks analysis on cross-dataset generalization — for instance, whether an SNN searched on ImageNet could transfer effectively to CIFAR10.

- Since the parameter spaces of ANNs and SNNs differ, it remains unclear how the TGS metric can meaningfully capture gradient alignment between teacher and student networks.

- Some implementation details are missing, such as the surrogate gradient function used for training the searched SNN, and the event data representations for CIFAR10-DVS and DVS128-Gesture (e.g., whether voxel grids are used).

- Typo: Line 240 — “SNN Teacher” should be “SNN Student.

### Questions
Please refer to the points raised in “Weaknesses”.

### Soundness
3

### Presentation
3

### Contribution
3
