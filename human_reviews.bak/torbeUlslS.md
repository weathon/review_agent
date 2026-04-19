# Rethinking Multiple-Instance Learning From Feature Space to Probability Space

- Decision: Accept (Poster)
- Scores: 8, 6, 6

## Abstract
Multiple-instance learning (MIL) was initially proposed to identify key instances within a set (bag) of instances when only one bag-level label is provided. Current deep MIL models mostly solve multi-instance problem in feature space. Nevertheless, with the increasing complexity of data, we found this paradigm faces significant risks in representation learning stage, which could lead to algorithm degradation in deep MIL models. We speculate that the degradation issue stems from the persistent drift of instances in feature space during learning. In this paper, we propose a novel Probability-Space MIL network (PSMIL) as a countermeasure. In PSMIL, a self-training alignment strategy is introduced in probability space to cope with the drift problem in feature space, and the alignment target objective is proven mathematically optimal. Furthermore, we reveal that the widely-used attention-based pooling mechanism in current deep MIL models is easily affected by the perturbation in feature space and further introduce an alternative called probability-space attention pooling. It effectively captures the key instance in each bag from feature space to probability space, and further eliminates the impact of selection drift in the pooling stage. To summarize, PSMIL seeks to solve a MIL problem in probability space rather than feature space. Experimental results illustrate that PSMIL could potentially achieve performance close to supervised learning level in complex tasks (gap within 5\%), with the incremental alignment in propability space bring more than 19\% accuracy improvements for current existing mainstream models in simulated CIFAR datasets. For  existing publicly available MIL benchmarks/datasets, attention in probability space also achieves competitive performance to the state-of-the-art deep MIL models. Codes are available at \url{https://github.com/LMBDA-design/PSAMIL}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The author contends that deep MIL models suffer from a drift in the representation of instances during training. This drift, which occurs in the feature space, negatively impacts the model's performance. To address this issue, the paper proposes the Probability-Space MIL network. This network introduces probability-space attention pooling and a probability-space alignment objective. By transforming instances from the feature space to the probability space using class prototypes, the network mitigates the effects of feature drift and consequently enhances the overall performance of the MIL model.

### Strengths
1-	The problem is described clearly.

2-	The investigated problem is important.

### Weaknesses
1-	Limited downstream evaluation. 

2-	The PSMIL approach seems similar to other approaches that use additional attention pooling, such as DSMIL [1] and DTFDMIL [2].

3-	 The paper does not discuss the computational complexity and scalability of the proposed approach in detail, which could be a concern for large-scale applications.

4-	Limited Ablation Studies: While the paper includes some ablation studies, more extensive ablations could strengthen the claims.

[1] Bin Li, Yin Li, and KevinWEliceiri. Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14318–14328, 2021.

[2] Hongrun Zhang, Yanda Meng, Yitian Zhao, Yihong Qiao, Xiaoyun Yang, Sarah E Coupland, and Yalin Zheng. Dtfd-mil: Double-tier feature distillation multiple instance learning for histopathology whole slide image classification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 18802–18812, 2022.

### Questions
1- Use another evaluation method, such as localization on WSI datasets, and compare it with the previous approaches. 

2-Illustrate the key differences between the proposed approach, DSMIL [1] and DTFDMIL [2].

3-The approach uses additional probability space with the class prototype. Thus, it is worth comparing the proposed approach's scalability and complexity against the previous approaches. 

4-We encourage the authors to do more ablation studies for the approach and see how that will impact the performance of the proposed approach. For example, manipulate the hyper-parameter in the equation (11).

### Soundness
3

### Presentation
3

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
This paper addresses the issue of drift in instance representation learning within multiple instance learning (MIL) models by introducing Probability-Space Multiple Instance Learning (PSMIL). In PSMIL, the authors present two main strategies: probability-space attention pooling and a probability-space alignment objective. Experimental results show that the proposed approach improves performance on challenging MIL tasks, achieving results closer to supervised learning standards, while maintaining competitive performance on bag-level classification benchmarks.

### Strengths
+ The approach of addressing drift issue in instance representation learning through pseudo-label inference-based probability-space alignment appears novel to me.
+ The authors have provided enough visualizations (e.g., Figure 2) along with the ablation study in Table 3 to justify the effectiveness of the proposed components PSAtt, and PSAli.
+ The strong performance of existing techniques like DTFDMIL is well-explained in the context of the FMNIST dataset.

### Weaknesses
+ The authors should ensure all symbols used in the equations are clearly defined and explained. For example, in Equation 5, it is unclear how ${p^{^\rightarrow}_s}(x^\prime)$i s derived. Similarly, in Equation 2, the steps from layerwise differentiation to decomposition and recombination require further clarification.
+ My main concern is the evaluation, as PS-MIL is tested only on synthetic datasets without including any challenging, real-world datasets. Furthermore,  in Table 2, PSMIL performs significantly lower on the FMNIST dataset compared to DTFDMIL, which requires further indepth justification.
+ In the ablation study (Table 3), the standalone contribution of PSAtt, independent of PSAli, is unclear. Including results for PSAtt alone would help clarify its individual impact on overall performance.
+ The impact of hyperparameters, such as $\lambda$ in Equation 9, is not explored. A detailed study on the effect of  $\lambda$ would offer insights into the relative importance of $L_{\text{bag}}$ and $L_{\text{ins}}$ on performance.
+ Video anomaly detection is a key application of the MIL approach [1, 2, 3], providing challenging, real-world datasets. The authors should demonstrate the performance of their method on these datasets in comparison with established baselines [1, 2, 3]. This would strengthen the paper, as only simpler synthetic datasets are currently considered, which do not fully represent real-world MIL challenges.
+ The process of pseudo-label inference in Equation 5 is unclear. Providing additional explanation beyond the equations would aid readers in understanding how pseudo-labels are inferred.

**References**
1. Wu et al. "VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection". AAAI2024. 
2. Tian et al. “Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning”. ICCV2021
3. Sultani et al. “Real-world Anomaly Detection in Surveillance Videos”. CVPR2018

### Questions
Please refer to Weaknesses section

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work observes that the core issues leading to performance degradation in MIL with Attention-based methods stem from selection shift and feature shift. To address these problems, it introduces prototype learning and feature alignment.

### Strengths
This work identifies the core issues leading to failures in Attention-based MIL approaches and proposes targeted solutions. It also provides clear theoretical proofs and achieves superior performance.

### Weaknesses
The article employs two core technologies to address the aforementioned issues; however, there are aspects of these technologies that require further clarification. Firstly, introducing prototypes to resolve selection bias is a technique that has been mentioned in "Rethinking Multiple Instance Learning for Whole Slide Image Classification: A Good Instance Classifier is All You Need." Secondly, feature alignment appears to be a specific application of contrastive learning (either bringing enhanced image features closer or aligning probability spaces). If this process were placed before the overall model training (e.g., as a pre-training step), would it achieve similar effects?
Additionally, it is necessary to highlight the differences between this approach and traditional contrastive learning methods.

### Questions
The authors need to supplement their work by addressing several issues mentioned in the aforementioned weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
