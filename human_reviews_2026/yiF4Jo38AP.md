# CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
Source-Free Domain Adaptive Object Detection (SF-DAOD) aims to adapt a detector trained on a labeled source domain to an unlabeled target domain without retaining any source data. Despite recent progress, most popular approaches focus on tuning pseudo-label thresholds or refining the teacher-student framework, while overlooking object-level structural cues within cross-domain data. In this work, we present CGSA, the first framework that brings Object-Centric Learning (OCL) into SF-DAOD by integrating slot-aware adaptation into the DETR-based detector. Specifically, our approach integrates a Hierarchical Slot Awareness (HSA) module into the detector to progressively disentangle images into slot representations that act as visual priors. These slots are then guided toward class semantics via a Class-Guided Slot Contrast (CGSC) module, maintaining semantic consistency and prompting domain-invariant adaptation. Extensive experiments on multiple cross-domain datasets demonstrate that our approach outperforms previous SF-DAOD methods, with theoretical derivations and experimental analysis further demonstrating the effectiveness of the proposed components and the framework, thereby indicating the promise of object-centric design in privacy-sensitive adaptation scenarios. Code is released at https://github.com/Michael-McQueen/CGSA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes CGSA, a novel source-free domain adaptive object detection framework that integrates object-centric learning into DETR-based models. It introduces two modules: Hierarchical Slot Awareness, which decomposes images into slot-based structural priors, and Class-Guided Slot Contrast, which aligns slots with class semantics through contrastive learning. The approach enables domain-invariant adaptation without source data. Experiments on multiple benchmarks show consistent and significant improvements over existing methods, supported by theoretical risk descent analysis.

### Strengths
1. The paper introduces CGSA, the first framework to combine Object-Centric Learning with Source-Free Domain Adaptive Object Detection, and integrates slot-based structural priors and class-guided contrastive alignment reasonably to achieve domain-invariant adaptation.
2. Extensive experiments show that CGSA significantly outperforms prior methods, and the authors also provide theoretical analysis demonstrating risk reduction and stable convergence during adaptation.

### Weaknesses
1. Nevertheless, the proposed method remains highly dependent on the DETR framework, showing limited generalizability. Evaluating its effectiveness on other architectures, such as Faster R-CNN, is still desireable.
2. There is no intuitive visualization or analysis of slot decomposition process.
3. The risk analysis is mathematically sound but lacks empirical validation of key variables in authors' description, e.g., cosine margin gain or reconstruction consistency.

### Questions
1. How sensitive is CGSA to the number of slots (n=5) or slot hierarchy depth?
2. What is the computational overhead compared to standard DETR?

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
4

### Summary
This paper proposes integrating object-centric learning into DETR-based detectors for source-free domain adaptive object detection. Specifically, the Hierarchical Slot Awareness (HAS) module decomposes an image into a set of slots, which are then fused with queries to inject object-level visual priors into the object decoder. Additionally, a Class-Guided Slot Contrast (CGSC) mechanism is introduced to guide the slots toward domain-invariant yet class-relevant object features by leveraging contrastive learning with class prototypes. Extensive experiments on multiple datasets demonstrate the effectiveness of the proposed method.

### Strengths
1. The method achieves state-of-the-art performance across multiple datasets.

2. The authors provide a solid theoretical analysis explaining why slot-based features can offer domain-invariant priors.

### Weaknesses
1. The HAS module introduces additional parameters and computation overhead. It would be beneficial to provide a comparison of speed and parameter size before and after adding HAS.

2. The authors did not provide the performance of the source-only model after adding HAS. This omission makes it unclear whether the performance improvement stems from a stronger source-only model or from HAS enhancing the adaptation capability.

3. HAS requires self-supervised pretraining on the COCO dataset. It would be useful to discuss the importance of this pretraining step.

### Questions
1. Why is CGSC not used during training on the source data?
2. Can this method be applied to source-available domain adaptive object detection (DAOD)?

### Soundness
3

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
This paper proposes a framework called CGSA, specifically designed for source-free domain adaptive object detection (SFA-OD). CGSA embeds object-centric slot representations into DETR queries, progressively disentangles target images into coarse-to-fine slots and aligns slot prototypes with online class prototypes. Experiments show the effectiveness of the proposed CGSA.

### Strengths
* The paper is easy to follow.
* Using a slot-aware framework for object-level alignment is a reasonable approach.
* Experiments on five cross-domain shows the effectiveness of the proposed method.

### Weaknesses
1. The performance of the base model should be reported(e.g., RT-DETR) for better evaluation.
2. As shown in Figure 3, the HSA module adapts the pre-trained DINO model. Therefore, the additional computation overhead needs to be analyzed.
3. Since the method uses a pre-trained model to inject feature knowledge, it is recommended to add some comparative introductions with existing methods that use VLM for knowledge injection, such as [1][2]. And recent DETR-based SFOD methods should also be discussed, such as [3].

[1] Da-ada: Learning domain-aware adapter for domain adaptive object detection. NeurIPS 2024

[2] SEEN-DA: SEmantic ENtropy guided Domain-aware Attention for Domain Adaptive Object Detection. CVPR 2025

[3] Source-Free Object Detection with Detection Transformer. IEEE TIP 2025

### Questions
Please refer to Weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the problem of Source-Free Domain Adaptive Object Detection (SF-DAOD), where a detector trained on a labeled source domain is adapted to an unlabeled target domain without accessing source data during adaptation. The authors argue that existing SF-DAOD approaches neglect object-level structural information across domains. To address this issue, they propose a new framework named CGSA, which introduces Object-Centric Learning (OCL) into SF-DAOD through slot-aware adaptation. The framework contains two main components: (1) Hierarchical Slot Awareness (HSA), which disentangles image features into slot representations; and (2) Class-Guided Slot Contrast (CGSC), which enhances semantic consistency and promotes domain-invariant adaptation. Experiments on five cross-domain object detection benchmarks show that CGSA achieves consistent performance gains compared to prior SF-DAOD methods. The paper also provides some theoretical analysis to justify the generalization of the proposed approach.

### Strengths
The authors conduct experiments on multiple benchmark datasets, showing that their method achieves better performance than previous approaches.

The paper includes theoretical discussions supporting the generalization ability of the proposed method.

### Weaknesses
The paper’s description of the proposed method, especially the role of slot attention, is unclear. From the current presentation, it appears that slot attention is applied to the queries in DETR. However, it is not clear how this differs in essence from standard attention mechanisms. The authors should provide a clear comparison or ablation study to demonstrate why slot attention is necessary in this context.

The paper claims that slot attention helps capture object-level features. However, the provided visualization (Figure G.1 in the supplementary materials) does not clearly show the effectiveness of slot-based decomposition. The authors should include more convincing visual evidence or quantitative analysis to illustrate how slot representations correspond to distinct object structures.

The proposed method is built on DETR, while several compared SF-DAOD methods are based on Faster R-CNN. Since DETR generally outperforms Faster R-CNN even without adaptation, the comparison may not be fully fair. The authors are encouraged to implement their framework on Faster R-CNN to ensure a fair comparison and isolate the improvement brought by the proposed adaptation mechanism from that of the detector architecture itself.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2
