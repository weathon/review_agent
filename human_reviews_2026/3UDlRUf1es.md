# DeCo-DETR: Decoupled Cognition DETR for efficient Open-Vocabulary Object Detection

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Open-Vocabulary Object Detection (OVOD) plays a critical role in autonomous driving and human-computer interaction by enabling perception beyond closed-set categories. However, current approaches predominantly rely on multimodal fusion, facing dual limitations: multimodal fusion methods incur heavy computational overhead from text encoders, while task-coupled designs compromise between detection precision and open-world generalization. To address these challenges, we propose Decoupled Cognition DETR, a vision framework featuring a three-stage cognitive distillation mechanism: Dynamic Hierarchical Concept Pool constructs self-evolving concept prototypes using LLaVA-generated region descriptions filtered by CLIP alignment, aiming to replace costly text encoders and reduce computational overhead; Hierarchical Knowledge Distillation decouples visual-semantic space mapping via prototype-centric projection, avoiding task coupling to enhance open-world generalization; Parametric Decoupling Training coordinates localization and cognition through dual-stream gradient isolation, further optimizing detection precision. Extensive experiments on the common OVOD evaluation protocol demonstrated that DeCo-DETR achieves state-of-the-art performance compared to existing OVOD methods. It provides a new paradigm for extending OVOD to more real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This article introduces DeCo-DETR (Decoupled Cognition DETR), a novel open-vocabulary object detection (OVOD) framework designed to overcome the limitations of existing methods, particularly high computational overhead from text encoders and a trade-off between detection precision and open-world generalization.

### Strengths
This paper identifies certain issues in current large model-based OVD models, proposes a viable approach to address them, and conducts comprehensive experiments that can serve as a valuable reference for future research.

### Weaknesses
The description on issues and challenges lacks persuasiveness, and the connection between the proposed method and the challenges it aims to address is weak. Furthermore, the explanation of the methodology lacks clarity, and the presentation of figures, tables, as well as certain parts of the writing, appears somewhat casual. Overall, the rigor of the work needs to be enhanced.

### Questions
1. In Section 1, Paragraph 1, the authors assert that “the emergence of large language models (LLMs) has significantly enhanced detector generalization by providing richer and more nuanced semantic supervision,” yet no supporting citation is provided. Similarly, the authors claim that existing distillation methods face latency and generalization trade-offs, again without citing any references. It is recommended that relevant citations or quantitative results be added to substantiate these claims.

2. In Section 1, Paragraph 2, the authors state that multimodal fusion designs inherently involve compromises, but they provide no explanation, citation, or quantitative evidence to support this claim. As a result, the argument lacks persuasiveness.

3. In Section 1, Paragraph 3, the DHCP module is introduced without clarifying how it addresses the challenge outlined in the preceding paragraph. Moreover, the mention of “momentum updates with attention weighting” is confusing, and the authors should elaborate on its relationship with DHCP.

4. A similar issue arises in Section 1, Paragraph 4, where the connection between the proposed design and the previously mentioned challenges is not clearly articulated.

5. Due to the issues raised in points 1–4, the first contribution claim in Section 1—“We reveal two critical flaws in existing open-vocabulary detection”—does not appear sufficiently supported.

6. In the second contribution statement in Section 1, the authors mention “dynamic concept pooling” and “hierarchical distillation and parametric isolation mechanisms” as solutions to the identified challenges. However, these terms appear only in this section and are likely referring to “DHCP” and “Hi-Know PDA” introduced later in the manuscript. The current wording is ambiguous and may cause confusion.

7. In Figure 1, the bottom-left subfigure seems unnecessary. Additionally, the caption indicates that Hi-Know PDA is part of the framework diagram, but it is not visually represented.

8. Several issues are present in Section 3. For instance, the “spectral clustering-based hierarchical compression algorithm” is mentioned for the first time in Section 3.1, yet its specific operation is not explained, nor is it illustrated in any figure or pseudocode.

9. Certain references are missing throughout the manuscript. For example, LLaVA and DBSCAN are not cited.

10. The results in Table 3 are unclear. DeCo-DETR does not appear to be the most efficient method according to the table. Moreover, comparing it with Deformable-DETR is confusing, as Deformable-DETR is not an open-vocabulary object detection (OVOD) method nor serves as the baseline for DeCo-DETR.

11. In Section 4.3, Table 2, the benchmarks V-OVD, G-OVD, C-OVD, and WS-OVD are neither introduced nor cited. While these may originate from OVCOCO (Bansal et al., 2018), they are not referenced in the paper.

### Soundness
2

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
This paper proposes DeCo-DETR, a framework for open-vocabulary object detection that aims to eliminate text encoder dependency during inference while improving generalization. The approach introduces three main components: (1) Dynamic Hierarchical Concept Pool (DHCP) that constructs visual prototypes using LLaVA-generated descriptions filtered by CLIP, (2) Hierarchical Knowledge Distillation (Hi-Know DPA) for visual-semantic alignment, and (3) Parametric Decoupling Training (PD-DuGi) with gradient isolation. The method achieves competitive results on OV-COCO and OV-LVIS benchmarks with low inference latency.

### Strengths
- Novel approach to eliminating text encoder dependency. The Dynamic Hierarchical Concept Pool is an interesting idea that pre-computes and maintains visual prototypes, eliminating the need for text encoders at inference time. 
- The results show strong empirical performance. The experiments demonstrates consistent improvements across multiple benchmarks and settings, achieving state-of-the-art result while maintaining reasonable computational cost.

### Weaknesses
- There exists several misleading claims. 1) The paper claims to eliminate "multimodal fusion" but DHCP construction still heavily relies on LLaVA and CLIP. 2) The framework is only vision-only at inference, not overall.
- Lack in-depth ablations. The ablation study in table 4 only compares 2 configurations, not isolating individual component contributions.

### Questions
N/A

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
5

### Summary
This manuscript targets open-vocabulary object detection (OVOD) and proposes DeCo-DETR, a three-stage decoupled cognition pipeline.

### Strengths
[1] DHCP: a dynamic, hierarchical concept pool that generates region descriptions with LLaVA and filters them with CLIP to build semantic prototypes (thus removing the text encoder at test time); 

[2] Hi-Know DPA: hierarchical knowledge distillation that projects decoder queries into the prototype space for prototype attention/alignment; 

[3] Parametric Decoupling Training: a dual-stream gradient isolation scheme that routes localization and semantic-alignment gradients separately. 

[4] DeCo-DETR reports strong gains (+3.5 to +7.2 AP on novel classes) while keeping inference at 135 ms/image.

### Weaknesses
[1 ]Missing scale ablation on queries and prototypes：There is no systematic ablation for the number of decoder queries N=2000and the total number of prototypes M=M_1+M_2(with M_1=1203 coarse anchors and M_2=4800 fine units). In DETR-style models, increasing the number of queries and prototypes generally improves accuracy, but drives memory usage up linearly;

[2] Fairness of the efficiency comparison (Table 3)：One of the paper’s main claims is removing multimodal computation at test time. For fairness, Table 3 should include methods with similar accuracy and comparable settings. As it stands, Table 3 under-represents strong multimodal fusion baselines near DeCo-DETR accuracy, making the efficiency story less convincing;

[3] Isolating the independent contribution of Parametric Decoupling Training：Section 4.4 shows that the cosine annealing weight adds ~+1.6 AP_50, but this does not quantify the benefit of Parametric Decoupling Training itself.

### Questions
See above.

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
4

### Summary
This paper proposes DeCo-DETR, an open-vocabulary object detector that removes dependence on text encoders and improves both precision and generalization. It introduces a three-stage cognitive distillation mechanism—a dynamic concept pool from LLaVA-CLIP filtering, hierarchical knowledge distillation for decoupled visual-semantic mapping, and parametric dual-stream training for coordinated localization and recognition.

### Strengths
1.	The paper introduces a three-stage cognitive distillation framework (DHCP, Hi-Know DPA, PD-DuGi) that provides a conceptually coherent and interpretable alternative to conventional multimodal fusion.
2.	The model achieves improvements in both detection accuracy and inference efficiency, effectively reducing computational cost while maintaining strong open-vocabulary generalization.

### Weaknesses
1.	Main weakness – Metric inconsistency (Table 1). The reported AP50 values are higher than both APNovel50 and APBase50, which violates standard OVOD evaluation logic. Since AP50 includes both base and novel categories, its score should theoretically lie between them. This inconsistency casts doubt on the correctness of the evaluation protocol or result reporting, and substantially weakens the empirical credibility of the paper’s main claims.
2.	Incomplete ablation (Table 4). Table 4 is missing key variants and does not include an ablation isolating the proposed Dual-stream Gradient Isolation Mechanism, leaving its effectiveness unverified.

### Questions
1.	The manuscript implicitly assumes the student prototypes A and teacher prototypes P share a one-to-one correspondence via the same M, but it is unclear how this mapping is defined or established, since A results from unsupervised clustering and P is text-derived.

### Soundness
2

### Presentation
2

### Contribution
2
