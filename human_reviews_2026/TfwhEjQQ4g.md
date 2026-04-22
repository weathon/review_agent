# VLAAR: Vision-Language Attribute-Aware Router for Pedestrian Attribute Recognition

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Pedestrian attribute recognition aims to identify multiple semantic attributes of individuals from visual data, a task critical for surveillance applications. However, existing methods often overlook the heterogeneity of pedestrian attributes and lack mechanisms for effectively modeling inter-attribute relationships. This paper proposes VLAAR, a parameter-efficient fine-tuning method for pedestrian attribute recognition that leverages the mixture-of-experts framework. Building upon a pre-trained CLIP model, our approach employs lightweight expert modules, forming a pool of specialized networks. At its core, our dual-input routing mechanism concurrently processes visual features alongside semantic cues derived from natural language prompts, guiding expert selection effectively. This dynamic routing facilitates the optimal allocation and efficient processing of complex attribute information while preserving computational efficiency. Extensive evaluations on image and video benchmarks demonstrate state-of-the-art performance for multi-label attribute recognition in surveillance and re-identification systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The proposed approach addresses the Pedestrian Attribute Recognition (PAR) problem by adopting a Mixture-of-Experts (MoE) framework that aggregates outputs from multiple models. Because PAR is a multi-label classification task, the suitable models may vary across attributes. Rather than relying on a single model to handle all attributes, the paper introduces expert models that specialize in detecting particular attributes and combines these experts conditioned on the input. A key challenge in MoE is how to select experts and combine them appropriately. To address this, the paper proposes a multimodal design that leverages both text and visual signals, and employs a sparse MoE so that only a small number of relevant experts are activated instead of engaging all experts. The method reduces training cost by training only shallow expert models while keeping the CLIP backbone frozen, avoiding full backbone retraining.

### Strengths
- The paper achieves performance gains by training only a shallow MoE, suggesting an efficient learning scheme. 
- In addition, the simple MoE design provides a flexible architecture that can be easily extended by adding new experts. 
- The proposed method attains the best results among the compared approaches.

### Weaknesses
- Although the MoE strategy shows strong results, extending the MoE mechanism to PAR limits the originality. The MoE is common in other applications.
- The paper proposes three expert types, but it is not clear how these three experts are complementary to one another. 
- There is also a concern that the observed improvements may stem from overfitting to the evaluated datasets; additional generalization tests on the PETA-ZA and RAP-ZS datasets proposed in [R1] would be helpful.

[R1] Jia J, Huang H, Yang W, Chen X, Huang K. Rethinking of pedestrian attribute recognition: Realistic datasets with efficient method. arXiv preprint arXiv:2005.11909. 2020 May 25.

### Questions
- It seems intuitive that the standard expert would excel at capturing global information, while the convolutional experts would be stronger at local cues. Is this the intended design? For the advanced experts, is the multi-layer structure meant to fuse global and local information via cross-region interactions?
- Analyzing, for each attribute, which experts are selected could provide additional insight. (i.e., the distribution between attributes and expert selection)
- How were the video experiments conducted? For a given video, were frames treated as independent images with results aggregated (e.g., voting) to produce a video-level attribute prediction?
- Considering the risk of overfitting, it would be beneficial to validate generalization on the PETA-ZA and RAP-ZS datasets introduced in [R1].

[R1] Jia J, Huang H, Yang W, Chen X, Huang K. Rethinking of pedestrian attribute recognition: Realistic datasets with efficient method. arXiv preprint arXiv:2005.11909. 2020 May 25.

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
The paper proposes a solid vision–language mixture-of-experts framework for multi-label pedestrian attribute recognition. It shows clear architectural novelty and strong empirical gains, but lacks analysis on generalization and training efficiency. Some formulations and ablations could be clarified further.

### Strengths
VLAAR introduces a clear and effective vision–language routing mechanism for expert selection. It achieves strong performance–efficiency trade-offs across both image and video benchmarks. The experiments are comprehensive and well-organized, providing solid empirical support.

### Weaknesses
see Questions.

### Questions
- It would be beneficial to report the statistical uncertainty of the results, for example by including the standard deviation across multiple runs, to make performance comparisons more reliable.

- To give a more complete view of efficiency, please consider adding more resource metrics (e.g., GPU hours, FLOPs) in a table alongside the inference metrics in Table 3.

- For better interpretability, adding t-SNE plots or attention heatmaps that visualize the vision–language gating process (Eq. 2) would help illustrate how the model integrates the two modalities.

- A brief discussion of optimization challenges in sparse routing, such as potential expert imbalance or convergence issues, would also clarify the training dynamics.

- Including a few qualitative failure cases (e.g., occlusion or low-light scenarios) could help highlight current limitations and guide future work.

- Finally, please comment on possible bias inherited from the pre-trained CLIP semantics, especially for demographic attributes, and mention potential mitigation strategies such as prompt regularization or balanced sampling.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes VLAAR, a new parameter-efficient fine-tuning method for pedestrian attribute recognition. By dynamically selecting expert modules through a dual-modality routing mechanism that considers both visual features and natural language semantics, VLAAR effectively addresses attribute heterogeneity and models inter-attribute relationships. This approach ensures efficient allocation of complex attribute information while maintaining computational efficiency.

### Strengths
- This paper presents a novel and well-motivated application of the MoE framework to the task of PAR. The key insight of employing architecturally diverse experts to explicitly address the inherent heterogeneity of attributes and to model their inter-relationships is a significant strength.

- The core idea of a dynamic router that integrates both visual features and semantic language cues for expert selection is innovative.

- As a PEFT method, it achieves strong performance with minimal trainable parameters, offering clear practical benefits for computational resource conservation and real-world deployment.

### Weaknesses
- While the related work section outlines the conceptual differences between VLAAR and existing paradigms like shared or separate experts, the paper would be significantly strengthened by including ablation studies that quantitatively demonstrate the superiority of this proposed architecture. 
- What is the difference between the SPARSE MIXTURE-OF-EXPERTS approach, which selectively activates only a subset of experts to process each input, and existing MOE methods that utilize Top-k Routers?
- The results in Table 1 are impressive, showing that VLAAR with a ViT-B/16 backbone surpasses EVSITP 's performance with a larger ViT-L/14. To better understand the source of these gains, it is crucial to provide a comprehensive ablation study on key datasets like PETA and PA100k. This analysis should dissect the individual contribution of each expert type within the proposed framework.
- The ablation study in Table 5 lacks an analysis of the synergistic effects when different expert modules are combined. An investigation into how and why specific experts are activated for different attributes would provide deeper insights into the model's decision-making process.

### Questions
- There is a discrepancy in the reported result for "PLIP" in Table 1. The value of 88.84 is listed under the Acc metric. However, in the cited reference, this value (88.84) is reported as the F1-score. Could the authors please clarify whether this entry is a data transcription error in the table, or if it represents a reproduced result obtained by the authors through their own experiments?


- Are the variable 'x' in Equation 1 and Equation 4 referring to the same entity? Additionally, the variable 'y' introduced in Equation 3 does not appear in subsequent formulations - could this be clarified?

### Soundness
3

### Presentation
2

### Contribution
2
