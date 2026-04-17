# StPR: Spatiotemporal Preservation and Routing for Exemplar-Free Video Class-Incremental Learning

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Video Class-Incremental Learning (VCIL) seeks to develop models that continuously learn new action categories over time without forgetting previously acquired knowledge. Unlike traditional Class-Incremental Learning (CIL), VCIL introduces the added complexity of spatiotemporal structures, making it particularly challenging to mitigate catastrophic forgetting while effectively capturing both frame-shared semantics and temporal dynamics. Existing approaches either rely on exemplar rehearsal, raising concerns over memory and privacy, or adapt static image-based methods that neglect temporal modeling. To address these limitations, we propose Spatiotemporal Preservation and Routing (StPR), a unified and exemplar-free VCIL framework that explicitly disentangles and preserves spatiotemporal information. We begin by introducing Frame-Shared Semantics Distillation (FSSD), which identifies semantically stable and meaningful channels by jointly considering channel-wise sensitivity and classification contribution. By selectively regularizing these important semantic channels, FSSD preserves prior knowledge while allowing for adaptation. Building on this preserved semantic space, we further design a Temporal Decomposition-based Mixture-of-Experts (TD-MoE), which dynamically routes task-specific experts according to temporal dynamics, thereby enabling inference without task IDs or stored exemplars. Through the synergy of FSSD and TD-MoE, StPR progressively leverages spatial semantics and temporal dynamics, culminating in a unified, exemplar-free VCIL framework. Extensive experiments on UCF101, HMDB51, SSv2 and Kinetics400 show that our method outperforms existing baselines while offering improved interpretability and efficiency in VCIL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The topic of this paper is about video class-incremental learning(VCIL). The authors propose an exemplar-free VCIL framework with the spatio-temporal Preservation Routing(STPR) mechanism. It consists of a frame-shared semantics distillation(FSSD) and a Temporal Decomposition-based Mixture-of-Experts(TD-MoE) strategy. The STPR framework has been evaluated on several datasets.

### Strengths
+ The idea of exploring temporal-based MoE in VCIL is interesting.
+ The performance of the proposed method StPR outperfoms that of other methods by a margin on several datasets.

### Weaknesses
- Concerns about the FSSD module. The proposed FSSD strategy is a plug-in method can be integrated to other VCIL methods. To show the effectivess of FSSD, can it be evaluated on different VCIL baselines? The authors have conducted ablation studies in Table 4 and the performance with only employing the FSSD method is needed.

- The presentation can be improved. For example, there are some minor spelling mistakes. “suppl.”->supplemental. The formulation “$A^b$” is not presented in Figure 2 of the overview framework.

- Lack of some important comparisons. The idea of decoupling spatiotemporal knowledge in VCIL has been discussed in previous methods (e.g., [a][b] ). Compare to these works and make a discussion in the related work. Besides, some works published in 2025(e.g., [b]-[c]) can be compared and discussed. For instance, the proposed method can utilize the complex large-scale video continual learning benchamrks [c] for evaluation.

[a] When Video Classification Meets Incremental Classes, MM2021

[b] Learning Conditional Space-Time Prompt Distributions for Video Class-Incremental Learning, CVPR 2025

[c] CRAM:Large-scale Video Continual Learning with Bootstrapped Compression, ICCV 2025

### Questions
My major concerns are inclued in the above weaknesses.

Q1. Different parameter-efficient finetuning(PEFT) strategies(e.g., prompt, adapter, MoE) have been utilized in VCIL scenarios. How to select appropriate PEFT strategy in different practical VCIL scenarios?

### Soundness
3

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
This paper introduces Spatiotemporal Preservation and Routing (StPR), a novel mechanism designed to tackle the challenging problem of Video Class-Incremental Learning (VCIL) in an exemplar-free (memory-free) setting. The proposed method explicitly disentangles and preserves critical spatial and temporal representations, effectively mitigating catastrophic forgetting while maintaining adaptability. StPR consists of two key components: Frame-Shared Semantics Distillation (FSSD) and the Temporal Decomposition-based Mixture of Experts (TD-MoE). FSSD identifies and preserves semantically stable and meaningful spatial channels by jointly considering channel-wise sensitivity and classification contribution. TD-MoE dynamically routes task-specific spatiotemporal experts based on decomposed temporal dynamics, enabling adaptive learning without task identifiers or stored exemplars. Experimental results demonstrate that this memory-free approach achieves superior performance compared to rehearsal-based baselines such as BiC and iCaRL.

### Strengths
1. This paper introduces an effective rehearsal free method that can obtain state-of-the-art performance in different benchmarks like vCLIMB and TCD, outperforming both exemplar-based (e.g., FrameMaker, HCE) and exemplar-free (e.g., STSP, ST-Prompt) methods.
2. Temporal Decomposition-based Mixture-of-Experts (TD-MoE) provides a routing based on temporal residuals and attention-weighted deviations which allows the inference without task IDs.
3. They provide a well-designed distillation loss FSSD that leverages the Fisher information and classification contribution to compute per-channel relevance which help preserving model stability-plasticity. 
4. The paper provides extensive evaluations across multiple benchmarks and metrics, along with a thorough ablation study isolating the contributions of each component.

### Weaknesses
1. The proposed TD-MoE instantiates a separate expert for each new task. Consequently, both memory and computation costs are expected to scale linearly with the number of tasks. The paper does not provide an explicit analysis or discussion on this scalability issue, nor on potential mitigation strategies.
2. It would be valuable to understand the limitations of the TD-MoE in temporal challenging datasets like SSv2. For instances, analizing the forgetting of the model with and without the TD-MoE module when the number of tasks increase.

### Questions
1. How does memory and compute cost scale with the number of tasks? Do you freeze old experts or allow fine-tuning?
2. How does FSSD compare to gradient-based importance as a measure of stability? 
3. Can you discuss the different between FSSD and the pricipales of EWC, which also use the Fisher information?

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
This paper proposes StPR (Spatiotemporal Preservation and Routing), a unified, exemplar-free framework for Video Class-Incremental Learning (VCIL). The key challenge in VCIL is mitigating catastrophic forgetting while preserving both frame-shared spatial semantics and temporal dynamics across incrementally arriving action classes—without storing past video exemplars due to memory/privacy constraints.

To address this, the authors introduce two core components:

Frame-Shared Semantics Distillation (FSSD): A channel-wise importance-aware distillation mechanism that preserves semantically stable and classification-relevant spatial features by combining semantic sensitivity (via Fisher Information) and classification contribution (via cosine alignment with CLIP text embeddings).

Temporal Decomposition-based Mixture-of-Experts (TD-MoE): A task-agnostic routing mechanism that decomposes spatiotemporal features into static (mean) and dynamic (residual) components. At inference, it routes inputs to task-specific temporal experts based solely on temporal dynamics, eliminating the need for task IDs or stored exemplars.

The framework is built on a frozen CLIP ViT backbone with lightweight adapters and a small spatiotemporal encoder per task. Experiments on UCF101, HMDB51, SSv2, and Kinetics400 under standard VCIL benchmarks (TCD and vCLIMB) show state-of-the-art performance, outperforming both exemplar-based and exemplar-free baselines. Ablation studies and analyses support the design choices.

### Strengths
Strengths

Originality: The disentanglement of frame-shared semantics and temporal dynamics for VCIL is novel. While MoE and distillation exist in CIL, their adaptation to video via temporal decomposition and semantic importance weighting is creative and domain-aware.

Quality: Rigorous experiments with strong baselines, multiple datasets, and thorough ablations. The use of CLIP aligns with modern vision-language trends.

Clarity: The framework is easy to follow (Figure 2), and algorithms are provided (Appendix A). The writing is concise and technical depth is balanced well.

Significance: Addresses a critical gap in VCIL—exemplar-free learning with temporal modeling. The method is efficient (low FLOPs), scalable, and applicable to real-world scenarios like edge devices (mentioned in conclusion).

### Weaknesses
Task-Specific Expert Scaling: TD-MoE allocates one spatiotemporal encoder per task. While efficient per expert (~9M params), this leads to linear growth in parameters with the number of tasks (e.g., 10 tasks → ~90M extra params). This may limit scalability in long-task sequences. The paper does not discuss parameter efficiency in the long run or compare total model size vs. baselines like L2P or ST-Prompt.

Dependence on CLIP: The method relies heavily on frozen CLIP features. While common, this raises questions about generalizability to non-pretrained or non-ViT backbones. A brief experiment with a non-CLIP backbone (e.g., VideoMAE, TimeSformer) would strengthen claims about framework generality.

Temporal Anchor Construction: During inference, TD-MoE uses class-wise mean temporal features (“anchors”) from the current task. This assumes access to current-task data during inference to build anchors. Clarification is needed: are anchors computed once after training and stored? If so, does this constitute a form of “exemplar” (albeit aggregated)? The paper claims “exemplar-free,” but storing per-class statistics could be seen as a lightweight memory buffer.

Kinetics-400 Results: On Kinetics-400 (Table 3), StPR achieves higher final accuracy but also higher BWF than some exemplar-based methods (e.g., CSTA). This trade-off should be discussed—does higher accuracy come at the cost of more forgetting on earlier tasks?

### Questions
Scalability: How does StPR scale to 50+ incremental tasks? Is there a plan to compress or merge experts (e.g., via expert pruning or clustering) to avoid linear parameter growth?

Anchor Storage: Are the temporal anchors  stored permanently after training each task? If yes, how much memory do they consume? Does this violate the “exemplar-free” claim, even if minimally?

Backbone Generality: Have you tested StPR with non-CLIP backbones (e.g., VideoSwin, TimeSformer)? Is the performance gain primarily due to CLIP’s strong priors or the StPR mechanism itself?

Task Boundaries: TD-MoE assumes known task boundaries during training (to assign experts). How would StPR handle task-agnostic or online VCIL where task boundaries are unknown?

Comparison to Prompt-Based Methods: ST-Prompt† (Pei et al., 2023) also uses spatiotemporal prompting. How does TD-MoE fundamentally differ? Is the gain due to routing or the disentanglement design?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the challenges of catastrophic forgetting and spatiotemporal modeling in Video Class-Incremental Learning (VCIL) by proposing an exemplar-free framework named StPR. Its contributions are twofold: 1）It proposes a Frame-Shared Semantics Distillation method (FSSD) that preserves frame-shared, semantically aligned spatial channels through semantic importance-aware regularization, optimizing the stability-plasticity trade-off in continual learning; 2）It designs a Temporal Decomposition based Mixture-of-Experts strategy (TD-MoE) that decomposes spatiotemporal features and uses temporal dynamics for expert combination, enabling task-id-free and dynamic adaptation.

### Strengths
1) The paper proposes an exemplar-free setting, which avoids the memory and privacy issues associated with storing historical data, making it more practical for real-world applications.
2) The theoretical analysis is substantial, providing derivations for the channel importance metrics (Fisher Information, classification contribution) used in FSSD.
3) The experimental analysis is thorough, validating the method's effectiveness across multiple datasets and various task partitions, and includes comprehensive ablation studies and analysis.

### Weaknesses
1) Line 70: "reuses these decomposed components to enhance the model’s ability to adapt continually, thereby reducing forgetting without storing extensive exemplars." Is this statement problematic, as the subsequent text does not mention reusing these decomposed components? Furthermore, the claim that decomposed components help enhance adaptability is also debatable.

2) Why are channels with high semantic importance preserved? The semantic importance from a previous session does not necessarily guarantee importance in subsequent sessions.

3) If the mixture-of-experts system can effectively address catastrophic forgetting and adaptive learning, why is the distillation module necessary? Moreover, mixture-of-experts has already been used in class-incremental learning[1], and no significant distinction from your method is evident. 
[1] Yu J, Zhuge Y, Zhang L, et al. Boosting continual learning of vision-language models via mixture-of-experts adapters[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 23219-23230.

4) On the Kinetics-400 dataset, there is a lack of comparison with results from the latest top-tier conference papers.

5) It is recommended to supplement experiments with longer video sequences (e.g., 16 or 32 frames) or higher frame rate inputs to analyze TD-MoE's capability in long-term temporal modeling and its impact on computational cost.

### Questions
Seen above.

### Soundness
2

### Presentation
3

### Contribution
2
