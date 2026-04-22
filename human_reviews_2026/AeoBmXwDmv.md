# SPKLIP: Aligning Spike Video Streams with Natural Language

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 2

## Abstract
Spike cameras offer unique sensing capabilities but their sparse, asynchronous output challenges semantic understanding, especially for Spike Video-Language Alignment (Spike-VLA) where models like CLIP underperform due to modality mismatch. We introduce SPKLIP, the first architecture specifically for Spike-VLA. SPKLIP employs a hierarchical spike feature extractor that adaptively models multi-scale temporal dynamics in event streams, and uses spike-text contrastive learning to directly align spike video with language, enabling effective few-shot learning. A full-spiking visual encoder variant, integrating SNN components into our pipeline, demonstrates enhanced energy efficiency. Experiments show state-of-the-art performance on benchmark spike datasets and strong few-shot generalization on a newly contributed real-world dataset. SPKLIP's energy efficiency highlights its potential for neuromorphic deployment, advancing event-based multimodal research. The source code and dataset are available at [link removed for anonymity].

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This study addresses the challenge of aligning sparse and asynchronous spike video outputs from spike cameras with natural language by proposing SPKLIP, the first architecture specifically designed for Spike Video-Language Alignment (Spike-VLA). SPKLIP adaptively models multi-scale temporal dynamics via a Hierarchical Spike Feature Extractor (HSFE), achieves direct cross-modal alignment through Spike-Text Contrastive Learning (STCL), and designs a Full-Spiking Visual Encoder (FSVE) to enhance energy efficiency. Experiments demonstrate that SPKLIP achieves state-of-the-art (SOTA) performance on benchmark datasets such as HMDB51-S and UCF101-S, exhibits strong few-shot generalization on a newly constructed real-world dataset, and provides potential for neuromorphic deployment with its energy efficiency.

### Strengths
1. It innovatively fills a gap in the field by proposing the first end-to-end architecture dedicated to Spike-VLA, effectively resolving the modality mismatch issue of traditional vision-language models on spike data.
2. The design of core components is highly targeted: HSFE adapts to the sparse and asynchronous characteristics of spike data, while STCL enables direct alignment between spike videos and text, and their synergy enhances cross-modal semantic understanding.
3. The experimental validation system is comprehensive, covering performance comparison on benchmark datasets, few-shot generalization testing on real-world datasets, and ablation experiments on key components, fully demonstrating the model’s effectiveness and robustness.
4. It considers practical deployment needs: the designed FSVE integrates SNN components, achieving an approximate 16.6% overall energy efficiency improvement and providing a feasible solution for neuromorphic hardware deployment.

### Weaknesses
1. Typos: Line 149: spikestatus(”0”or”1”). The formulation is confusing. Line 150, the letter of "H x W" is different from "H x W" in line 158.
I hope the authors can fix these typos.
2. The theoretical in-depth exploration of the photon conservation mechanism in HSFE is insufficient, and no comparative analysis with existing dynamic channel allocation methods (e.g., attention-driven channel selection) is conducted. Could you explain this?
3. Although the FSVE improves energy efficiency, it suffers from significant accuracy loss (when all components of the visual encoder are converted to SNNs, the accuracy on the UCF101-S dataset drops to 65.24%), leaving room for optimization in balancing accuracy and energy efficiency. Add discussion on this point.

### Questions
See weakness.
This paper is sound but needs polishing in writing.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents SPKLIP (Spike-based Cross-modal Learning with CLIP), the first end-to-end neural network architecture specifically designed for Spike Video-Language Alignment. SPKLIP employs a Hierarchical Spike Feature Extractor that adaptively models multi-scale temporal dynamics in event streams while leveraging photon conservation principles for efficient feature extraction. It further utilizes Spike-Text Contrastive Learning to align raw spike video with natural language directly, without intermediate frame conversion. A full-spiking visual encoder variant was introduced to integrate spiking neural network principles, enhancing energy efficiency for neuromorphic hardware. Extensive experiments demonstrate that SPKLIP achieves better performance on benchmark spike datasets, substantially outperforming adapted conventional models, and shows strong few-shot generalization on a newly contributed real-world spike video dataset.

### Strengths
1) Its originality is highlighted by SPKLIP, the first end-to-end framework specifically designed for Spike Video-Language Alignment, as well as the introduction of an energy-efficient Full-Spiking Visual Encoder;

2) A new real-world spike video dataset was also constructed. 

3) The experimental results show that the SPKLIP yielded substantial Top-1 accuracy improvements over baselines, robust few-shot generalization, and effective text-to-video retrieval.

### Weaknesses
1) While the paper highlights the Full-Spiking Visual Encoder as a significant contribution, its connection to the main SPKLIP framework and its direct effectiveness are not fully explored through experiments. It remains ambiguous how FSVE directly contributes to or could enhance or degrade the main SPKLIP framework's performance. 

2) For the Hierarchical Spike Feature Extractor, the decision to divide the input spike stream into "five temporally overlapping sub-blocks" raises questions. The rationale behind specifically choosing five blocks and the exact definition of block_i are not sufficiently clear in the main text. It is unclear whether this number of sub-blocks is fixed regardless of the input's total temporal length, or if it adapts. An ablation study or a more detailed theoretical justification for this specific choice, demonstrating its optimal performance over other configurations would significantly strengthen this architectural decision.

3) While Appendix A.7 fully details the custom real-world dataset's construction, a more concise summary of its methodology and characteristics should be integrated into the main text (e.g., Section 4.1) for better context. More importantly, the necessity of introducing this entirely new dataset, rather than utilizing existing or publicly available benchmarks, is not adequately justified, which somewhat diminishes the clarity of its specific contribution to the paper's claims.

4) Some necessary citations for external methodologies used should be more prominently placed in the main text. For instance, in Section 3.5, where the Full-Spiking Visual Encoder (FSVE) is introduced, the paper mentions integrating "Spiking ResNets with a Spiking Temporal Transformer for event stream processing." It specifically refers to components like "temporal-dependent normalization" (TDBN) and an "efficient E-SDSA module." While Appendix A.4 elaborates on these, their initial mention in the main text lacks immediate citation or brief explanatory context.

### Questions
1) Could the authors elaborate on the precise role and intended contribution of the FSVE within the overarching SPKLIP framework？

2) The paper highlights FSVE as a key contribution, but direct experimental validation of its impact on SPKLIP's primary performance metrics (e.g., Top-1 accuracy or retrieval performance) seems to be absent. The authors could provide experiments or a clearer qualitative analysis demonstrating how FSVE specifically influences the model's ability to perform Spike Video-Language Alignment.

3) In Section 3.2, the input spike stream is divided into "five temporally overlapping sub-blocks." What is the specific rationale behind choosing precisely five sub-blocks? Is this number derived from extensive empirical studies, consistently yielding the best performance across various input scales and tasks?

4) Are there ablation studies or theoretical justifications to support the choice of five sub-blocks over other configurations (e.g., three, seven, or a dynamically determined number), especially given the claim of adaptively modeling multi-scale temporal dynamics?

5) The paper introduces and validates SPKLIP on a newly contributed real-world dataset. Could the authors explicitly articulate the necessity of creating this new dataset? What specific limitations or gaps in existing public benchmarks  did this new dataset address that are not adequately covered by current resources?

6) How does this new dataset uniquely contribute to demonstrating SPKLIP's capabilities or addressing specific challenges in Spike-VLA that existing datasets could not? A clear justification in the main text would greatly enhance the perceived value and impact of this dataset contribution.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors in this paper propose an architecture for Spike Video-Language Alignment (Spike-VLA). The authors employ a hierarchical spike feature extractor to model multi-scale temporal dynamics and uses contrastive learning technique to align spike video with language. A new dataset is also proposed as part of the work and the performance of the model is competitive compared to other non-spiking baselines.

### Strengths
1) The motivation of the work is strong. Spiking Cameras offer advantages not present in conventional cameras however there has been limited work done on efficient processing of spiking video streams.

2) The HSFE module proposed in the paper seemed interesting.

### Weaknesses
1) The computations inside the HSFE module does not seem entirely spiking. 
2) Most of the other parts of the model proposed (text encoder, even parts of the visual encoder) can be derived from available literature. The contrastive learning loss proposed is used in most video-language models like UniVTG, etc. Thus, making the work seem more of an engineering endeavor.
3) Performance comparison with baselines might not be fair since they are evaluated on a spiking variant of the datasets.

### Questions
1) What is the reason of processing the input into five temporally overlapping sub-blocks?
2) It will be interesting to visualize the temporal dynamics of the model i.e. average spiking rate, etc.
3) The primary contribution of this work seems to be the HSFE module. How do the work compare to other encoding techniques [1]


References:

[1] Zhu, Lin, Xiao Wang, Yi Chang, Jianing Li, Tiejun Huang, and Yonghong Tian. "Event-based video reconstruction via potential-assisted spiking neural network." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3594-3604. 2022.

### Soundness
3

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
5

### Summary
This paper introduces SPKLIP, the first architecture for Spike Video-Language Alignment (Spike-VLA), addressing the challenge of aligning sparse, asynchronous spike camera data with natural language. The method features a Hierarchical Spike Feature Extractor (HSFE) with multi-scale temporal filtering and spatial attention, combined with Spike-Text Contrastive Learning (STCL) for cross-modal alignment. A Full-Spiking Visual Encoder (FSVE) variant demonstrates energy efficiency potential. Experiments show 91.15% Top-1 accuracy on HMDB51-S (versus 45.31% for adapted Vita-CLIP) and effective few-shot learning on a new real-world dataset. While the problem is novel and results are promising, the work has significant limitations including reliance on synthetic data conversion, substantial accuracy drops with the spiking variant, and limited real-world validation.

### Strengths
1. Novel Problem Formulation: First work addressing video-language alignment specifically for spike cameras, filling an important gap between neuromorphic vision and semantic understanding.
2. Well-Motivated Architecture: The HSFE module with multi-scale temporal filtering (MTF) and spatial attention (SA) is thoughtfully designed to handle spike data's unique characteristics—sparse, asynchronous, high-frequency event streams.
3. Multimodal Alignment Validation: Text-to-video retrieval experiments (Table 4: 31.94% R@1, 63.12% R@5) provide evidence beyond classification that the model learns meaningful cross-modal embeddings.

### Weaknesses
1. HMDB51-S and UCF101-S are synthetically generated from RGB videos using SpikeCV toolkit, not real spike camera data.
2. Full-spiking variant (FSVE) drops from 86.43% to 71.11% (CNN→SNN) and 65.24% (full SNN) on UCF101-S. This 21.19% drop undermines claims about neuromorphic deployment viability.
3. Real dataset contains only 384 samples (96×4) across 4 simple actions.
4. Contrastive loss (Eq. 6) is standard CLIP loss—limited novelty.
5. No comparison with simple baselines like averaging spike frames or using histogram features.

### Questions
1. Can you provide quantitative comparison between real spike camera output and your synthetically converted data? Do spike frequency distributions, temporal statistics, and noise patterns match?
2. Have you ablated temporal window size T for FSVE? What accuracy is achievable with T=5, 10, 20?

### Soundness
2

### Presentation
2

### Contribution
2
