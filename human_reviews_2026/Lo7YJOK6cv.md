# SNN-Driven Multimodal Human Action Recognition via Sparse Spatial-Temporal Data Fusion

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
Recent multimodal action recognition approaches that combine RGB and skeleton data have achieved strong performance, but their high computational cost and poor energy efficiency hinder deployment on edge devices. To address these limitations, we propose the first spiking neural network (SNN)-based framework for multimodal human action recognition, to the best of our knowledge, offering an energy-efficient and scalable solution that fuses sparse spatiotemporal data of event cameras and skeletons within a unified spiking architecture. The framework leverages the sparse and asynchronous nature of event and skeleton data and the energy-efficient properties of SNNs. It achieves this through a series of tailored components, including modality-specific feature extraction, a sparse semantic extractor, spiking-based cross-modal fusion via Spiking Cross Mamba, and task-relevant feature compression utilizing a Discretized Information Bottleneck (DIB). To support reproducible evaluation, we further introduce a data construction pipeline that generates temporally aligned event-skeleton pairs from existing RGB-skeleton datasets. Extensive experiments demonstrate that our approach achieves state-of-the-art accuracy among SNNs while significantly reducing energy consumption, providing a practical and scalable solution for neuromorphic multimodal action recognition.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an SNN-based framework for multimodal human action recognition by fusing event camera data and skeleton sequences. The approach addresses the high computational cost and energy inefficiency of traditional RGB-skeleton fusion methods by leveraging the sparse, asynchronous nature of both event and skeleton data within a unified spiking architecture. Key components include SGN (Spiking Graph Network), SSE (Sparse Semantic Extractor), SCM (Spiking Cross Mamba) DIB (Discretized Information Bottleneck). The authors construct event-skeleton datasets from existing RGB-skeleton benchmarks using V2E transformation. Experiments on NTU RGB+D, NTU RGB+D 120, and NW-UCLA show competitive accuracy among SNNs with reduced energy consumption.

### Strengths
1. First work to explore SNN-based multimodal fusion for action recognition, combining event and skeleton modalities

2. Comprehensive and competitive results. Most of the experiments achieve higher performance than previous works with iso-parameter architecture. Furthermore, authors implement extensive ablation studies and analysis.

3. Appendix A rigorously analyzes why classical Gaussian IB fails for SNNs and justifies the DIB formulation with discrete KL divergence and cosine surrogates.

### Weaknesses
1. Novelty
- While the authors claim this as the first SNN-based multimodal action recognition framework, the novelty is questionable. Except for the DIB module, all components are directly adopted from prior works with minimal modification (Spiking Mamba, SGN, etc.). The contribution essentially reduces to replacing activation functions with spiking neurons and introducing DIB. This appears more like an ad-hoc engineering integration rather than a fundamental methodological advance. 

2. Pseudo-event data
The entire evaluation relies on V2E-synthesized events from RGB, not real DVS camera data. This is acknowledged in Appendix J, but undermines claims about "event camera" advantages like high dynamic range and low latency. 

2. Paper presentation
- Even though every module is well-explained in text, the figures are very hard to see due to small text. 
- I feel that the Figure 1 is redundant. The figure 1 does not convey any core technical contribution.

3. Energy calculation
- Several input tensors to Linear Projection (LP) layers are not binarized due to residual connections. However, based on equation (32), the authors used $E_{MAC}$ only for first layer. I strongly believe that the energy calculation should be corrected based on the architecture.

### Questions
1. Could you elaborate on the core innovations or novelties of this work beyond the DIB module? I am happy to discuss about this.

2. In the Spiking Cross Mamba (Figure 2e, Table 4), why do event features feed into the State Space Model (SSM) path while skeleton features go to the Selective Path (gate)? Is there a theoretical or empirical justification for this asymmetric design?

3. For better understanding of the model's efficiency and to validate energy calculations, could you provide firing rates for each module

4. Could the authors discuss what specific characteristics of real DVS data are not captured by V2E and how they might affect your model? Also, could you clarify whether the current framework would remain valid for real DVS inputs?

### Soundness
2

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
4

### Summary
This paper proposes a novel Spiking Neural Network (SNN) framework for multimodal human action recognition, fusing event camera data and skeleton sequences. Key contributions include:
--The first SNN-based multimodal fusion architecture for event-skeleton data.
--Introduction of Spiking Cross Mamba (SCM) for cross-modal interaction and a Discretized Information Bottleneck (DIB) for task-relevant feature compression under spiking constraints.
--A pipeline to construct aligned event-skeleton datasets from existing RGB-skeleton benchmarks.
--State-of-the-art accuracy among SNN methods with significantly lower energy consumption (1.73 mJ).

### Strengths
--Quality: Strong empirical results, thorough ablation, and theoretical grounding.
--Significance: Demonstrates a practical pathway for low-power multimodal recognition on edge devices.
--Clarity: The overall pipeline and experimental section are well-structured and described.

### Weaknesses
Motivation: The introduction does not convincingly establish a strong "why now" or "why this way" for the proposed method. The limitations of prior ANN and SNN works are stated but not used to build a powerful narrative for the current approach.

Originality: The architectural innovations (SCM, DIB) feel more like competent engineering integrations of existing ideas (cross-attention, Mamba, IB) into the SNN domain, rather than a fundamental conceptual breakthrough.

Presentation: Inconsistent reference formatting and occasionally dense technical passages reduce readability.

### Questions
Could you better motivate the specific choice of Spiking Mamba and the SCM fusion mechanism? What specific limitations of prior SNN or ANN fusion methods do they address that simpler baselines cannot?

The DIB is a key contribution. Beyond the theoretical derivation, can you provide more intuition or analysis on how it selectively compresses features and improves performance?

Could you compare your method with more ANN-based multimodal models under similar parameter budgets to better contextualize the performance-efficiency trade-off?

How would the performance change if real event camera data were used instead of V2E-simulated events?

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
The paper proposes a SNN‑based framework for multimodal human action recognition that fuses event and skeleton streams end‑to‑end in spikes. The architecture comprises: (i) spiking encoders for skeleton (SGN) and events (Spiking‑Mamba) from prior works, (ii) a Sparse Semantic Extractor (SSE) with hypergraph generators and Global Spiking Attention (GSA), (iii) Spiking Cross Mamba (SCM) for cross‑modal interaction, and (iv) a two‑stage Discretized Information Bottleneck (DIB) that performs spike‑compatible fusion.

### Strengths
1. Event/skeleton are both sparse temporal modalities; an SNN‑native fusion is a coherent direction.
2. Module‑wise gains are cleanly reported, and the DIB variants are systematically explored.

### Weaknesses
1. The highest Xs achieved by your model on NRD/NRD-120 is 85.0/74.6, which is substantially lower than the best-performing ANNs, such as VPN at 93.5/86.3, and MMNet at 94.2/92.9. 
2. ANN models operating on the same magnitude of computational cost also perform better, eg., CTR-GCN at 89.9/84.9 with 1.97 G FLOPs, and Shift-GCN at 87.8/80.9 with 2.5 G FLOPs. The efficiency gain claim is week.

### Questions
What is the compute profile? Please report GPU type&number, GPU hours, and peak memory to train your model.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the first spiking neural network–based framework for multimodal human action recognition, combining event camera and skeleton data for energy-efficient, real-time recognition on edge devices. The proposed system, termed SNN-driven multimodal fusion, integrates several novel components: a Spiking Graph Network (SGN) for skeleton encoding, Spiking Mamba for event encoding, a Sparse Semantic Extractor (SSE) for structured attention, Spiking Cross Mamba (SCM) for cross-modal fusion, and a Discretized Information Bottleneck (DIB) for task-relevant feature compression under spiking constraints. The model achieves strong performance across NTU RGB+D and NW-UCLA benchmarks, outperforming prior SNNs in accuracy.

### Strengths
1. The paper introduces the first multimodal SNN framework for human action recognition, representing a novel direction in neuromorphic computing. The use of event and skeleton modalities is well-motivated, making them well-suited for low-power, energy-efficient computation on edge devices.

2. The paper is technically thorough and clearly presented. 

3. Achieves state-of-the-art SNN accuracy with drastically reduced energy consumption compared to ANN baseline.

### Weaknesses
1. My main concern lies in the degree of technical novelty. Each component (Mamba, SNNs, and the Information Bottleneck) appears to be based on existing techniques, and the overall contribution could be viewed as a careful integration rather than a fundamentally new design. Could the authors clarify what specific aspects of the proposed framework go beyond a modular combination of known components?

2.  Although the paper reports improved fusion accuracy and energy efficiency, it provides limited qualitative or interpretive analysis (e.g., failure cases, feature attribution, or modality interaction visualization) to illustrate how the model effectively leverages the complementary cues of event and skeleton data. Including such analyses would significantly enhance interpretability and reader confidence in the proposed fusion mechanism.

3. The experiments rely on synthetic event-skeleton pairs converted from RGB videos rather than real event-camera datasets. This limits the validity of the claimed neuromorphic efficiency. Results on a genuine event-based dataset would substantially strengthen the empirical contribution.

4. Minor comment – Please consider enlarging Figures 2 and 4 for better readability and visual clarity.

### Questions
Please address points in weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
