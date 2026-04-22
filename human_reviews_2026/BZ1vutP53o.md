# TEN-DM: Topology-Enhanced Diffusion Model for Spatio-Temporal Event Prediction

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 6

## Abstract
Spatio-temporal point process (STPP) data appear in many domains. A natural way to model them is to describe how the instantaneous event rate varies over space and time given the observed history which enables interpretation, interaction detection, and forecasting. Traditional parametric kernel-based models, while historically dominant, struggle to capture complex nonlinear patterns. In contrast, deep learning methods leverage the representational power of neural networks to aggregate historical events and integrate spatio-temporal point processes. However, existing deep learning methods often process space and time independently, overlooking the spatio-temporal dependencies. To address this limitation, we propose a novel method called Topology-ENhanced Diffusion Model (TEN-DM), including two key components namely spatio-temporal graph construction and multimodal topological feature representation learning. Further, we use temporal query technique to effectively capture periodic temporal patterns for learning effective temporal representations. Extensive experiments show the effectiveness of TEN-DM on multiple STPP datasets compared to state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a TEN-DM framework for modeling Spatio-Temporal Point Processes (STPPs). The method integrates three key components: a Graph Construction and Learning (GCL) module to represent STPP data as graphs, a Temporal Topological Learning (TTL) framework using zigzag persistence to capture evolving topological features, and a Temporal Query-guided Self-Attention (TQ-SA) mechanism. The authors combine these into a diffusion model to learn complex spatio-temporal dependencies, particularly under sparse and noisy regimes. Extensive experiments on five real-world datasets are presented with promising results.

### Strengths
Strengths:

- The paper proposes several advanced paradigms, such as the graph neural networks and diffusion models, into a single framework for STPP modeling.

- The paper provides a thorough experimental section, benchmarking TEN-DM against a set of baselines across multiple real-world datasets. 

- The proposed model is decomposed into clear, modular components, making the architecture easy to understand.

### Weaknesses
Weaknesses:

- One of the main issues of this paper is the lack of motivation and problem analysis. Although there are some discussions for the choices of the key components, the justifications are somewhat insufficient. For example, the rationale for using graph abstraction is "graph abstraction offers a flexible... framework" and "never been used". This does not articulate what specific challenges or limitations exist in current STPP methods.

- Similarly, the motivation for using diffusion models is that they are "a new powerful machinery" and have not been applied to STPPs before, which is somewhat insufficient. Thus, I suggest that to provide a theoretical analysis to show that under what conditions the proposed method is better than the previous methods.

- Furthermore, it would be better if it could provide some case studies to illustrate the key increment and its intuition compared with other STTP methods.

### Questions
See the weaknesses above.

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
4

### Summary
This paper presents TEN-DM, a Topology-Enhanced Diffusion Model for spatio-temporal point process (STPP) prediction. The authors aim to address the limitations of existing deep learning models in capturing complex, non-stationary spatio-temporal dependencies, especially under sparse and noisy conditions. TEN-DM introduces three core components: (i) a graph construction and learning module to model event interactions, (ii) a temporal topological learning (TTL) framework based on zigzag persistence to extract dynamic topological features from time-series images, and (iii) a temporal query-guided self-attention (TQ-SA) mechanism to capture periodic patterns. The model is evaluated on five real-world datasets and outperforms the baselines in both spatial and temporal prediction tasks.

### Strengths
- Originality: This is the first work to integrate zigzag persistence and diffusion models for STPP forecasting. The use of topological data analysis (TDA) in the form of zigzag persistence images (ZPI) to capture time-evolving shape patterns is novel and well-motivated. The idea of converting STPP data into image time-series and analyzing their topological evolution is creative and technically sound.

- Technical Quality: The paper is mathematically rigorous, with formal definitions of cubical complexes, filtrations, and zigzag persistence. The stability theorem (Theorem 3.2) provides theoretical grounding for the robustness of ZPI under noise. The Lipschitz bound for the proposed attention mechanism (TST-MHA) further demonstrates the model’s theoretical controllability.

- Clarity: Despite the complexity of the methodology, the paper is well-organized and clearly written. Each module is introduced with both intuition and formalization. Figures (e.g., Fig. 1, Fig. 2) effectively illustrate the pipeline and help readers understand the workflow.

- Significance: The paper addresses a real-world problem, forecasting discrete events in space and time (e.g., earthquakes, crimes, disease outbreaks), and proposes a unified, interpretable, and robust solution. The integration of geometry, topology, and diffusion opens a new research direction in spatio-temporal modeling, especially for sparse and irregular data.

### Weaknesses
- Scalability and Efficiency Concerns: While the model is effective on small-scale datasets (e.g., ~10K events), its scalability to large-scale urban data (e.g., millions of taxi trips or tweets) is unclear. The zigzag persistence computation and image rasterization steps may become prohibitively expensive for high-resolution or long-duration data. A complexity breakdown or runtime scaling analysis is missing.

- Limited Ablation on Topological Hyperparameters: The paper does not thoroughly explore the sensitivity of ZPI parameters, such as filtration resolution, patch size, or zigzag directionality. The multi-scale mixing uses fixed weights (βq = 0.25), but adaptive or learned weighting could be more effective. An ablation on these choices would strengthen the contribution.

- Generalization Across Domains: Although the model is tested on five datasets, they are all from the US or Japan, and mostly urban or seismic events. There is no evaluation on human mobility, social media, or climate events, which are also common STPP scenarios. A cross-domain generalization test would better support the claim of universality.

- Baseline Diversity: While 17 baselines are included, few recent graph-based or transformer-based STPP models are missing. Also, no comparison with other TDA-based methods is provided. This limits the completeness of the empirical evaluation.

### Questions
- Scalability: How does TEN-DM scale to larger datasets (e.g., >1M events)? What is the time and memory complexity of ZPI generation and TTL module with respect to image resolution and sequence length?

- Topological Sensitivity: How does the model performance change with different filtration functions, patch sizes, or zigzag directions? Have you tried adaptive weighting for multi-scale ZPI fusion?

- Cross-domain Evaluation: Have you tested TEN-DM on non-urban, non-seismic data, such as animal movement, social media check-ins, or climate anomalies? How general is the topological assumption?

- Comparison with TDA Baselines: Why not compare with other TDA-enhanced models? This would better highlight the unique value of zigzag persistence.

- Real-time Forecasting: Is TEN-DM suitable for real-time or online forecasting? Can the ZPI and TTL modules be incrementally updated as new events arrive?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Topology-ENhanced Diffusion Model (TEN-DM) for predicting future events in spatio-temporal point process (STPP) data.

The central problem the authors address is that existing methods often fail to capture the complex, higher-order dependencies between the spatial and temporal dimensions. Their solution is a conditional diffusion model where the conditioning signal is a sophisticated, fused representation of the event history. It fuses Spatio-Temporal Graph, Spatial information, temporal information and Topological Learning (converts the STPP data into a time-series of 2D images). This fused embedding guides the diffusion model’s denoising process to accurately predict both the time and location of the next event.

Experiments on five real-world datasets (e.g., earthquakes, crime, COVID-19) show that TEN-DM achieves state-of-the-art performance

### Strengths
The motivation is straightforward: conditioning a spatio-temporal generative model on more available information generally improves performance. The approach proves highly effective, achieving top results across five diverse real-world datasets and outperforming 17 baselines.

### Weaknesses
The proposed pipeline is exceptionally complex. It involves GNN pre-training, data-to-image conversion, multi-scale cubical zigzag persistence computation (which is notoriously expensive), a CNN on persistence images, and a conditional diffusion model. This complexity may make the model impractical.

The conversion of event data to a 2D image (Section 3.2) is a critical step, but it is underspecified and potentially lossy. The paper states, "we rasterize the events geo-coordinates onto the 2D image by recording as each pixel's value the associated temporal attribute." But What happens if two or more events fall into the same grid and same time patch? How to choose the grid resolution?

The GCL module feels less developed than the TTL module. The graph is described as a similarity-based $\epsilon$-graph (using cosine similarity), which is known to be highly sensitive to the choice of the threshold $\epsilon$ (or $\mathbb{R}^r$ in the paper). The mechanism by which the aggregation weights $\{\alpha_r\}$ are "updated adaptively" is not explained

### Questions
Could the author please clarify the rasterization process in Section 3.2?

Why did you choose the image-based cubical complex representation?

In Section 3.1, how are the adjacency matrix aggregation weights $\{\alpha_r\}$ "updated adaptively"? Are these learnable parameters optimized end-to-end, or set via a separate process?

### Soundness
2

### Presentation
3

### Contribution
2
