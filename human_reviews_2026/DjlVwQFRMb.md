# Tramba: Mamba with Adaptive Attention for Traffic Speed Forecasting

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 6, 4, 8

## Abstract
We introduce \textbf{Tramba}, a novel deep learning model for traffic speed forecasting in complex urban road networks. Unlike conventional methods that rely heavily on short-term trends or local spatial proximity (e.g., upstream and downstream links), Tramba captures dynamic, long-range dependencies across both time and space. It does so by integrating two key components: a Mamba-based temporal encoder that models long-term historical patterns of the target link, and an adaptive attention mechanism that learns temporally similar patterns from non-adjacent road links across the network. We evaluate Tramba on a real-world dataset from Gangnam-gu, Seoul, comprising 5-minute interval speed measurements across 366 road segments. Tramba is tested over forecasting horizons from 1 to 36 steps and compared with six strong baselines. It consistently outperforms all alternatives, achieving an average MAPE of 10.17\%, MAE of 2.80~km/h, and MSE of 20.50~(km/h)$^2$ on the three datasets for 12-step forecasting. These results highlight Tramba’s ability to model long-range dependencies and detect non-local influences in complex urban networks, reducing prediction lag and improving robustness in dynamic traffic conditions. Code is available at~\url{https://github.com/tr-anon-users/tramba-code}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Tramba, a traffic speed forecasting framework that combines a Mamba-based temporal encoder for long-range memory with an adaptive attention module that captures non-local spatial dependencies based on temporal similarity. The model integrates these two signals via a gating mechanism to balance temporal continuity and spatial alignment. Experiments on PEMS-BAY, METR-LA, and Seoul TOPIS show consistent improvements over strong Transformer- and Mamba-based baselines, with ablations highlighting the importance of adaptive attention.

### Strengths
1. The attempt to apply a new modeling paradigm like Mamba to traffic speed forecasting is interesting and contributes to methodological diversity in this area.
2. The release of code and datasets improves reproducibility and supports further research.
3. The paper is generally well-organized, and the experimental design is clear, making it easy to follow and understand the proposed approach.

### Weaknesses
1. The motivation for introducing Mamba is not clearly articulated, and its role relative to the attention module remains vague. Moreover, the so-called “adaptive attention” relies on similarity across temporal sequences to generate weights, which limits its adaptiveness and makes its ability to capture spatial dependencies questionable. The paper does not provide sufficient justification for this design choice or demonstrate its clear advantages over standard attention mechanisms.
2. The paper’s claim that existing methods neglect spatial dependencies is inaccurate, as many prior work has already modeled spatial relationships in different flexible ways. The proposed adaptive attention lacks a clear justification for why its approach is more effective, and deriving spatial relations purely from temporal similarity is insufficiently motivated.
3. The evaluation compares only Transformer- and Mamba-based models, omitting key state-of-the-art approaches such as LLM-based and diffusion-based methods. This narrow selection limits the credibility of the reported improvements and makes it difficult to assess the method’s generality and competitiveness.

### Questions
1. The paper should provide a clearer motivation for introducing Mamba and explain why it is necessary in this setting. What specific limitations of attention-based models does Mamba address, and how does it complement or outperform attention in this task?
2. The current experiments compare mainly Transformer- and Mamba-based models. It would strengthen the paper to include more diverse baselines, such as LLM-based and diffusion-based approaches, to better position the method relative to the state of the art. Additionally, the paper notes that some methods model spatial dependencies using fixed adjacency matrices, but it is unclear which baselines fall into this category and which do not. Please clarify these distinctions and discuss how they affect the comparison.
3. The paper describes the components of the proposed model but lacks sufficient explanation for why each part is designed this way. In particular, the rationale for deriving spatial dependencies from temporal similarity, and for the design of the “adaptive attention” mechanism, should be discussed in more depth.

### Soundness
2

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
2

### Summary
The paper proposes Tramba, a traffic forecasting framework that combines a per-link Mamba backbone for efficient long-range temporal modeling with an adaptive attention branch that retrieves information from non-adjacent road segments under learnable time shifts. A gating module fuses the temporal (Mamba) and non-local (attention) signals to generate multi-step predictions. Evaluations on standard benchmarks (e.g., METR-LA, PEMS-BAY, TOPIS) show that Tramba delivers consistent improvements—especially at medium-to-long horizons—over strong Transformer- and Mamba-based baselines. Ablations indicate that both the adaptive attention and the gating mechanism are critical to the gains, and qualitative analyses visualize how the model prioritizes time-aligned yet spatially distant segments.

### Strengths
- Targets a realistic pain point in road networks—non-local dependencies with temporal misalignment—and addresses it by pairing Mamba’s long-sequence modeling with shift-aware attention.
- Temporal dynamics are handled by the per-link Mamba backbone, while the adaptive attention captures dynamic, non-adjacent relations; a simple learnable gate provides robust fusion.
- Demonstrates stable gains at longer horizons across multiple datasets and steps, suggesting better robustness for challenging prediction ranges.
- Removing or altering the adaptive attention, similarity function, or gating leads to notable degradation, clearly attributing improvements to the proposed components.

### Weaknesses
- Please provide a fine-grained computational/parametric complexity breakdown for each module (Mamba branch, adaptive attention over (segments × shifts), and gating), along with GPU memory usage curves under large sequence lengths **T** and large numbers of segments **L** . The article only provides training time, which does not fully reflect efficiency.

- Please add representative GNN models (e.g., GCN/GAT/temporal-graph variants) or explicitly justify their exclusion, and clarify how Tramba’s shift-aware non-local attention differs from GNN-style spatiotemporal modeling in terms of inductive bias, expressivity, and computational trade-offs.

- To strengthen the causal attribution of gains, please replace the Mamba backbone with a Transformer backbone (keeping the rest of the architecture intact) and report the performance and efficiency deltas. This would disentangle the contribution of the proposed adaptive attention/gating from the choice of temporal encoder.

### Questions
See Weakness.

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
4

### Summary
This paper proposes Tramba, which uses the selective state space model Mamba as a long-term memory encoder for each road segment, and introduces adaptive non-local attention with time-shift alignment to retrieve “non-adjacent yet temporally co-modulated” road contexts across the entire network. A learnable fusion gate then combines the two representations to output multi-step forecasts in a single pass. The method aims to simultaneously mitigate prediction lag caused by short-term-only modeling and missed dependencies caused by relying solely on physical adjacency. On METR-LA, PEMS-BAY, and TOPIS, across multiple horizons, Tramba achieves consistent improvements over strong baselines including several Mamba and Transformer variants; ablations indicate that non-local co-modulation retrieval and gated fusion are the primary sources of gain.

### Strengths
S1. The model architecture and mathematical formulations are clearly presented.

S2. Tramba achieves solid performance on all three datasets.

S3. The released code facilitates reproducibility.

### Weaknesses
W1. The motivation is not sufficiently compelling. Spatial modeling limited to adjacent segments has been studied long ago; many recent works already use global attention to handle long-range spatial dependencies, but the paper does not discuss how Tramba differs from these approaches.

W2. Many classic traffic forecasting baselines are missing; such as DCRNN, GWNet, and AGCRN should be included.

W3. The three components of Tramba—MambaBlock, Adaptive Attention, and the Fusion Gate—do not introduce novel designs. In particular, the MambaBlock is directly implemented with the mambapy.mamba library without modification.

W4. The application scope is narrow: evaluation is limited to traffic speed datasets. Most existing traffic models cover both speed and flow prediction (e.g., PeMS03, PeMS08).

### Questions
Q1. Compared with the NeurIPS submission version, there are no technical differences described, so why are the results improved so much?

Q2. Can Tramba generalize to generic traffic flow forecasting? What are the key challenges?

Q3. Although the paper reports FLOPs and parameter counts, it lacks clear training/inference time and memory usage. Please include these comparisons in tabular form.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents Tramba, a deep learning model for traffic speed forecasting in complex urban road networks. Unlike traditional methods that focus on short-term or local dependencies, they capture long-range spatiotemporal relationships by combining a Mamba-based temporal encoder for long-term trends with an adaptive attention mechanism that identifies temporally similar patterns from non-adjacent road links. Evaluated on real-world traffic data, Tramba consistently outperforms six strong baselines across multiple forecasting horizons.

### Strengths
* The writing is clear, well-organized, and easy to follow throughout the paper.
* The experiments are thorough and clearly demonstrate the effectiveness of the proposed method, including baseline comparisons across different regions and forecasting horizons to validate its generality.
* The paper includes an ablation study that justifies the contribution of each model component.
* The paper provides a thorough visual interpretation of Tramba’s attention behavior, which makes it easy to understand how the model captures spatial and temporal dependencies.

### Weaknesses
* The organization of the experimental section could be improved. The current structure contains too many subsections, which makes it harder to follow. It may be clearer to group the results into broader categories, such as Main Results and Attention Analysis.
* Section 4.4 (Confidence Analysis) is missing a results table. Include a table reporting the baseline numbers alongside the proposed method.

### Questions
Addressed in the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3
