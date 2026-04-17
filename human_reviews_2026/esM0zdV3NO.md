# Heterogeneous Graph Temporal Fusion Transformer for Time Series Forecasting in Multi-Domain Physical Systems

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Existing Transformer-based models effectively capture multivariate dependencies, while pre-trained large models achieve strong generalization but are often confined to single-object or single-physics settings. Spatial-temporal approaches leverage graph structures but fall short in modeling heterogeneous entities with diverse inter-variable interactions, and they often lack mechanisms to enforce physical consistency.
To address these challenges, we propose the Heterogeneous Graph Temporal Fusion Transformer (HGTFT), a pre-training and fine-tuning framework tailored for spatially and temporally structured physical environments. HGTFT tokenizes observation points and generates embeddings that capture both temporal patterns and spatial correlations, enabling the integration of heterogeneous static and dynamic information. 
We further introduce optimized normalization and physics-informed loss functions that enhance predictive accuracy while improving physical plausibility. Applied to temperature, flow, and energy-related datasets in building environments, our approach demonstrates strong zero-shot generalization and achieves substantial accuracy gains through few-shot fine-tuning with domain-specific data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the Heterogeneous Graph Temporal Fusion Transformer (HGTFT), a pre-training and fine-tuning framework designed for time series forecasting in multi-domain physical systems. The HGTFT model tokenizes observation points and uses a graph-temporal architecture to integrate heterogeneous static and dynamic information. The framework also introduces a novel Multi-Instance Normalization technique and a multi-stage training pipeline. The main findings suggest that HGTFT achieves strong zero-shot and few-shot performance, outperforming various baselines in both prediction accuracy and physical plausibility.

### Strengths
1. This paper addresses forecasting in complex real-world physical systems, which is a challenging and valuable problem. Furthermore, the focus on multi-domain interactions and physical consistency represents an important research area.
2. The authors conducted extensive ablation studies on the model architecture, provided a detailed analysis of model scalability, and compared different training strategies and normalization methods.
3. The introduction and release of the Multiphysics Building System (MBS) dataset is a valuable contribution to the community, providing a large-scale, complex benchmark dataset for this research field.

### Weaknesses
1. The baselines compared in the experiment are weak. The authors should compare against more recent and SOTA models, such as the deep learning models PatchTST [1], DLinear [2], and TQNet [3], as well as time series foundation models like Sundial [4].
2. The RCS introduced in the paper is not a general or objective evaluation metric. It is essentially a set of heuristics defined by the authors themselves. The authors use these rules as a loss function during the training stage and then use them again during the evaluation stage to demonstrate the model's physical consistency. Therefore, the model will naturally perform better on the RCS metric compared to the baselines. However, this does not fully prove that the model truly understands the physics; it only proves that it has learned to adhere to these hard-coded rules.
3. The scalability of the proposed architecture appears to be poor. According to the Scaling Study in Table 7, increasing the model parameter count from 310M to 1.26B results in a minimal improvement in the primary MSE metric (from 0.0027 to 0.0025). Meanwhile, the FDS (Frequency Domain Similarity) metric actually worsens (increasing from 0.405 to 0.416).

[1] A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. (ICLR 2023)

[2] Are Transformers Effective for Time Series Forecasting? (AAAI 2023)

[3] Temporal Query Network for Efficient Multivariate Time Series Forecasting. (ICML 2025)

[4] Sundial: A Family of Highly Capable Time Series Foundation Models. (ICML 2025)

### Questions
1. For the channel-independent baselines in the experiment, how was the pre-training or few-shot learning conducted? Were the models trained using only the target variable, or were all variables used in the training process?
2. Is the framework truly generalizable? The paper only conducts sufficient experiments in a single physical domain (the building domain). Would migrating it to other domains require significant effort (e.g., redesigning the RCS loss), and what kind of performance changes could be expected?
3. In Appendix A, the authors use a simplified FCU physical model as an example. They first derive differential equations, use them to generate simulated data, and then show HGTFT can perfectly fit this data. However, the inputs in this example are perfect sine waves. Successfully fitting this toy problem does not prove that HGTFT understands the physics. Is there any demonstration on truly complex and chaotic real-world data?
4. According to Appendix B, the simulated data contains 600B time points, while the real project data has only 16B time points. Does this imply that the pre-trained model spends the vast majority of its time learning the simulator's behavior? It is possible the model is merely overfitting to the simulator's simplified rules and specific dynamics rather than learning the truly complex physical laws of the real world.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes HGTFT, a transformer framework for forecasting in heterogeneous multi-domain physical systems. It fuses static and dynamic variables in a heterogeneous graph, integrates temporal attention with relation-specific aggregation. Evaluations show substantial improvements over baselines and strong zero/few-shot transfer across realistic multiphysics systems.

### Strengths
1. the topic is timely and valuable.

2. The paper clearly defines the new setting of heterogeneous graph forecasting in multi-domain physical systems, extending beyond conventional data.

3. The proposed graph-temporal fusion with physics-aligned losses is technically well motivated and addresses both accuracy and physical consistency, which many data-driven models ignore

4. Comprehensive experiments across synthetic and real-world datasets support the method’s claims

### Weaknesses
1. The technical increment over existing graph transformers or physics-informed forecasting models is limited. This paper introduced: a heterogeneous graph encoder, a temporal transformer, and physics-inspired regularizers. However, each individual piece has been seen in earlier spatiotemporal or physics-informed learning work. The paper’s novelty lies more in integration and application to building energy systems than in a new architectural mechanism.

2. The proposed physics-informed losses (RCS/CRS/FDS) are heuristic rather than derived from governing equations. The extent to which they enforce true physical constraints?

### Questions
1. Table 2 shows large RCS improvements (e.g., 0.0158 → 0.0018 zero shot); could the authors how this metric generalizes to unseen physics domains?

2. In Section 5.3, the weighting of the four loss terms (MSE, RCS, CRS, FDS) appears fixed; could the authors show how performance changes when these weights are learned or tuned, to verify robustness?

3. The multi-instance normalization in Eq. (9) aggregates across percentile bounds. Could the authors explain how this compares against standard per-feature normalization across node types?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces HGTFT, a graph transformer architecture for time series forecasting in multi-physics domains. This approach improves upon drawbacks of existing methods in multi-physics settings, which often struggle to perform under the complexity of disparate spatiotemporal dynamics. The proposed architecture attempts to overcome this challenge by piecing together relevant neural modules (e.g., the temporal, graph, and subtask layers) capable of jointly capturing complex dynamics. This method is compared to many competing baseline models on several common time series benchmarks, as well as on a newly proposed Multiphysics Building System (MBS) dataset.

### Strengths
- The paper's stated contributions are clear and address a difficult, high-impact problem in the domain of multi-physics systems. The approach potentially lays the groundwork for tackling broader challenges across connected physics models (not just the building environment).
- The presentation of the paper is clear and well-organized. I appreciated the comprehensive literature review and logically grouped discussions across Sections 1-4, which made problem setup and methodology easy to compartmentalize and digest.
- The reported evaluation is very extensive, covering a variety of important dimensions that help position the model's utility. For instance, the model is compared to several baseline methods on common time series datasets (highlighting its comparative advantages), key ablations are reported (justifying architectural decisions), and different model sizes are evaluated, highlighting the impact of parameter scaling.

### Weaknesses
- The empirical evaluation of the method on multi-physics settings is somewhat limited, provided only the multi-physics building setting is explored. While results appear strong and there are diverse dynamics present, it is difficult to assess the proposed architecture's general utility as a multi-physics model beyond this domain.
- The analysis of empirical results would benefit from a discussion that characterizes when a complex architecture like HGTFT is justified, compared to a less complex model, such as LSTMs. It would be very insightful to see practical tradeoff considerations, for instance, highlighting model differences in performance at fixed parameter counts or time spent training.
- It is claimed that the more common time series benchmark datasets don't capture the multi-domain complexity that HGTFT targets, but there is little to no explanation behind why the architecture underperforms other methods on these benchmarks (e.g., in Table 13). Presumably many of the mechanisms relevant for capturing complex multi-physics interactions would be beneficial in modeling complex multi-variate time series more broadly. Additionally, it is not particularly clear why the other datasets, e.g., traffic, are not considered to exhibit multi-scale dynamics among diverse entities (stated in Section 6.3) when these are common qualities of traffic forecasting settings. Characterizing the performance differences across these settings in more depth would go some way to helping bridge the empirical gap and help characterize model behavior in lieu of an additional multi-physics dataset.

### Questions
- In Table 2, there are a few counter-intuitive fluctuations in performance across zero-shot and few-shot settings. For instance, the RCS metric is better for HGTFT zero-shot than it is in the few-shot (in both the "50 MBS" and "Full MBS" settings). I understand minor fluctuations could very well be noise, but is there a more principled reason behind this?
- In the main multi-physics evaluation setup, building samples from the MBS dataset are used for pre-training. Is there a significant amount of overlap between MBS and BTS? What dynamics does MBS capture that are expected to be helpful for BTS, or perhaps more importantly, what are the meaningful differences (potentially highlighting ability to generalize to new dynamics seen only in BTS)?

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
The paper proposes a multiheaded GAT to model heterogeneous spatiotemporal graph data. The authors also examine the effectiveness of masked pre-training. The training pipeline is validated on a simulated heterogeneous spatiotemporal dataset.

### Strengths
- Originality and Significance: The paper addresses the lack of heterogeneous graph datasets in the field of spatiotemporal forecasting.
- Clarity: The multiple technical contributions are well explained.
- Significance: The paper touches on masked pre-training and analyzes zero-shot and few-shot performances.

### Weaknesses
- Literature Review:
  - The paper does not provide physics-prioritized methods
  - The paper misses a few important related works. E.g. 
    - DCRNN: Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2017). Diffusion convolutional recurrent neural network: Data-driven traffic forecasting. arXiv preprint arXiv:1707.01926.
    - STEP: Shao, Z., Zhang, Z., Wang, F., & Xu, Y. (2022, August). Pre-training enhanced spatial-temporal graph neural network for multivariate time series forecasting. In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining (pp. 1567-1577).
- Quality:
  - The paper has too many directions, making it hard to follow the central contribution. The authors might benefit from focusing on heterogeneous graphs only.
- Clarity:
  - The paper's wording is vague, making it hard to follow. E.g. in the following text, the authors could provide examples of different node types interacting simultaneously.
  > More generally, multi-domain physical systems such as power grids and building operations, where heterogeneous entities interact across multiple physical fields. Accurate forecasting in such systems is critical for efficiency, safety, and sustainability, yet remains challenging due to diverse data modalities, structural dependencies, and domain-specific physical mechanisms. 
  - Some typos. e.g. in section 6.1, 
  > Mult-domain physical System Datasets

### Questions
- How is LSTM pretrained and evaluated zero-shot in your framework?
- Could we use the output from the pre-trained models as input to other baseline methods? Would this approach increase the accuracy compared to using a linear projection from node values to the hidden space?

### Soundness
2

### Presentation
1

### Contribution
4
