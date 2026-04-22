# Inductive Reasoning for Temporal Knowledge Graphs with Emerging Entities

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Reasoning on Temporal Knowledge Graphs (TKGs) is essential for predicting future events and time-aware facts. While existing methods are effective at capturing relational dynamics, their performance is limited by a closed-world assumption, which fails to account for emerging entities not present in the training. Notably, these entities continuously join the network without historical interactions. Empirical study reveals that emerging entities are widespread in TKGs, comprising roughly 25\% of all entities. The absence of historical interactions of these entities leads to significant performance degradation in reasoning tasks. Whereas, we observe that entities with semantic similarities often exhibit comparable interaction histories, suggesting the presence of transferable temporal patterns. Inspired by this insight, we propose TransFIR (Transferable Inductive Reasoning), a novel framework that leverages historical interaction sequences from semantically similar known entities to support inductive reasoning. Specifically, we propose a codebook-based classifier that categorizes emerging entities into latent semantic clusters, allowing them to adopt reasoning patterns from similar entities. Experimental results demonstrate that TransFIR outperforms all baselines in reasoning on emerging entities, achieving an average improvement of 28.6\%  in Mean Reciprocal Rank (MRR) across multiple datasets. The implementations are available at https://anonymous.4open.science/r/TransFIR-C72F.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel framework tailored to the unseen entity link prediction in temporal knowledge graphs. The authors observed that entities sharing similar semantics often have comparable interaction histories and interaction patterns. Inspired by this, the authors propose the TransFIR framework that uses the semantically similar known entities to augment the unseen entity reasoning, where a codebook-based classifier is used to map entities to semantic clusters, and the semantics of unseen entities will be augmented by other entities within the cluster. Extensive experimental results showcasing the effectiveness of the proposed method.

### Strengths
S1. The paper is well-written and easy to follow

S2. Learning the reasoning strategy for emerging entities is a challenging and valuable direction in the field of temporal knowledge graphs.

S3. Technical details of the proposed framework are well-motivated and justified.

S4. Extensive experimental results are provided, offering a comprehensive understanding of the model performance.

### Weaknesses
W1. The current framework assumes static cluster assignments for entities after training. However, in reality, entity semantics often evolve over time, leading to potential shifts in their associated clusters. This inherent limitation is likely to impair the model's performance in long-term prediction scenarios, where semantic changes can become more pronounced.

W2. Under the open-world assumption, emerging entities may belong to entirely new categories that exhibit no discernible similarities to existing ones. It is therefore worthwhile to examine how the framework performs in handling such entities.

### Questions
None

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
This paper introduces TransFIR, a transferable inductive reasoning framework for Temporal Knowledge Graphs that enables effective reasoning about emerging entities by leveraging historical interaction patterns of semantically similar known entities via a codebook-based semantic clustering approach, achieving significant performance gains over baselines in predicting facts involving new entities.

### Strengths
1. Intuitive experimental results clarify the motivation, making the paper easier to understand.

2. A novel codebook-based approach is proposed to address emerging entities in temporal knowledge graphs.

3. Experimental results comprehensively and clearly demonstrate the effectiveness of the proposed method.

### Weaknesses
1. The related work section omits some recent inductive reasoning methods for temporal knowledge graphs.

2. Lines 157–158 state, “after training, emerging entities deviate sharply from known entities in the embedding space.” Since emerging entities rarely appear in the training set and are updated less frequently, this phenomenon is unsurprising.

3. With BERT-encoded and frozen entity embeddings, the method likely relies on BERT’s semantic encoding to address emerging entities. Ablation results on ICEWS18 in Figure 5 support this. It is recommended to provide additional experiments to further assess the impact of LM on performance.

4. The method depends on having a reliable textual description for each entity to generate initial BERT embeddings . In domains where such text is unavailable, noisy, or ambiguous, the quality of the codebook clustering could degrade significantly, weakening the entire framework.

5. The complexity analysis shows a time complexity of $O(n_t L(k^2d + kd^2))$ for the IC encoder, which could become a bottleneck for graphs with very long interaction histories.

### Questions
See Weakness

### Soundness
3

### Presentation
4

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
Existing Temporal Knowledge Graph (TKG) reasoning methods primarily focus on modeling relation dynamics but typically assume a closed entity set. In the real world, new entities are continuously added to the graph but lack historical interaction data, leading to a significant drop in reasoning performance for these entities. TRANSFIR offers a systematic solution to the inductive reasoning problem for emerging entities without historical interactions. It enables transferable temporal reasoning through semantic similarity transfer and a codebook-based classification mechanism, achieving significant progress in both performance and scalability.

### Strengths
1. The paper introduces the concept of semantic similarity transfer, providing an effective solution to prevent representation collapse.
2. Through empirical research, the paper demonstrates the widespread presence of emerging entities in Temporal Knowledge Graphs (TKGs), with approximately 25% of entities being new. The study also shows that existing methods experience a significant performance degradation when handling these emerging entities. This provides strong theoretical and experimental support for the proposed TRANSFIR framework.

### Weaknesses
1. The evaluation could be more comprehensive. It only includes one large-model-based method, whereas other relevant approaches like ICL [1] and PPT [2] are not considered.
2. Unclear novelty over existing similarity-based approaches. The main innovation of the proposed TRANSFIR framework lies in leveraging the behavioral evolution patterns of similar entities to assist in predicting emerging entities. However, similar approaches already exist — for example, MGESL[3] also considers the similarity between entities and analyzes the behavioral evolution patterns of semantically related entities. Moreover, MGESL discusses both settings where candidate entities are known and unknown.

    [1] Dong-Ho Lee, Kian Ahrabian, Woojeong Jin, Fred Morstatter, and Jay Pujara. 2023. Temporal Knowledge Graph Forecasting Without Knowledge Using In-Context Learning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 544–557, Singapore. Association for Computational Linguistics.

    [2] Wenjie Xu, Ben Liu, Miao Peng, Xu Jia, and Min Peng. 2023. Pre-trained Language Model with Prompts for Temporal Knowledge Graph Completion. In Findings of the Association for Computational Linguistics: ACL 2023, pages 7790–7803, Toronto, Canada. Association for Computational Linguistics.

    [3] Shi Mingcong, Chunjiang Zhu, Detian Zhang, Shiting Wen, and Li Qing. 2024. Multi-Granularity History and Entity Similarity Learning for Temporal Knowledge Graph Reasoning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5232–5243, Miami, Florida, USA. Association for Computational Linguistics.

### Questions
1. In the ablation experiment on the GDELT dataset, is the performance without the textual encoding module better than TransFIR? This is difficult to determine from the figure. If the performance without the textual encoding module is better than TransFIR, what could explain this result?
2. How does TRANSFIR fundamentally differ from existing similarity-based models such as MGESL[3]? Would including MGESL[3] in the experimental comparison change the relative performance ranking of TRANSFIR?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes TRANSFIR, an inductive reasoning framework for temporal knowledge graphs, designed to handle emerging entities that appear without historical interactions. The authors first conduct an empirical investigation showing that approximately 25% of entities in common TKG benchmarks are unseen during training, leading to severe performance degradation and representation collapse. To address this, TRANSFIR introduces a Classification–Representation–Generalization pipeline:
1.	Codebook Mapping via a learnable vector-quantized (VQ) codebook that clusters entities into latent semantic categories, even for unseen ones.
2.	Interaction Chain Encoding, which models temporal dynamics as ordered interaction sequences instead of unordered neighborhoods.
3.	Pattern Transfer, which propagates learned temporal dynamics within semantic clusters, preventing collapse and enabling inductive generalization.
Experiments across four standard datasets (ICEWS14, ICEWS18, ICEWS05-15, and GDELT) demonstrate significant performance improvements (average +28.6% MRR) compared to strong baselines such as LogCL, REGCN, and InGram. Ablation, sensitivity, and visualization analyses confirm the contribution of each component and show how TRANSFIR prevents embedding degeneration. The paper provides theoretical motivation, detailed methodology, and strong empirical validation.

### Strengths
1.	Clear Problem Definition and Motivation
The paper explicitly defines inductive reasoning on emerging entities — a setting rarely formalized before. The authors provide convincing empirical evidence that around one-quarter of TKG entities lack training interactions, motivating the need for inductive treatment. This establishes a meaningful gap between existing “closed-world” assumptions and real-world scenarios.
2.	Well-Designed Methodology
The Classification–Representation–Generalization pipeline is logically structured and technically coherent. Each stage (codebook clustering, interaction chain encoding, and pattern transfer) addresses a distinct aspect of the emerging-entity problem: type-level priors, temporal dynamics, and generalization.
3.	Empirical Rigor and Breadth
The experimental setup is comprehensive: four datasets, multiple categories of baselines (graph-based, path-based, inductive), and both strict Emerging and relaxed Unknown evaluation settings. Quantitative improvements and stable results across hyperparameters demonstrate robustness.
4.	Insightful Analysis and Visualization
The inclusion of t-SNE visualizations and the quantitative Collapse Ratio metric provides clear evidence that TRANSFIR effectively mitigates representation collapse. The cluster case study concretely illustrates transferable reasoning patterns.
5.	Clarity and Organization
The writing is technically clear, equations are well formatted, and the pipeline diagram helps convey the overall structure. The ablation and sensitivity analyses provide transparency regarding the influence of each module and hyperparameter.

### Weaknesses
1.	Limited Theoretical Explanation of Codebook Semantics 
The VQ-based codebook serves as the foundation for TRANSFIR’s semantic generalization, yet the paper offers limited theoretical or empirical analysis of what these latent clusters truly capture. Beyond a few illustrative examples, there is no quantitative assessment of the semantic coherence or stability of the learned clusters. It remains unclear whether the grouping behavior arises from shared linguistic semantics, co-occurrence frequency, or inductive biases in the embedding space. A more explicit discussion of how the codebook representation links to underlying entity semantics would strengthen the interpretability claim.
2.	Incomplete Scalability and Efficiency Evaluation
Although Appendix D.3 presents an asymptotic complexity discussion, the main text lacks direct empirical comparisons of runtime and memory usage with strong baselines such as REGCN and LogCL. Given that TRANSFIR integrates multiple computational stages—including codebook updating, transformer-based interaction encoding, and intra-cluster pattern propagation—a detailed runtime profile and resource breakdown on large-scale datasets would be valuable for assessing its real-world feasibility and computational efficiency.
3.	Sensitivity to Textual Initialization and Encoder Choice
The model initializes entity representations using fixed BERT-based textual embeddings, yet the influence of these pretrained representations is not examined. The paper does not analyze whether the model’s performance depends on the semantic quality of textual inputs, nor whether substituting alternative or domain-specific encoders would change outcomes. Since the codebook mapping step relies heavily on the textual embedding space, understanding this dependency is important for assessing generalization across domains or datasets with varying textual richness.
4.	Limited Exploration of Temporal Chain Configuration
The Interaction Chain length parameter defines the temporal window used for reasoning, but the paper provides minimal empirical or theoretical discussion on its effect. The impact of varying chain length on information propagation, noise accumulation, and temporal dependency modeling remains underexplored. A systematic analysis of how chain truncation influences accuracy and stability across datasets would clarify how TRANSFIR balances temporal coverage with computational overhead.
5.	Absence of Detailed Error and Failure Case Analysis
The qualitative examples focus on successful transfer cases and reduced collapse, but the paper omits analysis of failure conditions. Instances where semantic clusters merge unrelated entity types or where temporal transfer fails due to inconsistent interaction histories are not discussed. Identifying and characterizing such failure modes—especially on heterogeneous datasets like GDELT—would provide important diagnostic insights and demonstrate a more complete understanding of model behavior.

### Questions
1.	Causal Path Discovery Assumptions
The paper defines causal path discovery as the foundation of CausER’s reasoning process, but the assumptions that guarantee the validity of discovered causal paths remain implicit. Could the authors specify under what structural or temporal conditions the learned paths can be regarded as causally valid rather than correlational? Clarifying how the model ensures causal sufficiency and mechanism stability in multi-relational temporal graphs would help readers understand the theoretical boundary of the proposed intervention objective.
2.	Identifiability and Theoretical Guarantees
The theoretical section presents an identifiable counterfactual objective but does not detail how identifiability is maintained under partially observed temporal data. Are there specific assumptions—such as temporal faithfulness or stable mechanism transitions—that must hold for the causal estimator to remain unbiased? A more explicit discussion of these conditions and their relation to the structural causal model defined in Section 3.2 would strengthen the theoretical contribution.
3.	Causal Path Generator Efficiency and Scalability
The causal path generator explores multi-hop relational paths using differentiable interventions, which can be computationally intensive on dense graphs. Could the authors provide empirical runtime and memory profiles for this module on larger datasets such as GDELT? Including a quantitative comparison with baselines in terms of cost per epoch or per sample would clarify whether the causal discovery process scales efficiently to real-world graph sizes.
4.	Effect and Behavior of the Counterfactual Regularizer
The counterfactual regularizer is presented as a key mechanism that improves robustness to temporal confounding, yet its operational behavior is described qualitatively. Could the authors further explain how this regularizer alters the score distribution during training? For instance, how does it affect the relative weighting of causal versus spurious temporal correlations over epochs? More detailed training dynamics or representative examples would make its impact on model behavior clearer.
5.	Evaluation Protocol and Emerging Entity Setting
The paper emphasizes inductive generalization to unseen entities and uses chronological splits for evaluation. Could the authors clarify whether the evaluation explicitly separates emerging entities from known ones and whether metrics are reported both for emerging and overall subsets? Such clarification would allow more precise comparison with other inductive temporal reasoning frameworks and highlight how CausER handles first-appearance nodes.

### Soundness
3

### Presentation
3

### Contribution
3
