# Graph-Enhanced EEG-to-Text Decoding: A Spatio-Temporal Relational Embedding Framework for Brain Signal Translation

- Avg Score: 2.00
- Decision: Reject
- Scores: 0, 2, 2, 4

## Abstract
Despite recent progress in brain–computer interfaces (BCIs), decoding natural language directly from EEG remains a critical challenge. Existing EEG-to-text models primarily treat signals as sequential time series, which severely limits their ability to capture the spatial and temporal relationships among electrodes and limits the possibility of generalization in low-data regimes. To address this challenge, we propose a novel graph-enhanced framework to explicitly model relational information in brain signals. The key idea of our framework is to construct Spectro-Topographic Relational Graphs (STRG) that jointly encode static electrode topology and dynamic inter-channel functional connectivity. From these graphs, we derive Spatio-Temporal Relational Embeddings (STRE), which provide graph-aware representations for downstream sequence-to-sequence decoding. Specifically, (i) STRG captures spatial adjacency and frequency-specific connectivity, (ii) STRE transforms these relational structures into embeddings aligned with text decoding, and (iii) the overall framework integrates these embeddings with a neural decoder to generate natural language outputs. To the best of our knowledge, this is the first graph-enhanced approach for EEG-to-text decoding that explicitly uses graph-based representations of EEG signals. Empirical results show that our framework delivers substantial improvements over strong recurrent and Transformer baselines. In particular, our Graph-Enhanced EEG-to-Text Decoding achieves up to 16% relative gains on BLEU-4, which highlights the effectiveness of relational graph modeling for advancing neural decoding.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The authors propose a graph-enhanced framework to explicitly model relational information in brain signals and decode natural language directly from EEG. Specifically, spectro-topographic relational graphs are constructed, followed by the spatio-temporal relational embeddings for downstream decoding. However, the significance of the work is not clear and the technical details, dataset splitting, and experimental settings are missing. Besides, the technical contributions are limited.

### Strengths
Graph representation learning for a specific domain of EEG: EEG-to-text decoding

### Weaknesses
[1] Graph representation learning – Graph representation learning has been extensively explored in the EEG field. The authors should clarify their unique technical contributions.\
[2] Related works – The authors should significantly improve their related works.\
[3] Equations – The authors should describe all the notations in their work. For example, the authors should indicate the covariance and standard deviations in Equation (1).\
[4] Methodology – The method that the authors presented (Graph – GAT – Transformer – CLIP-like contrastive learning) is unfortunately not novel in the EEG field and is not enough for publication in ICLR.\
[5] Baseline – The authors are encouraged to compare their work with models in the field, such as EEG2TEXT
Liu et al., EEG2TEXT: Open Vocabulary EEG-to-Text Decoding with EEG Pre-Training and Multi-View Transformer.\
[6] Metrics – The authors should indicate which specific BERT model they used for BERTScore. And indicate which ROUGE did they use, e.g., ROUGE-1 or ROUGE-L? More details should be revealed.\
[7] Experimental settings – The dataset splitting and experimental settings are missing in the manuscript. The authors should clearly indicate the training, validation, and testing sets. Besides, the specific experimental settings are missing, e.g., subject-dependent cross-session experiments, subject-independent experiments, etc.\
[8] Significance – The motivation and significance of the work, as well as the topic, is not convincing.

### Questions
[1] How exactly did the authors build the graph with the encoding of the functional connectivity? And how exactly is the “dynamic” functional connectivity built?\
[2] What is the significance of the work and the topic?\
[3] What is the dataset splitting and experimental setting?\
[4] What is the technical contribution of the authors’ work? For example, what is the contribution to the graph representation learning?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores incorporating graph structures into EEG-to-text decoding, aiming to explicitly model spatial and functional relationships among EEG channels. The authors construct an electrode graph based on channel topology and correlation, apply a graph neural encoder to extract spatio-temporal features, and align EEG and text embeddings through contrastive learning for downstream text generation. While the motivation of introducing graph topology into EEG modeling is relevant and well justified, the work appears incomplete in terms of design, experimentation, and presentation.

### Strengths
1.	The motivation is valid and well aligned with the challenges of EEG decoding. Incorporating electrode topology and inter-channel relationships is a meaningful direction that has been underrepresented in EEG-to-text studies.
2.	The idea of using graph representations to capture spatial dependencies could, if properly developed, lead to more neurophysiologically grounded models.
3.	The paper identifies a real limitation of current transformer-based EEG encoders, which generally ignore spatial structure.

### Weaknesses
1.	The work is clearly incomplete. Figures and tables appear preliminary, and the narrative lacks coherence. Many implementation details are missing, and visualizations are too rough to illustrate the model’s behavior.
2.	The proposed graph modeling is not well justified. The dynamic edge definition is problematic—high correlation between EEG channels may arise from artifacts or shared noise sources rather than genuine functional coupling. Without physiological constraints or validation, the learned graph structure is difficult to interpret.
3.	Experimental validation is insufficient. The dataset is small, baselines are limited, and there are no meaningful ablations or cross-subject evaluations. Reported improvements are minimal and may fall within statistical noise.

### Questions
1.	How are dynamic edges defined in practice, and how do you control for correlations driven by noise or volume conduction rather than functional connectivity?
2.	What measures did you take to ensure the graph structure reflects physiological plausibility rather than arbitrary channel correlations?
3.	Do you plan to extend this work with richer data (e.g., multi-session EEG or MEG) and stronger baselines to validate the approach?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a framework that explicitly models spatial and spectral relationships in EEG signals to improve natural language generation from brain data. The authors introduce Spectro-Topographic Relational Graphs (STRG), where each node corresponds to an electrode–frequency band pair and edges encode both static electrode topology and dynamic functional connectivity among channels. They then derive Spatio-Temporal Relational Embeddings (STRE) by applying graph neural networks (Graph Attention Networks) on STRGs and feeding the resulting graph-aware features into a temporal Transformer encoder. A Transformer-based decoder finally generates text from these embeddings. The approach is evaluated on the ZuCo dataset.

### Strengths
- The idea of Spectro-Topographic Relational Graphs (STRG) is novel and motivation is good. By explicitly modeling electrode adjacency and functional connectivity in each EEG segment, the model captures spatial patterns that sequential models ignore. It's a nice attempt of injecting prior layout knowledge to the encoder. Previous attempts are mostly in sleeping stage prediction.

### Weaknesses
- The weakness is also related to graph based encoder, use node to represent node and edges to suggest the layout and spatial relationships are not very novel, at least it has been applied on to other domains of EEG for times, such as sleeping stage prediction and driving drowness prediction. The novelty is somewhat incremental. Meanwhile, considering EEG-to-Text domain has apears a bunch of papers pointing out the alignment and probabaly some training schemas plays more improtant role. 

- Experimental results are not strong enough for support the claim. Ablation study is not very convincing, with graph and without graph results are some but not determinastic. Meanwhile, these follow-up works mentioned "teacher forcing" setting and the necessity of comparing with random input for two years. The random fluctuation of performance is around that range as well.

### Questions
1. When designing the connectivity of the graph, how the edges been defined between nodes, are the edges are desided with human prior knowledge?

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
4

### Summary
This paper proposes a graph-enhanced framework for decoding natural language from EEG signals. The authors address a key limitation of existing EEG-to-text models, which treat signals as sequential time series and thereby ignore spatial relationships among electrodes and frequency-specific connectivity patterns. Their approach constructs Spectro-Topographic Relational Graphs (STRG) that jointly encode static electrode topology (based on physical scalp placement) and dynamic functional connectivity (derived from inter-channel correlations). These graphs are then processed through Graph Attention Networks (GATs) to generate Spatio-Temporal Relational Embeddings (STRE), which serve as input to a Transformer-based decoder for text generation. The framework is evaluated on ZuCo datasets and reports improvements of up to 16% in BLEU-4 over baseline methods including BiLSTM, BART, DeWave, and E2T-PTR.

### Strengths
1. The paper effectively identifies the limitation of sequential models in capturing spatial relationships, which is a legitimate gap in EEG-to-text decoding.
2. The approach makes sense since STRG design reflects known spatial and functional EEG properties.
3. Comparison against four different baseline paradigms (recurrent, Transformer, discretized embedding, contrastive pretraining) provides reasonable coverage.

### Weaknesses
1. Only two datasets of reading EEG; unclear if results generalize to spontaneous speech or other subjects.
2. The paper lacks statistical analysis, e.g. error bars, significance tests, or multi-seed evaluation. Given small sample sizes, results could be statistically insignificant.
3. No visualization or neuroscientific analysis of learned graphs: interpretability claims remain unsubstantiated.
4. The hyperparameters $\alpha$, $\beta$, $\lambda_{1}$, and $\lambda_{2}$ are introduced in Sections 4.2 - 4.4 within Eqs.~(3) and the text following Eq(7), but their specific values or selection procedure are not reported. Moreover, no sensitivity analysis or tuning discussion is provided to assess the impact of these parameters on performance.
5. Limited reproducibility details. Missing information includes: exact hyperparameters (learning rate, batch size, number of epochs, GAT layers, Transformer layers), data preprocessing pipeline specifics, training procedure (optimizer, scheduling, early stopping criteria). Code availability not mentioned

### Questions
1. How is overfitting controlled given small data? Was early stopping or dropout used?
2. Please clarify if the contrastive loss uses frozen or jointly trained text embeddings?
3. How were the hyperparameters $\alpha$ and $\beta$ in Equation~(3) chosen? What is their sensitivity to performance?

### Soundness
3

### Presentation
2

### Contribution
3
