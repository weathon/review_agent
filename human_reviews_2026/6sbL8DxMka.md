# Gated Graph Attention Networks with Multichannel Fusion for Disease Comorbidity Prediction

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 2

## Abstract
The co-occurrence of multiple diseases, or comorbidity, significantly complicates clinical management and worsens patient outcomes. Comorbidity is believed to arise from genetic mutations functionally connected through protein-protein interactions (PPIs) within the human interactome. Unraveling these intricate PPI networks is essential for understanding disease progression and addressing the challenges posed by comorbid conditions. In this study, we propose a novel Gated Graph Attention Network (GGAT) framework tailored for disease comorbidity prediction by addressing issues that hinder the existing methods via three key aspects: (1) applying attention over local neighbors rather than global pairwise attention among all protein nodes, enabling more biologically meaningful aggregation; (2) incorporating a gating mechanism to adaptively regulate information flow and enhance representation learning for comorbidity prediction; and (3) introducing a multichannel fusion strategy that integrates connectivity based and disease association based embeddings, both of which have been shown to be important for disease comorbidity prediction. Experimental results on the benchmark dataset demonstrate that GGAT significantly outperforms the Transformer baselines across all metrics (AUROC, AUPRC, accuracy, and MCC), with the multichannel gated fusion variant achieving the best overall performance. These findings highlight the importance of integrating complementary biological features through graph structure and indicate that the proposed GGAT provides a generalizable graph learning framework applicable beyond disease comorbidity prediction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents a gated graph attention network (GGAT)  that predicts disease comorbidity using human protein-protein interaction graphs. GGAT utilize GAT layers to define attention based neighbor information aggragation. To avoid over smoothing, they insert GRU style gates between the GAT layers.These gates control how much new neighbor information to accept and how much of previous information to keep. Each protein has 2 types of initial embeddings, one connectivity embedding resulting from Node2vec with randomwalk on PPI network and one disease-association embedding resulting from Label2vec. The model combines these 2 embeddings with 2 different fusion model:EmbedFusion for early fusion, where it merges them into one embedding then they run GGAT or GatedFusion for late fusion, where it runs a GGAT for each type (in parallel)  then use a learned gate to fuse them into one embedding. To predict the comorbidity for a disease pair, the model gets the proteins linked to each one of the diseases and produce a disease-pair embedding using adaptive pooling, which is passed to a MLP classifier to output the probability of the 2 diseases being comorbid. The experiments showed that the GatedFusion based GGAT has the best results overall, compared to the Transformer (TSPE) baseline

### Strengths
- the usage of GRU gates between GAT layers to regulate information flow to see how much new neighbor information to accept and how much of the previous representation to keep is an interesting idea. 
 - they introduce the GatedFusion module that combines connectivity and disease-association embeddings for proteins with a learned gate
-leveraging protein–protein interaction networks for this Disease comorbidity prediction is well-motivated.

### Weaknesses
-Constribution is very limited as combining GAT and GRU, both the gating and fusion components are well-established concepts in prior graph learning works (e.g., GatedGCN, GCN-GRU).
-The claimed novelty of “multichannel fusion” is essentially feature concatenation followed by gating, which offers minimal conceptual advancement over prior multimodal GNN frameworks.
- The state of the art used for comparison only focus on TSPE which may not be enough to validate the model. It is not including gated networks or GAT models results.
- There is no clear theoretical analysis or ablation to demonstrate why gating specifically benefits comorbidity prediction beyond empirical results.
-Only one baseline (TSPE) is compared; missing comparisons with GCN, GatedGCN, and other biomedical GNN models limits the strength of the evaluation.
Ablations and hyperparameters study missing. There is no comparison with or without GRU gates, the different pooling types (why adaptive pooling in particular). It helps us see which parts matter most. 
- Paper writing should be improved. disease-association information is mentioned but not explained. How they obtained.
- including disease-association information information as the initial embeding may cause label leakege.

### Questions
Q1. Did you check if there is any information leakage when splitting into training and testing? Does a disease appear in both training and testing?
Q2. Are  Node2Vec and Label2Vec computed inside each training fold?
Q3. Could be usufull to Include an ablation study section where you show what happens if you remove GRU, if you replace adaptive pooling with mean or another type.
Q4. Could you please explain why GatedFusion is better than EmbedFusion
Q5. Did you try Node2Vec with different parameters values (p,q)? You can include a sensitivity study about it. Same thing applies to Label2Vec
Q6. You mentioned that TSPE has a concatenation cost, can you compare it with GGAT ?
Q7. Could you clarify the role of the Label2Vec embeddings? How are these embeddings trained — jointly with GGAT or pre-trained and fixed? How are diseases represented in the embedding space (e.g., one embedding per protein, per disease, or per disease pair)?

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
This paper studies disease comorbidity prediction on the human interactome using a Gated Graph Attention Network (GGAT) that couples local GAT message passing with GRU-style gating and introduces two fusion strategies (EmbedFusion and GatedFusion) to combine connectivity- and disease-association–derived embeddings

### Strengths
- Clear decomposition of signals (connectivity vs. disease association) and a simple, reproducible fusion design (single-channel vs. dual-channel) make the study easy to follow and reimplement.
- Consistent improvements across four metrics were observed.

### Weaknesses
- Much of the model is built from well-established components (GAT layers, GRU gating, etc.), so the methodological novelty appears limited.

- Since GAT and GRU are well known, enumerating nearly all equations is unnecessary; streamlining the math would improve focus on the new contributions.

- Protein connectivity and disease associations could be complemented by pretrained single-entity embeddings (e.g., ESM for proteins, BioBERT for disease text) to enrich representations.

- Experiments rely on a single data source with a general evaluation; deeper analyses of generalizability (e.g., across alternative interactomes or disease subsets) are missing.

- Important experimental details are insufficiently specified (e.g., data splitting protocol, hyperparameter selection/search), making it hard to assess rigor.

- The manuscript does not discuss any limitations.

### Questions
None

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The authors propose a multi–channel fusion-based Gated Graph Attention Network (GGAT) mechanism to predict disease comorbidity. The problem addressed is impactful and challenging. Authors use a number of graph attention networks and associated data representation. However, some of the complexity of the task are yet to be addressed. Survey of related work that appears in computational biology literature is incomplete.

### Strengths
1) Disease comorbidity prediction is a mature but challenging problem. Hence it is still relevant to the research community.
2) Paper introduces a Gated transformer based architecture, which regulates the information flow with respect to their importance.
3) Authors integrate protein-protein interaction network topology and disease-protein association connectivity for better feature learning in comorbidity prediction.

### Weaknesses
1) Disease comorbidity is a challenging problem, as comorbidity can never be explained through one or two types of biological or clinical modality. PPI interaction itself is a quite sparse, context dependent, and undirected network that fails to capture the complete etiological landscape.
2) Comorbidity arises from multilayer determinant features like genetic, epigenetic, transcriptomic, metabolic, immune, environmental stress and clinical. Apart from that longitudinal EHR data combined with pathway, regulatory, and phenotypic networks are crucial to incorporate with cohort specific PPI data.
3) The rationale of the proposed work is so naive as the incompleteness in PPI and over-represented well studied disease-associated proteins only give bias to the outcomes.
4) Relying on a Gated mechanism to “let-in” the most crucial features from the past information risks encoding temporal and modality bias: as the learned attention may overweight that are plentiful in the training graph, i.e., typically PPI topology and disease-associations while ignoring the heterogeneous context specific determinants of comorbidity (e.g., age, sex, medication, immune-state etc.).
5) As comorbidity is multi-causal and context dependent, so attention restricted to PPI and its well known disease association can yield spuriously high edge weights for well-studied protein-disease associations. To mitigate these, the authors should introduce multi-channel fusion on heterogeneous modalities explicitly, limit the attention to reflect recency context and causal plausibility.
6) The learned gate over multiple modality can well decide the crucial information to let in the model. That lacks the proposed work.
7) Node2vec embedding with protein-disease association mostly encodes PPI topology and disease-wise co-occurrence, thereby overemphasizing the popular hubs and homophily while ignoring context, directionality, clinical, and environmental stressors - an approach is too naive for the mechanistic complexity of disease comorbidity. Hence, making such embeddings too naive to predict comorbidity beyond superficial network co-localization.
8) The proposed work appears to equate comorbidity “robustness” with relative-risk weighted links between disease pairs. However, RR is a marginal, prevalence dependent association that (i) is even confounded by multi-dimensional factors like age, sex, treatments etc. (ii) is also vulnerable to Berkson’s bias, Simpson’s paradox, (iii) it lacks directionality and temporality (onset order lag), and (iv) unstable across hospitals, period, and sub-populations. Using RR as the sole input network risks mistaking exposure patterns for biology and overstates pair “robustness” without causal adjustment or tune-to-event modeling.

### Questions
1) Disease comorbidity prediction is a well studied problem, which is not only associated with genetic mutations and only PPI information never able to give insight to address the fact too. Even PPI information and disease-protein pairs never give insights about the proper disease progression mechanisms. Authors should provide a discussion on these challenges.

2) From line 75-77 what is the other information integrated and how?

3) Under sec. 2.3, there are plenty of approaches done on comorbidity prediction and analysis like tensor factorisation, gene co-expression based analysis. But the authors are quite naive into their literature survey. The literature survey should be expanded.

4) Under sec. 3.1 how the authors design the attention pooling on which information perspectives? 

5) How the "simple node2vec random path" and "protein-disease" association set can give most effective information in embedding learning, as it only provides PPI based topological and disease stratified protein information, which quite naive to define the complex trait of comorbid diseases?

6) from line 233-244, how do the authors think of letting entry be the most important feature to incorporate from the past information using the GATED approach ? As, there is plausible bias present in their embeddings to get learned from past (only) protein topological sequences and disease associations, as comorbidity depends on multiple heterogeneous factors.

7) In line 285. what are the different biological signals?

8) node2vec treats graph as a homogeneous one, hence it captures the PPI-based node-node connectivity information only, which is never a confirmatory way to give insights about novel comorbid diseases.

9) In line 319. how do the authors explain the reliability and robustness of the complementary information?

10) How do authors define the disease-disease pair robustness with relative risk scores only?

11) Heterogeneous biological signals can never be accumulated from PPI and disease association information only. Hence, the comorbidity prediction is supposed to be highly biased to Protein connectivity information only.

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
4

### Summary
This paper proposes a Gated Graph Attention Network (GGAT) framework for disease comorbidity prediction, aiming to address limitations of the state-of-the-art (SOTA) Graph Transformer with Subgraph Positional Encoding (TSPE) method. The framework’s core design includes three key innovations compared to TSPE: applying attention over local neighbors (instead of global pairwise attention) for more biologically meaningful feature aggregation, integrating a gating mechanism to adaptively regulate information flow and enhance representation learning, and introducing a multichannel fusion strategy to combine connectivity-based and disease association-based embeddings. Experiments show the effectiveness of the proposed model.

### Strengths
1. The paper has a clear structure and is easy to read

2. This paper proposes a Gated Graph Attention Network (GGAT) framework for disease comorbidity prediction. The overall model architecture seems reasonable.

### Weaknesses
1. Core components of GGAT lack originality. Local neighbor attention is a defining feature of standard GATs, the GRU-based gating mechanism replicates prior work on Gated GCN, and multichannel fusion adapts existing fusion techniques from GNNs and computer vision. The manuscript does not introduce innovations to these components (e.g., disease-specific attention weighting, task-aware gating) or provide theoretical justification for their combination beyond incremental performance gains.

2. The experiments are not convincing enough. The authors only compare to TSPE variants, omitting critical baselines like standard GNNs (GCN, GraphSAGE), recent comorbidity-focused models, and vanilla GAT. The baselines in 2025 should be compared. 

3. A key motivation for GGAT is addressing TSPE’s high computational cost from positional encodings. Can you provide quantitative comparisons of computational overhead (FLOPs, training/inference time, memory usage) between GGAT (especially GatedFusion with dual channels) and TSPE on the RR1 dataset? Does GGAT’s efficiency scale to larger interactomes or datasets?

### Questions
Please see the weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
