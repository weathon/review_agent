# HiBio-ST: A Hierarchical Multimodal Foundation Model with Biological Prior Anchors for Spatial Transcriptomics

- Decision: Reject
- Scores: 4, 4, 4, 6, 4

## Abstract
Spatial transcriptomics (ST) enables medical computer vision researchers to uncover the molecular relationships underlying tissue morphology. However, most existing vision–omics models are built on limited and homogeneous datasets, rendering them task-specific and with poor generalizability. Recent multimodal foundation models attempt to bridge histology and gene expression via contrastive objectives; however, they fail to effectively model spot-specific molecular context and overlook spatial dependencies by treating each spot–patch pair in isolation. To bridge these gaps, we present HiBio-ST, a novel hierarchical multimodal foundation model guided by biological prior anchors for ST analysis. HiBio-ST employs a progressive multi-level alignment pretraining pipeline to harmonize visual context with molecular identities. A TF–IDF reweighting strategy is first applied to highlight spatially informative “keyword” genes within ST profiles, reducing the dominance of ubiquitous housekeeping signals. Curated pathway anchors are then incorporated to inject global biological knowledge into the representation space. Moreover, hierarchical region-aware clustering united contiguous meso-scale regions into coherent structural patterns, allowing the model to capture higher-order spatial organization. We evaluated HiBio-ST on four downstream tasks across multiple datasets. Experimental results demonstrate that HiBio-ST consistently achieves state-of-the-art performance, underscoring its broad applicability in spatial transcriptomics modeling.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors propose HiBio-ST, a hierarchical multimodal foundation model for ST that: (1) reweights genes via TF-IDF to emphasize “keyword” genes and use ranked gene names as gene sentence , (2) provides a way to construct pathway priors using curated gene-set “anchors,” and (3) adds region-aware clustering to encourage meso-scale spatial coherence. While the results show improved compared to several SOTA methods, I am not fully convinced by the results that the gene ranking methods are more efficient and powerful than some current methods like spatial variable gene detection. Additional experiments are needed to support the claim.

### Strengths
The authors provide comprehensive experiments for various tasks. The idea of use external information like pathway gene sets is interesting and potential useful.

### Weaknesses
Additional experiments and discussion need to be added for a comprehensive comparison with SOTA methods.

1. The authors claim that "the high-expression-only strategy overlooks spot-specific signals and instead favors ubiquitous housekeeping genes". There are no evidence in the paper that support this claim, the authors should provide examples in certain datasets comparing the genes highlighted by TF-IDF or pathway and HVGs.
2. In practice, in a lot of spatial clustering methods, people use spatial variable genes (SVGs) instead of HVGs. How does TF-IDF ranked genes compare with SVGs? The author could consider benchmark against SVGs extracted using SPARK, Moran's I, and nnSVG.
3. It has been discussed in several papers (Genept, SpatialAgent) that the gene could be ranked in a better way than just using HVGs or mean/max gene expression. Please add discussion about them if they are not discussed. How does TF-IDF or pathway informed gene importance/weights compared to the gene importance proposed in Genept and SpatialAgent?
4. There are many overlapping pathways relevant to a given tissue (especially for tumors: immune, tumor invasion, and tumor–immune interaction). Please explain how you selected which pathways to include and whether this choice generalizes across tissues. And for healthy tissue like DLPFC, how should users choose pathway? Please add experiment about pathway informed genes for the DLPFC analysis.
5. The authors should add more comparison with recent multimodal or fine-tuned baselines, such as: fine-tuned CLIP (e.g., from STimage-1K4M), fine-tuned CONCH (e.g., from HEST-1K), and several other multi-modal framework, for example, UMPIRE, and STPath frameworks. These comparisons are essential to position your method against current SOTA.

### Questions
See Weaknesses for major questions. 

Minor questions:
1. For clustering task, the authors only evaluate zero-shot clustering task on DLPFC but not HER2+ breast cancer, while they evaluate linear probing on the HER2+ but not DLPFC. Please include the result for both datasets.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes HiBio-ST, a multimodal foundation model that integrates pathology images and spatial transcriptomics (ST) data for diverse ST analyses. It aims to overcome the poor generalizability of task-specific models and the limitations of existing multi-modal foundation models, which often (1) overlook spot-specific signals and (2) fail to capture the microenvironmental context in spot-patch pairs.

To address these issues, HiBio-ST introduces three components:
1. TF-IDF for Gene Selection: Selects genes based on relative scarcity rather than absolute expression, capturing important spot-specific signals.
2. Pathway-guided integration in spot-patch pairs: Incorporates pathway information to apply global biological constraints to spot-patch pairs.
3. Region-Aware Hierarchical Clustering: Uses Leiden clustering enhanced by edge weights derived from both spatial distance and feature cosine distance to better reflect biological structures. 

These components are aligned using a contrastive loss objective. Applies contrastive loss objectives to simultaneously align visual and gene embeddings at the local (spot), functional (pathway), and meso-scale (region) levels.

### Strengths
* The paper clearly explains its key motivation. (e.g., needs for foundation model in this ST domain, limitation of gene selection)
* The paper is well-written and easy-to-follow.

### Weaknesses
* While the claim that a foundation model is needed for better generalizability compared to task-specific models is valid, the zero-shot clustering performance presented to substantiate this is very poor. Specifically, the performance is significantly lower than that of representative task-specific models [1, 2] that use only spatial transcriptomics. Consequently, the proposed method does not appear to support its stated motivation.
* The methodology is considered incremental, as it primarily combines existing TF-IDF gene selection from the single-cell domain [3] with community detection [4, 5].
* The method section lacks a clear explanation of how the individual components of HiBio-ST are integrated, particularly in Section 2.4. For instance, it is unclear how the pathway prototype representation, which is an aggregation of gene embeddings, is applied in Equation (11).
---
[1] Deciphering spatial domains from spatially resolved transcriptomics with an adaptive graph attention auto-encoder. Nature communications. 2022.

[2] Global Context-aware Representation Learning for Spatially Resolved Transcriptomics. ICML. 2025.

[3] Single cell RNA-seq data clustering using TF-IDF based methods. BMC genomics. 2018.

[4] Modularity and community structure in networks. PNAS. 2006.

[5] From Louvain to Leiden: guaranteeing well-connected communities. Scientific Reports. 2019.

### Questions
* There is no definition for $G_s$. Is it the set of expressed genes within a specific spot?
* The paper argues that a foundation model is necessary for generalizability, but the reported zero-shot performance is poor. Even if it is inferior to task-specific models, shouldn't the performance be at a comparable level? Is a direct comparison with these task-specific models available?
* Alternatively, are there other ways to demonstrate generalizability from a different perspective?
* With technological advancements, newer, higher-resolution spatial transcriptomics technologies like Xenium (which is part of HEST-1k) are available. How does HiBio-ST perform on this type of data?
* TF-IDF-based gene selection performs better than existing HVG (highly variable genes) and SVG (spatially variable genes) selection?
* How about parameter sizes and time complexity for pretraining and inference comparing with baselines?

### Soundness
2

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
4

### Summary
This paper introduces the HiBio-ST pretraining framework, designed to integrate histology images and gene expression for spatial transcriptomics analysis. The model first employs a TF-IDF reweighting scheme to highlight spatially informative genes. It then incorporates curated pathways to inject biological priors and guide the alignment. Finally, a hierarchical, region-aware clustering method groups spots into coherent meso-scale structures, allowing the model to capture higher-order spatial patterns. The authors evaluate their model on multiple downstream tasks and achieve better performance than the baseline method.

### Strengths
1. The idea of using the TF-IDF reweighting to find important genes and integrating pathway anchors to inject biological priors is novel.
2. The authors evaluate their method on several datasets, including gene expression prediction, zero-shot layer clustering, and image-to-sentence retrieval, with good performance.

### Weaknesses
1. The evaluation is limited to a few datasets/organs. Why not evaluate the HEST benchmark to show the model’s generalizability? Consequently, it’s insufficient validation for a called "foundation model".
2. The paper lacks an analysis of important hyperparameters
3. The choice of K in the TF-IDF reweighting scheme and its impact on performance.
- The threshold used for the pathway-guided alignment, and how different values (higher vs. lower) would affect the results.
- The paper would be significantly strengthened by providing visualizations that illustrate how the region identifiers evolve during the training process.
4. The authors should provide an ablation study on the weights for different loss component to show their contribution to the final performance.

**Minor**:
1. The reported performance of some baselines (such as TRIPLEX) appears unusually low. The authors should verify their implementation or justify the result.
2. The paper lacks justification for the chosen image encoder. It is unclear why more recent, domain-specific vision foundation models (e.g., UNI or Virchow) were not considered, as they might offer stronger performance.
3. The authors should include a comparison against more recent, relevant models (such as CONCH [1]) as a baseline for the image-to-ST sentence retrieval task.
4. Clarification for Eq. 11: The equation computes a KL divergence using "the model's predicted probabilities," but it is not apparent which component of the model (as shown in Figure 1) generates these probabilities. The authors should clarify this part.

[1] Lu, Ming Y., et al. "A visual-language foundation model for computational pathology." Nature Medicine 30.3 (2024): 863-874.

### Questions
Please see the Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces HiBio-ST, a new pretraining method for spatial transcriptomics foundation model. Instead of normally used SVG, HVG embedding/processing of gene table, each site read out is first ranked by TF-IDF algorithm on the gene sections, then chosen the top-k most genes from this ranking as a gene sentence. After this set of sentences is constructed, these sentences are being cross-referenced with the ‘pathway’ which are known biological functions of sets of genes (KEGG pathways is what the authors chose). On the imaging side, instead of just learning the site level embeddings independently, the authors try to incorporate regions that are essentially clusters generated from spatial locations and image-gene similarity. The dataset

### Strengths
1. This paper is well-written. It is clear, easy to follow, and well-motivated.
2. This paper tries to improve the foundation model pre-training specifically on the ST dataset. It introduces several novel points. First it treats the gene as sentences and uses the NLP method to conduct initial dimension reduction. Secondly, it has a biology reference bit where it incorporates biological knowledge as priors which prior works lack. Finally, it improves the imaging side to allow spots to have regions or a spatial sense rather than being treated independently.
3. The reviewer really appreciates the clear and logical flow of the ablations study or the flow of paper writing in general. While reading the methods sections, the reviewer writes down a few questions on the different design bits, but they are mostly answered or at least experimented in the ablation section.

### Weaknesses
1. The reviewer thinks Figure 3 should be in appendix as it is a zero-shot layer clustering task where the qualitative results are all pretty bad or different from the human annotation. Putting it here takes a lot of space but not telling too much information.
2. The figures shown in the paper are in general too small, at least in an arms length distance.
3. If treating gene readout as keywords or sentences, there should be some other way to conduct such dimension reduction or ranking or selection. This work does not consider or discuss those possibilities.
4. Clustering into regions does provide spatial correlations between spots but there is a great amount of work incorporating graphs for spatial correlations, this work does not mention why or why not those were not being considered.

### Questions
1. For the four tasks, since the reviewer does not come from a computational biology background, it would be hard for the reviewer to assess even excelling in these tasks. How useful would a ST foundation model be applied if given to all the cancer researchers?
2. Is the KEGG pathway the only or complete definition of genes? Is the field of gene pathways still involved and at what speed? Do the KEGG pathways be useful or correlated for the gene set in the dataset? If Visum changed their generation or readout protocols would these prior still hold? Will biological knowledge shift or change? (all the above is quite similar question, the reviewer thinks that the authors should illustrate more on the importance or stability of such priors)
3. Is the choice of top-K would be similar to the experiment for the 100-500 tokens?
4. How does the choice of ViTs and Text encoders or image/gene encoders in general affect the performance or pretraining? Would foundation models from each modality (pathology and RNA) help here?


The reviewer is holding a positive view on this paper as it flows nicely and the ablation study is well-done. The reviewer is giving a borderline acceptance and willing to change scores during the discussion session.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This study presents a novel visual-omics foundation model, HiBio-ST, that integrates spatial and biological pathway information through a TF-IDF reweighting scheme and hierarchical modeling. The framework highlights spatially informative genes, incorporates pathway-level priors, and introduces a region-aware contrastive alignment mechanism. The model is evaluated on several downstream tasks, including clustering, and gene expression prediction.

### Strengths
1. This paper lies in its unique incorporation of spatial and biological pathway information into a visual-omics foundation model, effectively bridging morphology and molecular representation learning. A key innovation over existing approaches is the TF-IDF reweighting strategy, which selects genes that are locally enriched (captured by the TF term) but not globally expressed (filtering out housekeeping genes), a biologically meaningful and effective approach.

2. The paper also presents a series of ablation studies that demonstrate the contribution of each model component to the overall framework.

### Weaknesses
Despite the methodological novelty, the experimental validation leaves several important aspects unconvincing. In particular, the spatial clustering results on the DLPFC dataset are not strong, the predicted domains fail to recover clear laminar structures and appear inconsistent with the known cortical anatomy. This raises questions about whether the proposed biological priors truly enhance spatial representation quality. The manuscript would benefit from additional sanity checks showing that the embeddings preserve biologically meaningful relationships (e.g., spots from the same cortical layer or with similar cell-type compositions clustering together).

Furthermore, while the benchmarking includes comparisons with other foundation models, it omits direct comparisons with task-specific spatial analysis tools such as GraphST or STAGATE, which are designed for spatial domain detection. Without these baselines, it is difficult to assess whether HiBio-ST offers practical improvements for biologically relevant tasks.

Finally, although the method is systematically evaluated across multiple tasks, the presentation tends to emphasize quantitative improvements without sufficient biological interpretation or qualitative analysis, making it difficult to judge whether the gains are meaningful from a biological perspective.

### Questions
1. Could the authors clarify, in the ablation study under w/o TF-IDF, whether the same number of genes (250) are used but selected based on top-expressed genes? This clarification is important to demonstrate the advantage of the TF-IDF strategy over the commonly used selection approach.
2. Could the authors perform a sensitivity analysis on the threshold used for the indicator of the pathway-patch alignment score?
3. Could the authors explain the specific purpose and biological interpretation of each downstream benchmarking analysis?
4. Could the authors clarify the transformations applied (e.g., log(1 + x)) for each benchmarking method and state whether the implementations are consistent and fair?
5. Could the authors repeat the experiments with different random seeds to evaluate the stability of the results under randomness?
6. Could the authors report the results of spatial clustering without the spatial label-smoothing mechanism, as HiBio-ST theoretically already captures spatial information and should outperform other methods without redundant spatial regularization?
7. It is unclear whether the authors considered simpler baselines or alignment strategies, such as using a single-scale InfoNCE loss without hierarchical extensions. The proposed framework appears to stack multiple existing components (TF-IDF weighting, pathway priors, spatial clustering, and multi-level contrastive learning) without a clear unifying principle or demonstrated necessity. The ablation studies only show marginal differences and do not isolate whether the hierarchical contrastive design truly contributes beyond increased model complexity. Could the authors justify why this multi-stage architecture is required, and provide evidence that its improvements are not merely due to overparameterization or better hyperparameter tuning?

### Soundness
2

### Presentation
3

### Contribution
2
