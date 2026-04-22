# From Spatial Transcriptomics to Tokens: Generative Pre-Training with Byte-Pair Encoding

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Generative pre-trained models have achieved remarkable success in natural language processing and computer vision, while spatial single-cell transcriptomics has emerged as a powerful tool for investigating disease mechanisms. The current methods largely overlook the impact of RNA spatial organization on cellular identity and disease processes, which may lead to the loss of RNA co-localization information, incomplete spatial transcriptome analysis, and insufficient investigation of disease mechanisms, thereby missing critical strategies for clinical diagnosis. To address the above issues, we propose STBPE (Spatial Transcriptomics Byte Pair Encoding)，a pre-training framework that focuses on subcellular resolution. This framework innovatively integrates “spatially aware byte pair encoding strategies”, by converting subcellular localization information of RNA within a single cell into serialized token units, achieving precise digital representation of RNA spatial distribution patterns. Specifically, it first uses a spatial omics data-driven word segmentation algorithm to encode the spatial coordinates and transcript features of RNA into a unified byte pair sequence. Then, it adopts the BERT style masked self supervised learning paradigm to randomly mask partially spatially aware labels and reconstruct the original sequence, forcing the model to learn deep embedding representations that contain spatial position information. This design enables STBPE to capture the potential association between RNA spatial distribution and gene expression, significantly improve cell type annotation, uncover co-localized RNAs associated with cellular identity from a new perspective, and pave the way for building multimodal foundation models that integrate spatial transcriptomics with natural language.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents STBPE, a tokenization strategy that adopts the BPE algorithm to facilitate representation learning with subcellular spatial transcriptomics. The authors introduce relative orientation and distance as additional features for each token and adopt CBOW to extract the low-dimensional embedding for each token. Experiments show that STBPE outperforms prior spatial clustering methods on cell-type annotation.

### Strengths
- Representation learning for subcellular spatial transcriptomics is a novel research direction.
- A series of qualitative and quantitative analyses is conducted, which reveals several co-occurrence patterns of RNA molecules and how they are correlated to cell identity.

### Weaknesses
- The motivation for using BPE in subcellular spatial transcriptomics is unclear. It appears that the ultimate goal is to perform self-supervised learning on the multi-modal ST data with 2D coordinates and gene identity. Such data closely resemble 3D molecules, proteins, or cell slices, and a broad range of works have proposed to address the problem [1, 2, 3]. The authors should justify the adoption of BPE rather than strategies like vector quantization [4] or auto-encoding [5] for handling ST data. Moreover, the cell representation model is directly trained on downstream tasks **without pre-training**, raising concerns about generality on data obtained from different species, slices, and ST sequencing technologies. 
- The presentation is inconsistent and confusing. For example:
  - In Lines 47-49, why do existing approaches fall short in handling new subcellular ST data?
  - In Line 106, what are the limitations of existing methods?
  - In Lines 115-116, the authors claimed that "graph-based approaches rely solely on topology or adjacency". However, it's a common practice for graph models to encode distances and orientations with edge embeddings [6].
  - The authors mentioned a BERT-style masked modeling, which is not presented in the methodology section.
  - The thresholds for relative positions are unknown.
  - In Lines 260-264, what's the definition of "merging criteria", "relevant attributes", and "keep on line, delete one line"?
  - The training objective and hyperparameters for CBOW and cell-type classification are unknown. Details of the position embeddings and network architecture are missing.
  - The baselines are not properly cited or introduced in detail. The evaluation metrics are not introduced.
- The proposed framework is technically unsound. For example, the relative position introduced in Table 1 is not robust to different orientations of the cell or the whole slice (e.g., if the cell is rotated clockwise by 90°, the "Above" position becomes "Right"). Besides, the adoption of rectangular distance rather than Euclidean distance is questionable, especially when the X or Y coordinates of RNA molecules are close. Moreover, the RNA pair token is expanded by searching for the nearest RNA molecule, even if it's far from the current position.
- The baselines are weak. Comparisons between single-cell models like scGPT [7] and scGPT-spatial [8] should be presented.
- The paper violates the double-blind review by providing an unanonymous GitHub link in Appendix C.

Refs.

[1] Uni-Mol: A Universal 3D Molecular Representation Learning Framework

[2] SaProt: Protein Language Modeling with Structure-aware Vocabulary 

[3] SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics

[4] Neural Discrete Representation Learning

[5] Recent Advances in Autoencoder-Based Representation Learning

[6] Position-aware Graph Neural Networks

[7] scGPT: toward building a foundation model for single-cell multi-omics using generative AI

[8] scGPT-spatial: Continual Pretraining of Single-Cell Foundation Model for Spatial Transcriptomics

### Questions
My major concerns have been listed in the weaknesses above. Here are some minor questions:
- What's the ultimate vocabulary size? What's the distribution for the number of RNA molecules in the dictionary? How is the compression rate of the input data using STBPE?
- In Table 3, why does a larger embedding size lead to suboptimal performance (e.g., dimension collapse [1])? How is the performance of STBPE-128 and STBPE-512 with position embeddings?
- What's the meaning of the different colors in Figure 6?

Refs.

[1] Understanding Dimensional Collapse in Contrastive Self-supervised Learning

### Soundness
2

### Presentation
1

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
This paper proposes a novel method named STBPE, aimed at addressing the challenges of analyzing subcellular-resolution spatial transcriptomics data. The core idea, inspired by Byte-Pair Encoding from natural language processing, is to serialize the spatial distribution of RNA molecules within a cell into "spatial tokens" while preserving relative positional information between RNAs, and to build a generative pre-training model.

### Strengths
1.The adaptation of the BPE algorithm from NLP to spatial transcriptomics, incorporating geometric information (distance, angle), is a creative and worthwhile direction. The problem definition is clear, and a distinction is made from traditional GraphBPE.

2.The writing is generally fluent, figures are abundant, and the methodological workflow is illustrated with diagrams, making the core ideas easy to understand.

### Weaknesses
1.The core of the algorithm, "statistical high-frequency RNA pair merging" described on Page 5, is unclear in its specific mechanism. for instance, what exactly is the "merging criteria" (pure frequency threshold, or combined with spatial distance)? how is information loss or bias prevented due to greedy merging? The specific implementation and necessity of the direction reversal function are also unexplained.

2.The "dual spatial position encoding" mentioned in Equation (1) is only briefly referenced in the main text. its specific composition, how it is combined with the initial gene embedding, and why it is more effective than incorporating positional information during preprocessing are not detailed. This makes it resemble a "black-box" operation, severely compromising the method's understandability and reproducibility.

3.Figure 4 is too unclear, and in Page 7, the association between listed tokens (e.g Apq4+Cxc114) and cell types is merely observational description, without providing any statistical significance testing (such as enrichment analysis p-value), which casts doubt on the reliability of these findings.

### Questions
1.Why was the evaluation limited to cell type annotation? how does STBPE perform on tasks that truly reflect spatial modeling capabilities, such as spatial domain identification or ligand-receptor co-localization analysis?

2..The paper refers to this work as "generative pre-training," but the described tasks are context prediction and masked reconstruction. please clarify where the "generative" aspect of the framework lies? alternatively, consider revising the terminology to avoid confusion.

### Soundness
3

### Presentation
2

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
The paper introduces STBPE, a spatial tokenizer for subcellular spatial transcriptomics that serializes RNA coordinates and expression into tokens. Embeddings are learned with CBOW/Word2Vec and a BERT-style masked prediction objective; the tokens are then used for cell-type annotation and interpretability analyses.

### Strengths
- A new way to tokenize subcellular spatial data, which could potentially be a fundamental element for building the spatial model.
- Evaluated at scale with ablations indicating the importance of positional encoding; interpretability analyses (IG) highlight biologically coherent token patterns.

### Weaknesses
Although the idea is interesting, the presentation is not very clear and the nessarity of using BPE instead of normal tokenization method is lack.

### Questions
1. You describe the framework as “generative pre-training,” yet the methods emphasize CBOW and BERT-style masking. Is any generative objective used? If not, please revise the terminology and abstract for precision.

2. In Section 3, what is the role of distance and angle features? How do they contribute to the problem formulation and downstream performance?

3. Why not directly encode each transcript with 2D sinusoidal positional embeddings combined with gene embeddings? This approach has been explored before in [1].

4. Related to Q3, what is the performance of this baseline strategy on the cell type annotation task compared to your method?

5. Why restrict the model to only eight positional relationships? More ablation studies would clarify the impact of this design choice.

6. How are the iteration numbers in the tokenization process determined? Is there a stopping criterion or sensitivity analysis?

7. In Figure 4, cells appear with different shapes. Could these morphological differences act as confounding covariates that bias the cell type annotation results?

[1] Wen, Hongzhi, et al. "CellPLM: Pre-training of Cell Language Model Beyond Single Cells." The Twelfth International Conference on Learning Representations.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose STBPE (Spatial Transcriptomics Byte Pair Encoding), a novel generative pre-training framework designed to analyze subcellular resolution spatial transcriptomics data. The core contribution is a "spatially aware byte pair encoding" strategy that converts the spatial coordinates and transcript features of individual RNAs within a cell into serialized token sequences. This tokenization aims to preserve the spatial relationships between RNAs. The paper then employs a BERT-style masked self-supervised learning paradigm to learn deep embeddings from these token sequences. The method is evaluated on a MERFISH dataset for the task of cell type annotation, where it reportedly outperforms several other methods and uncovers biologically relevant RNA co-localization patterns.

### Strengths
The primary strength of this paper is its **originality**. The core idea of tokenizing subcellular RNA distributions is novel and distinct from existing bert\t5\gpt style RNA Foundation Models. By attempting to convert sparse 2D point-cloud data (RNA locations) into a 1D sequence suitable for transformer-based models, the paper introduces a new and creative perspective for applying powerful generative pre-training paradigms to the spatial biology domain. This "spatially aware" BPE-like algorithm is a clever conceptual contribution.

### Weaknesses
Despite the novel idea, the paper suffers from several weaknesses that limit its current impact and the soundness of its conclusions:

1. **Insufficient Baseline Comparisons:** The experimental evaluation is missing comparisons against several key and recent baselines that also model spatial context or utilize large-scale pre-training for spatial data. Notably, methods like **Nicheformer**, **CellPLM**, and **GeneCompass** are not included. Without comparing against these relevant works, it is difficult to accurately assess whether STBPE represents a state-of-the-art contribution.
2. **Limited Downstream Task Evaluation:** The model's capabilities are only demonstrated on a single downstream task: cell type prediction. While this is a standard benchmark, the paper's abstract and introduction promise broader insights into "disease mechanisms" and "cellular identity." A model that captures detailed subcellular spatial patterns should theoretically be applicable to other, more spatially-native tasks, such as **niche prediction** or **neighborhood gene expression generation**. The limited evaluation makes the learned representations feel under-utilized.
3. **Single Dataset and Assay:** The model is only trained and evaluated on a single MERFISH dataset. The generalizability of this tokenization strategy to other subcellular spatial technologies, such as **10x Xenium** (which also provides transcript-level spatial coordinates), is not explored. This makes it unclear if the method is robust to data from different platforms or tissues.
4. **Lack of Scalability Analysis:** The paper does not provide any analysis of the **scalability** of the STBPE tokenization algorithm or the subsequent model training. How does the method's performance (in terms of compute time and memory) scale with the increasing number of transcripts per cell, gene panel size, and total number of cells? This is a critical factor for a pre-training framework.
5. **Clarity of Figures:** Figure 1, which illustrates the overall framework, is overly simplistic and lacks sufficient detail. Specifically, **Figure 1A** and **Figure 1C** are presented as high-level concepts without adequate description of the actual processes, making it difficult for the reader to fully grasp the technical workflow.

### Questions
**Baseline Comparisons:** Can the authors justify the omission of Nicheformer, CellPLM, and GeneCompass as baselines? Or, preferably, could they provide a comparison against them?

**Methodological Clarity (Sec 4.1.1):** Could you please elaborate on the problem of "non-uniform RNA coordinates across cells"? What is the specific nature of this non-uniformity, and what are the limitations of simpler approaches (e.g., simple centering) that necessitate the proposed grid-quantization framework?

**Generalizability (Datasets):** How do you expect STBPE to perform on an Xenium dataset, which may have different characteristics (e.g., density, panel size) than MERFISH? Have you considered training a model on a **mixture of MERFISH and Xenium data** to test its robustness and potential as a cross-platform foundation model?

**Generalizability (Tasks):** While dataset labels may be a limitation, could you discuss the feasibility of applying STBPE to other downstream tasks? For example, could the learned token representations be used to predict cell-cell interactions or to generate *in silico* gene expression patterns in a spatially-aware manner?

### Soundness
3

### Presentation
2

### Contribution
2
