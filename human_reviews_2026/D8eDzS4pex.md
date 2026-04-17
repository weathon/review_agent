# LEARNING DISCRETE REPRESENTATIONS TO UNDER- STAND AND PREDICT TISSUE BIOLOGY

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Learning tissue-level representations that capture the organization of entire tissues while preserving cellular and microenvironmental detail is a central challenge in spatial biology. While graph autoencoders have been employed to learn spatially aware continuous representations, they have limited utility for tissue-level generation, lack inherent interpretability for biological analysis, and are not readily reusable across contexts and modeling architectures. To address this challenge, we present SQUINT, a discrete representation learning framework for spatially-resolved transcriptomics that encodes tissues into a finite vocabulary of interpretable discrete codes. SQUINT achieves this by combining graph neural networks with vector quantization, conditioning on relative spatial distances, and employing a masking strategy during training. Cells are then represented by assignments to this shared vocabulary, allowing whole tissues to be modeled as sequences of discrete tokens. At inference, SQUINT codes enable gene expression imputation at arbitrary spatial locations outperforming state-of-the-art generative methods across diverse datasets. Further, we demonstrate the interpretability of these discrete tokens in capturing meaningful tissue structures beyond individual cells and reflecting recurrent mi-
croenvironmental organization patterns through downstream applications including 3D imputation, tumour stratification, and perturbation analysis.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes SQUINT, a discrete representation learning framework for spatially resolved transcriptomics (SRT). The method combines a GNN encoder, vector quantization, and conditional masking to tokenize cells into discrete codes. The learned tokens are used for tissue-level generative modeling, tumor stratification, and perturbation analysis.

### Strengths
- The paper is easy to follow.
- The idea of learning discrete spatially-aware cell tokens for SRT is conceptually interesting.

### Weaknesses
- The evaluation of SQUINT is weak, which can be summarized in several aspects: 
  - For Task A (2D imputation), the author includes the Wasserstein Flow Matching (WFM) as a baseline for comparison. However,  it is apparent that WFM is not a suitable baseline as there are too many "NA" in the table. I suggest that the authors try simpler baselines such as Gaussian-process-based spatial interpolation methods, which have been demonstrated to perform effectively for spatial imputation in [2–4].
  - For Task B-D, quantitative comparison with baseline methods seems to be completely missing.
  - The ablation studies for the critical component in SQUINT are too simple.SQUINTw/o C model doesn't have much information for algorithmic insight.
- The literature review on relevant work is notably incomplete. Discrete representation learning on transcriptomic data is not new [1] and the comparison with many existing works is missing. A method [2] for spatial imputation and perturbation is also ignored, which makes the claim "To the best of our knowledge, SQUINT is the first model to address this task on SRT data." overstated.
- Claims about batch effect correction are made but not supported by any experimental results.

[1] Li, Y. MetaQ: fast, scalable and accurate metacell inference via single-cell quantization.

[2] Hao, M. *et al.* GeST: Towards Building A Generative Pretrained Transformer for Learning Cellular Spatial Context.

[3] Shang, L. & Zhou, X. Spatially aware dimension reduction for spatial transcriptomics. *Nat Commun* **13**, 7203 (2022).

[4] Tian, T., Zhang, J., Lin, X., Wei, Z. & Hakonarson, H. Dependency-aware deep generative models for multitasking analysis of spatial omics data. *Nat Methods* **21**, 1501–1513 (2024).

### Questions
See weakness.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents SQUINT, a spatially-aware discrete representation learning framework for spatial transcriptomics (SRT).
 Unlike continuous embedding approaches such as graph autoencoders or transformers, SQUINT encodes tissues into a finite vocabulary of interpretable discrete codes using a graph neural network encoder combined with vector quantization conditioned on relative spatial distances.
 Each cell is assigned to a shared codebook entry, allowing tissues to be represented as sequences of discrete tokens.
 The learned codes enable gene expression imputation at arbitrary spatial locations and capture recurrent microenvironmental motifs across samples.
 Empirical results on multiple datasets, including 3D skin sections and kidney tumor tissues, show that SQUINT achieves competitive reconstruction accuracy while providing interpretable symbolic representations applicable to downstream tasks such as tumor stratification and perturbation analysis.

### Strengths
**Discretization resolution and codebook stability not analyzed:**
 The impact of codebook size, token granularity, and quantization error on tissue representation quality is not systematically evaluated.


**Clear architecture and presentation:**
 The model design—encoder, vector quantization bottleneck, decoder—is well illustrated and easy to follow. Figures and explanations are well aligned with the mathematical description.

**Biological insight:**
 The discrete tokens reveal spatially recurrent structures and microenvironmental organization patterns that correspond to known biological phenomena (e.g., immune infiltration, tumor aggressiveness).

### Weaknesses
**Limited novelty relative to existing VQ approaches:**

 The overall framework follows the standard vector quantization autoencoder pipeline (VQ-VAE), with the main innovation being its application to SRT. The methodological contribution may be incremental compared to recent graph-based SRT embedding models.

 **Limited robustness and spatial consistency:**

While SQUINT performs well in 2D spatial imputation, its robustness across 3D tissue sections is limited.
 As acknowledged by the authors, imputation quality drops in the middle of the Z-stack, likely due to misalignment and uneven section spacing.
 This indicates that the current formulation lacks explicit mechanisms to enforce spatial continuity or cross-section alignment, effectively treating 3D data as independent 2D slices.
 Consequently, the model’s generalization and robustness to spatial distortions or imperfect registration remain limited.
 Future work should consider alignment-aware architectures or continuous spatial encoders to achieve true volumetric reconstruction.

**Scalability and computational cost not discussed:**

 Given the large number of cells in modern SRT datasets, it is unclear whether the model scales efficiently to millions of spatial spots or whole-organ datasets.

**Limited cross-platform validation:**

Although SQUINT claims general applicability to spatial transcriptomics data, the experiments are restricted to a small set of platforms (Visium, Xenium, CosMx).
 No evaluation is provided on high-resolution single-cell or subcellular assays such as MERFISH, seqFISH, or Stereo-seq, which differ substantially in spatial density, noise structure, and coordinate geometry.
 Without such validation, it remains unclear whether SQUINT can generalize across measurement technologies or maintain consistent token semantics under varying data modalities.
 This limitation raises questions about the model’s robustness and practical usability for diverse spatial omics datasets.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors present a discrete representation learning framework for ST data that combines GNNs with vector quantization to encode cells as discrete tokens from a learned codebook. The authors claim these tokens enable gene expression imputation at arbitrary spatial locations and demonstrate applications in imputation, tumor stratification, and perturbation analysis. The manuscript in its current state has important conceptual, experimental, and presentation deficiencies that prevent me from recommending acceptance. The core concerns are: weak baseline comparisons, insufficient ablations and controls for important design choices, potential data/experimental leakage, over-claiming interpretability/clinical relevance from tiny cohorts, and missing reproducibility details.

### Strengths
- The paper demonstrates the utility of learned discrete tokens across multiple biologically relevant applications (2D/3D imputation, tumor stratification, perturbation analysis), showing that the representations capture meaningful information beyond simple reconstruction. 
- Table 1 shows SQUINT achieves substantially better reconstruction metrics than the baseline across multiple datasets and species.

### Weaknesses
- Authors claim the method enable generation of expression profiles. However the evaluation is limited to reconstruction metrics (MSE, SSIM, Pearson correlation). The authors should present examples of samples from the generative model, assessments of sample diversity, evaluations of biological validity beyond held-out reconstruction and comparisons to baseline generative models.
- The authors fail to cite or compare against the relevant prior work. The paper claims to be "the first spatially-aware discrete tokenizer tailored to SRT". These claims are demonstrably false e.g., Yarlagadda et. al. [1] which presents a similar approach using VQ-VAE for ST data. This is oversight undermines claims of novelty. 
- The only generative baseline is WFM, which requires cell-type annotations & cannot perform location-specific imputation. The authors exclude other GNN-based ST methods like SpaGCN, STAGATE, GraphST, recent transformer approaches like CellPLM and scGPT-spatial, NMF-/ topic-modeling based approaches, or scVI-like VAEs adapted for spatial data.
- How important are the different components like masking and FiLM conditioning? An ablation study assessing their impact is lacking. The "custom masking strategy" (Eq. 4) is just replacing masked cells with a learnable vector plus noise.
- The generative model formulation (Section 4.1) is unnecessarily complex for what amounts to a conditional VQ-VAE
- Why is GraphSAGE with mean pooling used here? The single-layer GNN seems simplistic. Fig. 8 shows GIN performs comparably—was this actually optimized or just chosen arbitrarily?
- The MSSIM + NB loss combination (Eq. 5) lacks justification. Why MSSIM for sparse gene expression data? The λ weights appear hand-tuned without principled selection.
- The imputation experiments define patches within one section as imputation sites while training on other cells from that section as well as remaining sections. That phrasing suggests the training set may include spatially adjacent cells to the held-out patches, risking trivial interpolation rather than true out-of-region generalization. The authors must precisely define the train/test partitioning (are held-out patches contiguous? how far from training cells?), and ideally evaluate far-away masking splits (e.g., hold out entire anatomical regions or whole sections) to test true generalization. Without this, the strong numerical gains could reflect spatial proximity rather than real generative power.

[1]. Yarlagadda, D. V. K., Massagué, J., & Leslie, C. (2023). Discrete representation learning for modeling imaging-based spatial transcriptomics data. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 3846-3855).

### Questions
- How is the codebook initialized? The authors state using a codebook size of 5000 with multiple heads in some experiments (and claim it “avoids collapse”), but in the 3D skin experiment they use a codebook size of 50 (and 200 latent dims) — no explanation for the dramatic difference or guidance on choosing K. There is no systematic ablation of codebook size, head count, code utilization statistics, or metrics of codebook collapse.
- What is the masking schedule (0.2 to 0.6 annealing)?
- How stable is training with the straight-through estimator? What is the variance across random seeds?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SQUINT, a novel framework designed to learn discrete representations from spatially-resolved transcriptomics (SRT) data. The primary goal is to capture tissue-level organization and microenvironmental details within a finite, interpretable vocabulary of discrete codes. The method combines a Graph Neural Network (GNN) encoder with vector quantization (VQ), conditioned on relative spatial distances and trained using a masking strategy. Cells are thus represented by token assignments, enabling whole tissues to be modeled as sequences. The authors validate their approach on several downstream tasks, including 2D and 3D gene expression imputation, tumour stratification, and in silico perturbation analysis, claiming SQUINT provides interpretable codes and outperforms existing generative methods.

### Strengths
- The paper tackles the important and challenging problem of learning compact, interpretable, and reusable representations for complex SRT data, which is a significant bottleneck in the field.
- The proposed method of combining a GNN with vector quantization to create a "tissue tokenizer" is a novel and interesting approach.
- The authors demonstrate the utility of the learned discrete codes across a diverse range of downstream applications (imputation, stratification, perturbation), showing the potential versatility of the framework.

### Weaknesses
1. **Overstated Novelty and Missing Baselines:** The authors claim that SQUINT is "the first model to address" the task of imputing gene expression at specific spatial locations. This is factually incorrect. For example, scGPT-spatial already supports this task and explicitly includes it as a pre-training objective. A direct performance comparison with this highly relevant baseline is missing, which makes it difficult to assess the actual performance and contribution of SQUINT for spatial imputation.
2. **Unaddressed Scalability Concerns:** The paper does not sufficiently address the scalability of the GNN-based approach. The largest dataset mentioned (Sec 5.2) contains 280K cells, which is modest by modern SRT standards. It is unclear how SQUINT would perform in terms of memory and runtime on datasets with millions of cells. The authors do not report detailed computational overhead nor do they discuss potential GNN-specific issues like over-smoothing, which can be a significant problem in large, densely-connected graphs.
3. **Insufficient Justification for Architectural Choices:** The justification for key architectural choices is lacking. The paper asserts that a two-layer MLP is sufficient to handle data sparsity, but this claim is not substantiated with ablation studies or other evidence. It is unclear how this conclusion was reached or why this specific design is optimal.
4. **Lack of Detailed Dataset Information:** The paper omits a clear, consolidated summary of the datasets. For reproducibility, it is essential to provide specific details for *each* dataset, including the precise cell counts, the sequencing technology used (e.g., Xenium, Visium), and the number of genes measured.

### Questions
1. Could the authors please clarify the claim of novelty regarding spatial imputation, given that models like scGPT-spatial already perform this task? More importantly, could you provide a direct performance comparison against scGPT-spatial for the 2D expression imputation task?
2. Regarding scalability:
   - What are the specific runtime and peak memory usage for training SQUINT on the 280K-cell dataset (Task B)? How do you anticipate these resources scaling to datasets with >1 million cells?
   - Have you investigated the potential for GNN over-smoothing in your framework, particularly as the graph size and neighborhood depth increase?
3. Could the authors provide an ablation study or further justification for the claim that a two-layer MLP is sufficient for handling data sparsity?
4. Could you please add a table summarizing the key statistics for *each* dataset used in the experiments (e.g., specific cell count, sequencing method, gene count, data source)?

### Soundness
2

### Presentation
3

### Contribution
2
