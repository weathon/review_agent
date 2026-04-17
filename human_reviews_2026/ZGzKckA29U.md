# STAGE: A Foundation Model for Spatial Transcriptomics Analysis via Graph Embeddings with Hierarchical Prototypes

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Spatial transcriptomics offers an unprecedented opportunity to elucidate the spatial organization of tissues by capturing gene expression profiles while preserving tissue architecture. This enables the identification of spatial niches and deepens our understanding of tissue function and disease-associated microenvironments. However, consistent identification of spatial domains across samples, tissues, and even technological platforms remains a formidable challenge, due to low-dimensional and heterogeneous gene panels across platforms, pronounced batch effects, and substantial biological variability between samples. To address these limitations, we propose STAGE, a generalizable foundation model for spatial transcriptomics via graph embeddings. At its core, STAGE introduces a hierarchical prototype mechanism to capture global semantic representations of spatial niches, alongside an efficient online expectation-maximization algorithm to enable scalable learning from large-scale heterogeneous data. Pretrained on a large dataset comprising 32 million cells from 18 tissue types, STAGE learns robust cell representations within their neighborhood graphs and supports niche inference for domain recognition. Comprehensive evaluations on multiple benchmark datasets demonstrate that STAGE substantially enhances domain consistency in cross-platform, cross-sample, and cross-tissue spatial domain identification tasks, outperforming existing state-of-the-art methods. Furthermore, STAGE supports critical downstream biological analyses, highlighting its strong potential as a powerful tool in biological research.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces a self-supervised framework for learning tissue zonation patterns from spatial transcriptomics data. Drawing inspiration from the classical SwAV framework [2020], the authors propose a clustering-based approach in which a prototype network is trained to align multiple augmented views of the same data point to a shared set of prototypes. Unlike SwAV, however, the presented method extends this principle to the spatial-omics domain and incorporates a hierarchical prototype structure, allowing the model to capture both global tissue organization and finer-grained local zones.

### Strengths
The hierarchical prototype assignment is arguably the most compelling aspect of the paper. However, similar extensions of the SwAV framework have already been explored in the computer vision domain \cite{guo_hcsc_2022, xu_hirl_2022}, and related ideas of multiscale niche definition have also been introduced in spatial transcriptomics, for example in MENDER \cite{yuan_mender_2024}. Therefore, while the implementation presented here is technically interesting, it cannot be regarded as a fundamentally novel contribution. Nonetheless, I would encourage the authors to further analyze and interpret the resulting hierarchies, as they may reveal biologically meaningful patterns or inspire new downstream applications.

Authors then benchmark the method against general clustering algorithms (louvain, leiden), cell reprsentation models (scGPT) and a previous generation of niche identification methods (STAGATE [2022], GraphST [2023], Novae [2024]). The focus of benchmarking is primarily on the geometrical properties of the idetnified regions (continuity as measured by FIDE, PAS and ASW scores) and slide integration (estimated using JSD). The paper tries to emphasize that the approach is able to integrate between the tissue slices of (i) one tissue, (ii) multiple tissues and (iii) across the experimental platforms (CosMX and Xenium). The latter is achieved through scGPT-inspired representation of gene expression as a sequence of gene tokens. Though the benchmarking of the approaches for niche identification is not trivial due to the absence of ground truth data in most cases, the paper would benefit from extending (A) the range of the metrics and (B) the suite of the methods tested.

### Weaknesses
Numerous existing frameworks already address domain or niche identification, ranging from graph neural networks to simpler yet effective static feature aggregation methods. Moreover, the proposed mechanism for tissue integration is not original to this work but directly borrowed from scGPT; in principle, scGPT-derived embeddings could be integrated into any existing pipeline, the same effect can also be reached with cell type based approach.

The paper’s effort to integrate datasets across tissues and assay platforms is commendable, particularly given the limited development of multitissue models—apart from frameworks such as NicheFormer \cite{schaar_nicheformer_2024}. Nevertheless, the presented results do not convincingly demonstrate the success of this integration. The UMAP visualizations (Figure 8) reveal little to no alignment between CosMX and Xenium datasets, casting doubt on whether joint training across these sources yields any tangible benefit. Moreover, in quantitative comparisons, STAGATE achieves substantially better integration (as reflected by lower JSD values) while employing a considerably simpler modeling approach. 

It should also be noted that, from a biological standpoint, cross-tissue integration is a nuanced objective. While certain structural elements recur across organs—such as tertiary lymphoid structures or components of the connective tissue—many aspects of niche organization remain inherently tissue-specific and should not be  overintegrated. I therefore recommend that the authors adjust their benchmarking procedure to account for this nuance. A possible step in this direction could involve comparing slices originating from the same tissue (but critically different donors) versus those from distinct tissues, showcasing that the tissue specific differences are well-preserved.

### Questions
Below I suggest modifications that can help improve the paper.

1. **Extend the benchmarking.**
   - Acknowledge that **Louvain**, **Leiden**, and **scGPT** are *not* niche-identification methods; because they ignore explicit cell neighborhoods, they naturally recover *cell identity* rather than *niche identity*. While **STAGATE (2022)**, **GraphST (2023)**, and **Novae (2024)** are genuine niche-discovery algorithms, they are now outdated. Please compare against more recent methods such as *MENDER*, *CellCharter*, *scNICHE*, and additional 2025-era approaches.  
   - Explore a range of hyperparameters for all baselines, *especially the neighborhood size*, as spatial-continuity metrics are directly influenced by this choice.  
   - Include diverse datasets in the benchmark (e.g., the **MERFISH brain atlas**). Note that cancer may be suboptimal for measuring continuity: connective tissues often form branching, fractal structures that will *naturally* show lower continuity.  
   - Demonstrate the **biological plausibility** of identified regions: Do they exhibit unique gene-expression signatures or cell-composition profiles compared to regions obtained by other methods?  
   - Use a brain atlas to show that the method recapitulates known ground-truth zonation, as is customary in this literature.  
   - Show that the model not only *integrates* tissues but also *discriminates* between them, and that it does so better than other methods. Comment on regions identified as similar across tissues and whether such cross-tissue integration is biologically sensible.  

2. **Improve the method description.**
   - Clarify the difference to the SwAV: clearly separate canonical *SwAV* in a **Background** section from your modifications. Highlight how your **prototype hierarchy** differs from existing methods and attribute credit appropriately.  
   - Provide a clear architectural diagram and a precise textual description of all modules.  
   - Describe the training procedure unambiguously. As understood, there are two stages using the classical SwAV objective and an extended objective, and there is also an EM component. Is the E-step part of gradient-descent training within each step, or is it decoupled? Please clarify.  
   - Explain graph augmentations in detail: how are different views generated?  
   - Explain the gene-expression encoder beyond the token-embedding procedure. Are any biological priors used? If so, specify where and how.  
   - The statement “K-means clustering on learned representations initializes the bottom level” is obscure. If this is tied to a two-stage training procedure, state it explicitly and describe the rationale.  
   - The manuscript focuses on niche identification, yet Figure 1 advertises trajectory inference, variable-gene identification, and pathway analysis. Either provide concrete examples of these use cases or remove them from the figure.  

3. **Comment on biological parameters.**
   - Justify the choice of neighborhood size (report both the number of cells and the physical scale in microns).  
   - Analyze how the number of prototypes affects segmentation quality and stability.  

4. **Minor errors and clarifications.**
   - “ASW ranges from 1 to 1” → **ASW ranges from 0 to 1.**  
   - “Nonetheless, these models still focus mainly on local neighborhoods and lack mechanisms to capture global …” → please acknowledge published methods that integrate local and global context (e.g., *MENDER*, *CellCharter*, and many GNN-based approaches) and attribute credit to those works.  
   - *Figure 2:* Panels A–B–B appear mislabeled relative to the captions; moreover, the current content offers limited intuition about the method. Consider revising for clarity.  
   - In the Supplementary Materials, include explicit examples of *cross-tissue* integration.  

---

### Outlook

Some general feedback that may help the authors to improve the paper (I do not expect these to be implemented for the revision):

1. The method builds on the **SwAV** framework, which, although influential in early self-supervised learning, has since been succeeded by more expressive approaches. Try to explore methods that perform better in computer vision.  
2. The field of **niche identification** is already highly saturated, with several established methods demonstrating strong performance. The work would benefit from identifying a more distinctive angle or reframing the problem to address an underexplored aspect of spatial biology. Exploring alternative use cases or novel problem formulations could better highlight the unique strengths of the proposed approach.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a foundation model, pretrained on 32 million cells from 18 tissue types, for identifying spatial domains in spatial transcriptomics data. To my understanding, this is essentially a community-detection problem, but the application context makes it particularly challenging. Specifically, batch effects and technological platform biases (including varying gene panels) can make it difficult to compare recorded gene expression values, which impacts transferability and applicability in inductive settings. The authors identify shortcomings of existing methods and propose a model, which they call STAGE. STAGE uses a "Gene Embedder" for learning representations in a joint embedding space, that is, across different technologies and batches, which, to my understanding, aims to remove systematic biases (at least to some extent). With an attention mechanism, STAGE then aims to learn "local cellular interactions and broader spatial context", which, to my understanding, is necessary because cells interact with each other at different scales, that is, locally with cells in their immediate neighbourhood, but also more globally with cells further away. Furthermore, STAGE involves a "hierarchical prototype head" for discovering hierarchical communities in the tissue samples; The authors use an EM algorithm to learn those hierarchical communities with a loss formulation based on optimal transport. In an empirical evaluation on three datasets, the authors benchmark STAGE's performance against a range of baselines, showing that STAGE outperforms the baselines. The considered scenarios are cross-platform, same-tissue same-platform, and paired normal-tumour tissues.

I found the paper generally well written and the author's motivation and goals clear. However, I found some sections harder to follow because the authors use jargon that I believe is rather unfamiliar to a machine learning audience.

### Strengths
- The authors consider an important research question in spatial transcriptomics analysis and propose a novel model that outperforms current methods on the task of spatial domain identification.
- The authors use statistical significance tests to verify that STAGE outperforms the baselines significantly (I believe this is only mentioned in the appendix, but it may be worthwhile to mention this in the main text).
- The authors have included an ablation study to investigate the effect of some of STAGE's parameters on its performance.

### Weaknesses
- I believe the work could be somewhat more self-contained. Specifically, the authors mention that they extend the SwAV framework but provide only a reference, which forces the reader to look up that paper. I believe that a short paragraph on how the SwAV framework works would have helped the reader to understand the work better.
- I found some parts of the text a bit difficult to access due to jargon, and believe the same may be true for a general ML audience.

### Questions
There are a couple of things that remained unclear to me, which may be because I am not an expert in spatial transcriptomics, so I may simply not recognise some terms or concepts. I hope the authors can help me clarify those points.

1. If I understood it correctly, your work is essentially about community detection in a "difficult scenario" (caused by different types of biases). Would it be correct to think about "niches" as something microscopic, "domains" as something mesoscopic, and "regions" as something macroscopic?
2. Could you briefly describe what the SwAV framework is and how it works? And what are the extensions you propose that go beyond SwAV?
3. Did I understand it correctly that STAGE proposes several different ways of clustering the data? That is, for a given dataset, does it propose several (hierarchical) partitions?
4. I am not sure I understand what exactly a "prototype" is. I have the vague feeling that a prototype is a community and that the hierarchical organisation of prototypes form a partition. Could you explain?
5. I don't quite understand why constructing a neighbourhood graph via Delaunay triangulation involves setting a radius (Appendix F.3.). To my understanding, a Delaunay triangulation should be fully defined by the spatial distribution of the points. For a radius graph, however, a radius is needed. Could you clarify?
6. As far as I am aware, deciphering cell-cell communication is still an active research area. So it seems to me that the model used for spatial neighbourhood construction (section 3.3) makes some simplifications. Specifically, the construction of two different neighbourhoods for each node: (i) a local neighbourhood with "nearby" cells, and (ii) a more distant neighbourhood, further away than the nearby local cells, but no more than $r_view$ (by the way, unless I have missed it, I believe that $r_\text{local}$ and $r_\text{view}$ are not defined). Is there a biological motivation behind this model, or does it simply turn out to work well? And how did you choose values for $r_\text{local}$ and $r_\text{view}$?
7. Connected to the previous question, I am wondering whether these two different neighbourhoods are what make it possible for STAGE to "capture both local cellular interactions and broader spatial context"?
8. You mention that your "formulation eliminates costly pairwise comparisons while preserving their discriminative power". Could you elaborate on what things are traditionally compared in a pairwise manner? And what part of the formulation is it that removes this need? And how do you still maintain the discriminative power despite not performing those comparisons that are necessary in other methods?
9. Could you explain what "spatially related subgraphs", which you mention in section 3.4.2, are? Does the fact that they are separated by $n_\text{view}$ edges make them related? And does "separated by $n_\text{view} edges" refer to the minimum of shortest paths between any pair of nodes from the two graphs? Or is it some sort of edit distance?
10. How did you determine that $\lambda = 0.1$ is a good setting? Should this setting be understood as a general recommendation, or did it just turn out to work well in the present case?
11. Since the results in Table 1 do not contain any standard deviations, I am wondering whether STAGE returns deterministic results (I wouldn't really expect that) or whether each experiment has only been repeated once? I am also wondering why the JSD and FIDE measures are not included in the table?
12. What is the significance of the dashed horizontal and vertical lines in Figures 3+4?
13. You mention that STAGE "[demonstrates] strong resistance to batch effects", and I assume that this conclusion is based on the results in Table 1, is that right? But then I am wondering how you verified that STAGE is indeed resistant to batch effect? I suppose there could be other reasons that explain STAGE's good performance. How can we pinpoint that it is truly due to its resistance against batch effects?


Minor points
- It is somewhat unusual to have an empty section in the appendix simply titled "APPENDIX".
- I believe something went wrong with the LaTeX command for typesetting STAGE's training algorithm in Appendix G.1.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes STAGE, a foundation model with enhanced generalizability, to address critical challenges in spatial transcriptomics (ST) that arise from technological and biological variability. These challenges include: (1) gene panel discrepancies across platforms , (2) the need for per-sample retraining , and (3) a lack of unified semantic representation.

Fundamentally, STAGE builds upon the SwAV framework but introduces a hierarchical prototype mechanism. This approach imposes explicit constraints within its Optimal Transport (OT)-enhanced online EM algorithm, ensuring that assignments follow the defined tree structure and maintain hierarchical consistency.

The effectiveness of STAGE is validated through a series of experiments, including: Ablation studies, Cross-platform consistency evaluations, Cross-batch robustness tests, Pathological condition comparisons to identify both shared and disease-specific spatial domains

### Strengths
* The paper has a clear motivation that appropriately targets the generalizability problem faced by various ST models, including existing foundation models.

### Weaknesses
* Lack of Novelty: The methodology is highly incremental, fundamentally building on SwAV by adding a hierarchical prototype structure and constraints to ensure assignments explicitly follow this structure.
* Misalignment of Motivation and Methodology: The paper is unconvincing as to how this specific hierarchical prototype structure directly solves the three key problems cited for achieving generalizability: (1) Gene panel discrepancies, (2) Per-sample retraining, and (3) Lack of unified semantic representation.
* Unverified Biological Relevance: The paper asserts that this hierarchical structure "reflects biological reality", but it does not perform a direct Ground Truth comparison (e.g., sub-annotation validation) to support this claim. The hierarchy depth ($L_p$) was experimentally selected as 3 because it yielded good performance on validation metrics, not based on biological precedent. Therefore, it is unclear whether the "hierarchical sub-structure" claimed by this methodology has true biological meaning or if it is simply a computational construct that best optimizes the final clustering consistency (JSD) and continuity (FIDE) metrics. See reference [1]
---
[1] Limitations of cell embedding metrics assessed using drifting islands. Nature Biotechnology. 2025.

### Questions
* How about the clustering measures (e.g., ARI, NMI)?

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
This study introduces STAGE, a foundation model designed to enhance the generalizability of spatial transcriptomics (ST) by mitigating technological and biological variability. The model addresses challenges such as inconsistencies in gene panels across different platforms, the need for model retraining on individual samples, and the lack of a unified semantic representation framework.

Specifically, STAGE incorporates a hierarchical prototype mechanism integrated with an Optimal Transport-based online EM algorithm. This design ensures structured and consistent assignments within a hierarchical tree, improving representation alignment across various modalities and datasets.

The authors conduct comprehensive experiments to demonstrate the robustness and versatility of STAGE with diverse tasks such as cross-platform and cross-batch assessments.

### Strengths
- The authors effectively adapt a self-supervised learning strategy from the computer vision domain to enhance representation learning in spatial transcriptomics data.

- The proposed hierarchical prototype mechanism is conceptually novel and presents an interesting extension to the SwAV framework.

### Weaknesses
- Although the paper claims to address challenges such as cross-platform gene panel inconsistencies and the lack of a unified semantic representation, it remains unclear how these issues are specifically handled within the proposed methodology.

- The hyperparameter sensitivity analysis is limited. While ablations on hierarchical depth (L) and batch size are provided, key parameters such as the slide-specific prototype selection threshold (θ), temperature (τ), OT regularization coefficient (ϵ), and global-loss weight (λ) are not explored.

- The relationship between zero-shot and supervised settings is ambiguous. In several cases, the zero-shot variant outperforms the fine-tuned model (e.g., lower PAS values are not consistently improved after fine-tuning), suggesting that fine-tuning may sometimes degrade spatial coherence. The paper does not provide sufficient analysis to explain this behavior or the conditions under which it occurs.

### Questions
- Please include benchmarking results on widely used datasets such as DLPFC and compare the performance against current state-of-the-art methods.

- Currently, the graph construction relies solely on spatial coordinates. Do the authors plan to incorporate histological image information (e.g., H&E or IF)?

### Soundness
2

### Presentation
3

### Contribution
2
