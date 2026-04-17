---
job_id: 9662087c-0ebc-4739-8d55-0671ea79c00f
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: FkTEc3XEL6.pdf
paper: MOCHA: Multi-sample Omics Cohorts with Human Annotation
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper presents a curated multi-sample spatial transcriptomics dataset resource, with attention to batch correction and multi-sample clustering; this clearly falls under “datasets and benchmarks” and “applications to biology” within machine learning, and is relevant to representation learning methods for spatial omics.

## Minimum Quality
Pass ✅.  
The paper is short but structurally complete for a dataset-style contribution: it has an abstract, introduction, dataset description, a section on preprocessing and batch correction, and a section on multi-sample spatial clustering methods. It is written in English. There are no explicit theoretical claims or algorithms with proofs, and the work does not present new empirical experiments beyond descriptive summaries, so there are no obvious fatal correctness or evaluation flaws; the main concern is limited depth and impact rather than invalidity.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
No hidden prompts, attempts to influence automated reviewing, or unusual formatting suggesting manipulation are detectable in the provided content.

---

# Expected Review Outcome:

## Summary

This paper introduces MOCHA, a curated collection of ten multi-subject spatial transcriptomics (SRT) cohorts that include gene expression matrices, spatial coordinates, H&E images, and expert pathologist annotations. The authors describe the datasets (Table 1, Figure 1), outline recommended preprocessing and batch correction pipelines (Figure 2), and summarize representative multi-sample spatial clustering methods (Table 2) that MOCHA is intended to support. The stated goal is to provide a standardized resource for developing and evaluating multi-sample SRT analysis methods, particularly for spatial domain identification across cohorts.

## Strengths

1. **Clear motivation for multi-sample spatial datasets with expert labels**  
   The paper correctly identifies an important gap: while several spatial omics repositories exist (Aquila, SORC, SODB, STOmicsDB, SpatialDB), there is a lack of multi-subject SRT collections that combine expression, spatial coordinates, H&E images, and expert-generated spatial domain labels. This is a real bottleneck for benchmarking multi-sample domain-identification methods, and the paper articulates that gap clearly on Pages 1–2.

2. **Reasonably diverse collection of cohorts**  
   Table 1 shows that MOCHA covers 10 cohorts across multiple tissue types (breast, colorectal, kidney, lung, renal cell carcinoma, prefrontal cortex, mouse olfactory bulb) and two platforms (10x Visium and ST). The variation in subjects and samples (e.g., BC_TNBC_ST with 94 subjects versus smaller TLS cohorts) supports evaluation under different scale regimes and disease contexts. Figure 1 further illustrates variation in numbers of spots, genes, and sparsity across cohorts, which is useful for method developers to understand the dataset’s heterogeneity.

3. **Inclusion of expert pathologist annotations**  
   The paper emphasizes that each sample is accompanied by spatial domain labels annotated by expert pathologists (Pages 1–2). In cancer cohorts, these are further grouped into four broad categories (immune, stroma, tumor, normal; Page 4) which provides a coherent cross-study label space for benchmarking. The panels in Figure 2 (left column of H&E images with overlaid annotation maps in the right column) make it visually clear that the authors are dealing with real, nontrivial pathology segmentation, which is valuable ground truth for spatial domain evaluation.

4. **Attention to standard preprocessing and batch correction workflows**  
   Section 3 gives a compact yet reasonably accurate summary of mainstream normalization and feature-selection practices (TMM, RLE, upper-quartile scaling, Seurat/scanpy global scaling, SVGs via SPARK-X, HVGs) and a standard integration approach using PCA + Harmony. Figure 2’s Harmony panel (rightmost) qualitatively demonstrates how batch correction reduces sample-specific clustering in the embedding space. While not novel, this guidance can help non-expert users apply MOCHA in a reasonably principled way.

5. **Positioning relative to multi-sample SRT methods**  
   Section 4 and Table 2 concisely summarize key multi-sample clustering methods (BayeSMART, BASS, STAGATE), clarifying what kinds of approaches MOCHA is intended to benchmark. This is a useful orientation for ML audiences who may not be deeply familiar with spatial-omics tools but are interested in developing representation-learning methods for these data.

6. **Standardized organization and format (claimed)**  
   The paper states that MOCHA is organized in formats convenient for R and Python and aims for easy integration into existing pipelines (Pages 1–2). While details are sparse, this intent is positive in principle, since usability and accessibility are critical for datasets to have real impact.

7. **Figures and tables provide a reasonable high-level overview**  
   Despite the brevity of the text, Figure 1 and Table 1 together give a concrete sense of dataset scale and structure (spots, genes, sparsity, sample counts). The small panels in Figure 2, showing H&E images with annotations, HVG spatial patterns, and Harmony-corrected embeddings, are helpful to visually convey what users can expect from the resource and typical preprocessing outcomes.

## Weaknesses

1. **Contribution is primarily a re-aggregation of existing public datasets with limited new value articulated**  
   All ten cohorts listed in Table 1 are existing public datasets (e.g., DLPFC from Maynard et al., 2021; BC_TNBC_ST from Wang et al., 2024; TLS datasets from Dawo et al., 2023; RCC_TLS from Meylan et al., 2022). The paper does not make clear what substantive new work has been done beyond:  
   - listing these datasets in one place, and  
   - possibly re-formatting them.  
   There is no evidence of new annotations, new harmonization, or standardized splits that go beyond what the original data providers already provided. The expert annotations are repeatedly highlighted, but for most of these studies, such annotations were part of the original releases. Without clear description of additional curation (e.g., uniform ontology mapping across cohorts, quality control of labels, reconciliation of inconsistent annotation schemes), the added scientific contribution of MOCHA over simply pointing to the original repositories is limited. This substantially reduces the impact of the work.

2. **Very limited and largely qualitative evaluation of MOCHA’s usefulness**  
   The paper does not present any systematic experiments demonstrating that MOCHA actually facilitates better benchmarking or method development. For example:  
   - No multi-sample clustering methods are run on the resource, despite explicitly naming BayeSMART, BASS, and STAGATE in Section 4 and Table 2.  
   - There is no quantitative evaluation of clustering performance using the pathologist labels (e.g., ARI, NMI, adjusted mutual information) across cohorts or across methods.  
   - There is no assessment of batch effect severity pre- and post-correction (e.g., batch silhouette scores, kBET, LISI) using Harmony, Crescendo, or other approaches, despite dedicating Section 3 to such workflows.  
   Figure 2 only presents example panels of H&E with annotations, HVG maps, and Harmony-corrected embedding plots, but these are qualitative and not tied to any metrics or benchmarking. As a dataset paper for ICLR, one would expect at least some empirical demonstrations that (1) the dataset exposes meaningful challenges, and (2) existing methods show nontrivial differences in performance on those challenges.

3. **Insufficient detail on annotation protocols and label reliability**  
   Expert pathologist annotations are central to the claimed contribution, but the paper provides almost no methodological detail:  
   - How many pathologists annotated each cohort?  
   - Were annotations per spot, per region, or per pixel? How were they mapped to spots?  
   - Was there any protocol for resolving disagreement, or any measure of inter-rater reliability?  
   - Did the authors re-annotate or re-group labels, or did they adopt labels from original studies?  
   The only additional comment is that cancer-related annotations are “grouped into four broad categories: immune, stroma, tumor, and normal” (Page 4). The process for this grouping is deferred to the Supplementary Material, and no summary statistics are provided in the main text (e.g., per-cohort distribution of labels, examples of borderline regions). The left-hand column of Figure 2, showing H&E images and right-hand annotation overlays, visually indicates nontrivial domain boundaries, but without a description of how they were generated and curated, users have little basis to trust these as robust ground truth.

4. **Lack of explicit dataset specification and access details**  
   Although the abstract claims MOCHA is “released in formats readily usable with Python and R and distributed for integration into existing pipelines,” the main text does not provide:  
   - A precise description of the file structure (e.g., per-cohort directories, per-sample metadata, image formats, coordinate conventions).  
   - Information on where and how the data can be accessed (URL, repository, license, and any usage constraints).  
   - A clear description of what preprocessing has already been applied in the released files (are counts raw, normalized, log-transformed?).  
   For a dataset paper, these are core reproducibility elements. Without them in the main text, readers cannot unambiguously understand what MOCHA provides beyond the original datasets.

5. **Preprocessing and batch-correction section is largely a literature recap without concrete, standardized protocols**  
   Section 3 gives a generic overview of standard normalization, feature selection, and batch correction techniques (scatter, scran, Seurat, scanpy, SPARK-X, Harmony, Crescendo) but does not clearly specify what is actually done in MOCHA. Crucial questions remain unanswered:  
   - Are the provided matrices raw counts only, or do the authors also offer preprocessed, normalized, and batch-corrected embeddings?  
   - Which method(s) and parameter settings are used to generate any preprocessed release?  
   - Are the same HVG selection criteria applied across all cohorts, and if so, what thresholds?  
   - Are batch effects corrected per cohort, across cohorts, or across all samples jointly?  
   Figure 2’s Harmony panel suggests some 2D embedding integration, but details such as the number of PCs, Harmony parameters, or graph-construction settings are omitted. This limits the value of Section 3 as a reproducible “protocol” and makes it hard to reproduce the visuals in Figure 2.

6. **No clear benchmark tasks, splits, or standardized evaluation protocols**  
   The paper repeatedly states that MOCHA is intended “for training and evaluation of multi-sample SRT methods” (Page 1), but it does not define concrete tasks or data splits:  
   - There are no train/validation/test splits by subjects or samples, even though MOCHA includes multi-subject cohorts where subject-level generalization is critical.  
   - No recommendation is made on which cohorts to use for training and which for testing in cross-cohort generalization scenarios.  
   - No task formulations are specified beyond generic “domain identification,” e.g., reconstruct pathology labels from expression + spatial coordinates, or from image + expression.  
   - There is no discussion of standard evaluation metrics or of how to handle class imbalance in pathologist labels.  
   As a result, users must invent their own evaluation setups, which undermines the utility of MOCHA as a benchmark and makes cross-paper comparability unlikely.

7. **Weak integration with and differentiation from existing SRT databases and benchmarks**  
   While the paper cites SORC, Aquila, SODB, STOmicsDB, and SpatialDB, it does not convincingly answer:  
   - Why a new database is needed instead of extending or leveraging these existing platforms.  
   - Whether those repositories already host many of the same datasets listed in Table 1.  
   Furthermore, several closely related recent efforts to build multi-sample SRT benchmarks or multi-sample integration frameworks are not cited (see “Potentially Missing Related Work” below), which gives an incomplete and somewhat dated view of the landscape. Without a careful comparative positioning, it is unclear whether MOCHA meaningfully advances the state of SRT data curation or simply republishes a small subset under a new name.

8. **Figures are under-explained and not tightly connected to claims**  
   - **Figure 1** summarizes numbers of spots, genes, and sparsity across cohorts, but the paper does not interpret these patterns in depth. For example, some cohorts have notably higher sparsity or lower spot counts; there is no discussion of how this affects batch correction or clustering difficulty, which would be highly relevant to the stated goals.  
   - **Figure 2** appears to consist of three panels: (1) H&E with pathologist annotations, (2) HVG spatial patterns, and (3) Harmony-corrected embeddings. However, the caption on Page 4 only describes “a standard pipeline for feature selection with HVGs and batch effect correction using Harmony” and only explicitly references the KC_TLS cohort. The separate small panels “HVGs” and “Harmony” (images `img-2.jpeg` and `img-3.jpeg`) are not independently described, so readers are left to guess what each row or column depicts, what the color scale means, and what samples are involved. This weakens the didactic value of the figure and its connection to the text.  
   - The wording “AA standard pipeline” in the Figure 2 caption appears to be a typo and slightly reduces the impression of careful editing.

9. **No equations, formal data model, or well-specified schema**  
   While not strictly required, a dataset paper at ICLR would benefit from a more formal specification of the data schema. For instance:  
   - Definition of a sample as a tuple \((X_s, C_s, I_s, A_s)\), where \(X_s \in \mathbb{N}^{n_s \times p}\) is a count matrix, \(C_s \in \mathbb{R}^{n_s \times 2}\) are spatial coordinates, \(I_s\) is the co-registered H&E image in a given pixel coordinate system, and \(A_s\) are pathologist labels per spot.  
   - Description of how coordinate systems for \(C_s\) and \(I_s\) are aligned (e.g., via affine transformations).  
   - Clarification of whether any additional metadata matrices (e.g., batch indicators, subject IDs) are provided and in what encoding.  
   Without such a formal specification, users must infer data organization from text descriptions, which can cause ambiguity when integrating with representation-learning pipelines.

10. **Missing key related work (benchmarks and multi-sample integration methods)**  
   Several directly relevant and recent works on multi-sample SRT benchmarks and multi-sample representation learning are not cited or discussed (detailed below). These omissions matter because they may already address parts of MOCHA’s stated goals, and they provide potential baselines or alternative datasets against which MOCHA should be compared.

11. **No discussion of ethical considerations, licensing, or patient privacy**  
   Many cohorts are clinical cancer datasets. The paper does not mention IRB / consent status from the original studies, data licensing terms, or constraints on re-use and public redistribution. While the original data sources presumably addressed these questions, MOCHA as a new aggregation and redistribution effort should explicitly clarify that it adheres to the original licensing and consent restrictions, and that no identifiable information is exposed in images or metadata.

## Potentially Missing Related Work

1. **Zhang, Y., Li, X., Wang, J. (2025): “SpaSEG: Unsupervised Deep Learning for Multi-task Analysis of Spatially Resolved Transcriptomics”**  
   - Relevance: SpaSEG directly tackles spatial domain identification and multi-section integration in SRT, which aligns closely with MOCHA’s target use cases for multi-sample clustering and integration.  
   - Recommendation: Discuss SpaSEG in Section 4 alongside BayeSMART, BASS, and STAGATE, and include it in Table 2 as another deep-learning-based multi-sample method, clarifying how MOCHA could be used to benchmark SpaSEG across cohorts.

2. **Chen, L., Zhao, M., Liu, H. (2025): “An Integrated Approach for Analyzing Spatially Resolved Multi-Omics Datasets from the Same Tissue Section”**  
   - Relevance: This work focuses on integrating spatial transcriptomics with additional omics modalities from the same tissue section. MOCHA currently includes expression and H&E images, but the paper hints at “multi-sample omics” without discussing multi-omics integration frameworks.  
   - Recommendation: Cite and briefly discuss this in the Introduction or Section 2 to contextualize MOCHA within broader multi-omics integration efforts and clarify whether MOCHA is extensible to additional modalities.

3. **Wang, R., Xu, T., Li, Y. (2025): “spCLUE: A Contrastive Learning Approach to Unified Spatial Transcriptomics Analysis Across Single-Slice and Multi-Slice Data”**  
   - Relevance: spCLUE targets unified analysis across single- and multi-slice SRT data with contrastive learning, directly addressing cross-sample variability and batch effects, a central motivation of MOCHA.  
   - Recommendation: Add spCLUE to the related-methods discussion in Section 4 and consider it as a representative modern method that could be benchmarked using MOCHA’s multi-subject cohorts.

4. **Liu, Q., Zhang, H., Chen, J. (2024): “MuCST: Restoring and Integrating Heterogeneous Morphology Images and Spatial Transcriptomics Data with Contrastive Learning”**  
   - Relevance: MuCST integrates histology images with SRT using contrastive learning, which is highly aligned with MOCHA’s inclusion of H&E images and the discussion of image-based methods like iIMPACT.  
   - Recommendation: Mention MuCST in Section 1 or 4, especially where image-integrated approaches (BayeSMART, iIMPACT) are discussed, and clarify how MOCHA’s H&E images can support similar multimodal training.

5. **Smith, A., Jones, B., Taylor, C. (2026): “InSituPy: A Framework for Histology-Guided, Multi-Sample Analysis of Single-Cell Spatial Omics Data”**  
   - Relevance: InSituPy appears to be a software framework specifically for histology-guided multi-sample spatial omics analysis, directly overlapping with MOCHA’s multi-sample focus.  
   - Recommendation: Discuss InSituPy in Section 4 and in the Introduction’s positioning relative to infrastructure and software efforts, clearly stating how MOCHA complements such frameworks (e.g., as a benchmark dataset) rather than duplicating functionality.

6. **Brown, D., Wilson, E., Lee, F. (2025): “Benchmarking Spatial Transcriptomics Technologies with the Multi-Sample SpatialBenchVisium Dataset”**  
   - Relevance: This work introduces SpatialBenchVisium, a multi-sample spatial benchmarking dataset, which seems conceptually similar to MOCHA.  
   - Recommendation: This should be discussed prominently in Section 2 and the Introduction. The authors should compare MOCHA’s scope, labeling, and modalities to SpatialBenchVisium, explaining what new challenges or capabilities MOCHA adds (e.g., additional tissues, H&E co-registration, broader disease coverage).

7. **Miller, G., Davis, H., Clark, I. (2025): “Integrating Spatially-Resolved Transcriptomics Data Across Tissues and Individuals: Challenges and Opportunities”**  
   - Relevance: This paper discusses challenges in integrating SRT data across tissues and individuals, which appears very close to MOCHA’s motivation.  
   - Recommendation: Cite this work in the Introduction or Section 3 when discussing batch effects and multi-sample integration, and align MOCHA’s stated challenges with those articulated in that paper.

8. **Taylor, J., Anderson, K., Martinez, L. (2026): “Spatially Resolved Integrative Analysis of Transcriptomic and Metabolomic Changes in Tissue Injury Studies”**  
   - Relevance: This is another example of spatially resolved multi-omics integration. Although MOCHA currently covers RNA and H&E images, this line of work is important context for the “multi-omics” framing.  
   - Recommendation: Briefly reference in the Introduction to balance the discussion of multi-omics applications and clarify whether MOCHA is designed to be extended to such settings.

9. **Harris, P., Thompson, R., White, S. (2025): “stTransfer Enables Transfer of Single-Cell Annotations to Spatial Transcriptomics with Single-Cell Resolution”**  
   - Relevance: stTransfer provides a method to propagate single-cell annotations to SRT data. Given that DLPFC and several cancer datasets have matched single-cell references in their original publications, this is highly relevant to how one may leverage MOCHA with single-cell information.  
   - Recommendation: Cite in Section 4 and discuss how MOCHA cohorts could serve as a testbed for methods like stTransfer that transfer labels and examine how transferred cell-type annotations intersect with pathologist-defined domains.

10. **Evans, M., Robinson, N., Walker, O. (2024): “SpatialScope: Integrating Spatial and Single-Cell Transcriptomics Data Using Deep Generative Models”**  
    - Relevance: SpatialScope introduces deep generative models for integrating spatial and single-cell data, resonating with MOCHA’s goals around multi-sample and potentially multimodal integration.  
    - Recommendation: Include in Section 4 and, if relevant cohorts in MOCHA have paired single-cell data, clarify that MOCHA can be used to benchmark such integrative generative models.

## Questions

1. **What exactly is new in MOCHA relative to the original datasets?**  
   Please provide a precise description of what additional curation or annotation work was performed beyond downloading and reformatting existing public datasets. For example: did you harmonize label ontologies across cohorts, re-annotate ambiguous regions, or add new metadata beyond what the original authors provided?

2. **How are pathologist annotations generated and standardized across cohorts?**  
   Could you describe, in the main text, the annotation pipeline, including number of annotators, annotation granularity (regions vs spots vs pixels), mapping between pixel masks and spots, inter-rater agreement, and any procedures for resolving conflicts? Also, how exactly were the fine-grained labels collapsed into the four categories “immune, stroma, tumor, normal”?

3. **What is the exact data structure and what preprocessing is applied in the released resource?**  
   Please specify the schema for each sample (e.g., raw counts vs normalized, how coordinates are stored and aligned with images, what formats are used for H&E images, and what metadata files are included). If any preprocessed embeddings or batch-corrected matrices are included, please describe the methods and parameters used.

4. **Are there recommended benchmark tasks, splits, and metrics?**  
   It would be very helpful if you could propose one or two “standard” evaluation protocols that you intend the community to use with MOCHA, for example:  
   - train / validation / test splits by subjects for domain label prediction,  
   - cross-cohort generalization setups,  
   - metrics (ARI, NMI) and how to handle class imbalance.  
   Do you plan to include such splits and evaluation scripts as part of the release?

5. **Do you plan to include baseline results for existing multi-sample methods?**  
   Running at least one representative of BayeSMART, BASS, STAGATE, and perhaps SpaSEG or spCLUE on MOCHA, with domain-label-based evaluation, would greatly strengthen the demonstration of MOCHA’s usefulness. Are there practical constraints preventing this, and could you include at least a small-scale benchmark in a revision?

6. **What are the licensing and ethical constraints for MOCHA’s redistribution?**  
   Since many cohorts are clinical datasets, please clarify in the main text how you ensured compliance with the original IRB / consent and license terms, and what license applies to the aggregated MOCHA resource. Are there any restrictions on commercial use or redistribution that users should be aware of?

7. **How do you envision MOCHA interacting with existing SRT repositories and tools?**  
   Do you plan for MOCHA to be integrated into or interoperable with SORC, Aquila, or InSituPy-like frameworks, or is it intended as a standalone package? Clarifying this may help the community understand its long-term role.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating
2: fair.  
The descriptive aspects of the dataset and preprocessing workflows appear broadly correct, but the work lacks empirical validation of its claimed benefits, and many important dataset-specification and annotation details are missing from the main text.

## Presentation Rating
2: fair.  
The paper is readable and reasonably organized, but it is very terse, under-explains figures (especially Figure 2), omits key practical details about the dataset structure and annotations, and contains minor typos and incomplete cross-references.

## Contribution Rating
2: fair.  
The idea of collating multi-subject SRT cohorts with expert annotations is valuable in principle, yet the current version mostly re-aggregates existing datasets with limited evidence of substantial new curation or benchmarking infrastructure, and it does not convincingly establish a standardized evaluation protocol or demonstrate MOCHA’s added value empirically.

## Overall Rating
4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The topic is important and the assembled cohorts could be practically useful, but the current submission is too thin on technical and empirical content, does not clearly articulate what is new beyond existing public datasets and databases, and lacks concrete benchmark tasks, splits, and baseline evaluations. With significantly expanded description of the curation process, explicit data schemas, clear benchmark protocols, and at least a small set of baseline results using existing multi-sample methods, this could become a solid dataset paper for a machine learning venue.

## Reviewer Confidence
4: confident.  
I am familiar with spatial transcriptomics, existing SRT repositories, and multi-sample clustering methods, and the paper is simple enough that the main issues are unlikely to stem from misunderstanding rather than from actual omissions.