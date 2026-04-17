Now I have a good sense of comparison papers. Let me synthesize my review of the MOCHA paper.

Key observations:
- This is a resource/benchmark paper that curates 10 existing SRT datasets with expert annotations
- The paper describes the datasets, preprocessing, and multi-sample methods but does NOT include any experiments or empirical demonstrations
- The key differentiator is "expert pathologist annotations" but the paper lacks details on annotation harmonization
- No data access details (URL, DOI, repository)
- Sections 3-4 are mini-reviews of existing methods rather than concrete protocols

Comparing to calibration papers:
- LST-Bench (scores 3,3,3,1 - Reject): Had actual benchmarking experiments but was still rejected for limited novelty and lack of insights
- GlucoBench (scores 5,6,8 - Accept poster): Had curated datasets + standardized tasks + benchmark models with baseline experiments
- COMET (scores 5,5,5,8 - Reject): Had benchmarking framework but was criticized for dataset selection and lack of depth
- Scenario-Wise Rec (scores 3,3,3,6 - Reject): Had experiments but still rejected for limited novelty
- CUS3D (scores 5,5,5,6 - Withdrawn/Reject): Had actual dataset collection with experiments but still rejected for annotation issues and unclear contribution

The MOCHA paper is weaker than most of these because it has NO experimental results at all — just a catalog of datasets and two mini-review sections. This is substantially below the bar even for resource papers that were rejected.</think>Now I have sufficient calibration context. Let me write the final review.

## Summary
MOCHA presents a curated collection of 10 publicly available spatially resolved transcriptomics (SRT) cohorts, each paired with expert pathologist-derived spatial domain annotations and co-registered H&E images. The paper also reviews standard preprocessing pipelines and multi-sample spatial clustering methods, positioning the resource for developing and evaluating multi-sample SRT integration methods.

## Strengths
- **Addresses a genuine gap.** Multi-subject SRT datasets with expert spatial domain annotations suitable for benchmarking are indeed scarce. Existing SRT databases (SODB, SpatialDB, SORC, Aquila) primarily aggregate raw data without harmonized expert annotations for cross-sample evaluation. This is a real need in the community.
- **Diverse cohort coverage.** The 10 cohorts span multiple cancer types, brain tissue, and olfactory bulb across two technology platforms (10x Visium and ST), with subject counts ranging from 1 to 94, providing breadth for method evaluation.
- **The concept of harmonized annotations across cohorts has inherent value.** If properly executed, grouping heterogeneous pathologist labels into a consistent taxonomy (e.g., immune, stroma, tumor, normal) would enable cross-cohort evaluation that is currently not possible with existing databases.

## Weaknesses

### Major:
1. **No empirical demonstration or benchmarking experiments.** Despite claiming to provide a resource "for developing and evaluating multi-sample SRT methods," the paper runs zero methods on the curated data. There are no clustering results, no ARI/NMI metrics, no visualizations of domain segmentation, and no before/after batch correction analysis. For a resource/benchmark paper, this is a fundamental gap — even papers that were ultimately *rejected* at similar venues (LST-Bench, Scenario-Wise Rec, COMET) all included actual experimental results. A benchmark paper without benchmarks has an unsupported core claim.

2. **Insufficient specification of what MOCHA actually provides.** The paper claims "standardized data organization" and "efficient storage formats for large-scale processing" but never describes: the data schema, file formats (AnnData? HDF5? loom?), folder structure, metadata model, or how raw data was standardized (gene ID harmonization, coordinate alignment with images, QC thresholds). Without this, the "resource" is indistinguishable from a list of links to existing GEO/Zenodo entries, which existing databases (SODB, Aquila, SORC) already provide.

3. **Annotation harmonization is underspecified.** The paper's primary differentiator is "expert pathologist annotations," yet it provides almost no detail on: whether annotations come from the original publications (and thus from different pathologists with different label taxonomies) or were newly generated; inter-annotator agreement; how heterogeneous fine-grained labels were mapped to the four-category scheme (immune, stroma, tumor, normal); or how label consistency across cohorts was validated. The mapping is deferred to Supplementary Material that is not available in the main text. For a resource whose main novelty is annotation quality, this is a critical omission.

### Minor:
4. **Cohort size imbalance not discussed.** KC TLS (3 subjects), LC TLS (5 subjects), and MOB (1 subject) may be insufficient for meaningful multi-sample integration evaluation. The paper does not acknowledge this limitation or recommend how these cohorts should be used.

5. **Sections 3 and 4 are generic literature reviews, not concrete MOCHA protocols.** Section 3 lists normalization methods (TMM, RLE, scran, Seurat, scanpy) and batch correction approaches (Harmony, Crescendo) without specifying which MOCHA recommends or provides. Similarly, Section 4 briefly describes BayeSMART, BASS, and STAGATE without defining any specific benchmark task (train/test splits, metrics, evaluation procedures). This means "protocols for handling batch effects in multi-sample integration" as claimed in the abstract are not actually provided.

6. **No data access information.** The paper mentions MOCHA is released "in formats readily usable with Python and R" but provides no URL, repository link, DOI, or any access mechanism. Without this, the resource cannot actually be used or verified.

7. **Limited technology and tissue diversity.** All 10 cohorts use either 10x Visium or original ST; no imaging-based platforms (MERFISH, STARmap, Slide-seq) are included despite being mentioned in the introduction. Eight of ten cohorts are cancer-related.

### Trivial:
- Two Stahl et al. 2016 references appear in the bibliography (likely a duplication artifact).

## Nice-to-Haves
- A benchmarking study running even 2-3 methods (BayeSMART, BASS, STAGATE) on 2-3 cohorts with standard metrics would dramatically strengthen the paper and provide baseline numbers for future work.
- Include at least one imaging-based SRT dataset to broaden platform coverage.
- Provide a concrete data format specification (e.g., AnnData schema with required fields for expression, coordinates, images, and annotations) and example notebooks.
- Show at least one example visualization of H&E + annotation + expression overlaid to demonstrate data quality and usability.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **"No new data or methods"**: Several reviewers characterized the paper as merely curation with no new data or methods. While true that all datasets are public and Sections 3-4 review existing methods, this is inherent to the resource paper genre. The question is whether the curation and annotation effort adds value, not whether new data/methods were created. The weakness is preserved above as "insufficient specification" rather than "mere curation."
- **"MERFISH/STARmap not included"**: One reviewer demanded inclusion of imaging-based platforms. This is nice-to-have rather than a core flaw, as the paper never promises to cover all platforms.
- **"Code/software availability details absent"**: While a valid concern, this falls under reproducibility/accessibility and is captured in weakness #6. The broader "no code" complaint from the human finder is partially a nitpick about submission format for double-blind review.

## Novel Insights
None beyond the paper's own contributions. The paper identifies a real need but does not deliver sufficient evidence that the resource fills it.

## Suggestions
1. **Run benchmarking experiments** on at least 3 MOCHA cohorts using the methods reviewed in Section 4 (BayeSMART, BASS, STAGATE). Report ARI, NMI, and visualizations against pathologist annotations. This is the single most impactful change.
2. **Include the four-category label mapping in the main text** with per-cohort label distributions, inter-rater agreement (if available), and explicit discussion of information loss from collapsing fine-grained labels.
3. **Provide a concrete data access URL and file format specification** so that the resource can actually be used and verified.
4. **Define at least one concrete benchmark protocol**: e.g., "Use cohorts X, Y, Z; train on subjects {1,...,N-1} per cohort; evaluate ARI on held-out subject N; report results."

## Score and Decision

**Calibration references:**
- LST-Bench (scores 3,3,3,1 — Reject): Had benchmarking experiments with 11 models on 14 datasets, but rejected for limited novelty and shallow analysis. MOCHA is weaker — it has zero experiments.
- GlucoBench (scores 5,6,8 — Accept poster): Had curated datasets + standardized tasks + actual benchmarking with baseline results. MOCHA lacks all of these.
- COMET (scores 5,5,5,8 — Reject): Had experimental evaluation but was rejected for dataset selection bias and overclaims. MOCHA doesn't even have experiments.
- Scenario-Wise Rec (scores 3,3,3,6 — Reject): Had 12 baseline model implementations. MOCHA has none.
- CUS3D (scores 5,5,5,6 — Withdrawn/Reject): Had new data collection + 6 baseline experiments. Still rejected.

MOCHA is below essentially all these comparison papers because it provides no experiments at all, no concrete data specification, and an underspecified annotation harmonization process. Its main contribution (curated multi-sample SRT resource with expert annotations) is conceptually sound but the paper reads as a preliminary description rather than a complete contribution. Even among rejected benchmark/resource papers, those that included experiments scored in the 3-5 range; MOCHA, without any, falls below that.

MY FINAL SCORE: <pineapple>2.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>