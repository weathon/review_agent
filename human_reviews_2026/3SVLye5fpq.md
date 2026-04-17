# EarthScape: A Multimodal Dataset for Surficial Geologic Mapping and Earth Surface Analysis

- Decision: Reject
- Scores: 2, 4, 4, 0

## Abstract
Surficial geologic (SG) maps are essential for understanding surface processes and supporting infrastructure planning, but current workflows are labor-intensive and difficult to scale. We introduce EarthScape, an AI-ready multimodal dataset for SG mapping that integrates digital elevation models, aerial imagery, multi-scale terrain features, and hydrologic and infrastructure vector data within a unified, reproducible pipeline. We report baseline benchmarks across single-modality, multi-scale, and multimodal configurations. In our experiments, terrain-derived features provide the most reliable predictive signal, while spectral inputs and raw elevation degrade substantially under cross-region evaluation. Cross-generalization and multimodal fusion remain challenging, underscoring the need for models that capture shape-driven surface processes. EarthScape offers a geographically compact but modality-rich benchmark for multimodal fusion, domain adaptation, and surface-process modeling.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces EarthScape, a multimodal, multi-scale benchmark dataset designed for surface geology (SG) mapping and broader surface analysis tasks. It integrates diverse data sources, including RGB and NIR imagery, digital elevation models (DEMs), DEM-derived shape features at multiple scales, and vector GIS layers for transportation and hydrological networks. The authors also provide baselines for unimodal, multi-scale, and multimodal configurations using a custom model architecture as well as some FMs using the multi-label classification task.

### Strengths
-	Solid motivation and well-grounded related work. 
-	Inclusion of a geographic hold-out set to evaluate generalization. 
-	Novel and interesting dataset; no similar resource appears to exist according to the related work. 
-	Extensive experimental evaluation, with many results (primarily reported in the appendix).

### Weaknesses
-	The main paper evaluates only one custom architecture. While additional tests with EO foundation models appear in the appendix, this critical section lacks detailed information and ablation studies. 
-	All experiments are confined to Kentucky, leaving uncertainty about performance in other geographic regions. 
-	The paper asserts applicability to segmentation, multi-label classification, and regression, but only provides baselines for the multi-label task. 
-	The main paper includes only a single results table and omits comprehensive ablation studies. Visual diagrams would strengthen the insights presented in Section 4.2. 
-	Model names in tables are inadequately explained, making them difficult to interpret.

### Questions
-	Several prior works on ML for SG mapping are cited, but the Related Work section does not mention any specific SG datasets. What datasets were used in those ML papers? Clarifying this would help position EarthScape relative to existing resources. 
-	The dataset merges multiple modalities, including several DEM-derived features that may carry overlapping information. Which modalities contribute most to performance, and which appear redundant? Figure 10 suggests some redundancy—why not filter out modalities that do not improve results? An ablation or modality importance analysis would strengthen the paper.

### Soundness
3

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
This paper introduces EarthScape, the first AI-ready multimodal dataset for surficial geologic mapping, comprising 31k 256×256 patches with 38 co-registered channels of RGB/NIR imagery, LiDAR DEM, multi-scale terrain derivatives, and hydro-infrastructure vectors. Using the lightweight SGMap-Net baseline, we systematically evaluate unimodal, multi-scale, and multimodal fusion strategies, demonstrating that slope-based features generalize best across geologically distinct regions. Multi-scale channel stacking of elevation percentile, slope, and slope-standard-deviation achieves the top macro-F1 (0.657) with the smallest domain shift (ΔF1 0.059), establishing a reproducible and extensible benchmark for automated surficial geologic mapping.

### Strengths
EarthScape pioneers the first AI-ready surficial-geologic dataset with 31 k rigorously aligned 38-channel image-terrain-vector patches, and its exhaustive single-/multi-scale and multimodal ablations fill the large-scale label gap while quantitatively demonstrating that slope-based features offer the strongest cross-domain generalization.

### Weaknesses
1. The coverage of the dataset is limited.
The dataset only includes data from two counties Warren and Hardin, missing other geomorphological zones such as glacial, arid, and tropical regions, which limits the external validity of cross-domain conclusions

2. The dataset has issues of imbalanced data categories.
Seven types of SG units were sampled according to their natural occurrence frequency. Minority classes (Qaf, Qat) have very few samples, which easily leads to model bias toward majority classes, affecting the precision and recall of minority classes.

3. The architecture of SGMap-Net is overly simple, without any novelty. The choice of aggregation method is not verified by experiments.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces EarthScape, a multimodal dataset designed for surficial geologic (SG) mapping and broader Earth surface analysis. The dataset integrates digital elevation models (DEMs), RGB+NIR aerial imagery, multi-scale terrain derivatives (e.g., slope, curvature), and vector-based features for hydrology (NHD) and infrastructure (OSM). It is sourced from publicly available U.S. Geological Survey (USGS) SG maps covering limited regions of the U.S., processed into 256x256 patches with 38 co-registered channels. The authors provide a reproducible processing pipeline and establish baselines using standard segmentation models (e.g., U-Net variants) across unimodal, multi-scale, and multimodal setups. Key challenges highlighted include class imbalance, geographic heterogeneity, and the need for multimodal fusion. Experiments demonstrate that terrain derivatives are highly predictive, but cross-region generalization remains poor, positioning EarthScape as a benchmark for domain adaptation and fusion techniques.

### Strengths
1. The dataset addresses a practical gap in geospatial domain by focusing on surficial geology, which has applications in hazard mitigation, urban planning, and climate studies.

2. Integration of multi-scale terrain features and vector data adds a layer of physical interpretability, potentially benefiting downstream tasks beyond geology, such as autonomous navigation or medical imaging analogies.

3. The emphasis on reproducibility via an end-to-end pipeline and use of public sources is commendable, and the baselines provide a clear starting point for future comparisons.

### Weaknesses
1. Limited Novelty and Broader Contribution: Dataset papers are increasingly common in machine learning, particularly in remote sensing and geospatial domains. EarthScape, while tailored to SG mapping, does not introduce particularly unique elements—its multimodal fusion of imagery, DEMs, and derivatives echoes existing RS benchmarks, and the focus on surface morphology is not sufficiently differentiated from prior works on landslides or land cover. The "living resource" claim lacks specifics on expansion plans, and the dataset's U.S.-centric nature limits its global representativeness, potentially reducing its appeal as a general benchmark for multimodal learning or domain adaptation.

2. Scope and Scale Concerns: The dataset appears constrained in geographic diversity, relying on site-specific SG maps that may not capture global variability in geologic processes. Class imbalance and long-tail issues are mentioned but not quantified in detail, which is a missed opportunity to highlight uniqueness. Additionally, the processing pipeline, while reproducible, relies on standard tools, and the choice of 256x256 patches at fixed scales may not adequately address multi-resolution challenges in real-world RS tasks.

3. Baseline Experiments and Evaluation: The baselines seem preliminary, focusing on standard architectures without exploring state-of-the-art multimodal models (e.g., CLIP variants, Perceiver IO, or recent geospatial transformers like GeoCLIP). Results emphasize terrain predictiveness but lack ablation on modality contributions or comparisons to non-ML methods (e.g., traditional GIS-based mapping). Cross-region generalization is identified as a challenge, but without metrics like domain shift quantification (e.g., MMD or adversarial scores), it's unclear how EarthScape advances beyond existing datasets. Moreover, no discussion of computational efficiency or scalability for large-scale deployment.

4. Minor Issues: The writing occasionally overstates impact (e.g., "unusually challenging benchmark") without empirical evidence comparing difficulty to other datasets.

### Questions
1. Why were the baselines limited to basic configurations? Could you include comparisons to recent multimodal foundation models (e.g., Prithvi or SatMAE) or non-deep learning approaches to better contextualize performance?

2. How was class imbalance addressed in training (e.g., weighted losses, oversampling)? Did you evaluate sensitivity to patch size or scale selection in the terrain derivatives?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes a dataset called EarthScape for surficial geological mapping using multimodal remote sensing data. There are many datasets for remote sensing and land cover mapping, disaster detection, and other anthropomorphic or single-event phenomena, there are few datasets like EarthScape for surface geology. The dataset covers sites in two counties in Kentucky. It includes aerial RGB+NIR imagery and LiDAR-derived DEMs with classification and segmentation labels, both provided by Kentucky agencies. The paper also proposes a model, SGMap-Net, to model the dataset. The paper shows experiments using the dataset for multiple fusion strategies and architecture choices and shows that SGMap-Net performs best overall.

### Strengths
- The paper is well written and easy to read.
- There is a gap in datasets for surface geology in the remote sensing + ML research space, which this dataset helps fill.
- There are not many multimodal becnhmark remote sensing datasets that include DEMs, especially where DEMs are an important variable, so this dataset helps fill that gap.

### Weaknesses
- The dataset is called EarthScape, implying a global scale, but only represents two counties in Kentucky. The paper says that more sites will be added by 2026, but it’s unclear how that will be done and this work does not seem complete or high enough impact in its current form (before the broader geographic scope).
    - The paper says the limitation of two regions in Kentucky reflects the availability of 1:24k scale SG maps, but it seems there are a ton available from the USGS: https://ngmdb.usgs.gov/mapview/?center=-112.161,38.452&zoom=9
    - If this is meant to be a “living resource” how will it be updated, versioned, and kept consistent for use as an effective research benchmark? It doesn’t seem like the planned pace of its expansion will keep up with multimodal research in ML.
- The paper states that one of the challenges of machine learning for surficial geologic maps is that the classes on the surface are continuous and geologic maps are often subjective depending on the expertise or preferences of the mapper. How is this class ambiguity handled in the dataset? It seems like this aspect is not addressed.
- The aerial and lidar datasets both come from Kentucky agencies, so it’s not clear how it will be expanded or applied to other regions where those same datasets may not exist.
- The paper states that the dataset is a challenging benchmark for multimodal surface-aware learning and will contribute to multimodal learning research with remote sensing, but the experiments don’t engage with any recent work in multimodal learning for remote sensing (for example, geospatial foundation models that have RGB+NIR and DEM input channels).

### Questions
- If all of the code for creating/reproducing the dataset is done, why will it take until end of 2026 to add more regions?
- What prevents the use of the many 1:24k scale SG maps on the USGS site?
- If this is meant to be a “living resource” how will it be updated, versioned, and kept consistent for use as an effective research benchmark?
- How is this class ambiguity stemming from subjective geologic mapping and continuous landforms/materials handled in the dataset?

### Soundness
2

### Presentation
3

### Contribution
1
