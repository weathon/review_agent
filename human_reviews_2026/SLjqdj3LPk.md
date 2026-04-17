# CHAMMI-75: Pre-training multi-channel models with heterogeneous microscopy images

- Decision: Accept (Poster)
- Scores: 4, 8, 6

## Abstract
Quantifying cell morphology using images and machine learning has proven to be a powerful tool to study the response of cells to treatments. However, models used to quantify cellular morphology are typically trained with a single microscopy imaging type. This results in specialized models that cannot be reused across biological studies because the technical specifications do not match (e.g., different number of channels). Here, we present CHAMMI-75, an open access dataset of heterogeneous, multi-channel microscopy images from 75 diverse biological studies. We curated this resource from publicly available sources to investigate cellular morphology models that are
channel-adaptive and can process any microscopy image type. Our experiments show that training with CHAMMI-75 can improve performance in multi-channel bioimaging tasks primarily because of its high diversity in microscopy modalities. This work paves the way to create the next generation of cellular morphology models for biological studies.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents CHAMMI-75, a large-scale multi-channel microscopy dataset that aggregates 75 public biological studies and 18 imaging platforms. The authors provide detailed documentation of data collection, cleaning, redundancy control, and scaling studies comparing SSL frameworks on this dataset.
While CHAMMI-75 fills an important data gap and is technically well executed, the paper focuses primarily on dataset construction and benchmarking. It lacks methodological innovations addressing the intrinsic challenge of multi-channel adaptation and provides limited analysis on how dataset diversity affects model generalization.

### Strengths
1. The CHAMMI-75 dataset integrates 75 biological projects and 18 imaging platforms, providing unprecedented scale and diversity for microscopy SSL. This makes it a valuable benchmark for future research.
2. The authors conduct careful cleaning, redundancy reduction, and balanced sampling, improving representativeness and reproducibility.
3. The paper includes a thorough comparison of SSL methods and analyzes performance scaling with dataset size and channel configurations, validating the dataset’s utility.

### Weaknesses
1. The paper’s main contribution lies in data curation and benchmarking. The experimental pipeline relies heavily on existing models without proposing new architectures or training strategies tailored to multi-channel adaptation.
2. Although the dataset is multi-source and multi-channel, the study does not analyze how factors such as number of channels, organism diversity, or imaging modality affect representation learning. This limits understanding of what drives cross-domain generalization.
3. The reported SSL benchmarks focus on general downstream accuracy rather than the core motivation of channel generalization or cross-modality transfer, leaving a gap between dataset design and evaluation goals.

### Questions
The questions can be found in the weakness part. I understand this is a dataset and benchmark paper so the method part is not the research focus. However, additional validation experiments are essential for better understanding the benchmark. I am open to raise my score if more analysis is included in the rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents CHAMMI-75, a dataset of heterogeneous multi-channel microscopy images for pre-training channel-adaptive cellular image analysis models. The dataset comprises 2.8 million multi-channel images from 75 diverse sources, featuring more than 1.8 billion cells, resulting in a curated, high-quality resource that exhibits greater biological and technical variation than other microscopy datasets used in prior work. The authors train vision transformer models using self-supervised methods, such as DINO, MAE, and SimCLR. The trained models are then evaluated on several downstream tasks, like cell-cycle stage classification and protein localization classification, among others.

### Strengths
The dataset’s large scale is impressive and would be a great unified resource for other researchers. The results for downstream tasks provide real value. The authors also added metadata information, specifically creating 22 metadata fields. Additionally, they segmented the nucleus of 1.8 billion cells using CellPose, which is very useful for the community. The benchmarks across different microscopy tasks demonstrate the generalizability of the proposed models. The statement from the authors to make the code and the dataset publicly available is highly appreciated and will help the community.

### Weaknesses
* While multiple benchmarks are used, the analysis could be strengthened by deeper biological interpretation of learned representations e.g., probing whether CHAMMI-75 pre-training improves biological feature disentanglement or transfer to unseen modalities

* The paper acknowledges missing and inconsistent metadata, but the implications for training robustness and domain bias are missing/underexplored.

* The evaluation seems to be focusing primarily on fluorescence microscopy. Brightfield datasets like LIVECell and EVICAN are part of the training set but aren’t used as an evaluation benchmark. Including label-free imaging in the evaluation would better display the transfer of learned representations across different modalities.
There are other recent papers like Segment Anything for Microscopy (Nature Methods, 2025), which discuss the cross-modality generalization in microscopy using foundation models for the task of segmentation. A clarification on how CHAMMI-75 complements or differs from such methods would highlight the contribution of this paper much more.

### Questions
* How does CHAMMI-75 handle cases where biological metadata (e.g., organism or stain identity) is ambiguous or inconsistent? 

* Did the authors attempt cross-domain zero-shot evaluations (e.g., predicting unseen channel types)?

* How do the learned embeddings compare qualitatively (e.g., via t-SNE or UMAP) across datasets with different channel counts?
To better support the claim that CHAMMI-75 enables models to learn domain-invariant and biologically meaningful representations, I recommend including qualitative embedding visualizations such as UMAPs or t-SNE plots of the learned single-cell features. These could illustrate how samples cluster across different datasets, channel configurations, or biological conditions (e.g., cell type, treatment, protein localization). Comparing embeddings from representative models would provide intuitive evidence for cross-domain consistency and morphology awareness, complementing the quantitative benchmarks

* Can authors provide guidance for researchers who want to finetune models on their own microscopy data (e.g., normalization or channel alignment practices)?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents CHAMMI-75, a curated dataset of multi-channel microscopy images from 75 studies with standardized formats, and rich metadata for content-aware sampling. The authors focus on data acquisition, metadata integration, and redundancy reduction, then use the dataset to pre-train ViT models. Benchmarks span five tasks, including two newly constructed evaluations. A DINO-based BoC ViT-small pretrained on CHAMMI-75 delivers the strongest SSL results across most tasks, shows favorable data/model scaling trends, and outperforms models trained on narrower sources.

### Strengths
- The dataset has unmatched scale and heterogeneity for multi-channel microscopy, which supports broad generalization.
- The benchmarks cover diverse, realistic tasks and include novel channel combinations, which enhances external validity.
- The scaling analysis is careful and practical, which offers guidance on data size, model size, and multi-channel strategy.
- The SSL results are consistently strong across tasks, which validates CHAMMI-75 as a useful pre-training resource. And the release plan supports reproducibility and downstream use.

### Weaknesses
- The work offers limited methodological novelty, which centers contributions on data and benchmarking.
- The comparison to SubCell mixes training regimes and model sizes, which hinders clean conclusions about capability gaps.
- The LLM-assisted metadata extraction lacks error quantification, which raises concerns about label noise in curation.

### Questions
Please refer to my weakness section.

### Soundness
4

### Presentation
4

### Contribution
3
