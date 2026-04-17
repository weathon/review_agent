# OmniSTVG: Toward Spatio-Temporal Omni-Object Video Grounding

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
We introduce spatio-temporal omni-object video grounding, dubbed $\textbf{OmniSTVG}$, a new STVG task aiming to localize spatially and temporally all targets mentioned in the textual query within videos. Compared to classic STVG locating only a single target, OmniSTVG enables localization of not only an arbitrary number of text-referred targets but also their interacting counterparts in the query from the video, making it more flexible and practical in real scenarios for comprehensive understanding. In order to facilitate exploration of OmniSTVG, we propose $\textbf{BOSTVG}$, a large-scale benchmark dedicated to OmniSTVG. Specifically, BOSTVG contains 10,018 videos with 10.2M frames and covers a wide selection of 287 classes from diverse scenarios. Each sequence, paired with a free-form textual query, encompasses a varying number of targets ranging from 1 to 10. To ensure high quality, each video is manually annotated with meticulous inspection and refinement. To our best knowledge, BOSTVG, to date, is the first and the largest benchmark for OmniSTVG. To encourage future research, we present a simple yet effective approach, named $\textbf{OmniTube}$, which, drawing inspiration from Transformer-based STVG methods, is specially designed for OmniSTVG and demonstrates promising results. By releasing BOSTVG, we hope to go beyond classic STVG by locating every object appearing in the query for more comprehensive understanding, opening up a new direction for STVG. Our project is released at https://jellyyao3000.github.io/OmniSTVG/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the OmniSTVG task where multiple objects mentioned in the text query have to be grounded spatially and temporally. The BOSTVG dataset is collected, consisting of 10K videos manually annotated with spatio-temporal tubes for 1-10 objects, including 287 object classes in total. The OmniTube model is presented, including separate spatial and temporal decoders, text-guided query generation and multi-tube prediction. This model outperforms single-object methods adapted to the new task, and is also about 4x fater than naive approaches of running a single-object model for each object in the query.

### Strengths
- A new dataset that addresses an important limitation of previous STVG approaches
- OmniTube performs well compared to prior approaches and has been ablated extensively

### Weaknesses
- Assuming all objects share the same temporal segmentation is a strong assumption
- It is confusing what the “baseline” is meant to show in the paper: it excludes various features present in prior work, e.g. text-guided query generation in https://arxiv.org/abs/2502.11168 or alternative spatial and temporal blocks in https://arxiv.org/abs/2203.16434. Note that these features are rightfully not being presented as contributions in the introduction.

### Questions
- Out of curiosity, is the architecture able in practice to ground “zero-shot” objects not seen in the training set?
- How much is each of ResNet and VidSwin feature helpful for the STVG performance?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Existing Spatio-Temporal Video Grounding (STVG) tasks are impractical, as they are conventionally trained to localize only a single target, even when multiple targets are referenced in the text.

This paper proposes a more practical new task, termed OmniSTVG (Omni-target Spatio-Temporal Video Grounding), which aims to localize all objects mentioned in the textual query. The primary contributions include:

New Task (OmniSTVG): Defining the aforementioned new direction of "omni-target" localization.

New Dataset (BOSTVG): To support this task, a large-scale, high-quality dataset comprising 10,000 videos was constructed, which has undergone multiple rounds of manual refinement.

New Baseline (OmniTube): An effective Transformer-based model is provided, which utilizes "text-guided queries" to concurrently localize all targets.

### Strengths
Pioneering a New Direction: The work identifies the critical limitation of existing STVG tasks (i.e., single-target grounding) and defines a more complex and practical new research direction (i.e., multi-target grounding).

Contribution of a Core Resource (BOSTVG): It provides the field's first large-scale (10,000 videos), high-quality benchmark dataset specifically dedicated to "omni-target" localization, establishing a core asset for advancing subsequent research.

Provision of a Strong Baseline (OmniTube): The study does not merely pose the problem but also delivers a well-designed (e.g., "text-guided queries") and empirically validated solution, offering a solid baseline for future work.

Rigorous and Solid Experimentation: The effectiveness of the model's constituent components is validated through exhaustive ablation studies, while comparative experiments underscore the uniqueness and necessity of the new BOSTVG dataset.

The manuscript is clearly and normatively written; the figures are highly comprehensible and meticulously detail the proposed methodology.

### Weaknesses
**Insufficient Analysis**:Regarding the benchmark, an analysis of the task's inherent difficulties is absent. The inclusion of illustrative examples to demonstrate these challenges would be beneficial.Furthermore, a deeper analysis is required as to why existing algorithms, including those compared in this study, cannot be directly or effectively applied to this benchmark.In the ablation study, the paper merely enumerates the functional contributions of individual algorithmic components without providing further in-depth analysis.

**Insufficient Representativeness of "Omni" (Data Distribution Imbalance)**:In Section 3.4 (Dataset Splits), the test set is partitioned into three groups: BOSTVG-Low (1-3 targets, 1566 samples), BOSTVG-Medium (4-6 targets, 273 samples), and BOSTVG-High (>7 targets, only 73 samples).The core objective of the paper is to address the "omni-target" problem; however, the proportion of high-density, genuinely complex "omni-target" samples (73) within the test set is extremely low (approximately 3.8% of the test set).This implies that the model's capability in handling complex, multi-target scenarios has not been sufficiently validated. The reported overall average performance (e.g., $m\_vIoU$) is likely dominated by the simpler "Low" group.

**Questionable Model Generalizability and Data Compatibility (Poor Generalizability)**:In Section 5.2 (Table 8: Ablation on Training Data), when attempting to merge BOSTVG with the existing single-target dataset HCSTVG-v2 for training (Row 3), performance paradoxically decreases slightly compared to training on BOSTVG alone (Row 1).Typically, augmenting the training data, even with related datasets, is expected to enhance model robustness.This performance degradation (which the authors attribute to "data inconsistency") may suggest that the model (OmniTube) or the annotation style of the dataset (BOSTVG) is prone to overfitting, hindering its ability to generalize or maintain compatibility with other data sources.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a new video grounding task, OmniSTVG, which differs from the traditional STVG task in that it can localize all targets mentioned in the textual query as well as the interactive relationships existing among these targets. To this end, the paper constructs a large-scale dataset BOSTVG and presents a simple yet effective model named OmniTube.

### Strengths
- The task proposed in the paper expands the scope of the traditional STVG task. Moreover, the proposed dataset has a wide range of sources and is built using a relatively rigorous manual annotation method, combining both scale and quality.
- The paper puts forward a simple and effective baseline, and has implemented and compared several public models.
- The paper provides detailed descriptions of details, making it easy to follow.

### Weaknesses
The paper does not mention the performance of multimodal large language models (MLLMs) on this task.

### Questions
Why is the performance of any multimodal large language models on this dataset not provided?

### Soundness
3

### Presentation
4

### Contribution
3
