# OmniWorld: A Multi-Domain and Multi-Modal Dataset for 4D World Modeling

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
The field of 4D world modeling—aiming to jointly capture spatial geometry and temporal dynamics—has witnessed  remarkable progress in recent years, driven by advances in large-scale generative models and multimodal learning.  However, the development of truly general 4D world models remains fundamentally constrained by the availability  of high-quality data. Existing datasets and benchmarks often lack the dynamic complexity, multi-domain diversity,  and spatial-temporal annotations required to support key tasks such as 4D geometric reconstruction, future  prediction, and camera-controlled video generation. To address this gap, we introduce OmniWorld, a large-scale,  multi-domain, multi-modal dataset specifically designed for 4D world modeling. OmniWorld consists of a newly  collected OmniWorld-Game dataset and several curated public datasets spanning diverse domains. Compared with  existing synthetic datasets, OmniWorld-Game provides richer modality coverage, larger scale, and more realistic  dynamic interactions. Based on this dataset, we establish a challenging benchmark that exposes the limitations of  current state-of-the-art (SOTA) approaches in modeling complex 4D environments. Moreover, fine-tuning existing  SOTA methods on OmniWorld leads to significant performance gains across 4D reconstruction and video generation  tasks, strongly validating OmniWorld as a powerful resource for training and evaluation. We envision OmniWorld  as a catalyst for accelerating the development of general-purpose 4D world models, ultimately advancing machines’  holistic understanding of the physical world.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes OmniWorld, a large-scale dataset designed to address the lack of high-quality data for 4D world modeling.
This paper proposes OmniWorld-Game, a benchmark for 3D geometry prediction and camera control video generation.
Experiments show that fine-tuning existing state-of-the-art models on OmniWorld significantly improves their performance in 4D reconstruction and video generation tasks, validating the value of this dataset.

### Strengths
OmniWorld covers four major areas: simulators, robotics, humans, and the internet, and contains over 600,000 video sequences and 300 million frames of data. OmniWorld-Game significantly surpasses existing synthetic datasets in modal diversity (such as depth maps and optical flow) and data volume, providing a rich resource for 4D modeling.

Challenging benchmarks are constructed for 3D geometry prediction and camera control video generation. Multi-model evaluation reveals current state-of-the-art shortcomings. Fine-tuning experiments demonstrate that the dataset effectively improves model performance and verifies the integrity of the logic.

### Weaknesses
Human-domain data accounts for the largest proportion in OmniWorld, while data from other domains (such as robotics and the internet) is relatively scarce. This may lead to insufficient training of the model's generalization ability in non-human-related scenarios.

The core subset, OmniWorld-Game, is derived from a gaming environment. While it strives to simulate real-world scenarios, differences exist between game rendering and the real physical world, potentially affecting the model's transferability to real-world scenarios. Furthermore, the impact of this "simulation-to-reality" gap on task performance has not been thoroughly analyzed.

In the evaluation of 3D geometry prediction and camera control video generation, the main reliance is on quantitative indicators (such as Abs Rel, FVD) and some qualitative comparisons. There is a lack of evaluation of the model's performance in extreme scenarios (such as fast motion and complex occlusion), which cannot fully reflect the model's robustness.

### Questions
Regarding the problem of uneven data distribution, are there any plans to supplement data from fields such as robotics and the Internet, or use data augmentation technology to balance the data proportions in each field to improve the model's generalization capabilities in multiple scenarios?

OmniWorld-Game is derived from a game environment. How can we quantify the difference between this synthetic data and real-world data? Are there any experiments validating the performance of models trained on OmniWorld when transferred to real-world datasets (such as real indoor/outdoor 4D datasets)?

The current benchmark does not involve the evaluation of extreme scenarios (such as intense motion, severe occlusion, and low light). Will you consider expanding the benchmark's scene coverage in the future to more comprehensively measure the model's 4D modeling capabilities under complex real-world conditions?

### Soundness
3

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
The paper introduces OmniWorld, a large-scale multi-domain and multi-modal dataset (RGB, depth, pose, text, optical flow, mask) designed for 4D world modeling. It unifies new synthetic data (OmniWorld-Game) and re-annotated public sets, supporting benchmarks for 3D geometry prediction and camera-controllable video generation. The dataset features higher resolution, richer modalities, and more complex dynamics than existing resources like Sintel or TartanAir. Its automated annotation pipeline and unified benchmark design expose clear weaknesses of current geometric and generative models, and fine-tuning on OmniWorld improves performance across multiple datasets.

However, the paper lacks clarity on how dynamic objects and camera motions are modeled, leaving its 4D claim partially unsubstantiated. The work also relies heavily on synthetic data, with limited validation on real-world domains (human, robot, internet). Moreover, the quality and uncertainty of automatically generated annotations are not quantitatively analyzed.

Overall, OmniWorld is a valuable and ambitious contribution that could serve as a strong foundation for 3D and 4D perception research. Clearer handling of dynamic scenes, stronger real-world evaluation, and annotation-quality validation would further improve the paper. My current recommendation is Borderline Accept.

### Strengths
1. **Significant scale and modality improvement.**
Compared with existing synthetic datasets, OmniWorld-Game offers clear advantages in resolution, frame count, and modality diversity. It also covers more dynamic and complex scenes. Table 1’s comparison against Sintel, TartanAir, and HyperSim highlights this superiority.

2. **Cross-domain integration and automated annotation pipeline.**
The paper clearly details an end-to-end annotation pipeline, from four-domain collection, video slicing and filtering, to modality annotation, and provides domain composition statistics, which support reproducibility and future extension.

3. **Challenging benchmark design.**
Compared with prior benchmarks featuring short sequences and smooth camera motion, OmniWorld-Game introduces longer sequences, complex trajectories, and dynamic scenes. On this benchmark, most state-of-the-art GFM/video generation models exhibit weaknesses in temporal consistency and camera control (as shown in Table 3/Table 4 and qualitative examples).

4. **Demonstrated training value.**
Models such as DUSt3R, CUT3R, and Reloc3R, when fine-tuned on OmniWorld, show consistent performance gains on standard public benchmarks (Sintel, KITTI, NYUv2), indicating that the dataset provides practical benefits for improving 3D geometry and temporal coherence.

### Weaknesses
1. **Unclear handling of dynamic 4D content.**
Given the central claim of 4D world modeling, the paper should more explicitly describe how dynamic objects and camera motions are represented during dataset construction, particularly in the synthetic (simulation) domains. It is unclear whether the dataset systematically varies object motion patterns or camera trajectories, or includes a taxonomy distinguishing static versus dynamic regions. In evaluation, the study focuses mainly on static 3D foundation models; incorporating 4D reconstruction or dynamic scene methods (e.g., MegaSAM) would make the benchmark more consistent with its 4D ambition and strengthen its empirical validity.

2. **High reliance on synthetic data and limited real-world evaluation.**
Although the authors highlight the multi-domain nature of OmniWorld, both training and evaluation are largely centered on OmniWorld-Game, the synthetic/game-based subset. Systematic experiments on the human, robot, and internet domains are limited, which makes it difficult to assess how well the dataset and benchmarks generalize to real-world conditions. A stronger cross-domain analysis would clarify the practical robustness of the proposed data.

3. **Lack of annotation-quality assessment and uncertainty analysis.**
Many modalities (depth, pose, optical flow, and foreground masks) rely on automatic or pseudo-labeling pipelines, yet the paper provides no quantitative evaluation of their accuracy or reliability. It would be valuable to include error quantification (e.g., comparisons with manual ground truth or high-confidence models), failure-case analysis, and sensitivity studies showing how annotation noise affects training and evaluation outcomes. These analyses would help validate the trustworthiness of the dataset’s supervision signals.

### Questions
See weaknesses.

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
4

### Summary
This paper introduces a large-scale, multi-domain, multi-modal 4D dataset. It comprises over 214 hours of videos, includes simulation, human, robot and internet, and provides rich annotations including depth, camera, text, flow and masks. Based on this dataset, the paper also establishes a challenging 4D benchmark and validates the dataset as a powerful resource for training.

### Strengths
* The dataset is large-scale, multi-domain and multi-modal, while previous datasets are limited.
* The paper proposes an annotation pipeline for the data from different domains and different outputs.
* The paper validates that the dataset can be a powerful resource for both training and evaluation.

### Weaknesses
* It's hard to validate the quality of the dataset. The paper only leverages the pretrained models for annotation. The open-source real-world datasets are very noisy and the pretrained models can not handle them perfectly. How to filter the bad cases?
* The paper lacks the ablation analysis of the annotation pipeline. Which pretrained model is most suitable for each dataset and why.
* The paper lacks the analysis of how to select the benchmark samples.

### Questions
See Weakness.

### Soundness
2

### Presentation
3

### Contribution
2
