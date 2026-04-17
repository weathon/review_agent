# InternSpatial: A Comprehensive Dataset for Spatial Reasoning in Vision-Language Models

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Recent benchmarks and datasets have been proposed to improve spatial reasoning in vision-language models (VLMs), yet existing open resources remain constrained by limited scale, narrow visual diversity, and restricted instruction expressiveness. To address these gaps, we present InternSpatial---the largest open-source dataset for spatial reasoning in VLMs---alongside InternSpatial-Bench, a comprehensive evaluation benchmark designed to assess spatial understanding across diverse instruction formats. InternSpatial contains 12 million question-answer(QA) pairs covering both single-view and multi-view scenarios, sourced from varied visual environments and supporting 19 distinct instruction formats that mirror real-world query patterns. InternSpatial-Bench aims to single-view assessment and also extends multi-view reasoning through a novel rotation estimation task. Experimental validation demonstrates that models trained on \trainset achieve substantial performance improvement of 12.1% on InternSpatial-Bench and 10.7% on VSI-Bench, while preserving competitive performance on general-purpose benchmarks. We expect these resources can advance the development of spatially-capable VLMs for practical applications in robotics and embodied AI systems. Our codes and datasets are publicly available at https://github.com/dengnianchen/intern-spatial.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Authors introduce a 12M QA-pair dataset containing both single and multi view images. This dataset aims to improve spatial reasoning in vision-language models (VLMs). The data is sourced from diverse domains (in-the-wild, indoor, driving, object-centric, embodied) and also contains an evaluation benchmark of 6,000 QAs. 
The data generation pipeline uses automated annotation (e.g. object masks, depth estimation) with pretrained model followed by template-based QA generation.
The authors finetune an InternVL2.5-8B models on their dataset and obtain clear performance gains on spatial reasoning benchmarks including their own, while maintaining performance on general VQA benchmarks.

### Strengths
1. Clear improvement over baseline when trained on new data. 
2. Thorough analysis of dataset statistics.
3. Large-scale open-source dataset contribution

### Weaknesses
1. Only `InternVL2.5-8B` is used as a baseline. How does this dataset help other models in general? 
2. Only an 8B scale model is used. Will this data improve smaller (e.g. 2B-3B models), large (14B-70B), and MoE models? 
3. Minimal description of data generation process. How do the automated tools (e.g. SAM for bbox / mask) perform? What are the error rates? These are not clearly discussed in the paper. 
4. QA generation: diversity? These is little analysis on the dataset. Maybe calculate diversity metrics on the text and image data. Also, given the template based generation (as opposed to human or LLM), this data diversity concern is amplified. 
5. Test data leakage: The training dataset is created using 3D information of some datasets, "integrated multi-view data derived from the training splits of the ScanNet/MultiScan/R2R/Objaverse". At the same time, their benchmark uses some of these same datasets' test splits to evaluate. Even other benchmark (e.g. VSI contains ScanNet data) use this common data. Could the performance improvement be due to this similar domain data being used to create the training dataset?

### Questions
See weaknesses. 

1) In Table 1, the authors mention their dataset "InternSpatial" as Open-source. Is the dataset already open-source? If not, this claim is false? 

2) See related work sections of below papers for 2D spatial reasoning. Consider discussing prior 2D spatial reasoning works in detail?
  - Ferretv2: https://arxiv.org/abs/2404.07973 
  - LocVLM: https://arxiv.org/abs/2404.07449 
  - Shikra: https://arxiv.org/pdf/2306.15195

### Soundness
2

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
This paper introduces InternSpatial and the corresponding benchmark for spatial reasoning. The dataset contains 12 million QA pairs spanning both single-view and multi-view scenarios, covering 19 instruction formats (textual and visual). It also proposes InternSpatial-Bench, a new benchmark with a new rotation angle prediction task. Experiments are performed on various benchmarks.

### Strengths
1.	The proposed dataset is large-scale especially regarding the number of QA pairs.
2.	The data generation pipeline is sound.
3.	It shows promising performance leveraging the curated data.

### Weaknesses
1.	Could the author compare the proposed dataset with previous ones in terms of scenarios, question types, etc.?
2.	Could the authors validate the effectiveness of the proposed data on more open-source frameworks?
3.	Could the authors explain the performance show limited gain on rotation estimation and object counting?
4.	Could the generated QA pairs reflect the complexity or ambiguity of human spatial questions.
5.	Will data generation pipeline and data be publicly available?

### Questions
Please refer to Weakness.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes InternSpatial, which is claimed to be the largest spatial QA dataset with 12M data. They sourced the data from 2D images as well as 3D datasets of various sources, generating single-view and multi-view QA pairs. They report the scores of various models on the InternSpatial Bench, including InternVL-8B model trained on their datasets. They also report results on VSI-Bench as well as other general benchmarks.

### Strengths
In general, I think a12M dataset is a quite significant improvement from previous QA datasets in terms of data quantity. The dataset comes from a wide variety of data, as shown by Figure 4. Results on InternSpatial-Bench, VSI-Bench as well as other general benchmark results show that training on this dataset brings a lot of improvements.

### Weaknesses
It would be great to understand how much the image datasets are helping with the training in general. The alignment to view space from 2D images requires depth estimation followed by camera estimation, both of which could potentially introduce significant errors. I wonder if it would be possible to see the improvements based on InternVL-Spatial-8B trained with only 3D datasets and/or only 2D datasets. Also, this would give more insights on whether there are domain gaps within the training dataset itself.

The paper could also be strengthened by showing more baselines of models specialized in 3D on InternSpatial-Bench (e.g., SpatialMLLM, SpaceR, etc). Currently, the results shown are mainly on general VLMs and not VLMs specifically trained for spatial reasoning. 

The paper would also be improved by showing results of more methods finetuned with InternSpatial dataset to further show the effectiveness of the dataset on different model architectures/training methods.

### Questions
Overall, my main questions of the paper are two-fold:

1. Are both dataset sources (2D and 3D images) helpful in contributing to the training when evaluating on other benchmarks such as VSI-Bench? Are there potential ways to filter out results of bad prediction during 2D data preprocessing?

2.Are there results of other methods finetuned on this dataset? Does that bring any further improvements?

Overall, I do see this dataset being beneficial to the community, so I am leaning towards accept. The questions I believe would significantly add to the contribution.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents InternSpatial, a large-scale open-source dataset (12M QA pairs) designed to improve spatial reasoning in Vision-Language Models (VLMs). It addresses key limitations in prior works, such as limited scene diversity, narrow instruction formats, and lack of multi-view supervision, by aggregating data from a wide range of sources (COCO, Visual Genome, ScanNet, Cityscapes, Objaverse, R2R, etc.) and generating question-answer pairs with 19 instruction formats spanning both textual and visual variations. The authors also propose InternSpatial-Bench, a benchmark comprising 6,008 QA pairs that evaluate single-view and multi-view spatial reasoning, including a new rotation angle prediction task.

Models trained on InternSpatial (notably, InternVL-Spatial-8B) show large performance gains +12.1% on InternSpatial-Bench and +10.7% on VSI-Bench, while maintaining comparable performance on general VQA tasks, confirming that spatial reasoning gains do not come at the expense of general multimodal ability

### Strengths
### Reasonable Dataset Design
- The dataset covers diverse visual domains (indoor, outdoor, object-centric, embodied, urban) and both single-view and multi-view reasoning setups.
- It supports a wide variety of instruction modalities, text, bounding boxes, masks, numeric indicators, coordinate-based prompts, totaling 19 instruction types, a major advancement over prior datasets like SpatialVLM or OSD.
### Data Generation Process
- The data pipeline integrates multiple pretrained modules for depth, segmentation, and camera parameter estimation (SAM2, Metric3Dv2, PerspectiveFields, WildCamera) to lift 2D annotations into 3D canonical view space. The pipeline is modular and reproducible, allowing flexible annotation generation and QA synthesis without relying on expensive LLM prompting for each sample.

### Novel Multi-view Reasoning Component
- The addition of rotation angle prediction is new and well-motivated for embodied AI and robotics. Multi-view QA construction uses geometric consistency (e.g., Alpha Shape–based room estimation, OrientedBoundingBox fitting) to ensure physically grounded reasoning.

### Strong Experimental Results and Benchmarking
- The evaluation suite is broad: InternSpatial-Bench, VSI-Bench, and five standard multimodal tasks.
- Results show large, consistent improvements in spatial reasoning, including outperforming commercial VLMs like GPT-4o and Claude 3.7 Sonnet in several spatial tasks.
- Ablation studies isolate the effects of instruction format diversity and confirm its value for cross-format generalization.

### Weaknesses
### The limit of Template-Driven QA Generation. 

While efficient, the template-based QA generation may lead to limited linguistic diversity and potential overfitting to templated phrasing. The authors acknowledge this, but do not quantify how template rigidity affects generalization to natural human queries.

### Lack of Qualitative Error Analysis

The evaluation focuses almost exclusively on quantitative metrics. There is little qualitative examination of failure modes (e.g., reasoning about occluded objects, symmetry, or ambiguous rotations).

### Over-Reliance on InternVL2.5 Backbone

Experiments are restricted to fine-tuning InternVL2.5-8B. The generality of the dataset across architectures (e.g., LLaVA, Qwen2.5-VL) is not tested. This limits claims of dataset generalizability.

### Rotation Task Evaluation Unclear

The “rotation angle prediction” task is introduced as novel but evaluated using classification accuracy, without specifying the label granularity (e.g., 15° bins?). Clarifying this would help interpret improvements.

### Questions
### Template Diversity:
How many distinct QA templates were used, and how was linguistic diversity ensured across 12M samples? Were any human validation steps introduced beyond filtering for ambiguity?

### Instruction Format Sampling:
Given 19 formats, how were the subsets for each training batch sampled? Is there a weighting scheme, or are they uniformly sampled?

### Multi-View Ground Truth Validation:
For the rotation estimation task, how were ground-truth rotation angles derived or verified in scenes where camera calibration might be uncertain?

### Cross-Model Evaluation:
Have the authors tested whether models other than InternVL2.5 (e.g., LLaVA-OneVision, Qwen-VL) benefit similarly from InternSpatial training?

### Human Baseline or Difficulty Assessment:
Has any human performance baseline been measured on InternSpatial-Bench to contextualize the difficulty of the tasks?

### Soundness
3

### Presentation
3

### Contribution
3
