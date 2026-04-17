# FETAL-GAUGE: A BENCHMARK FOR ASSESSING VISION-LANGUAGE MODELS IN FETAL ULTRASOUND

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
The growing demand for prenatal ultrasound imaging has intensified a global shortage of trained sonographers, creating barriers to essential fetal health monitoring. Deep learning has the potential to enhance sonographers' efficiency and support the training of new practitioners. Vision-Language Models (VLMs) are particularly promising for ultrasound interpretation, as they can jointly process images and text to perform multiple clinical tasks within a single framework. However, despite the expansion of VLMs, no standardized benchmark exists to evaluate their performance in fetal ultrasound imaging. This gap is primarily due to the modality’s challenging nature, operator dependency, and the limited public availability of datasets. To address this gap, we present Fetal-Gauge, the first and largest visual question answering benchmark specifically designed to evaluate VLMs across various fetal ultrasound tasks. Our benchmark comprises over 42,000 images and 93,000 question-answer pairs, spanning anatomical plane identification, visual grounding of anatomical structures, fetal orientation assessment, clinical view conformity, and clinical diagnosis. We systematically evaluate several state-of-the-art VLMs, including general-purpose and medical-specific models, and reveal a substantial performance gap: the best-performing model achieves only 55\% accuracy, far below clinical requirements. Our analysis identifies critical limitations of current VLMs in fetal ultrasound interpretation, highlighting the urgent need for domain-adapted architectures and specialized training approaches. Fetal-Gauge establishes a rigorous foundation for advancing multimodal deep learning in prenatal care and provides a pathway toward addressing global healthcare accessibility challenges. Our benchmark is publicly available at https://github.com/BioMedIA-MBZUAI/FETAL-GAUGE

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper contributes Fetal-Gauge, a large-scale VQA benchmark for fetal ultrasound tasks, by integrating and processing 13 existing datasets. The authors evaluated 15 SOTA VLMs (including general-purpose and medical-specific models) on Fetal-Gauge. The primary finding is that even the best-performing closed-source model (GPT-5) achieved only 55% overall accuracy, while most open-source models performed near random chance. This reveals a substantial performance gap for current VLMs in this highly specialized domain.

### Strengths
1. The paper aggregates and standardizes 13 heterogeneous datasets from different sources, which have varying formats and annotations, into a unified VQA benchmark. It demonstrates a thoughtful design in data partitioning, such as preserving original dataset splits and allocating some datasets entirely to the test set to evaluate generalization.

2. The paper's experimental analysis provides valuable preliminary insights, such as the substantial performance gap of existing VLMs, the critical importance of domain adaptation for future research, and specific model limitations, including poor performance on small anatomical structures and inadequate generalization to phantom images.

### Weaknesses
1. The paper's contribution is primarily an aggregation of existing datasets. The described method of 'Task and Annotation Unification' (Sec 3.2), which converts various annotation types (e.g., image-level labels, segmentation masks, and bounding box annotations) into a unified VQA format, does not demonstrate significant novelty or innovation compared to established data curation practices in the medical imaging domain.

2. The 'Vocabulary Normalization' operation (Sec 3.2) is presented as a potentially important methodological contribution. However, this section is described too briefly, making it difficult to assess its impact. It is unclear which specific datasets are affected and how their vocabularies are unified (e.g., are answer vocabularies from different datasets merged?). Concrete examples would be necessary for comprehension. Furthermore, given that the benchmark uses an MCQ format where all answers are provided as options, the necessity of this normalization step is uncertain. An ablation study is required to empirically validate the importance of this operation.

### Questions
1. Figure 3 shows the distribution of the five task types per anatomical region. Providing an overall distribution of the five task types would seem more intuitive. Could you provide this? 

2. How are the incorrect options (distractors) for the MCQs generated? Are they sampled from an answer pool internal to the source dataset, or are they sampled globally from an answer pool spanning all datasets?

3. For a given MCQ question, is the order of the options fixed, or is it randomized?

4. From which source dataset(s) was the 'Clinical View Conformity (VC)' task derived? 

5. Is the content of Section 4.4 incomplete? The text appears to cut off. 

6. In Figure 3 and Table 1, should the abbreviation 'SV' be 'VC'?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents **Fetal-Gauge**, the first benchmark specifically designed for evaluating **Vision-Language Models (VLMs)** in fetal ultrasound imaging. The benchmark consists of **42,000 ultrasound images** and **93,000 question–answer pairs**, covering five clinically relevant tasks:

1. Anatomical plane identification  
2. Visual grounding of key anatomical structures  
3. Fetal position and orientation assessment  
4. Clinical view conformity evaluation  
5. Clinical diagnosis question answering  

The authors evaluate **15 representative VLMs**, including general-purpose models (e.g., GPT-4V, Gemini, LLaVA) and medical-specific ones (e.g., Med-Flamingo, BioVLM, PMC-VLM). The results show that the best-performing model achieves only **55% accuracy**, which is far below clinical requirements. Through analyses of error types, modality correlation, and task difficulty, the authors reveal major limitations of current multimodal models in handling the complex visual characteristics of fetal ultrasound—such as high noise, frequent occlusion, and operator dependency. This work aims to promote the development of **domain-adapted VLMs for medical imaging** through standardized datasets and evaluation protocols.

### Strengths
1. This work establishes the **first large-scale benchmark** for fetal ultrasound with multi-task and multi-question–answer settings, covering the entire pipeline from basic recognition to diagnostic reasoning.  
2. The dataset contains **42k images and 93k QA pairs**, each accompanied by standardized labeling schemes and metadata for ultrasound view types, ensuring clear task hierarchy and structure.  
3. It evaluates both **general-purpose and medical-specific VLMs** under a unified QA interface, ensuring fair comparison across models.  
4. The authors summarize **failure modes** of current multimodal models—such as boundary ambiguity, structural occlusion, and linguistic ambiguity—which provide valuable insights for developing safer and more reliable medical VLMs.  
5. If the dataset and codes are fully released as planned, it will likely become an **important evaluation standard** for future medical VLM research.

### Weaknesses
1. The paper does not clarify the **data origin**, **IRB approval numbers**, **patient consent process**, or **de-identification pipeline** for the 42k images. Since obstetric data involve pregnant women and fetuses, the authors should describe the ethical approvals and data-sharing protocols either in the main text or the appendix.  
2. The benchmark currently lacks **structured diagnostic**, **multi-view fusion**, and **temporal decision-making** tasks, which limits its representativeness for real-world AI-assisted prenatal screening. It is unclear whether the authors plan to include these components.  
3. There is no quantitative analysis of **error sources** (e.g., modality, gestational stage, or device variation), nor is there an evaluation of confidence calibration, output stability, or safety metrics.  
4. The **clinical evaluation metrics** are limited.

### Questions
1. Please clarify whether the data are collected from **real patients**. If so, provide IRB approval identifiers, the informed consent procedure (prospective/retrospective/waiver), de-identification process, and cross-institutional data-use agreements. (If the data are entirely from public sources, this is not required.)  
2. The highest reported accuracy is only **55%**. Could you elaborate on the **clinical interpretability** and potential **risk-control mechanisms** associated with such performance levels?  
3. Do the authors plan to build **domain-adapted models** on top of this benchmark (e.g., through medical text corpus fine-tuning, image–text alignment enhancement, or multimodal RLHF)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This manuscript introduces Fetal-Gauge, a large, multi-task visual question answering (VQA) benchmark for assessing vision-language models (VLMs) on fetal ultrasound. The benchmark aggregates 13 public datasets into 42,036 images and 93,451 MCQ question–answer pairs spanning five clinically motivated tasks: Anatomical Plane Identification (PI), Clinical View Conformity (VC), Fetal Orientation (FO), Clinical Diagnosis (CD), and Visual Grounding (VG). The authors standardize labels and unify heterogeneous annotations (including converting segmentations to bounding-box-based grounding questions). They enforce patient-wise splits and include a phantom subset (~19k images).

The authors evaluate 15 VLMs (general and medical-specific; plus one commercial model) with accuracy as the main metric, reporting substantial headroom: the best overall model attains ~55% accuracy; most others hover near task-specific random baselines. Fine-tuning two open models markedly improves performance (up to ~85% overall), and analyses highlight weaknesses on phantom data and on small structure grounding.

### Strengths
- The paper addresses an imperative need for a standardized fetal ultrasound VLM benchmark. 
- The benchmark’s size and breadth (five tasks that mirror real clinical workflows) are substantial and thoughtfully justified. The MCQ format prioritizes objective, scalable evaluation and curbs hallucinations.
- Unifying disparate labels/annotations (including mask-to-box conversion) and vocabulary harmonization is valuable for reproducible evaluation across datasets.
- Evaluating 15 models (medical and general) under a unified protocol, plus showing that domain-specific fine-tuning can unlock large gains, gives the community clear baselines and targets.
- The phantom vs. clinical breakdown and bounding-box-size analysis offer actionable insights (e.g., small-structure failures).
- Cleary writing and presentation.

### Weaknesses
- Aggregating 13 datasets raises questions about licenses and usage permissions. Please include a table listing the license, citation, and redistribution policy for each source.
- Although multiple datasets/hospitals are used, please report scanner vendors, probe types, gestational age ranges, and geographic distribution to understand generalizability.
- Converting masks to boxes and asking “what’s in the box?” evaluates recognition, not localization fidelity. Consider adding a pointing game (models output coordinates of bounding box).
- “normal/benign/malignant” in fetal ultrasound is non-trivial; please clarify labeling sources, consensus rules, and whether images are frame-level vs case-level labels.
- Several source datasets include video sweeps; current tasks are frame-centric. Adding temporal subtasks would align with real ultrasound operation and may favor models with spatiotemporal reasoning.
- Multiple typos. Section 4.4 seems to be incomplete.

### Questions
- Is a single-frame image enough for FO or CD tasks? Sonographers may rely on multi-view reasoning, temporal sweep, and 3D relationships.

### Soundness
3

### Presentation
3

### Contribution
3
