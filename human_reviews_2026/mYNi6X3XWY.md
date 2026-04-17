# ViewSpatial-Bench: Evaluating Multi-perspective Spatial Localization in Vision-Language Models

- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
Vision-language models (VLMs) have demonstrated remarkable capabilities in understanding and reasoning about visual content, but significant challenges persist in tasks requiring cross-viewpoint understanding and spatial reasoning. We identify a critical limitation: current VLMs excel primarily at egocentric spatial reasoning (from the camera's perspective) but fail to generalize to allocentric viewpoints when required to adopt another entity's spatial frame of reference. We introduce ViewSpatial-Bench, the most comprehensive benchmark designed specifically for multi-viewpoint spatial localization recognition evaluation across five distinct task types, supported by an automated 3D annotation pipeline that generates precise directional labels. Comprehensive evaluation of diverse VLMs on ViewSpatial-Bench reveals a significant performance disparity: models demonstrate reasonable performance on camera-perspective tasks but exhibit reduced accuracy when reasoning from a human viewpoint. By fine-tuning VLMs on our multi-perspective spatial dataset, we achieve an overall performance improvement of 46.24% across tasks, highlighting the efficacy of our approach. Our work establishes a crucial benchmark for spatial intelligence in embodied AI systems and provides empirical evidence that modeling 3D spatial relationships enhances VLMs' corresponding spatial comprehension capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduce ViewSpatial-Bench, a comprehensive new benchmark with an automated 3D annotation pipeline, to systematically evaluate the capability of multi-perspective spatial localization with five different tasks. The paper also shows by training on the curated large-scale dataset, the model performance on this benchmark improves by 40%, with generalizability into embodied interaction scenarios.

### Strengths
1. Comprehensive benchmark ViewSpatial-Bench covering five different tasks from both camera and human perspectives. 
2. Extensive experiments revealing the common failure of current models in multi-perspective spatial localization and performance improvement with fine-tuning. 
3. The paper is well written.

### Weaknesses
1. Concerns regarding overfitting and generalization. The performance leap of MVSM is suspicious and raises the question of whether the model has learned a generalizable skill of perspective-taking or has simply memorized the patterns in the ViewSpatial-Bench training set. The results on VSI-Bench show a much more modest improvement (+2.37% average), suggesting that the learned skill may not transfer as effectively to an out-of-distribution benchmark. 
2. Comparison with previous work. As shown in Table 2, the difference between ViewSpatial-Bench and SPHERE only lies in 3D-Coord, which seemingly is not used in the experiments. Considering other similar benchmarks such as MindCube, this raises concerns regarding the novelty of the work. In addition, the benchmark only covers in-door scenes (from ScanNet), which is acknowledged by the author at L459. This raises concerns when it comes to outdoor environments. 
3. The paper introduces the "Multi-View Spatial Model (MVSM)", which is not a novel model design, but a VLM fine-tuned on the proposed dataset. This is a bit overclaiming and should be made clearer.

### Questions
See as above in weaknesses. I'm happy to adjust the scores if the author can address the three concerns in the weaknesses.

Format: table title should always appear before the table.

### Soundness
2

### Presentation
3

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
This work leverages two well known datasets (ScanNet and CoCo) to build a new resource focused on perspective taking.  Questions are phrased as from the perspective of individuals in the image (their right vs right side of image).

### Strengths
Clearly we want visual models to have the same abilities as humans -- building representations of the implied 3D scene so that they can reason about relations and perspectives. The work is evaluated on a suite of appropriate models and a model is trained on the task directly.

### Weaknesses
1. See below, I'm unclear on what we learn from the FT-ing experiments 
2. I have a slight concern about the diversity of the data and the domains chosen, both of which are very canonical.  Given that what might be a small amount of training nearly solves this dataset, the longevity of the work is in question and makes the reader wonder what could be done to build a more robust evaluation.

### Questions
- Data generation describes the presence of distractors (also shown in Fig 4), should I interpret evaluation as multiple choice? (random baseline shifts by category)

*Evaluation*
I'm having trouble understanding the training and corresponding evaluation claims.  
- "Applicability of our training methodology" just means that you did SFT in domain, correct? Why is it surprising that this improved performance? 
- You then only perform ZST comparisons for the pretrained models.  Is there a reason that few shot cannot be run to give the model the basic task structure?  
- Related concern is how often the models generated answers that weren't in the set of options? For example "back" or "left" when "back-left" was required
- Can training curves be provided for the fine-tuned model, or additional details on how many samples were required?

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
4

### Summary
This paper introduces ViewSpatial-Bench, a new benchmark for evaluating multi-perspective spatial localization in VLMs across five tasks covering camera-centric and human-centric viewpoints. The authors develop an automated 3D spatial annotation pipeline to construct QA pairs from images in ScanNet and MS-CoCo.  Experiments show that competitive VLM struggle with cross-viewpoint spatial reasoning. To address this, they train a Multi-View Spatial Model on spatially annotated samples, achieving ~46% absolute improvement over the base model. In addition, to validate MVSM's generalization abilities in practical applications, the authors evaluated on VSI-Bench and VSI-App benchmarks and showed improved performance over baselines.

### Strengths
- The paper is logically coherent overall.
- The paper is easy to understand. 
- The research problem is important in the VLM community.
- The proposed MVSM method is effective.

### Weaknesses
As a dataset and benchmark paper, the overall quality control and evaluation rigor fall short of expectations for publication at a top venue like ICLR.

**Insufficient Human Annotation & Lack of Quality Control**
- Only 864 out of 5,712 samples received human annotation, and no inter-annotator agreement or reliability metrics are reported. This is highly concerning, particularly since the authors acknowledge that automated annotation is unreliable for human-perspective tasks. Stating that annotation is “complex” is not an excuse for minimal human involvement—if the majority of samples are not manually verified, the benchmark’s label quality is questionable, and reported model performance numbers may not be trusted. A rigorous benchmark should involve multi-annotator labeling (potentially combined with the automated pipeline), iterative refinement to eliminate systematic annotation errors, and reporting of agreement scores alongside clear annotation protocols. I would be happy to reconsider once the majority of the test samples are carefully annotated. 

**Answer Validity from Images Alone**
- Since questions are generated from metadata rather than from human interpretation of the images, it is unclear whether the image alone always contains sufficient visual information to yield a unique and unambiguous answer. It remains questionable whether a model—or even a human—can reliably answer some questions solely based on the provided image without access to metadata. 

**Problematic or Ambiguous Questions**
- Several questions appear under-specified, or ambiguous. For example, in Figure 9, the prompt “Standing at table, gazing at chair, where should books be?” has the provided answer “front”. However, the desk is pretty large, and depending on where one stands around the table, “right” could be an alternative answer. Such cases indicate inconsistencies in spatial frame-of-reference grounding and suggest the benchmark may introduce artificial or unclear phrasing not aligned with natural human spatial reasoning.

**Lack of Statistical Rigor in Reporting Results**
- Table 2 and Table 3 present accuracy values without confidence intervals, variance, or statistical testing. As this is a benchmark paper, stronger evidence of robustness and significance is needed. Reporting standard deviations or significance tests is essential to support claims of superiority and to ensure results are reliable.

**The Dataset Is Not Realistic**
- The dataset only covers elementary elements of spatial direction (front, back, left, right, etc.). A useful dataset should cover more realistic tasks such as asking the distances and navigation.

### Questions
- The human performance baseline is missing. What is the performance of humans in these tasks? 
- How to ensure that visual information is sufficient and the answer is uniquely determined (without relying on metadata)?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates the limitations of current vision-language models in reasoning about spatial relationships from different viewpoints. While these models handle spatial reasoning from their own (camera-centered) perspective, they often fail to interpret scenes from another entity’s point of view. To study this issue, the authors introduce ViewSpatial-Bench, a benchmark designed to evaluate multi-viewpoint spatial localization through five types of tasks that involve both camera and human perspectives. The benchmark is constructed using an automated 3D annotation pipeline that produces directional labels for a diverse set of images. Experiments on various vision-language models show that performance declines when models are required to reason from non-egocentric perspectives. The authors further fine-tune a model on their dataset, referred to as the Multi-View Spatial Model, which achieves higher accuracy, suggesting that incorporating explicit 3D spatial information can improve perspective-based reasoning.

### Strengths
1. The paper presents a clear and well-motivated problem formulation, identifying the lack of perspective-taking ability in current multimodal systems and linking it convincingly to challenges in embodied AI and human–robot interaction.

2. The proposed ViewSpatial-Bench is a systematically designed benchmark encompassing five tasks that jointly evaluate egocentric and allocentric reasoning. Its integration of automated 3D annotation with human verification provides a scalable and reliable framework for assessing spatial understanding.

3. The experimental evaluation is thorough, covering a range of major VLMs (e.g., GPT-4o, Gemini-2.0) and revealing consistent performance asymmetries between egocentric and allocentric settings. The accompanying analysis offers useful diagnostic insights into model limitations.

4. The fine-tuned Multi-View Spatial Model (MVSM) demonstrates consistent and interpretable improvements across tasks and generalizes to external benchmarks (VSI-Bench, VSI-App). Additional analyses on backbone variation and answer formats support the robustness of the findings.

5. Overall, the paper is well-written and clearly presented, with informative figures, tables, and methodological descriptions that facilitate reproducibility.

### Weaknesses
1. The methodological novelty is limited. While the benchmark is comprehensive, its conceptual basis—evaluating and fine-tuning for 3D spatial reasoning—builds directly on existing work. The MVSM primarily extends a Qwen-VL baseline through additional spatially annotated data rather than introducing new model architectures or explicit 3D reasoning mechanisms.

2. The comparative analysis omits several relevant baselines, including specialized models such as SpatialVLM, SpatialPin, SpatialReasoner, Space-Qwen, and the Gemini 2.5 series, which would strengthen the empirical claims.

3. The paper’s assertion of being the “first comprehensive benchmark” is somewhat overstated, as prior works (e.g., 3DSRBench, SPHERE, All-Angles Bench) already address similar multi-view evaluation objectives. The contribution lies more in the dataset scale and integration than in conceptual originality.

### Questions
Please refer to the weakness

### Soundness
3

### Presentation
3

### Contribution
2
