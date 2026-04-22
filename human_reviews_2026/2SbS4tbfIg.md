# OODBench: Out-of-Distribution Benchmark for Large Vision-Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Existing Visual-Language Models (VLMs) have achieved significant progress by being trained on massive amounts of data and with the assumption that the data are Independent and Identically Distributed, IID. However, we must embrace the fact that, in real-world scenarios, it is not only difficult but also impractical to expect that all data processed by an AI system would satisfy this assumption. Furthermore, if an AI system chooses to ignore out-of-distribution, OOD, objects during processing, it may cause safety hazards or even lead to catastrophic consequences in real-world applications (e.g., autonomous driving, medical assistance scenarios, etc.). Unfortunately, current research has not yet provided valid benchmarks that can comprehensively assess the performance of VLMs in response to OOD data. Therefore, we propose OODBench, a predominantly automated method with minimal human verification, for constructing new benchmarks and evaluating the ability of VLMs to process OOD data. OODBench contains 40K instance-level OOD instance category pairs, and we also show that VLMs still struggle to process natural image categories in OODBench, despite those categories are common. In addition, we propose a reliable automated assessment metric that employs a Basic-to-Advanced Progression of prompted questions to assess the impact of OOD data on questions of varying difficulty more fully. Lastly, we summarize substantial findings and insights to facilitate future research in the acquisition and evaluation of OOD data. The dataset is open to the public for research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces OODBench, a novel and automated benchmark designed to evaluate the performance of large Vision-Language Models (VLMs) on out-of-distribution (OOD) data. The authors propose an effective pipeline to collect 40,000 instance-level OOD pairs by using robust classifiers as OOD detectors. The submission also presents a multi-faceted evaluation metric, the Basic-to-Advanced Progression (BAP), to probe model capabilities in recognition, counting, and reasoning. Through extensive experiments, the paper compellingly demonstrates that even state-of-the-art VLMs suffer significant performance degradation on OODBench, highlighting a fundamental challenge in model generalization.

### Strengths
1. The paper tackles the crucial issue of OOD robustness in VLMs, an area vital for real-world applications yet relatively underexplored. It successfully moves beyond simple misclassifications to create a more nuanced benchmark for generative vision-language tasks.

2. The proposed automated pipeline for OOD data collection is a significant strength. The use of multiple detectors in a cross-validation-style approach is a principled method to mitigate the biases of any single detector, ensuring the curated dataset is a more generalized and reliable testbed for OOD performance.

3. The authors conduct a thorough evaluation across a wide and relevant selection of eight VLMs, including recent state-of-the-art models. This broad analysis ensures the findings are generalizable and provides a valuable snapshot of the current landscape of VLM robustness.

### Weaknesses
1. While the quantitative results are strong, the paper would be significantly enhanced by a deeper qualitative analysis. Including visual examples of the OOD samples and discussing common failure patterns would provide the community with more intuitive insights into why these models fail.

2. The evaluation focuses on a diverse set of models but misses the opportunity to analyze the effect of model scale on OOD robustness. Including experiments on a model family with varying parameter counts (e.g., small, medium, and large versions of the same architecture) would be highly insightful. 

3. The paper's experimental results strongly differentiate OOD from hallucination. However, the qualitative examples presented in the paper (e.g., Figure 4) are paradigmatic cases used to test for hallucination.

4. Discussing the potential boundaries of the OODBench framework—for instance, the types of distributional shifts it primarily captures or its reliance on existing classifiers—would provide a more complete and balanced perspective on the work.

### Questions
see above

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
### Method
- The authors discuss the definition of OOD data for VLMs, and they define two types of OOD data:
    - Objects in images that are neither main objects nor semantically related to the main semantic object;
    - Variants or anomalous forms of target objects.

- Second, they introduce OODBench, a new OOD benchmark for VLMs.
    - They create an automated and efficient OOD data division pipeline with minimal human verification.
    - Along with OODBench, they propose the Basic-to-Advanced Progression Metric
### Results
- Based on OODBench, the authors evaluate SoTA VLMs, and show that current models still struggle with OOD data.

### Strengths
- This study explores an interesting question on OOD definition for VLMs.
- They introduce a new benchmark and show insights into current VLMs.

### Weaknesses
### Major
- Definition of OOD data for VLMs. The authors define two categories: (1) Objects in images that are neither main objects nor semantically related to the main semantic object;  (2) Variants or anomalous forms of target objects.
    - However, it is difficult to assert that these data types are truly “OOD” for current VLMs. These VLMs are highly likely trained on datasets that already contain such cases to improve their robustness.
    - Second, the definition of OOD is somehow ambiguous. For example, how to tell which objects are the main ones in (1), and what form is anomalous in (2).  In prior OOD settings, class labels are used to clearly separate ID vs. OOD, while the current definition relies on subjective judgments, making it difficult to apply consistently.
    - A possible case is that the data is not OOD but just harder than normal questions.

- Could the authors explain more about how they do the one-shot overlap rate experiments (lines 372-399)?
    - I'm wondering how to use VLMs as model-specific detectors to flag their OOD samples?
    - Are VLM reliable to flag their own OOD samples? Do the authors validate this method on some open-source models like LLaVA?

- Could the authors show some insights or opinions on how to solve the proposed OOD problem?
   - Did the authors try some methods to resolve the OOD problem? Like finetuning the models on the data, changing prompts, or using chain-of-thought? 

### Minor
- Some sections of the paper seem slightly redundant. For instance, almost half of the abstract explains the importance of OOD in VLMs, and there is repeated content in the introduction. It may be helpful to refine the structure for better clarity and conciseness.
- The format of citations is not correct. Please try to distinguish between \citep and \citet.
- The format of tables can be improved. For example, the left part of Table 1 is sparse, while the right part is very dense, making it hard to tell the numbers.

### Questions
Please see the Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes OODBench, an instance-level OOD benchmark for VLMs targeting covariate shift. It uses multiple generalized classifiers (e.g., CLIP, BLIP2) with a “purify” step and cross-validation to label OOD-S (symmetric difference) and OOD-H (intersection), with light human verification. Data spans COCO/LVIS (natural) and nuScenes/Cityscapes (driving).
A Basic-to-Advanced Progression (BAP) metric evaluates existential, counting, and logical comparison questions.
Extensive experiments across open/closed models show large performance drops on OOD-H vs. ID and mixed effects of CoT prompting. The paper also differentiates OODBench from hard samples and hallucination via overlap, variance/correlation analyses, and CoT effects.

### Strengths
1. Timely problem with safety relevance: focuses on covariate shifts common in real-world deployment (especially driving).
2. Mostly automated, reproducible pipeline with reasonable cross-detector validation and public release.
3. BAP design offers layered diagnostics (recognition → counting → logic).
4. Broad, careful experimentation and ablations (detector types/number, threshold T); analysis distinguishing OOD vs. hard samples/hallucination is thoughtful.

### Weaknesses
1. Conceptual clarity: The paper claims a focus on covariate shift but operationally defines OOD as “misclassified/low-confidence” by generalized detectors. Are misclassified samples assumed equivalent to covariate-shifted samples? Misclassification can stem from ID hard cases, label noise, ambiguity, or prompt wording. Please clarify the formal relationship and provide quantitative checks (e.g., factors like scale/occlusion/illumination) to support covariate-shift attribution.
2. Task-level distinction from hallucination: The yes/no queries (“Does this image contain a truck?”) resemble hallucination benchmarks. Although the appendix argues differences (e.g., CoT helps hallucination but not OOD here), clearer, earlier, and more direct contrasts in the main paper (definitions, data construction, and quantitative comparisons) would avoid confusion.
3. Detector dependence and external validity: While detector replaceability is tested, the approach still inherits biases from detector vocabularies/prompts. Sensitivity to text templates/synonyms and guidance for out-of-domain transfer (e.g., medical/remote sensing) are limited. A brief protocol or pilot in a specialized domain would strengthen generality claims.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work addresses the lack of a comprehensive benchmark to evaluate the performance of Vision-Language Models (VLMs) when dealing with out-of-distribution (OOD) data.

The paper hypothesizes that existing VLMs are primarily trained on the most common classes of visual data. Based on this assumption, the authors define two types of OOD data from a human perception perspective:

1. Objects in images that are neither main objects nor semantically related to the main semantic object.

2. Variants or anomalous forms of target objects.

To operationalize this, the paper proposes a succinct and efficient OOD data division process and introduces OODBench — a benchmark for evaluating VLMs. OODBench includes:

An automatically constructed OOD dataset.

1. A VLM-oriented OOD data division pipeline.

2. A Basic-to-Advanced Progression (BAP) Metric for evaluation.

3. A complete experimental setup for testing multiple state-of-the-art VLMs.

### Strengths
1. The pipeline of this paper is clear and easy to be understand.

2. This work proposes that VLMs has seen normal patterns and various instances and it proposes that VLMs should has a new descriptions which is reasonable.

3. This work proposes a new framework in generating data.

4. The bad performance of various VLMs proposes the hard problems of OODbench. Basic-to-Advanced Progression is reasonable for current proposes of LLM-based VLM.

### Weaknesses
1. The core idea of OOD benchmark in to evaluting whether VLM can understand the images in In-or-Out training distribution. This work aims in find the images making the VLMs wrongly predict which is not reasonable.
2. This work proposes a BAP evaluating metric but it doese not show in the the first table.
3. In constructing the OODBench, the pipeline is about reducing human cost but there is still human labor involed which is non-reasonable.
4. For the pipeline in OOD collection, it finds the object which makes the base VLMs (CLIP, BLIP-2) wrongly predict. Thus, the OOD data constructed is hard to tell which contributes the bad performance of VLLMs in benchmark. 
5. In generating OOD data with pipeline he proposed, there is a clear bottleneck because of the VLMs used in Fig.3.

### Questions
The propose directions of semantic shift and covariate shift is not orginally proposed in your paper while there lack of citation.

Writing problem in line 67-71, which is not clear about "unlike".

In Table1 for evaluating performance, there is division of "Open-source Models" but there is no private model.

Random Chance in Table 1 is non-sense and there is extra line at the top for all the tables.

In the tables, there are some models that performs worse than random guess which worth exploration which makes the performance less reasonable.

### Soundness
2

### Presentation
2

### Contribution
3
