# A Structured, Tagged, and Localized Visual Question Answering Dataset with Full Sentence Answers and Scene Graphs for Chest X-ray Images

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Visual Question Answering (VQA) enables targeted and context-dependent analysis of medical images, such as chest X-rays (CXRs). However, existing VQA datasets for CXRs are typically constrained by simplistic and brief answer formats, lacking localization annotations (e.g., bounding boxes) and structured tags (e.g., region or radiological finding/disease tags). To address these limitations, we introduce MIMIC-Ext-CXR-QBA (abbr. CXR-QBA), a large-scale CXR VQA dataset derived from MIMIC-CXR, comprising 42 million QA-pairs with multi-granular, multi-part answers, detailed bounding boxes, and structured tags. 
We automatically generated our VQA dataset from scene graphs (also made available), which we constructed using LLM-based information extraction from radiology reports. After automatic quality assessment, we identified 31M pre-training and 7.5M fine-tuning grade QA-pairs, providing the largest and most sophisticated VQA dataset for CXRs to date. Tools for using our dataset and the construction pipeline are available at https://github.com/philip-mueller/mimic-ext-cxr-qba/ .

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes MIMIC-Ext-CXR-QBA (or CXR-QBA), a large VQA datset derived from MIMIC-CXR. Using an automated pipeline—LLM-based information extraction —the authors first construct scene graphs for chest X-rays and then automatically generate question–answer (QA) pairs with multi-part, structured answers, bounding boxes, and structured tags. An automatic quality assessment filters these into ~31.2M pre-training and ~7.5M fine-tuning pairs. The paper also reports expert validation steps, introduces a structured VQA formulation for CXRs, and releases a baseline model and tools.

### Strengths
- If substantiated, a CXR VQA dataset of 31M pre-training and 7.5M fine-tuning grade QA-pairs (w. box-localized, tag-structured), might be valuable to the community
- The dataset construction pipeline might also be helpful.

### Weaknesses
- Reliance on LLM-information extraction and LLM-as-judge without deep radiologist ground truth
- The issue of dataset balance, diversity, and repetition

### Questions
## LLM reliability and bias

> L243-244: More specifically, we use Llama 3.1 8B (Grattafiori et al., 2024) to rate questions and answers by the following five criteria

1. LLM hallucination and judgement bias are well-documented in general / medical LLM-as-judge literature; 8B judges in particular are fragile compared with larger or ensemble judges.

2. The proposed pipeline is based on scene-graph information extraction on general-domain LLaMA-3.1. Why didn’t you use (or at least cross-check against) medically tuned LLMs/IE models? The authors should at least provide a head-to-head on a radiologist-labeled dataset (entity/relation P/R/F1, negation/uncertainty, ...), and justify keeping a general model if it underperforms medical LLMs.

3. If the LLM extractor and LLM judge share architecture/family (Llama 3.1), what safeguards prevent model-family bias or circular validation (the judge favoring its own style)?

## Dataset

4. Generally, the ontology, negation/uncertainty handling, and mapping rules need rigorous specification to be reproducible and comparable to RadGraph and ImaGenome schemas. What are the macro/micro P/R/F1 of your LLM-IE compared with RadGraph baselines? Where do errors concentrate?

> Figure 5, Distribution of tags (finding subcategories, regions, findings) mentioned in answers of different question types (indication, study abnormality, region abnormality, finding). 

5. CXR is highly long-tailed; without balancing and leakage controls, models might learn shortcuts and inflate metrics. 

    5.1 In some of these questions, the question types have all positive or all negative (e.g., in  region abnormality or finding). How do the authors solve the issue of imbalanced dataset? 

    5.2 Additionally, the authors should also disclose patient/study/report-level splits to rule out report -> QA leakage and measure sensitivity to template priors. 

6. Large auto-generated corpora often contain heavy redundancy that distorts training and evaluation. What fraction of QAs are near-duplicates (by text, tags, question types, and box layout)? The authors should provide more about the dedup pipeline (e.g., MinHash/SimHash + ANN, or semantic-hashing) and show how dedup affects baseline performance and diversity.

>  For comparison, we thus select MAIRA-2 (Bannur et al., 2024), the only publicly available CXR report generation model that supports bounding box prediction. 

7. Beyond MAIRA-2, how does the structured-VQA model transfer to other independent Med-VQA/generation datasets and phrase-grounding tasks without template leakage? E.g., results on some common medical benchmarks (e.g., VQA-RAD, SLAKE, and PathVQA)?

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
This paper proposes CXR-QBA, a new large-scale VQA dataset for chest X-rays. Based on MIMIC-CXR, the authors employed a three-stage automated pipeline: (1) scene-graph construction (including an information extraction model, an embedding model, and localization models), (2) template-based QA generation, and (3) automated quality assessment (mainly for deciding whether to use samples for fine-tuning) using a judge model (Llama-3.1-8B). Through this pipeline, the authors generated a total of 42.2M QBA (question, bounding box, answer) pairs. The main contribution is richly detailed automatic annotations for MIMIC-CXR at large scale, which might be better than the Chest ImaGenome dataset.

### Strengths
- The dataset’s scale is a massive contribution, with 42.2M pairs, including a 7.5M high-quality fine-tuning subset. The answer details are a key strength, providing rich, full-sentence responses derived from reports.
- The three-stage automated pipeline is a significant contribution in itself.
- The authors quantitatively benchmarked their scene graphs against the existing Chest ImaGenome dataset. They demonstrated that their pipeline produces superior results, notably a 20% performance improvement on long-tail (rare) disease classes, proving the value of their extraction method.
- The paper’s presentation is excellent. It is clearly written, logically structured, and uses numerous figures and tables to effectively explain the complex pipeline, dataset, and results.

### Weaknesses
* While the paper claims “detailed” or “fine-grained” localization, its bounding boxes correspond to 257 broad anatomical structures, not the pathologies themselves.
* The “Grade B” (pre-training) data accounts for 58.8% (24.8M) of the dataset but is defined by a permissive standard (e.g., “may have issues”). Critically, the paper lacks an ablation study to prove that this massive, known-to-be-flawed data is actually beneficial for pre-training. The provided ablations (Table 14) only test task formulation, not the impact of data quality grades (A vs. B).
* The pipeline lacks validation for individual stages, making it vulnerable to error propagation. Specifically: (a) the segmentation accuracy of the foundational CXAS model (for 257 regions) is not reported; (b) the information extraction performance of Llama-3.1-70B is not quantified (e.g., F1 score); (c) or benchmarked against RadGraph or Chest ImaGenome by using their gold standard dataset.

### Questions
Q1 (Regarding the utility of “Grade B” data): 
The “Grade B” (pre-training) data comprises 58.8% (24.8M) of the dataset, and proving its utility is essential. Could you provide an ablation study comparing the baseline model’s (Sec. 5) performance under these three conditions:
  * (A) Fine-tuned on “Grade A” data only.
  * (B) Pre-trained on “Grade B” data, then fine-tuned on “Grade A” data.
  * (C) Pre-trained on the full (A+B+…) dataset, then fine-tuned on “Grade A” data.

Q2 (Regarding the Validation of Scene Graph Accuracy): 
The pipeline depends heavily on the Scene Graph Construction step, yet that step lacks independent validation. While benchmarks against MIMIC-CXR-JPG Test, CXR-LT, MS-CXR, and REFLACX are useful, they cover only overlapping, smaller scopes. Given the expanded coverage (257 regions, 221 findings), was a small internal gold standard (e.g., radiologist-annotated) created to validate factual accuracy of scene-graph outputs? If so, please report:
- (a) Segmentation accuracy of the 158 CXAS masks (e.g., Dice).
- (b) IE performance of Llama-3.1-70B (e.g., F1) against this internal gold standard.

Additionally, could you provide component-level benchmarking results against the public gold-standard datasets from RadGraph and Chest ImaGenome?

### Soundness
3

### Presentation
4

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
This paper introduces MIMIC-Ext-CXR-QBA (CXR-QBA), a large-scale VQA dataset for CXRs, derived from the MIMIC-CXR dataset. It comprises 42 million automatically generated question-answer pairs with multi-part, multi-granular answers styled like radiology reports. Each study includes detailed annotations, such as bounding boxes and structured tags (findings, regions). The dataset was constructed using an automatic scene graph construction pipeline, which leverages LLMs, semantic entity mapping, and localization models. The authors also define a structured VQA task, propose evaluation metrics, and provide a baseline model to demonstrate the utility of the dataset and guide future research.

### Strengths
1. Introduces MIMIC-Ext-CXR-QBA, a large-scale CXR-VQA dataset with 42 million QA pairs.
2. Provides rich and detailed answers in radiology report style, along with fine-grained annotations including bounding boxes and structured tags.
3. Implements robust quality control, including validation against expert annotations and assessment of automatically generated outputs.
4. Proposes a structured VQA task for CXRs and provides a baseline model to showcase dataset effectiveness and support further model development.

### Weaknesses
1. The dataset is automatically generated using large language models (LLMs). While the authors employ various quality control, there remains the potential for subtle errors or inconsistencies that may not be present in datasets fully curated by human experts. In particular, it is unclear whether the LLaMA-8B model provides sufficiently reliable quality assessment compared to larger, more capable LLMs.
2. Since the dataset is derived from MIMIC-CXR, it likely inherits demographic and clinical biases inherent to that specific patient population, which may limit the generalizability of models trained on this dataset.
3. The baseline model (MAIRA-2) is fundamentally a radiology report generation model, rather than a specialized VQA model or medical foundation model. This raises questions about the appropriateness of the baseline for evaluating performance on a VQA-specific task, as it may not accurately reflect the potential or limitations of models explicitly designed for structured question answering.
4. The evaluation process relies on the LLaMA-8B model to determine entailment in the RadStrucVQA metric. Given the model’s relatively small size, there are reasonable concerns about the accuracy and robustness of its entailment judgments.

### Questions
1. Are bounding boxes derived from CXAS masks always higher in quality than those from Chest ImaGenome? Given that both sources are used in the dataset, what criteria determine when one is preferred over the other?
2. What are the criteria for defining Parent Regions and Fusion Regions?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces CXR-QBA, a very large chest X-ray VQA dataset built from MIMIC-CXR that contains 42.2M QA pairs with full-sentence, multi-part answers, bounding boxes for grounding, and structured tags for findings and regions. The authors construct visually grounded scene graphs from radiology reports using LLM-based information extraction, semantic entity mapping, and region localization, then generate questions and answers from these graphs. They automatically rate QA quality with an LLM judge and release two subsets: pre-training grade (31.2M) and fine-tuning grade (7.5M). They also propose a structured VQA task that requires models to output text, boxes, and tags, and report a baseline LLaVA-style model with Rad-DINO and Llama-3.2-3B that outperforms an adapted MAIRA-2 reporting model on their new metric, RadStrucVQA.

### Strengths
- The dataset is orders of magnitude larger than prior CXR VQA resources and includes sentence-level answers, localization, and detailed tags.
- The scene-graph construction is well described, including CXAS-based region masks, Chest ImaGenome boxes, LLM extraction, and BioLORD-based concept mapping, followed by LLM-as-judge quality assessment and graded splits.
- The authors compare derived tags and boxes to radiologist gold standards and public annotations.
- The structured VQA task and RadStrucVQA metric encourage outputs that are logically correct and visually grounded with tags.

### Weaknesses
- I am not sure how reliable the LLM-as-a-judge quality assessment is when the judge is Llama 3.1 8B. The paper should calibrate this with stronger judges and report robustness checks like cross-judge agreement.
- There are plenty of recent multimodal LLMs that could serve as reference points. Incorporating strong open-source baselines such as Qwen3-VL or LLaVA-Med v1.5 would better situate this work relative to current progress.
- Human–LLM agreement is only fair to moderate.
- The authors mention co-design with radiology collaborators, but they do not clearly report the number of participants, and their roles are vaguely described.

### Questions
- Does the pre-training grade include only B-rated samples, or does it also include A and A+?

### Soundness
4

### Presentation
4

### Contribution
3
