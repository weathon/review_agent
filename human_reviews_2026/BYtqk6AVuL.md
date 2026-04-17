# MedLesionVQA: A Multimodal Benchmark Emulating Clinical Visual Diagnosis for Body Surface Health

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Body-surface health conditions, spanning diverse clinical departments, represent some of
the most frequent diagnostic scenarios and a primary target for medical multimodal large
language models (MLLMs). Yet existing medical benchmarks are either built from publicly
available sources with limited expert curation or focus narrowly on disease classification,
failing to reflect the stepwise recognition and reasoning processes physicians follow in
real practice. To address this gap, we introduce MedLesionVQA, the first benchmark
explicitly designed to evaluate MLLMs on the visual diagnostic workflow for body-surface
conditions in large scale. All questions are derived from authentic clinical visual diagnosis
scenarios and verified by medical experts with over 20 years of experience, while the
data are drawn from 10k+ real patient visits, ensuring authenticity, clinical reality and
diversity. MedLesionVQA consists of 12K in-house images (never publicly leaked) and
19K expert-verified question–answer pairs, with fine-grained annotations of 94 lesion types,
110 body regions, and 96 diseases. We evaluate 20+ state-of-the-art MLLMs against
human physicians: the best model reaches 56.2% accuracy, far below primary physicians
(61.4%) and senior specialists (73.2%). These results expose the persistent gap between
MLLMs and clinical expertise, underscoring the need for the multimodal benchmarks to
drive trustworthy medical AI. The dataset can be found in https://github.com/bytedance/MedLesionVQA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel benchmark dataset for evaluating the capabilities of multimodal LLMs on diagnosing body-surface conditions. Data has been collected from in-house volunteers and the derived question-answer pairs are reviewed by medical professionals. The paper evaluates state-of-the-art models, including both proprietary and open-source models, and demonstrates that current AI techniques fall far behind senior professionals on diagnosing body-surface conditions.

### Strengths
- This work contributes valuable new data collected in-house that can be leveraged to fairly evaluate existing techniques as it is guaranteed that such models are not trained on the dataset. 

- The benchmark is generated based on physician annotated data, and all QA pairs are manually reviewed. This provides solid guarantees on its reliability and correctness.

- The paper evaluates human performance on the benchmark as well, thus we can gauge the gap between existing models and medical professionals.

### Weaknesses
- It is unclear if the physicians participating in annotation and QA verification overlap with the medical professionals asked to solve the benchmark for human-performance baseline. If there is an overlap, then I am doubtful about the methodology as the human performance may be inflated.

- The part about automatic scoring needs more clarification. For multiple-choice evaluation, why cannot models simply output the selected answer option, which is then compared to the ground truth?

- It would be great to have more analysis on *how* the models fail on the benchmark, leading to a discussion on how to improve current models to close the gap with senior medical professionals.

Minor:
- Typos on lines 198, 392.
- The Appendix is missing.

### Questions
- Has there been an overlap between the experts creating the dataset and those solving the benchmark for the reported numbers?
- Have authors observed consistent failure modes on the benchmark? If so, what is the take-away in terms of future directions?

### Soundness
3

### Presentation
3

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
The paper introduces MedLesionVQA, a multimodal benchmark aimed at emulating the real clinical visual diagnostic workflow for body-surface health (dermatology, dentistry, surgery). It comprises ~12K in-house volunteer images and ~19K expert-verified QA pairs, annotated across 94 lesion types, 96 diseases, and over 110 body regions, and organized into seven stepwise abilities: lesion recognition, attribute recognition, location recognition, spatial relation, lesion reasoning, disease diagnosis, and suggestion & treatment. The authors propose an automated scoring pipeline for MCQ and open-ended items (with a judge-LLM) and compare over 20 MLLMs against primary and senior physician baselines, showing a persistent performance gap.

### Strengths
- The paper provides a comprehensive benchmark of over 20 general-domain MLLMs and physicians over fine-grained body-surface VQA tasks, with concrete findings on the specialized capabilities of these models.
- The dataset is of good quality with manual quality check, covering a broad range of cases, question types, body-region ontologies, and lesion attributes, supporting comprehensive assessment.
- The writing highlights practical insights that can guide future model design and training.

### Weaknesses
**1. About the queries solvable by text-only LLM**

I am curious why the authors do not drop VQA items that can be easily answered by a text-only LLM, which indicates that the question itself can be hacked without using the visual evidence and should not be included for VQA benchmarking. This is a common practice to avoid language shortcuts in curating VQA benchmarks [1--3].

[1] MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark (ACL 2025)

[2] PMC-VQA: Development of a large-scale medical visual question-answering dataset (Communication Medicine 2024)

[3] MicroVQA: A Multimodal Reasoning Benchmark for Microscopy-Based Scientific Research (CVPR 2025)


**2. About evaluation of medical-domain models**

Currently the benchmark only assesses general-domain MLLMs. It would provide more insights if including medical-domain [4-6] or dermatology-specialized MLLMs [7]. 

[4] MedGemma Technical Report (arxiv 2025)

[5] HuatuoGPT-vision, towards injecting medical visual knowledge into multimodal llms at scale (EMNLP 2024)

[6] GMAI-VL & GMAI-VL-5.5M: A Large Vision-Language Model and A Comprehensive Multimodal Dataset Towards General Medical AI (arxiv 2024)

[7] Pre-trained Multimodal Large Language Model Enhances Dermatological Diagnosis using SkinGPT-4 (Nature Communications, 2024)


**3. About more details** 
- The paper notes large gaps in visually demanding tasks. I am wondering if there are per-ability error taxonomies for physicians versus models. That would sharpen which capabilities (e.g., boundary/size/color perception vs. region localization) most need architectural or training changes.
- The authors mention that they use prompt tweaks for the open-ended scoring based on GPT-4 after observing disagreements (e.g., color/size strictness). This introduces circularity and instability: results hinge on a proprietary model’s behavior and prompt details. There seems no rigorous inter-rater reliability report (e.g., Cohen’s κ) between physicians and the judge-LLM. It seems that the authors did not upload the appendix; there is no content at the end of the PDF nor supplementary material.


**4. Discussion with highly related works is missing**. For example, DermVQA [8] and Derm1M [9].

[8] DermaVQA: A Multilingual Visual Question Answering Dataset for Dermatology (MICCAI 2024)

[9] Derm1M: A Million‑Scale Vision‑Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology (ICCV 2025)


**5. Question on the LLM-based scoring**

The use of commercial proprietary LLMs (GPT-4) might entail model/version drift risk since neither the API nor any specific snapshot is guaranteed to be available forever. Given this, I am wondering whether open-source models (e.g. Qwen series) are able to serve this and how the evaluation results will vary (e.g., will the scores differ drastically, or will there be any bias?). 

---

Minor issues: 
- The “multiple-choice question” (selecting one single answer from multiple choices) in Line 277 seems to be “multiple-response question” (selecting all correct answers).
- The first sentence in the Introduction section is a strong assertion (“Taking a photo and consulting multimodal large language models has become a main approach for addressing body surface health concerns”). I am not sure if this is the case for the global world. Any references/statistics to support such solution as “a main approach”?
- Most of the references in this manuscript should use “\citep” with parentheses to avoid confusion with the main texts.
- In Figure 1, upper right part, “Leision Recognition” → “Lesion Recognition”.

### Questions
Will the dataset be made publicly available?

### Soundness
3

### Presentation
3

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
This paper introduces MedLesionVQA, a multimodal benchmark designed to test how well AI systems can perform medical visual diagnosis for body surface conditions such as skin, oral, and nail diseases. It includes over 12k volunteer images and VQA pairs that cover seven diagnostic steps including lesion recognition to treatment recommendation. The dataset is curated with annotations across several lesion types, diseases, and body regions, reflecting real clinical workflows validated by clinical experts. The authors evaluate around 20 MLLMs and find that the best model achieves 56.2% accuracy, still well below physician performance. This work provides interesting insights evaluating MLLMs for body surface diagnosis.

### Strengths
- The authors experimentally show limitations of current MLLMs by comparing them against human physicians across realistic diagnostic tasks, providing quantitative evidence that even the best models are behind expert clinicians.
- The paper offers valuable empirical analysis on scaling and specialization trade-offs, showing through systematic evaluation that while larger models generally perform better, domain specific fine-tuning can sometimes reduce performance.

### Weaknesses
- The paper does not disclose information about the diversity of participants involved in the data collection process. Metrics such as the country of origin and age groups of participants would help assess whether the benchmark is truly diverse or if its biased to any particular demographic or target group. 
- Need to include comparisons about other Medical MLLMs like – Google’s MedGemma [1], Huatuogpt [2], Bimedix2 [3]. This would help us understand if domain specific medical instruction tuning affects the model’s generalizability and how these models compare on MedLesionVQA.
- The Appendix and the supplementary material section are missing from the submission. The authors make multiple references to it in the main paper.  Detailed prompts for the evaluation framework are missing. 
- In Figure 1 Suggestion and Treatment section “list at least two topical anti-infective drugs?”. Questions like these would help models respond without properly analyzing the image. This reduces the credibility of the benchmark. 

[1] *Sellergren, Andrew, et al. "Medgemma technical report." arXiv preprint arXiv:2507.05201 (2025).*

[2] *Chen, Junying, et al. "Huatuogpt-vision, towards injecting medical visual knowledge into multimodal llms at scale." arXiv preprint arXiv:2406.19280 (2024).*

[3] *Mullappilly, Sahal Shaji, et al. "Bimedix2: Bio-medical expert lmm for diverse medical modalities." arXiv preprint arXiv:2412.07769 (2024).*

### Questions
Please address the above weaknesses. 
- What instructions were given to the clinical experts for verification of the samples?
- Please include the clinical lexicon tree in the appendix.

### Soundness
3

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
2

### Summary
This paper introduces MedLesionVQA, a multimodal benchmark explicitly designed to evaluate multimodal large language models (MLLMs) in body-surface health diagnosis. Unlike existing datasets focusing on isolated disease classification or lesion recognition, MedLesionVQA covers  seven clinical ability dimensions.
The dataset includes 12K ethically collected volunteer images and 19K expert-verified question–answer pairs, covering 94 lesion types, 96 diseases, and 110 body regions. Over 20 MLLMs, including GPT-5, Gemini 2.5 Pro, and Qwen2.5-VL-72B, are evaluated, alongside human baselines. Results show that the best-performing MLLM (Gemini 2.5 Pro) achieves 56.2% accuracy, lagging behind primary physicians (61.4%) and senior physicians (73.2%), indicating persistent gaps between current MLLMs and clinical reasoning capabilities.

### Strengths
The benchmark is well-motivated and methodologically thorough, with a clear focus on aligning AI evaluation with authentic clinical workflows. The data collection process is commendably rigorous: recruiting real volunteers under ethical review and performing multi-stage expert validation with high inter-annotator reliability. The seven diagnostic categories mirror the stepwise reasoning of real physicians, offering a structured and interpretable assessment framework. The inclusion of both open- and closed-source MLLMs, together with human baselines, provides a well-rounded evaluation. The paper is clearly written, integrates comprehensive error analysis, and successfully highlights key challenges in general-purpose MLLMs’ ability to handle fine-grained lesion understanding and multimodal reasoning.

### Weaknesses
The benchmark’s contribution primarily lies in data curation and clinical alignment rather than introducing a new evaluation paradigm. 
The evaluation setup, though comprehensive, leans heavily on performance reporting without deeper interpretability or diagnostic insights. For instance, while the authors provide category-wise breakdowns, the analysis stops short of identifying failure modes beyond recognition vs. reasoning. The results are descriptive but not analytical: there is little exploration of why models fail (e.g., domain shift, visual ambiguity, prompt misalignment) or how these findings could inform model development.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3
