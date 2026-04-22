# CL2GEC: A Multi-Discipline Benchmark for Continual Learning in Chinese Literature Grammatical Error Correction

- Avg Score: 2.80
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 2, 6

## Abstract
The growing demand for automated writing assistance in scientific domains highlights the need for robust Chinese Grammatical Error Correction (CGEC) systems that can adapt across disciplines. However, existing CGEC research lacks dedicated benchmarks for academic writing and overlooks continual learning as a solution to handle domain-specific variation. To fill this gap, we introduce CL2 GEC, a Continual Learning benchmark for Chinese Literature Grammatical Error Correction, designed to evaluate adaptive CGEC across multiple academic fields. Our benchmark includes 10,000 human-annotated sentences spanning 10 disciplines, each exhibiting distinct linguistic styles and error patterns. We evaluate large language models under sequential tuning, parameter-efficient adaptation, and representative continual learning strategies, using both standard GEC metrics and continual learning metrics adapted to task-level variation. Experimental results show that regularization-based continual learning methods, such as OGD and GEM, outperform replay-based and sequential approaches in both grammatical accuracy and knowledge retention. These findings underscore the feasibility and importance of integrating continual learning into CGEC and position our benchmark as a foundation for future research on adaptive scientific writing assistance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces CL2GEC, the first benchmark and evaluation suite for continual learning in Chinese Grammatical Error Correction (CGEC) across 10 academic disciplines. The benchmark consists of 10,000 manually curated and annotated sentences, each associated with up to three human references, drawn from diverse domains such as law, science, and art. CL²GEC enables controlled assessment of catastrophic forgetting and cross-domain transfer using both standard GEC metrics and continual learning metrics adapted for task-sequence settings. The paper provides extensive empirical results for large language models (LLMs) under various adaptation and continual learning strategies, including sequential fine-tuning, LoRA, replay-based, and regularization-based continual learning methods.

### Strengths
**Comprehensive Academic Chinese GEC Benchmark**: The CL²GEC dataset is not only substantial in size (10,000 sentences) but also features rigorous multi-stage curation, leveraging both automatic detectors and in-domain human experts. The annotated data covered multiple research fields, which ensure the diversity of the benchmark.

**First Chinese GEC benchmark for continue learning**: To the best of my knowledge, this is the first work that study the scenario of continue learning for Chinese GEC. **However**, the target of continue learning for such a GEC task is ambiguous.

**Rich Experimental Settings**: The authors provide detailed experimental setups, including both random and semantically informed task sequences, ablation on replay buffer size, and systematic results across multiple LLMs and adaptation strategies.
In-depth Evaluation and Metrics: The use of both standard GEC metrics (Precision, Recall, F₀.₅) and continual learning-specific metrics (Backward Transfer, Average Task Performance) brings nuance to the empirical analysis. Equation formalizations are transparent and align with continual learning literature (see Page 4, loss and optimization formulations).

### Weaknesses
**Missing Annotation Principles**: The paper did not describe the annotation principles for GEC task. The grammatical error can be corrected in multiple ways. In most previous works, the principle of minimal edit is applied in the data annotation. Without annotation principles or annotation guidelines, the quality and the consistency of labeled data can not be ensured.

**Problematic Data Filtering** In section 3.2 Data Annotation, the authors described "Only sentences flagged consistently by all  6 grammatical error detectors are kept".  However, the sentences that are judged as erroneous sentences by all models are simple samples, while the difficult samples may be excluded by such a principle. This will significantly influence the distribution of the benchmark.

**Limited Research Significance for Continue Learning**: GEC task is not a long-context task and the dataset has only 10000 samples in total, which means the train cost on CL2GEC is not that unaffordable. Why not just shuffle the dataset randomly and directly train the model using all training data? The application of continue learning seems meaningless in such a task.

**Limited Theoretical Insights into Catastrophic Forgetting**: While the empirical results are substantial, there is little in-depth mathematical or theoretical exposition regarding why certain CL strategies succeed or fail in this linguistic, multi-domain context. For example, the paper lacks a formal analysis of error distribution shift or domain overlap between disciplines.

**Ambiguous Treatment of Semantic Task Ordering**: The computation of semantic similarity (Appendix A.1.2) is described, but the similarity is a metric for every pair of sub-dataset. The paper did not describe how to organize them into a absolute sorted list. From the Appendix A.1.2, I cannot know the specific rules for the 3 groups and how to sort them.

**Typos**: In line 462, there is reference error (Figure ??)

### Questions
- Could you please provide the annotation guidelines?
- How do you ensure the data quality?
- Could you please clarify the purpose of continue learning in GEC task? Can the continue learning reach a higher performance than direct training using all data?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents the CL^2GEC dataset, which contains 10,000 high-quality, human-annotated samples for grammatical error correction, featuring a diverse range of error patterns. In addition, the paper conducts continual learning experiments on the CL^2GEC dataset and establish solid baselines for future work.

### Strengths
1. A large-scale, high-quality GEC dataset has been constructed.

2. Comprehensive analytical experiments are conducted to evaluate the performance of existing models.

### Weaknesses
1. The motivation for applying continual learning (CL) in the GEC domain is not clearly justified. Since grammatical errors across different domains of the same language share certain common patterns, it is unclear whether adopting a CL framework provides practical value.

2. As GEC dataset annotation is inherently challenging, the paper lacks consistency metrics to demonstrate the reliability of human annotations.

### Questions
1. As mentioned above, considering the commonality of error patterns in GEC, the necessity of CL in this domain still requires further validation. Adding a full-data fine-tuning result in Table 2 would make the argument more convincing.

2. Typo issue: there is a citation error between lines 43–46.

### Soundness
2

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
This paper introduces $CL^{2}GEC$, the first continual learning (CL) benchmark for multi-disciplinary academic writing. The benchmark comprises 10,000 human-annotated sentences from 10 academic fields, designed to evaluate models in a sequential, domain-incremental setting. The authors benchmarked large language models using sequential fine-tuning, parameter-efficient adaptation, and four CL algorithms.

### Strengths
1. This paper first proposes the Continual Learning benchmark for Chinese Literature Grammatical Error Correction, which could comprehensively evaluate the continual learning ability of LLMs in the chinese literature grammatical error correction task.

### Weaknesses
1. This paper only re-implement classic continual learning methods, which are not specifically designed for LLMs' continual learning. This paper should report the newest LLM continual learning methods[1,2] and provide convincing experiments to demonstrate the value of the dataset and the pros and cons of different methods.

2. It is unclear what its core differences from other datasets are. The dataset is built with grammatical error data from 10 different domains, but the connection between this domain categorization and continual learning is not immediately obvious.

3. As detailed in ICLR call for papers, the main text should be 9 pages or fewer, and additional pages are only allowed for the bibliography/references. Thus, the limitations should be controlled in 9 pages.

4. This paper only evaluates two common large language models, it fails to meet the acceptance standards for ICLR in terms of evaluation comprehensiveness, dataset indispensability.

[1] He, Jinghan, et al. "Continual instruction tuning for large multimodal models." arXiv preprint arXiv:2311.16206 (2023).

[2] Smith, James Seale, et al. "Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces **CL2GEC**, the first continual learning benchmark for **Chinese Grammatical Error Correction (CGEC)** across academic disciplines. It studies how large language models adapt to new domains sequentially, highlighting the issue of **catastrophic forgetting**. While the benchmark assumes that models cannot perform **multi-task training** or access all domains simultaneously, this constraint may be unrealistic for modern LLMs that already demonstrate strong **cross-domain generalization and in-context learning**. Experiments on Qwen2.5 and LLaMA3 show that **regularization-based continual learning methods** outperform naive fine-tuning and replay strategies. Overall, CL2GEC provides a valuable research framework for studying lifelong adaptation, though its sequential learning assumption may limit its real-world applicability.

### Strengths
1. The paper introduces a **novel multi-disciplinary Chinese Grammatical Error Correction (GEC) dataset** with a **comprehensive and well-structured evaluation framework**.

2. Experiments on Qwen2.5 and LLaMA3 show the proposed method outperforms naive fine-tuning and replay strategies.

### Weaknesses
1. The paper assumes that large language models (LLMs) can only acquire multi-domain GEC capabilities through **continual learning**, without comparing other plausible approaches such as **multi-task fine-tuning, retrieval-augmented generation (RAG), or in-context learning**. Given the strong generalization ability of LLMs, continual learning may not be strictly necessary in this setting.

2. The model comparison is limited; it should include **more open-source models with stronger Chinese capabilities**, such as **different sizes of the Qwen2.5 series**, to provide a fairer evaluation.

3. The study **does not compare smaller encoder–decoder models**, for which continual learning might actually be **more relevant and effective** than for large instruction-tuned LLMs.

### Questions
N/A

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes CL2GEC which is a benchmark for Chinese Literature Grammatical Error Correction across multiple disciplines. In particular, CL2GEC is designed to support continual learning. The paper explores various continual learning algorithms on this dataset, and the authors perform various experiements to highlight important dynamics (e.g., task ordering, backbone differences) that practitioners should be aware of. Evaluation is done using both standard GEC metrics, as well as continual learning metrics.

## Dataset
Dataset was crawled from China National Knowledge Infrastructure, with 10 disciplines: Law, Management, Education, Economics, Science, History, Agriculture, Literature, Art, and Philosophy. Random sample of 1000 questions per discipline to form 10k questions in total. Then some cleaning steps are done, such as sentence extraction, noise removal, and anonymization. Finally, the data is annotated both by LLMs (initial filter), then manually reviewed by expert annotators.

### Strengths
- baselines are thorough
- it's a big effort to collect 10,000 human annotated sources, so this is a valuable resource
    - Thorough data collection process, with manual human reviews, so the dataset is likely high quality
- experiments had good coverage of various methods.
- ablations are interesting. For example, I liked the section on task order

### Weaknesses
- This is a very narrow domain and not easily generalizable to some of the bigger topics that the community really cares about. I imagine the subset of researchers who care about Chinese Literature GEC might not be that large.
- I would have wanted the authors to flesh out more what makes this task special from other GEC tasks. Are there any nuances specific to this task that are less common in other GEC tasks?
- I don't fully see how this dataset itself is connected to continual learning. It somehow feels like the authors just stitched two somewhat disjointed topics together (GEC + continual learning).

### Questions
- How much or how little did you ablate on the different filtering steps? For example, in the "Noise Removal" step, did you iterate much on the parameters here? Or did you just use standard reasonable assumptions? I'm quite curious on how some of these filtering parameters may affect the performance.
- curious on how this will transfer to other architectures (e.g. state sapace models), and also other model sizes (No need for extra experiments! Just curious if this is something you've done)
- How do you ensure annotators are annotating with the same metric/criteria in mind?

### Soundness
3

### Presentation
3

### Contribution
2
