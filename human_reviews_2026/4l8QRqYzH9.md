# GDI-Bench: A Benchmark for General Document Intelligence with Vision and Reasoning Decoupling

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 2, 6

## Abstract
The rapid advancement of multimodal large language models (MLLMs) has profoundly impacted the document domain, creating a wide array of application scenarios. This progress highlights the need for a comprehensive benchmark to evaluate these models' capabilities across various document-specific tasks. However, existing benchmarks often fail to locate specific model weaknesses or guide systematic improvements. To bridge this gap, we introduce a General Document Intelligence Benchmark (GDI-Bench), featuring 2.3k images across 9 key scenarios and 19 document-specific tasks. By decoupling visual complexity and reasoning complexity, the GDI-Bench structures graded tasks that allow performance assessment by difficulty, aiding in model weakness identification and optimization guidance. We evaluate various open-source and closed-source models on GDI-Bench, conducting decoupled analyses in the visual and reasoning domains, revealing their strengths and weaknesses. To address the diverse tasks and domains in the GDI-Bench, we propose a GDI-Model that mitigates catastrophic forgetting during the supervised fine-tuning (SFT) process through an intelligence-preserving training strategy, thereby reinforcing the inherent weaknesses of the base model. Our model achieves state-of-the-art performance on previous benchmarks and the GDI-Bench. Both our benchmark and models are or will be open-sourced on \url{https://huggingface.co/GDIBench}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents two primary contributions:

GDI-Bench: A new benchmark for document intelligence. Its core innovation is the decoupling of document understanding tasks into two orthogonal difficulty dimensions: visual complexity and reasoning complexity. The benchmark features 2.3k images across 9 key scenarios and 19 document-specific tasks.

GDI-Model: A model developed based on InternVL3-8B , which introduces a training strategy called "Layer-wise Adaptive Freeze-Tuning" (LW-AFT). This strategy aims to address the issue of catastrophic forgetting encountered during SFT on document tasks.

Experimental results show that the 8B-parameter GDI-Model achieves overall performance on GDI-Bench that surpasses larger closed-source models, including GPT-4o, Claude-3-7-Sonnet, and Gemini-2.0-Flash.

### Strengths
The paper's core contribution—decoupling visual and reasoning complexity—is well-motivated. The authors correctly point out that an MLLM's failure on a document task could stem from inaccurate visual recognition, limited language organization, or both.

The LW-AFT method is an interesting and novel contribution that addresses a well-known problem in SFT: catastrophic forgetting.

The paper is well-written. Figures like Figure 1 (benchmark framework) and Figure 7 (LW-AFT method) greatly aid in understanding the core ideas.

### Weaknesses
The R1 and R2 task QA pairs were generated using GPT-4o. This is a common practice but introduces a potential "LLM-on-LLM" evaluation bias. The benchmark might, to some extent, be measuring the consistency of other models with GPT-4o's output style.

The evaluation of the models is limited. It is necessary to additionally assess Qwen2.5-VL-7B, Qwen3-VL-8B, Ovis2, and others.

I am quite curious about the performance of the recently popular work, DFT, on catastrophic forgetting in the document application domain.

DFT: On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification

### Questions
See Weaknesses

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
This paper presents the General Document Intelligence Benchmark (GDI-Bench), covering 9 key scenarios, 2,300 images, and 19 document-specific tasks. By decoupling visual complexity from reasoning complexity, it creates graded tasks to identify model weaknesses and guide optimization . The study evaluates open-source and closed-source models on GDI-Bench, finding issues like the InternVL3-8B—performing well in the R0 domain but degrading significantly in R1 and R2 . To address these weaknesses and catastrophic forgetting in supervised fine-tuning (SFT), the Layer-wise Adaptive Freeze-Tuning (LW-AFT) method is proposed, leading to GDI-Model. This model achieves state-of-the-art (SOTA) performance on GDI-Bench and other benchmarks.

### Strengths
1. GDI-Bench covers 9 key document scenarios, 19 specific tasks, and 2.3k images, ensuring scenario representativeness via multi-source data integration.
2. To address "catastrophic forgetting" in SFT, the study proposes Layer-wise Adaptive Freeze-Tuning (LW-AFT), which improves models’ reasoning performance while preserving basic capabilities.
3. The study pioneers decoupling the complexity of multimodal document understanding into visual complexity (V0-V2) and reasoning complexity (R0-R2), with a difficulty grading mechanism.

### Weaknesses
1. GDI-Bench currently only supports single-image document understanding tasks and excludes multi-image or multi-document tasks, limiting its ability to evaluate models in complex real-world scenarios that require cross-image information integration .
2. The proposed Layer-wise Adaptive Freeze-Tuning (LW-AFT) method has mainly been validated on the InternVL3-8B model, with insufficient verification of its generality across models with different architectures (e.g., closed-source models like GPT-4o or other open-source models of varying sizes) .
3. GDI-Bench’s data sources are primarily from OmniDocBench and in-house collections of mainstream domain documents (e.g., scientific papers, newspapers), lacking coverage of tasks in niche domains (e.g., specialized technical manuals, rare language documents), which restricts its adaptability to more diverse practical document processing needs .

### Questions
1. How does the Layer-wise Adaptive Freeze-Tuning (LW-AFT) method proposed in the paper alleviate catastrophic forgetting during the supervised fine-tuning (SFT) process?
2.Compared with its base model InternVL3-8B, on which reasoning complexity levels (R0/R1/R2) of GDI-Bench does the GDI-Model show significant performance improvement?

### Soundness
3

### Presentation
2

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
The paper describes two different and independent contributions. On one hand, the creation of a new benchmark for document understanding, where the main novelty is the categorization of documents and tasks across two different axes: visual complexity of document and reasoning difficulty of the tasks. This can allow to analyze the performance of the evaluated models according to this two variables. On the other hand, the proposal of a continual learning strategy to mitigate catastrophic forgetting when training the model for different tasks and domains of documents. Both contributions are partially related through the application of the continual learning approach to the training of different tasks in the proposed GDI_Bench.

### Strengths
- The paper introduces a new benchmark for document understanding that includes diverse types of documents and reasoning tasks, that can be useful to advance in the research of generic document understanding models, able to perform well across different types of documents and tasks.
- The paper analyzes the problem of catastrophic forgetting in the context of learning differnet tasks for document understandin and proposes a training procedure to overcome this problem.

### Weaknesses
- The categorization of documents using visual complexity and reasoning difficulty seems to abstract. It is difficult to know exactly what documents and tasks are behind categories V0,v1 and V2, and R0, R1, R2. I think it is better a categorization based on more focused characteristics of documents and tasks, as it is already done in other existing benchmarks. For instance, categorizing by the type of visual evidences contained in a document or required to answer the question, or by the type of reasoning required (single-page or cross-page, multi-hop reasoning, arithmetic operation, ... )
- Visual complexity is defined according to the performance of current models in OCR conversion, not to the intrinsic characteristics of the documents. OCR performance depends on current benchmarks and models, and thus, it might not reflect actual complexity of documents, but complexity of the documents in that specific benchamark and the ability of models to deal with those types of documents. For instance, in principle, it would seem that slides or magazines can contain more visual complexity than newspapers, but slides have much better score than newspapers. 
- In general, the definition of the tasks for each level of difficulty is not clear, especially for R1 and R2. According to table 2, there are 19 different tasks. Which are these 19 different tasks? How are they distributed accross the different levels of difficulty. In addition, R2 tasks are defined as multi-choice QAs. Maybe it is not the best way to evaluate complex reasoning abilities of the models, since a multi-choice question is giving some hints to answer the model. It seems more appropiate to have open QAs
- Tasks in different levels of difficulty have different evaluation metrics and thus, results are not comparable across levels of difficulty. For instance, one would expect that results for R2 tasks are worse than for R1 and R0 tasks, but according to figure 2, they are better. Also results for V1 documents in R1 tasks are better than results for V0 documents which, in principle are less complex. Some discussion on this effect would be necessary. 
- The results shown in figure 2 are not discussed in the text of the paper. Only results for InternVL are discussed in section 4. A global joint discussion of these results for all models across different types of documents and tasks would be necessary. 
- The initial hypothesis that motivates the proposal of the fine-tuning method in secion 4 is not fully justified. According to table 5, fine-tuning with the full model not only degrades performance for R0 tasks, but also for R2 tasks, that are supposed to be included in the training set. This would suggest that the only explanation for the degradation of performance would not be catastrophic forgetting. 
- The proposed fine-tuning approach is only compared with LoRA (with very similar results). A better comparison with other SoA for continual learning would be necessary.
- Section on related work focuses the analysis of document understanding on OCR tasks. Document understanding implies much more than simple OCR.

### Questions
- In terms of paper format, figures and tables should be located in the document closer to the section of the text where they are discussed. For instance, figure 2 and table 5 are far away from the text where they are cited. 
- How the combination of the results across different models in OmniDocBench is done to get the score that is used to decide the category for visual complexity? How thresholds for V1 and V2 are determined?
- It is not clear what figure 3 exactly shows. From the context I guess that it shows average edit distance score for OmniDocBenc for each type of documents, but more details are needed in the text in order to better explain it. 
- What is the training set used to obtain the results in table 3? It is the combination of all training sets for all datasets?

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
The submission introduces GDI-Bench, a new benchmark for General Document Intelligence that decouples visual complexity (V0–V2) and reasoning complexity (R0–R2) to diagnose weaknesses of multimodal LLMs on document tasks. It covers 9 document scenarios, 19 task types, and 3,660 test cases, with graded difficulty and a unified scoring protocol. The authors also propose LW-AFT (Layer-wise Adaptive Freeze-Tuning), a parameter-freezing strategy guided by layer-wise sensitivity to mitigate catastrophic forgetting during SFT; they fine-tune InternVL3-8B to create GDI-Model and show improved R1/R2 performance while retaining R0 capabilities. Extensive comparisons with open/closed-source models and document-oriented MLLMs support the claims, and the benchmark and model are promised to be open-sourced.

### Strengths
- Clear decoupling of visual vs. reasoning complexity; practical difficulty grading.

- Broad coverage: 9 scenarios, 19 tasks; supports MLLMs, OCR+LLM, and parsers.

- Sound evaluation protocol (mix of NED and accuracy), reproducibility details provided.

- LW-AFT is simple, effective, and well-motivated by parameter-change analyses.

- Strong empirical gains on R1/R2 with minimal degradation on R0; cross-domain/task evidence.

### Weaknesses
(1) Some reliance on LLM-generated annotations; limited auditing statistics on annotation quality, inter-annotator agreement, and remaining bias/noise.

(2) Visual complexity taxonomy (V0/V1/V2) is partly derived from OmniDocBench SOTA gaps; may overfit to current model weaknesses rather than intrinsic document properties. Lacks a formal metric or blind human assessment of visual difficulty.

(3) Reasoning complexity (R0/R1/R2) boundaries could be sharper; e.g., clearer operational definitions and multi-step vs. single-hop distinctions, and examples with verified reasoning chains.

(4) LW-AFT selection relies on sensitivity from small α expert models; ablation on alternative selection signals (e.g., Fisher information, gradient norms, magnitude pruning baselines) is missing.

(5) Comparisons: some strong closed-source baselines (o3, Claude-3.7, Gemini) are reported but details on prompting, context length, and image preprocessing parity should be elaborated to ensure fairness.

(6) Catastrophic forgetting analysis is mostly aggregate; per-subskill breakdown (e.g., table structure recovery vs. formula transcription) and longer continual-learning sequences would strengthen claims.

(7) Currently single-image only; multi-page/multi-document workflows (common in PDFs) not yet covered.

### Questions
(1) Visual complexity: Can you provide a model-agnostic, quantitative metric (e.g., entropy of layout graph, density of non-textual elements, columnar complexity) to assign V1/V2, rather than relying on prior SOTA gaps?

(2) Reasoning complexity: How do you ensure R2 questions truly require reasoning beyond pattern matching? Any human-vs-model ablation verifying necessity (e.g., answerability with text-only vs. layout-only inputs)?

(3) Annotation quality: What are the inter-annotator agreement statistics after PhD-level verification? What proportion of synthetic items were rejected? Any audit on language bias (CN/EN) and OCR noise tolerance?

(4) LW-AFT: How robust is the parameter mask when changing α or training data domains? Can masks learned on domain A transfer to B, or do you need per-domain masks? Any memory/computation trade-offs when maintaining multiple masks?

(5) Baselines: Did you evaluate other selective-update CL strategies (EWC, L2-SP, Fisher-based pruning, adapters+freeze, Task Vectors) under the same setup?

(6) Fairness: For closed-source models, what were the exact prompts, temperature, max tokens, image resolutions, and chunking procedures? Were multiple trials averaged to reduce sampling variance?

(7) Generalization: How does GDI-Model perform on long multi-page PDFs with cross-page references, footnotes, and figure-to-text linking?

(8) Metric sensitivity: For NED-based tasks, how do you handle minor formatting differences (e.g., whitespace, punctuation, case)? Any robustness checks?

### Soundness
3

### Presentation
3

### Contribution
3
