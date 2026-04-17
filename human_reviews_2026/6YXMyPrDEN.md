# End-to-End Document Understanding via Chain-of-Reading

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Intelligent Document Analysis (IDA) is a formidable task owing to documents’ complex layouts, dense tables, charts, and mixed modalities. Conventional pipelines apply OCR before large language model reasoning but suffer from error propagation. End-to-end multimodal models avoid explicit pipelines yet struggle to scale to multi-page documents, where information dilution and evidence localization remain major bottlenecks. We propose Chain-of-Reading (CoR), an end-to-end framework that transforms traditional text-centric reading into a native multimodal paradigm. CoR directly consumes PDF pages as visual input, mimicking human eyes, and performs document-level question answering through a chain-of-thought process. It first localizes relevant evidence, then selectively applies OCR, and finally performs reasoning over the localized content. To further enhance comprehension of visual elements such as charts and scientific figures—which exacerbate information dilution and impede pinpointing evidence—we introduce Masked Auto-Regression (Mask-AR), a self-supervised method for multimodal grounding. CoR achieves a 14.3% improvement over the base model on the MMLongBench-Doc benchmark. We will release the CoR-Dataset and our fine-tuned model, Qwen2.5-VL-CoR.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a workflow designed to enhance LLMs ability to locate supporting evidence within long PDF documents. The proposed training strategies demonstrate promising improvements in long-document understanding across two benchmark datasets. However, the paper raises several concerns regarding its overall structure, methodological novelty, and the depth of both quantitative and qualitative analyses.

### Strengths
- The motivation of this paper is meaningful.
- The introduced methods demonstrate effectiveness in achieving the claimed contributions through experimental results.
- The proposed training paradigm shows potential for practical application in domain-specific scenarios.

### Weaknesses
The main concerns for this work include:

- Paper structure and organization: The paper would benefit from substantial restructuring. Too much essential information, such as dataset creation and method details, is placed in the appendix, making the paper less self-contained. Figures 2 and 3 appear somewhat repetitive. Additionally, the qualitative analysis examples are presented too early—before readers fully understand the context. Moving these examples to the end, alongside result discussions, would improve clarity and better highlight the contributions.

- Technical novelty: While the paper achieves promising results from a practical standpoint, it lacks sufficient technical innovation. The improvement appears to stem mainly from using more data and manually verified knowledge rather than introducing fundamentally new approaches. 

- Generalization concern: It remains unclear whether the proposed training approach preserves the model’s general-domain and single-page document understanding abilities. Evaluating the model on general-domain and single-page VQA tasks would be valuable to ensure that the specialized training does not compromise the LLM’s broader reasoning or knowledge capabilities. As well as, the effectiveness on various scale LLM backbones are also underexplored.

### Questions
- What general error patterns might occur?

- Can your framework still perform effectively on general or single-page VQA tasks?

- Are there any methods to evaluate the model’s reasoning process, and if incorrect reasoning occurs, how can it be detected and mitigated to prevent error accumulation?

### Soundness
3

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
4

### Summary
The paper introduces the “Chain-of-Reading” (CoR), an end-to-end framework for document understanding, which enhances comprehension of complex documents that combine text, tables, charts, and images. The CoR framework is designed to address two major challenges in document understanding: evidence localization (finding relevant information within a document) and information dilution (maintaining focus across large documents). The model processes raw PDF pages directly as visual input and performs document-level question answering (QA) through a chain-of-thought reasoning process. Additionally, the paper presents Masked Auto-Regression (Mask-AR), a self-supervised method that improves the model’s understanding of visual elements like charts and figures by grounding them with text. Through extensive experiments, the CoR framework outperforms baseline models and achieves state-of-the-art performance on the MMLongBench-Doc and LongDocURL benchmarks, even surpassing some proprietary models.

### Strengths
1. Innovation in Reasoning Paradigm: Chain-of-Reading (CoR) introduces a fundamentally new approach to document understanding, inspired by how humans read and reason through documents. Unlike traditional systems that rely on separate stages (OCR, layout analysis, etc.), CoR integrates evidence localization and reasoning into a single, trainable end-to-end framework.

2. Targeted Improvement for Visual Complexity: The introduction of Masked Auto-Regression (Mask-AR) specifically targets the challenge of visual grounding (e.g., charts), which is known to be a weakness in current Multimodal Large Language Models (MLLMs).

3. Data Generation and Evaluation: The paper details the creation of the CoR-Dataset, specifically designed to train and evaluate CoR. This dataset includes 26,088 high-quality QA pairs, each annotated with reasoning traces that explicitly follow the CoR paradigm.

### Weaknesses
- Incomplete Baseline Comparison: The comparison lacks recent open-source models, such as MinerU, MinerU-2.5. For proprietary models, the comparisons are limited and somewhat outdated, with the latest inclusion being Gemini 1.5 Pro.

- Novelty of Visual Chain-of-Thought: There are numerous existing works on Chain-of-Thought (CoT) related to vision; thus, applying it to Document Understanding might not be entirely novel, perhaps suggesting more novelty in prompt design tailored for multi-page scenarios.

- Insufficient Solution for Key Information Dilution: The authors claim to address key information dilution, but the proposed methods—CoR for localization difficulty and Mask-AR for better visual understanding—do not appear to fully resolve the core issue of information dilution across lengthy documents.

### Questions
- Although the paper proposes Masked Auto-Regression (Mask-AR) to enhance the understanding of visual elements, especially complex charts and scientific graphics, its effectiveness seems more pronounced on standardized, well-structured charts (e.g., those in scientific literature). Given that the dataset also has a high proportion of structured data like academic papers and government reports, we are curious if the authors can provide a case study demonstrating the model's performance in understanding and locating non-standardized visual elements.

- The authors identify key information dilution as a problem, but I believe it is not adequately solved. CoR addresses evidence localization difficulty, and Mask-AR better understands visual elements, but neither method seems to fully resolve the core issue of key information dilution across long documents. Could the authors elaborate on how CoR and Mask-AR directly mitigate the information dilution effect, rather than just improving component steps?

- I find the comparison with other methods incomplete. The current comparison lacks more recent visual grounding baselines or open-source models like MinerU, and the proprietary model comparison is still stuck at Gemini 1.5 Pro, which is somewhat outdated. Will there be future work to incorporate a more comprehensive and contemporary set of baselines?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the challenge of end-to-end document understanding for long, multi-page, visually-rich documents (like PDFs). The authors identify two primary failures in existing systems: 1) traditional pipelines (e.g., OCR $\rightarrow$ LLM) suffer from cascading errors, and 2) current end-to-end Multimodal LLMs (MLLMs) struggle with information dilution and evidence localization in long contexts.

To address this, the paper proposes a framework with three main contributions:
1.  **Chain-of-Reading (CoR):** A training paradigm that fine-tunes an MLLM to first generate an explicit, step-by-step reasoning trace *before* answering a question. This trace mimics a human's "Plan $\rightarrow$ Locate $\rightarrow$ Extract $\rightarrow$ Synthesize" process, forcing the model to articulate *where* it is looking (e.g., page numbers, table coordinates) and *what* it finds.
2.  **Masked Auto-Regression (Mask-AR):** A self-supervised task designed to improve the model's comprehension of visual elements like charts and figures. The model is trained to reconstruct masked-out portions of figure captions by reasoning over the visual content and the surrounding document text.
3.  **A Curated Training Strategy:** The authors develop a 3-stage training recipe (foundational LoRA tuning, specialized SFT on new CoR and Mask-AR datasets, and DPO alignment). They also created and will release the **CoR-Dataset** (26,088 QA-trace pairs) to train this behavior.

Their resulting model, **Qwen2.5-VL-COR** (a fine-tuned 7B model), achieves significant performance gains over its base model (+14.3% on MMLongBench-Doc) and notably outperforms larger, proprietary models like GPT-4V and Gemini 1.5 Pro on long-document benchmarks.

### Strengths
1.  **Originality of the Core Idea:** The "Chain-of-Reading" (CoR) paradigm is a novel and highly effective way to structure reasoning for document analysis. It operationalizes the human intuition of "find it first, then read it" into a trainable format. This bridges the gap between simple Chain-of-Thought (which often fails at localization) and complex, multi-tool agentic frameworks (which suffer from high latency).
2.  **Methodological Rigor:** The 3-stage training strategy is comprehensive and well-justified. It correctly identifies that foundational skills, task-specific reasoning patterns, and preference alignment are all necessary. The ablation studies clearly support this, showing that each component (CoR SFT, Mask-AR, and DPO) adds complementary and significant value.
3.  **Excellent Qualitative Analysis:** The paper's clarity is massively boosted by its qualitative examples. The appendix (A.9) is a model for how to demonstrate a model's reasoning process. It showcases success on diverse and challenging tasks, including parsing irregular layouts, navigating repetitive pages, and, most importantly, identifying and rejecting "hallucination traps" (Example 11).
4.  **Significant and Surprising Results:** The paper's most significant strength is its results. The final 7B model (Qwen2.5-VL-COR) doesn't just get a small boost; it achieves a **+14.3%** gain on MMLongBench-Doc over its base. More impressively, it sets a new SOTA for open-source models and surpasses proprietary giants like GPT-4V and Gemini-1.5-Pro on these benchmarks.

### Weaknesses
1.  **Limited Evaluation on Diverse, 'Real-World' Documents:** The paper's evaluation is impressive but narrow. The CoR-Dataset is heavily skewed (75%+) towards academic papers and government reports (Figure 6). The evaluation benchmarks (MMLongBench, LongDocURL) share this academic focus. The authors miss a key opportunity to prove the model's *generalization*. They fail to benchmark against a practically-oriented, multi-industry, multi-domain dataset like **DUDE**, which was specifically designed to test usability on diverse, long documents and serve as an easy way to debug performance in real-world generalization scenarios. This omission makes it unclear if the CoR paradigm works on invoices, legal contracts, or other messy document types not well-represented in the training data.
2.  **No Performance/Cost Analysis:** The CoR paradigm requires the model to generate a very long, explicit reasoning trace *before* providing the final answer (e.g.,). This dramatically increases the number of generated tokens, which directly impacts inference latency and cost. The paper provides **zero benchmarks or discussion on this practical trade-off**.
3.  **Missing Simple Baselines:** The paper only compares against the base model (a zero-shot baseline). A critical missing experiment is a **few-shot prompting baseline**. How much of this 14% gain comes from the 26,088-sample SFT/DPO, and how much could be unlocked by simply showing the base Qwen2.5-VL model the CoR format in a 5-shot or 10-shot prompt?
4.  **Ambiguity on Data Contamination:** The authors admit to using a "curated mixture of publicly available document analysis datasets" in Stage 1. They *must* explicitly state whether they filtered out all samples from the MMLongBench and LongDocURL test sets from this mixture. The lack of a clear "data hygiene" statement is a recurring problem in LLM-era research.

### Questions
1.  Can the authors please clarify their data contamination protocol? Did they explicitly search for and remove all documents and questions from the MMLongBench and LongDocURL benchmarks from their Stage 1 training mixture?
2.  Could the authors provide a simple baseline comparing their full Qwen2.5-VL-COR model to the base Qwen2.5-VL model when guided by a 5-shot or 10-shot prompt demonstrating the CoR trace format? This is crucial for justifying the necessity of the 26k-sample SFT pipeline.
3.  What is the practical inference cost of the CoR model? Could the authors please provide data on the **average token length of a CoR trace** and the **end-to-end latency** (in tokens/sec or sec/query) compared to the base model's direct-answering?
4.  Given the CoR-Dataset's 75%+ skew towards academic/government reports, have the authors performed any zero-shot generalization tests on out-of-domain document types, such as those in the **DUDE** benchmark, to test the model's robustness and debug its performance on more diverse, practical layouts?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Chain-of-Reading (CoR), an end-to-end framework for multi-page, visual document QA task.  It treats PDFs as visual inputs and uses a workflow which starts with planning, localizing evidence, then reasoning.  CoR reduces error propagation from OCR pipelines and mitigate information dilution in long documents. To better handle complex charts and scientific figures, the authors add a self-supervised Masked Auto-Regression (Mask-AR) objective to improve the multimodal comprehension. The authors also construct a CoR-Dataset to supervise the reading-chain paradigm, curating multi-page PDFs with Q&A and stepwise localization traces, annotated via a semi-automated pipeline, and plan to release it alongside Qwen2.5-VL-CoR. They train a new model with constructed dataset using DPO and CoR paradigm. Experiments results show that on the long visual document benchmark MMLongBench-Doc, CoR reports a +14.3% improvement over its base model Qwen2.5-VL-7B.  And the improvement of the model on LongDocURL is +12.3%.

### Strengths
1. This paper proposes a clear “locate-then-reason” training paradigm (CoR) that supervises evidence localization before reasoning. This is different from traditional end-to-end approaches that focus on image-to-text modelling without an explicit evidence-first chain.  This paper also introduces Mask-AR as a self-supervised task to improve chart/figure comprehension. 

2. The three-stage pipeline, 1. SFT on public data 2. task-specific SFT on CoR and Mask-AR 3. DPO alignment on pair-wise data, is coherent and ablation studies clearly isolate gains from each stage.  Experiments on two long-document benchmarks, MMLongBench-Doc and LongDocURL, are conducted. The paper reports strong, clearly tabulated improvements over the base model (e.g., +14.3% Acc on MMLongBench-Doc; +12.3% on LongDocURL). 

3 The CoR stages (planning, focused search, cross-modal integration, reasoning) are described clearly with figures and multiple examples.  Tables include per-modality and per-page-count breakdowns, helping readers to understand in which areas the performances gains are greater. 

4. This paper demonstrates that a 7B OCR-free model with structured supervision can approach the performance of larger and stronger LLMs on long-document QA tasks. This paper constructs a well-defined dataset that targets the evidence localization.

### Weaknesses
1.	The framework Chain-of-Reading’s core idea, planning, locating, extracting, and reasoning is not very innovative compared to existing framework like DocReact.

2.	The Mask-AR training is based on scientific figure–caption pairs, while generalization to non-scholarly documents may not be included. You may consider adding tasks beyond academic PDFs to test domain transfer a common focus in earlier OCR-free Document Understanding like Donut and follow-ups. 

3.	While the three-stage training is described, practical costs vs. alternatives are not compared. 

4.	Core baselines include Qwen2.5-VL-7b, it would be better to include other models as baselines, like larger models or non-Qwen models to showcase the generalizability of the method.

### Questions
In the training stage, from the datasets that you’ve constructed, there has been a large number of documents for chartQA, DocVQA, or academic papers included (as shown in Tables 6 and 7). While the test sets MMlongbench-doc and LongDoc-URL also include documents of these types. Is there any overlap of any document between your constructed datasets and the test benchmarks?

### Soundness
3

### Presentation
4

### Contribution
3
