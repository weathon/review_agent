# Judge a Book by its Cover: Investigating Multi-Modal LLMs for Multi-Page Handwritten Document Transcription

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 8, 2, 4

## Abstract
Handwriting text recognition (HTR) remains a challenging task. Existing approaches require fine-tuning on labeled data, which is impractical to obtain for real-world problems, or rely on zero-shot tools such as OCR engines and multi-modal LLMs (MLLMs). MLLMs have shown promise both as end-to-end transcribers and as OCR post-processors, but to date there is little empirical research evaluating different MLLM prompting strategies for HTR, particularly for the case of *multi-page documents*. Most handwritten documents are multi-page, and share context such as semantic content and handwriting style across pages, yet MLLMs are typically used for transcription at the page level, meaning they throw away this shared context. They are also typically used as either as text-only post-processors or image-only OCR alternatives, rather than leveraging multiple modes.
This paper investigates a suite of methods combining OCR, LLM post-processing and MLLM end-to-end transcription, for the task of zero-shot multi-page handwritten document transcription.
We introduce a benchmark for this task from existing single-page datasets, including a new dataset, `Malvern-Hills`. Finally, we introduce **OCR+PAGE-1** and **OCR+PAGE-N**, prompting strategies for multi-page transcription that outperform existing methods by sharing content across pages while minimizing prompt complexity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces a benchmark framework for transcribing multi-page handwritten documents through a two-step process: initial transcription using OCR engines, followed by refinement with multimodal large language models (MLLMs). While OCR provides a preliminary draft, zero-shot prompting enables MLLMs to detect and correct errors at both character and semantic levels. The study proposes two architectural variants—OCR+PAGE1 and OCR+PAGEN—which combine OCR output with a selected document image to guide the correction process. The key distinction lies in the choice of image used for prompting. By incorporating visual context, the system is hypothesized to learn and generalize transcription errors across pages. Evaluation is conducted across multiple benchmark tasks, including the proposed methods, and is supported by the release of a new dataset: Malvern-Hills.

### Strengths
Originality:
Tackles the challenging problem of multi-page handwritten document transcription, emphasizing semantic continuity across pages. 
Introduces a zero-shot prompting strategy for MLLMs, which adds a novel layer of semantic error correction beyond character-level transcription. 
Proposes two architectural variants, OCR+PAGE1 and OCR+PAGEN, that leverage partial visual context to guide transcription refinement. 

Quality: 
Benchmarks include single-page and multi-page setups, offering a comprehensive view of system performance under varied conditions. 
The experimental design, comparing multiple configurations across OCR and MLLM modules, supports the architectural choices and hypotheses, demonstrating empirical rigor. 

Clarity:
The paper is clearly written and well-structured, with hypotheses and contributions articulated in an accessible manner.
The benchmark structure and dataset contributions are presented in a way that is accessible and reproducible. 

Significance:
Contributes a new dataset (Malvern-Hills) and multi-page extensions of existing datasets, which will be valuable resources for the research community. 
Establishes a benchmark framework that can guide future work in multi-page handwriting transcription.

### Weaknesses
The proposed approach relies heavily on off-the-shelf commercial components (OCR engines and MLLMs), which limits the novelty from a methodological standpoint. While the architectural design and prompting strategy are thoughtfully constructed, the paper would benefit from deeper algorithmic innovation or theoretical grounding.

The multi-page dataset construction primarily involves grouping existing pages, which may not fully reflect the diversity and complexity of real-world multi-page documents. In practice, documents vary significantly in writer identity, layout structure, and semantic content. The paper would be strengthened by a dedicated analysis of how performance varies across different document types or heterogeneous sources (e.g., letters vs. forms, single-writer vs. multi-writer).

The Malvern-Hills dataset, while a valuable addition, is somehow underutilized in the analysis. Given its novelty, the authors should consider to provide: ablation studies isolating the impact depending on variations in the types of documents. Metadata analysis (e.g., writer diversity, layout variation) to demonstrate its relevance and richness compared to existing datasets.

### Questions
Please take into account the comments outlined above. Overall, the paper presents several positive contributions. However, for a more definitive assessment, I would like to see stronger evidence of scientific and methodological innovation beyond the integration of existing components. Additionally, the proposed dataset is a valuable asset, and its internal heterogeneity offers an opportunity for deeper evaluation and discussion. A more comprehensive analysis of its characteristics and impact would significantly enhance the paper’s contribution.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces new methods that use MLLM to improve the OCR transcription of handwritten documents. The authors compare several inputs for the MLLM, including the use of OCR transcriptions and a combination of images with their corresponding OCR transcriptions. The proposed approach improves the results of existing OCR systems, which often perform poorly on handwritten documents when not fine-tuned. The approach also exploits the redundancy of handwriting styles across pages to improve the overall transcription quality while reducing post-processing costs. The authors demonstrate the effectiveness of their approach on three datasets, including one specially designed for evaluation purposes.

### Strengths
The paper is well written, clearly structured and easy to follow, except for the explanation of the methods on page 5, which may slightly confuse the reader. The arguments are well illustrated and justified. In addition, it is particularly important to address costs when working with MLLMs, and this aspect is discussed throughout the paper. While not really emphasised, the evaluation of the different types of errors using an LLM is also very interesting and is rarely presented in papers. It is crucial to understand which errors persist and are added by the model in order to identify how they can be solved.
Furthermore, the proposed approach is easily reusable, as it is straightforward to implement. It seems that the prompts are available in the GitHub repository, which enhances the reproducibility of the approach. The paper is sufficiently clear to allow readers to reproduce the results and adapt the approach to their own data.

### Weaknesses
The pages selected as input for the post-OCR correction model are either the first page or a page chosen by an intermediate LLM. It would have been interesting to compare these approaches with a random page selection. This would be a compromise, avoiding the systematic use of the first page, which may only contain a title, while simplifying the processing pipeline by removing the need to use an LLM only for page selection.
Furthermore, the paper frequently refers to “multi-page” processing. However, most of the documents used in the experiments contain only two pages. While this demonstrates some improvement, the use of the pageN to select a single page out of two seems rather complex. Evaluating the proposed methods on longer documents (e.g., at least five or ten pages) would provide a more realistic evaluation of their effectiveness

### Questions
The costs of the models were discussed throughout the paper. It would also have been interesting to analyse inference times, as these can sometimes be critical when processing long documents or large amount of documents.

### Soundness
4

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
5

### Summary
This paper evaluates using multi-modal LLMs for handwritten document transcription. Specifically, the paper focuses on the problem of multi-page OCR. In this context, it evaluates several alternatives for presenting the LLM with image + text information - OCR + images, OCR + page1, and OCR + pageN, where the difference in the image modality is having the specific page to be transcribed, the first page, and a single page chosen by an upstream LLM, respectively. 

Results over three datasets show the proposed OCR + page1 and OCR + pageN to be among the top performing methods, and generally comparable to OCR + images, thus demonstrating that information from a single page can be extrapolated to improve OCR overall, at a level similar to having access to the full set of images, thereby helping to reduce total transcription cost.

### Strengths
Thorough evaluation and discussion.  In particular, the proposed method is tested on three separate datasets, including one dataset constructed by the authors.

### Weaknesses
Technical novelty/contribution is minor, as the method is a small modification to the baseline of OCR + images, the problem is somewhat niche (minimizing cost in multi-page handwriting OCR), and the improvement/benefit is somewhat small. Paper is probably better suited to a more text/OCR-specific conference.

In addition, the multi-page documents are generally short, the majority being 2 pages, and only one dataset having a small number of 4 page documents, so both the challenge and benefit of using one page versus the whole document seems limited.  It would potentially be more interesting if longer documents were used (eg 10+ pages).  One potential question would be how performance scales with page length.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose an interesting idea, suggesting that one page image can provide enough visual context to correct OCR across the document. From this claim, the paper studies zero-shot transcription of multi-page handwritten documents with multimodal LLMs. Two different prompting strategies are proposed to evaluate the impact of giving to an MLLM the OCR text for the whole document plus one page image, considering that a single image captures cross-page regularities (writer style, OCR error patterns) and it is sufficient to correct OCR on all pages. 

The paper also presents a multi-page benchmark from IAM and Bentham and releases a new historical dataset (Malvern-Hills). Across three datasets, they compare page-by-page vs. all-at-once pipelines and text-only vs. image-only vs. hybrid inputs using GPT-4o, Gemini-2.5-Pro, and Gemma-3-27B. 

Results show that OCR+PAGE1/PAGEN are consistently on the cost/accuracy Pareto frontier, often outperforming full image inputs in character error rate (CER) while being cheaper in tokens. The authors also present an LLM-based semantic error report showing fewer major errors for the proposed methods.

### Strengths
Treating multi-page handwriting recognition as a cross-page context sharing problem and asking an MLLM to learn the OCR to image error mapping from a single page is an original and practical idea.

The demonstration that one page’s visual signal often suffices to correct systematic OCR errors across unseen pages is both novel and impactful for cost-constrained digitization workflows.

The release of a multi-page benchmark is valuable.

The broad comparison across methods (PBP vs. AAO; OCR, images, hybrids) and models (Gemini, GPT-4o, Gemma) on three datasets, with cost accounting and a semantic error audit is important.

### Weaknesses
Despite the interesting results, the study does not yet position itself against the strongest non-MLLM baselines (e.g., modern HTR models with few-shot adaptation or lexicon-aware decoders) nor against competitive open MLLMs specialized for document OCR. The authors should argue more in this direction in the paper.

I am not sure I understand correctly, but it seems to me that combining single pages by writer ID likely increases lexical overlap across pages, making cross-page extrapolation easier than in organically multi-page documents.

Costs sometimes favor OCR+PAGE methods despite extra OCR tokens. In this case, the per-stage token counts, image-token policy, and retry/failure rates are not tabulated.

All datasets are English/Latin scripts. I suggest an evaluation on diverse scripts or heavily structured forms or arguments in this direction in the paper. 

The use of LLM-as-judge where one vendor model is used to assess outputs of competing vendor models is interesting, but it could be interesting to expand the auditor set (cross-vendor) and include limited human adjudication beyond 5 docs.

Costs appear to mix OCR and LLM usage. I think it is important to report per-stage token counts, image tokenization policy, truncation events, and failure/retry rates.

### Questions
Did you use OCR bounding boxes/layout?

Have you tried OCR text + token-level image crops only where the OCR is uncertain? This could lower costs further while isolating where images help.

How do accuracy and cost scale for 5–10 page documents?

### Soundness
3

### Presentation
3

### Contribution
3
