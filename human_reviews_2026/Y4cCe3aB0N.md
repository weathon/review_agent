# Benchmarking Multimodal LLMs on Recognition and Understanding over Chemical Tables

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
With the widespread application of multimodal large language models in scientific intelligence, there is an urgent need for more challenging evaluation benchmarks to assess their ability to understand complex scientific data. Scientific tables, as core carriers of knowledge representation, combine text, symbols, and graphics, forming a typical multimodal reasoning scenario. However, existing benchmarks are mostly focused on general domains, failing to reflect the unique structural complexity and domain-specific semantics inherent in scientific research. Chemical tables are particularly representative: they intertwine structured variables such as reagents, conditions, and yields with visual symbols like molecular structures and chemical formulas, posing significant challenges to models in cross-modal alignment and semantic parsing.
To address this, we propose ChemTable—a large-scale benchmark of chemical tables constructed from real-world literature, containing expert-annotated cell layouts, logical structures, and domain-specific labels. It supports two core tasks: (1) table recognition (structure and content extraction); and (2) table understanding (descriptive and reasoning-based question answering). Evaluation on ChemTable shows that while mainstream multimodal models perform reasonably well in layout parsing, they still face significant limitations when handling critical elements such as molecular structures and symbolic conventions. Closed-source models lead overall but still fall short of human-level performance.
This work provides a realistic testing platform for evaluating scientific multimodal understanding, revealing the current bottlenecks in domain-specific reasoning and advancing the development of intelligent systems for scientific research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces ChemTable, a large-scale benchmark for evaluating multimodal large language models (MLLMs) on chemical table recognition and understanding. The dataset includes 1,382 real-world chemical tables extracted from top-tier chemistry journals, annotated with structure, content, and over 9,000 QA pairs. The authors evaluate both open-source and proprietary MLLMs on tasks such as table structure extraction, molecular recognition, and reasoning over chemical data. The key claim is that current MLLMs fall short of human-level performance, especially in domain-specific reasoning and molecular structure understanding.

### Strengths
This paper presents ChemTable, a carefully constructed and richly annotated benchmark focused on chemical table recognition and understanding, which fills a notable gap in the evaluation of multimodal large language models (MLLMs) on domain-specific scientific content. The dataset is derived from real-world chemistry literature and includes over 1,300 tables and nearly 10,000 question-answer pairs, supporting both structural extraction and reasoning tasks. The authors provide a comprehensive evaluation pipeline and assess a wide range of open-source and proprietary models, offering a useful reference for the community. The work is clearly written, well-organized.

### Weaknesses
- The experimental conclusions lack depth and novelty. The finding that proprietary models outperform open-source ones is widely acknowledged and not specific to this domain. The paper does not provide detailed error analysis or insights into why models fail, making the conclusions too generic to guide future model development.

- The paper does not sufficiently justify the necessity of a chemistry-specific benchmark in light of existing general-purpose benchmarks like MMMU or HLE, which also include chemistry-related content. There is no comparative analysis showing that ChemTable introduces uniquely challenging or uncovered tasks, weakening the motivation for a new dataset.

- The evaluation metrics, while standard, are not diagnostic. High-level accuracy scores do not reveal whether models truly understand chemical content or are relying on superficial cues. There is no attempt to evaluate intermediate reasoning steps or semantic correctness, especially in molecular recognition and multi-hop reasoning tasks.

- The paper lacks exploration of training strategies or model behavior. All models are evaluated in a zero-shot or frozen setting, with no investigation into whether domain-specific pretraining or fine-tuning improves performance. This limits the benchmark’s utility as a tool for driving model development rather than just evaluation.

- The generalizability of the findings is not discussed. While the benchmark is chemistry-specific, there is no attempt to assess whether insights or improvements from ChemTable transfer to other scientific domains or table types, limiting its broader impact.

### Questions
Please see weaknesses.

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
4

### Summary
This paper presents ChemTable. It is a large-scale benchmark designed to evaluate the recognition and understanding abilities of MLLMs when interacting with real-world chemical tables. 
ChemTable includes more than 1300 tables collected from top chemistry journals.
Each table is enriched with pixel-level annotations, logical layout structures, and domain-specific labels.
The benchmark supports two major tasks: table recognition (structure parsing and content extraction) and table understanding (descriptive and reasoning QA).
The authors conduct extensive experiments on both proprietary and open-source MLLMs. They benchmark these models across a diverse set of tasks and compare their ability with human performance.

### Strengths
- This paper makes a valuable contribution by introducing a benchmark on chemical tables, an area previously lacking multimodal datasets.
Unlike prior table benchmarks (e.g., FinQA, SciTab, MMTab), ChemTable focuses on chemistry-specific content that includes molecular structures, chemical symbols, and experimental conditions.
- The dataset contains over 1300 real-world chemistry tables and 9000 question–answer pairs, covering a wide range of tasks such as table recognition, structure parsing, and reasoning-based question answering. 
The annotation process is comprehensive, including layout, text, molecular graphics, and metadata. 
- The authors conduct an extensive comparison across multiple leading open-source (e.g., Qwen-VL, Llama, InternVL) and proprietary models (e.g., GPT-4.1, Gemini-2.5-Flash). 
This benchmarking provides a clear and balanced view of current model capabilities and limitations in chemical table understanding.

### Weaknesses
- The core evaluation framework of ChemTable mainly adopts existing standard metrics such as TEDS, Edit Distance, and Accuracy. While these metrics are reliable, they are not tailored to capture the unique characteristics of chemical tables (e.g., molecular structures, symbolic notation, multimodal relationships). The evaluation approach appears to be a direct transfer from general table recognition tasks, lacking methodological innovation specific to the chemistry domain.
- As a benchmark, ChemTable focuses almost exclusively on general-purpose multimodal large language models (MLLMs), while lacking systematic evaluation of domain-specific models for chemistry or broader scientific applications. This omission limits its value as a professional reference benchmark for “chemical table” understanding and recognition.
- Many tasks within the benchmark (such as Table Recognition, Title Description, Annotation Description, Yield and Conditions, Value Comparison, Find Min/Max, and Multi-hop Retrieval) already achieve high performance across mainstream MLLMs, with most models scoring above 80. This raises a key concern: if general models already perform well on these tasks, do these benchmarks still provide sufficient discriminative power or research value?
- Notably, Gemini-2.5-Flash, a small-sized model, achieves the best or near-best performance in most tasks, which raises an important question: if a “smaller” model performs this well, would flagship models such as Gemini-2.5-Pro, GPT-5, or Claude-4 easily surpass the current results? If so, does the benchmark still provide meaningful evaluation or differentiation between stronger models?

### Questions
Please refer to the weakness part above.

### Soundness
2

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
3

### Summary
The paper proposes a benchmark for evaluating multimodal LLMs on chemical table recognition and understanding. The dataset spans ~1,300 annotated chemical tables from literature with 9000 QA instances to measure table understanding capabilities. The work evaluates multiple MLLMs on the benchmark, revealing performance gaps in molecular structure recognition and domain reasoning compared to human performance.

### Strengths
Novel benchmark focused on a challenging domain of chemical table recognition and reasoning

Rigorous annotational protocol incorporating both manual and synthetic data generation

Comprehensive evaluation across multiple open-source and proprietary MLLMs

### Weaknesses
Limited dataset scale. Although the benchmark incorporates multiple table types, the overall number of samples remains modest, with 41.4% of the tables related to “Condition Optimization”. These limitations can constrain the generalizability of the findings. It would be good if the authors can provide some discussion of sampling bias and coverage across chemistry subfields.

Questions distribution. The paper does not report the distribution of QA instances across tables. Given that multiple filtering steps are applied, how did the authors ensure that evaluation metrics are not biased by skewed QA density? It would be helpful to include per-table or per-category QA statistics after filtering.

Qualitative error analysis. Although the paper provides metrics for different subdomains of the benchmark (e.g. by question type, molecular complexity), it would be helpful to include a few concrete failure case studies with visualizations to clarify where the models break.

### Questions
I'm not so sure about the contribution—the authors propose a dataset for OCR in chemistry; overall, it seems useful, but not for the broader community. Looks like a paper for Chemoinformaics journals.

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
3

### Summary
The paper proposes ChemTable, a benchmark consists of chemical tables with various contents. The core tasks include table recognition and table understanding. Evaluations are carried out for several closed-source or open-source multi-modal models, showing that they consistently fall short in complex tasks such as handling molecular structures and symbolic conventions.

### Strengths
The authors construct a real-world chemical table benchmark with expert annotations. The evaluated tasks are diverse, and the paper presents adequate number of experimental observations that seems reasonable.

### Weaknesses
Major points:

* The size of benchmark seems to be inadequately large. 
* The evaluated models are not state-of-the-art. What are the performance of more powerful models, such as Claude-4.0, GPT-5, Gemini-2.5-Pro? Also, the paper should include MLLMs specifically finetuned for chemistry tasks, such as ChemVLM.
* The presentation form of tables is always image. Did you try out tabular data form and study the performance of relational foundation models? Or using text descriptions for symbolic elements / graph representations for molecular structures when applicable?
* This is a purely benchmarking paper, and the authors fail to provide theoretical justifications or insights in depth. In particular, for those open-sourced models where one can observe the reasoning patterns inside the models, can you provide any analysis?

Minor points: 

* Typo: "Claude-3-7-Sonnet" should be "Claude-3.7-Sonnet" in Table 3.

### Questions
See above

### Soundness
2

### Presentation
3

### Contribution
2
