# LiveProteinBench: A Contamination-Free Benchmark for Assessing Models' Specialized Capabilities in Protein Science

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
In contrast to their remarkable performance on general knowledge QA, the true abilities of Large Language Models (LLMs) in tasks demanding deep, specialized reasoning, such as in protein biology, have yet to be thoroughly investigated. Current benchmarks suffer from critical deficiencies, such as data contamination due to outdated test sets, insufficient focus on essential protein-specific tasks, and a neglect of multimodal assessments. To resolve these issues, we introduce LiveProteinBench, a contamination-free, multimodal benchmark of 12 tasks for evaluating LLM performance on protein property and function prediction. Its central innovation lies in a test set composed exclusively of proteins validated after the start of 2025, guaranteeing that the data is novel to all tested models. We benchmarked a suite of prominent general-purpose LLMs and specialized biological LLMs using both unimodal and multimodal input schemes. Our results show that: 1) General-purpose proprietary large models demonstrate superior zero-shot performance when encountering new protein data, outperforming their open-source and domain-specific counterparts by over 20\% accuracy. 2) The effective use of multi-view structural information remains a significant challenge, as the inclusion of structural images often fails to provide a consistent benefit and can even degrade performance. This highlights the limitations of current models in effectively fusing information across different modalities. 3) Models' performance scales more directly with the computational cost during inference than with its parameter count, underscoring the critical role of Chain-of-Thought reasoning capabilities for protein-specific tasks. 
LiveProteinBench delineates the current performance frontiers for LLMs in bioinformatics and presents new challenges for the development of future multimodal foundation models for biology.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This manuscript presents LiveProteinBench, a benchmark designed to evaluate large language models (LLMs) on protein science tasks using a strictly contamination-free framework. It includes 12 biologically diverse tasks—ranging from function prediction to structural reasoning—based on proteins validated after January 1, 2025, ensuring no overlap with pretraining data. However, the paper suffers from key weaknesses, including limited methodological innovation, poor writing quality, and a lack of clarity regarding the dataset.

### Strengths
1-	The use of post-2025 protein entries ensures that none of the test data overlaps with pretraining corpora, addressing a critical issue in LLM evaluation: data leakage.
2-	Broad evaluation across general and domain-Specific LLMs.

### Weaknesses
1-	Lack of methodological innovation and dataset accessibility. The manuscript does not present any clear methodological innovation beyond the temporal filtering strategy. Furthermore, the authors do not provide access to the benchmark dataset. 
2-	Authors didn’t provide details on the proteins analyzed. They just mentioned selection criteria from public database, but any further biological information (e.g., biological diversity, sequence novelty, representative of real challenges, …etc). Therefore, it is challenging to assess the contamination-free claims. 
3-	The benchmark’s dataset size is limited, with only ~2,000 proteins. This is substantially smaller than other recent benchmarks such as PROBE and ProteinLMBench, which use tens of thousands of proteins and provide public access to data and evaluation pipelines. The small scale of LiveProteinBench may limit its generalizability and statistical robustness.
4-	Lack of fine-tuning or adaptation experiments especially for small-scale specialized models (SLLMs). Only zero-shot evaluation is reported; no exploration of how models could improve with task-specific training. This is a significant limitation, as recent literature [Schmirler et al. Nature Comm 2024] demonstrated that fine-tuning can substantially improve performance on protein-specific tasks
5-	The current version of the manuscript needs significant improvement in writing quality. The paper is densely written and lacks clear presentation and methodological articulation of key contributions such as multi-modal integration. For example, how were the protein images paired with sequence data? How multi-modal input was provided to the model? etc. In addition to vague sentences such as “Genuinely unlocking the secrets of life requires these models to move beyond merely processing sequence information and to demonstrate multiple advanced capabilities.”

### Questions
N/A

### Soundness
2

### Presentation
1

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
The paper introduces LiveProteinBench, a contamination-free and multimodal benchmark for assessing large language models’ (LLMs) capabilities in protein science. 

It features 12 structured tasks spanning protein function, structure, and physicochemical property prediction, built exclusively from UniProt entries validated after January 2025 to ensure no data leakage. 

The authors benchmark over 10 general-purpose and protein-specific models, revealing that general-purpose models (e.g., GPT-5) outperform specialized ones and that multimodal integration of 3D protein structures remains a significant challenge.

### Strengths
The paper is original in proposing a live, contamination-free design for benchmarking LLMs in biology. 

The methodology is rigorous, with carefully defined tasks, fair temporal splits, and reproducibility ensured through public databases. 

The clarity of the presentation and experimental analyses is high, and the results are significant

### Weaknesses
The multimodal evaluation relies on 2D structure projections, which may not fully capture 3D relationships; alternative encodings could be discussed. 

The benchmark focuses only on single-protein properties, omitting interactions or dynamics that are crucial in biological contexts. 

Evaluation metrics are limited to accuracy.

Limited discussions of related works such as [1, 2, 3]

[1] STELLA: Towards Protein Function Prediction with Multimodal LLMs Integrating Sequence-Structure Representations

[2] Proteingpt: Multimodal llm for protein property prediction and structure understanding

[3] Prot2Text-V2: Protein Function Prediction with Multimodal Contrastive Alignment

### Questions
How will LiveProteinBench be maintained to ensure continued contamination-free status as future models update? 

Could the authors provide evidence that the cutoff date fully excludes pretraining data from foundation model updates?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
LiveProteinBench is a contamination-free, multimodal benchmark specifically designed to assess LLM capabilities in protein science. The benchmark includes 12 diverse tasks across functional annotation, structural localization, and physicochemical property prediction, using only post-2025 data from UniProt to ensure no pretraining contamination. The authors evaluate >10 general-purpose and domain-specific models using both sequence-only and sequence+structure modalities.

### Strengths
- LiveProteinBench addresses major flaws in existing protein evaluation (contamination, outdated tasks, lack of multimodality) with rigorous dataset construction and “live data” principle.

- The benchmark offers 12 well-structured tasks grounded in validated annotations; task variety enables broad assessment of biological reasoning.

### Weaknesses
- The evaluations are zero-shot. It would be valuable to see whether task-tuned or instruction-fine-tuned models can close the generalist-specialist gap.

### Questions
- Are there some qualitative examples where structure helped vs. harmed performance to better understand the fusion bottleneck?

### Soundness
2

### Presentation
3

### Contribution
2
