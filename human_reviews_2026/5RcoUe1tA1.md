# SC-Arena: A Natural Language Benchmark for Single-Cell Reasoning with Knowledge-Augmented Evaluation

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 2

## Abstract
Large language models (LLMs) are increasingly applied in scientific research, offering new capabilities for knowledge discovery and reasoning. In single-cell biology, however, evaluation practices for both general and specialized LLMs remain inadequate: existing benchmarks are fragmented across tasks, adopt formats such as multiple-choice classification that diverge from real-world usage, and rely on metrics lacking interpretability and biological grounding.  We present **SC-ARENA**, a natural language evaluation framework tailored to single-cell foundation models. SC-ARENA formalizes a *virtual cell* abstraction that unifies evaluation targets by representing both intrinsic attributes and gene-level interactions. Within this paradigm, we define five natural language tasks — cell type annotation, captioning, generation, perturbation prediction, and scientific QA — that probe core reasoning capabilities in cellular biology.  To overcome the limitations of brittle string-matching metrics, we introduce **knowledge-augmented evaluation**, which incorporates external ontologies, marker databases, and scientific literature to support biologically faithful and interpretable judgments.  Experiments and analysis across both general-purpose and domain-specialized LLMs demonstrate that: (i) under the *Virtual Cell* unified evaluation paradigm, current models achieve uneven performance on biologically complex tasks, particularly those demanding mechanistic or causal understanding; and (ii) our knowledge-augmented evaluation framework ensures biological correctness, provides interpretable, evidence-grounded rationales, and achieves high discriminative capacity, overcoming the brittleness and opacity of conventional metrics. **SC-ARENA** thus provides a unified and interpretable framework for assessing LLMs in single-cell biology, pointing toward the development of biology-aligned, generalizable foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SC-ARENA, a new benchmark to evaluate large language models (LLMs) in single-cell biology. The core ideas are:
1. Virtual Cell abstraction
Treats a model as if it were a “virtual cell.”
The “cell” has attributes (identity/state) and methods (how it responds to the environment).
This unifies evaluation across different biological tasks instead of testing each task in isolation. 
2. Five natural language tasks
Each one probes something biologically meaningful:
Cell Type Annotation (CTA): Given expression → predict ontology cell type.
Cell Captioning (CC): Given expression → describe the cell in natural language.
Cell Generation (CG): Given a cell type description → generate a plausible “cell sentence” (i.e. pseudo-expression profile).
Perturbation Prediction (PP): Given baseline + perturbation, predict changes and post-perturbation state.
Scientific QA (SQA): Answer mechanistic, literature-grounded biological questions. 
These map both static identity and dynamic behavior (how a cell changes under interventions), which is exactly what one would expect from a “virtual cell.” 
3. Knowledge-augmented evaluation
Instead of BLEU / ROUGE / exact match, which fail badly in biology, SC-ARENA uses an LLM-as-a-judge but grounds the judge in external biological knowledge (Cell Ontology, Gene Ontology, UniProt, CellMarker, PubMed evidence, etc.).
The evaluator scores answers and produces an interpretable rationale with references, not just a number. 
They show this judgment correlates with real biological hierarchy (e.g. closer ontology terms → higher score; Spearman ρ≈0.62, p<0.001). 
4. Empirical study
Benchmarks general LLMs (Qwen2.5/3, GPT-4o, DeepSeek-R1, Kimi-K2) and domain-specific single-cell models (C2S, scGPT, scGenePT, Cell-O1).
Finds: no model is good at everything; general models are fluent but biologically shaky, and domain models are precise in narrow skills but weak elsewhere. 
For example, captioning and science QA get the best scores (~60–74/100 for top models), but perturbation prediction and mechanistic reasoning remain very weak (<38/100). 
Small domain models like C2S actually beat giant general LLMs on cell type annotation, which shows specialization can outperform sheer parameter count for grounded biology.

### Strengths
1. Unified evaluation via the Virtual Cell abstraction
Instead of 5 unrelated leaderboards, SC-ARENA frames everything around one object: the Virtual Cell. The “attributes” vs “methods” split is elegant, and it matches how experimentalists think (cell identity vs response to perturbation). This is novel and, importantly, extensible. 
This framing makes it possible to talk about “does an LLM qualify as a virtual cell model?” instead of “did it get 2% better BLEU.” That’s a conceptual contribution, not just engineering.
2. Knowledge-augmented evaluation is thoughtful and genuinely useful
The work directly addresses a known failure mode of LLM evals in science: surface-similarity metrics like BLEU and ROUGE often (a) reward buzzwords, (b) fail to punish mechanistic nonsense.
Here, the evaluator:
retrieves ontology / marker / pathway / literature evidence, explains why it scored something, and aligns with biological hierarchy (ρ=0.6212 between ontology distance and evaluator score).

### Weaknesses
1. The benchmark leans heavily on LLM-as-a-judge, which is itself another model
Yes, they mitigate this with knowledge retrieval and ontologies. Yes, they validate correlation with ontology distance. But the scoring model is still an LLM (GPT-4o-mini), which raises standard questions:
How stable is the score across judge variants / seeds?
Could a model “game” the judge by mimicking citation-sounding language and pathway buzzwords?
Is there any adversarial testing (e.g., hallucinated but authoritative-sounding nonsense)?
They partially address alignment with expert judgments and ontology distance, but a deeper robustness audit (or inter-judge agreement across two different judges) would make the claim stronger. 

2.Ground truth for some tasks is underspecified
For perturbation prediction (PP), the model is supposed to say:
which genes go up/down, and
produce a plausible “post-perturbation cell sentence.”
But evaluating free-form gene-level differential expression is biologically thorny:
They say they use DEGs from Norman/Adamson and external knowledge bases (GO, UniProt, NCBI) to judge plausibility. 

What isn’t fully clear is: how do you distinguish a true novel hypothesis from a hallucination? If a model proposes a plausible but previously unreported compensatory pathway, does it get penalized or rewarded?

### Questions
1. Knowledge-Augmented Judging
What prevents the LLM-as-judge from being biased by language fluency rather than biological accuracy?
How reproducible are scores across judges (e.g., GPT-4o vs DeepSeek-R1) or seeds?
How do you ensure the judge doesn’t “reward” models that use familiar ontology terms but misstate mechanisms?

2. Dataset and Coverage
Are the 600-sample CellxGene subset and 138 perturbations enough to represent cellular diversity?
Could SC-ARENA generalize to datasets from other species, tissues, or modalities?
Are there biases toward well-studied cell types (e.g., immune, epithelial) due to ontology density?

3. Benchmark Design
Why were all five tasks weighted equally in the “Total Score”?
→ Would weighting causal tasks (e.g., perturbation prediction) more heavily yield a different ranking?
Did the authors test whether models overfit to the language style of prompts rather than underlying biology?
How much variance exists across repeated evaluations (inter-run consistency)?

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
3

### Summary
This paper introduces SC-Arena, a natural language benchmark for evaluating large language models (LLMs) in single-cell biology. The authors propose a unified “Virtual Cell” abstraction and design five biologically grounded tasks: cell type annotation, captioning, generation, perturbation prediction, and scientific QA. The evaluation leverages a knowledge-augmented LLM-as-a-judge framework, integrating external ontologies and databases to ensure interpretability and biological fidelity. Experiments compare both general-purpose and domain-specialized LLMs, revealing strengths and limitations in biological reasoning.

### Strengths
-  The Virtual Cell abstraction and multi-task natural language evaluation is an interesting idea to more objectively test different models’ capacity to understand cellular processes. 

- Integrating external biological knowledge (Cell Ontology, UniProt, GO, CellMarker, PubMed) into the evaluation pipeline is a major strength and a clever way to address the limitations of string-matching metrics.

- The paper benchmarks a wide range of models (Qwen, GPT-4o, DeepSeek-R1, Kimi-K2, C2S-Scale, scGenePT, scGPT, Cell-O1) and analyzes performance across tasks, model scales, and domains, using only open-source datasets.

### Weaknesses
- While the benchmark covers different tasks and the datasets are open-source, the paper does not address the risk of benchmark dataset leakage; such as, whether the datasets used to construct the SC-Arena benchmark were present in the pretraining or fine-tuning data of the evaluated models. 

- The rationale for model selection and the fairness of comparisons (e.g., fine-tuning protocols, input formats) could be better discussed. 

- The knowledge-augmented LLM-as-a-judge is promising, but its reliability and potential biases should be discussed further.

### Questions
- While the use of open-source datasets is commendable, how do the authors plan to address the risk of benchmark dataset leakage? For example, CS2 used the CELLXGENE dataset for training. Could this explain their performance on the CTA task?  

- Are there plans to expand the benchmark to include additional modalities or more challenging reasoning tasks? 

- It’s unclear how the architecture of the models will influence performance in these tasks. My main concern is that performance differences may reflect experimental artifacts rather than true differences in model capability. For example, a domain-specific model might outperform others simply because it was fine-tuned on data similar to the benchmark, or because its input format more closely matches the evaluation protocol. How do the authors plan to address this potential bias? Will this influence which models can or cannot be used for the benchmarking?
 
- The assessment of the knowledge augmented evaluator is shown only for the CTA task. How do the authors plan to validate the evaluator for the other tasks? How can we be confident that the evaluator is reliable and interpretable for these other tasks, which may involve different types of reasoning, output formats, and biological knowledge?

- As the evaluator is also a model, a proper evaluation of the model should be performed. For example, are there cases where the evaluator produces scores that do not align with expert human judgment, or where it fails to recognize biologically implausible or trivial answers? Does the evaluator systematically favor certain model architectures, output styles, or biological domains? Does it reward verbosity, penalize concise but correct answers, or show preference for models trained on similar data as the evaluator itself?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces SC-Arena, a benchmark designed to evaluate large language models (LLMs) as “virtual cells” capable of reasoning over biological knowledge and single-cell data. The benchmark includes multiple question types testing biological plausibility, reasoning consistency, and interpretability. It also introduces a knowledge-augmented evaluation strategy (Eval-RAG) that uses retrieval-augmented generation to penalize biologically implausible answers and reward semantically coherent responses beyond exact-match metrics.

### Strengths
•	Very creative and well-motivated benchmark that treats LLMs as reasoning agents over biological cell states.
•	The Eval-RAG strategy is an elegant idea that improves evaluation by incorporating biological context and semantic plausibility, moving beyond token-level correctness.
•	The paper provides a valuable framework for comparing different LLMs under biologically grounded tasks.
•	The paper does extensive evaluation of general and domain-specific models.
•	Discussion includes relevant next steps for producing a more biologically robust evaluation benchmark.

### Weaknesses
The paper would benefit from more detailed examples—for instance, elaborating on the process shown in Figure 2, panel B, to clearly explain how the biological plausibility scoring is computed step by step.

### Questions
How sensitive is Eval-RAG to the retrieval source—does the choice of biological database or text corpus significantly change the evaluation outcome?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work proposes an evaluation framework called Virtual Cell which seeks to unify the assessment of LLMs performance on various sub tasks important for single cell analysis like cell type annotation, captioning etc. The authors then evaluated various LLMs on a dataset derived from combining publicly available single-cell databases.

### Strengths
1. Combining vast amounts of single-cell data with natural language based knowledge available to gain insights into cellular function is beneficial to the biology community so this is a timely topic. 
2. Authors have benchmarked their proposed framework for evaluating LLM performance across many different LLM models or other domain specific models. 
3. Definition of the knowledge cell class is well thought out in considering multiple sources of information available for analyzing cellular dynamics.

### Weaknesses
- The novelty/value of this framework for evaluation is unclear, many current models like Cell2Sentence already combine single cell rna data with text based information and have shown use cases for downstream tasks like cell type prediction, perturbation response prediction etc. 
- Considering existing methods that can perform some of the tasks mentioned in the multi-task benchmark like Cell2Sentence, CellReasoning etc, to fully evaluate this work, performance of existing methods on individual tasks should be included and discussed.

### Questions
- This research area is a useful application of LLMs for biological science but while the authors mention the previous works, proper comparisons to these methods is lacking. Authors should consider expanded the related work section needs to clarify the contributions of this paper in comparing to some of the works mentioned here. 
- Authors should consider incorporating some of the ideas suggested in the discussion on modeling, evaluating and scoring into this work to improve the contribution and novelty of this work.

### Soundness
2

### Presentation
2

### Contribution
2
