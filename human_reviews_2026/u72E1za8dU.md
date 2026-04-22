# $S^3$-Bench: A Comprehensive Study of Multimodal LLMs for Scientific Discovery with Benchmarking

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 0, 2, 6

## Abstract
Recent advances in artificial intelligence (AI), especially large language models, have accelerated the integration of multimodal data in scientific research. Given that scientific fields involve diverse data types, ranging from text and images to complex biological sequences and structures, multimodal large language models (MLLMs) have emerged as powerful tools to bridge these modalities, enabling more comprehensive data analysis and intelligent decision-making. This work, $\text{S}^3\text{-Bench}$, provides a comprehensive overview of recent advances in MLLMs, focusing on their diverse applications across science. We systematically review the progress of MLLMs in key scientific domains, including drug discovery, molecular \& protein design, materials science, and genomics. The work highlights model architectures, domain-specific adaptations, benchmark datasets, and promising future directions.  More importantly, we also conducted benchmarking evaluations of open-source models on several highly significant tasks, such as molecular property prediction and protein function prediction. Our work aims to serve as a valuable resource for both researchers and practitioners interested in the rapidly evolving landscape of multimodal AI for science.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a comprehensive overview of the application of Multimodal Large Language Models (MLLMs) in scientific discovery. The main body of the work surveys and categorizes existing MLLMs across four major scientific domains: Drug & Molecule science, Protein science, Genomics, and Materials science. The authors aim to provide a valuable resource for researchers by summarizing the rapidly evolving landscape of MLLMs for science.

### Strengths
The strengths are summarized below:

- The paper's primary strength is its breadth, as it covers four distinct scientific domains and organizes them under the unified theme of MLLMs. 
- The tables and figures represent a clear comparison of different MLLMs in terms of their release date, scale, and architecture. It could be a valuable resource for researchers new to these fields.
- The discussion of emerging topics, such as dLLMs for science, seems interesting.

### Weaknesses
The weaknesses of this work are shown below:

- This paper is fundamentally a survey paper, not an original research contribution. The vast majority of the paper (Sections 2-5) is a description and categorization of prior works. Therefore, this paper may not be suitable for the ICLR main track.
- The title is misleading. This paper does not propose any new benchmark. The "benchmarking" section is a limited-scope experimental study on pre-existing datasets, i.e., MoleculeNet and TAPE. This is a standard experimental validation section, not the contribution of a new benchmark.
- While this papers cover different areas, the review in the main sections often is more like a list of models rather than a deep, critical analysis. It provides little novel insight into why certain architectures are more successful than others, what the common failure modes are, or what fundamental principles unite MLLM design across these different scientific domains.

### Questions
Please refer to the weaknesses part above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The work does not present any new methodology or dataset but looks like a survey paper mainly focusing on existing model's metadata such as parameter count, release date, modalities trained on etc. As such I personally don't find it suitable for ICLR.

### Strengths
N/A

### Weaknesses
N/A

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents an extensive survey and empirical study of multimodal large language models (MLLMs) in scientific research and systematically consolidates developments across multiple scientific domains—including molecular and drug design, protein science, genomics, and materials science. The authors summarize representative model and methods, domain-specific adaptations, and commonly used datasets, providing ataxonomy of how MLLMs integrate textual, visual, structural, and 2D, 3D geometric modalities for each specific domain. The paper also highlights emerging research frontiers such as diffusion-based MLLMs. Its benchmarking component evaluates selected models (e.g., MoMu, MoleculeSTM, Token-Mol) on two tasks: molecular property and protein function prediction, illustrating comparative strengths and limitations.

### Strengths
1. The paper presents a systematic and interdisciplinary survey of multimodal large language models (MLLMs) for scientific discovery, covering four major domains—molecular science, protein science, genomics, and materials science. This breadth of coverage reflects both the maturity and the integrative nature of the review.

2.  The work provides an in-depth examination of diverse modalities involved in MLLMs, including textual sequences, 2D and 3D structural representations, and sequence-to-language transformations. It systematically summarizes representative methods and evolutionary trends within each modality, offering a well-organized reference framework for future research.

3. Beyond summarizing existing architectures, the paper also discusses emerging paradigms, expecially diffusion-based multimodal generation models and analyzes their potential applications across different scientific domains.

### Weaknesses
1. My major concern is the limited evaluation task coverage. The title of the benchmark is Scientific Discovery, but the task coverage is very limited. It seems that there is an overclaim about the paper. Although the authors conducted a comprehensive literature review across various subfields, the experimental evaluation is restricted to only two tasks—molecular property prediction and protein function prediction. This narrow focus limits the paper’s contribution to understanding and integrating domain-specific challenges and results. Expanding the benchmarking to include additional representative tasks (e.g., gene function prediction, materials property modeling) would significantly enhance the empirical depth and generalizability of the work. 

2. Insufficient analysis of benchmarking results. The paper reports the benchmarking outcomes with only brief descriptions and lacks detailed comparative or diagnostic analysis. It does not examine the reasons behind performance differences among models or analyze design factors such as architecture, modality fusion strategy, or training paradigm. Consequently, the study provides limited insight into task-specific findings. Future work should include more granular comparisons and discuss their respective advantages and limitations across modalities and tasks.

3. Absence of evaluation on leading proprietary models. The current experiments are conducted solely on open-source models, without including comparisons against strong proprietary models such as GPT-5 or Gemini-2.5, which have demonstrated advanced multimodal reasoning capabilities. The lack of such baselines weakens the completeness and external validity of the conclusions, as it remains unclear how these closed-source models would perform or what the potential upper bound of performance might be on such scientific multimodal tasks.

### Questions
1. Most existing MLLMs remain focused on text generation, rather than directly producing multimodal outputs such as images or structured visualizations. However, in scientific discovery, non-textual outputs can often be more intuitive and interpretable. Do the authors believe that multimodal output generation (e.g., visual depictions, molecular structures) is necessary for advancing scientific research? What do they consider to be the main challenges in achieving such multimodal outputs?

### Soundness
1

### Presentation
3

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
This paper presents a comprehensive survey of multimodal large language models (MLLMs) applied to molecular, protein, materials, and genomic scientific discovery. The authors catalogue and categorize recent approaches, discuss their conceptual differences and applicability, and highlight emerging trends across these domains.

### Strengths
The survey covers a broad set of application domains—molecules, proteins, materials, and genomics—and provides a useful overview of widely used and well-established methods.

The authors make an effort to explain conceptual distinctions between approaches, which helps readers understand trade-offs and design choices at a high level.

The manuscript is well written and structured in the manner typical for high-quality survey papers; it is easy to follow and adequate as an entry point into the field.

### Weaknesses
The manuscript would be substantially strengthened by adding more practical, quantitative comparisons and clearer systematization of methods.

The comparisons currently presented (Tables D1, D2 in the Supplementary) cover only a small subset of molecular and protein prediction tasks and omit materials and genomics.Generative capabilities of the surveyed models are not addressed. Many surveyed works report results on established benchmarks (e.g., MOSES, TDC, USPTO for 2D molecular tasks; GEOM-DRUGS, CROSSDOCKED for 3D molecular tasks). Compiling these publicly reported results into summary tables in the Supplementary Materials would provide essential, practitioner-oriented value and make the survey far more actionable.


I recommend a dedicated section that systematizes approaches by how they integrate modalities, for example: text-only LLMs [a], frozen domain-specific encoders [b, c], trainable domain-specific encoders [d, e], and fully custom multimodal architectures. Such a taxonomy will make conceptual differences clearer and simplify comparison across papers.


Separately overview the data representations used in the literature (e.g., SMILES/SELFIES/IUPAC for 2D molecules; atom coordinates/torsional-angle representations for 3D structures; sequence and structural encodings for proteins; genomic tokenizations). For each representation, briefly state strengths and limitations. This will lower the barrier for newcomers and improve the survey’s utility as a reference.

a. BindGPT: A Scalable Framework for 3D Molecular Design via Language Modeling and Reinforcement Learning, Zholus et al.

b. 3D-MOLM: Towards 3D Molecule-Text Interpretation in Language Models, Li et al.

c. Structure Language Models for Protein Conformation Generation, Lu et al.

d. nach0: Multimodal Natural and Chemical Languages Foundation Model, Kuznetsov et al.

e. 3DSMILES-GPT: 3D molecular pocket-based generation with token-only large language model, Wang et al.

### Questions
All important questions and suggestions are stated in Weaknesses section.

Minor:
Typo on lines 74–75: evluation → evaluation.

### Soundness
3

### Presentation
3

### Contribution
2
