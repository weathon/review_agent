# HLStrans: Dataset for C-to-HLS Hardware Code Synthesis

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 4, 2, 4

## Abstract
High-Level Synthesis (HLS) enables hardware design from C/C++ kernels but requires extensive transformations, such as restructuring code, inserting pragmas, adapting data types, and repairing non-synthesizable constructs, to achieve efficient FPGA implementations. While large language models (LLMs) show promise in automating these transformations, progress has been limited by the absence of large-scale, well-structured datasets. Existing HLS datasets focus primarily on resource estimation, lack paired C/HLS examples with testbenches, and cover only a narrow set of optimizations.
We introduce HLStrans, the first benchmark-scale dataset for LLM-driven C-to-HLS synthesis. HLStrans contains over 124K paired C/HLS programs for real-world applications, with full testbenches and synthesis-based annotations of latency and resource usage. The dataset systematically captures five categories of transformations and is enriched by an automated augmentation pipeline combining LLMs, Monte Carlo Tree Search (MCTS), and Design Space Exploration (DSE). We benchmark state-of-the-art LLMs on HLStrans, demonstrating that retrieval and fine-tuning significantly improve success rates and performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces HLStrans, a dataset containing 124K paired C/HLS programs for training LLMs to transform C code into optimized High-Level Synthesis (HLS) code for FPGAs. The dataset is created by collecting 309 base programs from various sources and augmenting them using an automated pipeline combining LLMs, Monte Carlo Tree Search (MCTS), and Design Space Exploration (DSE). The authors benchmark various LLMs showing improvements from retrieval and fine-tuning.

### Strengths
(1) Target is real and timely: LLMs for C→HLS need paired before/after code plus testbenches.
(2) Dataset is well-scaffolded around real HLS transformations, which is closer to what actual HLS engineers do than “just insert a pragma.”
(3) They try to close the loop with EDA feedback (Vitis), which is the right direction for LLM-in-the-loop hardware optimization.

### Weaknesses
(1) The augmentation relies entirely on DeepSeek-R1 and automated synthesis. There is no mention of expert validation of the generated samples' quality or correctness.
(2) All synthesis on Xilinx Alveo U55C, and thus it is unclear if insights transfer to other FPGAs (Intel, Lattice, etc.).

### Questions
(1) Please make an explicit comparison to Forgebench (2025): what does HLStrans offer that Forgebench doesn’t, besides testbenches? Do you cover more transformation types, or just organize them differently?

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
This paper introduces HLStrans, a large-scale benchmark dataset for C-to-HLS (high-level synthesis) code generation, targeting the automatic transformation of standard C/C++ kernels into synthesizable, hardware-optimized HLS code. The dataset comprises over 124,000 paired C/HLS programs, complete with testbenches and synthesis-based performance/resource annotations. The authors present a three-stage pipeline for dataset construction involving open-source collection, automated augmentation using LLMs, Monte Carlo Tree Search (MCTS), and design space exploration (DSE). The paper benchmarks multiple LLMs and demonstrates that fine-tuning and retrieval-augmented techniques using HLStrans result in clear improvements in synthesis rates, optimization, and code quality.

### Strengths
1. HLStrans provides a benchmark-scale, well-structured dataset for C-to-HLS transformation that includes paired pre/post-HLS code, testbenches, and synthesis-based resource/latency feedback. Compared to previous datasets (Table 1), HLStrans has greater diversity, size, and supports a broader range of transformation tasks.

2. Figure 1 concretely illustrates the transformation space covered (T1–T5), moving beyond mere pragma insertion (the focus of most prior works) to include code restructuring, data-type adaptation, algorithm repair, and function remapping.

### Weaknesses
1. While empirical results are solid, there is a lack of explicit theoretical analysis linking the diversity/scope of the dataset to expected generalization properties of LLMs trained/fine-tuned on it. There is little quantitative discussion on statistical diversity or representativeness of the underlying C/HLS patterns, which is crucial if the dataset is to become a benchmark standard.

2. The paper overlooks Wan et al. (2024), which introduced the Chrysalis dataset — an LLM-aided framework for HLS defect generation and functional verification. While Chrysalis focuses on bug injection rather than optimization, it is one of the earliest datasets coupling LLMs with HLS code transformation, synthesis feedback, and verification. Failing to acknowledge or contrast with this work substantially weakens the claimed novelty of HLStrans as the “first LLM-oriented C-to-HLS dataset.”

3. Figure 5 reports extremely large “speedup (×)” values, but the paper does not specify how latency was measured or controlled. Since HLS latency varies across synthesis runs and tool settings, the lack of a fixed evaluation protocol or success-rate reporting makes these results difficult to interpret or reproduce.

### Questions
1. Can the authors provide more quantitative evidence about the dataset’s diversity/coverage—e.g., statistical measures on code structure, transformation frequency, or functional domain clustering?

2. How was the latency in Figure 5 obtained—were multiple synthesis runs averaged, and were failed or unsynthesizable cases excluded from the reported speedups?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents HLStrans, a dataset for LLM-driven C-to-HLS synthesis. The main contributions of HLStrans include: open-source HLS examples with testbenches, an augmentation pipeline to generate diverse designs, and evaluation of LLMs on success rates and performance.

### Strengths
* Datasets for HLS are important and urgently needed for the EDA community
* Well-written background about HLS toolflows

### Weaknesses
* This work specifically evaluates a single HLS tool; how about other HLS tools, such as Bambu HLS, Cadence Stratus HLS, and Siemens Catapult HLS?
* The paper is missing reasoning about functional failure and synthesis failure. Are they caused by the same problem? It would be helpful to have classifications of failures and break down the importance of these failures over the benchmarks.
* This work tries to cover the whole HLS flow but misses a lot of ablation studies at each HLS pipeline stage.
* I am not sure if the data augmentation evaluation is sound. Why is a higher percentage better in this case? For example, 100% of the programs involve T2 - not all programs need to be unrolled. It seems less diverse to me.
* The paper is missing details about test bench generation. How are the test data generated for the test bench? What is the coverage over different hardware interfaces?

### Questions
Please see above.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents HLStrans, a large-scale dataset for C-to-HLS code transformation. The dataset includes over 124K paired C and HLS programs with testbenches, covering varying transformation types. It introduces an automated augmentation framework that combines LLMs, MCTS, and DSE to generate optimized HLS variants. Experiments demonstrate that retrieval-based prompting and finetuning on HLStrans can notably improve synthesis success rates and latency reduction.

### Strengths
1. It proposes the first large-scale benchmark for C-to-HLS transformation, filling a missing piece in the LLM-assisted EDA field.

2. The automated augmentation framework demonstrates good soundness, and the integration of MCTS and DSE may provide insights for other LLM-assisted EDA tasks.

3. It conducts comprehensive benchmarking of multiple models and prompting strategies using meaningful synthesis-level metrics.

### Weaknesses
1. The major concern lies in the limited generalizable insights and scientific contributions this work provides to the community. Although it successfully demonstrates the application of LLMs, MCTS, and DSE tools in crafting an EDA dataset, it remains unclear how the findings can generalize to other problems or advance the broader fields of LLM and EDA research.

2. There already exist LLM4EDA datasets created through strategic prompting or design space search. The authors are expected to provide a methodological comparison to clarify whether their proposed approach represents a superior or more scalable pipeline for future LLM4EDA dataset construction.

3. It is also unclear whether C-to-HLS is a sufficiently meaningful task for researchers and practitioners in this domain. Given that HLS closely resembles C, HLS experts can readily add pragmas to convert C to HLS, while algorithm developers typically do not work on such conversions. Therefore, the C-to-HLS task, due to the strong similarity between the two languages, may be less impactful compared to more practical tasks like Verilog generation.

### Questions
My questions have been included in the weakness section. In addition, I would like the authors to address the following two questions:

1. Do the authors think that future LLM4EDA datasets should be crafted using the pipeline proposed in this work?

2. Which group of users would most benefit from the proposed C-to-HLS benchmark and the finetuned LLM?

### Soundness
3

### Presentation
3

### Contribution
2
