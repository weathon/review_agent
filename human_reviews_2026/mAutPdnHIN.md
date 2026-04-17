# ASyMOB: Algebraic Symbolic Mathematical Operations Benchmark

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Large language models (LLMs) are increasingly applied to symbolic mathematics, yet existing evaluations often conflate pattern memorization with genuine reasoning. To address this gap, we present **ASyMOB**, a high-resolution dataset of **35,368** validated symbolic math problems spanning integration, limits, differential equations, series, and hypergeometrics. Unlike prior benchmarks, **ASyMOB** systematically perturbs each seed problem using symbolic, numeric, and equivalence-preserving transformations, enabling a fine-grained assessment of generalization and robustness.
Our evaluation reveals three key findings: (1) most models’ performance collapses under minor perturbations, while frontier systems exhibit substantial robustness, suggesting an emerging *"phase transition"* from memorization to generalization; (2) integrated code tools stabilize performance, particularly for weaker models; and (3) we identify examples where Computer Algebra Systems (CAS) fail while LLMs succeed, as well as problems solved only via a hybrid LLM-CAS approach, highlighting a promising integration frontier.
**ASyMOB** serves as a principled diagnostic tool for measuring and accelerating progress toward building verifiable, trustworthy AI for scientific discovery.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel benchmark for LLM mathematical reasoning, namely ASyMOB. This benchmark perturbs existing problems using symbolic, numeric, and equivalence-preserving transformations to ensure robust evaluation. Several key insights are identified to guide the future development of mathematical LLM.

### Strengths
- The benchmark is reasonable, as the symbolic and numeric versions can fully evaluate the ability of LLMs to address mathematical reasoning.
- The provided examples are well-motivated, as identifying cases where both LLMs and symbolic systems do not perform well can help guide further research directions.

### Weaknesses
- The novelty of this paper requires further clarification. As noted at the end of this paper, GSM-Symbolic has conducted similar research and reached comparable conclusions. Therefore, it is important for the authors to clearly articulate the unique contribution and positioning of this work within the field, especially given the prior work, i.e., GSM-Symbolic. The authors should carefully clarify the difference between their benchmark and existing work. Additionally, some closely related studies are missing from the discussion, such as [1], which also employs neuro-symbolic methods for data generation. Including and discussing such relevant literature would further strengthen the paper. 
- The conclusions and findings presented in this paper are trivial. For example, the integration of code tools has already been discussed in [2], and this is not a significant discovery or a new insight within the domain of LLM reasoning. In fact, numerous studies on agentic workflows [3] have previously explored approaches that enable LLMs to utilize a variety of tools.
- The presentation of this paper has significant room for improvement. For example, the paper ends with a lot of blank space. Some technical details are missing, such as the details of equivalence-preserving transformations, which are not clearly presented.

[1] Zenan Li, Zhi Zhou, Yuan Yao, Xian Zhang, Yu-Feng Li, Chun Cao, Fan Yang, Xiaoxing Ma. Neuro-Symbolic Data Generation for Math Reasoning. NeurIPS 2024.

[2] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, Graham Neubig. PAL: Program-aided Language Models. ICML 2023.

[3] Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, Chenglin Wu. AFlow: Automating Agentic Workflow Generation. ICLR 2025.

### Questions
Please refer to the weaknesses section. Additionally,
1. The list of equivalence-preserving transformations provided in the appendix appears to be too limited. It is unclear whether this small set of transformations is sufficient to comprehensively evaluate LLM capabilities, or whether it merely identifies a few isolated cases that might temporarily "hack" or exploit LLM behavior.
2. The evaluation costs associated with this benchmark should be further clarified. As a benchmark, the evaluation process should be both easy and affordable for researchers to carry out. It should also be clear whether the benchmark is sufficiently inexpensive to allow for comprehensive evaluation, or whether results obtained from evaluating only a subset of the benchmark are truly representative of overall performance.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents ASyMOB (Algebraic Symbolic Mathematical Operations Benchmark), a large-scale benchmark for evaluating large language models (LLMs) on symbolic mathematical reasoning. ASyMOB contains 35,368 verified problems generated from 100 seed questions through systematic symbolic, numeric, and equivalence-preserving perturbations, covering integration, limits, differential equations, series, and hypergeometric functions. These perturbations maintain mathematical equivalence while testing robustness to structural and numerical variation. The authors evaluate multiple open- and closed-weight LLMs (e.g., GPT-4o, Gemini 2.5, DeepSeek-R1) and find substantial performance degradation under minor perturbations (average accuracy drops from 74.6% to 46.8%).  However, frontier models show notably higher robustness, suggesting an emerging transition from memorization to true symbolic generalization. Additionally, ASyMOB reveals that LLMs sometimes outperform traditional CAS systems, and that hybrid LLM + CAS approaches can solve problems unsolvable by either alone.

### Strengths
- ASyMOB isolates symbolic mathematical reasoning from linguistic understanding, providing a clean test of algebraic manipulation skills.
- The symbolic, numeric, and equivalence perturbations enable fine-grained evaluation of robustness and generalization.
- Dual symbolic–numeric verification ensures reliability, and the findings reveal meaningful trends such as a "phase transition" toward genuine reasoning in frontier LLMs.

### Weaknesses
- The scope is somehow limited. The benchmark focuses narrowly on algebraic operations, omitting other mathematical reasoning domains, such as geometry or proofs.
- Some generated variants may be mathematically artificial and not representative of real-world symbolic problems.
- Several key conclusions, such as the role of code integration and hybrid tool use in improving LLM reasoning, have already been explored in prior work on tool-augmented or agentic LLMs, making the contributions more incremental than novel.

### Questions
1. How would the proposed perturbation framework generalize to other mathematical domains, such as geometry, proofs, or word problems that involve both symbolic and linguistic reasoning?
2. The paper interprets the robustness of frontier models as evidence of a "phase transition" from memorization to genuine symbolic reasoning. However, how do the authors rule out the possibility that these models simply memorize or interpolate over a much larger region of symbolic patterns, rather than performing true reasoning?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces ASyMOB, a 35,368 problem benchmark for symbolic mathematics (integration, limits, differential equations, series, etc.) built to test symbolic reasoning by systematically perturbing seed problems and then validating answers via equivalence-aware 
symbolic and numeric checks. For every seed problem, they obtain variants of controlled difficulty through three methods: 1) symbolic perturbations, which perturb N symbols; numeric variants, which replace all or exactly one symbol with N-digit integers; and equivalence variants, which insert identities equal to 1 (trigonometric, hyperbolic, logarithmic, complex-exponential, and series). They further note that the dataset can be re-generated before assessing a new LLM, making their dataset more resilient against benchmark hacking or memorization. Finally, they find that 1) models' performance drops even under small perturbations, with frontier models being more robust, 2) incorporating code tools stabilizes performance, especially for weaker models, and 3) they observe instances where CAS fails but LLMs succeed and problems that can only be solved by combining both.

### Strengths
The work clearly explains how problems are built and expanded into symbolic, numeric, and equivalence variants, with worked examples for each. 
* ASyMOB fills gaps in existing literature dataset, targeting symbolic manipulation (integration, limits, DEs, series, hypergeometrics) rather than text-to-math. It offers controlled difficulty via systematic perturbations and broad university-level problem coverage that previous benchmarks lack. 
* Dataset instances are created with random transforms, and the dataset can be re-generated before evaluation. This design reduces leakage/memorization risk compared to static test sets. 
* The work illustrates that models’ performance degrades sharply under small perturbations, while frontier models are more robust. Tool use helps weaker models, and hybrid LLM + CAS solves cases where either alone fails.

### Weaknesses
* The work documents qualitative examples where CAS fails but LLMs succeed, and a case solvable only by an LLM + CAS hybrid (Figure 6). Further, it argues that symbolics hurt CAS more than LLMs. What’s missing is a dataset-level percentage/table partitioning successes into LLM-only, CAS-only, and hybrid categories across perturbations. Adding this would substantively strengthen the claim.
* Some of the perturbations appear to be somewhat contrived. This may not necessarily be a bad thing, but it could mean that the benchmark is testing perturbations that would not reasonably appear in the real world. This is related to my first question below.

### Questions
* Does performance on ASyMOB correlate with performance on other real-world tasks? E.g., performance on math contests that occurred after model releases.
* Would training/fine-tuning on ASyMOB perturbations lead to stronger models?

### Soundness
3

### Presentation
2

### Contribution
2
