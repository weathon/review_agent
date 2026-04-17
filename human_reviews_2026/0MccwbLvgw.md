# FMC: Formalization of Natural Language Mathematical Competition Problems

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Efficient and accurate autoformalization methods, which leverage large-scale datasets of extensive natural language mathematical problems to construct formal language datasets, are key to advancing formal mathematical reasoning. In this paper, we propose an autoformalization pipeline based on large language models with error feedback for syntax verification and cost-free problem decomposition for semantic alignment check, achieving a fully automatic and training-free formalization approach. Using this pipeline, we curate an Olympiad-level dataset aligning natural language problems with Lean formalizations. The dataset contains $3,214$ natural language mathematical problems and $6,994$ corresponding Lean statements, indicating a one-to-many relationship where a single problem may map to multiple formal representations. This dataset is well-suited as a benchmark for automated theorem provers. Additionally, we investigate the formalization and reasoning capabilities of various LLMs and empirically demonstrate that problem decomposition, few-shot learning and error feedback are key components to enhance the autoformalization process.
Experiments of three automated theorem provers on the \dataset\ dataset also highlight its challenging nature and its value as a benchmark for formal reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces FMC, an LLM-based autoformalization framework that integrates few-shot prompting, error feedback for syntactic verification, and problem decomposition for semantic alignment. The authors also construct a dataset of 6,994 Lean statements derived from 3,214 Olympiad-level math problems. Experimental results demonstrate that few-shot prompting, error feedback, and decomposition each contribute effectively to improving LLM-based autoformalization performance.

### Strengths
* The paper is easy to follow, with a clear motivation.
* The proposed dataset could be valuable for training LLMs in both autoformalization and theorem proving tasks.

### Weaknesses
The proposed method lacks novelty. Techniques such as few-shot prompting, syntactic correction through error feedback, and semantic verification have already been explored in prior work [1, 2, 3, 4].

Moreover, the autoformalized statements produced by the system are only guaranteed to be syntactically valid, with semantic consistency checked solely by LLMs. There is no manual verification to ensure that the formal statements truly align with their original natural-language counterparts. In many cases, autoformalization may omit implicit assumptions present in the original problem statements, leading to incorrect or unprovable formulations—despite being syntactically correct. This issue has also been observed in prior work such as DeepSeek Prover, which demonstrates that some automatically formalized statements are actually false (which can be disproved). This could partially explain why existing LLMs achieve relatively low accuracy on the FMC benchmark.

In addition, the baseline comparison is limited to a vanilla LLM, which may only serve as an ablation rather than a true comparison. Stronger and more representative baselines in this area should be considered to provide a fair evaluation.

Finally, the lack of manual validation across the entire dataset raises concerns about the reliability of the proposed benchmark. As shown in Table 4, the automated semantic checking is not highly reliable and shows only weak alignment with human judgments, which further limits the scope and credibility of the dataset.

References:

[1] Autoformalization with Large Language Models

[2] Autoformalize Mathematical Statements by Symbolic Equivalence and Semantic Consistency

[3] Improving Autoformalization using Type Checking

[4] KELPS: A Framework for Verified Multi-Language Autoformalization via Semantic-Syntactic Alignment

### Questions
How many samples were used for manual evaluation in Section 5.2?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces an autoformalization pipeline that integrates problem decomposition, few-shot learning, and error feedback to translate natural language mathematical competition problems into Lean 4 formalizations. Using this approach, the authors build FMC, an Olympiad-level dataset aligning natural language problems with their formal Lean counterparts. The FMC dataset contains 3,214 problems and 6,994 corresponding Lean statements. Experimental results highlight its difficulty and demonstrate its potential as a valuable benchmark for advancing research in formal reasoning.

### Strengths
- The overall idea of the paper is well-motivated and technically sound. The combination of few-shot learning and error feedback effectively enriches contextual information, which can improve the robustness and accuracy of the autoformalization process. Moreover, problem decomposition bridges the gap between natural language expressions and their formal counterparts in Lean, facilitating more reliable semantic alignment and verification.
- The paper conducts a thorough and systematic ablation study. The experiments clearly isolate and quantify the contributions of each component—problem decomposition, few-shot learning, and error feedback—providing strong empirical evidence for the effectiveness of the proposed pipeline.
- The work also offers practical value beyond methodology. By producing a large-scale Olympiad-level dataset (FMC), it contributes a challenging and well-structured benchmark for evaluating formal reasoning systems and automated theorem provers.

### Weaknesses
- The paper contains several presentation inconsistencies that detract from readability and professionalism. For instance, Table 4 is never cited in the main text, and the captions for some figures and tables are overly minimal. Captions such as that of Figure 1 should include a more detailed explanation of the depicted pipeline. Additionally, there is a numerical inconsistency in Section 4.3: the values 3,922 and 87.14% appear only once, whereas 3,214 and 66.99% are used elsewhere. This discrepancy should be clarified and corrected.
- The workflow for this method lacks sufficient detail. Conceptually, it involves two distinct semantic judgments:
(a) verifying whether the decomposed subproblems remain faithful to the original natural language statement, and
(b) checking whether these decomposed elements semantically correspond to the resulting Lean 4 formalization.
The current text does not clearly distinguish these steps, leaving the reader uncertain about the precise evaluation process. Section 5.2 should be expanded to provide a clearer methodological description and include supporting experimental evidence.
- The paper would benefit from a dedicated section describing the dataset itself: its composition, domain coverage (e.g., algebra, number theory, combinatorics), and its comparison with existing datasets. Although comparisons are provided in the appendix, they are never referenced or summarized in the main body, which weakens the paper’s contribution as a benchmark dataset.
- The appendix contains valuable analyses, including dataset comparisons and case studies, but these are not mentioned or cited in the main paper. This omission results in a structural disconnect, rendering the supplementary material underutilized. Integrating references to the appendix within the main text would improve coherence and readability.

### Questions
1. The discussion of prior autoformalization research appears rather limited, referencing only [1]. Several important related works are missing; please refer to the recent surveys [2, 3] for a broader overview. In addition, comparative experiments with other recent autoformalization pipelines and tools would strengthen the paper by clarifying how the proposed approach differs from and improves upon existing frameworks.

2. The paper states that state-of-the-art provers include DeepSeek-Prover-V2 and Goedel-Prover-V2, yet the experiments rely on DeepSeek-Prover-V1.5-RL and Goedel-Prover. Could the authors elaborate on the rationale behind this choice? For completeness, it would also be valuable to include STP, which is cited in the paper and represents another leading system. Incorporating results from these newer models would provide a more comprehensive and up-to-date evaluation of the proposed FMC benchmark.

[1] Autoformalization with large language models, NeurIPS 2022.

[2] A Survey on Deep Learning for Theorem Proving, COLM 2024.

[3] Autoformalization in the Era of Large Language Models: A Survey, arXiv 2025.

### Soundness
3

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
This paper presents FMC, an autoformalization pipeline that translates natural-language mathematical competition problems (mainly Olympiad-level) into the Lean formal language. The proposed approach integrates LLMs with error feedback and problem decomposition to achieve a fully automatic, training-free formalization process. Using this pipeline, the authors construct a dataset of 3,214 natural-language problems and 6,994 corresponding Lean statements, achieving 93.4% syntactic validity and 66.9% semantic consistency, both surpassing prior work (e.g., StepFun-Formalizer). The paper further evaluates multiple LLMs (DeepSeek-R1, GPT-4o-mini, Claude 3.7 Sonnet) and several automated theorem provers on the dataset, showing FMC’s challenging nature and its potential as a benchmark for formal reasoning.

### Strengths
- The proposed combination of error feedback and problem decomposition provides a well-motivated and empirically validated improvement over prior autoformalization frameworks. The training-free design is elegant and practical.
- The dataset focuses on Olympiad-level problems, ensuring both semantic richness and difficulty. The reported statistics (93% syntactic, 67% semantic accuracy) are impressive given the full automation and task complexity.
- The paper conducts detailed comparisons across multiple models (DeepSeek, GPT-4, Claude) and theorem provers (Kimina, Goedel, DeepSeek-Prover). This breadth of evaluation makes FMC a valuable benchmark for future research in formal mathematical reasoning.

### Weaknesses
- While the combination of known components (few-shot prompting, error feedback, decomposition) works well, the conceptual novelty is somewhat incremental compared to existing autoformalization frameworks such as StepFun-Formalizer or KELPS.
- The paper includes some case studies, but a deeper quantitative or typological analysis of the 33% semantic failures would strengthen the understanding of model limitations and inform future improvements.
- The paper does not specify whether the dataset, code, or prompts will be released. Given its claimed benchmark status, public release (and clear licensing details) would be essential for meaningful community adoption.

### Questions
The proposed pipeline focuses on Olympiad-level problems with relatively well-structured text.

1. How well does FMC generalize to non-Olympiad mathematical texts (e.g., research papers, textbooks)?

2. Would the error feedback and decomposition mechanisms still work effectively on less formulaic or multi-step narrative problems?

The paper highlights that syntax error feedback improves formalization accuracy, while semantic error feedback may introduce noise. 

1. Could the authors elaborate on the computational cost and iteration depth of the feedback loop? 
2. How many rounds of retranslation were typically needed before convergence, and how does this trade off with the observed accuracy gains? 
3. Have the authors explored adaptive stopping criteria or selective feedback strategies to balance quality and efficiency?

### Soundness
2

### Presentation
2

### Contribution
2
