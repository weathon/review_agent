# PRISM-Physics: Causal DAG-Based Process Evaluation for Physics Reasoning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Benchmarks for competition-style reasoning have advanced evaluation in mathematics and programming, yet physics remains comparatively underexplored. Most existing physics benchmarks evaluate only final answers, which fail to capture reasoning processes, while recent stepwise methods rely on heuristic LLM-as-judge scoring or restrictive linear assumptions, limiting reliability and diagnostic validity.
We introduce PRISM-Physics, a process-level evaluation framework and benchmark for complex physics reasoning problems. Solutions are represented as directed acyclic graphs (DAGs) of formulas, explicitly encoding causal dependencies among intermediate steps to enable fine-grained, interpretable, and theoretically grounded scoring. 
We prove the optimality of the DAG representation and the corresponding scoring policy. Combining with a fully rule-based method for symbolic formula equivalence matching that we developed, we ensure consistent validation across diverse formulations without heuristic judgments. Results show that our evaluation framework is more aligned with human experts' scoring. 
Experiments on state-of-the-art LLMs reveal persistent reasoning failures in physics, while step-level scoring offers both diagnostic insight and rich signals for later training. By combining structural rigor, theoretical guarantees, and symbolic validation, PRISM-Physics provides a principled foundation for advancing process-level evaluation and guiding the development of models with deeper scientific reasoning capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces PRISM-PHYSICS, a benchmark and a process-level evaluation framework that encodes physics solutions as DAGs and employs rule-based symbolic equivalence checking for reliable, fine-grained scoring.

### Strengths
1. A large-scale benchmark of competition-level physics problems with carefully curated, DAG-structured solutions. 
2. A DAG-based scoring policy that explicitly models causal dependencies among formulas, enabling fine-grained and interpretable process-level evaluation.
3. A fully rule-based symbolic formula equivalence checker to reliably validate diverse mathematical expressions, ensuring consistent comparison across alternative formulations and eliminating reliance on heuristic LLM-as-judge scoring.

### Weaknesses
1. My main concern is that Figures 1 through 5 are very unclear, and even when enlarged twice, they are still hard to read. These figures should ideally be the most direct representation of the data analysis in this study, PRISM-PHYSICS. I hope the authors can improve the clarity of these figures.
2. Figure 1 appears on page 2, but there is no corresponding content on the first two pages. Is its placement here inappropriate? Additionally, Figure 1 is not referenced anywhere in the text. The same issue applies to Figure 2.
3. I feel that the originality of the article is somewhat limited. Could the authors provide an explanation of the connection between the challenges presented in this study and the research methods? At the moment, the challenges and methods do not seem to align well.
4. Regarding the analysis of the experimental section, I hope the authors can summarize it more clearly. The current summary of the experiments is not very clear.
5. How was the step level notation done? I would appreciate clarification on this point.

### Questions
Refer to the weaknesses.

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
This paper introduces PRISM-PHYSICS, a framework for evaluating physics problems at the process level. Solutions are represented as directed acyclic graphs (DAGs) to capture dependencies between steps. The key innovation is the ancestor-closure scoring, which allocates partial credit based on intermediate steps. A rule-based symbolic equivalence checker ensures accurate comparison of formulas. Experimental results show that PRISM-PHYSICS provides detailed and reliable evaluations.

### Strengths
- PRISM-PHYSICS provides a large-scale, competition-level benchmark with carefully curated, DAG-structured solutions to complex physics problems.
- A fully rule-based symbolic equivalence checker ensures consistent validation of diverse mathematical expressions, eliminating reliance on heuristic LLM scoring and offering a more reliable comparison across alternative formulations.
- The ancestor-closure scoring policy allows for partial credit on intermediate steps, offering a more nuanced and fair assessment of student reasoning.

### Weaknesses
- Does the system account for context-dependent variations in formulas? (e.g., solving a problem from both kinematics and dynamics perspectives, or analyzing it through momentum and energy considerations)
- In physics problems, certain expressions may be contextually equivalent, but the strict analysis in this algorithm might overlook such context-dependent equivalence.   Does the current framework account for these context-sensitive variations?
- If skips over intermediate, simpler steps during the solution process, would this result in incorrect evaluation by the proposed method?  
- The summary of the experiment is not clear enough. It is hoped that the author will have a more clear and organized discussion of the experimental results.

### Questions
See weakness.

### Soundness
3

### Presentation
2

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
This paper introduces PRISM-Physics, a large-scale physics reasoning benchmark with a proposed DAG-based evaluation protocol that addresses the limitations of existing LLM-as-judge scoring methods. The evaluation framework includes a fully rule-based symbolic formula equivalence checker to ensure consistent validation across diverse mathematical formulations, thereby eliminating reliance on subjective judgments. In the experiments, the paper investigates a diverse set of leading LLMs on PRISM-Physics and demonstrates the superiority of the proposed evaluation protocol compared to the LLM-as-judge method.

### Strengths
1. The idea of using DAG to judge the correctness of the final answer and intermediate steps is reasonable, and I agree that using the LLM-as-judge method to evaluate the correctness of physics problems is challenging and prone to errors.
2. The theoretical analysis part of the paper is solid.
3. The experiment is comprehensive and convincing.

### Weaknesses
1. My main concern is that, although PRISM-Physics can conduct rule-based judgments to determine the correctness of the final answer and intermediate steps using a DAG, the construction of the DAG still heavily relies on LLM-based extraction and rewriting. Compared to the existing LLM-as-judge method, the uncertainty introduced by LLMs seems to have merely shifted from the judgment stage to the preprocessing stage.
2. Another concern lies in the scalability and additional computational cost of the proposed evaluation protocol. Compared to existing benchmarks, PRISM-Physics requires an annotated DAG in addition to the final answer for each question in order to perform a more rigorous evaluation. Thus, the scalability of the proposed protocol appears limited. If we aim to extend this rigorous protocol to other existing benchmarks, what additional requirements would those questions need to meet? Furthermore, if we intend to construct a DAG for a new physics problem, how much extra computational cost would this introduce in the preprocessing stage?
3. Typo: in Line 362: "zero-shot **COTzheg** prompts". In Table 1, it would be better to retain the same number of digits after the decimal point and to bold the best results.
4. The results in Figure 8 (Appendix E.2) are difficult to understand. The authors should at least explain the meaning of each rectangle in the text and clarify whether the difference shown represents "multimodal – text" or "text – multimodal".

### Questions
See Weaknesses 1, 2.

### Soundness
2

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
This paper introduces PRISM-Physics, a physics benchmark with particular focus on process-level evaluation. The authors model reasoning process as a Directed Acyclic Graph, where each node represents a formula in the resoning chain, and edges denote logical dependencies. They then propose to score a student answer by counting the ancestor closure of student's nodes within that of reference answer. Besides the scoring scheme, the authors established a way to reliably transform natural language reasoning into DAG with normalized formulas, allowing for robust formula matching during evaluation. Lastly, they curated PRISM-Physics, contributing to process-level evaluation of physics problem solving.

### Strengths
1. The paper is well written and very detailed.
2. Experiments and analysis are comprehensive and insightful, covering a wide range of LLMs and problem types.
3. The proposed DAG-based scoring scheme is novel and captures certain logical dependencies in the reasoning path.

### Weaknesses
1. The Ancestor Closure Scoring Policy seemed too forgiving for rigorous process-level evaluation. If I understand correctly, this scoring scheme overlooks skipped steps in the students reasoning path, as long as a targeted formula is attained downstream. Neither does this scheme check for validity of derivation, since it does not check whether assumptions leading to the formula are correct.
2. Following point 2, it would be nice to verify how well the scoring scheme aligns with true logical evaluation.
3. While the DAG-based scoring system does not require a student's answer to have the same sequential steps, it also ignores the structure in the student's answer. If I understand correctly, the student's answer is extracted as a bag of formulas with no logical dependencies extracted from context; only the reference answer is represented as DAG with logical dependency. Again, this seemed quite forgiving.

### Questions
1. Since you assume all ancestor nodes in the reference DAG are scored, have you checked how often a student answer actually misses those ancestor nodes? Would missing these nodes break the logical soundness of the student's answer?
2. In Section 6.4, did human experts score with Ancestor Closure Scoring as well? If yes, have you tried alternative evaluation methods (that is based on true logic instead of formula matching) to see how Ancestor Closure Scoring aligns with that scheme?
3. In Appendix E.1, did you use 8K context and zero temperature for evaluation of reasoning LLMs?
4. Have you tried extracting the student's answer as a DAG as well, and compare it to the reference DAG for a more rigorous scoring?

### Soundness
3

### Presentation
4

### Contribution
3
