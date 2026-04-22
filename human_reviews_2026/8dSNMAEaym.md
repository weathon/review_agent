# Concept Concentration for Faithful Representation Intervention

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 6, 2, 4

## Abstract
Representation intervention aims to locate and modify the representations that encode the underlying concepts in Large Language Models (LLMs) to elicit the aligned and expected behaviors. Despite the empirical success, it has never been examined whether one could locate the faithful concepts for intervention. In this work, we explore the question in safety alignment. If the interventions are faithful, the intervened LLMs should erase the harmful concepts and be robust to both in-distribution adversarial prompts and the \textit{out-of-distribution} (OOD) jailbreaks. While it is feasible to erase harmful concepts without degrading the benign functionalities of LLMs in linear settings, we show that it is \textit{infeasible} in the general non-linear setting. To tackle the issue, we propose \texttt{Concept Concentration} (\texttt{COCA}). Instead of identifying the faithful locations to intervene, \texttt{COCA} refactors the training data with an explicit reasoning process, which first identifies the potential unsafe concepts and then decides the responses. Essentially, \texttt{COCA} simplifies the decision boundary between harmful and benign representations, enabling more effective linear erasure. Extensive experiments with multiple representation intervention methods and model architectures demonstrate that \texttt{COCA} significantly reduces both in-distribution and OOD jailbreak success rates, and meanwhile maintaining strong performance on regular tasks such as math and code generation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the limitations of existing representation intervention methods for safety alignment in LLMs, showing that perfect harmful concept erasure is theoretically impossible in non-linear representation spaces. To address this, it proposes COCA (COncept ConcentrAtion), which introduces structured reasoning annotations to linearize harmful concepts, enabling effective and faithful erasure. Experiments across four LLMs demonstrate that COCA improves robustness against out-of-distribution jailbreaks while preserving benign capabilities.

### Strengths
1. **Clear Theoretical Contribution.** The paper rigorously identifies the root cause of representation intervention failure as non-linear entanglement between harmful and benign concepts and mathematically proves the impossibility of perfect erasure under this regime (Theorem 2.2).
2. **Novel Conceptual Shift.** Instead of seeking ideal intervention points in a complex space, it introduces Concept Concentration (COCA), a new paradigm that simplifies the representation space at the data level.
3. **Strong Empirical Results.** COCA substantially reduces OOD jailbreak success rates across four base models (LLaMA-3.1-8B, Qwen-2.5-7B, etc.) while preserving performance on math and code reasoning tasks, demonstrating a balance between safety and utility

### Weaknesses
1. **Annotator Bias.** The method depends on fine-grained concept annotations, so residual bias or subjectivity in defining “unsafe” content may remain.
2. **Absence of Large Reasoning Model Evaluation.** The paper does not include experiments on Large Reasoning Models (LRMs), which are increasingly important for assessing safety and concept alignment. Because the proposed method relies on explicit structural annotations, its applicability to LRMs such as DeepSeek-R1-Qwen-7B or DeepSeek-R1-LLaMA-8B remains unclear. Without such validation, it is difficult to confirm whether the approach generalizes beyond standard instruction-tuned LLMs. Comparison with recent LRM baselines [1,2] would strengthen the evaluation.

[1] Jeung, Wonje, et al. "SAFEPATH: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment." NeurIPS (2025).\
[2] Wang, Zijun, et al. "Star-1: Safer alignment of reasoning llms with 1k data." arXiv preprint arXiv:2504.01903 (2025).

### Questions
I don't have additional questions. Solid work.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes COCA, a presentation intervention approach to improve the safety alignment robustness against out-of-distribution jailbreaks. It first conducts a theoretical analysis to reveal the infeasibility of existing representation intervention techniques to erase harmful concepts without degrading utility under non-linear setup. It then proposes a data refinement approach to explicitly decompose the safety data into reasoning steps and prove that such approach is able to simplify the decision boundary and make linear erase feasible. Evaluation results on several jailbreaks and comparison with SOTA concept erase and safety alignment baselines demonstrate the effectiveness of the COCA.

### Strengths
- The paper tackles an important problem in LLM safety.
- The theoretical analysis is insightful and valuable to the safety community, as it exposes fundamental limitations of existing methods and motivates the proposed approach.
- The proposed data refinement method is intuitive and easy to follow.

### Weaknesses
- The empirical improvement over existing methods is relatively small.
- The presentation could be improved for better clarity and readability.

### Questions
- The presentation, especially in Tables 1 and 2, could be improved. The current layout is hard to read and makes it difficult to match results with the proposed approach.
- When compared with state-of-the-art alignment methods such as STAIR, the improvement achieved by COCA-structured data appears marginal and, in some cases, even falls below baseline performance.  For example, STAIR achieves 4.3% ASR on PAIR when applied to LLaMA-3.1-8B, while COCA+RR yields 7.8%.
- How are ID and OOD attacks refined? What is the rationale for treating all jailbreak attacks as OOD? Some, like PAIR, do not involve unreadable tokens, what characteristic makes them OOD?
- It would be helpful to include evaluations against gradient-based jailbreaks, such as GCG [1], to further demonstrate robustness.

---
Reference 
----

[1] Zou A, Wang Z, Carlini N, et al. Universal and transferable adversarial attacks on aligned language models[J]. arXiv preprint arXiv:2307.15043, 2023.

### Soundness
3

### Presentation
2

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
This paper studies the safety alignment problem of LLMs. The authors propose COCA, which first indentifies the potential unsafe concepts and then decides the response. The anthor introduce a well-designed system prompt annotated by teacher model (GPT-4o) into the training data. It aims to induce the student LLMs to reflect and indentify on the potential unsafe the concepts before the standard response. The authors validate the effectiveness of the proposed approach in multiple experiments.

### Strengths
1. The problem studied in this paper is valuable. 
2. The paper is well written. 
3. The proposed approach is effective.

### Weaknesses
1. This paper tells a good story, from the standard safety alignment, concept-centric alignment to the linear representation hypothesis. After reading that, I expect some amazing ideas to solve the non-linear representation alignment problem. However, the method is just like a prompt or CoT engineer with well-designed prompts, and fine-tune the model to generate safety thinking before the standard response. We believe this is a very normal and robust method in real applications. 
2. The evidence of my first point can be found in Table 3. The "self-generated" method can also significantly improve the safety performance. And I believe that the LLMs can also achieve a good safety performance by adding this "concept prompt" in the LLMs without any training.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

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
This paper proposes COCA, a data pre-processing framework designed to make safety reasoning explicit in fine-tuning data. COCA reformats training examples into a structured reasoning sequence using five tags, aiming to encourage the model to reason about and remove harmful content before producing final outputs. The authors provide theoretical analysis suggesting that such structured reasoning can concentrate harmful concepts into a more linearly separable subspace in the model’s internal representation space. Comprehensive experiments on several open-weight LLMs show that COCA enhances model's resistance to both in-distribution and out-of-distribution attacks while maintaining utility.

### Strengths
1. The proposed method is easy-to-integrate in practice. Since it requires no system changes and only modifies the training data, COCA could be easily applied to existing fine-tuning framework.
2. The paper provides a formal analysis demonstrating that COCA encourages concentration of harmful concepts into a linearly separable subspace.
3. The authors present visualization result to illustrate the better internal separation after fine-tuning with COCA's data.

### Weaknesses
1. The authors evaluate COCA primarily on medium-sized models (7B-9B). Including smaller (<=3B) and larger (>=14B) models would help assess COCA's generalizability.
2. The paper claims that COCA is orthogonal to SRG and STAIR and could be combined with them, yet no experiments demonstrate this. Moreover, COCA’s reported performance in Table 1 does not show clear improvement over these baselines.
3. COCA relies on five hardcoded tags. The paper does not clarify how these tags were defined or whether performance depends on their semantics or order. It remains unclear if changing, removing, or reordering tags would affect performance.
4. The theoretical analysis models COCA’s data as a generic structured reasoning sequence and does not differentiate the role of individual tags. As a result, the theory does not explain how specific tags contribute to concept concentration. It is unknown that whether the effectiveness of COCA comes from its five-tag design or only from its longer reasoning context provided during the fine-tuning.
5. COCA's main contribution is related to train the model to follow a specific reasoning pattern through fine-tuning. Therefore, comparisons with similar reasoning-based alignment methods, such as Deliberative Alignment [1], are necessary.
6. The paper presents internal-state separability results only for COCA, while ignoring the baselines. Without this comparison, it is unclear whether the observed separation is unique to COCA or a general outcome of fine-tuning.
7. COCA evaluates internal-state separability but does not use these states to classify or guide safety decisions. Higher separability in internal representations does not necessarily imply improved output safety and quality of the model.

[1] Guan, Melody Y., et al. "Deliberative alignment: Reasoning enables safer language models." arXiv preprint arXiv:2412.16339 (2024).

### Questions
Please check my questions in the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2
