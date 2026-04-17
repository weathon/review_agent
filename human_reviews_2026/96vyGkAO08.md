# Anchoring Entities: Retrieval-Augmented Hallucination Detection

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 6

## Abstract
Hallucination detection is crucial for large language models (LLMs), as hallucinated content creates significant barriers in applications requiring factual accuracy. Current detection methods mainly depend on internal signals like uncertainty and self-consistency checks, using the model's pre-trained knowledge to identify unreliable outputs. However, pre-trained knowledge may become outdated and has coverage limitations, especially for specialized or recent information. To address these limitations, retrieval-augmented generation (RAG) has emerged as a promising solution that grounds model outputs in external evidence. In this paper, we target a critical and practical learning problem RAG-based hallucination detection (RHD), where RAG is employed to enhance hallucination detection by addressing information updating challenges. To address RHD, we propose a novel method Evidence-Aligned Entity Verification (EAEV), which detects entity-level hallucinations by leveraging RAG to align generated entities with retrieved evidence contexts. Specifically, EAEV evaluates entity-evidence alignment through three complementary dimensions and introduces counterfactual stability analysis to ensure robust alignments under evidence perturbations. Experiments across multiple RAG benchmarks demonstrate that EAEV achieves consistent improvements over existing methods with strong generalization capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduced RAG-based hallucination detection, a new task that aims to detect hallucination based on the alignment between LLM-generated text and retrieved documents. The authors then proposed Evidence-Alighed Entity Verification, a new method to detect entity-level hallucination with retrieved documents. The approach consists of alignment assessment, stability analysis, and entity-centric aggregation. The authors conducted experiments on four datasets with three models, showing a better performance against other baselines.

### Strengths
1. The authors introduced a novel approach for hallucination detection at the entity level, which will be useful in real-world applications to localize hallucinations
2. The proposed approach achieves the highest performance on almost all datasets and models

### Weaknesses
1. **Novelty of RAG-based hallucination detection (RHD).** The author claims that they proposed RHD as a novel task. However, detecting hallucination based on retrieved evidence is not a new task. People have been using NLI models or LLMs to check if the answer contradicts provided evidence [1, 2]. In addition, this idea has also been adopted at the entity level [3].
2. **Unjustified statements.** One major concern of this paper is that it contains many unjustified statements, weakening the sound of this paper. For example, in Line 118, the authors claimed that "[Existing methods] face challenges when evidence is explicitly available yet underutilized in RAG setting" without further explanation or citations. In addition, in Line 132 (and also 161), the authors claimed that "factual errors in RAG settings manifest primarily at the entity level" without providing any citations or empirical evidence in later Sections. As the approach of entity-level RAG-based hallucination detection is the core of this paper, I think these statements should be properly justified.
3. **Lack details of the proposed approach.** Another major concern is that many details of EAEV are missing. For example:
    - Lines 205 and 239: The authors did not explain how they extract $s$. The authors also didn't explain why they only focus on certain types of entity (i.e., ENT, NUM, NP)
    - Line 229: The authors did not explain what $\text{anchor}(q,e^\ast)$ is
    - Line 245: It is unclear what the rule-based patterns are for identifying explicit conflicts
    - Lines 231, 241, and 246: The authors did not explain how they found the hyperparameters
    - The authors should properly justify these design choices or provide experimental/ablation results to support the choices.
4. **Basic assumption.** The basic assumption of the proposed approach is that the evidence retrieved by a RAG system is correct and sufficient to detect hallucination. However, a RAG system can return noise or incorrect data, which may degrade the performance of the proposed approach. The authors should conduct experiments or analyses in such setting, or mention it as a limitation.
5. **Lack entity-level verification.** The authors claim that they proposed an entity-level hallucination detection approach. However, all the experiments are conducted at the answer level. I believe some experiments/analyses at the entity level are necessary to justify their approach.

[1]: Fast and Accurate Factual Inconsistency Detection Over Long Documents. (2023)

[2]: FACTSCORE: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation (2023)

[3]: HalluEntity: Benchmarking and Understanding Entity-Level Hallucination Detection (2025)

### Questions
1. What is the purpose of the supervised learning model in the whole pipeline? The authors discussed the supervised model in Sec 3.6 but did not conduct any experiment on it. There are also other details about the model (e.g., what dataset is used to train the model). It is unclear to me why the authors included this section.
2. Fig 3: For the left figure, there is no relationship between AUROC, Accuracy, and F1. Thus, using a line plot to connect these three doesn't make sense to me. The authors should consider using a bar plot instead. On the other hand, it would be better to use a line plot in the right figure.

### Soundness
1

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
4

### Summary
This paper introduces Evidence-Aligned Entity Verification (EAEV), a method for detecting hallucinations in retrieval-augmented generation (RAG) systems. The key innovation is entity-level verification that leverages three alignment dimensions (identity, semantic, consistency) combined with counterfactual stability analysis to distinguish genuine evidence support from spurious correlations. The method is evaluated on three RAG benchmarks (RAGTruth, HotpotQA, DelucionQA) across three model architectures, achieving 87.55% average AUROC on LLaMA2-13B with consistent improvements over 11 baseline methods.

### Strengths
1. Entity-level verification in RAG addresses a real gap in existing methods
2. Innovative approach to distinguish genuine evidence support from spurious correlations
3. Identity, semantic, and consistency dimensions provide complementary verification signals
4. Multiple models (Qwen2.5-7B, LLaMA2-7B/13B), datasets (RAGTruth, HotpotQA, DelucionQA), and 11 baselines
5. Consistent improvements across settings (87.55% avg AUROC on LLaMA2-13B, +3-4 points over best baseline)

### Weaknesses
1. Identity = string matching, semantic = embedding similarity, consistency = numerical IoU; main novelty is the combination.

2. The concerns in Ad-hoc design choices. Four perturbation types lack principled selection criteria. Multiplicative combination in Eq. 6 not justified. Type-adaptive weights appear hand-tuned without sensitivity analysis.

3. The paper has some missing critical analyses, computational cost comparison vs. baselines, human evaluation of entity-level detection quality, and failure mode analysis or error propagation study.

4. Only English QA/summarization tasks; no dialogue, code generation, or cross-domain evaluation
Methodological concerns:

5. Statistical significance testing absent. Entity extraction pipeline glossed over and claims "entity-level" but only evaluates answer-level metrics.

6. The work contribute in incremental improvements: +3-4 AUROC points is meaningful but not dramatic given added complexity
Theoretical gaps: No justification for why these three alignment dimensions are sufficient/complete

### Questions
How do you handle cases where the entity is correct but the relation is wrong? (e.g., "Paris is the capital of Germany" - both entities are real)
Why use multiplicative combination in Eq. 6 rather than additive or learned combination?
What happens when relevant evidence is NOT retrieved? Does EAEV flag everything as hallucination?
Can you provide examples where counterfactual stability catches spurious correlations that multi-dimensional alignment alone misses?
How does performance vary with retrieval quality (e.g., at different top-k settings)?
The supervised learning component (Sec 3.6) seems disconnected - is it an alternative method or complementary approach?
How do you determine the "primary evidence e*" when multiple evidence pieces support the entity?
What is the computational overhead compared to lightweight baselines like SelfCheckGPT?

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
4

### Summary
This paper introduces Evidence-Aligned Entity Verification (EAEV), which detects entity-level hallucinations by leveraging RAG to align generated entities with retrieved evidence contexts. EAEV first performs entity-evidence alignment through three complementary dimensions and introduces counterfactual stability analysis to ensure robust alignments under evidence perturbations. The experiments demonstrate superior performance of EAEV across multiple RAG benchmarks.

### Strengths
1. This work propose to leverage RAG for entity-level verification within retrieved contexts in hallucination detection, where previous works  rely on internal uncertainty or external judges without evidence traceability.

2. The proposed method combines multidimensional alignment with counterfactual stability analysis to distinguish genuine evidence support from spurious correlations in RAG settings.

3. The experiments on RAG benchmarks show that the proposed method works well on Qwen2.5-7B, LLaMA2-7B and LLaMA2-13B.

### Weaknesses
1. The experiments are conducted on outdated models, with the most recent being Qwen2.5. Could you provide additional experimental results on newer models such as Qwen 3 or Llama 3.2 to demonstrate the consistent superiority of your approach?

2. The multi-dimensional alignment assessment in Sec. 3.3 is introduced to evaluate the alignment between entity mentions and supporting evidence. Have the authors considered prompting SOTA LLMs (e.g., GPT-4.1, Claude 4, Gemini 2.5) to do this task as an alternative to the proposed machine learning pipeline? A comparative analysis between the LLM-based approach and your method would strengthen the paper by demonstrating the advantages of your proposed pipeline over these readily available alternatives.

### Questions
1. Related works are missing, for example: Enhancing Uncertainty-Based Hallucination Detection with Stronger Focus by Zhang et al.

2. The evaluation resource cost of this work is missing.

### Soundness
2

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
4

### Summary
The paper tackles hallucination detection in retrieval-augmented generation (RAG) and proposes Evidence-Aligned Entity Verification (EAEV). EAEV checks each entity in a model’s answer against retrieved passages along three axes—identity (direct matches), semantic (paraphrase similarity), and consistency (numbers/attributes, contradictions)—and adds counterfactual stability tests to filter spurious matches.

### Strengths
1. Sound motivation: The motivation is well formulated and clearly justified. I like how the authors approach RHD from a different angle and redefine it.

2. Robust framework: The authors clearly justify the design choices behind each component, and the ablation study demonstrates why those components matter.

3. Exhaustive experiments: I appreciate the inclusion of 11 baselines and the rigor of the experimental evaluation.

### Weaknesses
A few suggestions:

1. Figure 1 caption: Expand the caption to explain the figure so readers can grasp it at a glance.

2. Figure 3 readability: It’s difficult to read in its current form -- please increase the font size (and consider improving contrast).

3. Citations/references: There are some inaccuracies. For example, the LLM-Check paper lists different author names than the original. Please be more mindful of citations and references, and double-check them.

### Questions
I have no questions so far.

### Soundness
3

### Presentation
4

### Contribution
3
