# MAD-Logic: Multi-Agent Debate Enhances Symbolic Translation and Reasoning

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Large language models (LLMs) struggle with complex logical reasoning. Previous methods can be briefly summarized into two pipelines: (1) translating natural language (NL) to symbolic language (SL) then reasoning via external solvers, and (2) adopting LLMs to reason directly in NL based on prompting or fine-tuning.  However, we point out that on the one hand, the translation relying on a specific SL often fails to capture different important features of raw NL, leading to information loss or translation errors. On the other hand, both two pipelines have unignorable limitations. For example, the former (SL-based) methods are highly sensitive to imperfect translation, and the latter (NL-based) methods are prone to hallucinations.   Motivated by this, we are the first to propose a multi-agent debate framework to leverage the strengths of different SLs and reasoning methods, achieving better performance in both translation and reasoning stages. Specifically, in the translation stage, multiple agents translate the NL into different SL and refine translations through debate. In the reasoning stage, multiple agents based on SL (obtained by the corresponding solver) and NL debate multiple rounds, with the final answer determined by majority vote. In addition, to address the inefficiency of multi-agent debates, we introduce an adaptive sparse communication strategy that prunes unnecessary interactions based on agent confidence and information gains. Extensive experiments on three datasets show that our method enhances logical QA performance while reducing computational cost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The manuscript proposes a sparse multi-agent debate framework that first translates natural-language logical questions into three symbolic languages (LP, FOL, SAT) via agent debate, then lets symbolic and natural-language agents debate the answers, finally voting on the result. An adaptive communication gate prunes low-value inter-agent messages. Experiments on three logical-QA benchmarks report accuracy improvements over a leaderboard that has already reached a degree of saturation.

### Strengths
Table 1 shows +0.92 pp average gain over the fully-connected multi-agent baseline on GPT-4, demonstrating that merging symbolic and NL reasoning is more accurate than either alone.

Table 3 ablation adding each solver (LP→SAT→FOL) steadily lifts accuracy (88.06 % → 95.44 %), confirming that diversity of symbolic systems matters.

Algorithm 1 (Appendix E) gives reproducible pseudo-code, enhancing technical soundness.

### Weaknesses
The experiments are confined to three English synthetic datasets—all of which have already reached saturation. Their small size further undermines the significance of the findings, and no evidence is provided for performance on natural text, whether from specialized domains (e.g., legal, medical) or multilingual contexts.

Table 1 presents performance percentages without accompanying variance estimates (e.g., standard deviation, confidence intervals). Given the dataset’s size, small discrepancies in reported performance could easily be attributed to random noise rather than meaningful differences.

No token cost comparison is provided against two critical baselines: human-only reasoning, or single-agent Chain-of-Thought (CoT) operating within the same budget. This makes the "token saving" claim questionable, as the reference point for comparison remains unspecified.

No timing analysis: how often do Pyke/Prover9/Z3 timeout?

Eq. (1) introduces a preference score but no lemma guarantees that pruning preserves consensus or convergence. No ablation on λ beyond a single value (1.0).

I'm concerned with the prompt sensitivity. Appendix F shows one fixed prompt per role; no robustness check with alternative prompt wordings.

GPT-4 and Claude may have seen ProofWriter during pre-training; no sanitisation check is mentioned.

### Questions
Suggestions:
Run pipeline on FOLIO multilingual, Chinese LogiQA-V2 and LogiEval-Hard; report delta and failure rates.
Repeat main experiments with three prompt seeds; report mean and std-dev in Table 1.

### Soundness
1

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
2

### Summary
This paper proposes a sparse multi-agent debate framework  to improve the performance of large language models in complex symbolic logic reasoning tasks. This approach combines the advantages of symbolic language (SL) and natural language (NL) reasoning: multiple agents translate the question into different formal logics (LP, FOL, SAT) and optimize the translation through debate. The SL and NL agents then debate and vote on the final answer over multiple rounds. To improve efficiency, the authors design an adaptive sparse communication mechanism, which reduces token consumption while improving accuracy.

My opinion may be negative at this stage, if the author can address my concerns, I will increase my score.

### Strengths
1. Achieved very good performance on multiple data sets, with many indicators reaching 100%.

2. The experiments are very substantial, and a large number of experiments are done to verify the performance and efficiency of the proposed method.

### Weaknesses
1. The paper repeatedly states that LLMs struggle with complex logical reasoning. However, the metrics shown in Table 1 are all very high, which appears inconsistent with this claim. Moreover, I am curious about the results of directly testing the LLMs—i.e., having the LLM answer the questions without any additional framework—as such baseline results might better support the authors’ argument.

2. While most of the metrics in Table 1 are high, the proposed method does not achieve significant improvements over the baseline. The numerical metrics indicate that all three datasets appear simple, with 100% accuracy achieved multiple times, making the experiment less than convincing.

3. The paper uses the latest powerful baseline models. Does this mean that the method proposed in this paper has high requirements for the base model? Is it applicable to those ordinary models or smaller models such as qwen 2.5 7B?

### Questions
1. Is there a more challenging dataset now? This is necessary to verify the effectiveness of the method in this paper.

2. How is the confidence score obtained in sparse communication?

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
The authors introduce a multi-agentic debate system to elicit logical reasoning in LLMs, by integrating symbolic and natural language reasoning. They do this through translating natural language to symbolic language using multiple agents, with the refinement process done through agentic debate. For reasoning, the symbolic output from the translation stage and natural language are used to determine an output through majority voting. To enhance computational efficiency, the authors use a sparse communication strategy within the debate framework.

### Strengths
- The use of general purpose (i.e. GPT) and reasoning-enhanced (i.e. deepseek) LLMs helps the reader understand the generalizability of the proposed approach. 
- The results are impressive, as they achieve 100% across several set-ups. 
- The number of baselines used to compare to the proposed approach is sufficient, showcasing the true performance capability of the framework. 
- The use of an adaptive sparse communication strategy to enhance computational efficiency is a very important advantage of this framework.

### Weaknesses
- The whole pipeline depends on the translation quality in the first stage. There is some missing validation of the symbolic translations to enhance the validity of the pipeline. 
- The majority-vote process could highlight systematic biases if multiple agents share similar pretraining biases or translation tendencies.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a two-stage, sparse multi-agent debate framework to improve complex logical reasoning with LLMs by explicitly combining symbolic-language (SL) and natural-language (NL) paradigms: multiple agents first translate an NL problem into diverse SL formalisms (LP, FOL, SAT) and refine these translations via debate; then SL-solver agents (Pyke/Prover9/Z3 results verbalized) and NL-reasoning agents (e.g., CoT, Plan-and-Solve) debate for several rounds before a majority vote determines the answer. To curb the high cost of multi-agent discussion, the authors introduce an adaptive sparse communication mechanism that gates interactions using a preference score mixing agent confidence and information gain (cosine dissimilarity), coupled with selective memory updates. Experiments on ProntoQA, ProofWriter (depth-5 subset), and BIG-Bench LogicalDeduction across GPT-4, Claude 3.7 Sonnet, and DeepSeek-V3 report new SOTA accuracies while reducing token usage compared to strong single- and multi-agent baselines, with ablations showing gains from both the translation-stage debate and heterogeneous SL+NL agent composition.

### Strengths
- Novel combination of symbolic and natural-language reasoning in a multi-agent debate.
- Sparse-communication mechanism effectively reduces token cost with minimal accuracy loss.
- Strong empirical performance and ablation studies across several reasoning benchmarks.

### Weaknesses
- The paper’s two main ideas extend well-known patterns (multi-agent debate; sparse topologies like SparseMAD/CortexDebate) rather than fundamentally rethinking them; the conceptual gap from past methods looks narrow.
- No theoretical framework proving that the proposed agents can perform valid NL to FOL/LP/SAT translation.
- With heterogeneous agents, simple majority voting can entrench shared biases or spurious agreement.
- Critical components (e.g., $C_i^d$, Algorithm 1) are under-defined or inconsistent.
- Lack of significance testing and limited evaluation scope on real-world reasoning tasks.

### Questions
Please concretely differentiate your contributions from recent multi-agent debate and sparse/topology papers (methodologically, not just empirically). Could you add an ablation that (a) removes the SL-NL cross-paradigm stage and (b) replaces your sparse gating with a strongest prior baseline, to quantify each idea’s standalone lift?

How often do translation mistakes occur, and how does the system recover when all SL agents share the same systematic error?

Please include sensitivity analyses for λ and the similarity metric, and compare majority vote against confidence-weighted or learned adjudicators.

### Soundness
3

### Presentation
3

### Contribution
2
