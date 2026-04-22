# CodeSense: a Real-World Benchmark and Dataset for Code Semantic Reasoning

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 2, 2, 6

## Abstract
Understanding and reasoning about code semantics is essential for enhancing code LLMs' abilities to solve real-world software engineering (SE) tasks. Although several code reasoning benchmarks exist, most rely on synthetic datasets or educational coding problems and focus on coarse-grained reasoning tasks such as input/output prediction, limiting their effectiveness in evaluating LLMs in practical SE contexts. To bridge this gap, we propose CodeSense, the first benchmark that makes available a spectrum of fine-grained code reasoning tasks concerned with the software engineering of real-world code. We collected Python, C and Java software projects from real-world repositories. We executed tests from these repositories, collected their execution traces, and constructed a ground truth dataset for fine-grained semantic reasoning tasks.  We then performed comprehensive evaluations on state-of-the-art LLMs. Our results show a clear performance gap for the models to handle fine-grained reasoning tasks. Although prompting techniques such as chain-of-thought and in-context learning helped, the lack of code semantics in LLMs fundamentally limit models' capabilities of code reasoning. Besides dataset, benchmark and evaluation, our work produced an execution tracing framework and tool set that make it easy to collect ground truth for fine-grained SE reasoning tasks, offering a strong basis for future benchmark construction and model post training. Our code and data are located at \url{https://codesense-bench.github.io/}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The author propose a benchmark CodeSense that makes available a spectrum of fine-grained code reasoning tasks concerned with the software engineering of real-world code. The results show a clear performance gap for the models to handle finegrained reasoning tasks.

### Strengths
* The coverage over Python and C are good. 

* Sourced from real-world project

* Comprehensive testing over wide range of models.

### Weaknesses
* A bit of overselling: Java examples are only 74, which is negligible compared to the total number.

* Figure 3b: What is “smaller” or “larger” function size? Any quantitative number?

* How does author measure accuracy for API call? For something like `time`, it seems impossible to get it right, e.g. time interval for executing code blocks are simply random.

* Can you give example of “abstract values”? For Figure 7, was the few-shot examples with abstract or concrete values?

### Questions
* Figure 4: Models are bad at arithmetic so, it seems low accuracy is expected. 

* Line 375-376: What if the model is prompted with different type of statement, how would statement prediction perform?

* Could the author include a non-reasoning and reasoning mode for Loop Properties, given the same model (e.g. Qwen3)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Obvious template tampering. Recommended to desk reject.

### Strengths
N/A

### Weaknesses
N/A

### Questions
N/A

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes CodeSense, a benchmark for fine-grained code semantic reasoning constructed from real-world GitHub projects in Python, C, and Java. The benchmark introduces reasoning tasks at statement, code-block, and function levels, targeting semantic properties such as variable values, loop iteration counts, pointer aliasing, and branch conditions. The authors develop an execution tracing framework to automatically generate ground truth from real-world tests, and evaluate 14 state-of-the-art LLMs across six research questions. Results show that models struggle with fine-grained semantic reasoning, particularly on C code and arithmetic operations, with limited improvements from prompting techniques.

This paper addresses an important problem but lacks critical validation of the benchmark's value relative to existing work. While Table 1 compares features with CruxEval, CruxEval-X, REval, and CodeMind, no experiments demonstrate whether CodeSense provides unique insights or advantages that existing benchmarks cannot capture.

### Strengths
1. Important problem and motivation: Fine-grained semantic reasoning is crucial for real-world SE tasks like test generation, vulnerability detection, and bug repair. The motivation examples in Figure 1 effectively illustrate this.

2. The paper provides clear presentation and informative figures. 

3.Releasing the framework, dataset, and leaderboard supports reproducibility and future work.

### Weaknesses
1. Missing Direct Comparison with Existing Benchmarks
The paper claims CodeSense provides advantages over existing benchmarks (Table 1) but provides no experimental validation. Table 1 only compares features (Real-world Projects, Fine-grained Reasoning) without demonstrating whether these features translate to better evaluation quality. The 14 models should be evaluated on both CodeSense and existing benchmarks (CruxEval, REval, CodeMind) to show: (1) whether CodeSense reveals insights that other benchmarks miss, (2) whether fine-grained tasks are more diagnostic than coarse-grained I/O prediction, and (3) whether real-world code introduces challenges absent in synthetic benchmarks. Without this comparison, the benchmark's unique contribution is unproven.


2. Unclear Novelty of Research Findings
The six RQs (RQ1-RQ6) investigate important questions, but without baseline comparison, their novelty is unclear. For example, RQ4 finds "chain-of-thought offers limited benefit" and RQ6 shows "models perform better on Java/Python than C". I may have questions like are these findings unique to CodeSense's fine-grained tasks, or would they also be observed on CruxEval? The paper positions findings as new discoveries without demonstrating what existing benchmarks cannot reveal. 


3.The paper does not adequately justify why fine-grained semantic reasoning (statement-level, loop counts, pointer aliasing) is more valuable than existing coarse-grained tasks. While Section 1 motivates the need for semantic reasoning, no evidence shows that fine-grained evaluation better predicts performance on downstream SE tasks. For instance, does a model's ability to predict loop iteration counts correlate with its ability to generate test inputs or detect vulnerabilities? Without this validation, it's unclear whether the added complexity of fine-grained tasks provides practical benefits beyond existing benchmarks.

### Questions
Do you have any existing comparison data (even partial) showing how the same models perform on CodeSense vs CruxEval or REval?

Can you provide specific examples of insights that CodeSense reveals but existing benchmarks cannot?

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
This paper proposes CodeSense, a benchmark for code semantic reasoning sourced from real-world code. The authors curate benchmark samples in Python, C, and Java from GitHub repositories and study a variety of research questions  such as block-level semantics, statement-level semantics.

### Strengths
- The benchmark is interesting, addressing the weaknesses of previous benchmarks (CruxEval, REval, CodeMind).
- The authors study a diverse array of code reasoning and program analysis tasks. RQ3 and RQ5 are particularly interesting and have not been studied before (to my knowledge). Many other RQ's are also studied from a fresh perspective, and there is some analysis for each one (some extended in the Appendix)
- A wide variety of models are evaluated and studied, and there is a lot of room for improvement among the studied models.

### Weaknesses
- The models studied in the paper are generally weaker than those on the frontier line, even taking into account the lag between the review date and the ICLR submission deadline. The paper would be stronger if it drew insights from failure modes of today's frontier models such as GPT-5, Gemini-2.5-Pro as well as open models like DeepSeek-R1, Qwen3. This would differentiate which of the paper's findings still holds true for the strongest models.
- The research questions are interesting and open up the potential for in-depth exploration, but analysis is only done at a surface level. For example, the CoT's could be analyzed to understand where models are reasoning incorrectly. This is especially interesting for the strongest models. 
- Another example is that RQ3 could be studied more carefully, analyzing these trends for different programs (similar to the flavor of https://arxiv.org/abs/2402.05980). RQ5 is very interesting and could also benefit from an analysis of how the type and abstraction level of approximations affects accuracy.
- RQ6 should be studied with paired data, to minimize the effect of confounding variables other than language.
- No variance numbers are reported for the results

### Questions
- To what extent do you expect training on traces (e.g. SemCoder, Code World Model) to improve the performance of models? Can you do an analysis to see if these models are significantly better than their non-trace-trained counterparts, or a simple experiment to measure this effect?
- In your opinion, what kinds of insights does your benchmark enable that were not apparent from previous benchmarks?
- In RQ6, how do you know the results are due to the language difference rather than the difference of the problems in the language?
- How do the findings in this work relate to previous work? For example, CruxEval-X studies something very close to RQ6, can you compare the findings? I believe some other RQ's have works studying similar questions.

### Soundness
3

### Presentation
3

### Contribution
3
