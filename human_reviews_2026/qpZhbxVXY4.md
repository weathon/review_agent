# CodeChemist: Test-Time Scaling for Low-Resource Code Generation via Functional Knowledge Transfer

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Code Large Language Models (CodeLLMs) have been widely adopted for code generation, powering applications with large user bases.
Their performance, however, varies sharply across programming languages (PLs) and is particularly suboptimal for low-resource PLs due to data scarcity, limiting their overall usability. In this work, we introduce CodeChemist, a simple yet effective test-time scaling framework that transfers the model's functional knowledge from high-resource to low-resource PLs via synthesized test cases.
Specifically, CodeChemist first performs code generation and execution in high-resource PLs to derive test cases that capture functional knowledge, then applies multi-temperature hedged sampling to produce candidate code snippets in the low-resource PL, and finally selects the best candidate by executing the synthesized test cases. Extensive experiments demonstrate that CodeChemist significantly outperforms existing test-time scaling methods, improving code generation for low-resource PLs without retraining.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CodeChemist, a test-time framework that transfers functional knowledge from high-resource programming languages to low-resource ones by synthesizing executable, language-agnostic test cases; it then applies hedged multi-temperature sampling in the target language and selects the candidate maximizing test pass rate—without retraining. Experiments across multiple models and benchmarks report consistent gains and show compatibility with other test-time scaling methods.

### Strengths
1. The paper is clearly written and easy to follow.
2. The method is training-free,and the efficiency study indicates that its runtime is acceptable.

### Weaknesses
1. The proposed approach appears to primarily compose existing test-time techniques, without introducing a clearly novel algorithmic mechanism or theoretical insight.
2. While the motivation for the proposed multi-temperature hedged sampling is understandable, the rationale for the specific hyperparameter choices is not sufficiently justified.
3. In practice, inputs differ substantially across programming languages—for example, when a function expects a struct—so how should such inputs be converted across languages?
4. In many practical settings, a Python implementation is infeasible, necessitating the use of other languages; conversely, if the task can already be solved in Python, the added value of generating code in another language is unclear.
5. In your paper, you have characterized C++ as a low-resource language. Given its many application scenarios, such as its translation to Rust, we would be more interested in seeing your method applied to a C++-to-Rust context, which represents a more common real-world application.
6. The improvements appear modest, and the set of comparative baselines is insufficient.
7. The quality of the generated test cases is questionable.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
3

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
This paper introduces CodeChemist, a test-time scaling (TTS) framework designed to improve the performance of CodeLLMs on low-resource programming languages (PLs). The authors define high-resource and low-resource PLs based on differences in CodeLLM performance on the same benchmark problems written in different languages. For example, CodeLLMs typically perform well in Python, making it a high-resource language, whereas C++ tends to have poorer results and is thus considered low-resource.

The proposed method proceeds as follows:

1. Generate multiple code solutions in a high-resource PL (e.g., Python).
2. Generate a set of $n$ input test cases.
3. Execute the Python solutions on these inputs to collect the corresponding outputs.
4. Use majority voting across Python solutions to determine the “correct” outputs.
5. Generate multiple candidate solutions in the target low-resource PL (e.g., C++) using hedged sampling (i.e., varying generation temperatures).
6. Execute the C++ solutions on the same $n$ inputs to obtain their outputs.
7. Compare the outputs of the C++ candidates with the “correct” outputs and select the best-performing one.

In their experiments, the authors evaluate CodeChemist on three benchmarks: MultiPL-E (including HumanEval and MBPP tasks) and Ag-LiveCodeBench-X (derived from LiveCodeBench), and cross three low-resource languages: Lua, C++, and Java. They compare CodeChemist against both the baseline model (without TTS) and other TTS methods using seven models in total: four sizes of Qwen 2.5-Coder, Llama 3 3B, GPT-4o-mini, and DeepSeek-V3.1. The results show consistent improvements over the base models across all benchmarks.

### Strengths
1. This paper presents a fresh and insightful perspective on test-time scaling for code generation, exploring a fundamental question about the limiting factor in CodeLLMs: is their weakness primarily due to understanding program semantics (the underlying logic) or handling syntax (the language-specific structure)? By disentangling these aspects through cross-language functional transfer, the work provides a novel way to examine how test-time scaling can bridge gaps between high- and low-resource programming languages.

2. CodeChemist proves is shown effective on code contest benchmarks, demonstrating consistent improvements across all evaluated models and benchmarks. The gains are particularly pronounced for smaller models, such as Qwen-2.5-Coder (1.5B) and Llama 3B, indicating that the proposed method not only scales well but also significantly enhances low-resource performance where model capacity is limited.

### Weaknesses
1. Scope: This method seems only useful when generating simple and small programs (i.e., coding contests) that run in the command line. They have to be easy to build and must take only one input and produce only one output.
2. Missing details in design: The authors have not explained some important details in the hyperparameters, such as the number of generated solutions in the high-resource PL and the number of test cases. The *Test Case Generation* paragraph in the ablation studies only justifies the need for sampling 10 high-resource PL solutions for voting to create test oracles, instead of explaining why 10 were sampled and how many inputs were generated.
3. The proposed method seems less effective on larger models.
4. Missing evaluation: The other TTS methods are only evaluated on HumanEval but are missing on the other two benchmarks.
5. Before discussing whether this approach works on these benchmarks, let's consider one of the first principles of programming languages: why do we design different programming languages? Programming languages are designed for different purposes, not just because some are easy to write and others are not. The development of different programming languages aims to address different real-world use cases. In many cases, it is not reasonable or practical to generate a Python solution for tasks such as low-level system or memory control, which are typically the use cases of the "low-resource" languages.

### Questions
1. Please address the concerns listed in **Weaknesses**.
2. Please explain the design choice of $n$. What is the number of $n$ when generating inputs, and how is this number determined? Intuitively, a larger $n$ would lead to a more precise selection in the final result.
3. What is the token cost of CodeChemist? The authors discuss the time cost of CodeChemist in comparison to other TTS methods and the vanilla LLM in Section 4.5, but the other and perhaps most important aspect of LLM TTS is missing: how many additional tokens are consumed by this method.
4. The authors mention that the experiments are performed on a single A100 GPU. However, the latency of the execution-based selection method also, and perhaps largely, depends on CPU clock speed.
5. Now that reasoning models are widely adopted, it would be valuable if the authors could compare the performance of a model with CodeChemist versus the same model with reasoning (thinking) enabled.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Code Chemist, a test-time scaling framework designed to improve the performance of CodeLLMs on low-resource PLs. The core problem addressed is the performance gap where models excel in high-resource PLs (like Python) but falter in low-resource ones (like Lua) due to data scarcity. The authors conduct experiments across multiple models (Qwen, Llama3, GPT-4o mini, DeepSeek) and benchmarks (MultiPL-HumanEval, MultiPL-MBPP, Ag-LiveCodeBench-X), claiming that CodeChemist significantly outperforms baseline and other test-time scaling methods, especially on low-resource PLs.

### Strengths
The paper tackles the problem of poor CodeLLM performance on low-resource programming languages.

The method is a test-time-only framework, which does not require costly model fine-tuning.

The evaluation across a wide variety of model families and sizes (Qwen, Llama, GPT-4o mini, DeepSeek) and multiple code benchmarks (MultiPL-HumanEval, MBPP, Ag-LiveCodeBench-X) .

### Weaknesses
The paper's main weakness is its limited originality. The method is an assemblage of existing techniques: Best-of-N sampling (re-branded as "hedged sampling"), test-case-based validation, and cross-language transfer. The contribution is an engineering one, not a fundamental research advancement, which overstates its novelty.

### Questions
The test oracle generation relies on majority voting from 10 high-resource samples. What analysis was done to ensure these oracles are correct? How does the method handle a systematic model failure, where the majority of the 10 samples are logically incorrect in the same way? Have you analyzed the failure cases where CodeChemist succeeds or fails based on the correctness of this generated "ground truth"?

Can you provide a clear and rigorous justification for categorizing C++ and Java as "low-resource" PLs in the context of "data scarcity"?

How does MTHS compare to a standard, well-tuned Best-of-N sampling baseline (e.g., all 10 samples at an optimal fixed $\tau$)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents CodeChemist, a test-time framework intended to improve code generation for low-resource programming languages. The core idea is to transfer "functional knowledge" from a high-resource language (like Python) to a low-resource one without any model retraining. The method uses a high-resource language (Python) to first generate code and derive a set of input-output test cases. These test cases are then used as a filter to select the best solution from multiple candidates generated in the target low-resource language. The authors conduct extensive experiments on benchmarks like MultiPL-E and Ag-LiveCodeBench-X, demonstrating that CodeChemist significantly outperforms vanilla generation and other test-time scaling baselines

### Strengths
1. The paper tackles the important problem of improving code generation for less-represented languages, which is crucial for the broader applicability of these models. 
2. The paper is well-written, and the proposed CodeChemist framework is explained with great clarity. The three-stage process is logical and easy to comprehend, making the work highly accessible.

### Weaknesses
1. Inappropriate baseline. The proposed method relies heavily on an external code execution environment to generate and validate solutions. However, the chosen baselines do not leverage this execution feedback in the same way. A more direct comparison would be against methods that also utilize execution feedback. For instance, a baseline could involve an iterative refinement loop where the model corrects its own code in the low-resource language based on feedback from failing test cases. Without such a baseline, it is difficult to determine whether the performance gain truly stems from the cross-language knowledge transfer from Python, or simply from the powerful signal provided by the test cases and the execution environment itself.
2. The idea of using test cases to enhance code generation is a well-established practice in the field. (e.g., GenX, RepoCoder, Epicoder) use test execution to filter, validate, or refine generated code. The authors should clarify the key differences between CodeChemist and these works in the paper.
3. The method for generating test oracles has potential weaknesses. It relies on an LLM to generate inputs and uses majority voting over LLM-generated high-resource code snippets to determine the correct output. This process lacks formal guarantees of correctness or coverage.

### Questions
1. How does the framework ensure that the test cases generated via the high-resource PL are sufficient to cover corner cases and complex logic?
2.  The experiments exclusively use Python as the high-resource "teacher" language. How dependent is CodeChemist's success on this specific choice? Have you experimented with other high-resource languages?
3. How do you envision CodeChemist scaling to more complex, multi-function, or modular code generation tasks where correctness depends not just on input-output behavior but also on state modifications and inter-function interactions?

### Soundness
2

### Presentation
3

### Contribution
2
