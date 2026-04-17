# CrossPL: Systematic Evaluation of Large Language Models for Cross Programming Language Interoperating Code Generation

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6, 6

## Abstract
Large language models (LLMs) have shown strong performance in single-language code generation, but how well they produce cross-programming-language (CPL) interoperating code, which is widely used in cross-platform and complex software systems, remains underexplored. Therefore, a benchmark for evaluating CPL interaction code generation is essential. However, Constructing such a benchmark is challenging owing to sparse interoperating code in real-world multi-programming-language projects, diverse Inter-process Communication (IPC) mechanisms, vast Foreign Function Interface (FFI) language pairs, and the difficulty of evaluation. To address this gap, we introduce CrossPL, the first benchmark for systematically assessing LLM performance of CPL code generation across two primary interoperation modes and 2534 tasks, specifically 1,982 IPC tasks spanning six languages and 522 Python–C FFI tasks. Its construction involved a review of CPL documentation, 156 finite state machines, and analysis of 19,169 multi-language GitHub repositories. Two LLM-based workflows are designed for automating the benchmark construction and evaluation, and assess 20 state-of-the-art LLMs. Results reveal clear limitations: the best model achieves only 19.5\% Pass@1 and 26.46\% Pass@5 on the FFI subset, in sharp contrast to the strong performance of these models on single-language benchmarks. These findings underscore the urgent need for improving LLMs regarding CPL interoperating code generation. The benchmark and code are available at https://github.com/newxzh/crosspl.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The goal of this paper is to build a benchmark to evaluate the ability of LLMs
to generate code that spans multiple programming languages. It evaluates both
language interoperability using both IPC and FFI. For IPC, it considers several
techniques and language pairs. For FFI, it only considers FFI between Python
and the GNU Scientific Library (GSL) which is written in C.

The benchmark has 2,000+ tasks, based on existing GitHub repositories. It
achieves this scale with LLM-authored prompts.

### Strengths
- An important task to benchmark. To the best of my knowledge, this is not
  a well-studied task.

### Weaknesses
I really think the paper is on to something by working on this problem. However,
I am not convinced by the benchmark construction methodology. There are two
main weaknesses:

- Very high likelihood of benchmark contamination: the paper starts with
  existing repositories, which makes contamination likely. I am aware that
  there are other, popular LLM benchmarks with the same flaw, but that doesn't
  mean that they are good benchmarks, even if they are very popular.

  To be concrete, consider that the all the Python-C FFI tasks involve the
  GNU Scientific Library (GSL). Unsurprisingly, there is a well-established
  Python FFI to the GSL on GitHub: https://github.com/pygsl/pygsl

  So, to what extent is the benchmark just measuring the willingness of LLM
  developers to train on GPL code?

- Model generated prompts can be unnatural: the problem with an LLM generated
  prompt is that it can leak very peculiar implementation details unless it is
  carefully crafted. For example, in my opinion, the prompt on Page 34 (Appendix) 
  literally specifies all implementation details. In fact, the prompt is
  longer than the canonical solution.

### Questions
See the weaknesses listed -- there are two primary concerns that I have with the paper.

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
This paper presents CrossPL, a benchmark suite for systematically evaluating large language models (LLMs) for generating cross-programming language interoperating code (CPL interoperating code). The benchmark covers two major interoperability mechanisms: IPC and FFI, encompassing 2,534 tasks (1,982 IPC tasks and 522 Python–C FFI tasks). The authors modeled interfaces based on 156 FSMs, mined 19,169 multi-language GitHub repositories, and designed two LLM-driven build and evaluation workflows (one each for IPC and FFI), ultimately evaluating 20 representative LLMs. Key findings: On the FFI subset, the best model achieved only Pass@1=19.54% and Pass@5=26.46%, in stark contrast to the strong performance of single-language code generation, revealing the significant shortcomings of current LLMs in generating CPL interoperating code.

### Strengths
- The problem definition is novel and important: this is the first systematic focus on "cross-language interoperable code generation" rather than translation/monolinguistic operation.
- The experiments are comprehensive, making good use of FSM formalization, large-scale repository mining, LLM pipelines, and FFI executable environments for experiments.
- The experimental results can reveal the difficulties of FFI and the heterogeneity of IPC in the cross-model/language/protocol panoramic comparison.

### Weaknesses
- The experiments and methods limit the scope of generalization, making it difficult to prove that the current approach is effective for multiple libraries, platforms, and languages simultaneously.
- IPC correctness is determined only by FSM matching and runtime robustness is not measured, which lacks important evaluation metrics.
- There is no systematic exploration of the impact of hyperparameters and decoding strategies on the overall method. Only the impact of fixed temperature and top-p on the experimental method is given.
- The paper doesn't propose a novel technical path or implementation; its technical paths rely on elements of established, mature implementations. Furthermore, the FFI is limited to Python-C + GSL, and the IPC covers only six languages/seven technology categories, limiting the originality of this "first system benchmark" in broad CPL interoperability.
- The paper's most important conclusion is a negative result (Pass@1 ≈ 19.5% in the FFI subset), but this finding is primarily confirmatory rather than surprising: given the narrow Python-C/GSL setting and the IPC caliber of FSM-compliant simulation, the conclusions are difficult to extrapolate to the broader CPL ecosystem.

### Questions
- IPC currently defines "correct" based on FSM matching. Can you provide end-to-end run pass rates and abnormal scenario results?
- Whether statistical significance tests (e.g., paired tests/confidence intervals) were performed for the improvement or decrease in Fig. 3/4
- Why does think mode have a "limited or even negative" effect on IPC but is beneficial for FFI? I think you need to provide a mechanistic analysis of the failure example.

### Soundness
2

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
2

### Summary
CrossPL introduces a benchmark for cross-programming-language (CPL) interoperating code generation, covering two modes: IPC (1,982 tasks across 6 languages and 7 techniques) and Python↔C FFI (522 tasks). The benchmark is largely auto-constructed with 156 FSMs plus two LLM-driven workflows (for mining/templating IPC snippets and for building/validating FFI tasks from GSL). Evaluation of 20 LLMs shows stark gaps versus single-language coding: best Pass@1 is ~80%+ on some IPC slices but only 19.5% Pass@1 on FFI (GPT-4o), highlighting interoperability as an unsolved frontier. The paper also provides error taxonomies and an analysis of “think” mode (helpful for FFI, mixed for IPC). Overall, this is a timely benchmark with solid engineering, though several choices (Python-C-only FFI, FSM-only IPC validation, LLM-generated instructions) raise concerns about coverage, validity, and potential bias/contamination.

### Strengths
Clear problem framing: Shifts evaluation from translation/single-language coding to true cross-language interaction.

Scale & breadth: 2,534 tasks; IPC spans 6 languages × 7 techniques; rare in this area.

Methodological novelty: Use of 156 FSMs to mine/validate IPC protocols is systematic and reusable.

### Weaknesses
FFI narrowness: FFI limited to Python↔C (GSL); may not generalize to JNI, CFFI, Rust FFI, JS↔C++ addons, Swift bridging, etc.

IPC validation fidelity: FSM conformance ≠ functional correctness; many IPC tasks are not executed end-to-end (risk of false positives).

Dataset skew: Language/technique distribution is uneven (e.g., heavy Java/HTTP), which can bias aggregate metrics.

Contamination risk: Mining popular repos and widely documented patterns without a contamination analysis could inflate results.

### Questions
PC validation fidelity: What fraction of IPC tasks are executed end-to-end vs only FSM-matched, and what are the FSM false-accept/false-reject rates?

Contamination/time split: Did you create a time-based holdout (repos/docs post-training) to estimate training-data leakage and its impact on scores?

FFI generalization: Why restrict to Python↔C (GSL)? Any pilots for JNI/Rust/Node add-ons/CFFI—and a plan to broaden FFI coverage in v2?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces CrossPL, the first benchmark for systematically assessing LLM performance in Cross Programming Language (CPL) interoperating code generation.

### Strengths
- The writing is fluent and logically clear.
﻿
- The dataset and evaluation code are open-sourced, facilitating replication of the work and future use by the community.
﻿
- The first benchmark for CPL interoperating code generation
﻿
- It highlights the widespread inadequacy of large models in this task, providing a new perspective for improving LLM capabilities.
﻿
- The benchmark is large-scale and provides a methodology for automated benchmark construction.
﻿
- The use of FSM-based validation simplifies the evaluation process.

### Weaknesses
- CrossPL-FFI is limited to Python–C tasks, and even focuses on the GSL library, raising concerns that the conclusions might be overly assertive or not generalizable.

- There remains a gap between FSM-based validation and actual execution, which may affect the accuracy of performance assessment.

Tips

- The abbreviations IPC and FFI are not fully defined upon their first appearance in the abstract, which could hinder readability.

### Questions
- Was there any manual review process for the 156 FSMs? Could insufficient coverage of the FSMs lead to an underestimation of model performance?

- For IPC tasks, the FSM validation focuses on state transitions, but does it overlook performance aspects (such as latency or concurrency) in the evaluation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents a cross-programming-language benchmark designed to systematically evaluate large language models (LLMs) on cross-language code generation tasks. The benchmark includes 2,534,982 interprocess communication tasks across six programming languages, as well as 522 Python–C foreign function interface (FFI) tasks. The study is motivated by the growing prevalence of real-world systems that integrate two or more programming languages. Through extensive experimentation, the authors provide insightful analyses of LLM performance, highlighting differences in capabilities across languages and programming techniques.

### Strengths
* The benchmark is large in scale, with the IPC task subset covering multiple real-world repositories and the FFI subset derived from a high-quality open-source repository.
* The benchmark construction pipeline is fully automated, leveraging established software engineering methods to ensure correctness.
* Although the benchmark is synthetically generated, the authors validate correctness through stochastic verification, including compilation and execution tests and a comprehensive suite of test cases.
* The evaluation spans a diverse set of LLMs, encompassing both proprietary and large open-source models, providing a balanced comparative analysis.

### Weaknesses
* The manuscript employs an excessive number of acronyms within individual paragraphs across multiple sections, which negatively affects readability. Including a dedicated background or terminology section early in the paper would help introduce key terms and abbreviations, improving overall clarity.
* The literature review omits several major code benchmarks, including SWE-bench, Multi-SWE-bench, and MBPP, which are essential for contextualizing the contribution.
* Although a manual review was conducted, the methodology and outcomes are not reported, limiting transparency and reproducibility.
* The pipeline’s performance appears to rely heavily on large, high-capacity LLMs, which may limit its reproducibility and scalability when using smaller open-source models.
* Despite identifying outliers, most LLMs achieve similarly high scores on both IPC and FFI tasks, suggesting the need for further task annotation and filtering to better capture differences in task complexity.

### Questions
* Benchmark design: Multi-programming-language tasks can often be represented within repository-level benchmarks. Why do you believe your benchmark design offers a superior approach compared to, for instance, extending SWE-bench to include GitHub repositories that natively incorporate multiple programming languages?
* FSM creation: Based on the main paper, I understand that the finite state machines (FSMs) were not generated by LLMs. Could you please confirm this?
* Manual review details: What was the size of the manually sampled subsets used for human review (referenced in Lines 254 and 287)? Could you describe the review process in more detail and specify how many reviewers participated?
* Filtering and model accuracy: Could you provide detailed percentages of instances that were filtered or pruned at each LLM-judging step? Additionally, do you believe that smaller or less powerful LLMs could achieve comparable accuracy on these tasks?

### Soundness
2

### Presentation
2

### Contribution
3
