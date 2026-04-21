# xCodeEval: An Execution based Large Scale Multilingual Multitask Benchmark for Code Understanding, Generation, Translation and Retrieval

- Avg Score: 4.75
- Decision: Reject
- Scores: 6, 3, 5, 5

## Abstract
Recently, pre-trained large language models (LLMs) have shown impressive abilities in generating codes from natural language descriptions, repairing buggy codes, translating codes between languages, and retrieving relevant code segments. However, the evaluation of these models has often been performed in a scattered way on only one or two specific tasks, in a few languages, at a partial granularity (e.g., function) level, and in many cases without proper training data. Even more concerning is that in most cases the evaluation of generated codes has been done in terms of mere lexical overlap with a reference code rather than actual execution. We introduce **xCodeEval**, the largest executable multilingual multitask benchmark to date consisting of $25$M document-level coding examples ($16.5$B tokens) from about $7.5$K unique problems covering up to $11$ programming languages with execution-level parallelism. It features a total of $7$ tasks involving code understanding, generation, translation and retrieval. **xCodeEval** adopts an execution-based evaluation and offers a multilingual code execution engine, **ExecEval** that supports unit test based execution in all the $11$ languages. To address the challenge of balancing the distributions of text-code samples over multiple attributes in validation/test sets, we propose a novel data splitting and a data selection schema based on the geometric mean and graph-theoretic principle. Our experiments with OpenAI's LLMs and open-sourced LLMs on the tasks and languages demonstrate **xCodeEval** to be quite challenging as per the current advancements in language models. Both [xCodeEval](https://github.com/ntunlp/xCodeEval) and [ExecEval](https://github.com/ntunlp/ExecEval) are freely available at [Hugging Face](https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval) and [Github](https://github.com/ntunlp/ExecEval).

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a multi-programming-language code dataset based on algorithmic problems and natural language descriptions taken from codeforces.com. The dataset includes 17 languages, with 7.5K problems, where problems have solutions in multiple languages. The problems also have test cases, 63K in total, which can be used in program synthesis (code generation) tasks to verify the correctness of model-generated solutions. The dataset also supports code translation, classification (tagging, executability), retrieval (text<->code or multilingual code<->code retrieval), and program repair. The paper introduces a Docker-based execution testing environment, ExecEval, for the languages and problems. Finally, the paper evaluates ChatGPT, a fine-tuned StarCoder-3B model, and Code-Llama variants on the various tasks from the dataset, finding that performance varies across languages and ChatGPT performance is substantially higher than the other models.

### Strengths
S1) The paper uses execution-based testing for its program synthesis tasks, which is important when generating long/complex code, with multiple correct solutions, like the algorithmic tasks here.

S2) The paper's dataset is large-scale and consists of >10 programming languages. For the program synthesis task, this fills a gap in existing execution-based datasets that are either smaller-scale (e.g. MBXP, MultiPL-E) or consist of only a few languages (e.g. APPS, CodeContests).

S3) The EvalExec framework for executing generated code will likely be useful for other work constructing execution-based benchmarks. I appreciated the information in the Appendix about the API server, instrumentation of failure cases, and security measures / configurable resource limits.

S4) While the paper's contribution was mostly on the dataset side, some of the experimental analysis was also interesting, in particular the effect of ChatGPT's knowledge cutoff on performance.

### Weaknesses
W1) The contribution of the paper is spread a bit thin, in my opinion. On the dataset side, I think that the dataset will likely be useful for program synthesis, but given the existence of other similar datasets (although they either have fewer languages or smaller scale, see S1), I would want to see a bit more work verifying that the scale and multi-lingual nature of the dataset is a useful feature:

W1a) For scale, it would be helpful to verify that the problems and tests are high quality and consider automatically generating more tests, given that work like CodeContests [Li et al.] find that many naturally-occurring tests have spotty coverage, leading to false positives. 

W1b) For multi-lingual, it would help to do more analysis of difference in model performance across languages, perhaps as a function of the data the models were trained on (for open-source models with known training data, like StarCoder). 

W2) The analysis and model evaluation was a bit thin:

W2a) The experiments consisted mainly of ChatGPT with some additional experiments on open-source models. While I appreciated the use of open-source models, they were relatively small: CodeLlama-Instruct (up to 13B) and StarCoder (3B, fine-tuned) models, and given the very low performance of these models, and that a different model family was fine-tuned than used zero shot, I don't know that much can be drawn from the finding that StarCoder-3b fine-tuned outperformed CodeLlama-7b instruct.

W2b) I didn't feel that the temperature analysis or "reasoning spectrum" added much to the paper, as it was difficult to find a clear takeaway from them.

W2c) The other tasks beyond program synthesis were under-explored. Code translation and APR, in particular, seem potentially useful to me (in particular, I think it's exciting that APR is based on chronological submissions by a participant, as explained in E.5), but both of these had very limited experimental results. The difficulty of the retrieval task also seems to depend heavily on the size of the retrieval dataset (as evidenced in the text description of Table 4, with high vs low resourced languages), and I think future work here might need to introduce other metrics or account for the size of the datasets in some way.

W2d) I was curious about the distribution of tags in the dataset, as this will give a lot of information about the fine-grained types of algorithms that are involved.

W3) The writing of the paper could be improved. In particular, I'm afraid I didn't fully understand the motivation for or implementation of the data balancing method in section 2.1. I think that this might be better suited to the appendix, with the main text focusing on addressing some of the weaknesses above (e.g. additional experiments, or evaluation), or moving some of the appendix text on the Task Construction process to the main text.

### Questions
Q1)  I was confused by the \sum_{v \in V} f(u, v) = 0 in section 2.1, since the range of the flow is only non-negative integers. Does the direction of the edges negate some edge weights in the sum?

Q2) How do the top-k retrieval accuracy metrics account for there being multiple solutions in a given language per problem (if I understand correctly)?

Other clarification points (not necessary to answer in the author response):
- In the abstract, specify the source of the data (code contests problems)
- "parallelism of multilingual data" in the intro was unclear; might convey parallel programming.
- "Evaluation and its granularity" section of the intro was unclear about what global level meant or how to produce unit tests for it. 
- The description in section 2.1 had a lot of detail about the algorithm, but I felt it would be better served to give some high-level intuition instead. It was unclear to me why the method does val/test division before filtering (as filtering might affect the number of samples in val and test). 
- Figure 3 is hard to read since C and Rust have the same color. 
- Is Rust the only rare language? What about e.g. Go and Kotlin? 
- The description of "reasoning path" on page 8 was confusing, since it only evaluates the output of the code (PASSED vs WRONG ANSWER, etc), but to me "reasoning" conveys different algorithms or code implementations that all can produce PASSED solutions.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces xCodeEval, a large-scale, multilingual, multitask benchmark designed to evaluate code language models on ode understanding, generation, translation and retrieval.  The benchmark includes 25M coding examples from about 7.5K unique problems in 11 programming languages. It employs execution-based evaluation with both training and test sets. The authors conduct various benchmarking on leading models and show that xCodeEval presents a significant challenge to the current state of LLMs.

### Strengths
- The benchmark covers a wide range of code tasks.
- The docker-based execution-based evaluation can streamline the evaluation across platforms and potentially compliment many existing widely-used lexical-match-based benchmarks. 
- The authors presented various anaylsis and clearly discussed limitations and risks, which are crucial for benchmarks.

### Weaknesses
- While the paper appears to be comprehensive, it is extremely dense and not self-contained. It appears that the authors aimed at covering a ton of work and have to skip most details, resulting in most sections being cursory. Figures and tables are not presented clearly and barely readable. What's worse, the related work section is completely missing in the main body of the paper, which is unacceptable. The authors should consider how to streamline the content and make sure that the main paper is self-contained. 

- The novelty is limited. The authors use codeforces as the source dataset. On one side, it is similar to various existing code-competition execution-based benchmarks from HumanEval and MBPP to APPS, CodeContests, DS-1000, MBXP, HumanEval-X, MultiPL-E, BabelCode, etc. which cover a wide range of problems in different level of difficulty and/or multiple languages. Among all, xCodeEval overlaps a lot with CodeContests which was partially built on Codeforces too. On the other side, it is unclear how the rest of tasks, e.g., code retrieval, can benefit the current set of evaluations. The authors should clearly articulate the novelty of the work.

- The dataset comes with finetuning data, however, it wasn't explained well why we need finetuning data at the era of LLMs. How would it add value in benchmarking an LLM?

- The license of the dataset is CC-BY-NC, which significantly limits the usability of the dataset, especially given most LLMs come from the industry. Is this coming from the authors or codeforces? If the former, it would be great if the authors can re-consider the decision.

### Questions
See Weaknesses for questions. In addition,

- How does xCodeEval ensure the quality of its benchmarks across different languages?
- How would the new split algorithm (sec 2.1) improve over the baselines (e.g., random selection and/or time-based selection)?
- Did the authors obtain hidden tests (if any) from codeforces?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper presents xCodeEval, a benchmark designed for evaluating code generation models. This benchmark, notable for its size and scope, encompasses over 25 million coding examples from approximately 7,500 unique problems and extends support to 11 programming languages. A key feature of xCodeEval is its emphasis on execution-based evaluation, operationalized through the introduction of ExecEval—a multilingual code execution engine that uses unit tests to evaluate code in all supported languages. The authors test the benchmark with several pre-trained language models.

### Strengths
1. The paper introduces a large-scale, execution-based benchmark, xCodeEval, filling a gap in existing evaluation protocols that often rely on smaller datasets (HumanEval, MBPP) or non-execution metrics like BLEU and exact match.

2. A standout feature of this work is the parallel, distributed execution framework, ExecEval. This innovation allows for efficient and scalable evaluation of code across multiple programming languages, which is crucial for large-scale benchmarks.

3. The detailed analysis of OpenAI's LLM performance on xCodeEval offers valuable insights. It provides a clear picture of where current LLMs excel and where they struggle.

### Weaknesses
1. The benchmark draws exclusively from Codeforces, which may limit the novelty of the dataset, as similar approaches have been used in other benchmarks like APPS.

2. The reliance on a single platform like Codeforces means the benchmark might not capture the full spectrum of coding tasks. Basically the benchmark is limited to algorithm contest questions.

3. The high difficulty level of the benchmark, as evidenced by low pass rates (less than 4%) across several 3B/7B/15B LLMs in Table 5, suggests it may be too challenging for all models with less than 30B parameters. This limits its utility for evaluating a broader range of model sizes and capabilities.

4. There are concerns about data contamination when evaluating models such as GPT-3.5, which have been trained on extensive web data that may include the very solutions from Codeforces used in the benchmark, potentially skewing the results in favor of GPT-3.5. This is because xCodeEval contains questions from 2010 - 2022. It would be interesting to see whether GPT-3.5 has consistent performance on questions after 2022.

5. The visual presentation of the paper could be enhanced for better clarity and accessibility. Specifically, some graphics and tables, such as the legend in Figure 4, are difficult to read in print form.

In summary, while the paper contributes a valuable tool for advancing code LLMs, these aspects should be addressed to fully realize its potential.

### Questions
It would be interesting to see whether GPT-3.5 has consistent performance on questions after 2022.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents work done to enhance code LLM evaluation abilities, through a new benchmark that supports additional languages, includes multiple tasks for which code LLMs are being used today and additional framework to enable automatic evaluation through code execution.

### Strengths
Code LLMs are now mainstream and to go beyond small interactive code snippets, we need reliable evaluation mechanisms, especially those using code's advantage over other text - verifiable executability. This work furthers the body of paired samples enhanced with useful metadata which helps support multiple tasks. The work further validates the thesis by trying relatively smaller SoTA models like *Coder-base and Llama all the way to OpenAI's offerings.

### Weaknesses
While the work is important and highly relevant, the contribution feels incremental:
- there are plenty of enhancements coming into code datasets with additional processing of github based datasets like Stack v2
- the additional languages are also mostly mainstream and available in different datasets; it would have helped if we add really low resource languages (Stack v2 has shown that too with even languages like COBOL)
- execution based evaluation has been around for multiple years now, especially unit tests, The distributed execution as well as additional metadata is a great value add, but it's been meshed with other features in one paper - it might have been better to split these into separate submissions to be able to evaluate each on its own merit
- execution oriented evaluation also has its limitations and can lead to incorrect code being validated correctly especially if there's no return value or booleans; it doesn't help sufficiently with secure code generation or deprecated API etc.

Maybe the issue is with combining several concepts into one paper which limits the amount of presentation you can provide to each of the concepts - separating benchmark, execution framework, and multiple languages might have helped at least one to rise sufficiently.

### Questions
- What would it take to expand the work to include other low resource languages including from a different domain?
- Since you used StarCoderBase as one of comparison points, could you provide value differentiation of the new benchmark over the github content in Stack used to train it? One would expect most internet code to also be available in github and so all of the models used to compare might have been exposed to most of this code.
-

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
