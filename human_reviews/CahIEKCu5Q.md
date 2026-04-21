# CodeMMLU: A Multi-Task Benchmark for Assessing Code Understanding & Reasoning Capabilities of CodeLLMs

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 6, 5, 3

## Abstract
Recent advances in Code Large Language Models (CodeLLMs) have primarily focused on open-ended code generation, often overlooking the crucial aspect of code understanding & reasoning. To bridge this gap, we introduce CodeMMLU, a comprehensive multiple-choice benchmark designed to evaluate the depth of software and code comprehension in LLMs. CodeMMLU includes nearly 20,000 questions spanning diverse domains, including code analysis, defect detection, and software engineering principles across multiple programming languages. Unlike traditional benchmarks that emphasize code generation, CodeMMLU assesses a model’s ability to reason about programs across a wide-range of tasks such as code repair, execution reasoning, and fill-in-the-blank challenges. Our extensive evaluation reveals that even state-of-the-art models struggle with CodeMMLU, highlighting significant gaps in comprehension beyond generation. By emphasizing the essential connection between code understanding and effective AI-assisted development, CodeMMLU provides a critical resource for advancing more reliable and capable coding assistants.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors present CodeMMLU, a multiple-choice question-answer benchmark for code, consistenting of thousands of code/software-related questions, spanning syntactic knowledge (e.g., API and Frameworks, PL syntax), semantic knowledge (e.g., OOP, compiler design), and real-world tasks (e.g., code completion, code repair). This is inspired by the MMLU evaluation set used in NLP as well as programmer comprehension behavior models (Shneiderman & Mayber, 1979). The knowledge-based tasks are derived from W3Schools, Geeks4Geeks, and Sanfoundry, with LLM-based filtering. The real-world tasks are derived by re-purposing existing evaluation sets, with additional steps for synthesizing distractors with LLMs and execution-based filtering. The authors benchmark a large number of closed-source and open-source LLMs with CodeMMLU.

### Strengths
- This is a very large and diverse evaluation set, spanning many different topics, concepts, tasks, and programming languages. This could potentially be useful for the research community, providing a way forward for hill-climbing to improve code understanding in LLMs.
- The authors evaluate several open-source and closed-source code LLMs on this new benchmark, reporting numbers separately for syntactic, semantic, and real-world tasks. This is useful for understanding the weaknesses of current models, also paving a path forward for improvement.
- I particularly like the way the authors have generated hard distractors for the real-world tasks. This makes the task more difficult.
- There are some key insights from this work that are quite interesting. For instance, the authors show that CoT prompting is often not effective for CodeMMLU tasks, whereas few-shot prompting seems to consistently perform well. Additionally, the authors compare performance of HumanEval as a generative task versus as a MCQA task, through which they show that MCQA performance can be much lower sometimes. This suggests that generative tasks do not adequately evaluate a model's code reasoning capabilities.

### Weaknesses
- **Not clear whether this is a reliable evaluation set.** The correlation with human judgement has not been measured. The authors motivate this work by highlighting the issues of potential data leakage with existing benchmarks (L036). However, it seems that CodeMMLU is susceptible to the same issue. Data sources like W3Schools, Geeks4Geeks, and Sanfoundry are likely already in the pretraining sets of existing models. Additionally, the real-world tasks are based on existing benchmarks, which have leakage issues, as the authors claimed. Next, Figure 7 and Table 4 suggest that the performance is very sensitive to the position of the correct option, which suggests that there are factors beyond code comprehension at play in MCQA. Therefore, it is not clear whether we can rely on this for evaluation code comprehension.


- **Many missing details and also some details which are inconsistent.** First, it is not clear how the authors generated MCQA questions and hard alternative options for the knowledge-based tasks. Next, 10K is approximation given in the abstract for the number of examples in CodeMMLU. However, the sum across subjects in Table 2 is 20, 281. Does that mean there are some duplicates? Furthermore, the number of models that have been benchmarked is not clear. In Section 4.1, the authors say 35 open-source models (L312) and 3 closed-source models (L319). However, the number of rows in Table 3 do not align with this. In L375, the authors say they have evaluated 43 models across 10 model families. In L976, the authors say they have experimented on 48 LLM models from over 12 families. Additionally, some of the results are difficult to interpret. For example, there is no y-axis for Figure 5 and also the prompting style is not actually labeled in Figure 9.


**Suggestions**:
- Is 3.2 mis-labeled? Should it correspond to "Real-world problem set"
- Place Table 3 before Figure 4.
- Currently, Figure 5 is referred to before Figure 4. Maybe switch the numbering?
- L345: Detaill $\rightarrow$ Detail 
- Table 3 is confusing. CodeMMLU is the aggregate score across Syntacic, Semantic, and Real-world tasks? Make this clear by saying "Combined" instead. CodeMMLU includes all.
- L426: There is no Table 8. Seems that the authors intended Figure 7.

### Questions
Please address the concerns raised in the Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes CodeMMLU, a benchmark for evaluating LLMs in code understanding using multiple-choice questions. CodeMMLU consists of a group of different tasks across knowledge-based tests and real-world programming questions. The authors evaluated 35 LLMs on the CodeMMLU benchmark, and the results suggests that the performances of LLMs on CodeMMLU is not always consistent with the performances on code generation benchmarks.

### Strengths
+: The authors proposed a new benchmark in code understanding, which is long ignored in LLM for code evaluation.

+: CodeMMLU consists of a wide variety of code comprehension tasks, from syntax/semantic understanding to code repair and defect prediction.

+: The authors conducted extensive experiments on CodeMMLU with various LLMs.

### Weaknesses
-: For most tasks, the authors use LLMs to generate distractors. The quality of these generated distractors should be discussed.

-: The code completion and fill-in-the-blank tasks are more related to code generation instead of code understanding. Especially, the code completion task is based on the existing HumanEval dataset.

### Questions
- The decline in accuracy with COT prompts is interesting. Perhaps it's better to analyze the LLMs' answers with COT in detail.

- In section 3.2, why is predicting execution output under the same category as defect prediction?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
his paper presents CodeMMLU, a collection of almost 20,000 multiple choice questions and answers on coding, collected from the web.

The paper covers three areas of related work: benchmarks for program comprehension, models for program comprehension, and multi-task question/answering benchmarks.

The paper collects programming-relaed multiple-choice questions from web sites like GeeksForGeeks, W3Schools, and Sanfoundry. The questions cover syntactic (programming languages and API) and semantic (algorithms, design, OO) knowledge, as well tasks (code completion, fill-in-the-blank, defect detection, and code repair).

These 20,000 questions were evaluated on 35 "open source" code LLMs. 
Key reported insights include:

- Performance on knowledge questions and task questions is correlated
- LLM's preference for certain answers (e.g., avoid 'A') in multiple-choice questions is also present in code models
- Chain-of-Thought is not helpful for these questions
- Turning HumanEval into multiple choice questions changes model performance

### Strengths
The dataset is impressive, with almost 20,000 questions, and represents a substantial amount of (manual) work

The grouping of questions is meaningful

The paper is easy to read and follow

The dataset puts code language model performance in a different perspective.

### Weaknesses
While I liked the paper, my main concerns are:

- While filters and manual processing are applied, these are described only very briefly. As a result, the quality of the questions is unclear. It appears the paper favors quantity over quality. Fewer questions but of guaranteed quality would have been better.

- The questions for the 'real world tasks', e.g., for defect detection or line completion, are very artificial

- The treatment of multiple-choice bias (models prefering to avoid option A) in the paper is unsatisfactory

- It is unclear to what extent the LLMs were exposed to the underlying resources for this new data set (leakage). This risk is not discussed nor mitigated.


The writing and presentation are generally good, yet is sloppy at places (the abstract speaks about "over 10,000" questions -- there are 19,900, which is more like 20,000, 3.2 speaks about "five distinct" tasks, but there are four, there is no table 8 (only a figure 8), ...). It is confusing that the text summarizing table 2 gives very different numbers from what is in the table ('over 3000' when in the table it appears to be closer to 5000, and 6000 when it is in fact 7000). I'm not sure why section A.3 is entitled "visualization" (nothing is visualized -- examples are given).

The filtering process is described, but the exact numbers involved (before/after filtering) are not provided. The filtering involves various manual steps -- applied to how many cases? Deep learners are used here, but no details are provided.

Referring to the tasks as "real world performance" is misleading. The tasks are still highly artificial. Concerning the tasks themselves:

- I was surprised to see 'code completion' tasks based on HumanEval -- HumanEval suffers from many problems. There is a vast amount of literature on LLM assisted code-completion using data that is better than HumanEval.
- The defect detection task appears to be about predicting the correct execution result -- which is a different task from defect detection. Again, there are lots of defect benchmarks around, with real bugs collected from actual systems (e.g., defects4j). It is not so clear what these mutliple choice questions add to that, especially with the weak distractros (like compile error, "internal error" (??))

The reporting of the results about preference for option A (figure 7, table 4) is very minimalistic. The bias is stated, but not really studied / explained, nor are mitigation measures such as proposed by Zheng et al applied. The paper writes that 'we experimented' with multiple answer order, but what exactly was done is unclear.
I must say these findings also undermine the whole endeavour. If the multiple choice format itself is a problem, what is the point of having a large multiple choice data set?

It is unclear how the dataset will be distributed. I would believe some of the data is copyright protected (e.g., W3Schools). This would mean you cannot redistribute their multiple choice questions.

### Questions
Do all questions have four alternatives (one correct and 3 distractors)? At any rate, a random guesser would get 25% right, which makes the results in Table 3 less impressive.

Were the questions or the sources they were derived from included in the training data of the language models benchmarked? How are such risks mitigated?

Can you explain the "CodeMMLU" column in Table 3? I thought it was the overall performance, but it's not explained. How can the Meta-Llama-3.170B-Instruct overall be 60, while the other three columns are all above 64? (Also please align around decimal point instead of centering numeric columns).

Can you explain how you will distribute the dataset and under what licence the original material was made available, and which license you intend to use?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The authors propose CodeMMLU, a new benchmark designed specifically to evaluate LLMs' code understanding capabilities. This benchmark includes over 10,000 questions covering various domains and programming languages. The question types include knowledge-based evaluations (such as programming language syntax, API usage, and software development principles) and practical programming tasks (such as code completion, fixing, and defect detection). The authors test various LLMs using this benchmark and provide insights into model performance, prompting strategies, and their correlation with practical programming skills.

### Strengths
1. CodeMMLU abandons traditional code generation evaluation methods and adopts a multiple-choice question format, shifting the focus from code generation to code understanding to evaluate LLMs' code understanding abilities.

2. The CodeMMLU benchmark includes over 10,000 diverse questions with broad coverage and high-quality sources (such as GeeksforGeeks, LeetCode, etc.). The authors have put in substantial work overall.

3. The authors conduct extensive experiments on CodeMMLU, providing experimental insights into multiple aspects such as selection bias in multiple-choice questions and correlations between LLMs' software knowledge and real-world applications. The authors also provide numerous tables, figures, and other visualizations to help readers understand the paper.

### Weaknesses
1. In the Related Work section, the authors simply list some code evaluation benchmarks without clearly articulating the specific differences between CodeMMLU and each existing benchmark, nor do they explain what issues CodeMMLU addresses that current benchmarks do not. In contrast, there are many comprehensive and thorough benchmarks for code understanding and code generation, such as CodeXGlue, XLCoST, xCodeEval, CodeScope, LiveCodeBench, and BigCodeBench. Notably, CodeScope (ACL 2024) has constructed a code understanding benchmark that includes four tasks and multiple-choice questions. Additionally, LiveCodeBench addresses the issue of data leakage through dynamic data set updates. CodeMMLU lacks detailed and thorough comparative analysis of key related works (a good example of related work analysis can be found in ClassEval). Moreover, the paper does not offer any solutions to the data leakage problem mentioned in line 36. Overall, although the paper involves considerable effort in data labeling and other aspects, it lacks novelty, with many conclusions already available in previous literature. The innovation and actual contributions of the paper remain unclear.

2. In paper, the authors discuss how LLMs are sensitive to the order of options in multiple-choice questions, which can lead to fluctuations in performance and thereby affect the accuracy of the results. Although the issue is acknowledged, unfortunately, no strategies for eliminating this bias are provided. This raises concerns about the reliability and robustness of benchmark tests. Moreover, the paper "Large Language Models Are Not Robust Multiple Choice Selectors," published at ICLR 2024, has demonstrated that using multiple-choice questions to evaluate LLMs is not stable and introduces significant biases. Given this, I am curious about how the authors address this issue.

3. The overall writing of this paper needs improvement, particularly in the areas of data handling and presentation where essential detailed explanations are lacking. Specifically, in Figure 3, titled "Overview of the CodeMMLU Data Creation Pipeline", the authors fail to clearly explain the process of constructing multiple-choice questions from various data forms after filtering. Moreover, the inclusion of the LLMs evaluation in Figure 3 is not well-explained. In line 319 of the text, although four models "GPT-3.5, GPT-4, Claude-3-opus, and Claude-3.5-sonnet" are tested, the manuscript inaccurately mentions "including three proprietary models." Additionally, the citations for Claude-3-opus, Claude-3.5-sonnet, and Qwen2 among others are incorrect and urgently need correction. Concerning the topic categorization in Table 2, the paper does not provide a valid explanation or methodology for the classification. Several key steps in constructing the evaluation benchmarks are also lacking thorough explanations and supportive descriptions, even in the appendix. Overall, these issues raise concerns about the quality of the manuscript, and it is recommended that the authors give more attention and detailed exposition to these critical areas in the revision.

### Questions
1. Can the authors provide additional insights or data to illustrate the correlation between the model's performance on CodeMMLU and its real-world application in software development environments, where code generation is more prevalent?

2. In line 212, the authors claim to use a deep learning-based filtering model to automatically remove low-quality or irrelevant questions. I do not understand how the authors ensure data quality. Is it solely based on prompts? What model was used for checking? Are there any rule-based verification methods involved? I have reviewed sections A.2.1 and A.2.2 of the appendix and found no clear explanations. In line 1043, the authors state, “Followed by a manual validation step to ensure their appropriateness for the benchmark.” Did the authors really manually review over 10,000 samples? This seems hard to believe.

### Soundness
2

### Presentation
1

### Contribution
2
