## Human Reviewer 1

### Summary
This paper introduces EditBench, a novel benchmark designed to evaluate Large Language Models' (LLMs) capabilities in instructed code editing. Unlike existing benchmarks that rely on synthetic or competitive programming problems, EditBench's key contribution is its data, which is sourced from real-world developer workflows. Data was collected via a custom VS Code extension from nearly 500 users, capturing user instructions and code context in the wild. The benchmark comprises 545 problems spanning diverse real-world use cases, from bug fixes to feature additions. EditBench is the first to systematically require models to leverage rich contextual cues—including the user instruction, the entire code file, highlighted code snippets, and cursor position—to solve problems, thus more accurately simulating a developer's environment. It features diversity across 5 natural languages (including Chinese, English, etc.) and Python/Javascript programming languages. The authors evaluated 40 different LLMs, finding that EditBench is highly challenging; even the best-performing model (claude-sonnet-4) achieved only 66.67% pass@1. The study further demonstrates the significant impact of contextual information and task categories on model performance.

### Strengths
S1: The primary strength is the in-the-wild data collection methodology, which captures non-templated, natural user instructions and context from a real IDE environment, reflecting authentic developer challenges. 

S2: The use of a rigorous process involving human experts and secondary review to create the test suite addresses the inherent difficulty of creating a verifiable benchmark from raw user behavioral data, ensuring the high quality and trustworthiness of the results.

S3: The design explicitly tests the models' ability to integrate contextual cues (file content, highlights, cursor), which is shown to be crucial for performance, thus providing valuable insights for future LLM development.

### Weaknesses
W1: EditBench currently supports only Python and Javascript. For a benchmark aiming for real-world representation, the lack of support for other major industry languages like Java, C++, Go, C#, and TypeScript limits its broad applicability.

W2: The extended version, EditBench-complete, relies on GPT-4o for translating non-English instructions and comments. While this increases linguistic diversity, the paper lacks detailed evidence of human quality validation for the LLM-translated content. This raises a concern about potential loss of naturalness or subtle nuance of user intent in non-English problems, potentially diminishing the "in-the-wild" authenticity of this subset of data.

W3: The paper mentions code context lengths can be long (average file length 4.5k tokens) but provides minimal dedicated analysis on model performance differences concerning context length. Given that LLM performance on long-context tasks is a major research area, a specialized analysis contrasting performance across different context length bins (e.g., short, medium, long) would have been highly valuable.

### Questions
Q1: The paper notes that models perform best when given highlighted code but not the cursor position, and that context can affect performance by up to 11%. Could the authors provide a more detailed ablation study in the appendix or rebuttal, clearly listing performance metrics for all four combinations: (a) No extra context; (b) Highlighted code only; (c) Cursor position only; (d) Highlighted code and cursor position. This would clarify why the cursor position appears to have a detrimental or unhelpful effect and quantify the exact utility of each cue.

Q2: Given that non-English problems in EditBench-complete were generated via GPT-4o translation, what specific quality control steps were taken to verify the naturalness and preservation of the original user intent in these translations? Were the non-English problems (e.g., Chinese, Russian, etc.) reviewed by native speakers to ensure quality?

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper introduces a new benchmark, EditBench, for evaluating LLM code editing capabilities grounded in real-world usage, i.e., user instructions and code contexts collected in the wild. EditBench comprises of 545 problems, multiple natural and programming languages, and a diverse set of real-world use cases, ranging from resolving errors to adding features. 40 diverse LLMs have been evaluated and only 3 models score over 60% on EditBench, indicating it is a challenge set of problems. Also, empirical results and analyses show that, varying levels of contextual information greatly affect task success rate.

### Strengths
1. This paper proposes a new benchmark EditBench for evaluating LLM code editing capabilities grounded in real-world usage.
2. Only 3 models score over 60\% on EditBench, indicating it is a challenge set of problems.
3. The presentation is good and the paper is easy to follow.

### Weaknesses
1. No codebase is released or provided to confirm the reproducibility.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper presents a benchmark for code editing, where code editing
means working with a VSCode extension and using a sidebar LLM chat window
to edit code. The paper recruited 500+ participants to use a VSCode extension
that was developed by the authors. They received  appropriate IRB
approval and take care to manage PII.

The paper then puts in the effort to turn the raw data into a benchmark
suite with test cases. The benchmark is quite challenging and much more
realistic than prior work in this space.

### Strengths
I think this paper sets a new bar for coding benchmarks. Almost every other
benchmark that I can think of for any coding tasks (and not just code editing)
uses artificial tasks and prompts. The only related work I can think of that
uses natural prompts is the StudentEval benchmark (Babe et al, 2023), but the
tasks in that are quite trivial compared to what is presented here.

I have gotten quite tired of evaluating LLMs on made up tasks that I know are
trivial compared to the kinds of prompts that get used in the wild. This paper
helps change that. I would personally use this benchmark in my own work. It
is easy to see that other researchers would also find it very useful.

### Weaknesses
- The only weakness I can think of is that the context provided to the
  models seems to be just the current editor window. Tools like Cursor
  are well beyond this and supply significantly more context. However, 
  I think this simplification is the right one for a benchmark.

### Questions
- I don't quite understand Appendix A.1. This looks like the system prompt
  for editing. What does "collecting user votes" means? Actually, section D
  makes it clear that this is not the prompt template for the model, so I
  am quite confused.

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
10

### Confidence
5

---

## Human Reviewer 4

### Summary
The paper presents EditBench, a benchmark designed to evaluate large language models (LLMs) on real-world instructed code editing, where LLMs are guided by natural language to modify existing code. To address the lack of realistic interactions, the authors built a VS Code extension that collects in-the-wild user instructions, highlighted code, and cursor positions from nearly 500 developers. After filtering and manual annotation, they curated 545 problems across five natural languages (English, Spanish, Russian, Chinese, Portuguese) and two programming languages (Python, JavaScript), each paired with test harnesses for automated evaluation. Evaluating on 40 LLMs show that EditBench is highly challenging where only 3 models (Claude-Sonnet-4, GPT-o3-mini-high, Claude-3.5-Sonnet) exceed 60% pass@1. Including highlighted code improves performance by up to 7%, while cursor position lowers it.

### Strengths
1. EditBench is collected directly from real-world interactions, which makes it more representative of practical coding workflows.
2. EditBench incorporates diverse user instructions, code context, highlighted code, and cursor position.
3. Multilingual benchmark across five natural and two programming languages broadens applicability.
4. Comprehensive evaluation with 40 LLMs spanning reasoning and non-reasoning models.

### Weaknesses
1. The authors keep stating that they include "both interesting and challenging (line 158)" or removing "ambiguous (line 178)" problems, but there is a lack of explanation what problem refers to bad or good problem.
2. In the evaluation, the authors split EditBench into easy and hard groups, but they do not provide clear criteria explaining how this division was made.
3. Following up on the previous concern, the authors claim that “\emph{hard instructions tend to have shorter instructions (by nearly 5 times) but slightly longer highlighted code}.” It is unclear whether the shorter instructions are compared to those in easy problems or to the length of the highlighted code itself. For either perspective, there is a lack of in-depth analysis to figure out the root cause.
4. From the example shown in Table 5, the instructions collected by EditBench appear less clear than the annotator-provided prompts in other benchmarks. This may explain why all LLMs perform poorly with such instructions. It would be helpful to know what the upper bound of performance is when following these instructions.
5. It is notable that older reasoning models such as o3 and o3-mini-high outperform GPT-5. A deeper analysis of this result would be valuable to understand why newer models lag behind in this setting.

### Questions
1. What specific criteria were used to determine which problems were considered “interesting and challenging”?
2. How did the authors define and measure problem difficulty when splitting EditBench into easy and hard groups?
3. What is the estimated upper bound of model performance when following the instructions in EditBench, and was any human or expert-based evaluation used to assess the clarity and followability of these instructions?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
3