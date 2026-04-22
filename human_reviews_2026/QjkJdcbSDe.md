# The Matthew Effect of AI Programming Assistants: A Hidden Bias in Software Evolution

- Avg Score: 4.40
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4, 2

## Abstract
AI-assisted programming is rapidly reshaping software development, with large language models (LLMs) enabling new paradigms such as vibe coding and agentic coding. While prior works have focused on prompt design and code generation quality, the broader impact of LLM-driven development on the iterative dynamics of software engineering remains underexplored. In this paper, we conduct large-scale experiments on thousands of algorithmic programming tasks and hundreds of framework selection tasks to systematically investigate how AI-assisted programming interacts with the software ecosystem. Our analysis quantifies a substantial performance asymmetry: mainstream languages and frameworks achieve significantly higher success rates than niche ones. This disparity suggests a feedback loop consistent with the Matthew Effect, where data-rich ecosystems gain superior AI support. While not the sole driver of adoption, current models introduce a non-negligible productivity friction for niche technologies, representing a hidden bias in software evolution.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper explores how AI has influenced or will influence software development by investigating AI's performance asymmetry across different programming languages and frameworks. To this end, they constructed a large-scale benchmark to evaluate models' coding capabilities across eight languages and five categories of software development tasks. They reveal a Matthew effect: the more popular a programming language or framework is, the higher the success rate of LLM-generated code.

### Strengths
- They construct a large-scale benchmark covering various programming languages. Considering that existing benchmarks mainly focus on a limited set of languages such as C++ or Python, their benchmark is valuable for assessing models' coding capabilities across diverse languages.
- Their evaluation is systematic, including various metrics and statistical analyses.
- They connect their evaluation findings to a broader societal perspective by discussing the Matthew effect in AI programming assistants.

### Weaknesses
I don't think the results themselves are very surprising. It's well known that models trained with more examples in their training data tend to show higher performance on those tasks. Even though the paper describes the situation as a "striking" Matthew effect, I personally don’t find it surprising. In fact, as models evolve over time, such an effect might diminish due to their generally higher capabilities. Moreover, even if the Matthew effect persists, one could create specialized models for each programming language through fine-tuning to mitigate the issue.

Another question is how significantly the Matthew effect will impact software development. In any case, whether or not there is a Matthew effect in AI assistants, software development itself has long been centralized (as shown in Table 1). It is unclear whether the Matthew effect from AI will have a substantial influence on software development, or what the negative consequences of this situation might be. I think the paper needs to discuss the meaning of the Matthew effect, its negative consequences, and whether the effect will persist.

### Questions
Please see the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors of this paper proved empirically the existence of training data gaps/bias for full-stack engineering and algorithmic code generation tasks across various programming languages, software frameworks using five commercial LLMs. They have contributed a comprehensive benchmark dataset to track progress/improvement of this bias over time.

### Strengths
- Excellent job on highlighting the gaps in training data for full-stack engineering and algorithmic code generation tasks across various programming languages, frameworks in the leading commercial LLMs
- Good call out on LLMs potential to amplify these training biases in the emerging coding paradigms such as vibe coding.
- Good presentation of the topic overall.

### Weaknesses
- The paper does not include justification for why the leetcode & CRUD task datasets were chosen as valid proxies for the larger programming ecosystem. Empirical scope of the study based on datasets explanation in section 3 seems narrow for below reasons 
    - LeetCode algorithmic problems scope is narrow as leetcode problems are self-contained exercises that capture coding ability rather than the broader aspects of software ecosystem (or) programming language evolution which is much broader with non-algorithmic aspects like software configuration, security, performance, maintenance for example.
    - Similarly, the full-stack benchmark comprising only CRUD-style web-application tasks does not fully represent the breadth of the software ecosystem such as data engineering, distributed systems, embedded systems, or system integrations.

- The discussion in the paper does not mention the existing structural biases in the software ecosystems pre-dating the LLMs. The software industry has favored certain languages/frameworks ecosystems for structural reasons (standardizations, secure implementations, maturity, ecosystem support). Without presenting the pre-LLM baselines for “Matthew Effect”, it is difficult to establish/quantify AI-induced amplification from historical continuation of these biases.

- The paper simplifies language and framework selection by treating popularity and by extension AI compatibility as a dominant determinant for language/framework adoption and suggests - (Lines 20-23) “The phenomenon suggests that AI systems may reinforce existing popularity hierarchies, accelerating convergence around dominant tools while hindering diversity and innovation.” However, technical decision-making in the software ecosystem is not driven by popularity alone. It's a multi-dimensional decision carefully evaluated by senior engineers on the projects based on various factors like language/framework features, runtime performance, maintainability, ecosystem support, and business needs. So this claim need to be proven by considering the aforementioned aspects before reaching to conclusions.

### Questions
Supplementary material seems to be missing in the submission, Where can i find the links to the code base and dataset to replicate this experiments?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper evaluates different LLMs on coding tasks taken from LeetCode.
Bot surprisingly the paper reveals that the more popular a programming language or framework, the higher the success rate of LLM-generated code. Besides LeetCode, the authors also evaluate "vibe coding" revealing that successful code generation is related to a few popular frameworks.

In terms of languages, the authors evaluate as follows: Python, C++, Java, JavaScript, Go, Rust, Erlang, and Racket for five models: GPT-4o-mini, DeepSeek-V3, Gemini-2.0-Flash, Gemini-2.5-Flash, Qwen3-Turbo.

For the framework selection tasks, three AI programming tools are used: Cursor Pro (using Claude-4-Sonnet), CodeBuddy (using Claude-4-Sonnet), and Visual Studio Code with GitHub Copilot (using GPT-5).

### Strengths
With the widespread use of llms in coding tasks it is important to understand their strengths and limitations.

The paper contributes new benchmarks that would be useful for evaluating future models particularly on less popular languages and frameworks.

### Weaknesses
The results are not surprising so it is not clear what this paper brings in terms of novelty.

### Questions
It is not clear to me what metrics were used for evaluating vibe coding (section 5.2). Was the evaluation done manually?

Is there a reason you do not evaluate C? Or is that subsumed by C++?

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
The paper examines the "Matthew Effect" in AI programming assistants. The authors demonstrate that popular programming languages and frameworks receive disproportionately better support from LLMs compared to less popular alternatives. Through large-scale experiments (120,440 code generations across 8 languages and 5 models), the authors demonstrate that language popularity strongly correlates with AI code generation success rates, with performance gaps widening as problem difficulty increases.

### Strengths
To my knowledge, it's the first systematic demonstration of popularity bias in AI coding assistants at both language and framework levels, with clear quantitative characterization.

### Weaknesses
1. The authors tested GPT-4o-mini, DeepSeek-V3, Gemini-2.0-Flash, Gemini-2.5-Flash, and Qwen3-Turbo, but excluded reasoning-specialized models like o4 or DeepSeek-R1 (although Qwen3-Turbo could be reasoning model). Given that most complex coding problems today are solved using these reasoning models, I'm a bit concerned about the generalizability.

2. I wonder about the effects of Chain-of-Thoughts as well. CoT reasoning is utilized in most coding cases - would the presence or absence of chain-of-thought reasoning alter the popularity bias patterns?

3. The paper shows correlational evidence between language popularity and model performance, but the underlying causal mechanism driving this relationship is a bit unclear. For instance, many popular languages tend to be easier, more user-friendly, and simpler to use (e.g., Python). Perhaps it's not language popularity but rather "language difficulty" that determines model performance - models might simply perform better on easier languages.

4. The assumed link between training data representation and performance is not empirically verified. This leads to the concern about the causality of the claim.

### Questions
1. How can you disentangle language difficulty from popularity? Have you considered matched comparisons of "languages with similar complexity" but different popularity? (e.g., Python is not only popular but also easy - so it's hard to know whether the model does Python well because of its popularity or complexity)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
- This paper highlights the Matthew Effect in Coding LLMs, where models do better on programming languages that are predominant in use and therefore predominant in the training set
- The paper studies it in two ways: (1) Leetcode problems and (2) building CRUD applications
- Creating a dataset of 3011 Leetcode problems across 8 languages, the paper tests 5 models and shows that models perform poorly on less-used languages. Similar observations are made for CRUD applications

### Strengths
- Notwithstanding the issues highlighted below, the papers' focus on studying model performance on low-resource languages is important.
- The paper also does substantial experiments to highlight model performances on varying languages

### Weaknesses
- LeetCode does not have a public API. The paper mentions in Line 174 that the GraphQL endpoint is used to retrieve problem information. Can the authors comment on what this endpoint is, and whether it is the official endpoint?
- LeetCode additionally prohibits scraping problems, which raises Terms of Service violation concerns. For this reason, I have flagged the paper for an ethical review.
- To evaluate model-generated solutions, the authors create 15 user accounts on LeetCode and then submit solutions using these accounts with rate-limiting and other mechanisms to avoid detection. This also raises concerns about the misuse of the LeetCode platform
- Due to the above issues, this dataset cannot be used to benchmark future models that aim to fix this issue. This limits reproducibility.
- Section 4.1 mentions the list of post-processing steps taken on model-generated output to strip them of comments and other steps. I am not sure why these are required, and it seemed to me that the post-processing steps were excessive. Comments do not alter the program behavior, so why remove them?
- What is the difference between the three graphs in Figure 2? There is no indication of how they differ
- How are the vibe coding tasks evaluated? Are there automated test cases to evaluate the various functionalities?
- Additionally, why propose a new dataset and not use an existing one with automated evaluation, such as the SWE-Bench Multilingual dataset (https://www.swebench.com/multilingual.html) for this purpose?
- The SWE-Bench Multilingual results do not show the Matthew effect. Resolution rate on C/C++ is the lowest, while Rust is the highest. How would the authors reconcile the findings in this work, and what SWE-Bench Multilingual shows?

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2
