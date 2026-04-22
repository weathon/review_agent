# Vibe Checker: Aligning Code Evaluation with Human Preference

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Large Language Models (LLMs) have catalyzed vibe coding, where users leverage LLMs to generate and iteratively refine code through natural language interactions until it passes their *vibe check*. *Vibe check* is tied to real-world human preference and goes beyond functionality: the solution should feel right, read cleanly, preserve intent, and remain correct. However, current code evaluation remains anchored to pass@k and captures only functional correctness, overlooking the non-functional instructions that users routinely apply. In this paper, we hypothesize that instruction following is the missing piece underlying *vibe check* that represents human preference in coding besides functional correctness. To quantify models' code instruction following capabilities with measurable signals, we present **VeriCode**, a taxonomy of 30 verifiable code instructions together with corresponding deterministic verifiers. We use the taxonomy to augment established evaluation suites, resulting in **Vibe Checker**, a testbed to assess both code instruction following and functional correctness. Upon evaluating 31 leading LLMs, we show that even the strongest models struggle to comply with multiple instructions and exhibit clear functional regression. Most importantly, *a composite score of functional correctness and instruction following correlates the best with human preference*, with the latter emerging as the primary differentiator among advanced LLMs on real-world programming tasks. Our work identifies core factors of the *vibe check*, providing a concrete path for benchmarking and developing models that better align with user preferences in coding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ‘vibe checker’, aiming at evaluating the instruction following ability to represent human preferences besides the function correctness. They create a taxonomy of 30 coding instructions and their corresponding deterministic verifiers, which can be used to augment existing evaluation suites. The taxonomy covers five categories, including coding style & conventions, logic & code patterns, documentation & commenting, error handling & exception management, and library & API constraints. The curated instructions are mainly for Python. Through evaluation on 31 leading LLMs, the authors find that non-functional instructions can cause noticeable performance regressions in functional correctness, revealing gaps in models’ instruction-following ability.

### Strengths
- By using executable, deterministic verifiers, the proposed method grounds qualitative “vibe” in reproducible and measurable criteria, improving reliability and comparability of evaluations. 
- The evaluation results show that it’s still challenging for advanced LLMs to follow non-functional instructions while maintaining functional correctness, offering a valuable insight for future model development.

### Weaknesses
- The benchmarks used in this paper, BigCodeBench and LiveCodeBench, are all function-level code generation tasks for single files. However, “vibe coding” instructions often become more practical and complex at the project or multi-file level, where style consistency, documentation practices, and dependency management interact. For project-level instruction following, if the code generation task involves multiple files, curating relevant and non-conflict instructions will also be challenging. 
- While the authors mention “practice grounding” as a design principle, the process remains vaguely described. It lacks evidence and reference on how manual review by the author team ensures real-world relevance. 
- The results could be made more actionable by breaking down which categories or instruction types most strongly correlate with functional regressions, and why. This would help identify specific model weaknesses and guide targeted improvements.

### Questions
During benchmark construction, the paper states that an LLM was used to select relevant and non-conflicting instructions. How was this LLM-based selector evaluated for accuracy and reliability?

### Soundness
3

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
Vibe Checker brings an important question to the table---is evaluating pass@1 (functional correctness) sufficient for coding models? The obvious answer is that it is not. Vibe Checker provides some initial evidence and methodology indicating how to convert benchmarks into a new benchmark that checks for coding style and other points of interest.

### Strengths
The problem is not original, but the quality and clarity of the paper is high. Additionally, this is an important aspect of code generation and I believe the paper would have high impact, especially if it can be extended to other benchmarks.

### Weaknesses
The biggest weakness is that the results seem to naturally conflict with each other / hold no trend.
For example, in single instruction some models (claude) seem to *improve* on functional testing with extra instructions.
Another example is that the regression trend sseems to be much more significant for livevibebench which, according to the paper, is the *less* realistic benchmark problems.
There's also no explanation provided in the paper as to *why* these phenomena are occurring.
The paper could explain why some models are better than others / what the errors look like.

An additional weakness is that there's no statistical significance in any of the results. it's unclear if any of the regressions are meaningful and the coloring for percentages seem fairly arbitrary.

### Questions
In addition to answering the points under weaknesses, how easily can this extend to other benchmarks?

### Soundness
3

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
The paper proposes a new framework for evaluating large language models in code generation. The authors argue that current benchmarks only focus on functional correctness. At the same time, they propose to evaluate both functionality and instruction following. First, they introduce the VERICODE, a set of 30 verifiable code instructions across five categories (coding style, logic, documentation, error handling, and API usage) with corresponding deterministic verifiers. Then, they construct VIBECHECKER from current benchmarks(BigCodeBench and LiveCodeBench) to access functional correctness and instruction following in single-turn and multi-turn code generation. Experiment on 31 LLMs illustrating that non-functional instructions cause functional regression; multi-turn editing improves instruction following but reduces functional accuracy; and a mixture of functional and instruction following metrics best correlates with human preferences.

### Strengths
1. The paper is timely and articulates "vibe check" as a measurable composite of functionality and instruction following.

2. The paper augments existing benchmarks, including BigCodeBench and LiveCodeBench, with verifiable instructions to better simulate real-world interactions.

3. The method is carefully designed and curated in three stages, including sourcing a candidate pool, multi-stage filtering, and review and verification to ensure the reach of developers' expectations.

4. The experiment spanned 31 LLMs from 10 model families in two settings, including single-run generation and multi-turn editing. Providing a comprehensive evaluation across interaction contexts.

### Weaknesses
1. Honestly, I am not fully convinced by the motivation behind this work. I do not believe coding should be a subjective matter. Code is a tool, and the most important aspect is whether it achieves its intended purpose and all of these can be measured by test cases. Talking back to the example in Figure 1, the choice between using a for loop or recursion is not the goal itself. The critical point is that recursion may exceed memory limits in certain cases (so programmers may not want to use it). All of this can be verified through test cases, because it is simple to add assertion conditions in the test code (e.g., memory ≤ limit). I do not believe there is a specific group of people who inherently prefer a particular coding style. If there is, they are not programmer but artists. This is increasingly obvious in "vibe coding" stage, because there is an increasing amount of AI-generated code and less needs for humans to read all the code. The only aspect of "vibe" that I believe truly matters is readability. However, it is already well discussed in software engineering and has its own metrics (e.g., the Flesch-Kincaid score).

2. Line 180 mentioned that the framework is language-agnostic, but the experiment only focuses on Python.

3. Line 207 mentioned the instruction selection uses an LLM-based selector, but no human inspection was discussed.

4. The taxonomy is not generalizable. There can be numerous target vibes in practice.

5. The writing style is not like ML paper (this is more suitable to submit to HCI conferneces).

### Questions
How was the accuracy for instruction selection?

### Soundness
3

### Presentation
3

### Contribution
2
