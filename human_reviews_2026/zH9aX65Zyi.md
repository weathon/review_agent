# Benchmarking Correctness and Security in Multi-Turn Code Generation

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
AI coding assistants powered by large language models (LLMs) have transformed software development, significantly boosting productivity. While existing benchmarks evaluate the correctness and security of LLM-generated code, they are typically limited to single-turn tasks that do not reflect the iterative nature of real-world development. We introduce MT-Sec, the first benchmark to systematically evaluate both correctness and security in multi-turn coding scenarios.  We construct this using a synthetic data pipeline that transforms existing single-turn tasks into semantically aligned multi-turn interaction sequences, allowing reuse of original test suites while modeling the complexity of real-world coding processes. We evaluate 32 open- and closed-source models, and 3 agent-scaffolding on MT-Sec and observe a consistent 20-27% drop in “correct & secure" outputs from single-turn to multi-turn settings--even among state-of-the-art models. Beyond full-program generation, we also evaluate models on multi-turn code-diff generation--an unexplored yet practically relevant setting--and find that models perform worse here, with increased rates of functionally incorrect and insecure outputs. Finally, we find that while agent scaffoldings boost single-turn code generation performance, they are not quite as effective in multi-turn evaluations. Together, these findings highlight the need for benchmarks that jointly evaluate correctness and security in multi-turn, real-world coding workflows.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces MT-Sec, the first benchmark that systematically evaluates both functional correctness and security in multi-turn coding scenarios. It employs a synthetic data pipeline that transforms existing single-turn tasks into semantically aligned multi-turn interaction sequences, and comprehensively evaluates 32 open- and closed-source models as well as three agent-scaffolding frameworks on MT-Sec.
In addition, this work is the first to assess models’ ability to generate correct and secure code diffs, providing new insights into incremental code-generation safety.

### Strengths
1. The benchmark’s three synthetic formats—Expansion, Editing, and Refactoring—are practical and realistic, capturing common multi-turn coding workflows.
2. The inclusion of a manual verification step to ensure the validity of multi-turn requests makes the test case generation process reliable and credible.
2. The benchmark supports multiple programming languages and covers a wide range of CWEs, enhancing its comprehensiveness and generality.
3. The experiments are extensive, evaluating a large number of LLMs, which makes the empirical analysis thorough and convincing.

### Weaknesses
The comprehensive and large-scale experiments are appreciated. However, since the overall pipeline design is relatively straightforward and both the seed code and test suites are derived from existing benchmarks (e.g., SecCodePLT and BaxBench), the technical and methodological novelty appears somewhat limited.

Given that the core contribution lies in transforming single-turn tasks into multi-turn code generation, its scope appears limited to only three interaction types—Expansion, Editing, and Refactoring. Exploring a broader range of interaction patterns could further strengthen the benchmark’s coverage and realism.

### Questions
1. Do the authors plan to extend MT-Sec beyond the three existing formats (Expansion, Editing, Refactoring) to include more realistic multi-turn coding scenarios, such as debugging cycles, version recovery/recall, or other scenarios? 

2. Why do some results in Table 1 show large discrepancies among the three categories (MT-Expansion, MT-Editing, and MT-Refactor)? Could the authors explain the distinct challenges or characteristics that lead to such performance gaps across these task types?
Additionally, would it be appropriate to incorporate a difficulty hierarchy across tasks to better reflect the varying complexity levels of different multi-turn interactions?

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
5

### Summary
This paper proposes a secure coding benchmark, which quantifies how large language models can produce correct and secure code in both single-turn and multi-turn settings. This paper designs three common multi-turn interaction ways between users and AI. Experimental results show that LLMs can not handle secure coding very well.

### Strengths
This paper selects an important task, secure coding, to evaluate the performance of modern LLMs, since nowadays AI coding agents have been widely used. 

The setting of this benchmark is rational, for example, the choice of three multi-turn query generation ways.

### Weaknesses
There seems to be a difference in the potential risks of insecure code generation across different large models, but it's not analyzed in detail. Although the probability of generating insecure code might be similar, which model is more brittle hasn't been discussed.

Additionally, there is no explanation for why multi-turn interactions result in a higher rate of insecure code generation. This is a crucial issue, as without understanding this, the community won't know how to improve.

Also, I have concerns about how well this benchmark aligns with real-world user scenarios. For example, how close are the multi-turn queries in the benchmark to actual use cases? Do real users often mix interactions, such as expansion and editing?

### Questions
See the limitations.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The author introduce MT-Sec a multi-turn security benchmark derived from SecCodePLT and BaxBench (hence, not python only). An non-LLM-based first-prompt-selector is introduced which filters out instances that do not contain dynamic security checks. Then an LLM is introduced to create 3-turn conversations from each seed prompt. Different types of multi-turn conversations are generated 1) refactoring, 2) editing, and 3) expansion dialogs. The authors then manually review and verify the dialogs and edit them when necessary. On the experimental side, the authors measure the fraction of correct and secure (C&S) and correct and insecure (C&I) solutions. The authors show that SOTA scaffold + LLMs' performance drop significantly compared to the single turn setting. In addition, it is shown that some thinking models tend to over-refuse and that placement of hints regarding the desired security policy may depend on the base LLM. Lastly, the authors perform a random permutation of the first 2 turns to rule out that context length is the reason for performance degradation by showing that this performance is roughly on par with the single turn set up.

### Strengths
* Multi-language benchmark (rather than python-only)
* Evaluation of a solid amount of base models (Claude, GPT, Qwen, Deepseek) and scaffolds (OpenHands, Codex, Aider)
* Interesting observations around over-refusal
* MT-random is a great ablation to illustrate the challenge with the data set
* Interesting insight that multi-turn interactions may trigger overly cautious refusals
* Thorough experimental comparison and good visualizations which make the work easy to grasp
* Usage of human verification for quality control

### Weaknesses
* The split into expansion, editing, and refactoring seems largely arbitrary and potentially overlapping. From the author's description it sounds like the editing and expansion tasks are not that different. Also, it seems that an editing task could fall under the refactoring category.
* The multi-turn nature of this benchmark may be confusing by construction for LLMs as it is not adaptive. A true multi-turn setting would consider the initial response from the LLM and in adjust the second turn question accordingly. Here, the flow of questions is determined through the data set and question 2 may be totally unrelated to the LLM's solution to question 1 (maybe it asks a follow up question?). I think this may be a reason why performance is not that great.
* Solvability: I interpret the results around decreased performance overall and in MR-random differently than the authors. I fear the non-adaptability of the dataset hurts performance as the LLM may receive an instruction that is not in line with its initial response. You report that some of the LLMs generate C&S solutions after the first then but then get confused. I think these instances should be considered a success as in the real-world I would not follow up if I am happy with the first solution. Can you recompute success rates accordingly?
* Performance metrics are point estimates without error bars which makes it impossible to understand whether differences in performance are statistically significant.

### Questions
* Can you cleanly lay out whether the distinction between expansion, editing, and refactoring results in non overlapping sets? Could a turn in editing not contain request related to refactoring? And isn't expansion a form of editing? Where exactly are the differences and can you prove that there is no overlap?
* Tables 2 & 3: Which scaffold is used?
* Can you compute CIs or error bars to assess whether performance differences between ST and MT-* are statistically significantly different?
* Could you add a "Scaffold" column in Table 1 to more allow for better representation of individual scaffolds? What is your interpretation around scaffolds, which perform better/worse and why?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper enriches two existing benchmarks (SecCodePLT and BaxBench) by converting them into multi-turn versions. This is done by 3 strategies: *Expanding* a first - more limited - request, *Editing* previous versions and *Refacotoring* (a way of Editing that focuses on stylistic changes). This is achieved with a model-in-the-loop procedure (GPT4o), using intermediate checks ("Consistency Guardrails") to guide a correct rollout. All datapoints are human-verified, approving >90 of them and rewriting the remaining ones

### Strengths
This is a large dataset as it goes for test (2376 datapoints), covering 6 programming languages (and not only python as is often common unfortunately)

Focusing on multi-turn is a missing area in code. Multi-turn has special challenges, due to requiring long context retrieval capability, the possibility of adapting instructions and in general a better resilience to error accumulation

The paper include some good analysis, including trying to disentangle multi-turn from long input. The authors report additional experiment using diffs instead of full file generation

### Weaknesses
The main weakness is that the evaluation set is machine generated instead of fully human, potentially including data artefacts or incorrect hints. It seems however that the authors put enough checks in place to control that

nitpick: first time the acronym CWE is mentioned (ln 103) it is not defined

### Questions
Could you report on the number of difficult (those not solved by any model) and easy datapoints?

Did you perform some diversity analysis (length, CWEs types, etc)?

### Soundness
3

### Presentation
4

### Contribution
3
