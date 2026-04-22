# C$^3$-Bench: Evaluating and Achieving Controllable Code Completion in Code LLM

- Avg Score: 4.40
- Decision: Reject
- Scores: 4, 2, 6, 4, 6

## Abstract
Code completion has become a central task, gaining significant attention with the rise of large language model (LLM)-based tools in software engineering. Although recent advances have greatly improved LLMs' code completion abilities, evaluation methods have not advanced equally. Most current benchmarks focus solely on functional correctness of code completions based on given context, overlooking models' ability to follow user instructions during completion\textemdash a common scenario in LLM-assisted programming. To address this limitation, we present the first instruction-guided code completion benchmark, \textbf{\underline{C}}ontrollable \textbf{\underline{C}}ode \textbf{\underline{C}}ompletion Benchmark (C$^3$-Bench), comprising 2,195 carefully designed completion tasks. Through comprehensive evaluation of over 40 mainstream LLMs across C$^3$-Bench and conventional benchmarks, we reveal substantial gaps in instruction-following capabilities between open-source and advanced proprietary models during code completion tasks. Moreover, we develop a straightforward data synthesis pipeline that leverages Qwen2.5-Coder to generate high-quality instruction-completion pairs for supervised fine-tuning (SFT). The resulting model, Qwen2.5-Coder-C$^3$, achieves state-of-the-art performance on C$^3$-Bench. Our findings provide valuable insights for enhancing LLMs' code completion and instruction-following capabilities, establishing new directions for future research in code LLMs. To facilitate reproducibility and foster further research in code LLMs, we open-source all code, datasets, and models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces C$^{3}$-Bench, a benchmark designed to evaluate the instruction-following capabilities of Large Language Models in code completion tasks. C$^{3}$-Bench encompasses a diverse range of code completion scenarios and instructions. The evaluation results reveal a substantial gap in instruction-following abilities between open-source and advanced proprietary models. Furthermore, the authors propose a data synthesis method to generate data aimed at enhancing model performance on C$^{3}$-Bench.

### Strengths
1. The paper is well-motivated. It aims to evaluate the instruction-following capabilities of LLMs in code completion tasks, which is a valuable and timely research topic.

2. The paper is well-structured and comprehensive. The overall workflow, which includes problem definition, benchmark construction, and a proposed solution, is complete and presented with clarity.

3. The consistency observed between the results from C$^{3}$-Bench and exsiting benchmarks like Copilot Arena is impressive.

### Weaknesses
1. The paper's core contribution lies in the evaluation of instruction-following capabilities. However, a primary concern is that the evaluation uses a single model (Claude 3.5). While the authors have validated Claude's judgments against human annotations for consistency, the use of an LLM-as-a-judge for such a subjective task can inevitably introduce bias [1], particularly for those ambiguous model-generated responses. Is it possible to incorporate more controllable, quantitative metrics for the evaluation? Alternatively, at a minimum, could the authors provide results using other LLMs as judges to demonstrate the consistency and robustness of the evaluation results?

2. Following up on the previous point, the performance of the fine-tuned Qwen2.5-Coder-C$^{3}$ on the ICC task is understandably constrained by the capabilities of its base model, making the modest improvement in pass@1 seem reasonable. However, the remarkably significant increase in the IF score raises a critical question: Does this suggest that the IF score can be easily somehow hacked? Specifically, could using the same model that generated the benchmark data as the judge introduce a systemic bias, potentially rewarding models that mimic the judge's stylistic preferences?

3. The scope of C$^{3}$-Bench is currently limited to in-file tasks in Python. The impact of this work could be significantly enhanced by extending the benchmark to support repository-level contexts, and to encompass multiple programming languages.

[1] Wei H, He S, Xia T, et al. Systematic evaluation of llm-as-a-judge in llm alignment tasks: Explainable metrics and diverse prompt templates[J]. arXiv preprint arXiv:2408.13006, 2024.

### Questions
Please refer to weaknesses.

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
5

### Summary
The premise of this paper is that well known benchmarks for programming tasks
do not adequately test the "instruction following" capabilities of LLMs, where
"instruction following" in this paper means testing non-functional requirements
in the prompt. Non-functional requirements are anything that cannot be
tested using simple test cases of the form "assert pred(f(x))" where f is the
synthesized program, x is a test input, and pred some predicate of the output.

The paper presents a synthetic dataset of programming tasks, derived from
HumanEval and SAFIM, that test models' non-functional requirements.
In addition, the paper uses the same benchmark generation pipeline to generate
a SFT dataset. The paper decontaminates the SFT dataset and uses it to fine-tune
a Qwen Coder model, which performs very well on the benchmark.

### Strengths
Important problem with a large evaluation.

### Weaknesses
At this point, there are probably hundreds of benchmarks that evaluate the
coding abilities of LLMs in different ways. Although the exact prompt format
that this paper uses may be new (though just a prompt format is not a
contribution), there are plenty of other papers that test the ability of models
to complete code in context with instructions that have non-functional requirements.
From my memory, here are a few:

- NoFunEval: Funny How Code LMs Falter on Requirements Beyond Functional Correctness
   https://openreview.net/forum?id=h5umhm6mzj

  NoFunEval is HumanEval-derived, similar to some of the benchmarks in
  the current submission.

-  Can Large Language Models Write Parallel Code?
   https://dl.acm.org/doi/10.1145/3625549.3658689

   ParEval goes beyond functional correctness, which matters for parallel code
   I recall it has some other metrics that go beyond pass@1. There is a lot
   of follow up work from the ParEval authors.

- D3: https://openreview.net/forum?id=Ksq7fgagId

  This is a dataset, perhaps not a benchmark. But, the contribution overlaps.

Overall, I think the contribution of this paper is very small. This is particularly
the case since the benchmark is partially synthetic, with portions of prompts
generated by Claude 3.5 Sonnet.

Some further notes on the writing:

- I wish the main body of the paper gave the reader some examples of the
  kinds of problems in the benchmark. I think the only example is in the appendix
  (Figure 7). Moreover, I don't find that example very compelling. 

- L160: "Multi-line completion mandates the generation of a predetermined number of
  complete code lines" It seems peculiar to require a solution in X lines of
  code.

- L202: "The original HumanEval [..] datasets primarily contain single-line implementations
  of ground truth middle code." Is this really true of HumanEval? Just
  scanning the canonical solution column here:

  https://huggingface.co/datasets/openai/openai_humaneval

  There are a number of solutions that begin with "return ...", but most of
  them begin with loops, variable definitions, if statements, etc.

Finally, the prompt format that the paper uses seems peculiar. It is possible
that scores will go up significantly by picking a format that is more in
distribution. E.g., just having a single code block with "### INSERT CODE HERE ###"
to fill in.

### Questions
See weaknesses.

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
This paper proposes C3-Bench, a new benchmark for evaluating large language models on controllable code completion, focusing on their ability to follow detailed implementation instructions—an aspect overlooked by existing benchmarks like HumanEval and CrossCodeEval.
C3-Bench defines two tasks: Implementation-Control Completion (ICC) and Scale-Control Completion (SCC), covering functional, structural, and size constraints. The benchmark contains 2,195 Python tasks, measuring both functional correctness and instruction adherence.
The authors also introduce a data synthesis pipeline using Qwen2.5-Coder, and fine-tune it to create Qwen2.5-Coder-C3, achieving state-of-the-art results on C3-Bench. Experiments on over 40 LLMs (open and commercial) show that current benchmarks overestimate model capability, and C3-Bench correlates better with real-world developer evaluations.

### Strengths
The paper identifies instruction-following as a crucial yet underexplored aspect of real-world code completion. 

The division into ICC and SCC tasks provides comprehensive coverage of different types of control in code generation, going beyond traditional correctness-based metrics.

The authors detail a hybrid process (AST extraction, variant generation, instruction synthesis, filtering) that ensures data diversity and precision.

Evaluation across 40+ models and multiple benchmarks provides a rich empirical analysis, showing that C3-Bench aligns well with practical code-assistance performance.

### Weaknesses
The benchmark focuses solely on Python and single-file completions, which underrepresents real-world multi-language, multi-file development complexity.

Qwen2.5-Coder-C3’s gain mainly comes from instruction-tuning rather than architectural advances; ICC performance still saturates at base-model limits.

Even with consistency checks, using an LLM as the primary instruction-following evaluator may introduce subtle bias or inconsistency compared to human judgment.

### Questions
Do the authors plan to extend C3-Bench to multi-language or multi-file tasks (e.g., C++, JavaScript)?

How is prompt sensitivity of the LLM judge controlled?
Have the authors evaluated inter-judge consistency (e.g., Claude vs. GPT-based judges)?

Are there notable differences between ICC and SCC results across models (e.g., algorithmic control vs. scale control)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel benchmark to evaluate the instruction-following capabilities of LLMs in code completion tasks. Specifically, the benchmark adds specific requirements to the prompts of code competition tasks, and evaluates how well LLMs follow these instructions using LLM-as-a-Judge and rule-based methods, beyond functional correctness tests. Furthermore, this paper proposes a straightforward data synthesis pipeline that leverages Qwen2.5-Coder to generate high-quality instruction-completion pairs for fine-tuning.

### Strengths
- This paper explores an important and interesting research direction.
- The paper is well-written and easy to follow.

### Weaknesses
- Extracting the middle code by selecting nodes from the AST tree is not novel. SAFIM has explored syntax-aware completion within code’s AST including algorithmic blocks that targets at multi-line completion and serve as the major part of SAFIM, which however fails to get acknowledged in Section 2.3.1.
- The benchmark focuses on function-level code completion, which however is different from real-world scenarios where code completion always happens in the large repositories with cross-file context. It’s also important to study whether the instruction-following capabilities of LLMs will change in repository-level code completion tasks.

### Questions
None beyond the above.

### Soundness
3

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
3

### Summary
The authors introduce $C^3$, an instruction-guide code completion dataset, that is designed to address a fundamental limitation of existing  code completion benchmarks which only look for functional correctness. The paper also discusses results across 40 LLMs on this benchmark exposing gaps in instruction following capabilities of closed vs open models. The authors curated high quality SFT data which is used to train Qwen2.5-Coder-$C^3$ model, resulting in SoTA results on $C^3$ dataset. Interestingly, it is also shown that the performance on this dataset, correlates with results from Copilot arena, demonstrating its practical significance.

### Strengths
- LLM assisted coding in real world applications need to adhere to user instructions. However, most of the publicly available datasets just account for functional correctness and do not measure the instruction following (IF) ability. This paper introduced $C^3$ dataset, which is a novel and important contribution to the field. 
- The authors conducted extensive studies across different models, sizes, exposing gaps in the IF abilities of these models, thereby providing guidance to the open source community to work on improving this capability.
- Through SFT data curation and training  Qwen2.5-Coder-$C^3$, they achieved SoTA on $C^3$ bench, demonstrating the importance of high quality and relevant dataset to improve IF capability.

### Weaknesses
- For Semantic Validation for ICC, the authors used LLM-based judging system with Claude3.5-Sonnet. They mentioned that this has 98% agreement with senior Python developers across 10 independent assessment rounds. However, the detail on whether this has been done across all the examples in ICC or a subset of them is unspecified. 
- As acknowledged by the authors, the dataset only comprises of in-file python tasks, limiting the scope of usage for this dataset.

### Questions
- Can you share more details on how the agreement of 98% between LLM judge and human developers is achieved?
- Why is this paper (https://www.arxiv.org/pdf/2507.22462) not referenced anywhere?

### Soundness
3

### Presentation
3

### Contribution
3
