# BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 4

## Abstract
Evaluating language models fairly is becoming harder as static benchmarks risk contamination by training data, making it unclear whether models are truly reasoning or just recalling answers. We introduce **BeyondBench**, an evaluation framework that avoids this problem by using **algorithmic problem generation**. Unlike traditional benchmarks that risk contamination from internet-scale training data, BeyondBench creates mathematically grounded problems on the fly, ensuring each test remains fresh and uncontaminated. Our framework covers **44 algorithmic tasks** with a total of **117 variations**, grouped into three difficulty levels: the *Easy Suite* (29 tasks) for basic arithmetic and statistics, the *Medium Suite* (5 tasks, 49 variations) for sequence patterns and reasoning, and the *Hard Suite* (10 tasks, 68 variations) tackling NP-complete and constraint satisfaction problems. Each task generates problems from a combinatorial space larger than $10^{15}$ unique instances, with solutions verified deterministically by mathematical proofs. We evaluated **101 language models**, including 85 open-source and 16 closed-source models, spanning sizes from 0.5B to 141B parameters and multiple quantization schemes. All evaluations use three-fold evaluation to ensure statistical robustness. Our results show consistent reasoning deficiencies across model families, with performance degrading sharply as problem complexity increases from polynomial to exponential. In our Hard Suite evaluations, models such as Gemini-2.5-pro, Llama-3.3-70B, and Qwen2.5-72B achieved average accuracies of **56.21%, 27.16%, and 33.37%,** respectively. Moreover, we observe that performance drops drastically without tool usage, with GPT-5, GPT-5-mini, and GPT-5-nano showing a **decline** of **16.81%, 15.86%, and 43.95%** in overall accuracy without tool access. The contamination resistance of BeyondBench rests on three guarantees: (i) the problem space is vastly larger than any static dataset, (ii) every instance has a deterministically verifiable solution (unique or fully enumerated), and (iii) isomorphic transformations generate semantically equivalent but syntactically new problems. BeyondBench redefines reasoning evaluation through genuine algorithmic problem-solving, ensuring fair and meaningful evaluation. Our public leaderboard is available at https://ctrl-gaurav.github.io/BeyondBench/. Our open-source Python package is available at https://pypi.org/project/beyondbench/, and the codebase can be found at https://github.com/ctrl-gaurav/BeyondBench for easy and reproducible evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces BeyondBench, a novel evaluation framework designed to address the critical issue of data contamination in LLM assessment.

The core problem BeyondBench tackles is that traditional, static benchmarks are often found on the internet and inadvertently included in LLM training data. This leads to models achieving high scores through simple recall rather than demonstrating genuine reasoning capabilities. It becomes unclear whether a model is solving a problem or merely retrieving a memorized answer. BeyondBench solves this by implementing a benchmark-free evaluation paradigm based on algorithmic problem generation. Instead of relying on a fixed set of questions, the framework dynamically and perpetually generates new, unique, mathematically grounded test problems on the fly. This "freshness" ensures that no test instance has been seen during the model’s training, thereby eliminating the risk of data contamination.

The primary contribution of this work is the development of a contamination-resistant and reliable method for testing the core reasoning skills of LLMs. By covering a variety of difficulty levels and task types, BeyondBench provides a dynamic, rigorous, and fairer mechanism to evaluate and compare the true intelligence and problem-solving capacity of different language models, moving the field of LLM assessment beyond the limitations of static benchmarks.

### Strengths
1. The paper tackles the critical and growing issue of data contamination in LLM evaluation. It provides a robust solution to move past static benchmarks。 This shift is vital for accurately assessing the true capabilities and progress of frontier language models.

2.  Excellent Execution and Contamination-Free Design. By employing algorithmic problem generation, it ensures that every test instance is unique, mathematically grounded, and perpetually fresh, thereby guaranteeing a contamination-free evaluation. This design covers a wide range of difficulty levels and task types, making it a rigorous and challenging test of core reasoning skills.

3. The methodology is presented clearly and logically, effectively demonstrating how the new framework can accurately distinguish between different models' true problem-solving capabilities.

### Weaknesses
1.  Reduced User-Friendliness and Increased Score Variance: Compared to using static datasets, dynamically generated evaluations are inherently less user-friendly for the general public and researchers. They require a bit more complex infrastructure for on-the-fly problem creation and may lead to higher score variance between evaluations due to the non-static nature of the test instances, making direct comparisons of models more challenging than with a fixed set.

2.  Missing Discussion of Prior Work: The paper's core idea—using dynamically generated environments/problems for evaluation—is not entirely novel, especially in related fields like RL and other LLM evaluation efforts (e.g., Reasoning Gym [1]).

References

1. Stojanovski, Zafir, et al. "REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards." arXiv preprint arXiv:2505.24760 (2025).

### Questions
1. Will programmatically generated problems bias towards LLMs with coding tools capabilities? Since the problems are generated via code, is there a risk that LLMs with strong coding skills will have an unfair advantage in solving these problems compared to those without such capabilities?

### Soundness
4

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
4

### Summary
The authors introduce a new dataset namely **BeyondBench** which benchmarks LLMs performance on algorithimic tasks and evaluate the performance of both open open source and proprietary on their dataset

### Strengths
The paper is easy to read and authors have provided sound mathematical justification for their arguements in the paper. The appendix is detailed and is very nicely written to accompany the main text.

### Weaknesses
1. The fundamental weakness of the paper is with respect to novelty, there are already exact same work done in this direction and the authors unfortunately fail to cite them as well as evaluate them against their benchmark which would be a fairer comparision. [1], [2], [3]

2. Problems in hard suite like crypt arithemtic are known NP complete problems, so the benchmarking of LLMs/LRMs directly only with prompting is in itself an unreasonable task. is it even a fair experiment to expect LLMs which are atmost polynomial time methods to solve problems of NP complete complexity class?

3. A more reasonable approach would be to evaluate methods like PAL [4] or logic lm [5], but these works are neither cited nor are these methods evaluated. There is a section which talks about augmenting LLMs with tools and with some numbers but there is no detail as to what these "tools" were and as to what was the LLMs asked to generate for a given instance using these "tools".

[1] NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes

[2] FCoReBench: Can Large Language Models Solve Challenging First-Order Combinatorial Reasoning Problems?

[3] PUZZLEPLEX: Benchmarking Foundation Models on Reasoning and Planning with Puzzles

[4] PAL: Program-aided Language Models

[5] Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning

### Questions
Please address the weakness section

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
3

### Summary
BeyondBench is a dynamically generated benchmark for evaluating reasoning in language models. It addresses data contamination by algorithmically generating problems from a combinatorial space of >10^15 instances across 44 tasks in three difficulty tiers. Each problem has verifiable solutions using SAT/CSP solvers. The authors evaluate 101 models and find that most open-source models plateau around 30-35% accuracy on hard tasks, quantization has minimal impact, and tool-augmented models substantially outperform reasoning-focused models.

### Strengths
S1. Data contamination in static benchmarks is well-documented, and this paper provides a solution with mathematical guarantees about solution uniqueness and problem space size.

S2. The evaluation covers 101 models across different sizes (0.5B-141B parameters), architectures, and quantization schemes, providing broad coverage of the current model landscape.

S3. The use of SAT/CSP solvers to verify solution uniqueness and enumerate all valid solutions when multiple exist shows proper attention to evaluation correctness.

S4. The framework dynamically scales problem difficulty based on model context windows and checks for token overflow, preventing unfair comparisons and catching models that "overthink" simple problems.

S5. The empirical findings about performance plateaus on hard tasks, minimal quantization impact, and superiority of tool-augmented approaches over pure reasoning models provide actionable insights for model development.

S6. The contamination probability analysis in Appendix E gives formal backing to the motivation.

### Weaknesses
W1. The 44 tasks focus almost entirely on algorithmic puzzles, arithmetic, and constraint satisfaction, missing many reasoning types relevant to LLM applications: causal reasoning, analogical reasoning, argument evaluation, multi-step logical inference in natural contexts. The paper doesn't justify why performance on Sudoku or Tower of Hanoi reflects general reasoning capability. Their choice to view reasoning to be interchangeable with algorithmic computation needs to be better motivated. (To the authors' credit, this is touched on in their limitations section but should be highlighted)

W2. While specific problem instances are novel, models could still learn superficial algorithmic strategies from similar problems in training data. The paper doesn't distinguish between "memorizing specific solutions" and "learning heuristic algorithms", and their method seems to only expose the former. If allowing memorization of fixed, superficial procedures (e.g. a problem which requires addition, followed by multiplication, followed by subtraction in that exact order) is not considered as part of contamination, this should be made clear.

W3. No confidence intervals, significance tests, or reporting of variance across different problem instances appears in the paper. Claims about performance differences lack statistical backing. The results are interesting, so it would be best to give readers more confidence in the findings.

W4. Problem generation algorithms, solution verification procedures, and task specifications are relegated to appendices, making it hard to assess the approach without reading 20+ pages of supplementary material. Although the methods appear sound, readers will benefit from more explicit presentation of key technical methodological details (especially section 3.1, where there is not enough explanation on how problems are generated and along what dimensions the problems are being varied. The 'parameter space' is presented too abstractly.).

W5. Calling this 'Benchmark-Free' seems confusing. Readers might expect some kind of mechanistic interpretability/probing method, or some metric derived from the model which does not require inference on a dataset of test problems.

### Questions
In addition to questions implied by the weaknesses section, here are some additional questions

Q1. Could the authors provide information about if there were any differences in failure modes between vanilla models and reasoning models, even if they performed similarly in raw metrics? This is one of the most interesting findings of this paper but it is not emphasized and the analysis on it seems thin.

Q2. Criteria for graceful vs sudden degradation needs more care, for example using number of disks for Tower of Hanoi may not make sense as number of steps are not linear in number of disks. The degradation may be linear (and hence 'graceful') w.r.t reasoning steps, why should we expect linear degradation on arbitrary complexity metrics? Could the authors provide arguments for why 'n-disks', 'grid size', 'vertices', 'board size', 'word length' etc. are comparable complexity metrics? Otherwise the conclusions about differences in gradual vs sudden degradation across tasks lack any significance.

Q3. Could the authors provide more justification that avoiding parameter space collisions with high probability is enough to be confident that structural/semantic space collisions are avoided? This is related to the earlier point about superficial heuristics (W2), does this approach prevent collisions akin to, say, IMO questions requiring the same integration trick multiple years in a row?

### Soundness
3

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
3

### Summary
This paper introduces a framework for evaluating the reasoning capabilities of LLMs. It addresses the issue of data contamination in existing static benchmarks by generating mathematically grounded problems algorithmically and on-the-fly, ensuring each test instance is fresh and uncontaminated. BeyondBench comprises 44 algorithmic tasks with 117 variations across three difficulty suites (Easy, Medium, Hard) and offers provable unique solutions verified by mathematical proofs. It also incorporates a token-aware evaluation system that dynamically scales problem complexity based on a model's context window.

### Strengths
1. Contamination Resistance: BeyondBench uses algorithmic problem generation to create mathematically grounded problems on the fly, ensuring each test is fresh and uncontaminated. 
2. Task and difficulty: This work covers 44 algorithmic tasks with 117 variations, grouped into three difficulty levels. The complexity of each task can be controlled by scaling parameters.
3. Token-Aware Evaluation Framework: The framework dynamically calibrates problem complexity based on a model's token budget.

### Weaknesses
1. The benchmark questions do not sufficiently assess the level of intelligence. Existing competition benchmarks often cannot be solved by fixed algorithms and instead require special techniques or constructions. However, the problems used in this benchmark are almost all based on fixed algorithms, with difficulty controlled by adjusting complexity and computational difficulty. This approach fails to evaluate the "cleverness" or "ingenuity" of a model in solving problems.
2. The benchmark seems to be easily hacked by tool-augmented models. Fixed algorithms and complex computations can be easily solved by calculators or code. The performance gap of GPT-5 family with/without tool shows the evidence.
3. This work only theoretically argues that the combinatorial variations of this benchmark ensures the test is uncontaminated, but it lacks experimental validation. If related data is constructed for SFT or RL targeting this benchmark, the influence is not studied.

### Questions
1. If data is constructed for SFT or RL targeting this benchmark, how will the evaluation results change? In cases of deliberate hacking, does BeyondBench have any advantages compared to static evaluation and other dynamic evaluation methods?
2. A more in-depth analysis should be conducted on the phenomenon where tool-augmented models achieve extremely high results. Is it possible that certain tasks can be entirely solved by tools, or is the improvement in computation more significant?

### Soundness
2

### Presentation
2

### Contribution
3
