# TALES: Text Adventure Learning Environment Suite

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Reasoning is an essential skill to enable Large Language Models (LLMs) to interact with the world. As tasks become more complex, they demand increasingly sophisticated and diverse reasoning capabilities for sequential decision-making, requiring structured reasoning over the context history to determine the next best action. We introduce TALES, a diverse collection of synthetic and human-written text-adventure games designed to challenge and evaluate diverse reasoning capabilities. We present results over a range of LLMs, open- and closed-weights, performing a qualitative analysis on the top performing models. Despite an impressive showing on synthetic games, even the top LLM-driven agents fail to achieve 20% on games designed for human enjoyment. Visualization of the experiments can be found at https://github.com/tale-suite/tale-suite-anonymized.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces TALES, a unified benchmark spanning five text-adventure frameworks, with minimal scaffolding to probe raw, compositional reasoning. The authors evaluate 42 models in a zero-shot setting and find that while systems perform well on synthetic environments, they struggle markedly on human-written interactive fiction like JERICHO.

### Strengths
1. The definition of the four reasoning skills feels well-grounded and convincing, and the descriptions and setups of the tasks are clear and reasonable.
2. This paper contributes to evaluation stability and reproducibility, which are often overlooked but crucial for benchmarks.
3. The benchmark covers models very comprehensively, giving a broad and useful picture of how current models perform in different types of reasoning.
4. The paper reads smoothly overall

### Weaknesses
1. The benchmark imposes a fixed 100-step cap for all environments with different complexity. A dynamic step setting may be more meaningful. And Table 5 shows extremely low scores of all the LLMs on Jericho. This raises concerns about whether the reported performance reflects actual reasoning limitations or just under-allocated interaction budgets.
2. TEXTWORLD tasks seem trivial for modern LLMs, which all reach near 100%. It's unclear whether this task provides discriminative metric.
3. Lack of justification on the benchmark’s validity through, for example, ablative studies, consistency analyses. Clear empirical evidence of construct validity would strengthen the benchmark.
4. The final average score seems to be computed uniformly across environments. However, Table 8 show variance on different tasks. And the score in different task can be quite high or quite low. The paper does not discuss weighting strategies like whether certain environments dominate the final score distribution.
5. Although the paper mentions “strong evidence of data contamination” in human-written games, it's not clear how the reported results can be interpreted as measurements of reasoning rather than memorization.
6. Since the reward sparsity varies across the tasks. It is unclear whether observed performance differences reflect reasoning or reward shaping artifacts.
6. Reporting results in Table 6 to two decimal places is unnecessary.

### Questions
1. Although it’s necessary to evaluate the LLMs’ raw capabilities, I still wonder what’s their performance under the same rule instruction setting.
2. How does TALES ensure construct validity that the measured scores truly reflect the four proposed reasoning skills rather than other factors?
3. Has the benchmark been validated against human baselines or expert heuristics to confirm the intended difficulty hierarchy?
4. Is the results sensitive to small perturbations like the change in prompt?
5. More analysis on the four proposed skills in the experiments is needed

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
The authors have introduced the Text Adventure Learning Environment Suite (TALES), a collection of text-adventure games designed to rigorously evaluate the reasoning abilities of LLMs. TALES presents a challenge in reasoning for the current state-of-the-art AI, revealing that even the most advanced models struggle with complex reasoning problem.

Key Contributions of the TALES paper:

A Unified Evaluation Framework: TALES provides a standardized benchmark, allowing for consistent and comparable evaluation across different models. The suite includes games from various frameworks like Jericho, TextWorld, TextWorldExpress, ALFWorld,
ScienceWorld, offering a broad spectrum of challenges.

Benchmarking Leading LLMs: The authors tested a wide range of both open- and closed-weight LLMs on the TALES benchmark. 
Analysis of Model Failures: They also conduct a qualitative analysis of the top-performing models identified common failure points.

### Strengths
Both of the strengths and the weaknesses of the paper are quite evident.

1. The text-adventure games provide a good test bed to assess LLM's reasoning ability, which are verifiable (whether succeed in completing the tasks) and challenging enough with long-horizon tasks.
2. The paper provides a unified suite for the existing benchmarks like TextWorld, TextWorldExpress, ALFWorld, ScienceWorld, Jericho, which would make it more convenient for a systematic evaluation in various text adventure based games.
3. The paper includes a large amount of experiments across these games and mainstreaming open-source and close-source LLMs, and provides detailed results and solid setup for evaluation.

### Weaknesses
The provided experiments are robust and the provided unifying of the existing text-adventure benchmarks would be beneficial from a engineering perspective. However, from the novelty view, this paper doesn't include new design of gaming benchmark. This work does not include new games, nor new design of evaluation. 

Meanwhile, leveraging interactive environments to assess various reasoning abilities is not new. And it is not surprising to see LLMs would fail in gaming where complex reasoning is needed. [1][2][3]

[1] Jinhao Duan, Renming Zhang, James Diffenderfer, Bhavya Kailkhura, Lichao Sun, Elias Stengel-Eskin, Mohit Bansal, Tianlong Chen, and Kaidi Xu. 2024b. Gtbench: Uncovering the strategic reasoning capabilities of llms via game-theoretic evaluations. In NeurIPS.

[2] Jen-tse Huang, Eric John Li, Man Ho Lam, Tian Liang, Wenxuan Wang, Youliang Yuan, Wenxiang Jiao, Xing Wang, Zhaopeng Tu, and Michael R Lyu. 2024. How far are we on the decision-making of llms? evaluating llms’ gaming ability in multi-agent environments. arXiv preprint arXiv:2403.11807.

[3] Wenye Lin, Jonathan Roberts, Yunhan Yang, Samuel Albanie, Zongqing Lu, and Kai Han. 2025. GAMEBoT: Transparent Assessment of LLM Reasoning in Games. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7656–7682, Vienna, Austria. Association for Computational Linguistics.

### Questions
Except convenience, what are the other advantages of TALES compared to respectively evaluating LLMs on each of these benchmarks: TextWorld, TextWorldExpress, ALFWorld, ScienceWorld, Jericho and combine the results?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a unified benchmark suite integrating five existing text-adventure learning environments. The proposed benchmark is used to assess the zero-shot capabilities of LLMs. The evaluation measures the maximum score attainable within a fixed number of turns. Additionally, the analysis identifies common failure modes related to spatial, deductive, inductive, and grounded reasoning. The findings indicate that LLM-based agents can struggle even with a simplified "Simon says"-style game—a task considerably simpler than solving complex virtual puzzles. The results also suggest that these agents remain far from achieving optimal performance in games designed for human players.

### Strengths
1. The topic of this work is both compelling and valuable; the integration of diverse reasoning skills into a unified framework offers significant practical utility.

2. The experiment section demonstrates thorough engineering effort, encompassing a wide range of models, and clearly highlights a critical limitation in their long-horizon reasoning capabilities.

3. The design of the SIMON SAYS task offers a natural and well-grounded method for evaluating a model's ability to follow instructions.

4. With multiple concrete examples and detailed explanations provided in the appendix, this paper delivers a benchmark that will greatly benefit the research community and facilitate future studies.

### Weaknesses
1. The four reasoning skills proposed by the authors raise concerns regarding comprehensiveness, and there is a lack of discussion on the interrelationships among these skills. The classification comes across as more of an intuitive listing rather than a systematically constructed framework.
2. The discussion of "TO THINK OR NOT TO THINK" could be more thorough; it would benefit from a comparative analysis between the reasoning modes in this benchmark and those commonly found in pre-training data for LLMs.

### Questions
1. I want to know whether It would be valuable if the authors could provide results and analysis from simple fine-tuning of open-source models on this benchmark, which could further encourage the research community to adopt it for training and optimization purposes.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Note: This is a review by an emergency reviewer.


This paper presents the TALES benchmark for evaluating LLMs on text-adventure game environments that require 4 core reasoning abilities: deductive, inductive, spatial, and grounded reasoning. The authors argue that the compositional reasoning capabilities needed to perform well on TALES are critical for real-world applications of LLM-based agents. The benchmark is created from 5 existing frameworks, each comprised of multiple games, and introduces an initial instruction-following task (Simon Says).  The authors present the scores for 10 strong LLMs on TALES, finding low scores on the games from one framework but very high scores on the others. Further analysis evaluates the reasoning traces of some of the evaluated models, identifies reasoning failures, and compares thinking vs non-thinking models. Finally, prompting strategies are compared for a weak open-source model.

### Strengths
**Concept**: the idea of integrating existing text-based adventure games into a reasoning-focused benchmark as an evaluation of reasoning for real-world applications is interesting.

**Evaluation breadth**: the authors evaluate a wide range of frontier LLMs on TALES with results for additional weaker models also reported in the appendix (42 LLMs in total). This gives a comprehensive overview of current capabilities.

**Evaluation approach**: the decision to use a standardised, lightweight evaluation approach that does not include domain knowledge strengthens the evaluation. 

**Limitations**: the authors acknowledge and discuss several limitations with their approach.

**Visualizations**: the anonymized github link includes numerous visualizations of per-game performance, along with a measure of score spread across different runs.

### Weaknesses
**Difficulty**: for me, a major limitation of this benchmark lies in its difficulty. Specifically, frontier models score very highly on 4 of the 5 frameworks (88-100% for o3 medium) – the games in these frameworks are either approaching saturation or are saturated, and evaluating on them offers limited insights. Therefore, I see the core value lies in the results of the games from the Jericho framework, which do prove challenging for current frontier models (though they are acknowledged by the authors as potentially suffering from data contamination). Given this, the significance of the contribution of TALES seems limited. Down-selecting just the most challenging subset of games from the 4 high-scoring frameworks could increase the overall difficulty, but given the lack of headroom, this would filter out most games.

**Curation**: the description of TALES lacks reasoning for why the 5 frameworks were selected. Statistics on the reasoning types covered by each game would be useful here, as would a clearer overview of the types of games included in each. How do these games relate to real-world applications, and what specific real-world applications do you think attaining strong performance on TALES unlocks?

**Analysis**: the analysis (Sec. 5) would be strengthened by adding example reasoning traces with failures (from the Appendix). This would help contextualise the qualitative insights. 

**Clarity**: several aspects of the paper could be made clearer:

-	The authors compare the results of synthetic games vs games “designed for human enjoyment” but do not specify which games/frameworks correspond to which.

-	The TALES score calculation is not defined in the paper.

-	In Tab. 1, the units for walkthrough length are not stated

-	More examples of the games from the different frameworks (e.g., Fig. 1) would help provide context.


**Minor errors/typos**: the following lists some of the minor errors, inconsistencies and typos in the paper:

-	Line 156: “Figure 1 illustrates a simple task in a text-adventure game where multiple reasoning skills are required at each step…”. Most steps in the figure don’t require multiple reasoning skills

-	Line 40 missing word in “need apply”

-	Lines 49, 401 missing space between text and citation e.g., “task(Paglieri…)”

-	Line 93 missing word in “a collection games”

-	Line 214: “the player receive” -> receives

-	Line 239: remove comma after 54

-	Line 257: missing word: “coming from human expert”

-	Line 288: missing word: “but find”

-	Line 351: missing word “when a subgoal was and”

-	Line 358: missing word “failures still when multiple…”

-	Line 406: missing word: “are same as main results”

-	Line 434: missing word: “reduce the space possible commands”

-	Line 446: missing full stop/period.

-	Table 1 is not referenced in the paper.

-	Line 243: “For example, 9:05 follows the morning of an ordinary office worker where ANCHORHEAD is a Lovecraftian Horror Story” – where should be while?

-	Introduction: “games” and “tasks” are used interchangeably to refer to the suite of 122 games.

-	citep / citet usage is inconsistent

-	TALES is presented as both a collection of frameworks (Line 179) and also described as a framework itself (Line 99).

### Questions
How are the TALES scores calculated? It’s not the mean of the average scores per framework but appears to be a weighted average based on the number of games in each framework.

Have you explored evaluating a human baseline on TALES or the Jericho games in particular?

### Soundness
2

### Presentation
2

### Contribution
1
