# lmgame-Bench: How Good are LLMs at Playing Games?

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 2, 8, 6, 4, 6

## Abstract
Playing video games requires perception, reasoning, memory, and long-horizon planning—exactly the faculties expected of modern large language and vision–language models (LLMs/VLMs). We introduce LMGame-Bench, a benchmark built on six popular games spanning platformer, puzzle, and narrative games through a unified Gym‑style API. Unlike prior game benchmarks that entangle multiple skills, LMGame-Bench employs a modular harness—including perception, memory, and reasoning modules—that can be toggled to selectively probe distinct capabilities. The benchmark further improves robustness through prompt standardization and contamination mitigation. Evaluation of 13 state-of-the-art models demonstrates that LMGame-Bench remains challenging yet effectively discriminates among models. Correlation analysis reveals that individual games align with core LLM capabilities, providing a quantitative framework for interpreting performance. Finally, LMGame-Bench exposes models’ limitations in visual state extraction, reflection, spatiotemporal reasoning, and long-context reasoning, pointing to concrete directions for model improvement.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a benchmark for LLMs on six game environments, they evaluated a variety of popular LLMs. The paper also studies contamination and attempts to create standardize prompting.

### Strengths
Having the human average for each of the games and random baselines is good. The study on contamination detection is interesting, and so is the attempt at standardizing the prompting with  DSPy standardization

### Weaknesses
I would argue that testing LLM without harness doesn’t make much sense at all. In the context of LLM Agents, the LLM is the brain (without memory module), the agentic scaffolding, or so called “harness” in the paper, is equivalent to affordances and tools such as arms, legs and eyes, memory. It’s not surprising to me at all how a naked LLM without these affordances (scaffolding or harness) often doesn’t outperform random. Memory particularly is very important for partially observable MDPs. 

The benchmark could have been valuable around a year prior to submission, but now comes in a much more crowded space of benchmarks for LLM/VLM Agents, and the the papers struggles to find novelty both in the methodology as well as in insights provided.  The variances in most of the results are extremely large, and it’s difficult to disentangle real results from noise. The paper attempts to do something very similar to existing benchmarks specifically Balrog [Paglieri et al. ICLR 2025], and given the many similarities a more thorough comparison would be needed to explain what novelty (whether methodological or new insights) this paper brings. 

Using o3 to generate perception traces also seems like a big methodological mistake as the rest of the models' capabilities will be influenced by another model.

### Questions
Could the author better argue what’s the difference between their “harness” compared to typical scaffoldings used in most related benchmarks?
Could the author try to better argue where the novelty of the benchmark comes from, and what insights are actually new, especially when compared to the many existing benchmarks in the area?
Why using o3 for perception? This feels like a methodological flaw.
How many seeds were tested for each game?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces LMGame-Bench, a new benchmark designed to evaluate large language
and vision–language models (LLMs/VLMs) through six popular video games: Super Mario Bros.,
Tetris, Sokoban, 2048, Candy Crush, and Ace Attorney. The benchmark uses additional modular
gaming harness composed of Perception, Memory and Reasoning modules, which are used to
assess specific model capabilities, and allow the models to perform at the level of human
players. The benchmark also takes advantage of the DSPy’s SIMBA optimizer to standardize
the prompts.
LMGame-Bench allows for a controlled evaluation of LLMs on video games and offers a
comprehensive evaluation of 13 state-of-the-art models, while relying on robust techniques such
as prompt standardization and contamination detection.

### Strengths
1. Gaming Harness: The design of the gaming harness is highly effective and it represents a
robust method for isolating and accurately evaluating specific model capabilities and skills.
2. Data Contamination Detection: The inclusion of an explicit mechanism for detecting data
contamination is a significant strength
3. Prompts Standardization: The decision to standardize prompts by leveraging the DSPy
framework is highly commendable. This approach ensures consistency and reproducibility
across experiments.

### Weaknesses
The primary weakness lies in the overall presentation and clarity of the paper. I recommend
repositioning the Related Work section to immediately follow the Introduction. This structural
change would more effectively contextualize the work and highlight the paper's novel
contributions earlier. Also, given that this is a benchmark paper, the explanation of the Metrics
(specifically the Raw and Aggregated Scores) should be expanded, possibly reserving one
subsection just for this point.

Also, there are some typos in the main paper. For example line 53:
"To address this issue and also enable controlled evaluation, we enriches our evaluation
settings by developing gaming harness". "we enriches" should be "we enrich".

And line 425:
"Super Mario Bros. is excluded" should be "Super Mario Bros is excluded".

### Questions
Regarding the gaming harness. It is not clear to me if you can "toggle on or off" them selectively.
For example, is it possible to only activate the Perception Module without activating the Memory
Module, or viceversa?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new benchmark for evaluating LLMs on game-playing tasks, testing 13 state-of-the-art models using a direct screenshot-to-action setup. The authors find that these models perform poorly, often close to random, highlighting their weaknesses in visual perception and long-horizon decision-making. The study also explores where and how these models fail as task difficulty increases and demonstrates the effective use of the proposed harness.

### Strengths
1. A new benchmark consisting of complex goal-driven games is introduced in this study. 
2. An extensive suite of models is evaluated, covering 13 state-of-the-art architectures.
3. The problem statement and the experimental framework are well designed and presented.
4. The authors perform detailed and consistent evaluations across difficulty levels, revealing how and where models fail.

### Weaknesses
While the work is interesting and systematically executed, many of its findings align with prior studies that have already established similar limitations of LLMs and explored methods to overcome them (e.g., Chain-of-Thought reasoning, embedding API calls, or memory modules/database access). The novelty and contribution of this work, therefore, feel limited unless the authors can better justify what new insights their benchmark offers.

Additionally, while the authors show that adding different modules enhances performance, they do not explain why these modules lead to improvement. If such reasoning is provided, the authors should indicate the corresponding line numbers.

### Questions
1. How does this benchmark fundamentally differ from existing benchmarks that also test LLMs or VLMs on interactive or game-based tasks? Specifically, how are these selected games different in terms of difficulty and multi-modality from the other existing benchmarks/games?

2. Is there any difficulty level or stage where even the module-based models fail to improve? The authors should clarify how far these modules can enhance performance and at what point their effect saturates or diminishes.

### Soundness
4

### Presentation
4

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
This paper introduces LMGame-Bench, a benchmark for evaluating large language models (LLMs) and vision-language models (VLMs) on six popular video games (Super Mario Bros., Tetris, Sokoban, Candy Crush, 2048, Ace Attorney) via a unified Gym-style API. Unlike prior game benchmarks that entangle multiple skills, LMGame-Bench uses a modular harness (perception, memory, reasoning modules) to isolate specific capabilities, supports both scaffolded and unscaffolded evaluations, and enhances robustness through data contamination mitigation and prompt standardization. Evaluations of 13 state-of-the-art models show the benchmark effectively discriminates performance (o3 and o1 lead), reveals correlations between games and core LLM capabilities (e.g., Sokoban aligns with math/coding, Ace Attorney with language understanding), and identifies model limitations in visual state extraction, spatiotemporal reasoning, and long-context processing.

### Strengths
1. Modular Harness Design: Addresses a key limitation of prior game benchmarks (entangled skills) by enabling selective activation of perception, memory, and reasoning modules. This allows fine-grained diagnosis of model strengths/weaknesses (e.g., separating perception failures from planning gaps) that was previously unachievable.
2. Rigorous Experimental Design: Evaluates 13 models across 6 diverse games (platformer, puzzle, narrative) with standardized metrics (progression/long-horizon rewards) and statistical validation (paired-sample t-tests, Glass’s δ, coefficient of variation). Results are consistent and reproducible, with detailed ablation studies for harness modules.
3. Insights for Model Improvement: By identifying specific failure modes (e.g., VLMs struggle with board state extraction from images, non-reasoning models lack self-correction), the paper guides concrete advancements in model architecture and agentic design.

### Weaknesses
1. Limited Game Diversity: While the 6 games cover 3 genres, they lack representation of real-time strategy (RTS), open-world, or multiplayer games—domains that test collaboration, dynamic resource management, or complex opponent adaptation. This limits the benchmark’s generalizability to broader game-based agentic tasks.
2. Computational Cost Opacity: While the paper mentions high computational costs (Appendix B.4), it does not provide concrete guidance for scaling evaluations (e.g., cost-saving strategies beyond vague suggestions like "bounding trajectories"). This may limit accessibility for smaller research teams.
3. Perception Module Efficacy: For games like Super Mario Bros., how accurate is the textual representation generated by the perception module in capturing dynamic spatiotemporal cues (e.g., enemy speed, jump physics)? Could gaps in this representation explain why the harness provides limited gains for this game?

### Questions
See Weaknesses.

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
## Summary  
This paper introduces **LMGame-Bench**, a modular, Gym-style benchmark constructed around six popular games (Super Mario Bros., Tetris, Sokoban, Candy Crush, 2048, and Ace Attorney). A core design feature is a **toggleable “gaming harness”** (comprising perception, memory, and reasoning modules) that isolates distinct capabilities and expands performance headroom. Complementing this, the benchmark integrates **contamination checks** and **prompt standardization** to reduce evaluation variance. Experiments conducted on 13 state-of-the-art models demonstrate that the harness significantly improves the benchmark’s ability to discriminate between models while uncovering key failure modes—including limitations in visual state extraction, spatiotemporal control, self-reflection, and long-context reasoning. The paper further employs **correlation and low-rank decomposition analyses** to link game-specific performance to broader clusters of LLM capabilities.  


## Strengths  
1. **Original and Impactful Contribution**: LMGame-Bench addresses a critical limitation of existing game benchmarks—their tendency to entangle multiple skills—by introducing a modular harness that isolates distinct LLM/VLM capabilities (perception, memory, reasoning). This design enables fine-grained diagnosis of models’ strengths and weaknesses, making it a valuable tool for guiding model development.  
2. **Rigorous Benchmark Design**: The benchmark strikes a balance in difficulty (avoiding both premature saturation and excessive hardness) and covers diverse game genres (platformers, puzzles, narrative games), ensuring it effectively discriminates between state-of-the-art models. The inclusion of contamination mitigation (e.g., entity masking, paraphrasing) and prompt standardization (via DSPy’s SIMBA optimizer) further enhances its robustness—a key prerequisite for reliable LLM evaluation.  
3. **Comprehensive Experimental Design**: The evaluation of 13 models (under both harnessed and unharnessed settings) combines quantitative methods (paired-sample t-tests, Glass’s δ effect sizes, correlation analysis) and qualitative insights (failure mode analysis). By linking game performance to core LLM capabilities through low-rank factorization, the work delivers actionable insights beyond mere raw score comparisons.  
4. **Candid Limitation Identification**: The paper openly highlights models’ weaknesses—such as challenges in visual state extraction, spatiotemporal reasoning, and long-context retrieval—and proposes concrete directions for improvement. This avoids the common pitfall of overemphasizing benchmark performance without addressing actionable gaps in model capability.  


## Weaknesses  
1. **Lack of Quantitative Support for Qualitative Analysis**: While Section 3.2 presents a qualitative analysis of model failures, it lacks accompanying quantitative validation. For instance, the paper could address this gap by using LLMs to annotate a subset of game trajectories, identifying the key failure reasons (e.g., “incorrect visual state parsing” vs. “poor long-horizon planning”) for each episode, and calculating the statistical proportion of failures attributed to each cause. Conducting this ablation experiment on a small scale would significantly enhance the credibility of the benchmark’s diagnostic claims.  
2. **Limited Diversity in Evaluation Metrics**: The paper relies primarily on raw scores to evaluate model performance. While raw scores effectively reflect game progress from a human perspective, they provide limited procedural feedback for LLMs—failing to capture nuanced capabilities like reaching critical game nodes (e.g., accessing a bonus area in Super Mario Bros.) or acquiring key information (e.g., identifying critical evidence in Ace Attorney). Incorporating such procedural metrics would offer a more holistic view of model capabilities.  


## Questions  
Q1: Could the authors supplement the qualitative analysis with small-scale quantitative validation (e.g., trajectory annotation and failure reason statistics) as suggested in Weakness 1?  
Q2: Could the authors propose additional, more diverse evaluation metrics—including procedural feedback indicators—to better capture nuanced model capabilities, as outlined in Weakness 2?

### Strengths
See Summary

### Weaknesses
See Summary

### Questions
See Summary

### Soundness
2

### Presentation
3

### Contribution
3
