# Sparks of Cooperative Reasoning: LLMs as Strategic Hanabi Agents

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 8, 4, 2

## Abstract
Cooperative reasoning under incomplete information is a significant challenge for both humans and multi‑agent systems. The card game Hanabi embodies this challenge, demanding theory of mind reasoning and strategic communication. We present the largest evaluation to date of Large Language Models (LLMs) as Hanabi playing agents, assessing 17 state‑of‑the‑art LLMs in 2 to 5‑player cooperative multi-agent settings. We investigate why multi-agent coordination failures persist by systematically evaluating the impact of context engineering, from simple game state (Watson) tracking to scaffolding reasoning with explicit card deductions motivated by Bayesian inference (Sherlock) across a wide range of LLM capability (from 4B to 600B+ parameters). To our knowledge for the first time, we show 1) agents can maintain a working memory to track game state (Mycroft) instead of being explicitly provided engine deductions  2) a smooth interpolation of cross-play performance between different LLMs. In the Sherlock setting, the strongest reasoning models exceed 15 points out of 25 on average across all player counts, yet they still trail experienced human players and specialist Hanabi agents, both of which consistently score above 20. Lastly, we release the first public Hanabi datasets with move utilities and annotated game trajectories:  1) HanabiLogs: 1,520 full game logs for instruction tuning and 2) HanabiRewards: 560 games with dense move‑level value annotations (rewards) for all candidate moves. We demonstrate that supervised and RL finetuning of a 4B open-weight model (Qwen3-Instruct) on our datasets improves cooperative play on Hanabi by 21% and 156% respectively, which is within ~3 points of a strong proprietary reasoning model (o4-mini), and surpasses the best non-reasoning model (GPT-4.1) by 52%. The HanabiRewards RL finetuned model generalizes beyond Hanabi, improving performance on the recent cooperative group-guessing game benchmark by 11%, temporal reasoning on EventQA by 6.4%, instruction-following on IFBench-800K by 1.7 Pass@10, and identical mathematical reasoning Pass@10 on AIME 2025.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the use of Hanabi as a benchmark for evaluating large language models (LLMs) in cooperative reasoning. The authors adapt the game into a multi-turn, multi-agent format and evaluate a diverse set of LLMs under three prompting schemes (Watson, Sherlock, and Mycroft) which vary the degree of explicit and implicit information accessible to each agent. Beyond evaluation, the paper introduces two new Hanabi datasets. The authors further show that a smaller model fine-tuned (SFT) on this new data substantially outperforms strong baselines and existing state-of-the-art models on their cooperative reasoning benchmark. Overall, the paper positions Hanabi as a controlled yet expressive environment for testing an LLM’s ability to deduce hidden information from evolving states and coordinate with others based on incomplete knowledge.

### Strengths
- Extensive model coverage: The paper evaluates a wide range of LLMs under three distinct prompting setups (Watson, Sherlock, Mycroft) and multiple random seeds, providing a thorough view of model performance across configurations.
- Benchmark insight: The work effectively demonstrates that Hanabi, when reformulated under their multi-agent prompting scheme, can serve as a useful benchmark for studying implicit reasoning and cooperative inference in LLMs.
- Transparency and reproducibility: The experimental setup, prompts, and evaluation details are well-documented, ensuring replicability.
- Clear motivation: The paper makes a compelling case for moving beyond single-agent reasoning benchmarks toward cooperative, belief-dependent tasks.

### Weaknesses
- The novelty relative to prior benchmark studies (like those mentioned LLM-Arena and SPIN-Bench) is not discussed in depth enough beyond their introduction in the related works sections.
- The generalisation value of fine-tuning a specialised Hanabi model remains limited. It is unclear whether such training improves reasoning in other cooperative or belief-based tasks, and thus the benefit of both the datasets and the SFT results remain unclear.

### Questions
1. Could you clarify how your approach differs from a “single prompt per turn” setup, as stated in the related works section.
2. You mention “adding Hanabi strategies.” Could you elaborate on what these strategies are, and how they are represented or introduced into the model’s reasoning process?
3. It would be helpful to understand why non-reasoning models performed worse with added contextual information. Does this suggest overfitting to irrelevant details or difficulty managing longer prompts? 
4. You note that there are exceptions to the performance vs number of agents pattern. performance. Do you have any insight into why these exceptions occur? Might it have to do with the unique strategies each agent uses?
5. Regarding Mycroft: why not rely on the model’s own deductions to update its knowledge about its hand?
    - Related: If a model fails to deduce something correctly or omits key information, is that information permanently lost from its belief state, or can it be recovered later.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a large-scale evaluation of 17 Large Language Models (LLMs) on their ability to perform cooperative reasoning in the challenging incomplete-information card game, Hanabi. The authors investigate performance across 2-5 player settings and explore the impact of different context scaffolding techniques: MinCon (minimal state), DeductCon (state plus explicit deductions from the game engine), and Mycroft (a multi-turn setup requiring implicit state tracking by the LLM) . Key findings indicate that reasoning-enhanced LLMs outperform non-reasoning ones, and explicit deductions (DeductCon) significantly boost performance (average scores >15/25), although still falling short of expert humans and specialized Hanabi agents (>20/25). The Mycroft setting proved difficult due to state-tracking challenges. The paper also contributes two valuable datasets: HanabiLogs (1,520 game trajectories) and HanabiRewards (560 games with dense move-level value annotations). Instruction-tuning a small LLM (Qwen-3-4B) on HanabiLogs demonstrated a 21% score improvement, surpassing some larger closed models in the DeductCon setting.

### Strengths
**Scale and Scope:** The evaluation is comprehensive, covering a wide range of recent LLMs (17 models), multiple player counts (2-5), and includes cross-play experiments. This provides a robust snapshot of current LLM capabilities in Hanabi.

**Dataset Contributions:** HanabiLogs and HanabiRewards are significant contributions, providing richly annotated data with move utilities, which can fuel future research in instruction tuning and reinforcement learning for cooperative agents. The successful SFT demonstration validates HanabiLogs' utility.


**Systematic Scaffolding Analysis:** The comparison between MinCon, DeductCon, and Mycroft offers clear insights into how context, explicit knowledge, and implicit state tracking affect LLM performance in cooperative reasoning tasks.


**Transparency and Positioning:** The paper clearly articulates its methodology, acknowledges limitations of prior work regarding reproducibility , and provides context relative to existing benchmarks and agents .

### Weaknesses
**Scaffolding Choices:** While insightful, the performance heavily depends on the chosen scaffolding (MinCon, DeductCon). It remains unclear why these specific strategies are chosen, and makes readers wonder about generalization to settings without such explicit support, different context, unseen partners and open-ended conventions.

**Implicit State Tracking Analysis:** The Mycroft setting highlights difficulties with implicit state tracking, but the analysis doesn't quantify the source of error (e.g., context limits vs. reasoning failure) or evaluate the quality of the state/deductions the models attempted to track.

**Inconsistencies:** In the abstract on OpenReview, the improvement via instruction tuning on HanabiLogs is 24%, whereas it becomes 21% in the abstract of the pdf version paper.

### Questions
1. Could you provide a more detailed ablation study isolating the contribution of each component within the Sherlock (DeductCon) prompt (e.g., strategic advice vs. explicit deductions)? Are the full prompts used for Watson, Sherlock, and Mycroft available?

2. Could you report contamination checks and take the fact that models may be exposed to Hanabi tutorials/solutions into consideration?

3. Have you considered or conducted experiments involving human-LLM cross-play, particularly with human players unfamiliar with typical LLM strategies, to assess adaptability and convention formation?

4. How were the dense move utilities in HanabiRewards generated? Do they rely on heuristics, model estimates, search algorithms, or human annotations? Clarity on their origin is important for their use in RL.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors evaluate a suite of frontier language models on the Hanabi Learning Environment. The HLE is a popular framework in multi-agent reinforcement learning to evaluate agents' abilities to coordinate and cooperate. The authors reveal a significant discrepancy in performance between frontier models, although with seemingly high variance. They also show that additional information is not necessarily beneficial to performance for all models. However, for the models where it is beneficial, these models struggle to maintain a game history themselves, and performance declines, although it is unclear whether this is statistically significant.

### Strengths
1. The authors are transparent about the metrics they report and the hyperparameters they use.
2. The plots are mostly well-formatted and easy to read.
3. The paper follows a logical flow and is easy to follow.
4. The experiments seem reasonable.
5. Evaluating LLMs on coordination and cooperation tasks is highly significant for the community, and reporting trustworthy results on a popular benchmark is very useful. Furthermore, fine-tuning data can also be highly impactful and useful.

### Weaknesses
## Clarity
1. Citation formatting is inconsistent.
2. Sometimes citations are missing, like for BAD and SAD.
3. The writing tense changes

## Quality
1. Overall, while the authors provide the standard deviations, they do not seem to account for them when making conclusions. It appears that most standard deviations heavily overlap, and it's unclear whether any of the conclusions hold up under a statistical significance test. Especially with such high variance, the authors might want to consider using libraries like rliable (https://agarwl.github.io/rliable/) to make conclusions confidently. For example, the authors state

> Our high-level goal is to evaluate and improve the cooperative capabilities of LLMs in multi-agent settings, which we do in this work through the lens of Hanabi.

But none of the improvements via the fine-tuning dataset appear statistically significant. 

## Soundness
1. The fine-tuning is motivated by an increase in performance and matching the performance of "strong" closed-source models like GPT-4o and Grok 3. However, these models rank among the lowest performing models, as shown in Figure 3 in DeductCon. The authors should consider adapting the language.

### Questions
1. Please consider increasing the size of the legend in Figure 4.
2. Would the authors consider doing a more thorough analysis if the performance rankings and changes in average performance hold up under any statistical significance test? Currently, it's extremely challenging to discuss experiments and propose potential improvements when most of them do not discriminate between models or game settings.

If the authors could address the above concerns, I'm looking forward to continuing the discussion during the rebuttal period.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper evaluates the cooperative reasoning abilities of 17 state-of-the-art LLMs in the challenging multi-agent card game Hanabi. By designing and investigating various prompting strategies, deductive context with engine-provided inferences (Sherlock/DeductCon), and a novel multi-turn implicit deduction regime (Mycroft). The authors aim to dissect the limitations and capabilities of LLMs in strategic, theory-of-mind based cooperation under incomplete information. The study includes extensive empirical benchmarking across 2-5 player settings, a broad cross-section of LLMs, ablation studies on context engineering and team composition, and the creation of two new Hanabi datasets (HanabiLogs and HanabiRewards) for instruction tuning and reward optimization.

### Strengths
1. This paper is well-written and well-organized.
2. The paper provides a comprehensive benchmark, which includes several structured context designs, and the dataset could support future studies.

### Weaknesses
1. The analysis in this paper remains superficial, just showing the scores without the in-depth analysis.
2. The entire work is constrained to a single game domain and does not show how this benchmark generalizes to the real-world setting.
3. No mention of sampling temperature, top-k/p values or seed control. 
4. It is unclear whether models play as independent agents or as a shared model controlling all players sequentially.

### Questions
1. How do you operationally define “cooperative reasoning” in this context? What measurable behavior distinguishes it from simple rule-following or pattern matching?
2. How do you justify general claims about “multi-agent reasoning” from a single domain (Hanabi)? Have you tested transfer to any other cooperative tasks?

### Soundness
2

### Presentation
3

### Contribution
2
