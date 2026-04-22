# Reasoning Boosts Opinion Alignment in LLMs

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Opinion modeling aims to capture individual or group political preferences, enabling applications such as digital democracies, where models could help shape fairer and more popular policies. Given their versatility, strong generalization capabilities, and demonstrated success across diverse text-to-text applications, large language models (LLMs) are natural candidates for this task. However, due to their statistical nature and limited causal understanding, they tend to produce biased opinions when prompted naively. In this work, we study whether reasoning can improve opinion alignment. Motivated by the recent advancement in mathematical reasoning enabled by reinforcement learning (RL), we train models to produce profile-consistent answers through structured reasoning. We evaluate our approach on three datasets covering U.S., European, and Swiss politics. Results indicate that reasoning enhances opinion modeling and is competitive with strong baselines, but does not fully remove bias, highlighting the need for additional mechanisms to build faithful political digital twins using LLMs. By releasing both our method and datasets, we establish a solid baseline to support future research on LLM opinion alignment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors present a novel approach toward aligning LLMs with individual opinions in a political context, which could enable downstream applications such as more representative digital democracies. In contrast to prior work that focuses on in-context learning, they use reinforcement learning to train models to align to the political opinions of individual candidates, finding gains in opinion alignment when compared to certain baselines. They also perform a detailed analysis of results and find that models are more easily aligned to left-wing political groups, but that trained agents tend to lean more centrist.

### Strengths
- The authors identify a key problem in opinion alignment that current prompting-based approaches fail to adequately solve, and that could enable highly prosocial applications if solved, and propose a novel solution.
- I found the political ideology analysis (sections 4.5, 4.7, 4.8) really interesting, especially the experiment with flipped labels in 4.8 that showed that left-leaning opinion profiles are easier to learn than synthetic left-leaning profiles constructed by inverting right-leaning opinion profiles. The findings that right and center positions are harder to learn than left ones, and that fine-tuning tends to push agents to the center, make interesting additions to the dialogue around opinion modeling with LLMs.

### Weaknesses
- The novel contribution of this work (using RL to train models to reason for opinion alignment) doesn’t seem to yield much improvement over SFT on seen opinions. In Table 3, SFT+GRPO is only a statistically significant improvement over SFT in three of the 12 base model - dataset combinations (Qwen3 8B + WoM, Qwen3 8B + ANES, Magistral 24B + ANES). The authors note that this relates to dataset scale, but SFT+GRPO doesn’t seem to consistently improve over just SFT even on the largest WoM dataset. Given the difficulty of collecting individual opinions, it may be difficult to apply the proposed method to new users.
- Given the amount of prior work on opinion modeling with LLMs in survey environments, I found “Our work lays the groundwork and introduces the first benchmark for systematic opinion modeling with LLMs” to be a bit of an overclaim.

### Questions
- How does opinion alignment scale with the number of questions personalized models are trained on (SFT, GRPO) or prompted with (ICL)? Analyzing the scaling behavior could help us understand what kind of approaches are most likely to work with current datasets, or inform the collection of future datasets.
- Why do you think your trained agents are generally more centrist and conservative than the opinions they are trained on (Figure 2), and how does this vary across different training/prompting settings?

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
3

### Summary
This paper looks at whether reasoning helps align LLMs with human political opinions. The authors fine-tune LLMs on political survey data from the US, Germany, and Switzerland, and then train them further using Group Relative Policy Optimization (GRPO) — a reinforcement learning method that encourages structured reasoning before giving an answer. The goal is to make models answer policy questions like real individuals or parties would. Experiments show that reasoning-based training (SFT+GRPO) outperforms prompt-only and SFT-only baselines, improving F1 scores across all datasets. Overall, the paper argues that reasoning improves opinion alignment but isn’t enough to eliminate political bias entirely.

### Strengths
1. The paper studies a very relevant question — how reasoning affects opinion alignment, which feels timely and important.

2. The method is clear and builds nicely on recent RL reasoning techniques.

3. The experiments are well-organized and tested across multiple real-world political datasets.

4. The analysis is thoughtful, especially the discussion of ideological bias and neutral-class difficulty.

### Weaknesses
1. The clarity of paper should be further refined, and there are several typos in the content.

2. It’s not entirely clear whether GRPO genuinely improves reasoning or just fits survey patterns better.

3. The paper shows biases but doesn’t really explain their sources or propose fixes.

### Questions
1. Could you please provide a more detailed explanation of Figure 4, especially the panels on the right? I would like to understand what those plots specifically represent and what the error bars indicate in this context.

2. Can the authors show evidence that GRPO improves reasoning quality rather than just accuracy?

3. Did you analyze why right-leaning or neutral profiles are harder to learn — is it data imbalance or model bias?

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
3

### Summary
This paper investigates whether reasoning-based approaches can improve LLMs' ability to model individual political opinions. The authors train models using reinforcement learning (specifically GRPO - Group Relative Policy Optimization) to generate reasoning traces before providing political stances, using survey data from three political contexts: Swiss candidates (smartvote), German parties (Wahl-O-Mat), and U.S. voters (ANES 2020). The key contribution is showing that structured reasoning through SFT+GRPO outperforms baseline approaches, though significant challenges remain with neutral stances and right-leaning political positions. The work establishes a benchmark for opinion modeling and releases datasets for future research.

### Strengths
1. First systematic study of using RL-based reasoning for individual-level political opinion modeling, moving beyond demographic-based approaches
2. Three datasets from different political systems (US, Germany, Switzerland) provide robust cross-cultural validation
3. The paper identifies a gap between demographic-prompted political simulation (Santurkar et al., 2023; Argyle et al., 2023) and individual-level agents that must stay consistent with a known survey profile.

### Weaknesses
1. Training one model per individual is computationally prohibitive for real-world deployment
2. The method still trains one model per persona, which is expensive and does not scale to population-level simulation; the paper acknowledges this, but the main method remains hard to apply in real digital-democracy settings where thousands of agents are needed.

### Questions
The current pipeline does SFT on synthetic arguments generated by Llama 3.1 70B. How sensitive is the final SFT+GRPO model to the style of these synthetic arguments? For example, if the initial CoT is written in a more partisan tone, will GRPO converge to a different political center?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores whether explicit reasoning improves *individual-level* opinion alignment in LLMs. The task is to predict a persona’s stance $(y\in{\text{Yes, Neutral, No}})$ on survey questions (q), given the persona’s past answers. The model outputs `<reasoning>...</reasoning><answer>...</answer>` and is trained with GRPO (group-relative PPO) using a composite reward for format correctness, rationale length, and answer agreement. An optional SFT warm-start with synthetic rationales is applied before GRPO. Experiments span smartvote (CH), Wahl-o-Mat (DE), and ANES (US), showing that SFT+GRPO improves macro-F1 (e.g., 70.7 on smartvote) and that Neutral responses and right-leaning personas are harder to predict. Code and data are released.

### Strengths
* **Clear, reproducible method and reward design.** The structured format and composite reward are explicit, with training hyperparameters (LoRA, steps, schedulers) fully reported.
* **Multiple datasets and ideologies.** The paper evaluates on smartvote (binary), Wahl-o-Mat (ternary), and ANES (ternary), with clear unit definitions and splits.
* **Variance reporting across 8 stochastic runs.** Results include mean ± s.d., improving robustness.
* **Insightful analyses of failure modes and class imbalance.** Figures analyzing ideology and Neutral class are carefully interpreted; removing Neutral boosts F1 but preserves ideological gaps.
* **Resource release.** Code and datasets are linked, supporting reproducibility and follow-up work.

### Weaknesses
1. **Missing strong baselines for political alignment.** The paper omits comparisons against recent specialized methods that align LLMs to political viewpoints using supervised preference optimization or domain corpora (e.g., Stammbach et al. 2024).

2. **Limited persona-conditioning alternatives.** The method trains *one model per persona*, not comparing to shared-parameter persona embedding methods.

3. **Synthetic SFT rationales risk leakage/priors.** Synthetic arguments generated by a large LLM may encode political priors.

4. **Scope of SOTA claim is limited.** No head-to-head with specialized political fine-tuning models.

5. **Evaluation depth on ANES is modest.** Only 21 respondents modeled, limiting statistical power.

6. **Fairness and calibration analyses are limited.** The paper lacks per-persona calibration or reliability metrics.

### Questions
1. Please compare against at least one specialized political-alignment baseline (e.g., Stammbach et al. 2024) and/or ORPO-style preference optimization.
2. Can you provide a *shared model* baseline that conditions on persona embeddings instead of per-persona training?
3. How sensitive are results to the synthetic SFT data source?
4. Can you expand ANES to more respondents and report per-respondent confidence intervals?
5. How did you prevent test leakage from SFT synthetic data?
6. Could you include calibration metrics (ECE, coverage-accuracy)?
7. What is the compute footprint per persona for GRPO training?

### Soundness
3

### Presentation
3

### Contribution
3
