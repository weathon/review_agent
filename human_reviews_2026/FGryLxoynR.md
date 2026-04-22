# COPER: Agentic Context Significantly Improves and Stabilizes LLM in Multi-Player Game

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 2, 6

## Abstract
Recent multi-player game benchmarks can be sensitive: modest changes to role, system, or judge prompts often flip win-rate rankings under identical decoding; and static, read-once descriptions fail to impart the game-specific priors (rules, legality, action→transition effects) needed for consistent play. We document this context-induced instability and argue evaluation should be agentic: let interaction surface and solidify priors, then evaluate models for both their strength (performance) and reliability (consistency under perturbations). To establish more reliable baselines, we present COPER, a backbone-agnostic, tuning-free self-play recipe that (i) evolves prompts using a conservative TrueSkill lower-confidence bound, (ii) writes structured reflections into a persistent experience bank retrieved across turns to supply rule-aware priors, and (iii) uses prioritized replay to revisit rare, informative states for sample-efficient stabilization. Across five text games, COPER raises mean win rate from 24.9% → 49.5% (GPT-4o-mini) and 21.7% → 44.3% (Qwen-2.5-7B-Instruct) with a small budget (5×400 self-play games per task), and stabilizes agent performance under evaluation. These results show that much of today's LLM game headroom can be unlocked by context rather than weight updates, with COPER yielding strong improvements in negotiation games, competitive results in some imperfect-information settings, and RL remaining more effective in perfect-information games.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces COPER, a training-free and backbone-agnostic framework designed to enhance LLMs' gaming performance through a three-part mechanism: prompt, experience, and replay. It optimizes reliable prompt evolution using the TrueSkill lower confidence bound, maintains a persistent experience bank that writes and retrieves structured reflections, and utilizes prioritized replay to revisit rare and informative states. COPER significantly improves both the winning rate and robustness of GPT-4o-Mini and Qwen-2.5-7B-Instruct across 5 games spanning 3 types of game (Negotiation, Imperfect Information, and Perfect Information), outperforming baseline and other non-RL methods.

### Strengths
1. COPER's mechanism design is comprehensive and well-motivated. By considering evaluation as agentic context construction rather than fixed-prompt play, it integrates experience memory and prioritizes replay under a reliability-oriented objective.
2. This paper's presentation is rigorous. Each component is formalized and defined through precise mathematical formulations describing the mechanism and state transitions, with detailed prompt examples in the Appendix.
3. The authors have clear ablation experiments that isolate the contributions of the experience bank and replay components, providing valuable insights into the effectiveness of each element in the overall mechanism.

### Weaknesses
1. **Limited Breadth of Evaluation:**
While the experiments are sound, the evaluation is restricted to two relatively small models (GPT-4o-Mini and Qwen-2.5-7B-Instruct; the models in Table 4 are also small). Since COPER is a training-free framework, it would be valuable to examine its effectiveness on stronger and larger models such as Qwen-2.5-72Band Claude-3.7-Sonnet, or even on reasoning models like DeepSeek-R1 and GPT-5. Without these larger-scale references, it is unclear whether the proposed mechanism generalizes beyond small models. It is possible that COPER's benefits mainly arise from compensating for small models' limited reasoning and rule comprehension abilities (since some weaker models may even struggle to fully understand game rules and thus rely on additional "experience" component for reasoning support).
2. **Lack of Analysis on Experience Content and Dynamics:**
The persistent experience bank is the core of COPER, but the authors provide limited insight into what kind of knowledge or reflections are actually stored and how they evolve. A qualitative analysis of stored entries (e.g., rule abstractions, strategies, case studies) would make the mechanism's learning dynamics more transparent.

3. **Insufficient Comparison with RL Methods:**
The comparison with RL baselines is limited, although they include the UnstableBaseline and SPIRAL. The paper mainly contrasts COPER with lightweight RL variants but lacks results against stronger full-parameter RL agents such as PPO. Without these comparisons, it is difficult to learn how close COPER approaches the performance frontier of RL methods.

### Questions
1. Could COPER be extended to handle multiple games simultaneously, allowing the model to evolve its context and experience across different gaming environments at once? If so, would the experience memory module need to be redesigned?
2. Could you provide a deeper analysis or a few case studies of the experience bank to understand? For example, which types of stored reflections contribute most to improving the model's gaming performance?
3. Have you analyzed the invalid or rule-breaking actions produced during gameplay? How were such cases detected and handled? If yes, what is the number of invalid actions before and after applying COPER?

Minor suggestions and typos:
1. Line 315 should be in Tab. 1, not 10.
2. Table descriptions for Tab. 3 and Tab. 4 are too short.

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
The authors claim that popular multi-player game benchmarks to measure agents based on language models (LMs) can be sensitive to prompt choices. The resulting instability in outcomes and subsequent rankings harms the validity of such benchmarks to compare model performance. To alleviate this problem, the paper proposes a novel prompt-optimization protocol called COPER that combines three components: (i) evolutionary prompt optimization, (ii) model-generated insights based on game trajectories, and (iii) prioritized experience replay to balance state coverage. Using COPER, an LM-based agent plays a target game under several generations of self-play to optimize its game context. The paper evaluates COPER on five games using two models (gpt-4o mini and Qwen-2.5-7B-Instruct) against several baselines and three non-optimized models. The authors claim that COPER reaches higher performance and is more efficient than alternative methods.

### Strengths
[**significance**] Prompt optimization is an important area of research for LM-based agents. Due to the large size and/or black-box nature of many models, weight optimization is often impractical or impossible. The authors propose a practical method to improve absolute performance of an LM-based agent on a known task under (unlimited) offline sample collection.


[**quality**] The paper evaluates on three game types with different, relevant characteristics. The paper further attempts to conduct relevant ablations and stratifications.

### Weaknesses
[**clarity**] The authors introduce a lot of notation that can be challenging to follow. For example, line 128 defines “p” as the player index, while line 134 redefines it as the instruction prompt. Or “S” as the state space (l123) and the capacity in l204. Other variables and concepts are under- or undefined entirely, e.g., “method” (l140), “run” (l146), “the collector” (l231). This lack of clarity extends to in-line references, e.g., “this template” (l151).


[**quality**]
- A number of statements lack citations or supporting evidence. For example, (l62) “Unfortunately, current game-LLM [...] under-agentic”, (l46) “Because prompts [...] can flip ELO comparisons and reorder models under identical decoding”, (l72) “Secondly, read-once descriptions [...] games states and pay-offs”, “While benchmarks provide [...] single reading”, “Without interaction-driven [...] strategic play”, (l156) “KUHNPOKER”

- The authors claim their protocol is “backbone-agnostic” and “training-free” (l109, l480). The protocol is certainly a form of training, it just does not involve weight updates. It is unclear to this reviewer what “backbone-agnostic” refers to. While the authors cite budget constraints, the small model suite of two appears insufficient to support the strong statements.

- Throughout the paper, reported results (i) lack confidence intervals, (ii) “winning” means something different for different games. For example, in zero-sum negotiations, ending a game 0.51 vs. 0.49 likely should be given a different interpretation than ending a game 0.99 vs. 0.01.

- The core of COPER relies on self-play optimization. As pointed out by e.g., [1], self-play can lead to distinctly different behaviors and considerations compared to heterogeneous interactions. Step 1 of the protocol creates different contexts optimized for specific models. If both models are not given the same optimization budget, this leads to an unfair comparison across models in multi-player settings. This is also concluded in section 5, observation 4, putting in question the “win rate” statistics presented throughout.

- The COPER method has three distinct components (i) evolutionary prompt optimization, (ii) reasoning-based insights over full trajectories, and (iii) experience replay to sample rare states. The experimental setup fails to evaluate these components separately, thus making it impossible to attribute changes in performance to the different components. In Table 2, components (ii) and (iii) are only shown in combination with (i). Table 2 also omits important details like number of samples and confidence intervals. Crucially, experiments do not control for the confounding effect of (much) longer context lengths on downstream performance. This could also explain the rather surprising results of observation 3 (section 5) – see questions.
- While the central stated purpose of COPER is to improve stability of game-based benchmarks caused by prompt choices, it appears that the meta optimizations done by COPER to optimize prompts also rely on a variety of (subjective) prompt choices.

[**significance**] Overall, this framework appears to measure something orthogonal to the original multi-player games it is designed to stabilize. Specifically, it measures the ability of models to create and learn from in-context insights, rather than play a given game in a single forward pass. While certainly useful, it changes the required experimental setup and representation of results.

### Questions
q1. Observation 3 is very confusing and potentially damning: why would insights regarding the rules of game A transfer to a completely different game B? This suggests that gains might not come from the prompts at all, but simply from an increase in context length. 


q2. Observation 5 should be stratified by model, game, and hyperparameters. As is, the claim can not be evaluated fairly.


q3. How were hyperparameters chosen throughout this work? For example, how were the initial prompts initialized, the permutation options chosen, the number of generations, etc.


q4. Under evolutionary optimization, how are valid game constraints/instructions guaranteed?


q5. This framework appears to have a combinatorial explosion built-in. Specifically, how would you create a single-set of prompts across multiple models? And, how would this work for slightly more complex games that have larger state spaces and/or more complicated instructions?


Other suggestions:
- It would be useful to share qualitative samples of “learned” contexts and compare them to “base” contexts
- I suggest looking at [1] for the related work on LLM for games and self-play/cross-play evaluation. 
- Given the stated budget constraints, perhaps extend the evaluation to smaller open-source models?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes **COPER**, an engineering pipeline that treats context as an optimizable, persistent resource to improve large language models in multi-agent/multi-round text games. COPER combines (1) prompt evolution using a selection rule based on TrueSkill and lower-confidence bounds, (2) an experience bank (CRUD-style reflection storage and retrieval), and (3) prioritized replay to revisit rare/high-information episodes. The authors show performance and stability improvements across several text-game tasks, reporting large win-rate gains and reduced ranking variance for models like GPT-4o-mini under relatively small self-play budgets.

### Strengths
1. **Tackles an important empirical problem**, considering the instability and prompt-sensitivity in multi-agent text games
2. The COPER pipeline is **concretely specified**, making reproduction feasible; authors provide pseudocode and appendices with additional details.
3. Contains **multiple ablations and baselines** across a variety of game types (negotiation, imperfect/complete-information), showing **clear numerical improvements** in presented settings.

### Weaknesses
1. The **RL comparison is not sufficiently clear**, the paper lacks budget-matched RL baselines and a full accounting of token and wall-clock costs.
2. There may be **generator–evaluator bias** due to heavy reliance on self-play and reflections from the same model family could produce generator-specific artifacts rather than broadly useful strategies.
3. The **quality of the experience bank is unknown** because the paper does not report statistics on entry accuracy, growth over time, or examples of incorrect or harmful entries and their impact.
4. **Replay hyperparameter sensitivity has not been tested** because the authors use defaults for α, β, and B without performing a sweep to demonstrate robustness.
5. Claims about **broad generalization are overstated** because the limited and mixed cross-model and cross-game transfer results do not yet support such sweeping conclusions.

### Questions
1. Please describe in detail the training and hyperparameter settings for the RL baselines in the main comparisons (including seeds, compute, token counts). Are those baselines given sufficient budget and hyperparameter tuning?
2. How are prompt variants constructed for the ranking-stability experiments? Can you expand the set of perturbations (paraphrase, length, adversarial phrasing) and report stability under those?
3. Provide examples of experience-bank entries (both high-quality and mistaken). How often do mistaken entries occur and what is their impact?  
4. Why were the replay parameters chosen (α, β, B)? Can you provide evidence that your default values are near-optimal or at least robust?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles prompt instability in multi-player LLM game benchmarks, where small prompt changes flip model rankings. The authors propose COPER, which combines prompt evolution, an experience bank of gameplay insights, and prioritized replay. Experiments across five games show substantial win-rate improvements.

### Strengths
The paper does a good job highlighting prompt sensitivity in multi-agent settings, a real problem that doesn't get enough attention. The experimental work is solid, covering multiple games, models, and runs with proper ablations. The core idea of building game understanding through experience rather than weight updates makes sense and the results back it up. The method is also way more efficient than RL baselines

### Weaknesses
### Major comments


- Table 2 shows replay adds only +2.1% on average, which looks within SE. COPER already has three moving parts, so either run more experiments to prove replay matters or just drop it? The 22.7% token savings on SimpleTak is neat but you'd need to show this holds for other games too without losing performance.

- Everything's averaged across three opponents with no breakdowns. This matters because (a) opponent choice clearly affects your numbers, (b) seeing which opponents COPER struggles with would show when it's actually useful, and (c) you can't tell if you're getting better at games or just exploiting specific model weaknesses.

- Five games and two smaller models (7B and GPT-4o-mini) isn't much to claim broad applicability. Need to test on more diverse games and bigger models with better context processing. 

- The authors advocate for multi-prompt evaluation but in the experiments, only fixed prompts are used for the opponent. Ideally, experiments should include variants of the prompts for the opponents as well. 

---
### Minor comments:

- Throwing around "TrueSkill lower-confidence bound" in the intro without any context might lose readers who don't already know these methods. Either explain them briefly or at least point to where they're defined later.

- It is not clear from Figure 2 what CP vs C_g actually are visually.

- [Line 380] If the columns represent the source tasks, shouldn’t it be the other way around? For example, KuhnPoker → SimpleTak shows a +25.9% improvement, and similarly for the other cases.

### Questions
- [Line 239]: Why only give experience to fraction π of agents? What happens if you change π?

- Does M saturate? How do you balance old vs new insights?

- Practical prompt stratification: Any insight on how many prompt variants are sufficient for evaluation and how should they be selected?

### Soundness
4

### Presentation
4

### Contribution
3
