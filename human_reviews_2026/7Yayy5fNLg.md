# SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Recent advances in reinforcement learning have shown that language models can develop sophisticated reasoning through training on tasks with verifiable rewards, but these approaches depend on human-curated problem-answer pairs and domain-specific reward engineering. We introduce SPIRAL, a self-play framework where models learn by playing multi-turn, zero-sum games against continuously improving versions of themselves, generating an automatic curriculum of stronger opponents, and eliminating the need for human supervision. To enable this self-play training at scale, we implement a fully online, multi-turn, multi-agent reinforcement learning system for LLMs and propose role-conditioned advantage estimation (RAE) to stabilize multi-agent training. SPIRAL produces reasoning capabilities that transfer broadly, improving performance by up to 10\% across a suite of 8 reasoning benchmarks on 4 different models spanning Qwen and Llama model families, outperforming supervised fine-tuning on 25,000 expert game trajectories. Multi-game training (TicTacToe, Kuhn Poker, Simple Negotiation) yields the strongest results, with improvements observed across both base and instruction-tuned models. Analysis of chain-of-thought traces reveals that games develop distinct cognitive patterns that transfer to improve reasoning performance, with different games developing complementary strengths. Even models which have already been trained on reasoning tasks using RLVR, like DeepSeek-R1-Distill-Qwen-7B, still benefit from our approach. These results demonstrate that zero-sum games naturally develop transferable reasoning capabilities across diverse model architectures and training stages, highlighting a promising direction for autonomous reasoning development.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SPIRAL, a self-play reinforcement learning framework that enhances LLM reasoning by training models through multi-turn, zero-sum language games (e.g., TicTacToe, Kuhn Poker) against evolving versions of themselves. The key insight is that competitive game dynamics generate an automatic curriculum of increasingly challenging reasoning tasks without requiring human-curated problems or domain-specific reward engineering. To stabilize training, the authors introduce Role-conditioned Advantage Estimation (RAE), which mitigates variance in multi-agent policy gradients. Experimental results show significant improvements on mathematical reasoning benchmarks, with evidence of reasoning pattern transfer from games to academic tasks. The approach is evaluated across multiple model families, demonstrating broad applicability.

### Strengths
**The work proposes a novel training paradigm**: instead of relying on human-curated problems with verifiable answers, SPIRAL uses self-play in zero-sum language games to create an autonomous curriculum of increasingly complex reasoning challenges. While the game outcomes are still verifiable, the strategies and subproblems emerge dynamically from competition, enabling the model to develop transferable reasoning skills without direct exposure to target domains.

**Empirically Promising Results**: SPIRAL achieves substantial gains on math reasoning benchmarks compared to base models, even without exposure to target problems during training. The qualitative analysis of reasoning pattern transfer (e.g., case analysis, expected value calculation) provides mechanistic insights beyond mere performance metrics.

**Broad Applicability**: The method demonstrates consistent benefits across diverse model architectures (Qwen, Llama, Octothinker) and training stages (base, instruction-tuned, and even RLVR-trained models), suggesting strong generalization potential.

### Weaknesses
**Limited Benchmark Coverage**: Evaluation is narrowly focused on mathematical reasoning. There is no evidence of improvement on other critical reasoning domains such as code generation, logical deduction, causal reasoning, or multi-hop QA. This limits claims about general reasoning enhancement.

**Missing Pipeline-Level Experiments**: The paper suggests SPIRAL can serve as a foundational or complementary stage in modern training pipelines (e.g., pre-training before RLVR or fine-tuning after). However, no experiments compare:

1) Base → SPIRAL → RLVR vs Base → RLVR

2) RLVR-Trained Model → SPIRAL vs continued RLVR/SFT Without these, the practical value of SPIRAL as a booster or enhancer remains speculative rather than demonstrated.

**No Assessment of General Capability Preservation**: The paper does not evaluate whether SPIRAL training degrades performance on non-reasoning tasks (e.g., instruction following, commonsense reasoning, language modeling). This raises concerns about catastrophic forgetting or capability misalignment, which are crucial for real-world deployment.

**Claims Outrun Evidence**: While the title and abstract imply broad applicability to "reasoning," the experimental validation is largely confined to math. The claim that SPIRAL develops "transferable reasoning capabilities" would be stronger with more diverse downstream evaluations.

### Questions
1) Has the impact of SPIRAL training on general capabilities been evaluated? Specifically, what happens to performance on MMLU, AlpacaEval, or MT-Bench after SPIRAL post-training? Is there any degradation in instruction-following or commonsense reasoning?

2) Have the authors conducted experiments where SPIRAL is used as a pre-training stage before RLVR (e.g., Base → SPIRAL → RLVR)? If so, could you include a comparison with Base → RLVR in terms of final performance and convergence speed?

3) When stating that models like DeepSeek-R1-Distill still benefit from SPIRAL, does this refer to further training after RLVR? If yes, how does it affect performance on the original RLVR benchmarks?

4) Can the authors provide ablation studies on game diversity? For example, does combining multiple games (TicTacToe + Kuhn Poker) outperform single-game training or simple data mixing? Do different games induce complementary reasoning skills?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces SPIRAL, a novel and compelling framework for improving the reasoning capabilities of large language models through multi-agent, multi-turn self-play in zero-sum games. The core contribution is the demonstration that complex, transferable reasoning skills can be developed without reliance on human-curated problem-answer datasets, which represents a significant step towards more autonomous and scalable AI development. By leveraging an adaptive curriculum of ever-improving opponents, SPIRAL effectively teaches models to reason strategically. The technical innovations, particularly Role-conditioned Advantage Estimation (RAE), and the extensive empirical results provide strong evidence for the viability of this approach. This is a high-quality paper with major findings, and I recommend its acceptance.

### Strengths
The paper is written in a clear and understandable manner, with a well-defined methodology and simple yet effective improvement strategies that are easy to follow.

●	This work makes a major contribution toward the goal of self-improving LLMs by reducing the dependence on human-curated data. By using self-play as a source of unlimited training signal, SPIRAL facilitates significant model self-improvement. This approach could represent the next paradigm for RLVR, moving beyond the domains of math and code reasoning.

●	The experiments are comprehensive, conducted on five different models (Qwen3-4B, Qwen3-8B, Octothinker-8B, DeepSeek-Distill-Qwen-7B, and Llama-3.1-8B) from two model families (Qwen, Llama), including base models, instruction-tuned models, and those already fine-tuned for reasoning. The consistency of improvements across eight challenging benchmarks is a testament to the generalizability of the skills learned through SPIRAL.

●	The paper provides valuable resources for the community by delivering a complete, fully online, multi-turn, multi-agent RL framework. The core technical contribution, RAE, effectively solves the critical "thinking collapse" problem and is broadly applicable to future MARL experiments. Furthermore, the framework is orthogonal to other emerging techniques; for instance, the auto-curriculum from SPIRAL could be combined with prolonged RL methods [1] to continuously push reasoning boundaries.

●	The paper goes beyond simply reporting benchmark scores by providing a deep and insightful analysis of skill transfer. Identifying and tracking specific cognitive patterns [2] (e.g., Case-by-Case Analysis), provides a compelling explanation for why the method is effective and builds confidence that the model is learning genuinely useful strategies.

[1] https://arxiv.org/abs/2505.24864
[2] https://arxiv.org/abs/2503.01307

### Weaknesses
While this is a great paper, there are several areas where further discussion or exploration could enhance its contribution. These points are primarily intended as constructive feedback instead of reasons to reject.

●	The training resources are demanding, which could be a barrier to broader adoption. A discussion on potential avenues for improving computational efficiency (i.e. LoRA) would be a valuable addition.

●	In line 412, it would be helpful if the authors could further elaborate on how the fixed-opponent baselines (vs. Gemini and random) address the issue of spurious rewards. The connection is not clear.

### Questions
1.	What do you think about the results on training with SFT (Qwen3-4B-SFT) is better than RL against Gemini and Mistral models (Qwen3-4B-Gemini and Qwen3-4B-Mistral)?
2.	How do you see the SPIRAL framework extending beyond the zero-sum setting? For example, could it be adapted for cooperative or mixed-motive games to elicit other valuable skills like collaboration and negotiation?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper uses online policy optimization (REINFORCE) with LLM self-play in zero-sum games (Tic-Tac-Toe, Kuhn Poker, and Simple Negotiations). The authors demonstrate how their approach improves downstream reasoning across multiple models (including a model that was already trained using RLVR - DeepSeek-R1-Distill-Qwen-7B). While games offer verifiable, human-free rewards, they shift complexity to game environment design and validation, which can shape emergent strategies, generalization, and scalability of this method.
They introduce a role-conditioned advantage to reduce variance and adjust it for asymmetric reward expectations (e.g., first-move effects in Tic-Tac-Toe and Kuhn Poker) and show it prevents policy collapse.
They present a systematic bottom-up method for automated evaluation of reasoning traces to discover common ("transferable") patterns (Appendix F).
Overall, the authors demonstrate up to 10% gain in average on downstream tasks and a considerable margin compared to SFT on “expert” trajectories.

### Strengths
* The paper systematically shows transfer from simple games to harder reasoning tasks, strengthening prior evidence that self-play improves LLM reasoning. 

* At face value, this paper provides strong gains across models, as shown in Table 1, with a relatively simple online policy optimization algorithm. These gains present an interesting avenue to future research, leveraging more robust and sample-efficient RL algorithms and game environment design.

* The automatic evaluation methodology could be refined to address some of its current weaknesses (mentioned below) and serve as a blueprint for evaluating fuzzy LLM statistics in future work.

### Weaknesses
* Major issue: the authors did not report seeded policy optimization, and results do not include mean±STD for critical experiments. This is a must to assess reproducibility and statistical robustness, while results may appear cherry-picked.

* Concerns about SFT dataset quality and size for a fair baseline:
  * Quality: The “expert” is unevaluated. Compare the generator to SPIRAL-trained models and to classic RL agents to establish competence. Report diversity of winning traces (e.g., % unique trajectories vs. coverage seen during SPIRAL training).
  * Size: With GBS=128 and 200 steps (or 2 epochs over 400 steps), the SFT baseline is underpowered versus multi-model self-play data.
  * Example for a fair setup: sample 400×128 trajectories from multiple “expert” models to match self-play scale and diversity.

* Section 4.1, LLM-as-verifier limits qualitative reliability. The paper is also missing key details to evaluate logic and reproduce the results.
  * Reliability issues are known to be exacerbated by long context: F.2. suggests the authors insert multiple full traces into the judge model's context to find patterns, while F.6 suggests the authors used "batch-mode" for classification of multiple traces together. This seems like a loose method for a central claim in the paper without additional guardrails.
Perhaps the authors can mitigate this using the judge to evaluate trace-by-trace, identify distinct reasoning approaches in each trace, then cluster similar patterns together as opposed to the full context approach used in F.2. Finally, reclassify each trace independently to detect the dominant patterns. This approach should allow the authors to report clear, verifiable statistics.
  * F.3 cross-domain transfer: what model sourced the traces for the analysis? If the authors used a SPIRAL model to produce math traces, it is reasonable to assume these traces would be biased to common patterns seen in the game traces (as opposed to observing the reasoning traces of an unrelated strong baseline).
  * How are traces evaluated under sequence-length limits?

* Clarity: Although the method is relatively simple to understand, the paper feels a bit chaotic to read, forcing back and forth between the appendix and different sections of the paper to find and extrapolate critical information.

### Questions
Main:
* Can the authors provide multi-seeded evaluation for Table-1 models (at least partial coverage for one or two models, including SFT and SPIRAL)? 
* Can the authors provide mean and STD using multi-seeded plots in Figs. 5. 6. (and 9.)?
* Can the authors clarify which model sourced reasoning traces for Section 4.1?
* Can the authors add a random policy and a simple RL expert to Tables 4 and 5 to contextualize self-play/Gemini Flash win-rates? 
* Other frameworks exist for large-scale multi-turn trajectory sampling (e.g., Nemo RL). Can the authors explain how this framework is different (since the authors consider their framework part of their main contribution)?
* How well does the SPIRAL model’s reasoning generalize in increased complexity environments (e.g., impact on winrate in 5x5 grid Tic-Tac-Toe, 5 card Kuhn Poker, etc)? Discussing this sort of generalization can help understand how well SPIRAL performs in a smaller controlled domain, before leaping into academic reasoning and math benchmarks.

Secondary:
* Can the authors provide some statistics on the game trajectories during training? (e.g., game length (number of moves), avg reasoning tokens per step, self-play player 1 win-rate, win-rate vs. baseline agent, etc.)
* Please explain how “transferable patterns” were discovered in the main text, or add explicit appendix pointers.
* Please define the τ notation in the text and in Algorithm 1. Also, specify the exact input structure used to train the policy model (and which part of the input tokens gets a gradient during policy optimization)
* Please report GPU hours per model and game/multi-game (reproduction compute budget).
* Can the authors provide some justifications for the choice of games (including OOD games)? Also, justify the chosen training regime?

### Soundness
2

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
4

### Summary
This paper presents SPIRAL, a self-play reinforcement learning framework where LLMs learn reasoning by playing multi-turn zero-sum games against themselves. It introduces Role-conditioned Advantage Estimation (RAE) to stabilize multi-agent training. Experiment results show that SPIRAL improves models' performance on reasoning benchmarks like math and reasoning.

### Strengths
1. Solid idea and algorithm design: Using self-play to construct an automatic curriculum for improving LLMs’ reasoning ability is a reasonable idea, and the proposed multi-turn, multi-agent RL framework is a novel approach.
2. Good empirical results: The method is thoroughly evaluated across multiple models and benchmarks, showing consistent and meaningful performance improvements.
3. Good clarity and presentation: The manuscript is well written and easy to follow. Figures and tables effectively illustrate both the proposed method and the experimental outcomes.

### Weaknesses
1. The second contribution (RAE) is relatively weak: Normalizing advantages separately for different agents in multi-agent settings is a common practice in classical MARL [1, 2]. Applying it to LLM-based multi-agent systems is quite straightforward, which makes the second contribution less substantial.
2. Limited performance on instruct models: In Table 1, SPIRAL shows a significant improvement on base models but much smaller gains (1–2%) on instruct models. This raises the question of whether the improvement on base models truly comes from enhanced reasoning ability or merely better instruction-following.
3. Missing related work: For LLMs in gaming, the authors do not discuss closely related work on LLM agents in multi-agent games such as Diplomacy [3], Werewolf [4, 5], and Avalon [6]. Some of these works also use self-play with RL to train agents, and should be referenced.

[1] Lowe, Ryan, et al. “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.” arXiv preprint arXiv:1706.02275 (2017).

[2] Yu, Chao, et al. “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.” Advances in Neural Information Processing Systems (2022).

[3] Bakhtin, Anton, et al. “Human-Level Play in the Game of Diplomacy by Combining Language Models with Strategic Reasoning.” Science 378.6624 (2022): eade9097.

[4] Xu, Yuzhuang, et al. “Exploring Large Language Models for Communication Games: An Empirical Study on Werewolf.” arXiv preprint arXiv:2309.04658 (2023).

[5] Xu, Zelai, et al. “Language Agents with Reinforcement Learning for Strategic Play in the Werewolf Game.” arXiv preprint arXiv:2310.18940 (2023).

[6] Wang, Shenzhi, et al. “Avalon’s Game of Thoughts: Battle Against Deception through Recursive Contemplation.” arXiv preprint arXiv:2310.01320 (2023).

### Questions
compare them with the corresponding base model, to better analyze how much of the improvement on base models comes from genuine reasoning enhancement?
2. Please analyze why training on games improves QA benchmarks. The paper provides detailed analysis on why SPIRAL enhances performance on math benchmarks, but the improvement on QA seems less straightforward and lacks detailed explanation.
3. During training, the base model cannot directly use the <think> special token, but the pattern analysis section mentions the “thinking token.” How was this implemented in practice—was it through CoT prompting or another mechanism?

### Soundness
3

### Presentation
3

### Contribution
3
