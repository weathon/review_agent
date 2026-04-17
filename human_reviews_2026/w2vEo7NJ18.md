# Cogito, Ergo Ludo: An Agent that Learns to Play by Reasoning and Planning

- Decision: Reject
- Scores: 4, 4, 6, 8

## Abstract
The pursuit of artificial agents that can learn to master complex environments has led to remarkable successes, yet prevailing deep reinforcement learning methods often rely on immense experience, encoding their knowledge opaquely within neural network weights. We propose a different paradigm, one in which an agent learns to play by reasoning and planning. We introduce Cogito, Ergo Ludo (CEL), a novel agent architecture that leverages a Large Language Model (LLM) to build an explicit, language-based understanding of its environment's mechanics and its own strategy. Starting from a tabula rasa state with no prior knowledge (except action set), CEL operates on a cycle of interaction and reflection. After each episode, the agent analyzes its complete trajectory to perform two concurrent learning processes: Rule Induction, where it refines its explicit model of the environment's dynamics, and Strategy and Playbook Summarization, where it distills experiences into an actionable strategic playbook. We evaluate CEL on diverse grid-world tasks (i.e., Minesweeper, Frozen Lake, and Sokoban), and show that the CEL agent successfully learns to master these games by autonomously discovering their rules and developing effective policies from sparse rewards. Ablation studies confirm that the iterative process is critical for sustained learning. Our work demonstrates a path toward more general and interpretable agents that not only act effectively but also build a transparent and improving model of their world through explicit reasoning on raw experience.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors present Cogito, Ergo Ludo (CEL), an LLM-based learning agent that learns to play by using a structured two-pronged algorithm for reasoning and planning, which involves constructing a semi-abstract world model and a “strategic playbook” – essentially policy advice – out of natural language. The authors show that their algorithm outperforms multiple baseline language models that do not use the proposed algorithm.

### Strengths
The connection to existing RL literature is thoughtful and sensible, with most relevant literature touched on. The writing style is clear and easy to follow. The idea has perhaps been explored lightly before, but is more thoughtfully integrated into RL than most similar works.

### Weaknesses
The improvement of the CEL agent over the Qwen+game rules is modest for some environments. 95% confidence intervals should be plotted, not just average returns. There are unanswered questions about the source of the language descriptions from states.

### Questions
Where do the natural language interpretations of game states come from? Are they provided by the environment? Is there a layer between the environment and the LLM where you engineered natural language observations? If so, it would hardly make the agents *tabula rasa.*

In the Generated-Rule Definition tables in the appendix, are the rules in blue the only rules that are learned? Why are they highlighted?

I would guess that the agent performance has high variance, judging by the plots. Why were these experiments not plotted with a proper confidence interval?

**Initial Recommendation:** Reject

**Reasons:** It’s customary to plot confidence intervals for agent performance so we can see the variance of the learning algorithm, please plot them. Also, please address the questions I pose and I will be happy to revise my recommendation to accept.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Cogito, Ergo Ludo (CEL) , an LLM-centric game-playing agent that learns by reasoning and planning. CEL runs a two-phase loop: during play it performs natural-language lookahead with a Language-based World Model and evaluates options via a Language-based Value Function; after each episode it conducts post-episode reflection to induce/refresh explicit game rules and summarize a strategic playbook, both persisted as text and used in subsequent episodes. Starting tabula rasa (action set only), CEL jointly acts, reasons, and updates its knowledge while also training the LLM parameters (e.g., via GRPO). Evaluated on Minesweeper, Frozen Lake, and Sokoban, CEL discovers rules from interaction, learns effective policies from sparse rewards, and shows generalization beyond seen layouts, pointing to procedure-level transfer.

### Strengths
1. The paper is well-structured and easy to follow: the problem setup, agent loop (in-episode reasoning + post-episode reflection), and training details are clearly delineated.

2. Framing gameplay as learning rules and strategies from interaction via an LLM—then persisting them as explicit, reusable knowledge—is a creative twist on world-model RL. The language-based world/value modules and post-episode “playbook” distillation offer a fresh, interpretable alternative to purely latent approaches.

3. The empirical section demonstrates care for robustness: multiple random seeds, ablations on key components (e.g., rule induction, value guidance), and comparisons against sensible baselines.

4. The environments and interfaces appear lightweight and standardized, making it feasible for the community to replicate results and extend the benchmark (new levels or variants) without heavy engineering overhead.

### Weaknesses
1. The environments (Minesweeper, FrozenLake, Sokoban) are classic and likely known to pretrained LLMs. This weakens the “tabula rasa” claim: the agent may be recalling prior heuristics instead of actually inducing rules online. The paper should make explicit what prior knowledge the model is allowed to use.

2. The method relies on GRPO to train the LLM, but the optimization setup is under-specified (reward shaping, rollout collection, frequency of distillation from reflection, etc.). It’s difficult to attribute where the gains come from or to reproduce the system.

3. There is no imitation-learning / supervised baseline where the LLM is trained on expert demonstrations. Without this, it’s hard to conclude CEL is the most effective way to make LLMs play these games.

4. The games are small and single-agent puzzles. The paper does not convincingly show (or argue in detail) how CEL scales to harder settings. The contribution would be stronger with either an explicit scaling path or a clearer statement that CEL is currently a controlled proof-of-concept rather than a generally applicable agent paradigm.

### Questions
1. What concrete steps ensure the LLM isn’t recalling prior knowledge of Minesweeper/FrozenLake/Sokoban?

2. Could you provide a training diagram + full hyperparameter table and compute budget?

3. How does CEL compare to supervised LLMs trained on a small set of expert demonstrations (BC/DAgger or scripted experts)?

4. To which game classes do you expect CEL to scale, and what modifications are required?

5. Do you expect CEL’s “reason–act–reflect” loop to extend to (a) long-horizon, combinatorial, perfect-information board games like chess and Go, and/or (b) high-bandwidth, partially observed, continuous-control settings like FPS?

6. If CEL cannot straightforwardly handle games like chess/Go or FPS-style environments, how should we interpret its core contribution? Is CEL intended primarily as (i) a self-explaining rule/strategy inducer for small, symbolic environments, or (ii) a generally useful module in a larger agent stack?

7. How would you compose CEL with other methods (tree search, learned motor policies, memory/belief tracking) in a full system?

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
The paper proposes the CEL agent. A small LLM explicitly summarizes environment rules , then performs one-step lookahead with value estimation to decide actions. On text-based grid environments (Minesweeper, FrozenLake, Sokoban) with sparse terminal rewards and no ground-truth rules provided, the authors report steadily improving success rates and ablations suggesting that continual rule induction is critical. The core value is coupling readable, auditable, language-level knowledge with the action policy, offering an interpretable path to “learning by thinking” on small but challenging tasks.

### Strengths
1. Experiments showed that CEL successfully learns to master these tasks
by autonomously discovering their rules and developing effective strategies.
2. Strict evaluation setup (sparse rewards, no ground-truth rules) across three representative grid tasks, with ablations supporting the claim that continual rule induction matters.
3. CEL provide qualitative evidence of the architecture’s unique interpretability, showcasing the comprehensive, human-readable rulebooks and sophisticated strategic heuristics.

### Weaknesses
1. Limited evaluation scope: confined to small, text-based grid environments; external validity to larger/complex tasks, higher stochasticity, and distribution shift is unclear.
2. The paper lacks comparisons with baselines like MuZero、Dreamer and other reinforced search discussed in the related work.
3. No verification for induced rules: the loop relies on induction and reuse; there is no explicit verification mechanism (e.g., executable consistency checks, counterexample-driven falsification, precision/recall against ground-truth rules), so systematic errors may persist. I wonder if this will lead to performance bottlenecks in larger systems.

### Questions
1. Can you Report results on larger/longer-horizon variants and more stochastic settings. 
2.  I wonder if there is a brief failure analysis on hardest cases.
3. Can you add some other baselines like MuZero and Dreamer under matched budgets (steps/compute/tokens) ?
4.  Can you explain why there is no verification for induced rules and whether this will lead to performance bottlenecks in larger systems.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a novel approach to leverage LLMs to learn world models directly in natural language. 
The proposed agent architecture uses a loop of planning and interaction, and world model updates through rule induction from episodic experience.
The authors evaluate their method directly using text-based versions of Minesweeper, Sokoban and Frozen Lake.

### Strengths
- The paper proposes quite an interesting approach to world modeling and planning. The paper is generally well written. 
- The paper proposes a novel way to do model-based learning in language space and leverage LLMs/GRPO to learn interactively.
- The natural interpretable process provides great insight about the abilities the agent learned while building a world model. Authors showed generalization capabilities, and strategy and rule induction directly from interaction and sparse rewards.
- The authors show quantitatively that the agent performance improves significantly over zero-shot approaches and include ablations that show that the reflection/thinking part of their framework in contributing importantly to the performance gain.

### Weaknesses
- In terms of quantitive results, I’m only concerned about the lack of a fine-tuning baseline. Wouldn’t the naive thing be to directly use the reward and GRPO to learn to the play the game? I would expect this to be part of the baselines. 
- The evaluation domains are limited to simple games and I’d be interested to understand how this would scale to more complicated domains. However, the proposed idea, imho, is interesting and novel enough that these domains show potential. 
- However, it would be really interesting to understand how the performance behaves when the game size grows. Typical RL would struggle. I wonder if the generalization capabilities would shine is this case. Is this considered within the 256 instances used for evaluating generalization or just the change of layout?

### Questions
It's very interesting to me the use of natural language to describe a kind of qualitative value function. I wonder if this approach can give us an expressive enough value for more complicated settings? Wouldn't the model be limited by the discrete nature of language?

### Soundness
4

### Presentation
3

### Contribution
4
