# ResT: Reshaping Token-Level Policy Gradients for Tool-Use Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 4, 8

## Abstract
Large language models (LLMs) transcend passive generation and act as goal-directed agents by invoking external tools. Reinforcement learning (RL) offers a principled framework for optimizing these emergent tool-use policies, yet the prevailing paradigm relies exclusively on sparse outcome rewards and lacks consideration of the particularity of tool-use tasks, inflating policy-gradient variance and resulting in inefficient training.
To better understand and address these challenges, we first establish a theoretical link between policy entropy and training stability of tool-use tasks, which reveals that structured, low-entropy tokens are primary determinants of rewards. Motivated by this insight, we propose Reshaped Token-level policy gradients (ResT) for tool-use tasks. ResT reshapes the policy gradient through entropy-informed token reweighting, progressively upweighting reasoning tokens as training proceeds. This scheme enables a smooth shift from structural correctness to semantic reasoning and stabilizes convergence in multi-turn tool-use tasks. Evaluation on BFCL and API-Bank shows that ResT outperforms other strong baselines, outperforming prior methods by up to 8.76%. When fine-tuned on a 4B base LLM, ResT further surpasses GPT-4o by 4.11% on single-turn tasks and 1.50% on multi-turn base tasks. Code is available at https://github.com/1229095296/ResT_Tool_use_LLM.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work is a modification on GRPO that reshapes token-level policy gradients using entropy-aware weights computed over regions of the response that are of interest to a tool call, such as format tags, tool names, parameters, chain-of-thought, so that so lower-entropy, reward-critical tokens get higher weight. This is combined with a curriculum that increases weights on reasoning/parameter tokens as training progresses, smoothly shifting focus from structural correctness to sophistication of reasoning. Building on top of veRL, this technique achieves achieves state-of-the-art on BFCL and API-Bank.

### Strengths
* The work is reasonably well written with no major grammatical or formatting errors. The schematics are easy to understand and the paper is easy to read. 
* The work theoretically establishes a principled connection between token-level entropy and gradient variance which grounds the approach well. I've checked the proofs roughly and they appear correct
* The problem is important (tool-call agents are of great interest in the community, and sparse reward is an on-going problem) and the essential ideas intuitively make sense (we need to get the calling conventions right before focusing on argument semantics)

### Weaknesses
The experimental results, though extensive, does not unequivocally establish ResT as a state of the art. It's unclear whether the accuracy gap falls within the variance or not (https://www.anthropic.com/research/statistical-approach-to-model-evals), and I cannot find an authoritative benchmark on either BFCL multi-turn or API-Bank that can support the SOTA claim. The results do look strong, just not conclusively strong.

### Questions
veRL is a complex codebase. I would appreciate the authors highlighting the changes they have made branching off of v0.5 to facilitate future replication and adoption efforts.

### Soundness
3

### Presentation
3

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
This paper introduces ResT (Reshaped Token-level policy gradients), a reinforcement learning approach to improve LLM tool-use capabilities by addressing the high-variance credit assignment problem in multi-turn tasks. The authors observe that in tool-using scenarios, success hinges on generating a few critical structured tokens. ResT decomposes a multi-turn interaction into single-turn steps and redistributes reward feedback at the token level, with an emphasis on low-entropy tokens that determine task success. ResT therefore reweights the policy gradient to focus on these high-signal tokens, and employs a lightweight curriculum: early training strongly rewards structural correctness (format, tool invocation tokens), then gradually up-weights semantic reasoning tokens as the agent improves. Experiments on the BFCL and API-Bank benchmarks demonstrate state-of-the-art results.

### Strengths
1. A clear identification of the high-variance reward problem in tool-use and a principled solution grounded in theory.
2. A method that is relatively simple to implement, no additional model modules, just weighting of gradient contributions, yet effective, which is a practical advantage.

### Weaknesses
1. Missing discussion of token-level feedback and curriculum RL literature, which raises concerns that the contribution might be incremental relative to those.
2. The approach is somewhat domain-specific in its current form: it relies on known ground truth outputs to compute rewards for each token category, which may not directly extend to tasks without a deterministic reference solution.
3. While the theoretical results are solid, the practical benefit of the optimal weighting vs. simpler heuristics (e.g. just heavily reward tool tokens) could be discussed more; the authors did compare to a static weighting baseline and showed improvement.

### Questions
1. By design, ResT upweights low-entropy tokens (tool names, parameters) which makes the policy more deterministic on those critical parts early on. While this improves short-term stability, is there a risk of reduced exploration or diversity in the model’s behavior? For instance, could focusing too much on format and tool tokens lead the agent to ignore alternative valid strategies or outputs (especially if the initial reference is suboptimal)? The current experiments show longer outputs and no collapse, which is encouraging. Still, could the authors clarify how ResT balances exploration vs. exploitation? Was any entropy bonus or regularization used alongside the token reweighting (common in PPO-based methods), or any schedule to anneal the focus to avoid overly greedy policies?
2. The method uniquely includes the model’s chain-of-thought (reasoning) tokens in the training signal, gradually increasing their weight later in training. This is an interesting design choice – essentially encouraging the model to not only get the final tool API call correct, but also to produce the intermediate reasoning. Could the authors provide more intuition or evidence on why weighting the reasoning tokens helps? The ablation, turning off CoT gradients, shows a performance drop, confirming its utility. Is the idea that by training the model to articulate its reasoning, even though those tokens don’t directly affect the environment, we guide it to a better policy for the tool calls? If so, might this relate to known benefits of chain-of-thought prompting for decomposing tasks?
3. This is practical for tasks like API calls where an exact answer is known, but how would ResT apply to more open-ended tasks or learned rewards?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces an entropy-aware RL (ResT) method to improve tool-use in large language models. Traditional approaches rely on sparse, outcome-based rewards that cause instable and inefficient training. ResT tackles this by linking token entropy to policy-gradient variance, showing that emphasizing low-entropy tokens (like tool names and parameters) stabilizes learning. It reshapes gradients to weight these structured tokens more early on, then progressively increases focus on reasoning tokens through a curriculum-learning schedule. Experiments on benchmarks like BFCL and API-Bank show ResT outperforms prior methods (including GRPO and GPT-4o).

### Strengths
* Strong theoretical foundations: This paper provides a thorough theoretical motivations and proofs to the method, rigorously deriving the relationship between token-level entropy and policy-gradient variance,
* Extensive experiments: This paper provides extensive experiments, including different sizes of models and training methods, which provides a well-established evidence to the effectiveness of the method.
* Innovation methods by considering different types of tokens, providing a different view of Tool RL.

### Weaknesses
Limitations:
* The paper does not compare SFT+ResT against ResT alone. Since SFT can produce a different token distribution, it may alter the entropy of certain token types. For instance, some function words might exhibit lower entropy after SFT, potentially requiring adjustments to the training curriculum.
* It remains unclear whether larger models inherently generate lower-entropy outputs—particularly for chain-of-thought tokens—which could reduce the benefit of grouping tokens by entropy type.
* Regarding the partition mechanism, the theoretical framework does not constrain how tokens are grouped, as long as distinct entropy characteristics exist. Other grouping strategies, such as those based on word frequency, might also be effective.

### Questions
See weaknesses

### Soundness
4

### Presentation
4

### Contribution
4
