# ReJump: A Tree-Jump Representation for Analyzing and Improving LLM Reasoning

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Large Language Models (LLMs) typically reason via Chain-of-Thought (CoT) prompting or explicit training. Though many LLMs achieve similar accuracy on challenging tasks, such as math problem solving and programming, how their underlying reasoning "algorithms" compare remains poorly understood. To investigate this, we propose **ReJump**, which represents a reasoning trace as a visitation order over nodes in a tree of intermediate problem-solving steps. ReJump allows *tree jumps*, non-adjacent transitions between nodes that capture reasoning behaviors such as backtracking, verification, and calculation. This representation enables analyzing LLM reasoning with diverse and intuitive metrics that capture exploration, exploitation, overthinking, forgetting, and verification. We apply ReJump to analyze state-of-the-art Large Reasoning Models (LRMs), which are LLMs explicitly trained for long-form CoTs, and find that models with comparable final accuracy can nonetheless display distinct reasoning behaviors. We further compare distilled LRMs with their teachers, CoT-prompted LLMs with LRMs, and investigate how reasoning examples influence reasoning behavior.  Finally, we show that ReJump can improve reasoning quality at test time through strategies such as ReJump-guided Best-of-N selection and prompt selection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ReJump, a framework designed to analyze and enhance LLM)reasoning by transforming generated reasoning traces into a structured two-layer representation: a tree layer that captures the hierarchical organization of partial solutions and dependencies, and a jump layer that traces the sequential flow of reasoning steps. Through this representation, the framework enables quantitative analysis of reasoning behaviors such as exploration and exploitation balance, overthinking, and forgetting, facilitating comparison of reasoning dynamics across models, tasks, and settings beyond final accuracy metrics. By revealing reasoning weaknesses independent of task performance, ReJump provides insights for improving model training and inference strategies. Additionally, the framework demonstrates practical utility by improving reasoning quality in applications such as Best-of-N response selection and prompt optimization.

### Strengths
This is a work about a proposed framework for converting LLM-generated reasoning traces. This is an interesting direction for refining the understanding on LLM reasoning. 

It is an interesting design to define both the two types of similarity measurement, covering both the content semiotics and reasoning jump patterns

A series experimental studied were conducted using the proposed framework, and the authors shared insights/findings from the work

### Weaknesses
In the Tree similarity (Sim_T) definition, the authors introduces tree edit distance, a variant of graph edit distance.  How to handle the difference in the nodes, say, corresponding to the newly def

The author should elaborate how to assess the set of metrics proposed for the evaluation. All the proposed metrics look fine, but how to justify they are not overlapping, and jointly cover all key aspects for the evaluation.

It is unclear if the metrics proposed, or the evaluation approach overall, would be sensitive to prompting, for example, if the observed overthinking or increased reasoning would be displayed or not during the LM inference?

The effectiveness of the max flow based approach may be more stronger when the graph. The authors should provide some description/observations on that, so that the value of the work can be better demonstrated.

### Questions
n the Tree similarity (Sim_T) definition, the authors introduces tree edit distance, a variant of graph edit distance.  How to handle the difference in the nodes, say, corresponding to the newly def

The author should elaborate how to assess the set of metrics proposed for the evaluation. All the proposed metrics look fine, but how to justify they are not overlapping, and jointly cover all key aspects for the evaluation.

It is a bit unclear if the metrics proposed, or the evaluation approach overall, would be sensitive to prompting, for example, if the observed overthinking or increased reasoning would be displayed or not during the LM inference?

The effectiveness of the max flow based approach may be more stronger when the graph. The authors should provide some description/observations on that, so that the value of the work can be better demonstrated.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper "ReJump: A Tree-Jump Representation for Analyzing and Improving LLM Reasoning" introduces a novel framework to systematically analyze and compare the reasoning processes of Large Language Models (LLMs), particularly focusing on Large Reasoning Models (LRMs) that generate long-form Chain-of-Thought (CoT) outputs.

At its core, ReJump represents a model's reasoning trace as a tree of intermediate solution steps, augmented with a jump layer that captures non-adjacent transitions—such as backtracking, verification, and recalculation—which are common in human-like reasoning but hard to quantify in raw text. This dual-layer structure enables the extraction of six interpretable metrics: solution count, jump distance (exploration vs. exploitation), success rate, verification rate, overthinking rate, and forgetting rate.

The authors apply ReJump to analyze state-of-the-art LRMs like DeepSeek-R1, QwQ-32B, and Grok 3 Mini Beta across tasks such as MATH-500 and Game of 24. A key finding is that models with similar final accuracy can exhibit fundamentally different reasoning strategies—some favoring broad exploration, others focused exploitation. ReJump also reveals that distilled models inherit reasoning behaviors from their teachers, and that in-context reasoning examples influence action-level behavior more than high-level problem decomposition.
Beyond analysis, ReJump is shown to improve reasoning at test time by enabling Best-of-N selection and prompt selection based on desired reasoning characteristics (e.g., encouraging exploration in Game of 24), without requiring ground-truth labels.

### Strengths
1. The approach is both methodologically novel and conceptually comprehensive. The paper introduces ReJump, a dual-layer representation (tree + jump) that models reasoning both structurally and dynamically, offering a novel and interpretable lens on LLM reasoning. Alongside this, the authors define six behavioral metrics (e.g., jump distance, verification rate, overthinking rate) and two similarity metrics that together form a comprehensive toolkit for quantitatively analyzing exploration–exploitation balance and cognitive behaviors such as verification and forgetting.

2. The experiments are solid, covering multiple reasoning models (DeepSeek-R1, Grok 3 Mini Beta, QwQ-32B, Phi-4-Reasoning-Plus, and Claude 3.7 Sonnet) and datasets (MATH-500 and Game of 24), revealing nuanced differences between models that go beyond mere accuracy.

3. A technically sound enhancement built upon meaningful discoveries, ReJump is not only analytical but also practical — it improves reasoning performance through Best-of-N and prompt selection strategies guided by its own metrics, demonstrating clear and measurable gains without the need for supervision.

### Weaknesses
1. ReJump extraction requires prompting a large LLM (e.g., Gemini 2.5 Pro), leading to high cost and poor scalability for large-scale or real-time analysis.

2. Experiments focus narrowly on mathematical reasoning and arithmetic-style problems; results on commonsense, coding, or multi-hop reasoning would strengthen generality.

3. Task-specific prompt engineering is still needed to define what a “partial solution node” is; automatic adaptation to new domains (e.g., logic, coding) is not yet solved.

### Questions
1. ReJump shows promise as an evaluation and inference-time enhancement tool. Do the authors envision incorporating its metrics as intermediate supervision signals (e.g., via reinforcement learning or reward modeling)?

2. The definitions of overthinking and forgetting are conceptually intuitive, but how are they validated? For example, did human annotators agree with these automatic flags, or were they cross-checked with qualitative examples?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors aim to better understand the underlying reasoning process of current reasoning LLMs by representing the reasoning process as a walk over a graph, where each node represents a partial solution, and each parent node is a prerequisite of the child node. The authors use Gemini 2.5 Pro to parse and analyze textual LLM outputs into this graph + visitation order representation. Based on this, the authors derive metrics that quantify the model's reasoning behavior. Notably, e.g., the authors identify (1) the level of overthinking by counting the number of extra partial solutions identified by the model, *after* deriving the first correct partial solution. Another example is (2) the level of forgetting, which is derived by the number of times the same node is visited by the model in its reasoning.

The authors apply this method to the MATH-500 and Game-of-24 tasks to derive general insights on LLM reasoning behaviors, including the effect of distillation and R1-style RL training, as well as insights on specific reasoning strategies used for each task.

### Strengths
- S1. The graph representation methodology is interesting and the proposed metrics based on this graph representation are intuitive and reasonable.
- S2. The graph representations derived by Gemini 2.5 Pro are verified by human evaluation, and the correlation with human judgement is reasonable.
- S3. Comparison on five main model families and various model variants
- S4. The results quantitatively explain existing insights on *general* reasoning behaviors and show novel *task-specific* insights.

### Weaknesses
- W1. The method is complex and costly, given that it is task specific. Applying this analysis to a novel task requires (1) a set of samples for the given task (the authors mention that they use 70 samples from each task), (2) funds for API costs (the authors mention they used approximately $2000 across all experiments), and (3) hand-crafting new prompts to adapt the methodology according to the task (mentioned in the limitations).
- Regarding the findings on *general* reasoning patterns of reasoning LLMs, highlighted in pages 6, 7, 8:
  - W2. The finding on model behaviors are not too surprising, and the quantification of those behaviors offers limited actionable conclusions. Regarding actionability, the study of direct applications of ReJump as inference time strategies in Section 5 does not sufficiently demonstrate strong real-world potential, given the complexity of the method, and limited investigation (of task performance comparison with baselines, cost analysis, etc.).
  - W3. It is unclear if the proposed method (graph-based analysis) had a key role in deriving these findings. E.g., it could be possible to conduct large scale behavioral analysis by simply employing Gemini 2.5 Pro to identify these behaviors from the reasoning paths directly and calculating statistics. How is the proposed method better than this simpler baseline, as a tool for analysis?

I'm willing to increase my score if the authors can provide *strong* arguments against W2 and W3.

Please note that my assessment differs on the paper's *task-specific* insights vs *general* insights. To clarify my assessment: the former are novel and actionable (S4), but hard to apply on new tasks (W1). The later have limited novelty and not very actionable (W2, W3).

### Questions
Suggestions on presentation:
- I think that reducing the information density of the figures could make them more effective at delivering key insights. Examples:
  - Font sizes could be larger to improve legibility (they are too small)
  - There are too many models in Figure 4, making it hard to read the radar chart, and recognizing the difference in shape between models. The main figure could include the 2-3 top models that achieve similar performance, to better highlight the point that `even when models achieve similar final performance, their underlying reasoning processes can differ significantly`.
  - In Figure 4 and 5, the metric labels could be made bigger, and written in plain English, similar to those used in the caption: Solution Count, Jump Distance, Success Rate, Inv. Forget Rate, Verification Rate, Inv. Overthinking Rate.
  - Figure 6 is too small and dense.

### Soundness
3

### Presentation
3

### Contribution
2
