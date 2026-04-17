# RuleEdit: Benchmarking Rule-Level Knowledge Editing in Large Language Models

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Knowledge editing seeks to update language models without full retraining. While most prior work focuses on isolated factual or instance-level edits, we explore a more structured domain: mathematical rules. We introduce \textbf{RULE-EDIT}, the first benchmark explicitly designed for editing and evaluating rule-level abstract knowledge in LLMs. Beyond measuring direct edit accuracy, our benchmark is designed to encourage deeper investigation into the interpretability and symbolic reasoning capabilities of LLMs: (1) To what extent do edits to abstract rules propagate to derived instances? and (2) How well do token-level updates align with higher-level symbolic structures across formats? To evaluate this, we propose two new metrics:\emph{Instance Portability} and \emph{Rule Understanding} that quantify whether edits correctly generalize to rule-governed examples and maintain consistency across symbolic and natural language representations. Through experiments on best-performing open-source LLMs using representative editing methods, we find that while models can often overwrite formula-level knowledge, they frequently struggle to propagate these edits to rule-derived instances and to maintain consistency across different forms of a rule.  For example, several methods achieve nearly 100\% reliability on direct rule queries, yet their rule-specific scores remain unsatisfactory (Instance Portability never exceeds 52\% and Rule Understanding stays below 26\%).  Our findings highlight the limits of current editing methods and motivate rule editing as a testbed for controllable knowledge in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces RuleEdit which is a new benchmark that evaluates rule-level knowledge editing in LLMs. Unlike majority of prior work that focus on factual or instance level updates, RuleEdit focuses on mathematical rule based edits. Authors also propose two new metrics suitable for rule based edits, namely Instance Portability and Rule Understanding. Lastly, authors use this benchmark and perform experiments using various models.

### Strengths
1. The problem is novel and interesting. It introduces a new perspective on knowledge editing.
2. This work can open new windows for newer work that can build upon this work.
3. The paper is clearly written and easy to follow.
4. Authors considered various model editing approaches in their experiments.

### Weaknesses
1. The benchmark lacks realism. It would have been better if authors could have designed the benchmark in a more realistic setup. In mathematical setting, rules are not changed, so this makes the setup not realistic. It would have been better if authors designed the benchmark for more realistic setups.
2. The benchmark was really small which makes one doubt the statistical significance of the results.
3. The scope/domain which benchmark covers is really small (e.g., Euclidean geometry and only focusing on fundamental geometric rules).
4. Since the data is not realistic, it makes me doubt about the practicality, usefulness, and feasibility of the work. Perhaps if the benchmark was more realistic all these doubts could have been cleared.
5. it would have been nice if authors could also comment about more closed sourced and powerful models.
6. The evaluations are based on automatic evaluators using DeepSeek. It would have been better if some human verification was done on the results.
7. The data was generated synthetically and then human verified. This might make the data not diverse and rigorous enough.

### Questions
Can you think about a more realistic setup where rule-edits might be found beneficial and applicable to?

### Soundness
2

### Presentation
2

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
This paper propose a new benchmark to explicitly test how good editing methods are in injecting rule-level abstract knowledge. They find that while models can often overwrite formula-level knowledge, they frequently struggle to *propagate* these edits to rule-derived instances and to maintain consistency across different forms of a rule

### Strengths
* This benchmark target an important problem in knowledge editing --- propagation. The perspective of having a common rule is interesting.

* The work finds an important observation that exiting knowledge editing techniques can mostly regurgitate what's injected rather than generalize the knowledge across instances/query, similarly in prior work [1].

* Problem is proposing a f


[1] PropMEND: Hypernetworks for Knowledge Propagation in LLMs

### Weaknesses
* Failure at propagation is more of an established observation from prior work [1, 2]. No method is proposed in the work to resolve the problem. Or insights for how to propose a better method.

* The setting is not realistic: 1. what if there's conflicting rules; 2. multi-edit setting


[1] Evaluating the Ripple Effects of Knowledge Editing in Language Models

[2] CodeUpdateArena: Benchmarking Knowledge Editing on API Updates

### Questions
* How does the author think of the difference from prior work [1]?

* Choosing math rules feels hard, where the model knows math pretty well. To some extent, the model seems done to fail --- a few gradient descent from editing methods seems hard to overwrite what the model learns from pretraining. 


[1] CodeUpdateArena: Benchmarking Knowledge Editing on API Updates

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
This paper introduces RULE-EDIT, a benchmark for evaluating rule-level knowledge editing in large language models (LLMs), with a focus on mathematical rules such as geometric formulas. The authors argue that while existing benchmarks like CounterFact or ConceptEdit mostly target instance-level factual edits, rule-level knowledge, more abstract, generalizable, and interpretable, remains underexplored.

They propose two new metrics: Instance Portability (IP): measures whether edits propagate to rule-derived instances. Rule Understanding (RU): measures cross-view consistency between symbolic and natural language representations.

Experiments across five editing methods (LoRA, ROME, MEMIT, GRACE, PROMPT) and four LLMs (GPT-J, LLaMA-3, Qwen2, Qwen2.5) reveal that existing methods achieve high edit reliability but poor generalization, locality, and symbolic consistency. The paper concludes that current editing techniques mostly perform surface overwriting rather than rule internalization.

### Strengths
The idea of studying rule-level editing is intuitively appealing and helps connect factual editing with symbolic reasoning.

The work contributes to a more interpretable, structured view of knowledge editing, a direction of increasing importance for controllable and reliable model updates.

### Weaknesses
The key findings that edited knowledge fails to generalize, that locality collapses, and that edits do not propagate coherently are not new. Prior work in knowledge editing (e.g., ROME, MEMIT, GRACE, EasyEdit studies) has repeatedly reported the same issues: distributed representations prevent clean locality, and disjoint parameter subspaces block propagation.
This paper re-observes these phenomena in the domain of geometric rules, but does not analyze whether rule-level editing fails for the same or different reasons. As a result, the study feels confirmatory rather than revealing new mechanisms.

The paper lacks deeper analysis explaining why rule edits fail.
There is no representational probing, causal tracing, or neuron-level localization to connect rule propagation failure with parameter entanglement or symbolic abstraction.
Without such insight, the work remains descriptive ("models fail to internalize") rather than diagnostic ("models fail because knowledge subspaces are orthogonal / overlapping").

Connections to known reasoning phenomena such as the "reverse curse" where models can answer A→B but not B→A would have strengthened the interpretation, showing that symbolic generalization failures extend consistently across both inference and editing.

The abstract poses two questions "(1) how edits propagate" and "(2) how token-level updates align with symbolic structures", which are conceptually redundant. Both address the same propagation-consistency problem, and should be reframed as orthogonal dimensions (e.g., vertical propagation vs. horizontal alignment).
The main text also contains substantial repetition. Phrases like “models act as surface-level overwriting mechanisms rather than internalizing rules” appear multiple times. The paper could be significantly condensed for higher information density and clearer logical flow.

Overall, the experiments demonstrate breadth but limited depth or interpretability. The results largely restate known patterns ("LoRA overfits", "GRACE memorizes", "PROMPT generalizes weakly") without advancing understanding.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2
