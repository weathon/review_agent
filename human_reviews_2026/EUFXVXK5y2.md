# FOR-Prompting: From Objection to Revision via an Asymmetric Prompting Protocol

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Reasoning protocols such as Chain of Thought (CoT) and Tree of Thought (ToT) organize internal deliberation but lack an explicit mechanism for external questioning that elicits self-revision. We present FOR-Prompting (From Objection to Revision Prompting), an asymmetric protocol where a Defender proposes an answer, an Debater (Questioner) raises question-style objections with no direct fixes, and a Host optionally synthesizes the final output. Across GSM8K, FOR-Prompting matches the accuracy of CoT and consistently improves over single-prompting when evaluated under identical model backbones. On small-scale open-source models (e.g., LLaMA-3.2-1B), FOR-Prompting yields substantial gains over direct prompting and performs comparably to lightweight reasoning baselines, highlighting its promise for low-resource and on-device settings. Cross-model role-swapping further shows that performance is primarily determined by the Defender, enabling small models to act effectively as Questioners. Beyond structured math tasks, FOR-Prompting supports refinement in open-ended and multi-stage tasks: qualitative analysis shows improved exploration, coverage, and specificity, and a blind human preference study found that participants preferred FOR-Prompting outputs over strong LLM baselines in an itinerary-planning scenario. The protocol is model-agnostic and operates purely through role-structured prompting, requiring no training, access to model internals, or symmetrically strong agents. FOR-Prompting therefore enables scalable study of objection-driven reasoning and offers a practical mechanism for automated iterative refinement across both hosted and local LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces FOR-Prompting, a novel, model-agnostic prompting protocol that structures LLM reasoning as an asymmetric dialogue between a "Defender" agent that proposes answers and a "Debater" agent that raises question-style objections without providing solutions. The core contribution is the formalization of questioning as the sole mechanism for external intervention, which aims to elicit self-revision while preserving a single, accountable line of reasoning.

### Strengths
(1) Conceptual Novelty: The protocol's design, which strictly limits external intervention to questioning alone, is a novel and well-motivated contribution.

(2) Effectiveness on Small Models: The experiments conducted on the LLaMA-3.2:1B model are a key strength.

(3) Demonstrated Utility on Open-Ended Tasks: Case Study 4 provides a compelling qualitative demonstration of the protocol's value beyond factual question-answering.

### Weaknesses
(1) In Case Study 1, the main results on GSM8K compare FOR-Prompting (using gpt-4o) against CoT (using gpt-4o-mini). But since those two setups use different backend models, the comparison is confounded. It’s hard to tell whether the better reasoning and coherence actually come from the FOR-Prompting method itself or just from the stronger model (gpt-4o).

(2) In Case Study 2, the authors only compare FOR-Prompting on the LLaMA-3.2-1B model with a single-prompt baseline. They don’t test against other lightweight reasoning methods—like a CoT version run on the same 1B model—so we can’t really tell how much benefit comes from FOR-Prompting versus other simple prompting tricks.

(3) Section 4 introduces three roles in the framework, including a “Host” that combines the dialogue into a final answer. But there’s no ablation study showing what happens if you remove that Host. Without testing its impact, we can’t see how much that synthesis step actually matters.

(4) Case Study 1’s reasoning and coherence scores all come from GPT-4.1 acting as a judge. That’s fine as a quick heuristic, but these LLM-as-a-judge metrics are subjective and inherit the biases of the evaluating model. They’re weaker evidence than direct, task-based or human-verified performance metrics.

### Questions
please see weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new prompting method named FOR-Prompting, where a Defender and an Objectioner discuss questions centered on the original answer, with a host deciding the closure. The authors show the gain on GSM8K with GPT-4o and Llama 3.2 1b with high scores on the reasoning traces judged by GPT 4.1. The authors also show qualitative analysis on open-ended tasks and highlight the model's effectiveness on local device usage.

### Strengths
1.	The authors provide a comprehensive overview of the relevant literature in Section 3.

2.	The authors show evaluation on reasoning trace besides simple accuracy

3.	Besides the model performance on math reasoning, the authors also consider studying the features of open-ended tasks

### Weaknesses
1.	There are insufficient experiments to support the claims. (1) Only GSM8K is used for the main experiment. The open-ended task is not well-grounded or introduced in detail. The reviewer would suggest that the authors check OLMES (https://github.com/allenai/olmes) to extend the tasks; (2) the model selection is arbitrary. Multiple GPT models are selected (GPT 4/5) without clear reasoning. There is no ablation on whether the method works on other open/closed-source models; (3) Only CoT is considered as a baseline. Other important baselines are missing, e.g., self-ask (https://arxiv.org/abs/2210.03350) and Least-to-Most prompting (https://arxiv.org/abs/2205.10625) 

2.	The proposed method lacks technical novelty. The introduced workflow is mostly covered by multi-agent debate (https://arxiv.org/pdf/2305.14325) or Self-Refine (https://arxiv.org/abs/2303.17651). The empirical performance is also similar to CoT (as in Figure 2), which further limits the effectiveness.

3.	The comparison between FOR-Prompting and Single-prompt can be unfair. The authors should at least show that the number of tokens is in a similar range. If not, methods like self-consistency (https://arxiv.org/abs/2203.11171) should be considered to make the comparison fair.

4.	The evaluation is not well grounded. GPT 4.1 is considered valid to evaluate the solution quality when only GPT models are used in Section 4.1. There should be a consistency check among models or human annotations to support the use of LLM to judge the solution quality.

### Questions
1.	Line 47: What do you mean by external pressure?

2.	Why is automatic HITL considered HITL (Line 64)? This is not grounded if there is no experiment showing that human-Questioner/Defender performance in the proposed framework

3.	In related work, how is your method relevant or different from existing ones, e.g., how is it different from multi-agent debate

4.	Citation problem of ReAct on Line 121

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes FOR-Prompting, an asymmetric prompting protocol designed to improve LLM reasoning via external questioning rather than direct answer correction. The framework defines three roles: Defender, Debater, and Host. The key insight is that asking clarification/adversarial questions—without offering solutions—induces models to refine reasoning while preserving a single accountable chain of thought.

### Strengths
1. Clear conceptual separation between questioning and answer replacement.
2. Preserves single-author chain of reasoning and transparency.
3. Demonstrates large gains for small models, relevant for edge / on-device scenarios.
4. Works on both factual reasoning (math) and open-ended planning tasks.

### Weaknesses
1. Benchmark scope narrow: mainly GSM8K and anecdotal open-ended tasks.
2. Open-ended evaluation lacks systematic human preference studies.
3. CoT baseline backend mismatch raises fairness questions.
4. Limited theoretical exploration of why questioning aids reasoning or convergence behaviors.

### Questions
1. How does FOR-Prompting perform with the same backbone as CoT?
2. Can the method scale to long-horizon planning or multi-stage tasks requiring memory?
3. How sensitive is performance to the quality of the Debater model?
4. Can you consider adding human preference evaluations for open-ended tasks?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces FOR-Prompting, a novel asymmetric prompting protocol designed to improve reasoning in large language models (LLMs) through external questioning rather than answer substitution. The protocol comprises three roles: Defender (proposes answers), Debater (asks external questions), and Host (synthesizes the final output). The paper empirically evaluates FOR-Prompting on several tasks, including the GSM8K math word problems, demonstrating significant improvements in accuracy and reasoning quality. Notably, the approach also benefits small-scale models, achieving up to 19% accuracy gains on the GSM8K task. The protocol is model-agnostic and does not require retraining.

### Strengths
1. Novelty: The protocol's emphasis on external questioning and its role-structured design is a unique approach, distinguishing it from other prompting strategies that focus on self-reflection or debate-based solutions.

2. Empirical Validation: The experiments on GSM8K, including comparisons with single-prompt and CoT baselines, show clear performance improvements, particularly in reasoning quality and coherence.

3. Model-Agnostic: The ability to apply FOR-Prompting across different model sizes without retraining is a significant strength, especially for small-scale models where traditional methods may fail to achieve robust reasoning.

### Weaknesses
1. Complexity in Application: While the approach is promising, the need for a "Debater" to only raise questions without proposing fixes might introduce inefficiencies, especially for tasks requiring quick responses.

2. Generalizability: While the experiments show promise, the method’s applicability in non-mathematical tasks or more complex real-world scenarios (e.g., open-ended creative reasoning) remains unclear. Further case studies would help validate its robustness across diverse domains.

3. Scalability Issues: The need for multiple rounds of questioning, particularly in small-scale models, could lead to token usage and latency issues. The authors mention that the cost overhead is controllable, but this would benefit from deeper exploration, particularly for real-time systems.

### Questions
No

### Soundness
3

### Presentation
3

### Contribution
3
