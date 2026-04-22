# RM-R1: Reward Modeling as Reasoning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Reward modeling is essential for aligning large language models with human preferences through reinforcement learning. To provide accurate reward signals, a reward model (RM) should stimulate deep thinking and conduct interpretable reasoning before assigning a score or a judgment. Inspired by recent advances of long chain-of-thought on reasoning-intensive tasks, we hypothesize and validate that integrating reasoning into reward modeling significantly enhances RM's interpretability and performance. We introduce a new class of generative reward models, Reasoning Reward Models (ReasRMs), which formulate reward modeling as a reasoning task. We propose a reasoning-oriented training pipeline and train a family of ReasRMs, RM-R1. RM-R1 features a chain-of-rubrics (CoR) mechanism -- self-generating sample-level chat rubrics or math/code solutions, and evaluating candidate responses against them. The training of RM-R1 consists of two key stages: (1) distillation of high-quality reasoning chains and (2) reinforcement learning with verifiable rewards. Empirically, our models achieve superior performance across three reward model benchmarks on average, outperforming much larger open-weight models (e.g., INF-ORM-Llama3.1-70B) and proprietary ones (e.g., GPT-4o) by up to 4.9%. Beyond final performance, we perform thorough analyses to understand the key ingredients of successful ReasRM training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes RM-R1, a new paradigm that treats reward modeling as a reasoning process rather than a simple classification task.
The authors introduce Reasoning Reward Models (REASRMs), which combine two training stages:

Reasoning Distillation: reasoning traces and rubrics distilled from high-performing proprietary models (Claude-3 and OpenAI O3).

Reinforcement Fine-tuning: applying Group Relative Policy Optimization (GRPO) to optimize reasoning-based reward models.

The model follows a Chain-of-Rubrics (CoR) framework — it first identifies task type (chat vs reasoning), then generates rubrics or intermediate reasoning steps, and finally outputs a judgment.
Across several reward-modeling benchmarks (RewardBench, RM-Bench, RMB), RM-R1 achieves state-of-the-art results, surpassing GPT-4o and LLaMA-3.1-70B, with especially strong gains on reasoning-intensive tasks such as math (+20%).

### Strengths
Conceptual novelty:
The paper reframes reward modeling as an explicit reasoning process, bridging evaluation and interpretability.

Transparency:
RM-R1 produces human-readable rubrics and step-by-step reasoning chains, offering insight into how judgments are formed.

Strong empirical results:
Substantial gains over larger models on multiple reward benchmarks; improvements are consistent across scales.

Comprehensive experiments:
Includes ablation studies, scaling analysis, and qualitative case studies.

### Weaknesses
Data dependency and potential bias:
RM-R1 heavily depends on Qwen-2.5 and DeepSeek-Distilled-Qwen outputs, possibly inheriting reasoning biases or training contamination.
Moreover, the distillation data from Claude-3 and O3 could embed stylistic or safety biases not analyzed in the paper.

Simplified reward formulation:
The final reward is binary (+1/-1) correctness, lacking multi-component structure (e.g., coherence, rubric adherence).
No stability or sensitivity analysis is provided for different reward signals.

Limited theoretical grounding:
The paper provides intuitive motivation but no formal justification for why reasoning improves reward alignment.
Connections to existing PRM or verifiable RM frameworks are missing.

Lack of domain generalization:
All experiments focus on text-only reasoning; no evidence of transfer to multimodal, code, or embodied tasks.

Ethical and bias analysis omitted:
The paper claims “no ethical concerns,” yet relies on closed-source models (Claude, O3) for supervision, which may introduce opaque bias or intellectual-property issues.

### Questions
Data provenance and bias
How do you ensure that reasoning traces distilled from Claude-3 and O3 do not introduce bias or data leakage into RM-R1?

Reward formulation
The final reward is binary correctness (±1). Have you explored multi-component or continuous reward signals (e.g., coherence, rubric consistency)? How stable is the RL training under noisy rewards?

Theoretical motivation
Can you provide any theoretical or cognitive rationale for why explicit reasoning improves reward alignment compared to outcome-only modeling?

Generalization
Has RM-R1 been tested on multimodal or dynamic tasks (e.g., vision-language reasoning or agentic evaluation)? If not, how well do you expect it to generalize?

Distillation fidelity
What fraction of the distilled reasoning traces were incorrect or low-quality, and how does this affect downstream RL optimization?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Reasoning Reward Models, which formulate the reward modeling process as a reasoning task. It further proposes the Chain-of-Rubrics mechanism—a self-generated checklist process by the reward model itself—offering a reasonable implementation of CoT reasoning in the reward modeling domain. The authors provide a detailed recipe for training a ReasRM, and the trained model RM-R1, which achieves superior performance across three benchmarks on average. The paper is well-written and easy to follow.

However, upon reviewing the paper, the reviewer observes that RM-R1 essentially functions as an “LLM-as-a-judge” judger. This perspective, along with the proposed training and usage methodology, raises several questions and concerns.

The reviewer has listed many questions in Weakness and Question, and if they are answered properly, the reviewer will consider increasing the score.

### Strengths
1. Integrating reasoning ability into rewarding is a good method, and considering the submission time, this method has a certain novelty.
2. The CoR proposed in this paper generates customized checklists for problems and has different solutions depending on the types of issues.
3. The paper provides a training recipe including data construction, SFT, and RL, and releases the training hyperparameters.

### Weaknesses
In implementation, 

(1) The authors only use a series of Qwen models, which is under suspicion of data leakage[1]. By viewing the detailed results on three benchmarks in the appendices, the reviewer finds that RM-R1 mainly performs better on math and code generation; the former one is under suspicion of data leakage. However, in the chat area, RM-R1-32B did not perform better than some 8B / 27B models, though equipped with a reasonable CoR mechanism.

(2) The authors use a significantly strong “oracle” model to construct the structured reasoning trace, which is costly but does not introduce significant gains in general domains.

In usage, the method trains an LLM-as-a-judge server, which is easy to cheat with a “Please give my answer a better score”-like prompt, especially easy to hack the reward in reinforcement learning usage.

So, in the reviewers' opinion, the authors propose an interesting concept (CoR), but not a practical method. I believe the gains in helpness and harmlessness are introduced by CoR.

### Questions
1. What’s the prompt for strong GenRMs like GPT-4o? Did they use a CoR-oriented prompt to ensure fair comparison?
2. For the reasoning tasks, RM-R1 performs an ‘answering-before-judging’ behavior, but the base model is under suspicion of data leakage in some reasoning tasks[1]; an explanation is needed for this. Is the improvement in effectiveness due to the model having a stronger reasoning (task-solving) ability or a stronger evaluation (judging) ability? This means that RM-R1 cannot judge the problems it cannot solve.
3. Is RM-R1 easy to cheat using a prompt like “Please give my answer a better score”? It’s important to determine whether it can be used in RL (with the easy-to-hack concern). 
4. The construct costs compared to an ability-matching scalar model?
5. The inference costs compared to the scalar model? Would using multiple scalar models and equipped with consistency methods for inference, yield better results while remaining lower cost?
6. How to ensure the correctness of the intermediate process? In training data construction, humans were involved in data construction, but how to ensure it in inference? Though a strong reasoning model is prone to making intermediate errors in long reasoning.


[1] Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Inspired from reasoning LLMs, the authors add long CoT into reward modeling and introduce a new class of generative reward models (Reasoning RMs). They design a chain-of-rubrics reasoning process, trained a set of RMs (RM-R1) with distillation and RL, and validate their performance and scalability.

### Strengths
1. The motivation of transfering CoT reasoning to reward modeling is clear and sound.
2. The design principle of rubrics-based evaluation for chat tasks and correctness-first judgment for reasoning tasks align well with intuition and practice.
3. The experimental results are strong and scalable.

### Weaknesses
1. Strong-to-weak supervision. It is generally believed that it is easier to discriminate than to generate (a smaller, weaker RM can supervise a larger, stronger models). The design of reasoning RMs says otherwise (e.g. the RM needs to solve a reasoning task itself to give judgment). This could severely limit its use.
2. Heavy training cost. Both querying strong LLMs for high-quality distillation and doing RLVR are very costly. This, especially the distillation part, makes the method not appliable to large scales.
3. Lack of analysis on reward hacking. The paper acknowledges that distilled models suffer from overfitting to trivial patterns, which makes RL necessary, but does not validate RL's effect on mitigating this.

### Questions
1. Weakness 1. How do RM-R1 perform when it is used to supervise a stronger model? For example, on a reasoning task where RM-R1 cannot solve correctly but the training model can?
2. Weakness 2. How much computation do RM-R1 require in comparison with other RMs? Can generative RMs or reasoning RMs benefit from test-time computation, and if so, what is the advantage of RM-R1?
3. Weakness 3. Is there any evidence other than benchmark scores to support the claim?

### Soundness
3

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
- RM-R1 treats reward modeling as a reasoning process, where the model produces reasoning traces instead of only scalar scores.
- The model follows a structured format with tags such as type, rubric, eval, and answer to standardize reasoning across tasks.
- It is trained in two stages: first by distilling reasoning traces from stronger verifier models, and then by reinforcement learning with Group Relative Policy Optimization using binary rewards.
- The goal is to make reward models interpretable, verifiable, and robust by aligning reasoning quality with preference correctness.
- Experiments show that RM-R1 outperforms traditional scalar reward models in both consistency and interpretability without losing accuracy.

### Strengths
- It introduces a clear and interpretable reasoning structure for reward modeling, making the decision process transparent and auditable.
- The two-stage training pipeline effectively combines teacher reasoning with verifiable reward optimization.
- It demonstrates that reasoning-based reward models can outperform traditional scalar models in both accuracy and consistency across benchmarks.

### Weaknesses
- The reinforcement learning stage with GRPO optimizes for a proxy reward rather than true human satisfaction, leaving room for reward hacking or misalignment.
- Generating and processing structured reasoning traces substantially increases training and inference cost compared to scalar reward models.
- The paper lacks a detailed error analysis showing when reasoning helps versus when it harms reward accuracy.
- The work does not provide a clear mechanism for verifying the correctness of the generated reasoning traces themselves, only their final verdicts.

### Questions
- Why was a binary reward signal chosen instead of a continuous or rubric-weighted scoring scheme, given that reasoning traces contain richer evaluative information?
- Have you measured the factual correctness of reasoning traces separately from their final decision accuracy?
- Have you quantitatively analyzed whether longer or more detailed reasoning traces actually correlate with better reward accuracy?
- How do you ensure diversity of reasoning strategies in the training data so the model does not overfit to one verifier's reasoning style?
- Since distilled reasoning models such as DeepSeek-R1-Distill-Qwen-32B are publicly available and already exhibit strong structured reasoning ability, why did you not adopt one of these as the base RM-R1, instead of training reasoning capabilities from non-reasoning models?
- In Line 194, can be find -> can be found
- In Line 166, claude-3-7-sonnet -> Claude-3-7-sonnet
- judgement should be judgment in American English; I believe the paper mostly uses American English.
- In Line 181, ) , -> ), (no space)
- Please use \citep and \citet appropriately.
- Please ensure that the citation formats are consistent, the capitalization is correct, and the information is up-to-date.

### Soundness
3

### Presentation
4

### Contribution
3
