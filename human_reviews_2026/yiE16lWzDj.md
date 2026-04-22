# Nemotron-Research-Tool-N1: Exploring Tool-Using Language Models with Reinforced Reasoning

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Enabling large language models with external tools has become a pivotal strategy for extending their functionality beyond text space. To enhance LLMs' tool-calling abilities, previous approaches primarily rely on supervised fine-tuning (SFT) with trajectories distilled from stronger models, often resulting in imitative reasoning that limits generalization. In this work, we explore rule-based reinforcement learning to enhance tool-calling in LLMs, resulting in Nemotron-Research-Tool-N1, a series of tool-calling reasoning models. Rather than enforcing supervision over intermediate distilled reasoning traces, Tool-N1 is trained with a binary RL reward that assesses only the format validity and functional correctness of tool invocations. This lightweight supervision allows the model to develop reasoning strategies independently, without relying on annotated trajectories. Experiments on several major benchmarks show that Tool-N1-7B/14B clearly outperform GPT-4o. We conduct a systematic study on the design of rule-based reinforcement learning strategies for training tool-calling models. Using 5,518 distilled reasoning trajectories, we compare SFT, RL, and the SFT-then-RL pipeline, finding that the widely adopted SFT-then-RL paradigm does not necessarily outperform pure RL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a family of open-source tool-calling LLMs trained via rule-based reinforcement learning (R1-style RL) rather than conventional supervised fine-tuning (SFT). The key innovation lies in using a binary reward that verifies only output format and tool-call correctness, omitting supervision on reasoning trajectories. Besides the main experiments, the authors further conduct systematic ablations, finding that pure RL outperforms SFT-then-RL, that binary rewards outperform fine-grained ones, and that structured reasoning tags (<think>, <tool_call>) are essential for generalization.

### Strengths
1. This paper demonstrates that pure rule-based RL can outperform traditional SFT and SFT+RL pipelines for tool-use, which is an important insight contradicting existing practice.

2. Tool-N1 models surpass GPT-4o and other closed- and open-source tool models (e.g., Hammer2.1, xLAM-2) on all tested benchmarks, validating the method’s effectiveness.

2. This paper includes ablations on reward granularity, data composition, model scaling, and backbone choice, offering valuable design guidance for future RL-based tool-use research.

### Weaknesses
1. The paper’s contribution is largely empirical; it lacks theoretical insight into why rule-based RL generalizes better than SFT or how binary rewards shape reasoning policies.

2. A binary {0,1} reward could cause credit assignment inefficiency or instability, yet no analysis or mitigation (e.g., shaping, curriculum) is discussed.

### Questions
No extra question. See weakness above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents Nemotron-Research-Tool-N1 (Tool-N1), a series of tool-calling language models trained using a rule-based reinforcement learning (RL) paradigm. The core contribution is the application of a "R1-style" RL approach, which uses a simple binary reward to judge the format and functional correctness of tool calls, without supervising the intermediate reasoning steps. The authors conduct a systematic empirical study, demonstrating that their method outperforms strong baselines, and intriguingly, that a pure RL training pipeline can surpass the commonly adopted "SFT-then-RL" paradigm in this domain.

### Strengths
- The paper effectively argues that supervised fine-tuning (SFT) on distilled reasoning traces can lead to "imitative" or "pseudo" reasoning. The proposed rule-based RL method is a compelling alternative that incentivizes the model to develop its own internal reasoning strategies, promoting better generalization. The motivation is clear and grounded in current limitations of the field.
- Strong Empirical Evidence.
- Rigorous and Insightful Ablation Studies.

### Weaknesses
- There is no computational effiency analysis. How fast is it running compared to baselines? Especially, how the speed becomes when the number of candidate tools becomes large? (since LLM receives the textual input with all the tools included)
- There should be some brief introductions to recent tool-calling baselines, either in Section 4.1 or a Related Work section.
- There is no public code or models which makes the experiment replication difficult.
- There is no explicit paragraph of major contributions in Introduction.

### Questions
- How the name "NEMOTRON-RESEARCH-TOOL-N1" is determined? There seems to be no evident hint in the paper. Considering to use a shorter name instead?
- How is the Reward Model trained? There should be some explicit introduction (even simply refering to some popular methods such as Bladley-Terry) to make the methodology self-contained.
- The training is conducted by GRPO. How would the performance change if using PPO instead? Is there any insights compared these two RL methods?

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
The authors propose a rule-based reinforcement learning method to enhance the tool calling abilities of LLM based on GRPO. Their method uses a binary reward function with light supervision, e.g. it only checks functional correctness and format validity. They claim that it improves the generalization abilities vs conducting SFT (no experiments are provided to back up such claim). The authors present results that their method improves performances by conducting evaluation using a variety of tool calling benchmarks: BFCL, ACEBench and API bank. The authors also present a thorough set of ablations to validate their design choices.

### Strengths
- The authors present a very thorough set of ablations to validate their design choices: 
- SFT vs RL vs SFT+RL, 
- Scaling laws that showcase that using a more powerful model is better and good generalization accross different LLMs as backbones
- Ablation on reasoning format vs not using it
- Ablation on binary vs more fine grained reward function
- Ablation on data decomposition

- The paper is well written and easy to follow

### Weaknesses
- Novelty and originality: It is hard to assess the overall contribution since no other baselines are presented of works that provide ways to interweave reasoning with tool calling.

### Questions
- What percentage of data gets filtered out when you remove the samples with invalid tool calls?
- You claim throughout the paper that it improves reasoning and doesn't lead to pseudo-reasoning. However, there is no experiment to back up this claim. How can you show case that this is indeed the case experimentally?
- The QWEN model used is outdated, it would be good to present some results with more recent models.

### Soundness
3

### Presentation
3

### Contribution
3
