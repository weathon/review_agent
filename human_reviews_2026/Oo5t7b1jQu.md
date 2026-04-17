# SituatedThinker: Grounding LLM Reasoning with Real-World through Situated Thinking

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Recent advances in large language models (LLMs) demonstrate their impressive reasoning capabilities. However, the reasoning confined to internal parametric space limits LLMs' access to real-time information and understanding of the physical world. To overcome this constraint, we introduce SituatedThinker, a novel framework that enables LLMs to ground their reasoning in real-world contexts through situated thinking, which adaptively combines both internal knowledge and external information with predefined interfaces. By utilizing reinforcement learning, SituatedThinker incentivizes deliberate reasoning with the real world to acquire information and feedback, allowing LLMs to surpass their knowledge boundaries and enhance reasoning. Experimental results demonstrate significant performance improvements on multi-hop question-answering and mathematical reasoning benchmarks. Furthermore, SituatedThinker demonstrates strong performance on unseen tasks, such as KBQA, TableQA, and text-based games, showcasing the generalizable real-world grounded reasoning capability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SituatedThinner, a novel framework that grounds LLM reasoning in real-world contexts through a paradigm termed "situated thinking." The core idea is to enable LLMs to adaptively interleave internal reasoning steps ("internal actions") with calls to external interfaces ("situated actions") during a single, deliberate reasoning trace. These interfaces provide a unified way to interact with diverse external environments like knowledge bases, code executors, and simulated worlds. The model is trained using a reinforcement learning objective (an adaptation of GRPO) with a simple reward based on final answer correctness, which incentivizes the LLM to autonomously learn when and how to invoke interfaces to acquire information, verify reasoning, and correct errors. Extensive experiments demonstrate significant improvements over strong baselines on multi-hop QA and mathematical reasoning, and, more importantly, show compelling generalization to unseen tasks (KBQA, TableQA) and interfaces (text-based games).

### Strengths
**Novel and Well-Motivated Paradigm**: The concept of "situated thinking" is a meaningful and timely contribution. It moves beyond simply injecting external knowledge (as in RAG) towards enabling a dynamic, interactive reasoning process where the model actively queries its environment. This addresses a key limitation of closed-world, parametric reasoning in modern LLMs.

**Impressive Generalization**: The framework's ability to generalize zero-shot to entirely new domains (MedQA, GPQA), tasks (WebQSP, WTQ), and interfaces (TextWorld) is a major strength. This suggests the model learns a generalizable skill of "tool-use" rather than overfitting to specific tasks seen during training.

**Effective and Simple Training Approach**: The use of RL with only a final-answer reward to successfully teach complex behaviors like interface invocation, reflection, and planning is elegant and effective. The ablation studies convincingly show that the model learns these capabilities without explicit supervision on intermediate steps.

**Rigorous Experimental Setup:** The paper provides comprehensive evaluations across a wide range of benchmarks, comparing against a strong set of baselines (including multi-step RAG and other RL-for-reasoning models). The inclusion of training dynamics analysis and qualitative case studies adds depth to the empirical validation.

### Weaknesses
**Limited Discussion on Relation to RAG**: While the paper positions itself against "tool-calling" and "search-enhanced" models, a more direct and thorough discussion comparing and contrasting the proposed paradigm with classic and multi-step Retrieval-Augmented Generation (RAG) is lacking. A dedicated subsection explicitly analyzing the fundamental differences (e.g., static knowledge injection vs. dynamic, agentic interaction) would significantly sharpen the contribution and make it more accessible to a broader audience.

**Limited Scope of Evaluation**: The current evaluation is confined to the textual modality and the English language. The limitations section rightly notes this, but it remains a weakness of the current empirical evidence. The framework's performance in multimodal settings or with non-English interfaces remains an open and important question.

**Focus on Deterministic Tasks**: The work primarily addresses tasks with definitive, verifiable answers. Its applicability to more open-ended, creative, or non-deterministic reasoning tasks (e.g., essay writing, complex strategic planning) is not explored and represents a significant boundary for the current method.

**Lack of Assessment on General Capabilities**: The idea of employing a single, powerful LLM for situated reasoning, as opposed to constructing a multi-agent framework, is highly appealing, and I appreciate this elegant approach. However, when training one model to handle such a complex and diverse set of reasoning tasks, its generalization and the preservation of core capabilities become paramount. In this context, the paper currently lacks an assessment of whether the RL fine-tuning causes catastrophic forgetting of the model's general capabilities. It remains unclear if performance on standard benchmarks (e.g., MMLU, BBH) or its general instruction-following ability has degraded after the specialized training for situated thinking. An ablation study comparing the base model and the trained SituatedThinner on such general tasks is crucial to ensure that the acquisition of these novel, specialized skills does not come at the cost of the model's broader competency and linguistic fluency.

It's better to discuss the related works of the agent frameworks for reasoning.

### Questions
1) The decision to omit the KL penalty in the GRPO objective is a significant one. To better understand its impact, did you conduct an ablation study on this choice? Specifically, how critical was omitting the KL penalty for achieving the final performance, and what was its effect on the model's behavior and stability during training?

2) Given that the model successfully generalizes to new interfaces, did you observe any cases of "interface confusion" where the model used the wrong interface (e.g., using code execution for a simple fact lookup)? A case study and analysis of such failure modes could be insightful.

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
4

### Summary
This paper proposes “Situated Thinker”, which grounds the reasoning process of long chain-of-thought (CoT) LLMs with results returned from external tools. The LLM system is allowed to invoke tools through “interfaces” while generating a long CoT, with results dynamically inserted into the CoT. Two models of 8B and 32B size are trained and evaluated on multi-hop QA, math, and other out-of-distribution reasoning benchmarks

### Strengths
- The single-turn formulation of situated thinking paradigm is, to my knowledge, novel. This formulation should greatly simplify the RL training pipeline as compared to more complex multi-turn tool-use enhanced reasoning models.
- The resulting models seem to generalize to unseen domains not explicitly covered in the training data.
- The paper is generally well-written and easy to understand.

### Weaknesses
- I find some of the statements about novelty over-claimed:
  - It’s unclear how the claim that situated thinking, which is described as “a new paradigm that allows LLMs to adaptively engage with external environments” in line 91, is different from several LLM-based agents that use retrieval tools, like web-search, e.g., [1,2,3,4], or even baselines ReSearch and Search-R1 (beyond number of tools used). 
  - The exact way tool results are incorporated (directly within a single-turn long CoT) may be only differentiator, but I don’t believe this constitutes a fundamentally new paradigm. Rather, as described in Strengths, I consider this a formulation that has some benefits, e.g., simplified training pipelines
- Some issues with baselines:
  - Situated Thinker is trained from Qwen3 family, but the most direct baselines, Search-R1 and ReSearch are trained from Qwen2.5-7B. Given that Qwen3 are generally stronger models, it’s unclear how much the performance difference stems from base model difference vs. methodological improvements. A more reasonable comparison would be to conduct RL training from the same base model if a methodological comparison is the aim, then demonstrating that such approach can improve the performance of better starting models, as is done with Qwen3 experiments.
  - Why do you compare against Qwen3 base LLMs on math benchmarks, and not the post-trained models that are capable of long CoT? Same question for ReAct baselines for OOD tasks?In particular, most current implementations of ReAct agents use instruction-tuned models capable of better instruction following

[1] https://arxiv.org/abs/2509.06283

[2] https://arxiv.org/abs/2507.02592

[3] https://arxiv.org/abs/2507.15061

[4] https://arxiv.org/abs/2508.13167

### Questions
- Can you provide more details about how retrieval is done for SituatedThinker? What retriever did you use? Is this retriever the same for RAG-based baselines? How does SituatedThinker's ability depend on the quality of retrievals?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the situated thinker, an RL-trained framework that enables large language model chain-of-thought reasoning with predefined tool calls (external interfaces), including retrieval, code execution, and other APIs. The authors use GRPO-style RL training to inject tool call capabilities into smaller language models. Trained on MusiQue and Big-Math, the model reports strong gains on mathematical reasoning datasets (e.g., AIME24/25 and MATH500).

### Strengths
- This paper is well-motivated and well-written. It’s good to show how general tool-use capabilities can be injected into smaller language models via RL to boost reasoning.
- The results on MATH and knowledge-based reasoning datasets seem strong.

### Weaknesses
- Interface design: he authors customized a new interface design for external tool calls. The specifications look highly similar to the Model Context Protocol (MCP). Why not just use MCP, since it’s more universal and enables a wider range of tool use?

- It’s hard to see what actually works. From Table 8, we can see that the tool calls hardly affect model performance on one of the major claimed domains—mathematical reasoning—while on knowledge-based tasks, it’s unclear how retrieval-only methods contribute to the performance. For example, it’s not clear how state-of-the-art retrieval or search-based methods contribute—row 3 in Table 6 might simply show that the proposed situated thinker interface is not effective for retrieval-based tasks.

- Novelty and broader implications: Marrying language models with tool use is not novel and is already a widely recognized direction. I wonder what the broader implications of this project are.

### Questions
For TextWorld, how much depends on interface instruction quality vs learned policy?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a framework, SituatedThinker, which aims to ground the reasoning of LLMs in the real world. The core of this method is to enable LLMs to learn to adaptively combine their internal knowledge with external information obtained through predefined "Interfaces". The authors use RL to train the LLM and achieve a good performance in multiple tasks.

### Strengths
1. The problem of grounding the LLM in real-world situations is important to explore.

2. On multi-hop question-answering and mathematical reasoning, the model significantly outperforms baseline methods.

### Weaknesses
1. I am still unclear about the distinction between "Situated Thinking" and standard tool learning. The "Interfaces" defined in the paper, which include the description, the specified input format, and outputs, do not seem fundamentally different from the LLM agent function calling used in the community today. The only difference is the representation, such as text-based tags versus JSON Schema, which makes the contribution incremental.

2. One of the contributions is that the paper claims that the model can generalize to unseen interfaces. However, this does not seem particularly compelling. Most current instruction-tuned models already possess a "zero-shot tool use" capability: as long as a clear description and usage format for a new tool (e.g., the format for OpenAI's function calling) is provided in the message, the model can follow these instructions to use it. This may weaken the contribution as well.

### Questions
1. Would performing SFT first to equip the model with better format-following capabilities make the RL training smoother? Why was this setting not considered?

2. There is a spelling error in the Interface Template in Section 2.1.

### Soundness
3

### Presentation
2

### Contribution
2
