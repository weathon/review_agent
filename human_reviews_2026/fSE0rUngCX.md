# Multimodal Policy Internalization for Conversational Agents

- Avg Score: 7.33
- Decision: Accept (Poster)
- Scores: 6, 8, 8

## Abstract
Modern conversational agents such as ChatGPT and Alexa+ have become indispensable in everyday life. To handle diverse business requirements and enable agentic capabilities, these LLM-based systems often rely on predefined policies, which specify instructions such as model metadata, response styles, and tool-using rules. These policies, typically implemented as in-context prompts, are becoming increasingly complex and lengthy, posing challenges for models in faithfully following them. Moreover, they impose a large fixed computational cost regardless of the input query. As multimodal conversational agents emerge, complex policies that govern multimodal tasks and even involve visual instructions are becoming increasingly necessary, yet they have been rarely studied in previous work. In particular, prior work on prompt compression has focused solely on reducing the length of task templates and demonstrations, which require limited reasoning compared to policies. Meanwhile, related work on policy alignment has been limited to internalizing text-only safety instructions. To bridge this gap, we introduce Multimodal Policy Internalization (MPI), a new task that aims to internalize reasoning-intensive multimodal policies into the parameters of a large multimodal model, enabling stronger policy-following behavior without requiring the policy to be included in-context during inference. MPI presents unique challenges from both data and algorithmic perspectives. We construct two new datasets that cover complex decision-making and tool-using tasks across both synthetic and real-world visual inputs. We investigate diverse internalization strategies and propose a novel three-stage training framework, TriMPI, which enables stronger guidance from the original policy during internalization. Specifically, we first introduce a continual pretraining stage before supervised finetuning, which directly injects policy knowledge into the model. We then propose PolicyRollout, a simple yet effective extension to GRPO-style RL algorithms, which enables more grounded exploration by augmenting the rollout space with policy-aware responses. We show significant improvements of TriMPI over strong baselines in end-to-end performance, generalization capability, and robustness to catastrophic forgetting. As the first work on multimodal policy internalization, we aim to build a strong foundation for future research by providing datasets, training recipes, and comprehensive evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work addresses the challenge of Multimodal Policy Internalization. Policy internalization is a known problem in LLMs, which involves learning policy information through model parameters, instead of providing it in context during inference. This work highlights that this in-context policy can significantly increase inference costs, more so for simpler user queries. This work concretely explores these issues with Multimodal models and proposes a benchmark and a training paradigm for multimodal policy internalization. Overall, on the proposed benchmark, with their framework TriMPI, the multimodal models are much better at following policy and accomplishing a task, without requiring the policy in context during inference.

### Strengths
- Firstly, I think the paper was well written, and most of the content was very clearly understandable. Great job on that!
- The problem of multimodal policy internalization is a well-motivated one, especially in scenarios like legal or financial applications, where agentics systems might need to reference large documents with text and images to make decisions. 
- The idea of policy rollout, to include policy-aware algorithms in the response space, is a clever one and seems to have good empirical benefits. 
- The authors have ablated various stages of their pipeline, CPT, RL finetuning, and the role of policy rollout. 
- Additional analysis on computational efficiency and evaluating policy referral is interesting and should provide useful advice for future practitioners when adopting the ideas from this work.

### Weaknesses
One key question I have about this work is that, although it is geared towards multimodal models, it seems that most ideas are in some way bootstrapped on ideas in the language domain, and they have, after some tuning, shown benefits. 
- This is not necessarily a bad thing, but were there any explorations as to how using policy images can benefit policy internalization?
- The lack of discussion on the vision side of things is my only major complaint about this work.

### Questions
- L304, it's claimed that the policy remains insufficiently grounded in the policy. Was this a qualitative observation? Is there a way it can be measured?
- Just as an additional baseline, what if the model did have access to the prompt at inference time?
  - Does MPI still work better?
  - Or does having the prompt, essentially, make all the baselines and MPI the same in terms of performance?
  - This could be a good tradeoff to understand when the cost of MPI tuning may be higher than just putting the policy in the prompt. Note that this does not take any away from the contribution of the work. 
- At what point does the in-context policy tradeoff wrt the user query become negligible?
  - Specifically, are there scenarios where the user queries may be large enough that the policy tokens are negligible with respect to them?
- In the policy rollout, the way I understand it, is that the RL algorithm now has more policy-aware options, but a question comes to mind, which might be naive:
  - How are these policy-aware responses affecting the rollouts? Is it because they’re more correct, hence GRPO will force the Policy model to produce more similar trajectories? 
- Is RAG an alternative? One piece of related work that I get curious about when thinking of the problem of multimodal policy internalization is what if the LLM could leverage a RAG-like system to retrieve necessary policy-based information.
  - Maybe this is a different way of implementing MPI, but this can maybe reduce the amount of in-context information needed during inference, by only retrieving the relevant one. I would be curious to hear the author’s thoughts on this one.
- Layers in CLEVR policy: are these referring to the layers in the decision tree? Is that how the complexity of tasks is defined?
- Minor: There are often references to figures and tables in the supplement; perhaps it might become easier to parse if it were stated, like Table 5 in the Supplement, etc.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper tackles the problem of internalizing multimodal policies---complex, reasoning-intensive system prompts that guide agent behavior---into the parameters of large multimodal conversational models to improve efficiency and policy adherence. It introduces TriMPI, a three-stage framework combining Visually-Masked Continual Pretraining (VM-CPT), Chain-of-Thought Supervised Fine-Tuning (CoT-SFT), and PolicyRollout, a policy-aware reinforcement learning stage extending GRPO/DAPO for grounded exploration. Experiments on two proposed benchmarks, ClevrPolicy (synthetic decision-making) and GTAPolicy (tool-use), show significant performance gains over SFT and in-context baselines, while maintaining minimal degradation on general reasoning benchmarks (MMMU-Pro and MMLU-Pro).

### Strengths
- The paper is easy to follow, with strong conceptual motivation and clear examples of datasets and methods.
- The paper addresses an important and relevant challenge of conversational agents under long system prompts.
- Extensive experiments, ablations, and analysis (including generalization, efficiency, and catastrophic forgetting) lend strong empirical evidence to claims.

### Weaknesses
- It requires multi-stage SFT and RL with explicit SFT data; less straightforward than prompt-based alignment approaches.
- It is evaluated on self-constructed datasets (ClevrPolicy, GTAPolicy) rather than realistic multimodal policy data.
- It is unclear how the benefit of “multimodal policy” over “text-only policy” translates to conversation agent use cases.

### Questions
- *L048:* The estimated range of policy prompt lengths (1K–50K tokens) is broad—are there empirical or open-source statistics supporting this claim?
- What are practical scenarios where multimodal policy adherence is critical over text-only policy for conversational agents?
- What’s the distinction with deliberative alignment (Guan et al., 2024) apart from the input modality?
- Writing
    - *L072—073:* The phrase “…, which requires minimal reasoning” can be more clear, example,  “…, which requires minimal reasoning to adhere to the policy.”
    - There’s seems to be inconsistency in Section 4: Line 256 describes two stages while Line 259 and Figure 4 show three. The method description be made more consistent across sections.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces Multimodal Policy Internalization (MPI), a task and methods for training multimodal conversational agents to follow complex, reasoning‑intensive policies without carrying those policies in the inference prompt. The authors propose TriMPI, a three‑stage training recipe: (1) Visually‑Masked Continual Pretraining (VM‑CPT) to inject policy knowledge by language modeling the combined (policy + task) streams while masking visual tokens; (2) CoT‑SFT, which teaches explicit policy‑guided reasoning on task data; and (3) RL fine‑tuning with a new rollout augmentation called PolicyRollout (PoRo) that adds policy‑aware trajectories during exploration but keeps training/inference aligned by applying policy gradients only to the no‑policy path. They also release two benchmarks: ClevrPolicy (synthetic, controllable policy complexity; a text‑only variant and an image‑augmented policy variant) and GTAPolicy (real‑world images and queries for tool‑use with versioned, user‑conditional rules; low‑data regime). TriMPI yields large gains over SFT and in‑context baselines, maintains general capabilities, and substantially reduces prompt tokens and prefil time once policies are internalized.

### Strengths
The main strength of the paper is as following:

1. The authors address a practical, under‑explored problem: handling long, reasoning‑intensive policies for decoder‑only models. The motivation is clear and quantified.
2. Two new datasets are introduced. ClevrPolicy (synthetic, controllable complexity; text‑only and image‑augmented policy variants) and GTAPolicy (real images and queries; tool metadata and versioned rules in a low‑data regime) provide controlled benchmarks.
3. The three components (VM‑CPT, CoT‑SFT, RL) are well motivated and ablated; PolicyRollout improves over GRPO/DAPO by enabling policy‑grounded exploration. And we can clearly observed that SFT part is not enough to train the model to compress the long prompts.
4.  The paper evaluates policy updates, policy knowledge referral, catastrophic forgetting, and efficiency, which is crucial for this method, because, generally, on-inference prompt extension with the policy description seems more robust approach in comparison to the specific training.

### Weaknesses
Despite the strong points of the research, I noticed a few weaknesses

1. Risk of overfitting and external validity. While TriMPI maintains general abilities (MMMU‑Pro/MMLU‑Pro) and handles policy updates (Policy Override), broader real‑policy evaluations (e.g., long stylistic/safety guidelines) would further validate external generalization.
2. The three‑stage pipeline, particularly RL, increases implementation and tuning burden; DAPO vs. GRPO behavior on small data underscores this.
3. Although the authors argue soft prompts are task‑specific and hurt robustness, a direct empirical comparison to strong prompt‑tuning/gist‑token baselines would strengthen the case.

### Questions
My questions to the authors are the following:

1. I could miss something, but you notice that you apply vision masking in the first training stage, isn't it a common practice for almost all adapter-based multimodal training, to mask vision input from calculating cross-entropy loss? Or there is some specific trick applied in your research. Could you, please, provide more detailed explanation for this stage?
2. Beyond Policy Override, do you evaluate OOD generalization across visual domains or unseen tool types?
3. How do failure modes break down (policy‑branching vs. perception vs. tool‑argument formatting), especially on GTAPolicy?
4. What are the most common overfitting patterns you observed during training, and can VM‑CPT or PoRo be adapted to mitigate them further?

### Soundness
3

### Presentation
3

### Contribution
3
