# Micro-Macro Retrieval: Reducing Long-Form Hallucination in Large Language Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Large Language Models (LLMs) achieve impressive performance across many tasks but remain prone to hallucination, especially in long-form generation where redundant retrieved contexts and lengthy reasoning chains amplify factual errors. Recent studies highlight a critical phenomenon: the closer key information appears to the model outputs, the higher the factual accuracy. However, existing retrieval-augmented language models (RALMs) lack effective mechanisms to ensure this proximity — external evidence is injected into reasoning via multi-turn retrieval, but this cannot ensure key information stays close to the outputs. We propose Micro–Macro Retrieval ($M^2R$), a novel retrieve-while-generate framework to fill this gap. At the macro level, $M^2R$ retrieves coarse-grained evidence from external sources; at the micro level, it extracts essential results from a key information repository built during reasoning and reuses them while generating answers. This design directly addresses the key-information–to-output proximity bottleneck, effectively reducing hallucination in long-form tasks. $M^2R$ is trained with a curriculum learning–based reinforcement learning strategy using customized rule-based rewards, enabling stable acquisition of retrieval and grounding skills.  Extensive experiments across different benchmarks demonstrate the effectiveness of $M^2R$, especially in lengthy-context settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to reduce LLM hallucinations in long-context settings, especially in the context of RAG.
The key limitation of current RAG approaches is that, either the long context makes it hard for the model to accurately identify key info, or model fails to retrieve intermediate results from its own reasoning chains.
To address this limitation, the paper introduces **micro-macro retrieval (M$^2$R)**, a retrieve-while-generate method.
M$^2$R hinges on the positive correlation between key information proximity to model outputs and factual accuracy, which was reported previous works.
M$^2$R directly enforces this proximity mechanism into the LLM through curriculum-learning with GRPO, such that *macro retrieval* (also `<think>` phase) implements traditional RAG and maintains a key info repo while *micro retrieval* (also `<answer>` phase) extracts key info from the established repo to ground model outputs.

Experiment evaluation on multi-hop QA benchmarks show that M$^2$R generally outperforms most baselines and its comparative advantage is more evident in challenging scenarios (HotpotQA-2/3Q).

### Strengths
1. Originality

   This paper is novel in that it alleviates hallucinations in RAG based directly on empirical insights from [1-2] that key information position methods in long-form generation.
   The application of GRPO + curriculum learning to realize the insight above also makes sense.
2. Quality

   The experiment design is sound and directly supports central claims of this paper.
   The authors have also done an excellent job by releasing the source code with clear documentation.
3. Clarity

   The paper clearly explains its motivation, key insights (lines 53-55, Appendix B), as well as discussions on the rationale of key designs (lines 120-122, 136-137, 161, 249-256, etc).
   The case study is useful for readers to understand the method intuitively.
4. Significance

   This paper could be of significance to the field by grounding in empirical findings of previous works, combining GRPO and curriculum learning and achieve a compelling performance boost in challenging multi-hop QA tasks.

References:

[1] Lost in the middle: How language models use long contexts. (2023)  
[2] Found in the middle: How language models use long contexts better via plug-and-play positional encoding. (2024)

### Weaknesses
(Authors do **not** need to refer to points raised in this section since the main points are already mentioned in *"Questions" section*.)
1. W1: Limited model family

   The paper only involves experiments on Qwen-2.5-3B/7B models; results on more diverse model families could strengthen the paper's arguments.
2. W2: Missing discussions on costs

   Since the core method introduces additional storage requirements (key info repo) and requires retrieving key info during answer phase, there are concerns regarding whether these components induce heavy storage/time costs.

### Questions
**Major questions (that could affect rating)**
1. **Question 1**: Limited model family

   M$^2$R is only tested on two models of different sizes (3B, 7B) but the same model family (Qwen2.5). This limitation does *not* directly undermine the central claims of the paper, but results on more diverse model families could greatly enhance the general utility of the proposed method.
2. **Question 2**: Cost analysis

   Inference efficiency is a critical concern in RAG applications. The proposed method requires maintaining a key info repo during macro retrieval and retrieving key infos during micro retrieval. Therefore, a natural concern arises as to whether the performance boost is worth the cost:

   *How does this micro-macro framework affect inference latency, and what are the storage costs of the key info repo?*

   A detailed analysis (either theoretical or empirical) could be useful in deciding whether M$^2$R is usable in practice.

**Minor questions and suggestions (that are not considered to affect rating)**
1. **Minor question 1**: Additional details for reproducibility

   Although the authors have provided source code and some experiment details in the paper, additional details such as hardware requirements and seeds could be useful for reproductions of results.
2. **Suggestion 1**: Notations

   The method name, M$^2$R, is not consistently presented: it is usually written in normal font but sometimes written in italics (lines 206-261, 267, etc).
3. **Suggestion 2**: Paper organization

   Related Work section could help readers set the context, but currently it is placed in the Appendix. Therefore I recommend move Table 1 (M$^2$R prompt template) to the Appendix and move Related Work section to the main body instead.
4. **Suggestion 3**: Details on training stability

   The paper mentions at lines 242-243 that, directly optimizing macro/micro retrieval leads to poor convergence. Detailed results (preferably placed in the Appendix) could help future researchers gain in-depth understandings regarding the significance of curriculum learning.

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
4

### Summary
This paper proposes Micro–Macro Retrieval (M2R), a two-level retrieve-while-generate framework designed to reduce hallucination in long-form generation. By combining macro retrieval of coarse evidence during reasoning with micro retrieval of key information during answer generation, M2R ensures that essential evidence remains proximal to output tokens. Trained with a curriculum-based reinforcement learning strategy using rule-based rewards, the method achieves significant improvements in factual consistency and robustness across multi-hop QA and long-context benchmarks compared to strong RAG baselines.

### Strengths
1. The idea of Micro–Macro Retrieval surprisingly natural and well-motivated — it addresses one of the most persistent issues in RAG systems (long-form hallucination) with a solution that feels both principled and minimal. The “retrieve-while-generate” framing elegantly captures how reasoning and retrieval should co-evolve.

2. The paper is very carefully written. I particularly like how the authors formalize the two retrieval levels and the transition between \<macro_tool_call>, \<key_info_save>, and \<micro_tool_call>. It feels like reading a well-designed system that could actually be implemented in production without hidden tricks.

3. I really appreciate that the system is interpretable by design: it shows that we can literally see the reasoning flow: what it retrieved, what it saved, what it reused. That’s a refreshing contrast to the black-box nature of most retrieval-augmented models. It also feels cognitively aligned with how humans solve tasks: note things down, then recall them precisely.

4. Compared with Self-RAG[1] this work feels like a thoughtful continuation rather than simple imitation. Self-RAG let the model decide when to retrieve; M2R turns that spark into a full reasoning routine. It not only detects when retrieval is needed but also explicitly manages what to keep and reuse, maintaining a long-term internal memory grounded in already verified facts. I find this progression deeply satisfying—the model isn’t merely “asking for help” anymore; it’s learning to remember what it already knows to be true. That evolution from reactive retrieval to proactive self-memory feels like a genuine step forward.

[1] Asai, Akari, et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." The Twelfth International Conference on Learning Representations.

### Weaknesses
I think this paper has no particularly obvious weaknesses. Unlike the naïve combination of RAG with RL or GRPO, this work takes a much more principled approach—from memory to retrieval to the overall training strategy, making it a significant step forward in improving RAG performance. However, I do have a few questions that I’d like to raise briefly.

1. In your gradient computation implementation, how did you mask out the information from the retrieval part? Is the positional information handled relative to the retrieval step, or do you directly remove the masked portion?
2. When invoking retrieval, are macro-retrieval and micro-retrieval mutually exclusive, or can they be used jointly?
3. When is **key_info_save** called? In your experiments, is it triggered only when the macro retrieval is considered contextually relevant, and thus the macro information is saved? If there are identical or similar **micro_tool_calls**, does the model jointly retrieve them?
4. If the model chooses to invoke **micro**, but there is no stored memory or relevant content, does that lead to hallucination? How do you handle cases where micro-retrieval fails or retrieves incorrect results?
5. Are there situations where neither **micro** nor **macro** is used?
6. What are the overall training costs and latency characteristics? If the database is large or the query is long, does it lead to bottlenecks?
7. If provides a main figure, it will help more readers to fastly grasp your methodology.

### Questions
See above, I am warmly welcome to discuss further detailed on this paper.

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
This paper tackles hallucination in large language models (LLMs), especially in long-form generation where redundant contexts and extended reasoning amplify factual errors. The authors observe that factual accuracy improves when key information appears closer to the generated output, yet existing retrieval-augmented LMs (RALMs) lack mechanisms to ensure this proximity.
To address this, they propose Micro–Macro Retrieval (M2R), a retrieve-while-generate framework combining macro-level external retrieval with micro-level key information reuse from an internal repository built during reasoning. M2R directly maintains evidence proximity to outputs and reduces hallucination in long-context tasks.
Trained via curriculum-based reinforcement learning with rule-based rewards, M2R achieves consistent gains in factual accuracy and grounding across multiple benchmarks, showing good effectiveness in lengthy-context scenarios.

### Strengths
1. The topic is valuable, especially for mitigating hallucinations in long-form reasoning models.

2. The method is straightforward and conceptually sound.

3. The paper is well written and easy to follow.

### Weaknesses
1. The core methodological details are underspecified. If I understand correctly, the approach hinges on constructing and maintaining a key-information repository, and then:

- (1) For macro retrieval: per Lines 060–061, how is “the reasoning process yields answer-aligned evidence” detected, and how exactly is it inserted into the repository(i.e., what constitutes the key and the value)?

- (2) For micro retrieval: what query is used to retrieve answer-related information from the repository?

- (3) Does GRPO merely teach the LLM when/how to invoke the macro- and micro-retrieval tools, rather than optimizing the retrieved content?

2. Empirical coverage is limited. Training is conducted only on Qwen2.5-3B/7B and evaluated on four relatively simple QA datasets. How does the method perform on other model sizes/families and on more challenging reasoning benchmarks? Moreover, Figure 2 shows substantial reward oscillation and a low mean (~0.4), suggesting training stability or sufficiency may be a concern.

3. FlashRAG details and ablations require clarification, including the knowledge base size, and the effects of chunk size and retrieve number on performance. Reporting token statistics of input/output during inference would further clarify whether the approach practically alleviates long-form reasoning constraints.

### Questions
Please see the weaknesses,

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
A "retrieve-while-generate" framework that performs macro retrieval (external sources during reasoning) and micro retrieval (from key-information repository during answer generation) to ensure evidence proximity and reduce hallucination.

### Strengths
1. Strong Empirical MotivationLost in the Middle Phenomenon (Liu et al. 2023):

GPT-3.5 accuracy drops from 75% → 55% when answer-bearing evidence moves from context start to middle (Figure 3)
Empirically validated across multiple models and tasks
Theoretical Grounding (Appendix B):

Analyzes RoPE positional encoding: attention ∝ q^T R_{θ,m-n} k
High-frequency components cancel at large distances Δ
Formal claim: "Evidence contribution decreases monotonically with distance"
This foundation is solid — problem is real and well-documented.

2. Explicit Key Information ManagementKey-Value Repository Design:

Advantages:

- Explicitly separates "what to remember" from "how to answer"
- Forces model to extract atomic facts rather than rely on context attention
- Reduces cognitive load during answer generation
This is cleaner than implicit reasoning traces (e.g., ReSearch where key info is buried in <think> text).

### Weaknesses
1. "Retrieve-While-Generate" might be Misleading

What the Paper Claims:

"M²R is the first framework to introduce a retrieve-while-generate paradigm during the answer phase."
This suggests a novel generation mechanism — e.g., retrieval happening during the forward pass.What Actually Happens:Multi-Turn Generation with Tool Calls:

This is identical to:

OpenAI's function calling
Anthropic's tool use
ReAct (Yao et al. 2022)
Self-RAG (Asai et al. 2023) — which also retrieves during answer generation!
M²R requires 5-10 sequential model invocations per query.

2. Cost Analysis Completely Absent

### Questions
1. How many model invocations does M²R require per query on average?

Please report separately for: (a) think phase, (b) answer phase, (c) total
Break down by dataset (HotpotQA, MuSiQue, etc.)
What is the range (min/max)?



2. What is the end-to-end latency in realistic deployment?

Table 2 shows batch inference time (0.67s), but what about non-batched API calls?
Assuming 100ms per forward pass: how long does a typical query take?
How does this scale with question complexity?

### Soundness
2

### Presentation
3

### Contribution
3
