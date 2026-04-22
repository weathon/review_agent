# StreamMem: Query-Agnostic KV Cache Memory for Streaming Video Understanding

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Multimodal large language models (MLLMs) have made significant progress in visual-language reasoning, but their ability to efficiently handle long videos remains limited. Despite recent advances in long-context MLLMs, storing and attending to the key-value (KV) cache for long visual contexts incurs substantial memory and computational overhead. Existing visual compression methods require either encoding the entire visual context before compression or having access to the questions in advance, which is impractical for long video understanding and multi-turn conversational settings. In this work, we propose StreamMem, a query-agnostic KV cache memory mechanism for streaming video understanding. Specifically, StreamMem encodes new video frames in a streaming manner, compressing the KV cache using attention scores between visual tokens and generic query tokens, while maintaining a fixed-size KV memory to enable efficient question answering (QA) in memory-constrained, long-video scenarios. Evaluation on three long video understanding and two streaming video question answering benchmarks shows that StreamMem achieves state-of-the-art performance in query-agnostic KV cache compression and is competitive with query-aware compression approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes StreamMem, a training-free, query-agnostic memory compression framework designed for streaming video understanding with MLLMs. It processes video frames incrementally as they arrive, without requiring knowledge of the full video length or the user query in advance. The key innovation of StreamMem is its use of attention scores between visual tokens and generic proxy tokens to compress the KV cache, maintaining a fixed-size memory footprint for efficient question answering in memory-constrained scenarios. Additionally, StreamMem filters out redundant frames before encoding and merges visual tokens frame-wise into compact prototype representations, further optimizing memory usage. Unlike previous methods, it does not rely on query-aware compression, making it highly suitable for real-world streaming and multi-turn conversational applications. (This paragraph's writing was polished using LLMs)

### Strengths
The paper is interesting and concerns a very hot and difficult topic for long video understanding. The most important part of it in my opinion is the idea to go query agnostic, a very interesting direction, so the main strengths of the paper according to my understanding are:

1 - Pushing the research towards query agnostic long video understanding, crucial for streaming media.

2 - The method is training free, which makes it easier to use as a plug and play with existing MLLMs

3 - Addressig the kv cache compression which can grow very quick and make video processing unfeasible in memory constrain scenarios

4 - The method is validated across many different datasets improving the baseline results.

### Weaknesses
I believe the paper has the following main weaknesses:

1 - There is not much novelty in the proposed method. Attention based selection and compression have been explored in previous works, come cited by the authors and some not. This includes also the frame selection strategy, is very similar to the memory construction of MovieChat which merges frames and representations based on similarity. The particularity of it in my opinion stand on the composition of the KV cache memory with the prototypes representing coarse global description complementing the 'salient visual tokens'.

2 - Salient tokens: there is a process in the frame selection where the methods reduces redundancy by merging similar frames, then later on the method does a sort of kv cache pruning by weighting token importance with a cross-attention between auxiliary tokens and the content visual tokens. My doubt here is 1) why is similarity to the proxy query <|im end|><|im start|>assistant\n considered as an importance score where the query itself only induces some bias to resemble training format, and 2) by keeping the top-k most important tokens, what happens to redundancy, in the end attention scores are similarity measures, does it mean tokens similar to <|im end|><|im start|>assistant\n contain more important information? Why not form the auxiliary query with context from the frames themself?

3 - Validation: While validation is extensive, the improvements are marginals, although the task is difficult. Additionally, while the method is designed for streaming, when testing in non-streaming data, the method has to be compared against at least more previous works that are also training free or use external memory approaches (the comparison with SOTA, if possible, will not affect as a negative on my scoring, I can understand that it might require extra experiments that might not be feasible for the rebuttal, but would benefit the paper). While these works or approaches are framed differently, they are basically trying to solve the same problem, long video understanding.

4 - Limitations: While the method is effective in some scenarios as shown in the paper, i think is important to consider the limitations of it also. For example, while the frame selection reduces computation, what happens to the temporal modeling of the events? It is very likely that the model this way is able only to understand semantic context (coarsely) and maybe some order but not temporal dynamics characterizing each event. This, depending on the downstream application (or even the question itself), might be a big limitation of the method.

### Questions
Refer to the weakness section.

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
This paper proposes StreamMem, a training-free, query-agnostic KV cache compression framework for streaming video understanding with multimodal large language models (MLLMs). The key innovation lies in addressing the memory bottleneck when processing long videos in real-time scenarios where queries are unknown in advance.

Main Contributions:
  - A novel attention-based pruning mechanism using cross-attention scores between visual tokens and chat template tokens as proxy queries
  - A frame-wise KV merging strategy that creates prototype representations for spatial compression

The method maintains fixed memory footprint while achieving competitive performance compared to methods with significantly larger memory budgets.

### Strengths
This paper addresses a practically important problem in streaming video understanding with MLLMs and provides a systematic integration of existing techniques. The work demonstrates reasonable technical execution with comprehensive experiments across multiple benchmarks and three different MLLMs, supported by ablation studies that validate component contributions. The paper is clearly written with effective visualizations and covers relevant literature adequately. The training-free, plug-and-play nature makes it applicable to existing MLLMs while maintaining competitive performance in memory-constrained environments. However, the technical novelty is limited as the core components (cosine similarity-based frame filtering, cross-attention based KV compression) largely build upon existing approaches from LiveVLM, InfiniPot-V, and related work, with the main contribution being their combination rather than fundamental algorithmic innovation.

### Weaknesses
**Incremental improvements over existing work**: The core technical components heavily overlap with prior methods. Cosine similarity-based frame filtering is similar to temporal compression in LongVU, cross-attention based KV pruning follows established patterns in LiveVLM and other KV compression works, and frame-wise merging has been explored in MA-LMM and related papers. The differences from InfiniPot-V appear incremental, mainly in the choice of proxy queries and specific implementation details rather than conceptual breakthroughs.

**Intuition-based design**: The method relies heavily on intuitive assumptions (e.g., chat templates implicitly prompting video descriptions) without rigorous theoretical analysis or empirical validation of these assumptions.

**Significant Methodological Limitations**: 
- Oversimplified compression strategy: Even memory distribution across transformer layers ignores the well-established fact that different layers capture different levels of visual abstractions and may require different compression ratios
- Crude frame filtering: Cosine similarity in embedding space may miss subtle but semantically important inter-frame changes, potentially losing critical temporal dynamics
- Limited proxy query justification: The assumption that chat templates serve as effective generic queries lacks rigorous validation and may not generalize across different video domains or question types

### Questions
1. In Equation (1), how exactly do you aggregate attention scores across multiple query tokens (q dimension)? Is it simple averaging, max pooling, or weighted combination?

2. Why is the memory budget evenly distributed across all transformer layers? Have you experimented with layer-wise importance-based allocation?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces StreamMem, a novel query-agnostic KV cache memory mechanism designed for streaming video understanding with MLLMs. Based on the attention score, it prune the less important KV pairs and boost the efficiency. The evaluation demonstrate it works on the several long video benchmarks.

### Strengths
1. The paper is well-presented, and the core idea is illustrated very clearly.

2. The technical approach makes sense. The use of kv-cache based on the chat-template is well-motivated. I found the ablation study in Table 4 particularly insightful. It demonstrates a reasonable trade-off between performance and generalization.

### Weaknesses
1. The paper lacks evaluation on several important streaming benchmarks, which weakens its overall contribution. As a training-free method, comprehensive evaluation is crucial. Consider including results on OVO-Bench [1], OV-Bench [2], StreamBench [3], and StreamingBench [4].

2. KV pruning based on attention is not new to the community; several prior works have explored this area [5,6]. I feel like the contribution is not sufficient if only making it both query-agnostic and streaming mode.



[1] OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?
[2] Online video understanding: A comprehensive benchmark and memory-augmented method.
[3] Streaming video understanding and multi-round interaction with memory-enhanced knowledge.
[4] Streamingbench: Assessing the gap for mllms to achieve streaming video understanding.
[5] Cross-Self KV Cache Pruning for Efficient Vision-Language Inference
[6] Snapkv: Llm knows what you are looking for before generation.

### Questions
1. In the implementation, it appears that pruning is performed first, followed by merging. What would happen if only merging is applied without pruning, or if merging is performed before pruning? Would these variations lead to different performance outcomes?

2. The trade-off between budget and performance is interesting. Is there any observed trend or threshold for the budget parameter? Is there a "sweet spot" for the value of $M$?

### Soundness
3

### Presentation
3

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
The authors present StreamMem, a training-free method for streaming video understanding with MLLMs under a fixed memory budget. The paper's main contribution is a novel query-agnostic method for compressing the visual KV cache. It cleverly leverages the model's own chat template tokens as a proxy query to identify and preserve the most salient visual information over time. And the proposed method is shown to be effective.

### Strengths
1.	This paper is well-written.
2.	Using chat template tokens to extract universal visual information is a simply but interesting idea.
3.	StreamMem shows great performance in long video understanding task.

### Weaknesses
1.	The paper lacks comparison against essential KV Cache compression baselines, such as StreamingLLM[1] and H2O[2], hindering a clear assessment of its relative performance.
2.	An ablation study for the INPUT FRAME FILTERING module is necessary to validate its specific contribution and effectiveness.
3.	The manuscript requires a deeper analysis of why the chat template tokens have the observed effect, as the current explanation is insufficient.
4.	The contribution appears limited. Both INPUT FRAME FILTERING and the chosen POSITIONAL EMBEDDING are widely adopted techniques, and the unique contribution is not clearly articulated.

[1] Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2023). Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453.
[2] Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., ... & Chen, B. (2023). H2o: Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems, 36, 34661-34710.

### Questions
1.	The analysis of token effects is limited. It is unclear if similar phenomena exist for other common tokens associated with video-chat, such as other vision-related template tokens (e.g., <|vision_start|>), standard system prompts (e.g., "You are a helpful assistant."), or high-frequency query words (e.g., "what," "where," "video").
2.	The Figure 3 fails to clearly distinguish the impact of different queries. It lacks a clear conclusion regarding the mechanism of how these chat template tokens function. Furthermore, the claims are not substantiated with sufficient quantitative results to prove the hypothesis.

### Soundness
3

### Presentation
3

### Contribution
2
