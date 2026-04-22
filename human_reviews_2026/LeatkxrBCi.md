# Cache-to-Cache: Direct Semantic Communication Between Large Language Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
Multi-LLM systems harness the complementary strengths of diverse Large Language Models, achieving performance and efficiency gains that are not attainable by a single model. In existing designs, LLMs communicate through text, forcing internal representations to be transformed into output token sequences. This process both loses rich semantic information and incurs token-by-token generation latency. Motivated by these limitations, we ask: Can LLMs communicate beyond text? Oracle experiments show that enriching the KV-Cache semantics can improve response quality without increasing cache size, supporting KV-Cache as an effective medium for inter-model communication. Thus, we propose Cache-to-Cache (C2C), a new paradigm for direct semantic communication between LLMs. C2C uses a neural network to project and fuse the source model's KV-cache with that of the target model to enable direct semantic transfer. A learnable gating mechanism selects the target layers that benefit from cache communication. Compared with text communication, C2C utilizes the deep, specialized semantics from both models, while avoiding explicit intermediate text generation. Experiments show that C2C achieves 6.4-14.2% higher average accuracy than individual models. It further outperforms the text communication paradigm by approximately 3.1-5.4%, while delivering an average 2.5x speedup in latency. Our code is available at https://github.com/thu-nics/C2C.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper “Cache-to-Cache: Direct Semantic Communication Between Large Language Models” proposes a new paradigm (C2C) that lets multiple LLMs exchange information directly via their KV-caches instead of text. By projecting and fusing one model’s cache into another through a learnable neural “Fuser” with gating, C2C enables richer, faster inter-model communication that preserves deep semantics. Experiments show 3–5 % higher accuracy and about 2× speedup over text-to-text interaction. It generalizes across model families and sizes, outperforming routing and text collaboration baselines while supporting heterogeneous model cooperation.

### Strengths
1. Novel idea of transferring cache with little restriction on the type of model.
2. Does not require change of the receiver or sender. 
3. Shows improvement over text to text and cache to cache baselines.
4. Proposes an interesting argument. Although the reviewer personally do not support this claim (see comment below), I think the initial results are nice.

### Weaknesses
1. Lack of experiments on bigger models. 
2. The theory is lacking. To me, there is an implicit assumption that generating KV cache from text is more costly than the fusing operation introduced. However, transforming between two different model classes is intuitively a more complex operation. Can authors explain why this might be cheaper than KV recomputation?

### Questions
1. What is the training overhead for the fuser? Transforming a llama 3.2 1B's KV into Qwen 0.5B's KV seems to be a computational heavy idea.
2. Is this idea reasonable? If you have two models of different architectures, transferring the KV cache does not seem like a good idea to me. The main reason is that the difference in the numeric value of KV are probably too high such that transforming one into another will be more difficult than generating the KV from text.

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
This paper proposes a method, Cache-to-Cache (C2C), which enables two LLMs to communicate through the transfer of KV cache representations, bypassing the need to explicitly decode text. C2C projects and fuses the KV cache of a sharer model into a receiver model using a learned fuser that involves a projection, dynamic weighting, and learnable gating modules. The approach is transformable across model families and experiments show that receiver models in the C2C setup observe an improved accuracy and achieve a significant speedup.

### Strengths
1. The paper's primary strength is the novelty of it core idea, in using KV cache as a direct communication channel between two models, bypassing text generation. While KV cache sharing/re-use ideas exist for single-model inference, this paper proposes a novel latent communication between heterogeneous models.
2. The cache enrichment oracle is an interesting observation to decouple the effect of cache quality from cache length, demonstrating cache enrichment at fixed length improves quality.
3. The C2C trained MLP enables broad setups across model families and sizes.
4. Consistent accuracy and latency gains across models and tasks tested.

### Weaknesses
1. While the oracle experiments motivate the approach, there is limited analysis and justification for why KV-Cache is the optimal representation for inter-model communication, compared to other modes models may communicate with each other, such as through hidden-states.
2. Presentation quality needs to be improved. Some important design details such as alignment to enable heterogeneous communication are only briefly mentioned in main text. Figures and tables not being referenced also make it difficult to follow the paper (Figures not referenced in lines 205-206. Table 3 not referenced in the main text). 
3. The performance benefits from C2C seem to diminish significantly when both the receiver model and sharer model size grow larger (Figure 6). A three layer MLP or the limited training may not be sufficient to capture the shift in distribution in KV cache representation for larger models.
4. The C2C fuser module must be trained for each specific sharer and receiver pair. This is a significant limitation that is not discussed. For $N$ distinct models in a multi-agent workflow, this implies training $O(N^2)$ fuser modules.
5. The paper's main evaluations are single-turn, single-answer benchmarks that do not require multi-agent collaboration. The paper would be stronger if C2C can be demonstrated to also work in tasks that involve models sharing responsibilities such as in planning or negotiation.

### Questions
1. The effective rank increases after fusion, but does this always correspond to "richer semantics"? Could you analyze what types of information are being transferred?
2. Table 4 shows performance gap between C2C and baseline approaches drop with longer contexts. Is this due to cache projection errors accumulating, or fundamental limitations of the approach?
3. What is the training time and GPU memory requirement for training a fuser? How does this compare to the potential inference savings?
4. Could you please clarify the latency calculation for C2C in Table 3? Does the reported C2C time (e.g., 0.40s for Qwen2.5-0.5B Sharer) include the time for both the Sharer's forward pass (0.34s) and the Receiver's forward pass (0.29s), plus the Fuser's computation? The C2C time seems lower than the sum of its parts.

### Soundness
3

### Presentation
2

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
The paper proposes Cache-to-Cache (C2C), a way for one LLM to pass internal semantics to another by projecting and fusing their KV caches, instead of sending text messages. A small “cache projector” merges the sharer’s cache into the receiver’s cache, with a learned per-layer gate that decides where fusion helps. Two “oracle” studies motivate the design: cache enrichment without longer sequences, and cross-model cache transformability.

### Strengths
1) The two pilot studies are interesting and motivate the design: (a) cache enrichment at fixed cache length and (b) cross-model cache mapping that shows convertibility. They make the later choices like layer gating and fusion easier to trust.  
2) Communicating through caches instead of text has clear novelty. It avoids extra decoding and can lower end-to-end latency while letting models share internal signals that plain text cannot carry.

### Weaknesses
1) Training cost vs benefit is not fully quantified. Please report GPU hours, peak memory, and wall time for training the fusion module per model pair, plus the speedup and accuracy gain at serving time. A clear trade-off table would help readers judge whether the training cost is paid back by latency savings and accuracy gains in real use.
2) Writing still need improvement. Some tables are not referenced in the paper.

### Questions
1) How does the method scale to more than two agents. For example, if there are N sharers and one receiver, do you fuse pairwise, do you use a learned mixer over all caches, or do you route by head or layer. What is the complexity and does interference grow with N.  
2) Have you considered multi-agent benchmarks to test this setting. Examples include collaborative QA, team reasoning tasks, or tool-use suites where several models with different skills must coordinate. If not available, can you adapt existing agent benchmarks and report success rate, latency, and cost.

I would like to increase my score if the questions and weakness are addressed.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes C2C, a novel paradigm for direct semantic communication between LLMs. Current multi-LLM systems communicate through a text-to-text (T2T) interface that requires re-encoding of conversation histories and therefore leads to loss in information. C2C proposes a way to let model exchange KV caches that encode richer contextual representations.

The key component of C2C is a cache fuser (a neural network) that projects the sharer model's KV-cache into the latent space of the receiver model. Then the fusing occurs through a dynamic weighting and gating mechanism to decide which layers are fused. C2C module can be trained using NTP with LLMs remain frozen.

Empirical result shows increase accuracy in various benchmark along with faster inference. Further behavior analysis shows increase in effective rank of the receiver's KV cache, verifying information gain.

### Strengths
1. Communication through KV-Cache is indeed a very interesting approach in improving communication between LLMs! KV-cache is known to be containing semantically rich, model-specific view of the context and this paper gives KV-cache a new role than being re-used for accelerating inference. And the motivation is well-justified through the oracle experiments.

2. The main experiments between homogeneous and heterogenous pairs of sharer and receivers showcase consistency: in almost all settings we see both improvement in accuracy compared to T2T and latency. I also like that the author conducts the effective rank analysis to show that there is indeed an information gain.

3. Clear ablation to justify every module: The design is based on findings from oracle experiment and each component seems to be adding performance gain gradually.

### Weaknesses
1. Pair-specific training and limited story on reuse: It seems that a retraining is required for every pair of sharer receiver models. I wonder how this additional cost scales up in more complex multi-LLM systems. There is likely cases where we would have more than two models and the communication between all of them are bidirectional. I also wonder how the cost of the C2C module changes as we have larger models it seems like the training cost is left undiscussed. In addition there is no experiment of showing how fusers can be "merged" (like training a single fuser by jointly optimize on a mixture of sharer and receivers)

2. Lack of failure mode analysis: Despite the accuracy gain, it seems unclear that what exact kind of question will fusion hurt, and how well this idea generalize to OOD settings. Right now this is justified through increased effective rank but some concrete examples would help for better understanding (similar to the <p> example but that is also slightly confusing in the way it's being presented right now). 

3. Presentation needs to be improved: there seems to be unfinished section (for example missing figure in line 205-207)

### Questions
From weakness 1:

Q1: Is there a possibility for fuser reuse? Can you show, for example from the same family, that some of the fusers might be similar?

Q2: In addition is it possible to train a fuser for a pair of sharer group and receiver group?

From weakness 2:

Q3: Can you provide more granular breakdown of your evaluation to show which categories benefit/hurt using C2C versus T2T?

Q4: Some more qualitative example would be nice to have.

### Soundness
4

### Presentation
3

### Contribution
4
