# Fast and Accurate Language Model Decoding via Parallel Token Processing

- Decision: Reject
- Scores: 3, 5, 5, 3

## Abstract
Autoregressive decoding suffers from an inherent efficiency bottleneck due to its sequential token generation process, where each token must be generated before the next can be processed. This sequential dependency significantly limits the ability to fully exploit the parallel processing power of modern hardware. While speculative decoding and layer skipping offer promising speedups, both approaches come with drawbacks. Speculative decoding relies on a secondary small ''drafter'' model, which not only increases memory overhead but may also be unavailable in many cases---the drafter must share the same tokenizer and vocabulary as the main model for compatibility between generated and verified tokens. Layer skipping, on the other hand, can cause discrepancies in the generated output compared to standard autoregressive decoding, as skipped layers do not compute the key-value (KV) cache that plays a crucial role in predicting future tokens.
In this work, we introduce a fast and accurate decoding method, ParaDecode, which accelerates autoregressive decoding while ensuring output parity, without the need for auxiliary models or changes to original model parameters. Our approach is driven by the observation that many tokens---particularly simple or highly-predictable ones---can be accurately predicted using intermediate layer representations, without requiring computation through the entire model. Once the model reaches a certain confidence, further layers are unlikely to significantly alter the prediction. ParaDecode generates tokens at an intermediate layer when confidence is sufficiently high. 
This allows the next token computation to commence immediately, in parallel with the completion of the KV cache computation for the early-predicted token in its remaining layers. This parallelism, implemented using batched matrix operations, enables simultaneous processing of multiple tokens across different layers, thereby maximizing hardware utilization and reducing overall decoding latency. To ensure output consistency, a final verification step is applied to guarantee that the early-predicted tokens match the results of standard autoregressive decoding. Experiments across diverse generation tasks, including text summarization, code generation, and mathematical reasoning, demonstrate that ParaDecode consistently achieves superior decoding throughput compared to baselines with up to 1.53$\times$ speedup, while guaranteeing output parity with standard autoregressive decoding.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper argues that in language model, some easy tokens can already be predicted using intermediate layer representations. Based on this observation, this paper proposes a novel decoding algorithm to accelerate LLM inference. It borrows the idea from speculative decoding by using output prediction from intermediate layer representation as draft tokens. Once a draft token is generated, the language model can immediate start predict the next token based on the draft token. Then it usese a verification scheme to ensure the accuracy of the generated tokens.

### Strengths
1. The paper is easy to understand and the figures are illustrative.

2. The observation that "some easy tokens can already be predicted using intermediate layer representations" is interesting.

### Weaknesses
1. Although the paper claims that the proposed method can accelerate LLM inference with an auxiliary model, it still requires training additional LM head, which introduces extra parameters and memory overhead.  I do not find specific data in the paper about how many extra parameters are introduced for the models used in the experiments.  But according to Section 2.1, each LM head for a 8B model introduces 16M parameters. And according to some previous studies [1,2]. Llama-2-7B and Llama-2-13B can be accelerated by a Llama-68M model, which is about 4 extra LM heads. So it seems that the proposed method does not have advantages in reducing memory overhead introduced by a small model.

2. I tried to analyze the theoretical upper bound of speed-up of the proposed method, and it does not look promising. Assume the tokens can be accurately predicted with first $\alpha$ (0<$\alpha$<1) of whole layers.  And assume we want to generate $N$ tokens in total. Let the latency of generate one token be $t$. Then based on the ideal case depicted in Figure 2, the total time to generate $N$ tokens is $((N-1)\alpha+1)t/N$, when $N$ approaches infinity, the speed up is $1/\alpha$. According to Figure 3(b) in the paper, I think $\alpha$ is about 0.5-0.75, so the proposed method can at best be 2 times faster than autoregressive decoding. And some speculative decoding with powerful drafters (e.g., Medusa and EAGLE) can achieve 4 times speed-up, while only introducing 370M extra parameters.

3. Even if $\alpha$ can be really small, the proposed method still has a potential problem with memory. In order to achieve the ideal case, the proposed method needs to compute $1/\alpha$ tokens in parallel, but these tokens are executed at different layers (as illustrated in Figure 3). And based on the batched matrix operations, the proposed method needs to load parameters of all $1/\alpha$ layers simultaneously, which introduces additional memory overhead compared to running a single layer. Notice that the speculative decoding does not have this problem. In a word, the batched matrix operation introduces additional overhead during computation, which might not be parallelized perfectly.

[1] Specinfer: Accelerating large language model serving with tree-based speculative inference and verification
[2] Multi-Token Joint Speculative Decoding for Accelerating Large Language Model Inference

### Questions
1. Can authors report how much additional parameters are introduced for each model used in the experiments?

2. In Table 1, why self-specdecode and the proposed method have different target models? They are not comparable in my opinion.

3. The early-prediction verification mechanism does not make sense to me. i think it only works for deterministic greedy setting. In stochastic setting, even when the two distributions are same, there is still a high chance that t and t* are different. For example, p_i=(0,5,0.5) and p*=(0.5,0.5), then there is a 50% chance of rejection.

4. I think the layout of Table 1 needs significant improvement. 

5. I think the choices of draft models for SpecDecode in the experiments are bad. The draft model needs to be significant faster than the target model, but 7B and 13B model are not significant faster than 34B model, so naturally SpecDecode cannot be faster. I suggest authors following previous studies (e.g., [1,2]) to choose appropriate target and draft model (e.g., Llama-68M and Llama-2-7B, Llama-2-7B and Llama-2-70B, Llama-68M and Llama-2-13B).

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
In this work, the authors propose to combine layer-skipping based decoding with parallel cache computation for the purpose of efficient and accurate text generation. Traditional layer-skipping decoding schemes do not ensure parity with standard auto-regressive decoding as they require modification to the model weights / computation performed. To perform ParaDecoding, light-weight LM heads are attached to intermediate model layers. If an intermediate LM head is sufficiently confident in a next token prediction, ParaDecode takes that token as the output and continues on to start decoding the next token. In order to preserve the KV cache, in parallel to decoding the next token in the sequence, they continue to compute the representation of the early-exited tokens. ParaDecode also employs a verification scheme where the early predicted token’s identity is verified against the token that would have been predicted had decoding not exited early.

### Strengths
* The ParaDecode methodology is interesting and novel, allowing arbitrary pretrained language models to benefit from layer-skipping decoding while maintaining consistency.

* The memory saving lightweight LM head methodology seems novel and broadly useful for other techniques that require additional LM heads such as Medusa decoding [1].

* The presentation of the ParaDecode method is very clear throughout the paper. The figures describing the method (Figure 1, Figure 4) make the method very explicit.

[1] Cai, Tianle, et al. "Medusa: Simple llm inference acceleration framework with multiple decoding heads." arXiv preprint arXiv:2401.10774 (2024).

### Weaknesses
* The vanilla SpecDecode baseline does not seem like a fair comparison on tasks other than code as the ParaDecode LM heads are trained on each task while a frozen CodeLlama model is used as the drafter for SpecDecode.

* There is no discussion or comparison with draft-head based speculative decoding (see Medusa as mentioned above) even though such methods achieve high speed ups while not requiring separate pre-trained draft models.

* While non-greedy sampling is an important setting for language generation, the paper only looks at greedy verification strategies. If possible it would be useful to include results for how ParaDecode performs with non-greedy sampling.

### Questions
* Would you be able to speak more towards how ParaDecode achieves a speedup when different layers still need to be fetched from memory for each token being decoded in parallel. While I understand that the actual computation is batched, as each token has a corresponding unique weight operating on it, it seems as though the weight movement per token generated would not be decreased. Any clarification would be appreciated.

* Tree based speculative decoding ([1], [2]]) sets the current state-of-the-art for speculative decoding techniques. Would tree based decoding work for ParaDecode?

[2] Miao, Xupeng, et al. "SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification." arXiv preprint arXiv:2305.09781 (2023).

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
Autoregressive decoding method without auxiliary model, based on the observation that simple tokens can be predicted using intermediate layer representations. Paradecode generate tokens at the intermediate layer to allow next token generation immediately. The computation of next token generation and current token computation can be parallelized with batched matrix multiplication.

### Strengths
* Paper is well written and easy to follow.
* Combining layer skipping and speculative decoding is an interesting idea and makes sense. Both approaches compliment each other. 
* Using per-layer LM head seems a right choice to generate a next token based on intermediate with high fidelity.

### Weaknesses
* Missing experiments to claim the generalization of lightwight layerwise LM head. Also it will increase the computation overhead by adding additional parameters. How does latency change with the LM heads?
* Latency is critical in LM inference and paper doesn’t include any latency or throughput experiments.

### Questions
* Is this method compatible with continuous batching? continuous batching is standard of LLM inference so i want to know how this method can seamlessly integrated with standard inference systems. 
* What is the overhead from LM Head at each intermediate layer?
* How to determine Confidence threshold? It says it’s a hyperparameter but I wonder how it can be set in practice. 
* How does the confidence threshold per layer differ? for instance I expect confidence threshold of early layers should be very high because it haven’t passed many layers yet. Also could you provide the average statistics of when the next token generation could start by layers?  (e.g., X tokens started next token gen after layer 0, Y tokens started next token gen after layer 1, ..) 
* Is BMM applicable when weights are different? (across layer, weights are different but is it possible?) As far as I understand, bmm only works for batch dimension, so for input of size (b,n,m) and weight (b,m,p), output becomes (b,n,p). if weight is different for each input, isn’t it just concatenated matrix multiplication?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes ParaDecode, a new approach for speeding up LLM decoding by processing several tokens in parallel. It does so by: (1) adding lightweight LM heads at intermediate layers in an LLM, (2) While processing token $t$, if one of the intermediate LM heads is confident enough in token $t+1$, it samples that token and begins processing it at the bottom of the network, while continuing to process token $t$ with the later layers of the network, (3) finally, once token $t$ has been fully processed by the network, ParaDecode samples the next token $t'+1$, and if it disagrees with token $t+1$, it discards the partial computation of $t+1$ and begins processing $t'+1$. In this way, the processing of several tokens can be overlapped, thus speeding up decoding. Empirically, ParaDecode demonstrates speedups of up to 1.53x relative to autoregressive decoding.

### Strengths
- ParaDecode is a novel algorithm for speedup up LLM decoding by processing several tokens in parallel.
- The approach of training lightweight LM heads $E^{(i)}$ by just training a rotation matrix on the model's frozen LM is nice. $E^{(i)} = E^* T^{(i)}$, where $E^*$ is the models pre-trained LM head. This way, only $d^2$ parameters must be trained per intermediate LM head, instead of $d * vocab$.

### Weaknesses
- **Most importantly**: I do not see why this method gives speedups relative to autoregressive decoding.  In particular, I'm confused by why the batched matrix-vector multiply is able to give meaningful speedups relative to sequential autoregressive decoding. It seems the time to execute this batched operation should essentially be equal to the time to load all the parameter matrices into memory (because it is a memory-bound operation); thus it's unclear to me why batching this operation is faster than performing these operations sequentially (which is what normal autoregressive decoding does).
- The baselines seem quite weak. Speculative decoding can generally give 1.5x-3x speedups (e.g., using Llama-3.1-8B as a speculator for Llama-3.1-70B). The choice of draft/target model combinations is a bit unusual (CodeLlama, Llama 2)---why not just use Llama 3.1 and 3.2 models, which come in many sizes?
- The paper claims ParaDecode is easier to get working than speculative decoding when a small draft model is not readily available. However, the intermediate LM heads must still be trained. Therefore, it's unclear to me why this training is simpler than, for example, the LM head training in Medusa. Medusa gives meaningfully larger speedups than those presented in this paper.
- This paper bears meaningful resemblance to the speculative decoding algorithm in LayerSkip, which is not adequately discussed or compared to in my opinion (although in LayerSkip the whole model is trained, a meaningful difference).

### Questions
- See my first bullet in the weaknesses section: Why is batched matrix-vector multiply faster than sequential matrix-vector multiplies?
- Can you explain the consistency results in more detail (Table 2)? Are these with greedy decoding? I'm confused by the results being so low.
- Is there an intermediate LM head at every single layer? If so, I'm confused by how using the threshold $\gamma = 0$ can work, as this would mean all tokens early exit after layer 1, which seems like it would perform quite poorly (many rejected tokens).
- In Figure 5, what are the thresholds that are references in the legends of the figures? Speculative decoding does not have any thresholds that I'm aware of.
- What batch sizes are used in the experiments? Is it always batch size of 1 during ParaDecode inference?
- NIT: On line 247, shouldn't it be $l^{(t_1)} > l^{(t_2)} > \ldots$?

### Soundness
1

### Presentation
3

### Contribution
2
