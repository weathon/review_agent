# Hymba: A Hybrid-head Architecture for Small Language Models

- Decision: Accept (Spotlight)
- Scores: 8, 6, 8, 8

## Abstract
We propose Hymba, a family of small language models featuring a hybrid-head parallel architecture that integrates attention mechanisms and state space models (SSMs) within the same layer, offering parallel and complementary processing of the same inputs. In this hybrid-head module, attention heads provide high-resolution recall, while SSM heads facilitate efficient context summarization. Additionally, we introduce learnable meta tokens, which are prepended to prompts to store critical meta information, guiding subsequent tokens and alleviating the “forced-to-attend” burden associated with attention mechanisms. Thanks to the global context summarized by SSMs, the attention heads in our model can be further optimized through cross-layer key-value (KV) sharing and a mix of global and local attention, resulting in a compact cache size without compromising accuracy. Notably, Hymba achieves state-of-the-art performance among small LMs: Our Hymba-1.5B-Base model surpasses all sub-2B public models and even outperforms Llama-3.2-3B, achieving 1.32\% higher average accuracy, an 11.67$\times$ reduction in cache size, and 3.49$\times$ higher throughput.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces a new hybrid model named Hymba that integrates attention mechanisms with SSMs in a hybrid-head manner. The main difference between existing models like Samba is that Hymba enables hybrids to operate in parallel rather than sequentially.

The authors progressively propose several augmentations to the hybrid-head framework, including local/global attention, KV cache sharing, and meta tokens. They found that Hymba performs strongly against existing pure attention-based models and other hybrid models. By training the small-scale Hymba on trillions of tokens, Hymba performs well on common benchmarks and achieves near-perfect results on NIAH tests.

### Strengths
* This is a very solid work, with numerous experimental results and ablation studies verifying the effectiveness of Hymba, which is quite convincing.
* The results on NIAH tests are very impressive, especially given the model scales.

### Weaknesses
* I suggest the authors to add discussions with InfiniteTransformer, which fuses LA and attn in similar manners (Eq. 10)
* Why not the authors conduct experiments on Mamba2 rather than Mamba?
* If possible, I suggest the authors to add some discussions with more existing linear attention works like RetNet/GLA/HGRN2/YOCO

InfiniteTransformer: Efficient Infinite Context Transformers with Infini-attention

### Questions
* In Table 2, why are the ARC-C scores reported for 25 shots? I believe the common choice is zero shot.
* I am curious about how the throughputs in Table 2 are measured. Given that RWKV6 is reported to be much faster than others, this does not match my impressions. What is the input to the model? Can A100 GPUs with 80GB of memory handle an input size of 128 * 8K?

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
3

### Summary
This paper introduces Hymba, a hybrid architecture that combines Transformer and SSM within a single layer. The authors also propose several additional techniques to enhance further efficiency and performance, such as combining global and local attention, cross-layer KV cache sharing, and introducing meta-tokens, which act as a learned prefix. Through extensive evaluation, the authors show that Hymba performs best among small LMs while being significantly more efficient.

### Strengths
1. The paper introduces several useful designs that could be empirically used for training hybrid Transformer + SSM models.
2. The proposed method shows high performance, outperforming most small LMs while achieving better computation efficiency.
3. The authors perform extensive evaluations across diverse tasks and setups.a

### Weaknesses
1. Limited novelty. The paper seems to suggest a combination of implementation tricks rather than proposing a significant idea.
2. The hybrid head design seems to be the most significant component of the proposed method, but the evaluation justifying its efficacy is confusing. In Figure 3, the authors compare their method with Samba and claim they achieved a larger ERF. However, it is unclear if the gain comes from the parallel design or the introduction of global attention heads (not present in Samba).

### Questions
1. Does the Samba baseline in Figure 3 also use the same number of global attention layers? If not, how can we tell if the performance gain comes from the parallel design or the introduction of global attention layers?
2. How does the parallel design impact the throughput? Empirically, are the SSM heads and Attention heads computed in parallel or sequentially? (e.g., if you forward the input through the SSM heads, then forward the input through the attn heads, and then aggregate them, then the implementation is done sequentially, even if the design is conceptually ‘parallel’) Would ‘true’ parallel computation require a specialized GPU kernel?
3. Is the concept of meta-tokens useful for SSMs only, or would general Transformer models also benefit from the technique?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Hymba,  a small language model that combines attention mechanisms with SSMs in a hybrid-head architecture. Authors did the following, 

1. A hybrid-head architecture that processes inputs through parallel attention and SSM heads in each layer, leveraging attention's high-resolution recall and SSM's efficient context summarization

2. Learnable meta tokens prepended to input sequences that act as learned cache initialization to modulate token processing

3. Optimization techniques including local/global attention combination and cross-layer KV sharing to improve efficiency

The authors validate their approach through extensive experiments showing Hymba1.5B achieves comparable performance to larger models while being 3x faster and using 15x less cache yielding memory gain.

### Strengths
1. Empirical study is comprehensive and clear, particularly the ablation studies;

2. Consistent gains for models with different sizes under 1.5B

3. The paper is easy to follow

### Weaknesses
In general I think this is a strong paper. I have the following comments and questions. 

- Some implementation details can be added
1. I am a bit lost while reading the cache optimization, and meta-token, maybe worth more explanations or pseudocode?

- How many meta tokens are needed, and how they are related to the performance in downstream tasks?

- The ratio between SSM and Attention is not clear. And I understand that recent papers demonstrate that it is important to integrate attention for linear RNN models, but attention layer still added overhead, though coupled with all those techniques. A fair comparison could be a pure attention model with all methods proposed in this paper, comparing their efficiency gain and performance curve.

### Questions
1. The interplay between SSM and Attention, see weakness 

2. Consider a more fair comparison for efficiency and performance gain? see weakness

3. How does the model perform for long context tasks as this is the where the gain of Hymba goes significant?

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
The paper introduces **Hymba**, a new family of small language models designed with a hybrid-head architecture that merges attention mechanisms with state space models (SSMs) for improved memory functions. Hymba utilizes attention heads for precise recall of high-resolution information and SSM heads to summarize broader context efficiently, mirroring aspects of human memory. A significant innovation is the introduction of learnable meta tokens, which act as a dynamic cache initialization, enhancing focus on key information during inference. 

The authors outline a systematic approach to developing Hymba, from creating fused hybrid modules to scaling model and data sizes. Experimental results show that Hymba sets new benchmarks for small language models, achieving an improved accuracy-efficiency balance. Notably, *Hymba-1.5B* matches the commonsense reasoning performance of larger models, such as *LLaMA 3.2 3B*, while operating more efficiently. The meta tokens also reduce attention map entropy, potentially aiding the model in identifying and focusing on salient tokens. Hymba’s design offers promising advances in both the performance and efficiency of compact language models.

### Strengths
1. this is a solid and well-written paper.
2. the hybrid-head design, which combines attention and state space models, is an innovative approach that provides Hymba with both fine-grained recall and efficient long-range summarization. The introduction of learnable meta tokens as a dynamic cache initialization mechanism is also novel, drawing a parallel to human metamemory functions.
3. the experiments are extensive and well-documented, including ablation studies that thoroughly evaluate the impact of each component, such as the hybrid heads and meta tokens. The benchmarks are comprehensive and competitive, providing a robust demonstration of Hymba's capabilities.

### Weaknesses
1. It would be even better if the effectiveness of the Hymba could be validated on image or speech modalities.

### Questions
1. Equation 1 does not mention a scaling factor. Is it included in the actual implementation?

### Soundness
4

### Presentation
4

### Contribution
3
