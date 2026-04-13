## Human Reviewer 1

### Summary
SeerAttention proposes an efficient attention implementation that reduces attention computation. It uses the learned attention-gating network, which is the same attention mechanism but with a pooled sequence.

### Strengths
1. Simple yet effective.

The overall paper is crystal clear and intuitive. I have no concerns about the presentation and methodology explanation.

### Weaknesses
1. Require of training.

Even this paper claims that sparsity should be learned, but in most cases, it is quite expensive to do so with a large-scale language model such as llama3 70B or 405B.
I think the requirement of learning sparsity patterns is the extreme downside of this methodology, and this direction has also been tried in the past years and always not chosen in practice scenarios due to required training and additional engineering (TDSA: https://arxiv.org/abs/2110.11299, Reformer: https://arxiv.org/abs/2001.04451)

I have concerns that the efficiency of this approach is really better than MInference with training free context extension methods (Self-Extend) during the whole life cycle of LLM. I think if the method should be trained every time a new LoRA or new model is coming, its effectiveness will be quite limited and will have the same limitations as previous methods (TDSA, Reformer, Mamba: https://arxiv.org/abs/2312.00752, etc. (training requires methods).) Therefore, I suggest you add how the trained SeerAttention can adapt to new tasks or unseen weights (LoRA).

Furthermore, is there really no way to pool tokens and approximate attention patterns without training or minimal training (a few steps)? I think there should be some great way to do something like QUEST (https://arxiv.org/abs/2406.10774). I strongly suggest authors try to investigate or at least compare with simple heuristics (averaging, absolute maximum, QUEST-style reducing)

Additionally, the requirement for training may lead to bad generalizability, as shown by the perplexity exploding in Table 1's 128k context length, even if 128k is inside of Llama 3.1 's pretrained context window. 

2. Asymptotic memory complexity is now quadratic.

One of the most important parts of FlashAttention was reducing the memory complexity of the attention mechanism to linear from quadratic by fusing attention score computation. However, because of the requirement of top-k for each row, this method has to store an O (T^2) buffer before converting it to a sparse mask (indices). This peak memory consumption of the temporary buffer will be around 15.625GB for a 1M (Gemini's context length) sequence with 64 block sizes. And if we have to allocate additional buffers such as indices and sorted values for top-k operation (let's think the backend of top-k is merge-sort), we have to store at most 93.72GB (15.62GB for input values, 15.62GB for sorted values, 62.48 for indices). Therefore, I think authors should investigate the memory consumption of the algorithm more carefully and should investigate the way to reduce peak memory (e.g., increasing the block size of gating)

3. Limited methodological contribution.

I think the overall approach is quite similar with TDSA, but just pooling dimension is different. (hidden dim -> sequence dim). I think if there is some kind of justification or comparision between various pooling dim (hidden, head, sequence) and showing why pooling sequence dim is the most effective should be better than current form.

The reducing sequence dim approaches for generating attention masks already exist (QUEST, MInfernece), so I think we need some more justification for additional training in attention-gating networks.

4. Limited downstream task evaluation.

The only downstream task evaluation is the LongBench in Table 2. (I do not think PPL of PG19 can demonstrate actual downstream task performance because perplexity is well-known to have a weak correlation between instruction tasks.)

The context length of Table 2 is around 8 to 32k, which is within the range of their speedup claim (x5.67 on 32k). However, I think if we can show that the performance degradation is minimal or negligible in a longer context, such as 128k, the impact of this work can be more powerful (InfiniteBench: https://github.com/OpenBMB/InfiniteBench, LOFT: https://github.com/google-deepmind/loft, RULER: https://github.com/NVIDIA/RULER, etc.).

### Questions
- Can you show how long the training took with 400 steps? I have a huge concern about the cost, and it may be unrealistic with a long context model. I want to ask you to add FLOPs measures or wall clock GPU-hour measures of training. And the training context length is quite short compared to up-dated long context LLMs (GLM4 = 1M, Llama3.1 = 128k). Therefore, I cannot agree that training will always end in a few hours because the training time grows quadratically.

- Can you show the generalizability of the attention gating network by extending the context window further than the trained (attention gate trained context, 64k~32k)? In Table 1, attention gating tends to fail to estimate exceeding trained context length (64k), which is still inside of LLM pretrained context length.

- Can you add a hyperparameter study about block size? I think the block size of each query and key dimension will show different patterns and have different sensitivity.

---

Therefore, while I agree that the paper is well-written and organized, I think it requires more revisions. I hope we can fix the problems and improve the paper partially during the discussion period.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
5

### Confidence
5

---

## Human Reviewer 2

### Summary
This paper introduces a learnable method to leverage attention sparsity, enhancing the efficiency of large language models (LLMs) in long-context inference. Specifically, the authors developed a learnable gate that adaptively selects significant blocks within the attention map, treating non-significant blocks as sparse. Furthermore, to facilitate efficient training of this gate, the paper incorporates a customized FlashAttention implementation that efficiently extracts block-level ground truth from the attention map with minimal overhead. Experimental results demonstrate that the proposed technique achieves a superior accuracy-efficiency trade-off when processing long-context inputs compared to baseline approaches.

### Strengths
- Addressing the accuracy-efficiency trade-off of LLMs during long-context inference is a critical challenge, especially given the recent trend of employing LLMs to tackle increasingly complex problems with sophisticated inference processes.

- The proposed approach, which learns a sparse attention distribution rather than relying on pre-defined attention patterns or heuristics to approximate sparsity, is intuitively effective in achieving a superior accuracy-efficiency trade-off.

- The customized FlashAttention modification enhances real-device efficiency, providing tangible performance improvements over the naïve implementation and reinforcing the feasibility of applying this approach in real-world applications.

- The paper is well-organized, making it accessible and straightforward to read and comprehend.

### Weaknesses
- **Limitations in Related Work Discussion**: Alleviating attention sparsity to enable long-context inference through attention optimization has been a significant area of research, with numerous discussions surrounding it. To provide a more comprehensive background, it would be beneficial for the authors to include additional related works on attention sparsity, particularly approaches that leverage KV cache eviction, encompassing both pre-defined patterns [1] and dynamic pattern attention sparsity [2,3,4]. 

- **Limited Context Length Evaluated**: Recent works utilizing attention optimization to enhance long-context inference capabilities have successfully extended the context length to settings up to 1M, significantly exceeding the pretrained context limits of LLMs, even without training adjustments [1,2]. However, this paper evaluates only up to a 128K context length. Given that the pretraining context length of the target Llama-3.1 model is also 128K, further experiments with longer context windows would help elucidate the proposed approach’s potential in handling extended contexts.

- **Deployment Cost of the Proposed Approach**: While existing approaches aim to improve the long-context capabilities of LLMs without requiring additional training by exploiting attention map sparsity, this work entails additional model tuning. Benchmarking both the post-training and fine-tuning variants of the proposed approach against existing training-free techniques would offer greater insight into the effectiveness of the required additional training.

- **Setting Mismatch Between Training and Memory Profiling**: In the pooling selection section, the authors state that they use average pooling for Q and a combination of max and min pooling for K. However, in the memory profiling results presented in Figure 8, only max pooling is considered in the training kernel. Would incorporating average pooling, as implemented in the proposed approach, significantly increase memory overhead? Additionally, what would the memory overhead look like during the inference process? How does the additional training introduced by this approach impact memory usage?

[1] Xiao, Guangxuan, et al. "Efficient streaming language models with attention sinks." arXiv preprint arXiv:2309.17453 (2023).

[2] Zhang, Zhenyu, et al. "H2o: Heavy-hitter oracle for efficient generative inference of large language models." Advances in Neural Information Processing Systems 36 (2024).

[3] Ge, Suyu, et al. "Model tells you what to discard: Adaptive kv cache compression for llms." arXiv preprint arXiv:2310.01801 (2023).

[4] Wang, Zheng, et al. "Model tells you where to merge: Adaptive kv cache merging for llms on long-context tasks." arXiv preprint arXiv:2407.08454 (2024).

### Questions
- **Theoretical or Intuitive Explanation for Pooling Selection**: In the pooling selection section, the proposed approach utilizes average pooling on Q and a combination of max and min pooling on K. Could the authors provide a more theoretical or intuitive explanation for this choice? Specifically, given that many existing studies have highlighted the presence of excessively high attention scores on tokens with limited semantic information [1,2,3], could this phenomenon support the rationale behind the pooling selection?

- **Row-Normalization of Max-Pooled Attention Map**: In Section 4.1, the authors mention scaling the max-pooled attention map by row-normalizing it to sum to 1. Could the authors further elaborate on this step? As previous studies [4] suggest, it is critical to maintain the values of the attention distribution and avoid out-of-distribution attention scores (i.e., scores that are too large or too small) for informative tokens. Given this, might the proposed normalization inadvertently produce out-of-distribution attention scores? Have the authors experimented with alternative approaches, such as leaving the attention map unaltered or redirecting unused attention to the first attention sink token?

- **Applicability to Other LLM Architectures**: Can the proposed approach be adapted for other types of large language models, such as models based on the Mixture of Experts (MoE) architecture?

[1] Yu, Zhongzhi, et al. "Unveiling and harnessing hidden attention sinks: Enhancing large language models without training through attention calibration." arXiv preprint arXiv:2406.15765 (2024).

[2] Sun, Mingjie, et al. "Massive activations in large language models." arXiv preprint arXiv:2402.17762 (2024).

[3] Ge, Suyu, et al. "Model tells you what to discard: Adaptive kv cache compression for llms." arXiv preprint arXiv:2310.01801 (2023).

[4] Xiao, Guangxuan, et al. "Efficient streaming language models with attention sinks." arXiv preprint arXiv:2309.17453 (2023).

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper presents SeerAttention, a attention mechanism learning intrinsic sparse attention in LLMs. It outperforms existing methods in post-training and fine-tuning, with efficient implementation and good adaptability to different context lengths and sparsity ratios.

### Strengths
- The proposed SeerAttention mechanism learns attention sparsity instead of relying on predefined patterns. This allows for better adaptation to different language tasks and models, as demonstrated by its performance across various experiments.
- The development of a customized FlashAttention implementation enables efficient learning of the gating network by extracting the block-level ground truth of the attention map with minimum overhead. This not only improves the training process but also contributes to the efficiency of the model.

### Weaknesses
- In section 3.1, the description is insufficient. It is hard to understand the proposed method only with the description in Section 3.1, but I found Figure 2 is easy to understand. I suggrest that more details around Figure 2 should be added in Section 3.1 to enhance understanding. For instance, the operations and significance of each component in the AttnGate module need to be elaborated.
- In section 4.2, many symbols related to FlashAttention are used without prior explanation, making it difficult for readers unfamiliar with FlashAttention to follow.
- The meaning of "max and min pooling on K matrix" is not clear. It should be precisely defined whether it is applied to different dimensions or in a specific way.
- When seq cannot be evenly divided by B (as the size of Q and K after downsampling is [seq/B,d]) during the growth of seq in token generation, the proposed method does not adequately discuss how to handle these situations. It is also unclear if the method is only applicable to the Prefilling stage and not suitable for the token decoding stage.
- After reading the paper, I found that only the blocks with larger attention scores from Atten Gate are used for calculation with flash attention during infernece. If an inference stage Triton kernel is implemented, its pseudo code should be included in the paper.
- There are inconsistencies in algorithm symbols, such as the use of A in Figure 3 and D in the appendix to represent Attention Score, which is confusing.
- Using only a part of the blocks for attention calculation raises questions about information retention. In very long sequences, discarding a large number of tokens may affect the model's ability to perform various text tasks. Since most experimental results report PPL metric, which may not be sensitive to token information sufficiency, additional experiments like Needle-In-A-Haystack or PassKey Retrivel could be added for a more comprehensive evaluation.
- It is not clear if the results in Figure 8 report to Prefilling or decoding stage delays.
- The data requirements of SeerAttention during Post-training and Fine-tuning are not specified.
- The inference delay of the model could be reported in Table 1 for a more complete comparison.

### Questions
NA

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
5

### Confidence
4

---

## Human Reviewer 4

### Summary
This work introduces SeerAttention, which uses pooled embedding to compute attention block mask and uses the mask to do block sparse attention for reduced latency. To enable efficient learning of the gating network, a customized FlashAttention kernel is implemented that extracts the block-level ground
truth of attention map with minimum overhead. Experiments show that SeerAttention can achieve 90% sparsity ratio at 32k context length and offere 5.67x speedup over FlashAttention-2.

### Strengths
* SeerAttention operates at the block level, leading to high efficiency potential.
* SeerAttention learns the sparsity pattern during fine-tuning, which is more flexible than heuristic sparsity.
* An efficient SeerAttention kernel is provided and leads to a 5+x speedup over FlashAttention.

### Weaknesses
* I understand that SeerAttention allows users to adjust the balance between sparsity and accuracy. However, how to determine the sparsity (or Top-k) in practice is not clear to me. I suggest the authors include a discussion on how to choose sparsity in order to maintain high accuracy.

### Questions
* What if the pooling is done on the RoPE'd QK embeddings? Do we still need the RoPE in the AttnGate?
* I assume that using MSE loss to train AttnGate requires fewer training steps and samples. However, what if we train AttnGate using end-to-end LM loss?
* What is the sparsity pattern? Does the max-pooled attention pool prefer recent tokens (like sliding window attention) or does it selectively preserve "important tokens" (like needles in the haystack)?
* Will SeerAttention impact general benchmarks like MMLU?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4