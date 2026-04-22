# FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Existing key-value (KV) cache compression methods for large language models (LLMs) often rely on token eviction, which risks losing critical local information in both long prefilling and decoding scenarios. When extrapolating beyond the pretrained context length, their performance degrades sharply on long-context benchmarks. Motivated by the observation in the frequency domain that the context information is concentrated in the low-frequency components, we propose FreqKV, a parameter-free and architecture-agnostic approach. It iteratively compresses the increasing KV cache in the frequency domain, allowing models to process lengthy contexts efficiently. With minimal training at 8K length, FreqKV extends the context window of LLaMA-2-7B up to 256K tokens while maintaining stable perplexity. Extensive experiments on both prefilling and decoding stages demonstrate that FreqKV enables robust context window extension and consistently outperforms existing KV cache compression methods, highlighting its effectiveness for both understanding and generation in long contexts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a form of cache compression based on converting the tokens in the cache into the frequency domain for compressed storage. This results in lower storage requirements, lower attention complexity in decoding, and a kind of natural context extension since positional embeddings since the compression projects the cache into a fixed cardinality which will not expand to use up all of the positional embeddings.

### Strengths
- KV cache compression is an important topic due to the efficiency gains and power consumption concerns of modern transformers.

- In addition to compression the cache, the method also seems to cause an efficiency gain in the decoding attention operation.

### Weaknesses
- Llama 3 is used as a baseline model. This is important because I believe the only reason some baselines show poor performance is because they have exceeded the number of training positional embeddings. However, Llama 3 is already 1.5 years old at this point. There are already 3 releases of Llama3 which go up to 3.3 and have 131K native positional embeddings. Can FreqKV be applied to these models and show the same good performance past 131K?

- There is no comparison of latency with baselines such as SnapKV.

---

Overall, it would be more convincing if the authors could provide a case on what happens with extremely long contexts. Due to the compressive nature of the cache, it may only be able to hold information up to a certain point before the compression becomes noise. However, if the same trend witnessed at 8K-->16K can be witnessed on a 131K-->232K model, I would find this very compelling. This could be done with the exact same experimental setup and swapping in the Llama 3.x models.

### Questions
For a clearer understanding of how the DCT and IDCT matrices transform the inputs, can you add the dimensions for both the DCT and IDCT matrices?

### Soundness
3

### Presentation
2

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
The paper proposed FreqKV, a novel KV cache compression method. FreqKV performs KV compression in the frequency domain using the DCT, retaining only the low-frequency components. After the fine-tuning, LLMs (Llama2 and Llama3) show comparable performance to the uncompressed one, and are superior to existing KV compression methods. The paper also reports generation speed improvements.

### Strengths
* The idea of applying frequency-domain compression to the KV cache is both intuitive and novel. The similarity to JPEG compression makes the concept easy to understand, and the empirical results demonstrate that this approach is competitive with or superior to existing baselines.
* The paper includes extensive comparisons with a wide range of prior methods on multiple datasets.

### Weaknesses
* The paper lacks a detailed analysis or intuitive explanation of why low-frequency components dominate in the KV cache. Moreover, the large magnitude of low-frequency components does not necessarily imply that they are semantically important, yet the paper seems to make this assumption. While the empirical evidence supports the method’s motivation and effectiveness, a more analytical approach would strengthen the argument.
* Although the authors claim speed improvements in both the prefill and decoding stages, the experiments seem to focus on decoding latency. The paper should provide either empirical results or additional clarification regarding the prefill stage speed-ups.
* In Table 3, competing methods fail completely (near zero point) at the 16K context length. This might not solely reflect a generalization failure but could be caused by an implementation difference: FreqKV applies pre-RoPE compression (on-the-fly RoPE just before inference), whereas others employ post-RoPE compression. It would be essential to clarify this difference.
* FreqKV requires a training phase for parameter learning; however, it remains unclear how it performs without such fine-tuning. Furthermore, comparisons in Table 3 should also include other training-based methods (e.g., LoCoCo).

### Questions
* The related work section would benefit from a broader discussion of recent streaming-based long-context KV management methods, such as InfLLM, InfiniPot, and Minference.
* The rescaling formulation (Eq. 5) is somewhat questionable. Instead of scaling by the number of retained coefficients, it might be more reasonable to adopt an (spectral) energy-preserving normalization. 
* (minor) The abstract could be updated to refer to Llama 3 rather than Llama 2.

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
authors propos FreqKV, a parameter-free, architecture-agnostic KV-cache compression method for LLMs. It applies a DCT along the sequence axis to KV tensors, keeping low-frequency components, then IDCTs back, with an iterative schedule that re-compresses older tokens as the cache grows.

### Strengths
1. Clear idea with strong intuition that low-frequency energy concentration in KV states

2. Comprehensive experiments on multiple benchmarks illustrated the effectiveness of the proposed method.

3. The ablation study provides more detailed information on frequency choice.

### Weaknesses
1. While the low-frequency concentration is a great motivation, the paper doesn't explore why this happens or what information is stored in which frequency bands. For instance, is the low-frequency "global context" and the high-frequency "local token-specific details"? Authors may provide a deeper analysis here to provide valuable insights

2. Attention heads, layers may carry different spectral content. A per-head adaptive $\gamma$ or power-based cutoff might outperform fixed ratios. Do the authors have ablation or adaptive strategies here?

3. Authors may need to compare the most recent baseline [1]

References:

[1] LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models, ICML'25

### Questions
see. weaknesses

### Soundness
2

### Presentation
2

### Contribution
3
