# KV-Prune: Key–Value Similarity for Online Structured Pruning for Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Pruning has emerged as a promising direction for accelerating large language model (LLM) inference, yet existing approaches often suffer from instability because they rely on offline calibration data that may not generalize across inputs. In this work, we introduce Token Filtering, a lightweight online structured pruning technique that makes pruning decisions directly during inference without any calibration data. The key idea is to measure token redundancy via joint key–value similarity and skip redundant attention computations, thereby reducing inference cost while preserving critical information. To further enhance stability, we design a variance-aware fusion strategy that adaptively weights key and value similarity across heads, ensuring that informative tokens are retained even under high pruning ratios. This design introduces no additional memory overhead and provides a more reliable criterion for token importance. Extensive experiments on LLaMA-2 (7B/13B), LLaMA-3 (8B), and Mistral (7B) demonstrate that Token Filtering consistently outperforms prior structured pruning methods, preserving accuracy on commonsense reasoning benchmarks and maintaining strong performance on challenging tasks such as MMLU, even with 50% pruning

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Token Filtering, a fully online structured pruning method for LLMs that accelerates inference without relying on global profiling or calibration data. The key idea is to measure token redundancy through the cosine similarity between current and historical key–value representations, allowing the model to skip redundant attention computations during decoding. To ensure stability, the method employs a variance-aware fusion strategy that adaptively weights key and value similarity across attention heads. It further adopts a tail-focused pruning scheme, applying pruning mainly to later layers with dynamically adjusted thresholds to minimize latency overhead. Experiments show its effectiveness and efficiency.

### Strengths
S1. [Originality and Clarity]: The paper proposes a novel and promising online pruning framework that removes the dependency on global profiling or calibration data. The overall methodology is easy to follow. 

S2. [Experimental adequacy]: The experiments are comprehensive and convincing, covering multiple model scales (LLaMA-2/3, Mistral, Phi-4) and diverse benchmarks.

### Weaknesses
W1. [Inconsistent notation]: Some notations are inconsistent; for example, $alpha$ is used in Eq. (1)(2) and Eq. (4) with different meanings, which could create ambiguity.

W2. [Unclear implementation details]: The paper does not specify how residual connections and LayerNorm are handled when an attention layer is skipped—whether the attention output is zeroed out or bypassed directly.

W3. [Concerns about cold-start issue]: During the early steps of inference, when historical keys/values are insufficient, the similarity estimate may be unstable. The paper does not mention any warm-up or minimum-token safeguard. In addition, an ablation study on the threshold $\eta$ in threshold adaptation is necessary.

W4. [Confusing ablation naming]: Appendix tables use both “Key-Sim” and “Token Filtering” to describe variants, which can mislead readers about whether results correspond to the full model or ablations.

W5. [Overstated claims]: The claim of “no additional memory overhead” is inaccurate, since each layer must still store anchor and variance statistics. Therefore, a quantitative report of actual memory usage should be provided.

### Questions
I am particularly interested in the clarifications related to W2, W3, and W5 in WEAKNESSES, and I hope the authors can provide additional details on these aspects through the following questions:

Q1. How are residual connections and LayerNorm handled when an attention layer is skipped? Is the attention output zeroed out or directly bypassed?

Q2. How does the method address the cold-start issue when historical keys/values are insufficient? Is there any warm-up strategy or minimum context length before pruning is applied?

Q3. Could the authors provide quantitative evidence regarding the actual memory overhead introduced by storing anchor and variance statistics to support the claim of “no additional memory overhead”?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the important problem of LLM inference latency by proposing Token Filtering, a novel online, zero-shot structured pruning method. Instead of relying on calibration data , it dynamically identifies redundant tokens during inference by measuring joint key-value (KV) similarity against the mean of past context. To improve stability, it uses a variance-aware fusion strategy that weights K and V similarity based on their consistency across heads. To minimize overhead, it employs a tail-focused pruning strategy, applying pruning only to the later, more redundant layers of the network. It studies LLaMA-2/3, Mistral, and Phi-4 models and demonstrates significant accuracy preservation and run-time inference speedups, especially at high pruning ratios.

### Strengths
The method is fully online and zero-shot, requiring no calibration dataset whatsoever. This simplifies deployment and avoids the generalization issues of offline pruning.

It demonstrates exceptional robustness at high pruning ratios (e.g., 50%). While baselines suffer from "severe model collapse," Token Filtering maintains strong performance, especially on complex tasks like MMLU .

### Weaknesses
The method's benefits are heavily skewed towards large batch sizes. At a small batch size (e.g., 8), the latency and memory reductions are modest (~12.5% and ~6%, respectively, per Figure 3).This is a significant limitation, as many real-world inference applications (like single-user chatbots) operate at a batch size of 1.

No Runtime Comparison to Baselines: The efficiency evaluation in Figure 3 only compares Token Filtering against the dense (unpruned) model. It lacks a direct runtime and latency comparison against the other pruning methods (SlimGPT, FLAP, PP) that are used in the accuracy tables. Without this comparison, it is impossible to evaluate the true efficiency-accuracy trade-off. For example, a baseline might have slightly lower accuracy but be significantly faster, which could be a preferable trade-off.

While task accuracy is consistently superior, the perplexity (PPL) is often slightly worse than the best baseline (Probe Pruning). For example, on LLaMA-2-13B at 50% pruning, Token Filtering's PPL is 29.22 vs. PP's 28.86.

The incremental averaging strategy for the anchor uses a fixed smoothing factor α=0.9, which seems heuristic. The paper does not include a sensitivity analysis for this hyperparameter.

### Questions
Listed as above in the weakness.

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
This paper proposes Token Filtering, a method to dynamically skip attention computations on tokens that have high cosine similarity to the average key/value in the preceding tokens. The method achieves higher accuracy than the compared baselines while reducing latency and memory overhead, particularly at large batch sizes.

### Strengths
* The proposed method helps reduce the quadratic complexity of the attention module, a source of significant overhead in LLM inference. 
* Token Filtering is calibration-free, ensuring general applicability across a broad range of contexts. 
* Token Filtering achieves high accuracy compared to the baselines and suffers less degradation at higher compression rates in particular. 
* The variance-aware metric helps ensure that token representations that are particularly different within specific attention heads remain unpruned. 
* The paper includes ablations on which layers to prune and the similarity metric.

### Weaknesses
# Major concerns
The following represent key weaknesses that must be addressed to increase the rating:
* Support for static computational graph: Dynamic conditional computation is challenging to integrate with modern JIT compilers, which generally require a static computational graph. Prior work such as [1] incorporates a top-k routing approach to ensure that exactly k tokens are routed through the conditional branches. Can such a scheme be incorporated into Token Filtering? And if so, how does top-K routing affect the accuracy and performance of the Token Filtering? [1] is a closely related work that should be cited in this paper. 
* K/V anchor value causality: The paper primarily refers to using Token Filtering in the decoding stage, it’s a little ambiguous whether the method can be applied to prefill as well. If it can be, how the anchor values are determined during prefill is unclear. Are the K/V anchor values averaged over the entire input sequence or only on the tokens prior to the current query token?
* Token pruning baselines: The baselines include layer-pruning and structured pruning  (neuron/channel) methods; however, Token Filtering shares some features with Token Pruning methods such as LazyLLM. While I understand the authors argument that their method is more closely related to structured pruning / conditional computation methods, a direct comparison with an established token pruning method is crucial to better understanding the accuracy/latency trade-off between these approaches. 
* Long-context performance: The quantitative evaluations are conducted on standard downstream QA tasks or 128 token long sequences of WikiText. These are relatively short prompts. Whether Token Filtering retains its high accuracy at long contexts remains to be evaluated. In particular, tasks from RULER such as Needle-in-a-Haystack may represent a significant challenge for Token Filtering when relying on K/V similarity. 
* Overhead quantification: While the tail-focused strategy is introduced as a way to mitigate the overhead of online similarity computations, quantification of the overhead is not provided. 
* Hyperparameter tuning: The method introduces new hyperparameters such as $Y$ and two $\alpha$ variables. It’s unclear whether these parameters required tuning and how sensitive Token Filtering is to them. 
* Additional benchmarking information: The benchmarking settings are lacking some important details such as the sequence length, time-to-first-token (TTFT), and tokens-output-per-second (TOPS). Based on the increasing latency of attention w.r.t. batch size, it appears that the benchmark setting may be a single input with increasing input sequence length? 

# Minor concerns
The following are minor concerns, typos, etc. which would improve the work but do not affect the final rating: 
* PPL != Text Generation: Table 2 and associated text introduce WikiText PPL as indicative of text generation. In general, PPL can be a misleading metric when evaluating compressed LLMs and language modelling metrics such as PPL are not necessarily strong proxies for open-ended generative tasks. 
* Extension to hybrid attention models: Many modern open-weight LLMs employ hybrid attention with interleaved sparse (local) / dense attention. Token Filtering is not evaluated on such architectures. 

[1] D. Raposo, S. Ritter, B. Richards, T. Lillicrap, P. C. Humphreys, and A. Santoro, “Mixture-of-Depths: Dynamically allocating compute in transformer-based language models,” Apr. 02, 2024, arXiv: arXiv:2404.02258. [Online]. Available: http://arxiv.org/abs/2404.02258

### Questions
* Can TokenFiltering be applied to prefill? How are the K/V anchor values determined in this setting?
* How does LazyLLM compare with Token Filtering in terms of accuracy, latency, and memory?
* How does Token Filtering compare to baseline on RULER or other long-context benchmarks? 
* What is the overhead of Token Filtering when 50% of tokens are skipped across a given decoder? What is the overhead when no tokens are skipped? 
* How sensitive is Token Filtering to $Y$? What about $\alpha$ for anchoring and $\alpha$ for calculating $T_l$? How were the selected values of 0.5, and 0.9 determined? What value is used for $\alpha$ in the $T_l$ update expression? 
* What input sequence length is used for the benchmark results? Is the latency measured across both prefill and decoding or only on one phase? What is the TTFT and TOPS for the various batch sizes? What did the input consist of? 
* Given the dynamic nature of Token Filtering, some inputs are likely to include more tokens with similarity exceeding $T_l$. Is $T_l$ calculated on a per input basis or a global average across all prior inputs? Empirically, what is the mean and variance of the number of tokens filtered across a range of typical inputs?

### Soundness
2

### Presentation
3

### Contribution
2
