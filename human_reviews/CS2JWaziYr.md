# MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 5

## Abstract
Large Language Models (LLMs) have become more prevalent in long-context applications such as interactive chatbots, document analysis, and agent workflows, but it is challenging to serve long-context requests with low latency and high throughput. Speculative decoding (SD) is a widely used technique to reduce latency losslessly, but the conventional wisdom suggests that its efficacy is limited to small batch sizes. In MagicDec, we show that surprisingly SD can achieve speedup even for a high throughput inference regime for moderate to long sequences. More interestingly, an intelligent drafting strategy can achieve better speedup with increasing batch size based on our rigorous analysis. MagicDec first identifies the bottleneck shifts with increasing batch size and sequence length, and uses these insights to deploy SD more effectively for high throughput inference. We leverage draft model with sparse KV cache to address the KV bottleneck, which scales with both sequence length and batch size. Additionally, we propose a theoretical model to select the optimal drafting strategy for maximum speedup. Our work highlights the broad applicability of speculative decoding in long-context serving, as it can enhance throughput and reduce latency without compromising accuracy. For moderate to long sequences, we demonstrate up to 2.51x speedup for LLaMA-3.1-8B when serving batch sizes ranging from 32 to 256 on various types of hardware and tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
Conventional understanding suggests that **speculative decoding (SD)** enhances performance primarily in scenarios with small batch sizes. This paper presents a novel theoretical analysis demonstrating that SD can also yield performance improvements in settings with large batch sizes and extended prompt lengths. The analysis in this paper identifies how performance bottlenecks shift with increasing batch size and prompt length. To address these bottlenecks, this paper proposes using a draft model with a compressed key-value (KV) cache, effectively alleviating the new constraints. The theoretical framework provided by this paper enables an optimal drafting strategy tailored to specific draft-target model pairs, making this approach particularly valuable for SD applications in long-context scenarios.

### Strengths
1.The paper is well-written.
2. The paper presents a clear and logical progression of ideas.
3. The paper provides rigorous theoretical analysis supported by extensive experimental results.

### Weaknesses
I appreciate the quality of this paper, and I have only one minor suggestion.

1. **Potential limitations in real-world application**: The theoretical analysis is thorough and effectively clarifies when the conventional view—that speculative decoding (SD) enhances performance primarily with small batch sizes—applies, as well as when the new discovery on SD’s performance benefits with larger batch sizes and extended prompt lengths holds true. However, in real-world cloud environments, request configurations are often limited to smaller batch sizes and shorter prompt lengths (see https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/). Could the authors kindly discuss how this theoretical analysis might be applied within the constraints of typical LLM request configurations?

### Questions
Please answer the following question.

1. **Potential limitations in real-world application**: The theoretical analysis is thorough and effectively clarifies when the conventional view—that speculative decoding (SD) enhances performance primarily with small batch sizes—applies, as well as when the new discovery on SD’s performance benefits with larger batch sizes and extended prompt lengths holds true. However, in real-world cloud environments, request configurations are often limited to smaller batch sizes and shorter prompt lengths (see https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/). Could the authors kindly discuss how this theoretical analysis might be applied within the constraints of typical LLM request configurations?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses the challenge of improving latency and throughput in LLM inference for long-context tasks. Traditional speculative decoding (SD) literature tends to focus on smaller batch sizes, but MagicDec demonstrates how SD can also benefit high-throughput scenarios involving long sequences. By combining theoretical and empirical approaches, MagicDec identifies and mitigates bottlenecks in memory access using a draft model with sparse Key-Value (KV) caching. It introduces a theoretical model for selecting optimal draft strategies, achieving speedups of up to 2.51x in large batch settings for the LLaMA-3.1-8B model across diverse hardware configurations.

### Strengths
+ The work introduces novel improvements in speculative decoding for large-batch, long-sequence LLMs, challenging existing assumptions and offering a fresh perspective.
+ Theoretical insights are well-validated through high-quality experimental results, with robust data supporting the effectiveness of MagicDec across diverse scenarios.
+ While some sections could benefit from increased clarity, the paper's main findings are well-articulated, with sufficient depth to support the claims.
+ MagicDec offers notable improvements in both latency and throughput for LLMs, with potential impacts across a range of long-context applications.

### Weaknesses
- Complexity in Explanation: Sections discussing KV caching and speculative decoding speedup factors could be streamlined to improve readability, especially for a broader audience.

### Questions
## Questions
1. Can you clarify the potential trade-offs in performance if MagicDec were applied to significantly smaller LLM models?
2. Is the performance gain from MagicDec sustained across varying types of long-context tasks, particularly those that require variable batch sizes or non-standard hardware configurations?
3. Have you considered additional draft model selection criteria that might further alleviate the KV bottleneck?

## Comments
### Comment on Lines 52-74 and Figure 1a
In the section describing how the "KV Cache Bottleneck Enables SD Speedup Even For Large Batches," the authors assert that KV cache loading time increases significantly in long-context, large-batch scenarios, leading to a more memory-bound process. This serves as evidence for the memory bottleneck supporting speculative decoding at scale. However, Figure 1a appears to represent the combined "KV load and store" time rather than isolating the KV load time alone. Could you clarify what portion of this time is allocated to "KV store"? Understanding the distinction between load and store times would help validate the impact of the bottleneck more precisely.

### Comment on Lines 203-211 ("Expected Generation Length Per Step")
The preprint by Timor, Nadav, et al., titled *"Distributed Speculative Inference of Large Language Models"* (arXiv preprint arXiv:2405.14105, 2024), appears to be highly relevant to your study, particularly in examining the regime where SD leads to either speedups or slowdowns based on the drafting latency budget. Consider discussing this work to strengthen the related literature section.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper discusses the batched speculative decoding in the long context scenario. It finds a critical sequence length that is the threshold for batched speculative above which it can show speedups compared to the original autoregressive decoding. The paper further examines the compressed kv cache and tries to find the best compressed KV-based drafting with the best strategy.

### Strengths
- Interesting insights on the significant sequence length and how the nature of memory bound translates to speedups in batched speculative decoding setting.

- Insights that changing the KV drafting instead of the smaller draft models can potentially have better acceptance rate and thus better speedups.

### Weaknesses
- Perhaps on the incremental side since the main new ideas are not all that large. The idea of compression KV is already in the literature.

- Perhaps the paper jump to some conclusions too fast without enough explanations, which makes it a little hard to follow.
  
  - What does Figure 1(b) try to show?

  - Maybe consider adding more details about the effects of increasing batch size with the original speculative decoding.


- Some terms are vague in the texts. Authors may consider more clearly define them.

   - What does Figure 1(b) try to show?

  - What does ``self-speculation" in the texts mean?

  -  What does KV budget mean?

- Some important design details can be more explicitly clarified.

  - Where does the time breakdown such as Figure 1(a) come from? GPU kernels can hide latency among threads so the end-to-end time does not necessarily equals to the combination of the memory loading and computing time. Nevertheless, the intuition of memory bound of KV cache when batch size is large makes sense.

  - Is KV compression used for both the draft and target model?

  - What is the baseline for results in Figure 4?

  - As for this lossy KV compression selection method, perhaps consider adding the evaluation that shows the selection in KV compression strategy has advantage (e.g., speedups) over the original speculative decoding.

  - How the framework guide the selection of KV compression method remains unclear to me.

- The applications for the very long contexts prompts in large batch size still need further justification.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces a speculative decoding technique designed to enhance throughput and reduce latency in long-context Large Language Models (LLMs), addressing a common bottleneck in high-batch, memory-bound inference tasks. By optimizing the Key-Value (KV) cache through sparse KV configurations and employing self-speculation, the method improves memory efficiency, making speculative decoding effective for large batch sizes and moderate-to-long sequence lengths. Empirical results on high-performance GPUs, such as A100s and H100s, show up to a 2.51x speedup compared to traditional autoregressive decoding, especially in scenarios with long sequences benchmarking. While promising, the approach lacks comparisons of some baselines, evaluating a limited spectrum of model sets, lack of case study or worse case analysis, and lacks a discussion of the scope/limitation section. If the authors could address the concerns in revision, I would be willing to raise the score.

### Strengths
1. The writing is pretty good and easy to follow.

2. Sufficient experiments conducted on high-performance GPUs (e.g., A100, H100) show up to a 2x speedup compared to autoregressive decoding, demonstrating speculative decoding’s efficiency for long sequences.

3. The authors provide a detailed mathematical analysis showing how speculative decoding can be effective even for large batch sizes in memory-bound regimes, particularly by addressing the KV cache bottleneck.

### Weaknesses
1. Missing discussion and comparisons of reasonable baselines. Although the authors briefly discuss TriForce, which demonstrates the effectiveness of self-speculation with compressed KV, they did not compare it with TriForce in experimentation for some reasons. Besides, there exists some self-speculation work aiming to accelerate LLM inference, e.g.,  Xia et al. [1] and Zhang et al. [2]. Could such existing works be applicable to / integrated with the proposed solution? Please either provide satisfactory reasons why forgiving such baselines or add comparisons in experiments and clear this work's novelty. While the paper references approaches like vLLM, it lacks a side-by-side comparison, especially regarding memory efficiency versus other batch-processing strategies. Including metrics from existing solutions in similar conditions would solidify the paper’s contributions.

[1] SWIFT: ON-THE-FLY SELF-SPECULATIVE DECODING FOR LLM INFERENCE ACCELERATION

[2] Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding

2. Evaluation of a limited spectrum of models. The experimental settings focus on LLaMA series models, yet the performance of proposed methods on other models remains unknown.

3. Lack of Case study or worse case analysis. The paper describes handling variability in token acceptance rates during speculation, but it could provide more details on failure cases or worst-case scenarios where rejection rates could impair throughput significantly. This is especially important for practical deployments in heterogeneous batches where sequence lengths vary widely.

3. Lack of discussion and limitation. Providing a brief discussion of limitations would help clear the scope of this work and make broader impacts. For instance, this work focuses on high-end modern GPUs, so the proposed solution may not perform that well on desktop/low-end GPUs, or even worse at resource-constrained embedded systems. Including tests on more widely accessible hardware (e.g., T4 or V100 GPUs) would demonstrate the method’s practical viability across a broader range of settings. The authors could elaborate more directions in the revisions.

### Questions
See weakness part.

### Soundness
2

### Presentation
4

### Contribution
3
