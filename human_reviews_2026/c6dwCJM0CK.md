# Coverage-Driven KV Cache Eviction for Efficient and Improved Inference of LLM

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 4

## Abstract
Large language models (LLMs) excel at complex tasks like question answering and summarization, thanks to their ability to handle long-context inputs. However, deploying LLMs is costly, not only due to the high computational demands of quadratic complexity of self-attention and auto-regressive generation, but also because of the significant memory overhead required for storing the key-value (KV) cache during inference. To reduce the memory cost, existing KV-cache eviction strategies leverage the sparsity in attention to selectively store a subset of tokens. While reducing the memory footprint, such approaches show a considerable drop in performance, especially in tasks that require long-context reasoning. We identify that the drop in performance is linked to a reduction in the coverage of unique tokens. Additionally, we theoretically show that reduced coverage limits the mutual information between inputs and outputs, thereby impairing predictive accuracy. To this end, we introduce K-VEC, a novel coverage-aware KV-cache eviction strategy that prioritizes token coverage while evicting tokens in the cache. K-VEC introduces a cross-head and a cross-layer coverage module to enhance token retention across attention heads and model layers, mitigating performance degradation caused by low coverage. Evaluated on 16 LongBench subsets, K-VEC exhibit up to 10.35 points improvement over the existing methods under the same eviction rate and memory constraint. Comprehensive evaluations validate the effectiveness of our approach and demonstrate its potential for efficient LLM deployment in resource-constrained settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper identifies that performance degradation in KV-cache eviction stems from reduced token coverage. The authors propose K-VEC, a coverage-aware eviction strategy with cross-head and cross-layer coverage modules that prioritizes retaining unique tokens.

### Strengths
The paper is clearly written and straightforward.

### Weaknesses
1. The attribution of performance degradation to coverage is not sufficiently convincing. I personally believe that the pattern skewed towards the end, as mentioned in Figure 2, is actually the key factor rather than coverage itself. This is a well-known issue in SnapKV-like approaches that use a local window for score computation. This problem has been demonstrated in related work and addressed through Global-Local Importance strategies.

Reference:

EMS: Adaptive Evict-then-Merge Strategy for Head-wise KV Cache Compression Based on Global-Local Importance

2. The experimental setup is inconsistent and raises concerns. For instance, while baselines such as Pyramid, SnapKV, Ada-Pyramid, Ada-SnapKV, HeadKV, and GemFilter exist, most tables (e.g., Tables 1, 2, and 3) omit certain baselines. In the Qwen2.5-7B experiments, only SnapKV is retained. Could the authors explain the rationale behind this experimental arrangement?

3. The algorithm involves numerous hyperparameters, and the results demonstrate insufficient robustness. Although the authors provide ablation studies, the results show significant sensitivity. The main experiments report an average improvement of less than two points, yet the ablation studies show that any parameter change causes fluctuations of approximately one point. This substantial impact on results raises doubts about the algorithm's practical robustness.

4. The algorithm appears to severely compromise prefill efficiency. The experiments show that prefill throughput drops from 5,672 to 3,440-a reduction of 39%. The authors' argument that "the prefill stage is a one-time operation" is unconvincing. On the contrary, prefill time is a critical overhead, and existing KV cache compression methods typically preserve this crucial efficiency with minimal impact.

### Questions
The questions raised are detailed in the weaknesses above. 

Additionally, the authors need to present more convincing arguments, analysis, or experimental evidence to establish that coverage is indeed the root cause of performance degradation, rather than simply the problem of local scoring induced by local window mechanisms. Optimization should target the fundamental cause rather than a proximate symptom in the causal chain. This is my primary concern and the basis for my rejection.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a new perspective by examining KV cache eviction through the  token coverage. Based on this insight, the authors propose K-VEC, a coverage-aware eviction strategy that incorporates cross-head and cross-layer coverage modules to preserve diverse tokens across attention heads and model layers. Experimental results demonstrate that K-VEC achieves consistent improvements across LongBench.

### Strengths
1. The paper proposes a novel perspective by examining the problem from the token coverage.

2. K-VEC demonstrates consistent and effective improvements on LongBench.

### Weaknesses
1. In my view, the authors' claim that coverage is the root cause of performance degradation is insufficiently justified. If the issue stems primarily from attention being skewed towards the end, why not directly optimize token selection across different positions rather than focusing on coverage?

2. Following the above point, the theoretical analysis appears insufficient. The analysis lacks grounding in the underlying mechanisms of LLMs and seems hastily constructed, offering limited insight. It fails to convincingly establish why coverage is the fundamental cause of performance degradation.

3. The proposed algorithm introduces numerous hyperparameters, raising concerns about hyperparameter sensitivity.

4. Expanding the observation window from O to O' cannot be considered a genuine improvement. Previous work selected observation windows primarily to control the computational overhead of the compression algorithm itself. For instance, in the efficiency tests, K-VEC significantly increases this overhead, which would severely impact TTFT  in practical deployment-a critical concern in long-context LLM serving.

5. The experimental evaluation is insufficient, as experiments are conducted only on Llama-3.1-8B. Additionally, why do LongBench and Needle-in-a-Haystack use different baselines? I recommend providing complete results across all baselines for both benchmarks.

### Questions
n/a

### Soundness
2

### Presentation
2

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
This paper proposes K-VEC, a coverage-driven KV cache eviction policy that reduces redundant token selection across heads and layers. It expands attention for low-focus heads and prioritizes under-represented tokens across layers. Based on benchmark LongBench with Llama-3.1-8B-Instruct, K-VEC improves performance under tight KV budgets with minor prefill overhead.

### Strengths
1. Give empirical evidence that existing strategies exhibit recency bias and low coverage.
2. Well-motivated decomposition into cross-head vs. cross-layer effects.
3. Improvements are consistent across multiple memory budgets.

### Weaknesses
1. Coverage as a general assumption, the paper’s own limitation section acknowledges that many tokens “contain no relevant information” and that full coverage is not beneficial. This weakens the universality of the central premise. It is better to demonstrate failure modes where coverage hurts.
2. The paper does not clarify whether SnapKV’s sliding-window smoothing is applied before STD computation. Since STD is noise-sensitive and may affect head ranking stability, an analysis with and without smoothing is needed to assess robustness.
3. LongBench includes many tasks where coverage matters weakly. To support the hypothesis, the paper should include more information dense tasks, such as NeedleBench.
4. The key novelty is STD-based head reweighting, yet the paper lacks comparisons against random, top-STD, or fully entropy-based selection. Without these baselines, it is unclear whether the signal is truly informative or arbitrary.

### Questions
1. Did we smooth attention scores before STD computation? What is the effect without smoothing?
2. How many heads actually classify as low STD in practice (distribution across layers)? Is this model dependent or a general observation.
3. Can coverage ever degrade performance on tasks where few key spans dominate relevance?
4. Algorithm 1 returns updated eviction scores but does not explain how they are thresholded, masked, or applied over time. Practical guidance for stability—especially regarding the γ focus term—is missing.
5. Does early-layer coverage matter more? How consistent are the selected low-STD heads across layers and prompts? Are they stable or input-dependent?

### Soundness
2

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
The idea of promoting coverage across heads and layers for KV eviction is interesting and makes intuitive sense, and the results look reasonable in those settings.

My main concern is that, if I understand correctly, the method is stated as being applied during prefilling. However, in practice, the cache is still fully built first, and pruning only occurs once after the prefill stage completes. Since prefilling is parallel, there is no actual token-by-token eviction taking place. This makes the approach less general and not directly applicable to streaming or incremental scenarios.

The evaluation is also relatively narrow. All experiments are based on a single LLaMA model, while KV-cache behavior can vary significantly across different architectures such as Qwen, DeepSeek, and Mistral. Without broader evidence, it’s difficult to be fully convinced that the method will generalize well.

Finally, the design feels more aligned with retrieval style tasks than reasoning tasks. For math, where only a small part of the prompt matters, “coverage” may not be the right signal.

Overall, the idea is promising, but the one shot pruning, limited model coverage, and unclear benefits beyond retrieval tasks make me want to see more evidence before fully buying in.

### Strengths
1. The core idea is simple and intuitive, preserving diverse parts of the prompt instead of just the highest-attention tokens. 
2. The method is easy to plug into existing models and it shows steady gains under tight KV budgets. 
3. The paper grounds the idea in a clear empirical observation.

### Weaknesses
1. The evaluation is mostly tied to Llama model, so it’s hard to tell if the method really generalizes across architectures like Qwen, DeepSeek. 
2. The paper says eviction happens during prefill, but in practice the full KV cache is still built first and pruning only happens once, which limits the method in streaming or incremental settings. 
3. It’s unclear how well it transfers to reasoning or math problems where only a small part of the prompt matters.

### Questions
1. How would this approach behave on tasks where only a small portion of the prompt matters, like math or reasoning, where broad coverage may not be as meaningful?
2. Do the authors expect the same behavior on Qwen/DeepSeek architectures, which often have different attention patterns? Any reasons to think the coverage heuristic generalizes beyond Llama?
3. Since coverage statistics are only computed once after prefill, have the authors looked at scenarios where the output is extremely long, or multi-turn?

### Soundness
2

### Presentation
3

### Contribution
3
