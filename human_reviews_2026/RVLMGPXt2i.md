# LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future without Generation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 4

## Abstract
Transformer-based large language models (LLMs) rely on key–value (KV) caching to avoid redundant computation during autoregressive inference. While this mechanism greatly improves efficiency, the cache size grows linearly with the input sequence length, quickly becoming a bottleneck for long‑context tasks. Existing solutions mitigate this problem by evicting prompt KV that are deemed unimportant, guided by estimated importance scores. Notably, a recent line of work proposes to improve eviction quality by “glimpsing into the future”, in which a draft generator produces a surrogate future response approximating the target model's true response, and this surrogate is subsequently used to estimate the importance of cached KV more accurately. However, these approaches rely on computationally expensive draft generation, which introduces substantial prefilling overhead and limits their practicality in real-world deployment. To address this challenge, we propose LookaheadKV, a lightweight eviction framework that leverages the strength of surrogate future response without requiring explicit draft generation. LookaheadKV augments transformer layers with parameter‑efficient modules trained to predict true importance scores with high accuracy. Our design ensures negligible runtime overhead comparable to existing inexpensive heuristics, while achieving accuracy superior to more costly approximation methods. Extensive experiments on long-context understanding benchmarks, across a wide range of models, demonstrate that our method not only outperforms recent competitive baselines in various long-context understanding tasks, but also reduces the eviction cost by up to $14.5$×, leading to significantly faster time-to-first-token. Our code is available at https://github.com/SamsungLabs/LookaheadKV.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes to use LoRA and learnable special tokens during the attention step. These components assist the prefill phase in determining which parts of the KV cache should be retained and which can be evicted. The approach ensures accuracy and model performance without significantly increasing decode overhead.

### Strengths
1. The problem is well defined. The paper addresses a clearly defined and critical problem: how to efficiently predict the attention patterns of future tokens in the input context for KV Cache eviction. 
2. Concise Solution. The proposed solution is concise and seems to work well according to experiments.
3. Comprehensive Experiments. The paper is supported by a thorough experimental evaluation. The accuracy tests cover multiple models and diverse scenarios, including long-input and long-output tasks. The efficiency analysis provides both theoretical cost models and empirical TTFT measurements.

### Weaknesses
1. Limited Practicality for Long-Form Generation: While LKV outperforms baselines on long-output tasks, its absolute performance still degrades significantly (as shown in Section 4.3, "Long-Form output Evaluation"). 
2. Prefill-Only Eviction Limits Applicability: The method's focus exclusively on prefill-stage eviction is a major limitation. Many SOTA reasoning models rely on a large number of decoding steps, during which the KV cache continues to grow. Ignoring decoding phase significantly reduces its overall utility in these increasingly important use cases.
3. Performance Advantage is Concentrated in Extreme Low-Budget Scenarios: The most significant performance gains of LKV over strong baselines like LAQ are demonstrated in extremely low-budget settings (e.g., a cache budget of 64 or 128). These settings, while useful for stress-testing, are not always representative of practical deployments where budgets might be more generous. In higher-budget scenarios, the performance gap narrows considerably, questioning the method's marginal benefit when resources are less constrained.

### Questions
1. Could the authors elaborate on the detailed training cost for proposed LKV? Although this process is a one-time effort, it currently appears that training is required for each different model.
2. The training process utilizes greedy decoding to generate ground-truth. But in practice, the most popular sampling strategy is not always greedy decoding. How does LKV's performance generalize to stochastic decoding strategies such as beam-search or top-p? Is there a risk that the learned modules overfit to the deterministic attention patterns of greedy decoding, leading to a performance drop when the generation path becomes less predictable?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes LookaheadKV, a training-time method that learns to predict token importance for KV cache eviction without running a draft generation. During prefilling, the model appends a small number of learnable lookahead tokens and activates a lightweight LoRA only on these tokens. The attention induced by these lookahead tokens is trained to match the attention that would be observed if the future response were known. At inference time, the method uses the learned mechanism to rank prompt tokens and evict low-importance items from the KV cache. Experiments on long-context benchmarks report accuracy that is competitive or better than recent baselines, with low overhead at prefilling and no added cost during decoding.

### Strengths
The central idea is interesting: use a pretrained module with a small LoRA applied on a short learnable window to substitute for running an explicit generate step. This reduces prefilling cost while still capturing signals about future attention. The method is simple to integrate, has low inference overhead, and shows consistent gains in low-budget regimes. The empirical results cover several model families and tasks, and the latency accounting is practical.

### Weaknesses
The presentation has gaps. I can understand the LoRA training and the alignment objective, but I do not understand precisely how the lookahead embeddings are obtained. It is not fully clear whether these lookahead tokens are new learned embeddings, adapted from existing vocabulary embeddings, or derived from another module. The paper would benefit from a precise definition of the parameterization, initialization, and update path of these embeddings.

### Questions
1. How long does training take in time for the main models, and on what hardware budget. Do you believe the small TTFT improvements and decoding-time stability justify this training cost. How do you view this trade off in practical deployments.
2. Have you tried combining the importance score from LookaheadKV with a SnapKV score. I am thinking of using both the lookahead tokens and the last few prompt tokens as the observation window, so that the score contains both prompt-side and generate-side information. Do you expect this to help, and can you share preliminary results if any.
3. For extreme extrapolation, for example the 128k context window of Llama 3.2, can you report RULER results at 64k and 128k. Even a reduced subset would be helpful to show the scaling trend.

### Soundness
3

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
5

### Summary
LookaheadKV proposes a lightweight, training-based framework for predicting token importance during prefill to enable accurate KV-cache eviction without draft generation or additional inference steps. The key idea is to introduce a small set of learnable “lookahead tokens”, activated only during prefill, whose attention interactions approximate the model’s future decoding behavior. These tokens are enhanced by LoRA-based low-rank adapters that learn to predict future attention distributions via a KL-divergence ranking objective, requiring no model modification or retraining of base weights. LookaheadKV achieves near-oracle token importance estimation, outperforming SnapKV, SpecKV, and StreamingLLM across benchmarks.

### Strengths
1. Strong empirical results with broad coverage: Outperforms baselines on multiple benchmarks
2. Comprehensive ablations and analysis: Includes detailed experiments on number of lookahead tokens, LoRA layer coverage, training context length, and budget scaling

### Weaknesses
1. Unclear generalization to stochastic decoding: The method is trained and evaluated primarily under greedy decoding, assuming deterministic next-token prediction. It remains untested under sampling-based decoding
2. Absence of multi-turn or instruction-following benchmarks: Evaluations mostly use single-turn or synthetic long-context datasets. Multi-turn reasoning or conversational tasks, where future tokens are highly context-dependent, are missing
3. Lack of Subtask-Level Analysis on LongBench: The paper evaluates on LongBench but reports only averaged or composite scores, omitting per-subtask breakdowns

### Questions
1. Would LookaheadKV’s importance predictions remain accurate when decoding randomness changes token trajectories?
2. How to select retained tokens with lookahead tokens. 
3. What is the breakdown scores for LongBench? How Lookahead performs on different tasks. 
4. Do different LLMs need different LoRA?
5. How does this method differ from works such as “Learning to Compress Prompts with Gist Tokens”, which train models to summarize and compress prompt tokens to reduce KV-cache size, rather than to learn predictive lookahead behavior for future token importance?

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
3

### Summary
The paper introduces a learned KV cache management framework, LookaheadKV, that selectively evicts tokens during prefill stage by predicting future token importance using a small set of trainable tokens and LoRA adapters that are only active for these special tokens. The approach aims to achieve the accuracy benefits of similar draft-based approaches like LAQ and SpecKV, without their latency penalty from generating a draft response. LookaheadKV is trained with a normalized KL divergence loss and use the predicted importance to select top-$k$ KV cache entries. Experiments across various models show that LookaheadKV outperforms baselines and faster than draft-based approaches.

### Strengths
1. Presentation is overall clear and problem is motivated by the accuracy overhead trade-off from other draft-based methods. The latency problem is directly tackled through a learned approach. LookaheadKV provides an interesting application of parameter-efficient fine-tuning methodology for KV cache purposes.
2. Experiments performed on various model families and sizes, and also long-context benchmarks. Inclusion of LongProc provides some support that LookaheadKV can preserve the model's reasoning ability beyond recalling isolated facts.
3. Demonstrates efficiency by showing that method's impact on TTFT is almost negligible and competitive with SnapKV, while being significantly fast than both SpecKV and LAQ.

### Weaknesses
1. All ground-truth attention scores are generated using greedy decoding. As a result, lookaheadKV is effectively trained to predict attention patterns of a greedy future only, which may limit its applicability. In many practical applications, LLM inference is not deterministic and benefits immensely from stochastic sampling strategies such as temperature scaling to produce more diverse outputs. The attention patterns can differ substantially, which could lead to degradation in eviction quality when stochastic decoding is used such as in complex reasoning intensive tasks with longer thoughts.
2. Training lookaheadKV is expensive, since it requires for each model trained, to first generate a number of decoded tokens to use as ground truth.
3. The method defines ground-truth importance based on the attention of response tokens generated. However, attention scores do not always correspond to actual causal influence, so LookaheadKV may evict tokens whose attention is low but whose semantic contribution is high.
4. The need for requiring both lookahead tokens and lookahead lora is not made clear. While the ablation studies show the benefits, some additional justification is needed.

### Questions
1. Does inclusion of LoRA adapters enable using shorter length lookahead tokens? Could you achieve similar or better performance by simply increasing the number of lookahead tokens (e.g., n=64 or n=128) without LoRA adapters? What is the fundamental trade-off between lookahead token count and LoRA expressiveness in your framework?
2. How does lookaheadKV's performance vary when using stochastic decoding strategies at inference time?
3. The RULER evaluation (Figure 4, bottom) only reports results at budget=128, while LongBench shows multiple budgets (64, 256, 1024). How does LookaheadKV perform on RULER at these other budget settings, and are the performance advantages consistent across different budget constraints?
4. Results show from Table 3 that at n=32, performance mostly saturates. What happens to both performance and latency overhead when number of lookahead tokens exceeds 32 (e.g., 64, 128, 256)?

### Soundness
2

### Presentation
3

### Contribution
2
