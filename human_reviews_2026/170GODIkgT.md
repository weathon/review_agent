# SpecExtend: A Drop-in Enhancement for Speculative Decoding of Long Sequences

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
Speculative decoding is a widely used technique for accelerating inference in large language models (LLMs), but its performance degrades as input length grows, with significant drops even at moderate lengths. Yet, this early degradation has remained largely underexplored. We introduce SpecExtend, a drop-in enhancement that improves speculative decoding on long sequences without additional training. SpecExtend integrates efficient attention mechanisms such as FlashAttention and Hybrid Tree Attention to accelerate prefill and verification steps. To improve both draft accuracy and speed on long inputs without retraining, we propose Cross-model Retrieval, a novel KV cache eviction strategy that leverages the target model’s attention scores to dynamically select relevant context for the smaller draft model. Extensive evaluations show that SpecExtend accelerates speculative decoding by up to 2.84× on 16K-token long summarization and up to 3.86× on long reasoning, while preserving the short-input performance of state-of-the-art frameworks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents SpecExtend, a training-free, drop-in enhancement for speculative decoding that significantly improves inference efficiency on long sequences in LLMs. The proposed method integrates efficient attention mechanisms (FlashAttention for prefill and Hybrid Tree Attention for verification) and introduces Cross-Model Retrieval, a novel KV cache eviction strategy that dynamically updates the draft model’s cache using the target model’s attention scores.

### Strengths
1. Innovative cache strategy: The Cross-Model Retrieval method effectively improves both speed and accuracy without retraining.

2. Strong empirical results: Demonstrates consistent, significant speedups and robustness across models and tasks.

3. Practicality and generality: Works as a plug-and-play enhancement compatible with existing speculative decoding frameworks.

### Weaknesses
1. The proposed CMR mechanism feels somewhat lightweight. It mainly relies on reusing attention scores for cache selection, which may limit novelty compared to prior works.

2. The paper uses Hybrid Tree Attention, which appears to originate from LongSpec. It would be helpful to clarify whether this component has been modified or is directly adopted.

3. Experiments are primarily evaluated up to 16K tokens, which might still be short for assessing scalability in modern LLMs. It would strengthen the work to include results on 32K or 64K contexts.

### Questions
1. The Hybrid Tree Attention component appears to originate from prior work (LongSpec). Could the authors clarify whether any modification or optimization is introduced here, or if it is directly adopted?

2. Experiments are mainly conducted up to 16K tokens, which may not fully reflect long-context behavior for modern LLMs. Have the authors tried evaluating on 32K or 64K inputs to assess scalability and generalization to truly long contexts?

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
This paper introduces SpecExtend, a drop-in enhancement for speculative decoding on long sequences that requires no additional training. It accelerates prefill and verification by integrating efficient attention mechanisms (e.g., FlashAttention, Hybrid Tree Attention). To improve draft accuracy and speed on long inputs without retraining, it proposes Cross-model Retrieval, a KV-cache eviction strategy that uses the target model’s attention scores to select relevant context for the smaller draft model.

### Strengths
1. This paper presents a practical, training-free augmentation that combines hybrid tree attention with KV-cache eviction to speed up speculative decoding on long inputs.
2. This paper proposes Cross-model Retrieval, leveraging target-model attention to guide KV compression for the draft model, aiming to improve both drafting accuracy and end-to-end latency.

### Weaknesses
1. The contribution reads as an engineering integration of known components—hybrid attention and KV cache eviction—for long-sequence acceleration, with limited new algorithmic insights beyond composing these pieces.
2. The novelty of Cross-model Retrieval appears limited: similar ideas using target-model attention to prune draft-side redundancy have been explored (e.g., works in EMNLP-2025-SpecVLM that analyze long-context drafting latency and use the verifier’s attention maps to guide pruning). Although the application domain differs, the mechanism seems closely related.
3. The method seems difficult to apply to the decoding phase with FlashAttention (as presented), and reported acceptance rate gains at 16k context over the streamingLLM baseline are small (≈+0.5), which weakens the practical impact claims.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the performance degradation of speculative decoding on moderately long sequences, a problem that occurs even before the KV cache becomes the primary system bottleneck. The authors introduce SpecExtend, a training-free, drop-in enhancement designed to solve this issue. The method consists of two main components: (1) integrating efficient attention mechanisms like FlashAttention and Hybrid Tree Attention to accelerate the prefill and verification stages, and (2) a novel KV cache eviction strategy for the draft model called Cross-model Retrieval (CMR). CMR leverages the attention scores from the larger target model's verification step to dynamically select and retain the most relevant context chunks for the smaller draft model. Extensive evaluations show that SpecExtend significantly accelerates speculative decoding on long summarization (up to 2.84x) and long reasoning (up to 3.86x) tasks while preserving the strong short-input performance of existing frameworks.

### Strengths
1. The paper successfully identifies and tackles a specific, important, and largely underexplored problem: the early performance drop of speculative decoding in the moderate-length regime. This is a valuable contribution that moves the field beyond focusing solely on the extreme-length memory bottleneck.

2. The core idea of Cross-model Retrieval is novel and intuitive. Using the more powerful target model as an "oracle" to guide the context management of the smaller, less capable draft model is a clever, training-free way to improve draft accuracy where it is most needed.

3. The proposed solution is practical and immediately applicable. As a drop-in enhancement that requires no retraining of the draft or target models, SpecExtend can be integrated into existing speculative decoding pipelines, offering a low-friction path to significant performance gains.


4. The experimental results are comprehensive and demonstrate substantial, consistent speedups on relevant long-context tasks. The method's ability to boost performance on both off-the-shelf and highly optimized draft models (like EAGLE) showcases its robustness and wide applicability.

### Weaknesses
1. A significant concern is the reliance on the target model's attention scores as an objective measure of context importance. Attention mechanisms are known to exhibit idiosyncratic model-specific behaviors, such as "attention sinks," where high scores are assigned to initial tokens regardless of their semantic relevance. By using these scores to guide the draft model's cache, there is a risk that CMR simply teaches the draft model to replicate the target model's attentional biases rather than focusing on the truly important context. This could create a system that is highly tuned to the target model's quirks, potentially limiting the robustness and generalizability of the approach.


2. The effectiveness of the Cross-model Retrieval strategy seems highly dependent on the assumption that attention patterns are transferable and useful from a large, complex model to a much smaller, architecturally simpler one. This may hold true for models within the same family (e.g., Llama-7B and Llama-68M), but it is questionable how well this would work if the draft and target models were from entirely different architectural families, potentially leading to a "biased" or unhelpful retrieval signal.

### Questions
1. The paper focuses on models from the same family. How do you expect CMR to perform if the target and draft models are from different architectural families with potentially very different attention patterns.

2. Is the  Cross-model Retrieval strategy better than the RAG used in RAPID [1]? It seems to be a very related work and should make discussions.

[1] RAPID: Long-Context Inference with Retrieval-Augmented Speculative Decoding

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SpecExtend, a training-free, drop-in enhancement for speculative decoding that targets performance degradation in moderate-length and long-sequence scenarios. It integrates efficient attention mechanisms (FlashAttention and Hybrid Tree Attention) with a novel Cross-model Retrieval strategy that uses the target model’s attention scores to dynamically update the draft model’s KV cache, thereby improving both draft speed and accuracy without retraining. Experiments on long summarization and long reasoning tasks show up to 2.84× and 3.86× speedups, while preserving short-input performance and broad compatibility with existing speculative decoding frameworks.

### Strengths
1. Identifies a valuable and underexplored problem: the sharp performance drop of EAGLE-based speculative decoding on long sequences.
2. Presents a rich and comprehensive experimental evaluation across diverse tasks, model variants, and input lengths, with extensive comparisons to multiple strong baselines.
3. The proposed Cross-model Retrieval method is integrated into an overall framework that remains training-free, broadly compatible, and achieves notable speedups without sacrificing short-input performance.

### Weaknesses
1. The use of efficient attention is mainly an implementation detail rather than a core contribution, which makes the baseline comparisons somewhat unfair.
2. Lacks discussion and empirical analysis on the choice of using the target model’s last-layer attention for draft KV retrieval.

### Questions
1. Although the use of efficient attention mechanisms helps address the quadratic complexity of standard attention in long-context settings, the baselines in the comparison should also adopt these techniques for fairness, since this is not the primary factor behind EAGLE’s performance drop on long sequences. Furthermore, I recommend including the standard+StreamingLLM setting in the main results table, reporting both its end-to-end speedup and acceptance length. One intuitive explanation for the long-sequence accuracy drop is that the model has never seen such long contexts during training, leading to position ID generalization issues; StreamingLLM offers a direct approach to mitigate the training–inference mismatch in position IDs.
2. What is the rationale for using the target model’s last-layer attention to guide draft KV retrieval? Is there empirical evidence or analysis comparing different layer choices, and does the last layer consistently yield the best results?
3. Could the retrieval frequency be made adaptive depending on how much the target model’s attention distribution changes during decoding? It would be informative to see how different update frequencies affect acceptance length and end-to-end acceleration.

I believe this paper addresses an important and practically relevant problem, so if the authors can adequately address these concerns, I would be inclined to raise my score.

### Soundness
3

### Presentation
3

### Contribution
3
