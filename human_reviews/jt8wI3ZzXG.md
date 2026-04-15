# Speculative Streaming: Fast LLM Inference without Auxiliary Models

- Decision: Reject
- Scores: 3, 8, 6, 5

## Abstract
Speculative decoding is a prominent technique to accelerate large language model inference by leveraging predictions from an auxiliary draft model. While effective, in application-specific settings, it often involves fine-tuning both draft and target models to achieve high acceptance rates. As the number of downstream tasks grows, draft models add significant complexity to inference systems. Recently
several single model architectures viz. Medusa have been proposed to speculate tokens in non-autoregressive manner, however, their effectiveness is limited due to lack of dependency between speculated tokens. We introduce a novel speculative decoding method that integrates drafting within the target model by using Multi-stream attention and incorporates future token planning into supervised finetuning objective. To the best of our knowledge, this is the first parameter-efficient approach that scales well with an increasing number of downstream tasks while enhancing downstream metrics and achieving high acceptance rates, attributable to the interdependence among the speculated tokens. Speculative Streaming speeds up decoding by 1.9 - 3X in a diverse set of tasks, such as Summarization, Structured Queries, and Meaning Representation, while improving generation quality and using ∼10000X fewer extra parameters than alternative architectures, making it ideal for resource-constrained devices. Our approach can also be effectively deployed in
lossless settings for generic chatbot applications that do not necessitate supervised fine-tuning. In such setups, we achieve 2.9 - 3.2X speedup while maintaining the integrity of the base model’s output.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces Speculative Streaming, a method that integrates multiple soft embeddings before the $N-N_s$ multi-head attention layers of target LLMs and fine-tunes LLMs using LoRA, enabling the parallel prediction of these additional positions. Compared with prior methods like Medusa which modifies the MLP layers, Speculative Streaming focuses on modifications of LLM attention, providing a unique and investigative perspective on speculative decoding. The approach’s efficacy is validated through evaluations across various tasks and LLMs, achieving a 1.9 to 3x overall speedup.

### Strengths
1. The paper presents a novel investigation to speculative decoding by modifying the attention mechanism. Compared to prior methods like Medusa and Eagle, the authors claim that Speculative Streaming requires fewer parameters while achieving superior speedup performance.
2. This work provides an insightful explanation for Speculative Streaming’s effectiveness, which is illustrated in Section 4.1.2. The findings suggest that by enabling LLMs to attend to future tokens, the attention mechanism would enhance its ability to anticipate and plan for the nearest next token, thus improving generation quality. This result holds promise for future research directions.
3. The authors provide extensive experimental details and analysis in the Appendix, covering aspects like the tree drafting process, training costs, parameter overhead, and etc. These additions effectively address some of my initial concerns.

### Weaknesses
1. **The original LLM distribution is altered**. Unlike most speculative decoding such as [1], Medusa-1[2], and Eagle[3] that maintains the original output distribution of target LLMs, Speculative streaming (SS) modifies the target LLM's attention, enabling the main stream to attend to speculative streams and thus altering the original output distribution. This modification would largely impact the application of this paradigm since the generation quality may not be guaranteed. Although comparative performance is shown in Tables 1 and 2, SS performance is likely sensitive to fine-tuning procedures and evaluation domains.
2. **Lack of Experimental Detail**. In the abstract and results of Section 4.1.1, the authors claim that in **lossless** settings for generic chatbot applications that **do not necessitate supervised fine-tuning**, SS could achieve 2.9 - 3.2X speedup while maintaining the integrity of the base model’s output. However, I found no detailed description of the experimental setup for this configuration. Given that SS introduces stream embeddings and LoRA tuning, clarification on how SS accomplishes lossless speedup without fine-tuning is necessary. Detailed settings should be provided. Besides, could SS maintain the same output distribution as the original LLM?
3. **Some demonstrations are confusing**. In Section 3.1.4 (Training) and Appendix D, the authors mention using LoRA to train the model with the objective in Equation (5). However, LoRA-tuned parameters are not highlighted in Figure 2, which suggests that only stream embeddings are tuned. A clear discussion is needed on which parameters are subject to LoRA tuning—does LoRA tuning apply to all multi-stream attention (MSA) layers?
4. **Computation overhead of SS**. In Table 1, SS is shown to require fewer parameters than Eagle and Medusa. However, SS appears to incur higher computation overhead in drafting than Medusa, as the extra stream representations require forward passes through the last $N_s$ MSA layers, including both the MLP and attention layers. This adds an additional computational burden to the whole framework.
5. **Clarification of Multi-Stream Attention**. The term "multi-stream attention" may be somewhat misleading. SS resembles a combination of soft-prompting (via stream embeddings) and LoRA tuning, with the drafting process involving not only MSA layers but also the MLP layers in the last $N_s$ layers. Consideration of alternative wording might improve clarity.

[1] Fast Inference from Transformers via Speculative Decoding. Leviathan et.al. ICML 2023.

[2] Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. Can et.al. ICML 2024.

[3] EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty. Li et.al. ICML 2024.

### Questions
Most of my primary concerns are outlined in the weaknesses section above. Here are additional minor concerns:

- In line 107, the phrase "Moreover, using a dedicated layer leads to significant parameter overhead" is somewhat misleading. Compared to Medusa, which incorporates multiple MLP heads, Eagle requires only one autoregressive head, thereby **reducing parameter overhead**, as shown in Table 1.
- The authors investigate a tree drafting strategy in Section 3.1.1 and Appendix A.1. Compared to Medusa’s fixed tree structure of 64 candidate tokens, SS increases the number of candidates to a batch size of $(1+\gamma) \cdot \left(1+\sum_{g=1}^\gamma k^g\right)$ (where $\gamma=3$ and $k=3$), while limiting exploration breadth and depth to 3. Have the authors compared this tree implementation with that of Medusa?
- In line 321, the authors mention that "For generic chatbot-like settings, where fine-tuning is not required, we train speculative streams while keeping the base model frozen as noted in Section 3.1.4." The specific settings for this configuration should be clarified: is LoRA tuning still applied, or are only the stream embeddings tuned?
- Please provide a more detailed explanation of the CR (call reduction) ratio metric. Does it correspond to the mean number of generated tokens per step?
- The reporting of extra parameters in Table 1 appears imprecise, as Medusa-2 requires tuning of the original LLM. Comparisons should focus on the number of tuned parameters rather than extra parameters.
- In line 403, the authors claim that "the generation quality of Speculative Streaming consistently outperforms that of next-token prediction-based fine-tuning, positioning it as a compelling alternative to LoRA-based fine-tuning approaches." However, no comparison with next-token prediction-based fine-tuning methods (such as LoRA tuning) is provided, which makes this an overstatement.

### Soundness
3

### Presentation
2

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
This paper introduces Speculative Streaming, a novel approach to make LLM inference faster without needing extra models. It combine speculation and checking in one model with multi-stream attention and put future token planning in fine-tuning. Speculative Streaming make inference 1.9-3X faster for different tasks, make generation better, and use ~10000X less extra parameters than other methods. It good for devices with not much resource and can be used for chatbots without losing quality.

### Strengths
1. The novelty is good. This paper presents an innovative approach to speculative decoding that eliminates the need for auxiliary draft models, addressing a significant limitation in existing methods.
2. The method is efficient. Speculative Streaming demonstrates impressive speedups (1.9-3X) while using substantially fewer parameters than competing approaches, making it highly efficient and suitable for resource-constrained environments. The speedup is consistent  across diverse tasks, including summarization, structured queries, and meaning representation, indicating its broad applicability.
3. The method works well for both fine-tuned and no-loss settings, so it useful for many applications, from specific tasks to general chatbots.

### Weaknesses
The method was tested only on model with a size < 7B parameters, a more detailed analysis of how it scales with very large models (e.g., 100B+ parameters) would be valuable. And a more in-depth theoretical analysis of why Speculative Streaming works would be helpful, particularly in terms of the interplay between multi-stream attention and future token planning.

### Questions
1. How does Speculative Streaming perform on very long sequences (e.g., 10,000+ tokens)? Does the efficiency gain remain consistent?
2. Can you provide more insights into how the multi-stream attention mechanism affects the model's understanding of context and coherence in generated text?
3. How sensitive is the method to hyperparameters, particularly the number of speculative streams? Is there an optimal range for different model sizes or tasks?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents speculative streaming, a parameter-efficient method for speculative decoding. Speculative streaming leverages multi-stream attention to integrate representations of future token predictions directly into the language modeling process. Unlike traditional approaches, it uses a multi-token prediction strategy during training rather than the standard next-token prediction. In evaluations, the proposed method demonstrated superior performance in both latency and generation quality compared to several baseline methods.

### Strengths
**Parameter Efficiency**: The proposed multi-stream attention method effectively reduces the parameter requirements for speculative decoding.


**GPU Utilization**: By design, the multi-stream attention architecture achieves higher GPU kernel utilization, providing a notable advantage over popular methods like Medusa.


**Evaluation Result**: Speculative streaming requires significantly fewer parameters than other speculative decoding approaches, enhancing both efficiency and practicality.

### Weaknesses
**Model Evaluation**: Since the proposed training method modifies the original model output, a thorough evaluation is essential to verify the claimed integrity of the target model. While model performance is reported on the SqlContext, Dialogsum, and e2e-nlg datasets using metrics like exact match and ROUGE score, these evaluations may not fully capture the performance of a modern LLM, such as LLaMA.

**Formatting Issues**: The citation format does not meet ICLR requirements, and Figure 17 appears distorted. The titles in the headers are not rendered.


**Lack of Throughput Experiments**: The experiment is conducted on an Nvidia A100 GPU with a batch size of 1, and an ablation study with batch sizes of 2 and 4 is included in the appendix. However, using an A100 for inference with a batch size of 1 is uncommon in practice. An evaluation of throughput (at maximum batch size) compared to other methods is necessary to demonstrate the practical performance of speculative streaming.


**Lack of Motivation**: The paper emphasizes parameter efficiency in the speculative process, though a draft model may contain as little as 1% of the parameters of the target model. The significance of this efficiency focus is not shown and would benefit from further justification.

### Questions
1. In Figure 3 and Figure 4, do you achieve a 3-time speedup in comparison to 2-model SD?
2. What prompts are used during the evaluations?
3. Are the target model weights frozen in all experiments? Why do you choose to “fine-tuning only LoRA parameters of MSA layers”?
4. You mentioned “Furthermore, it is important to highlight that the generation quality of Speculative Streaming consistently outperforms that of next-token prediction-based finetuning, positioning it as a compelling alternative to LoRA-based fine-tuning approaches.” in line 405.” What is the evidence of this?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a non-autoregressive draft embedding to accelerate speculative decoding.

### Strengths
This paper is generally well-written and easy to follow.

The experiments of this paper are detailed and the authors test different llms, which is good.

### Weaknesses
1) The method of this paper is very similar to BiTA, https://arxiv.org/html/2401.12522v2.

2) Due to the params of LLM are also updated, I just wondering if the training dataset will introduce bias. For example, if the training set is SQL, the metric of speculative streaming may be very good but fails at other tasks, e.g., machine translation. Can the authors add lossless acceleration result like eagle?

### Questions
see cons

### Soundness
2

### Presentation
3

### Contribution
2
