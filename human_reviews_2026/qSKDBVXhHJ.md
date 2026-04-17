# EffComp: Efficient Prompt Compression via Hybrid Reinforcement & Supervised Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 2, 4

## Abstract
Prompt compression is aimed at reducing input prompt lengths to enable cheaper and faster LLM predictions. However, existing prompt compression methods are often limited by modest compression gains, a risk of hallucination, and/or high compression latency. This paper proposes EffComp, an $\textbf{Eff}$icient prompt $\textbf{Comp}$ression framework using a hybrid reinforcement and supervised learning approach for RAG-based open-domain question answering (QA). EffComp employs BERT-style document reranker and sentence selector models to allow fast extractive prompt compression at the sentence level. Its extractive nature prevents hallucinations in the compressed prompts. Additionally, the training process is designed to optimize the compression ratio while preserving LLM accuracy. Experiments on four open-domain QA datasets demonstrate that EffComp outperforms state-of-the-art prompt compression methods in terms of prediction accuracy and achieves competitive compression ratios (up to 78.4x) with minimal latency, making it practical for real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces EFFCOMP, a hybrid model designed to solve the trade-off in RAG systems by providing high extractive compression while maintaining LLM accuracy and minimizing latency. The main contribution is a novel hybrid of reinforcement and supervised learning for training the sentence selector, optimizing the trade-off between compression ratio and LLM accuracy. EFFCOMP achieves higher QA accuracy through extensive extractive compression and notable reduction in total latency, showing its practical effectiveness.

### Strengths
EFFCOMP effectively overcomes the main weakness of previous extractive methods, which only achieved modest compression ratios. It provides significant extractive compression, reaching ratios up to 78.4x while preserving factual accuracy and human readability.

### Weaknesses
1. Lack of Quantitative Evidence for Addressing the Core Challenge

>The paper's main goal is to address the inherent trade-off between abstractive compressors (high compression, high hallucination risk) and extractive compressors (low compression, low hallucination risk). The proposed EFFCOMP hybrid method aims to achieve this by offering the advantages of both (high compression and low hallucination). However, the experimental results only report end-to-end QA metrics (Acc/F1) and the compression ratio, failing to quantify the claimed advantage over the competing approaches.

>The paper asserts that EFFCOMP prevents hallucination due to its extractive nature, but this is never quantitatively proven against abstractive methods (like Recomp-Abstractive or CompAct). A rigorous comparison requires quantitatively evaluating their ability to retain factual integrity better than abstractive approaches. The qualitative example in Table 2 is insufficient proof.

>In addition, the paper claims EFFCOMP preserves more important information than other extractive methods, implying high efficiency in selection. This should be demonstrated by showing that the compressed contexts are semantically richer or less redundant than those produced by extractive methods.

>In short, the experiments show that EFFCOMP performs well overall, but they lack the necessary diagnostic analysis to quantitatively validate how it successfully addresses the fundamental trade-off between abstractive hallucination risk and extractive compression limitations, which was the paper's main focus.

2. Lack of Justification for Hybrid RL/SL Architecture

>The paper's core technical contribution is the hybrid Reinforcement Learning (RL) and Supervised Learning (SL) approach used for fine-tuning the sentence selector. While the authors describe the functional purpose of combining the two (RL for optimizing the objective, SL for efficiency via memory), they fail to provide a rigorous justification for why this specific hybrid method is optimal for the task.

>The authors state that the RL loss is needed because initial pseudo-labels are imperfect, and the SL loss (with best-trajectory memory) is needed for efficiency. However, the paper lacks a clear, upfront theoretical or empirical argument explaining why a complex alternating hybrid structure is necessary over simpler, established alternatives. 

>Although the paper includes an ablation study (Figure 4, Appendix D.4) comparing RL-only, SL-only, and Hybrid, the comparison is limited to showing which one performs best overall. It does not provide the diagnostic clarity needed to confirm that the benefits outweigh the overhead. Specifically, it doesn't clearly explain why the hybrid method handles zero-reward signals or exploration better than a modified pure-RL structure designed for contextual bandits.

>In essence, the paper proposes a complex solution without fully demonstrating why the problem demands this specific complex solution over more parsimonious alternatives.

### Questions
Questions on Latency Metrics and Measurement
> Table 4 provides a detailed breakdown of efficiency, but the units and measurement process need clear explanation to ensure reproducibility. What are the units for the Compression, LLM Read, and Total Latency metrics reported in Table 4? The text suggests they are in seconds, but this should be explicitly stated in the table caption or text. Could the authors provide a concise explanation of how latency was measured?

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
4

### Summary
This paper proposes a new sentence-level prompt compression framework for open-domain question answering, which is good for the high compression ratio and low compression latency. To achieve this goal, this paper designs a complex training framework, including supervised learning and end-to-end RL. Then the experiments over four open-domain QA datasets showed the advantages of high compression ratio and accuracy compared with many competitive baselines. This paper also provides many ablation study including the compression latency and fail explanations.

### Strengths
This manuscript has many strengths, including:

1. Reasonable method design, including sentence-level, end-to-end RL training
2. This paper conducts comprehensive experiments to show their advantages and provides detailed ablation studies to further investigate.
3. This paper has a good presentation and makes enough contribution to the area of prompt compression.

### Weaknesses
1. It's recommended to provide the code to reproduce the results.
2. The main results use F1 metric and EM accuracy. Why doesn't it use the LLM Judge to evaluate the answers?
3. There are some important train-free attention-based methods [1,2] requiring to compare or discussion. 

[1] Leveraging Attention to Effectively Compress Prompts for Long-Context LLMs
[2] Efficient Prompt Compression with Evaluator Heads for Long-Context Transformer Inference

### Questions
1. In my opinion, the compression ratio of the proposed method is determined adaptively. The compression ratio of selected baselines should be determined by the user. How do you select the compression ratio and what are the reasons?

2. Compression times are dependent on the input prompt. It's better to provide the length of used prompt in Table 4?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces EffComp, a two-step, extractive prompt compression framework designed for RAG-based QA.

### Strengths
- The framework achieves high compression ratios and outperforms baselines in QA accuracy and F1 score across four datasets.

- The paper includes generalization tests to an out-of-distribution LLM (Llama-3.1-8B) and a failure analysis.

### Weaknesses
- The baselines are all from 2024, and the paper lacks a comparison with newer methods from 2025 (such as CoLoR[1] and DAC[2]).

- While other prompt compression papers often demonstrate their method's effectiveness on summarization in addition to QA, this paper's evaluation is confined solely to QA.

- The authors state that 42% of the failure points are attributed to the evaluation metric and suggests that semantic-based evaluation metrics like BERTScore might be a better option. Therefore, the paper should have included evaluations using these metrics.

- Similarly, such an unreliable metric is used as the core signal for the GRPO reward function. This may lead to unstable training and spurious policy updates that do not reflect true semantic quality.

[1] Minju Seo, et al., Efficient Long Context Language Model Retrieval with Compression, ACL 2025.

[2] Yi Zhao, et al., DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression, ACL 2025.

### Questions
- The Phase I pretraining biases the policy against intermediate reasoning sentences by exclusively labeling sentences containing the final answer string. This creates a low probability for retaining crucial intermediate-reasoning sentences in multi-hop QA (like HotpotQA). It is questionable whether the GRPO sampling in Phase II can effectively explore and reinforce the retention of these sentences, or if it will struggle to overcome the flawed bias established during pretraining.

- The ablation study on RL and SL in the appendix should be placed in the main text, or at least be mentioned.

### Soundness
3

### Presentation
2

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
This paper introduces EFFCOMP, an efficient prompt compression framework for RAG-based open-domain question answering. The method employs a two-stage approach: (1) document-level reranking using BERT-style rerankers to filter and reorder retrieved documents, and (2) sentence-level selection using a BERT-based sentence selector trained through a hybrid reinforcement and supervised learning approach. The sentence selector is first pretrained using pseudo-labels based on answer string matching, then finetuned using GRPO with a reward function that balances QA accuracy and compression ratio. During finetuning, a best trajectory memory stores high-reward action sequences for supervised learning updates. Experiments on four QA datasets (Natural Questions, TriviaQA, HotpotQA, 2WikiMultiHopQA) with Gemma-2-9B-IT demonstrate compression ratios of 29.1x-78.4x while maintaining or improving accuracy compared to uncompressed baselines.

### Strengths
1. The combination of reinforcement learning with supervised learning using a best trajectory memory is sensible and addresses the instability issues common in pure RL approaches. The alternating RL-SL updates within each epoch provide a reasonable balance between exploration and exploitation.

2. Unlike many abstractive compression methods that require expensive LLM calls iteratively, EFFCOMP uses lightweight BERT-based models for both reranking and sentence selection, resulting in competitive latency.

3. By operating at the sentence level and preserving original text, EFFCOMP avoids the fabrication issues demonstrated in Table 2, where Recomp-Abstractive and CompAct generate factually incorrect compressed contexts.

### Weaknesses
* The paper compares primarily against LongLLMLingua, Recomp, and CompAct, but misses critical recent prompt compression methods that have shown superior performance:
    - LLMLingua-2 : A BERT-based token-level classifier trained via data distillation, achieving 3x-6x faster compression than LLMLingua with better task-agnostic performance. This is directly comparable to EFFCOMP's approach and should be included.
    - TACO-RL: Another RL-based prompt compression method using REINFORCE for task-aware optimization, which is methodologically similar and should be compared.
    - 500xCompressor: A soft prompt method achieving 6x-480x compression ratios.

* FlashAttention compatibility: EFFCOMP requires attention scores for the reranker, making it potentially incompatible with FlashAttention. The paper should compare memory usage and latency against FlashAttention-enabled baselines.


* Missing evaluation on more complex multi-hop reasoning datasets like MuSiQue or StrategyQA, and single-document QA datasets like SQuAD 2.0.

### Questions
* Can the authors include comparisons with LLMLingua-2, TACO-RL, and 500xCompressor?
* How does EFFCOMP's memory usage and latency compare to baselines using FlashAttention? Can the method be adapted to work without requiring explicit attention scores?
* Can the authors provide an ablation study on the reward weight α (currently 0.95)? What happens at α=0.5, 0.7, 0.9, and how does this affect the Pareto frontier of compression vs. accuracy?

### Soundness
2

### Presentation
3

### Contribution
2
