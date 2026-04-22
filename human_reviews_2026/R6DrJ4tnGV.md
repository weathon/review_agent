# Scaling Linear Attention Capacity with Sparse State Expansion

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
The Transformer architecture, despite its widespread success, struggles with long-context scenarios due to quadratic computation and linear memory growth. While various linear attention variants mitigate these efficiency constraints by compressing context into fixed-size states, they often degrade performance in tasks such as in-context retrieval and reasoning. To address this limitation and achieve more effective context compression, we propose two key innovations. First, we introduce a row-sparse update formulation for linear attention by conceptualizing state updating as information categorization. This enables sparse state updates via softmax-based top-$k$ row selection, thereby extending receptive fields and reducing information interference. Second, we present Sparse State Expansion (SSE) within the sparse framework, which expands the contextual state into multiple partitions, effectively decoupling parameter size from state capacity while maintaining the sparse row-selection paradigm. Supported by efficient parallelized implementations, our design achieves highly discriminative state representations. We extensively validate SSE in both pure linear and hybrid (SSE-H) architectures across language modeling, in-context retrieval, and mathematical reasoning benchmarks. SSE demonstrates strong retrieval performance and scales favorably with state size. Moreover, after reinforcement learning (RL) training, our 2B SSE-H model achieves state-of-the-art mathematical reasoning performance among small reasoning models, scoring 64.5 on AIME24 and 50.2 on AIME25, significantly outperforming similarly sized open-source Transformers. These results highlight SSE as a promising and efficient architecture for long-context modeling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new framework to enhance the efficiency and scalability of linear attention models for long-context processing. It introduces two innovations: (1) row-sparse state updates, which treat information storage as a classification task using top-k softmax selection to reduce interference and extend context range, and (2) Sparse State Expansion (SSE), which partitions the state into multiple sparsely updated segments to increase memory capacity without adding parameters. Experiments show that SSE and its hybrid variant (SSE-H) outperform earlier linear attentions across language modeling, retrieval, and reasoning tasks.

### Strengths
Clear motivation, sound implementation, and good performance with model at scale.

### Weaknesses
1. From the Fig.4 we can see that the wall time is not linear with respect to the sequence length. Why? Hope the authors could provide some more experiment results comparing the wall time of their architecture against earlier linear attentions, GLA, GDN, Mamba2, and full attention.
2. Hope the authors could also provide scaling experiments compare their architecture against earlier linear attentions, GLA, GDN, Mamba2, and full attention, considering training wall time vs perplexity.
3. Hope the authors could also discuss MoE for quadratic attention in their related work.

### Questions
On which datasets was the model trained? Will these data be open-sourced?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper improves context compression in linear attention models with row-sparse update and sparse state expansion (SSE). The row-sparse update learns a sparse mask with top-k and softmax to update only a few rows in the contextual states, which utilizes the state space more effectively than dense update. Since different tokens are associated with different rows, the row-sparse update eliminates the need of gating at each step, thereby avoiding the limited receptive field caused by gating. The authors then propose SSE to extend row-sparse update to larger state space. SSE first divides a large state space into N partitions, and then perform dense updates on k partitions. By sharing the attention parameters across partitions, SSE decouples the parameters size and the memory capacity, thereby solving the capacity bottleneck of linear attention models. Experiments on language modeling, needle-in-a-haystack and reasoning benchmarks show that SSE and the hybrid SSE-H achieves state-of-the-art performance among linear attention models.

### Strengths
1. This paper tackles the memory capacity problem, a key issue in linear attention models with clear motivations. The row-wise sparse update aims to improve the utilization of a fixed size state space, while the SSE extends the memory capacity without increasing the number of parameters.
2. Experiments are thorough and solid. The authors visualizes the cosine similarity of the state space and show that row-sparse update significantly improves state space utilization compared to existing designs. The final SSE model is evaluated against Transformer and linear attention baselines on a wide range of benchmarks, with a rigorous setup of fixed number of parameters.
3. The efficiency of SSE is not confined to theory. The authors implemented SSE by grouping tokens into subsequences according to their partitions and executing them with a linear attention kernel call. This makes SSE practical for real-world long context use.

### Weaknesses
1. The writing of this paper may be largely improved. The authors used quite a few terminologies or preliminaries without enough explanations, which make this paper hard to follow. For example, the authors didn’t explain how they computed the cosine similarity in Figure 1. Line 176 compares SSE against gated linear attention, but the form of gated linear attention is never mentioned in the paper. There aren’t ground truth classes nor a classification task, but the authors keep using the term classification to refer to state space utilization. In Line 245, the term segmented clustering is used, but this isn’t a commonsense for audience. Captions of figure 1 and 3 need to be extended with their implications. See more in questions.
2. The title doesn’t exactly reflect the contribution of this paper. “Scaling linear attention" sounds like this paper studies the model performance under different linear attention sizes. I would suggest modifying it to be “Extending linear attention capacity with sparse state expansion”.
3. While SSE focuses on improving linear attention models, there isn’t any comparison of wall time for SSE and baselines. Could you please report the performance-time trade-off curve for SSE, Transformer and other linear attention models? That will help audience know which model to use given a specific context length.

### Questions
1. Is the cosine similarity computed for a single step then averaged over a sequence or the whole dataset?
2. The logic in Line 174-178 is hard to understand. It’s hard to understand what Propositions 2-4 are without looking into the Appendix. Propositions 2-4 use the row-sparse update, which is introduced only in later sections. Besides, it’s not very clear how the conclusion of decay in gated variants is derived before looking into Proposition 4. As row-sparse update is complementary to gating and they may co-exist, it’s also hard to understand why SSE solves the issue caused by gating. You may discuss eliminating gating as a benefit of SSE after introducing the method.
3. Line 184-186: By theoretical analysis, do you mean Proposition 2? It’s a little bit hard to understand this without looking into the appendix. You need to add more details in the main paper.
4. The functions softmax and top-k produce a $k$ dimension vector by their definition, which is not correct. Do you mean softmax on the $k$ non-zero elements and keep the rest as 0? Then you need to re-define your softmax function.
5. Line 263-265: This key insight should be brought to early paragraphs of Sec 4.1. Otherwise, it’s hard to understand how SSE is connected with row-sparse update. You’re essentially factorizing a row-sparse update for N*c rows into two small parameter matrices.
6. Font size in Figure 3 & 4 should be increased.
7. Line 284 & 286: Why can the always-selected partition capture local interactions? The inputs are accumulated by addition, which doesn’t model interactions.
8. Line 294-296: Please explain why singular value entropy reflects the difficulty of compression. It’s not a commonsense to audience.
9. Line 301: Sequential computation is never explained in the main paper.
10. Line 308-310: I would recommend to draw a figure for grouping and varlen technique.
11. In Table 1, linear attention models are worse than Transformer. Is it because language modeling requires more pairwise interactions? Then why SSE becomes on par with Transformer in Table 3 & 4?
12. Figure 5: Which task is it?
13. Figure 6: Does SSE-Shared refer to the always-selected partition? Please be consistent in the terms.
14. Line 457: Is the sparsity measured within the selected partitions? For n4k1, I understand the upper bound of sparsity for all partitions should be 25%, right?

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Sparse State Expansion (SSE) for linear attention. Two ideas drive the method: (i) row‑sparse updates that treat state updates as a classification problem and write only to top‑k rows via a softmax head, and (ii) state expansion into N shared‑parameter partitions chosen by a write–read gate, so capacity (state size) scales without growing parameter count. Efficient masked/varlen implementations are provided. Empirically, SSE and its hybrid variant are tested on language modeling, retrieval, and math reasoning.

### Strengths
1. Modeling state updates as information classification is well argued and operationalized 
2. Decoupling the model’s parameters and state's parameters is important direction for improving linear transformers.

### Weaknesses
1. There is no convincing explanation on why SSE is only effective for linear attention but not deltanet.
2. Baselines seem to be cherry picked since they are neither the most powerful nor fundamentally relevant sequence models. 
3. The paper uses hard partition selection (top‑k) and softmax row selection, but the gradient treatment for the discrete top‑k isn’t described. Can you please clarify how gradients flow through Eqs. 7–9 and whether any implementation tricks are needed for stability.

### Questions
See weaknesses.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes row-selector to bottleneck the update of state and use multiple partitions to expand state size for improved expressiveness. The results show their methods can get comparable performance with Transformers.

### Strengths
1. results are strong: SOTA in 2B reasoning model.
2. preliminaries are well-organized, proofs are completed.
3. enable multiple efficient parallelized implementations.

### Weaknesses
1. phrasing can be simplified and changed for better delivery: information classification, row-sparse -> row-selector. (TBH information classification is very confusing).
2. miss an important baseline Mamba/Mamba2

### Questions
1. can you provide some efficiency analysis, especially empirical evidence, when compared to baseline linear attention models?

### Soundness
4

### Presentation
2

### Contribution
3
