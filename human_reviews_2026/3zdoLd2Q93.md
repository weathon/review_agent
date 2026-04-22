# RCStat: A Statistical Framework of Relative Contextualization in Transformers

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
Estimating the importance of input tokens and their activations in auto-regressive models is a fundamental requirement in many applications, such as key-value (KV) cache compression and attribution. Prior work computes token importance using attention weights, which are obtained by normalizing the raw attention logits (query-key inner products) with a softmax operation. However, the softmax normalization suppresses the rich information within the attention logits. We introduce RCStat, a statistical framework that harnesses the raw attention logits via Relative Contextualization (RC) -- a random variable measuring contextual influence from one subset of tokens to another. We derive computationally efficient bounds on the expected RC and demonstrate its utility in two applications:  (i) KV compression, where RC‐based adaptive thresholding evicts substantial portions of the KV cache with minimal quality loss in token generation; and (ii) Attribution, where attention heads with high expected RC yield accurate span‐level attribution. Across QA, summarization, and attribution benchmarks, RCStat achieves state-of-the-art performance, improving generation quality by 15–40\% and attribution accuracy by 2–16\%, all without any model retraining.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces RCStat, a statistical framework that leverages pre-softmax attention logits to quantify contextual influence between token groups through a measure called Relative Contextualization (RC). Unlike prior methods relying on post-softmax attention weights—which lose fine-grained relational information due to normalization—RCStat models raw attention logits as random variables and derives an efficient upper bound on their expected difference to estimate contextual relevance. This statistical formalism enables two key applications: adaptive KV-cache compression and token-level attribution, both achieved without retraining or auxiliary supervision.

### Strengths
1. The work provides a theoretical justification for their kv cache's importance quantification.

### Weaknesses
1. The authors claim "Despite this potential, the usage of pre-softmax attention remains largely underexplored, primarily due to the lack of statistical tools and frameworks to extract structured insights from unnormalized logits." I believe this is not not true. Please check out [1]: a kv cache selection paper that quantifies importance of kv based on pre-softmax scores, or [2].
2. Missing comparison with SOTA baselines such as [1][2].
3. The ultimate goal of most KV-cache compression techniques is inference efficiency (faster decoding, lower memory footprint) with minimal performance loss. While the proposed work emphasizes improved performance under compression (which may itself be questioned in competitiveness to SOTA), the work provides no experiments on efficiency metrics (latency, GPU memory, throughput) - thus its practical viability remains unclear.


[1] Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference, Tang et. al., ICML 2024.
[2] InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory, Xiao et. al., ArXiv.
[3] DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads, Xiao et. al., ICLR 2025.

### Questions
1. I'm wondering what is the performance of the proposed method under other long-context tasks such as needle in a haystack?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work proposes a method for quantifying the influence of attention logits by measuring the relative contribution of the prompt part of tokens toward the generation part of the tokens. Basic ideas is to treat the raw attention logits as values for a random variable in a probability density function, and defines two random variables defined for the queries during generation: cross-contextualization which represents the logits for the prompt part of keys and self-contextualization which represents the logits for the generation part of keys. The third variable, relative contextualization, is defined to measure the importance of cross-contextualization over self-contextualization motivated by the focus of prompt during generation. The metric is employed for two tasks: KV cache compression by pruning irrelevant keys and chunk-level prompt attribution, demonstrating superior performance over baselines.

### Strengths
- This work presents a novel view of the raw attention logits in order to represent the influence of two parts, i.e., prompt and generation, by introducing random variables, cross-contextualization, self-contextualization and relative-contextualization. Given this view, this work also shows that the upper-bound of expected relative contextualization could be computed by marginization with an efficient algorithm for the computation.
- Experiments are designed systematically with solid results. The KV cache experiments clearly show better tradeoff of the compression and qualities. Chunk-level accuracy for attribution also shows gains when compared with other methods.

### Weaknesses
- It takes time to understand the manuscript, given that several key explanation for Equations CC (9) and SC (10) is presented in Appendix. Probably a little bit more intuitive explanation will alleviate the issue.
- Similarly, the chunk-level prompt attribution task is not clearly explained, and thus, readers have to look up the prior studies to understand the setting.

### Questions
- $c$ appeared in Equation (SC), but it should be $s$ if my understanding is correct.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces RCStat, a method for measuring attribution to prior context by using unnormalized attention weights. It proposes both an exact form and a more efficient approximation of this measure. Then, the paper demonstrates the utility of RCStat by using it to develop both a KV cache eviction method and text attribution method.

### Strengths
S1. The idea of RCStat is intriguing-- I like the argument that looking at weights post-softmax loses valuable information. To the best of my knowledge, this is a novel way of measuring influence from prior context.

S2. The formulation of RCStat with separation of context into two contrastable segments is also interesting, and the way that this allows for span-level identification of importance is a nice property.

S3. Attribution is a really hard problem, and one that this paper makes an nice contribution towards. I think the analysis of attribution performance growing worse with adding too many irrelevant attention heads is an interesting one.

### Weaknesses
W1. The paper presents a lot of insights, which is nice, but because it takes on three different focuses (deriving RCStat, using RCStat to design a competitive-with-SOTA KV cache eviction strategy, and analyzing RCStat's usefulness for attribution), it sometimes struggles to provide sufficient evidence for each. I organize the rest of the critique into these three subparts. 

W2. The initial explanation of RCStat was a bit hard to follow. I found the shapes of density functions drawn in 2(d) to be a bit distracting, and something that it would have been helpful to have clarified earlier-- in particular, why do these density functions overlap only slightly? 

W3. To make an argument about applying RCStat as a new KV cache eviction method, you really must address latency in the main text; I think the results in the appendix are somewhat promising, but I don't agree with the statement that this is only "modestly slower" than methods that are 4-5x faster per layer in the compression step (I understand the argument that, compared to prefill+decode, the overall time difference is negligible, but if that is the case you want to make, you should additionally report some metrics of overall time!). ROUGE-1 is not a reasonable quality metric to use; even an older neural metric would make a stronger case. And it seems this is missing several KV cache methods that are prominent in the community -- what about H20 or Pyramid KV? Why is this particular set of baselines the most reasonable choice? 

W4. The attribution section is interesting, but I think could benefit from some presentation improvements. Can you report standard deviations and/or significance test the differences between scores in Table 1? L3.1-8B's HS baseline is in a different section from the other L3.1-3B results, but for the Llama 8B and Qwen numbers this is in the same grouping. The pre/post comparison for Llama 8B might be better positioned in a separate table. Are the "least RC" numbers listed for the other models the pre-softmax ones? I was also wishing for a bit more exploration of which heads were good predictors of attribution-- the relevance of depth is mentioned a few times (and discussed in other work), but it would be interesting to hear more about how this works for RCStat's measure of head relevance in particular.

### Questions
Q1. Do you have any intuition for why VeriGran performance drops so much more dramatically than QuoteSum performance with increasing number of heads for Qwen in Fig 8b? 

Q2. My main critique is that the paper, in trying to do a lot of different things, is not quite doing all of them to a sufficient level. In your conceptualization of this work, do you see the theoretical framing of RCStat itself as the main contribution (with RCStat-based KV cache eviction + RCStat-based attribution being two example applications), or would you place all three on equal footing? 

Q3. Like most attention-based methods, this requires explicitly instantiating the attention matrix in memory, instead of calling an efficient kernelized implementation, right? Can you discuss the VRAM requirements imposed by this? 

Q4. various specific questions/feedback on the sections, as detailed in the weaknesses above. 

Small typos:
- line 463: generalizability misspelled
- line 100: this citation for the phrase mechanistic interpretability seems strange to me -- not clear why this must be cited or that this is the right thing to cite? I'm not as familiar with this literature though so if there is a real reason for this please disregard.
- line 112: "up to 84%" is what that work observed, but this phrasing makes it seem like this is a hard-and-fast maximum, which of course is not true
- line 419: "the an"; more generally I don't really understand what this sentence is trying to say

### Soundness
2

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
3

### Summary
The attention module in Transformer introduces structural bias by sharpening attention toward dominant tokens while flattening the others. It may discard potentially meaningful contextual signals. Corresponding to this phenomenon, the authors posit that logits encode not only what the current layer attends to but also preserve upstream interactions, offering a richer statistical substrate for analysis. Based on this behavior, the authors quantify how different attention heads behave in KV-cache compression and token attribution. By using a newly proposed method, relative contextualization (RC), the authors investigate that most heads have compressible KV-caches, while the few resistant ones provide useful attribution signals. The experimental results on large language models (LLMs) show that the proposed RC-based framework, RCSTAT, can support the improvement of prompt attribution and KV cache reduction in summarization and QA tasks.

### Strengths
- The targeting issue of the information loss in attention is fundamental for Transformer-based models, including large language models (LLMs).
- The assumption of the relative contextualization (RC) is based on the actually observed cases.
- The authors provide the computational complexity of the expected RC.
- The experimental results show the effectiveness of RC-based KV cache reduction and prompt attribution on various tasks.
- The authors compared their RC-based method with strong baselines, including state-of-the-art methods.

### Weaknesses
- The paper is not self-contained because reading the main text part requires accessing content in appendices like Equations (9) and (10). This is a presentation issue.
- The used models are restricted to small language models that have less than 10B of parameters.
- Runtime comparison is not reported.

### Questions
In the KV cache reduction, norms of value vectors are important as well as attention weights shown in the following papers. Could you explain the potential of combining your approach with such methods?
- Alessio Devoto, Yu Zhao, Simone Scardapane, and Pasquale Minervini. 2024. A Simple and Effective L_2 Norm-Based Strategy for KV Cache Compression. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 18476–18499, Miami, Florida, USA. Association for Computational Linguistics.
- Zhiyu Guo, Hidetaka Kamigaito, and Taro Watanabe. 2024. Attention Score is not All You Need for Token Importance Indicator in KV Cache Reduction: Value Also Matters. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 21158–21166, Miami, Florida, USA. Association for Computational Linguistics.

### Soundness
3

### Presentation
2

### Contribution
2
