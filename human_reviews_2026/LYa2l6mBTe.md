# TokenButler: Token Importance is Predictable

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Large Language Models (LLMs) rely on the Key-Value (KV) Cache to store token history, enabling efficient decoding of tokens. 
As the KV-Cache grows, it becomes a major memory and computation bottleneck, however, there is an opportunity to alleviate this bottleneck, especially because prior research has shown that only a small subset of tokens contribute meaningfully to each decoding step. 
A key challenge in finding these _critical tokens_ is that they are dynamic, and heavily input query-dependent. 
Existing methods either risk quality by evicting tokens permanently, or retain the full KV-Cache but rely on retrieving chunks (pages) of tokens at generation, failing at dense, context-rich tasks. 
Additionally, many existing KV-Cache sparsity methods rely on inaccurate proxies for token importance.
To address these limitations, we introduce **TokenButler**, a high-granularity, query-aware predictor that learns to identify these critical tokens. 
By training a light-weight predictor with less than $1.2\\%$ parameter overhead, TokenButler prioritizes tokens based on their contextual, predicted importance. 
This improves perplexity \& downstream accuracy by upto $8\\%$ relative to SoTA methods for estimating token importance. We evaluate TokenButler on a novel synthetic small-context co-referential retrieval task, demonstrating near-oracle accuracy. Furthermore, we show that TokenButler minimizes the gap to the oracle throughput and outperforms prior methods by up to $3\times$. Code, models, dataset and benchmarks [are available](https://anonymous.4open.science/r/TokenButler-EAEF).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes TokenButler, a learned predictor that make KV caches sparse during decoding. TokenButler estimates the importance of the tokens. The method is plug-and-play at inference, requires modest compute overhead, and targets the decode phase (not prefill). Results show consistent perplexity/accuracy gains.

### Strengths
Disclaimer: reviewer is non-expert in the area

1. The paper is well presented.
2. The paper is addressing a real challenging research problem: KV cache cost for long context.
3. The method of the paper is intuitive and novel.
4. The paper has done solid experiments on short to mid context task.

### Weaknesses
Disclaimer: reviewer is non-expert in the area

The paper doesn't show experiment results comparing TokenButler with the  token-eviction methods and page-based token-selection methods on long-context cases. In the limitation section, the author explains that this is because his 'primary focus is on demonstrating simple cases where token-eviction methods and page-based token-selection methods can fail.' However, without experiment results for longer context, we simply don't get a complete picture of the actual performance of TokenButler in real case scenarios, especially considering the fact that KV cache cost is the most when the context is long.

### Questions
N/A

### Soundness
3

### Presentation
3

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
The paper proposes TokenButler that uses a tiny (<1.2\% extra parameters) predictor to decide which tokens to keep in the KV cache. On a new co-reference benchmark it retains context-critical tokens almost as well as an oracle, letting aggressive sparsity coexist with high accuracy. It lifts WikiText perplexity and downstream task scores by ≥8\% and yields up to 3× higher throughput than prior methods.

### Strengths
1. This paper aims to address a timely and important problem
2. Some of the evaluation results are great and insightful
3. The paper analyzes the weakness of eviction strategies (StreamingLLM and H2O) and large-page retrieval strategies (Quest) in the context of co-referential conversation.

### Weaknesses
1. The paper claims that token importance is predictable, but does not explain why. (See questions)

2. An important prior study is not discussed. IMPRESS [1] is a training-free adaptive dynamic strategy. It neither evicts tokens nor retrieves pages of tokens. IMPRESS efficiently finds dynamic and query-dependent critical tokens.

[1] Weijian Chen, Shuibing He, Haoyang Qu, Ruidong Zhang, Siling Yang, Ping Chen, Yi Zheng, Baoxing Huai, and Gang Chen. 2025. IMPRESS: an importance-informed multi-tier prefix KV storage system for large language model inference. In Proceedings of the 23rd USENIX Conference on File and Storage Technologies (FAST '25). USENIX Association, USA, Article 12, 187–201.

3. The experimental setting is unclear. For example, the metric "coverage" in Table 2 is defined properly, but the "coverage" in Figure 5 seems to have a different meaning. Moreover, the four tasks (HellaSwag, ARC-Easy, PIQA and WinoGrande) and their accuracy are not explained.

4. The synthetic task is similar to the Needle-in-a-Haystack test, which should be used to evaluate TokenButler. SnapKV performs excellently in this task as shown in the prior paper.

### Questions
1. LLMs consist of multiple transformer layers, because different layers capture relations between tokens from different perspectives. How can the single-layer TokenButler predict the importance for all layers, especially when its hidden dimension is reduced to d? As the number of LLM parameters increases (e.g., 13B or 30B), TokenButler may fail to make precise predictions.

2. Figures 3 and 4 show that TokenButler's accuracy is between 70-75\%, which is not high enough. What is the accuracy of prior works?

Minor comments:
- Line 483, In terms of throughput. We -> In terms of throughput, we

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces TokenButler, a lightweight, query-aware predictor intended to identify "critical" tokens in the Key-Value Cache for efficient decoding in large language models. Unlike prior static, eviction-based, or page-based approaches, TokenButler directly predicts per-token, per-head importance ahead of time, using only the first-layer hidden states. The authors train this predictor to match full attention maps and demonstrate its utility in reducing memory and compute bottlenecks, showing improved accuracy and perplexity on standard and synthetic, co-referential benchmarks compared to recent token sparsity methods.

### Strengths
The paper addresses a real and timely bottleneck in LLM inference, particularly as context lengths increase and traditional KV-cache management strategies become increasingly inadequate for context-rich tasks. TokenButler's approach—predicting token importance scores for fine-grained, per-head selection using first-layer representations—is clearly motivated and provides a practical balance between computational efficiency and contextual relevance. The technical design is explained with a reasonable level of transparency and thoughtful ablations.

### Weaknesses
1.  While the per-token, per-head predictor design is well-motivated, the idea of learning to predict token importance for sparsity and retrieval (from early-layer activations) builds on and is somewhat incremental to recent contextual/pruning/sparsity works, and the overall leap over existing public approaches like TokenSelect, is not as sharply differentiated as the manuscript sometimes suggests. The scope for genuine conceptual innovation is thus moderate.
2.  While Table 3 and Figure 6 include a selection of known sparse attention and token-eviction methods, TokenSelect is not included in the experiments or discussed. As TokenSelect is highly relevant and recent, its absence weakens the claims on comprehensiveness.
3.  A significant weakness is the mismatch between the problem the paper claims to address and the scope of its evaluation.  The introduction explicitly states that LLM contexts have expanded to 128k-1M tokens and that the KV Cache is the primary bottleneck. However, the paper's core accuracy and PPL evaluations Figure 6 are conducted on relatively short context lengths ("up to 1024"). Its novel synthetic task Figure 5 is also limited to "512 tokens".

### Questions
1.  Regarding the long-context evaluation, why doesn't the paper compare against other token-level prediction methods, such as TokenSelect, in the main accuracy and perplexity benchmarks? Can the authors also provide results for TokenButler on a standard long-context benchmark like InfiniteBench?

2.  The evaluation would be strengthened by including performance results from applying TokenButler to a wider range of backbone models (beyond those already tested) and evaluating them on other common datasets.

3.  As the results in Figure 5 and 6 is not compelling, could the authors provide more token accuracy results on longer

### Soundness
3

### Presentation
3

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
The paper presents TokenButler, a lightweight (~1–1.2% of base model parameters) predictor that estimates token importance per layer and per head using only first-layer hidden states. It predicts low-dimensional query/key pairs whose dot product approximates attention logits, allowing the model to selectively access tokens during decoding without discarding them from the KV cache. 

Unlike prior methods that evict tokens (H2O, SnapKV) or rely on page-level retrieval (Quest), TokenButler maintains all tokens in memory but reduces bandwidth by reading only the predicted important ones.

Across several Llama, Mistral, and Phi variants, it achieves 70–75% accuracy for identifying top-50% important tokens and 51% Recall@1% on WikiText2, with minimal overhead. On a synthetic co-reference task, TokenButler approaches oracle performance and outperforms page/eviction baselines. It also improves perplexity and downstream accuracy (e.g., HellaSwag, ARC, PIQA, WinoGrande) by up to 8% at similar sparsity.

Through calibration, TokenButler adapts per-head token budgets, outperforming TOVA, SnapKV, and H2O on long-context tasks such as Qasper and GovReport. The method’s bandwidth savings yield 3× faster decode throughput than TokenSelect. Limitations include full KV retention, dense prefill cost, and small-scale synthetic benchmarks.

### Strengths
**1) Clear problem framing and motivation**

The paper clearly explains why query-aware, fine-grained token selection is important and where static or page-level approaches fail. Figures 1 and Table 1 (pp. 2–3) effectively illustrate both the motivation and the limitations of prior methods.

**2) Methodological simplicity and clarity**

Predicting attention for all heads and layers using only first-layer hidden states is a simple and efficient design. This approach aligns with the observed low cross-head agreement (Fig. 11, p. 15), and the architecture in Fig. 2 is easy to implement and conceptually clear.

**3) Broad model coverage.**

Experiments are conducted on multiple model families including Llama-3.2-1B/3B/8B, Llama-2-7B, Mistral-7B, and Phi-3/3.5, as well as a reasoning-oriented model (Sec. 4, pp. 4–5; Fig. 7, p. 8). This variety supports the generality of the findings.

**4) Empirical results that support the main claims.**
Top-K recall and accuracy are strong given the small computational overhead (Figs. 3–4, p. 5).
On the synthetic co-reference benchmark, TokenButler performs close to an oracle and surpasses eviction- and page-based methods (Table 2, p. 6; examples pp. 18–22).
Under uniform-budget simulations, perplexity and downstream accuracy consistently outperform H2O, SnapKV, Quest, and StreamingLLM with up to eight-percent gains (Fig. 6, p. 7; p. 8).
Throughput remains high compared to full attention and TokenSelect (Fig. 10, p. 9).

**5) Comprehensive ablation studies.**

The paper carefully analyzes attachment depth, per-head necessity, predictor size scaling, and per-head budget calibration (Figs. 8, 11, 13; Table 4; pp. 8–9, 15–16).

**6) Transparency about scope and limitations.**

Limitations and trade-offs such as dense prefill cost, full-KV retention, and small latency overhead are clearly discussed (Appendix B, p. 14; Appendix E.1, p. 16), which enhances the paper’s credibility.

### Weaknesses
See question

### Questions
1) The idea of predictor attachment depth is interesting. Have you explored using multiple predictors attached at different layers simultaneously?

2) In Section 4.5 (Leveraging Top-K Recall), it seems that sparsity is adjusted per head. How does this compare to HEADKV (ICLR 2025) in terms of KV-cache budget management?

3) Is C4-RealNews really the best choice for training data? Could you show at least one more sample or alternative domain?

4) Since the predictor approximates attention, I’m curious how it represents attention sinks. Also, could you elaborate on how this relates to OrthoRank (ICML 2025)?

5) Please provide a clearer analysis of inference benefits, including latency, throughput, and peak memory.

### Soundness
4

### Presentation
3

### Contribution
4
