# LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Large Language Models (LLMs)-based text retrieval retrieves documents relevant to search queries based on vector similarities. Documents are pre-encoded offline, while queries arrive in real-time, necessitating an efficient online query encoder. Although LLMs significantly enhance retrieval capabilities, serving deeply parameterized LLMs slows down query inference throughput and increases demands for online deployment resources. In this paper, we propose LightRetriever, a novel LLM-based retriever with extremely lightweight query encoders. Our method retains a full-sized LLM for document encoding, but reduces the workload of query encoding to no more than an embedding lookup. Compared to serving a full LLM on an A800 GPU, our method achieves over a thousand times of speedup in query encoding and over 10× increase in end-to-end retrieval throughput. Extensive experiments on large-scale retrieval benchmarks show that LightRetriever generalizes well across diverse tasks, maintaining an average of 95% retrieval performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Proposes and evaluates an LLM-based retriever where the goal it to minimize query encoding time. The retriever has sparse and dense components.  For dense retrieval the query vector consists of the average of token vectors, and for sparse retrieval the query vector is a query term frequency count. Overall, this approach is basically the simplest hybrid approach possible. The evaluation employs BEIR and CMTEB-R. As one might expect, query encoding time is just a milliseconds, resulting in substantial improvements in throughput. NDCG values are comparable to acceptable first-stage baselines, including various query embeddings, BM25, and SPLADE.

I wonder why FIgure 5 (the ablation results) uses barcharts instead of presenting results in as a table, which is the format for the other results. The reason I wonder is because the sparse results look quite good, especially given that the query vector representation is nothing more than query term frequency, the same as what might be used with BM25. As I understand it, the green (hybrid) bars are taken from the bottom part of Table 2, so that the orange (sparse) bars are in the high 40s, maybe hitting even 50. This is a pretty remarkable result, given that BM25 results are in the low 40s. Since these queries are nothing more than BM25-style term frequencies, they would run very fast against the sparse document index, perhaps even faster than BM25 with the right query processing strategy.

The description of the training for sparse retrieval seems to be SPLADE more-or-less, so this seem unexpected, since splade uses more than term frequencies for the queries. I dug around on the Web a bit, and I think this is a known result, for example, "Joel Mackenzie, Shengyao Zhuang, and Guido Zuccon. 2023. Exploring the Representation Power of SPLADE Models. In Proceedings of the 2023 ACM SIGIR International Conference on Theory of Information Retrieval (ICTIR '23)."

On the other hand, the dense retrieval use the average of token vectors, which also seems very simplicist, dating back to the Word2Vec era.

### Strengths
I wrote more than I should have for the summary, so most of what I want to say is there. The paper is readable and complete, the experiments appear properly conducted and (with the exception of Figure 5) properly reported. The focus on efficiency is good.

### Weaknesses
I raised what I consider the main weakness in the summary. The dense part is only an average of token vectors, and the sparse part is only splade with query term frequencies. They are combined with a standard hybrid retrieval formula. Is this enough? It might be good from an engineering standpoint, but I'm not seeing the research-level insights.

### Questions
Why is this new? I don't see Mackenzie et al. (2023) referenced. The training for document vectors in section 2.3 is basically splade with term frequencies. Token vectors for retrieval were explored back in the Word2Vec era. Convince me there is something new here.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the issue of excessive inference costs in large language model (LLM)-based retrievers by proposing an asymmetric architecture named LightRetriever. This architecture employs a full LLM for offline document encoding ($Enc_d$), but replaces the online query encoder ($Enc_q$) with a simple embedding lookup. This lookup table is generated through an innovative training process: caching the outputs of the full $Enc_q$ for each instruction-word combination, i.e., $v_{t_i}^{den} = Enc_q(Inst; t_i)$. The final query vector is simply the average of these cached embeddings. This approach claims to achieve >1000x query acceleration and >10x end-to-end throughput improvement while retaining 95% of the performance of the full symmetric model.

### Strengths
1.  This work addresses a critical practical bottleneck in deploying LLM-based retrievers: high online query latency, while achieving substantial throughput gains.
2.  The "distill-to-embedding-bag" approach is novel. It ingeniously caches the word-level understanding of the instruction-aware $Enc_q$ into a simple lookup table, differing from standard knowledge distillation schemes.
3. The method is validated across multiple LLMs (Llama, Qwen) and benchmark datasets (BeIR, CMTEB-R), with fair comparisons against fully symmetric models trained under identical conditions.

### Weaknesses
1.  The dense query encoder employs a bag-of-words model, where $v_q^{den} = \frac{1}{n}\sum E[t_i]$. This model fails to capture query composability, meaning that for the model, the query vectors for "flights from Beijing to Shanghai" and "flights from Shanghai to Beijing" are identical. This fundamentally limits the model's ability to understand complex queries.
2.  This approach trades significant online memory consumption (an 8B model requires approximately 1.05GB of embedding tables) for reduced latency, yet never discusses this critical trade-off.
3. The paper contains inconsistencies. For example, the abstract and experimental settings mention acceleration on A800, but the conclusion refers to H800; the 2500x acceleration cited on line 378 also differs from the 1000x acceleration mentioned elsewhere.

### Questions
1. Could the average "95% performance" mask catastrophic failures in composite tasks (e.g., Case Study 3)? Can performance breakdowns be provided for word lookup versus composite query tasks?
2. How does the online memory footprint (RAM/VRAM) of the 1.05 GB embedding cache compare to standard KD student models or the "first-layer" baseline? The paper ignores this memory-latency tradeoff.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
LightRetriever proposes an asymmetric LLM-based retrieval architecture that keeps a full-sized LLM for document encoding while making query encoding extremely lightweight. For dense retrieval, the method trains token-level query embeddings end-to-end using a full LLM during training, then caches the entire vocabulary’s token embeddings so that online query vectors are computed by an embedding lookup followed by mean pooling. For sparse retrieval, the query vector is a simple term-count vector and the document vector is learned by projecting LLM hidden states into the vocabulary space with sparsity regularization.

### Strengths
(1) The paper introduces a clear and practical asymmetric design that eliminates deep query-side inference while preserving full LLM power on the document side, delivering extreme online speedups with modest accuracy trade-offs.

(2) The dense pathway’s cache-and-average mechanism and the sparse pathway’s LM-to-vocabulary projection with FLOPs-based sparsity are well formalized, and the training–caching–serving pipeline is technically sound and reproducible.

(3) Empirical coverage is broad, spanning 23 training datasets and two large benchmark suites in English and Chinese, multiple LLM backbones and sizes, and detailed speed breakdowns on 65k queries over 1M passages.

(4) Ablations substantiate key design choices, showing asymmetry is crucial, full query modeling during training is needed, and hybrid dense+sparse recovers most performance; additional results demonstrate controllable trade-offs via Matryoshka dimension truncation and sparse top-k.

### Weaknesses
(1) The approach still depends on full LLM query modeling during training; the paper does not explore reducing training-time cost through distillation, curricula, or lighter interim encoders while maintaining cached-token quality.

(2) Evaluation focuses on academic text benchmarks; robustness on production-like settings with noisy queries, long or heterogeneous documents, domain shifts, and adversarial inputs is not addressed, limiting external validity.

(3) Instruction conditioning is asymmetric: dense uses query instructions while sparse cannot, and the sensitivity to instruction templates and potential mismatches is underexplored, especially for instruction-heavy tasks.

(4) Mean-pooled token embeddings may miss fine-grained compositional constraints; beyond case studies, more systematic stress tests on compositionality, negation, and multi-attribute intents are needed to quantify failure modes and the extent to which hybrid signals compensate.

(5) Many real pipelines perform truncation, normalization, rewriting, or chunking; the robustness of the embedding-lookup query encoder under such preprocessing is not evaluated.

(6) Baseline breadth could be expanded with stronger asymmetric or distilled LLM retrievers and recent hybrid SOTA across both English and Chinese to contextualize the quality–efficiency trade-offs.

### Questions
(1) How expensive is training the dense token cache compared to a standard symmetric dual-encoder, and can you reduce training cost via distillation from the full query encoder to the token cache or a smaller interim encoder without sacrificing performance?

(2) Can you provide systematic evaluations on compositional and constraint-heavy queries beyond case studies, reporting category-wise results for multi-attribute intents, negation, and temporal constraints, and quantifying how hybrid signals mitigate dense-only failures?

(3) How sensitive are results to instruction templates on the dense side, and can sparse retrieval benefit from instruction conditioning indirectly, for example via vocabulary reweighting derived from instruction tokens?

(4) How robust is LightRetriever to production preprocessing such as truncation, lowercasing, punctuation stripping, query rewriting, and subword-vocabulary mismatches, and what are the measured impacts on speed and accuracy?

(5) Can you propose and evaluate an adaptive serving policy that tunes dense dimension top-k and sparse top-k per query based on estimated difficulty or latency SLAs, reporting accuracy–latency trade-off curves?

(6) How does the method transfer to long-document retrieval, code or log retrieval, and additional languages beyond English and Chinese, and does the sparse projection remain tractable and effective with domain-specific tokenization?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces a novel asymmetric methodology that trains the embedding layer of a LLM as efficient token-level retrieval embeddings, which can then be averaged to form a query embedding. During training, each token for a given query is prepended with a task specific instruction before being passed to the encoder. Final query embedding (mean of all instruction + token embeddings) is then used as a query representation for contrastive learning using document representations from the same encoder model.

This method shows a 1000x speed-up while maintaining quality (up to 95%) for a variety of queries across a suite of datasets.

### Strengths
- The paper's primary strength is its direct and effective solution to a practical, real-world bottleneck: the high cost and low throughput of online query encoding in LLM-based retrieval systems. The method reduces query encoding time by orders of magnitude (e.g., ~109s to 0.04s for an 8B model on a test batch), leading to a >10x increase in overall query-per-second (QPS) throughput.
- The training methodology is simple and effective. 
- The paper's claims are well-supported by its ablations. Ablation A2 confirms that the full-sized query LLM is essential during training and that a simple embedding bag is insufficient, validating the training-time complexity . Ablation A1 confirms the asymmetry is crucial, as a lightweight model on both the query and document sides leads to a severe performance collapse
- Overall, clear and well written paper.

### Weaknesses
- Evaluations on more challenging IR tasks representing live, real world traffic (maybe include CoIR, BRIGHT).
- Please add a discussion section (in Appendix if needed) that talks about performance within BeIR splits. For example, in the case of HotpotQA (multi-hop), FiQA, etc the performance drops are significant compared to full baselines. This aspect should be made clearer in the paper write-up (expanded more than L407 - L409). Currently, the paper reads as this approach being a one-solution fits all approach for retrieval.


- nit: the citations don't have brackets around them and neither are they highlighted in blue (for hyperlinks) which makes it slightly hard to read. I think it may be a rendering issue. Please fix.

### Questions
- Is the mean weighted for the results table?

### Soundness
2

### Presentation
3

### Contribution
2
