# OSCAR: Online Soft Compression for RAG

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by integrating external knowledge, leading to improved accuracy and relevance. However, scaling RAG pipelines remains computationally expensive as context length grows. On one hand, hard compression methods have recently proposed to prune the retrieved text on-the-fly with a limited compression ration. On the other hand, soft compression method performs a costly offline compression thanks a dedicated LLM but with a higher compression rate. In this paper, we introduce OSCAR, a novel query-dependent online soft compression method for RAG. OSCAR bridges the gap between online hard and offline soft compression methods, bringing the best of both: OSCAR dynamically compresses retrieved information at inference time, eliminating storage overhead and enabling higher compression rates than existing methods. Our experiments demonstrate state-of-the-art performance with a 2-5x speed-up in inference and minimal, if any, accuracy loss, for LLMs ranging from 1B to 24B parameters.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes OSCAR, an online, query-dependent soft compression mechanism for RAG. A compressor LLM maps each (query, document) pair to a small number of learned “memory” embeddings, which a generator LLM then consumes to answer the query. Two compressor backbones are explored: (i) OSCAR-N-Layers, using the first N layers of the generator; and (ii) OSCAR-llama, a small LLM with a learned hidden-space mapping. Across Mistral-7B, Qwen-7B, Llama-1B and Mistral-24B backbones, OSCAR yields 2–5× end-to-end speedups with minimal accuracy loss.

### Strengths
1. Propose a novel online soft compression method that compresses the input in a query-dependent fashion.

2. The training pipeline of OSCAR operates in a self-supervised way that does not rely on ground truth labels and generates them from no-compression RAG pipeline, providing flexibility over document choices.

3. Comprehensive evaluation with different choices of datasets, backbones and evaluation metrics. The results show a better accuracy-efficiency tradeoff for OSCAR over existing methods.

4. Detailed ablations on compression rate, query dependence, context length and compressor architecture.

### Weaknesses
1. OSCAR must be retrained per generator LLM to align hidden spaces; while the paper notes a few GPU-days for training, this could be operationally costly for trying out different backbone LLMs for different tasks.

2. Under some dataset setups OSCAR has worse accuracy compared to existing compression methods or the original backbone models. The paper does not provide a good investigation into the reasons causing this.

3. The baselines chosen by the paper are mainly hard compression and offline soft compression methods. There is a lack of comparison with existing online soft compression methods.

### Questions
1. How does OSCAR compare to existing online soft compression approaches such as FiD-light mentioned in the prior work?

2. In Figure 4 for BioASQ, why is the accuracy of OSCAR consistently worse than the baselines with the same compute T-FLOPs? 

3. In Figure 3 for the comparison of OSCAR with Mistral-7B, OSCAR also does not show clear wins as compared to Provence and PISCO. What are the possible reasons behind this?

4. For the N value taken in OSCAR-N-Layers, what is the sensitivity of it when the backbone model changes?

### Soundness
4

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
The paper proposes OSCAR, a query-dependent, online soft compression framework for RAG. Unlike prior hard or soft compression approaches, OSCAR compresses each (query, document) pair into a small set of embedding tokens on the fly and conditions generation on these embeddings. Across open-domain QA benchmarks, OSCAR targets 2–5× end-to-end inference FLOPs reduction with near-parity accuracy to uncompressed RAG, with stronger gains on larger backbones.

### Strengths
- The method is elegantly designed, conceptually bridging the gap between traditional hard compression and offline soft compression in terms of both performance and usability. 
- Strong empirical results, achieving near-parity accuracy with up to 2–5× efficiency gains. 
- The integration with reranking allows shared computation, improving system efficiency without additional overhead.

### Weaknesses
1. My main concern is that fine-tuning the generator may impair generalization. Because the compressor is query-dependent, the resulting features can encode domain-specific priors; joint training encourages the generator to exploit these priors, increasing the chance of specialization and consequent degradation on cross-domain retrieval and non-RAG tasks. This potential loss of generality could limit the method’s practicality in real-world deployments.

2. The experiments are mainly conducted on Mistral-7B and Qwen2-7B, while larger models such as Mistral-24B are only compared against an uncompressed version. Moreover, the evaluation focuses on accuracy and FLOPs, omitting critical system-level metrics such as end-to-end latency and throughput. Although Table 6 reports related values, it lacks direct comparisons to baseline methods.

3. The paper claims that query-dependent embeddings preserve task-relevant semantics but provides limited theoretical analysis or formal justification to support this claim.

### Questions
- How does the LoRA-fine-tuned generator perform on standard non-RAG benchmarks ? Does fine-tuning lead to a degradation in general-purpose reasoning or language understanding?

- The current cross-domain evidence is limited, focusing mainly on biomedical QA and retrieval-level BEIR, without end-to-end evaluation in other domains or non-QA RAG tasks. How does the proposed approach behave on cross-domain retrieval or non-QA RAG tasks?

- Could the authors report results when the generator is kept frozen during training to isolate the effect of compression?

- Providing a more comprehensive evaluation would offer a fuller assessment of the method’s contribution and generality.

### Soundness
3

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
3

### Summary
This paper propose a compression method for RAG, called OSCAR. This method includes two designed models; the first one is called compressor, which is used to convert document chunks as well as query into fixed-length embeddings. The second model is called generator, which leverages the query and the embeddings obtained from the compressor to generate an answer. Both compressor and generator can be trained in an end-to-end fashion, using the RAG results from a teacher model without compression as training signals. Experiment results show that on a series of QA datasets, the proposed method can achieve similar performance compared to no compression and slighter better inference cost.

### Strengths
1. The paper is well writen and easy to follow.
2. The proposed method combines the advantages of both hard-compression and soft-compression, allowing online query-aware compression while ensures a better compression rate. It is a pretty straightforward and intuitively nice idea.
3. The experiment results are conducted on a wide range of datasets, making the rseults look solid.

### Weaknesses
One possible concern is about the context length. This paper majorly conducts experiments on 128 token chunks, 5/10 chunks per query. This is one possible scenario, but other cases where longer context is applied is not tested in this paper. I would argue that results only under the settings from this paper is not comprehensive enough.

### Questions
Evidence that the method generalizes to longer context cases would be appreciated; does it maintain similar accuracy and end-to-end speed when k and document lengths increase?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces OSCAR (Online Soft Compression for RAG), which tries to "sit in the middle" between (i) online, query-aware hard pruning methods such as Provence and RECOMP, and (ii) offline, high-ratio soft compression methods such as PISCO / earlier context-embedding work. The key idea is to run, at inference time, a query-conditioned compressor LLM that takes (q, dᵢ, MEM tokens) and emits a small set of embedding tokens per retrieved document, which are then fed to the generator LLM instead of the original text. Two realizations are proposed: (1) OSCAR-N-Layers (reuse the first N layers of the generator) and (2) OSCAR-llama (a small 1B Llama with a learned projection). The whole pipeline is trained end-to-end by sequence-level distillation from an uncompressed RAG teacher (Mistral-7B) and can optionally output a reranking score from the same forward pass so that compression is "for free" On six RAG QA benchmarks and on several backbones (Mistral-7B, Qwen2-7B, Mistral-24B, Llama-1B) the authors report 2–5× FLOP savings with little or no accuracy loss relative to uncompressed RAG and better speed/quality than online hard pruning (Provence, RECOMP).

### Strengths
- Well-framed gap and plausible contribution. The paper correctly observes that current online methods are mostly text-level and top out around 2× compression (Provence, RECOMP) while soft methods (PISCO, xRAG/COCOM-style context embeddings) reach 16× but are practically offline and often query-independent. Positioning OSCAR as "query-dependent and online" is a nice conceptual sweet spot. 

- Strong and unusually broad empirical section. The paper does not just show "one backbone, one dataset": it repeats the story on four backbones (including a 24B model), six RAG QA tasks, and also checks harder settings such as BM25-only retrieval, more retrieved docs (up to 50), different compressor depths, and different compressor families. That breadth makes the claim “this is not overfit to one setup” fairly convincing. 

- Factoring compression and reranking in one pass is practically useful. Existing strong RAG pipelines already pay for a cross-encoder reranker; using (almost) the same computation to also produce compressed representations is exactly the kind of engineering win production RAG wants, and it is consistent with how Provence also couples pruning and reranking.

### Weaknesses
- Novelty is a bit overstated. The paper calls OSCAR the "first query-dependent online soft compression," but there is prior art that is close enough that it must be discussed carefully:

    - DODO already performs dynamic contextual compression for decoder-only LMs and can be used as a context compressor at inference. It also reduces the number of hidden states and is explicitly meant to lower long-context cost. 

    - PISCO (2025) explicitly targets 16× compression for RAG and is trained by sequence-level distillation; the main difference is "offline vs. online," but OSCAR’s evaluation makes PISCO pay an online cost, which weakens the fairness of the comparison. 

    - Provence / RECOMP already couple pruning with reranking or selective augmentation; OSCAR's "free compression + rerank" is in the same design space. 

- The paper needs to spell out why these do not qualify as "online soft, query-dependent" in the authors' definition.

- Comparison protocol slightly favors OSCAR. Counting PISCO’s compression as if it had to be done online makes OSCAR look 3–5× faster, but the whole point of PISCO/xRAG/COCOM-like approaches is that you pre-compute or at least pre-compress. Likewise, Provence and RECOMP are mostly LLM-agnostic and don't require per-backbone re-training, while OSCAR has to be retrained for every generator (the authors themselves retrain for Mistral-7B, Qwen2-7B, Mistral-24B, Llama-1B). Deployment cost and model storage are not reported, so the advantages are not end-to-end in a systems sense. 

- Missing or underdeveloped baselines / metrics. The related-work section knows about FINCH, TurboRAG / block-attention RAG, and long-context KV-cache tricks but they are not in the experiments, even though some of them also run online and trade KV size for speed. We also never see actual wall-clock latency or QPS-under-load numbers, only FLOPs on synthetic inputs, so it’s hard to tell whether "OSCAR compressor + generator" < "Provence + generator" on a real A100/H100 with retrieval in front. And everything is single-domain QA with 128-token chunks; real RAG often has heterogeneous, noisy, and much longer pieces.

### Questions
- Scope of "online." In a production RAG service that already runs a DeBERTa-v3 reranker, what is the measured latency (ms) of OSCAR-llama / OSCAR-8-Layers for 10×128-token docs on an A100, and how does that compare to that reranker? Please report wall-clock, not only FLOPs.

- Reusability across queries. Because OSCAR is query-conditioned, you can't pre-compress a corpus once. Did you try a two-stage variant (cheap query-agnostic soft compression at index time + light query adapter at run time)? If not, what prevents it?

- Positioning vs. DODO / TurboRAG / KV-cache compression. What concrete capability do those methods lack that forces you to call OSCAR "the first" online soft compression? In particular, DODO can also run at inference and reduce hidden states. 

- Sensitivity to retrieval quality. You show BM25-only experiments, but only aggregated. Can you provide per-dataset breakdowns and failure cases (e.g., when none of the top-k docs is actually relevant), and how compression interacts with that?

- Teacher choice. You always distill from Mistral-7B. If the teacher is smaller or of lower RAG quality, do you still beat Provence / RECOMP?

### Soundness
3

### Presentation
3

### Contribution
3
