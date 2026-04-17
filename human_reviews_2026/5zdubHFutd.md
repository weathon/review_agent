# GRO-RAG: Gradient-aware Re-rank Optimization for Multi-source Retrieval-Augmented Generation

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Retrieval-Augmented Generation (RAG) systems often rely on information retrieved from heterogeneous sources to support generation tasks. However, existing approaches typically either aggregate all sources uniformly or statically select a single source, neglecting semantic complementarity. Moreover, they commonly employ re-ranking models to obtain Top-k documents, without accounting for actual contribution to generation objective.
In this paper, we propose GRO-RAG, a training-free, gradient-aware re-ranking framework for multi-source RAG. 
Our method performs Top-k document selection by reading gradients from the language model, estimating each document’s contribution to the generation loss through a single backward pass. 
This enables re-ranking not by heuristic relevance, but by direct feedback from LLM's generation objective. 
At the source level, we incorporate inter-source redundancy and query relevance to select source combination prior to re-ranking.
Theoretically, we prove that this gradient-based Top-k selection approximates the optimal subset minimizing the generation loss, and aligns with minimizing the leave-one-out loss upper bound.
Experiments across multi-source QA and open-domain generation tasks demonstrate consistent improvements in generation quality, highlighting the importance of generation-aware retrieval selection in multi-source RAG.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose GRO-RAG, a training-free framework for improving multi-source RAG by integrating a gradient-aware re-ranking mechanism. The framework selects a subset of sources by optimizing for the minimization of redundancy, and performs a re-ranking of retrieved documents based on their contribution to the final generation output. The method is evaluated across several multi-hop QA datasets, showing consistent improvements over several baselines.

### Strengths
1. The method introduces a gradient-based re-ranking approach backed by solid theoretical analysis, with proofs showing that the gradient-based selection approximates the optimal utility-maximizing solution, providing strong guarantees for its effectiveness.
2. GRO-RAG is training-free,    making it practical for deployment in existing systems without requiring additional model training.
3. The experiments and ablation studies show consistent performance improvements.

### Weaknesses
1.  The gradient-based relevance estimation relies on the first-order Taylor expansion (Eq. 5), implicitly assuming local smoothness of the loss landscape. The paper does not provide quantitative evidence on the approximation error when the loss function exhibits strong nonlinearity.

2.Proposition 3.2 shows linear convergence under strong-convexity and L-smoothness assumptions. However, the generation loss of a frozen LLM is generally non-convex. It remains unclear whether monotonic improvement still holds empirically when these assumptions are violated.

3.The scoring function (Eq. 6) assumes additive document contributions, while semantically dependent passages may only be useful jointly. The treatment of such higher-order interaction effects is not addressed or evaluated.

### Questions
See weaknesses.

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
5

### Summary
This paper presents **GRO-RAG**, a gradient-aware and training-free framework for improving document selection in multi-source Retrieval-Augmented Generation (RAG). The method addresses a key limitation of existing RAG systems that rely on heuristic or similarity-based document ranking, which often fails to reflect how retrieved passages affect the downstream generation objective. GRO-RAG introduces a two-stage approach: first, a source combination selection module balances query relevance and inter-source redundancy through submodular optimization; second, a gradient-based re-ranking mechanism estimates each document’s contribution to generation loss using a single backward pass through a frozen language model.  

The framework is theoretically grounded. The authors show that gradient-derived importance scores approximate a leave-one-out loss bound and empirically validate the approach on multi-hop QA benchmarks. The method consistently outperforms standard RAG and multi-source baselines without requiring fine-tuning or additional parameters. The overall goal is to align retrieval selection more directly with the LLM’s generation behavior in a model-agnostic and computationally efficient manner.

### Strengths
* This work provides an elegant and principled perspective on retrieval optimization by connecting document ranking directly to the generation loss. 

* The central idea of using LLM gradients as feedback for re-ranking is conceptually novel yet intuitive, bridging the gap between retrieval relevance and downstream utility. 

* The proposed method remains entirely training-free, introducing no additional parameters and operating seamlessly with frozen LLMs, which makes it appealing for practical deployment in resource-limited or privacy-sensitive contexts.  

 Overall, the study introduces a solid and thought-provoking idea that advances the understanding of generation-aware retrieval.

### Weaknesses
* While the core contribution is technically sound, the evaluation and discussion could be broader. All experiments focus on question answering, leaving uncertainty about how well the method generalizes to other forms of retrieval-augmented generation such as summarization or dialogue where grounding is less explicit. Although the method itself is task-agnostic, the paper would benefit from more explicit reasoning about its broader applicability.  

* From a practical standpoint, the backward pass required for each query may raise latency concerns in large-scale applications. Although the discussion briefly mentions this issue, a clearer contextualization of the trade-off between computational cost and quality improvement would make the work stronger.  

* The gradient linearization assumption may also be optimistic given the non-linear nature of LLM loss landscapes. The theoretical discussion on convexity and smoothness provides a mathematical foundation but may not fully hold for transformer architectures. Clarifying how these assumptions should be interpreted in practice would improve the overall balance between theory and application.  

In summary, the idea is creative and meaningful, but the empirical scope and certain discussions fall slightly short of the depth expected for ICLR acceptance.

### Questions
1. Could the authors provide more intuition about how the gradient-based re-ranking signal behaves in practice? For example, how should one interpret documents with small or negative gradient alignments in terms of their contribution to generation quality?  
2. The paper mentions that a backward pass adds latency but remains training-free. Can the authors contextualize this cost relative to other optimization-free enhancements such as reflection-based or reranking-only pipelines?  
3. How was the relevance–redundancy trade-off parameter λ chosen? Is it generally robust, or does it require tuning across datasets?  
4. The theoretical analysis relies on convexity and smoothness assumptions that may not hold strictly for transformer-based models. Could the authors clarify how these assumptions should be viewed, primarily as guiding intuition or as formal guarantees?  
5. The gradient linearization is a key design choice. Could the authors discuss under what conditions it might fail, and how practitioners should interpret its limitations when applying GRO-RAG in different settings?

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
4

### Summary
The paper proposes a training-free re-ranking framework that selects sources by a relevance–redundancy tradeoff and then ranks documents by how strongly each aligns with the LLM’s generation-loss gradient. The re-ranking score is the inner product between a passage embedding and the loss gradient, which approximates loss reduction under a first-order expansion and yields Top-k selection in one backward pass. A theoretical result upper-bounds leave-one-out utility by the gradient score and a second result shows linear convergence for an optional iterative mixture update under smooth-strongly convex assumptions. Experiments on HotpotQA, 2WikiMultihopQA, and MuSiQue with Llama-3-8B and GLM-4 report consistent EM/F1 gains over vanilla, single-source, and multi-source baselines.

### Strengths
(1) The gradient-aware scorer ties ranking directly to the generator’s objective rather than query similarity, which better matches end-task utility. 

(2) The method is training-free and uses a single forward–backward pass per query, so it plugs into frozen LLMs without extra parameters. 

(3) As a re-ranker over a fixed pool, it approaches dense retrievers in NDCG and surpasses them on the hardest dataset (MuSiQue) in at least one backbone configuration.

### Weaknesses
(1) The linearization around a uniform mixture can misestimate interactions among passages when the loss landscape is highly non-linear. 

(2) The convergence guarantee assumes smooth-strongly convex behavior in the span of passage embeddings, which is unlikely to hold strictly for modern LLMs. 

(3) Latency increases materially because a backward pass is required, which is orders of magnitude slower than dense retrieval and may constrain interactive use. 

(4) The limitations section acknowledges dependence on local linear approximations and fixed document encoders, which could cap performance in complex settings.

### Questions
(1)How stable are gradient scores across decoding temperatures or prompt formats and do these scores correlate with true leave-one-out gains on held-out queries. 

(2)What is the end-to-end latency profile broken down into retrieval, gradient computation, and generation, including tail percentiles under web-augmented scenarios.

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
3

### Summary
The paper backs this with theory (a leave-one-out upper bound and a simple iterative refinement view) and shows the setup is training-free for frozen LLMs. It first picks which sources to use by balancing query relevance with cross-source redundancy so you don’t haul in a pile of near-duplicates. It then ranks candidate passages by how much each would drop the LLM’s loss, using one forward-backward pass and an inner-product score.

### Strengths
1. Ranking is tied to the generator’s actual objective instead of just query similarity, which is exactly what RAG needs. 
2. The source router uses a clean relevance-diversity trade-off with a greedy submodular pick that has a standard approximation guarantee. 
3. It slots into frozen models with no extra training and keeps the compute to a single backward pass per query.

### Weaknesses
1. A backward pass per query adds real latency compared to retriever-only pipelines, so the deployment fit needs clearer bounds.

2. The submodularity claim relies on a small trade-off weight and cosine redundancy, which may not hold up uniformly across domains.

3. Source selection builds on fixed document embeddings rather than a query-adaptive encoder, which can undercut routing on niche topics

### Questions
1. How sensitive is source selection to the redundancy weight and to the encoder choice used to summarize each source

2. Is there a simple recipe for the number of refinement steps that balances marginal gains against extra backward passes.

### Soundness
3

### Presentation
2

### Contribution
3
