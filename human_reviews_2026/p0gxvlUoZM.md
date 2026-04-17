# LLM-guided Hierarchical Retrieval

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Modern IR systems are increasingly tasked with answering complex, multi-faceted queries that require deep reasoning rather than simple keyword or semantic matching. While LLM based IR has shown great promise, the current retrieve-then-rerank paradigm inherits the limits of embedding-based retrieval, parametric generative approaches are difficult to adapt to new information, and long-in-context approaches that put the entire corpus in context are computationally infeasible for large document corpora due to the quadratic attention complexity. To this end, we introduce a hierarchical retrieval framework LATTICE that enables an LLM to reason and navigate a large corpus with logarithmic search complexity in the number of documents, achieved by imposing a semantic tree structure on the corpus.
Our approach comprises two stages: (1) an offline process where we organize the document collection into a semantic hierarchy - we explore two LLM-driven strategies for this: a bottom-up agglomerative approach and a top-down divisive approach using multi-level summaries;  (2) an online traversal stage where a "search LLM" navigates this tree. A central challenge in using LLMs for search is that the LLM's relevance judgments are *noisy, context-dependent, and unaware of the underlying hierarchy*, making it difficult to compare nodes across different branches and levels of the tree. To solve this, our traversal algorithm estimates calibrated latent relevance scores from the LLM's local outputs, which are combined into a path relevance metric to guide the search globally across the tree. Our training-free framework achieves state-of-the-art zero-shot performance on the reasoning-intensive BRIGHT benchmark (with up to 420K corpus size), demonstrating improvements of up to 9% in Recall@100 and 5% in nDCG@10. Moreover, compared to the highly specialized and fine-tuned SOTA method DIVER-v2, it achieves comparable results on BRIGHT subsets that use a static corpus for evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents LATTICE, a training-free hierarchical retrieval framework designed to help large language models (LLMs) efficiently search large document corpora with logarithmic search complexity. The authors aim to overcome key limitations in current LLM-based information retrieval (IR) systems—namely, the bottlenecks in retrieve-then-rerank pipelines and the scalability challenges of long-context methods.
LATTICE operates in two stages: (1) Offline Stage – Documents are organized into a semantic tree through either a bottom-up or top-down LLM-driven clustering strategy. (2) Online Stage – A search LLM navigates this tree using a traversal algorithm that estimates calibrated latent relevance scores from inherently noisy and context-dependent LLM judgments. These scores are aggregated into a path relevance metric that guides global search decisions.
Empirical results show that LATTICE achieves state-of-the-art zero-shot performance on the reasoning-heavy BRIGHT benchmark, outperforming strong baselines in Recall@100 and nDCG@10, and performing comparably to heavily fine-tuned systems on static corpus subsets.

### Strengths
1. The proposed tree search algorithm and offline tree construction methods are sound and empirically validated.
2. The paper provides comprehensive analyses illustrating the advantages and mechanisms of the approach.
3. The writing is clear and well-organized.

### Weaknesses
1. The proposed solution appears to perform well only under high online LLM budget settings. As shown in Figure 3, performance drops notably when the budget is low, falling behind the baselines.
2. As shown in Table 3, performance is highly sensitive to the choice of offline tree construction strategy (bottom-up vs. top-down). This suggests that selecting the right strategy requires prior knowledge of the corpus structure, which may not always be available in practice.
3. The experiments are only conducted on the Gemini-2.5 family. It would be helpful to see results on other or smaller open-source models for broader validation.
4. The model comparisons may not be entirely fair, since the proposed method uses Gemini-2.5-flash at all stages, while the baselines rely on GPT-4-based query expansion.
5. The evaluation is limited to the BRIGHT dataset. It would be useful to see how the method performs on other retrieval datasets.
6. Compared to other approaches that perform offline indexing on BRIGHT—such as
*Imagine All The Relevance: Scenario-Profiled Indexing with Knowledge Expansion for Dense Retrieval (Lee et al.)* and 
*EnrichIndex: Using LLMs to Enrich Retrieval Indices Offline (Chen et al.)*
—the previously built offline tree structure needs to be reclustered or redivided whenever new documents are added, which could lead to high maintenance costs over time.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenge of using LLMs for complex information retrieval over large corpora, where existing methods are either limited by embeddings, difficult to update, or computationally infeasible. 
The authors propose LATTICE, a hierarchical retrieval framework that structures a document corpus into a semantic tree. An LLM then navigates this tree  to guide the search. 
The experiments are employed on the BRIGHT benchmark, demonstrating good improvements in recall and nDCG compared with baselines.

### Strengths
1. This paper proposes a good LLM-guided hierarchical information retrieval framework. It designs multiple strategies to construct the original corpus into a hierarchical tree during the offline stage, and carefully designs navigation strategies for the online stage.  
2. The paper is  readable, making it easy for readers to understand the authors' motivation and methodology. The complex strategies in the online stage are explained through formulas and pseudocode.  
3. Experiments demonstrate that the proposed method achieves better performance under a zero-shot setting compared to baseline models.

### Weaknesses
1. The paper lacks obvious innovation, as previously published works have already explored similar ideas[1]: transforming large-scale knowledge corpora into hierarchical trees and designing navigation strategies for traversal and filtering. 
2. The paper only employs the BRIGHT benchmark for experiments. Although this benchmark contains multiple subsets, these subsets were all constructed by the same research team, resulting in a uniform pattern across them. This raises concerns about the generalizability of the proposed method.
3. Compared to the fine-tuned DIVER v2, the method proposed in this paper does not demonstrate a clear performance advantage. Although the proposed approach requires no training cost, it incurs offline construction costs. The paper fails to provide a clear comparison between the offline construction cost of their method and the training cost of fine-tuned DIVER v2. This leads to doubts about whether the proposed method achieves only marginal performance gains at a potentially higher overall cost.

[1] Hierarchical Document Refinement for Long-context Retrieval-augmented Generation. ACL 2025

### Questions
1. What is the core innovation of the proposed method compared to previous work [1] ?
2. Which one is higher: the offline construction cost of this method or the training cost of fine-tuned DIVER v2?
3. The offline construction phase uses Gemini-2.5-flash. Have other large language models been tried, and how much would using different models impact the final results?

[1] Hierarchical Document Refinement for Long-context Retrieval-augmented Generation. ACL 2025

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces LATTICE, a hierarchical retrieval framework that enables large language models (LLMs) to perform reasoning-driven search over large document corpora with logarithmic complexity. It organizes the corpus into a semantic tree structure using either a bottom-up agglomerative or top-down divisive strategy, and employs an LLM-guided traversal algorithm that estimates calibrated relevance scores to navigate the hierarchy effectively. The framework is training-free and demonstrates strong zero-shot retrieval performance on the reasoning-intensive benchmark BRIGHT.

### Strengths
1) The paper presents a hierarchical retrieval framework that integrates LLM reasoning with structured corpus organization.
2) The proposed method is clearly motivated and effectively described.
3) The framework shows potential for improving reasoning-oriented retrieval.

### Weaknesses
1) The approach has not been validated on large-scale, open-domain corpora, leaving its scalability and generalization uncertain.
2) The discussion of related work is incomplete, omitting several recent advances in hierarchical and structure-aware retrieval.
3) The paper lacks analysis of efficiency and computational cost during both tree construction and traversal stages.
4) The presentation could be improved. The paper introduces the search process before explaining tree construction and lacks a concluding section summarizing key insights and limitations.
5) The evaluation relies solely on the proprietary Gemini-2.5-flash model and a single reasoning-intensive benchmark, limiting the understanding of model dependence and robustness across different retrieval settings.

### Questions
1) How does the semantic hierarchy scale in both construction and traversal time when applied to large-scale, open-domain corpora (e.g., millions of documents)?
2) Beyond BRIGHT, have the authors evaluated LATTICE on more general retrieval datasets such as MS MARCO or Natural Questions to assess robustness and generalization?
3) How do smaller or open-source LLMs perform within this framework? Is the approach dependent on the reasoning strength of proprietary models like Gemini-2.5-flash?
4) Could the authors provide a quantitative or qualitative comparison of the bottom-up vs. top-down semantic tree construction strategies? In what data conditions should one prefer either method?
5) Since traversal efficiency is central to the claim of logarithmic complexity, can the authors provide empirical runtime comparisons against standard reranking pipelines?
6) How is the semantic hierarchy updated when new documents are added? Does the model require full reconstruction, or can it support incremental updates?

### Soundness
3

### Presentation
2

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
The paper proposes LATTICE, an LLM-guided hierarchical retrieval framework that organizes a corpus into a semantic tree offline and lets a “search LLM” traverse it online using calibrated path-relevance scores. On BRIGHT, it achieves strong zero-shot results and favorable cost–quality scaling compared to retrieve-then-rerank, with clear ablations on calibration and traversal design.

### Strengths
1. The proposed method is well-motivated and novel. The latent-score calibration and path-relevance update are interesting and reasonable.

2. Strong empirical results and thoughtful analysis.

3. Under larger token budgets, the method scales better than reranking.

### Weaknesses
1. Offline tree construction is expensive and appears data-sensitive (e.g., Table 3). Maintaining the tree for dynamic corpus (add/edit/delete) is nontrivial, as internal summaries can become stale. These issues may hinder real-world adoption.

2. This paper could benefit from more comparisons to agentic methods. The argument that “agents call a retrieval tool while LATTICE is the core retrieval mechanism” is not fully convincing. Both approaches rely on text embeddings but mainly in that LATTICE pre-clusters and has the LLM walk over clustered tree anchors, whereas an agent can pick an anchor (the query embedding) and check neighboring documents.  I believe more in-depth comparisons would help.

3. It is unclear how corpus size affects performance (both tree construction and search under a given budget). The BRIGHT corpus is relatively artificial and small, comparisons on larger datasets (eg, BEIR) would be helpful.

### Questions
1. How LATTICE maintains the diversity of retrieved results

### Soundness
3

### Presentation
3

### Contribution
3
