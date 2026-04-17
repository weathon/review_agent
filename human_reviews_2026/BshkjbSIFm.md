# Your Dense Retriever is Secretly an Expeditious Reasoner

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 6, 2

## Abstract
Dense retrievers enhance retrieval by encoding queries and documents into continuous vectors, but they often struggle with reasoning-intensive queries. Although Large Language Models (LLMs) can reformulate queries to capture complex reasoning, applying them universally incurs significant computational cost. In this work, we propose Adaptive Query Reasoning (AdaQR), a hybrid query rewriting framework. Within this framework, a Reasoner Router dynamically directs each query to either fast dense reasoning or deep LLM reasoning. 
The dense reasoning is achieved by the Dense Reasoner, which performs LLM-style reasoning directly in the embedding space, enabling a controllable trade-off between efficiency and accuracy. Experiments on large-scale retrieval benchmarks BRIGHT show that AdaQR reduces reasoning cost by 28% while preserving—or even improving—retrieval performance by 7%.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents AdaQR, a framework to improve the efficiency of query rewriting. The main idea is that for in-distribution queries, the system can directly map a query embedding to its corresponding “reasoned query” embedding. Normally, this would require using an LLM to generate a reasoning chain and then embedding that chain with the embedding model. Experiments on BRIGHT demonstrate both performance and cost benefits for the proposed approach.

### Strengths
1. The idea is novel. Directly navigating the query embedding space to obtain a reasoned query embedding is interesting and new. The paper provides solid empirical results demonstrating advantages in both efficiency and performance.
2. The motivation is strong. The pilot study analyzes the mean resultant length between the original query embedding and the transformed query embedding across seven different reasoners and five different embedding models, and shows strong alignment. This offers empirical evidence supporting the core idea of the paper.

### Weaknesses
1. Compared to an out-of-the-box retrieval framework, the method introduces new modules that require training for each LLM reasoner/embedding model combination.
2. The method adds computational overhead. Each retrieval now includes: embedding the query for the Reasoner Router to decide between LLM-enabled query rewriting and the Dense Reasoner; if routed to the dense path, (1) embed the query and (2) apply the Dense Reasoner to the query; if routed to the LLM path, (1) generate a reasoning chain with the LLM and (2) embed the enhanced query.
3. The paper does not clearly define how computational cost is measured. Although cost reduction is mentioned in several places, it is unclear whether this refers to end-to-end latency, FLOPs, dollar cost, or another metric. It is also unclear how this cost varies with the choice of LLM reasoner and embedding model.
4. The BRIGHT data split is unclear. The baseline LLM reasoning approach has no trainable components, while AdaQR does. It is therefore unsurprising that training on BRIGHT could improve nDCG relative to out-of-the-box LLM reasoners. I am concerned that the benefit may come from overfitting to BRIGHT’s query styles. BRIGHT is a meta-dataset with multiple data sources; lines 250–253 do not make it clear whether the 70/30 split is stratified by source or simply a random split across all instances.

### Questions
- The pilot study is helpful, but how should we conceptually understand the phenomenon of performing reasoning directly in the embedding space? An analysis of the router’s decisions—i.e., what types of queries are sent to the Dense Reasoner versus the LLM reasoner—would help interpret what the Dense Reasoner has learned.
- Conceptually, the Dense Reasoner seems to standardize queries and map them into a more canonical direction. It is unclear how much genuine “reasoning” it performs beyond normalization or transformation of embeddings.
- What is the architecture of the Dense Reasoner? I could not find a detailed description of its implementation in the paper.
- I was not able to find implementation and training details for the Dense Reasoner and the router in the appendix. Please provide comprehensive details (model architectures, training objectives, hyperparameters, and compute budget) to enable reproducibility.

### Soundness
2

### Presentation
2

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
To address the high cost of using LLMs for query rewriting in dense retrieval, the authors propose AdaQR. This framework dynamically routes queries: dynamically send queries to either a full LLM or a lightweight "Dense Reasoner" that operates in the embedding space, balancing performance and efficiency.

### Strengths
1. The design of the Dense Reasoner is interesting. By directly learning to approximate the embeddings of LLM-rewritten queries, it provides a lightweight rewriting mechanism
2. The Router appears capable of balancing the trade-off between the efficient but less accurate Dense Reasoner and the high-performance but costly LLM-based method.

### Weaknesses
According to Figure 6, AdaQR is shown to outperform both the LLM-only and Dense Reasoner-only baselines. This is surprising, as the performance of a router-based method like AdaQR is typically bounded by its components. This result suggests that the LLM and Dense Reasoner are complementary, with the router sending queries the LLM fails on to a successful Dense Reasoner. This I believe somehow contradicts the core design, where the Dense Reasoner is trained to mimic the LLM. The paper needs a deeper analysis to resolve this paradox and explain how a model trained for imitation can correct its teacher.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
To improve the efficiency of reasoning-intensive retrieval, this paper proposes adaptive query reasoning (AdaQR). The framework introduces a Dense Reasoner and a Reasoner Router to the retrieval pipeline. In particular, the Dense Reasoner transforms the embedding from the original query space to a reasoned query space. This bypasses the costly LLM rewriter + dense retriever approach. As the Dense Reasoner is small, a Reasoner Router is introduced to check whether the query should go through the Dense Reasoner or the LLM Reasoner. Empirically, on the BRIGHT reasoning-intensive retrieval dataset, AdaQR achieves significant cost reduction while surprisingly also results in improved performance.

### Strengths
1. Training a dense reasoner to induce reasoned embedding is a very cool idea. Pairing it with a router is capable of leveraging its efficiency benefit while even improving performance. 
2. It’s quite surprising to see that sometimes the dense reasoner can offer advantages over the LLM reasoner, so sometimes the queries just should not be rewritten at all, which can somewhat be captured by this dense reasoner. 
3. The evaluations and ablation studies are comprehensive with promising gains. 
4. The paper is generally well-written.

### Weaknesses
1. The Dense Reasoner requires accessing 70% BRIGHT’s ground-truth query and document as training data. The Reasoner Router requires knowing the in-domain query embedding as the Oracle Anchor beforehand. Both prevent the framework from generalizing to unseen domains. 
2. Some details are missing. For example, (a) it is unclear how the cost is calculated, (b) it is unclear how to ensure the pretraining corpus has no overlap with queries from BRIGHT.  Providing more details about that would make the results more convincing.

### Questions
1. How much traffic is routed to dense reasoner with respect to $\tau$? It would be useful to visualize this with a plot to see more details about the efficiency gain. 
2. The dense reasoner has a very small size. Do the authors have suggestions on how its size affects the performance and how its size should be determined? 
3. How does the Dense Reasoner perform if it is only pretrained without in-domain adaptation on ReasonIR? Does it also have only a slight decline? 
4. Other than ReasonIR-8B, has the author tried other reasoning-intensive retrieval models?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes an Adaptive Query Reasoning (AdaQR) framework that routes queries between an LLM based reasoning path vs a dense retriever trained to map an input query to an vector for an output query that represents a reasoned query. 

The three main components are:
1. LLM Reasoner: The standard, high-cost approach of using an LLM to auto-regressively rewrite the query text.
2. Dense Reasoner: This is a lightweight model (a two-layer MLP) trained to approximate the effect of the LLM Reasoner. It operates directly in the embedding space, learning a transformation that maps an original query's embedding to an embedding that is close to the embedding of the LLM-rewritten query. This is based on the paper's key empirical finding that these embedding-space transformations are not random but "systematic [and] structured."
3. Reasoner Router: A lightweight routing mechanism that directs incoming queries. It compares a query's embedding to a pre-computed "oracle anchor" (an average of queries known to be handled well by the DR). Queries similar to this anchor are sent to the fast Dense Reasoner; all others are routed to the expensive LLM Reasoner.

Experiments on the reasoning-intensive BRIGHT benchmark, conducted across 5 retrievers and 17 LLM reasoners, show that AdaQR, on average, reduces the reasoning cost by 28.12% while simultaneously improving retrieval performance (nDCG@10) by 7.24%.

### Strengths
1. Provides a trainable solution for reasoning routing showcasing cost reduction on a variety of tasks
2. Showcases that reasoned queries are not random transforms of the input query

### Weaknesses
1. Misleading title. It is not a single model but a multi-stage pipeline that requires: (1) pre-training a Dense Reasoner on an external corpus, (2) fine-tuning the Dense Reasoner on an in-domain dataset, (3) building an "oracle anchor" for the router from the same in-domain data, and (4) deploying and maintaining both the lightweight DR and the full, expensive LLM Reasoner in production. How is the dense retriever a secret reasoner if it has been explicitly trained to learn the mapping?
2. The Reasoner Router is not self-learning; it relies on an "oracle anchor" built from a set of queries that are already known to be "predictable" by the Dense Reasoner. This requires an in-domain, labeled training set (in this case, 70% of the BRIGHT dataset) to construct the oracle, introducing a significant data dependency for the routing component. How will this generalize to other domains? 
3. The routing mechanism is controlled by a similarity threshold that is "determine[d] empirically" and differs for each retriever backbone (e.g., 0.75 for BGE-Large, 0.6 for Qwen3-Embedding-0.6B). This is a sensitive, non-trivial hyper-parameter that requires manual tuning for any new deployment.
4. For custom deployments, the framework reduces the average reasoning cost but does not eliminate the peak deployment cost. The expensive LLM Reasoner must still be loaded in VRAM and available to serve "hard" queries, meaning the full hardware cost of the system is still incurred, even if the LLM is queried less frequently.

### Questions
na

### Soundness
2

### Presentation
2

### Contribution
2
