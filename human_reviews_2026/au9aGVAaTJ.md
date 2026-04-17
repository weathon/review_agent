# FREESON: Retriever-Free Retrieval-Augmented Reasoning via Corpus-Traversing MCTS

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Large Reasoning Models (LRMs) have demonstrated remarkable capabilities in multi-step reasoning and calling search engines at appropriate steps. However, existing retrieval-augmented reasoning approaches rely on separate retrieval models, limiting the LRM's role in retrieval to deciding when to retrieve and how to query. This separation not only increases hardware and operational costs but also leads to errors in the retrieval process due to the representation bottleneck, a phenomenon where the retriever's embedding space lacks sufficient expressiveness to capture the distinctions required by the generator. To address this, we shift our perspective on retrieval from sequence-to-sequence matching to locating the answer-containing paths within the corpus, and propose a novel framework called FREESON (Retriever-FREE Retrieval-Augmented ReaSONing). This framework enables LRMs to directly interact with external knowledge by acting as both a generator and a retriever, thereby autonomously acquiring relevant information. To achieve this, we introduce a variant of the MCTS algorithm specialized for the retrieval task, which we call CT-MCTS (Corpus-Traversing Monte Carlo Tree Search). In this algorithm, LRMs traverse through the corpus toward answer-containing regions. Experiments on five open-domain QA benchmarks covering both single-hop and multi-hop questions demonstrate that FREESON achieves an average improvement of 14.4% in EM and F1 over four multi-step reasoning models with a separate retriever, and it also performs comparably to the strongest baseline, surpassing it by 3% on PopQA and 2WikiMultihopQA, and by 12% on the fact-checking benchmark FEVER.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a new knowledge indexing and retrieval method for retrieval-augmented generation (RAG), especially for the large reasoning models, since the inference process depends on the strong reasoning ability over the corpus. 
The interesting contribution lies in the elimination of separate retrievers, unifying the retrieval and generation in a single model.
To facilitate this, a retrieval method called CT-MCTS (a variant of MCTS) is designed.
Experiments on five QA datasets show its effectiveness.

### Strengths
- This paper targets an important question in RAG, i.e., the representation bottleneck in retrieval.

- Eliminating the retriever using the unified generator itself is interesting.

### Weaknesses
- Although the proposed method demonstrates improved efficiency in knowledge indexing, this is essentially a one-time cost. During the more frequent inference phase, the method incurs significantly higher computational overhead (425×, i.e., 1.88e13 vs. 4.42e10), which limits its practical applicability. 

- The method relies heavily on multiple off-the-shelf components (e.g., TreeCorpus, UCT). Consequently, the overall inference process depends on the performance of these external algorithms, and it remains unclear whether errors from these components may propagate and affect the final results.

- The criterion for determining “when search is required” is not clearly defined. 

- A comparison of knowledge indexing and retrieval efficiency across corpora of varying sizes would be valuable for assessing the scalability of the proposed approach.

- In terms of main results, the method does not show consistent improvements and underperforms Search-R1 on TrivialQA, HotpotQA, and MuSiQue.

- The analysis experiments are conducted on different datasets, raising concerns about potential cherry-picking. Performing the analyses on a consistent set of datasets would provide a more comprehensive and reliable evaluation.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes FREESON, a retriever-free retrieval-augmented reasoning framework based on a Corpus-Traversing Monte Carlo Tree Search (CT-MCTS). Instead of using a separate dense retriever, FREESON enables a large reasoning model (LRM) to directly explore the corpus via prefix-constrained decoding over an FM-Index–based structure (CorpusTree). Each node in the search tree corresponds to a valid text prefix present in the corpus, and a learned value network estimates the reward (answer correctness) to guide the search. The approach is evaluated on several QA and fact-checking benchmarks, showing improved or comparable performance to retriever-based methods.

### Strengths
1. Conceptually interesting idea: Reformulating retrieval as a constrained search directly over the corpus is novel and theoretically appealing. It removes the embedding bottleneck and the need for a separate retriever.
2. Introduction of the CorpusTree abstraction: The paper creatively builds on the FM-Index to define a CorpusTree, which represents the corpus as a prefix-constrained traversal space. This abstraction is elegant and enables efficient legality checking of generated prefixes during decoding. Integrating such a structure with LLM-based reasoning and MCTS constitutes a clear system-level innovation.
3. Comprehensive empirical analysis: The paper provides ablations on the number of expansion nodes (M), token granularity (G), and value network training strategies (on-policy vs off-policy), helping to understand the method’s behavior.
4. Implementation detail clarity: The description of CT-MCTS integration with LLM decoding and FM-Index–based prefix constraints is technically solid and well-articulated.
5. Quantitative improvements: FREESON achieves notable accuracy gains on datasets such as PopQA, 2WikiMultihopQA, and FEVER.

### Weaknesses
1. Limited novelty over existing MCTS-based reasoning frameworks: While the idea of corpus traversal is positioned as new, the core algorithm still heavily relies on standard MCTS machinery. The adaptation to prefix-constrained decoding is incremental, and the novelty is primarily in system integration rather than algorithmic advancement. 
2. Severe inference-time inefficiency: The CT-MCTS search requires multiple expansions (M=2) and 32 simulations per query, resulting in about 1.88×10^13 FLOPs, over 400× higher than a dual-encoder baseline. The paper acknowledges this but does not provide any practical mitigation (e.g., pruning or speculative decoding). Thus, the method is unlikely to be feasible for real-world deployment.
3. Limited search coverage due to shallow traversal depth: The maximum search length is restricted to G * d_{max} = 6 * 16 = 96 tokens. This means CT-MCTS only traverses approximately one paragraph, which may suffice for small datasets like FEVER or PopQA but is clearly insufficient for real-world or web-scale retrieval, where key evidence may appear hundreds of tokens deep. Consequently, the framework is constrained to small, static corpora rather than scalable retrieval scenarios.
4. Poor cost–effectiveness: efficiency drops sharply without clear accuracy gains. Despite introducing massive computational overhead, FREESON does not show consistent superiority over strong baselines such as search–r1. On datasets like TriviaQA and HotpotQA, its performance is even lower than search–r1, indicating that the large increase in inference cost does not translate into meaningful quality improvement. This undermines the paper’s main claim that CT-MCTS leads to “retriever-free yet more accurate” reasoning.
5. Questionable scalability to dynamic or open-domain corpora: The FM-Index–based CorpusTree must be prebuilt and is not easily updatable. The paper provides no mechanism for incremental index updates, making the approach unsuitable for continuously evolving or web-scale text collections.
6. High engineering complexity and model dependency: The method requires access to LLM logits and hidden states for value estimation, which is not supported by most API-based LLMs. It also needs an on-policy value network trained with large-scale GPU resources (80GB A100). This greatly limits reproducibility and accessibility.
7. Unclear fairness in baseline comparisons: Some baseline configurations (retriever types, retrieval depth, index sizes) are insufficiently detailed. Without standardized resource budgets, the reported accuracy gains might not purely reflect algorithmic improvement.

### Questions
1. Provide detailed latency, FLOPs, and wall-clock comparisons against retriever-based baselines, especially search–r1.
2. Investigate lightweight or approximate variants of CT-MCTS (e.g., pruning, speculative decoding) to improve scalability.
3. Include per-dataset ablations showing where CT-MCTS fails or underperforms compared to standard retrieval-based search.
4. Discuss how the approach could handle longer documents or larger, dynamically updated corpora.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes to index documents in a tree structure for retrieval augmented generation (RAG) in order to retrieve arbitrary contexts for reasoning. Basic idea of RAG is to retrieve relevant passages using an external model, e.g., encoder, but it is suboptimal given the ambiguity of queries. The proposed approach leverages Monte Carlo Tree Search to retrieve relevant document by first indexing all the documents as a tree structure and by searching for the relevant ones with a value network learned separately. Experiments are carried out on standard benchmarks, general QA and multi-hop QA, demonstrating superior performance when compared with other relevant baselines, e.g., Search-R1.

### Strengths
- This work is presenting an interesting idea of indexing documents in a tree structure, which allows a more compact representation of documents by encoding multiple tokens as a single node. The search is carried out over the tree structure allowing multiple nodes in a single step for faster inference with fine-grained control by a value network trained separately.
- Experimental results show competitive results against several baselines of RAG combined with reasoning abilities, e.g., Search-R1. Analysis of the node granularity and the multiple expansion shows the advantage of the proposed method.
- The resource consumption measured by memory is smaller when compared with a conventional encoder-based retrieval, but demands more computation for inference, which could be future studies.

### Weaknesses
- Writing should be improved. There exist many terminologies, in particular, acronyms, not defined clearly, and thus, they could be interpreted arbitrary leading to misunderstanding. Examples are: ANN in line 119 and LRM in line 154 (which is only defined in abstract, but not in the main text). Also, $\mathcal{R}$ is not defined in line 171.

### Questions
None

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
FREESON lets one LLM both find and read the answer—no separate retriever. It builds a fast prefix index over the corpus so the model can only “type” sequences that actually exist in the docs, uses a smart search (CT-MCTS) to land on promising spots, then hands the full matching documents to the LLM to think and answer.

### Strengths
1. It uses one system instead of a separate retriever, so there’s less to train, tune, and break.
2. The prefix index keeps the model grounded in text that actually exists in the corpus.
3. The CT-MCTS search can explore and recover from early mistakes better than greedy/beam.

### Weaknesses
1. The compared baselines are a bit outdated that many newer works such as WebWalker, WebThinker, ASearcher, MiroThinker etc. are not compared with.
2. The paper is written in a way that is hard-to-understand the concrete implementation. Details of the method design, e.g., how the index is constructed and choosing the particular objective needs further expansion to facilitate understanding.
3. The search latency is in the **25-65 seconds** range, which is too slow for real applications. The author didn't compare the latency of the proposed method with baselines; a further discussion on the usability needs to be adjusted.

### Questions
1. In Figure 2, the decline from G=6 to G=10 is not obvious and seems within variance. Could you extend the ablation beyond G=10 to show when the performance shows significant drops?
2. Which 7B did you use for Freeson? Does it start with the same base model as the baselines? Details about the training setup is missing in the experiment section. If you're using Search-R1 checkpoint, how would you justify the performance drop on some of the benchmarks?
3. Could you discuss the scalability of the proposed method in terms of the Corpus size and compare the efficiency with baselines?

### Soundness
3

### Presentation
2

### Contribution
3
