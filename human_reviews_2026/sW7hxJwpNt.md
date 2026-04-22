# Directed Information $\gamma$-covering: An Information-Theoretic Framework for Context Engineering

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
We introduce \textbf{Directed Information $\gamma$-covering}, a simple but general framework for redundancy-aware context engineering. Directed information (DI), a causal analogue of mutual information, measures asymmetric predictiveness between chunks. If $\DI_{i \to j} \ge H(C_j) - \gamma$, then $C_i$ suffices to represent $C_j$ up to $\gamma$ bits. Building on this criterion, we formulate context selection as a $\gamma$-cover problem and propose a greedy algorithm with provable guarantees: it preserves query information within bounded slack, inherits $(1+\ln n)$ and $(1-1/e)$ approximations from submodular set cover, and enforces a diversity margin. Importantly, building the $\gamma$-cover is \emph{query-agnostic}: it incurs no online cost and can be computed once offline and amortized across all queries. Experiments on HotpotQA show that $\gamma$-covering consistently improves over BM25, a competitive baseline, and provides clear advantages in hard-decision regimes such as context compression and single-slot prompt selection. These results establish DI $\gamma$-covering as a principled, self-organizing backbone for modern LLM pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel and principled information-theoretic framework for context engineering, called Directed Information $\gamma$-Covering. The core idea is to leverage the asymmetric, predictive nature of Directed Information (DI) to build a query-agnostic "cover set" of context chunks. The authors formally define a $\gamma$-cover as a subset of chunks $S$ such that any chunk $j$ not in $S$ is "covered" by some chunk $i$ in $S$, meaning $DI_{i\rightarrow j} \ge H(C_j) - \gamma$. This criterion guarantees that $C_i$ can represent $C_j$ up to a $\gamma$-bit information loss.

The authors provide a crucial theoretical bridge (Lemma 3.1) linking this query-agnostic DI structure to query-dependent Pointwise Mutual Information (PMI), which is a common measure of relevance in RAG. They formulate the selection of this cover set as a submodular set cover problem, for which they propose a greedy algorithm (Algorithm 1) that inherits a $(1-1/e)$ approximation guarantee. The framework is supported by theoretical guarantees for information preservation (soundness, Theorem 3.5) and non-redundancy (diversity, Proposition 3.2). Empirically, the authors apply this framework to context compression, system prompt selection, and reranking on HotpotQA. The results show that $\gamma$-covering provides clear advantages in "hard-decision" regimes (e.g., high compression, single-prompt selection) and offers modest, consistent gains in reranking over a strong BM25 baseline, all while being a query-agnostic method that can be computed offline.

### Strengths
- The application of Directed Information to context engineering is, to my knowledge, novel. Using the asymmetry of DI ($DI_{i\rightarrow j}$ vs. $DI_{j\rightarrow i}$) to model predictive relationships is a clever and fine-grained approach, moving beyond more common symmetric measures like MI or perplexity. The paper is theoretically dense and well-constructed.

-  The authors do an excellent job of building the framework from the ground up. The bridge from query-dependent PMI to the query-agnostic DI covering criterion (Lemma 3.1 and its corollaries) is the theoretical cornerstone that justifies the entire offline approach. This is a significant contribution.

- Connecting the problem to submodular set cover is a major strength. This immediately allows the authors to import strong, well-understood approximation guarantees (Theorem 3.4) for their greedy selection algorithm. The additional formal guarantees for soundness (information preservation up to an additive slack) and diversity (an information-theoretic margin between representatives) provide a solid, principled foundation for why this method should work.

### Weaknesses
-  The theoretical guarantees (e.g., Theorem 3.5) are derived with respect to the true distributions ($p^{\*}$) and information measures ($DI^\*, H^\*$). The practical implementation, however, relies on empirical estimators from a model $p_\theta$.
   - The "Safe Pruning" guarantee (Theorem 3.3) shows that estimation errors from all components ($\delta_i, \delta_j, \eta_j, \epsilon_{ij}$) accumulate.
   - Vacuous Soundness Bound? This culminates in the final soundness bound (Theorem 3.5), where the total information slack is $|U \setminus S|(\gamma + 3\bar{\delta})$. In a setting with many chunks to be compressed (large $|U \setminus S|$), this additive error bound could become very loose or even vacuous, especially if the uniform estimation error $\bar{\delta}$ is non-trivial. The paper lacks a discussion or empirical characterization of this crucial estimation error, which is the primary link between the theory and the practice.

- The entire framework is parameterized by the tolerance $\gamma$. This parameter seems to be the most important hyperparameter, as it directly controls the trade-off between compression (cover size $|S|$) and information loss. How is $\gamma$ chosen? Is it tuned per task, per dataset, or set globally? How sensitive are the results to this choice? A small $\gamma$ could result in $|S| \approx |U|$ (no compression), while a large $\gamma$ would lead to high compression but potentially invalid soundess guarantees. This practical aspect is critical to the method's usability but is not discussed.

- The authors are honest about the results, but they are not a strong endorsement. The method only clearly outperforms the PMI baseline in "hard compression" regimes. In "soft compression" (Tables 1 & 2), the query-dependent PMI baseline is significantly better. The authors' defense (HotpotQA has query-dependent distractors but low redundancy) is plausible, but it also suggests that the practical benefits of this query-agnostic method may be limited to specific dataset types (high redundancy) or specific, "hard" tasks. This trade-off between offline computation and online performance needs to be explored more thoroughly.

- Disconnect of DIG-R (Reranking): Algorithm 2 (DIG-R) feels somewhat disconnected from the core $\gamma$-covering framework. It uses the raw DI weights ($\hat{w}_{i\rightarrow j}$) in a PageRank-style diffusion, not the $\gamma$-cover sets or the greedy algorithm (Algorithm 1). It's a graph-based reranker, but it doesn't seem to leverage the key theoretical contributions (set cover, soundness guarantees). The ablation in Table 6, which shows that a simple "$\gamma$-order" (presumably from Algorithm 1?) is not statistically different from DIG-R, further complicates this. The paper would be stronger if it either justified this connection better or focused the empirical section on applications that directly use the $\gamma$-cover set $S$.

### Questions
- On the Magnitude of Estimation Errors: Following Theorem 3.5, the soundness bound $I(q;U) \le I(q;S) + |U \setminus S|(\gamma + 3\bar{\delta})$ seems highly dependent on the number of compressed items. Could you provide an empirical analysis of the estimation slack $\bar{\delta}$ on a dataset like HotpotQA? How large is this term in practice, and at what corpus size $|U|$ would this error term dominate $\gamma$, potentially weakening the guarantee?

- On Selecting $\gamma$: How was the tolerance $\gamma$ selected for the experiments in Tables 1 and 2? Was it tuned on a validation set? Could you provide a sensitivity analysis showing how the cover size $|S|$ and the downstream EM/F1 scores vary with different choices of $\gamma$?

- On Scalability: The offline $O(M^2)$ computation to build the DI graph is a bottleneck for very large $M$. Have you considered or explored methods to approximate the $\gamma$-cover without computing all pairwise interactions? For example, could techniques from LSH or other similarity search methods be used to find a candidate set of $(i, j)$ pairs for which $DI_{i\rightarrow j}$ is likely to be high?

- On the MI vs. DI Ablation (Table 4): The finding that DI outperforms MI is very interesting. The MI criterion $I(i;j) \ge H(C_j) - \gamma$ is a stronger condition than the DI one (since $I \ge DI$). Intuitively, one might guess that a stronger condition would select "better" or more robust representatives. Why does it perform worse? My hypothesis is that it inefficiently "over-covers" (e.g., it might prefer a symmetric pair $(i, j)$ where $DI_{i\rightarrow j}$ and $DI_{j\rightarrow i}$ are both high, when a single asymmetric chunk $k$ with high $DI_{k\rightarrow i}$ and $DI_{k\rightarrow j}$ would have been a more efficient cover). Does this match your intuition?

### Soundness
4

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
3

### Summary
The authors formulate context/prompt selection as a γ-cover problem and propose Directed Information (DI) γ-covering, an information-theoretic framework for query-agnostic redundancy-aware “context diet” framework for LLMs. Unlike existing methods based on Pointwise Mutual Information (PMI), DI captures causal and asymmetric dependencies between text chunks, enabling self-organizing selection and compression of contextual information for LLMs. In this framework, DI is approximated by the negative log-likelihood of the LLM, and one chunk is said to γ-covers another if it can predict the other chunk within a margin of γ bits of uncertainty. The authors provide theoretical guarantees for γ-covering and empirically argue its effectiveness on the HotpotQA dataset, showing consistent improvements over PMI in context compression and single-slot prompt selection, and over vanilla BM25 in reranking tasks.

### Strengths
S1. The authors use an information-theoretic perspective to tackle the important problem of context and prompt engineering for LLMs.

S2. The authors formulate context/prompt selection as a γ-cover problem and propose Directed Information (DI) γ-Covering.

S3. The authors provide theoretical guarantees to support the effectiveness of the proposed DI γ-Covering.

### Weaknesses
W1. The authors should clarify the suitability of HotpotQA as the evaluation benchmark. Specifically, it is unclear whether HotpotQA is an appropriate dataset for evaluating the effectiveness of the query-agnostic DI γ-Covering framework. HotpotQA consists of two gold supporting paragraphs and eight distractors per query, making it inherently query-dependent. Since DI γ-Covering operates without query information, it is not evident how increasing DI among paragraphs can ensure that the two relevant supporting paragraphs are selected. The paper would benefit from either a clearer justification for using HotpotQA or additional experiments on datasets where query-agnostic selection is more natural.

W2. The experimental setup lacks clarity.

W2.1 In Section 4, the definition of the compression size is confusing. The authors define s as the number of gold supporting facts and d as the number of distractors. However, given that s=2 and d=8 in HotpotQA, some of the reported compression sizes s-2 become zero. This makes the reported settings (i.e., s+d-1, s+d-2, s, s-1, s-2) unintuitive and difficult to interpret.

W2.2 The authors should clarify which LLM was used in the context compression and system prompt selection experiments.

W3. The experiments should be more comprehensive.

W3.1 The authors should include a direct evaluation of the selection quality by reporting the recall of gold supporting facts (i.e., the percentage of annotated gold paragraphs retained after applying DI γ-covering). This would provide a clearer measure of how well the method preserves relevant evidence, complementing the end-to-end EM/F1 scores that reflect final answer quality.

W3.2 In Table 2, the authors should report EM and F1 scores when the LLM is given only the gold supporting paragraphs. This upper-bound accuracy comparison would contextualize the performance of γ-covering and help quantify its effectiveness relative to the ideal scenario.

### Questions
Please refer to W1, W2, and W3.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the problem of context selection and compression in large-scale language models. It proposes a Directed Information (DI) γ-covering approach, which efficiently selects, compresses, and diversifies context without losing essential information. The method introduces Directed Information to capture asymmetric predictive relationships between context chunks, forming a query-agnostic, self-organizing framework. Additionally, the authors design a greedy algorithm and provide theoretical guarantees on information retention, diversity, and approximation. Experimental results show that DI γ-covering outperforms traditional baseline methods in tasks like context compression and system prompt selection. This paper provides an effective solution for context engineering problems in large language models. The paper is logically sound, but there are some limitations in the experimental section.

### Strengths
1.	This paper offers a novel perspective by introducing Directed Information (DI) into context engineering tasks. The γ-covering method proposed provides a query-agnostic, self-organizing framework that offers a fresh approach to the retrieval-augmented generation (RAG) paradigm, which has traditionally relied on query-dependent reordering.

2.	The paper is theoretically robust, modeling asymmetric predictive relationships between context chunks via Directed Information (DI). It presents solid theoretical foundations and proofs, including guarantees on approximation and method soundness, making the entire framework reliable.

3.	The paper introduces a query-agnostic context selection method with DI γ-covering, reducing the need for expensive online computations. This significantly enhances computational efficiency, making the method scalable for large datasets and practical applications.

### Weaknesses
1.	The experiments are mainly focused on the HotpotQA dataset, which may not fully showcase the method's performance across other complex or diverse datasets. There is a lack of experimentation on multiple datasets (e.g., multi-document QA, summarization).Insufficient baselines. Comparisons focus mainly on PMI for ranking and BM25 for reranking, while several recent competitive methods are missing, which diminishes the strength of the empirical claims.

2.	The comparison of baseline methods is limited to PMI and BM25, with no comparison to more recent methods.

3.	It does not show visualizations comparing which context chunks are selected, merged, or removed by DI γ-covering versus PMI and other baselines.

### Questions
1.	Is the performance of this method sensitive to the choice of γ? How should γ be set or adjusted?

2.	Can the paper provide a visualization demonstrating the context chunks selected or removed by γ-covering compared to PMI and other baselines?

3.	The paper mentions that the method performs worse than PMI in soft compression. Could the authors provide specific examples where DI γ-covering made suboptimal choices in such scenarios and analyze why it happened?

### Soundness
3

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
This paper introduces Directed Information covering, a query-agnostic framework for context engineering. It leverages DI, an asymmetric measure of predictiveness between context chunks, to identify and remove redundancy. By formulating context selection as a gamma-cover problem, the authors propose a greedy algorithm that computes a "self-organizing" context structure offline, incurring no online cost. This method comes with provable guarantees for information preservation (soundness) and non-redundancy (diversity). Experiments on HotpotQA demonstrate that it significantly outperforms baselines in "hard-decision regimes," such as extreme context compression or single-prompt selection.

### Strengths
### Reduced Query Cost
The framework's primary strength is its novel approach to reducing online query costs by shifting computation offline. It pre-processes context by calculating the degree of predictive information between chunks, which minimizes search-time latency.

### Theoretical Soundness
The method is built on a strong theoretical foundation, offering provable guarantees for its performance, including information preservation and diversity

### Weaknesses
### Overhead in Dynamic Contexts
A major concern is the high computational overhead when new chunks are added or deleted. Adding even one new chunk would necessitate a costly $O(M)$ recalculation to update its DI relationships with all other chunks. Since retriever contexts in RAG systems often require frequent updates for information freshness, the framework's reliance on offline computation would necessitate costly recalculations of the DI relationships.


### Limited and Narrow Evaluation
The experimental validation is insufficient and relies almost exclusively on the HotpotQA dataset. The authors explicitly state that their sole evaluation dataset, HotpotQA, contains "relatively little redundancy". I think the framework was not tested in the "redundancy-rich settings" . To demonstrate that this method is generalizable and efficient, it must be validated across a more diverse range of QA and retrieval datasets.

### Questions
### Unaddressed Practicality of DI Estimation
The entire $O(M^2 T)$ offline computation depends on using a specific language model as a measurement tool. I think it's better to analyze how the choice of this model (e.g., its size or quality) impacts the resulting DI graph and final performance.

### Soundness
3

### Presentation
3

### Contribution
3
