# ProtoKV: Long-context Knowledges Are Already Well-Organized Before Your Query

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Modern Large Language Models (LLMs) face fundamental challenges in processing long text sequences due to the quadratic complexity of attention mechanisms. Key-Value (KV) cache retention strategies mitigate this issue by selectively preserving salient KV pairs for autoregressive generation. However, existing methods fail to adequately and efficiently preserve the semantic integrity of the compressed representations. In this paper, we discover a prevalent phenomenon in LLM: within the key embedding space, while most tokens exhibit similarity with their contextual neighbors (we term position-determined tokens), a small subset of special tokens serving as semantic anchors consistently show local deviation property and form one or several clusters (we term semantic-anchored tokens). Motivated by this observation, we propose ProtoKV that separately processes these two token categories for KV cache compression. Within this framework, we first construct semantic prototypes based on the inherent characteristics of the two token categories, and subsequently form clusters of semantically similar tokens as basic compression units. This approach preserves semantic integrity with high computational efficiency. Experiments on LongBench demonstrate that ProtoKV achieves 2.11% higher accuracy than state-of-the-art methods under matched memory constraints.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces ProtoKV, a KV-cache compression method for LLMs. It first exploits a key structural insight: most tokens (“position-determined”) cluster with their neighbors, while a small set of “semantic-anchored” tokens (SAT) consistently deviate and form clusters. By splitting tokens into these two classes, ProtoKV builds separate semantic prototypes and compresses each class into semantically coherent clusters, preserving meaning while cutting memory. On LongBench it outperforms prior state-of-the-art techniques by 2.11\% accuracy under the same memory budget.

### Strengths
1. The paper is well written and excellently presented. Figure 1 clearly illustrates the paper's novel findings.

2. The paper finds that semantic-anchored tokens, which exhibit the local deviation property, are important for generation. It provides exhaustive experimental results to demonstrate this.

3. ProtoKV builds separate semantic prototypes and compresses each class into semantically coherent clusters. This method improves accuracy compared with existing baselines.

### Weaknesses
1. The rationale behind locality-sensitive hashing applied to SAT is unclear. (Question 1)

2. The time cost of ProtoKV may hinder its applicability. (Question 2)

3. There are no error bounds for the key results. Quantitative robustness indicators such as standard deviations would help validate generalizable conclusions.

### Questions
1. The paper claims that SATs are salient for generation, so why not select all SATs directly? The results as shown in Figure 12 illustrate that the SAT prototype number does not influence the accuracy, because ProtoKV selects all SATs, even though they are classified into different clusters. Additionally, what type of RFF-based hashing is it, locality-sensitive or random?

2. Figure 14(b) shows that the average compression time exceeds half an hour. However, full attention computation usually takes several minutes for a long context. Do the users have to wait half an hour for KV cache compression?

3. As shown in Figure 34, none of the methods match FullKV in terms of accuracy. Could ProtoKV achieves FullKV’s level of precision with a larger budget size, e.g., 1024 tokens?



Minor comments:
- Line 461, Eq. equation 8 -> Eq. 8
- The caption of Figure 34 states that bold indicates the best performance and underline the second performance, but no text in the figure is bolded or underlined.

### Soundness
3

### Presentation
4

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
The paper introduces ProtoKV, a novel framework designed to enhance the efficiency of Key-Value (KV) cache retention in large language models (LLMs) when processing long text sequences. The authors identify two categories of tokens within the key embedding space: Position-Determined Tokens (PDTs), which maintain strong similarity with their contextual neighbors, and Semantic-Anchored Tokens (SATs), which exhibit local deviation and form clusters. By leveraging the unique properties of these two token types, ProtoKV constructs semantic prototypes that improve KV cache compression while preserving semantic integrity. Experimental results demonstrate that ProtoKV outperforms existing state-of-the-art methods by achieving an average accuracy improvement of 2.11% on the LongBench benchmark, showcasing its effectiveness in maintaining high retrieval accuracy with minimal KV cache retention.

### Strengths
- Good Presentation. The presentation of this paper is very clear, the structure is reasonable, and the presentation of the figures and tables is also very precise.
- Reasonable Idea:  The paper presents a novel perspective on token categorization in LLMs, particularly the identification and utilization of SATs as semantic anchors, which is a significant contribution to the field.
- Good Performance: The authors provide comprehensive experiments across various benchmarks, clearly demonstrating the advantages of ProtoKV over existing methods in terms of accuracy and efficiency.
- Insightful Analyses.

### Weaknesses
- Complexity of Implementation: The proposed method may introduce additional complexity in the implementation of LLMs, which could be a barrier for adoption in certain applications or by practitioners with limited resources.
- Limited Comparison with Other Methods: While the paper provides lots of experiments to evaluate ProtoKV, a more extensive analysis involving **more recent SOTA approaches** could strengthen the argument for its superiority.

### Questions
Please refer to the weakness part.

### Soundness
3

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
2

### Summary
The paper proposes ProtoKV, a semantic-aware KV cache compression framework for large language models (LLMs) that mitigates long-context inference costs. It introduces two token categories—Semantic-Anchored Tokens (SATs) and Position-Determined Tokens (PDTs)—and constructs hybrid semantic prototypes for each, guiding KV retention through cluster-based attention relevance. The method preserves semantic integrity while maintaining computational efficiency, outperforming baselines such as SnapKV, H2O, and ChunkKV by up to 2.11% on LongBench and achieving 97.3% retrieval accuracy in Needle-in-a-Haystack tests.

### Strengths
•	The identification of SATs as clustering outliers in the key embedding space (Fig. 4–6) provides a new lens for understanding token semantics in LLMs. This insight grounds ProtoKV’s prototype-based compression design and distinguishes it from previous attention- or position-driven methods.
•	The framework (Sec. 4.2–4.3) integrates Random Fourier Feature hashing (Eq. 7–8) and prototype-guided selection (Eq. 10–11), avoiding costly iterative clustering (Fig. 8). Pseudocode and reproducibility details are given (Appendix J), enhancing transparency.
•	ProtoKV is compared with multiple baselines across three architectures (LLaMA-2, LLaMA-3, Mistral) and two benchmarks (LongBench, Ruler), showing robustness under varying KV budgets (64–512) and across tasks (Fig. 9, 10, 12–14). The ablation studies further isolate the roles of prototype number and SAT count.

### Weaknesses
•	While the “local deviation property” (Eq. 4–6) is empirically supported, the causal explanation (Sec. 3.2) remains qualitative. There is no analytical or statistical validation that SATs correspond to meaningful semantic units across layers or models.
•	Key hyperparameters such as neighborhood window $\kappa$, prototype number, and threshold $\beta$ are only briefly tuned (Fig. 12–13) without robustness metrics or cross-dataset variance, limiting confidence in generalizability.
•	Although computational cost is compared (Fig. 14), there is no wall-clock latency or memory breakdown versus model size (e.g., >8 B models), and no statistical significance tests for accuracy gains (Table 2–4).
•	The contribution of each stage (SAT detection, LSH clustering, observation window) is only partially evaluated; removing or modifying these modules’ effects is not explicitly quantified.

### Questions
1.	Could the authors provide layer-wise or head-wise distributions of SATs to clarify whether semantic anchoring is model-general or architecture-specific?
2.	How does ProtoKV behave under streaming or dynamic decoding, where prefilling-only assumptions may not hold?
3.	Can the authors include significance analysis or confidence intervals to verify the reported 2.11 % average improvement?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors discover that while most tokens demonstrate high similarity (in key space) with their contextual neighbors (position-determined tokens, PDTs), a subset of tokens (dubbed "semantic-anchored tokens", SATs) deviate from this property while accumulating a significant amount of attention. The authors construct lsh-based prototypes for SATs and chunk-based protoypes for PDTs as compression units. Clusters are ranked according to an importance metric, tokens are assigned to these clusters, and tokens from the top-ranked clusters are retained until the budget is met. This approach (ProtoKV) outcompetes baselines by >2% on LongBench and outcompetes other baselines at a lower budget and ties others at higher budgets.

### Strengths
- This appears to be the first work to discover SATs. The experiment distinguishing them from sinks validates this unique token type. 

- The clustering approach is principled according to the token types. 

- ProtoKV defeats several popular strategies across varying model families on LongBench. 

- The success of the method at a very small budget (64 tokens) is attractive for severely resource-constrained systems.

### Weaknesses
- LongBench and RULER are known to stack lots of noisy context around sparsely distributed signals, thus possibly rendering the appearance of SATs as unique to these types of benchmarks. The appearance of this token type does not appear to be explored over a greater variety of long-context tasks.

- Besides Llama-3-8B Instruct, only older models are tested. The authors should consider evaluating their approach on newer Qwen, Phi models, and/or the latest Mistral-7B. 

- The performance gain on RULER is quite minimal. While the average improvement on LongBench is +2%, the individual numbers on LongBench in Figure 34 are far less significant, where ProtoKV either incrementally wins or even loses against other baselines on a variety of tasks. This makes it difficult to determine whether ProtoKV is truly a worthwhile compression strategy.

### Questions
- See weaknesses. 
 - How does the method perform on RULER 16K or 32K?
 - Is H2O truly **that** bad on RULER (Table 2)? This doesn't seem to concur with other literature. 
 - How does this approach fundamentally differ from the Reformer, which also chunks and groups tokens according to LSH buckets?

### Soundness
2

### Presentation
2

### Contribution
2
