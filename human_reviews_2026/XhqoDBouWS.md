# Why Attention Patterns Exist: A Unifying Temporal Perspective Analysis

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
Attention patterns play a crucial role in both training and inference of large language models (LLMs). Prior works have identified individual patterns such as retrieval heads, sink heads, and diagonal traces, yet these observations remain fragmented and lack a unifying explanation. To bridge this gap, we introduce **Temporal Attention Pattern Predictability Analysis (TAPPA), a unifying framework that explains diverse attention patterns by analyzing their underlying mathematical formulations** from a temporally continuous perspective.
TAPPA both deepens the understanding of attention behavior and guides inference acceleration approaches. Specifically, TAPPA characterizes attention patterns as predictable patterns with clear regularities and unpredictable patterns that appear effectively random. Our analysis further reveals that this distinction can be explained by the degree of query self-similarity along the temporal dimension.
Focusing on the predictable patterns, we further provide a detailed mathematical analysis of three representative cases through the joint effect of queries, keys, and Rotary Positional Embeddings (RoPE). We validate TAPPA by applying its insights to KV cache compression and LLM pruning tasks. Across these tasks, a simple metric motivated by TAPPA consistently improves performance over baseline methods. The code is available at [https://github.com/MIRALab-USTC/LLM-TAPPA](https://github.com/MIRALab-USTC/LLM-TAPPA).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a unifying temporal perspective to explain the emergence of diverse attention patterns in LLMs. The authors categorize patterns into predictable (re-access, sequential, seasonal) and unpredictable (retrieval-like) types, attributing the distinction to variations in query self-similarity over time. They provide theoretical analyses linking pattern formation to query-key continuity and RoPE, and validate their framework through applications in KV cache compression and LLM pruning. Experiments on models like Llama and Qwen show consistent improvements over baseline methods.

### Strengths
- The paper is well-organized, with clear motivations, methodical explanations, and thorough appendices.
- The paper offers a coherent perspective that integrates previously fragmented observations (e.g., sink heads, diagonal patterns, retrieval heads) under a single temporal continuity lens.
- The paper provides detailed mathematical proofs for each pattern type, clearly linking RoPE mechanics and query-key dynamics to observable attention structures.
- The proposed q-similarity metric is effectively applied to downstream tasks (KV cache compression, pruning), demonstrating improved performance over strong baselines.

### Weaknesses
- The entire framework hinges on the assumption of temporal continuity in queries and keys. While this is likely a reasonable assumption for many layers and tasks, its universality is not thoroughly explored. The analysis might be less applicable in layers or for inputs where representations change abruptly. A discussion of the boundaries of this assumption would strengthen the work.
- The q-similarity metric is central to the applications, but its specific formulation (e.g., the choice of cosine similarity) lacks a comprehensive ablation study. It remains unclear how sensitive the performance gains are to these choices, or if an even more effective metric derived from the same theory could be designed.
- The proofs provided in the appendix, while a valuable effort, contain significant weaknesses that undermine their theoretical rigor.

### Questions
1.How does q-similarity vary across different layers and heads? Is it consistent across models, or does it require per-model calibration?
2.The paper claims that high q-similarity implies redundancy in pruning. Is this always true? Could some stable patterns be critical for certain tasks (e.g., syntax parsing)?
3.In the proof of vertical stability (Theorem 5.1, Appendix B), a crucial step bounds the change in the angle between the query and a fixed key:
$$|\phi_{t+1,i}^{(m)}-\phi_{t,i}^{(m)}|\leq\frac{\|q_{t+1}^{(m)}-q_t^{(m)}\|}{r_m}$$
However, consider a simplified scenario in 2D: let q_{t}=(1,0) and q_{t+1}=(cos(2arcsin(ε/2)), sin(2arcsin(ε/2))). Here ||q_{t+1}-q_{t}|| ≤ ε， the angle change is 2arcsin(ε/2) ＞ ε. This suggests the above inequality does not hold. 

In the proof of Theorem 5.4 (Seasonal Pattern), the derivation for non-dominant channels contains a critical error. The term (i-t)θ_m is incorrectly repeated in both cosine functions when calculating |a_{t+L,i}^{(m)} - a_{t,i}^{(m)}|. The correct expression for a_{t+L,i}^{(m)} should have a phase of (i-t)θ_m - Lθ_m. More critically, the standard inequality used to bound the difference for non-dominant channels is misapplied. This inequality, | ||u|| ||v|| cos φ - ||u'|| ||v'|| cos φ' | ≤ ..., is valid only when φ and φ' are the geometric angles between the vector pairs (u,v) and (u',v'), respectively. In your proof, you apply it with angles φ = φ_{t,i}^{(m)} + (i-t)θ_m and φ' = φ_{t+L,i}^{(m)} + (i-t)θ_m. However, these are not pure geometric angles but include an additive positional phase. This misapplication renders the subsequent bound on non-dominant channels invalid.

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
3

### Summary
This paper analyzes sparse attention patterns from the perspective of query similarity. It derives theoretical relations between the similarity of consecutive query vectors and the corresponding changes in attention values. The paper further shows how certain query distributions lead to typical attention patterns. It shows potential applications of these insights in attention budget allocation and layer pruning.

### Strengths
1. The paper provides a new perspective for explaining existing attention patterns from the query point of view.

2. The paper demonstrates how the proposed query-level observations can inform the design of sparse attention and layer pruning, adding practical value to the theoretical analysis.

### Weaknesses
1. The claim of analyzing the *joint effect of input dynamics and positional encoding* seems overstated. While it would be valuable to disentangle and quantify their respective contributions, the paper instead merges them into the query with post-encoding. This makes the connection to the original input less clear than the abstract and introduction suggest.

2. Several assumptions used in the derivations are not carefully validated, which raises concerns about the reliability of the conclusions. For instance, the assumption of a dominant channel weight in Theorem 5.1 requires empirical support.

3. The empirical section lacks comparisons with more direct and recent baselines, such as DuoAttention [1], which explicitly distinguishes retrieval heads.

4. Some key concepts (e.g., continuity, predictability) are introduced without sufficient explanation in the introduction. Brief definitions would help prevent confusion.

[1] Xiao, Guangxuan, et al. "Duoattention: Efficient long-context llm inference with retrieval and streaming heads." arXiv preprint arXiv:2410.10819 (2024).

### Questions
1. Is the query similarity computed after applying RoPE?

2. Could you provide more justification for using attention patterns to guide layer-wise FFN pruning in Section 6.2? In particular, why does high query similarity (stability) imply that “the layer extracts less novel information”

### Soundness
2

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
4

### Summary
This paper proposes a framework to explain diverse attention patterns in LLMs through temporal continuity analysis. The authors categorize patterns as predictable (re-access, sequential, seasonal) or unpredictable, attributing the distinction to query self-similarity. They provide mathematical analysis of how query/key continuity and RoPE jointly produce these patterns, and validate their framework through KV cache compression and (fewer) LLM pruning experiments. While the paper makes a reasonable attempt to unify attention pattern analysis, the contributions are incremental over existing work (particularly AttentionPredictor). The theoretical analysis, though rigorous, doesn't yield sufficiently novel insights—the sequential pattern analysis overstates its departure from prior work, and the seasonal pattern analysis lacks empirical grounding. The downstream experiments show only marginal improvements.

### Strengths
- The temporal continuity perspective provides a systematic way to understand previously fragmented observations about attention patterns. The decomposition view connecting query similarity to pattern stability is intuitive.
- Rigorous mathematical treatment: the theorems provide formal proofs for the emergence of different pattern types, with explicit bounds relating pattern stability to query/key properties and RoPE parameters.
- Novel insight on periodic sequential patterns: The analysis of diagonal spacing and experimental validation by manipulating dominant channel locations is particularly interesting.
- The evaluation on KV Cache compression is performed with different budgets.

### Weaknesses
- Limited novelty: the observation that query continuity drives attention stability was already made by AttentionPredictor (and the authors acknowledge that). While this paper provides mathematical formalization, the fundamental insight is not new. 
- KV cache compression: the improvements over CAKE are marginal and seem to be within noise margins. Other state of art methods such as DuoAttention, Expected Attention could be a stronger baseline.
- LLM pruning is only compared against a single baseline. There are no other comparisons to other structured pruning methods. If the authors think this makes sense, could they explain why they considered only this one baseline.
- The hyperparameters  appear hand-tuned without ablation studies.

### Questions
- What is the computational overhead of computing q-similarity scores during inference?
- What is the precise impact of this KV cache compression method on memory footprint ? And on latency ? 
- How are the hyper-parameters selected ?

### Soundness
2

### Presentation
3

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
This paper proposes a framework for analysing attention patterns in Transformer-based models; author identify three types of patterns (namely re-access, sequential, and seasonal) from a subset of the heads (referred to as "unpredictable") based on query self-similarity and positional embeddings. Based on the proposed framework, authors then propose a method for efficiently allocate KV cache budgets and structured layer pruning, improving over baselines like CAKE and ShortGPT.

### Strengths
- The analysis in e.g. Proposition 4.1 that links attention stability to query self-similarity, and how query drift induces changes in the logit changes, is novel and interesting; likewise for Th. 5.2 and 5.3, which provide conditions under which sequential/periodic diagonals appear
- Using q-similarity in CAKE and ShortGPT yields sigificant improvements in several settings

### Weaknesses
- Improvements in CAKE (Tab. 1) seem very marginal, are they statistically significant? Averages are not clearly reported
- Computing q-similarities for every layer/head seems computationally expensive, but runtimes/costs are not discussed in-depth

### Questions
- Can you please expand more on the runtime/costs of computing q-similarities and what kind of overheads they add to methods like CAKE?
- Are there cases where q-similarity fails to misidentify retrieval heads? What does it happen in those scenarios?

### Soundness
2

### Presentation
2

### Contribution
2
