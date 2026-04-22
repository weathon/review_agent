# FASA: FREQUENCY-AWARE SPARSE ATTENTION

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
The deployment of Large Language Models (LLMs) faces a critical bottleneck when handling lengthy inputs: the prohibitive memory footprint of the Key Value (KV) cache. To address this bottleneck, the token pruning paradigm leverages attention sparsity to selectively retain a small, critical subset of tokens. However, existing approaches fall short, with static methods risking irreversible information loss and dynamic strategies employing heuristics that insufficiently capture the query-dependent nature of token importance.
We propose FASA, a novel framework that achieves query-aware token eviction by dynamically predicting token importance.
FASA stems from a novel insight into RoPE: the discovery of functional sparsity at the frequency-chunk (FC) level. Our key finding is that a small, identifiable subset of "dominant" FCs consistently exhibits high contextual agreement with the full attention head. This provides a robust and computationally free proxy for identifying salient tokens.
Building on this insight, FASA first identifies a critical set of tokens using dominant FCs, and then performs focused attention computation solely on this pruned subset.
Across a spectrum of long-context tasks, from sequence modeling to complex CoT reasoning, FASA consistently outperforms all token-eviction baselines and achieves near-oracle accuracy, demonstrating remarkable robustness even under constraint budgets. Notably, on LongBench-V1, FASA reaches nearly 100\% of full-KV performance when only keeping 256 tokens, and  achieves 2.56$\times$ speedup using just 18.9\% of the cache on AIME24.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes FASA (Frequency-Aware Sparse Attention), a training-free, query-aware framework for token eviction by dynamically predicting token importance in LLMs. The authors identify functional sparsity in frequency chunks (FCs) within the RoPE positional encoding, observing that only a small subset (“dominant FCs”) contributes significantly to contextual awareness. Leveraging this insight, FASA consists of two stages:
1. Token Importance Prediction (TIP): Uses dominant FCs to predict salient tokens.
2. Focused Attention Computation (FAC): Performs full attention only on these tokens.
Two variants are proposed—FASA-M (memory optimized) and FASA-C (computation optimized)—achieving up to 2.56× speedup while maintaining near full-KV accuracy across LongBench-V1, MATH500, and AIME24.

### Strengths
1. The discovery of functional sparsity at the frequency-chunk level in RoPE is both novel and conceptually insightful. The paper is clearly written and well-structured.
2. FASA’s training-free, model-agnostic, and task-independent design enables straightforward integration with existing LLMs, greatly enhancing its practical deployability.
3. The method demonstrates strong performance

### Weaknesses
1. Although FASA reduces computation by selective token attention, it still retains the full KV cache, meaning memory consumption remains a potential bottleneck for extremely long contexts.

### Questions
1. In Figure 10, it would strengthen the universality claim to evaluate across a broader range of popular models.
2. I would appreciate a more intuitive explanation of dominant frequency chunks (FCs). The Contextual Agreement (CA) score measures alignment in token ranking between the full-head and single-FC attention scores. A high CA score indicates that the FC captures a similar ranking of token importance. However, since CA reflects order rather than magnitude, it is possible that some non-dominant FCs might exhibit smaller agreement but larger attention scale. Showing the distribution of CA scores for selected dominant FCs could help demonstrate that they accurately capture both ranking and influence on token importance.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new method called FASA to achieve query-aware token eviction by dynamically predicting token importance. The authors state that they found only a small subset of frequency chunks (FCs) contributes significantly to the model's contextual awareness, thus allowing FASA to predict and use a critical set of tokens for focused attention computation. The authors conducted several experiments and concluded that FASA performs better than other attention sparsity algorithms.

### Strengths
1. Their insights about RoPE are interesting.
2. According to their experiments, FASA appears to be effective.

### Weaknesses
- The authors mention that their insights about frequency chunks relate to RoPE and cite previous works. However, since this aspect is critical to FASA, I believe they should include their theoretical analysis directly in the paper. It is difficult for readers to be convinced by the current content alone.
- Regarding the experiments conducted to support their insights about sparsity in frequency chunks, I think more demonstration is needed. The figure in the main text simply compares two heatmaps of FCs between different models and layers, which raises concerns about selective presentation. Although Figure 11 shows comparisons of different heatmaps across layers of the same model, and considering FASA's effectiveness, I am inclined to believe there might be solid evidence. However, this evidence must be clearly presented in future versions.
- The authors claim to have empirically proven that the selection of FCs is task-agnostic and insensitive to calibration data, which is important for the method's practicality. However, the results are only briefly explained in Table 6, which is insufficient to convince readers.
- Regarding the experiments on FASA's effectiveness, a potential issue lies in the choice of dominant FCs. In the Appendix, the authors explain that they used different FCs for LongBench experiments and Long-CoT reasoning. This seems contradictory to their claim about FC selection.

- The font size in the figures is too small and difficult to read. It should at least match the main text's font size.
- Some important experimental details are included in the Appendix, which can confuse readers and require frequent cross-referencing.

### Questions
See Weakness.

### Soundness
1

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
This paper proposes a novel, training-free, query-aware token elimination framework called FASA. Its core contribution stems from a novel discovery of RoPE: functional sparsity exists at the frequency-chunk (FC) level. Based on this insight, FASA employs a two-stage framework:
1. Token Importance Predictor (TIP): Utilizes a one-time offline-calibrated set of dominant FCs to efficiently estimate attention scores (aggregating only contributions from these FCs) and identify a critical subset of tokens.
2. Focused Attention Computation (FAC): Performs full-dimensional, accurate attention computation only on the significant subset of tokens selected in the TIP stage.

Furthermore, this paper provides two hardware-aware implementation variants: FASA-M (memory-optimized, offloading non-dominant K and V caches to the CPU) and FASA-C (computationally optimized, performing sparse access on the GPU). 

On LongBench-V1,FASA reaches nearly 100% of full-KV performance when only keeping 256 tokens, and achieves 2.56× speedup using just 18.9% of the cache on AIME24.

### Strengths
1. The discovery of "functional sparsity at the frequency block level" in RoPE provides a new, theoretically grounded perspective for understanding attention mechanisms and designing sparse attention models, rather than relying solely on heuristics.
2. The core hypotheses were sufficiently verified; in addition to verifying the sparsity of the dominant FCs, the universality and task-invariance of the CA index were also verified.
3. Strong practicality: The paper considers different hardware constraints and proposes two variants, FASA-M (memory-constrained) and FASA-C (computation-constrained), which significantly enhances the practical application value of the method.

### Weaknesses
1. Strong dependence on RoPE: The entire methodology and core findings are built upon the analysis of RoPE. This severely limits the method's generalizability. It remains unclear whether or how FASA can be generalized to models using other positional encodings or even those without explicit positional encoding.
2. Table 6 shows the robustness to the data. Only the TREC and MATH datasets are provided here, which is relatively limited in variety. Furthermore, the data size is not analyzed.

### Questions
1. Regarding generalization beyond RoPE: Have you considered how to extend this insight into "functional sparsity" to models using other positional encoding schemes, or partially rope models like MLA?
2. Why was N_{tip}=16 chosen as the primary configuration in the paper? Figure 6 seems to suggest that a smaller N_{tip} is also good enough in many cases.
3. Regarding the efficiency analysis of FASA-M, does the comparison in Figure 7 already include prefetching operations? Could you also provide a comparison of the benefits of prefetching techniques?

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
The authors propose to use functional sparsity in RoPE at frequency-chunk level, basically a small per-head subset of FCs that act as a ‘filter’ for the head on what matters most. They use their importance predictor to score tokens, and run full attn on only selected positions. With this they are near full-KV accuracy on longbench with just 256 tokens.

### Strengths
- novel idea (and observation) to use functional sparsity of RoPE 
- speedup demonstrated at long context, and can work with other KV compression schemes as well
- robust across datasets
- because they do not re-index token positions, original absolute positions of tokens are preserved

### Weaknesses
- Not applicable to non-RoPE variants, further discussion on that would help.
- The idea is impressive, but it took quite some effort to understand. I think a huge amount of math can be simplified in the paper, maybe shifted to appendix for ‘more details’.

### Questions
Comments on non-RoPE schemes would be appreciated. 
Page-based methods cannot use this efficiently, perhaps an ablation on page level sparsity could be useful to the reader, but not critical.

### Soundness
4

### Presentation
2

### Contribution
4
