# Mechanistic Interpretability of LLMs through Network Science

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Understanding scaling laws and emergent abilities in Large Language Models
(LLMs) remains a key challenge for interpretability. While much prior work in
mechanistic interpretability has focused on learned representations, the attention
matrix—which governs information flow—has received much less attention. Fur-
thermore, the analysis of the attention matrix from a theoretical network science
perspective has also not been done. In this work, we present a pipeline for dynamic
graph construction from attention matrices, introduce a novel head aggregation
technique based on entropy, and analyse the attention graphs from a network sci-
ence perspective to draw interpretability insights. Our experiments show that the
entropy-based head aggregation preserves attention details, and that key graph
metrics—specifically the clustering coefficient and maximum pagerank—correlate
with improved model correctness and emergent abilities in LLMs. Notably, our
findings indicate that larger models exhibit higher maximum pagerank and lower
clustering coefficients, suggesting they reason differently by attending more glob-
ally and selectively focusing on key hotspots.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This authors propose a graph-theoretic framework for mechanistic interpretability. The authors introduce Autoregressive Attention Graphs (AAGs) - dynamic directed graphs constructed from attention matrices across heads, layers, and autoregressive timesteps. They also propose a novel entropy-based head aggregation method designed to preserve high-frequency attention details compared to mean aggregation. Using graph-theoretic metrics (clustering coefficient, pagerank, node degree), they analyze LLMs of various scales and report correlations between these metrics and task correctness on selected BigBench prompts. The key finding is that larger models exhibit lower clustering coefficients and higher maximum pagerank, suggesting a shift toward more globally distributed attention.

### Strengths
- The authors introduce an original perspective by modeling LLM attention as dynamic autoregressive graphs, bridging mechanistic interpretability and network science.
- The proposed entropy-based head aggregation method provides a tunable mechanism to emphasize either concentrated or diffuse attention, offering a new way to study head diversity.
- The experiments cover several LLM families and scales (Llama, Mistral, Qwen, Gemma), showing the framework’s general applicability across architectures.
- Figures effectively illustrate the overall pipeline, aggregation process, and performance trends, aiding comprehension even for non-experts in graph theory.

### Weaknesses
- The entropy-based aggregation lacks theoretical grounding; the paper does not explain why selecting a single head per token captures meaningful structure.
- "Emergent abilities" is equated with improvements in accuracy, whereas emergence more specifically refers to capabilities of LLMs that appear suddenly and unpredictably as model size scale up [1]. A steady increase in accuracy as model size increases is not a convincing proxy of emergent abilities.
- The absence of explicit formulas for graph metrics or references makes it difficult for readers from outside network science to understand these metrics.
- The experiments are based on three chosen graph metrics and three prompts, but the criteria or range from which these were selected are not described.
- The work demonstrates interesting correlations but stops short of showing how these findings advance interpretability or guide practical use.

[1] Wei, Jason, et al. "Emergent abilities of large language models." arXiv preprint arXiv:2206.07682 (2022).

### Questions
- What is the intuition behind "high-frequency details" in attention map? Which features or interactions do they represent?
- Why selecting a single head per token captures meaningful structure of the attention map?
- How do you decide which $\alpha$ value to use for aggregating heads?
- Could you elaborate on what the attention sink phenomena is and why it only affects the first token?
- In line 311, what is the set of different graph features that you considered and what analysis have you done on them?
- In line 474, which three prompts were used for analysis?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors present an approach to understanding LLM scaling through network science analysis of attention patterns. Applying graph metrics to attention dynamics is a reasonable angle—the pipeline for constructing dynamic graphs from attention matrices is clearly articulated, and the visualization of how graph properties evolve during autoregressive generation provides an intuitive framework. However, this work suffers from two fundamental problems: it fails to situate itself within the substantial existing literature on attention analysis, and it presents a collection of correlations without establishing meaningful mechanistic insights. The paper reads as an exploratory analysis that found some patterns and then retrofitted a story about emergent abilities, rather than a principled investigation of how attention structure drives model capabilities.

A major issue is the paper's shallow engagement with prior work. The authors frame their contribution as novel while missing extensive relevant literature. There's no discussion of induction heads and their role in driving in-context learning at scale (e.g., Olsson et al., 2022), despite this being directly relevant to understanding how attention patterns change with model size. The removal of the first token citing "attention sink phenomena" is discussed perfunctorily, given the rich work on how attention sinks emerge and concentrate information flow. There is also existing work on attention entropy and its relationship to model behavior (eg https://arxiv.org/abs/2303.06296).

Most fundamentally, it's unclear what the actionable takeaway is. That larger models have lower clustering coefficients and higher maximum PageRank values in their attention graphs (after a particular processing pipeline). So what? Can we use this to improve training? Predict failure modes? The paper offers no path from observations to actionable insights.

Specific comments:
- Selection bias: This is a major concern. Analyzing prompts specifically chosen where small models fail and large models succeed, then claiming that the observed correlations are why the large models did better, is cherry-picking. Do the same patterns hold on prompts where all models do well, or where the large models underperform? 
- Statistical rigor: With 19 prompts total (but detailed analysis on just 3) and cherry-picked metrics, the multiple hypothesis problem is severe.
- Missing ablations: The methodological choices (alpha=0.5, choice of beta, top-k vs threshold) are fragile and under-justified.
- Graph metrics: The chosen metrics (clustering, PageRank, degree) are all correlated with sparsity. Where are the partial correlations controlling for graph density? Where are metrics less coupled to basic graph structure?
- Qwen: The Qwen model family doesn't follow the proposed patterns, yet the authors speculate--without evidence--that Chinese pretraining data explains this. This kind of post-hoc rationalization undermines the paper.

### Strengths
see above

### Weaknesses
see above

### Questions
see above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a method to interpret LLMs with autoregressive attention graphs. It aggregates the attention matrix across layers and heads to construct a graph and then analyzes several graph metrics as the model generates more tokens.

### Strengths
The paper writing is clear and easy to follow. The pipeline is described concretely with a figure.

### Weaknesses
- The motivation for the interpretability method is not clear to me. The paper do not justify why the entropy quantile \alpha should pick the 'most informative' head row-wise, nor why the choice should meaningfully reflect the mechanism rather than noise or scaling artefacts. Likewise, the rollout \beta, top-k vs threshold, threshold values, and row-normalization are all consequential choices with no principled argument or ablation demonstrating robustness. These choices can strongly alter degree, clustering, and centrality.

- The central claim is essentially correlational: clustering and max-PageRank associate with correctness/emergence. There is no causal test (e.g., edge/attention patching or targeted ablations) showing that the proposed graph structures drive performance or reflect real circuits, nor any link from these graph metrics to known mechanistic phenomena.

- The paper 'hand-picks' three graph features without clear justification for those. They presents graph metrics as interpretability insights without validating against ground-truth circuits or behavioral interventions. The leap from 'entropy preserves high frequency' to 'retains meaningful head information' is not clear.

### Questions
- Please provide comprehensive sensitivity analyses over alpha, beta, top-k, threshold
- Why entropy quantiles, row-wise? What is the theoretical rationale that low/median/high-entropy rows correspond to mechanistically meaningful heads for a given token, as opposed to amplifying spurious attention? Can you compare to simple alternatives (e.g., per-row max head, soft ensemble, learned weights) on the same analysis?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an interpretability framework for large language model (LLM) mechanisms based on network science. By constructing the Attention matrix in the Autoregressive generation process as a dynamic graph (autoregressive attention graph), and combining graph theory indicators to analyze the correlation between model size and emergent ability. The core idea is that the attention matrix can be regarded as a dynamic directed graph, where nodes represent tokens and edge weights represent the intensity of attention. High-frequency details are retained through the entropy-based attention head aggregation method, and graph metrics such as clustering coefficients and maximum PageRank are calculated to reveal the relationship between the model's reasoning mechanism and performance.

### Strengths
S1:The dynamic graph modeling framework of "Autoregressive attention map" is proposed, which formalizes the attention flow of Transformer into a time-evolved graph structure and is conceptually innovative.
S2：An entropy-based attention head aggregation method is proposed, which can better preserve high-frequency details compared with average aggregation.
S3:The experimental design was systematized, covering multiple model families and multiple scales , which enhanced the universality of the conclusions.

### Weaknesses
W1:Only three prompt words were used for analysis, and the sample size was seriously insufficient, making it difficult to support the conclusion.
W2:Insufficient explanation of the abnormal trend of the Qwen ,It is simply attributed to "differences in pre-training data", lacking further data or experimental support.

### Questions
Are there any plans to verify the correlation between image metrics and correctness on a larger-scale prompt set? Have you considered using automated evaluation methods to handle complex outputs?

### Soundness
2

### Presentation
3

### Contribution
2
