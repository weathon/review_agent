# Memory Retrieval in Transformers:  Insights from the Encoding Specificity Principle

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
While explainable artificial intelligence (XAI) for large language models (LLMs)
remains an evolving field with many unresolved questions, increasing regulatory
pressures have spurred interest in its role in ensuring transparency,
accountability, and privacy-preserving machine unlearning. Despite recent
advances in XAI have provided some insights, the specific role of attention
layers in transformer-based LLMs remains underexplored.

This study investigates the memory mechanisms instantiated by attention layers, drawing on prior research in psychology and computational psycholinguistics that links Transformer attention to cue-based retrieval in human memory.
In this view, queries encode the retrieval context, keys index candidate memory
traces, attention weights quantify cue–trace similarity, and values carry the
encoded content, jointly enabling the construction of a context representation
that precedes and facilitates memory retrieval.

Guided by the Encoding Specificity Principle, we hypothesize that the cues used in the initial stage of retrieval are instantiated as keywords. We provide converging evidence for this keywords-as-cues hypothesis.
In addition, we isolate neurons within attention layers whose activations selectively encode and facilitate the retrieval of context-defining keywords.

Consequently, these keywords can be extracted from identified neurons and further contribute to downstream applications such as unlearning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates how transformer attention layers function as memory retrieval mechanisms, grounding the analysis in the Encoding Specificity Principle (ESP) from cognitive psychology. The authors argue that attention performs cue-based retrieval analogous to human memory.

### Strengths
Conceptual: Establishes a cognitive analogy between human cue-based memory retrieval and transformer attention under the Encoding Specificity Principle.

Empirical: Demonstrates that Q, K, V implement distinct cognitive-like roles—context encoding, memory indexing, and content storage—validated through controlled swapping and perturbation experiments.

Mechanistic: Identifies specific attention-layer neurons encoding “keywords” that act as retrieval cues, offering a concrete locus for contextual memory inside LLMs.

Applied: Suggests practical applications for machine unlearning and privacy-aware data removal, by targeting or suppressing these keyword-linked neurons to erase specific memories.

### Weaknesses
This paper is conceptually interesting but offers limited substantive innovation. The proposed connection between transformer attention and the Encoding Specificity Principle is largely a loose analogy rather than a formal theoretical contribution. The authors do not provide a rigorous mathematical formulation or define concrete quantitative measures such as memory retrieval efficiency or cue–content overlap. As a result, the findings are primarily descriptive phenomena rather than statistically grounded or systematically analyzed results.

Furthermore, the paper lacks stronger visualization or causal interpretability analysis. It does not present attention-head–level retrieval trajectories or activation dynamics that could substantiate the proposed analogy. Incorporating feature visualization or attention circuit tracing would make the conclusions considerably more convincing. Overall, the work reads more as a conceptual or idea paper than a genuine mechanistic discovery.

### Questions
like weakness

### Soundness
3

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
3

### Summary
This paper investigates memory mechanisms in Transformer-based Large Language Models (LLMs), with a specific focus on the role of attention layers. The authors propose a conceptual framework based on principles from human psychology, chiefly the Encoding Specificity Principle (ESP) and cue-based retrieval theories. This framework leads to two core hypotheses. The paper presents two main experiments to empirically validate these hypotheses using several decoder-only LLMs. The authors identify specific neurons that are highly activated by these keywords and show that perturbing the K-projection for these keywords significantly impairs memory recall far more than perturbing random tokens. The paper concludes that this evidence supports the ESP framework and identifies a pathway for extracting memory-indexing keywords, which could be used for applications like machine unlearning.

### Strengths
1. The paper's primary strength is its novel conceptual bridge, connecting the well-established psychological theory of cue-based retrieval and the Encoding Specificity Principle directly to the architectural components of the Transformer. This provides an intuitive and human-centric lens for interpreting the "black-box" attention mechanism.

2. This conceptual framework is supported by a very strict and rigorous experimental design. Experiment 1 is particularly well-controlled. It carefully isolates the roles of Q, K, and V by swapping them between factual and counterfactual prompts that are constrained to have the exact same tokenized length. The intervention is also minimal, applying only to the first token generation, which cleanly tests the effect of context processing. 

3. Experiment 2 is equally rigorous. It validates the "keywords-as-cues" hypothesis not just by perturbing keywords, but by benchmarking this against a crucial control: perturbing an equivalent number of random tokens. The dramatic difference in outcomes, shown in Figure 4, strongly supports the claim that these keywords are functionally special.

### Weaknesses
1. A primary point of clarification is that the paper does not present a new formal, mathematical theory; rather, it provides an empirical validation of a conceptual mapping from psychology. Its support is based on experimental evidence, not mathematical proofs.

2. The authors themselves identify limitations in their methodology. For instance, the method for selecting top neurons and the number of keywords to target for perturbation is described as "naive" and "largely arbitrary," suggesting that the full potential of the unlearning application is not yet realized.

3. A further methodological limitation, also noted by the authors, is the inability of their keyword extraction method to handle compound words or multi-word terms as single cues. The paper notes that "White rabbit" is a better cue than "rabbit" alone, but the current method cannot group these tokens, meaning the extracted keyword list may not fully capture the ideal set of contextual cues. This could under-represent the true effect of these cues.

### Questions
1. The K-perturbation experiment successfully demonstrates that zeroing-out keywords impairs memory. How sensitive are these results to the type of perturbation? For instance, what would be the effect of replacing the keyword K-projections with random noise, or with an averaged K-vector from other keywords, instead of simply zeroing them out?

2. Figure 3 shows that for each model, a single layer-head-dimension triplet is consistently the most activated by keywords across different books. Does perturbing only this single, dominant neuron have a disproportionately large impact on memory recall? How much of the memory impairment effect is localized to this one neuron versus the other high-ranking neurons?

3. The paper provides strong evidence that the attention mechanism functions *like* a key-value memory. Does this framework offer any insights into how the model *learns* to associate specific keywords with the K-matrix (the index) during pre-training? Does this imply that the K-projections are specifically trained to act as a content-addressable index for salient tokens?

4. Finally, the experiments focus on factual recall from texts. How robust is the "keywords-as-cues" hypothesis to different types of memory? Would this same mechanism (keywords indexed by K) be expected to retrieve more abstract or thematic concepts, or is it specialized for concrete factual associations?

### Soundness
3

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
3

### Summary
This paper studies the hypothesis that the attention mechanism in Transformer implements memory-like functions analogous to those found in human cognition. The evidence for this hypothesis comes from two experiments. First, they swap the attention activations between two counterfactual prompts and observe the resulting outputs. Second, they decode and compare lists of keywords from the attention heads for different documents. Experimental results suggest that the attention mechanism performs and information retrieval role.

### Strengths
- Establishing similarities between artificial intelligence and biological intelligence is an important and interesting direction.
- The analogy between transformers and cue-based retrieval is clearly explained.
- The authors state their hypothesis clearly and support them with experiments.

### Weaknesses
- The paper appears to be concealing the important distinction between long-term and short-term memory. This makes the argued similarity appear somewhat odd. LLM short-term memory (prompt) si compared with human long-term memory (hippocampal subregion).
- An important motivation of the paper is *machine unlearning*, but this is usually a concern with regard to the LLM long-term memory (weights), not the short-term memory (prompt) studied in this paper.
- The experimental results for attention swapping are not surprising, mirroring many of the already existing causal interventions in the XAI literature.
- Understanding transformers from a memory-retrieval perspective is not a novel idea. [1]

[1] Bietti, Alberto, et al. "Birth of a transformer: A memory viewpoint." Advances in Neural Information Processing Systems 36 (2023): 1560-1588.

### Questions
- I see that the Encoding Specificity Principle is concerned with episodic (long-term) memory. Can the authors point to any evidence for a retrieval-like mechanism in human **short-term** memory?
- Do the authors see a way that their methods (or similar) could be applied for unlearning?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents two hypothesis and experiments around them.

H1 : Q is context encoder, K as trace memory index, and V as content store (This is obvious in how the Q,K,V are named)

H2 : Encoding Specificity Principle: Basically most effective circumstances for retrieval is most prominent cues during encoding are available at retrieval, so they believe this would be keywords 

The findings were not new and re-iterate what's already known.

### Strengths
Encoding-Specificity Argument, from a conceptual standpoint the arguments and experiments they conduct are sound and logical, rigorous testing using 6 different models.
The figures are clear in displaying the information and takeaways. Generally good contextualization before each figure as well.

### Weaknesses
The first experiment has very significant perturbation so the significance of their results does not support their idea too much. For the first experiment, they do not talk about how they swap the Q, K, and V matrices. So if the matrix V is swapped, what is it swapped with? 

Why does is H1 even a hypothesis ? (H1 - Q encodes retrieval cues, K indexes candidate traces by those cues, and V stores
retrievable content.) Isn't this why the matrices are called Query, Key and Value matrices? 

For H2 : The method for finding keywords is unclear. Also putting key-vector activations of certain keywords is same as zeroing out their attention scores. 

Overall, the experiments are not explained in enough detail. For H1 experiments, it would be nice to see examples of counterfactual and fact pairs used. The biggest weakness would be that unfortunately we don't learn anything new from the results of the experiments.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
