# Towards Understanding Primacy and Recency Effects in Mamba: A Mechanistic Perspective

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
We uncover a sparse subset of channels in Mamba's selective state-space block that serves as a substrate for early-input retention. Identified through structured recall tasks, ablating these channels selectively degrades early positional recall. Input periodicity systematically shifts Mamba’s discretization gate, amplifying the “lost-in-the-middle” effect by reallocating information across positions. Primacy and periodicity-driven effects, combined with recency, yield the characteristic U-shaped recall curve, aligning with effects known in Transformers but underexplored in state-space models. We further examine how distractor tokens affect Mamba’s temporal dynamics: recency, sustained by an exponential-decay mechanism, collapses under distraction as it moves the queried items deeper in the sequence. Finally, we demonstrate that the same sparse subset of channels transfers beyond recall. Intervening on them degrades the performance on downstream long-context understanding tasks, indicating that they function as data-agnostic long-term memory carriers. These results provide a common mechanistic picture of Mamba’s temporal profile, linking primacy, recency, and input periodicity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper identifies a subset of channels in Mamba models that are responsible for primacy and recency effects in recall tasks. These effects have previously been explored both behaviorally and mechanistically in Transformers, but are understudied in SSMs. They also show that recency effects "collapse" when the sequence model is shown distractor tokens, and that ablating the identified channels degrades performance on tasks "beyond" recall (e.g. long-context tasks).

### Strengths
* The mechanistic story is quite compelling, and I think that the experiments are well done
* I think it's really cool to see interp work done on non-Transformer models!

### Weaknesses
* The writing could use significant clarification; it took me a couple read throughs of the introduction to understand the main points (and I'm still not confident in my understanding). I also think that things should be restructured a bit, e.g. the experimental setup for 4.2/4.3/4.4 should be more clearly segmented from the results.
* I don't particularly know if these results are very significant. Primacy/recency effects are _interesting_, but it's not clear that developing a mechanistic understanding of these effects has any larger bearing on the field. I think the experiments that explore the effects of intervention on the recall-relevant channels are more interesting in this regard, but they aren't scoped out to the extent that I'd find compelling.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
1

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
This paper investigates position-dependent recall phenomena (primacy and recency effects) in Mamba, a state-space model architecture. The authors identify a sparse subset of channels responsible for early-input retention, demonstrate how input periodicity exacerbates "lost-in-the-middle" effects through the discretization mechanism, and show that recency emerges from exponential decay dynamics.

### Strengths
1. The long-term memory channels identification method is principled and is based on the cumulative recurrence matrix product.

2. Showing that identified channels impact real long-context benchmarks strengthens the claim that these are functionally important, not just artifacts of the synthetic task.

3. The paper is generally well-written with effective visualizations

### Weaknesses
1. Section 4.4 on recency largely reiterates known exponential decay dynamics from prior work (Wang et al., 2025). The distractor experiment adds empirical support but offers no new mechanistic insight. This section feels underdeveloped compared to the primacy analysis.

2. Mechanistic claims about $Δ_t$ and periodicity are underdeveloped.

### Questions
1. In Equation 3, you compute the product of diagonal entries. But $A^{(i)}_t ∈ R^{N×N}$ could have off-diagonal structure. How do you ensure you're only extracting diagonal recurrence?

2. For the initialization experiment (Figure 5a), why only Layer 31? Is this a long-term channel layer? What happens in other layers?

3. Table 1: The random ablation sometimes outperforms targeted ablation (e.g., 2WikiMultihopQA). This is counterintuitive—how do you explain this?

### Soundness
2

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
This paper focuses on the recency (high recall on recent tokens) and primacy (high recall on initial tokens) effects in SSMs. The main contribution is the finding that only a sparse subset of channels carries information from the first few tokens to the end of the sequence. The authors support this hypothesis with experiments on various datasets and models.

### Strengths
- The authors uncover a primacy mechanism in Mamba, where only a sparse subset of channels carries information from the first few tokens to the end of the sequence. 
- They then design a probing experiment to validate this claim.

### Weaknesses
- The empirical validation is somewhat limited in scope, as the experiments are confined to two specific model instances (Falcon Mamba 7B and Mamba 1.4B). 
- The paper's primary contribution appears to be incremental, as it largely builds upon existing frameworks. The central new discovery, while interesting, is presented without a substantial theoretical justification or a deep analysis of its underlying principles. The paper would be significantly strengthened by either providing a more rigorous theoretical grounding for this finding or by conducting a more in-depth empirical analysis.

Minor:
- Given that the paper's central focus is on the primacy effect, the authors might consider refining the title and abstract to more explicitly reflect this emphasis.
- The authors should consider exploring whether these effects influence overall model ability (e.g., performance on downstream tasks). If a significant impact is identified, the paper would be substantially strengthened by proposing or evaluating methods to mitigate these sequential biases.

### Questions
The original paper on recency finding [1] visualizes log-influential scores as part of its analysis. This paper, however, focuses solely on accuracy metrics. Could the authors elaborate on the decision to omit an analysis of log-influential scores?"

### Soundness
3

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
4

### Summary
This study investigates the primacy and recency effects (the "U-shaped recall curve") in SSMs. The authors focus on the primacy mechanism, identifying a sparse subset of internal channels that function as a long-term memory. They demonstrate that ablating these specific channels selectively degrades the model's recall of early information and impairs its overall performance on long-context tasks.

### Strengths
+ The authors test their sparse channel hypotheses under multiple conditions, strengthening their conclusions. This includes varying sequence lengths, comparing "Repeated Relation" inputs to "Random Relation" inputs, and using "distractor tokens" to test the fragility of the recency effect.

### Weaknesses
- The contribution can be somewhat limited. The paper's core novelty rests on identifying the "sparse channel hypothesis". However, the recency and primacy effects themselves are well-studied phenomena. The authors also acknowledge their observations are "common" (ln 24) and "investigated in early studies" (ln 447).

- Another central weakness is that the paper successfully demonstrates that this hypothesis holds but does not explore the reason or rationale why it exists. The contribution would be substantially stronger if the authors provided a theoretical grounding or an investigation into the architectural properties of the model that give rise to these observed sparse channels.

- While authors reveal the issues with Mamba, there is no solution provided. Do authors believe these problems are inherent and cannot be relieved easily?

### Questions
1. Have authors tried to reproduce the observation for more advanced SSM/linear attention architectures, e.g., Mamba2, DeltaNet?

### Soundness
2

### Presentation
3

### Contribution
2
