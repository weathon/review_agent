# On Theoretical Interpretations of Concept-Based In-Context Learning

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
In-Context Learning (ICL) has emerged as an important new paradigm in natural language processing and large language model (LLM) applications. However, the theoretical understanding of the ICL mechanism remains limited. This paper aims to investigate this issue by studying a particular ICL approach, called concept-based ICL (CB-ICL). In particular, we propose theoretical analyses on applying CB-ICL to ICL tasks, which explains why and when the CB-ICL performs well for predicting query labels in prompts with only a few demonstrations. In addition, the proposed theory quantifies the knowledge that can be leveraged by the LLMs to the prompt tasks, and leads to a similarity measure between the prompt demonstrations and the query input, which provides important insights and guidance for model pre-training and prompt engineering in ICL. Moreover,  the impact of the prompt demonstration size and the dimension of the LLM embeddings in ICL are also explored based on the proposed theory. Finally, several real-data experiments are conducted to validate the practical usefulness of CB-ICL and the corresponding theory.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This paper presents a rigorous theoretical analysis of in-context learning, developing mathematical frameworks to understand the excess risk and convergence properties of ICL. The authors establish several key theorems (4.3-4.5) that characterize the performance bounds of ICL under different conditions. The work analyzes both sufficient demonstrations (when F_n(x_n) is invertible) and insufficient demonstrations cases, as well as complete versus incomplete model scenarios. The theoretical framework employs matrix analysis, eigenvalue decompositions, and optimization techniques to derive tight bounds on the excess risk.

### Strengths
1. **Rigorous Mathematical Framework**: The paper provides comprehensive mathematical proofs with detailed lemmas and corollaries, establishing a solid theoretical foundation for understanding ICL behavior.

2. **Comprehensive Coverage**: The analysis covers multiple important cases - sufficient vs. insufficient demonstrations, complete vs. incomplete models, providing a nuanced understanding of different ICL scenarios.

3. **Novel Theoretical Insights**: The connection between eigenvalue properties (particularly λ_1(F(x_Q)F_n^(-1)(x_n))) and ICL performance provides new theoretical insights into when ICL succeeds or fails.

4. **Practical Relevance**: The bounds derived (e.g., Theorem 4.5 on label prediction error) have direct implications for understanding ICL's practical performance limits.

5. **Technical Depth**: The use of advanced techniques like KKT conditions, matrix perturbation theory, and generalized eigenvalue analysis demonstrates technical sophistication.

### Weaknesses
1. **Limited Empirical Validation**: From the provided content, there appears to be heavy emphasis on theory without corresponding experimental validation of the theoretical bounds.

2. **Restrictive Assumptions**: The analysis relies on specific assumptions (e.g., f_K(x,y) = 1/√M) that may not hold in practical ICL scenarios, potentially limiting the applicability.

3. **Accessibility**: The highly technical nature and dense mathematical notation may limit the paper's accessibility to practitioners and researchers outside theoretical ML.

4. **Gap to Practice**: The theoretical model may oversimplify real ICL mechanisms in large language models, where the actual learning dynamics are likely more complex.

### Questions
1. **Empirical Validation**: How well do the theoretical bounds match empirical ICL performance in actual language models? What experiments were conducted to validate these bounds?

2. **Assumption Justification**: What is the justification for the specific form f_K(x,y) = 1/√M? How sensitive are the results to violations of this assumption?

3. **Practical Implications**: Can the authors provide concrete guidelines for practitioners on how to use these theoretical insights to improve ICL performance?

4. **Comparison with Existing Work**: How do these bounds compare with existing theoretical analyses of ICL? What specific gaps in prior work does this address?

5. **Extension to Modern Models**: How do these results extend to transformer architectures and attention mechanisms that are central to modern ICL?

6. **Tightness of Bounds**: Are the derived bounds tight? Can you provide examples where the bounds are achieved?

7. **Computational Considerations**: What are the computational implications of computing these bounds in practice, especially for large-scale models?

### Soundness
2

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
This paper investigates the theoretical foundations of concept-based in-context learning (CB-ICL). It develops a formal analysis that explains why and when CB-ICL performs effectively with limited demonstrations. The proposed theory models the relationship between prompt representations and LLM embeddings, introduces a similarity measure for prompt and query selection, and characterizes factors such as embedding dimensionality and demonstration sufficiency. Experimental results on several benchmarks support the theoretical findings and show that CB-ICL performs comparably or better than standard in-context learning approaches.

### Strengths
1. Carefully framed theoretical contribution with explicit assumptions. The paper presents a coherent theoretical framework, states assumptions clearly, and supplies largely complete derivations as well as interpretable quantitative measures (e.g., mean-squared excessive risk and the similarity score) that connect embedding geometry to prediction error.
2. Substantial and reasonably broad empirical validation. Experiments cover multiple benchmarks and model families, and the proposed similarity-based “golden” selection consistently improves performance in several settings, providing practical evidence that the theory has explanatory power.
3. Thoughtful synthesis and targeted novelty over prior ingredients. While the paper builds on familiar analytical tools used in prior ICL work (e.g., linear-model style analyses, projection/basis arguments and residual risk notions), it integrates these elements into a concept-based CB-ICL formulation and derives a practical similarity metric and selection criteria.

### Weaknesses
1. Notation and presentation are cluttered and sometimes unclear. Many symbols are visually similar (multiple uses of f with different inputs, closely related subscripts/superscripts) and some variables (e.g., K, M) are not introduced at the first encounter, increasing the reader’s cognitive load and risk of misinterpretation.
2. Method exposition is limited because of emphasis on theory. The manuscript concentrates on mathematical derivations at the expense of a clear, step-by-step description of the practical pipeline due to page constraints. Key implementation and procedural details are only conveyed via a single flow diagram, and the diagram’s symbols seem to be inconsistently used relative to the main text, which impedes reproducibility and practical understanding.
3. Ambiguity about how LLM embeddings are obtained and pooled. The paper treats f(x,y) as the semantic embedding but gives no practical details on how embeddings are extracted from LLM internals (which layer(s), pooling over token sequence, handling of label tokens, etc.), nor empirical comparisons to off-the-shelf sentence encoders (BERT / sentence-BERT / specialized sentence encoders) that might provide alternative embeddings.
4. Limited statistical reporting and stronger baselines. Results are reported as point estimates (accuracy percentages) across models/datasets, but confidence intervals, standard deviations from repeated runs, or formal significance tests are missing; comparisons against stronger baselines or more relative work are limited.
5. Scope limited to classification. The theoretical framework and experiments focus on label prediction, so applicability to structured outputs, chain-of-thought (CoT) style reasoning, or sequence generative tasks is not established; the paper should either clearly state this limitation or outline how the framework could be extended.
6. Theory vs. empirical support for embedding dimension (K). The paper theoretically predicts excessive risk grows with embedding dimension K, but presents only a theoretical argument; there is little direct empirical ablation that systematically manipulates embedding dimension to validate this trade-off. 
7. No reported computational cost or runtime analysis. This paper does not provide wall-clock timing, memory, or computational-cost figures for constructing embeddings, computing similarity scores, or golden selection, which would be valuable for assessing practical adoption.
8. Normalization preprocessing steps are specified mathematically but not operationally. The normalization constraints on f(x,y) are stated (zero-mean, unit variance across labels and a padded bias dimension), yet the paper lacks an explicit description of how these normalizations are implemented in experiments.

### Questions
1. On harder examples (the “Pro/Diamond” tiers) the reported gains shrink and especially, for MMLU Pro, CB-ICL sometimes underperforms standard ICL. Can the authors explain this behavior and identify which dataset or model characteristics drive the reduced or negative benefit?
2. Could using dedicated sentence/semantic encoders (e.g., all-MiniLM or other off-the-shelf sentence embedding models) yield more accurate semantic vectors and change the method’s empirical performance? 
3. The paper frequently refers to f(x,y) as a “semantic embedding.” However, the representation produced by an LLM may capture not only semantic content but also aspects of reasoning, contextual adaptation, or latent pattern discovery. How should this representation be conceptually understood within the proposed framework — is it purely semantic, or does it implicitly encode reasoning dynamics as well?
4. From the methodological description, it seems feasible to construct embeddings without explicitly conditioning on the label — that is, to use f(x) instead of f(x, y) — since the subsequent computations could still be grouped by y. It would be helpful to clarify how this alternative would differ conceptually or empirically from the proposed formulation.
5. The reported correlation between residual risk and task accuracy appears substantially weaker on GPQA than on MMLU. What factors might explain this discrepancy? For example, could the difference arise from the reasoning-heavy nature of GPQA, smaller sample size, or less stable alignment between the embedding similarity and true task difficulty?
6. The theoretical analysis assumes that both the demonstrations and the query are drawn from the same conditional distribution. In practice, however, in-context examples are often selected non-randomly or from a slightly different sub-distribution due to prompt design or dataset bias. How sensitive are the theoretical bounds and conclusions to such distributional mismatches or selection biases?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies Concept-Based In-Context Learning (CB-ICL), a view of ICL where few-shot demonstrations are used to estimate a concept representation from a frozen LLM and then predict query labels by aligning the query embedding with that concept. The authors formalize a regression-style objective and analyze three regimes: (i) complete & sufficient (embedding contains the target concept and demos span it), (ii) complete but insufficient (embedding complete, demos rank-deficient), and (iii) incomplete & insufficient (embedding misses part of the concept and demos don’t cover it). They derive spectral risk bounds tied to the demo/embedding matrices, and present a risk→classification link via a margin argument. From the bounds, they introduce a demo–query similarity score to guide demonstration selection. Experiments across several LLM backbones and benchmarks report improvements when choosing demonstrations by the proposed score compared with random selection, and ablations illustrate how demo quality and alignment influence CB-ICL performance.

### Strengths
- The CB-ICL setup captures the “prototype” intuition behind many prompt designs, making an implicit practice explicit and analyzable.
- By distinguishing completeness from sufficiency, the analysis teases apart backbone capacity vs. prompt curation which is useful for diagnosing where performance originates.
- The bound induces a demo–query similarity measure: selecting demos that are more similar (in the sense derived from the matrices in the proofs) tightens the bound, offering a criterion for demo selection that aligns with the theory.

### Weaknesses
- The method treats a decoder-based LLM as an encoder to produce task-relevant representations for concept estimation. However, decoder-only models are not generally optimized for embedding quality, especially without contrastive representation training. The proposed CB-ICL framework assumes these embeddings are sufficient for reliable concept extraction and similarity-based prediction, this raises doubts about suboptimal representations from a decoder-only backbone.
- The paper’s conclusions are established within the concept-based ICL pipeline (estimating concept vectors from few-shot demos and predicting via similarity). It is unclear whether these results extend to general ICL behaviors. The manuscript does not delineate the boundary between what is specific to the linear view and what persists for broader ICL mechanisms, leaving the external validity of the conclusions uncertain.
- Theorem 4.5 says the label error probability is lower bounded under small excessive risk, yet the paragraph then claims this “leads to small error probability.” It seems an upper bound can certify small error not a lower bound.
- The text defines complete LLMs (residual = 0) and sufficient demos (invertible). However, no empirical diagnostics or proxies are provided to distinguish failure modes in practice. Readers cannot tell whether poor performance arises from missing concept coverage in the chosen demonstrations or from backbone incompleteness of the embedding itself.
- Section 5.2 compares random vs their similarity (“golden”) selection only, so the new score’s advantage over recent strong baselines (also considering coverage and diversity in selection) remains untested.

### Questions
- Could the authors give a formal derivation of the claimed connection between the proposed extractor and an attention-style computation with a quadratic (i.e., linear) function? (line 330)
- The extractor yields coefficients that weight the demonstrations when predicting the query. Are there constraints (non-negativity, sum-to-one) applied? If not, how should we interpret negative or >1 coefficients in the context of “attention-like” weighting?
- Could the author include ablation that prompt concept extractor where the extractor is replaced by a  attention-style computation.
- LLMs do
- line 331, attetion -> attention

### Soundness
3

### Presentation
3

### Contribution
3
