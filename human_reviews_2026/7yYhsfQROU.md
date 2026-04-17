# Benchmarking Diversity in Image Generation via Attribute-Conditional Human Evaluation

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Despite advances in photorealistic image generation, current text-to-image (T2I) models often lack diversity, generating homogeneous outputs. This work introduces a framework to address the need for robust diversity evaluation in T2I models. Our framework systematically assesses diversity by evaluating individual concepts and their relevant factors of variation. Key contributions include: (1) a novel human evaluation template for nuanced diversity assessment; (2) a curated prompt set covering diverse concepts with their identified factors of variation (e.g. prompt: An image of an apple, factor of variation: color); and (3) a methodology for comparing models in terms of human annotations via binomial tests. Furthermore, we rigorously compare various image embeddings for diversity measurement. Notably, our principled approach enables ranking of T2I models by diversity, identifying categories where they particularly struggle. This research offers a robust methodology and insights, paving the way for improvements in T2I model diversity and metric development.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenge of rigorously evaluating diversity in text-to-image generative models by proposing an attribute-conditional human evaluation framework. The authors argue that existing diversity assessments are ambiguous and often conflate fidelity and diversity, leading to unreliable conclusions when variation factors are unspecified. To fix this, they formalize diversity along specific concepts and their key factors of variation (e.g., color of apples), develop a systematically generated prompt set spanning 86 concept-attribute pairs, and design a validated human evaluation template that improves annotation accuracy by explicitly focusing raters on a defined attribute. Using this benchmark, they collect 24,591 human annotations comparing five leading T2I models and show statistically significant model ranking differences—identifying Imagen 3 and Flux 1.1 as the most diverse models. They also evaluate the Vendi Score with various embedding choices, finding that ~65–80% alignment with humans is possible depending on representation space, while revealing sensitivity to embedding selection. Overall, the work provides a principled, scalable method for measuring and comparing model diversity and establishes a valuable dataset for validating future automated metrics.

### Strengths
The paper convincingly shows that diversity evaluation is ambiguous without specifying both the concept and the factor of variation.

A systematically generated prompt set covering 86 concept-attribute pairs and a large-scale human-rated dataset (24,591 comparisons), filling an evaluation gap in T2I research. 

A well-designed human evaluation template that significantly improves annotation accuracy by focusing raters on explicit attributes.

### Weaknesses
Real-world diversity often involves interactions between multiple attributes (e.g., shape + color + background). The framework isolates each property and does not consider compositional or multi-attribute diversity, leaving a large portion of the diversity landscape unexplored.

The benchmark focuses on everyday categories (food, nature, manufactured objects), often static and object-centric. This might underrepresent scenarios like human attributes, fine-grained scenes, and spatial relations, and overfit evaluation to objects common in existing datasets (ImageNet bias). As a result, findings may not generalize to domain-specific or creative applications.

More diverse does not equal more realistic. A model generating incoherent or noisy variations could superficially score high, and the framework does not explicitly guard against trade-offs between diversity and quality.

### Questions
How does the framework generalize to multi-attribute or compositional variation?

What is the framework’s sensitivity to prompt rephrasing or category expansion (e.g., uncommon or artistic concepts)?

Would incorporating vision-language grounding models trained for attribute classification improve metric-human alignment?

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
The paper suggests that diversity in T2I should be measured relative to an explicit attribute (e.g., color, species) rather than globally. It proposes a simple human-evaluation template shows two 8-image sets for a concept, name the target attribute, has raters count distinct values, then choose Left/Right/Equal/Unable, and validates that this count-then-compare instruction improves reliability on a small golden set. A larger study across several models and many concept–attribute pairs aggregates per-concept wins with binomial tests (Imagen 3 and Flux 1.1 typically strongest). For automation, the authors reuse Vendi with different embeddings; image-only embeddings (ViT/DINO/Inception) align best with human judgments on clear-gap cases, while tie detection is weak, and text conditioning adds little.

### Strengths
1) Attribute-conditioned, count-anchored human evaluation reduces ambiguity and improves rater reliability; the procedure is easy to replicate.
2) Comparison shows image-only embeddings with Vendi track human judgments on large-gap cases, offering a usable baseline for fast iteration.

### Weaknesses
1) Limited Novelty; contributions are primarily protocol + dataset + empirical comparisons.
2) The paper does not thoroughly analyze where autoraters fail (e.g., ties and small gaps, rare attribute values, per-concept heatmaps or illustrative disagreements).

### Questions
1) Why were other diversity metrics or Vendi variants (e.g., quality-weighted) not considered?
2) Why are the autoraters' failure cases not probed thoroughly? If failures might be due to the data, using SigLip2 might help. In general, adding results with SigLip is better. Ideally, it would be better if a new metric were proposed after analysing the failure cases to overcome them.
3) Appendix E.6 suggests the MLLM judge (Gemini 2.5 Flash) can outperform the embedding-based Vendi setup on the golden set. Could you expand this and analyze whether, in practice, one should prefer Vendi or an MLLM judge for diversity measurement going forward?

### Soundness
3

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
3

### Summary
This paper presents a framework for evaluating diversity in T2I generation models, addressing the persistent ambiguity in how diversity is defined and measured. This paper proposes a three-part evaluation framework consisting of a novel human evaluation template for nuanced diversity assessment, a curated prompt set covering diverse concepts, and a methodology for comparing models
in terms of human annotations via binomial tests. The authors build a benchmark of 86 concept–attribute pairs and collect over 24,000 human annotations comparing major models such as Imagen, DALLE, and Flux. Results show strong annotation reliability, meaningful model ranking, and high correlation between human judgments and automatic diversity metrics like the Vendi Score.

### Strengths
The authors correctly identify a major gap, which is the lack of a principled, attribute-grounded approach to diversity evaluation.

Extensive annotation effort, containing 240000 samples with high inter-annotator reliability α > 0.8, provides a strong empirical foundation.

### Weaknesses
The prompt generation pipeline still requires extensive human verification and filtering despite LLM assistance, which limits scalability and reproducibility across domains.

The framework focuses primarily on visual diversity, without considering semantic or contextual diversity dimensions that may better reflect real-world generative quality.

The evaluation of automatic metrics is narrow, emphasizing the Vendi Score while omitting comparisons to other recent or theoretically grounded diversity measures.

The paper provides limited qualitative analysis explaining why certain models perform better or worse in diversity, which reduces interpretability and actionable insights for improving model design.

### Questions
Refer to Weaknesses

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
4

### Summary
The paper argues that diversity in T2I models is poorly defined unless the concept and attribute are both specified, and that underspecified human evaluation can reverse model rankings as shown in Figure 1. It contributes three main elements: first, a framework consisting of clearly defined per-attribute diversity, a systematically generated prompt set covering everyday concepts, and a tailored human evaluation template; second, a human study showing that the template, which includes an attribute cue and a counting anchor, significantly improves reliability on a golden set; third, a large-scale annotation effort with 24591 comparisons from 20 raters, used to rank five recent models in pairs and to validate automated raters based on the Vendi Score with various embeddings and conditioning methods.

### Strengths
S1: The proposed idea is quite interesting to me. The problem formulation helps to de-confound diversity. They formalize per-attribute diversity and explain why generic prompts or generic human templates mix up fidelity or content variation with true diversity. This approach is both conceptually clear and practical to implement.

S2: The human study design and the aggregation or statistical methods are serious and thorough. There are 24591 annotations across 5 models, with majority-vote aggregation, binomial tests for pairwise significance, and reliability above 0.8. This represents all good practice.

### Weaknesses
W1: The scope and external validity of the concepts and attributes are the main concern. The prompt set focuses on people-excluded, everyday, ImageNet-like concepts and explicitly leaves out person categories. While suitable for a proof of concept, this skips socially sensitive forms of diversity such as demographics or geographic context, as well as open-world composition where multiple attributes interact, like material combined with style and function. The selection is based on LLM-proposed attributes with light manual pruning, which may reflect LLM biases about what matters for a concept. A stratified audit of concept and attribute coverage, including negative cases where attributes are visually subtle, would help strengthen the claims.

W2: Accuracy drops from 4×4 to 8×8 when the counting anchor is removed. Even with counting, larger sets remain more difficult. However, the main study uses 8-image sets for pairwise model ranking. A full power analysis connecting set size, rater effort, and test sensitivity would be helpful. For example, how much does reliability or win-rate significance drop if the number of raters is halved or the set size is changed?


W3: Pairwise ranks depend on the prompt set. Appendix D.2 shows that shrinking the set reduces significance and can reverse some results. While this is acknowledged, the main conclusions would be more convincing with cross-prompt-set stability, such as bootstrapped subsamples or category-conditioned rankings.

### Questions
How broad is the domain of your concept–attribute pairs beyond ImageNet-like nouns? Could the absence of human-related or compositional prompts bias the conclusions about model diversity?

What procedures ensured that attributes proposed by the LLM reflect actual visual variation rather than linguistic association? Were any validation checks or human audits done before removing ambiguous pairs?

Since prompt-set sufficiency is only discussed in Appendix D.2, can you report cross-prompt or bootstrapped stability analyses showing that model rankings stay consistent across random subsets of concepts?

For the human evaluation, the golden reference sets appear to be generated by a specific T2I model. Could model-specific artifacts influence annotators’ perception of diversity? Have you considered using photographic or cross-generator golden sets to test generality?

### Soundness
3

### Presentation
4

### Contribution
4
