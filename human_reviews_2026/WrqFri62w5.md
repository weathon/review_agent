# UniRec: Unified Multimodal Encoding for LLM-Based Recommendations

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4

## Abstract
Large language models (LLMs) have recently shown promise for multimodal rec-
ommendation, particularly with text and image inputs. Yet real-world recom-
mendation signals extends far beyond these modalities. To reflect this, we for-
malize recommendation features into four modalities: text, images, categorical
features, and numerical attributes, and emphasize unique challenges this hetero-
geneity poses for LLMs in understanding multimodal information. In particular,
these challenges arise not only across modalities but also within them, as attributes
(e.g., price, rating, time) may all be numeric yet carry distinct meanings. Beyond
this intra-modality ambiguity, another major challenge is the nested structure of
recommendation signals, where user histories are sequences of items, each car-
rying multiple attributes. To address these challenges, we propose UniRec, a
unified multimodal encoder for LLM-based recommendation. UniRec first em-
ploys modality-specific encoders to produce consistent embeddings across het-
erogeneous signals. It then applies a triplet representation—comprising attribute
name, type, and value—to separate schema from raw inputs and preserve semantic
distinctions. Finally, a hierarchical Q-Former models the nested structure of user
interactions while maintaining their layered organization. On multiple real-world
benchmarks, UniRec outperforms state-of-the-art multimodal and LLM-based
recommenders by up to 15%, while extensive ablation studies further validate the
contributions of each component.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes UniRec, a unified multimodal encoder for LLM-based recommendation systems, which integrates heterogeneous signals (text, image, categorical, numerical) through modality-specific encoders, triplet-based schema representation, and a hierarchical Q-Former. The method is evaluated on three real-world datasets and shows improvements over some existing multimodal and LLM-based baselines.

### Strengths
- Novel use of triplet representation for schema-aware attribute encoding.

### Weaknesses
- [Mandatory] The numerical encoder lacks sufficient implementation details in Appendix A, hindering reproducibility.

- [Mandatory] The triplet representation via simple summation may not preserve the uniqueness of (name, type, value), as the same summed embedding could result from multiple triplets.

- [Mandatory] SThe meaning of subscript $t$ in $\mathbf{H}_{i_t}$ is not clearly defined.

- [Mandatory] The origin of review context embedding $\mathbf{c}$ and timestamp embedding $\mathbf{p}$ is not explained.

- [Mandatory] Dimensionality and structure of user/item representations are inconsistently described.

- [Mandatory] Implementation Details are missing the introduction of some hyperparameters, such as the number of queries for Item/User Q-Former.

- [Mandatory] Selected multimodal recommendation baselines are outdated; more recent SOTA models  (e.g., MENTOR [1], SMORE [2], COHESION [3], PGL [4], and GUME [5])

- [Mandatory] Comprehensive hyperparameter sensitivity analysis is needed.

- [Mandatory] No discussion of efficiency (training/inference time, memory footprint) is included.

### Questions
Please refer to Weaknesses. Btw, I have some optional questions:

- [Optional] Have you tested the model in cold-start or missing-modality scenarios?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes UniRec, a unified multimodal encoder that enables LLMs to process heterogeneous recommendation signals (text, image, categorical, numerical) for next-item prediction. It introduces (i) modality-specific encoders that map every attribute into a shared hidden space, (ii) a triplet representation <name, type, value> to keep schema semantics, and (iii) a two-level Q-Former that first compresses item attributes into an item embedding and then aggregates a sequence of such embeddings into a user embedding. A two-stage training scheme (contrastive + reconstruction pre-training, then LoRA fine-tuning) is used. Experiments on Amazon Beauty/Baby and Yelp show 8–16 % MRR gains over prior multimodal and LLM-based recommenders.

### Strengths
* Ablations on schema tokens, user tokens, fusion strategy, and token-count sensitivity are reported; statistical significance is provided.
* Clear writing and thorough reproducibility statement: including ethics, data licences, and detailed hyper-parameters.

### Weaknesses
1. Limited novelty
    * The triplet disentanglement idea is essentially the “field-name + field-type + value” trick already used in CTR prediction (e.g., FiBiNET) and in very recent LLM-rec works (e.g., NoteLLM-2).
    * Hierarchical Q-Former is a direct application of BLIP-2’s item→user two-stage aggregation; no new attention mechanism or theoretical insight is offered. Two-stage contrastive→fine-tune training mirrors BLIP-2; hence the methodological contribution is incremental.

2. Missing strong baselines
    * Many SOTA state-of-the-art multimodal LLM recommenders published are missed.
    * Classical ID-based sequential models with recent feature fusion tricks are omitted even though they outperform many LLM baselines on Amazon data.

3. Heuristic design choices with insufficient ablation
    * Triplet fusion by simple addition (h=a+t+v) is adopted without comparing concatenation, cross-attention; the paper only ablates “w/ vs w/o triplet”, not the fusion operator.
    * Earlier sections argue alignment is critical, yet no alignment quality metric (e.g., modality retrieval accuracy) is given.

4. Evaluation gaps
    * Offline next-item prediction only; no online A/B or reranking experiment, and no cold-start vs warm-start breakdown.
    * Datasets are public 5-core Amazon/Yelp; results may not generalise to industrial datasets with long-tail items and noisy attributes.

### Questions
See Weaknesses part.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents UniRec, a structured multimodal encoding framework that bridges complex recommendation data with LLM-based reasoning. Unlike prior works that either concatenate multimodal features or rely on token-based flattening, UniRec argues that recommendation data requires semantic structure preservation, both across and within modalities. The paper introduces three key contributions: (1) a triplet-based attribute representation that encodes attribute name, type, and value to avoid semantic ambiguity; (2) a hierarchical Q-Former architecture that first aggregates multimodal features into item-level semantics and then models user preference across items; and (3) a unified interface for feeding structured user–item embeddings into LLMs for recommendation.

Empirical results show meaningful improvements over multimodal and LLM-based baselines, with ablations validating the benefits of the triplet representation and hierarchical aggregation. The work is conceptually clear, practically relevant, and provides a modular approach that can serve as a front-end encoder for various LLMs. While the idea of modality-specific encoding is not entirely new, UniRec’s contribution lies in advancing it into a semantically structured and hierarchically aligned paradigm tailored for LLM-based recommendation.

### Strengths
1. The paper identifies a real and under-explored challenge in LLM-based recommendation: multimodal item data is not only heterogeneous across modalities (text vs. image vs. numeric vs. categorical) but also semantically inconsistent within modalities. This articulation is accurate, timely, and well-motivated.

2. UniRec acts as a “pre-LLM structured encoding layer”, enabling the LLM to focus on reasoning rather than raw multimodal fusion. The framework is modular, can plug into various LLMs, and is potentially useful for industry scenarios where different modalities evolve over time.

### Weaknesses
1. The core idea of separating multimodal signals and encoding them independently before fusion is not fundamentally new and has been widely explored in multimodal recommendation (e.g., MMGCN, LATTICE, MISSRec) and recent LLM-based approaches. The paper’s positioning currently risks overstating originality. To justify a top-tier contribution, the authors must more clearly demonstrate what UniRec achieves beyond “structured multimodal encoding,” and why the proposed triplet and hierarchical design represent a substantial conceptual advance rather than a refined engineering variant.

2. Encoding each attribute as (name, type, value) is intuitive but insufficiently grounded. There is no theoretical discussion or empirical evidence clarifying why this specific decomposition is optimal, whether all three components are required, or how it compares to alternative schema- or ontology-based representations. The absence of ablations isolating the role of each component (e.g., removing “type” or replacing with learned embedding) weakens the claim that triplet representation fundamentally addresses intra-modal semantic confusion.

3. The two-level Q-Former architecture (item-level then user-level) is plausible, but the paper provides no analysis of why two levels are sufficient, whether the hierarchy should reflect deeper interactions (e.g., session-level, aspect-level), or whether simpler architectures could achieve comparable results. Without structural ablations or complexity–performance tradeoff studies, it is unclear whether improvements stem from hierarchy or merely from additional attention computation.

### Questions
1. Several prior works (e.g., MMGCN, LATTICE, MISSRec, and recent LLM-based multimodal recommenders) already encode modalities separately before fusion. Could authors clearly articulate what is fundamentally novel in UniRec beyond structured multimodal encoding, and why triplet + hierarchical Q-Former constitutes a conceptual advance rather than an engineering refinement?

2. What empirical or theoretical evidence supports the necessity of encoding (attribute name, attribute type, attribute value) separately? Can authors provide ablations isolating the role of each component (e.g., removing “type” or merging “name + value”) to demonstrate that the full triplet is required?

3. Why is a two-level Q-Former hierarchy optimal? Did authors experiment with alternative hierarchies (e.g., session-level, aspect-level, deeper hierarchies)? Without such analysis, how can we be confident the design is not ad-hoc?

### Soundness
3

### Presentation
3

### Contribution
3
