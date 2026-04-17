# HGNet: Scalable Foundation Model for Automated Knowledge Graph Generation from Scientific Literature

- Decision: Accept (Poster)
- Scores: 4, 8, 2

## Abstract
Automated knowledge graph (KG) construction is essential for navigating the rapidly expanding body of scientific literature. However, existing approaches face persistent challenges: they struggle to recognize long multi-word entities, often fail to generalize across domains, and typically overlook the hierarchical and logically constrained nature of scientific knowledge. While general-purpose large language models (LLMs) offer some adaptability, they are computationally expensive and yield inconsistent accuracy on specialized, domain-heavy tasks such as scientific knowledge graph construction. As a result, current KGs are shallow and inconsistent, limiting their utility for exploration and synthesis. We propose a two-stage framework for scalable, zero-shot scientific KG construction. The first stage, Z-NERD, introduces (i) Orthogonal Semantic Decomposition (OSD), which promotes domain-agnostic entity recognition by isolating semantic “turns” in text, and (ii) a Multi-Scale TCQK attention mechanism that captures coherent multi-word entities through n-gram–aware attention heads. The second stage, HGNet, performs relation extraction with hierarchy-aware message passing, explicitly modeling parent, child, and peer relations. To enforce global consistency, we introduce two complementary objectives: a Differentiable Hierarchy Loss to discourage cycles and shortcut edges, and a Continuum Abstraction Field (CAF) Loss that embeds abstraction levels along a learnable axis in Euclidean space. To the best of our knowledge, this is the first approach to formalize hierarchical abstraction as a continuous property within standard Euclidean embeddings, offering a simpler and more interpretable alternative to hyperbolic methods. To address data scarcity, we also release SPHERE, a large-scale, multi-domain benchmark for hierarchical relation extraction. Our framework establishes a new state of the art on benchmarks such as SciERC, SciER and SPHERE benchmarks, improving named entity recognition (NER) by 8.08\% and relation extraction (RE) by 5.99\% on the official out-of-distrubtion test sets. In zero-shot settings, the gains are even more pronounced, with improvements of 10.76\% for NER and 26.2\% for RE, marking a significant step toward reliable and scalable scientific knowledge graph construction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a two‑stage framework for scientific KG construction: Z‑NERD for zero‑/low‑shot entity recognition via Orthogonal Semantic Decomposition (OSD) and a Multi‑Scale TCQK attention that specializes heads to different n‑gram lengths, and HGNet for hierarchy‑aware relation extraction with three message‑passing channels (parent/child/peer) and two global regularizers: a Differentiable Hierarchy Loss (acyclicity + shortcut penalties) and a Continuum Abstraction Field (CAF) that learns a single Euclidean “axis of abstraction.” Results show new SOTA on SciERC/SciER/BioRED and on a new multi‑domain SPHERE benchmark; Table 1 (p. 8) and Tables 2–4 (p. 9) report average gains of +8.08 F1 (NER) and +5.99 F1 (RE), with larger zero‑shot gains (+10.76 / +26.2). Figures 5–6 (pp. 20–21) illustrate Z‑NERD/ HGNet; Appendix A.7 details scalable acyclicity via Krylov methods.

### Strengths
- TCQK introduces architectural inductive bias for multi‑word entities; CAF imposes a simple, interpretable Euclidean ordering of abstraction; DHL neatly encodes DAG and anti‑shortcut constraints. 
- Solid ablations (Tables 1–3) and complexity notes for DHL 
- Clean factorization of challenges (entity coherence, domain generalization, hierarchy, global consistency) and matching components.
- Consistent gains on SciERC/SciER and large zero‑shot improvements on SPHERE

### Weaknesses
SPHERE is generated and self‑annotated by LLMs; may encode stylistic biases, inflated structure, or task leakage that favors the proposed inductive biases. Though Limited human validation is described. (App. A.3.)  
-  Large LLMs are evaluated “as is” (several OOM), and there is no strong hyperbolic/order‑embedding baseline for hierarchy—weakening the “simpler & better than hyperbolic” claim.
- CAF relies on anchors and topological depths—procedure unclear for standard datasets; OSD’s learning objective (beyond feature concatenation) is under‑specified.
- No runtime/throughput or memory comparisons despite “lightweight” claims; parameter counts not tabulated per component. Report wall‑clock, FLOPs, and memory vs. PL‑Marker/HGERE and vs. GCN/GAT; quantify DHL/CAF overhead and benefits from the Krylov approximation

### Questions
- How does HGNet behave if the true structure is not strictly hierarchical (e.g., multiple inheritance, cross‑links)? Any failure cases beyond A.8?
- SPHERE validation: What fraction was human‑audited? Provide inter‑annotator agreement and a human‑written subset to test robustness to LLM‑style.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this study, the authors proposed a two-stage framework for scalable, zero-shot scientific KG construction to resolve and mitgate the challenges of knowledge graph (KG) construction from massive literature data (especially for the challenge of long multi-word entity recognition. 
The evaluation results showed that the proposed model improved the entity recognition and relationship extraction reliablly.

### Strengths
Clear and appropriate model design to resolve the specific challenges in KG construction from literature data
Solid experimental evaluation with reliable improvement.

### Weaknesses
None

### Questions
How fast is the proposed model for processing massive literature data?
How to build complete sets of graphs of abstract accross diverse domains? Which can be important for learning/understanding the domain knowledge.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper tackles the problem of generating KGs from scientific literature, focusing on four challenges that require more care than in general purpose KG construction: 
(i) careful extraction of multi-word scientific entities, 
(ii) domain generalization for this task of scientific NER, 
(iii) hierarchical relation extraction, and 
(iv) logical consistency in relation extraction (e.g. avoiding cycles).

Their contributions are: 
(i) a novel methodology for NER that looks for signals of semantic shifts to more robustly extract entities across domains and that incorporates convolutional filtering at different scales to specifically look for entities with different lengths, 
(ii) a novel methodology for RE that uses an explicitly hierarchical GNN to learn entity embeddings, a differentiable loss to encourage the KG to be a DAG, and losses that explicitly encourage the entity embeddings to encode hierachical relationships along a specific axis, and
(iii) a novel large multi-domain scientific KG benchmark data set

### Strengths
- The proposed ideas are well presented in isolation, e.g. the several hypotheses that the authors present to justify their system design seem reasonable. 
- The ideas seem original in the area of scientific KG generation to me, and some are likely to be influential, especially the NER system and the use of the differentiable loss to ensure a logically consistent KG

### Weaknesses
- The architecture of the system is not sufficiently described in the main body of the paper to understand or reproduce the system. E.g. sections 3.1 and 3.2 that provide the methodologies for the NER and RE do not mention a specific architecture; rather they mention losses and modifications to attention heads. One must read the appendix to begin to understand the system.

### Questions
- In equation 7, what is d?
- In section 3.2.3 by pinning roots and leaves to 0 and 1, and requiring a delta margin between parents and children, doesn't this undesirably limit the number of levels of abstraction you could have to 1/delta?
- where do you get ground truth topological depth scores in (11)?
- In HGNet you have a message passing system with latent relational predictors, the differentiable hierarchy loss, and the hierarchical separation loss, as well as the continuum abstraction loss -- how do these pieces fit together and how are they co-trained? 
- The description of HGNet in section 3.2 only mentions taking the entities from the NER stage as input, but the scientific documents must be used in predicting the relationship between two entities. This is not mentioned anywhere in section 3.2. E.g. the latent relational predictor only take the entity embeddings as input.
- The description of HGNet in section 3.2 only mentions extracting relations of the type {parent-of, peer-of, no-edge}. Where in the KG generation does one extract the usual <head, relation, tail> triplets?

### Soundness
2

### Presentation
1

### Contribution
3
