# NAVI: Inductive Alignment for Generalizable Table Representation Learning

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Effective representation learning for tabular data is critical for downstream tasks such as information retrieval, classification, and missing value imputation. However, existing transformer-based models often fail to generalize across in-domain tables, either preserving schema–value semantics at the cost of robustness or enforcing stability while losing fidelity. We propose NAVI—Entropy-aware Alignment via Header–Value Induction—a framework that unifies both desiderata.  NAVI introduces header–value segments as the atomic unit of table representation, serialized in an order-independent manner and anchored by global header embeddings. Structure-aware masked segment modeling enforces schema–value dependencies via balanced masking over headers, values, and tokens, while entropy-driven segment alignment aligns low-entropy (domain-coherent) columns with global headers and high-entropy (entity-discriminative) columns with row-specific values. This joint design yields representations that are both consistent and semantically faithful. Extensive experiments on large-scale benchmarks show that NAVI consistently outperforms baselines in generative and discriminative tasks while mitigating schema-level inconsistencies. The source code of NAVI is available at https://anonymous.4open.science/r/navi.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces NAVI, a new framework for tabular representation learning that jointly optimizes for fidelity and consistency. Its core idea is to represent rows as unordered sets of header: value segments and employ a novel entropy-driven contrastive alignment. This mechanism aligns low-entropy (domain-coherent) columns to ensure consistency, while separating high-entropy (entity-specific) columns to maintain fidelity. Experiments demonstrate that NAVI significantly outperforms strong baselines across a range of downstream tasks.

### Strengths
1.	The proposed "Fidelity" and "Consistency" framework provides a highly useful and insightful lens for evaluating and designing table representation learning models.
2.	The concept of the Header-Value Segment is simple yet effective. The Entropy-driven Alignment is a brilliant idea that ingeniously connects statistical properties to semantic objectives.
3.	The experimental setup is sound, the evaluation is multi-faceted, and the results are significant. The ablation studies and qualitative analyses are highly persuasive.

### Weaknesses
1.	Comparison with Graph Neural Network (GNN) Approaches: A brief mention and comparison with GNN-based methods in the Related Work section could make the literature review more comprehensive.
2.	Scalability: For wide tables with a very large number of columns, the input sequence can become excessively long. It would be beneficial to discuss the model's potential bottlenecks with such tables and possible solutions.
3.	Entropy estimation based on empirical distributions might be unstable for columns with long-tail distributions or sparse data. I suggest the authors briefly discuss this potential limitation.
4.	A significant limitation of the NAVI framework lies in its handling of numerical data, a critical weakness given that numerical values are arguably the most prevalent and foundational data type in real-world tables. The entropy-driven mechanism will likely misclassify numerical columns (e.g., price, quantity) as high-entropy, "entity-discriminative" attributes due to their high cardinality. Consequently, the contrastive learning objective pushes their representations apart, ignoring the inherent ordinal and metric semantics between values (e.g., the model fails to learn that '10' is semantically closer to '11' than to '100'). This fundamentally undermines the model's ability to perform numerical reasoning, severely restricting its applicability for tasks like regression or range-based queries and confining its value to a minority of use cases dominated by categorical and textual data.

### Questions
My main question, which is central to my evaluation, concerns the treatment of numerical columns. The entropy-driven alignment mechanism appears to classify numerical columns (e.g., price, age, measurements) as high-entropy, thereby treating them as entity-discriminative. The contrastive objective would then push the representations of different numerical values (e.g., "10.5" and "10.6") apart, just as it would for distinct movie titles. This approach seems to neglect the crucial ordinal and metric relationships inherent in numerical data.
Could you clarify if the current NAVI framework has any mechanism to preserve these numerical semantics?
If not, how do you see this impacting the model's utility for common, numerically-grounded tasks like regression or range-based queries? Could you elaborate on how the framework might be extended to incorporate a type-aware objective that respects the unique properties of numerical values?

### Soundness
2

### Presentation
2

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
This paper addresses the trade-off between fidelity (preserving specific schema-value semantics) and consistency (robustness to schema variations) in transformer-based models for in-domain table representation. The authors propose NAVI (Entropy-aware Alignment via Header-Value Induction), a framework that introduces the "header-value segment" as the atomic unit of table representation. NAVI employs three core mechanisms: (1) Schema-aware Segment Induction (SSI), which uses a global, context-free header encoder to anchor segment semantics; (2) Masked Segment Modeling (MSM), which enforces schema-value dependencies through balanced masking of header and value tokens; and (3) Entropy-driven Segment Alignment (ESA), a novel contrastive learning objective that categorizes columns by value entropy. ESA aligns low-entropy (domain-coherent) columns with their global header embeddings to promote consistency, while aligning high-entropy (entity-discriminative) columns with their row-specific value embeddings to preserve fidelity. Extensive experiments show that NAVI outperforms baselines on generative (imputation) and discriminative (classification, clustering) tasks, successfully balancing the two desiderata.

### Strengths
S1. The paper provides a valuable conceptual contribution by formalizing the core challenge of in-domain table representation as a trade-off between "fidelity" and "consistency"—a key unsolved problem for creating generalizable tabular deep models. The paper further breaks this down into structural and domain-specific components, providing a clear and principled lens for evaluating and developing models in this space.

S2. The core mechanism, Entropy-driven Segment Alignment (ESA), is a novel and highly intuitive solution to the fidelity-consistency dilemma. Using column value entropy to dynamically determine the contrastive learning target (a stable global header for domain concepts vs. a specific local value for discriminative entities) is a clever and effective method for explicitly balancing these two competing objectives within a single model.

S3. The experimental evaluation is comprehensive and well-aligned with the paper's conceptual framework. The use of the Permutation Sensitivity Index (PSI) as a direct measure of structural consistency is particularly effective, and the reported near-zero PSI for NAVI is an impressive result.

### Weaknesses
W1. The Entropy-driven Segment Alignment (ESA) mechanism relies on the InfoNCE loss, which inherently assumes that for any query, there is only one positive sample and all other samples are true negatives. This assumption is frequently violated in real-world tabular data. For the domain consistency loss ($L_{dom}$), correlated columns or synonyms (e.g., `director` and `auteur`) are all treated as distinct negative samples, creating a "false negative" problem. While the Global Header Encoder is intended to mitigate this by mapping synonyms to close embeddings, this creates a conflicting objective with the InfoNCE loss, which is forced to push them apart. Similarly, for the entity fidelity loss ($L_{ent}$), two different rows that share the same value (e.g., two different products with the color `red`) would be incorrectly treated as negative pairs. The paper does not analyze the impact of this "false negative" discrepancy, which could degrade the quality of the learned embedding space.

W2. The paper's positioning against prior work, particularly "consistency-oriented" models like HAETAE, could be stronger. HAETAE also utilizes a context-free header anchoring mechanism, and the paper's claim that it suffers from "header-value misalignment" is asserted rather than deeply investigated. A more direct comparison of how NAVI's segment-based induction and alignment mechanistically differs from and improves upon HAETAE's header-anchoring would strengthen the paper's novelty claim.

W3. The experimental evaluation is missing a helpful baseline comparison. While the paper compares NAVI's embeddings against other transformer-based embeddings (e.g., BERT, TAPAS), it does not include a comparison against a traditional model like XGBoost trained directly on the raw, pre-processed features. NAVI's primary strength is handling schema variation, which GBDTs cannot. However, including a baseline on a "clean" version of the dataset would be valuable to quantify the performance gap that still exists between complex neural models and top-tier GBDTs on standard classification tasks. This would help position the work in the broader context of the 'NNs vs. GBDTs' debate for tabular data.

**Comments**

C1. The paper's core concepts of "fidelity" and "consistency" are introduced in the motivation, but their explanation remains somewhat abstract. The accompanying Figures 1 and 2, which are intended to visually clarify these concepts and their trade-offs, are dense and difficult to interpret, making it challenging to build a concrete intuition for the problem before the methodology is presented.

C2. The paper should clarify the exact mechanism for obtaining $H_{ctx}$ and $V_{ctx}$. It is stated that they are "extracted by pooling the contextualized token embeddings," but the connection to the initial $z_j^k$ (input) and the final $e_t$ (output) is implicit. An example would be very helpful: for the segment "director: danny", are the "header" tokens just director or do they include the :? This precise operational detail is important for understanding the model's architecture. Additionally, in Figure 4, it should be made clearer which plot corresponds to BERT, as the small titles are easy to miss.

### Questions
Q1. Regarding the Entropy-driven Segment Alignment, the categorization is based on quartiles (Q1 and Q3), which seems to create a "dead zone" for all medium-entropy columns between Q1 and Q3. These columns apparently do not contribute to the $\mathcal{L}_{align}$ loss at all. What is the ratio of columns that fall into this dead zone, and what is the theoretical or empirical impact of ignoring them during alignment? Does this not risk creating a representation where domain-coherent and entity-discriminative columns are well-structured, but the "average" columns are left in a poorly structured part of the embedding space?

Q2. The framework's reliance on a context-free global header encoder for domain consistency is a key contribution. How does this mechanism handle headers that were not seen during pretraining (i.e., OOV headers, synonyms, or typos)? Does the model's consistency and fidelity degrade gracefully? A robustness evaluation against OOV headers seems essential for a method that so heavily relies on them for anchoring domain semantics.

Q3. In the $L_{msm}$ objective function (Section 2.2), the denominator of the softmax is written as $\sum_{v\in V}exp(We_{v}+b)$. The variable $e_v$ is not defined, whereas the numerator uses $e_t$ (the contextualized output token). Is $e_v$ a typo and intended to be something else, for instance, a non-contextualized embedding for each word $v$ in the vocabulary $V$? Please clarify the exact formulation of this loss function.

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
4

### Summary
This paper proposes NAVI, a model that treats each *header–value segment* (header:value) as the atomic unit for table representation. NAVI integrates (a) a global header encoder, (b) Structure-aware Masked Segment Modeling (SMSM) for balanced masking of headers, values, and tokens, and (c) Entropy-driven Segment Alignment (ESA) for contrastive routing between low-entropy header-centric and high-entropy row-centric representations. The goal is to achieve fidelity (schema–value preservation and row distinctiveness) and consistency (robustness to schema, lexical, and structural variation). The paper provides theoretical analyses, extensive ablations, and evaluations on two WDC WebTables domains (Movie and Product).

### Strengths
- Well-motivated segment-level design. Treating (header:value) pairs as set elements with local positional encoding nicely combines permutation invariance and context awareness. The connection to DeepSets provides theoretical clarity.
- Entropy-based contrastive routing. The distinction between low- and high-entropy columns for header vs. row alignment is intuitive and empirically supported. The alignment/uniformity analysis is a good step toward geometric justification.
- Comprehensive evaluation axes. The paper evaluates discriminative (classification, clustering), generative (header prediction, value imputation), and invariance-based (PSI, header clustering) tasks, providing a holistic empirical picture.

### Weaknesses
- Limited domain diversity and possible bias.
   Experiments are restricted to two domains (Movies, Products) with the largest 100 tables per domain. This selection favors clean, high-quality schemas. The model’s robustness to smaller or noisier tables, numeric-heavy domains (e.g., finance), and unseen domains is unclear. Cross-domain and noisy-table experiments are needed.

- Under-specified training regimen.
   All models are trained for only 2 epochs with batch size 32 on datasets up to 3.9M rows. Such limited training may hinder convergence, confounding architectural effects with optimization noise. The paper should report learning curves, seed variance, and matched compute comparisons for baselines.

- Missing negative sampling details.
   InfoNCE-based contrastive results are sensitive to negative sample quality. Clarify negative sampling strategy (same/different table, same domain, batch size, memory bank use) and ablate negative set size.

- No comparison to classical tabular methods.
   The paper reports XGBoost only on top of embeddings. End-to-end baselines (e.g., XGBoost on raw features, TabPFN, TabNet) are missing, making it difficult to gauge absolute improvements.

### Questions
- Are header encoder parameters updated during training? Please include an ablation comparing frozen, partially fine-tuned, and lightweight alternatives.
- How sensitive is entropy-based routing to threshold settings? Compare fixed, percentile, and soft routing schemes.
- Could the author(s) ablate negative set size and temperature parameters (τ_dom, τ_ent) to analyze their effect on alignment/imputation?
- How does NAVI perform on numeric-heavy domains? Compare against numeric-specialized models (e.g., TP-BERTa, TabPFN).
- Can author(s) demonstrate cross-domain transfer (e.g. Product→Movie, Movie→Product) to substantiate domain-invariant anchor learning?
- How robust is NAVI under schema noise (renaming, typos, column swaps)?


I would consider raising my score if the authors can adequately address these questions.

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
This paper proposes NAVI: Entropy-aware Alignment via Header–Value Induction. NAVI captures the structural properties of tables through schema-aware segment induction and modeling. In addition, NAVI employs entropy-driven alignment of segments to selectively incorporate domain knowledge shared among in-domain tables. Through various experiments, the paper shows effectiveness of NAVI on various downstream tasks.

### Strengths
In general, the paper is easy to follow and the paper provides theoretical grounds on constructing the proposed method.

### Weaknesses
-	The figures does not help understand the proposed method. 
-	In figure 1, the readers cannot see what they are supposed understand. Also, clear explanation with concrete examples for distinctiveness (fidelity) and robustness (consistency) is required.
-	In figure 2, the paper states there are trade-offs, but it is really hard to visualize what the trade-offs are.
-	It would be great to have explanations of the concepts with the examples shown in the figures.
-	As the paper addresses the importance of table representation for downstream tasks, it would be interesting to see how NAVI compares to simple heuristics for encoding tables (eg., Tablevectorizer or TextEncoder in Skrub package) combined with tabular learning methods such as TabPFN, XGB, and LR.

### Questions
-	What characterizes the distinctiveness (fidelity) and robustness (consistency)? What are some concrete examples?
-	How does NAVI deal with numerical values?
-	Would there be more datasets to compare the performance of NAVI?
-	What is the ground for choosing Bert-style model? Could NAVI benefit from a more sophisticated architecture?

### Soundness
3

### Presentation
2

### Contribution
2
