# EMERGE: A Benchmark for Updating Knowledge Graphs with Emerging Textual Knowledge

- Decision: Reject
- Scores: 2, 6, 0

## Abstract
Knowledge Graphs (KGs) are structured knowledge repositories containing entities and relations between them. In this paper, we study the problem of automatically updating KGs over time in response to evolving knowledge in unstructured textual sources. Addressing this problem requires identifying a wide range of update operations based on the state of an existing KG at a given time and the information extracted from text. This contrasts with traditional information extraction pipelines, which extract knowledge from text independently of the current state of a KG. To address this challenge, we propose a method for construction of a dataset consisting of Wikidata KG snapshots over time and Wikipedia passages paired with the corresponding edit operations that they induce in a particular KG snapshot. We obtain these pairs by aligning annotated hyperlinked entity mentions in each Wikipedia passage with the corresponding entities involved in the updated Wikidata triples. We verify, using LLMs with human validation, that these textual passages contain the knowledge needed to support the associated KG edits. The resulting dataset comprises 233K Wikipedia passages aligned with a total of 1.45 million KG edits over 7 different yearly snapshots of Wikidata from 2019 to 2025. Our experimental results highlight key challenges in updating KG snapshots based on emerging textual knowledge, particularly in integrating knowledge expressed in text with the existing KG structure. These findings position the dataset as a valuable benchmark for future research. We will publicly release our dataset and model implementations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces EMERGE, a large-scale benchmark designed for studying how knowledge graphs can be automatically updated with new information from unstructured text sources such as Wikipedia. Unlike most static datasets, EMERGE aligns evolving Wikipedia passages with temporal Wikidata snapshots, producing over 233K passages and 1.45 million text-driven KG update operations. These operations include adding, modifying, and deprecating triples to reflect newly emerging knowledge.

### Strengths
1.	Comprehensive related works are mentioned to position it within the broader research landscape including knowledge graph completion, refinement, and information extraction.

2.	The experimental settings and evaluation details are provided, allowing for clearly understanding of the resource.

3.	The paper is generally clearly-written and easy to follow.

### Weaknesses
1.	The practical utility of the resource remains unclear. More concrete use cases, applications, or empirical demonstrations are needed to justify its real-world usefulness.

2.	The work focuses solely on factual updates (new or deprecated triples) while not considering schema-level or structural modifications, such as ontology changes, entity merges, or property reorganizations.

3.	The dataset is limited to entities presented in Wikidata while omitting literal values (e.g., numerical or date attributes), constraining its applicability to many factual domains.

4.	The benchmark evaluation only reports recall/completeness without precision or F1, which reduces the comprehensiveness of performance assessment under open-world assumptions.

5.	The dataset assumes synchronized temporal windows between textual and KG updates, however in practice, Wikipedia and Wikidata could diverge significantly in update timing.

### Questions
1.	How can this resource be effectively leveraged by other research communities or real-world applications? In what ways does it offer measurable advantages over existing datasets?

2.	What is the unique benefit of EMERGE compared to directly utilizing sequential Wikidata snapshots for temporal KG analysis?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the problem of how to correctly map newly emerging textual facts to the required updates of a knowledge graph (KG) at a specific point in time. The authors formalize the Text-driven Knowledge Graph Updating (TKGU) task and define five update operation types: X-Triples, E-Triples, EE-Triples, EE-KG-Triples, and D-Triples. They further construct EMERGE, a large-scale and incrementally extensible benchmark aligned between Wikipedia text and weekly deltas of Wikidata, spanning 2019–2025 with 233K instances and ~1.45M KG update operations. Using two families of advanced information extraction baselines, they conduct systematic evaluation across multiple snapshots and temporal deltas, showing that existing IE/LLM methods struggle with connecting emerging entities back to the KG and with fact revocation operations. In addition, performance degradation increases with larger deltas, indicating that EMERGE reveals realistic capability gaps in dynamic KG maintenance.

### Strengths
S1: The paper not only proposes the direction of “text-driven KG updating” but also formalizes the task into five concrete TKGU operations (Section 3). With clear examples (Figure 1), the classification criteria are explicit and allow direct validation and quantification, going beyond generic triple extraction.

S2: EMERGE covers seven yearly snapshots (2019–2025) and up to five-week cumulative deltas per snapshot, totaling 233K instances and ~1.45M updates, with significant growth in KG entities and relations over time. This provides a solid foundation for studying robustness to temporal evolution and schema drift.

S3: The benchmark construction leverages Wikipedia/Wikidata historical dumps and includes weekly snapshot/delta generation, text–KG alignment, and data curation pipelines. Cleaning rules and rollback filtering strategies are documented, supporting extensibility and reproducibility.

S4: The evaluation is detailed across different models, snapshots, and deltas using completeness/recall metrics (Table 2, Figure 3, Appendix). Qualitative failure examples (Appendix E) help identify future research directions (e.g., the need for models to exploit KG structure rather than text alone).

### Weaknesses
W1: Although Meta-Llama-3.1 is used to filter alignment pairs efficiently, details of the LLM filtering process—thresholds, prompt designs, and rules for marking updates as “unsupported”—are insufficiently provided, affecting interpretability and reproducibility of the dataset.

W2: D-Triples constitute only 0.6% of the full dataset (3.3% in the subsampled test set), and their correctness often relies on KG-level reasoning. The imbalance and uncertainty make the evaluation unstable, and the paper does not demonstrate mitigation strategies for potential misjudgment or skewed metrics.

W3: EE-KG-Triples rely on external KG knowledge or schema assumptions (e.g., typing a new entity as Human). Many such links are not textually supported, yet the paper does not provide a clear evaluation protocol or manual validation rate to confirm their accuracy.

W4: Closed-IE ReLiK models and open-generation EDC+ differ significantly in the amount of KG information provided (entity/relation dictionaries vs. prompt-based access). The comparison risks conflating model capabilities with differences in provided prior knowledge.

W5: The definitions of EE-KG and D-Triples are introduced without intuitive examples in the main paper, leaving readers dependent on the Appendix for clarity.

W6: Section 4 (dataset construction) is disproportionately long, while Section 5 (analysis) offers limited error type exploration and insufficient discussion of challenge sources.

### Questions
Q1: How are the non-textually supported EE-KG links retained (fully Wikidata-driven?), and can the authors provide manual evaluation results (accuracy, common error types) for a small EE-KG sample?

Q2: The authors state that code/data will be released after acceptance, but even in the anonymized version, please provide: data format specification (JSON schema), alignment output examples.

Q3: Could the authors clarify whether D-Triples rely strictly on explicit deletion operations or also semantic overrides? Please also provide Cohen’s κ / F1 from human annotation if available.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes a pipeline as well as a benchmark dataset for the updates in the knowledge graphs.

### Strengths
--> The pipeline for automated updates from text could be a useful tool to integrate for the live updates of the major general purpose knowledge graphs such as Wikidata or DBpedia.

### Weaknesses
- In the abstract the authors could give a bit of the introduction of how is the framework implemented.
- The interchangeable use of the terms knowledge graphs and knowledge bases is quite well known, it does not need to be explicitly said in footnote 2.
- There are many other datasets available for KG completion such as LiterallyWikidata [1] or Codex [2], etc.
- The authors are discussing the SoTA on KG Completion but this task is not discussed later on or the framework is not evaluated on that task. Why are the authors relating this task to that family of algorithms?
- Why is the SoTA on information extraction is discussed since this is also not the main focus, the main focus is on how to handle updates in the knowledge graph.
- the authors should also discuss the methods for ontology versioning, etc. in the SoTA and make a clear distinction [3].
- The authors should keep in mind that Wikidata is also crowdsourced on top of containing the information from Wikipedia.
- The part on "Emerging entities to KG triples..." discusses about "instanceOf" relation which is a different problem, i.e., entity typing and adding new instances to an entity.
- The example of deprecated triple seems misleading. If a person stops being a member of a club then there should be added an end date because the fact doesn't change it just expires overtime.

[1] https://link.springer.com/chapter/10.1007/978-3-030-88361-4_30 
[2] https://arxiv.org/pdf/2009.07810 
[3] https://arxiv.org/pdf/2409.04572

### Questions
- The authors are discussing the SoTA on KG Completion but this task is not discussed later on or the framework is not evaluated on that task. Why are the authors relating this task to that family of algorithms?
- Why is the SoTA on information extraction is discussed since this is also not the main focus, the main focus is on how to handle updates in the knowledge graph? It should talk about Entity Linking algorithms and the existing dataset which suffer from the updates in the KG.
- Overall, the problem statement of the paper is a bit confusing. The pipeline is for updates in the knowledge graph with a dataset which enriches the Knowledge Graph with updated information but the evaluation seems to be on entity linking which is another problem linked with updated knowledge graphs and the text on which the entity linking is performed.
- the paper might be a better fit for the evaluations on dynamic knowledge graph completion.

### Soundness
1

### Presentation
1

### Contribution
1
