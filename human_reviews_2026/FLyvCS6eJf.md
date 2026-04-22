# AN ONTOLOGY ENRICHMENT FRAMEWORK USING RETRIEVAL-AUGMENTED LARGE LANGUAGE MODELS

- Avg Score: 1.00
- Decision: Reject
- Scores: 0, 2, 2, 0

## Abstract
Ontology enrichment, understood as the process of extending and refining existing ontologies with new concepts, relations, and instances, has become a critical task for building robust and up-to-date knowledge bases. The exponential growth of scientific publications, datasets, and multimodal resources makes manual enrichment highly impractical, creating the need for automated or semi-automated approaches. In this work, we propose a framework that leverages multimodal large language models and retrieval-augmented generation to support ontology enrichment. Our method systematically extracts semantic knowledge units, aligns them with existing ontological structures, and generates interlinked triples, thereby enhancing both the coverage and the expressivity of the ontology. This framework addresses the knowledge acquisition bottleneck by enabling scalable integration of heterogeneous resources and fostering cross-domain semantic interoperability. To illustrate its effectiveness, we apply the framework to the domain of 4D printing, a rapidly evolving field at the intersection of materials science, manufacturing, and design. By incorporating knowledge about materials, properties, stimuli interactions, process parameters, and design strategies, the framework enriches a domain-specific ontology and supports innovation in the development of programmable and multifunctional structures. The proposed framework follows a four-stage pipeline that combines multimodal retrieval of relevant text and figures from scientific literature with the ingestion of structured datasets and existing knowledge graphs, uses a fine-tuned multimodal LLM to extract ontology-aligned triplets, applies multi-criteria validation based on semantic relevance and consistency, and finally performs ontology population through symbolic reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper employs LLM and RAG to populate an ontology from a given set of keywords describing a domain, 4D prinitng in this case.

### Strengths
S1. The paper is easy to read.

### Weaknesses
W1. No technical contribution. The paper is basically a technical report describing a standard implementation of a LLM-based ontology population approach.

W2. Missing important technical details. For example, how does your symbolic reasoner work? How did you describe and cataloge a column? How did you extract object properties to link individuals? What are your relationship mapping strategies?How did you normalize and clean entity names?

W3. Lack of evaluation. How did you evaluate the quality of the generated ontology? How would you compare your approach with existing methods?

### Questions
Q1. What is the technical contribution of this paper?

Q2. See the questions in W2 and W3.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a framework for automatically enriching ontologies using multimodal large language models combined with retrieval-augmented generation. The authors extract semantic knowledge from scientific papers, datasets, and knowledge graphs to expand existing ontologies. They demonstrate their approach on the 4D printing domain, growing the HERMES ontology from 170 classes to 5849 classes and over 12.5 million instances.

### Strengths
1. The paper addresses an interesting problem - automatically updating knowledge bases.
2. Combining text and figures from papers makes sense for materials science, where diagrams and microscope images often contain information you can't get from text alone.

### Weaknesses
1. The paper evaluates the proposed framework in isolation without comparing against existing ontology learning methods. The paper cites Phrase2Onto, OntoGPT, and other LLM-based ontology extension approaches but provides no quantitative comparison. Without baselines, it is impossible to assess whether the performance (Graph BERTScore F1 = 0.7) represents an improvement over simpler alternatives or state-of-the-art methods.
2. The proposed framework consists of multiple components RAG retrieval, LLaVA fine-tuning, multimodal fusion, and multi-stage validation. However, the paper provides no ablation study showing the contribution of each component. The appendix only compares "fine-tuning alone" versus "fine-tuning + one-shot," which doesn't isolate the contributions of RAG, multimodal inputs, or individual validation criteria.
3. The paper claims the framework is "fully domain-agnostic", yet validation is performed exclusively on 4D printing. This contradicts the generalization claim and is insufficient to demonstrate transferability. 
4. The paper relies on a single metric (Graph BERTScore F1 = 0.7) mentioned briefly in the main text with details relegated to the appendix. This is insufficient for validating the core contribution.
5. Essential technical details are missing: the size of the fine-tuning dataset for LLaVA, threshold values for domain relevance, semantic coherence, and similarity-based filtering, and computational requirements for the system.
6. All figures are labeled "adapted from (Bougzime et al., 2025b)", a paper that is also under review. The paper should clarify what is novel here versus the cited work and whether this represents appropriate academic practice.
7. No code, no data availability statement, insufficient implementation details.

### Questions
1. Can you provide quantitative comparisons with at least 2-3 existing ontology learning methods (Phrase2Onto, OntoGPT, etc.) on standardized test sets?
2. Can you demonstrate generalization by applying the framework to at least two additional domains beyond 4D printing?
3. What's the size of your fine-tuning dataset?
4. What's the computational cost in GPU hours?
5. Can you provide ablation studies showing the contribution of RAG, multimodal inputs, and each validation criterion separately?
6. What are the most common errors? Can you show examples of false positives and false negatives with analysis?
7. Can you provide systematic ablation studies demonstrating the individual contribution of each framework component?
8. What proportion of initially extracted triplets are filtered at each validation stage? Can you provide concrete examples of hallucinated assertions that were successfully detected and removed by your validation pipeline?
9. What constitutes the core technical or scientific contribution beyond systems integration? Can you justify why this integrated pipeline yields superior results compared to direct prompting of frontier models like GPT-4 without the additional complexity?

### Soundness
2

### Presentation
2

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
The paper presents an automated framework for ontology enrichment that combines multimodal large-language-models with retrieval-augmented generation and symbolic reasoning. The pipeline ingests scholarly text and figures, retrieves relevant content, extracts candidate triples, filters and aligns them to a target ontology, then integrates results into the ontology store. The case study on a 4D-printing ontology reports large growth in classes, properties and instances, and explores metric-based validation and sensitivity to temperature and fine-tuning.

### Strengths
1. Coherent end-to-end framework that unifies retrieval, multimodal extraction and symbolic checks for ontology population.
2. Clear pipeline description with stages for ingestion, retrieval, extraction, validation, alignment and integration, which improves readability and reproducibility potential.
3. Multimodal design leverages figures alongside text, which is valuable for technical domains with schematics and tables.
4. Practical scale demonstrated: expansion from about 170 seed classes to thousands of classes and many millions of instances, showing the approach can run at scale.
5. Reasoning-based consistency checks provide structure and constraint validation beyond pure language-model extraction.
6. Initial parameter study comparing fine-tuned versus one-shot prompting and temperature settings shows attention to stability.

### Weaknesses
1. Evaluation is not adequate for the core claims. There is no expert-curated gold set and no triple-level precision/recall or ontology-level error taxonomy. Automatic similarity scores alone do not establish correctness.
2. No competitive baselines are included against established ontology enrichment tools or strong text-only extractors plus alignment. This leaves the value of multimodality, retrieval and fine-tuning unquantified.
3. Missing ablations for each component. The paper claims value from retrieval, images and fine-tuning, but does not quantify each component’s marginal contribution.
4. Domain generality is asserted but not shown. Results are limited to one domain. 
5. Reported scale for instances and classes lacks quality controls. No sampling audits, curator checks or constraint violation summaries are provided, so growth may include duplicates or noise.
6. The role of images is not sufficiently analyzed. The share of triples that need visual evidence and which relation types benefit remain unclear.
7. Reproducibility gaps. No released code, prompts, model checkpoints or ontology dumps, and limited reporting on compute cost and runtime make it hard to verify claims or adopt the method.
8. Novelty is partly diluted by overlap with prior lines of work and by visible self-referencing patterns that may risk double-blind compliance.
9. Metric choices are weakly justified. The mapping from graph-similarity or n-gram style metrics to human-judged accuracy is not established, and there are no confidence intervals.

### Questions
1. What is the measured gain from retrieval, images and fine-tuning relative to a strong text-only baseline? Please report triple-level precision, recall and F1 on an expert gold set for each configuration.
2. Did domain experts audit a stratified sample of triples, classes and relations? Please share sample size, agreement and an error taxonomy.
3. What fraction of correct triples requires visual evidence? Which relation types benefit most? Please provide examples where images change the decision.
4. Could you add baselines against competitive ontology-engineering pipelines and text-only triple extractors with ontology matching? Iw would like to see a report on extraction quality and end-to-end coherence.
5. Could you provide evidence to substantiate the domain-agnostic claims?
6. For the reported scale, please provide sampling-based precision estimates, duplicate rates and constraint violation counts and show distributional diagnostics for classes and properties.
7. Could you complement similarity-based metrics with triple precision/recall, relation-typing accuracy, constraint violation counts? The study would also benefit from a downstream task such as SPARQL-query success or retrieval over the enriched ontology.
8. Do you even intend to release code, prompts and at least a redacted subset of the enriched ontology? Which document model versions, licences and compute were used?
9. Could you provide one concrete downstream use-case that improves due to enrichment and quantify that improvement?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes a framework on enriching ontologies using LLMs with a use-case on 4D printing.

### Strengths
- the ontology enrichment framework based on textual and multimodal information from research articles which seems interesting.

### Weaknesses
- The details about the framework could be given in the abstract to strengthen the interest of the reader.
- There is a whole introduction on LLMs in the second paragraph which is not necessary. 
- Paragraph 3 talks about interpretability, are you focusing on interpretability.
- The authors talk about fine-tuning on page 4 which doesn't need an introduction here, there could be a separate section on this.
- Page 5 could use an example prompt from the dataset.
- Why exactly the authors chose the 4D printing as a scenario, if the authors mean to generalize the framework for any use-case, 2-3 related use-cases could be introduced.
- There is no evaluation of the generated ontology or the downstream task on the use of the ontology generated.
- The paper is quite verbose at times with unnecessary details. 
- Many acronyms were repeatedly introduced since LLM was used for grammar correction possibly.

### Questions
- Why exactly the authors chose the 4D printing as a scenario, if the authors mean to generalize the framework for any use-case, 2-3 related use-cases could be introduced.
- See other comments in the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
1
