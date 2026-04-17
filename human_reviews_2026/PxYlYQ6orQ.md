# Nested Text Labelling Structures to Organize Knowledge in AI Applications for the Humanities and Social Sciences

- Decision: Reject
- Scores: 2, 0, 4

## Abstract
Scientific literature has emerged to advance annotation frameworks incorporating multi-fragment and multi-assessor labelling protocols alongside contextual data. The application of such rigorously defined, expert-driven text annotations provides a foundation for developing machine learning models capable of performing automatic text markup. The paper aims to identify the knowledge representations suitable for both human annotators and machine learning processes, as well as various task types. Experience gained through a number of applied projects and research studies has shown that the answer is not that simple.
We propose a multi-level approach to the data models used for text annotation. Given its applicability for tasks involving context, multi-assessor labelling, and the extraction of subjective textual categories, this paper delineates its conceptual and logical foundations, alongside the associated cases. The proposed framework comprises three nested data models, each distinguished by its level of complexity. The relational representation of textual annotations is flexible enough for a variety of annotation scenarios. It supports named entity recognition, relation extraction, semantic analysis, co-reference resolution, frame semantics, multi-span matching, etc. - at least 17 types of tasks whose inputs and outputs have fundamentally different structural complexities. The framework includes a core model, an extended set of entities, and their relations. The same dataset can be related to various tasks of significantly different types. The broad applicability of our framework is supported by the survey of 21 datasets and related tasks found in more than a thousand publications. The proposed methodology extends the scope of structured text annotation, advances the standardisation of content analysis procedures, and facilitates solutions for a broader spectrum of natural language processing tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
- The paper introduces a unified data model/schema for text annotations 
- The schema consists of three levels 
    - Span Level: Highlight text fragments and tag them (eg. NER, POS, etc) 
    - Element Level: Group multiple related spans into elements that represent relationships or structures (more complex tasks like Relation Extraction (RE), Coreference Resolution, and Frame Extraction) 
    - Extended level: Add metadata like who annotated it, comments explaining decisions, and context notes on spans/elements (eg. context-dependent annotation such as human values dataset) 

- Validation of the schema  
    - The authors validate their framework's usefulness and broad applicability by demonstrating it can successfully model and categorize a wide range of existing, real-world annotation tasks and datasets. 
    - 21 diverse datasets (including CONLL 2012, ADE, and RuSentNE) were analyzed by reviewing "over a thousand" associated publications found via the Semantic Scholar API. 
    - Used LLM to extract what tasks each paper addresses (Identified 17 distinct task types) 
    - Map all these combinations of datasets and tasks to their proposed 3-level hierarchy (Table 1 of the paper)

### Strengths
- The paper's main contribution is a new hierarchical framework to standardize text annotation (more details in the summary above)  
- The paper claims to solve the "fundamental trade-off" that is an existing  problem text annotation acorss disciplines. The nested levels allow an expert to choose the simplest model possible for their task, while still providing the power to handle complex nuances (like multi-span elements or contextual comments) when needed. 
- The paper provides complete formal specifications: ER diagrams, relational schemas, entity definitions which also handles edge cases such as multi-annotator, comments, context, etc.

### Weaknesses
- The paper claims that the current models fail to capture "nuances" but doesn’t really discuss what specific nuances are lost? They just assert this without proof. And the entire premise of the paper is based on existence of the problem, without a sufficient evidence that the problem exits 
- There are claims that the schema is an attempt to solve "convenience vs expressiveness" trade-off but the paper never measures or provide evdience for how would it lead to better convenience (eg. user studies) or expressiveness (eg. ML experiments). In short there is no validation that the current schema would be useful. 
- The mapping of 21 datasets to the schema, shows that datasets can be mapped to the schema. The pipelines used LLM-as-judge to annotate the type of tasks in the paper, however, there is no validation that the LLM-as-judge actually works or any robustness checks, making the pipeline and subsequent results unreliable.

### Questions
- I am unsure about the who exactly does the schema help. Is it for the annotators to better understand the LLM generated annotations or it for better annotations using LLMs?   
- Why is a unified schema better (or more precisely why is having 3 levels better than having 1 flexible schema)? Different tasks might genuinely need different representations. Forcing everything into one framework could make things worse.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes a framework for performing knowledge management extracted from the textual information via annotation models. These can be extracted with the help of the tasks of entity and relation extraction.

### Strengths
Since the motivation and research questions are not described, it is hard to see the strengths of the paper.

### Weaknesses
--> There is no mention of humanities and social sciences in the rest of the paper other than the title.
--> The introduction is missing references to support the claims.
--> The exact research questions that the authors want to target are missing.
--> There is no related work.
--> After the generalized logical or relational model that is created with the NER and Relation extraction tasks, how is that useful in any of the downstream tasks?
--> The paper is missing the evaluation of the proposed knowledge management framework.
--> What is the broad impact of this work?

The paper is still in a premature stage and the contributions are not very clear.

### Questions
--> What are the research questions that authors are targeting?
--> After the generalized logical or relational model that is created with the NER and Relation extraction tasks, how is that useful in any of the downstream tasks?
--> How this framework can be evaluated?
--> What is the broad impact of this work?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a hierarchical data model with three levels of increasing complexity. The motivation is that existing text annotation models often fail to capture fine-grained expert knowledge, and there is a need to balance convenience for annotators with the expressive power of the annotation representation. The authors develop a semi-automated literature review pipeline by querying the Semantic Scholar API and extracting relevant task descriptions using a large language model. Through this process, they analyze 21 datasets and identify 17 distinct types of text-labeling tasks. Based on these findings, they design a nested text-labeling framework that organizes these tasks according to their structural complexity. The authors highlight the adaptability of the framework to datasets with varying levels of complexity, as well as its practical convenience for human annotators. Finally, they demonstrate how the framework can be implemented as relational database structures and propose a corresponding relational schema.

### Strengths
1. Clear motivation for using a hierarchical model that introduces additional complexity only when necessary
2. Good coverage of identified annotation task types across multiple datasets.
3. Potential for unifying datasets and standardizing annotation practices
4. Helpful explanation of functional dependencies and how the proposed framework maps to relational database schemas

### Weaknesses
1. **Missing references:** I am afraid but there are very few references provided. For example:
   - Lines 43–44: "In the humanities and social sciences, recent studies suggest that current text annotation models often fail to encapsulate the full nuance of expert knowledge, thereby limiting their utility for advanced AI applications." 
     This is your main motivation for proposing new data models, but no sources are cited to support this claim.
   - Line 223: "The data model described in this subsection is a traditional one and can be found in many different datasets."
     Please cite which datasets and sources.
   - Lines 80–81: "Existing methods proved inadequate, primarily due to limitations in scalability and coverage."
     Which methods?

2. **Missing related work section:** You do not have a related work section. You do position the contributions in the context of prior literature. What is the closest related work? How do other approaches model annotation structures? You rarely mention related work and do not contrast your work with existing models.

3. **Clearly state contributions:** I suggest including a dedicated paragraph that explicitly lists the paper’s contributions. For example, is the semi-automated literature review pipeline intended to be a main contribution as well?

4. **Significance of contributions:** Beyond the general motivation for improved annotation models, several contribution claims need to be moderated or substantiated:
   1. Lines 65–68 "The employment of the proposed model hierarchy during development, coupled with training on diverse task types, is posited to yield a new generation of machine learning models. These models would be characterised by greater usage simplicity and a broader range of application." This is a very strong statement. For a your proposed work of better organising annotation structures this claim seems a bit far-fetched.
   2. You emphasize convenience for annotators, but no empirical evaluation of annotation usability or efficiency is provided. When motivating your work with that and highlighting this in the conclusion section, please ensure you can substantiate that.
   3. You state that the structure “facilitates the selection of an appropriate AI model for any specific dataset” (lines 142–143). This may be true in principle, but for ICLR, this contribution is relatively limited unless comparative evidence is provided showing measurable improvements over existing data models.

5. **Validation:** The framework is derived from 17 identified tasks. You then conclude (lines 318–319) that the framework is widely applicable. To support this claim, I would argue to validate the system on tasks not used in the design process.

6. **Limitations:** You do not clearly articulate the paper's limitations. Which types of tasks or annotation scenarios are not well supported by the framework, and why? I would expect a dedicated section or paragrpah listing limitations.

7. **Semi-automated pipeline:** The pipeline uses an LLM, but the specific model is not identified. How do you ensure correct extraction? Was the output manually verified? This is particularly important since large models are known to struggle with “needle in the haystack” retrieval tasks (see: https://research.trychroma.com/context-rot).

8. **Vague**: Lines 18-19: "Experience gained through a number of applied projects and research studies has shown that there cannot be a single simple answer; [...]" That is very vague. Are these your own or do you draw this from the literature.

9. **Typos**: 
	- "Analisys" in Figure 1
	- "can label o and more" in line 295

### Questions
- Extended Level is only covered by three publications. This seems quite low compared to the other levels?
- Line 470: "A collection of tasks [...]" this refers to the 17 distinct tasks?
- Where exactly do you refer to the AI applications in the Humanities and Social Sciences in the paper?

### Soundness
2

### Presentation
2

### Contribution
2
