# Probing the Boundaries of Concepts in Language Models

- Decision: Reject
- Scores: 8, 2, 4, 4

## Abstract
Systematic investigation of the understanding of scientific concepts has not received much attention in language models. This gap can be bridged by a formalized theory of conceptual semantics that maps naturally to instruction templates for natural language agents. We propose a simple framework expressible in first-order logic to address the semantic compositionality of scientific concepts, noun phrases and conceptual hierarchies. The framework is used to derive a conceptual integrity benchmark with 6 tasks that are applied to a selection of 187 concepts from the domains of biology, chemistry and medicine. The performance of 15 state-of-the art language models is evaluated relative to baseline information collected from various knowledge repositories. We see a strong positive correlation between model size and performance. External validity of the benchmark is demonstrated by a high correlation with other benchmarks that measure related skills. It is suggested that the proposed framework and associated benchmark provide a practical template for developing conceptual integrity benchmarks in a wide array of technical or scientific domains.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors investigate concept understanding in some scientific domains. Information about a concept is scattered across documents, and the paper asks whether an LLM can synthesize across sources to form a complete model of a concept. They define a framework for what constitutes a concept, given by its label, definition/selection criteria, and referents. They also construct a benchmark on concept understanding by selecting several scientific concepts and defining a series of related tasks.

### Strengths
I think that the contribution of the benchmark and framework are valuable and timely. I especially see this work as useful toward the goal of evaluating the use of LLMs in producing novel scientific research. 

The definition of concepts based on label/definition/referents is interesting and novel (or at least, I have not heard of this idea). The definitions are formalized with some basic equivalence and compositionality properties. 

While the benchmark the authors have constructed is limited in scope, their framework and task definitions can be used to construct similar benchmarks for other domains. The tasks in the benchmark have a direct relationship to the theoretical framework, while also having obvious concrete applications to users of LLMs. 

The scope of the tasks is precisely defined, so a cross-comparison of performance between tasks is a meaningful test of ‘conceptual integrity’, as the authors put it.

### Weaknesses
I somewhat disagree with the presentation of this as a general investigation of conceptual understanding. I see it as a benchmark of scientific concept understanding (or even more specifically understanding of concepts in drug discovery). In my opinion, the understanding of general, everyday concepts is a separate line of inquiry that is not addressed here. 

I think conceptual understanding is a very broad and active line of work, with several threads that the authors do not mention. For example, I am thinking of work in modelling concepts via their representations/representation geometry, e.g.:

Luyten and van der Schaar. "A theoretical design of concept sets: improving the predictability of concept bottleneck models."  (NeurIPS 24)

Park et al., "The Geometry of Categorical and Hierarchical Concepts in Large Language Models" (ICLR 25)

Li et al.,  "The Geometry of Concepts: Sparse Autoencoder Feature Structure " 

I’m not at all implying that the results should be more general – I would just suggest that the scope of the paper and its relation to prior work on concept understanding be defined more precisely. 


While most of the paper was easy to follow, I had a few points of confusion. The last paragraph of Section 4 was difficult to understand - I am not sure if you are suggesting that there is some fundamental issue with categorizing biological concepts with hierarchies, or if there is just an error in the source data.

I also did not really understand why the semantic field size task was included - it seems to me from the discussion that the ground truth used to evaluate the responses was very inaccurate. In this case, what does the task tell us?

### Questions
How were the referents chosen? I assume based on the fact that some concepts have 100k+ referents that there is an automatic method. 

What is the meaning of point 5 on page 2, semantic compositionality on proposition level? It is mentioned as an aspirational goal, and I’m curious what you see as the gap/future improvement to the framework. 

Have you compared with performance on datasets restricted to biology/chemistry/medicine domains (e.g. in MMLU or GPQA the questions are labelled with the subject)? I would suspect that the correlation would be higher this way.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The current paper seeks to understand to what extent language models acquire conceptual integrity — the ability to learn the relationship between concept labels, definitions and referents. The authors set out to establish a metric that captures conceptual integrity based on a collection of tasks  — inferring concepts from definitions/ontologies, inferring concepts from the criteria (features) that act as ‘selection criteria’ for that concept, naming n referents of the concept, distinguishing valid members from non-members, and estimating the ‘semantic field size’, i.e., the cardinality of the number of concepts that fall into the conceptual bucket. The authors find variance across a suite of models evaluated on this metric and also find that this metric correlates with some existing LLM benchmarks but not others. They also provide an error analysis of select failure modes.

While the authors emphasize that the main contribution is a framework for understanding conceptual coherence, I find that this core claim is hard to accept for several reasons. In contrast to many domains where there is a structured ground truth in the world, ‘concepts’ are seemingly meaningful insofar as they align with human notions of the same. The current analyses have no human data at all. All model outputs are evaluated using LLM-as-a-judge, which in this particular case feels theoretically muddy, and there is no concrete sense of how these measures might look for human participants. In fact, there is a large literature in the cognitive sciences attempting to do what the authors are attempting here at larger scales in an ecologically valid (using concepts people think about in their day-to-day) manner. I note some examples below - 

McRae, K., Cree, G. S., Seidenberg, M. S., & McNorgan, C. (2005). Semantic feature production norms for a large set of living and nonliving things. Behavior research methods, 37(4), 547-559.

Hansen, H., & Hebart, M. N. (2022). Semantic features of object concepts generated with GPT-3. arXiv preprint arXiv:2202.03753.

Suresh, S., Mukherjee, K., Giallanza, T., Yu, X., Patil, M., Cohen, J. D., & Rogers, T. T. (2025). AI-enhanced semantic feature norms for 786 concepts. arXiv preprint arXiv:2505.10718.


Generally, these works have laboriously tried to collect human data to ground human and LLM semantic reasoning. I like the general approach taken by the authors here, but fail to see the current results as being theoretically interesting. This is compounded by a lack of serious discussion of what the current results mean. Since this benchmark seems to correspond closely with things like MMLU, maybe we should simply just use MMLU instead? Without theoretical grounding in human cognition and reliable ground truth, I find it difficult to justify why one would care about this benchmark over others. 

A firmly concrete suggestion would be to simply replicate the LLM experiments with people (though the specific domains chosen here might make that tricky) and to use that as a reference for model evaluation.

### Strengths
* Introduction of a novel framework for understanding concepts in LLMs (and potentially humans), which could be useful if deployed in stringent ways.
* Diverse set of tasks in evaluation suite
* Evaluation of a good mix of frontier language model systems
* Thoughtful error analysis on select responses.

### Weaknesses
* Insufficient engagement with the core cognitive science literature on related issues
* Lack of reliable ground truth to evaluate model responses
* Lack of theoretical consequences of performing well or poorly on this benchmark and how it distinguishes itself from existing benchmarks, whose scores are highly correlated.

### Questions
* Looseness in the use of concept. The Framework in Section 2 leads me to believe a concept is a definition but in many other contexts within this paper and in cognitive science, a single term (word) is often used to refer to a concept. I think the authors can get ahead of this by laying out definitions early on in the paper.
* Why this domain for concept understanding? I think its fine that the particular concepts are somewhat arbitrary but there needs to be theoretical motivations for studying something as general as concept learning/representation in the domain of something as highly specific as chemistry, biology, and medicine. 
* Why is SFS listed as its own column in Table 1? I assumed the SFS task was done within each domain (Bio/Med/Chem)? And since its a separate column here why is it not one for Table 2?
* I don’t think (non-perfect) correlations should be noted as indications of practical equivalence, re: `Very high correlation (0.87) between model ranks from the two ’decide-concepts’ tasks indicates the practical equivalence between definitions and corresponding selection criteria when naming concepts.`
* I had trouble parsing this sentence - `We suggest that conceptual integrity is a fundamental phenomenon that can be distinguished from and is subsumed by logical and factual reasoning.` If it can be distinguished from these other factors how is it also subsumed by them?

### Soundness
2

### Presentation
2

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
This paper assesses the conceptual knowledge in LLMs. Specifically, this paper proposes a simple framework expressible in first-order logic to address the semantic compositionality of concepts, noun phrases, and conceptual hierarchies. The authors construct a benchmark for this framework, which contains 6 tasks, 187 concepts from several domains, including biology, chemistry, and medicine. The authors conduct extensive experiments and evaluate 15 state-of-the-art LLMs. Experimental results reveal several interesting findings, such as a strong positive correlation between model size and performance.

### Strengths
1. The topic focused on in this paper is meaningful and interesting. Evaluating whether language models truly understand conceptual knowledge is fundamental to assessing their understanding of world knowledge and their abstract reasoning abilities. It provides valuable insights for the community’s broader understanding of LLMs.
2. The proposed evaluation framework based on first-order logic is novel and conceptually appealing. It offers a concise and formalized mapping from natural language to logical representations, which could inspire future evaluation methods and potentially be extended to other domains or tasks.
3. The experimental results may yield some interesting findings, such as the observed correlation between model size and conceptual understanding, which is also intuitive.
4. The presentation is clear and easy to follow.

### Weaknesses
1. The paper should include a comparison with previous studies on concept knowledge in LLMs [1, 2]. The authors should discuss how the proposed first-order logic–based approach differs from and improves upon existing methods. Clarifying the unique advantages and benefits of this framework would strengthen the contribution.
2. As acknowledged in the limitations, the study only uses 187 concepts and does not include concepts from more general domains. This limited scope may affect the validity and generalizability of the experimental conclusions. It would be beneficial for the authors to include a larger and more diverse set of concepts, at least covering more domains.
3. The paper provides extensive case studies, which are valuable for intuitively illustrating the model outputs. However, I think there needs a more quantitative analysis, such as identifying which types of concepts LLMs perform better on and exploring potential common patterns. The quantitative analysis would make the findings more insightful.

[1] Peng, Hao, et al. "Copen: Probing conceptual knowledge in pre-trained language models." *arXiv preprint arXiv:2211.04079* (2022).
[2] Wu, Weiqi, et al. "Do PLMs know and understand ontological knowledge?." *arXiv preprint arXiv:2309.05936* (2023).

### Questions
See Weaknesses.

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
3

### Summary
This paper introduces a logical framework of conceptual semantics that formalizes the relationship between concept labels, definitions, and referents. Based on this framework, the authors design a conceptual integrity benchmark comprising six tasks that test semantic mappings between these components. They evaluate 15 LLMs using 187 concepts from biology, chemistry, and medicine, finding that conceptual integrity correlates strongly with model size and with general benchmarks like MMLU. They also identify systematic failure modes (in annotation, scoring, and model reasoning) and discuss implications for ontology completeness and conceptual understanding.

### Strengths
* The formalization of conceptual integrity using first-order logic and the link to instruction-tuned benchmarks is original and timely.

* The benchmark spans multiple domains and models, with thoughtful correlation analysis to external metrics (MMLU, GPQA, etc.).

* The discussion on failure modes (especially semantic nuance and ontology incompleteness) is detailed and empirically motivated.

### Weaknesses
* Only 187 concepts across three domains, with hand-curated annotations. This limits generalizability.

* Scoring relies on incomplete and potentially inconsistent ontologies, leading to lower-bound estimates.

* Some benchmark tasks (JSON formatting, instruction following) test general compliance rather than conceptual understanding per se.

* The framework defines what to measure but doesn’t explain how models represent or manipulate conceptual structure.

### Questions
* Could the benchmark be expanded beyond biomedical/chemical domains to assess transferability?

* How might conceptual integrity relate to known phenomena like compositional generalization or systematicity?

* Could fine-tuning on ontology-style data improve conceptual integrity scores?

* How do annotation or scoring inconsistencies affect the reliability of correlations with other benchmarks?

### Soundness
3

### Presentation
3

### Contribution
3
