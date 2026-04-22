# Facts in Stats: Impacts of Pretraining Diversity on Language Model Generalization

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Language models are pretrained on sequences that blend statistical regularities (structures making text fluent) with factual associations between specific tokens (corresponding to world knowledge). While recent work suggests that the variability of their interaction, such as paraphrases of factual associations, critically determines generalization ability, we lack a systematic analysis of these multifaceted impacts. This paper introduces a flexible synthetic testbed that combines a statistical stream of generic tokens with an abstract factual stream of source-target token pairs, enabling fine-grained control over their interaction. Specifically, the design enables the independent control of diversity nature by manipulating stream composition (contextual structure) and the level of diversity by varying which statistical streams each fact appears in. Through controlled experiments, we find that while higher contextual diversity delays in-distribution (ID) factual accuracy, its effect on out-of-distribution (OOD) generalization depends critically on contextual structure. In some cases, OOD performance follows the same trend as ID, but in others, diversity becomes essential for non-trivial factual learning. Even when low diversity prohibits factual recall, optimal diversity levels depend on training duration. Beyond factual recall failures, we identify structures where statistical generalization fails independently, and others where both capabilities collapse simultaneously. This demonstrates how the interplay between contextual design and diversity level impacts different aspects of generalization. Through detailed mechanistic analysis of transformer components, we find that learned embeddings are key to successful generalization under high-diversity data. Overall, our synthetic framework allows us to isolate effects that would be confounded in large-scale studies, thus offering a controlled testbed for future investigations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates how language models learn and distinguish factual knowledge from statistical patterns during pretraining. It investigates how the diversity of contexts in which a fact appears (i.e., paraphrasing) impacts both factual recall and statistical generalization. The paper's primary contribution is the introduction of a synthetic testbed that decouples the training data into a "statistical stream" (templates with controllable statistical) and a "factual stream" (atomic source-target pairs). This framework allows for fine-grained, independent control over the level of diversity (how many different templates a fact is seen in) and the contextual structure (how the templates themselves vary), which the authors argue was a key limitation in previous, less-controlled studies.

Using this testbed, the paper investigates the relationship between diversity and generalization. For example, while high diversity (seeing a fact in many different templates) slows down in-distribution (ID) factual learning, it is often crucial for out-of-distribution (OOD) generalization. A mechanistic analysis identifies that learned embeddings and the unembedding layer are the primary components enabling factual generalization under high diversity, while attention mechanisms are key for statistical learning. This work moves beyond prior synthetic studies, which often used fixed templates and focused solely on factual recall, by being the first to systematically model and analyze the interplay and trade-offs between statistical learning and factual acquisition.

### Strengths
The paper's primary strength is its experimental design. By cleanly decoupling statistical and factual streams, it offers fine-grained control over diversity and contextual structure — a significant advance over prior work. This design compellingly allows the authors to move beyond just factual recall and be one of the first to systematically investigate the interplay and trade-offs between statistical generalization and factual acquisition.
What's more, the paper provides a strong mechanistic analysis to explain why these trade-offs occur, identifying distinct optimization bottlenecks for each learning type . The fact that this minimal, reproducible testbed can still capture these phenomena (like stage-wise learning) seen in larger-scale studies is a plus.

### Weaknesses
My main concerns with this paper relate to its translation to realistic LLM training. My biggest question is about the distinct, non-overlapping vocabularies for statistical and factual tokens. This design removes the real-world ambiguity where a single token must participate in both roles, forcing the model to rely on context. [cite_start]I wonder if this simplification artificially affects the learning dynamics, perhaps exacerbating the observed importance of the (un)embedding layers in separating these tokens, rather than testing the internal processing required in a more realistic, mixed-vocabulary setting.

A secondary concern is the deliberately small model size. While I appreciate the authors' open acknowledgment and justification for this (tractability and reproducibility), and their checks on 1- and 10-layer models, it does leave open the question of how these specific bottlenecks might shift in the larger architectures.

### Questions
Do I understand correctly that the models are trained single-epoch? Whenever a statistical template is used, the actual tokens are sampled anew; and only the fact tokens are repeatedly found at their respective positions?

What is effective token frequency for fact vs. statistical tokens / how often do they appear throughout the whole training set? It appears with so many fact tokens and only few statistical tokes, there might be a significant imbalance?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studied the learning and generalization of "statistical structure" (different Markov chain structures) vs "facts" (particular tokens that ) in small transformers, using a synthetic setting designed to emulate those aspects of natural language pretraining data.. They swept across many parameters controlling data diversity (and also freezing various parts of the network after training in different regimes), and investigated the effects on learning and generalization of the "statistics" and "facts", and learning dynamics.

### Strengths
* very nice coverage of the parameter space. it can be difficult to study the effects of parameters on training in an exhaustive way, which they did well
* nice breakdown of the three types of learning: "positions" vs "statistics" vs "facts"
* interesting results, esp how diversity doesn't monolithically affect the different types of generalization

### Weaknesses
* limited scale 
* Can you motivate more why it seems necessary to study the learning of facts and linguistic structure simultaneously? Why not e.g. facts vs reasoning, or facts vs learning larger scale statistical structure. Or each of them alone. Unclear why this is an important question
* I find the experimental setup very interesting. However, can you provide more precise motivation for why the Markov chain structure mirrors the statistical structure of natural language, while the two positions for "fact insertion" are a good representation of "facts"? Theoretically and/or empirically

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The author investigate how linguistic diversity influences factual and statistical learning in language models. They design a controllable synthetic testbed separating a statistical stream and a factual stream, allowing independent control of context structure and diversity level. They that higher diversity delays factual recall within distribution but is necessary for robust out-of-distribution generalization, while low diversity can cause failures in both factual and statistical learning.

### Strengths
- To study this, the authors design a controllable synthetic framework that separates statistical structures from factual associations, enabling the study of their interaction during training.
- They systematically examine how diversity level and contextual structure affect in-distribution.

### Weaknesses
- I think the setting is a bit overly simplified, Markov templates and atomic facts miss real linguistic hierarchy and semantic interference---this could limit the external validity.
- The uniform sampling of diversity overlooks long-tailed frequency patterns in real corpora, this could possibly misrepresent the true diversity effects.

### Questions
What diversity level is actually needed to improve generalization? Does the model learns true fact template separation or just memorizes within templates?

### Soundness
3

### Presentation
2

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
This paper investigates the impact of pretraining data diversity on language model generalization. The authors introduce a novel synthetic testbed that combines statistical streams (templates) with factual streams (source target pairs). This setup allows fine grained control over the diversity level, meaning how many different templates a fact appears in , and the contextual structure, meaning how templates vary statistically or positionally. Key findings show that while high diversity can slow in distribution fact learning , it is critical for out of distribution generalization. The specific effects depend heavily on the contextual structure.

### Strengths
1. The synthetic testbed is a primary strength. It is cleverly designed to isolate and control the interactions between statistical learning and factual memorization.
2. The paper provides a nuanced analysis of diversity. The finding that high diversity slows ID convergence but is essential for OOD generalization is a valuable insight.
3. The discovery of a temporal trade off where optimal diversity levels depend on the training duration is an interesting and practical finding.

### Weaknesses
1. The primary limitation is the simplicity of the synthetic data. While tractability is a goal , the first order Markov chain used for the statistical stream  is a very simplified model of language. It is unclear how these findings translate to the complex, long range dependencies of natural text.
2. The experiments are conducted on very small transformer models. Although the authors show results on 1 to 10 layer models , it remains an open question if these specific mechanisms and bottlenecks are the same in SotA models with hundreds of billions of parameters.
3. The paper's findings are dense. While the figures are informative, the interplay between the three contextual structures (MC10Pos1, MC1Pos10, MC10Pos10) and the three metrics  can be difficult to follow.

### Questions
1. The paper convincingly argues that low diversity creates a non generalizing minimizer . Could this be overcome with optimization changes, such as a different optimizer, learning rate schedule, or regularization, or is it a fundamental property of the low diversity data landscape?
2. How do you hypothesize these findings would change if the statistical stream was more complex than a first order Markov chain? For example, if it included simple grammatical structures, would the sharp distinction between statistical and factual learning bottlenecks  still hold?

### Soundness
3

### Presentation
2

### Contribution
2
