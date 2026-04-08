## Human Reviewer 1

### Summary
This paper proposes to study a novel concept of answer-set consistency in large language models (LLMs), which refers to whether an LLM's responses to related enumeration questions respect expected set-theoretic relations such as equality, containment, and disjointness. The authors introduced a new benchmark with 600 handcrafted quadruples of logically related questions and used it to evaluate 18 contemporary LLMs under several prompting strategies. As evaluating, explaining, and addressing answer-set inconsistency in LLMs' question-answering capability has immense practical value, the paper is quite valuable in this regard.

### Strengths
1. **Novelty:** The paper makes a novel contribution by formalizing the concept of answer-set inconsistency for enumeration questions. This establishes a new, well-defined dimension for evaluating LLM's reliability based on set-theoretic relations. 

2. **Thorogh empirical analysis**: I appreciate the paper's comprehensive and systematic empirical analysis, which helps understand the nature of answer-set inconsistencies across a wide variety of LLMs.

### Weaknesses
1. **Objectivity of some questions in the benchmark**: Some of the questions in the dataset are not objective. 
For instance, “On what video streaming services can I watch the Hunter x Hunter anime series?”-it depends on the region or country. 

2. **Incompatiblity of Problem formulation and benchmark**: In section 2.1, the authors assume E to be "the universe of entities from an arbitrary domain" from where both questions and answers originate.  This formulation implicitly assumes an open-world or extensible entity space -- that is, E could, in principle, include all possible entities that exist within a domain, even those not observed in the dataset. Under this assumption, answer-set consistency is defined in an idealized, domain-agnostic logical sense. But, in practice, the ASCB benchmark is constructed from knowledge graphs (KGs) such as Wikidata and DBpedia, which operate under a closed-world assumption. Meaning,
the benchmark enumerations are derived only from entities present in the KG; Missing facts or unobserved entities are treated as false rather than unknown; Consequently, the "universe"  E is finite and closed to the KG’s vocabulary. This means that while the paper’s definitions rely on the notion of “true set relations within an arbitrary domain,” the evaluations measure consistency only relative to KG-encoded truth -- not the broader, theoretically open universe.

3. **Lacking discussion on Synthetic**: Please discuss in detail the construction process of SYNTHETIC, in particular, what the prompts were, and how you came up with the questions in the first place.

4. **Deeper understanding of the results is missing:** Table 3, ~R4 column => Why CtE and Oracle are underperforming (lower score) compared to Base in many models? You identified this issue in Appendix E, but are there some inherent limitations to your mitigation strategies on disjoint set queries?  We also see this problem on GPT5 ~R5, CtE. In addition, we also see that on %R5, CtE strategy makes GPT-4.1-nano, and GPT-5-nano both perform worse. What are the reasons behind this?

### Questions
See W2 and W4.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
5

---

## Human Reviewer 2

### Summary
In this paper, the authors introduce answer-set inconsistency, in which large language models (LLMs) can produce contradictory answers to enumeration questions that should obey set-theoretic relations.
In this paper, the authors have focused on the set relationship such as equality, containment, and disjointness between question-answer sets.
To evaluate it, they constructed the Answer-Set Consistency Benchmark (ASCB) containing 600 handcrafted quadruples (a total of 2,400 questions) derived from knowledge-graph QA datasets.
Three approaches are defined 1) Base, 2) Classify-then-Enumerate (CtE), and 3) Oracle and these approaches are used to assess both relation classification and consistency of generated answers.
Experiments on 18 modern LLMs, including GPT-5, Gemini-2.5, and Llama-3, reveal pervasive inconsistency even under low temperature (greedy sampling) settings.
Containment and multi-set relations are found to be the most difficult, while equivalence and disjointness are easier.
The CtE strategy, which prompts the model to reason about relations before answering, significantly improves consistency across all models.

### Strengths
S1. The paper clearly formalizes an important concept, LLM answer-set consistency. This concept is quite interesting and has potential to provide a principled framework for LLM to be more logical coherent.

S2. The empirical analysis across 18 modern models and the 2400 questions dataset are valuable contributions to understand LLM behaviours.

### Weaknesses
W1. The proposed mitigation strategies, such as Classify-then-Enumerate, are relatively straightforward and simple. It would be great if the authors can provide more fundamental algorithmic improvements or mitigation strategy. 

W2. While the paper identifies sources of inconsistency, it provides limited theoretical analysis or deeper causal explanation beyond empirical observation.

W3. The symbols in this paper feels a bit overloaded. $Q_i$, $RQ_i$, $R_i$. Maybe create a symbol table for line 280-285 such that the readers can find the meaning of the different relationship easier. 

Comment: The author may find the following papers related to set/list semantic equivalence to be interesting: https://arxiv.org/abs/2312.10321 and https://arxiv.org/abs/2502.12466.

### Questions
Please see W1-W3.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper is concerned with answer-set consistency of LLMs during question answering. When asked an *enumerative* question, the LLMs answers with a set. LLMs may contradict themselves when enumerating all entities satisfying the question. To this end, the authors consider consistency evaluation from a set-theoretic perspective, by checking equivalence, disjointness, containment of set operations, etc.


The paper proposes a benchmark, and reveal inconsistencies of existing LLMs on answer-set setting.

### Strengths
The setup is well motivated, since LLMs may be inconsistent when asked to enumerate all entities in a set-focused question. The related work is explained well, with prior studies' definitions of inconsistencies. Empirical results are extensive, covering many models, and showing that they are not yet fully consistent during question answering. Several parts of the paper can be improved, as explained below.

### Weaknesses
- Section 3 requires rewriting. Mathematical notations could be simplified (e.g., table 2 should be as simple as possible). Some notations do not look standard, or not explained well. Since the authors consider a finite number of set operators, it is better to use in a *case by case* what it means to achieve consistency under equivalence, containment, disjointness or overlap. The notations for disjointness and overlap are mixed with Boolean True or False -- I have never seen these notations before. In summary, the consistency criteria should be explained in an intuitive manner, with illustrative examples.

- There is no discussion on what prompts are given to the LLM and how answers are parsed -- they have major influence on the evaluation of inconsistency.

- The mitigation strategy has a *conversational* aspect, which may improve consistency in a surface level, i.e., in-context, and not to the parametric knowledge of the LLM. What happens when the LLM does not know the enumerative answer?

- The construction of the dataset is explained in a complicated manner. Why there are quadruples in line 202, and onwards?

- Several sentences are repeated multiple times, but demonstrative examples are missing. Examples are line 182 to 185. Put differently, since the objective is to improve consistency, it is better to show examples of consistency/inconsistency without being too abstract in notations (refer to Ghosh et at (2025) for inspiration).

### Questions
Please address the points mentioned in the weakness.

I am willing to increase score after rebuttal.

### Soundness
3

### Presentation
1

### Contribution
2

### Rating
2

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper formalizes the problem of answer-set inconsistency in large language models (LLMs) for factual enumeration questions,
where LLMs generate responses that violate set-theoretic relations (e.g., equivalence, containment, disjoint-ness) between questions. The paper proposes a benchmark dataset (ASCB) with 600 handcrafted question quadruples (2,400 questions) and evaluate 18 state-of-the-art LLMs. Key contributions include a novel theoretical framework, comprehensive metrics, and mitigation strategies that significantly improve consistency. Experiments demonstrate pervasive inconsistency across models, even when they correctly recognize relations, with mitigation
strategies achieving statistical significance (p < 0.001).

### Strengths
S1. The paper introduces a rigorous theoretical foundation for answer-set consistency, formalizing it using set-theoretic relations
(Section 3.1). This includes definitions for consistency and contradictions, enabling a sampling-based evaluation approach
with error guarantees (e.g., control relation R* for stochasticity analysis in Section 3.4). The framework bridges LLM behavior with
database-like query containment principles, advancing beyond prior work focused on single-answer consistency.

S2. The empirical analysis is extensive, covering 18 LLMs (e.g., DeepSeek, Grok, Mistral, GPT, Gemini, Llama families) across
multiple tasks (Base evaluation, CtE, Oracle). Evaluation metrics (Consistency rates, Jaccard similarity) and statistical testing
(McNemar test) reveal pervasive inconsistency.

S3. The mitigation strategies (e.g., CtE) of inconsistency are practical and only require minimal prompt engineering, making
them accessible for real-world applications.

### Weaknesses
W1. The proposed ASCB dataset is limited to 600 quadruples (2,400 questions) in English, focusing on static, factual domains.
This scale may not generalize to dynamic environments or high stakes applications requiring larger scales datasets. The manual
curation, while ensuring quality, restricts diversity and scalability.

W2. Based on KGQA datasets, the experiments are conducted in isolated, single-turn contexts, ignoring real-world scenarios like
multi-turn dialogues. This limits the validity of consistency claims for interactive systems, where temporal dynamics might
exacerbate inconsistencies.

W3. While the paper highlights stochasticity as a cause of inconsistency in Section 3.4 and 4.2, it lacks accurate error reason analysis (e.g., entity-level semantic misunderstanding or knowledge gaps). For example, contradictions in Appendix F (Table 7) are not decomposed into specific error types, lacking of guidance to extend datasets and hindering targeted model improvements based on prompt strategy.

W4. The paper describes different relations (including primary and implied relations) in Table 2. However, there is an unclear
explanation about modeling 12 pairs of questions in Section 3.2 and even why paper chooses five primary relations (R1-R5) in
Section 3.3.

### Questions
Limitations in dataset diversity, dynamic scenario evaluation, and error analysis reduce its generalizability. See the weaknesses (W1-W4) above.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
3