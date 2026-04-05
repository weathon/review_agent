# ANSWER SET CONSISTENCY OF LLMS FOR QUESTION ANSWERING


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Large Language Models (LLMs) sometimes contradict themselves when answering factual questions, especially when asked to enumerate all entities that satisfy
the question. We formalize such self-contradiction as answer-set inconsistency:
Given two enumeration questions whose answers satisfy a set-theoretic relation
(equivalence, disjointness, containment, etc.), the LLM generates responses violating the relation. To diagnose this phenomenon, we create a benchmark dataset
comprising tuples of enumeration questions over which a variety of set-theoretic
relations hold, and propose related metrics to quantify answer-set inconsistency.
Our evaluation of several state-of-the-art LLMs reveals pervasive inconsistency
across models, even in cases where the LLM can identify the correct relation.
This leads us to further analyze potential causes and propose mitigation strategies
wherein the LLM is prompted to reason about such relations before answering,
which leads to improved answer-set consistency. This work thus provides both a
benchmark and a systematic approach for evaluating, explaining, and addressing
answer-set inconsistency in LLM question answering, towards deriving practical
insights to improve the reliability of LLMs.


1 INTRODUCTION


Large Language Models (LLMs) have demonstrated impressive capabilities not only in natural language understanding and generation, but also in question answering and other tasks involving complex reasoning (Tan et al., 2023; Ding et al., 2024; Li et al., 2024; Saxena et al., 2024; Sui et al.,
2024). However, in the context of the latter tasks, they are prone to various types of contradiction (Ghosh et al., 2025; Calanzone et al., 2025) since they are not based on computational techniques that guarantee formal notions of consistency, soundness, etc.


One type of inconsistency that LLMs exhibit relates to factual question answering. Specifically,
in the context of _enumeration_ _questions_, which ask to list all entities that satisfy a question, the
responses across different questions may exhibit inconsistency with respect to evident set-theoretic
relations that hold between such questions. Take, for example, the four questions in Table 1. All
such questions are enumeration questions: they expect a set of entities as answers. Let [[ _Q_ ]] denote
the expected set of answers for a question _Q_ . Among these four questions, we can see that certain
set-theoretic relations should be expected to hold, including _equality_ ([[ _Q_ 1]] = [[ _Q_ 2]] = [[ _Q_ 3]] [[ _Q_ 4]]),
_∪_
_containment_ ([[ _Q_ 3]] [[ _Q_ 1]] _,_ [[ _Q_ 4]] [[ _Q_ 1]] _,_ [[ _Q_ 3]] [[ _Q_ 2]] _,_ [[ _Q_ 4]] [[ _Q_ 2]], and such relations entailed
_⊆_ _⊆_ _⊆_ _⊆_
by equality), _disjointness_ ([[ _Q_ 3]] [[ _Q_ 4]] = ), etc. However, the answers returned by a particular
_∩_ _∅_
model may not satisfy these relations, even in cases where the model can recognize the correct
expected relation. Specifically, let [[ _Q_ ]] _M_ denote the set of answers enumerated by model _M_ . Then,
for example, given _Q_ 1 and _Q_ 3, when asked what relation holds between their answers, _M_ may
correctly recognize the containment of the latter in the former ([[ _Q_ 3]] [[ _Q_ 1]]), but still enumerate
_⊆_

[[ _Q_ 1]] _M_ and [[ _Q_ 3]] _M_ such that [[ _Q_ 3]] _M_ [[ _Q_ 1]] _M_, thus contradicting itself.
_̸⊆_

We formalize this issue as _answer-set consistency_, wherein _the answers for a tuple of factual enu-_
_meration_ _questions_ _generated_ _by_ _a_ _particular_ _model_ _do_ _satisfy_ _the_ _set-theoretic_ _relations_ _that_ _are_
_expected to hold for that tuple_ . We further consider an _answer-set contradiction_ whereby _the model_
_enumerates answers that do not satisfy the set-theoretic relation it itself predicts_ . Related topics have
been well-studied in database theory literature wherein the notions of _query_ _containment_, _equiva-_
_lence_, etc., are textbook topics (Abiteboul et al., 1995), with decades of theoretical and practical re

1


Table 1: Illustrative example with four enumeration questions forming different relationships (equivalence, containment, disjointness, etc.) with respect to the expected answer sets.


_Q_ 1 What are the tributaries of the Madeira River?
_Q_ 2 Which rivers and streams flow directly into the Madeira River?
_Q_ 3 What are the right-bank tributaries of the Madeira River?
_Q_ 4 What are the left-bank tributaries of the Madeira River?


sults covering a variety of query languages, data models, and reasoning formalisms. Unlike database
systems, generative A.I. models are not designed to guarantee satisfaction of such formal relations
in the answers they provide. Perhaps for this reason, answer-set (in)consistency has not been wellstudied in the context of generative A.I. To the best of our knowledge, the closest work is that of
Elazar et al. (2021), which evaluates consistency across paraphrased versions of cloze-style phrases,
but these permit only a single answer. Hogan et al. (2025) briefly discuss this issue as “ _coherence_ ”,
but do not investigate it further. Given that such models are increasingly used to answer users’
enumeration questions, we believe the topic merits more analysis.


**Research questions** In this paper, we address the following research questions (RQs):


**RQ1** To what extent do LLMs produce (in)consistent answer sets for enumeration questions?


**RQ2** Can LLMs recognize the set-theoretic relations that exist between enumeration questions?


**RQ3** Which set-theoretic relations cause the most difficulty for LLMs?


**RQ4** Which key factors cause answer-set inconsistency in LLMs?


**RQ5** Can we mitigate answer-set inconsistency with prompting strategies?


**Contributions.** This paper describes four main contributions: (1) We highlight and formalize the
notion of answer-set consistency for enumeration questions. (2) To quantify answer-set consistency,
we develop and release a novel handcrafted benchmark of 600 question quadruples, with 2,400 questions in total, where fixed relations are expected to hold between the questions of each tuple. We
further propose measures to quantify answer-set consistency with respect to such a dataset. (3) We
present an empirical analysis of 18 state-of-the-art LLMs on the benchmark. (4) We present preliminary prompting strategies to mitigate answer-set inconsistency and evaluate their effectiveness.


Our methodology allows for a systematic analysis of how model size, architecture, and prompting
strategy affect the logical coherence of LLM outputs, providing insights into both the strengths
and limitations of current models for generating consistent answers to enumeration questions. Our
findings reveal that LLMs exhibit significant answer-set inconsistency, the extent of which depends
on the particular model and the particular relation (interestingly, newer or bigger models do not
universally outperform older or smaller variants). Such inconsistency often occurs even when the
model is able to correctly recognize the relation expected to hold. The prompting strategies we
propose to mitigate this issue significantly improve consistency across all tested models.


The code, datasets, and results of experiments are available on (anonymous) GitHub (authors, 2025).


2 RELATED WORK


**Question answering datasets** A wide range of datasets have been proposed for different flavors
of question answering tasks (e.g., Tan et al. (2023); Wang et al. (2023); Lee & Kim (2024); Singhal
et al. (2024); Wang et al. (2024); Zheng et al. (2024); Zhou & Duan (2024); Zhu et al. (2024);
Allemang & Sequeda (2025); Ma et al. (2025)). These datasets and evaluation frameworks focus on
accuracy with respect to a predefined ground truth. While such benchmarks are essential to verify
the correctness and completeness of answers, our focus is rather on the internal consistency of the
models, which we see as complementary to these existing datasets.


**Consistency of LLMs** Various works have addressed different notions of consistency for LLMs.


2


Ghosh et al. (2025) studies the logical consistency of LLMs in a boolean fact-checking setting. They
define the logical consistency of an LLM in terms of the boolean response _LLM_ ( _p_ ) for some propositional statement _p_ representing a fact. Specifically, they test _negation_ ( _LLM_ ( _¬p_ ) = _¬LLM_ ( _p_ ),
i.e., the response of the LLM to a negated fact should be the negation of the response to the
original fact), _disjunction_ ( _LLM_ ( _p_ _∨_ _q_ ) = _LLM_ ( _p_ ) _∨_ _LLM_ ( _q_ ), i.e., the response to a conjunction of facts is the same as the conjunction of the responses to each fact), and _conjunction_
( _LLM_ ( _p ∧_ _q_ ) = _LLM_ ( _p_ ) _∧_ _LLM_ ( _q_ ), as before). The authors further construct complex statements using the combinations of these three boolean operators _{¬, ∨, ∧}_ . Their results showed that
consistency improved with an increase in the number of model parameters.


Similarly, Calanzone et al. (2025) investigated whether a fine-tuning approach that integrates neurosymbolic reasoning can enforce logical consistency in language models. The authors include negation consistency and also whether the LLM can follow implication rules. Empirical results demonstrate that the approach outperforms conventional fine-tuning and external solver-based methods.


While these works address statements in propositional logic, Liu et al. (2024c) investigates logical
consistency in LLMs as a property of the relationships between sets of items, rather than isolated predictions. Instead of evaluating responses in isolation, the model is queried across sets of comparisons
to assess whether its preferences form a coherent structure. They define consistency through properties like transitivity, order-invariance, and semantic negation, applied across sets of judgments. Their
experiments reveal that popular LLMs frequently violate these properties, even on simple domains,
and that improving consistency correlates with better downstream task performance.


Jang et al. (2022) propose a behavioral definition of consistency, categorizing it into semantic, logical, and factual types. They introduce BECEL, a benchmark designed to evaluate logical consistency
through controlled input changes across several NLP tasks.


Addressing factual questions and statements with a single response, Elazar et al. (2021) evaluates
the consistency of (L)LMs across paraphrased versions of cloze-style phrases (i.e., facts with entities
masked, such as “ _is_ _the_ _largest_ _tributary_ _of_ _the_ _Madeira_ _River_ ”). The authors found notable
inconsistency in the models tested at that time. Cohen et al. (2023) propose having one LLM crossexamine another LLM, asking it diverse follow-up questions on the facts it states – for example, by
rephrasing the fact and posing it as a question, asking about logical implications of the stated fact,
etc. - checking if the response of the latter model remains consistent under cross-examination.


Zhao et al. (2023) investigated whether LLMs can determine semantic equivalence between SQL
queries, while Wei et al. (2025) introduced a benchmark to evaluate LLMs’ reasoning about program
semantics through equivalence checking. These studies, though related, are limited to equivalence
relations, without addressing more general or asymmetric semantic relations.


Although the consistency of LLMs with respect to boolean statements, facts, and cloze-style phrases
has been studied, the consistency of their responses with respect to questions that permit sets of
answers has, to the best of our knowledge, not been explored in depth.


**Query containment** Given a database _D_ and a structured query _Q_, let [[ _Q_ ]] _D_ denote the answers
for the query on that database. Given two queries _Q_ 1 and _Q_ 2, query containment asks if, for any
database _D_ (whose schema is compatible with _Q_ ), it holds that [[ _Q_ 1]] _D_ [[ _Q_ 2]] _D_, while _query_
_⊆_
_equivalence_ asks whether or not [[ _Q_ 1]] _D_ = [[ _Q_ 2]] _D_ likewise holds for any database (Abiteboul et al.,
1995). These problems have been studied for a variety of query languages and database models.
While these problems are decidable for simple query formalisms like conjunctive queries under
set semantics (akin to queries that only allow relational-style joins), they are undecidable for more
expressive query languages (like the relational algebra, full SQL, etc.) (Abiteboul et al., 1995).
Referring back to the work of Zhao et al. (2023), LLMs thus cannot decide equivalence for the full
SQL language, but it is of interest to know where the limits lie, and how to improve performance.


3 ANSWER-SET CONSISTENCY


In this section, we define the notion of answer-set consistency, describe the dataset we create to
evaluate it, the metrics we use to quantify it, and the mitigation strategies we propose to address it.


3


3.1 DEFINITION


We say that _Q_ is an _enumeration question_ if it has a ground truth answer that is a set, denoted [[ _Q_ ]]
(see Table 1). Let _M_ be a model (e.g., an LLM, potentially primed with a prompt). We denote by

[[ _Q_ ]] _M_ the answer set generated by model _M_ for question _Q_ .


We now define the related notions of _answer-set consistency_ and _answer-set contradiction_ .


Given two enumeration questions _Q_ 1 and _Q_ 2 that are equivalent (i.e., [[ _Q_ 1]] = [[ _Q_ 2]]), we say that a
model _M_ is _answer-set consistent_ with respect to the equivalence of ( _Q_ 1, _Q_ 2) if and only if [[ _Q_ 1]] _M_ =

[[ _Q_ 2]] _M_ . If this property does not hold, we call the model _answer-set_ _inconsistent_ . We likewise
define answer-set (in)consistency for other relations, including _containment_ (given [[ _Q_ 1]] [[ _Q_ 2]],
_⊆_
check [[ _Q_ 1]] _M_ [[ _Q_ 2]] _M_ ), _disjointness_ (given [[ _Q_ 1]] [[ _Q_ 2]] =, check [[ _Q_ 1]] _M_ [[ _Q_ 2]] _M_ = ) and
_⊆_ _∩_ _∅_ _∩_ _∅_
_overlap_ (given [[ _Q_ 1]] [[ _Q_ 2]] =, check [[ _Q_ 1]] _M_ [[ _Q_ 2]] _M_ = ).
_∩_ _̸_ _∅_ _∩_ _∅_

We do not need ground-truth answer sets for questions in order to analyze answer-set consistency.


**Example 3.1.** Considering the four questions in Table 1, even without the ground-truth answer sets,
we see that [[ _Q_ 1]] = [[ _Q_ 2]] (equivalence), since _tributaries_ of a river are defined as _rivers and streams_
that _flow_ directly into that river. Similarly, we see that [[ _Q_ 3]] [[ _Q_ 1]] and [[ _Q_ 4]] [[ _Q_ 1]] (containment)
_⊆_ _⊆_
since both left- and right-bank tributaries should be included in the answer for all tributaries. Furthermore, we see that _Q_ 3 _Q_ 4 = (disjointness), as right-bank and left-bank tributaries are mutually
_∩_ _∅_
exclusive. Other implied relations exist, for example, [[ _Q_ 3]] [[ _Q_ 2]] (containment), [[ _Q_ 3]] [[ _Q_ 2]] =
_⊆_ _∩_ _̸_ _∅_
(overlap). Hence, for example, if a model would give overlapping answers for _Q_ 3 and _Q_ 4, we would
deem it answer-set inconsistent with respect to the disjointness of ( _Q_ 3 _, Q_ 4).


The presented definition is agnostic to the open world assumption, or closed world assumption, since
we are solely interested in the relationship between two answers, regardless of whether elements not
included in the answer are treated as false, or unknown to the model. However, under the open world
assumption, a model may return empty answers, which are trivially consistent; for this reason, in
our experiments, we will handle cases in which the models returns empty answers separately.


Next we define _answer-set_ _contradictions_ . Given two enumeration questions _Q_ 1 and _Q_ 2, let

[[ _Q_ 1 _, Q_ 2]] _M_ E _,_ C _,_ D _,_ O denote the prediction of model _M_ for which binary relations hold for
_⊆{_ _}_
( _Q_ 1 _, Q_ 2): Equivalence, Containment, Disjointness, or Overlap. We represent the prediction as a set
since multiple relations can hold. We say that a model _M_ gives an _answer-set_ _contradiction_ if it
predicts a relation _R_ [[ _Q_ 1 _, Q_ 2]] _M_ that does not hold for [[ _Q_ 1]] _M_ and [[ _Q_ 2]] _M_ . In other words, it
_∈_
predicts a relation for _Q_ 1 and _Q_ 2 that its own answers for _Q_ 1 and _Q_ 2 do not respect. This is a
self-contradiction: the model contradicts itself, rather than the expectation relative to a ground truth.


**Example** **3.2.** Considering the first two questions in Table 1, if a model predicts that these two
questions are equivalent (E), but gives answer sets [[ _Q_ 1]] _M_ and [[ _Q_ 2]] _M_ such that [[ _Q_ 1]] _M_ = [[ _Q_ 2]] _M_,
this represents an answer-set contradiction with respect to equivalence.


For reasons of space, we delegate discussion of _answer-set contradictions_ to Appendix F.


3.2 DATASET


In order to evaluate the _answer-set consistency_ of various LLMs, and to address our research questions, we require a collection of pairs of questions that satisfy relations such as equivalence, containment, etc. Given that we know of no such existing dataset, we design and construct one.


We aim for questions whose ground-truth answers are objective and non-empty. We further aim for
questions with between 2 and 100 results: having 2 results enables us to test (strict, non-empty) containment over such questions, and we set a limit of 100 results to avoid context-window issues when
testing smaller models. Example questions meeting these criteria are _Which countries are members_
_of_ _the_ _European_ _Union?_, _What_ _are_ _the_ _tributaries_ _of_ _the_ _Madeira_ _River?_, etc. Such questions can
be found in knowledge graph question answering (KGQA) datasets, which provide a good starting
point, since they contain mostly objective enumeration questions, and we can use the queries provided to filter cases with too few or too many results. Some datasets further provide paraphrases that
cover equivalence. Unfortunately, only one KGQA dataset, QAWIKI (Moya Loustaunau & Hogan,
2025), provides containment relations, and only a few.


4


We constructed our dataset, which we call the Answer-Set Consistency Benchmark (ASCB), from
three base KGQA datasets: LC-QUAD 2.0 (Dubey et al., 2019), QALD (Usbeck et al., 2024),
and QAWIKI (Moya Loustaunau & Hogan, 2025). From each dataset, we evaluated the structured
queries associated with questions to ensure that they are enumeration questions and to test that
they satisfy the cardinality bounds. A preliminary filtering step was conducted on the questions
contained in the LC-QUAD 2.0 and QALD datasets using LLMs. This step aimed to remove
questions that were not aligned with our criteria and to generate new questions that conform to the
logical relations under evaluation. The complete procedure for filtering and question generation
is described in Appendix B.2. This created a candidate list of base questions, which we manually
reviewed and filtered down to a smaller set of selected questions that best meet our criteria, as well as
additional desiderata prioritizing the “crispness” [1], diversity, fluency, and objectiveness of questions.
We enriched this candidate set with available paraphrases to cover the case of equivalence, creating
a set of pairs of equivalent questions of the form ( _Q_ 1 _, Q_ 2) such that [[ _Q_ 1]] = [[ _Q_ 2]].


We next looked to add examples for (strict) containment to our dataset. To reduce manual effort,
rather than start from scratch, we used LLMs to extend the equivalent question pairs ( _Q_ 1 _, Q_ 2)
created previously to suggest a third query _Q_ 3 such that [[ _Q_ 3]] [[ _Q_ 1]] (and, by implication,
_⊆_

[[ _Q_ 3]] [[ _Q_ 2]]). During this process, it required relatively little additional effort to define a fourth
_⊆_
query, _Q_ 4, that captures the answers of _Q_ 1 that _Q_ 3 does not, such that [[ _Q_ 4]] = [[ _Q_ 1]] _\_ [[ _Q_ 3]], providing
an additional example not only for strict containment, but also disjointness.


Thus our initial dataset consisted of quadruples of questions ( _Q_ 1 _, Q_ 2 _, Q_ 3 _, Q_ 4), where Table 1 exemplifies one of the quadruples in our dataset. From such a quadruple of questions, there are 12 primary
relations, where, for clarity, we call the superset containment _broader_, and its inverse subset containment _narrower_ . These relations are _equivalence_ for ( _Q_ 1 _, Q_ 2) and ( _Q_ 2 _, Q_ 1); _broader_ for ( _Q_ 1 _, Q_ 3),
( _Q_ 1 _, Q_ 4), ( _Q_ 2 _, Q_ 3) and ( _Q_ 2 _, Q_ 4); _narrower_ for ( _Q_ 3 _, Q_ 1), ( _Q_ 3 _, Q_ 2), ( _Q_ 4 _, Q_ 1) and ( _Q_ 4 _, Q_ 2), and
_disjointness_ for ( _Q_ 3 _, Q_ 4) and ( _Q_ 4 _, Q_ 3). There are also further implicit relations, such as broader
and narrower implied by equivalence, and overlaps implied by broader, narrower and equivalence,
assuming non-empty answer sets; for example, given the broader relation for ( _Q_ 1 _, Q_ 3) (whereby

[[ _Q_ 1]] [[ _Q_ 3]]), and assuming non-empty answer sets ([[ _Q_ 1]] =, [[ _Q_ 3]] = ), this implies that
_⊇_ _∅_ _∅_
overlap holds for the same question pair ([[ _Q_ 1]] [[ _Q_ 3]] = ).
_∩_ _̸_ _∅_

These candidate questions were manually revised, pruned, modified, etc., to ensure a high-quality
dataset, resulting in 150 question quadruples from each dataset. We further added an additional
source, which we call SYNTHETIC, of questions generated from scratch by LLMs, the generation
process is detailed in Appendix B. It is important to note that in many cases, the questions extracted
merely served as “inspiration” for the final quadruple, even for the base query: the questions in
many cases were heavily modified. Furthermore, suggestions by LLMs for _Q_ 3 and _Q_ 4, though
useful, often did not satisfy the formal relations expected [2], and needed to be modified. As a final
step, we used LLMs to revise the quadruples and suggest improvements for phrasing, corrections,
etc., which were revised manually and applied if deemed suitable. The manual revision, curation,
and modification of questions were conducted by three of the authors.


The resulting ASCB dataset (available online (authors, 2025)) comprises 600 such quadruples of
handcrafted questions in English, with 2 _,_ 400 questions in total.


3.3 EVALUATION TASKS AND MITIGATION STRATEGIES


To evaluate the answer-set consistency of LLMs using our ASCB, we perform three distinct tasks,
the latter two of which involve mitigation strategies to try to achieve better answer-set consistency
(rather than the completeness or correctness of answer sets). For all such tasks, the LLMs under test
are configured to use the lowest possible temperature. The prompts used are given in Appendix A.


**Task 1: Base evaluation.** Our first task evaluates the base answer-set consistency of the LLM under
test, and involves two subtasks.


1This relates to the results forming a crisp set: some questions, such as “ _What are the jobs in finance?_ ” in
LC-QuAD 2.0, do not clearly define a “crisp” base set for results, and are excluded.
2Often the suggestions for _Q_ 3 and _Q_ 4 did not form a true dichotomy, for example: _What counties of North_
_Dakota_ _use_ _the_ _Mountain_ _Time_ _Zone?_ and _What_ _counties_ _of_ _North_ _Dakota_ _use_ _a_ _timezone_ _other_ _than_ _the_
_Mountain Time Zone?_ is not a true dichotomy as some counties are in multiple time zones.


5


Table 2: The relations considered for experiments: their notation, definition and meaning


Notation Definition Meaning


_E_ 1 _,_ 2 [[ _Q_ 1]] = [[ _Q_ 2]] _Equivalence_ : Results of _Q_ 1 are the same as those of _Q_ 2

_N_ 3 _,_ 1 [[ _Q_ 3]] _⊆_ [[ _Q_ 1]] _Narrower_ : Results of _Q_ 3 are contained in those of _Q_ 1

_N_ 4 _,_ 1 [[ _Q_ 4]] _⊆_ [[ _Q_ 1]] _Narrower_ : Results of _Q_ 4 are contained in those of _Q_ 1

_D_ 3 _,_ 4 [[ _Q_ 3]] _∩_ [[ _Q_ 4]] = _∅_ _Disjointness_ : No result of _Q_ 3 is a result of _Q_ 4

_E_ 4 _,_ 1 _\_ 3 [[ _Q_ 4]] = [[ _Q_ 1]] _\_ [[ _Q_ 3]] _Equivalence_ : Results obtained by removing the answer set of _Q_ 3
from that of _Q_ 1 are the same as those of _Q_ 4
_E_ 1 _,∗_ [[ _Q_ 1]] = [[ _Q_ _[∗]_ 1 []]] _Equivalence_ : Results of _Q_ 1 are the same as those of _Q_ 1 posed in
a different context at a a different time


In Task 1.1, _Classification_, the models are presented pairs of questions ( _Qi, Qj_ ) taken from each
quadruple and asked to identify the _first_ set-theoretic relation that holds, in the following order,
from the answer set of _Qi_ to that of _Qj_ : equivalence ([[ _Qi_ ]] = [[ _Qj_ ]]), contained by ([[ _Qi_ ]] [[ _Qj_ ]]),
_⊆_
contains ([[ _Qi_ ]] [[ _Qj_ ]]), disjointness ([[ _Qi_ ]] [[ _Qj_ ]] = ) and overlap ([[ _Qi_ ]] [[ _Qj_ ]] = ). Although
_⊇_ _∩_ _∅_ _∩_ _∅_
any pair of questions must satisfy at least one such relation, we further add an _unknown_ option
(‘idk’) in case the model cannot confidently determine the relation.


In Task 1.2, _Enumeration_, all 2 _,_ 400 questions are posed to the models independently, each within
its own isolated context, requesting the LLM to enumerate all answers for the question. The prompt
furtherreturn ’idkinstructs’ in case it cannot answer, to return ‘the model to return an exhaustivenolistanswerseparated’ if there is no answer, and to expandby ‘ _|_ ’, to avoid additional text, to
acronyms and use full names whenever possible. The corresponding answers are then collected.


**Task 2:** **Classification-then-Enumeration (CtE).** Following initial experiments on Task 1, we investigated a mitigation strategy that operates within a conversational context. The model is first
asked to identify the relationship between the questions, and is then prompted to enumerate their
answers. Our hypothesis is that this will improve answer-set consistency by allowing the model to
first reason about the relation that the answer sets should satisfy before enumerating them.


**Task** **3:** **Oracle.** We first run Task 1.1, and if we detect an answer-set inconsistency with respect
to a given relation for a question pair ( _Qi, Qj_ ), we tell the LLM the correct relation that holds, and
request it to enumerate answers again. This task is an ideal version of Task 2, as it assumes an oracle
that knows what relation holds between the questions. This will give us insights into what the model
could achieve in Task 2 if it always classifies the relation correctly.


**Question** **and** **relations** **considered.** Our quadruples provide a large number of potential binary
relations to test. However, many such relations are redundant. Thus, we only consider each pair of
questions once (skipping symmetric and inverse relations), and only test for the primary relation.
The selected relations are presented in Table 2. The relations _E_ 1 _,_ 2, _N_ 3 _,_ 1, _N_ 4 _,_ 1, and _D_ 3 _,_ 4 cover
equivalence, containment (narrower) and disjointness. We also test an _n_ -ary relation _E_ 4 _,_ 1 _\_ 3 for
an answer set constructed from set difference as an example of a more complex task. Finally, to
establish a referential result for non-determinism of the model over time, we test _E_ 1 _,∗_, which runs
the same question _Q_ 1 at different times, in different contexts, which we will use as a control later.


3.4 EVALUATION MEASURES


We present various measures to evaluate the answer-set consistency of LLMs with respect to the
previous tasks. The first two address the performance for enumeration, while the second addresses
the performance for classification.


**Classification** **accuracy.** We first quantify the ability of LLMs to correctly classify the settheoretic relation that holds between the answer-sets of the questions. For each relation _R_ _∈_
_{_ ( _nE_ 1= _,_ 2 _, N_ 6003 _,_ 1for _, N_ ASCB)4 _,_ 1 _, D_ 3 _,_ 4is _, E_ defined4 _,_ 1 _\_ 3 _}_, asthe _Rclassification_ [ACC] ( _M_ ) = [# correct classif] _accuracy_ [cations of] of _n_ a model _[ R]_ [ by] _[ M]_ _M_ . Weoverfurther _n_ testdenoteinstancesby

ACC( _M_ ) the average of the accuracy of the five aforementioned relations for the model _M_ .


6


**Consistency rates.** For each pair of enumerated answer sets, we evaluate whether or not the six expected relations _R_ _∈{E_ 1 _,_ 2 _, N_ 3 _,_ 1 _, N_ 4 _,_ 1 _, D_ 3 _,_ 4 _, E_ 4 _,_ 1 _\_ 3 _, E_ 1 _,∗}_ are satisfied. Each such binary relation
_R_ is checked independently over _n_ test instances, where we say that an instance is consistent for _R_ if
the enumerated answers for the questions satisfy _R_ . The _consistency rate_ of _M_ for _R_ is then defined
as _R_ [CON] ( _M_ ) = [# consistent answer sets for] _n_ _[ R]_ [ by] _[ M]_ . Here we exclude empty answer sets and responses of

“idk”, which are reported separately. We further denote by CON( _M_ ) the average consistency rate
of _M_ over the five relations excluding _E_ 1 _,∗_ (which is intended as a control).

**Jaccard** **similarity.** The consistency rates consider each ( _Qi, Q_ _[′]_ _i_ _[, ⋆][i]_ [)] [as] [a] [discrete] [result] [(1] [for]
correct, or 0 for incorrect) - irrespective of how close the answer sets of the two questions are to
satisfying the relation. To complement the first measure, we consider a second continuous measure
that captures the degree to which an instance is satisfied. Specifically, we use the well-known _Jac-_
_card similarity_, defined for sets _S_ 1 and _S_ 2 as _J_ ( _S_ 1 _, S_ 2) = _[|][S]_ _S_ [1] 1 _[∩][S]_ _S_ [2] 2 _[|]_ [For a given model] _[ M]_ [, and test]

_|_ _∪_ _|_ [.]
instance ( _Qi, Qj_ ), we define the similarity of the test instance as _J_ ([[ _Qi_ ]] _M_ _,_ [[ _Qj_ ]] _M_ ). The Jaccard
similarity of a model _M_ for a relation _R_ _∈{E_ 1 _,_ 2 _, D_ 3 _,_ 4 _, E_ 4 _,_ 1 _\_ 3 _, E_ 1 _,∗}_ over _n_ test instances is then
defined as _R_ [SIM] ( _M_ ) = [sum of similarity of all test instances by] _n_ _[ M]_, i.e., as the average similarity over _R_ . Empty

answer sets and “idk” are excluded from these results and are reported separately. We apply this
measure for relations involving equivalence ( _E_ 1 _,_ 2, _E_ 4 _,_ 1 _\_ 3, _E_ 1 _,∗_ ) and disjointness ( _D_ 3 _,_ 4), where a
score close to 1 is good in the former case, and poor in the latter case.


**Empty rates.** We also measure the percentage of the 2 _,_ 400 questions that return “idk” or an empty
answer, denoted as %IDK.


**Hypotheses** **and** **significance** **testing.** We propose two hypotheses: ( _H_ 1) The **CtE** and **Oracle**
strategies yield less answer-set inconsistency than **Base** . ( _H_ 2) LLMs with better general performance produce more consistent responses.


To test statistical significance, we apply the one-sided McNemar test (Lachenbruch, 2014), which is
suitable for paired nominal data. We adopt a significance level of _α_ = 0 _._ 05. The null hypothesis of
no improvement is rejected if the one-sided p-value satisfies _p_ _<_ 0 _._ 05. In such cases, we conclude
that the alternative strategy or model yields a statistically significant improvement in consistency.


**Stochasticity** **control.** The causes of answer-set inconsistency in LLMs can be attributed to two
main factors. (1) _Stochasticity_ in the generation process, such as during token sampling, or latent
variability, leading the model to produce different outputs for the same query across runs. (2) _Seman-_
_tic misunderstanding_, where the model misinterprets or overlooks the logical or semantic relations
among queries, resulting in violations of restrictions. We introduce _E_ 1 _,∗_ precisely to estimate and
control for the effects of (1), which varies across models. Further discussion of the causes behind
LLM inconsistencies can be found in Appendix G.


The impact specifically of the two factors on the answer-set inconsistency of model _M_ for _E_ 1 _,_ 2
can be assessed by comparing _E_ 1 [CON] _,_ 2 [(] _[M]_ [)][ with] _[ E]_ 1 [CON] _,_ 1 _,_ 2 [(] _[M]_ [)][ with] _[ E]_ 1 [SIM] _,_ [a large gap]
_∗_ [(] _[M]_ [)][ and] _[ E]_ [SIM] _∗_ [(] _[M]_ [)][:]
indicates (2) plays a smaller role, and a small gap indicates (2) plays a larger role. On the other hand,
we can compare the difference of consistency for equivalence relations like _E_ 1 _,∗_, _E_ 1 _,_ 2 with other
relations like _N_ 3 _,_ 1, _N_ 4 _,_ 1, _D_ 3 _,_ 4 where (assuming stocasticity plays a similar role for such relations)
a large gap indicates (2) plays a larger role, and a small gap indicates (2) plays a smaller role.


4 RESULTS


We now present the results of our experiments on 18 LLMs from the DeepSeek, Gemini, Grok,
GPT, Llama, and Mistral families. For all models, the temperature was set to the lowest possible
value (zero, if possible). LLMs are ranked in ascending order based on their Global Average scores
reported by White et al. (2025) from _A_ (worst) to _R_ (best); open-source models not included in this
ranking are loosely positioned alongside models of similar parameter size. The results are reported
for the overall dataset, that is, the dataset obtained by merging the four sources presented in Section
3.2. Results for individual datasets are available (anonymously) on GitHub (authors, 2025). For
reasons of space, we provide figures that help to identify trends in these results in Appendix C.


7


4.1 CLASSIFICATION TASK


Appendix D reports the accuracy of different LLMs on the relation classification task. The highest
accuracy score in each column is indicated in bold. The results reveal substantial variability among
the evaluated models. Smaller-scale models such as Llama-3.1-8b exhibit poor performance across
all relations, with accuracy often below 20%. Similarly, GPT-oss-20b, GPT-4.1-nano, and Mistralsmall:24b perform inconsistently, though notably the relations they struggle with sometimes differ.
In contrast, larger models such as Gemini-2.5-pro, GPT-5-nano, GPT-5, GPT-o3, and Grok-3-mini
achieve accuracies consistently above (or close to) 90% across all relations. Among the evaluated
relations, _E_ 1 _,_ 2 and _D_ 3 _,_ 4 are the least challenging overall, while _N_ 4 _,_ 1 emerges as the most challenging, revealing the limitations of current models in reliably capturing the containment relation. [3]
GPT-5 demonstrates the best performance over all relations, followed closely by Gemini-2.5-pro.


4.2 ANSWER-SET CONSISTENCY


Table 3 lists the results for answer-set consistency, considering the control relation, all five test
relations, and the measures of consistency rate and Jaccard similarity. Recall that, exceptionally for
_D_ 3 [SIM] _,_ 4 [, a score lower than][ 0][ for Jaccard similarity is better as the answer sets should be disjoint.]


We see high rates of answer-set inconsistency for all relations, including even the control relation.
The small gap between _E_ 1 _,∗_ and _E_ 1 _,_ 2 suggests that stochasticity plays an important role in answerset inconsistency, even though temperature was lowered as much as possible in all such cases, this
phenomenon aligns with prior research demonstrating that LLMs produce inconsistent outputs even
with identical inputs. The literature identifies four primary sources of this variability: decoding
randomness Ackley et al. (1985); Li et al. (2025); Atil et al. (2024); Renze (2024), computational
nondeterminism Yuan et al.; Masoudnia & Ebrahimpour (2014); Dao et al. (2022); Atil et al. (2024),
order sensitivity Vaswani et al. (2017); Liu et al. (2024a); Lu et al. (2022), and data-level conflicts Xu
et al. (2024); Nakshatri et al. (2025); Xie et al. (2023). Beyond these inherent stochastic factors, we
observe additional systematic errors including terminological inconsistency and knowledge gaps
that lead to incomplete outputs (see Appendix G and H for more discussion). However, the gap
between _E_ 1 _,_ 2, _E_ 1 _,∗_ and containment / ternary relations suggests that semantic misunderstanding is
also a key cause of inconsistency for these latter relations in particular.

The most inconsistent relation is the ternary relation of _E_ 4 _,_ 1 _\_ 3, with the most consistent relation
overall being the disjointness relation _D_ 3 _,_ 4. The best model in terms of average consistency across
all relations for the base case is GPT-5 ( _∼_ 57%), with the important caveat that it exhibits a high
%IDK rate ( _∼_ 32%). Models such as Grok-3-mini and Gemini-2.5-flash arguably perform better,
with lower average consistency of _∼_ 48% and _∼_ 46%, resp., but also lower %IDKvalues of _∼_ 8%
and _∼_ 5%, respectively. There are notable improvements for the mitigation strategy Classify-thenEnumerate (CtE), which (surprisingly) even outperforms the Oracle in many cases, due to good
classification accuracy (see Appendix D), and perhaps due to forcing the LLM itself to reason about
the questions when classifying their relation. We also observed that the %IDK values for CtE are
generally higher than Base and Oracle. This suggests that under this strategy, LLMs tend to adopt
a safer approach by answering “idk” when uncertain, which may explain why CtE outperforms
the other two strategies. Improvement with CtE is not universal, however: CtE sometimes performs
worse than Base due to the model being unable to classify the relation (as is the case for _E_ 4 _,_ 1 _\_ 3
in GPT-4.1.-nano, which answers idk for each case when asked to identify the relation), or due to
the model misinterpreting the more complex prompts (e.g, GPT-5-nano often continues to return a
relation for _E_ 4 _,_ 1 3 _after_ classification, when later asked to enumerate results).
_\_


4.3 HYPOTHESIS TESTING


In Appendix E, we present an analysis of the statistical significance of our results. Regarding hypothesis _H_ 1, that “The **CtE** and **Oracle** strategies yield less answer-set inconsistency than **Base** ”,
this is confirmed by a _p_ -value _<_ 0 _._ 001 for almost all models for both strategies. Regarding hypothesis _H_ 2, the mitigation strategies significantly affect consistency and relative model rankings.


3Regarding why _N_ 4 _,_ 1 is more challenging than _N_ 3 _,_ 1, _N_ 3 _,_ 1 is based on _Q_ 3, and _N_ 4 _,_ 1 is based on _Q_ 4, where
_Q_ 4 questions tend to negate the restriction that _Q_ 3 adds over _Q_ 1, and this negation appears more challenging.


8


Table 3: Per-relation consistency (0-100 scale) for each model and strategy (Str).


**ID** **Model** **Str** _E_ 1 [CON] _,∗_ _E_ 1 [SIM] _,∗_ _E_ 1 [CON] _,_ 2 _N_ 3 [CON] _,_ 1 _N_ 4 [CON] _,_ 1 _D_ 3 [CON] _,_ 4 _E_ 4 [CON] _,_ 1 _\_ 3 CON _E_ 1 [SIM] _,_ 2 _D_ 3 [SIM] _,_ 4 _E_ 4 [SIM] _,_ 1 _\_ 3 %IDK


B GPT-oss-20b Base 22.71 44.44 21.67 38.00 32.50 83.00 10.00 37.03 44.13 7.75 29.27 12.54
CtE –”– –”– 67.17 **100.00** 73.17 84.67 38.00 72.60 79.39 10.73 60.99 37.04
Ora. –”– –”– 58.33 80.67 92.33 95.33 62.00 77.73 76.50 3.97 78.64 13.33


D Mistral-small:24b Base 43.01 64.10 44.07 50.34 45.25 50.85 1.53 38.41 63.99 34.47 20.57 29.19
CtE –”– –”– 84.50 **100.00** 80.67 79.83 42.33 77.47 92.57 17.74 63.71 33.58
Ora. –”– –”– 72.00 74.67 90.00 95.83 50.17 76.53 82.96 1.48 67.82 15.54


F Gemini-2.0-flash Base 48.03 70.08 32.67 41.50 35.33 62.67 5.00 35.43 60.08 13.00 29.14 0.42
CtE –”– –”– 88.17 **100.00** 94.50 64.33 53.83 80.17 93.84 27.82 62.68 40.58
Ora. –”– –”– 79.00 84.17 92.00 96.67 73.83 85.13 82.33 1.37 84.15 2.96


H GPT-4o Base 48.98 52.09 45.17 53.33 47.00 62.83 6.17 42.90 65.34 26.69 26.12 29.79
CtE –”– –”– **97.33** 99.33 98.33 36.33 30.67 72.40 58.67 62.79 35.08 66.66
Ora. –”– –”– 85.33 91.83 94.83 86.50 67.83 85.26 73.43 12.09 58.79 33.88


J Grok-3-mini Base 37.67 63.09 34.33 52.33 43.83 87.67 23.00 48.23 63.46 5.61 44.78 8.12
CtE –”– –”– 90.83 **100.00** 90.00 86.67 66.83 86.87 96.39 12.77 78.69 35.08
Ora. –”– –”– 88.17 92.67 **98.50** 95.50 81.00 91.17 93.63 4.00 79.16 13.46


L Gemini-2.5-flash Base 37.69 64.85 33.50 49.50 41.50 84.50 21.83 46.17 61.01 3.60 43.82 5.33
CtE –”– –”– 89.83 **100.00** 96.83 90.67 84.83 **92.43** 93.83 8.82 83.92 31.96
Ora. –”– –”– 89.17 90.17 95.33 95.67 **85.50** 91.17 92.56 4.08 83.49 8.00


N DeepSeek-reasoner Base 29.95 51.86 18.67 40.17 30.17 82.00 10.83 36.37 46.10 3.96 31.15 4.21
CtE –”– –”– 78.50 75.33 85.33 69.67 51.50 72.07 82.40 29.66 56.37 28.54
Ora. –”– –”– 68.33 75.67 86.33 93.00 58.33 76.33 82.21 5.37 79.55 6.25


P GPT-5-mini Base 63.15 76.22 65.33 63.83 68.17 61.50 14.33 54.63 77.18 35.93 37.25 47.17
CtE –”– –”– 88.00 **100.00** 75.00 67.33 34.00 72.87 92.58 32.67 44.64 55.08
Ora. –”– –”– 86.83 91.67 95.00 73.67 54.50 80.33 91.04 26.17 55.03 50.04


R GPT-5 Base 58.59 76.89 61.00 64.17 65.00 73.50 21.67 57.07 78.16 22.11 48.16 32.00
CtE –”– –”– 92.33 **100.00** 96.50 70.67 59.83 83.87 96.63 29.20 66.83 47.50
Ora. –”– –”– 90.67 94.00 98.17 79.00 65.33 85.43 96.11 21.00 69.17 34.59


Table 4 shows significant positive correlations between _D_ 3 [CON] _,_ 4 [,] _[E]_ 4 [CON] _,_ 1 3 [,] [and] _[D]_ 3 [SIM] _,_ 4 [and] [the] [average]
_\_
score of the models on the external benchmark (White et al., 2025). Models not in this benchmark are excluded. In particular, _E_ 4 _,_ 1 _\_ 3 exhibits strong positive correlations across both measures.
Since _E_ 4 _,_ 1 _\_ 3 is the most challenging relation to identify, this finding supports _H_ 2, suggesting that
consistency is more pronounced in the more complex answer-set tasks.


9


Table 4: Pearson correlations between external model scores and reasoning performance metrics.
Asterisks indicate statistical significance: - _p <_ 0 _._ 05, ** _p <_ 0 _._ 001.


_E_ 1 [CON] _,_ 2 _N_ 3 [CON] _,_ 1 _N_ 4 [CON] _,_ 1 _D_ 3 [CON] _,_ 4 _E_ 4 [CON] _,_ 1 _\_ 3 CON() _E_ 1 [SIM] _,_ 2 _D_ 3 [SIM] _,_ 4 _E_ 4 [SIM] _,_ 1 _\_ 3


_r_ 0.328 0.410 0.406 0.650* 0.821** 0.682* 0.425 -0.220 0.840**
_p_ 0.252 0.145 0.149 0.012 _<_ 0.001 0.007 0.130 0.449 _<_ 0.001


5 DISCUSSION


We have highlighted the phenomenon of answer-set inconsistency in LLMs, proposed a dataset to
evaluate it, defined various measures to quantify it, and presented the results for 18 LLMs.


**Research questions** We address the RQs presented in the introduction:


**RQ1** LLMs exhibit high degrees of answer-set inconsistency for enumeration questions, with the
particular degree depending on the model and relation (see Table 3).


**RQ2** Contemporary large LLMs can recognize the set-theoretic relations that hold between enumeration questions with accuracy often about 90%, though smaller/older models struggle
with many types of relations (see Appendix D).


**RQ3** Binary equivalence relations appear to be the easiest relations for the LLMs to reason about,
where they struggle with containment and _n_ -ary relations (see Appendix D and Table 3).

**RQ4** Based on our control ( _E_ 1 _,∗_ ), much of the answer-set inconsistency for equivalence relations
is due to the stochastic nature of LLMs, whereas semantic misunderstanding plays a more
dominant role for containment, disjointness and, _n_ -ary relations (Table 3).


**RQ5** Answer-set inconsistency can be mitigated (i.e., improved by a wide margin, with statistical
significance) by prompting strategies that ask the LLM to reason about the set-theoretic
relations that hold between enumeration questions and their answer sets (Table 3).


The performance of LLMs for enumeration questions is remarkable considering that the technology
was not designed for this sort of workload (unlike, say, databases). But their performance is far from
perfect. Users should exercise caution when using contemporary LLMs to answer enumeration
questions, and should not expect consistent responses (even for the same query at different times).
Further research is required to understand and address this issue, potentially combining LLMs with
other technologies that provide consistency guarantees.


**Limitations** **and** **Future** **Work** The notion of answer-set (in)consistency could be extended further, for example, to include set cardinality (counts). More work is needed on how to improve
the consistency of LLMs in this regard, which may include prompting strategies that instruct the
LLM to reason about relations such as containment, disjointness, etc., inherent to such questions.
More advanced mitigation strategies could also be explored, such as asking the LLM to parse and
reason about structured representations of the questions (perhaps related to work using LLMs to
decide SQL equivalence Wei et al. (2025)). Moreover, the experimental setting employed in this
study is restricted to isolated, single-turn interactions and does not incorporate multi-turn dialogue.
Investigating conversational sequences represents a promising direction for future work, as temporal dependencies may introduce additional inconsistencies or amplify existing ones. The proposed
dataset consists of 600 quadruples (2 _,_ 400 questions) in English and focuses on static, factual domains. While manual curation contributes to the quality and clarity of the question–answer pairs,
and the scale of the dataset is sufficient to derive conclusions with statistical significance, future
efforts could explore automated or semi-automated approaches for expanding and diversifying the
dataset, enabling broader coverage and improved generalizability. Finally, we plan to investigate
strategies for mitigating answer-set inconsistency by converting questions into structured representations and subsequently using either the LLM or an external reasoning service to determine the
relations between these representations. Overall, one cannot expect an LLM by itself to provide
consistency guarantees, so it is of interest to conduct further research on combining LLMs with
technologies that provide such guarantees by design.


10


REFERENCES


Serge Abiteboul, Richard Hull, and Victor Vianu. _Foundations_ _of_ _Databases_ . Addison-Wesley,
1995. ISBN 0-201-53771-0. [URL http://webdam.inria.fr/Alice/.](http://webdam.inria.fr/Alice/)


David H Ackley, Geoffrey E Hinton, and Terrence J Sejnowski. A learning algorithm for boltzmann
machines. _Cognitive science_, 9(1):147–169, 1985.


Dean Allemang and Juan Sequeda. Increasing the accuracy of LLM question-answering systems
with ontologies. In Gianluca Demartini, Katja Hose, Maribel Acosta, Matteo Palmonari, Gong
Cheng, Hala Skaf-Molli, Nicolas Ferranti, Daniel Hern´andez, and Aidan Hogan (eds.), _The_ _Se-_
_mantic Web – ISWC 2024_, pp. 324–339, Cham, 2025. Springer Nature Switzerland. ISBN 978-3031-77847-6. doi: 10.1007/978-3-031-77847-6 ~~1~~ 8.


Berk Atil, Sarp Aykent, Alexa Chittams, Lisheng Fu, Rebecca J Passonneau, Evan Radcliffe,
Guru Rajan Rajagopal, Adam Sloan, Tomasz Tudrej, Ferhan Ture, et al. Non-determinism of”
deterministic” llm settings. _arXiv preprint arXiv:2408.04667_, 2024.


[Anonymous authors. Answer set consistency of LLMs for question answering, 2025. URL https:](https://anonymous.4open.science/r/ASCS-7412/)
[//anonymous.4open.science/r/ASCS-7412/.](https://anonymous.4open.science/r/ASCS-7412/)


Diego Calanzone, Stefano Teso, and Antonio Vergari. Logically consistent language models via
neuro-symbolic integration. In _The Thirteenth International Conference on Learning Represen-_
_tations_, 2025. [URL https://openreview.net/forum?id=7PGluppo4k.](https://openreview.net/forum?id=7PGluppo4k)


Roi Cohen, May Hamri, Mor Geva, and Amir Globerson. LM vs LM: Detecting factual errors via
cross examination. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), _Proceedings of the 2023_
_Conference_ _on_ _Empirical_ _Methods_ _in_ _Natural_ _Language_ _Processing,_ _EMNLP_ _2023,_ _Singapore,_
_December 6-10, 2023_, pp. 12621–12640. Association for Computational Linguistics, 2023. doi:
10.18653/V1/2023.EMNLP-MAIN.778.


Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memoryefficient exact attention with io-awareness. _Advances in neural information processing systems_,
35:16344–16359, 2022.


Junhua Ding, Huyen Nguyen, and Haihua Chen. Evaluation of question-answering based text summarization using LLM invited paper. In _2024 IEEE International Conference on Artificial Intel-_
_ligence Testing (AITest)_, pp. 142–149, 2024. doi: 10.1109/AITest62860.2024.00025.


Mohnish Dubey, Debayan Banerjee, Abdelrahman Abdelkawi, and Jens Lehmann. LC-QuAD 2.0:
A large dataset for complex question answering over Wikidata and DBpedia. In Chiara Ghidini,
Olaf Hartig, Maria Maleshkova, Vojtˇech Sv´atek, Isabel Cruz, Aidan Hogan, Jie Song, Maxime
Lefranc¸ois, and Fabien Gandon (eds.), _The Semantic Web – ISWC 2019_, pp. 69–78, Cham, 2019.
Springer International Publishing. ISBN 978-3-030-30796-7. doi: 10.1007/978-3-030-30796-7
5.


Yanai Elazar, Nora Kassner, Shauli Ravfogel, Abhilasha Ravichander, Eduard Hovy, Hinrich
Sch¨utze, and Yoav Goldberg. Measuring and improving consistency in pretrained language models. _Transactions_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics_, 9:1012–1031, 2021. doi:
10.1162/tacl ~~a~~ ~~0~~ 0410. [URL https://aclanthology.org/2021.tacl-1.60/.](https://aclanthology.org/2021.tacl-1.60/)


Bishwamittra Ghosh, Sarah Hasan, Naheed Anjum Arafat, and Arijit Khan. Logical consistency of
large language models in fact-checking. In _The Thirteenth International Conference on Learning_
_Representations_, 2025. [URL https://openreview.net/forum?id=SimlDuN0YT.](https://openreview.net/forum?id=SimlDuN0YT)


Aidan Hogan, Xin Luna Dong, Denny Vrandeˇci´c, and Gerhard Weikum. Large language models,
knowledge graphs and search engines: A crossroads for answering users’ questions, 2025. URL
[https://arxiv.org/abs/2501.06699.](https://arxiv.org/abs/2501.06699)


Myeongjun Jang, Deuk Sin Kwon, and Thomas Lukasiewicz. BECEL: Benchmark for consistency
evaluation of language models. In _Proceedings of the 29th International Conference on Compu-_
_tational Linguistics_, pp. 3680–3696, 2022.


11


Peter A Lachenbruch. McNemar test. _Wiley_ _StatsRef:_ _Statistics_ _Reference_ _Online_, 2014. doi:
10.1002/9781118445112.stat04876.


Yanggyu Lee and Jihie Kim. Evaluating consistencies in LLM responses through a semantic clustering of question answering. _arXiv preprint arXiv:2410.15440_, 2024. doi: 10.48550/arXiv.2410.
15440.


Lujun Li, Lama Sleem, Niccolo’ Gentile, Geoffrey Nichil, and Radu State. Exploring the impact of
temperature on large language models: Hot or cold? _arXiv preprint arXiv:2506.07295_, 2025.


Nianqi Li, Jingping Liu, Sihang Jiang, Haiyun Jiang, Yanghua Xiao, Jiaqing Liang, Zujie Liang,
Feng Wei, Jinglei Chen, Zhenghong Hao, and Bing Han. CR-LLM: A dataset and optimization for
concept reasoning of large language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar
(eds.), _Findings of the Association for Computational Linguistics:_ _ACL 2024_, pp. 13737–13747,
Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/
v1/2024.findings-acl.815. [URL https://aclanthology.org/2024.findings-acl.](https://aclanthology.org/2024.findings-acl.815/)
[815/.](https://aclanthology.org/2024.findings-acl.815/)


Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and
Percy Liang. Lost in the middle: How language models use long contexts. _Transactions_ _of_ _the_
_Association for Computational Linguistics_, 12:157–173, 2024a.


Shicheng Liu, Sina J Semnani, Harold Triedman, Jialiang Xu, Isaac Dan Zhao, and Monica S Lam.
Spinach: SPARQL-based information navigation for challenging real-world questions. _arXiv_
_preprint arXiv:2407.11417_, 2024b. doi: 10.48550/arXiv.2407.11417.


Yinhong Liu, Zhijiang Guo, Tianya Liang, Ehsan Shareghi, Ivan Vuli´c, and Nigel Collier. Aligning
with logic: Measuring, evaluating and improving logical consistency in large language models.
_arXiv preprint arXiv:2410.02205_, 2024c. doi: 10.48550/arXiv.2410.02205.


Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered
prompts and where to find them: Overcoming few-shot prompt order sensitivity. In _Proceedings_
_of_ _the_ _60th_ _Annual_ _Meeting_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics_ _(Volume_ _1:_ _Long_
_Papers)_, pp. 8086–8098, 2022.


Jiatong Ma, Linmei Hu, Rang Li, and Wenbo Fu. Local: Logical and causal fact-checking with
LLM-based multi-agents. In _Proceedings_ _of_ _the_ _ACM_ _on_ _Web_ _Conference_ _2025_, WWW ’25,
pp. 1614–1625, New York, NY, USA, 2025. Association for Computing Machinery. ISBN
9798400712746. doi: 10.1145/3696410.3714748.


Saeed Masoudnia and Reza Ebrahimpour. Mixture of experts: a literature survey. _Artificial Intelli-_
_gence Review_, 42(2):275–293, 2014.


Alberto Moya Loustaunau and Aidan Hogan. QAWiki: A knowledge graph question answering
& SPARQL query generation dataset for wikidata. In _Wikidata Workshop, colocated with ISWC_
_2025_, 2025.


Nishanth Sridhar Nakshatri, Shamik Roy, Manoj Ghuhan Arivazhagan, Hanhan Zhou,
Vinayshekhar Bannihatti Kumar, and Rashmi Gangadharaiah. When facts change: Probing llms
on evolving knowledge with evolveqa. _arXiv preprint arXiv:2510.19172_, 2025.


Matthew Renze. The effect of sampling temperature on problem solving in large language models.
In _Findings of the association for computational linguistics: EMNLP 2024_, pp. 7346–7356, 2024.


Yash Saxena, Sarthak Chopra, and Arunendra Mani Tripathi. Evaluating consistency and reasoning
capabilities of large language models. In _2024 Second International Conference on Data Science_
_and Information System (ICDSIS)_, pp. 1–5, 2024. doi: 10.1109/ICDSIS61070.2024.10594233.


Aryan Singhal, Thomas Law, Coby Kassner, Ayushman Gupta, Evan Duan, Aviral Damle, and
Ryan Luo Li. Multilingual fact-checking using LLMs. In Daryna Dementieva, Oana Ignat,
Zhijing Jin, Rada Mihalcea, Giorgio Piatti, Joel Tetreault, Steven Wilson, and Jieyu Zhao (eds.),
_Proceedings of the Third Workshop on NLP for Positive Impact_, pp. 13–31, Miami, Florida, USA,
November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.nlp4pi-1.2.
[URL https://aclanthology.org/2024.nlp4pi-1.2/.](https://aclanthology.org/2024.nlp4pi-1.2/)


12


Yuan Sui, Yufei He, Zifeng Ding, and Bryan Hooi. Can knowledge graphs make large language
models more trustworthy? an empirical study over open-ended question answering. _arXiv preprint_
_arXiv:2410.08085_, 2024.


Yiming Tan, Dehai Min, Yu Li, Wenbo Li, Nan Hu, Yongrui Chen, and Guilin Qi. Can ChatGPT
replace traditional KBQA models? an in-depth analysis of the question answering performance
of the GPT LLM family. In Terry R. Payne, Valentina Presutti, Guilin Qi, Mar´ıa Poveda-Villal´on,
Giorgos Stoilos, Laura Hollink, Zoi Kaoudi, Gong Cheng, and Juanzi Li (eds.), _The Semantic Web_

_– ISWC 2023_, pp. 348–367, Cham, 2023. Springer Nature Switzerland. ISBN 978-3-031-47240-4.
doi: 10.1007/978-3-031-47240-4 19.


Ricardo Usbeck, Xi Yan, Aleksandr Perevalov, Longquan Jiang, Julius Schulz, Angelie Kraft, Cedric
M¨oller, Junbo Huang, Jan Reineke, Axel-Cyrille Ngonga Ngomo, Muhammad Saleem, and Andreas Both. QALD-10 – the 10th challenge on question answering over linked data: Shifting from
DBpedia to Wikidata as a KG for KGQA. _Semantic Web_, 15(6):2193–2207, 2024.


Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances_ _in_ _neural_ _informa-_
_tion processing systems_, 30, 2017.


Cunxiang Wang, Xiaoze Liu, Yuanhao Yue, Xiangru Tang, Tianhang Zhang, Cheng Jiayang, Yunzhi
Yao, Wenyang Gao, Xuming Hu, Zehan Qi, Yidong Wang, Linyi Yang, Jindong Wang, Xing
Xie, Zheng Zhang, and Yue Zhang. Survey on factuality in large language models: Knowledge,
retrieval and domain-specificity. _arXiv_ _preprint_ _arXiv:2310.07521_, 2023. doi: 10.48550/arXiv.
2310.07521. [URL https://arxiv.org/abs/2310.07521.](https://arxiv.org/abs/2310.07521)


Julian Junyan Wang and Victor Xiaoqi Wang. Assessing consistency and reproducibility in the
outputs of large language models: Evidence across diverse finance and accounting tasks. _arXiv_
_preprint arXiv:2503.16974_, 2025.


Yuxia Wang, Minghan Wang, Hasan Iqbal, Georgi N. Georgiev, Jiahui Geng, and Preslav Nakov.
Openfactcheck: A unified framework for factuality evaluation of LLMs. _CoRR_, abs/2405.05583,
2024. doi: 10.48550/ARXIV.2405.05583. URL [https://doi.org/10.48550/arXiv.](https://doi.org/10.48550/arXiv.2405.05583)
[2405.05583.](https://doi.org/10.48550/arXiv.2405.05583)


Anjiang Wei, Jiannan Cao, Ran Li, Hongyu Chen, Yuhui Zhang, Ziheng Wang, Yuan Liu, Thiago SFX Teixeira, Diyi Yang, Ke Wang, et al. Equibench: Benchmarking large language models’
reasoning about program semantics via equivalence checking. In _Proceedings of the 2025 Con-_
_ference on Empirical Methods in Natural Language Processing_, pp. 33856–33869, 2025.


Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Benjamin Feuer, Siddhartha Jain, Ravid
Shwartz-Ziv, Neel Jain, Khalid Saifullah, Sreemanti Dey, Shubh-Agrawal, Sandeep Singh
Sandha, Siddartha V. Naidu, Chinmay Hegde, Yann LeCun, Tom Goldstein, Willie Neiswanger,
and Micah Goldblum. LiveBench: A challenging, contamination-limited LLM benchmark. In
_The_ _Thirteenth_ _International_ _Conference_ _on_ _Learning_ _Representations,_ _ICLR_ _2025,_ _Singapore,_
_April 24-28, 2025_ . OpenReview.net, 2025. [URL https://openreview.net/forum?id=](https://openreview.net/forum?id=sKYHBTAxVa)
[sKYHBTAxVa.](https://openreview.net/forum?id=sKYHBTAxVa)


Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. Adaptive chameleon or stubborn sloth:
Revealing the behavior of large language models in knowledge conflicts. In _The Twelfth Interna-_
_tional Conference on Learning Representations_, 2023.


Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang, Hongru Wang, Yue Zhang, and Wei Xu.
Knowledge conflicts for llms: A survey. _arXiv preprint arXiv:2403.08319_, 2024.


Jiayi Yuan, Hao Li, Xinheng Ding, Wenya Xie, Yu-Jhe Li, Wentian Zhao, Kun Wan, Jing Shi, Xia
Hu, and Zirui Liu. Understanding and mitigating numerical sources of nondeterminism in llm
inference. In _The Thirty-ninth Annual Conference on Neural Information Processing Systems_ .


Fuheng Zhao, Lawrence Lim, Ishtiyaque Ahmad, Divyakant Agrawal, and Amr El Abbadi. LLMSQL-Solver: Can LLMs determine SQL equivalence? _arXiv preprint arXiv:2312.10321_, 2023.


13


Danna Zheng, Mirella Lapata, and Jeff Z Pan. How reliable are LLMs as knowledge bases? rethinking facutality and consistency. _arXiv_ _preprint_ _arXiv:2407.13578_, 2024. doi: 10.48550/
arXiv.2407.13578.


Wenjie Zhou and Xiangyu Duan. Exploring and improving consistency in large language models for
multiple-choice question assessment. In _2024 International Joint Conference on Neural Networks_
_(IJCNN)_, pp. 1–9, 2024. doi: 10.1109/IJCNN60899.2024.10650668.


Qinrui Zhu, Derui Lyu, Xi Fan, Xiangyu Wang, Qiang Tu, Yibin Zhan, and Huanhuan Chen. Multimodel consistency for LLMs’ evaluation. In _2024_ _International_ _Joint_ _Conference_ _on_ _Neural_
_Networks (IJCNN)_, pp. 1–8, 2024. doi: 10.1109/IJCNN60899.2024.10651158.


14


A PROMPTS AND PROCESSES USED WITH LLMS


This section presents the three prompting strategies employed to evaluate the LLMs. The placeholder question denotes the text segment that is replaced with a question from the dataset.


A.1 _Zero-shot_


1 {question}

2 If you can’t answer, return ’idk’.

3 If the question has no answer, return ’no answer’.

4 In the response, do not use abbreviations or acronyms, but spell out the

full terms, i.e. "United States of America" instead of "USA".

5 If the response contains numbers or digits, use Arabic numerals. For

example, if the answer contains Star Wars V, indicate it with Star
Wars 5. Do not use Roman numerals (such as V) or text (such as five).

6 Please, Return me an exhaustive list separated by the symbol ’|’ don’t

add any other text.


A.2 _Classification & Question_


1 You are given two questions, q1 and q2.

2 Your task is to determine the logical relationship between their

respective sets of correct answers

3 Choose only one of the following relations:

4 - Equivalence: The answer sets of q1 and q2 are exactly the same.

5 - Contains: All answers to q2 are also answers to q1, but q1 includes

additional answers.

6 - ContainedBy: All answers to q1 are also answers to q2, but q2 includes

additional answers.

7 - Overlap: q1 and q2 share some, but not all, answers. Neither fully

contains the other.

8 - Disjoint: q1 and q2 have no answers in common.

9 - Unknown: The relation between the answer sets cannot be confidently

determined based on the given questions.

10 Here are the two questions:

11 q1: {q1}

12 q2: {q2}

13 Return **only** the name of the most appropriate relation from the list

above.

14 Do **not** provide any explanation or commentary.


1 You are given three questions: q1, q2, and q3.

2 Each question is associated with a set of answers. Your task is to

identify the logical relation between the concepts of questions based

on their answer sets. Compare the relationship of concept between
the following two sets, s1 and s2:

3 - s1: the set of all answers for q1 that are not answers for q2

4 - s2: the set of answers for q3.


5

6 These are the three questions:

7 q1: {q1}

8 q2: {q2}

9 q3: {q3}

10 For each comparison, use **one of the following labels**:

11 - Equivalence

12 - Contains

13 - ContainedBy

14 - Disjoint

15 - Overlap

16 - Unknown

17 Return only the exactly relation label.

18 Do **not** include any explanation or extra text.


15


A.3 _Oracle_


Below is the oracle prompt used for the logical equivalence relation. For the other two other logical relations tested, the prompt is adapted by replacing the true ~~l~~ ogical ~~r~~ elation placeholder and
modifying certain parts of the sentence to ensure coherence.


1 Pay attention, the questions I asked you before are {

true_logical_relation}, but you returned me different values.

2 In the response, do not use abbreviations or acronyms, but spell out the

full terms, i.e. "United States of America" instead of "USA".

3 If the response contains numbers or digits, use Arabic numerals. For

example, if the answer contains Star Wars V, indicate it with Star
Wars 5. Do not use Roman numerals (such as V) or text (such as five).

4 Please, Return me an exhaustive list separated by the symbol ’|’ don’t

add any other text.


A.4 EVALUATION PIPELINE


After submitting the questions contained in the ASCB using the prompts illustrated previously, with
each treated as an independent question, the responses are stored in JSON files where the key corresponds to the question ID and the value is a list of elements representing the set of answers generated
by the LLM for that particular question. Once the benchmark has been executed across all models
and for all three task types (Base, CtE, and Oracle), all JSON files are provided as input to an evaluation pipeline. This pipeline produces TSV files, one for each evaluated LLM, containing for every
answer: the Jaccard similarity, consistency, and a boolean flag indicating whether the answer to the
logical relation was satisfied. Finally, as the last step, starting from these task-, model-, and datasetspecific TSV files, four distinct summary TSV files (one for each dataset) are generated. These
summary files report the following information: the type of logical relation evaluated, the LLM, the
action, the consistency rates, the average non-empty response rate, the average Jaccard similarity,
and the ratio of empty responses. All aforementioned TSV file are publicly available in the GitHub
repository provided as supplementary material authors (2025).


B DATASET CONSTRUCTION


We describe now the dataset construction, consisting of our extraction and modification of questions
from the base datasets, followed by a process to filter and identify suitable candidate questions.


B.1 BASE DATASETS


To construct the question sets, we relied on four data sources: QALD (Usbeck et al., 2024), a
handcrafted question-answering dataset for Wikidata and DBpedia; LC-QuAD 2.0, a large-scale
question-answering dataset providing SPARQL queries with corresponding answers from both Wikidata and DBpedia [4] ; and a Synthetic dataset, generated entirely through LLMs.


**QALD** **and** **LC-QuAD** **2.0.** To extract questions from QALD (Liu et al., 2024b; Usbeck et al.,
2024) and LC-QuAD 2.0 (Dubey et al., 2019) datasets that conformed to our criteria, we employed
a filtering step using an LLM-based pipeline, described in detail in the Section B.2. The resulting
questions were reviewed by three authors to ensure adherence to the defined logical relationships and
criteria; we remark that careful revisions and considerable adaptations were required by hand, with
many candidate questions being discarded. The final filtered QALD and LC-QuAD 2.0 datasets
each contain 150 distinct question quadruples (300 for both), for a total of 600 questions in each
dataset (1 _,_ 200 across both).


**QAWiki.** For the QAWiki dataset, many _Q_ 1, _Q_ 2, and _Q_ 3 pairs were directly retrieved via
SPARQL queries. However, only 54 _Q_ 1– _Q_ 3 pairs and 951 _Q_ 1– _Q_ 2 pairs were available. Therefore, for the 54 _Q_ 1– _Q_ 3 pairs, the corresponding _Q_ 2 and _Q_ 4 questions were manually constructed.
Conversely, for the 951 _Q_ 1– _Q_ 2 pairs, _Q_ 3 and _Q_ 4 were manually derived. As with the other sources,


4DBpedia: [https://www.dbpedia.org/](https://www.dbpedia.org/)


16


this dataset was manually curated to include 150 distinct question quadruples per the required relations and criteria, totaling 600 questions.


**Synthetic.** The Synthetic dataset was generated using _gpt-4.1-2025-04-14_, designed to
produce 500 complete sets of _Q_ 1, _Q_ 2, _Q_ 3, and _Q_ 4 questions. Each set was then manually reviewed
by three researchers to ensure correctness, adherence to the inclusion criteria, and to eliminate duplicates. This dataset was heavily curated and revised by hand to include 150 distinct question
quadruples, totaling 600 questions. In many cases, the base question tuple generated by the LLM
served only as inspiration for the final handcrafted question.


The prompt used to generate the pair of questions _Q_ 1 and _Q_ 2 is the following:


1 Generate {number_of_questions_to_generate} pairs of diverse questions

about different topics, every pair of questions must be semantically
equivalent.

2 The answer to every question that you formulate must be a list of values,

not an ordered list, not a paragraph of text, not a boolean value,
and not a single number.


3

4 This is an example of a possible pair of questions:

5 1. How many regions of France are there? | How many regions does France

have?

6 Follow the following format to return the questions:

7 1. Question1 | Equivalent_Question1

8 Do not add any other kind of text except questions.’


Following the initial generation of the questions, three of the authors conducted a first round of
revision to verify their correctness and semantic equivalence. Based on this refined pool of questions,
we then instructed the LLM to generate, for each pair of equivalent questions ( _Q_ 1) and ( _Q_ 2), a third
question ( _Q_ 3) whose answer set constitutes a subset of the answers to both ( _Q_ 1) and ( _Q_ 2). The
prompt used for this task is reported below:


1 {dataset_of_question_pairs}

2 Starting from the provided dataset of question pairs, where each pair

consists of two semantically equivalent questions, generate a third
question whose answer is a subset of the answers to the original two
questions.

3 The answer to the generated question must be a list of values (not an

ordered list, not a descriptive paragraph, not a Boolean value, and
not a single number).


4

5 Use the following output format, and provide only the third question:


6

7 1. Broader_Question_from_the_dataset | Subset_Question


8

9 For example, given the pair: "What countries are in the EU?" | "What

countries are in the western EU?" the generated question would be the

subset question.


10

11 Return only the formulated subset question and no additional text.


Once the third questions were generated, we manually reviewed each ( _Q_ 3) and constructed a corresponding fourth question ( _Q_ 4) to satisfy the disjointness relation, defined as ( _Q_ 4 = _Q_ 1 _Q_ 3). The
final dataset comprises 600 question quadruples _\_


B.2 QUESTION PIPELINE


The multi-agent pipeline designed to filter questions that do not satisfy our inclusion criteria was
instructed to evaluate each input question against the defined conditions and, upon validation, to
generate the corresponding _Q_ 2, _Q_ 3, and _Q_ 4 questions. The LLM employed for this task was GPT4.1-2025-04-14. The initial datasets consisted of 320 questions from QALD and 30 _,_ 000 questions
from LC-QuAD 2.0. The multi-agent workflow comprises four sequential stages:


17


- **Agent 1 (Validate Q1):** checks whether the input question _Q_ 1 is well-formed and suitable;
aborts if invalid.

      - **Agent 2 (Generate Q2):** produces a companion question _Q_ 2 consistent with the intent and
constraints of _Q_ 1.

      - **Agent** **3** **(Generate** **Q3** **&** **Q4):** creates _Q_ 3 and _Q_ 4 such that the quadruple
( _Q_ 1 _, Q_ 2 _, Q_ 3 _, Q_ 4) satisfies the desired logical relations (e.g., equivalence, containment, or
disjointness).

      - **Agent 4 (Validate All):** verifies that ( _Q_ 1 _, Q_ 2 _, Q_ 3 _, Q_ 4) meet the required logical and formatting rules, corrects minor issues, and outputs the final validated set.


The process aborts if any step fails.


Agent1:


1 Evaluate whether the following question meets all of the following
criteria for acceptable answer types:


2

3 1. Returns a limited number of distinct answers (between 2 to 50).

4 2. Does **not** return a binary answer (e.g., "yes" or "no").

5 3. Does **not** return a single specific value (e.g., a date, name,
or number).

6 4. Does **not** require multiple answer dimensions (e.g., combining "
what" and "where" in the same question).


7

8 Question: "{question}"


9

10 Answer only with "Yes" or "No".


Agent2:


1 You are a rephrasing expert. Generate question Q2 which means exactly

the same thing as the original question but uses different syntax
wording.


2

3 Original Question: "{question}"


4

5 Format your response as:

6 Q2: ...


Agent3:


1 Given the original question below, generate Q3 and Q4 to ensure:

2 - Q3 and Q4 are objective,

3 - answer set of Q1 equal to union of answer set of Q3 and Q4,

4 - answer set of Q3 disjoint from Q4,

5 - answer set of Q3 and Q4 subset of Q1 (more restrictive).


6

7 Original Question: "{question}"


8

9 Format your response as:

10 Q3: ...

11 Q4: ...


Agent4:


1 You are reviewing four related questions.


2

3 Q1: {q1}

4 Q2: {q2}

5 Q3: {q3}

6 Q4: {q4}


7

8 Tasks:

9 1. Check if Q2 is equivalent to Q1.


18


**1000**

**1001**

**1002**


**1003**

**1004**

**1005**

**1006**

**1007**

**1008**


**1009**

**1010**

**1011**

**1012**

**1013**


**1014**

**1015**

**1016**

**1017**

**1018**

**1019**


**1020**

**1021**

**1022**

**1023**

**1024**

**1025**


10 2. Check if Q3 is a more restrictive version of Q1.

11 3. Check if the union of Q3 and Q4 covers Q1 (Q1 = Q3 + Q4).

12 4. Check all Q1˜Q4 are objective questions.


13

14 If everything is correct, answer only:

15 "Yes"


16

17 If not, return a corrected version in this format:

18 Corrected Q1: ...

19 Corrected Q2: ...

20 Corrected Q3: ...

21 Corrected Q4: ...


C VISUALIZATIONS


In this section, we present visualizations of the main results for answer-set inconsistency across the
models.


Figure 1 presents the Jaccard similarity _E_ 1 [SIM] _,_ 2 [,] _[D]_ 3 [SIM] _,_ 4 [and] _[E]_ 4 [SIM] _,_ 1 3 [.] [Figure] [1a] [illustrates] [that] [for] _[E]_ 1 [SIM] _,_ 2
_\_
there is high consistency across models, particularly for CtE and Oracle, with Jaccard similarity
values frequently exceeding 0 _._ 8. In contrast, Figure 1b demonstrates very low consistency for _D_ 3 [SIM] _,_ 4 [,]
indicating that this is the most challenging relation for LLMs. Even when adjusting the prompting
strategy, the improvement remains limited and not comparable to the performance observed for
_E_ 1 [SIM] _,_ 2 [and] _[E]_ 4 [SIM] _,_ 1 3 [.] [Finally,] [similar] [to] _[E]_ 1 [SIM] _,_ 2 [,] [Figure] [1c] [shows] [higher] [consistency] [for] _[E]_ 4 [SIM] _,_ 1 3 [,] [with]
_\_ _\_
scores improving progressively from Base to Oracle. Moreover, for some models (such as GPT-4.1
and DeepSeek-V3.1), the CtE strategy proves more effective than Oracle.


Figure 2 shows an alternative view of the bar chart, using a line chart and adding the consistency
rate. It is interesting to see how the **Base** strategy (blue dot) is almost always below **CtE** and **Ora** .
Furthermore, the sorting of the models follows the benchmark (White et al., 2025), but there is no
linear increase in performance as the model performance scales (for all strategies) as we expected.


D CLASSIFICATION TASK RESULT


Table 5 reports in detail the accuracy of the different models in identifying the different relationships. Across all models, the performance varies from _∼_ 60–91% in terms of average accuracy,
depending on the relation. Interestingly, the models perform best, on average, for the disjointness
relation ( _D_ 3 _,_ 4), with an average accuracy across models of 60%. The most difficult relationship to
_∼_
recognize for all models was _N_ 4 _,_ 1, with an average accuracy of 91%. A number of models achieve
_∼_
average accuracy above 90%, including some smaller models, with the best performing (taking ACC:
the average across relations) being Gemini-2.5-pro, followed closely by GPT-5, Grok-3-mini, and
other models, all of which achieve average accuracy across relations of _∼_ 95%.


E STATISTICAL SIGNIFICANCE


Table 6 presents the McNemar _p_ -values, which assess whether the **CtE** and **Oracle** strategies significantly outperform the **Base** strategy on the combined dataset. The McNemar test is a paired,
non-parametric test that considers only cases where the two methods disagree. In our setting, each
case is binary: we check whether the correct relation holds between the given answer sets. Across
all quadruples, this results in a total of 2,400 cases.


The table shows that the two proposed strategies offer statistically significant improvements for
answer-set consistency on almost all tested models. A statistically-significant improvement occurs
in the fewest cases for the _D_ 3 _,_ 4 relationship and the **CtE** strategy. The **Oracle** strategy, on the other
hand, shows a statistically significant improvement for all cases with only two exceptions: _D_ 3 _,_ 4 for
the models LLama-3.1-8b and GPT-4.1-nano.


Similarly, we examined whether certain LLMs significantly outperform others to verify hypothesis
_H_ 2. In this setting, each pair of LLMs is compared, and a corresponding _p_ -value indicates the


19


**1026**

**1027**


**1028**

**1029**

**1030**

**1031**

**1032**

**1033**


**1034**

**1035**

**1036**

**1037**

**1038**

**1039**


**1040**

**1041**

**1042**

**1043**

**1044**


**1045**

**1046**

**1047**

**1048**

**1049**

**1050**


**1051**

**1052**

**1053**

**1054**

**1055**

**1056**


**1057**

**1058**

**1059**

**1060**

**1061**

**1062**


**1063**

**1064**

**1065**

**1066**

**1067**


**1068**

**1069**

**1070**

**1071**

**1072**

**1073**


**1074**

**1075**

**1076**

**1077**

**1078**

**1079**


(a) _E_ 1 [SIM] _,_ 2


1

0 _._ 8
0 _._ 6
0 _._ 4
0 _._ 2
0
Base CtE Oracle


(b) _D_ 3 [SIM] _,_ 4


1

0 _._ 8
0 _._ 6
0 _._ 4
0 _._ 2
0
Base CtE Oracle


(c) _E_ 4 [SIM] _,_ 1 _\_ 3

|1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|Col22|Col23|Col24|Col25|Col26|Col27|Col28|Col29|Col30|Col31|Col32|Col33|Col34|Col35|Col36|Col37|Col38|Col39|Col40|Col41|Col42|Col43|Col44|Col45|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0_._6<br>0_._8<br>1<br>|0_._6<br>0_._8<br>1<br>|0_._6<br>0_._8<br>1<br>|0_._6<br>0_._8<br>1<br>||||||||||||||||||||||||||||||||||||||||||
|0_._2<br>0_._4<br><br>|0_._2<br>0_._4<br><br>|0_._2<br>0_._4<br><br>|0_._2<br>0_._4<br><br>||||||||||||||||||||||||||||||||||||||||||
||||||||B|ase||||||||||C||||||||||||||||rac|le||||||||||
||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||


|Col1|Col2|Col3|
|---|---|---|
|llama3.1:8b<br>GPT-oss-20b<br>GPT-4.1-nano<br>mistral-small:24b<br>llama3.1:70b<br>Gemini-2.0-fash<br>GPT-4.1-mini<br>GPT-4o<br>GPT-4.1<br>grok-3-mini<br>DeepSeek-V3.1<br>Gemini-2.5-fash<br>GPT-5-nano<br>DeepSeek-reasoner<br>Gemini-2.5-pro<br>GPT-5-mini<br>GPT-o3<br>GPT-5|ama3.1:8b<br>GPT-oss-20b<br>GPT-4.1-nano<br>mistral-smal<br>ama3.1:70b<br>Gemini-2.0-fash<br>GPT-4.1-mini<br>GPT-4o<br>PT-4.1<br>grok-3-mini<br>DeepSeek-V3.1<br>Gemini-2.5-f|ama3.1:8b<br>GPT-oss-20b<br>GPT-4.1-nano<br>mistral-smal<br>ama3.1:70b<br>Gemini-2.0-fash<br>GPT-4.1-mini<br>GPT-4o<br>PT-4.1<br>grok-3-mini<br>DeepSeek-V3.1<br>Gemini-2.5-f|


Figure 1: Visualization of answer-set (in)consistency across models


significance level (see Figure 3). This analysis was conducted only for the **Base** strategy, and the
results tend to support hypothesis _H_ 2.


F ANWSER-SET CONTRADICTIONS


In the body of the paper, we have looked at answer-set inconsistencies with respect to gold-standard
relations between questions. We further define _answer-set contradictions_ as the case where – irrespective of the gold standard – the relation predicted by a model _M_ for questions _Qi_ and _Qj_ does not


20


**1080**

**1081**


**1082**

**1083**

**1084**

**1085**

**1086**

**1087**


**1088**

**1089**

**1090**

**1091**

**1092**

**1093**


**1094**

**1095**

**1096**

**1097**

**1098**


**1099**

**1100**

**1101**

**1102**

**1103**

**1104**


**1105**

**1106**

**1107**

**1108**

**1109**

**1110**


**1111**

**1112**

**1113**

**1114**

**1115**

**1116**


**1117**

**1118**

**1119**

**1120**

**1121**


**1122**

**1123**

**1124**

**1125**

**1126**

**1127**


**1128**

**1129**

**1130**

**1131**

**1132**

**1133**


Figure 2: Per-relation consistency for each LLM and strategy. LLMs use IDs from Table 5.


Table 5: Accuracy of different LLMs on the relation classification task across relations. In bold, the
model that achieves the highest accuracy for the relation(s). ACC indicates mean accuracy.


**ID** **Model** _E_ 1 [ACC] _,_ 2 _N_ 3 [ACC] _,_ 1 _N_ 4 [ACC] _,_ 1 _D_ 3 [ACC] _,_ 4 _E_ 4 [ACC] _,_ 1 _\_ 3 ACC


_Average_ 86.01 66.25 60.64 **90.48** 77.55 76.19


hold for the answer sets that _M_ itself generates for _Qi_ and _Qj_ . In these cases, the model contradicts
itself. We describe here some measures and results that we extracted to analyze this issue.


F.1 MEASURES


We consider the following measures to quantify the self-contradictions of the LLMs in this setting.


21


_E_ 1 [CON] _,_ 2

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_D_ 3 [CON] _,_ 4

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_D_ 3 [SIM] _,_ 4

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_N_ 3 [CON] _,_ 1

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_E_ 4 [CON] _,_ 1 _\_ 3

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_E_ 1 [SIM] _,_ 1 _\_ 3

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_N_ 4 [CON] _,_ 1

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_E_ 1 [SIM] _,_ 2

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


**1134**

**1135**


**1136**

**1137**

**1138**

**1139**

**1140**

**1141**


**1142**

**1143**

**1144**

**1145**

**1146**

**1147**


**1148**

**1149**

**1150**

**1151**

**1152**


**1153**

**1154**

**1155**

**1156**

**1157**

**1158**


**1159**

**1160**

**1161**

**1162**

**1163**

**1164**


**1165**

**1166**

**1167**

**1168**

**1169**

**1170**


**1171**

**1172**

**1173**

**1174**

**1175**


**1176**

**1177**

**1178**

**1179**

**1180**

**1181**


**1182**

**1183**

**1184**

**1185**

**1186**

**1187**


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|
|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|
|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||


|ECON 4, 1 3|Col2|Col3|Col4|Col5|Col6|3|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
|4, 1<br>|4, 1<br>|4, 1<br>|4, 1<br>|4, 1<br>|4, 1<br>|3|3|3|3|3|
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||


_R_ _R_ (+) _R_ ( _−_ )


Figure 4: Comparison of R and R(+/–) consistency for Base (1st and 2nd rows) and CtE (3rd row).


22


1.0


0.8


0.6


0.4


0.2


0.0


A
B
C
D

E
F
G
H

I
J
K

L
M

N
O

P
Q

R


A
B
C
D

E
F
G
H

I
J
K

L
M

N
O

P
Q

R


E1, 2 [CON]


D3, 4 [CON]


A
B
C
D

E
F
G
H

I
J
K

L
M

N
O

P
Q

R


A
B
C
D

E
F
G
H

I
J
K

L
M

N
O

P
Q

R


N3, 1 [CON]


A
B
C
D

E
F
G
H

I
J
K

L
M

N
O

P
Q

R


N4, 1 [CON]


Figure 3: _p_ -value heatmap of LLMs in overall datasets for Base strategy.


_N_ 4 [CON] _,_ 1 (Base)

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_D_ 3 [SIM] _,_ 4 (Base)

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_E_ 1 [SIM] _,_ 2 (CtE)

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_D_ 3 [CON] _,_ 4 (Base)

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_E_ 4 [SIM] _,_ 1 _\_ 3 [(Base)]

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_E_ 4 [SIM] _,_ 1 _\_ 3 [(CtE)]

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_E_ 1 [CON] _,_ 2 (Base)

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_E_ 4 [CON] _,_ 1 _\_ 3 [(Base)]

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_E_ 1 [CON] _,_ 2 (CtE)

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_N_ 3 [CON] _,_ 1 (Base)

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_E_ 1 [SIM] _,_ 2 (Base)

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


_E_ 4 [CON] _,_ 1 _\_ 3 [(CtE)]

1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
A B C D E F G H I J K L M N O P Q R


**1188**

**1189**


**1190**

**1191**

**1192**

**1193**

**1194**

**1195**


**1196**

**1197**

**1198**

**1199**

**1200**

**1201**


**1202**

**1203**

**1204**

**1205**

**1206**


**1207**

**1208**

**1209**

**1210**

**1211**

**1212**


**1213**

**1214**

**1215**

**1216**

**1217**

**1218**


**1219**

**1220**

**1221**

**1222**

**1223**

**1224**


**1225**

**1226**

**1227**

**1228**

**1229**


**1230**

**1231**

**1232**

**1233**

**1234**

**1235**


**1236**

**1237**

**1238**

**1239**

**1240**

**1241**


Table 6: McNemar _p_ -value which shows the statistically significant improvements over the zero-shot
prompting with the Classification-then-Enumerate (CtE) strategy and Oracle (Ora.) on the overall
dataset. Asterisks denote statistical significance: *** for _p_ _<_ 0 _._ 001, ** for _p_ _<_ 0 _._ 01, and - for
_p <_ 0 _._ 05. The empty cell indicates a p-value _≥_ 0 _._ 05


**CtE** **Ora.**


**LLM** _E_ 1 _,_ 2 _N_ 3 _,_ 1 _N_ 4 _,_ 1 _D_ 3 _,_ 4 _E_ 4 _,_ 1 _\_ 3 _E_ 1 _,_ 2 _N_ 3 _,_ 1 _N_ 4 _,_ 1 _D_ 3 _,_ 4 _E_ 4 _,_ 1 _\_ 3


**Contradiction-free rates** To complement classification accuracy and answer-set consistency, we
measure whether a model is _internally_ _consistent_ with respect to the logical relation it _predicts_ .
Given two questions _Qi, Qj_, We say the case is _contradiction-free_ iff the predicted relation is
satisfied by the model’s _own_ answers; otherwise, a _self-contradiction_ occurs. We define the _self-_
_contradiction rate_ as the percentage of cases where a model’s predicted relation is not satisfied by
its own answers. This measure distinguishes between models that are internally consistent but wrong
and those that are internally inconsistent, thereby offering a finer-grained perspective on model reliability. We denote such rates for a model _M_ and relation _R_ as _R_ [CFR] ( _M_ ).


**Consistency** **by** **relation** **correctness** To further analyze model behavior, we distinguish consistency depending on whether the logical relation between two questions is correctly identified with
respect to the gold standard. This breakdown reveals how much consistency stems from correctly
recognizing the relation versus how much persists even when the relation is misclassified.


F.2 RESULTS


In Table 7, we present the results of the contradiction-free rates for five relations. We see that the
models do tend to contradict themselves, i.e., the answers they return for questions do not respect
the relation between the questions that they themselves predict. A lot of variance is seen across the
models, with large models showing more consistency.


Furthermore, we examine how consistency varies depending on whether the relation classification
is correct. Table 8 presents the results for consistency and similarity when the predicted relation
is correct, while Table 9 shows the corresponding results when the relation is incorrect. Figure 4
compares consistency across all cases, as well as positive and negative subsets. The results indicate
that incorrect relation classification generally leads to lower consistency.


G CAUSAL ANALYSIS OF LLMS INCONSISTENCY


As we observed, LLMs often give varying responses to identical inputs, potentially due to four
factors: decoding randomness, computational nondeterminism, order sensitivity, and data-level conflicts.


23


**1242**

**1243**


**1244**

**1245**

**1246**

**1247**

**1248**

**1249**


**1250**

**1251**

**1252**

**1253**

**1254**

**1255**


**1256**

**1257**

**1258**

**1259**

**1260**


**1261**

**1262**

**1263**

**1264**

**1265**

**1266**


**1267**

**1268**

**1269**

**1270**

**1271**

**1272**


**1273**

**1274**

**1275**

**1276**

**1277**

**1278**


**1279**

**1280**

**1281**

**1282**

**1283**


**1284**

**1285**

**1286**

**1287**

**1288**

**1289**


**1290**

**1291**

**1292**

**1293**

**1294**

**1295**


Table 7: Per-relation contradiction-free rates _R_ [CFR] (0–100 scale).


**LLM** **Str** _E_ 1 [CFR] _,_ 2 _N_ 3 [CFR] _,_ 1 _N_ 4 [CFR] _,_ 1 _D_ 3 [CFR] _,_ 4 _E_ 4 [CFR] _,_ 1 _\_ 3


G.1 DECODING RANDOMNESS


The most immediate cause of nondeterminism in LLM output is the probabilistic nature of token
generation. The sampling temperature _T_ (Ackley et al., 1985; Li et al., 2025) acts as a hyperparameter that influences the output distribution. Formally, the probability of selecting token _wi_ is:
_e_ _[zi/T]_
_P_ ( _wi_ ) = - [,] [where] _[z][i]_ [denotes] [the] [logit] [for] [token] _[w][i]_ [.] [When] _[T]_ _[>]_ [0][,] [sampling] [methods]

_j_ _[e][zj /T]_

such as top- _k_ and nucleus (top- _p_ ) sampling apply stochasticity to improve diversity, ensuring that
identical inputs can yield semantically distinct outputs. Theoretically, setting _T_ = 0 (greedy decoding) should guarantee determinism. However, empirical studies (Li et al., 2025; Renze, 2024;
Wang & Wang, 2025; Atil et al., 2024) demonstrate that outputs often remain unstable even with the
temperature set to zero, pointing to other underlying causes of nondeterminism.


G.2 COMPUTATIONAL NONDETERMINISM


Nondeterminism at _T_ = 0 can arise from the non-associative nature of floating-point arithmetic in
GPUs, where ( _a_ + _b_ ) + _c ̸_ = _a_ + ( _b_ + _c_ ) due to rounding errors. As Yuan et al. demonstrates, minor
numerical discrepancies in early tokens can cascade into divergent outputs, particularly in reasoningheavy tasks. Furthermore, modern optimizations, such as Mixture-of-Experts (MoE) (Masoudnia &
Ebrahimpour, 2014), FlashAttention (Dao et al., 2022), and continuous batching often process data


24


**1296**

**1297**


**1298**

**1299**

**1300**

**1301**

**1302**

**1303**


**1304**

**1305**

**1306**

**1307**

**1308**

**1309**


**1310**

**1311**

**1312**

**1313**

**1314**


**1315**

**1316**

**1317**

**1318**

**1319**

**1320**


**1321**

**1322**

**1323**

**1324**

**1325**

**1326**


**1327**

**1328**

**1329**

**1330**

**1331**

**1332**


**1333**

**1334**

**1335**

**1336**

**1337**


**1338**

**1339**

**1340**

**1341**

**1342**

**1343**


**1344**

**1345**

**1346**

**1347**

**1348**

**1349**


Table 8: Per-relation consistency rate and similarity when predicted relation is correct according to
ground-truth (0–100 scale)


**LLM** **Act** _E_ 1 [CON] _,_ 2 _N_ 3 [CON] _,_ 1 _N_ 4 [CON] _,_ 1 _D_ 3 [CON] _,_ 4 _E_ 4 [CON] _,_ 1 _\_ 3 _E_ 1 [SIM] _,_ 2 _D_ 3 [SIM] _,_ 4 _E_ 4 [SIM] _,_ 1 _\_ 3


in non-deterministic orders to maximize throughput. Atil et al. (2024) highlight that while fixing
random seeds can mitigate multi-GPU randomness, engineering optimizations (e.g., chunk prefilling
or prefix caching) frequently reintroduce nondeterministic behavior, making exact reproducibility
fragile even in controlled environments.


G.3 ORDER SENSITIVITY


LLMs exhibit hypersensitivity to input presentation due to the self-attention mechanism (Vaswani
et al., 2017). Because attention weights rely on positional encodings, trivial formatting changes such as reordering few-shot examples or altering whitespace - can have notable effects on output.
Liu et al. (2024a) identifies a “lost in the middle” phenomenon, where models fail to access information located in the middle of a long context window. Similarly, Lu et al. (2022) shows that the
chosen order of few-shot examples can change performance from near-random to state-of-the-art.


G.4 DATA-LEVEL CONFLICTS


Nondeterminism may also arise due to differences between pre-trained weights (parametric memory) and the provided context (non-parametric memory). Xu et al. (2024) identifies knowledge


25


**1350**

**1351**


**1352**

**1353**

**1354**

**1355**

**1356**

**1357**


**1358**

**1359**

**1360**

**1361**

**1362**

**1363**


**1364**

**1365**

**1366**

**1367**

**1368**


**1369**

**1370**

**1371**

**1372**

**1373**

**1374**


**1375**

**1376**

**1377**

**1378**

**1379**

**1380**


**1381**

**1382**

**1383**

**1384**

**1385**

**1386**


**1387**

**1388**

**1389**

**1390**

**1391**


**1392**

**1393**

**1394**

**1395**

**1396**

**1397**


**1398**

**1399**

**1400**

**1401**

**1402**

**1403**


Table 9: Per-relation consistency rate and similarity when predicted relation is incorrect according
to ground-truth (0–100 scale)


**LLM** **Act** _E_ 1 [CON] _,_ 2 _N_ 3 [CON] _,_ 1 _N_ 4 [CON] _,_ 1 _D_ 3 [CON] _,_ 4 _E_ 4 [CON] _,_ 1 _\_ 3 _E_ 1 [SIM] _,_ 2 _D_ 3 [SIM] _,_ 4 _E_ 4 [SIM] _,_ 1 _\_ 3


conflicts, where externally retrieved context contradicts internal training data, as a source of nondeterminism. Another issue is due to temporal misalignment; Nakshatri et al. (2025) finds that while
models may contain updated knowledge, they often struggle to retrieve it reliably against outdated
but strongly-weighted internal facts. Consequently, models may switch in an unpredictable manner
between “local optima,” prioritizing helpfulness over correctness (Xie et al., 2023).


H INCONSISTENCY ERROR ANALYSIS


We conducted a detailed analysis of the errors that continue to cause inconsistencies in the LLMs
responses, even when applying the CtE and Oracle mitigation strategies. We identified the following recurring error patterns, which can be classified into the categories listed below, excluding
those errors attributable to the stochastic nature of the model, which have already been discussed in
Appendix G.


    - Use of different terminology to refer to the same concept or entity: In some questions involving country lists, the model refers to the same country using different formulations, for
instance, “Spain” versus “Kingdom of Spain”. This is despite the fact that our prompting
guides the LLM to provide full names, avoid abbreviations, etc., to minimize such cases.


26


**1404**

**1405**


**1406**

**1407**

**1408**

**1409**

**1410**

**1411**


**1412**

**1413**

**1414**

**1415**

**1416**

**1417**


**1418**

**1419**

**1420**

**1421**

**1422**


**1423**

**1424**

**1425**

**1426**

**1427**

**1428**


**1429**

**1430**

**1431**

**1432**

**1433**

**1434**


**1435**

**1436**

**1437**

**1438**

**1439**

**1440**


**1441**

**1442**

**1443**

**1444**

**1445**


**1446**

**1447**

**1448**

**1449**

**1450**

**1451**


**1452**

**1453**

**1454**

**1455**

**1456**

**1457**


- Completeness of the response: For the case of relation _E_ 1 _,_ 2, we observed instances in
which the same question produced answers with different set cardinalities across different
executions.


    - Incapability of capturing implicit logic: As discussed in Appendix F.2, LLMs that misclassify the relation often show lower consistency in subsequent answers. When the relation is
incorrect, follow-up reasoning becomes unreliable. This highlights that the ability to correctly grasp the implicit logic and contextual relationships behind the questions is crucial
for achieving higher consistency.


    - Lack option for uncertainty: The strategy of permitting LLMs to answer “IDK” when uncertain generally improves consistency. Without this option, LLMs may attempt to answer
despite uncertainty, often leading to hallucinated responses that cause inconsistency.


I GENAI USAGE DISCLOSURE


GenAI is used for text refinement, assist code debugging, dataset creation.


27