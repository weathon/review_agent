# REMem: Reasoning with Episodic Memory in Language Agent

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6, 4

## Abstract
Humans excel at remembering concrete experiences along spatiotemporal contexts and performing reasoning across those events, i.e., the capacity for episodic memory. In contrast, memory in language agents remains mainly semantic, and current agents are not yet capable of effectively recollecting and reasoning over interaction histories. We identify and formalize the core challenges of episodic recollection and reasoning from this gap, and observe that existing work often overlooks episodicity, lacks explicit event modeling, or overemphasizes simple retrieval rather than complex reasoning. We present REMem, a two-phase framework for constructing and reasoning with episodic memory: 1) Offline indexing, where REMem converts experiences into a hybrid memory graph that flexibly links time-aware gists and facts. 2) Online inference, where REMem employs an agentic retriever with carefully curated tools for iterative retrieval over the memory graph. Comprehensive evaluation across four episodic memory benchmarks shows that REMem substantially outperforms state-of-the-art memory systems such as Mem0 and HippoRAG 2, showing 3.4% and 13.4% absolute improvements on episodic recollection and reasoning tasks, respectively. Moreover, REMem also demonstrates more robust refusal behavior for unanswerable questions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes REMem, a memory + reasoning framework for LLM agents. Offline, it converts past interactions into a “hybrid memory graph” containing (i) timestamped natural-language event summaries (“gists”) and (ii) structured temporal fact triples (subject, relation, object + validity interval). Online, a controller LLM acts as an agent: it issues tool calls to retrieve and traverse this graph (semantic retrieval, temporal filtering, entity-centric queries), iteratively gathers evidence, and then answers. The system targets both episodic recollection (“what happened when X visited?”) and episodic reasoning (“who lived there first?”, “how long did that last?”). REMem is evaluated on four benchmarks (LoCoMo, REALTALK, Complex-TR, Test of Time) and outperforms strong RAG-style baselines, structured memory systems (Mem0, Graphiti, HippoRAG 2), and even full-context prompting. On temporal reasoning, REMem-I is the only reported method to surpass 90% exact match on Test of Time, and it does so with lower average token cost than feeding the entire history to the model. The system also shows better-calibrated refusals on unanswerable questions.

### Strengths
1. Conceptual clarity of task focus. The paper explicitly distinguishes two abilities—episodic recollection vs. episodic reasoning—and builds the system to address both. This sharpens what “memory for agents” should mean.
2. Consolidation-style memory construction. While your paper targets episodic memory, I would argue that what your methodology does is more akin to systems consolidation in the neuroscientific sense. Systems consolidation in the brain entails that raw episodic memories are gradually distilled is distilled into schematic structures, which can be recalled in a compositional manner that may or may not include temporal details. This is a compelling stance that links agent memory to well-studied ideas in neuroscience. I think the paper would benefit from explicitly linking to this - I have left it as a strength as I believe it is one of the more interesting ways to interpret your work.
3. Agentic temporal reasoning. The agent can iteratively call tools that (a) pull semantically relevant gists and (b) query the graph with temporal/entity constraints (“who occupied Apt B before Casey moved in?”), enabling before/after, duration, and ordering queries that vanilla RAG typically fails at. This also ties in to compositional retrieval as seen in the brain.
4. Empirical results. REMem variants consistently outperform baselines on four datasets, including challenging temporal QA. The gains over “full context” are large, suggesting real reasoning benefits, not just better recall. The approach is also more token-efficient and shows more calibrated abstention (“I can’t answer”) than prior memory systems.
5. Ablations and human eval. The paper ablates removal of gists vs. fact triples and shows both are important, especially for multi-hop temporal reasoning. It also shows that its LLM-as-a-judge metric correlates better with human judgment than F1/BLEU, which strengthens trust in the reported numbers.

### Weaknesses
1. Connection to memory consolidation. As mentioned earlier, I would argue that your methodology specifically targets memory consolidation - reframing from this perspective would give a stronger theoretical connection to neuroscience.

2. Lack or discussion on parametric memory stores. In the introduction you mention that “parametric memory, embedded in modelweights during pre-training or fine-tuning, lacks adaptability and contextual grounding in specific experiences”. A whole breadth of literature covers parametric episodic memory stores that exist outside of the model weights. Some of these methods focus on restructuring internal model states (see [1]), others focus on event compression into a fast weight matrix [2]. I would argue that such methods are relevant and should be commented on/compared to as baselines.

3. Lack of mathematical/formal specification. The methodology reads descriptively but lacks a clear formalism. There is no precise definition of the memory graph schema (node/edge sets, attribute spaces, temporal operators), the retrieval operators as functions over that schema, or the agent loop as a constrained decision process. Even a lightweight formalization (typed multigraph with timestamped edge attributes and well-defined filter/composition operators) would greatly improve clarity and reproducibility.

4. Graph construction is under-specified. Even with the figure, it remains unclear:
    * whether gist nodes and phrase/concept nodes live in a single unified graph or in two stores queried separately;
    * where timestamps/intervals reside (on gist nodes, on relation edges, or on event-instance nodes) and how overlaps/validity are represented;
    * how entity canonicalization across mentions works (e.g., “I”→“Alice”; handling other arguments like “Peter”);
    * how synonymy links between gists interact with temporal distinctness (risk of over-merging similar but distinct events).
Since retrieval operates over this graph, a precise schema is important.

5. Redundant first-stage retrieval functions. During initial recall the system exposes both a semantic retriever (embeddings) and a lexical retriever (BM25). Both serve the same purpose, yet the paper doesn’t justify why both are needed, how they’re combined, or quantify their individual contributions. If REMem benefits from simply having two parallel high-recall channels, that should be isolated and reported.

6. No end-to-end example trajectories. Given the centrality of a ReAct-style loop, the paper should print at least one full trace on a hard temporal query (sequence of tool calls, parameters, intermediate results, final answer). Without this, it’s hard to see how multi-step reasoning actually unfolds.

7. Breadth of evaluation. Most evaluation focuses on temporally framed questions (before/after, first/last, duration). It’s less clear how well REMem handles non-temporal personalization or causal/relational queries where “when” isn’t central, despite claims that gists capture richer situational context.

8. Memory complexity and system cost. The paper does not analyse space/time complexity of the memory store or retrieval: expected growth of gist nodes, phrase nodes, and relational edges with conversation length; frequency/impact of synonymy links; and the asymptotic/empirical cost of Stage-1 retrieval and graph exploration as histories reach hundreds of thousands or millions of tokens. A brief complexity discussion (e.g., storage per event, index sizes, average degree) plus empirical reporting (graph size, tool-call count, latency) would strengthen deployability claims. 

[1] Z. Fountas, M. A. Benfeghoul, A. Oomerjee, F. Christopoulou, G. Lampouras, H. Bou-Ammar, and J. Wang, “Human-inspired Episodic Memory for Infinite Context LLMs,” arXiv:2407.09450 [cs.AI], 2025.

[2] M. Zhang, S. Arora, R. Chalamala, A. Wu, B. Spector, A. Singhal, K. Ramesh, and C. Ré, “LoLCATs: On Low-Rank Linearizing of Large Language Models,” arXiv:2410.10254 [cs.LG], 2025.

### Questions
* Can you provide a compact formal schema: node/edge types; where timestamps/intervals live; how contradiction and multi-interval states are represented?
* Why are both semantic (embedding) and lexical (BM25) retrievers needed in Stage-1 recall, and how are they combined? What happens if one is removed?
* Can you include one full ReAct trajectory on a representative complex query (tool calls, parameters, intermediate results, final answer)?
* Could you add a small slice evaluating non-temporal questions to demonstrate breadth beyond time-centric tasks?
* Please report storage and runtime characteristics—e.g., growth of gists/phrase nodes/edges with history length, index sizes, average node degree, effect of synonymy edges on sparsity/density, average tool-calls per query, and latency curves.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper examines the memory capabilities of language agents, focusing on the challenges of episodic recollection and reasoning. To tackle these issues, the authors propose REMem, a two-phase framework for memory construction and utilization: (1) offline indexing, and (2) online inference. The offline phase converts experienced events into a hybrid memory structure consisting of (a) gists — concise representations that capture the key contextual details of each episode, and (b) facts — structured triples in the form of (subject, predicate, object). During online inference, the system retrieves and composes relevant information from the memory graph to support reasoning. REMem achieves significant improvements across various episodic recollection and reasoning benchmarks, demonstrating the effectiveness of explicitly modeling when and how events occurred. Additionally, the method enhances refusal behavior by reducing hallucinations when the necessary information is missing.

### Strengths
- **S1.** REMem provides a well-motivated and well-structured approach to episodic memory, explicitly incorporating temporal and contextual information that prior non-parametric memory systems typically lack.
- **S2.** The hybrid structure of gists and facts, the method for constructing this structure, and the use of multiple retrieval tools to enable multi-step reasoning together represent a novel contribution.
- **S3.** The paper evaluates the model on both episodic recollection and episodic reasoning tasks, demonstrating significant performance improvements. Additionally, the authors conduct a comprehensive analysis, including refusal behavior, error analysis, and human evaluation, which provides deeper insight into the strengths and limitations of the method.

### Weaknesses
- **W1.** Since gist and fact extraction rely heavily on LLMs, errors introduced during this process may propagate and negatively impact the entire REMem framework. In particular, gist extraction is an abstractive procedure that can be prone to hallucination, making error detection increasingly important as the memory store expands.
- **W2.** The distinction from prior KG-based memory systems (e.g., HippoRAG’s offline indexing and online retrieval pipeline) is not clearly articulated. While the incorporation of temporal modeling is highlighted as a contribution, further elaboration would help readers better understand the core novelty and value of the proposed approach.
- **W3.** The paper would benefit from more detailed descriptions and concrete examples of how its core components — such as retrieval tools and graph traversal — operate together during inference. Additional clarification could significantly enhance the reader’s understanding of the system’s practical workflow (see Q1–Q3).
- **W4.** (Minor formatting issue) In lines 211–215, the numbering format repeats “1)” twice. Using alternative markers such as “(a)” and “(b)” would improve readability.

### Questions
- **Q1.** How are synonymy edges between gist nodes utilized in retrieval or reasoning? Do they improve multi-hop recall or only expand local connectivity?
- **Q2.** In the retrieval tools, are semantic_retrieve and lexical_retrieve applied to both gist and fact entities? If so, does this imply that gists function at the sentence level, while facts operate at the keyword/entity level?
- **Q3.** During graph exploration, once a relevant node is identified, are both its associated gists and facts incorporated into the reasoning workflow?
- **Q4.** Fact extraction employs a schemaless approach for predicate construction. How does the system handle inconsistency in predicate expressions (e.g., synonymy, phrasing differences)?
- **Q5.** When gists or facts extracted by the LLM contain errors, how does REMem detect these issues, and what strategies could be used for correction or quality improvement?

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
4

### Summary
This paper proposes REMem, a two-phase episodic memory system for language agents: (1) offline indexing that converts experiences into a hybrid memory graph containing time-aware gists and structured facts, and (2) online agentic inference using carefully curated retrieval and graph exploration tools. The authors evaluate on four benchmarks spanning conversational QA (LoCoMo, RealTalk) and temporal reasoning (Complex-TR, Test of Time), showing 3.4% and 13.4% absolute improvements on episodic recollection and reasoning tasks respectively over three SoTA systems (Mem0, Graphiti, and HippoRAG 2).

### Strengths
- Presentation: Very well motivated and very well written paper (clear problem framing, separating episodic recollection from episodic reasoning and evaluates both). Figures are also nice and helpful.

- Good execution: solid engineering, reasonable baselines, ablations that show both gists and facts matter, and an efficiency comparison. Refusal analysis is an added bonus.

### Weaknesses
- (main concern) The paper evaluates on relatively controlled settings where proper knowledge extraction is assumed to work. The most critical and challenging aspect of the work (how robustly the gist/fact extraction generalizes to noisy, ambiguous, real-world text) receives minimal treatment in the paper. Once you have clean, well-structured graphs, the superior performance on episodic reasoning tasks becomes somewhat predictable rather than surprising. Unfortunately, the paper spent most of its efforts narrating a success story on this predictable part. As said earlier, the ablation studies are helpful but don't address extraction robustness. 

- The graph+LLM combination is now ubiquitous in LLM research. Numerous commercial and academic systems combine knowledge graphs with RAG/memory, including commercial products, and including for temporal reasoning. What makes REMem's specific formulation necessary or superior isn't sufficiently differentiated from this crowded landscape, leading to a feeling of : yet another LLM + graph paper.

- The paper rightfully motivates the need for episodic reasoning, and uses four datasets. However, although they do have episodic characteristics, none of these datasets was originally designed for episodic memory. Other recent benchmarks could strengthen claims about the method's episodic capabilities and generalization. e.g. (keyword "episodic memory benchmark" yields:
https://arxiv.org/abs/2501.13121
https://huggingface.co/papers/2410.08133 )

- The code is not available yet (written: upon acceptance)

### Questions
- A plethora of approaches mix knowledge graphs and RAG/memory retrieval in LLMs, including commercial products (graphrag, entigraph by memory but keyword "Graph LLM" "Graph LLM memory|rag" yieds much more), including temporal reasoning. Only a tiny subset is mentioned in the related work. Why were these methods omitted?  

- How robust is extraction to real-world noise? What percentage of extractions are incorrect or incomplete? How does performance degrade with noisier inputs? Can you provide extraction accuracy metrics on held-out data? How does the extraction perform on regular text outside the 4 datasets you tested?

- perhaps for the discussion section: LLMs succeeded where decades of knowledge graph methods failed, largely because their learned representations are more flexible than rigid symbolic structures. Why should we return to explicit graph extraction? The paper doesn't engage with this question. Is the implicit claim that episodic memory specifically benefits from explicit temporal structure?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce a framework named REMem for indexing and reasoning with episodic memory. A two step approach is proposed: first, an *offline indexing* convert experiences to a time-aware memory graph, then an agentic retriever (in a single step or in iterative steps) performs the *online inference*.

In the offline phase, a memory graph is created (from the "event statements") containing both: *gists*, that are event summaries with timestamp, and *facts*, that (subject, relation, object) triplets with additional time information (date or range). Example of gist "Alice bought 3 cows from Peter on Jan 17th" with the associated timestamp (17 Jan, 2024). Example of fact:  (subject Alice, relation purchased, object 3 cows)

In the online phase, the graph enables logic composition (e.g., time range filtering, neighbor exploration). An agent retriever uses retrieval, graph exploration and flow control tools to extract the answer of the question.

The framework is evaluated on 4 benchmarks including LoCoMo and Test of time, and compared against Mem0 and HippoRAG2 among others, on 2 tasks (recollection and reasoning). The experiments show that the methodology outperforms Mem0 and HippoRAG2 for both tasks, especially for refusal behavior and is on-par compared to full context performance.

### Strengths
- Grounding the events in time by design (compared to e.g., Mem0)
- Better performance over HippoRAG2 and Mem0 for most experiments
- Error analysis explanation in Sec. 6.4 regarding the remaining gap of performance in this experiment
- Correct evaluation metrics

### Weaknesses
- Limitations have not been highlighted
- No conflict detection and delete/update, compared to Mem0. This is not discussed at all in the paper
- Experimental settings can be discussed, in particular the chunking / short size of each session
- Modeling of events with grounding in time, but other aspects like spatial location are not systematically modeled
- No data/code provided at submission time

### Questions
My current grading for the contributions is "fair". I may reassess this aspect after discussing the limitations of the experimental settings, that are currently not discussed by the authors. In particular, the limitations of the work are only discussed w.r.t. the current experiment (like error analysis). But the main limitations regarding "more complex environments" (l. 486) are never explained:

- the chunking is simplistic (well segmented short chat sessions): the chunking of each event (event statement, chat session) is given by the benchmark, but in realistic scenario it might not be the case: the absolute date may not be located in the current chunk or linked to the current date (it may be implicit or not available); same for location, or other entities. Relative temporal expressions may not be resolved locally (while in the paper, the model seems to be forced to give an absolute date).

Q. what are the assumptions needed on the input data ("event statement") for ensuring the correctness of the methodology? How the "event statement" have been obtained/defined?

Q. the entities e.g. "Alice" can be recurring. How this is handled? Is there a risk of confusion for large graphs? Can the model grounds two different "Alice"? How to resolve conflicts? Another example can be for referring to a book and a movie sharing the same name but with slight adaptation.
 
- in-context setting gives the same performance.

Q. If I understood correctly, all the documents are integrated in a single large graph (that might be composed of multiple non connected subgraphs). Can you confirm (and highlight) that for building the graph, you consider one for the whole corpus of experiences (instead of one separated graph per chat session)?

Q. Can you describe the graph obtained? How many entities, how many links? Is it mainly a single connected graph (recurring entities), or an union of many disconnected graphs? Is it related to Table 2 (present but not referenced in the paper)?

Q. l.262: I don't get: "Due to its limited context window", what is it? If you are able to perform the experiment, why the context window is a limiting factor?

Q. Can you show a table with the time taken for building the offline graph as a function of the dataset size, and the time/cost for inference?


## Clarification questions and minor comments:

Q. Can you show a more complex facts graph in the appendix? The Fig. 2 represents two (s, r, o) relations. Is it possible for an object (like "3 cows") to be the subject in a following session? Is it possible to retrieve information from the object instead of the subject?

Q. l.200 what is "phrase node" there?

Q. What is the impact of the synonymy edges? (there is no ablation with and without it)? Is there a risk of confusion between events when the synonymy edges parameter increases?

I agree that on this experiment, the results are better overall, from the table, compared to Mem0 and HippoRAG2 for recollection.

Q. Does TISER or -I strategies applicable for HippoRAG2, for which performance seems comparable in Tab. 4 (against REMem-S)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
the paper presents REMem, a prompt-based approach for instructing LLMs encoding episodic memory as a graph of ``gists'' and facts; indexing is performed offline and is queried at inference with either a single-step or iterative procedure. (This decoupling raise serious concenrs on the usefulness of the approach)

author compare on 4 datasets against a set of alternatives (essentially, 
Mem0, Graphiti and HippoRAG) and naive full-context/oracle baselines.

analysis is quite detailed, with analysis of errors and comparison of 100 samples to LLM as a judge answers, though it lacks statistically relevant depth.

summarizing, the paper tackles an important problem. it offers a relatively simple  solution (some doubts remain concerning practical relevance), and performs some experimental evaluation, with good  (deep dive, error analysis, human comparison)  as well as bad (lack f statitical rigour) aspects -- overall, the paper is therefore in a borderline position

### Strengths
- balanced comparison with 4 datasets and 3 non-naive baselines
- sufficiently detailed analysis of results

### Weaknesses
(check details in the question section)

- an apparent oximoron?
- lightweigth contribution (with apparent oximoron?) 
- lack of statistical rigour in the analysis
- doubtful numbers reported, with lower than publicly reported baselines

### Questions
the paper tackles an important problem. it offers a relatively simple  solution, and performs some experimental evaluation. 


## an apparent oximoron?

two main comments 

- on the one hand,  the paper claims the contribution to be to "directly instruct the LLM" to create memories. I am not sure this is a real benefit, as at the end of the day, this is created through a prompt -- whereas an algorithm could be analyzed under some hypotesis on the distribution/properties graph of gists/facts, a prompt-algorithm through an LLM can only be observed through its output.

- on the other hand, the paper claims this indexing is still an offline phase, which created in my view an oximoron: if there is an advantage of having the LLM creating memories through a prompt, then the advantage is that this is part of its instruction set. if however, such indexing need to be done in batch over time, then (1) the advantage disappears (2) it is far less clear how these episodic memories are created, as the whole content has to enter context for Gist extraction + Fact extraction + Memory graph construction

if that's the case, then the whole construct seems a bit too artificial, as the system cannot be used to dynamically and naturally construct episodic memories (as it is presented), but can just be evaluated as an adademic toy 


## lightweight contribution 

irrespectively of the above conudrum, the paper contribution remains quite lightweight, as it essentiually leverages such offline-prompt-built-graph through an iterative (or signle shot, which in most cases work fine) retrieval procedure (akin to a walk into the offline-built graph)

as details of the contributions are lacking, and anyway not completely new as per authors own admiossion (e.g.,  graph is conflated with synonymous edges whose similarity increase above a threshold as in HyppoRAG), the whole construct remains quite heuristic

additionally, no analysis of the basic properties of the  constructed graph (connectivity, betweenness centrality, diameter, degree, anything) is addressed (let alone mentionoing an analysis of how such graph would differ from similar grapical abstraction of  graphRAG hyppoRAG). 

with such graph information, one could study properties of the random walk (from initial random seeds) as intermediate step before reporting results of the downstream task.

lacking the above, the methodological contribution is akin to "we have engineered an heuristic"

## lack of statistical rigour in the analysis

tables report numbers that are often very close and would need a statistically relevant analysis to tell if they are apart.  

for instance, my gut feeling is that Rem-I and Rem-S are statistically equivalent, i.e., a single shot suffice for most cases as the margin is too thin. this may of course depend on the hardness of the task, as intuitively a one-shot retrieval cannot be sufficient for more involved retrieval; howerver, part of the lack of advantage on Rem-I may be rooted in a poorly designed inference/retrieval strategies (as the gap from oracle is wider and so possibly statistically relevant).

the lack of statistical rigor leaves these questions unaddressed  

## doubtful numbers reported, with lower than publicly reported baselines

 Mem0 reports significantly higher than 25.1 F1 score reported in Tab 3.

 Mem0 reports significantly higher than 25.1 F1 score reported in Tab 3.

		Method Single Hop Multi-Hop Open Domain Temporal
		Mem0 	38.72	 28.64 	 47.65 	 48.93
		Mem0g 	38.09	 24.32 	 49.27	 40.28 

https://arxiv.org/pdf/2504.19413

### Soundness
2

### Presentation
3

### Contribution
2
