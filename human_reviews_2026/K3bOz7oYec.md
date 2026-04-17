# O-Mem: Omni Memory System for Personalized, Long Horizon, Self-Evolving Agents

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 2, 2

## Abstract
Recent advancements in LLM-powered agents have demonstrated significant potential in generating human-like responses; however, they continue to face challenges in maintaining long-term interactions within complex environments, primarily due to limitations in contextual consistency and dynamic personalization. Existing memory systems often depend on semantic grouping and the retrieval of past interaction groupings, which can overlook semantically irrelevant yet critical user information and introduce retrieval noise. To address these issues, we propose O-Mem, a novel memory framework based on active user profiling that dynamically extracts and updates user characteristics and event records from interactions. O-Mem supports hierarchical retrieval of persona attributes and topic-related context, enabling more adaptive and coherent personalized responses. Additionally, we introduce a new dataset designed to evaluate personalized long-text generation in memory-augmented agents. Experiments across three personalized tasks demonstrate that O-Mem consistently improves long-term human–AI interaction by scaling memory-time within interactions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents O-Mem, a memory framework designed to help LLM-based agents maintain long-term, personalized conversations. Unlike chunk-based semantic retrieval systems that simply group and retrieve messages, O-Mem builds and updates a user profile, event memory, and topic memory. These three modules are queried together to form context at every turn. The method is evaluated on LoCoMo, and a new small benchmark called Personalized Deep Research Bench, showing improved personalization and efficiency.

### Strengths
- Clear motivation and good problem definition.
- The tri-memory structure (working, episodic, persona) is easy to understand.
- Empirical results are consistent across several benchmarks and show clear efficiency improvements (up to 80% lower latency and 94% fewer tokens).
- The token-controlled ablation is a nice touch to show that gains are not simply due to longer context.
- Well-written and logically structured paper; figures and tables support the narrative.

### Weaknesses
- Episodic retrieval is defined by selecting a single clue word, but the paper does not specify how it behaves for unseen or multi-word clues.
- The retrieval hyperparameters (top-k, thresholds, similarity cutoffs) are not clearly stated. Since efficiency is a main claim, this missing detail makes replication difficult.
- The new Personalized Deep Research Bench dataset is relatively small and not publicly available, which limits the impact towards the community.
- Reproducibility: while the paper provides prompts, the code is not made public.
- No detailed error analysis or failure cases are presented (e.g., wrong persona merges, noisy clues, context drift).
- The deep research benchmark evaluation relies on LLM-as-judge scores without human evaluators, which weakens the reliability of reported gains.

### Questions
- Provide all retrieval parameters (k, thresholds, backend) and report multi-seed variance.
- Give more information about the new dataset (size, annotation, examples).
- Add a short qualitative error analysis to understand where O-Mem fails.
- Include a short paragraph on privacy and limitations of method.
- Consider adding human evaluation rather than purely LLM-based

### Soundness
2

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
3

### Summary
This paper proposes O-Mem, a hierarchical memory framework for LLM-based agents that dynamically builds and updates user profiles through active persona extraction and event recording. By combining persona, episodic, and working memories, the system enables long-term personalized reasoning and shows consistent improvements over existing memory systems on three benchmarks.

### Strengths
1. The motivation of the paper is clear: it clearly illustrates why existing chunk-based or semantic retrieval memories fail at dynamic personalization and positions O-Mem as a principled solution.

2. The integration of persona, episodic, and working memories into a unified retrieval pipeline (Eqs. 8–12) with active LLM-driven updates is interesting and reasonable.

3. Experiments across three datasets (LoCoMo, PERSONAMEM, Personalized Deep Research Bench) show consistent gains.

### Weaknesses
1. Several metrics rely on “LLM-as-a-Judge,” which introduces bias; no human validation or inter-rater reliability checks are reported.

2. The efficiency and ablation results are strong, but in-depth qualitative analyses of what the model remembers or how errors occur are missing. 

3. More discussion on how often the persona-update operation (Op(ai) / Op(ei)) introduces noise or incorrect updates, and how error accumulation is mitigated, is not thoroughly discussed.

### Questions
Some minor suggestions on the presentation:

- “Sysytem” → “System” in Section 2 heading

- Some inconsistent spacing in equations (3) and (4) (“ApplyOp(Pf , ei, Op(ei))”).

- “we propose O-Mem” repeated across Abstract and Intro

How sensitive is O-Mem to the choice of the base embedding model (e.g., all-MiniLM-L6-v2)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates the personalized response generation task in conversational ai scenario. Different from existing memory systems that relies solely on semantic retrieval, the authors propose a new memory system named O-Mem. O-Mem consists of three memories: user persona memory, working memory, episodic memory, and coupled with three retrieval strategies for obtaining context when responding to a user’s query. A new dataset is constructed and introduced.

### Strengths
•	A new dataset is introduced

•	The designed memory system differentiates user past histories into three different types which aligns with personalization purpose

### Weaknesses
•	The presentation of this paper needs to be improved, e.g., the tables in experiment sections are sparse; a lot of content such as in line209-214 in Section 3 should be moved to appendix; the formulas in Section 3.2 and Section 3.3 are not necessary (they can either be put into appendix or in line with text), lacking more organized descriptions; in line 34-41, this would better be put into related work (intro would better contain more conclusion-like statements to explain current solutions to the limitations rather than stating who did what). There are more but I stop here, readers can check more in this stream.

•	The O-Mem constructs three memories (user persona memory, working memory, episodic memory) via LLM and perform retrieval over the three memories when a user query raised. This is not new and the experiments lack the latency/cost comparisons on memory constructions, especially in line72-73 O-Mem seems actively updating user memory in the runtime.

•	In line 82-85, the limitations of chunk size of retrieval etc. are not your contributions, these are commonly research questions in the community.

•	Are all baselines the same embedding model i.e., all-MiniLM-L6-v2 as O-Mem did? If they are, more embedding models should be tested as in their original design since this would help readers to understand if the improvements brought by the O-Mem design or the model used. If they are not, then the comparisons are not fair.

•	Datasets statistics such as utterance length, dialogue session numbers etc. are missing, even in the appendix. If the dataset is constructed casually and lacks evaluation/validation, this contribution is weak.

### Questions
•	In Table 1, LangMemory beats MemoryOS in Open category with GPT4.1 but the situation reversed when with GPT-4o-mini, why is that?

•	In Table 1, O-Mem is suboptimal in Open Category questions, why?

•	In Table 1 temporal category with GPT-4o-mini, is it possible that the F1 score of O-Mem is better than MemoryOS while the B1 score is suboptimal than MemoryOS?

•	The evaluation metrics are not convincing, the BLEU score is not a good indicator, how about BertScore/Faithfulness?

•	LoCoMo contains images, how do the authors deal with them?

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces O-Mem, a memory framework aimed at improving long-term human-AI interaction by leveraging dynamic user profiling and hierarchical memory retrieval. The system aims to enhance personalization by maintaining evolving user profiles and incorporating them into response generation.

### Strengths
1. The combination of memory systems with dynamic user profiling is an interesting and promising direction.

2. The paper is well-structured and clearly written, making it easy to follow the technical concepts and their practical applications.

### Weaknesses
1. The main innovation in the paper appears to be the creation of a dedicated persona memory system that retrieves information based on user attributes. This seems like an incremental improvement over existing systems, with limited novel insights or breakthroughs. The contribution could be seen as a modification.

2. The paper primarily evaluates the framework on only two models—GPT-4.1 and GPT-4o-mini. Expanding the evaluation to include other model families ( both open-source and closed-source, chat and reasoning models from different model families ) would provide a more comprehensive understanding of the framework's generalizability and robustness.

3. The Personalized Deep Research Bench, a self-constructed key benchmark for evaluating the framework, lacks detailed construction explanations. It would be beneficial for readers to understand how the dataset was created and how it compares to existing benchmarks in terms of coverage and difficulty.

4. Minor: In Table 1, there is an inappropriate use of bolding. The temporal F1 scores for MEMOS should be marked as the best model.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
2
