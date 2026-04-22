# LLMs Process Lists With General Filter Heads

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 8

## Abstract
We investigate the mechanisms underlying a range of list-processing tasks in LLMs, and we find that they have learned to encode a compact, causal representation of a general filtering operation that mirrors the generic ``filter'' function of functional programming. Using causal mediation analysis on a diverse set of list-processing tasks, we find that a small number of attention heads, which we dub *filter heads*, encode a compact representation of the filtering predicate in their query states at certain tokens. We demonstrate that this predicate representation is general and portable: it can be extracted and reapplied to execute the same filtering operation on different collections, presented in different formats, languages, or even in tasks. However, we also identify situations where LMs can exploit a different strategy for filtering: eagerly evaluating if an item satisfies the predicate and storing this intermediate result as a flag directly in the item representations. Our results reveal that transformer LMs can develop human-interpretable implementations of abstract computational operations that generalize in ways that are surprisingly similar to strategies used in traditional functional programming patterns.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the mechanisms by which Large Language Models (LLMs) process lists, identifying a small set of "filter heads" in the middle layers that implement a general, abstract filtering operation akin to functional programming1. Using causal mediation analysis, the authors demonstrate that these heads encode a filtering predicate (e.g., "is this a fruit?") into their query ($q$) state at the final answer token. This query state then interacts with the key ($k$) states, which carry the semantic properties of the list items, to select the correct items. This predicate representation is shown to be portable and general, as it can be extracted and applied to different lists, formats, and languages to trigger the same filtering operation. The paper also discovers a complementary "eager evaluation" strategy: when the question (predicate) is presented before the list, the model stores an is_match flag directly in the items' representations, bypassing the "lazy" filter head mechanism.

### Strengths
- The work isolates a very specific but important phenomenon—list processing—and shows that LLMs exhibit strategy shifts with longer contexts, a behavior previously undocumented. 
- The paper connects internal mechanisms (representation compression) to observable generalization failures in a logical, evidence-based way, through causal interventions (activation patching): patching $q$, swapping $k$, and manipulating "flags" provide a very convincing, mechanistic explanation.
- Synthetic list tasks make it easy to disentangle algorithmic difficulty from length effects and the discovered behavior likely extends beyond toy tasks to real-world list-like inputs (e.g., tables, document parsing, code).

### Weaknesses
- While controlled tasks are valuable for analysis, it remains unclear whether similar behavior holds for natural language or multimodal data (although it is potentially fit for real-world data, the corresponding experiments are in lack).
- Experiments appear limited to LLaMA-70B and Gemma-27B, which are quite old models (in terms of the development of LLMs). How about the results with recent models (e.g., LLaMA3, Qwen3, ...)? Besides, does this mechanism hold for models of different size? (LLaMA-7B, LLaMA-13B and LLaMA-70B)
- The filter head mechanism explains the Select* tasks well, but the authors note it has low causality for Counting and CheckPresence tasks. This suggests these tasks (which are still filter-reduce operations) rely on a different, un-investigated mechanism. The paper's core claim is thus limited to selection-based filtering.
- The compared baselines (e.g., Function Vectors, Induction Heads): how did the authors implement them? The details are not clear. The reviewer thinks some approaches (e.g., Function Vectors) mentioned here are not fit for the tasks studied in the paper inherently.
- The logic of the paper is clear. However some acronyms might be quite improper (e.g., line 155, what is "DCM", I did not find the full name of DCM throughout the paper...). The clear definitions for tasks like SelectOne, SelectFrist are not in the main text, somehow confusing.

### Questions
Please refer to Weaknesses part.

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
The authors study how LLMs solve filtering tasks, in which the model must select one element from a list of options. Using differentiable binary masking, they identify the "filter heads" responsible for performing this operation. They show that these heads are reasonably general, and can transfer between different tasks, formatting types, etc. They also discovered that if the question is presented before the options, then the model relies on a different mechanism: storing flags for successful matches and aggregating based on them.

### Strengths
- Detailed study: ablations, transfer to other examples, and generalization across tasks/presentation types are all studied
- Based on causal analysis
- Mostly clear writing

### Weaknesses
- There are some unclear parts, see questions.

### Questions
- L155: What is DCM? This should also be described in the paper.
- L175: "We use a sparsity regularizer..." - Why a sparsity regularizer? What is the exact formulation of this regularizer? What is the target sparsity? Since the goal is to select the minimal set of heads, why not just minimize the logit magnitude, like in [1]?
- How does the first eq. in eq. 4 work exactly? M was defined earlier as "source run", "destination run", etc, which are concepts and not mathematical equations, but here it is used as a scalar to maximize over it. Is this the loss at each token? What is the index t? The index of the target token?
- L210:  The authors mention "for evaluation we consider last 2 tokens ({“\Answer”, “:” }) to reduce information leakage". Can you elaborate on this? Why is there a leakage? Why, considering the last 2 tokens solve it, and what is "considering" mathematically?
- L379: "queries encode "what to look for" (the predicate) and keys bring "what is there"" - this is a somewhat trivial conclusion: this is what the attention was designed to do.


[1] Csordas et al, 2021: Are Neural Nets Modular? Inspecting Functional Modularity Through Differentiable Weight Masks

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the mechanism used by Large Language Models to solve a variety of simple filtering tasks that involve a list and a predicate. The authors identify the set of heads responsible by learning a sparse binary mask for activation patching. The uncover set of heads generalizes well across similar selection tasks, especially (1) selecting the only matching item from the list, (2) selecting the last matching item, or (3) selecting the first matching item. They also study the impact of placing the question before the list, revealing a different mechanism reminiscent of *eager* evaluation. An interesting application is that the query of the identified heads can be used to obtain a training-free probe.

### Strengths
- The studied task (selecting an item from a list based on a given predicate) is important and relevant in the current state of LLMs interpretability.
- The comparison of placing the predicate before and after the options is insightful and sheds light on the ways that causal masking affects the learned mechanisms in LLMs.
- The mechamism is explained clearly. The paper is easy to understand and figures are good.
- Experiments are comprehensive.
- The training-free probe is an interesting and important result.

### Weaknesses
1. The set of heads identified is quite large and the precise role of each head is unclear
2. The identified heads have low portability to aggregation tasks (presence checking, counting) which suggests a limited relevance.

### Questions
1. How many heads are identified for each of the tasks?
2. Do LLMs perform better when the question comes before or after the list?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces an internals-based explanation for LLM list processing mechanisms. They find evidence for "filter heads" in Llama 70B, which encode representations of a predicate relevant for a list processing query (e.g. "is this fruit" given the query "find the fruit"). They show that causal interventions on these filter heads (e.g. patching query states) result in the expected behaviors.

### Strengths
* The paper is very well written, and has some really clean experiments
* The experiments cover their bases quite well (good ablations, good generalization experiments)
* I think the results from Section 5 are particularly insightful and also just really cool.
* I think there's still a lot of value in doing this kind of mech interp :)

### Weaknesses
* I would like to see more tasks, especially given the CheckPresence results; it seems like this is a really nice explanation but I'm worried it won't actually generalize / you maybe got lucky with the tasks you chose.
* Similarly, I don't know what's going on with cross task transfer for SelectFirst/SelectLast. I think this deserves more time.

### Questions
See weaknesses above.

### Soundness
4

### Presentation
3

### Contribution
3
