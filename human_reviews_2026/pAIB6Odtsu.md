# Holistic Prompting: Joint Reasoning with Reusable States and Shortcut Discovery

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
Large Language Models (LLMs) have demonstrated significant capabilities in complex reasoning tasks, often employing frameworks like Tree of Thoughts (ToT) and Chain-of-Thought (CoT). 
However, such methods typically rely on trajectory-based state representations, where each state encapsulates the entire history of reasoning steps. 
For tasks with inherent solution spaces where overlapping reasoning paths frequently emerge, this inherently restricts the reuse of intermediate computations, leading to redundant exploration.
The effect is even more dramatic when samples can be solved jointly, as overlapping reasoning paths frequently occur across different problem instances.
We present Holistic Prompting, a novel framework that empowers LLMs to reuse intermediate results both within and across problem instances. 
Designed for tasks that exhibit reusable sub-structures, Holistic Prompting unifies shared access to intermediate thoughts with an active shortcut-discovery mechanism, enabling focused search between unsolved and solved subproblems and aggressively pruning reasoning paths. 
Our experiments show that reuse is highly profitable in ToT-style breadth-first search on the math puzzle Game24 and in AlphaZero-style Monte-Carlo tree search in retrosynthetic planning.
Here, Holistic Prompting achieves both higher success rates, while at the same time requiring fewer model invocations and outputs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Holistic Prompting (HP). The authors argue that conventional "trajectory-based state representations", where each state encodes its entire reasoning history, are redundant and prevent the reuse of intermediate computations, especially when tasks share overlapping subproblems. HP addresses this by processing multiple problem instances jointly within a shared And-Or graph structure, utilizing "collapsed states" that are Markovian and self-contained. This representation allows different reasoning paths to converge on and reuse identical subproblems, both within a single sample and across different instances. A core innovation of HP is an active "shortcut-discovery" mechanism, a type of inverse search that finds actions to connect existing unsolved subproblems to known, previously solved states, thereby aggressively pruning the search. Experiments demonstrate HP's effectiveness in the Game24 math puzzle and retrosynthetic planning.

### Strengths
1. The paper presents a novel idea on reusing reasoning states.
2. The proposed method is efficient in terms of tokens generated compared to ToT
3. The proposed method finds better performance over ToT
4. The shortcut discovery to intentionally arrive at already solved paths is interesting

### Weaknesses
1. The methodology seems to require common states that are exactly the same, so that different tasks could lead to common intermediate states, and the previous approach can be reused. Such tasks are rare, and the work only evaluates on 2 specific tasks.
2. The presentation is not clear. The descriptions are filled with jargon and not simple, intuitive explanations or illustrations of the underlying meaning.
3. The proposed methodology lacks a memory component. Thus, it is forced to solve all input problems simultaneously and could solve problems consecutively, storing intermediate results from prior steps.

### Questions
1. What other types of artificial or real-world problems can the proposed method solve?
2. Can a memory module be designed to store intermediate results for later reuse?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Holistic Prompting, a prompting framework that enables Large Language Models (LLMs) to reuse intermediate reasoning results both within and across problem instances. Existing multi-step reasoning frameworks, such as Chain-of-Thought (CoT) and Tree-of-Thoughts (ToT), usually use trajectory-based state representations. Each state encodes the full reasoning history, preventing the reuse of partial reasoning outcomes and leading to redundant computation. To address this, Holistic Prompting constructs a shared state space of intermediate thoughts, supporting cross-instance reuse and shortcut discovery between solved and unsolved subproblems.
The proposed framework is empirically evaluated on two tasks (Game24 and retrosynthetic planning), showing improved success rates.

### Strengths
This paper introduces a unified framework for reasoning reuse and shortcut discovery, which conceptually bridges CoT/ToT-style prompting with retrieval-augmented reasoning paradigms.

### Weaknesses
- Limited Evaluation. The experiments focus on two specialized domains. For example, Game24 is a quite old synthetic dataset (used in ToT). As most results of the experiment and the Appendix are reported on this dataset, it remains a question whether the proposed method can be applied to more practical domains, such as tool-use tasks [3] and coding tasks [2].

- Comparison with Retrieval-Augmented Methods. The paper claims conceptual similarity to Retrieval-Augmented Generation (RAG) but does not include direct comparisons or ablations against RAG-based baselines that could also leverage reusable intermediate results [1].

[1]. Buffer of thoughts: thought augmented reasoning with large language models. NeurIPS 2024.

[2]. SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution.

[3]. ToolRL: Reward is All Tool Learning Needs.

### Questions
The authors are encouraged to address the concerns above.

### Soundness
2

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
4

### Summary
The authors focus on a "memoization" opportunity in LLM reasoning. Instead of making the LLM do its reasoning from scratch for each problem, they aim to discover and reuse common intermediate steps/results.

### Strengths
- The proposed approach connects unsolved problem instances to already-explored reasoning paths from other samples, which is a good contribution. It is similar to dynamic programming and can support more efficient decoding and token usage.
- The results show that while the success rates are similar, the required steps (in their domain captured as reaction and molecule nodes) are fewer.
- The error analysis and ablation results are good.

### Weaknesses
- The effectiveness of this approach depends on the presence of reusable sub-structures in the targeted task class; it is not clear how much overhead is introduced when overlap is minimal (or non-trivial to adapt).
- Aggressive pruning, while controlling complexity, risks prematurely discarding valuable reasoning paths for atypical instances, potentially missing correct or novel solutions. The authors should think of situations where this can happen.
- There is a general assumption of using the LLM for batch or clustered problem solving rather than one-shot, highly individualized queries, potentially limiting its applicability in interactive or open-ended settings. (This is fine, but it needs to be acknowledged and mentioned).
- It would have been ideal if the authors could have connected this to RAG architectures and talked about situations where sub-structures are re-used even across problem settings or domains.

### Questions
- Please address the points raised in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new reasoning method for LLMs as an alternative to chain of thought (CoT) and tree of thought (ToT). The novel contribution is to build a graph of thoughts which is shared across input samples, where edges are built between intermediate reasoning states such that the states can be reused across different samples, thereby cutting down the length of reasoning traces. The states can only be reused as exact matches (as opposed to clusters or other abstractions). Two experiments are presented, the first a simple arithmetic problem using LLMs as the base predictor and the second a chemical synthesis problem, where existing domain-specific predictors were used instead of LLMs, due to LLMs giving high errors in this domain. The results on the arithmetic problem noticeably outperformed CoT and ToT with significantly fewer intermediate states and model calls. In the chemistry task, it matched the already high performance of the existing baselines but with fewer intermediate states.


The paper seeks to tackle an important problem and the idea of reusing intermediate reasoning states across inputs is definitely promising. However, in its current form, I am not convinced that this method will allow such an architecture to actually scale to standard LLM text-based reasoning problems, due to a combination of my intuition about the architecture and the lack of results on complex text-based domains. If the equality test was abstracted into some form of clustering or high-level concept correspondence, that may be a different story as this could potentially compress complex state spaces. For now, I don’t believe the method is competitive.

### Strengths
* The paper is clearly written
* It tackles the important and well-motivated problem of intermediate state representation and reuse in LLM reasoning
* The main methodological contribution, a sample-shared graph allowing reuse seems novel, though bear in mind that some very recent (last couple of months) methods tackle the reuse problem (e.g. metacognitive reuse, cross-question method reuse)

### Weaknesses
* The scope seems very limited. Since the matched intermediate states must be very/exactly similar and are low-level states without any abstraction, it is hard to see how this method could extend beyond problems with very simple input and intermediate token sequences. If the intermediate states were whole paragraphs or even sentences, how could these be reused at all?
* This paper is presented as a method for reasoning over LLMs, yet the second experiment didn’t use LLMs at all. If the problem precludes LLMs, you might as well use a domain-specific method rather than reasoning.
* I’m not sure why Game24 was tested with simple baselines, without considering higher performing ones like Graph of Thoughts or Self-discover, which may also be more relevant methodologically. It’s reasonably likely these methods would have matched your performance.

### Questions
See Weakness section.

### Soundness
2

### Presentation
2

### Contribution
2
