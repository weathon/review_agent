# ABSINT-AI: Agentic Heap Abstractions for Abstract Interpretation

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 2, 8, 8

## Abstract
Static program analysis is a foundational technique in software engineering for reasoning about program behavior. Traditional static analysis algorithms model programs as logical systems with well-defined semantics, but rely on uniform, hard-coded heap abstractions. This limits their precision and flexibility, especially in dynamic languages like JavaScript, where heap structures are heterogeneous and difficult to analyze statically. In this work, we introduce ABSINT-AI, a language-model-guided static analysis framework that augments abstract interpretation with adaptive, per-object heap abstractions for Javascript. This enables the analysis to leverage high-level cues, such as naming conventions and access patterns, without requiring brittle, hand-engineered heuristics. Importantly, the LM agent operates within a bounded interface and never directly manipulates program state, preserving the soundness guarantees of abstract interpretation. To evaluate our approach, we focus on a soundness-critical task: determining whether object property accesses may result in undefined or null dereferences. This task directly models a common requirement in compiler optimizations, where proving that an access is safe enables the removal of dynamic checks or simplifies code motion. On this task, ABSINT-AI reduces false positives by up to 34% compared to traditional static analyses with fixed heap abstractions, while preserving formal guarantees. Our ablations show that the LM’s ability to interact agentically with the analysis environment is crucial, outperforming non-agentic LM predictions by 25%.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Abstract interpretation is a technique to prove properties of programs. It is a sound approach but incomplete, so it can often fail to prove a property even when the property is true. Such cases are deemed "false positives". One of the central goals in static analysis has been to reduce the number of false positives (that a technique generates), but without compromising soundness (i.e. the number of false negatives should continue to remain zero).

The false positives are a result of abstracting too much. Statically, it is difficult to determine when and how much to abstract. Not abstracting causes non-convergence (of abstract interpretation), but abstracting too much results in increased false positives.

In this paper, the authors propose to use large language models to help with certain choices that an abstract interpreter makes to determine how much to abstract. The main point is that LLMs can take cues from the variable and function names to know what to abstract and what not to. This is a very reasonable hypothesis, and this paper demonstrates that it actually works for "picking heap abstraction" during static analysis of javascript programs. 

The paper picks a set of 17 javascript programs on which it compares their approach with baselines that do not use LLMs (and use fixed abstractions). Their approach generates fewer false positives. There are ablations too that show that agentic LLMs do better than just using (non-agentic) LLMs with static prompt.

### Strengths
Strengths:
1. The use of language models to help with picking the right abstraction for static analysis is a good and reasonable idea.
2. The experiments clearly show that the proposed approach helps, and alternatives perform worse.

### Weaknesses
Weaknesses:
1. While the high-level approach is clear, the details are less clear since the paper lacks explanation on some of the most important bits in the paper. 
2. The second weakness is mentioned in the paper -- the approach is limited in scalability. It inherits this non-scalability from static analysis in general. Even the 17 benchmarks had to be carefully chosen (Lines 359 and 362). These are some significant bottlenecks for real use of the approach.

### Questions
Questions:
1. All safety violations can be deemed false positives only when the program is known to be safe. How was the safety of the 17 programs established? Was it by manual inspection/verification?
2. Expanding more on the Weakness 1 above, Algorithm 1 provides a good high-level description, but Lines 21 and 22 of Algorithm 1 were not clear. Lines 283-307 is where I expected more details. (I couldn't find those details in the Appendix either).

2.1 First, we have to compute Join at control flow merge points. This happens in loops, but also at if-then-else. So, if we are at if-then-else merge point and doing a Join there, would the "EXEC" action (Loop Execution) make sense there?

2.2 Also, what is meant by "interpreter executes on iteration of the loop" -- do you mean the "abstract interpreter" or is there some concrete "interpreter" also running collecting concrete states?

2.3 The largest program is 570 odd lines. Instead of using "INFO", could we not include the full program in the prompt? In other words, what is the set of "queries" that the agent makes "for program information" on Line 6 of Algorithm 1? Based on Lines 275-- 276, it seems there are two INFO queries: (1) variable inspection and (2) function introspection. Providing the full program can potentially answer (2), and presenting the full abstract state can answer (1) -- these both seem like natural things to provide to the LLM agent? Why not provide them rather than asking the model to query for them? Querying for loop execution seems reasonable.

2.4 Lines 286-303 provide too little detail to make a good judgement on whether the approach is reasonable.

3. Line 427-430: As I mentioned above, when you remove the ability to query and inspect, do you add the "program" and the "abstract state" in full to the context?

Minor typos and errors?:
- line 101: "without affecting soundness" -- I don't think you meant "soundness" there.
- line 134: mentions "even proposing domain-informed widenings" : Widening is tricky because if the LLM proposes a new predicate during widening, the abstract interpreter will not be able to handle that predicate going forward unless it also has abstract transformers for that predicate. Do you really mean "widening" here?
- line 161: "context-sensitive abstraction" -- what is "context-sensitive abstraction" and what is a "context-insensitive abstraction"?
- line 186: "Additional details on our abstract domain...":  Are the abstract domains fixed or are we picking from a collection of abstract domains? In abstract interpretation, if the abstract domain is fixed, then performing analysis on that domain is mostly fixed except for the "widening" step. So, is the LLM being used for widening? Or do we have multiple abstract transformers (for a fixed abstract domain) and we are using the LLM to pick one.
- line 405-408: "Most of the overhead ..." This is also unclear. Out of 500 seconds, 189 are spent on "agent interaction" -- what is "agent interaction" -- is that "querying the agent"? So, 500-189 is on the "abstract interpreter"? But TAJS and WALA abstract interpreters take only 20 seconds? So, even the abstract interpreter is now slower (rather than getting faster because it is presumably being smart about the abstractions it is making)? This is again one of those things that is unclear.
- line 449-450: Did the model have access to the name "conways game of life", maybe it is mentioned somewhere in the code, which helps it propose that abstraction?  
- line 462-463: I will be surprised if there is no prior work in program analysis that uses LLM in a way that preserves soundness. In program synthesis, for example, there is plenty of work that uses LLM or ML to only provide heuristics. In theorem proving, LLMs can provide the choice of proof rule, and overall soundness is preserved.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents AbsInt-AI, a language-model-guided static analysis framework for JavaScript that augments traditional abstract interpretation with adaptive, per-object heap abstractions. By leveraging an LLM to guide abstraction selection, merging, and widening decisions, the system dynamically tailors its analysis to program context. This approach reduces false positives while preserving soundness in the static analysis process.

### Strengths
Using domain knowledge of variable naming conventions, LLMs can provide knowldge that purely symbolic systems cannot infer.The method of using the LLM is not fully described and the results feel incremental.

### Weaknesses
The method of using the LLM is not fully described and the results feel incremental. In addition, the paper makes a strong claim of soundness of the resultant analysis but do not provide a proof of this. The LLM is providing hints and assumptions to the analysis that can be incorrect, and thus the outputs can potentially be unsound. The performance is poptentially slow (25x slower than alternatives and requires many iterations). The evaluation is performed on a very small dataset.

### Questions
The idea that an LLM can take advantage of naming conventions when doing program analysis and understanding tasks is well accepted and not new. Unfortunately, I couldn't fully understand exactly how the LLM is used to make merge or widening decisions. 

The description in Section 3.3 is vague and incomplete. Is the agent just given a menu of strategies to chose from for a particular symbol? This seems arbitrary and it is unclear if this is an optimal strategy, or why this even works.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents an LLM agentic framework for Java script analysis via abstract interpretation. The LLM agent is used to suggest heap abstractions  -- in particular before fixpoint computations in unbounded loops, where the choice of the abstraction influences the performance of the abstract interpreter. The agent inspects the current analysis state, including the heap, code, and abstraction history and selects abstraction strategies from pre-defined possibilities, preserving soundness.

### Strengths
The paper presents a clever use of agentic LLMs for sound program analysis for a challenging programming language such as JavaScript.
The use of LLMs and agentic frameworks in this context is new, although the work is similar in spirit with other approaches taht use AI to improve theorem proving.

The paper is well written and a pleasure to read.

The experiments show the merits of the approach compared to traditional symbolic approaches. They also highlight the use of the agentic bit.

### Weaknesses
no important weaknesses

### Questions
Please comment on the use of such a framework for other programming languages.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces an agentic framework (called ABSINT-AI)  that assists extends what appear to be sound, essentially conventional static methods by leveraging AI to choose among various heap abstraction approaches at different points in the program. This appears to be a novel way of combining the predictive but unreliable power of LLMs with traditional semantically sound methods for abstract interpretation. The results are promising for a representative analysis problem determining whether a program execution may result in undefined or null dereferences. In comparison with two prior approaches, the method developed in this paper achieves up to a 34% reduction in false positives while maintaining soundness. Further examination shows that the LM’s ability to interact agentically with the analysis environment is crucial, outperforming non-agentic LM predictions by 25%.

### Strengths
This is a novel approach, at least to this reviewer. Further. the results are compelling.  It is particularly attractive that the LLM is used only to choose between different abstraction methods; soundess of the analysis does not depend on the LLM.  The paper also seems to open up compelling directions for future work -- can the set of analysis choices be expanded, are there other ways to describe the program setting to the LLM, what are the performance/precision tradeoffs, etc etc.

### Weaknesses
WIthin the scope of the paper, there are no evident weaknesses. The work is described adequately, given page constraints, and source code to reproduce these resultrs is given in teh supplementary material.

### Questions
Is there additional related work on using LLMs to direct static analysis that should be highlighted for the reader? Perhaps using LLMs to choose proof strategies for verification could be considered analogous.

### Soundness
4

### Presentation
4

### Contribution
4
