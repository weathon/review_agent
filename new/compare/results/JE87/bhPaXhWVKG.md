# Review

## Summary
This paper presents MermaidFlow, a framework that transforms agentic workflow generation by encoding workflows as statically typed, semantically annotated, and compiler-verifiable graphs using the Mermaid language. MermaidFlow defines workflows as declarative graphs, where nodes represent prompting agents and edges specify information flow. This high-level representation enables structural and semantic properties with static verification, e.g., structure feasibility, and type-safe connections, can be enforced at the graph level, offering a clear plan that is both human-readable and programmatically analyzable.

## Soundness
2

## Presentation
3

## Contribution
2

## Strengths
1. The writing is clear and easy to understand.
2. The method is simple and easy to implement.

## Weaknesses
1. The novelty is limited. MermaidFlow is very similar to AFlow, except that MermaidFlow uses a better graph description language and some common sense to improve the search process. It is foreseeable that using a better graph description language will improve the performance of the search algorithm.
2. The experimental results are not convincing. The baseline in the experiment is relatively weak. For example, the AFlow used in the experiment is only the weakly constrained version. The strongly constrained version of AFlow can achieve an 80% success rate on MATH. In addition, the experiment lacks the comparison of the number of tokens used, which is important for evaluating the efficiency of the search algorithm.
3. The value of the method is questionable. MermaidFlow requires the pre-definition of the node types and edges types, which limits the generalization ability of the method. In addition, the method requires the LLM to have strong skills to design a good workflow. For example, the workflow in Figure 4 requires the LLM to have strong skills in designing an ensemble. However, the general LLM lacks such ability, and it is difficult for the LLM to design a good workflow without prior knowledge. Therefore, it is difficult to say that the method can be widely used.

## Questions
See weakness.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4