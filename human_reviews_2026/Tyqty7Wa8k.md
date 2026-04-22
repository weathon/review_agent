# LLM-Guided Search for Deletion-Correcting Codes

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 6, 2

## Abstract
Finding deletion-correcting codes of maximum size has been an open problem for over 70 years, even for a single deletion. In this paper, we propose a novel approach for constructing deletion-correcting codes. A code is a set of sequences satisfying certain constraints, and we construct it by greedily adding the highest-priority sequence according to a priority function. To find good priority functions, we leverage FunSearch, a large language model (LLM)-guided evolutionary search proposed by Romera et al., 2024. FunSearch iteratively generates, evaluates, and refines priority functions to construct large deletion-correcting codes. For a single deletion, our evolutionary search finds functions that construct codes which match known maximum sizes, reach the size of the largest (conjectured optimal) Varshamov-Tenengolts codes where the maximum is unknown, and independently rediscover them in equivalent form. For two deletions, we find functions that construct codes with new best-known sizes for code lengths n = 12, 13, and 16, establishing improved lower bounds. These results demonstrate the potential of LLM-guided search for information theory and code design and represent the first application of such methods for constructing error-correcting codes.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper adapts FunSearch to construct deletion‑correcting codes. The core idea is to search for a priority function that greedily builds a large independent set in a graph whose vertices are binary strings and edges connect strings sharing a length $n−s$ subsequence. The authors add a deduplication step (hashing the per‑sequence priority vector) to avoid storing functionally identical candidates. Empirically, they (i) match known optima for single deletion up to $n\leq 11$ and reach $VT_0$ sizes for larger $n$; (ii) report new lower bounds for two deletions at $n=12,13,16$.

### Strengths
Applying LLM (FunSearch) to coding theory. Casting single/multiple deletion code construction as independent‑set search with an LLM‑discovered priority function is a clean mapping and is described systematically.

### Weaknesses
1. The method is essentially *vanilla FunSearch* plus a simple deduplication based on hashing the priority scores a function assigns to the evaluation inputs. While *Fig. 12* shows 20 % duplicates without the filter and a faster trajectory with dedup, there’s no head-to-head comparison against recent FunSearch variants under matched resources to establish a genuine sample-efficiency gain. The contribution feels primarily engineering rather than algorithmic.  

2. Practical impact on coding theory is small.  For $s = 1$, the system mostly re-discovers $VT_0$ (or achieves the same sizes); the truly new outcomes are modest lower-bound improvements for $s = 2$ at a few lengths—$n = 12 : 32 \rightarrow 34$, $n = 13 : 49  \rightarrow 50$, $n = 16 : 201 \rightarrow 204$. Given the stated budget (~400 K functions per run; 350 GPU hours), the improvement-per-compute looks low.  

3.  On the combinatorial side: no comparisons to ILP / SAT / CP-SAT or LP-relaxation + heuristics that are common for maximum-independent-set / coding-theory graphs at these sizes. On the LLM-guided side: no matched comparisons to ReEvo / EoH style search with the same evaluator and budget. The only ablations are within-prompt / model variants, which do not establish that the framework is stronger than alternatives.

### Questions
1. Can you provide a fully formal argument for Appx. H, especially the conditions under which lexicographic tie‑breaking ensures that remainder $r=0$ sequences are always exhausted first at a fixed quotient? and discuss whether the discovered forms say anything structural for $s>1$.

2. Please compare to graph/LP‑based solvers on the same $n,s$, with wall‑clock and solution quality, to justify LLM‑guided search as the preferred tool.

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
2

### Summary
The paper studies the classical problem of constructing error-correcting codes that can correct a fixed number $s$ of deletions in binary sequences of length $n$. The proposed method builds on the existing \textit{FunSearch} framework, which uses an LLM-guided evolutionary search to identify effective priority functions. The authors further introduce a deduplication step to remove functions that differ only syntactically. Using this approach, the method discovers priority functions that yield codes with new best-known sizes for code lengths $n = 12, 13,$ and $16$, thereby improving existing lower bounds.

### Strengths
1. Combining an LLM with an evolutionary search to design priority functions for greedy code construction is an interesting and promising direction that links learning-based methods with classical coding theory.

2. The empirical results demonstrate that the proposed method identifies codes with new best-known sizes for code lengths $n = 12, 13,$ and $16$, thereby improving the existing lower bounds. This provides supporting evidence for the effectiveness of the approach.

### Weaknesses
1. A large portion of the proposed method appears to rely on the existing FunSearch framework, making it unclear what specific technical contributions or novel insights are introduced in this work.
2. The approach is computationally expensive --- the paper reports approximately 350 GPU hours for processing around 400k functions for small $n$. For larger $n$, the evaluator becomes infeasible due to the exponential growth of the sequence space, which significantly limits the method’s practical applicability. Although the authors acknowledge this issue, the limitation remains substantial.  
3. The description of some concepts in this paper, such as the islands/clustering logic and priority function representation, could be clearer.

### Questions
1. The LLM occasionally generates invalid or non-executable code. How frequently does this issue occur, and what specific strategies are used to detect and handle such cases?
2. The paper introduces a deduplication step to promote diversity among generated functions. However, this step appears to be relatively simple. Could the authors clarify the technical insights or contributions associated with this component?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces an LLM-guided evolutionary search approach for constructing deletion-correcting codes—an open combinatorial problem in information theory. Building upon FunSearch (Romera-Paredes et al., 2024), the authors adapt the framework to generate and iteratively improve priority functions that define greedy algorithms for building large deletion-correcting codes. Using models like StarCoder2 and GPT-4o mini, they rediscover known optimal codes (Varshamov–Tenengolts, VT) and find new best-known lower bounds for two-deletion cases. The method effectively searches function space rather than code space, demonstrating LLMs’ potential in theoretical code design.

### Strengths
1. First known use of LLM-guided program synthesis for designing error-correcting codes—a fresh bridge between information theory and AI-driven heuristic search.
2. Successfully rediscovered VT codes and found new best-known constructions for small two-deletion cases (n = 12, 13, 16).
3. The deduplication step for function-space search improves efficiency and diversity, addressing a key limitation of prior FunSearch implementations.
4. Code release and transparent description of distributed implementation (RabbitMQ-based) facilitate future research.
5. The idea extends to other types of combinatorial or coding-theory problems beyond deletions.

### Weaknesses
1. Evaluation cost grows exponentially with sequence length n, making the approach infeasible for medium-to-large codes. Usually in practice long codes are required, so this algorithm cannot be applied.
2. Success heavily depends on hand-crafted prompts and model choice; reproducibility across models and random seeds may vary.
3. Experiments mostly cover short sequences (n ≤ 25); performance for longer lengths or higher deletion counts is unexplored.
4. Please compare to the existing methods for deletion channels.

### Questions
1. How sensitive are the discovered priority functions to LLM architecture or sampling temperature?
2. Can the approach be extended to insertion- or substitution-correcting codes with minimal modification?
3. Is there a way to approximate evaluation (e.g., sampling or probabilistic scoring) to mitigate exponential scaling?

### Soundness
3

### Presentation
3

### Contribution
3
