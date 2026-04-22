# Causal Path Tracing in Transformers

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
We propose a causal path tracing framework to understand how information causally flows through the internal structures of transformers for a given decision. By unfolding each block into a causal graph of path nodes and applying a minimality-based subset search, our method identifies all possible causal paths within each block, with polynomial-time complexity on average. Furthermore, we demonstrate the reliability of a union-based causal path reference strategy, enabling efficient and reliable causal tracing throughout the model. The key contributions of this work are: (1) an automated, efficient framework for causal path tracing that exhaustively searches paths along direct dependencies; (2) theoretical and empirical validation demonstrating exhaustive search with polynomial-time complexity on average; (3) experimental findings showing that self-repair effects occur far less frequently along the identified causal paths, that certain paths are uniquely activated for specific classes, and that the traced paths are both accurate and faithful.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a causal path tracing framework for explaining decisions of transformer models. The path-level approach allows for causal referencing, a technique that avoids incorrect explanations due to the self-repair behavior of these models. To make the path-level approach feasible, it employs a minimality-based subset search for identifying causal paths. Experimentally, the paper finds that self-repair happens less frequently along the found causal paths and that these paths accurately recover the model decision.

### Strengths
The paper is generally well-written. It extends a recent line-of-work on causal interpretability of transformer models, certainly a fundamental topic in the current AI literature. It has a clear story of making path tracing methods efficient by doing a minimal subset search which is effective for keeping the run-time manageable in practice. Moreover, it provides evidence on small-ish models to corroborate main claims (e.g. that self-repair happens outside of "causal paths").

### Weaknesses
The main weakness of the paper is that it lacks clarity and mathematical rigor. 

Many of the main notions in the paper are defined rather loosely and their relation to standard graphical or causal notions seems unclear to me. For example, even the main definition of a path as "... a sequence of node sets connected through [...] edges" is not fully clear to me. Similarly, causal notions are also introduced quite loosely and hand-wavy. Connections to Pearl's theory of causality (causal paths) are not formalized (despite Pearl being cited as source of inspiration) and remain vague.

The treatment of the main algorithmic improvement (Algorithm 1) also uses theoretical notions too loosely. While the idea to work with minimal subsets and thus improve the practical run-time is nice (and it's intuitive that this yields run-time improvements), I find Theorem 1 to be too general to give much actual insight in the context of transformer models (with the artificial assumption that causal node sets are independently Bernoulli distributed). Moreover, it is claimed that the problem at hand is "NP-complete" without this being corroborated by a proof. The statement "... this subset search problem is NP-complete" is a bit of an inaccuracy in itself with NP containing decision problems and not search problems, but apart from that it is per se not clear (despite being of course plausible) whether the studied problem is NP-hard (for transformer models) and such a claim would necessitate a formal problem statement and a reduction from an NP-hard problem.

This also makes it hard to judge the merit of the method in general and compared to related work. I am not an expert in explainability (also reflected in my confidence score) and this may add to this issue, however, it remains at least partially unclear what (practical) advantages the proposed methods offer. Self-repair scores are low for "causal paths", but it would be interesting to learn more about the concrete insights this delivers or where this shows related interpretability methods to be lacking and giving (causally) incorrect interpretations.

### Questions
1. Regarding the definition of paths: Why are node sets connected and not individual nodes and what does "connected" mean here precisely? How does this notion compare to the standard definition of paths in directed graphs (as it seems to differ from that)? 

2. Regarding causal notions: Is it possible to more precisely characterize the connection to Pearls causal theory? For example, can definition 4 be rephrased with regard to causal effects and the do-notation (e.g. 4a and 4b)?

3. Regarding the NP-hardness of causal subset search: Can you provide a proof sketch of the NP-hardness/NP-completeness of this problem (for transformer architectures)?


Suggestions (things I stumbled over, you may or may not heed these):

- Def. 1: "each node is deterministically computed from its parent nodes. [...] conditionally independent of its non-descendants": clarify how "conditionally independent" can be interpreted despite deterministic relations
- Def. 4: "all nodes between V and the output", this (and other notions) can be made more precise by building on standard graph terminology, maybe you mean all nodes on a path between a node in V and the output?
- Citations: you cite the arxiv version of some papers that have been published (eg Rushing, Nanda 2024 and Zhang, Nanda 2023)

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
The authors present a new technique for tracing the reasoning behind Transformer-based decisions by identifying their core causal pathways. The method operates by deconstructing Transformer blocks into a graphical representation of path nodes. It then employs a minimality-guided causal search across these blocks to isolate the precise chain of operations responsible for a specific output. Experimental validation confirms the method's efficiency and its superiority over existing baselines, proving that the identified pathways are integral to the model's decision-making process.

### Strengths
1. This work tackles a core issue in modern AI which is of significant current interest to the research community.
2. The paper's construction is logical, and the core ideas are articulated with clarity.
3. A key positive finding is the strong empirical evidence showing that the discovered causal paths are non-trivial and genuinely drive the model's predictions.

### Weaknesses
1. A significant limitation is the gap between the proposed interpretability method and its concrete application. The authors do not persuasively demonstrate how this tracing technique translates into tangible benefits for downstream tasks. The evaluation is more focused on the phenomenon of interpretability itself, rather than its practical consequences.
2. The methodological contribution regarding the average-case polynomial-time search appears derivative. It leans heavily on the causal minimality condition from the established Halpern–Pearl framework, which makes the proposed search algorithm feel more like an incremental extension than a novel breakthrough.

### Questions
1. How do the authors envision this framework bridging the gap to real-world applications? For example, can it be practically employed to diagnose a model's erroneous prediction in a specific case, or could it streamline the process of auditing a model for biased decision-making?

### Soundness
4

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
3

### Summary
This work focuses on interpreting the internal mechanism of a deep neural network architecture known as transformers via causal referencing, iteratively evaluating each component conditioned on priorly identified causal components along direct computational dependencies. Particularly, the paper addresses the issue of combinational complexity in path-level patching for causal referencing by proposing an efficient framework for tracing causal paths within each block of a transformer. Each block is treated as a path node and the whole transformer can be viewed as a causal graph. It introduces a minimality-based subset search strategy for identifying all possible causal path node combinations per block. This strategy is claimed to reduce the exponential complexity to polynomial time on average.

### Strengths
-	The paper attempts to solve an NP-complete problem.
-	It gives the first path-level patching method that feasibly enumerates all decision paths given an output.
-	It uses some simple tricks to rewrite the expression of the nonlinear function in a transformer to gain computational advantages.
-	The experiment supports that the search algorithm can still be useful in certain language models despite an unknown $p$ in Theorem 1.
-	The experiment shows some interesting use cases of the causal node sets.

### Weaknesses
- The writing needs significant improvement.
    - The paper does not define causal node sets before it uses this term in the definition of causal path.
     - The paper mentions things out of the blue without context. For example, line 67 says ‘self-repair occurs primarily…’, this is not clear about what the message is without describing what self-repair means. It then goes on to say ‘thus, the path contains information essential…’ to emphasize how important the causal path is. It is hard to understand the connection between the two statements. Subsequently, in line 069, it suddenly describes how causal paths are uniquely associated with specific classes. It is unclear what these specific classes mean. It implicitly suggests classification as the main goal of the transformer. 
     - Definition 4 becomes confusing after reading lines 128-131. Please see the questions for details. 
     -  Lines 135-136 give a very confusing statement regarding the definition of causal node set and the definition of causal node set (possibly not minimal). First, by definition 4, if $V$ is not minimal, $V$ is not a causal node set. The paper uses the same term with the addition ‘(possibly not minimal)’ to describe a set that does not meet the conditions of Definition 4.
     - Line 134 mentions ‘sufficient intervention’, but it is defined in line 147. 
     - Definition 8 is not even mentioned in the main paper when it is used in line 301 as a part of Theorem 2.
- Some claims are not well-supported. Many definitions are introduced to support the design choice, but not justified why the definitions are necessary to achieve the goal. 
- The last term named ‘Attention + MLP (H paths)’ in equation 2 is inconsistent with the derivation provided in Appendix E. The term $\frac{b_{oa}W_{ln2}^{\top} W_{lm}^{\top}  \mathbin{\circ} D_{B} W_{om}^{\top} }{H}$ is missing from the equation in the main paper. 
- The pseudocode in Algorithm 1 seems to suggest the algorithm is exponential at first glance. The argument of having polynomial time complexity is weak given that $p$ is unknown in Theorem 1. For a simple transformer, that may not be an issue. However, issues could arise in modern-day transformer architecture with complex tasks as suggested by the performance in the model named Pythia-14m. 
- Figure 2 seems to suggest the performance of the proposed algorithm may also depend on the task for the model, based on the difference in performance between language models and vision models. This gap is not well addressed in the paper.

### Questions
- What is self-repair in line 67?
- Why is argmax emphasized in the conditions of 4a and 4b?
- In condition 4b, it seems to suggest the causal node set must include all the downstream nodes of V along P, but why should that be the case? Why can $V$ not be a causal node set even when condition 4a is satisfied and $V$ partially satisfies condition 4b for the set of nodes in $P$ that are ancestors of $V$? Is it because it will not be enough for causal referencing? 
- In lines 129-130, it says ‘any causal node set must have at least one parent node set that is also causal’. What is the definition of a parent node set? Does the parent node set also include the causal node set in line 129 in order to be causal? If not, it seems contradictive to the condition 4b for the following reason: let $V$ be the causal node set and $Pa(V)\setminus V$ be the parent node set, by definition of causal node set, $V$ must satisfy the condition 4a, but it contradicts with $Pa(V)\setminus V$ being causal by condition 4b. 
- How are definitions 4a and 4b related to definition 6? Do they need to be stated separately?  Does Definition 6 imply satisfaction of conditions 4a and 4b in Definition 4?
- Regarding definition 6, why is there a need for changing graphical structures in order to be qualified as a sufficient intervention?
- Lines 170-171 claim that it requires identifying the causal node sets from the decision in order to identify which structures within the transformer contribute to the decision as causal paths. Is there proof of this claim?
- Why is it a good idea to treat input-dependent statistics, mean, and variance as fixed in line 191?
- Based on lines 5-7 in Algorithm 1, doesn’t it mean that minimality-based causal subset search per block is still an exponential search?
- How come Pythia-1b has a polynomial bound of $p$ higher than that of GPT2-xs even when Pythia-1b is a larger model?

### Soundness
2

### Presentation
1

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
The paper introduces a method for identifying the causal paths that lead to an output decision. Specifically, it decomposes the transformer architecture into multiple traces (paths) and applies interventions to determine which paths actually cause the decision. The authors propose a concrete algorithm and analyze its efficiency both theoretically and empirically.

### Strengths
1. The paper provides analytical results on decomposing the transformer layer into multiple traces, which appears to be novel to my knowledge.
2. The paper tries to employ the actual causality framework (Halpern & Hitchcock 2011, Halpern 2015) to provide an interpretation of the machine learning framework (transformer). 
3. The paper provides both a theoretical analysis of the time complexity for finding causal paths and experimental results demonstrating the algorithm’s efficiency in practice.

### Weaknesses
Some key definitions are either missing or introduced in an unclear order, which makes the paper difficult to follow and verify. Specifically,
1. pg2 Def 1. It seems that the authors want to provide a general definition here. However, this definition is unclear. In particular, the definition of "internal component" is missing. One would naturally think of internal components as tensor operators in a neural network, and this is not the authors' purpose. I couldn’t see how the causal graphs are structured until Section 2.3, so I suggest removing this definition or moving it to a later section.
2. pg2 Def 4. "subpath reference" has not been defined yet, which caused a lot of confusion. I also have some questions regarding this definition, which I included in the Questions section.
3. pg3 Def 5. This definition is quite confusing. "consecutively connected downstream node sets" is undefined. Also, please provide examples for "subpath reference" and "causal subpath reference".
4. pg3 Property 1. "child node" and "parent node sets" are undefined. Again, offering an example would be helpful. Also, does the result hold under any parameterizations (weights) in the transformer? Why not call it a proposition/theorem and provide a proof? 
5. pg3 Def 6. The interplay between condition (6b) and Property 1 is not clear to me. How could an intervention produce a causal graph that violates Property 1? For condition (6a), does it basically require V to have a different set of parents? Condition (6c) is also quite confusing -- I can't quite understand its role here.
6. pg4 Equation (1). Missing definitions for $L_{ia}, L_{oa}$, etc.
7. pg5 Remark 2. The proof for average-case time complexity being polytime is missing.
8. pg6 Theorem 2. Is "Reliability" a new notion introduced by the authors or an existing notion? If it exists, please cite properly. Otherwise, I don't think it is acceptable to put the definition in the Appendix without justifying its validity and importance.

### Questions
1. pg2 Def 4. I checked out the two papers mentioned above, and I didn't see how Def 4 here is identical to the original definition. For 4(a), the original definition in (Halpern & Hitchcock 2011) seems to allow $\hat{V}'$ to have the same value as $\hat{V}$. For 4(b), the original definition also intervenes on the initial values of $V$ in addition to $\hat{V}'.$ Please let me know if I missed anything.
2. pg4 Eq (2). Why include these bias terms ($b_{attn}$, etc.) as paths if they do not depend on $Z_{ib]$?
3. pg5 Algorithm 1. Can you actually specify $\hat{V}$ as input to the algorithm without knowing the specific subset $V$? 
4. Is the trick to rewrite the non-linear functions as Hadamard products something known? If so, please cite. Otherwise, please demonstrate.

### Soundness
2

### Presentation
2

### Contribution
3
