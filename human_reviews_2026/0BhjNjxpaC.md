# Limit Analysis for Symbolic Multi-step Reasoning Tasks with Information Propagation Rules Based on Transformers

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 8, 8, 2

## Abstract
Transformers have ability to perform reasoning tasks, however the intrinsic mechanism remains widely open. In this paper we propose a set of information propagation rules based on Transformers and utilize symbolic reasoning tasks to theoretically analyze the limit reasoning steps. 
We show that the number of reasoning steps has an upper bound of $O(3^{L-1})$ and a lower bound of $O(2^{L-1})$ for a model with $L$ attention layers.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work studies the question: "how many 'reasoning steps' can an $L$-layer Transformer carry out in a single forward pass?". To answer this question, the paper posits a formulation of "reasoning chains" as sequences of pairs of integers $a_i^1 \to a_i^2$, which can be permuted. It then formalizes an abstract model intended to represent multi-layer Transformers by defining rules of "information propagation", where the information at each token in the transformer is a set $V_i^l$ that expands at each layer (i.e., $V_i^{l+1} = V_{\mathcal{I}}^{l} \cup V_i^{l}$). The question of how many reasoning steps a transformer can carry out is mapped to the question of how big $C_i^l = | V_i^l|$ can be. The main result of the paper is to show that this is between $2^{l-1}$ and $3^{l-1}$.

### Strengths
This work tackles an important problem that is of great interest to the machine learning community, namely, characterizing the theoretical capabilities of Transformers to perform "reasoning". A key aspect that is highlighted is the potential to perform parallel computation across all tokens at each layer.

### Weaknesses
- The main weakness of this work lies in the modeling of its notions of "reasoning" and "Transformers", which underpins the main theoretical result. The framework is presented in a highly abstract (and arguably unnecessarily so) manner, lacking clear motivation or grounding in real tasks or implementations of Transformers. Formalizing vague concepts like “reasoning” is inherently difficult, but the formulation proposed here is neither clearly useful nor evidently interesting. It is unclear what class of problems this artificial setup captures.
- Moreover, the formulation of Transformer computation is also limited and artificial. On the one hand, it considers only a very limited subset of the types of computations Transformers can do (namely, "same token matching" and "adjacent position matching"). On the other hand, even those limited computations are modeled in an overly simplistic manner, which is disconnected from the parameterization of Transformers and the distributed vector representations at each layer.
- The central result is unsurprising given the artificial nature of the model. In particular, the toy model of Transformer computation crucially does not consider the capacity of finite-precision vector embeddings, which has been shown to be critical in determining the function class of neural networks, including Transformers, in prior work.
- The experimental section is limited and does not substantively test the theoretical results. While the experiments are designed around the paper’s own notion of “reasoning,” they only consider $L=3$ and yet claim empirical support for the theory. To meaningfully support the theoretical result, the authors would need to vary $L$ and demonstrate that the conclusions hold more generally. Since the theory concerns scaling with depth, fixing the depth in experiments fails to test the key claim.
- The Related Work section lacks substantive discussion. It merely lists a large number of papers with minimal analysis. In particular, relevant work on the approximation / representation capacity of Transformers is missing (e.g., Perez et al. 2019; Yun et al. 2020; Wei et al. 2022; Sanford et al. 2023; etc.). 
- Communication and structure are also weak points. The logical flow of the paper is difficult to follow. For instance, Section 3 presents an informal preview of the theoretical result before introducing the necessary definitions of “reasoning” or “information propagation”. The mathematical formalism is often redundant or unnecessarily complicated. For example, Definitions 4.2 and 4.4 are redundant, Definitions 4.6 and 4.8 are redundant, and Definition 4.5 appears unnecessary. Overall, much of the abstraction overcomplicates otherwise simple concepts.

### Questions
- Does this theory make any predictions about how the number of “reasoning steps” computable by a Transformer scales with the number of heads, in addition to the number of layers?
- How does the proposed model of “information propagation” relate to existing formal models of Transformer computation, such as those in RASP?
- In Rule 2 of the information propagation rules, why is it stipulated that “adjacent position matching” information must flow from odd-position nodes to subsequent even-position nodes? This appears to be an artificial constraint intended to align the model with the pair structure of the “reasoning chains.” How would such a rule correspond to realistic models of position-based attention?

Typos?
- line 200: $a_i^1 \to a_I^2$
- Line 285: $(\mathbf{a} a_m)_{1 \leq m \leq s}$

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the intrinsic single-pass reasoning depth of transformer decoders with L attention layers. Within a symbolic multi-step reasoning setup, the authors formalize “information propagation rules” (adjacent-position matching, same-token matching, residuals) and prove bounds on the maximum number of effective reasoning steps in one forward pass lies between. They give matching constructions showing a binary-tree lower bound and a ternary-tree upper bound, and provide experiments (e.g., a 3-layer model achieving perfect 3-step reasoning with sufficiently large hidden dimension) consistent with the theory.

### Strengths
A clear, formal definition of “reasoning step” tied to concrete token-level propagation primitives (adjacent and same-token matching). Theorems bound capacity as a function of depth.

The bounds are constructive and interpretable.

Empirical checks (3-layer transformer; accuracy vs. hidden size) align with the theoretical picture that wider representations help store intermediate facts but do not break the depth-dependent step limit.

### Weaknesses
It’s not yet clear (at least to me) how to map the notion of the reasoning step from the paper to what counts as a step in naturalistic prompting in reasoning experiments with LMs. For instance, Guzmán et al. (NAACL Findings 2024) document persistent failures of the transformer's logical generalization (recursivity and compositionality). Could such phenomena be explained through the main result of the paper?

### Questions
Do your bounds predict the specific deficits seen in compositional and recursive reasoning in transformers? 

How would the bounds change without the single pass assumption?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper provides a theoretical analysis of the symbolic multi-step reasoning limits of a Transformer in a single forward pass. By modeling information propagation through adjacent position and same-token matching, the authors show that an $L$-layer Transformer can support between $O(2^{L-1})$ and $O(3^{L-1})$ reasoning steps. Experiments on symbolic reasoning tasks confirm these bounds, showing that a 3-layer model succeeds on up to 4-step reasoning but fails beyond that. The work offers a formal and tight characterization of single-pass reasoning capacity, extending beyond prior empirical analyses.

### Strengths
This paper makes a strong contribution by providing a clear and rigorous theoretical understanding of the intrinsic reasoning capacity of Transformer models.

More specifically, it establishes formal upper and lower bounds on the number of reasoning steps a Transformer can perform in a single forward pass, moving beyond purely empirical observations. The information propagation framework is well structured, connecting low-level attention mechanisms to high-level symbolic reasoning, which enhances interpretability. The proof techniques are clean and logically sound, offering precise insights into how reasoning depth scales with model depth. The experimental results closely match the theoretical analysis, validating the tightness of the proposed bounds. Overall, the work addresses a fundamental and timely problem, making it both theoretically significant and practically relevant.

### Weaknesses
While the paper makes a solid theoretical contribution, it also has some limitations. The analysis is based on highly idealized symbolic reasoning tasks, which may not fully capture the complexity of real-world natural language reasoning scenarios. However, this might be understandable given the difficulty of rigorous theoretical analysis. 

In addition, although the theoretical bounds are tight, the paper offers limited discussion of how these limits might interact with optimization dynamics, initialization, and data distribution, and a deeper treatment of these aspects would make the results more practically relevant.

### Questions
How do you expect the theoretical bounds derived under idealized symbolic settings to translate to more realistic scenarios, where the reasoning process involves natural language, semantic ambiguity, and potentially noisy or unstructured inputs?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies the intrinsic, single-pass reasoning capacity of decoder-only transformers by introducing an idealized information-propagation model that abstracts attention and residual connections into set-based rules over “reasoning pairs.” The core task is symbolic: inputs are made of shuffled pairs (a_i^1, a_i^2) that form a reasoning chain, and the model must recover the target reached after multiple reasoning steps. Within this abstract model, the authors prove that for an L-layer system following their rules, the number of reasoning steps whose information can reach the final position is between a lower bound of 2^{L-1}-1 and an upper bound of \frac{3^{L-1}-1}{2}, i.e. exponential in depth with bases 2 and 3 respectively. These bounds are made concrete by constructing special input permutations that attain the two extremes. They further provide a constructive mapping from the abstract rules to a hand-designed single-head masked transformer that realizes adjacent-position matching and same-token matching with specific Q/K/V choices, arguing this shows the rules are implementable in principle. Experiments on synthetic data with a 3-layer transformer show: (i) near-perfect 3-step reasoning when width is large, (ii) partial success on 4-step when the ordering is favorable, and (iii) clear degradation on 5-step, which the authors present as consistent with their theoretical limit. However, the experiments do not test the theory’s central scaling claim in L, and the symbolic abstraction is stronger than what real, trained models are guaranteed to implement.

### Strengths
- The authors provide both a constructive lower bound and a constructive upper bound.
- Appendix D gives an explicit way to choose Q/K/V and use residuals/FFNs so that a single-head, causal Transformer can behave like the rule system.

### Weaknesses
- The theorems (Sec. 6) are proven for a set-union model of information, not for the actual vector-space, soft-attention transformer. It does not show that trained transformers implement those rules. The proofs are also dimensionality-agnostic, implicitly assuming each node can store arbitrarily many distinct reasoning items regardless of hidden size. However, the authors’ own experiments (Table 1) show that performance depends strongly on hidden size (larger widths are required for multi-step reasoning) indicating that representational capacity is a limiting factor.
- The empirical results insufficiently probe the core scaling law: the theory predicts reasoning capacity should grow exponentially with depth, yet all reported experiments evaluate only one depth (L=3).
- The assumption that adjacent position matching occurs only in layer 1 and same-token matching only in layers L >= 2 (Rule 2 & 3) is entirely artificial. There is no architectural or theoretical reason for this strict layering of function.

### Questions
- Can you add experiments for L=2,4,5 on the same symbolic task, with width sweeps, to show whether the “steps vs. depth” curve is actually exponential in practice?
- Your own results show strong dependence on hidden size. Can you give a dimensionality or capacity argument along the lines of “to realise N parallel reasoning items at layer L, you need at least O(N) (or similar) width”?

### Soundness
2

### Presentation
2

### Contribution
2
