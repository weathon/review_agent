# Constrained Decoding of Diffusion LLMs with Context-Free Grammars

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
Large language models (LLMs) have shown promising performance across diverse domains. Many practical applications of LLMs, such as code completion and structured data extraction, require adherence to syntactic constraints specified by a formal language. Yet, due to their probabilistic nature, LLM output is not guaranteed to adhere to such formal languages. To address this, prior work has proposed constrained decoding to restrict LLM generation to particular formal languages. However, existing works are not applicable to the emerging paradigm of diffusion LLMs, as this requires supporting token generation in arbitrary order instead of the traditional left-to-right order. In this paper, we address this challenge and present the first constrained decoding method for diffusion models, one that can handle formal languages captured by context-free grammars. We begin by reducing constrained decoding to the more general additive infilling problem, which asks whether a partial output with holes can be completed to a valid word in the target language. This problem also naturally subsumes the previously unaddressed multi-region infilling constrained decoding. We then reduce this problem to the task of deciding whether the intersection of the target language and a regular language is empty and present an efficient algorithm to solve this task for context-free languages. Empirical results on various applications, such as C++ code infilling and structured data extraction in JSON, demonstrate that our method achieves near-perfect syntactic correctness while consistently preserving or improving functional correctness. Importantly, our efficiency optimizations ensure that the computational overhead remains practical.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces an approach for constraining the output of diffusion LLMs (or in general any infilling model) based on a given context free grammar. The key idea of the work is to compute the space of realizable sequences of a infilling model as a regular language and then intersect it with a target grammar to check if the set is empty. If so the proposed completion is discarded. Some optimizations are presented to make the approach practical. The evaluation shows that the approach improves the quality of pass@1 rates for several models when using constraints.

### Strengths
- Constraining DLM is an emerging area and an important problem and this paper gives some initial solution
- Encouraging evaluation on existing constrained decoding benchmarks

### Weaknesses
- The paper mostly uses well-established ideas from automata theory. E.g. I believe that the "new" algorithm for intersecting regular languages and CFGs presented in this paper is already known (https://aclanthology.org/2023.eacl-main.52/). Also the algorithm is introduced to avoid a complexity stated on line 238, but the complexity of the presented algorithm is not given (at least the reviewer could not find it). The contribution of lines 250-255 is also a bit over claiming. No-one would check emptiness of a grammar with quadratic algorithm as stated by the paper. The reviewer was quite surprised to see the authors citing a random stackexchange post for the CFG emptiness algorithm where what they describe is the algorithm given in any textbook for theory of computation (e.g., Michael Sipser, Introduction to the Theory of Computation, Thm 4.8). 
- The caveats in section 3.3 are what worries me most. For Grammar Constrained Decoding, we have had many papers for which the implementations were incorrect because of how they handled the difference between tokens and lexemes. Some of these errors are reported in this paper (https://icml.cc/virtual/2025/poster/45613), which the authors should perhaps cite. The paper does not state correctness (specifically for Algorithm 3) so there is no guarantee that all and only all invalid masked sequences are rejected (something for which at the very least there should be assumptions about how the lexer operates).

### Questions
Is the presented algorithm sound/complete? Or when is it not? What does one need to assume about the lexer (1 lookahead?)

Is line 193 really true? If the distribution M is adversarial the algorithm will never terminate. For example, if the probability of e.g., adding an open parenthesis, is higher than ever closing it in the constrained distribution, the one will continue expanding the string without ever completing it. I don't think there is an easy way to always guarantee termination without some strong assumptions on M. This aspect is probably why the algorithm often takes many tokens and the authors need to resort to instead generating a random string in the grammar.

What does this mean: "only 7% of valid completions do not appear in the first 50 selected LLM updates?"

One of the known problems of constrained sampling is distribution distortion as discussed in this relevant related work (https://arxiv.org/abs/2405.21047). Suresh et al's Dingo have a theorem claiming that their approach to constraining DLM with regular expressions samples ("in some sense") optimally from the constrained distribution. What can the authors say about their approach?

How was the 256 tokens bound on line 336 chosen?

My understanding is that the proposed approach is not "friendly" for different sampling mechanisms (e.g., beam search) because it does not directly compute token masks? Why not directly compute token masks? The authors claim that this would require expensive preprocessing, but newer GCD algorithms do not require such expensive approaches (see llguidance and GreatGramma (https://icml.cc/virtual/2025/poster/45613), which are not cited in the paper) or at least require small-enough latency. I feel like masking would also drastically reduce the overhead as much of the overhead of the proposed method is caused by guessing and checking incorrect completions.

What do the underlined numbers mean in table 1 and 2?

For SMILES there are existing metrics for semantic quality (https://www.cs.jhu.edu/~jason/papers/lipkin+al.colm25.pdf) why using an LLM as a judge?

OTHER COMMENTS:
- I would report the absolute latency-per-token in the main body of the paper rather than the relative increase. The latter depends on the GPU on which one runs the LLM

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces the constrained decoding method for diffusion language models, enabling them to generate text that adheres to formal grammars like context-free grammars. Unlike traditional left-to-right generation, diffusion models generate tokens in arbitrary order, which existing constrained decoding methods cannot handle. The paper solves this by formulating an "additive infilling problem" that checks whether partial outputs with holes can be completed into a valid grammar-compliant text, reducing it to testing if the intersection of the target CFG and a regular language is empty. The method achieves strong results on syntactic correctness on tasks like C++ code generation and JSON extraction while maintaining or improving functional correctness, with reasonable computational overhead.

### Strengths
* The paper is the first work in ensuring CFG-constrained generation with diffusion LLMs.


* The paper is well-written and easy to follow. The formalism is solid, and the problem is presented with great detail. 


* The paper addressed a challenging technical problem. Additionally, there are several non-trivial technical contributions such as heuristics to reduce the size of the normalized CFG. 


* The empirical results are consistently strong, showing syntactical and. Functional improvement. And I appreciate the inclusion of confidence intervals.

### Weaknesses
* The MRI task is not natural. Removing the arbitrary character spans is not a realistic scenario in which one would expect to use an LLM. A more realistic code will remove semantically meaningful parts of the code. 


* The overhead of constraining can be large in some cases

### Questions
DINGO [Suresh et. al.] work ensures optimal decoding with regular grammar. How would the proposed approach compare against DINGO when using a regular grammar, both in terms of overhead and accuracy? 

> All MRI models were sampled with temperature 1 and greedy decoding.



If you are using greedy decoding, shouldn’t the temp be set to 0?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes the first constrained decoding method applicable to Diffusion Language Models (DLMs), which can handle formal languages defined by Context-Free Grammars (CFGs). The method works by transforming the constrained infilling problem into determining whether the intersection between the target language (CFG) and the completion language of partial outputs (a regular language) is non-empty. Incorporating strategies such as grammar optimization and rejection sampling, it significantly improves the syntactic and functional correctness of generated results in tasks like C++ code infilling and JSON structured data extraction, while incurring only moderate computational overhead. Additionally, it also supports the Multi-Region Infilling (MRI) scenario.

### Strengths
- It proposes the first constrained decoding method applicable to Diffusion Language Models (DLMs), filling the gap in existing technologies that fail to constrain DLMs using Context-Free Grammars (CFGs). Meanwhile, it naturally supports the previously unsolved Multi-Region Infilling (MRI) scenario, breaking through the limitation that traditional constrained decoding can only be applied to left-to-right Prefix generation (PRE) or simple Fill-In-the-Middle (FIM).
- It achieves the first implementation of constraining models with non-fixed-order generation (DLMs) using CFGs, capable of handling complex scenarios that rely on CFG-defined syntax, such as C++, JSON, and SMILES, thereby expanding the application scope of constrained decoding.

### Weaknesses
- In practice, models are limited by the number of tokens, which may lead to failure in meeting syntactic constraints (such as unclosed parentheses and incomplete molecular structures) and leave some residual syntactic errors. Currently, there is a lack of efficient solutions for accurately modeling the number of remaining tokens.
- In the lexing phase, if there are a large number of ambiguous terminal sequences, even though optimization via a "unified NFA" is applied, the risk of combinatorial explosion may still arise in extreme cases, which affects inference speed.
- For tasks with strong syntactic dependence (e.g., JSON, C++), the improvement in functional correctness is significant. However, for scenarios where correct syntax does not directly determine functionality (e.g., SMILES molecular generation), only a slight improvement can be achieved (an average of 0.2%). This indicates that the method is more effective for scenarios with "strong syntax-function correlation" but has limited ability to empower tasks that require semantic understanding.

### Questions
In the paper, "rejection sampling" is adopted to replace the traditional masking strategy to avoid pre-inference latency. However, the relationship between the "number of sample re-sampling attempts" and "model generation diversity" during the rejection sampling process is not clearly explained. Could you please elaborate on how the threshold for the number of re-sampling attempts (such as the 100 attempts set in the paper) is determined across different tasks (e.g., C++ code generation, SMILES molecular description)? Additionally, does there exist a scenario where "excessive re-sampling attempts lead to a decline in generation diversity"?

Need to add discussion on some related work:  
[1] Dong, Yixin, et al. "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models." Eighth Conference on Machine Learning and Systems.  
[2] Sun, Xintong, et al. "Earley-Driven Dynamic Pruning for Efficient Structured Decoding." Forty-second International Conference on Machine Learning.

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
This paper introduces a constrained decoding framework for diffusion language models, which enforces structural constraints during the infilling process.

### Strengths
The research question that this paper studied is timely and interesting.

### Weaknesses
Motivation and benefit: While the paper successfully enforces syntactic validity in diffusion language models, it remains unclear whether this constraint leads to better semantic or functional outputs. Improving syntax alone doesn’t necessarily improve model accuracy or usefulness, so it would help to clarify when grammatical correctness translates to real task gains and when it simply bounds decoding behavior.

Presentation and flow: The presentation of the core algorithm (Sec. 3) feels somewhat disconnected and difficult to follow. The exposition moves rapidly from defining the infilling intersection problem to dense formal descriptions (e.g., construction of regular language, grammar intersection, normalization, and emptiness checking) without sufficient intuitive explanation or consistent narrative flow. It is not always clear how these steps tie together in the overall decoding process. Including a running example throughout this section, showing how a concrete partial program or sentence evolves through each construction would make the method more accessible.

Flexibility limitation: It seems that the proposed method constrains the model to remain within a fixed grammar throughout diffusion, which limits flexible reasoning (reasoning first then provide answer). This restriction could prevent diffusion models from leveraging intermediate free-form reasoning steps, which are often beneficial in complex tasks like code synthesis or semantic parsing.

Baselines and comparisons: The evaluation would be stronger with additional baselines such as grammar prompting [1] (provide grammar information in the prompt without enforcing it), and recent CD method on text diffusion models[2]. Reporting results across functionality, syntax validity, and runtime overhead would help contextualize the real advantages of this approach.

Efficiency and scalability: Although the paper claims practical overhead, the computational trade-offs are not deeply analyzed. A breakdown of preprocessing vs. decoding time, memory usage, and how performance scales with grammar size, number of infilling regions, or diffusion steps would strengthen the empirical section. It would also help to discuss whether the rejection-based CFG intersection introduces runtime variance or instability, especially as grammar complexity increases and validity checks become harder to perform.

[1] https://arxiv.org/abs/2305.19234
[2] https://arxiv.org/abs/2505.23061

### Questions
See weakness.

### Soundness
3

### Presentation
1

### Contribution
2
