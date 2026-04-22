# Towards Sampling Data Structures for Tensor Products in Turnstile Streams

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
This paper studies the computational challenges of large-scale attention-based models in artificial intelligence by introducing innovative sampling methods in the streaming setting. Inspired by the classical definition of the $\ell_2$ sampler and the recent progress of the attention scheme in Large Language Models (LLMs), we propose the definition of the attention sampler. These attention samplers select the important coordinates in attention computation efficiently, bypassing the quadratic computational burden of computing the entire attention matrix. We demonstrate the effectiveness of the attention sampler from a theoretical perspective, including space and update time. Additionally, our framework exhibits scalability and broad applicability across various model architectures and domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes attention sampling - sampling a token index from a token stream from a distribution constructed according to the attention function. It gives a space lower bound for this problem and also provides upper bounds for similar L2 sampling questions.

### Strengths
1. The problem considered by the paper is interesting: sampling from a distribution according to the attention mechanism. Understanding the complexity of this problem and designing efficient algorithms for can have an important impact on efficient AI algorithms.
2. The methods proposed by the paper, though they generalize prior approaches and techniques, are applied with clarity and in a set of well-motivated theoretical settings.
3. The paper is rigorous and methodical in its approach. Its algorithms are well stated and proven.
4. The paper provides an extensive literature review.

### Weaknesses
1. The motivation behind "attention sampling" is not very well supported. The authors state that solving this problem would be positive towards developing efficient transformers, but this is not justified via experiments or suggestions of ways their algorithms can be used in practice. And without those, it is hard to see why attention sampling would be an important problem to solve. Furthermore, the authors show an $\Omega(n)$ lower bound for exponential sampling, meaning that even if attention sampling is very useful, the best thing we can do is calculate the underlying distribution explicitly. 
    * As a smaller comment, streaming space lower bounds for attention mechanisms have been proposed and analyzed before (also via communication complexity reductions). It would be nice for the paper to cite these works as well: [1], [2].
2. The L2 sampling contributions, while interesting, deviate from the main theme of the paper. Attention sampling, as proposed, should intuitively be related to the Transformer architecture or attention mechanism. Perhaps for different normalization functions (other than softmax), sampling in sublinear space is possible, which would give further motivation towards studying and using those functions. The L2 results seem like a natural extension of the well-studied problem of $L_p$ sampling in a turnstile stream, and though they make use of non-trivial sketch constructions to deal with updates in the $A$ matrix, they don't fit in very well to the paper's narrative in my opinion.

**References**
* [1] Han, I., Kapralov, M., Kochetkova, E., Sheth, K., & Zandieh, A. (2025). Streaming Attention Approximation via Discrepancy Theory. arXiv preprint arXiv:2502.07861.
* [2] Haris, Themistoklis, and Krzysztof Onak. "Compression barriers for autoregressive transformers." arXiv preprint arXiv:2502.15955 (2025).

### Questions
1. How are the entries of matrix $A$ updated? Is it row-by-row, or any kind of update is acceptable?

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
2

### Summary
The paper studies and characterizes a polynomial type sampling algorithms for the application of streaming LLMs where the model is
expected to engage in large inference sequence generations (as a long chat bot conversation).
Moreover, the authors derive a lower bound for the softmax distribution arguing for its "hardness", as well as upper and
lower bounds for polynomial type samples.

### Strengths
* The paper contains a comprehensive theoretical analysis for polynomial type samples proving several bounds and some
  lower bounds.

### Weaknesses
* The paper is motivated by the computational challenges of large-scale attention-based models which I'm familiar with.
However, the paper does not motivate why are we using particularly turnstile streams.
Some clearer motivation of a general Machine Learning audience would be highly valuable.
Moreover, Definition 1.1 talks about the attention sample for a matrix that is not square on the sequence length, which
deviates from the usual attention definition.
* I understand that the paper is a theoretical one where a thorough analysis is done to some polynomial type of
  algorithms. However, no empirical demonstration is done to corroborate its the core question of the paper:
  "instead of computing all the entries, can we recover the most important ones in efficient space and time?"
* I'm willing to revisit my score if the motivation for the work and its applicability is clearer to me.

### Questions
* In the abstract you claim: "Our approach significantly reduces the computational burden of traditional attention mechanisms"
Could you elaborate why this is true. My understanding is that the scope of streaming LLMs is not commonplace and
therefore the claim appears to encompass a wider scope. Am I missing something?
* In the conclusion you claim: "our framework identify the critical components in attention computation" could you
  succinctly state what does critical components are?
* Could any of this work be applied to pre-training a model or it can only be applied to inference?
* I might've missed this motivation, but what is the value of having a tensor version of the problem? What do we gain?
* Line 049. What do you mean by first asked? "A well-known example is the \ell_2 sampler first asked by" Maybe typo.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors propose to use attention sampler for efficient attention computation in the streaming setting. They primarily provide theoretical time & space lower and upper bound for L2 sampler. The authors also extend the analysis to the case of tensor multiplication.

### Strengths
The theoretical investigation seems solid and also quite comprehensive. They also extend the analysis to tensors, which might shed light on some tensor attention applications.

### Weaknesses
- Line 086-090: bullet point 1 and 2 are somewhat overlapping and provide different claims on complexity for the same underlying setting on A and x.
- The primary problem is that there are no experiments to demonstrate actual efficiency improvement in a Transformer, even though the paper is clearly motivated by solving the efficiency challenge of large-scale attention-based models computation.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2
