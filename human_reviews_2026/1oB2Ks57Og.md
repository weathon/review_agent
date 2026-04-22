# Fast RoPE Attention: Combining the Polynomial Method and Fast Fourier Transform

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
The transformer architecture has been widely applied to many machine learning tasks. A main bottleneck in the time to perform transformer computations is a task called attention computation. [Alman and Song, NeurIPS 2023] have shown that in the bounded entry regime, there is an almost linear time algorithm to approximate the attention computation. They also proved that the bounded entry assumption is necessary for a fast algorithm assuming the popular Strong Exponential Time Hypothesis.

A new version of transformer which uses position embeddings has recently been very successful. At a high level, position embedding enables the model to capture the correlations between tokens while taking into account their position in the sequence. Perhaps the most popular and effective version is Rotary Position Embedding (RoPE), which was proposed by [Su, Lu, Pan, Murtadha, Wen, and Liu, Neurocomputing 2024]. 

A main downside of RoPE is that it complicates the attention computation problem, so that previous techniques for designing almost linear time algorithms no longer seem to work. In this paper, we show how to overcome this issue, and give a new algorithm to compute the RoPE attention in almost linear time in the bounded entry regime. (Again, known lower bounds imply that bounded entries are necessary.) Our new algorithm combines two techniques in a novel way: the polynomial method, which was used in prior fast attention algorithms, and the Fast Fourier Transform.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a fast, almost linear-time algorithm for approximating RoPE-based attention. The main technique of the proposed method is a combination of the polynomial method and the Fast Fourier Transform (FFT) to handle the structure of RoPE. The paper also provides a computational hardness result.

### Strengths
RoPE is a cornerstone of many state-of-the-art LLMs (Llama, Claude, etc.), and developing faster algorithms for it is of significant practical interest.

### Weaknesses
1). The technical novelty is limited. The paper uses FFT to handle Toeplitz-like structures in positional encodings, which is also a known approach in existing models [1]. The primary contribution is the specific application of this technique to RoPE and combining it with the polynomial method which is yet another known method. While this is a valid contribution, the paper fails to articulate more generalizable algorithmic insight beyond this direct combination. 

[1] Qin, Zhen, et al. "Toeplitz Neural Network for Sequence Modeling." The Eleventh International Conference on Learning Representations.

2). The assumptions in the theorems, $B=o(\sqrt{\log n})$ and $d= O(\log n)$, are not validated. In particular I doubt if $d= O(\log n)$ is true in practice. For current LLMs, the embedding size is very large. For example, the embedding size of DeepSeek-R1 is 7168, which is unlikely to be of $\log n$ scale.

3). Although the research problem is driven by practical applications, the paper is purely theoretical. Without experiments, it is impossible to assess:
  - The practical speed-up over standard, hardware-accelerated attention.
  - The actual trade-offs between speed and approximation error.
  - How the method compares to other approximate attention mechanisms in terms of quality and performance.
  - The overhead introduced by the FFT and polynomial coefficient computations.
As a result, the paper's claims of efficiency cannot be translated into practical innovations.

4).The related work section is weirdly long and diverges from the main topic. For example, it is not clear why accelerating diffusion models or GNNs is relevant to this work.

### Questions
The paper mentions that Claude uses RoPE. How do you know that?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes the first almost linear-time algorithm for RoPE attention under bounded entries, matching known lower bounds. It introduces a generalized RoPE attention problem (ARAttC), proves an upper bound via a novel combo of the polynomial method with FFT on sums of rescaled Toeplitz matrices, and a SETH-based lower bound.

### Strengths
(1) The motivation of this paper is clear. The paper explains why classic polynomial-method low-rank arguments break under RoPE (Toeplitz-like structure rather than low rank) and why FFT is the right technique.

(2) The method incorporates the polynomial approximation and fast computation of FFTs.

### Weaknesses
(1) The theorems are asymptotic; it would help to expose the exact dependence on the polynomial degree and the number of rescaled-Toeplitz summands t after approximation. Here, n is the sequence length. In real applications, will it be approaching \infty? I think the real LLMs have a sliding window, and n is not very large, right?

(2) The paper does not conduct any experiments to show the improvement of the computation efficiency. I suggest including some experiments (even small synthetic ones) to compare the proposed method with Flashattention, etc.

### Questions
(1) Is the polynomial approximation to the exp function stable? Especially when we are using higher-order polynomials. 

(2) How does the algorithm extend across multi-head attention and batching? 

(3) How does the approach interact with causal masking and sliding-window attention often used with RoPE?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses a polynomial methods for efficient computation of attention, with **Rotary Position Embeddings** (RoPE).
While prior works achieved near-linear algorithms for standard attention(under bounded entry regimes), these methods do not extend to the increasingly popular RoPE variant.
The authors develop a new algorithm that achieves near-linear time for RoPE attention computation, combining the polynomial method (for low-rank approximations) with the Fast Fourier Transform
using rescaled Toeplitz formulation.
 They prove that the nearly linear regime's theoretical thresholds extend to RoPE and provide the first such provable algorithms for this popular class of attention mechanisms.

### Strengths
Clear Motivation & Relevance: The paper highlights the importance of efficient attention mechanisms, especially as RoPE becomes standard in large LLMs (Llama, Claude, Gemini, Apple, etc.).

Originality: The combination of the polynomial method with FFT for rescaled Toeplitz matrices is novel, and the authors identify why previous techniques fail in the RoPE case.

Theoretical Rigor: Strong upper and lower bounds are established. The authors are meticulous in showing tightness of their results, connecting them with SETH and prior complexity theory for attention.

Exposition: Section structure is logical and easy to follow. Notation, background, and step-wise algorithmic development are mostly clear. 
The proofs and more technical details are referenced for reproducibility.

### Weaknesses
Clarity: Some sections (esp. regarding structured matrix manipulations) assume a degree of background with FFT applications and polynomial approximations in algorithms.
 Additional diagrams or simplified intuition would make the work more accessible to a wider ML/AI audience.

Related Work Scope: The related work is comprehensive regarding theoretical literature, but more discussion about current practical/engineering solutions for fast attention 
(e.g., FlashAttention variants, hardware-accelerated solutions) might provide context for potential synergy or limitations.
Also, it is not clear if the RoPE issue is also present in Linear attention methods, where the attention is represented by matrix multiplication of (Mercer's) kernel functions. 

Generality: The main result holds under bounded norm assumptions and for certain embedding sizes relative to sequence length (O(log n)). While justified, practical consequences
 and possible relaxations/tighter practical bounds could be discussed more. Can you point in which cases this bound breaks? is it model/ data dependent?

Experimental Results Are Missing: Up to page 9 (end of main content), the paper is entirely theoretical.
 There is no validation of the algorithm's practical performance on real-world attention problems or large models (e.g., LLMs using RoPE in practice).
 While the theory is strong, empirical evidence for efficiency, accuracy, and scalability trade-offs is necessary for broader impact.

### Questions
$Q_1$. Precision/Accuracy Robustness: How does the practical choice of polynomial approximation error $\epsilon$ and norm bound $B$ affect downstream accuracy and speed?
 Are there scenarios (e.g., quantized or noisy activations) where theoretical advantages may not materialize?

$Q_2$. Scalability: while the paper highlights regime restrictions (O(logn)), is it possible to extend your approach to non-logarithmic scenarios in practice? or will a break in theoretical guarantees?

$Q_3$. Hardware implications: Given the rising importance of custom hw accelerators, is your method amenable to efficient hardware implementation/composability with existing frameworks?

$Q_4$. Empirical Validation: Do you plan to provide experiments on LLM inference/training with RoPE using your algorithm?
 How does the runtime and accuracy compare to current best practical methods (e.g., FlashAttention) on models like Llama-2/3?
 
$Q_5$. Linear attention extention: A vast class of fast attention methods are linear attention. Does the RoPE issue present also in this class of attentions? Does your method applicable to this case (e.g., Nystrom, Performer, etc.)? 

$Q_6$. Low-dregree polynomial approximation: In lines 181-190 the paragraph explains that exp function can be approximated by a polynomial function.
 However, softmax defintion is diffferent and will have a different approximation error when using low degree polynomial to apprixmate. 
 You should clarify the consequences of low-degree approximation of softmax rather than exp. 

$Q_7$. Minor comments : 
 - definition in line 63, is repeating again in line 68.
 - in line 60 "this lower bound" refers to ? 
 - equation (1) is per head (should be clarified) 
 - line 134, "changing the many parameters" - > "changing many parameters"
 - line 138, norm |S| should be defined (or referenced). 
 - line 151, typo ==> 1/sqrt(d) (this sqrt{d} term should appear also in eq (1))

### Soundness
3

### Presentation
2

### Contribution
3
