# Factor Graph Optimization for Belief Propagation Decoding

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 8, 2, 4

## Abstract
Belief Propagation (BP) is a highly efficient message-passing algorithm for inference on graphical models, famously applied to the decoding of sparse codes. The performance of BP, however, is critically dependent on the structure of the underlying factor graph. Designing a graph structure that is optimal for BP decoding remains a significant challenge, especially when constrained by short block lengths or novel channel models.
In this work, we introduce, for the first time, a gradient-based and data-driven framework to directly optimize the factor graph for the Belief Propagation algorithm. We learn locally optimal graph structures by running simulations under channel noise. This is enabled by a novel, complete graph tensor representation of the Belief Propagation algorithm, which makes the decoding process end-to-end differentiable. This representation allows us to optimize the graph structure over finite fields via backpropagation, coupled with an efficient line-search method. 
When applied to the design of sparse codes, the resulting BP-optimized factor graphs demonstrate decoding performance that outperforms existing popular codes and show the power of data-driven approaches for code design.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a new framework for optimizing the design of error-definite codes using gradient methods. Conventional design methods for LDPC codes, etc., have fixed the code structure and optimized performance based on analytical and combinatorial techniques. However, in recent years, there has been a demand for flexible code designs that can adapt to new communication channels such as short block lengths, IoT, and 5G.　
While many machine learning approaches have focused on designing "neural decoders," this research focuses instead on optimizing the "code itself (factor graph)."
The core idea of this research is, assuming a Belief Propagation (BP) decoder, to machine-learn the optimal code structure for the BP decoder.

### Strengths
Originality: This research is the first attempt to optimize a code structure (factor graph) using gradient descent while keeping the belief propagation decoder fixed.
Quality: As long as the authors examined, the superiority of the proposed method is shown objectively. 
Clarity: What was done is clear. 
Significance: Unlike conventional neural decoder-type ML-ECC, this method is compatible with existing BP implementations and has low implementation costs. The resulting code is sparse, practical, and performs well even in high SNR environments.

### Weaknesses
The relaxation method used for optimization has no theoretical basis. While significant performance improvements are observed for short and medium block lengths, this comparison is based solely on BP decoding. Unless the block length is sufficiently long, BP is strongly affected by cycles in the graph and is not necessarily a good decoding method. Performance comparisons with the current best codes for short and medium block lengths are desirable.

### Questions
How good is the performance obtained compared to the current best codes for short and medium block lengths?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper suggests learning the Tanner graph of an error correcting code by optimizing the performance of a BP decoder on that code. In order to achieve this, the BP decoding is formulated in a way that is differentiable. Results show significant improvements in bit error rate (BER) as a result of the optimization.

### Strengths
The paper is well-written and easy to follow. I agree with the authors that it is better to use "data driven" approaches to optimize the code for BP rather than using a gigantic transformer to learn how to decode. The results show a clear advantage when the search is well initialized.

### Weaknesses
The work could be improved by showing the importance of a good initialization for the code optimization. For example, figure 2 shows that using the optimization method starting from a highly designed 5G code will lead to a better code. But what happens if we initialize with a bad code? Since the optimization landscape is highly nonconvex, I would presume that a good initialization is crucial.

The work could also be improved by comparing to other numerical methods for optimizing codes. Although the authors call their approach "data driven", I see it more as a competitor to methods such as density evolution and EXIT diagrams which have existed for over 20 years. In such methods, the expected BER of a code with certain properties is estimated and code optimization is performed by numerically optimizing these performance as a function of the parameters. I think both approaches have their strengths and weaknesses but the paper could be improved by discussing these explicitly.

### Questions
See the section on weakenesses.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a gradient-based, data-driven method for designing sparse-graph codes tailored to belief propagation (BP) decoding. The main contribution is to learn the factor-graph structure through a differentiable representation that enables backpropagation. Specifically, the authors start from a complete bipartite graph where the edges are learnable.
The approach is interesting and, as the authors show, leads to codes that outperform some existing ones. However, I recommend rejecting the paper due to the limited scope of its contribution, the weak experimental comparisons, and the additional concerns detailed in the "Weaknesses" section.

### Strengths
The approach is interesting and, as the authors show, leads to codes that outperform some existing ones.

### Weaknesses
This paper was previously submitted to ICLR 2025 and rejected for well-founded reasons (see OpenReview). I would like to disclose that I served as one of the reviewers of that earlier submission. Upon inspection, this version appears nearly identical to the previous submission, with minimal or no substantive changes. Therefore, the same weaknesses identified in the earlier reviews remain unaddressed.
In short, the main issues are:

1. The writing of the paper requires significant improvement.
2. More importantly, the proposed codes do not outperform the state of the art. Yes, they do perform some good codes, but not the best ones.
3. While the work introduces some interesting ideas, it does not demonstrate sufficient impact or advancement to merit acceptance at a top-tier venue like ICLR. The contribution is more suitable for a specialized (not major) venue such as ISIT or ITW.

Overall, in my opinion does contain some interesting and potentially valuable ideas that deserve to be published. However, I also strongly believe that it does not merit publication in a top-tier conference/Journal. The decision made in the previous submission was well justified and should be upheld.

### Questions
I do not have additional questions beyond those raised in the previous submission. The paper is not technically flawed; however, its significance and potential impact are insufficient to merit publication in a venue of this caliber. In the opinion of this reviewer, the contribution is minor (besides the fact that the proposed approach does not surpass the state of the art)

### Soundness
3

### Presentation
2

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
The paper proposes to directly learnthe factorgraph for BP by backpropagating through a tensorized BP and updating the binary parity-check matrix with a discrete-aware (line-search) step, showing better BP decoding on several codes/channels.

### Strengths
Keeps standard BP as the final decoder (more deployable with respect to other methods)

Differentiable formulation of BP over a learnable adjacency is novel to my knowledge

A clever and novel binary-aware update to make STE actually work is designed and tested

Consistent empirical gains across multiple code families

### Weaknesses
1. The paper sounds a bit like it is the first work to learn the factor graph. It would be good to have a comparison to PEG, differentiable LDPC search, neural BP with learnable edges, etc.

2. You start from a dense bipartite graph and run tensor BP; the training numbers are heavy for relatively small codes. Is it possible to use this on realistic blocklengths or 5G-like structured graphs? It would be good to quantify the cost, show a lighter variant (e.g. start from structured sparse, optimize only a subset), or show it works on a realistically constrained graph

3. You claim “factor graph optimization,” but you don’t show girth or cycle-spectrum improvements. It would be good to add before-after cycle histograms, degree constraints kept... Otherwise it seems difficult to distinguish the method from small edge changes driven by the loss (or be convincing that the latter is enough).

4. optimize for 5 iteration, test at 15 iterations could look a bit tuned to the experiment

5. Ablations on the key trick (binary-aware line search) are missing.

### Questions
Please address the weaknesses, I will consider increasing my score upon convincingly addressing them.

### Soundness
3

### Presentation
3

### Contribution
3
