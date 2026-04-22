# On the Expressive Power of Weight Quantization in Deep Neural Networks

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
In recent years, weight quantization, which encodes the connection weights of neural networks in an $n$-bit format, has garnered significant attention due to its potential for model compression. Many implementation techniques have been developed; however, the theoretical understanding of many aspects, especially the approximation and degradation of expressive power as the number of quantization bits decreases, remains unclear. In this paper, we conduct a theoretical investigation into the expressive capability of deep neural networks relative to the number of quantization bits. We establish the universal approximation property of quantized neural networks with linear width and exponential depth. Additionally, we confirm that weight quantization leads to expressive degradation, in which the expressive capacity of quantized neural networks degrades polynomially as the number of quantization bits decreases. These theoretical findings provide a solid foundation for advancing weight quantization in the context of scaling laws and shed insights for future research in model compression and acceleration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a theoretical framework for analyzing the expressive power of weight-quantized neural networks, proving that networks with two or more bits retain universal approximation while 1-bit networks suffer expressive collapse, and that expressive power degrades polynomially as bit-width decreases, with experimental validation.

### Strengths
1. This paper establishes a formal mathematical link between quantization bit-width and expressive power, providing a solid theoretical basis for weight quantization.
2. The paper rigorously formulates universal approximation, expressive collapse, and polynomial degradation with clear and reproducible logic.
3. The theoretical results are validated through simulation and ImageNet experiments, enhancing the credibility and practical relevance of the work.

### Weaknesses
1. The metric ln(accuracy/model complexity) used in the ImageNet experiments appears to be uncommon; have the authors considered providing the raw values to facilitate the evaluation of fitting accuracy?
2. Would the main theorems still hold if weight quantization were modeled as a stochastic process rather than deterministic rounding?
3. The paper repeatedly mentions “linear width” and “exponential depth”; could the authors provide a one-sentence explanation of their physical or intuitive meanings?
4. In Experiment 1, 1000 sample points were generated, but the sampling method (uniform or normal) was not specified—could this affect the results?
5. The paper contains rich theoretical proofs; it is recommended that the authors include a table of symbols and terminology to help readers better follow the subsequent derivations.

### Questions
See the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the theoretical expressive power of neural networks with quantized weights. It shows that (1) quantized networks can still be universal approximators with sufficient depth and width, and (2) expressiveness degrades polynomially as the number of bits decreases.

### Strengths
* Provides a clear analysis of how weight quantization affects expressive power, which is important for model compression research.
* Establishes universal approximation for quantized networks and quantifies polynomial degradation with bit reduction.

### Weaknesses
* The paper does not cite Three Quantization Regimes for ReLU Networks (ca2024), which studies depth-precision trade-offs, minimax approximation error, and identifies under-, over-, and proper quantization regimes. This omission weakens both novelty and completeness of the literature review.
* While polynomial degradation is shown, the paper does not connect this to minimax approximation error or the mechanisms behind it, limiting practical guidance on bit allocation.

### Questions
* How does your polynomial degradation result relate to the three quantization regimes identified in ca2024? Why was this prior work not discussed?
* Can you provide bounds or rates for the universal approximation property relative to network width and depth, to make the results more actionable?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper explores how weight quantization influences the expressive power of deep neural networks. It shows that networks using two or more bits can still approximate any continuous function when sufficiently deep, while one-bit networks restricted to {0, 1} weights lose this ability entirely. The authors further derive a quantitative relationship between precision and representational accuracy, demonstrating that the approximation error between quantized and full-precision models grows polynomially as the number of bits decreases. Empirical tests on synthetic and image-classification tasks follow the same general pattern, though the experiments are limited in scope and mainly serve as qualitative support for the theoretical claims.

### Strengths
1.  The paper provides a unified theoritical framework connecting quantization to expressibity and approximation error. 
2. The constructive proof for universality is mathematically sound and leverages ideas from deep-narrow network theory (Kidger & Lyons, 2020).

### Weaknesses
The main weakness lies in the fact that the paper’s “expressive collapse” theorem for 1-bit neural networks only applies to models whose weights are restricted to the set {0, 1}, rather than the practically relevant signed case {−1, +1}. Because {0, 1} weights can only form non-negative linear combinations, such networks lack the ability to perform subtraction or cancellation, which makes their limited expressiveness somewhat inevitable. This means the negative result demonstrates the weakness of a degenerate, unsigned quantization scheme rather than establishing a general limitation of binary networks. Without extending the analysis to signed weights or reconciling it with prior work that found universal approximation in the {−1, +1} setting, the paper’s main claim risks overstating its generality and practical relevance.

Minor writing error (do not affect score):
Reference list duplicates Courbariaux et al., 2015a/2015b entries. They are identical. keep one.

### Questions
Would allowing signed 1-bit weights {−1, +1} restore universal approximation, or does the expressive-collapse phenomenon persist under that setting?

### Soundness
2

### Presentation
2

### Contribution
3
