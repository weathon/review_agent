# NorSA: Accelerate LLM Decoding via Normalized Sparse Activation

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Sparse activation accelerates  the decoding of large language models by eliminating redundant computations and reducing memory access during matrix multiplications. Current approaches have potential limitations as they rely on the strong assumption that "values across different dimensions of hidden states are drawn from independent and identically distributed random variables." Our research challenges this assumption by analyzing how causal dependencies exist between tokens and correlations exist between different dimensions of hidden states. Building on this insight, we introduce Normalized Sparse Activation (NorSA), a method that accounts for inter-dimensional relationships and integrates contextual information through rotation and norm-based thresholding. NorSA achieves superior performance while maintaining computational efficiency. Experiments across LLaMA, Mistral, and Qwen model series show that NorSA consistently outperforms existing methods. For LLaMA3-8B with 50% activation sparsity, NorSA narrows the perplexity gap to only 0.44 points relative to the dense model, while restricting the zero-shot accuracy decline to a mere 1.23%, surpassing La RoSA by 1.63% and TEAL by 3.9%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The study introduces Normalized Sparse Activation (NorSA), a method to speed up large language model decoding by reducing some of the unnecessary computations. The key insight of the method, different to others, is that it correctly does not assume that the generated tokens are IID, but they are casual (a token is conditioned in all the previous generated tokens). Therefore, the method does a simple adjustement when choosing the tokens to threshold. Furthermore, it also applies normalization, rotation, and thresholding to maintain meaningful activations. The method is tested in several benchmarks and using 3 family of LLMs (LLaMA, Mistral and Qwen) reaching results that outperform the other two methods it compares with.

### Strengths
I think these are the main strengths of the paper:

1) The main part of the method (normalized sparse attention) is well-motivated, makes very much sense, and is very simple to understand. This is actually quite nice, considering that most of the papers try to over-complicate things in order to make them sound novel.

2) The results are quite good. The method decisively outperforms TEAL and La RoSA in several benchmarks.

3) There are some interesting ablation studies shown in the method.

4) Very nice to see that the authors also do a hardware aware kernel for the method.

### Weaknesses
I think these parts of the paper can be further improved:

1) It is unclear to me the connection between sections 4.2 and 4.3. In particular, while I really like the motivation of 4.2, to some degree, 4.3 looks to me a bit forced, almost like trying to increase the complexity of the method. Furthermore, I think there must be an ablation that shows the performance of both 4.2 and 4.3 in isolation, without them being combined.

2) Presentation

2a) Related work can be further improved. Right now, it just mentions some papers, but without clarifying how do they work, why they are important and how do they connect with this work.

2b) Figures 1 and 2 can be massively improved. There is a lot of trivial information there (the attention mechanism) that can be collapsed.

2c) Table 2 comes before Table 1 in the paper, this should be rearranged.

### Questions
I would appreciate if the authors can clarify these potential issues:

1) Why the timing performance has been done in A100 GPUs? Is it just because that is what the authors have or some other reasons? It would be ideal if we could see some results in more modern H100 (or B-series) GPUs.

2) What does 100% sparsity even mean? 

3) Are all the results shown while also having standard speedup mechanisms such as FlashAttention and KV caching?

I am quite willing to increase my score based on the answers for Weaknesses and Questions.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors present that existing activation sparsification techniques (e.g., TEAL, LaRoSA) rely on the flawed assumption of i.i.d. activations. To address this, NorSA introduces norm-based thresholding (to incorporate contextual scale information) and rotation matrices (to decorrelate activation dimensions). Experiments across LLaMA, Mistral, and Qwen families show the efficacy of the approach.

### Strengths
- The motivation of iid assumption not holding is reasonable.
- Having numerical results across different model families and kernel implementation is good.

### Weaknesses
- Lack of novelty: The rotation idea is adopted from SliceGPT paper. 

- Lack of clarity. The context-aware selection seems equivalent to the layer-wise sparsity allocation. In other words, though other works select activated neurons upon some threshold, the thresholds could be assigned in the global cross-layer information, making them context-aware as well. 

- The writing needs to be improved. There are many inconsistencies of writing format throughout the paper.

### Questions
See the weakness.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces Normalized Sparse Activation (NorSA) and enhances prior approaches by considering the correlations between the dimensions of hidden states.

### Strengths
1. This work challenges the assumption that different activation dimensions are independent and suggests a way to incorporate that.
2. Combining the norm calculation with other operations is an effective way to reduce overhead.
3. The author invested considerable effort into the implementation and achieved impressive empirical results across different scales.

### Weaknesses
Some of the technical and experimental designs might need additional motivation; please see "Questions".

### Questions
1. I have some reservations about Equation (3). The authors compare activation entries to their norm, but these seem like different quantities, making the comparison questionable. Why not use $\tau$ standard deviations from the mean instead?

2. Regarding equations (5) and (6), is the PCA-based rotation matrix an approximate solution? Also, I'm unsure why a rotation matrix would be useful here, as I usually think of it for smoothing outliers.

3. In Section 5.2, how do the authors "learn" the rotational matrix?

4. Is there a typo after Equation (4)? It references `sparsify` twice.

5. I haven't seen any speed comparisons with other methods. Is the main focus that the improvements are mainly in quality with minimal latency impact? If so, that seems reasonable to me.

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
3

### Summary
NorSA is a training-free activation sparsity method for LLMs that relaxes the i.i.d. activation assumption used by TEAL / La RoSA. It uses a norm-normalized threshold (|xᵢ| > τ‖x‖) so sparsity decisions depend on the overall scale of each token’s hidden state, and introduces PCA-based rotation matrices to reduce linear correlations between dimensions before sparsification. Experiments on LLaMA-2/3, Mistral, and Qwen-2.5 show better perplexity and zero-shot / MMLU performance than TEAL and La RoSA at the same sparsity, plus real decoding speedups with Triton kernels.

### Strengths
- The normalized per-hidden-state thresholding rule is simple, intuitive, and easy to integrate into existing models.

- Strong empirical results across multiple model families and sparsity levels, consistently surpassing TEAL and La RoSA, especially at higher sparsity.

- Hardware-aware implementation (fused SwiGLU+norm, sparse GEMV) and ablations on rotations / covariance make the method feel practically usable.

### Weaknesses
- Conceptual novelty is moderate: norm-normalized thresholds + PCA rotations sit close to prior rotation-based sparsity work; the paper could be clearer about what is genuinely new.

- There is no clean, large-scale ablation of “NorSA without rotations” vs “NorSA with rotations”.

- The choice of PCA as the rotation mechanism is under-motivated: no comparison to learned rotations (e.g., via distillation), and limited discussion of calibration cost and scaling.

### Questions
- PCA vs learned rotations: Why did you choose PCA-based rotations over learning rotations with a small teacher–student distillation objective or gradient-based optimization? 
- TEAL and the i.i.d. assumption: You argue that prior work (e.g., TEAL) implicitly assumes i.i.d. activations across dimensions. Could you make this more explicit? What part of TEAL’s design depends on that assumption, and can you show empirical deviations from i.i.d. in real hidden states?

### Soundness
3

### Presentation
2

### Contribution
2
