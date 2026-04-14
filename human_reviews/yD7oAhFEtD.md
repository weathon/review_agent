# Conv-Basis: A New Paradigm for Efficient Attention Inference and Gradient Computation in Transformers

- Decision: Reject
- Scores: 3, 5, 6

## Abstract
The self-attention mechanism is the key to the success of transformers in recent Large Language Models (LLMs). However, the quadratic computational cost $O(n^2)$ in the input sequence length $n$ is a notorious obstacle for further improvement and scalability in longer contexts. In this work, we leverage the convolution-like structure of attention matrices to develop an efficient approximation method for attention computation using convolution matrices. We propose a $\mathsf{conv}$ basis system, analogous to the rank basis, and show that any lower triangular matrix can always be decomposed as a sum of structured convolution matrices in this basis. We then design a fast algorithm to approximate the attention matrix via a sum of such $k$ convolution matrices. This allows us to compute the attention {\it inference} via Fast Fourier Transforms (FFT) in $O(knd \log n)$ time, where $d$ is the hidden dimension, and thus achieve almost linear time $n^{1+o(1)}$ in the practical scenario where $kd = n^{o(1)}$. Furthermore, the attention {\it training forward} and {\it backward gradient} can be computed in $n^{1+o(1)}$ as well. We provide theoretical guarantees on the run time and approximation error and conduct preliminary experiments to evaluate its effectiveness. We hope our new paradigm for accelerating attention computation in transformer models can help their application to longer contexts.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This work investigates the problem of approximating matrices using a lower-triangular Toeplitz matrix basis, or "conv-basis", focusing specifically on the self-attention mechanism. The authors provide theoretical results about the approximability of lower triangular matrices into such a convolution basis, and provide algorithms for efficiently recovering this basis and using it to efficiently perform the self-attention operation using FFTs. Finally, the authors provide empirical results showing that their proposed method requires few basis elements to recover performance on the IMDB dataset when applied to Llama3 8B.

### Strengths
- The authors investigate an important problem: how to speed up the efficiency of the self-attention mechanism, the computational bottleneck in Transformers.
- The theoretical results are interesting, thorough, and clearly presented.

### Weaknesses
- I think the theoretical results are quite interesting and deserve to be presented. However, the main weakness seems to be that the empirical results are quite limited.
  - The empirical speedup results are limited. Although asymptotically, the authors argue that the conv-basis approximation is more efficient, as far as I can tell, an empirical comparison of their method's speed vs. standard self-attention is not provided. It could be clarifying to provide such a comparison in the case of e.g. Llama3 8B, so that it's clear at what sequence length such an approach becomes more efficient in practice.
  - Additionally, the expressivity vs. efficiency tradeoff inherent to the method could be explored more thoroughly. The main theoretical result in the paper is that any lower triangular matrix can be well approximated in the conv basis, but accurate recovery may take as many as n basis elements in the worst case. The authors further support this result with experimental validation on the IMDB dataset with Llama3 8B, but this is only one task and relatively simple. It could be clarifying to provide additional experimental results on more recent LM benchmarks (e.g. HellaSwag, WinoGrande?)

### Questions
- As the authors mention, the proposed concept of the conv-basis approximation to sequence models seems quite related to prior work around sub-quadratic alternatives to attention, e.g. SSMs [1], Mamba [2]. Is there some way a conv-approximation of an existing attention matrix can be interpreted as a set of specific kernels for a subquadratic model?
  - In particular, I wonder if the conv-basis concept relates to the semiseparable matrices of Mamba2 [3]?
- It would be interesting to see how the approximability of attention vs. number of conv-basis elements changes based on the task, e.g. do math reasoning tasks (GSM8K) require more basis elements than simple question-answering?

1. Efficiently Modeling Long Sequences with Structured State Spaces
2. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
3. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The authors propose a k-basis system for convolutional matrices which allows for sub-quadratic inference in attention with fewer constraints on the form of the attention as compared to previous works.

### Strengths
- The proposed method offers a way to reduce the prohibitive quadratic complexity of modern transformers.
- As compared to previous works, the method appears to be much less restrictive on the structure of the attention mask which allows it to be applied to modern LLM settings.

### Weaknesses
- The experiments section is rather light, and contains only one experiment. Given that the proposed method differs quite drastically from previous transformers, I would like to see more empirical validation and ablations on different aspects of the performance.

- It is not clear to me if the experiment which was performed on Llama3 required any training from the proposed method or not.
  - If it did not require training, I would be interested to see the performance of the proposed method which did include training or finetuning.

- The rank $k$ in k-conv is given, however after reading I am curious as to how $k$ relates to performance on long context tasks. Does $k$ need to scale commensurately with the context length?

- I am left confused by the following items in Algorithm 2:
  - What is $u$ and where is it introduced in the method?
  - What is $v$ and where is it introduced in the method?
  - What is the $s$ subscript in algorithm 2 ($\tilde{H}_s \leftarrow M_s \odot (QK^\top_s)$) supposed to denote?
  - What is happening on line 8 after achieving $\tilde{H}_s$

- The algorithmic complexity looks compelling, but what is the actual wall clock time of the experiment compared to the quadratic tranformer? Can you run an additional experiment which looks at this, even if it only utilizes random inputs? I suspect that at smaller context lengths such as 2K which is provided in the experiment, the proposed method might perform worse in terms of wall clock time, but it would be interesting to know at what length the crossover happens to where the lower complexity starts to show real gains in performance.

### Questions
- What relationship does the conv-basis rank ($k$) have to the context length?
- Can your method be applied on top of existing pretrained transformers without any retraining or finetuning?
- Why is the one provided experiment performed on such a small scale. Given that the method reduces the quadratic complexity of a transformer, shouldn't it be able to scale to huge context lengths without much trouble? It would be interesting to compare this to another method such as Streaming LLM [1] on a streaming perplexity task (like PG19) or other long context benchmark which scales to the order of millions of tokens. 

## Overall

Overall, I think the work attempts to make an ambitious change to the attention operation in order to lower the complexity. However, I believe anyone reading this work will be left with many questions about the workings of the algorithms as well as the performance characteristics. I believe adding more experiments, ablations, and answering the questions posed here will aid in understanding the proposed method more fully.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper examines the possibility to compute attention using Fast Fourier Transformer when attention matrix can be well approximated by the sum of some convolution matrices. They show that when the Attention matrix can be broken down in this way, both inference and training can be done in almost linear time when $kd = n^{o(1)}$.

### Strengths
1. This paper investigates the timely question on how to speed up attention calculation and offers a new perspective on when attention calculation can be speed up.

2. While focusing on the theoretical perspective, this paper present empirical validation on the correctness of the presented algorithm and required $k$ to approximate attention matrix.

### Weaknesses
1. The algorithm 3 is deferred to appendix while being a key building block of the algorithm. In general, a more detailed explaination on how the binary search is carried out will improve the clarity of the paper.

2. Because the current work is built upon the previous work with low rank assumption, a more direct comparison (either empirically or theoretically) between the current method and previous one should be presented.

### Questions
1. How would the author expect $k$ to scale with context length and the size of the model?

### Soundness
3

### Presentation
3

### Contribution
3
