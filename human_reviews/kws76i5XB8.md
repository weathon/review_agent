# Dobi-SVD: Differentiable SVD for LLM Compression and Some New Perspectives

- Decision: Accept (Poster)
- Scores: 8, 5, 6, 6, 6

## Abstract
Large language models (LLMs) have sparked a new wave of AI applications; however, their substantial computational costs and memory demands pose significant challenges to democratizing access to LLMs for a broader audience. Singular Value Decomposition (SVD), a technique studied for decades, offers a hardware-independent and flexibly tunable solution for LLM compression. In this paper, we present new directions using SVD: we first theoretically analyze the optimality of truncating weights and truncating activations, then we further identify three key issues on SVD-based LLM compression, including (1) How can we determine the optimal truncation position for each weight matrix in LLMs? (2) How can we efficiently update the weight matrices based on truncation position? (3) How can we address the inherent "injection" nature that results in the information loss of the SVD? We propose an effective approach, **Dobi-SVD**, to tackle the three issues. 
First, we propose a **differentiable** truncation-value learning mechanism, along with gradient-robust backpropagation, enabling the model to adaptively find the optimal truncation positions. Next, we utilize the Eckart-Young-Mirsky theorem to derive a theoretically **optimal** weight update formula through rigorous mathematical analysis. Lastly, by observing and leveraging the quantization-friendly nature of matrices after SVD decomposition, we reconstruct a mapping between truncation positions and memory requirements, establishing a **bijection** from truncation positions to memory. 
Experimental results show that with a 40\% parameter-compression rate, our method achieves a perplexity of 9.07 on the Wikitext2 dataset with the compressed LLama-7B model, a 78.7\% improvement over the state-of-the-art SVD for LLM compression method. 
We emphasize that Dobi-SVD is the first to achieve such a high-ratio LLM compression with minimal performance drop.  We also extend our Dobi-SVD to VLM compression, achieving a 20\% increase in throughput with minimal performance degradation. We hope that the inference speedup—up to 12.4x on 12GB NVIDIA Titan Xp GPUs and 3x on 80GB A100 GPUs for LLMs, and 1.2x on 80GB A100 GPUs for VLMs—will bring significant benefits to the broader community such as robotics.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper performs an in-depth investigation of post-training low-rank compression of LLMs. In brief, its contributions are as follows:

- It begins by showing that “activation-level” compression (i.e. minimizing || xW - xW^ ||_F rather than || W - W^ ||_F) is superior in the case of low-rank compression. While this is useful, there are already at least two prior papers investigating this type of compression, and this is standard for both quantization and pruning.

- It then performs a deep-dive into low-rank compression, and proposes a regularization-based approach for “learnable” truncation factors across layers, an approach for performing weight updates to correct for the truncation error, and a remapping quantization-based approach which removes some of the limitations of SVD in the low-compression regime. 

- These algorithmic proposals are validated experimentally via post-training compression of Llama-7B. This shows significant improvements relative to prior SVD and structured pruning-based methods. The authors also provide end-to-end speedup evaluations.

### Strengths
- The paper is reasonably well-written, easy to follow, and contains a systematic treatment of the SVD approach for LLMs 

- The approach and conclusions described in the paper seem valid

- Some of the techniques provided (learned truncation, remapping) are interesting and could prove generally useful

### Weaknesses
- The submission slightly overstates its novelty, as the “activation” SVD approach has already been investigated, and learned truncation can be seen as an instance of compression-aware regularization, which has been used in other settings. See e.g. ShearedLlama or the older Soft Threshold Reparametrization work, which should be cited. 


- Unfortunately though, the experimental results are a bit of a mess. They are badly presented, and I believe that the methodology may be flawed: see my comments below. Nevertheless, I do believe that the improvements being presented can probably be supported. 

Overall, the paper has interesting technical results, and with writing adjustments and improved methodology it should be acceptable to ICLR. 



Comments on the experiments: 

1. First, the reader has to dig into the Appendix to understand how exactly the experiments are performed (basically, they fine-tune the method parameters over WikiText2 calibration data). However, this leads to a number of questions: 

a. I would say that fine-tuning over WikiText2 data and then measuring PPL over the same dataset is not a great idea, and definitely gives you an advantage over prior work that seems to be purely one-shot. It doesn’t seem to me like you’ve run these comparisons fairly: you appear to be copy-pasting results from prior work, but wouldn’t it be fair to at least fine-tune their compressed models for a bit on Wiki2? Same for pruning methods. If you’d like a hard baseline to compare against, check out ShearedLlama. (They also provide their one-shot pruned model, without fine-tuning with dynamic batching.) 

b. It would be great to present the zero-shot eval numbers in e.g. Table 2 as NORMALIZED accuracy, i.e. where you subtract the random classification performance from the number you present. (Then, a random classification model would have 0% accuracy.) If you do that, you might notice that your technique has near-random accuracy on ARC-Challenge at higher compression rates. 

c. Metrics: Relatedly, the paper presents an unfortunate transition from the abstract which claims “minimal” accuracy loss, to the actual results tables, which show compression losses what are hardly “minimal,” from 4% at 20% compression to 23% or more to 60% compression. 
Could you please add MMLU evaluations on Llama2? Why don’t you include evaluations on the more recent Llama3 models? 

d. Generally, I would also suggest removing or rephrasing the discussions based on PPL from the intro: given that PPL isn’t really a metric that we can directly relate to any sort of practical task performance, a 78% reduction in perplexity increase doesn’t really mean anything. You could cite the average drop in zero-shot as a much more “actionable” metric. 

e. The quoted improvements over “the best pruning methods” are false, because you are not comparing against the best pruning methods, which, to my knowledge, are SliceGPT and ShearedLlama.

### Questions
In sum, I think this paper introduces some interesting ideas, but that the presentation and methodology are both currently lacking. I’d be happy to revise my opinion if additional data is presented. See major questions above. 

Additional comments: 

- Why is there no comparison with SliceGPT? 

- Lines 450-455: The K and V compressibility observation has been known at least since Scatterbrain.

- Line 515: This eval is weird and should be removed. Yes, you can get 10x or more speedup if you compare against a method with RAM offloading, but this is very apples-to-oranges. How about you compare against a 4-bit quantized method, which should fit fine on the GPU? 

- Line 521: Do you observe a reduction in potential speedups for quantization for a model that’s already been SVD-compressed, given that you are moving away from the pure memory-bound mode of the GPU? 

- The time quoted for fine-tuning fluctuates quite a lot (once it’s 8h, it’s 4h in another place…)

- L255: “the gradient is the devil”?! Please rephrase. 

- “causing g_A goes to infinite” This whole part of the paper needs A LOT of additional polish in terms of writing.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper mainly give a discussion about the three about low rank factorization  in LLM as following:
1. $\min rank(\hat{W})$, $s.t., $ the performance gap between $model_{LLM}(W)$and $model_{LLM}(\hat{W}) $ satisfies the requirements.
2. How to update the weight based on the the compressed activation.
3. Adjust the math algorithm into practice area.

I think the core idea in this paper is gained from the way to let 0-1 function (activation function) into sigmod function to let neural network have a gradient. They also build a multi-objective function to balance the 2 target.

The authors also give the analysis and operation details based on above main idea.

### Strengths
1.I personally like the style of paper writing. Compared with traditionaly paper structure, this paper structure shows a clear structure of how authors analysis problems and how to build the algorithms.

2.There is a few work which uses  traditional low-rank decomposition to analysis and design algorithm to compress the model. This work gives a good method to compress model.

3. The optimzation of compress method is the model performance instead of traditional $\|W-\hat{W}\|$

### Weaknesses
This article faces threee critical issues:

1. In Chapter 2, we believe that the author's choice of activation as the target for decomposition is incorrect. Specifically, the reasons are as follows: in Eq. 3, we know that $\|A-A_k|\ <\|A-xW_k\|$.

But $\Delta L_w = grad_w*（W-W_k) = grad_A*(A-xW_k) = \|grad_A\|\|A-xW_k\| cos<grad_A,A-xW_k>$ and  $\Delta L_A = grad_A*(A-A_k) =\|grad_A\|\|(A-A_k)\|cost<grad_A,A-A_k>$.  Then $\Delta L_w - \Delta L_A = \|grad_A\|(\|(A-A_k)\|cos<grad_A,A-A_k> -\|A-xW_k\| cos<grad_A,A-xW_k>)$. We cannot make sure about the relationship between $\Delta L_w$ and $\Delta L_A$ with the relationship $\|A-A_k|\ $ and $\|A-xW_k\|$.

What is more, I think the Eq.1 to Eq.3 have typos, $\frac{\partial L}{\partial A}$ should be $\frac{\partial L}{\partial W}$ in $\Delta L_w$ equation. 

2. In the methods for solving gradient explosion, the author lacks analysis in approximating the gradients. If the author considers this point important and unique, I think they should conduct an analysis on it, discussing the impact of this approach on convergence results. In fact, I believe that these methods are commonly used to solve gradient explosion, and the author does not need to further describe and discuss them in the article. Instead, they could simply cite some papers on common solutions to gradient explosion.

3. What is long overlooked limitation in paper? I know in low rank method, the half of the singular values need to be truncated and the degradation of model is rooted in the this kind of method. But how to overcome this problem in your method in section 3.3? I think I am lost in section 3.3.

### Questions
1.Please correct the analysis part in Eq.1 to Eq.3. 

2. Compress the content between line 254 to line 295, or show why your method is novel or important.

3. Give more details about why the overlooked limitation is so important and how to overcome it and analysis your algorithm. In current version, I cannot understand why your work can overcome this limitation.

### Soundness
2

### Presentation
4

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
This paper tries to open the capability of SVD for LLM compression. They introduce a differentiable mechanism for learning truncation values, combined with gradient-robust backpropagation, allowing the model to dynamically identify optimal truncation points. Then, they use the Eckart-Young-Mirsky theorem to derive a weight update formula.

### Strengths
1. The paper is generally well-written and easy to follow.
2. The idea of determining the optimal truncation position for each weight matrix is new.
3. They empirically show that their method outperforms previous methods with many datasets.

### Weaknesses
1. Could you please add the result of pure GPTQ in Table 5? For me, it looks like the pure GPTQ achieves much better performance than GPTQ + Dobi-SVD with still reasonably less GPU consumption. I think Dobi-SVD degrades the performance a lot when it is combined with the quantization methods.

### Questions
1. There are some typos in the equation (1), (2), (3). For example, for (1), it should be changed into $\Delta \mathcal{L}_W = \frac{\partial \mathcal{L}}{\partial W}\Delta W$.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies SVD decomposition for LLM compression. It proposes to 
- Learn the truncation points (the number of retained singular values) for each matrix's SVD decomposition.
- Sequentially extract and optimally update weight matrix features using incremental PCA.
- Apply quantization to the left and right singular vector matrices.

### Strengths
The effect of each of the three contributions are thoroughly ablation-studied. The method compares favorably against existing methods on SVD for LLM compression.

### Weaknesses
The term "differentiable optimization" may cause confusion, as it also refers to a class of method that performs back-propagation through optimization solvers. Perhaps "gradient-based learning" is closer to your intention.

### Questions
Question about the 2nd contribution with the IPCA: is memory footprint a common concern in SVD compression for LLMs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Dobi-SVD, a theoretically grounded method to compress LLMs by leveraging a low rank representation obtained via SVD. They identify three key issues - (i) how to decide the optimal number of singular values to choose for obtaining the low rank representation (referred to as truncation position) (ii) how to efficiently update the weight matrix (iii) how to address information/performance loss when using the SVD representation for the weight matrix. They introduce a continous differentiable approximation to the discrete singular value sequence which they optimize using conventional gradient based methods to solve the first problem. For the rest, they use well known tools from linear algebra to construct an algorithm which shows empirical and computational improvements over baseline methods.

### Strengths
- The authors introduce a computationally efficient method which handles the most common drawback of SVD based methods (high memory and computational complexity)
- They leverage EMY theorem to showcase rigorously that the optimal choice of truncation is activations not weights.
- They present a stable backpropagation technique to handle gradient instabilities often encountered when backpropagating across the SVD operator
- The method offers throughput improvements over using the uncompressed model

### Weaknesses
# Major
- The work has significant overlap with existing literature at the intersection of post training quantization and low rank methods (one of which was published in ICLR 2024) - [1],[2],[3],[4]. It is certainly true that some of the issues raised have not been discussed in earlier work. Specifically, the authors are correct in claiming (to the best of my knowledge) that theirs is the first work that attempts to identify the "optimal" truncation position. However, it remains unclear whether heuristics chosen in earlier works suffice for a good enough approximation or whether one requires a stricter more rigorous approach to the selection of the truncation position.
- The authors choose only papers which directly involve SVD as the chosen method of model compression as their baseline (and some comparisons with model pruning methods). The authors claim their method to be SOTA based on the baselines chosen, however due to the vast amount of work on post training quantization (which also fall under the bucket of model compression), a fair comparison would be to compare their method against them as well (some examples include SmoothQuant, AQLM, Quip# etc). It is certainly possible as the authors have shown in their paper that their method can be combined with the methods above (which merits more discussion and experiments than what is included in the paper). 


# Minor
- The paper equates the phrase "truncating the weights" with the notion of choosing a subset of singular values obtained via SVD to represent the matrix. While acceptable, I feel this may invite confusion for novice readers who may attribute truncation as an operation directly on the weight matrix instead of the svd representation. It may be worth clarifying this explicitly at the beginning.
- On L234-235, the authors mention that the size of the solution space is $(L * M)^n$. I believe the right expression is $n^{(L*M)}$.
- On L192 eq (1), I believe the authors intended to write $\Delta \mathcal{L}_W = \frac{\partial \mathcal{L}}{\partial W} \Delta W$ at the beginning
- On L275, the authors write decompression instead of decomposition

# References
1. Zhang, Cheng, et al. "LQER: Low-Rank Quantization Error Reconstruction for LLMs." arXiv preprint arXiv:2402.02446 (2024).
2. Saha, Rajarshi, Varun Srivastava, and Mert Pilanci. "Matrix compression via randomized low rank and low precision factorization." Advances in Neural Information Processing Systems 36 (2023).
3. Lee, Jung Hyun, et al. "LRQ: Optimizing Post-Training Quantization for Large Language Models by Learning Low-Rank Weight-Scaling Matrices." arXiv preprint arXiv:2407.11534 (2024).
4. Saha, Rajarshi, et al. "Compressing Large Language Models using Low Rank and Low Precision Decomposition." arXiv preprint arXiv:2405.18886 (2024).

### Questions
- How sensitive is the task performance to the choice of $k$?
- After completing the training required to find the truncation position, how are the final weights stored? Are they still kept as individual decomposed matrices or combined back into one weight matrix

### Soundness
3

### Presentation
3

### Contribution
3
