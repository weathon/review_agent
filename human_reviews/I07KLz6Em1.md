# QuantEase: Optimization-based Quantization for Large Language Models

- Avg Score: 3.67
- Decision: Reject
- Scores: 3, 3, 5

## Abstract
With the rising popularity of Large Language Models (LLMs), there has been an increasing interest in compression techniques that enable their efficient deployment. This study focuses on the Post-Training Quantization (PTQ) of LLMs. Drawing from recent advances, our work introduces QuantEase, a layer-wise quantization framework where individual layers undergo separate quantization. The problem is framed as a discrete-structured non-convex optimization, prompting the development of algorithms rooted in Coordinate Descent (CD) techniques. These CD-based methods provide high-quality solutions to the complex non-convex layer-wise quantization problems. Notably, our CD-based approach features straightforward updates, relying solely on matrix and vector operations, circumventing the need for matrix inversion or decomposition. We also explore an outlier-aware variant of our approach, allowing for retaining significant weights (outliers) with complete precision. Our proposal attains state-of-the-art performance regarding perplexity and zero-shot accuracy in empirical evaluations across various LLMs and datasets, with relative improvements of up to 15% over methods such as GPTQ. Particularly noteworthy is our outlier-aware algorithm’s capability to achieve near or sub-3-bit quantization of LLMs with an acceptable drop in accuracy, obviating the need for non-uniform quantization or grouping techniques, improving upon methods such as SpQR by up to two times in terms of perplexity.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces the method entitled QuantEase based on Coordinate Descent techniques that avoids matrix inversion and decomposition. Also, the paper proposes an outlier-aware approach by employing a sparse matrix with just a few non-zeros.

### Strengths
The authors propose derives a closed-form solution for fast update and presents a convergence analysis of QuantEase. Also, QuantEase can quantize up to OPT 66B on a single NVIDIA V100 32GB GPU.

### Weaknesses
It is dubious whether the outlier-aware approach could be also accelerated like other approaches such as GPTQ and AWQ due to the presence of a sparse matrix, $\hat{H}$. To validate the effectiveness of the outlier-aware approach, it seems to be required to measure the inference latency of the outlier-aware version of QuantEase.
 

All experiments are based on perplexity, which is insufficient to assess whether QuantEase is effective or not. The zero-shot performance of common sense reasoning tasks or the five-shot accuracy of MMLU seems to be needed. 
 

In addition, all experiments are conducted for OPT and BLOOM models. The experiments for Llama or Llama 2 models should be necessary to justify the effectiveness of QuantEase.

### Questions
N/A

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
QuantEase is novel quantization method similar to GPTQ [1], but instead of Hessian-based second order optimization, QuantEase uses Coordinate Decent (CD) for much faster training. In addition to the improvement over GPTQ, QuantEase also provide an outlier-aware solution with sparse integration similar to SPQR [2].

* [1] Frantar, E., Ashkboos, S., Hoefler, T. and Alistarh, D., 2022. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323.
* [2] Dettmers, T., Svirschevski, R., Egiazarian, V., Kuznedelev, D., Frantar, E., Ashkboos, S., Borzunov, A., Hoefler, T. and Alistarh, D., 2023. SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression. arXiv preprint arXiv:2306.03078.

### Strengths
1. QuantEase demonstrated the cheaper and faster coordinate decent method can achieve better result over Hessian-based GPTQ. GPTQ requires Cholesky decomposition of the Hessian matrix which is a big overhead for Neural Networks. Removing such overhead but achieving similar or better performance is a great contribution to quantization research.

2. QuantEase's coordinate decent approach is quite orthogonal to other modern quantization techniques, such as integrating with sparsity as this work demonstrated, AWQ [1], and sub-channel quantization (used in GPTQ as well). The addictive impact of this work is promising.  

* [1] Lin, J., Tang, J., Tang, H., Yang, S., Dang, X. and Han, S., 2023. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. arXiv preprint arXiv:2306.00978.

### Weaknesses
1. All the experiments were based on per-channel quantization. Nevertheless, SOTA quantization research (GPTQ and AWQ) set the baseline with sub-channel quantization with group size of 128. Author justified their choice of comparing per-channel baseline for computational efficiency, yet there were no quantitative support for the argument. Providing a runtime benchmark would be a good support.

2. Coordinate decent is often treated as approximation of Hessian based optimization. This work demonstrated CD performs better than GPTQ's Hessian. While it is encouraging, we'd like to see some explanation why it is the case.

### Questions
Please explain why CD outperforms GPTQ's Hessian based optimization.

### Soundness
3 good

### Presentation
2 fair

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Proposes a method which uses coordinate descent techniques to perform layer-wise quantization, achieving 3-bit quantization

### Strengths
- Interesting coordinate descent formulation, that seems to produce better quantization and is more efficient than another method, gptq.
- nice presentation of the method, and reasonable set of empirical results

### Weaknesses
- I know it may be difficult getting compute resources, but if at all possible I would have liked to see results for AWQ and GPTQ on OPT-66b rather than just “OOM”. GPU memory constraints in quantization are not common, in my view. I don’t want to fault the authors if they don’t have access to larger compute resources.

### Questions
- I am aware of another paper on arXiv called "QuIP: 2-Bit Quantization of Large Language Models With Guarantees" claiming 2 bit quantization for LLM models like OPT and LLama2. I know it was released recently over the summer, but could the authors comment on this work? How does AffineQuant perform at 2 bits?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good
