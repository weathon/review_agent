# Dynamic Sparse No Training:  Training-Free Fine-tuning for Sparse LLMs

- Avg Score: 6.00
- Decision: Accept (poster)
- Scores: 6, 6, 6

## Abstract
The ever-increasing large language models (LLMs), though opening a potential path for the upcoming artificial general intelligence, sadly drops a daunting obstacle on the way towards their on-device deployment. As one of the most well-established pre-LLMs approaches in reducing model complexity, network pruning appears to lag behind in the era of LLMs, due mostly to its costly fine-tuning (or re-training) necessity under the massive volumes of model parameter and training data. To close this industry-academia gap, we introduce Dynamic Sparse No Training ($\texttt{DSNT}$), a training-free fine-tuning approach that slightly updates sparse LLMs without the expensive backpropagation and any weight updates. Inspired by the Dynamic Sparse Training, $\texttt{DSNT}$ minimizes the reconstruction error between the dense and sparse LLMs, in the fashion of performing iterative weight pruning-and-growing on top of sparse LLMs. To accomplish this purpose, $\texttt{DSNT}$ particularly takes into account the anticipated reduction in reconstruction error for pruning and growing, as well as the variance w.r.t. different input data for growing each weight. This practice can be executed efficiently in linear time since its obviates the need of backpropagation for fine-tuning LLMs. Extensive experiments on LLaMA-V1/V2, Vicuna, and OPT across various benchmarks demonstrate the effectiveness of $\texttt{DSNT}$ in enhancing the performance of sparse LLMs, especially at high sparsity levels. For instance, $\texttt{DSNT}$ is able to outperform the state-of-the-art Wanda by 26.79 perplexity at 70% sparsity with LLaMA-7B. Our paper offers fresh insights into how to fine-tune sparse LLMs in an efficient training-free manner and open new venues to scale the great potential of sparsity to LLMs.  Codes are available at https://github.com/zyxxmu/DSnoT.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
Inspired by the weight pruning-and-growing method in dynamic sparse training, the authors propose a training-free fine-tuning method to sparsify LLMs. In practice, the proposed method iteratively performs weight pruning and growing with new importance metrics that take into account the expectation and variance of the reconstruction error reduction. The proposed metrics allow to eliminate the expensive backpropagation or any weight update in the original dynamic sparse training to enable training-free fine-tuning for LLMs. In the experiments, the authors conduct experiments on multiple benchmarks and show better performance when migrating into prior training-free methods.

### Strengths
1. The writing is clear and easy to follow.
2. Although the idea of pruning-and-growing is not new, the proposed method is novel to eliminate backpropagation and weight update with new importance metrics to enable training-free LLM sparsification
3. The proposed method consistently shows a clear performance gap on multiple benchmarks with varied sparsity rates compared to prior works.

### Weaknesses
1. It's unclear how to use the proposed dynamic sparse no training in the whole fine-tuning process. In the paper, the authors mainly illustrate the training-free pruning-and-growing method for one specific layer. It's unknown how the algorithm is used for sparsifying an entire LLM. 

2. It seems the proposed method is an improved technique for existing training-free methods. From the Related work, it can not tell the difference between SparseGPT and Wanda when integrating the proposed method. 

3. The proposed method involves extra computing and running time compared to Wanda.

### Questions
Detailed questions regarding Weakness:

1. It's unknown how the algorithm is used for sparsifying an entire LLM. 

    1.1  Regarding the entire LLM, since the proposed method aims to reduce the reconstruction error in layer-wise, will the proposed method progressive prune each layer or jointly prune all the layers for an entire LLM?

    1.2 How to assign the sparsity rate for each layer? 

2. It seems the proposed method is an improved technique for existing training-free methods. In the related work, it seems that SparseGPT and Wanda have different importance metrics to sparsify LLM, and this work proposes an orthogonal method. 

    2.1 What is the difference between SparseGPT and Wanda when integrating the proposed method? 

     2.2 Why the proposed method can not be considered as an independent method for training-free LLM sparsification?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents DS$\oslash$T, a novel training-free fine-tuning approach for sparse LLMs which edits sparse mask configuration inspired from DST. DS$\oslash$T revives weights which negatively contribute to reconstruction error between dense and sparse LLMs, and prunes weights based on Wanda metric and the sign of reconstruction error. By conducting experiments across a wide range of tasks, authors show that DS$\oslash$T can be seamlessly integrated with existing LLM-pruning techniques, achieving state-of-the-art results at  >50% sparsity regime with minimal computational overhead.

### Strengths
- The paper tackles a timely and practically-relevant problem supported by a fair amount of experiments conducted spanning different tasks and domains. Notably, this is the first work tackling the LLM pruning problem at >50% sparsity regime.
- The proposed method has two distinctive advantages: (i) it doesn't require gradient computation, resulting in minimal overhead costs, and (ii) it reconfigures existing sparse masks, making it compatible with existing LLM pruning methods.
- In general, the paper is well-written and easy to follow.

### Weaknesses
- Throughout the paper, the authors report performance at >50% sparsity regime. However, in most cases, DS$\oslash$T does not bring the performance of sparse neural networks even remotely close to that of the original dense network. To better understand DS$\oslash$T, I recommend reporting experimental results at a lower sparsity regime. For instance, as mentioned in the introduction, the authors stated that baselines start to lose performance at 20% sparsity with LLaMA-30B. Does the use of DS$\oslash$T preserve performance at 20% sparsity regime?
- The paper notes that the overhead cost of using Wanda+DS$\oslash$T is approximately 15 times greater than using Wanda alone in Table 3. However, it appears that the performance gain achieved in Tables 5 and 6 is relatively marginal. For instance, while a previous work [1] argues that the zero-shot classification performance of Wanda and SparseGPT is similar (as seen in Table 2 in [1]), the application of DS$\oslash$T to Wanda or SparseGPT does not consistently result in substantial improvements over the respective baselines. This observation raises questions about the trade-off between computational cost and performance improvement when employing DS$\oslash$T in combination with existing methods. 
- In Table 4, the authors make a comparison between DS$\oslash$T and LoRA. Since DS$\oslash$T alters network structure (pruning mask), the paper would benefit from analyzing whether the resulting sparse network structure from DS$\oslash$T can be further optimized with full-finetuning or LoRA. I wonder whether Wanda+DS$\oslash$T can achieve better fully fine-tuned accuracy compared to that of Wanda.

### Questions
- How many random seeds are used throughout the experiments? 
- Why is LLM-pruner [2] not included in the baselines while N:M structured pruning is included? 

[1] Sun et al., “A simple and effective pruning approach for large language models.” 2023.\
[2] Ma et al., “Llm-pruner: On the structural pruning of large language models.” 2023.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces Dynamic Sparse No Training, a training-free fine-tuning approach for sparse Language Model (LLM) deployment. It minimizes the reconstruction error between dense and sparse LLMs through iterative weight pruning-and-growing. This approach allows for updating sparse LLMs without the expensive backpropagation and weight updates, making it more efficient for on-device deployment. The paper demonstrates the effectiveness of the proposed method on several benchmark datasets, achieving comparable or better performance than traditional fine-tuning approaches while requiring significantly less computation. The contributions of this paper include the introduction of a training-free fine-tuning approach for sparse LLMs, and the demonstration of its effectiveness on several benchmark datasets.

### Strengths
In terms of originality, the paper introduces a novel approach to fine-tuning sparse LMs, called Dynamic Sparse No Training. This approach does not require backpropagation or weight updates, making it more efficient for on-device deployment.

In terms of quality, the authors provide detailed descriptions of the datasets and experimental setup, as well as a thorough analysis of the results. The paper also includes a comprehensive review of related work, highlighting the strengths and weaknesses of existing approaches.

In terms of clarity, this paper is well-written, with clear explanations of the proposed approach and experimental results.

In terms of significance, the paper addresses an important problem in the field of LMs, namely the challenge of deploying large models on resource-constrained devices.

### Weaknesses
1. For the inner loop, how does the threshold affect the final performance of the model, for both perplexity and efficiency?

2. The paper assesses the methods using a consistent sparsity rate of 60%. However, a 50% sparsity rate is more commonly employed in previous baselines. It would be beneficial to see the outcomes at this 50% sparsity level for comparison.

### Questions
please follow weaknesses

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
