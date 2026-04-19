# Sparse Spectral Training and Inference on Euclidean and Hyperbolic Neural Networks

- Decision: Reject
- Scores: 6, 5, 5, 5

## Abstract
The growing demands on GPU memory posed by the increasing number of neural network parameters call for training approaches that are more memory-efficient. Previous memory reduction training techniques, such as Low-Rank Adaptation (LoRA) and ReLoRA, face challenges, with LoRA being constrained by its low-rank structure, particularly during intensive tasks like pre-training, and ReLoRA suffering from saddle point issues. In this paper, we propose Sparse Spectral Training (SST) to optimize memory usage for pre-training. SST updates all singular values and selectively updates singular vectors through a multinomial sampling method weighted by the magnitude of the singular values. Furthermore, SST employs singular value decomposition to initialize and periodically reinitialize low-rank parameters, reducing distortion relative to full-rank training compared to other low-rank methods. Through comprehensive testing on both Euclidean and hyperbolic neural networks across various tasks, including natural language generation, machine translation, node classification, link prediction, and image classification, SST demonstrates its ability to outperform existing memory reduction training methods and is comparable to full-rank training in various cases. On LLaMA-1.3B, with only 18.7\% of the parameters trainable compared to full-rank training (using a rank equivalent to 6\% of the embedding dimension), SST reduces the perplexity gap between other low-rank methods and full-rank training by 97.4\%. This result highlights SST as an effective parameter-efficient technique for model pre-training, offering a promising new paradigm for achieving scalable and memory-efficient neural network training. Our code is available at https://anonymous.4open.science/r/sparse_spectral_training-6A2C/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
By selectively updating singular values ​​and vectors in the weight matrix, this paper proposes Sparse Spectral Training (SST), a memory-efficient technique for training large-scale neural networks. SST reduces memory usage while maintaining comparable performance to full-rank training by using SVD for startup and periodic reinitialization. SST outperforms current low-rank techniques such as LoRA and ReLoRA in machine translation and natural language processing, and shows competitive results when tested on a variety of neural network architectures and tasks

### Strengths
1. By focusing on selective updates, SST provides an efficient memory management technique that significantly reduces GPU memory usage during training.

2. The proposed method reduces the perplexity and BLEU score gaps on benchmarks, showing competitive or superior performance on a range of neural network architectures and tasks, often approaching the effect of full-rank training.

3. Regular use of SVD helps SST alleviate the saddle point problem of ReLoRA, bringing its training dynamics closer to full-rank training.

### Weaknesses
1. Frequent SVD reinitialization may affect scalability for very large models and incur significant computational overhead.

2. As with other low-rank techniques, stability issues can still occur in very low-rank models even with the sampling mechanism of SST.

3. Users who are not familiar with these techniques may find it challenging to get started, as memory efficiency optimizations (such as dividing active and frozen parts) and selective update methods may make the implementation more complex.

### Questions
1. How does periodic SVD reinitialization affect scalability as model size increases, especially in larger language models or high-dimensional settings?
2. How sensitive is the polynomial sampling strategy to the selection of singular values? Can the stability and efficiency of specific application scenarios be further improved by adopting different sampling methods?
3. Does periodic SVD reinitialization affect the convergence speed of models that are more constrained by low rank? In this case, is it possible that SST requires more training steps to achieve convergence compared to full-rank training?
4. Can the authors provide more practical guidance to help tune SST's hyperparameters (such as SVD reinitialization frequency) to achieve the best balance of memory efficiency and performance in different tasks?

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
4

### Summary
The paper introduces sparse spectral training (SST) to optimize memory usage for pre-training. SST updates all singular values and selectively updates singular vectors through a multinomial sampling method weighted by the magnitude of the singular values, and employs singular value decomposition to initialize and periodically reinitialize low rank parameters, therefore reducing distortion. The author of the paper conduct experiments on natural language generation, machine translation, node classification, link prediction, and image classification, and SST demonstrates its ability to outperform existing memory reduction training methods and is comparable to full-rank training in various cases.

### Strengths
1. This paper is the first to propose parameter-efficient pre-training process in hyperbolic space, in which the design of SST is quite novel and effective. 

2. Using SVD to approximate low-rank computation is somehow novel especially with the design of selective updates of singular vector and iterative reinitialization during training. 

3. The experimental results show promising performance of SST.

4. This paper is well organized with clear discussion on the disadvantages of the baseline methods, such as ReLoRA.

### Weaknesses
1. The overall framework and algorithm design are similar to ReLoRA, which limits the novelty aspect of the paper.

2. Trainable parameter count does not showing significant advantages over baseline methods, such as ReLoRA, and the GPU memory consumption is much larger compared with other methods. Considering the design is using SVD, this weakness is inevitable. Therefore, it limits the wide adoption of this method in the future.

3. Although the author of the paper discusses why the SVD decomposition approach is important, the discussion fails to convince the reader that this approach is necessary. According to the experimental results, ReLoRA achieves quite comparable results with lower memory consumption.

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper propose a novel memory-efficient LLM pre-training method SST, which employs singular value decomposition to initialize and periodically reinitialize lowrank parameters. The experimental results illustrate that the proposed can achieve a better performance than LoRA and its variants.

### Strengths
1. The paper focused on an important problem about memoey-efficient LLM pre-training. 

2. The proposed memory-efficient LLM pre-training method SST is easy to follow. 

3. This paper provides diverse results on different tasks to verify the performance of proposed method. The results illustrate that the proposed method can achieve a better performance than LoRA and its variants.

### Weaknesses
1. The motivation and intuition is not very clear. I'm not very clear about the reason why we need to use the proposed method Sparse Spectral Training (SST) for pre-training.  I mean why we need to leverage sparse updates within the spectral domain of neural network
weights. I understand the paper  provided some limitations of LoRA ReLoRA and Galore, but some related work tried to solve these limitations. For example, the limitation of ReLoRA about zero initialization of B can result in zero gradients of A, some related work has solved it, such as Pissa [1]. 

2. The additional training cost from SST could be increased with scaling the model size. For example, the results of Table 16 about overall training time shows the training time will be increased from 303.4h to 387.2h when using SST. 

3. The sensitivity analysis of the proposed method about hyper-parameters, such as the number of iterations and iteration interval. 

4. The experiments for pre-training task mainly focus on the small scale, not sure the results when sclaing to a larger dataset and model. 

[1] PISSA: PRINCIPAL SINGULAR VALUES AND SINGULAR VECTORS ADAPTATION OF LARGE LANGUAGE MODELS

### Questions
1. I would like to ask whether the authors tries to tune the hyper-parameters of baseline methods, such as learning rate, rank value.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes Sparse Spectral Training (SST), a memory-efficient training method designed for large neural networks. SST aims to reduce memory usage while maintaining performance close to full-rank models. The method updates all singular values of network weights and selectively updates singular vectors based on their significance, using a sampling strategy that prioritizes impactful components. SST is evaluated on tasks across Euclidean and hyperbolic neural networks, showing that it often outperforms existing memory reduction techniques and occasionally matches or exceeds full-rank training performance, especially in low-dimensional or resource-constrained settings.

### Strengths
Sparse Spectral Training (SST) introduces an interesting approach to memory-efficient training by updating all singular values while selectively focusing on significant singular vectors. This approach shows promise for reducing memory requirements and can be a creative improvement on existing low-rank adaptation methods.

### Weaknesses
While SST is promising, the current experiments lack rigor, especially as the performance comparisons rely on relatively high perplexity values and low zero-shot accuracies, limiting the conclusions that can be drawn about SST's effectiveness. For a more meaningful evaluation, it would be valuable to benchmark against models that achieve stronger results (e.g., perplexity below 10). Additionally, the selection of baselines could be strengthened to more comprehensively illustrate SST’s relative performance. These changes would help clarify SST’s potential as a competitive memory-efficient training method.

### Questions
- How would SST perform if compared against models capable of achieving lower perplexity scores (e.g., below 10)? What is the extent to which SST’s performance might vary when applied to higher-performing models with more rigorous baseline comparisons?
- Are there alternative sampling mechanisms that the authors considered, and could these impact the efficiency or stability of SST?

### Soundness
2

### Presentation
2

### Contribution
3
