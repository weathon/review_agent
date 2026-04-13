## Human Reviewer 1

### Summary
This paper proposes a rank-aware activation sparsity, including applying input sparsification and weight decomposition. Experiments show that the proposed R-Sparse improves end-to-end efficiency while maintaining comparable sparsity.

### Strengths
1. The proposed methods are easy to understand.
2. The experimental results are good.

### Weaknesses
1. The connections between motivations and the proposed methods are weak. For example, the first motivation claims that the outputs for the non-sparse inputs can be regarded as biases. However, it is not sure which part of the proposed methods is motivated by this. Please explicitly explain how the observation about non-sparse inputs being treated as biases directly informs specific components of their R-Sparse method.

2. The contributions are incremental, which only directly apply existing techniques, such as CATS for sparsity, SVD for weight decomposition, and genetic algorithm for hyperparameters searching. Please more clearly articulate the novelty of your approach. Is there any specific design in SVD?

3. The paper claims that existing non-ReLU activations such as SiLU and GELU introduce less sparsity. However, the experiments lack generality as Llama 2/3 and Mistral all adopt SiLU as activation functions. For example, adding results of models using GELU activation functions such as Gemma would be helpful.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
The authors propose a novel training-free activation sparsity method called R-sparse. This method is applicable to non-ReLU-based large language models (LLMs) and eliminates the need for prediction by utilizing input activation sparsity. Furthermore, it is a method that can be applied not only to MLP modules but also to attention modules.

### Strengths
- The authors successfully identified existing challenges and, in connection to these, suggested a training-free, prediction-free activation sparsity method. Furthermore, their method is applicable to attention modules.
- Through various experiments, they demonstrated performance improvements.

### Weaknesses
- difficult to read (clearity)
  - In introduction, how should Figure 1 be interpreted? It is difficult to understand the meaning of the figure until reviewing the explanation in Section 3.3.
  - In Section 3.2, what does the term "bias" mean? Is it interpreted as "bias" in the sense that performance is maintained even after replacing non-sparse values smaller than 0 with a constant? Does the term also include non-sparse components greater than 0? In Lines 183-185, why is it defined as "sparse", when $H_k \geq T_0$?

- the appropriateness of the title.
  - The proposed method appears to be "activation sparsity, then low-rank decomposition for non-important activations" rather than using both approaches simultaneously (rank-aware activation sparsity).



==== After Rebuttal ====
- I understand. However, the terms in Section 3.2 should be clarified more explicitly.
- It might be beneficial to add an explanation using examples, when $T_0 = 0$.
  - positive = sparse components = not pruned values
  - negative = non-sparse components = to be pruned values
- In fact, the most confusing term is "sparse components." It would be beneficial to clearly indicate, as suggested in the response, that it originates from prior research.

### Questions
- (Writing) Please switch the positions between mlp.up_proj and mlp.gate_proj of layer 0 in Figure 3.
- Recommended reference
  - Alizadeh, Keivan, et al. "Llm in a flash: Efficient large language model inference with limited memory."
- Definition of multi-phase ReLU.
  - Is the multi-phase ReLU expressed on line 170 correctly defined? Where is sparsity defined? Shouldn't there be a definition for when x $< T_{L}$? Moreover, the output should get closer to zero as it becomes more negative, but it is defined in the opposite direction.
  - For instance, $T_0 = 0, T_1=-1, T_2=-2$, please explain it.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper uses the inherent sparsity and low-rank properties of input activations in LLMs to accelerate the inference of LLMs. It sparsifies the input activations to remove unnecessary activations, and interprets unnecessary activations as a bias term, using SVD to compensate for this bias. The proposed method achieves higher sparsity than the baseline while maintaining overall performance.

### Strengths
- This paper proposes a training-free activation sparsification method to speed up LLM inference. The proposed method can maintain performance at the same level as full parameters at sparsity of about 50%.
- This paper proposes to design different sparsity ratios for different layers to improve overall performance.
- This paper is very clearly written and the proposed method is easy to follow.

### Weaknesses
- The analysis of "the contribution of each input channel and singular value component" in this paper is mainly focused on the C4 dataset (Figures 1 and 3). What are the similarities and differences between the analysis on other datasets and the C4 dataset? Especially on datasets that are very different from the C4 dataset. In addition, these analyses are mainly based on 16 randomly sampled training samples. When the number of samples increases or decreases, what changes will occur in the analysis results?
- As shown in Figure 3, there is a clear difference in the importance of different linear layers (such as self_attn.k_proj vs. self_attn.up_proj), what this mainly stems from, the authors can give more comments on this.
- What is the relationship between the sparsity ratio in the proposed R-Sparse and the final inference acceleration? For example, what is the corresponding acceleration for a certain sparsity ratio?
- In Figure 6, why under Dense, when the Generation Length becomes longer (1024->2048), the generation speed slows down, while when it is 128->256->512, the generation speed is accelerated.

### Questions
See the Weaknesses section

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 4

### Summary
This paper presents R-Sparse, a training-free activation sparsity approach for large language models (LLMs). Current activation sparsity methods face limitations with non-ReLU activation functions and have difficulties in predicting active channels and achieving high sparsity ratios. R-Sparse overcomes these challenges by leveraging the sparsity of input channels and singular value components. The authors conduct investigations and find that non-sparse components can be regarded as bias terms and full computation can be approximated by a combination of input channels and weight singular values. R-Sparse is applied to both attention and MLP modules of LLMs and an evolutionary search algorithm is used to find optimal sparse component ratios. Experiments on Llama-2/3 and Mistral models across ten tasks show that R-Sparse achieves 50% model-level sparsity with comparable performance and up to 43% end-to-end speed improvement with a customized kernel.

### Strengths
* The training-free method does not require extensive pre-training, making it more efficient and easier to implement compared to methods that need continual training.
* R-Sparse achieves high sparsity levels (50% model-level sparsity) without sacrificing performance, leading to significant improvements in efficiency.
* R-Sparse is compatible with weight quantization for further efficiency gains and can be applied to different LLM families and a variety of tasks.

### Weaknesses
* Table 1 is difficult to interpret. From the description of the authors, R-Sparse40% is compared with CATS22% and GRIFFIN33%, if my understanding is correct. Therefore, R-Sparse does not consistently outperform CATS across all tasks (e.g., PIQA 78.24 vs 79.00). The authors are suggested to refine the claim to avoid the misleading. Meanwhile, for certain cases, the performance increases with an even higher sparsity ratio (e.g., 79.49 vs 79.92 for R-Sparse40% and R-Sparse50%) on PIQA. Could the authors provide some insights into this phenomenon?
* The sensitivity analysis of hyperparameters should be added for a more thorough investigation of the effectiveness of R-Sparse.
* As [1] is discussed by the authors in the related works, why do the authors choose not to compare with [1] in the experiments?
* The authors are suggested to include the complexity analysis and running time comparison of R-Sparse, especially regarding the evolutionary search algorithm.
* The writing can be further improved. For example, the optimal sparse-rank $\alpha$ is not formally defined. From Algorithm 1, $\alpha$ seems to be fixed, how is it optimized?

[1] Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time, ICML 2023

### Questions
Please kindly refer to the Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3