# Reduce What You Use: Input‑Aware Matrix‑Multiplication Pruning for LLMs

- Avg Score: 3.60
- Decision: Reject
- Scores: 2, 8, 4, 2, 2

## Abstract
Transformer-based language models achieve strong performance but at high computational cost, raising the question of whether their full dimensional capacity is necessary at inference. We introduce Reduced Matrix-Multiplication (RMM), a training-free rule that adaptively prunes feature dimensions on the fly. Given current activations, RMM scores hidden channels with simple norms, retains a controlled fraction, and performs multiplications only within this reduced subspace—yielding deterministic approximations without altering model weights. Applied uniformly across all linear operations, RMM exposes a smooth accuracy–efficiency frontier governed by a single retention ratio. Across models ranging from 1B to 70B parameters and tasks spanning question answering, reasoning, math, coding, summarization, and vision–language benchmarks, RMM achieves substantial cost reductions with minimal accuracy loss. Larger models tolerate more aggressive pruning, highlighting increasing representational redundancy at scale. These findings demonstrate that high-dimensional computations in LLMs can be systematically compressed, offering a simple and general mechanism for controllable accuracy–efficiency tradeoffs

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Transformer-based language models deliver strong performance but incur substantial computational costs. To mitigate this, the authors introduce Reduced Matrix Multiplication (RMM)—a training-free approach that dynamically prunes feature dimensions during inference using activation norms. The method is motivated by principles from randomized linear algebra. A detailed implementation for a standard transformer architecture is provided in the methods section, along with a theoretical analysis of computational complexity. The experimental section presents evaluations of RMM across various large language models with different retention ratios, comparisons to other pruning techniques, vision–language model (VLM) assessments, and an ablation study.

### Strengths
* Extensive evaluations on diverse LLMs and benchmarks provide a thorough assessment of RMM’s empirical contribution.
* The authors present a well-founded motivation for their approach through an analysis grounded in randomized linear algebra.
* The paper is clearly written, well organized, and conceptually intuitive.

### Weaknesses
While the paper’s premise is to introduce a training-free approach, it remains somewhat difficult to fully assess its effectiveness without comparison to methods that incorporate post-training following pruning. For example, [1] statically prunes a 70B LLaMA model to 51B and 49B variants with minimal post-training, preserving 98.4% of the original model’s accuracy. This approach not only achieves higher accuracy retention than RMM but also reduces the total parameter count and operates without requiring a custom kernel for efficient inference.

As noted in the conclusion, developing an efficient kernel would allow for a more complete evaluation of RMM’s potential efficiency gains, especially when combined with an appropriate training strategy to demonstrate the benefits of dynamic pruning. However, without these components, the method’s contribution remains limited in significance compared to existing baselines.



[1] Puzzle: Distillation-Based NAS for Inference-Optimized LLMs, Bercovich et al. (https://arxiv.org/abs/2411.19146)

### Questions
* The method is described for single batch generation. For the case of multi batch generation, do the author believe that RMM, which will probably be based on the norm of features throughout the batch, can provide consistent results?

### Soundness
3

### Presentation
3

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
The paper introduces RMM, a novel method to reduce the high computational cost of LLMs at inference time . 
Authors argue that massive matrix multiplications in a transformer are redundant, and not all feature dimensions are necessary for every input .

### Strengths
Simple and elegant solution: The idea was simple but made a lot of sense. I personally enjoyed reading the paper. 

Training-free, general and input-adaptive: RMM is a "training-free rule" that can be applied to any pre-trained Transformer without altering model weights. 

Comprehensive and Insightful Ablations: The ablation study (RQ4) provides a interesting insight: attention modules are highly redundant and safe to prune, while MLP blocks are highly sensitive and pruning them causes "sharp degradation". This finding is valuable for any future work on hybrid pruning strategies.

### Weaknesses
New Computational Overhead Ablation: The proposed method "dynamically recomputes the retained subspace" at every layer and for every token. This involves calculating L2 norms and running a topk operation, which adds its own latency. The paper does not analyze the cost of this overhead.

Missing Baselines: There are lots of work in this area (as mentioned in the intro and related work), but authors only compare with H2O pruning, I would like to see some more baseline results for RQ3.

### Questions
1. Could you provide detail/ablation study item for latency of dynamically finding best features to prune? 
2. Could you add some recent papers to experiments for RQ3 (Sec 4.3)

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Reduced Matrix Multiplication (RMM), a training-free technique that replaces full matrix multiplication with an approximate variant capable of pruning feature dimensions on the fly. Given two matrices $A$ and $B$, and a pruning ratio $\text{RR}$, the method approximates their product $AB$ by sampling $k$ columns from $A$ (and the corresponding rows from $B$). The selection is guided by the column norms of $A$, retaining those with the largest norms so that the total proportion of used dimensions matches the specified ratio $\text{RR}$. The authors compare RMM against baseline pruning methods and a key–value (KV) cache optimization approach, and further analyze how both the pruning ratio and model scale affect overall performance.

### Strengths
-The RMM approximation is simple yet neat and intuitive, grounded in the principles of randomized linear algebra. It is easy to implement and enables structured feature pruning, which may, in the future, lead to actual speedups if efficiently implemented on GPUs.

-The method requires no training or fine-tuning. The authors demonstrate how this approach can be applied to compute both the attention product $QK^T$ and the MLP matrix product within transformer layers. However, the principle behind RMM can substitute any matrix multiplication, making it a highly general and versatile method.

-The experiments investigating RMM performance explore various pruning ratios (RRs), providing intuition about how the number of included rank-1 components influences network performance.

-Overall, the paper is generally clear and easy to follow.

### Weaknesses
Novelty:
The approximated matrix multiplication underlying RMM essentially corresponds to the sampling-based approximate matrix multiplication proposed by Drineas et al., which the authors also cite. The application of different approximate matrix multiplication (AMM) algorithms has already been surveyed in Zeng et al., including the works of Chen et al. and Blalock et al., which focus on developing and applying AMM methods in deep learning. Thus, the paper’s contribution is primarily the application of the algorithm from Drineas et al. to large language models (LLMs).

Experimental design and performance:
The analysis of RMM’s impact on generative models in Table 3 and Figure 2 is purely qualitative. While such visualizations provide useful intuition about the method’s behavior, they do not constitute valid evidence for drawing conclusions about the model’s benefits. Moreover, it would be helpful if the authors reported standard deviations, particularly since in some cases (e.g., RMM at RR = 0.8), the results are nearly identical (24.23 vs. 24.24).

Since the authors present RMM as a pruning method, a more extensive comparison with alternative pruning approaches would be advisable—for example, SparseGPT (Frantar et al.) or SliceGPT (Ashkboos Saleh et al.). On this note, the meaning of “static pruning” in the baseline is unclear. Does it refer to static magnitude pruning on weights, or pruning based on expected activations? This baseline should be explained more precisely.

Furthermore, the performance appears to degrade rather quickly—only pruning ratios of 0.9 or 0.8 maintain performance close to the baseline models. This limitation reduces the practicality of the method in scenarios requiring higher pruning ratios.

Finally, as far as can be understood, the reported speedup is purely theoretical. No wall-clock time measurements are provided, and there do not appear to be GPU-efficient implementations available at this stage.


References:

Chen, Yifan, et al. "Sketching as a tool for understanding and accelerating self-attention for long sequences." arXiv preprint arXiv:2112.05359 (2021).  
Zeng, Xianzhi, Wenchao Jiang, and Shuhao Zhang. "LibAMM: Empirical Insights into Approximate Computing for Accelerating Matrix Multiplication." Advances in Neural Information Processing Systems 37 (2024): 60517-60530.  
Blalock, Davis, and John Guttag. "Multiplying matrices without multiplying." International Conference on Machine Learning. PMLR, 2021.  
Drineas, Petros, Ravi Kannan, and Michael W. Mahoney. "Fast Monte Carlo algorithms for matrices I: Approximating matrix multiplication." SIAM Journal on Computing 36.1 (2006): 132-157.   
Frantar, Elias, and Dan Alistarh. "Sparsegpt: Massive language models can be accurately pruned in one-shot." International conference on machine learning. PMLR, 2023.  
Ashkboos, Saleh, et al. "Slicegpt: Compress large language models by deleting rows and columns." arXiv preprint arXiv:2401.15024 (2024).

### Questions
Questions:

Did the authors try to also train the models with RMM (not just use it at inference),, even for a high RR? Does such approach even allows for stable learning?

Why RMM is only looking at rows of matrix A, when equation (2) suggest a term based both on column and row norms?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a method, Reduced Matrix Multiplication, to reduce the inference time and computational cost of LLMs by reducing the number of matrix multiplication operations. RMM replaces the full matrix multiplication of two matrices A and B with an approximate solution, which uses only a subset of the columns of A and corresponding rows of B. RMM assignes score to columns dynamically on the fly making agnostic making it more robust to different inputs.

### Strengths
*The paper is well-written and provides a comprehensive overview of the background and related work.

*While several recent studies have demonstrated that LLMs perform a significant amount of redundant computation, most prior works have focused on different components of LLMs—such as sparse activation or model pruning. In contrast, the authors address a more fundamental question: how to reduce the cost of matrix multiplication in LLMs through approximation.

### Weaknesses
* There is no discussion about the computational cost of computing norm of column vector for each inference step in section 3. 

* The complexity analysis does not take into account the computational complexity for calculating norm. 

* The method is compared to static pruning, random pruning and H20 cache reduction. A comparison with other post-training pruning methods, such as SparseGPT, LLM Pruner, and Wanda, is lacking. 

* There is a significant drop in performance, even at higher Retention Rate (RR) as observed in Table 1 and Table 2. State-of-the-art pruning methods easily achieve a 50% sparse model with a minimum drop in performance. 

* Based on the results provided, there is no clear advantage for the proposed method in terms of efficiency. The method results in a significant drop in performance, making it hard to justify not using model pruning, quantization or distillation for improving the inference time efficiency.

### Questions
What advantage does RMM provide over other compression methods?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
## Overall Summary

Transformer-based language models achieve remarkable performance but incur high computational cost at inference. This paper introduces **Reduced Matrix-Multiplication (RMM)** — a *training-free*, adaptive pruning rule that reduces the dimensionality of intermediate computations on the fly.  

RMM works by computing scores for each hidden dimension based on the magnitude of activations, ranking them, and retaining only a controlled fraction of the most informative ones. The model then performs all linear operations within this reduced subspace. This approach yields deterministic approximations without altering model weights, offering a simple and efficient mechanism for reducing inference cost.

## Contributions

1. **Reduced Matrix-Multiplication (RMM):**  
   A simple, training-free mechanism that adaptively prunes hidden dimensions based on current activations. No retraining or architectural changes are required.

2. **Single control parameter:**  
   The retention ratio provides a smooth and interpretable *accuracy–efficiency frontier*, enabling users to balance computational savings and performance.

3. **Empirical results:**  
   RMM achieves significant inference cost reductions across transformer models ranging from **1B to 70B parameters**, with minimal accuracy loss. The method generalizes across diverse tasks, including question answering, reasoning, mathematics, code generation, summarization, and vision–language benchmarks.

4. **Scaling insight:**  
   Larger models tolerate more aggressive pruning, suggesting that representational redundancy increases with scale.

## Conclusion

RMM demonstrates that high-dimensional computations in large language models can be systematically compressed without retraining. This provides a simple and general mechanism for controllable *accuracy–efficiency trade-offs* and highlights the inherent redundancy of large-scale neural representations.

### Strengths
## Strengths - Originality and Quality
- The overall idea of pruning channels based on their importance has been studied quite widely in [1] and [2]
- The main difference or contribution in this paper is the dynamic computation of per-token dimension importance and the way the attention score is computed 
- The baselines considered in the paper are quite limited and relevant baselines like [1], [2], [3] are not considered.
- Questions:
    - Could the authors clarify the differences between RMM and structural pruning approaches like Minitron [2]?

## Strengths - Clarity
- The paper is written clearly in most parts
- I have the following clarification questions:
     - Have the authors considered pruning the width of the network or the embedding dimension?
     - Instead of head dimension, have the authors considered pruning the number of heads directly?
     - Models like llama-3.1-8B support grouped query attention, are the neurons/features pruned same across heads in the same group?
## Strengths - Significance
- The primary target domain of the paper ie. structural pruning and efficiency in LLMs is very relevant and significant


[1]  Molchanov, P., Mallya, A., Tyree, S., Frosio, I. and Kautz, J., 2019. Importance estimation for neural network pruning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 11264-11272).
[2] Muralidharan, S., Turuvekere Sreenivas, S., Joshi, R., Chochowski, M., Patwary, M., Shoeybi, M., Catanzaro, B., Kautz, J. and Molchanov, P., 2024. Compact language models via pruning and knowledge distillation. Advances in Neural Information Processing Systems, 37, pp.41076-41102.

### Weaknesses
- Check strengths 
## Weakness - Contribution
- I find the general premise of the proposed approach to be very similar to Minitron [2] with some differences in terms of how the importance scores are computed. I encourage the authors to discuss the differences between the two more thoroughly and also if possible compare against Minitron 

## Weakness - Experiments
- The baselines the method is compared to are quite limited. I recommend the authors to compare against approaches like SliceGPT [1] and Minitron [2]

## Weakness - On device latency gains
- To the best of my understanding of the paper, the important neurons/features are computed dynamically in a token dependent way. Does the method yield a single deployable model, with a fixed retention ratio? Since the activated neurons can change for every batch of data, the pruned architecture (with specified feature selection) changes for every input batch. Could the authors confirm if my understanding here is correct?

[1] Ashkboos, S., Croci, M.L., Nascimento, M.G.D., Hoefler, T. and Hensman, J., 2024. Slicegpt: Compress large language models by deleting rows and columns. arXiv preprint arXiv:2401.15024.
[2] Muralidharan, S., Turuvekere Sreenivas, S., Joshi, R., Chochowski, M., Patwary, M., Shoeybi, M., Catanzaro, B., Kautz, J. and Molchanov, P., 2024. Compact language models via pruning and knowledge distillation. Advances in Neural Information Processing Systems, 37, pp.41076-41102.

### Questions
Check strengths and weaknesses

### Soundness
3

### Presentation
3

### Contribution
1
